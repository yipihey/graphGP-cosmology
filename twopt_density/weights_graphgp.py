"""Part III: GP / Vecchia weights via graphgp.

Sec. 4.4 of ``twopt_density.pdf``: build per-point density weights using the
Vecchia approximation of a Gaussian process whose covariance kernel is the
measured xi(r). Cost is O(N * k^3) time and O(N * k) memory, scaling
linearly to N ~ 10^9 with the right hardware.

The graphgp pipeline is::

    cov   = tabulate_kernel(r_centers, xi_j)
    graph = gp.build_graph(positions, n0=N0, k=K)
    delta = gp.generate(graph, cov, xi_white)

where ``gp.generate`` computes ``L @ xi_white`` for the Vecchia Cholesky
factor ``L`` (so ``L L^T`` approximates the prior covariance). Feeding the
mean-centered, unit-variance KDE overdensity in as ``xi_white`` yields a
data-aware smoothed field whose pair correlations recover the input
xi(r) -- this is the "calibrated GP sample" form discussed in
IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import numpy as np


def fit_kernel(r_centers: np.ndarray, xi_j: np.ndarray) -> tuple[float, float, float]:
    """Fit a stretched-exponential ``k(r) = A exp(-(r/r0)^alpha)`` to xi(r).

    This guarantees a smooth, positive, monotone-decreasing kernel that
    Cholesky-decomposes cleanly inside graphgp's per-block refinement
    step. A simple unweighted least-squares is sufficient for the typical
    LS estimator (signal dominated up to ~50 Mpc, then noise).
    """
    from scipy.optimize import curve_fit

    def model(r, A, r0, alpha):
        return A * np.exp(-((r / r0) ** alpha))

    mask = (xi_j > 0)
    if mask.sum() < 4:
        # Fallback: monotonic decay from the largest bin.
        return float(xi_j.max()), float(r_centers[-1] / 2), 1.0
    A0 = float(xi_j[mask].max())
    r0_0 = float(r_centers[xi_j > 0.5 * A0][-1]) if (xi_j > 0.5 * A0).any() else float(r_centers[0])
    try:
        popt, _ = curve_fit(
            model, r_centers[mask], xi_j[mask], p0=[A0, r0_0, 1.5],
            bounds=([0.01, 0.5, 0.3], [1e3, 200.0, 3.0]),
            maxfev=2000,
        )
        A, r0, alpha = popt
    except Exception:
        A, r0, alpha = A0, r0_0, 1.5
    return float(A), float(r0), float(alpha)


def tabulate_kernel(
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    n_bins: int = 200,
    jitter: float = 1e-2,
):
    """Build a graphgp-format ``(cov_bins, cov_vals)`` tuple from xi(r).

    Fits a stretched-exponential parametric form to the measured xi(r) so
    the resulting kernel is guaranteed PSD; graphgp's per-block Cholesky
    inside ``refine`` requires this. Tabulates onto a log-spaced grid in
    graphgp's convention: ``cov_bins[0] = 0`` is the diagonal,
    ``cov_bins[1:]`` is logspace(r_min, r_max).

    Parameters
    ----------
    r_centers, xi_j
        From ``ls_corrfunc.xi_landy_szalay``.
    r_min, r_max
        Cover range for the discretized kernel. Default: spans
        ``r_centers``.
    n_bins
        Number of log-spaced bins (plus the implicit zero bin).
    jitter
        Multiplicative inflation on ``k(0)`` for PSD safety. graphgp's
        docstring: "If using your own covariance, inflate k(0) by a small
        factor to ensure positive definite."

    Returns
    -------
    (cov_bins, cov_vals) : pair of jax arrays in graphgp's expected form.
    fit_params : ``(A, r0, alpha)`` of the fitted ``A exp(-(r/r0)^alpha)``.
    """
    import jax.numpy as jnp

    r_min = r_min if r_min is not None else float(r_centers[0])
    r_max = r_max if r_max is not None else float(r_centers[-1])
    A, r0, alpha = fit_kernel(r_centers, xi_j)

    cov_bins_np = np.concatenate([
        [0.0],
        np.logspace(np.log10(r_min), np.log10(r_max), n_bins - 1),
    ])
    cov_vals_np = A * np.exp(-((cov_bins_np / r0) ** alpha))
    cov_vals_np[0] = A * (1.0 + jitter)
    return (jnp.asarray(cov_bins_np), jnp.asarray(cov_vals_np)), (A, r0, alpha)


def compute_2pt_weights(
    positions: np.ndarray,
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    nbar: np.ndarray | None = None,
    box_size: float | None = None,
    n0: int = 100,
    k: int = 30,
    r_kernel: float | None = None,
    mode: str = "prior_sample",
    seed: int = 0,
    n_kernel_bins: int = 200,
    return_diagnostics: bool = False,
):
    """Layer III per-point density weights via a Vecchia GP sample.

    Two modes are supported:

    ``"prior_sample"`` (default)
        Draw white noise ``xi ~ N(0, I)`` and compute
        ``delta = generate(graph, cov, xi) = L xi``. The result has prior
        covariance ``L L^T = Sigma`` exactly, so the weighted-DD pair sum
        recovers ``xi(r)`` in expectation. Data-agnostic in values, but
        evaluated AT the data positions, so per-point weights still
        encode the local correlation structure.

    ``"data_driven"``
        Use the (mean-centered, unit-variance) KDE overdensity as the
        white-noise input: ``delta = L * d_normalized``. Data-aware. The
        recovered xi has the same calibration relation as Layer I:
        scaled by ``<w>^2`` plus the weight-correlation term.

    Parameters
    ----------
    positions
        ``(N_D, 3)`` data positions.
    r_centers, xi_j
        Output of ``ls_corrfunc.xi_landy_szalay``.
    nbar
        Per-point local mean density. Required only for ``data_driven``.
    box_size, r_kernel
        For the KDE in ``data_driven`` mode (passed to Layer I helpers).
    n0, k
        graphgp Vecchia parameters: dense initial block size and number
        of conditional neighbors per point.
    mode
        ``"prior_sample"`` or ``"data_driven"``.
    seed
        Seed for the white-noise draw in ``prior_sample`` mode.
    n_kernel_bins
        Number of bins for the discretized covariance kernel.
    return_diagnostics
        If True, also return the fitted kernel params and the Graph.

    Returns
    -------
    weights
        ``(N_D,)`` numpy array of per-point density weights.
    diagnostics (optional)
        dict with keys ``'kernel_fit'`` (``(A, r0, alpha)``) and ``'graph'``.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import graphgp as gp

    N = len(positions)

    # 1. Tabulate kernel (parametric, PSD by construction).
    cov, fit_params = tabulate_kernel(
        r_centers, xi_j,
        r_min=float(max(r_centers[0], 0.5)),
        r_max=float(r_centers[-1]),
        n_bins=n_kernel_bins,
    )

    # 2. Build the Vecchia graph.
    points = jnp.asarray(positions, dtype=jnp.float64)
    graph = gp.build_graph(points, n0=min(n0, max(2, N // 2)),
                           k=min(k, N - 1))

    # 3. Build the white-noise input.
    if mode == "prior_sample":
        rng = np.random.default_rng(seed)
        xi_white = rng.standard_normal(N).astype(np.float64)
    elif mode == "data_driven":
        if nbar is None:
            raise ValueError("data_driven mode requires nbar")
        from .weights_binned import kde_overdensity, default_kernel_radius
        if r_kernel is None:
            r_kernel = default_kernel_radius(nbar)
        d = kde_overdensity(positions, nbar, r_kernel, box_size=box_size)
        d = d - d.mean()
        d_std = float(np.std(d))
        xi_white = (d / d_std) if d_std > 1e-12 else d
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # 4. Apply the Vecchia Cholesky factor.
    delta = np.asarray(gp.generate(graph, cov, jnp.asarray(xi_white)))
    weights = 1.0 + delta

    if return_diagnostics:
        return weights, {"kernel_fit": fit_params, "graph": graph}
    return weights
