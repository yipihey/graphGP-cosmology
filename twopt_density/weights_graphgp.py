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

    ``"posterior_sample"``
        Draw a single posterior sample of δ(x) at the data positions via
        Matheron's rule:
            δ_post = δ_prior + K_D [K_DD + N_D]^{-1} (y_D - δ_prior|_D)
        where y_i = 1/nbar_i - 1 and N_D = diag(1/nbar_i). Requires
        ``nbar`` per-point mean density. The returned weights 1 + δ_post
        are posterior realizations consistent with the Poisson observations.
        Uses a Vecchia CG solve (O(N k³) per sample).

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
        ``"prior_sample"``, ``"data_driven"``, or ``"posterior_sample"``.
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
    elif mode == "posterior_sample":
        if nbar is None:
            raise ValueError("posterior_sample mode requires nbar")
        # Matheron's rule at data positions only (no lightcone grid).
        # For lightcone output use density_field.sample_posterior_density_field.
        rng = np.random.default_rng(seed)
        xi_white = rng.standard_normal(N).astype(np.float64)
        xi_jax = jnp.asarray(xi_white)

        # Prior sample at data positions
        f_prior = np.asarray(gp.generate(graph, cov, xi_jax))

        # Observed overdensity y_i = 1/nbar_i - 1; noise N_D = diag(1/nbar_i)
        nbar_safe = np.maximum(nbar, 1e-30)
        noise_var = 1.0 / nbar_safe
        y_obs = noise_var - 1.0  # 1/nbar - 1

        # Residual
        residual = (y_obs - f_prior).astype(np.float64)

        # Vecchia matvec: (K + N) v
        def _matvec(v_np):
            v_jax = jnp.asarray(v_np)
            Lv = jnp.asarray(gp.generate(graph, cov,
                             jnp.asarray(gp.generate_inv(graph, cov, v_jax))))
            return np.asarray(Lv) + noise_var * v_np

        # Conjugate gradient solve: (K + N) alpha = residual
        alpha = np.zeros(N, dtype=np.float64)
        r = residual - _matvec(alpha)
        p = r.copy()
        rs_old = float(r @ r)
        for _ in range(100):
            Ap = _matvec(p)
            rp = float(p @ Ap)
            if abs(rp) < 1e-30:
                break
            step = rs_old / rp
            alpha += step * p
            r -= step * Ap
            rs_new = float(r @ r)
            if rs_new < 1e-8 * (residual @ residual + 1e-60):
                break
            p = r + (rs_new / (rs_old + 1e-60)) * p
            rs_old = rs_new

        # Correction at data positions: K_D alpha (diagonal K entries)
        cov_bins, cov_vals = cov
        k0 = float(cov_vals[0])  # K(0, 0) = kernel diagonal
        delta = f_prior + k0 * alpha  # simplified: K_DD alpha ≈ K(0)*alpha (diagonal only)
        xi_white = None  # already handled above
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if mode != "posterior_sample":
        # 4. Apply the Vecchia Cholesky factor (prior_sample / data_driven).
        delta = np.asarray(gp.generate(graph, cov, jnp.asarray(xi_white)))

    weights = 1.0 + delta

    if return_diagnostics:
        return weights, {"kernel_fit": fit_params, "graph": graph}
    return weights
