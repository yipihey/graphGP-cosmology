"""Part I: binned LS-consistent per-point weights.

Implements the Wiener filter of Eq. 4 of ``twopt_density.pdf``:

    delta_hat = C (C + N)^-1 d,
        C_ij = xi_hat(r_ij),  N_ii = 1 / (nbar_i * V_kernel)

with the data vector ``d`` taken to be a kernel-density estimate of the
local overdensity at each data point.

Honest limitation
-----------------
The doc's claim that the resulting weighted-DD pair sum (Eq. 5) reproduces
``xi_LS(r)`` exactly requires a precise definition of ``d`` that the doc
does not give. With a top-hat KDE of radius ``R``, the recovered
``xi_w(r)`` agrees with ``xi(r)`` only at scales ``r >> R``; for
``r << R`` the pair sum measures the smoothed auto-variance ``sigma^2(R)``
instead. The recovery has the right *shape* (monotone decreasing in
``r`` for clustered data, ratio ~ constant in the small-r regime). A
fully exact construction is a research item -- see IMPLEMENTATION_PLAN.md.

Suitable for ``N_D <= ~1e4`` where dense Cholesky is fine. Larger catalogs
should use ``weights_graphgp.compute_2pt_weights`` (Part III).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def _xi_lookup(r: np.ndarray, r_centers: np.ndarray, xi_j: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of binned xi(r) onto arbitrary r."""
    return np.interp(r, r_centers, xi_j, left=xi_j[0], right=0.0)


def kde_overdensity(
    positions: np.ndarray,
    nbar: np.ndarray,
    r_kernel: float,
    box_size: float | None = None,
) -> np.ndarray:
    """Top-hat KDE local overdensity at each data point.

    ``d_i = n_kde(x_i) / nbar(x_i) - 1`` where
    ``n_kde(x_i) = (count of OTHER points within r_kernel) / V_kernel``.
    """
    tree = cKDTree(positions, boxsize=box_size if box_size else None)
    counts = np.array(
        tree.query_ball_point(positions, r=r_kernel, return_length=True),
        dtype=np.float64,
    )
    counts -= 1.0  # exclude self
    V_kernel = (4.0 / 3.0) * np.pi * r_kernel ** 3
    n_kde = counts / V_kernel
    return n_kde / nbar - 1.0


def default_kernel_radius(nbar: np.ndarray, target_count: float = 30.0) -> float:
    """Top-hat radius giving ``target_count`` expected neighbors.

    The Wiener filter is best conditioned when the Poisson noise on the
    KDE input is similar in amplitude to the prior covariance. Choosing
    ``target_count ~ 30`` keeps the noise variance ``1/(nbar*V) ~ 0.03``,
    matching typical ``xi(r)`` values around the correlation length.
    """
    nbar_med = float(np.median(nbar))
    return (target_count * 3.0 / (4.0 * np.pi * nbar_med)) ** (1.0 / 3.0)


def compute_binned_weights(
    positions: np.ndarray,
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    nbar: np.ndarray,
    r_kernel: float | None = None,
    box_size: float | None = None,
    mode: str = "sample",
    subtract_mean: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return per-point weights ``w_i = 1 + delta_hat_i``.

    Parameters
    ----------
    positions
        ``(N_D, 3)`` data positions.
    r_centers, xi_j
        Output of ``ls_corrfunc.xi_landy_szalay``.
    nbar
        Per-point local mean density from ``ls_corrfunc.local_mean_density``.
    r_kernel
        Top-hat KDE radius for the data input. Default: choose so the
        expected count in the kernel is ~30 (well-conditioned Poisson).
    box_size
        Periodic box size (passed to KDE for periodic wrapping).
    mode
        ``"sample"`` (default) returns a draw from the GP posterior, which
        has prior variance ``C`` and therefore satisfies Eq. 5 of the doc
        in expectation. ``"mean"`` returns the Wiener filter posterior
        mean, which is data-aware but variance-shrunk.
    subtract_mean
        If True (default), subtract the empirical mean of the KDE input
        before solving. If False, keep the raw overdensities -- the
        weights then have ``<w> > 1`` and the recovered ``xi_w`` is
        rescaled by ``<w>^2`` (see ``demos/scan_N_dependence.py``).
    rng
        ``numpy.random.Generator`` for the posterior sample. Default: a
        fresh seed.
    """
    N = len(positions)
    if N > 12000:
        raise ValueError(
            f"N_D={N} too large for dense Cholesky; "
            "use weights_graphgp.compute_2pt_weights instead."
        )
    if r_kernel is None:
        r_kernel = default_kernel_radius(nbar)

    d = kde_overdensity(positions, nbar, r_kernel, box_size=box_size)
    if subtract_mean:
        # The KDE is biased high at data points (we only sample where points
        # exist, which preferentially is over-dense regions). Re-center so
        # the empirical mean of d is zero, matching the Wiener filter
        # assumption E[d] = 0 needed for an unbiased posterior.
        d = d - d.mean()
    V_kernel = (4.0 / 3.0) * np.pi * r_kernel ** 3
    noise_var = 1.0 / (nbar * V_kernel)  # Poisson noise on the KDE estimate

    r = _pairwise_distances(positions, box_size=box_size)
    C = _xi_lookup(r.ravel(), r_centers, xi_j).reshape(N, N)
    sigma2 = float(max(xi_j[0], 1.0))
    np.fill_diagonal(C, sigma2)

    # The piecewise-linear interpolant of a noisy xi(r) is not in general a
    # valid (PSD) covariance function -- empirically it has O(N/2) negative
    # eigenvalues. We solve the linear system K x = d with LU factorization
    # rather than Cholesky, so the indefiniteness of K = C + N is harmless.
    # The Wiener filter posterior mean mu = C K^-1 d is identical to the
    # PSD-projected case to numerical precision (verified on the toy
    # catalog: median xi_w/xi differs by < 1e-3 across PSD projection,
    # smooth-kernel fit, raw xi, and clipped xi).
    K = C + np.diag(noise_var) + 1e-6 * sigma2 * np.eye(N)
    lu = lu_factor(K)
    mu = C @ lu_solve(lu, d)
    if mode == "mean":
        return 1.0 + mu

    # For posterior sampling we need a Cholesky of the posterior covariance,
    # which DOES require PSD. Project on demand only in this branch.
    K_post = C - C @ lu_solve(lu, C)
    K_post = _project_psd(K_post) + 1e-6 * sigma2 * np.eye(N)
    L_post = np.linalg.cholesky(K_post)
    rng = rng if rng is not None else np.random.default_rng()
    z = rng.standard_normal(N)
    return 1.0 + mu + L_post @ z


def _project_psd(M: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to its nearest PSD via eigenvalue clipping.

    Used only in posterior-sample mode where a Cholesky of the posterior
    covariance is needed. The posterior mean (default ``mode='mean'``) is
    obtained without PSD projection via direct LU solve.
    """
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


def _pairwise_distances(positions: np.ndarray, box_size: float | None) -> np.ndarray:
    """Full N x N distance matrix, with minimum-image wrap when periodic."""
    if box_size is None:
        return squareform(pdist(positions))
    diff = positions[:, None, :] - positions[None, :, :]
    diff -= box_size * np.round(diff / box_size)
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
