"""Part II: basis-form Wiener filter weights.

Same dense solve as ``weights_binned`` but with a smooth covariance from the
SFH basis: ``C_ij = sum_alpha theta_hat_alpha f_alpha(r_ij)``. Removes the
bin-edge artifacts of Part I.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.distance import pdist, squareform

from .basis import Basis
from .basis_projection import xi_from_basis
from .weights_binned import kde_overdensity


def compute_basis_weights(
    positions: np.ndarray,
    theta_hat: np.ndarray,
    basis: Basis,
    nbar: np.ndarray,
    r_kernel: float | None = None,
    box_size: float | None = None,
    subtract_mean: bool = True,
) -> np.ndarray:
    """Return per-point weights ``w_i = 1 + delta_hat_i`` (basis form).

    Same Wiener-filter solve as ``weights_binned`` with a smooth
    ``C_ij = sum_alpha theta_alpha f_alpha(r_ij)``. Solved with LU
    factorization (no PSD projection) -- the indefiniteness of the
    smoothed-xi kernel is harmless for the posterior mean.
    """
    N = len(positions)
    if N > 12000:
        raise ValueError(
            f"N_D={N} too large for dense Cholesky; "
            "use weights_graphgp.compute_2pt_weights instead."
        )
    from .weights_binned import default_kernel_radius
    if r_kernel is None:
        r_kernel = default_kernel_radius(nbar)

    d = kde_overdensity(positions, nbar, r_kernel, box_size=box_size)
    if subtract_mean:
        d = d - d.mean()
    V_kernel = (4.0 / 3.0) * np.pi * r_kernel ** 3
    noise_var = 1.0 / (nbar * V_kernel)

    r = squareform(pdist(positions))
    C = xi_from_basis(theta_hat, basis, r.ravel()).reshape(N, N)
    diag_xi = float(xi_from_basis(theta_hat, basis, np.array([basis.r_min]))[0])
    sigma2 = max(diag_xi, 1.0)
    np.fill_diagonal(C, sigma2)
    K = C + np.diag(noise_var) + 1e-6 * sigma2 * np.eye(N)
    lu = lu_factor(K)
    delta_hat = C @ lu_solve(lu, d)
    return 1.0 + delta_hat
