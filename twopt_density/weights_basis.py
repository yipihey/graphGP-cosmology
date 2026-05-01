"""Part II: basis-form Wiener filter weights.

Same dense solve as ``weights_binned`` but with a smooth covariance from the
SFH basis: ``C_ij = sum_alpha theta_hat_alpha f_alpha(r_ij)``. Removes the
bin-edge artifacts of Part I.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
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
) -> np.ndarray:
    """Return per-point weights ``w_i = 1 + delta_hat_i`` (basis form).

    Same Wiener-filter solve as ``weights_binned`` with a smooth
    ``C_ij = sum_alpha theta_alpha f_alpha(r_ij)``.
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
    d = d - d.mean()
    V_kernel = (4.0 / 3.0) * np.pi * r_kernel ** 3
    noise_var = 1.0 / (nbar * V_kernel)

    from .weights_binned import _project_psd

    r = squareform(pdist(positions))
    C = np.maximum(
        xi_from_basis(theta_hat, basis, r.ravel()).reshape(N, N), 0.0
    )
    diag_xi = float(xi_from_basis(theta_hat, basis, np.array([basis.r_min]))[0])
    sigma2 = max(diag_xi, 1.0)
    np.fill_diagonal(C, sigma2)
    C = _project_psd(C)
    L = cho_factor(C + np.diag(noise_var) + 1e-6 * sigma2 * np.eye(N), lower=True)
    delta_hat = C @ cho_solve(L, d)
    return 1.0 + delta_hat
