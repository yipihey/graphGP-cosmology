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


def compute_basis_weights(
    positions: np.ndarray,
    theta_hat: np.ndarray,
    basis: Basis,
    nbar: np.ndarray,
) -> np.ndarray:
    """Return per-point weights ``w_i = 1 + delta_hat_i`` (basis form)."""
    N = len(positions)
    if N > 12000:
        raise ValueError(
            f"N_D={N} too large for dense Cholesky; "
            "use weights_graphgp.compute_2pt_weights instead."
        )
    r = squareform(pdist(positions))
    C = xi_from_basis(theta_hat, basis, r.ravel()).reshape(N, N)
    diag = float(xi_from_basis(theta_hat, basis, np.array([basis.r_min]))[0])
    np.fill_diagonal(C, diag)
    Ninv = np.diag(1.0 / nbar)
    L = cho_factor(C + Ninv, lower=True)
    n_minus_nbar = (1.0 - nbar / nbar.mean())
    delta_hat = C @ cho_solve(L, n_minus_nbar)
    return 1.0 + delta_hat
