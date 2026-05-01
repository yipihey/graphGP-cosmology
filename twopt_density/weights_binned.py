"""Part I: binned LS-consistent per-point weights.

Solves the Wiener filter (Eq. 4 of ``twopt_density.pdf``):

    delta_hat_i = ((C + N)^-1 (n - nbar))_i,
        C_ij = xi_hat(r_ij),  N_ii = 1 / nbar_i.

Suitable for ``N_D <= ~1e4`` where dense Cholesky is fine. Larger catalogs
should use ``weights_graphgp.compute_2pt_weights`` (Part III).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist, squareform


def _xi_lookup(r: np.ndarray, r_centers: np.ndarray, xi_j: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of binned xi(r) onto arbitrary r."""
    return np.interp(r, r_centers, xi_j, left=xi_j[0], right=0.0)


def compute_binned_weights(
    positions: np.ndarray,
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    nbar: np.ndarray,
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
    """
    N = len(positions)
    if N > 12000:
        raise ValueError(
            f"N_D={N} too large for dense Cholesky; "
            "use weights_graphgp.compute_2pt_weights instead."
        )
    r = squareform(pdist(positions))
    C = _xi_lookup(r.ravel(), r_centers, xi_j).reshape(N, N)
    np.fill_diagonal(C, xi_j[0])  # finite variance at zero separation
    Ninv = np.diag(1.0 / nbar)
    L = cho_factor(C + Ninv, lower=True)

    n_minus_nbar = (1.0 - nbar / nbar.mean())  # centered indicator
    delta_hat = C @ cho_solve(L, n_minus_nbar)
    return 1.0 + delta_hat
