"""Project fine-bin Corrfunc pair counts onto an SFH basis.

Implements Sec. 3.1-3.2 of ``twopt_density.pdf``:

    DD_alpha = sum_{i<k} f_alpha(r_ik)
             ~ sum_j  f_alpha(r_j_center) * DD_j        (fine-bin limit)
    theta_hat = (F^T F)^-1 F^T y,  y_j = (DD_j - 2 DR_j + RR_j)/RR_j

A future ``suave`` backend can be plugged in if ``suave`` is importable.
"""

from __future__ import annotations

import numpy as np

from .basis import Basis


def project_pair_counts(
    r_centers: np.ndarray,
    DD_j: np.ndarray,
    DR_j: np.ndarray,
    RR_j: np.ndarray,
    basis: Basis,
):
    """Return ``(DD_alpha, DR_alpha, RR_alpha, theta_hat)``.

    ``theta_hat`` is the least-squares basis decomposition of the binned
    LS estimator ``y_j = (DD - 2 DR + RR)/RR_j`` (Eq. 6 of the doc).
    """
    F = basis.evaluate(r_centers).T  # shape (n_bins, n_basis)
    DD_alpha = F.T @ DD_j
    DR_alpha = F.T @ DR_j
    RR_alpha = F.T @ RR_j

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(RR_j > 0, (DD_j - 2.0 * DR_j + RR_j) / RR_j, 0.0)
    # Weighted least-squares with RR_j as the natural weight.
    W = np.diag(RR_j)
    theta_hat, *_ = np.linalg.lstsq(F.T @ W @ F, F.T @ W @ y, rcond=None)
    return DD_alpha, DR_alpha, RR_alpha, theta_hat


def xi_from_basis(theta_hat: np.ndarray, basis: Basis, r: np.ndarray) -> np.ndarray:
    """Evaluate the smooth ``xi_hat(r) = sum_alpha theta_alpha f_alpha(r)``."""
    return basis.evaluate(r).T @ theta_hat
