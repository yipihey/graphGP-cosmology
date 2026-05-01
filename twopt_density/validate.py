"""Weighted-DD re-pairing check (Eq. 5 of ``twopt_density.pdf``)."""

from __future__ import annotations

import numpy as np

from .ls_corrfunc import xi_landy_szalay


def weighted_xi(
    positions: np.ndarray,
    weights: np.ndarray,
    r_edges: np.ndarray,
    box_size: float | None,
    nthreads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute xi(r) from weighted DD pair sums on data only.

    Returns ``(r_centers, xi_w)`` where
    ``xi_w(r) = DD_w(r) / RR(r) - 1``.
    """
    r_centers, _, RR_j, DD_w, _ = xi_landy_szalay(
        positions, randoms=None, r_edges=r_edges, box_size=box_size,
        nthreads=nthreads, weights=weights,
    )
    xi_w = np.where(RR_j > 0, DD_w / RR_j - 1.0, 0.0)
    return r_centers, xi_w


def assert_recovery(
    xi_target: np.ndarray, xi_w: np.ndarray, rtol: float = 0.1
) -> None:
    """Raise if recovered xi_w deviates from xi_target by more than rtol."""
    err = np.abs(xi_w - xi_target) / np.maximum(np.abs(xi_target), 1e-3)
    if np.any(err > rtol):
        raise AssertionError(
            f"weighted-DD recovery failed: max relative error {err.max():.3f} > {rtol}"
        )
