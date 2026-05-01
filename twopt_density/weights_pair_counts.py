"""Baseline: per-particle weights from per-particle pair-count histograms.

Idea
----
For each particle ``i`` and bin ``j`` we count ``b_i^(j)`` -- the number of
data partners of ``i`` whose separation falls in bin ``j``. The total
``DD_j = (1/2) sum_i b_i^(j)`` (each unordered pair counted twice in the
per-particle sum).

The per-particle, per-bin overdensity is

    delta_i^(j) = b_i^(j) / E[b^(j)] - 1,
    E[b^(j)]   = (N - 1) * V_shell_j / V_box   (uniform-Poisson expectation)

This satisfies

    < delta_i^(j) >_i = (1/N) sum_i b_i^(j) / E[b^(j)] - 1
                     = 2 DD_j / (N E[b^(j)]) - 1
                     = DD_j / RR_j - 1
                     = xi_simple(r_j),

so the *average* of the per-particle overdensities matches the simple
Landy-Szalay-equivalent estimator (modulo the 2DR/RR correction for
non-periodic surveys).

A single per-particle weight is then aggregated across bins:

    delta_i = (sum_j w_j delta_i^(j)) / (sum_j w_j),
    w_i     = 1 + delta_i.

The default aggregator weights bins by ``RR_j`` (Poisson-noise optimal)
and restricts to the strongly-clustered range ``xi_LS(r) > xi_floor``.

Cost
----
``O(N k_max)`` for the per-particle pair counting via a periodic
KD-tree, where ``k_max`` is the typical number of neighbors within the
maximum bin radius. For Quijote-like 5000-halo catalogs that is
< 1 s; for N = 1e5 with the same density it is ~10 s. The Corrfunc
``DD`` call is faster but does not expose per-particle counts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _shell_volumes(r_edges: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)


def per_particle_pair_counts(
    positions: np.ndarray,
    r_edges: np.ndarray,
    box_size: float | None = None,
) -> np.ndarray:
    """Return ``(N, n_bins)`` array of per-particle pair counts.

    ``b[i, j]`` is the count of partners ``k != i`` whose separation
    ``r_ik`` falls in ``[r_edges[j], r_edges[j+1])``. Each unordered pair
    contributes to both ``b[i, j]`` and ``b[k, j]``, so
    ``sum_i b[i, j] = 2 * DD_j_unordered``.

    Implementation: ``cKDTree.query_pairs`` returns every pair within
    ``r_max`` once; we histogram and accumulate into the per-particle
    count array using ``numpy.add.at``. Cost is dominated by the tree
    query (``O(N log N)``); the histogram is essentially free.
    """
    N = len(positions)
    n_bins = len(r_edges) - 1
    r_max = float(r_edges[-1])
    tree = cKDTree(positions, boxsize=box_size if box_size else None)

    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((N, n_bins), dtype=np.float64)

    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    if box_size is not None:
        diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    bin_idx = np.searchsorted(r_edges, d, side="right") - 1
    mask = (bin_idx >= 0) & (bin_idx < n_bins)
    pi = pairs[mask, 0]
    pk = pairs[mask, 1]
    bj = bin_idx[mask]

    b = np.zeros((N, n_bins), dtype=np.float64)
    np.add.at(b, (pi, bj), 1.0)
    np.add.at(b, (pk, bj), 1.0)
    return b


def per_particle_overdensity(
    b_per_particle: np.ndarray,
    r_edges: np.ndarray,
    N: int,
    box_size: float,
) -> np.ndarray:
    """Return ``(N, n_bins)`` per-particle, per-bin overdensity.

    ``delta_i^(j) = b_i^(j) / E[b^(j)] - 1``, with the uniform-Poisson
    expectation ``E[b^(j)] = (N-1) V_shell_j / V_box``.
    """
    V_box = box_size ** 3
    expected = (N - 1) * _shell_volumes(r_edges) / V_box  # shape (n_bins,)
    return b_per_particle / expected[None, :] - 1.0


def aggregate_weights(
    delta_per_bin: np.ndarray,
    xi_j: np.ndarray,
    RR_j: np.ndarray,
    mode: str = "RR_xi",
    xi_floor: float = 0.0,
) -> np.ndarray:
    """Aggregate per-bin per-particle deltas into a single weight per particle.

    ``delta_i = sum_j a_j * delta_i^(j) / sum_j a_j`` with the bin
    coefficients ``a_j`` chosen by ``mode``:

    ``"RR_xi"`` (default)
        ``a_j = RR_j * |xi_j|`` for ``xi_j > xi_floor``, else 0. This
        weights informative bins by their pair count (volume) and signal.
    ``"RR"``
        ``a_j = RR_j`` (volume average; least biased to clustering scale).
    ``"smallest_bin"``
        ``a_j = delta_{j,0}`` (use the smallest bin only; noisy but fully
        local).
    """
    if mode == "smallest_bin":
        return 1.0 + delta_per_bin[:, 0]
    if mode == "RR":
        a = np.maximum(RR_j, 0.0)
    elif mode == "RR_xi":
        a = np.where(xi_j > xi_floor, RR_j * np.abs(xi_j), 0.0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    if a.sum() == 0:
        a = np.maximum(RR_j, 0.0)
    delta = (delta_per_bin * a[None, :]).sum(axis=1) / a.sum()
    return 1.0 + delta


def compute_pair_count_weights(
    positions: np.ndarray,
    r_edges: np.ndarray,
    xi_j: np.ndarray,
    RR_j: np.ndarray,
    box_size: float,
    mode: str = "RR_xi",
    xi_floor: float = 0.0,
):
    """End-to-end baseline: positions + LS output -> per-particle weights.

    Returns
    -------
    weights : ``(N,)`` numpy array.
    delta_per_bin : ``(N, n_bins)`` per-particle overdensity at each scale
        (returned for diagnostics / multi-scale analyses).
    """
    N = len(positions)
    b = per_particle_pair_counts(positions, r_edges, box_size=box_size)
    delta_per_bin = per_particle_overdensity(b, r_edges, N, box_size)
    weights = aggregate_weights(
        delta_per_bin, xi_j, RR_j, mode=mode, xi_floor=xi_floor,
    )
    return weights, delta_per_bin
