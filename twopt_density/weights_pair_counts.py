"""Baseline: per-particle weights from per-particle pair-count histograms.

Idea
----
For each particle ``i`` and bin ``j`` we count ``b_i^(j)`` -- the number of
data partners of ``i`` whose separation falls in bin ``j``. The total
``DD_j = (1/2) sum_i b_i^(j)`` (each unordered pair counted twice in the
per-particle sum).

The per-particle, per-bin overdensity (uniform-Poisson form) is

    delta_i^(j) = b_i^(j) / E[b^(j)] - 1,
    E[b^(j)]   = (N - 1) * V_shell_j / V_box

This satisfies

    < delta_i^(j) >_i = (1/N) sum_i b_i^(j) / E[b^(j)] - 1
                     = 2 DD_j / (N E[b^(j)]) - 1
                     = DD_j / RR_j - 1
                     = xi_simple(r_j),

so the *average* of the per-particle overdensities matches the simple
Landy-Szalay-equivalent estimator (modulo the 2DR/RR correction for
non-periodic surveys).

Window-aware form
-----------------
For a survey with a non-trivial window we cannot use the uniform-Poisson
expectation. The natural per-particle generalization of LS is

    delta_i^(j) = (b_DD_i^(j) * N_R) / (b_DR_i^(j) * N_D) - 1,

where ``b_DR_i^(j)`` counts pairs of data point ``i`` with the random
catalog. This cancels both the global density and the local window
shape at each particle: at the survey boundary, both ``b_DD`` and
``b_DR`` are reduced equally, so the ratio is unbiased. The identity
becomes

    < delta_i^(j) >_i = DD_j / DR_j  *  N_R / N_D  -  1
                     = xi_simple_LS(r_j),

i.e., the simple ``DD/DR - 1`` estimator (Davis-Peebles); for full
LS replace ``b_DR_i`` with the LS denominator
``(2 b_DR_i - b_RR_i_proxy)``.

A single per-particle weight is then aggregated across bins; see
``aggregate_weights``.

Cost
----
``O(N log N)`` per pair-counter call. The window-aware path runs the
counter twice (DD and DR). For Quijote-like 5000-halo catalogs that is
< 1 s; for N = 1e5 with the same density it is ~10 s. Corrfunc's
``DD`` call is faster but does not expose per-particle counts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _shell_volumes(r_edges: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)


def _count_dtype(n_max: int) -> np.dtype:
    """Smallest unsigned-int dtype that can hold counts up to ``n_max``."""
    if n_max < 256:
        return np.uint8
    if n_max < 65_536:
        return np.uint16
    if n_max < 4_294_967_296:
        return np.uint32
    return np.uint64


def per_particle_pair_counts(
    positions: np.ndarray,
    r_edges: np.ndarray,
    box_size: float | None = None,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Return ``(N, n_bins)`` array of per-particle pair counts.

    ``b[i, j]`` is the count of partners ``k != i`` whose separation
    ``r_ik`` falls in ``[r_edges[j], r_edges[j+1])``. Each unordered pair
    contributes to both ``b[i, j]`` and ``b[k, j]``, so
    ``sum_i b[i, j] = 2 * DD_j_unordered``.

    The output uses an unsigned-integer dtype sized to ``N`` by default
    (uint8 / uint16 / uint32 / uint64 chosen automatically). Caller can
    override via ``dtype``. Halving the storage from float64 -> uint32
    is a free win because per-particle counts are bounded by ``N - 1``.

    Implementation: ``cKDTree.query_pairs`` returns every pair within
    ``r_max`` once; we histogram and accumulate into the per-particle
    count array using ``numpy.add.at``. Cost is dominated by the tree
    query (``O(N log N)``); the histogram is essentially free.
    """
    N = len(positions)
    n_bins = len(r_edges) - 1
    r_max = float(r_edges[-1])
    tree = cKDTree(positions, boxsize=box_size if box_size else None)

    if dtype is None:
        dtype = _count_dtype(N)

    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((N, n_bins), dtype=dtype)

    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    if box_size is not None:
        diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    bin_idx = np.searchsorted(r_edges, d, side="right") - 1
    mask = (bin_idx >= 0) & (bin_idx < n_bins)
    pi = pairs[mask, 0]
    pk = pairs[mask, 1]
    bj = bin_idx[mask]

    b = np.zeros((N, n_bins), dtype=dtype)
    np.add.at(b, (pi, bj), np.array(1, dtype=dtype))
    np.add.at(b, (pk, bj), np.array(1, dtype=dtype))
    return b


def per_particle_cross_counts(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    r_edges: np.ndarray,
    box_size: float | None = None,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Cross-catalog per-particle pair counts.

    For each point ``a`` in ``positions_a``, count partners ``b`` in
    ``positions_b`` with separation in each bin. Self-pairs are NOT
    excluded (the two catalogs are assumed disjoint, e.g. data x random).

    The dtype is sized to ``len(positions_b)`` (max possible count).
    """
    Na = len(positions_a)
    Nb = len(positions_b)
    n_bins = len(r_edges) - 1
    r_max = float(r_edges[-1])
    tree_b = cKDTree(positions_b, boxsize=box_size if box_size else None)

    if dtype is None:
        dtype = _count_dtype(Nb)

    neighbors = tree_b.query_ball_point(positions_a, r=r_max)
    b_AB = np.zeros((Na, n_bins), dtype=dtype)

    flat_a = np.repeat(np.arange(Na), [len(L) for L in neighbors])
    flat_b = np.fromiter(
        (j for L in neighbors for j in L),
        dtype=np.intp,
        count=int(sum(len(L) for L in neighbors)),
    )
    if flat_a.size == 0:
        return b_AB

    diff = positions_a[flat_a] - positions_b[flat_b]
    if box_size is not None:
        diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    bin_idx = np.searchsorted(r_edges, d, side="right") - 1
    mask = (bin_idx >= 0) & (bin_idx < n_bins)
    np.add.at(b_AB, (flat_a[mask], bin_idx[mask]),
              np.array(1, dtype=dtype))
    return b_AB


def per_particle_overdensity_windowed(
    b_DD_per_particle: np.ndarray,
    b_DR_per_particle: np.ndarray,
    N_D: int,
    N_R: int,
) -> np.ndarray:
    """Window-aware per-particle, per-bin overdensity (Davis-Peebles form).

    ``delta_i^(j) = (b_DD_i^(j) * N_R) / (b_DR_i^(j) * N_D) - 1``

    At the survey boundary both ``b_DD`` and ``b_DR`` are reduced
    proportionally by the same window factor, so the ratio is unbiased.
    Returns NaN at points where ``b_DR_i^(j) = 0`` (genuinely outside
    the survey for that scale).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (b_DD_per_particle * N_R) / (b_DR_per_particle * N_D)
        delta = ratio - 1.0
    return delta


def per_particle_overdensity(
    b_per_particle: np.ndarray,
    r_edges: np.ndarray,
    N: int,
    box_size: float,
) -> np.ndarray:
    """Periodic, uniform-Poisson per-particle, per-bin overdensity.

    ``delta_i^(j) = b_i^(j) / E[b^(j)] - 1``, with
    ``E[b^(j)] = (N-1) V_shell_j / V_box``. Use this only for periodic
    boxes with no survey window; for survey geometry call
    ``per_particle_overdensity_windowed`` instead.
    """
    V_box = box_size ** 3
    expected = (N - 1) * _shell_volumes(r_edges) / V_box
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
    box_size: float | None = None,
    randoms: np.ndarray | None = None,
    mode: str = "RR_xi",
    xi_floor: float = 0.0,
):
    """End-to-end baseline: positions + LS output -> per-particle weights.

    If ``randoms`` is given, uses the window-aware Davis-Peebles form
    ``delta_i^(j) = (b_DD_i N_R) / (b_DR_i N_D) - 1`` -- robust to
    survey edges. Otherwise uses the periodic uniform-Poisson form (and
    requires ``box_size``).

    Returns
    -------
    weights : ``(N,)`` numpy array.
    delta_per_bin : ``(N, n_bins)`` per-particle overdensity at each scale.
    """
    N = len(positions)
    b_DD = per_particle_pair_counts(positions, r_edges, box_size=box_size)
    if randoms is None:
        if box_size is None:
            raise ValueError("Either ``randoms`` or ``box_size`` must be given.")
        delta_per_bin = per_particle_overdensity(b_DD, r_edges, N, box_size)
    else:
        b_DR = per_particle_cross_counts(
            positions, randoms, r_edges, box_size=box_size,
        )
        delta_per_bin = per_particle_overdensity_windowed(
            b_DD, b_DR, N, len(randoms),
        )
        # Guard against NaNs (points with no R neighbors at any scale).
        delta_per_bin = np.where(
            np.isfinite(delta_per_bin), delta_per_bin, 0.0,
        )
    weights = aggregate_weights(
        delta_per_bin, xi_j, RR_j, mode=mode, xi_floor=xi_floor,
    )
    return weights, delta_per_bin
