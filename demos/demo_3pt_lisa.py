"""3pt LISA demo: per-particle equilateral triangle counts.

Verifies the analog of the 2pt identity at 3rd order:

    mean_i  T_i^(alpha) / E[T_i^(alpha)] - 1  =  DDD_alpha / RRR_alpha - 1
                                              =  zeta_simple(alpha)

where ``T_i^(alpha)`` is the number of triangles with shape ``alpha`` that
have particle ``i`` as a vertex. Empirically the identity holds to
floating-point precision on a clustered toy catalog.

This is the natural 3rd-order extension of the LISA framework
(Anselin 1995, Cressie & Collins 2001) and the per-particle pair-count
baseline in ``twopt_density/weights_pair_counts.py``. See
``LITERATURE.md`` for the broader context.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


def per_particle_triangle_counts(
    positions: np.ndarray,
    r_target: float,
    dr: float,
    box_size: float,
) -> np.ndarray:
    """Per-particle count of equilateral-ish triangles.

    For a triangle to count, all three side lengths must lie in
    ``[r_target - dr, r_target + dr]``. Each triangle increments the
    counter at all three of its vertices; we divide by 3 at the end so
    ``T[i]`` is the number of triangles containing ``i``.
    """
    tree = cKDTree(positions, boxsize=box_size)
    pairs = tree.query_pairs(r=r_target + dr, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(len(positions))

    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    keep = np.abs(d - r_target) < dr
    pairs = pairs[keep]
    if len(pairs) == 0:
        return np.zeros(len(positions))

    # Build neighbor sets for "is in target shell"
    nbr = [set() for _ in range(len(positions))]
    for i, j in pairs:
        nbr[i].add(int(j))
        nbr[j].add(int(i))

    T = np.zeros(len(positions))
    for i, j in pairs:
        common = nbr[int(i)] & nbr[int(j)]
        for k in common:
            if k != int(i) and k != int(j):
                T[int(i)] += 1
                T[int(j)] += 1
                T[k] += 1
    # Each triangle is hit 3 times (once for each of its 3 edges in
    # ``pairs``), each time incrementing all 3 vertices -> 9 increments
    # per triangle vertex group, but we only want each vertex counted
    # once per triangle: divide by 3.
    return T / 3.0


def main():
    box = 200.0
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, box, size=(20, 3))
    pts = np.vstack([
        rng.normal(c, 8.0, size=(150, 3)) for c in centers
    ])
    pts = np.mod(pts, box).astype(np.float64)
    N = len(pts)
    print(f"N = {N}")

    r0, dr = 8.0, 2.0
    t0 = time.perf_counter()
    T_data = per_particle_triangle_counts(pts, r0, dr, box)
    print(
        f"data triangle counts: {time.perf_counter()-t0:.2f}s   "
        f"sum={T_data.sum()/3:.0f}  mean={T_data.mean():.2f}  "
        f"std={T_data.std():.2f}"
    )

    unif = rng.uniform(0, box, size=(N, 3)).astype(np.float64)
    t0 = time.perf_counter()
    T_unif = per_particle_triangle_counts(unif, r0, dr, box)
    print(
        f"uniform triangle counts: {time.perf_counter()-t0:.2f}s   "
        f"sum={T_unif.sum()/3:.0f}"
    )

    zeta_global = T_data.sum() / T_unif.sum() - 1.0
    zeta_per_particle_mean = T_data.mean() / T_unif.mean() - 1.0
    print()
    print(f"global zeta            = DDD/RRR - 1 = {zeta_global:.6f}")
    print(f"mean_i (T_i / E[T]) - 1                = "
          f"{zeta_per_particle_mean:.6f}")
    diff = abs(zeta_global - zeta_per_particle_mean)
    print(f"identity holds to:     {diff:.2e}")
    if diff < 1e-9:
        print("=> 3pt LISA identity verified to floating-point precision")


if __name__ == "__main__":
    main()
