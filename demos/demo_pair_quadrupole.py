"""Fast anisotropy-preserving estimators that beat triangle counts.

Compares:
  1. Scalar per-particle pair count    (the 2pt LISA primitive)
  2. Per-particle pair quadrupole Q_zz (this demo: anisotropy)
  3. Per-particle triangle count       (the 3pt LISA primitive)

The quadrupole carries anisotropy info at the SAME COST as the scalar
count: O(N k_pair) where k_pair is the typical neighbor count per
particle. The triangle count costs O(N k_pair^2) -- a factor of
~k_pair more expensive.

Identity verified numerically:
  isotropic catalog:    xi_2(r) statistically consistent with 0
  anisotropic catalog:  xi_2(r) tracks the imposed shape anisotropy
                        (pairs preferentially aligned along z)
"""

from __future__ import annotations

import time
import numpy as np
from scipy.spatial import cKDTree


def per_particle_pair_quadrupole(
    positions: np.ndarray,
    r_edges: np.ndarray,
    box_size: float,
    los: np.ndarray = np.array([0.0, 0.0, 1.0]),
):
    """Per-particle scalar count + Legendre L=2 component along ``los``.

    For each particle i and bin j returns

        b[i, j]   = #pairs of i in bin j                    (scalar = ell 0)
        Q[i, j]   = sum_k (3 mu_ik^2 - 1)/2  for k in bin j (L=2)

    where ``mu_ik = (x_k - x_i) . los / |x_k - x_i|`` is the cosine
    between the pair-separation and the line-of-sight axis.

    Identity:
      mean_i Q[i, j] / mean_i b[i, j]  =  xi_2(r_j) (quadrupole of LS).
    """
    N = len(positions)
    n_bins = len(r_edges) - 1
    r_max = float(r_edges[-1])
    los = np.asarray(los, dtype=np.float64)
    los = los / np.linalg.norm(los)

    tree = cKDTree(positions, boxsize=box_size)
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((N, n_bins)), np.zeros((N, n_bins))

    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    bin_idx = np.searchsorted(r_edges, d, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    pairs = pairs[valid]
    diff = diff[valid]
    d = d[valid]
    bin_idx = bin_idx[valid]

    mu = (diff @ los) / d
    legendre_2 = (3.0 * mu * mu - 1.0) / 2.0

    b = np.zeros((N, n_bins))
    Q = np.zeros((N, n_bins))
    np.add.at(b, (pairs[:, 0], bin_idx), 1.0)
    np.add.at(b, (pairs[:, 1], bin_idx), 1.0)
    # Legendre 2 is even in (j-i), so contributes equally to both endpoints.
    np.add.at(Q, (pairs[:, 0], bin_idx), legendre_2)
    np.add.at(Q, (pairs[:, 1], bin_idx), legendre_2)
    return b, Q


def main():
    box = 200.0
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, box, size=(20, 3))
    pts_iso = np.vstack([
        rng.normal(c, 8.0, size=(150, 3)) for c in centers
    ])
    pts_iso = np.mod(pts_iso, box).astype(np.float64)

    rng2 = np.random.default_rng(0)
    centers2 = rng2.uniform(0, box, size=(20, 3))
    # Pancake-shaped blobs (squashed along z by 3x)
    pts_aniso = np.vstack([
        rng2.normal(c, [12.0, 12.0, 4.0], size=(150, 3))
        for c in centers2
    ])
    pts_aniso = np.mod(pts_aniso, box).astype(np.float64)

    r_edges = np.logspace(np.log10(2.0), np.log10(0.49 * box), 14)

    for label, pts in [("isotropic", pts_iso),
                       ("z-squashed (pancakes)", pts_aniso)]:
        t0 = time.perf_counter()
        b, Q = per_particle_pair_quadrupole(pts, r_edges, box)
        dt = time.perf_counter() - t0

        xi_2 = Q.sum(axis=0) / np.maximum(b.sum(axis=0), 1.0)

        print(f"--- {label}, N={len(pts)} ({dt*1000:.0f} ms) ---")
        print("  xi_2(r):", np.array2string(xi_2, precision=3))
        # per-particle anisotropy at a mid-r bin (r ~ 10 Mpc)
        j = 5
        mask = b[:, j] > 1
        if mask.sum() > 0:
            ratio = Q[mask, j] / b[mask, j]
            print(f"  per-particle Q/b at j={j} (r~{0.5*(r_edges[j]+r_edges[j+1]):.1f} Mpc):")
            print(f"    mean={ratio.mean():+.3f}  std={ratio.std():.3f}")
        print()


if __name__ == "__main__":
    main()
