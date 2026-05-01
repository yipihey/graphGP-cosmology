"""Per-particle pair tensor: trace-tidal split.

For each particle ``i`` and bin ``j``, compute the symmetric 3x3 tensor

    T_i^(j)_{ab} = sum_{k != i, r_ik in B_j}  n_ik^a * n_ik^b

with ``n_ik = (x_k - x_i) / |x_k - x_i|``.

Trace-tidal decomposition:
    Tr(T_i^(j))                  = b_i^(j)        scalar pair count (= density)
    T_i^(j) - Tr(T_i^(j))/3 * I  = traceless     local tidal/anisotropy
                                                  tensor at scale r_j

Eigendecomposition of the traceless part:
    eig pattern  (1 large +, 2 ~equal -)  -> filament (1D stretched)
    eig pattern  (1 large -, 2 ~equal +)  -> sheet/pancake (1D compressed)
    eig pattern  (~0, ~0, ~0)             -> isotropic / sphere (cluster core)

The construction is the per-particle, multi-scale, discrete-sample analog
of the smoothed-Hessian tidal tensor of Hahn+ (2007) and Forero-Romero+
(2009) -- but computed directly from pair counts in O(N * k_pair), no
density-field reconstruction required.

Identity (with isotropic clustering, periodic):
    < (T_i^(j) - Tr/3 I) >_i  -> 0   (no preferred axis on average)
"""

from __future__ import annotations

import time
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree


def per_particle_pair_tensor(
    positions: np.ndarray,
    r_edges: np.ndarray,
    box_size: float,
):
    """Per-particle pair tensor.

    Returns
    -------
    b : ``(N, n_bins)``
        Scalar pair count = ``Tr(T)``.
    T : ``(N, n_bins, 3, 3)``
        Sum of ``n_ik n_ik^T`` over partners in each bin.
    """
    N = len(positions)
    n_bins = len(r_edges) - 1
    r_max = float(r_edges[-1])
    tree = cKDTree(positions, boxsize=box_size)
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    b = np.zeros((N, n_bins))
    T = np.zeros((N, n_bins, 3, 3))
    if len(pairs) == 0:
        return b, T

    diff = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    diff -= box_size * np.round(diff / box_size)
    d = np.linalg.norm(diff, axis=1)
    bin_idx = np.searchsorted(r_edges, d, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    pairs = pairs[valid]
    diff = diff[valid]
    d = d[valid]
    bin_idx = bin_idx[valid]
    n = diff / d[:, None]
    nn = n[:, :, None] * n[:, None, :]

    np.add.at(b, (pairs[:, 0], bin_idx), 1.0)
    np.add.at(b, (pairs[:, 1], bin_idx), 1.0)
    np.add.at(T, (pairs[:, 0], bin_idx), nn)
    np.add.at(T, (pairs[:, 1], bin_idx), nn)
    return b, T


def trace_tidal_split(b: np.ndarray, T: np.ndarray):
    """Decompose per-particle pair tensor into trace and traceless parts.

    Returns
    -------
    delta_iso : ``(N, n_bins)``
        Trace-derived isotropic overdensity = b / E[b] - 1 (same as the
        scalar pair-count LISA primitive).
    T_traceless : ``(N, n_bins, 3, 3)``
        Normalized traceless tensor: ``T/b - (1/3) I``. Encodes the
        local tidal/anisotropy structure at scale r_j.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        T_norm = T / np.where(b[..., None, None] > 0, b[..., None, None], 1.0)
    eye = np.broadcast_to(np.eye(3), T_norm.shape)
    T_traceless = T_norm - (1.0 / 3.0) * eye
    return T_traceless


def classify_traceless(eigs: np.ndarray, eps: float = 0.02) -> str:
    """Classify a traceless eigenvalue triplet (sum = 0) by shape pattern.

    eigs are sorted ascending.

      filament : 1 large +,  2 ~equal -          (matter elongated along 1 axis)
      sheet    : 1 large -,  2 ~equal +          (matter compressed along 1 axis)
      triaxial : 3 distinct (+, ~0, -)
      isotropic: all eigs within ``eps`` of 0    (true cluster / void)
    """
    e = np.asarray(eigs)
    if np.max(np.abs(e)) < eps:
        return "isotropic"
    if e[2] > 0 and e[0] < 0 and abs(e[0] - e[1]) < eps:
        return "filament"
    if e[0] < 0 and e[2] > 0 and abs(e[1] - e[2]) < eps:
        return "sheet"
    return "triaxial"


def main():
    box = 200.0
    geometries = []
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, box, size=(20, 3))
    geometries.append(("isotropic", np.vstack([
        rng.normal(c, 8.0, size=(150, 3)) for c in centers
    ])))

    rng2 = np.random.default_rng(0)
    centers2 = rng2.uniform(0, box, size=(20, 3))
    geometries.append(("z-pancakes (sheet)", np.vstack([
        rng2.normal(c, [12.0, 12.0, 4.0], size=(150, 3)) for c in centers2
    ])))

    rng3 = np.random.default_rng(0)
    centers3 = rng3.uniform(0, box, size=(20, 3))
    geometries.append(("z-filaments", np.vstack([
        rng3.normal(c, [3.0, 3.0, 18.0], size=(150, 3)) for c in centers3
    ])))

    r_edges = np.logspace(np.log10(2.0), np.log10(0.49 * box), 14)
    j_test = 5

    for label, pts in geometries:
        pts = np.mod(pts, box).astype(np.float64)
        t0 = time.perf_counter()
        b, T = per_particle_pair_tensor(pts, r_edges, box)
        T_traceless = trace_tidal_split(b, T)
        dt = time.perf_counter() - t0

        # Per-particle eigendecomposition at j_test
        Tj = T_traceless[:, j_test]
        bj = b[:, j_test]
        keep = bj > 3
        eigs = np.linalg.eigvalsh(Tj[keep])  # ascending

        labels = [classify_traceless(e) for e in eigs]
        cnt = Counter(labels)

        # Mean traceless tensor (anisotropy averaged over particles)
        Tj_mean = Tj[keep].mean(axis=0)
        eigs_mean = np.linalg.eigvalsh(Tj_mean)

        print(f"--- {label}, N={len(pts)} ({dt:.2f}s) ---")
        print(f"  per-particle classification at r={0.5*(r_edges[j_test]+r_edges[j_test+1]):.1f} Mpc:")
        n = sum(cnt.values())
        for k in ("isotropic", "filament", "sheet", "triaxial"):
            v = cnt.get(k, 0)
            print(f"    {k:12s}: {v:5d}  ({100*v/n:5.1f}%)")
        print(f"  mean traceless eigenvalues (sum -> 0): {eigs_mean}")
        print()


if __name__ == "__main__":
    main()
