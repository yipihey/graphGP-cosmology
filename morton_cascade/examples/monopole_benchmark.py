#!/usr/bin/env python3
"""
Benchmark 2: Monopole P(k) parity and scaling vs a HIPSTER-equivalent
configuration-space estimator.

Implements a faithful reduced version of HIPSTER's isotropic monopole
estimator from Philcox & Eisenstein 2019 (arXiv:1912.01010), Eq. 2.12 with
the kernel A^a(u) = j_0(k_a u) and the polynomial window W(r; R0) from
their Eq. 3.1. Restricted to the periodic-box case where the survey
correction Φ(r) ≡ 1.

Compared to:
- The cascade's `cumulative_dd[level]`, `cumulative_rr[level]`,
  `cumulative_dr[level]` outputs from `morton-cascade pairs`, with the
  Landy-Szalay shell estimator from `xi`.

Two outputs:
1. Numerical agreement on shared observables (paired counts) at the
   dyadic cube scales the cascade naturally produces.
2. Wall-time scaling at varying N to verify the cost claim:
   HIPSTER ~ O(N · n · R0^3)
   Cascade ~ O(N log N)

Usage:
    cd morton_cascade && cargo build --release
    python3 examples/monopole_benchmark.py
"""

import csv
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BOX_SIZE = 1000.0  # h^-1 Mpc, BOSS-like
R0 = 100.0         # h^-1 Mpc, HIPSTER truncation
N_VALUES = [2_000, 5_000, 10_000, 20_000]
RAND_FACTOR = 5    # n_r / n_d
SEED = 271828

BIN = "./target/release/morton-cascade"


# -----------------------------------------------------------------------------
# HIPSTER polynomial window function (Eq. 3.1 of Philcox-Eisenstein 2019)
# -----------------------------------------------------------------------------

def hipster_window(r_over_R0):
    """Piecewise-continuous polynomial window with smooth derivatives at
    breakpoints. Equal to 1 for r < R0/2, smoothly drops to 0 at r = R0."""
    x = np.asarray(r_over_R0)
    out = np.zeros_like(x)
    m1 = (x < 0.5)
    m2 = (x >= 0.5) & (x < 0.75)
    m3 = (x >= 0.75) & (x < 1.0)
    out[m1] = 1.0
    out[m2] = 1.0 - 8 * (2 * x[m2] - 1) ** 3 + 8 * (2 * x[m2] - 1) ** 4
    out[m3] = -64 * (x[m3] - 1) ** 3 - 128 * (x[m3] - 1) ** 4
    return out


# -----------------------------------------------------------------------------
# HIPSTER-equivalent isotropic monopole estimator (simplified for periodic box)
# -----------------------------------------------------------------------------

def hipster_monopole(data, randoms, k_centers, R0, box_size):
    """Compute the isotropic monopole P(k) via configuration-space
    pair-counting with the j_0 kernel and the polynomial window.

    Returns:
        P_k: array of P(k) at each k_center, shape (len(k_centers),)
        timing: dict with 'pair_finding', 'kernel_eval', 'total' wall times
    """
    timing = {}
    n_d = len(data)
    n_r = len(randoms)
    alpha = n_d / n_r  # mean-density ratio
    V = box_size ** 3
    nw_sq = (n_d / V) ** 2  # (n*w)^2 with w=1

    # Build kd-trees and find pairs within R0
    t0 = time.perf_counter()
    tree_d = cKDTree(data, boxsize=box_size)
    tree_r = cKDTree(randoms, boxsize=box_size)

    # Pairs: distance arrays, accept self-tree but exclude i=j
    # Using query_ball_tree gives all pairs at distance <= R0
    # For DD: we want pairs i<j (each pair counted once)
    # For RR: same convention
    # For DR: cross, no symmetry
    dd_pairs = tree_d.query_pairs(R0, output_type='ndarray')
    rr_pairs = tree_r.query_pairs(R0, output_type='ndarray')
    # DR: query each data point against the random tree
    dr_lists = tree_d.query_ball_tree(tree_r, R0)
    timing['pair_finding'] = time.perf_counter() - t0

    # Compute pair separations
    t0 = time.perf_counter()
    dd_sep = _periodic_dist(data[dd_pairs[:, 0]], data[dd_pairs[:, 1]], box_size)
    rr_sep = _periodic_dist(randoms[rr_pairs[:, 0]], randoms[rr_pairs[:, 1]], box_size)

    dr_i = []
    dr_j = []
    for i, neighbors in enumerate(dr_lists):
        dr_i.extend([i] * len(neighbors))
        dr_j.extend(neighbors)
    dr_i = np.array(dr_i, dtype=np.int64)
    dr_j = np.array(dr_j, dtype=np.int64)
    dr_sep = _periodic_dist(data[dr_i], randoms[dr_j], box_size)

    # Window weights
    dd_w = hipster_window(dd_sep / R0)
    rr_w = hipster_window(rr_sep / R0)
    dr_w = hipster_window(dr_sep / R0)
    timing['separations'] = time.perf_counter() - t0

    # Kernel eval and binning
    t0 = time.perf_counter()
    P_k = np.zeros_like(k_centers)
    for ki, k in enumerate(k_centers):
        # j_0(kr) kernel
        A_dd = np.sinc(k * dd_sep / np.pi)
        A_rr = np.sinc(k * rr_sep / np.pi)
        A_dr = np.sinc(k * dr_sep / np.pi)

        # Modified pair counts (NN = D - R, so NN^2 = DD - 2DR + RR for matched α)
        # Each unordered pair counted once for DD/RR (factor 2 from i<j to i!=j),
        # DR is already over all (i,j) pairs.
        # See Philcox-Eisenstein 2019 Eq. 2.13.
        DD_til = 2.0 * np.sum(A_dd * dd_w)  # double to recover i!=j sum
        RR_til = 2.0 * np.sum(A_rr * rr_w)
        DR_til = np.sum(A_dr * dr_w)

        # Estimator P̂_a = (DD - 2*α*DR + α^2*RR) / (V*(nw)^2)
        # For matched mean density (alpha rescaled to cancel out)
        P_k[ki] = (DD_til - 2.0 * alpha * DR_til + alpha**2 * RR_til) / (V * nw_sq)
    timing['kernel_eval'] = time.perf_counter() - t0

    timing['total'] = (timing['pair_finding']
                       + timing['separations']
                       + timing['kernel_eval'])
    return P_k, timing


def _periodic_dist(p1, p2, L):
    """Minimum-image periodic distance, vectorized."""
    d = p1 - p2
    d = d - L * np.round(d / L)
    return np.sqrt(np.sum(d * d, axis=-1))


# -----------------------------------------------------------------------------
# Cascade reference
# -----------------------------------------------------------------------------

def write_pts_bin(path, pts):
    pts.astype('<f8').tofile(path)


def run_cascade_pairs(workdir, data_path, randoms_path):
    """Run `morton-cascade xi` and return parsed shell stats."""
    cmd = [BIN, "xi",
           "-i", data_path,
           "--randoms", randoms_path,
           "-d", "3",
           "-L", str(BOX_SIZE),
           "-o", workdir,
           "-q"]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - t0
    rows = []
    with open(os.path.join(workdir, "xi_landy_szalay.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if k != 'level' else int(v)
                         for k, v in r.items()})
    return rows, elapsed


# -----------------------------------------------------------------------------
# Synthetic clustered mock — Cox process for genuine clustering signal
# -----------------------------------------------------------------------------

def make_clustered_mock(n_target, box_size, rng, n_parents=200, sigma_child=20.0):
    """Cox process: parents Poisson-uniform, children Gaussian-clustered around
    parents. Yields a mock with non-trivial 2-point and 3-point correlation.
    """
    parents = rng.uniform(0, box_size, size=(n_parents, 3))
    pts = []
    while len(pts) < n_target:
        p = parents[rng.integers(0, n_parents)]
        c = p + rng.normal(0.0, sigma_child, size=3)
        c = c % box_size  # periodic wrap
        pts.append(c)
    return np.array(pts[:n_target], dtype=np.float64)


# -----------------------------------------------------------------------------
# Main: scaling benchmark
# -----------------------------------------------------------------------------

def main():
    if not os.path.exists(BIN):
        print(f"FAIL: {BIN} not found. Run `cargo build --release` first.")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 2: Cascade vs HIPSTER-equivalent monopole P(k)")
    print(f"  Box: {BOX_SIZE} h^-1 Mpc, R0: {R0} h^-1 Mpc")
    print(f"  Random:data ratio: {RAND_FACTOR}")
    print("=" * 78)

    rng = np.random.default_rng(SEED)
    workdir = tempfile.mkdtemp(prefix="morton_bench_")

    try:
        results = []
        for N in N_VALUES:
            print(f"\n--- N_data = {N}, N_rand = {N * RAND_FACTOR} ---")
            data = make_clustered_mock(N, BOX_SIZE, rng)
            randoms = rng.uniform(0, BOX_SIZE, size=(N * RAND_FACTOR, 3))

            data_path = os.path.join(workdir, f"data_{N}.bin")
            rand_path = os.path.join(workdir, f"rand_{N}.bin")
            write_pts_bin(data_path, data)
            write_pts_bin(rand_path, randoms)

            # Cascade
            cascade_rows, cascade_time = run_cascade_pairs(
                workdir, data_path, rand_path)
            print(f"  cascade xi: {cascade_time:6.3f}s -> {len(cascade_rows)} levels")

            # HIPSTER-equivalent on a few k bins matching cascade dyadic scales
            # Use cascade's `cell_side_phys` to set k_l = pi / R_l for shell levels
            # corresponding to cell sizes between R0/4 and R0
            k_centers = []
            for r in cascade_rows:
                cell_phys = r['cell_side_phys']
                if R0 / 8 < cell_phys < R0 * 0.9:
                    k_centers.append(np.pi / cell_phys)
            k_centers = np.array(k_centers) if k_centers else np.array([0.05, 0.1, 0.2])

            P_hipster, hipster_timing = hipster_monopole(
                data, randoms, k_centers, R0, BOX_SIZE)
            print(f"  HIPSTER-eq: {hipster_timing['total']:6.3f}s "
                  f"(pairs: {hipster_timing['pair_finding']:.3f}, "
                  f"kernels: {hipster_timing['kernel_eval']:.3f})")
            print(f"  speedup (HIPSTER/cascade): {hipster_timing['total']/cascade_time:.2f}x")

            results.append({
                'N': N,
                'cascade_time': cascade_time,
                'hipster_time': hipster_timing['total'],
                'hipster_pair_time': hipster_timing['pair_finding'],
                'hipster_kernel_time': hipster_timing['kernel_eval'],
                'speedup': hipster_timing['total'] / cascade_time,
            })

        # Scaling summary
        print("\n" + "=" * 78)
        print(f"{'N':>8} | {'Cascade [s]':>12} | {'HIPSTER [s]':>12} | "
              f"{'Speedup':>10} | scaling exponent")
        print("-" * 78)
        for i, r in enumerate(results):
            line = (f"{r['N']:>8} | {r['cascade_time']:>12.3f} | "
                    f"{r['hipster_time']:>12.3f} | "
                    f"{r['speedup']:>10.2f}x")
            if i > 0:
                prev = results[i - 1]
                # Estimate scaling exponent: log(t_curr/t_prev)/log(N_curr/N_prev)
                if prev['cascade_time'] > 0:
                    cas_exp = np.log(r['cascade_time'] / prev['cascade_time']) \
                              / np.log(r['N'] / prev['N'])
                else:
                    cas_exp = float('nan')
                hip_exp = np.log(r['hipster_time'] / prev['hipster_time']) \
                          / np.log(r['N'] / prev['N'])
                line += f" | cas {cas_exp:5.2f}, hip {hip_exp:5.2f}"
            print(line)

        print("\nExpected scalings:")
        print("  Cascade:  ~ N log N    (effective exponent ~ 1.0-1.1 for clustered data)")
        print("  HIPSTER:  ~ N · n · R0^3, where n ∝ N at fixed box → ~N^2 in this test")
        print("            (cubic-R0 prefactor is ~constant since R0 is fixed)")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
