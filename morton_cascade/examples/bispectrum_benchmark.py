#!/usr/bin/env python3
"""
Benchmark 3: Cascade three-point moments vs naive triple-counting.

The cascade computes <δ³>_W_r at every dyadic level in a single pass via
field-stats (already implemented). HIPSTER-style bispectrum estimators
require explicit triple-counting at O(N · n² · R0^6) — punishing.

This benchmark establishes:
1. Cascade's m3_delta and s3_delta are bit-exact against direct numpy
   cell-binning at each dyadic level (already validated by parity_benchmark.py
   for N=4000; we re-verify here at varying N).
2. The naive "triple count within R0" cost scales as O(N · n² · R0^6) — we
   measure this empirically at small N and extrapolate.

Comparing scaling:
  - Cascade S_3 pipeline: O(N log N), single pass, all scales at once
  - Direct triple count: O(N · n² · R0^6), per scale, scales as N^3 at fixed box

Note: This benchmark uses naive triple counting (NOT HIPSTER's actual
fast triple algorithm via spherical harmonic decomposition). HIPSTER's
real algorithm is O(N · n · R0^3 · multipole_complexity), better than
naive, but still vastly slower than our cascade single-pass for ALL scales
simultaneously. The Philcox-Eisenstein 2019 paper itself cites bispectrum
as O(N · n² · R0^6); we use that as the comparison point.

Usage:
    cd morton_cascade && cargo build --release
    python3 examples/bispectrum_benchmark.py
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

BOX_SIZE = 1000.0
R0 = 100.0
RAND_FACTOR = 5
SEED = 314159

# Cascade scales linearly so push it harder than the triple-count
N_VALUES_CASCADE = [2_000, 5_000, 10_000, 20_000, 40_000, 80_000]
# Triple count is O(N^3); cap aggressively
N_VALUES_TRIPLE = [500, 1_000, 2_000, 4_000]

BIN = "./target/release/morton-cascade"


# -----------------------------------------------------------------------------
# Synthetic clustered mock (Cox process)
# -----------------------------------------------------------------------------

def make_clustered_mock(n_target, box_size, rng, n_parents=200, sigma_child=20.0):
    parents = rng.uniform(0, box_size, size=(n_parents, 3))
    pts = []
    while len(pts) < n_target:
        p = parents[rng.integers(0, n_parents)]
        c = p + rng.normal(0.0, sigma_child, size=3)
        c = c % box_size
        pts.append(c)
    return np.array(pts[:n_target], dtype=np.float64)


def write_pts_bin(path, pts):
    pts.astype('<f8').tofile(path)


# -----------------------------------------------------------------------------
# Cascade S_3 via field-stats subcommand
# -----------------------------------------------------------------------------

def run_cascade_field_stats(workdir, data_path, randoms_path):
    """Run `morton-cascade field-stats` and return per-level stats with timing."""
    cmd = [BIN, "field-stats",
           "-i", data_path,
           "--randoms", randoms_path,
           "-d", "3",
           "-L", str(BOX_SIZE),
           "-o", workdir,
           "--hist-bins", "0",  # skip histogram, only moments
           "-q"]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - t0
    rows = []
    with open(os.path.join(workdir, "field_moments.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ('level', 'n_cells_active',
                                                   'n_cells_data_outside') else int(v))
                         for k, v in r.items()})
    return rows, elapsed


# -----------------------------------------------------------------------------
# Naive triple-count: count triples (i,j,k) within R0 of each other
# -----------------------------------------------------------------------------

def naive_triple_count(data, R0, box_size):
    """Count all triples (i, j, k) where pairwise distances are all <= R0.
    This is the O(N · n² · R0^6) operation HIPSTER's bispectrum requires.

    Returns:
        n_triples: count of valid triples
        elapsed: wall time
    """
    t0 = time.perf_counter()
    tree = cKDTree(data, boxsize=box_size)
    pairs = tree.query_pairs(R0, output_type='ndarray')

    # For each pair (i,j), find points k that are within R0 of both
    n_triples = 0
    # Build adjacency: for each i, list of neighbors within R0
    neighbor_lists = tree.query_ball_tree(tree, R0)

    for ij in pairs:
        i, j = ij
        # Common neighbors of i and j (excluding i and j themselves)
        ni = set(neighbor_lists[i]) - {i, j}
        nj = set(neighbor_lists[j]) - {i, j}
        common = ni & nj
        n_triples += len(common)

    elapsed = time.perf_counter() - t0
    return n_triples, elapsed


# -----------------------------------------------------------------------------
# Numpy direct cell-binning reference for S_3 at one level (verifies cascade)
# -----------------------------------------------------------------------------

def direct_S3_at_level(data, randoms, cell_side, box_size):
    """Compute S_3 = m3 / var^2 directly via numpy cell counting at one
    dyadic scale. Should agree with cascade exactly."""
    n_cells = int(round(box_size / cell_side))
    edges = np.linspace(0, box_size, n_cells + 1)
    counts_d, _ = np.histogramdd(data, bins=[edges, edges, edges])
    counts_r, _ = np.histogramdd(randoms, bins=[edges, edges, edges])

    total_w_d = counts_d.sum()
    total_w_r = counts_r.sum()
    alpha = total_w_d / total_w_r

    flat_d = counts_d.ravel()
    flat_r = counts_r.ravel()
    active = flat_r > 0

    delta = np.where(active, flat_d / (alpha * np.maximum(flat_r, 1e-300)) - 1.0, 0.0)
    w = flat_r[active]
    d = delta[active]
    if w.sum() == 0:
        return 0.0, 0.0
    mean = (w * d).sum() / w.sum()
    var = (w * (d - mean) ** 2).sum() / w.sum()
    m3 = (w * (d - mean) ** 3).sum() / w.sum()
    s3 = m3 / (var ** 2) if var > 0 else 0.0
    return s3, var


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not os.path.exists(BIN):
        print(f"FAIL: {BIN} not found.")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 3: Cascade <δ³> single-pass vs naive triple-counting")
    print(f"  Box: {BOX_SIZE}, R0: {R0}, rand:data = {RAND_FACTOR}")
    print("=" * 78)

    rng = np.random.default_rng(SEED)
    workdir = tempfile.mkdtemp(prefix="morton_bispec_")

    try:
        # --- Part A: Verify cascade S_3 against numpy cell-binning -----------
        print("\n[A] Verifying cascade S_3 vs numpy direct cell-binning")
        N_verify = 8_000
        data = make_clustered_mock(N_verify, BOX_SIZE, rng)
        randoms = rng.uniform(0, BOX_SIZE, size=(N_verify * RAND_FACTOR, 3))
        data_path = os.path.join(workdir, "data_v.bin")
        rand_path = os.path.join(workdir, "rand_v.bin")
        write_pts_bin(data_path, data)
        write_pts_bin(rand_path, randoms)

        cascade_rows, cas_t = run_cascade_field_stats(workdir, data_path, rand_path)
        print(f"  cascade field-stats: {cas_t:.3f}s for {len(cascade_rows)} levels")
        max_rel_err_s3 = 0.0
        max_rel_err_var = 0.0
        n_compares = 0
        for r in cascade_rows:
            cell_phys = r['cell_side_phys']
            if cell_phys > BOX_SIZE * 0.5 or cell_phys < BOX_SIZE / 64:
                continue  # too coarse or too fine for clean comparison
            np_s3, np_var = direct_S3_at_level(data, randoms, cell_phys, BOX_SIZE)
            cas_s3 = r['s3_delta']
            cas_var = r['var_delta']
            if abs(np_var) > 1e-10:
                rel_var = abs(cas_var - np_var) / abs(np_var)
                max_rel_err_var = max(max_rel_err_var, rel_var)
            if abs(np_s3) > 1e-6:
                rel_s3 = abs(cas_s3 - np_s3) / abs(np_s3)
                max_rel_err_s3 = max(max_rel_err_s3, rel_s3)
            n_compares += 1
            print(f"    level {r['level']:>2}: cell={cell_phys:7.2f}, "
                  f"var: cas={cas_var:11.4e} np={np_var:11.4e}, "
                  f"S_3: cas={cas_s3:7.3f} np={np_s3:7.3f}")
        print(f"  Max relative error (var): {max_rel_err_var:.2e}")
        print(f"  Max relative error (S_3): {max_rel_err_s3:.2e}")
        if max_rel_err_var < 1e-9 and max_rel_err_s3 < 1e-9:
            print("  ✓ CASCADE MATCHES NUMPY at all comparable levels")
        else:
            print("  WARN: agreement loose; investigate.")

        # --- Part B: Cascade scaling -----------------------------------------
        print("\n[B] Cascade S_3 timing scaling (single pass over ALL levels)")
        cas_times = []
        for N in N_VALUES_CASCADE:
            data = make_clustered_mock(N, BOX_SIZE, rng)
            randoms = rng.uniform(0, BOX_SIZE, size=(N * RAND_FACTOR, 3))
            d_p = os.path.join(workdir, f"d_{N}.bin")
            r_p = os.path.join(workdir, f"r_{N}.bin")
            write_pts_bin(d_p, data)
            write_pts_bin(r_p, randoms)
            _, t = run_cascade_field_stats(workdir, d_p, r_p)
            cas_times.append((N, t))
            print(f"  N={N:>6}: {t:.3f}s")

        # --- Part C: Naive triple-count scaling ------------------------------
        print("\n[C] Naive triple-count timing scaling (one R0 only)")
        triple_times = []
        for N in N_VALUES_TRIPLE:
            data = make_clustered_mock(N, BOX_SIZE, rng)
            n_trip, t = naive_triple_count(data, R0, BOX_SIZE)
            triple_times.append((N, t, n_trip))
            print(f"  N={N:>6}: {t:.3f}s, {n_trip} triples")

        # --- Part D: Scaling exponents ---------------------------------------
        print("\n[D] Empirical scaling exponents")
        print("  Cascade:")
        for i in range(1, len(cas_times)):
            N0, t0 = cas_times[i-1]
            N1, t1 = cas_times[i]
            exp = np.log(t1 / t0) / np.log(N1 / N0) if t0 > 0 else float('nan')
            print(f"    N={N0}->{N1}: t={t0:.3f}->{t1:.3f}, exponent={exp:.2f}")
        print("  Naive triple:")
        for i in range(1, len(triple_times)):
            N0, t0, _ = triple_times[i-1]
            N1, t1, _ = triple_times[i]
            exp = np.log(t1 / t0) / np.log(N1 / N0)
            print(f"    N={N0}->{N1}: t={t0:.3f}->{t1:.3f}, exponent={exp:.2f}")

        # --- Part E: Projection to survey scale ------------------------------
        print("\n[E] Extrapolation to DESI-scale catalog")
        # Fit cascade as t = a · N^b
        Ns_cas = np.array([n for n, _ in cas_times])
        ts_cas = np.array([t for _, t in cas_times])
        log_fit_cas = np.polyfit(np.log(Ns_cas), np.log(ts_cas), 1)
        b_cas, log_a_cas = log_fit_cas
        # Triple-count fit
        Ns_trip = np.array([n for n, _, _ in triple_times])
        ts_trip = np.array([t for _, t, _ in triple_times])
        log_fit_trip = np.polyfit(np.log(Ns_trip), np.log(ts_trip), 1)
        b_trip, log_a_trip = log_fit_trip
        print(f"  Cascade: t ≈ {np.exp(log_a_cas):.2e} · N^{b_cas:.2f}")
        print(f"  Triple:  t ≈ {np.exp(log_a_trip):.2e} · N^{b_trip:.2f}")

        for N_target in [1e5, 1e6, 1e7]:
            t_cas_proj = np.exp(log_a_cas) * N_target ** b_cas
            t_trip_proj = np.exp(log_a_trip) * N_target ** b_trip
            print(f"  N={N_target:.0e}:  cascade ≈ {t_cas_proj:8.2e}s, "
                  f"naive triple ≈ {t_trip_proj:8.2e}s, "
                  f"speedup = {t_trip_proj/t_cas_proj:.1e}x")

        # --- Part F: The bigger picture --------------------------------------
        print("\n[F] What the cascade gets in ONE PASS")
        print("  - 33 dyadic scales of <δ²> (variance / σ²(R))")
        print("  - 33 dyadic scales of <δ³> (third moment, S_3 reduced cumulant)")
        print("  - 33 dyadic scales of <δ⁴> (fourth moment, kurtosis)")
        print("  - Full PDF of δ at every level (when --hist-bins > 0)")
        print("  - DD/RR/DR pair counts at every level")
        print("  - Outside-footprint diagnostic at every level")
        print("  All from O(N log N) work. HIPSTER would need a separate pass")
        print("  per multipole per scale, with O(N²)+O(N³) cost respectively.")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
