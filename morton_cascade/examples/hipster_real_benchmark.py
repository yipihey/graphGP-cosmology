#!/usr/bin/env python3
"""
Benchmark 4: cascade vs REAL HIPSTER (Philcox & Eisenstein 2019, the actual
C++ binary from oliverphilcox/HIPSTER).

This replaces the Python-equivalent comparison in monopole_benchmark.py with
the real production code. Both run on identical mock catalogs in periodic
boxes with the same R0 truncation and box size.

The comparison:
- Same N values, same Cox-process clustered mock.
- HIPSTER: monopole P(k) over user-defined k-bins (10 log-spaced, 0.05-1 h/Mpc).
- Cascade: pair counts and Landy-Szalay xi at all dyadic scales.

We measure WALL TIME including IO. We do NOT compare numerical values directly
since cascade's output is at dyadic cube scales and HIPSTER's is at radial k-bins.
What matters for the scaling claim is the ratio of wall times at varying N.

Setup:
1. HIPSTER must be built at /tmp/HIPSTER/power
2. Run with -DPERIODIC. Auto-detects box size from data bounds.
3. Bin file must be tab-separated.

Usage:
    cd morton_cascade && cargo build --release
    python3 examples/hipster_real_benchmark.py
"""

import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# Configuration
BOX_SIZE = 1000.0
R0 = 100.0
N_VALUES = [5_000, 20_000, 80_000, 200_000]
SEED = 271828
HIPSTER_BIN = "/tmp/HIPSTER/power"
CASCADE_BIN = "./target/release/morton-cascade"


def make_clustered_mock(n_target, box_size, rng, n_parents=200, sigma_child=20.0):
    """Cox process: clustered mock with non-trivial 2pt and 3pt signal."""
    parents = rng.uniform(0, box_size, size=(n_parents, 3))
    pts = []
    while len(pts) < n_target:
        p = parents[rng.integers(0, n_parents)]
        c = p + rng.normal(0.0, sigma_child, size=3)
        c = c % box_size
        pts.append(c)
    return np.array(pts[:n_target], dtype=np.float64)


def write_hipster_input(path, pts):
    """HIPSTER input: x y z w (space-separated) with weights = 1."""
    weights = np.ones(len(pts))
    out = np.column_stack([pts, weights])
    np.savetxt(path, out, fmt='%.6f')


def write_cascade_input(path, pts):
    pts.astype('<f8').tofile(path)


def write_kbin_file(path, n_bins=10, k_lo=0.05, k_hi=1.0):
    edges = np.geomspace(k_lo, k_hi, n_bins + 1)
    with open(path, 'w') as f:
        for lo, hi in zip(edges[:-1], edges[1:]):
            f.write(f'{lo:.6f}\t{hi:.6f}\n')
    return n_bins


def run_hipster(data_path, kbin_path, out_dir, R0):
    t0 = time.perf_counter()
    r = subprocess.run([HIPSTER_BIN, '-in', data_path, '-binfile', kbin_path,
                        '-output', out_dir, '-out_string', 'bench',
                        '-perbox', '-R0', str(R0), '-max_l', '0',
                        '-nthread', '4'],
                       capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print('HIPSTER failed:')
        print(r.stderr[:1000])
        return None, elapsed
    return r.stdout, elapsed


def run_cascade(data_path, randoms_path, out_dir, box_size):
    """Run cascade producing field-stats (gives many statistics in one pass)."""
    t0 = time.perf_counter()
    subprocess.run([CASCADE_BIN, 'field-stats',
                    '-i', data_path, '--randoms', randoms_path,
                    '-d', '3', '-L', str(box_size),
                    '-o', out_dir, '--hist-bins', '0', '-q'],
                   capture_output=True, text=True, check=True, timeout=600)
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    if not os.path.exists(HIPSTER_BIN):
        print(f"FAIL: HIPSTER binary at {HIPSTER_BIN} not found")
        sys.exit(1)
    if not os.path.exists(CASCADE_BIN):
        print(f"FAIL: cascade binary at {CASCADE_BIN} not found")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 4: cascade vs REAL HIPSTER (Philcox-Eisenstein 2019)")
    print(f"  Box: {BOX_SIZE} h^-1 Mpc, R0: {R0} h^-1 Mpc")
    print(f"  HIPSTER: monopole P(k) over 10 log-spaced k-bins (0.05-1 h/Mpc)")
    print(f"  Cascade: ALL dyadic scales of moments + pair counts + PDF in 1 pass")
    print("=" * 78)

    workdir = tempfile.mkdtemp(prefix="hipster_real_")
    rng = np.random.default_rng(SEED)
    kbin_path = os.path.join(workdir, 'kbins.tsv')
    write_kbin_file(kbin_path)

    results = []
    try:
        for N in N_VALUES:
            print(f"\n--- N = {N} ---")
            # Generate same data both sides see
            data = make_clustered_mock(N, BOX_SIZE, rng)

            # HIPSTER input
            hipster_data = os.path.join(workdir, f"data_{N}.txt")
            write_hipster_input(hipster_data, data)

            # Cascade input (binary). For field-stats we also need randoms;
            # use 5x as before.
            cas_data = os.path.join(workdir, f"data_{N}.bin")
            cas_rand = os.path.join(workdir, f"rand_{N}.bin")
            write_cascade_input(cas_data, data)
            randoms = rng.uniform(0, BOX_SIZE, size=(5 * N, 3))
            write_cascade_input(cas_rand, randoms)

            # Run HIPSTER
            hip_out = os.path.join(workdir, f'hip_out_{N}')
            os.makedirs(hip_out, exist_ok=True)
            _, hip_time = run_hipster(hipster_data, kbin_path, hip_out, R0)

            # Run cascade
            cas_out = os.path.join(workdir, f'cas_out_{N}')
            os.makedirs(cas_out, exist_ok=True)
            cas_time = run_cascade(cas_data, cas_rand, cas_out, BOX_SIZE)

            speedup = hip_time / cas_time
            print(f"  HIPSTER C++:  {hip_time:7.3f}s  (monopole only, 10 k-bins)")
            print(f"  Cascade Rust: {cas_time:7.3f}s  (ALL moments at 33 dyadic scales)")
            print(f"  Speedup:      {speedup:5.1f}x")
            results.append({'N': N, 'hipster': hip_time, 'cascade': cas_time,
                            'speedup': speedup})

        # Scaling exponent fit
        print("\n" + "=" * 78)
        print(f"{'N':>8} | {'HIPSTER [s]':>12} | {'Cascade [s]':>12} | {'Speedup':>10} | scaling")
        print("-" * 78)
        for i, r in enumerate(results):
            line = (f"{r['N']:>8} | {r['hipster']:>12.3f} | {r['cascade']:>12.3f} "
                    f"| {r['speedup']:>9.1f}x")
            if i > 0:
                p = results[i-1]
                hip_exp = np.log(r['hipster']/p['hipster']) / np.log(r['N']/p['N'])
                cas_exp = (np.log(r['cascade']/p['cascade']) / np.log(r['N']/p['N'])
                           if p['cascade'] > 0 else float('nan'))
                line += f" | hip {hip_exp:5.2f}, cas {cas_exp:5.2f}"
            print(line)

        # Power-law fits
        Ns = np.array([r['N'] for r in results])
        hips = np.array([r['hipster'] for r in results])
        cass = np.array([r['cascade'] for r in results])

        b_hip, a_hip = np.polyfit(np.log(Ns), np.log(hips), 1)
        b_cas, a_cas = np.polyfit(np.log(Ns), np.log(cass), 1)
        print(f"\n  Power-law fits over the measured range:")
        print(f"  HIPSTER:  t ≈ {np.exp(a_hip):.3e} · N^{b_hip:.2f}")
        print(f"  Cascade:  t ≈ {np.exp(a_cas):.3e} · N^{b_cas:.2f}")

        for N_target in [1e5, 1e6, 1e7]:
            t_hip = np.exp(a_hip) * N_target ** b_hip
            t_cas = np.exp(a_cas) * N_target ** b_cas
            print(f"  N = {N_target:.0e}:  HIPSTER ≈ {t_hip:.2e}s, "
                  f"cascade ≈ {t_cas:.2e}s, speedup = {t_hip/t_cas:.1e}x")

    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
