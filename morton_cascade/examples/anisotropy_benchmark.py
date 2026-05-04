#!/usr/bin/env python3
"""
Benchmark 7: Cell-wavelet axis-aligned anisotropy moments.

Demonstrates the cascade's native anisotropy observable Q_2:

  Q_2 = <w_z²> − ½(<w_x²> + <w_y²>)

where w_x, w_y, w_z are the axis-aligned Haar wavelet coefficients of the
density-contrast field at each dyadic level. The line-of-sight (LoS)
convention is z-axis (third coordinate).

For redshift-space distortions:
  - Kaiser regime (large scales): coherent infall enhances LoS density → Q_2 > 0
  - FoG regime (small scales): velocity dispersion smears LoS → Q_2 < 0
  - Isotropic (no RSD): Q_2 ≈ 0 within shot noise

This benchmark constructs three synthetic mocks:
  1. ISOTROPIC: clusters with sigma_x=sigma_y=sigma_z. Q_2 ≈ 0 expected.
  2. KAISER-LIKE: clusters tight in z, loose in x, y. Q_2 > 0 expected.
  3. FOG-LIKE: clusters elongated in z. Q_2 < 0 expected.

We run the cascade anisotropy CLI on each and verify the signs come out right.
We also time the run vs the standard field-stats run to confirm it's
practically free.

Usage:
    cd morton_cascade && cargo build --release
    python3 examples/anisotropy_benchmark.py
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

BIN = "./target/release/morton-cascade"
BOX_SIZE = 1000.0
N_DATA = 30_000
N_RAND = 150_000
N_CLUSTERS = 200
SEED = 42


def make_anisotropic_mock(label, sigmas, n=N_DATA, rng=None):
    """Build clustered mock with per-axis Gaussian cluster widths."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    centers = rng.uniform(0, BOX_SIZE, size=(N_CLUSTERS, 3))
    sx, sy, sz = sigmas
    pts = []
    while len(pts) < n:
        c = centers[rng.integers(0, N_CLUSTERS)]
        p = c + rng.normal(0.0, [sx, sy, sz], size=3)
        p = p % BOX_SIZE
        pts.append(p)
    return np.array(pts[:n], dtype=np.float64), label


def write_bin(path, pts):
    pts.astype('<f8').tofile(path)


def run_anisotropy(workdir, data_path, rand_path):
    out = os.path.join(workdir, 'aniso_out')
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    r = subprocess.run([BIN, 'anisotropy',
                        '-i', data_path, '--randoms', rand_path,
                        '-d', '3', '-L', str(BOX_SIZE),
                        '-o', out, '-q'],
                       capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print('FAIL:', r.stderr[:500])
        sys.exit(1)
    rows = []
    with open(os.path.join(out, 'field_anisotropy.csv')) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != 'level' and k != 'n_parents' else int(v))
                         for k, v in row.items()})
    return rows, elapsed


def run_field_stats(workdir, data_path, rand_path):
    """Time comparison: just standard field-stats (no anisotropy)."""
    out = os.path.join(workdir, 'fs_out')
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run([BIN, 'field-stats',
                    '-i', data_path, '--randoms', rand_path,
                    '-d', '3', '-L', str(BOX_SIZE),
                    '-o', out, '--hist-bins', '0', '-q'],
                   capture_output=True, text=True, check=True, timeout=120)
    return time.perf_counter() - t0


def main():
    if not os.path.exists(BIN):
        print(f"FAIL: binary {BIN} not found")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 7: Cell-wavelet axis-aligned anisotropy")
    print(f"  Box: {BOX_SIZE} h^-1 Mpc")
    print(f"  N_data: {N_DATA}, N_rand: {N_RAND}, N_clusters: {N_CLUSTERS}")
    print(f"  LoS: z-axis (cosmology convention)")
    print("=" * 78)

    workdir = tempfile.mkdtemp(prefix='aniso_bench_')
    rng = np.random.default_rng(SEED)

    try:
        # Three mocks: isotropic, Kaiser-like (z tight), FoG-like (z elongated)
        cases = [
            ("ISOTROPIC      ", [20.0, 20.0, 20.0]),
            ("KAISER-LIKE    ", [25.0, 25.0,  4.0]),  # tight in z
            ("FoG-LIKE       ", [10.0, 10.0, 50.0]),  # elongated in z
        ]

        # Common randoms (uniform)
        randoms = rng.uniform(0, BOX_SIZE, size=(N_RAND, 3))
        rand_path = os.path.join(workdir, 'r.bin')
        write_bin(rand_path, randoms)

        # Run field-stats once on first case to time it
        first_data, _ = make_anisotropic_mock(*cases[0], rng=rng)
        d0_path = os.path.join(workdir, 'd0.bin')
        write_bin(d0_path, first_data)
        # Warmup
        run_field_stats(workdir, d0_path, rand_path)
        fs_time = run_field_stats(workdir, d0_path, rand_path)
        print(f"\nReference: field-stats wall time = {fs_time:.3f}s\n")

        all_rows = {}
        for label, sigmas in cases:
            rng_case = np.random.default_rng(SEED)  # same seed for fair comparison
            data, _ = make_anisotropic_mock(label, sigmas, rng=rng_case)
            d_path = os.path.join(workdir, f'd_{label.strip()}.bin')
            write_bin(d_path, data)
            rows, t = run_anisotropy(workdir, d_path, rand_path)
            all_rows[label.strip()] = rows
            print(f"{label} sigmas=[{sigmas[0]:.0f},{sigmas[1]:.0f},{sigmas[2]:.0f}]: "
                  f"anisotropy {t:.3f}s ({t/fs_time:.2f}x field-stats time)")

        # Display Q_2 at every meaningful scale, side-by-side
        print(f"\n{'Cell [h^-1Mpc]':>15} | {'ISOTROPIC Q_2':>16} | {'KAISER Q_2':>14} | {'FoG Q_2':>14}")
        print("-" * 78)
        # All cases produce same number of levels with same cell_side
        for i in range(len(all_rows['ISOTROPIC'])):
            iso = all_rows['ISOTROPIC'][i]
            kai = all_rows['KAISER-LIKE'][i]
            fog = all_rows['FoG-LIKE'][i]
            cell = iso['cell_side_phys']
            n_par = iso['n_parents']
            if cell < 8 or cell > 500: continue
            if n_par < 30: continue
            print(f"{cell:>15.2f} | {iso['quadrupole_z']:>16.3e} "
                  f"| {kai['quadrupole_z']:>14.3e} "
                  f"| {fog['quadrupole_z']:>14.3e}")

        print(f"\n{'Cell [h^-1Mpc]':>15} | {'ISOTROPIC reduced Q_2':>22} | {'KAISER':>10} | {'FoG':>10}")
        print("-" * 78)
        for i in range(len(all_rows['ISOTROPIC'])):
            iso = all_rows['ISOTROPIC'][i]
            kai = all_rows['KAISER-LIKE'][i]
            fog = all_rows['FoG-LIKE'][i]
            cell = iso['cell_side_phys']
            n_par = iso['n_parents']
            if cell < 8 or cell > 500: continue
            if n_par < 30: continue
            print(f"{cell:>15.2f} | {iso['reduced_quadrupole_z']:>22.3f} "
                  f"| {kai['reduced_quadrupole_z']:>10.3f} "
                  f"| {fog['reduced_quadrupole_z']:>10.3f}")

        # Sanity check: verify expected signs
        print("\n--- Sanity: check signs at cluster-scale levels ---")
        anomalies = 0
        for i in range(len(all_rows['ISOTROPIC'])):
            iso = all_rows['ISOTROPIC'][i]
            kai = all_rows['KAISER-LIKE'][i]
            fog = all_rows['FoG-LIKE'][i]
            cell = iso['cell_side_phys']
            n_par = iso['n_parents']
            # Use scales comparable to or larger than largest sigma (50)
            if cell < 30 or cell > 250: continue
            if n_par < 100: continue
            kai_q = kai['reduced_quadrupole_z']
            fog_q = fog['reduced_quadrupole_z']
            kai_ok = kai_q > 0.1
            fog_ok = fog_q < -0.1
            tag_kai = "✓" if kai_ok else "✗"
            tag_fog = "✓" if fog_ok else "✗"
            print(f"  cell={cell:7.2f}: KAISER reduced_Q_2={kai_q:+.3f} {tag_kai}, "
                  f"FoG reduced_Q_2={fog_q:+.3f} {tag_fog}")
            if not kai_ok: anomalies += 1
            if not fog_ok: anomalies += 1
        if anomalies == 0:
            print("\n  ✓ All sign expectations met. Q_2 reliably distinguishes")
            print("    Kaiser-like (positive) from FoG-like (negative) anisotropy.")
        else:
            print(f"\n  WARN: {anomalies} unexpected signs at cluster scales.")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
