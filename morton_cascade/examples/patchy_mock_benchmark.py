#!/usr/bin/env python3
"""
Benchmark 6: Patchy-style mock with non-trivial survey geometry.

Real surveys (BOSS, DESI, eBOSS) are not periodic boxes. The selection
function is encoded by the random catalog, with non-uniform density due to
fiber completeness, imaging systematics, dust correction, redshift selection.
This benchmark exercises that machinery with a realistic survey-like geometry:
a slab+footprint mock where data lives only inside a complex region.

What we exercise:
1. Cascade native footprint handling via random-catalog cell counts. The
   field-stats output reports n_cells_data_outside as a diagnostic.
2. Wall time scaling at varying N for a non-trivial-geometry survey.

Note: HIPSTER does NOT run on this case since we're not in periodic mode.
The non-periodic HIPSTER configuration requires a separately-fit Phi(r,mu)
correction function (their Eq. 2.5) computed from exhaustive RR pair counts
via Corrfunc, which is itself O(N²) work — and is exactly what HIPSTER
needs to avoid quadratic cost in production. Cascade gets footprint handling
natively from the random-catalog cell counts in the cascade itself, no
separate Phi fit step needed.

So this benchmark establishes:
- Cascade scales linearly even with non-trivial geometry.
- Cascade produces meaningful output that flags catalog-quality issues
  (n_cells_data_outside) at all dyadic scales.

Setup: clustered data inside an octant of the box, plus a small
contamination cluster outside the octant to test the diagnostic.
"""

import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

BOX_SIZE = 1000.0
SEED = 271828
CASCADE_BIN = "./target/release/morton-cascade"
N_VALUES = [5_000, 10_000, 20_000, 40_000, 80_000]


def make_survey_mock(n_target, box_size, rng, n_clusters=200, sigma=15.0):
    """Survey mock: data clustered in an upper-half-Z slab + an octant
    footprint; small contamination cluster placed deliberately OUTSIDE the
    footprint to test the diagnostic.

    Geometry:
        - Footprint: 0.5L < z < L, x and y unrestricted.
        - 95% of data: clustered Gaussian blobs inside footprint.
        - 5% of data: contamination cluster at z < 0.3L (outside footprint).
    """
    L = box_size
    n_main = int(0.95 * n_target)
    n_contam = n_target - n_main

    # Cluster centers inside footprint
    centers = np.column_stack([
        rng.uniform(0, L, n_clusters),
        rng.uniform(0, L, n_clusters),
        rng.uniform(0.5 * L, L, n_clusters),
    ])
    main = []
    while len(main) < n_main:
        c = centers[rng.integers(0, n_clusters)]
        p = c + rng.normal(0.0, sigma, size=3)
        # Periodic in x,y only
        p[0] %= L
        p[1] %= L
        if 0.5 * L < p[2] < L:
            main.append(p)

    # Contamination cluster outside footprint (in lower z)
    contam_center = np.array([0.5 * L, 0.5 * L, 0.15 * L])
    contam = []
    while len(contam) < n_contam:
        p = contam_center + rng.normal(0.0, sigma, size=3)
        p[0] %= L
        p[1] %= L
        if 0 < p[2] < 0.3 * L:
            contam.append(p)

    data = np.array(main + contam, dtype=np.float64)
    return data


def make_survey_randoms(n_data, box_size, rng, factor=10):
    """Random catalog: uniform in the SAME footprint as the main data.
    Factor of 10 randoms typical for survey work (DESI uses ~50x).
    """
    L = box_size
    n_r = factor * n_data
    randoms = np.column_stack([
        rng.uniform(0, L, n_r),
        rng.uniform(0, L, n_r),
        rng.uniform(0.5 * L, L, n_r),
    ])
    return randoms


def run_cascade(data_path, randoms_path, out_dir, box_size):
    t0 = time.perf_counter()
    subprocess.run([CASCADE_BIN, 'field-stats',
                    '-i', data_path, '--randoms', randoms_path,
                    '-d', '3', '-L', str(box_size),
                    '-o', out_dir, '--hist-bins', '30', '-q'],
                   capture_output=True, text=True, check=True, timeout=600)
    return time.perf_counter() - t0


def parse_field_moments(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ('level', 'n_cells_active',
                                                   'n_cells_data_outside') else int(v))
                         for k, v in r.items()})
    return rows


def main():
    if not os.path.exists(CASCADE_BIN):
        print(f"FAIL: cascade not built at {CASCADE_BIN}")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 6: Patchy-style survey mock with non-trivial geometry")
    print(f"  Box: {BOX_SIZE} h^-1 Mpc")
    print(f"  Geometry: upper-half-z slab footprint")
    print(f"  Data: 95% clustered in footprint, 5% contamination outside")
    print(f"  Randoms: 10x data, uniform in footprint")
    print("=" * 78)
    print("\nNote: HIPSTER non-periodic mode requires a separate O(N²) Phi fit step.")
    print("Cascade handles footprint natively from random-catalog cell counts.\n")

    workdir = tempfile.mkdtemp(prefix='patchy_bench_')
    rng = np.random.default_rng(SEED)

    # Warmup
    warm_d = make_survey_mock(2000, BOX_SIZE, rng)
    warm_r = make_survey_randoms(2000, BOX_SIZE, rng)
    warm_dp = os.path.join(workdir, 'warm_d.bin')
    warm_rp = os.path.join(workdir, 'warm_r.bin')
    warm_d.astype('<f8').tofile(warm_dp)
    warm_r.astype('<f8').tofile(warm_rp)
    warm_o = os.path.join(workdir, 'warm_out')
    os.makedirs(warm_o, exist_ok=True)
    subprocess.run([CASCADE_BIN, 'field-stats', '-i', warm_dp, '--randoms', warm_rp,
                    '-d', '3', '-L', str(BOX_SIZE), '-o', warm_o,
                    '--hist-bins', '0', '-q'],
                   capture_output=True, check=True, timeout=30)

    results = []
    detailed_first = None
    try:
        for N in N_VALUES:
            data = make_survey_mock(N, BOX_SIZE, rng)
            randoms = make_survey_randoms(N, BOX_SIZE, rng)
            data_path = os.path.join(workdir, f'd_{N}.bin')
            rand_path = os.path.join(workdir, f'r_{N}.bin')
            data.astype('<f8').tofile(data_path)
            randoms.astype('<f8').tofile(rand_path)
            out_dir = os.path.join(workdir, f'out_{N}')
            os.makedirs(out_dir, exist_ok=True)

            t = run_cascade(data_path, rand_path, out_dir, BOX_SIZE)
            mom_rows = parse_field_moments(
                os.path.join(out_dir, 'field_moments.csv'))

            # How much data ended up flagged outside footprint at meaningful scale?
            # Use the level where ~ box/8 (matches contamination cluster scale).
            target_cell = BOX_SIZE / 8
            best_level = min(mom_rows, key=lambda r: abs(r['cell_side_phys'] - target_cell))
            n_outside = best_level['n_cells_data_outside']
            sw_outside = best_level['sum_w_d_outside']
            n_active = best_level['n_cells_active']

            print(f"N={N}: cascade {t:.3f}s "
                  f"(data outside footprint at l={best_level['level']}: "
                  f"{n_outside} cells, W_d={sw_outside:.0f}, "
                  f"active cells: {n_active})")
            results.append({'N': N, 'time': t, 'n_outside': n_outside,
                            'sw_outside': sw_outside})
            if detailed_first is None:
                detailed_first = mom_rows

        # Power-law fit
        Ns = np.array([r['N'] for r in results])
        ts = np.array([r['time'] for r in results])
        b, a = np.polyfit(np.log(Ns), np.log(ts), 1)
        print(f"\nCascade scaling: t ≈ {np.exp(a):.3e} · N^{b:.2f}")

        # Show full diagnostic at first N
        print(f"\n--- Per-level diagnostic for N={results[0]['N']} ---")
        print(f"{'level':>5} {'cell':>8} {'active':>8} {'mean δ':>10} "
              f"{'var δ':>10} {'S_3':>8} {'data_out':>10} {'W_d_out':>10}")
        for r in detailed_first[:12]:
            cell = r['cell_side_phys']
            if cell < 5: continue
            print(f"{r['level']:>5} {cell:>8.1f} {r['n_cells_active']:>8} "
                  f"{r['mean_delta']:>10.3f} {r['var_delta']:>10.3f} "
                  f"{r['s3_delta']:>8.2f} {r['n_cells_data_outside']:>10} "
                  f"{r['sum_w_d_outside']:>10.0f}")

        print("\n--- Interpretation ---")
        n5 = results[0]['N']
        contam_expected = int(0.05 * n5)
        print(f"For N={n5}: {contam_expected} contamination points placed at z<0.3L.")
        print(f"At fine cell scales these get flagged as outside-footprint.")
        print("This is the diagnostic that real survey pipelines need but that")
        print("FFT-based estimators cannot provide natively.")

    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
