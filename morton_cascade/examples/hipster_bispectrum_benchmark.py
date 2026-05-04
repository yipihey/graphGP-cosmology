#!/usr/bin/env python3
"""
Benchmark 5: cascade <delta^3> vs REAL HIPSTER bispectrum.

HIPSTER's bispectrum is documented as O(N · n^2 · R0^6) in the paper, but
their actual implementation uses spherical harmonic decomposition tricks
that improve this. We measure their real wall time and compare to cascade
field-stats which produces all moments at all dyadic scales in one pass.

The two computations measure related but not identical observables:
- HIPSTER: B_ell(k1, k2) Legendre multipoles of the bispectrum (truncated at R0)
- Cascade: <delta^3>_W_r at every dyadic cell scale (cube-window-averaged)

Both are 3-point statistics; both give scale-dependent information about
non-Gaussianity. The cascade output includes <delta^3> at 33 dyadic scales;
HIPSTER produces multipoles for the user-specified k-bins.

What we're measuring is wall-time scaling and the ratio of statistics-per-second.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

BOX_SIZE = 1000.0
R0 = 100.0
N_VALUES = [5_000, 10_000, 20_000]
SEED = 271828
HIPSTER_BIN = "/tmp/HIPSTER/power"
CASCADE_BIN = "./target/release/morton-cascade"


def make_clustered(n, box, rng, n_par=200, sig=20.0):
    parents = rng.uniform(0, box, size=(n_par, 3))
    pts = []
    while len(pts) < n:
        p = parents[rng.integers(0, n_par)]
        c = (p + rng.normal(0.0, sig, size=3)) % box
        pts.append(c)
    return np.array(pts[:n], dtype=np.float64)


def write_kbin_file(path, n_bins=10, k_lo=0.05, k_hi=1.0):
    edges = np.geomspace(k_lo, k_hi, n_bins + 1)
    with open(path, 'w') as f:
        for lo, hi in zip(edges[:-1], edges[1:]):
            f.write(f'{lo:.6f}\t{hi:.6f}\n')


def main():
    if not os.path.exists(HIPSTER_BIN):
        print(f"HIPSTER not built at {HIPSTER_BIN}")
        sys.exit(1)

    print("=" * 78)
    print("Benchmark 5: cascade <δ³> at all dyadic scales vs HIPSTER bispectrum")
    print(f"  Box: {BOX_SIZE} h^-1 Mpc, R0: {R0} h^-1 Mpc")
    print("=" * 78)
    print("\nNote: HIPSTER must be built with -DPERIODIC -DBISPECTRUM.")
    print("Run: cd /tmp/HIPSTER && rm -f *.o power && \\")
    print("     make Periodic=-DPERIODIC Bispectrum=-DBISPECTRUM\n")

    workdir = tempfile.mkdtemp(prefix='bispec_bench_')
    rng = np.random.default_rng(SEED)
    kbin_path = os.path.join(workdir, 'kbins.tsv')
    write_kbin_file(kbin_path)

    # Warmup run to prime caches and avoid cold-start bias on first measurement
    print("Warming up cascade binary...")
    warm_data = make_clustered(2000, BOX_SIZE, rng)
    warm_d = os.path.join(workdir, 'warm_d.bin')
    warm_r = os.path.join(workdir, 'warm_r.bin')
    warm_data.astype('<f8').tofile(warm_d)
    rng.uniform(0, BOX_SIZE, size=(10000, 3)).astype('<f8').tofile(warm_r)
    warm_out = os.path.join(workdir, 'warm_out')
    os.makedirs(warm_out, exist_ok=True)
    subprocess.run([CASCADE_BIN, 'field-stats', '-i', warm_d, '--randoms', warm_r,
                    '-d', '3', '-L', str(BOX_SIZE), '-o', warm_out,
                    '--hist-bins', '0', '-q'],
                   capture_output=True, check=True, timeout=30)
    print("Warmup done.\n")

    results = []
    try:
        for N in N_VALUES:
            print(f"--- N = {N} ---")
            data = make_clustered(N, BOX_SIZE, rng)

            # HIPSTER input
            hip_data = os.path.join(workdir, f'd_{N}.txt')
            np.savetxt(hip_data, np.column_stack([data, np.ones(N)]), fmt='%.6f')
            hip_out = os.path.join(workdir, f'hip_{N}')
            os.makedirs(hip_out, exist_ok=True)

            t0 = time.perf_counter()
            r = subprocess.run([HIPSTER_BIN, '-in', hip_data,
                '-binfile', kbin_path, '-output', hip_out,
                '-out_string', f'b{N}', '-perbox',
                '-R0', str(R0), '-max_l', '0', '-f_rand', '3',
                '-nthread', '4'],
                capture_output=True, text=True, timeout=3600)
            hip_time = time.perf_counter() - t0
            if r.returncode != 0:
                print(f"  HIPSTER failed: {r.stderr[:300]}")
                continue

            # Cascade input
            cas_data = os.path.join(workdir, f'cd_{N}.bin')
            cas_rand = os.path.join(workdir, f'cr_{N}.bin')
            data.astype('<f8').tofile(cas_data)
            randoms = rng.uniform(0, BOX_SIZE, size=(5 * N, 3)).astype('<f8')
            randoms.tofile(cas_rand)
            cas_out = os.path.join(workdir, f'cas_{N}')
            os.makedirs(cas_out, exist_ok=True)

            t0 = time.perf_counter()
            subprocess.run([CASCADE_BIN, 'field-stats',
                '-i', cas_data, '--randoms', cas_rand,
                '-d', '3', '-L', str(BOX_SIZE), '-o', cas_out,
                '--hist-bins', '0', '-q'],
                capture_output=True, text=True, check=True, timeout=600)
            cas_time = time.perf_counter() - t0

            speedup = hip_time / cas_time
            print(f"  HIPSTER bispectrum:  {hip_time:7.2f}s  (B_0(k) at 10 k-bins)")
            print(f"  Cascade <δ³> + more: {cas_time:7.3f}s  (m2,m3,m4,S_3,P(δ) at 33 scales)")
            print(f"  Speedup:             {speedup:6.1f}x\n")
            results.append({'N': N, 'hipster': hip_time, 'cascade': cas_time,
                            'speedup': speedup})

        print("=" * 78)
        print(f"{'N':>8} | {'HIPSTER B(k) [s]':>17} | {'Cascade <δ³>+ [s]':>18} | {'Speedup':>10} | exp")
        print("-" * 78)
        for i, r in enumerate(results):
            line = (f"{r['N']:>8} | {r['hipster']:>17.2f} | {r['cascade']:>18.3f} "
                    f"| {r['speedup']:>9.1f}x")
            if i > 0:
                p = results[i-1]
                hip_exp = np.log(r['hipster']/p['hipster']) / np.log(r['N']/p['N'])
                cas_exp = np.log(r['cascade']/p['cascade']) / np.log(r['N']/p['N'])
                line += f" | hip {hip_exp:.2f} cas {cas_exp:.2f}"
            print(line)

        # Power-law fits
        Ns = np.array([r['N'] for r in results])
        hips = np.array([r['hipster'] for r in results])
        cass = np.array([r['cascade'] for r in results])
        b_hip, a_hip = np.polyfit(np.log(Ns), np.log(hips), 1)
        b_cas, a_cas = np.polyfit(np.log(Ns), np.log(cass), 1)
        print(f"\nPower-law fits:")
        print(f"  HIPSTER:  t ≈ {np.exp(a_hip):.2e} · N^{b_hip:.2f}")
        print(f"  Cascade:  t ≈ {np.exp(a_cas):.2e} · N^{b_cas:.2f}")

    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
