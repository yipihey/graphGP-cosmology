#!/usr/bin/env python3
"""
3D comparison of cascade ξ vs Corrfunc xi (3D, native), in both periodic
and non-periodic mode.

Cox points loaded from /tmp/cascade_3d_run/. Box is [0, 128) tree-coord.
For non-periodic mode we generate a random catalog uniformly in [0, 128)^3
and use Landy-Szalay (DD - 2*DR + RR) / RR.
"""
import numpy as np
import pandas as pd
import glob
import os
import time
from Corrfunc.theory.DD import DD
from Corrfunc.theory.xi import xi as cf_xi
from Corrfunc.utils import convert_3d_counts_to_cf

BOXSIZE = 128.0
DIR = "/tmp/cascade_3d_run"

points_files = sorted(glob.glob(f"{DIR}/points_*.bin"))
casc_files   = sorted(glob.glob(f"{DIR}/tpcf_*.csv"))
n_real = min(len(points_files), len(casc_files))
print(f"Found {n_real} 3D realizations")

# Cascade lags: 1, 2, 4, 8, 16, 32, 64
unique_rs = [1, 2, 4, 8, 16, 32]   # cap below L/2 = 64 for periodic
log2_rs = np.log2(unique_rs)
edges = []
for lr in log2_rs:
    edges.extend([2**(lr - 0.025), 2**(lr + 0.025)])
edges = sorted(set(edges))
edges = np.array(edges)

cf_periodic_per_real = []
cf_nonper_per_real   = []
casc_per_real        = []
times_periodic = []
times_nonper   = []

for i in range(n_real):
    raw = np.fromfile(points_files[i], dtype=np.float64)
    pts = raw.reshape(-1, 3)
    N = pts.shape[0]
    X = np.ascontiguousarray(pts[:, 0])
    Y = np.ascontiguousarray(pts[:, 1])
    Z = np.ascontiguousarray(pts[:, 2])

    # ---- Corrfunc periodic xi (uses analytic RR) ----
    t0 = time.time()
    res_per = cf_xi(boxsize=BOXSIZE, nthreads=1, binfile=edges,
                    X=X, Y=Y, Z=Z, verbose=False)
    times_periodic.append(time.time() - t0)
    cf_per = {}
    for r in res_per:
        rmid = 0.5 * (r['rmin'] + r['rmax'])
        idx = np.argmin(np.abs(np.log(np.array(unique_rs)) - np.log(rmid)))
        cf_per[unique_rs[idx]] = r['xi']
    cf_periodic_per_real.append(cf_per)

    # ---- Corrfunc non-periodic xi via Landy-Szalay with random catalog ----
    # Generate random catalog of same size N uniformly in [0, BOXSIZE)^3
    Nr = N
    rng = np.random.default_rng(seed=1000 + i)
    Xr = rng.uniform(0, BOXSIZE, Nr)
    Yr = rng.uniform(0, BOXSIZE, Nr)
    Zr = rng.uniform(0, BOXSIZE, Nr)

    t0 = time.time()
    DD_dd = DD(autocorr=1, nthreads=1, binfile=edges,
               X1=X, Y1=Y, Z1=Z, periodic=False, verbose=False)
    DR_dd = DD(autocorr=0, nthreads=1, binfile=edges,
               X1=X, Y1=Y, Z1=Z,
               X2=Xr, Y2=Yr, Z2=Zr, periodic=False, verbose=False)
    RR_dd = DD(autocorr=1, nthreads=1, binfile=edges,
               X1=Xr, Y1=Yr, Z1=Zr, periodic=False, verbose=False)
    xi_ls = convert_3d_counts_to_cf(N, N, Nr, Nr,
                                     DD_dd, DR_dd, DR_dd, RR_dd)
    times_nonper.append(time.time() - t0)
    cf_np = {}
    for j, r in enumerate(DD_dd):
        rmid = 0.5 * (r['rmin'] + r['rmax'])
        idx = np.argmin(np.abs(np.log(np.array(unique_rs)) - np.log(rmid)))
        cf_np[unique_rs[idx]] = xi_ls[j]
    cf_nonper_per_real.append(cf_np)

    # ---- Cascade ----
    casc = pd.read_csv(casc_files[i])
    cx = {(int(r['level']), int(r['k'])): r['xi_measured'] for _, r in casc.iterrows()}
    casc_per_real.append(cx)

    print(f"  realization {i+1}/{n_real}: N={N}, "
          f"corrfunc periodic {times_periodic[-1]*1000:.0f} ms, "
          f"non-periodic (DD+DR+RR) {times_nonper[-1]*1000:.0f} ms")

def avg_dict(lst):
    keys = lst[0].keys()
    return {k: np.mean([d.get(k, np.nan) for d in lst]) for k in keys}

cf_per_avg = avg_dict(cf_periodic_per_real)
cf_np_avg  = avg_dict(cf_nonper_per_real)
casc_avg   = avg_dict(casc_per_real)

print()
print("=" * 80)
print(f"3D CASCADE vs CORRFUNC (averaged over {n_real} realizations)")
print("=" * 80)
print(f"Cascade time per realization: ~4000 ms (from Rust output)")
print(f"Corrfunc periodic xi:          {np.mean(times_periodic)*1000:.0f} ms")
print(f"Corrfunc non-periodic LS:      {np.mean(times_nonper)*1000:.0f} ms")
print()
print(f"{'r':>4} {'lvl':>4} {'k':>4} {'h_l':>6} "
      f"{'<casc>':>14} {'<cf periodic>':>14} {'<cf nonper>':>14} "
      f"{'rel(per)':>10} {'rel(np)':>10}")

# Match cascade entries (level, k) to r = k * 128 / 2^level
for r in unique_rs:
    if r not in cf_per_avg:
        continue
    # Pick best cascade entry: smallest h_l
    best = None; best_h = float('inf')
    for (lvl, k), xi in casc_avg.items():
        if k * (128 // (1 << lvl)) == r:
            h = 128.0 / (1 << lvl)
            if h < best_h:
                best_h = h
                best = (lvl, k, xi)
    if best is None: continue
    lvl, k, xi_c = best
    xi_per = cf_per_avg[r]
    xi_np  = cf_np_avg.get(r, float('nan'))
    rel_per = (xi_c - xi_per) / xi_per if abs(xi_per) > 1e-12 else float('nan')
    rel_np  = (xi_c - xi_np)  / xi_np  if abs(xi_np)  > 1e-12 else float('nan')
    print(f"{r:>4} {lvl:>4} {k:>4} {best_h:>6.2f} "
          f"{xi_c:>14.5e} {xi_per:>14.5e} {xi_np:>14.5e} "
          f"{rel_per:>10.4f} {rel_np:>10.4f}")
