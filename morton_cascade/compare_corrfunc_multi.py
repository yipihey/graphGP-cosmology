#!/usr/bin/env python3
"""
Multi-realization comparison: cascade ξ vs Corrfunc ξ averaged over many realizations.
This tests whether the per-realization disagreement at r=64 is sample variance from
the cascade's axis-only sampling vs Corrfunc's full angular averaging.

The Rust binary writes one points file per realization to /tmp/cascade_xi_run/,
along with cascade ξ values. We re-load each, run Corrfunc, and average.
"""
import numpy as np
import pandas as pd
from Corrfunc.theory.DD import DD
import time
import os
import glob
import sys

BOXSIZE = 256.0
DIR = "/tmp/cascade_xi_run"

if not os.path.isdir(DIR):
    print(f"Run directory {DIR} not found. Run the Rust binary first to populate it.", file=sys.stderr)
    sys.exit(1)

# Bin edges shared across all realizations
unique_rs = [1, 2, 4, 8, 16, 32, 64]
log2_rs = np.log2(unique_rs)
edges = []
for lr in log2_rs:
    edges.extend([2**(lr - 0.025), 2**(lr + 0.025)])
edges = sorted(set(edges))
edges = np.array(edges)
edges_centers = []
for lr in log2_rs:
    edges_centers.append(2**lr)

# For each realization: load points, run Corrfunc, also load cascade xi values
real_files = sorted(glob.glob(f"{DIR}/points_*.bin"))
xi_files   = sorted(glob.glob(f"{DIR}/tpcf_*.csv"))
n_real = min(len(real_files), len(xi_files))
print(f"Found {n_real} realizations", file=sys.stderr)

cascade_xi_per_real = []   # list of dicts {(level, k): xi}
corrfunc_xi_per_real = []  # list of dicts {r: xi}
corrfunc_times = []

for i in range(n_real):
    raw = np.fromfile(real_files[i], dtype=np.float64)
    pts = raw.reshape(-1, 2)
    N = pts.shape[0]
    X = np.ascontiguousarray(pts[:, 0])
    Y = np.ascontiguousarray(pts[:, 1])
    Z = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    dd = DD(autocorr=1, nthreads=1, binfile=edges,
            X1=X, Y1=Y, Z1=Z, periodic=True, boxsize=BOXSIZE,
            verbose=False)
    dt = time.time() - t0
    corrfunc_times.append(dt)

    L = BOXSIZE
    cf_xi_at_r = {}
    for d in dd:
        rmin, rmax = d['rmin'], d['rmax']
        rmid = 0.5 * (rmin + rmax)
        rr = N * (N - 1) / (L * L) * np.pi * (rmax**2 - rmin**2)
        xi = d['npairs'] / rr - 1.0
        # Match to nearest unique r
        idx = np.argmin(np.abs(np.log(np.array(unique_rs)) - np.log(rmid)))
        cf_xi_at_r[unique_rs[idx]] = xi
    corrfunc_xi_per_real.append(cf_xi_at_r)

    casc = pd.read_csv(xi_files[i])
    casc_xi = {}
    for _, row in casc.iterrows():
        casc_xi[(int(row['level']), int(row['k']))] = row['xi_measured']
    cascade_xi_per_real.append(casc_xi)
    print(f"  realization {i+1}/{n_real}: N={N}, corrfunc {dt*1000:.0f} ms", file=sys.stderr)

# Average each estimator across realizations
def avg_dict(list_of_dicts):
    if not list_of_dicts:
        return {}
    keys = list_of_dicts[0].keys()
    out = {}
    for k in keys:
        out[k] = np.mean([d.get(k, np.nan) for d in list_of_dicts])
    return out

cf_avg = avg_dict(corrfunc_xi_per_real)
casc_avg = avg_dict(cascade_xi_per_real)

# Build comparison table: cascade (level, k) vs corrfunc at corresponding r
print()
print(f"Averaged over {n_real} realizations.")
print(f"Mean Corrfunc time per realization: {np.mean(corrfunc_times)*1000:.1f} ms")
print()
print(f"{'lvl':>5} {'k':>5} {'r_tree':>8} {'h_l':>8} {'<xi(casc)>':>14} {'<xi(corrfunc)>':>14} {'rel diff':>12}")

# Generate same (level, k) pairs as before
levels = sorted(set(k[0] for k in casc_avg.keys()))
for lvl in sorted(levels, reverse=True):
    h_l = 256.0 / (1 << lvl)   # tree-coord cell side
    for (l_x, k_x), xi in sorted(casc_avg.items()):
        if l_x != lvl: continue
        r_tree = k_x * (256 / (1 << lvl))
        if r_tree not in cf_avg: continue
        xi_cf = cf_avg[r_tree]
        rel = (xi - xi_cf) / xi_cf if abs(xi_cf) > 1e-12 else float('nan')
        print(f"{lvl:>5} {k_x:>5} {r_tree:>8.1f} {h_l:>8.2f} {xi:>14.5e} {xi_cf:>14.5e} {rel:>12.4f}")
