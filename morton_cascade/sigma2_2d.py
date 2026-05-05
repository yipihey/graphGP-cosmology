#!/usr/bin/env python3
"""2D sigma^2(R) test: cascade per-level stats vs direct shift-averaged Python."""
import numpy as np
import pandas as pd
import glob
import time

BOXSIZE_2D = 256.0
DIR = "/tmp/cascade_xi_run"

points_files = sorted(glob.glob(f"{DIR}/points_*.bin"))
stats_files  = sorted(glob.glob(f"{DIR}/stats_*.csv"))
n_real = min(len(points_files), len(stats_files))
print(f"Found {n_real} 2D realizations")
if n_real == 0:
    raise SystemExit(0)

df0 = pd.read_csv(stats_files[0])
R_values = sorted([r for r in df0['R_tree'].values if 1 <= r <= 128], reverse=True)
print(f"R values: {R_values}")

cascade_s2 = {R: [] for R in R_values}
for f in stats_files:
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        R = row['R_tree']
        if R in cascade_s2:
            cascade_s2[R].append(row['sigma2_field'])

def direct_shift_avg_sigma2_2d(pts, R, n_shifts=20):
    n_per_axis = int(round(BOXSIZE_2D / R))
    if n_per_axis < 2: return float('nan')
    rng = np.random.default_rng(seed=42)
    s2s = []
    for _ in range(n_shifts):
        sx, sy = rng.uniform(0, R, 2)
        pts_s = np.empty_like(pts)
        pts_s[:, 0] = (pts[:, 0] + sx) % BOXSIZE_2D
        pts_s[:, 1] = (pts[:, 1] + sy) % BOXSIZE_2D
        H, _, _ = np.histogram2d(pts_s[:, 0], pts_s[:, 1],
                                 bins=[np.linspace(0, BOXSIZE_2D, n_per_axis+1)] * 2)
        counts = H.flatten()
        mean = counts.mean()
        var = counts.var()
        if mean > 1e-12:
            s2s.append((var - mean) / (mean * mean))
    return np.mean(s2s) if s2s else float('nan')

direct_s2 = {R: [] for R in R_values}
print("\nComputing direct shift-averaged sigma^2 in 2D...")
t0 = time.time()
for fp in points_files:
    raw = np.fromfile(fp, dtype=np.float64)
    pts = raw.reshape(-1, 2)
    for R in R_values:
        direct_s2[R].append(direct_shift_avg_sigma2_2d(pts, R, n_shifts=20))
print(f"Direct sigma^2 done in {time.time()-t0:.1f} s")

print()
print(f"2D sigma^2(R) cascade vs direct (mean of {n_real} Cox realizations, N=50k each):")
print(f"{'R':>5} {'<casc>':>15} {'<direct>':>15} {'rel diff':>12} {'casc SE':>12} {'direct SE':>12}")
for R in R_values:
    c = np.array(cascade_s2[R]); d = np.array(direct_s2[R])
    cm, csd = c.mean(), c.std() / np.sqrt(len(c))
    dm, dsd = d.mean(), d.std() / np.sqrt(len(d))
    rel = (cm - dm) / dm if abs(dm) > 1e-12 else float('nan')
    print(f"{R:>5.1f} {cm:>15.5e} {dm:>15.5e} {rel:>12.5f} {csd:>12.2e} {dsd:>12.2e}")
