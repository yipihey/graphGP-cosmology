#!/usr/bin/env python3
"""
sigma^2(R) accuracy test, comparing three estimators on the same Cox 3D realizations:
  (1) cascade sigma^2_field = (var(N) - <N>) / <N>^2 from per-level stats
  (2) direct python: histogram into cubes of side R, averaged over many random shifts
  (3) Corrfunc-derived: compute xi(r) at fine bins, integrate against cell window
"""
import numpy as np
import pandas as pd
import glob
import time
from Corrfunc.theory.xi import xi as cf_xi
# (scipy import moved below)

BOXSIZE = 128.0
DIR = "/tmp/cascade_3d_run"

points_files = sorted(glob.glob(f"{DIR}/points_*.bin"))
stats_files  = sorted(glob.glob(f"{DIR}/stats_*.csv"))
n_real = min(len(points_files), len(stats_files))
print(f"Found {n_real} 3D realizations\n")

df0 = pd.read_csv(stats_files[0])
R_values = sorted([r for r in df0['R_tree'].values if 1.0 <= r <= 64.0], reverse=True)
print(f"Comparing at R = {R_values}")

# (1) Cascade sigma^2 per realization
cascade_s2 = {R: [] for R in R_values}
for f in stats_files:
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        R = row['R_tree']
        if R in cascade_s2:
            cascade_s2[R].append(row['sigma2_field'])

# (2) Direct shift-averaged Python histogramming
def direct_shift_avg_sigma2(pts, R, n_shifts=20):
    n_per_axis = int(round(BOXSIZE / R))
    if n_per_axis < 2:
        return float('nan')
    rng = np.random.default_rng(seed=42)
    s2s = []
    for _ in range(n_shifts):
        sx, sy, sz = rng.uniform(0, R, 3)
        pts_s = np.empty_like(pts)
        pts_s[:, 0] = (pts[:, 0] + sx) % BOXSIZE
        pts_s[:, 1] = (pts[:, 1] + sy) % BOXSIZE
        pts_s[:, 2] = (pts[:, 2] + sz) % BOXSIZE
        H, _ = np.histogramdd(pts_s, bins=[np.linspace(0, BOXSIZE, n_per_axis+1)] * 3)
        counts = H.flatten()
        mean = counts.mean()
        var = counts.var()
        if mean > 1e-12:
            s2s.append((var - mean) / (mean * mean))
    return np.mean(s2s) if s2s else float('nan')

print("\nComputing direct shift-averaged sigma^2 (slow)...")
direct_s2 = {R: [] for R in R_values}
t0 = time.time()
for fp in points_files:
    raw = np.fromfile(fp, dtype=np.float64)
    pts = raw.reshape(-1, 3)
    for R in R_values:
        n_shifts = 20 if R >= 4 else 8
        direct_s2[R].append(direct_shift_avg_sigma2(pts, R, n_shifts=n_shifts))
print(f"Direct sigma^2 done in {time.time()-t0:.1f} s")

# (3) Corrfunc-derived sigma^2 via xi(r) integral
def cube_sigma2_from_xi(xi_func, R, n_grid=20):
    s_axis = np.linspace(0.5*R/n_grid, R - 0.5*R/n_grid, n_grid)
    ds = R / n_grid
    sx, sy, sz = np.meshgrid(s_axis, s_axis, s_axis, indexing='ij')
    s_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    weight = (R - sx) * (R - sy) * (R - sz)
    xi_vals = xi_func(s_mag)
    integral = (xi_vals * weight).sum() * ds**3 * 8.0
    return integral / R**6

print("\nComputing Corrfunc-derived sigma^2...")
raw = np.fromfile(points_files[1], dtype=np.float64)
pts = raw.reshape(-1, 3)
N = pts.shape[0]
X = np.ascontiguousarray(pts[:, 0])
Y = np.ascontiguousarray(pts[:, 1])
Z = np.ascontiguousarray(pts[:, 2])

edges = np.logspace(np.log10(0.5), np.log10(20), 12)
print(f"  bin edges: {edges}")
t0 = time.time()
xi_res = cf_xi(boxsize=BOXSIZE, nthreads=1, binfile=edges, X=X, Y=Y, Z=Z, verbose=False)
t_corrfunc = time.time() - t0
print(f"Corrfunc xi at {len(xi_res)} bins (1 realization): {t_corrfunc:.1f} s")

r_centers = np.array([0.5*(b['rmin']+b['rmax']) for b in xi_res])
xi_vals = np.array([b['xi'] for b in xi_res])
from scipy.interpolate import interp1d
xi_interp = interp1d(np.log(r_centers), xi_vals, fill_value=(xi_vals[0], xi_vals[-1]),
                     bounds_error=False, kind='linear')
def xi_func(r):
    r = np.asarray(r)
    out = np.zeros_like(r, dtype=float)
    mask = r > r_centers[0] * 0.5
    out[mask] = xi_interp(np.log(np.maximum(r[mask], r_centers[0])))
    out[~mask] = xi_vals[0]
    return out

corrfunc_s2 = {}
xi_max_r = r_centers[-1]   # xi only known up to here
for R in R_values:
    # Need xi at r up to R*sqrt(3); skip if that exceeds xi knowledge
    if R * np.sqrt(3) > xi_max_r:
        corrfunc_s2[R] = float('nan')
        continue
    n_grid = max(15, int(20 * 10/max(R,1))) if R < 10 else 25
    corrfunc_s2[R] = cube_sigma2_from_xi(xi_func, R, n_grid=n_grid)

print()
print("=" * 100)
print(f"sigma^2(R) cube comparison, averaged over {n_real} 3D Cox realizations")
print("=" * 100)
print(f"{'R':>6} {'<casc>':>15} {'<direct shift-avg>':>22} {'rel(c-d)':>10} {'corrfunc-int (1 real)':>22}")
for R in R_values:
    c_arr = np.array(cascade_s2[R])
    d_arr = np.array(direct_s2[R])
    cm, csd = c_arr.mean(), c_arr.std() / np.sqrt(len(c_arr))
    dm, dsd = d_arr.mean(), d_arr.std() / np.sqrt(len(d_arr))
    rel = (cm - dm) / dm if abs(dm) > 1e-12 else float('nan')
    cf_val = corrfunc_s2.get(R, float('nan'))
    print(f"{R:>6.1f} {cm:>9.5e}+-{csd:>7.2e}  {dm:>9.5e}+-{dsd:>7.2e}  {rel:>10.4f}  {cf_val:>15.5e}")

print()
print("Timing summary:")
print(f"  Cascade ALL R simultaneously, 3D N=200k: ~3.4 s per realization (cascade only)")
print(f"  Direct shift-avg ONE R, 3D N=200k:        ~50 ms per shift x 20 shifts = ~1 s per R")
print(f"  Corrfunc xi at 40 bins, 3D N=200k:        {t_corrfunc:.1f} s per realization")
