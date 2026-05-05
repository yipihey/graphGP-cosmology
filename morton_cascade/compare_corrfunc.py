#!/usr/bin/env python3
"""
Compare cascade ξ(r) against Corrfunc DD pair counting on the same Cox realization.

Cascade output: tpcf.csv  (level, k, r_tree, smoothing_h_fine, xi_measured)
Cox points:     points_xi.bin  (binary f64 (x, y) pairs, box [0, 256))

We run Corrfunc.theory.DD on the points with z=0, periodic boxsize=256,
in r bins matching the cascade's lag grid. Then convert DD -> xi using
the analytic 2D RR.
"""
import numpy as np
import pandas as pd
from Corrfunc.theory.DD import DD
import time
import sys

BOXSIZE = 256.0  # tree-coord box

# Load points
print("Loading points...", file=sys.stderr)
raw = np.fromfile("/home/claude/morton_cascade/points_xi.bin", dtype=np.float64)
pts = raw.reshape(-1, 2)
N = pts.shape[0]
X = np.ascontiguousarray(pts[:, 0])
Y = np.ascontiguousarray(pts[:, 1])
Z = np.zeros(N, dtype=np.float64)
print(f"Loaded {N} points in [0, {BOXSIZE}) box", file=sys.stderr)

# Load cascade results
print("Loading cascade tpcf.csv...", file=sys.stderr)
casc = pd.read_csv("/home/claude/morton_cascade/tpcf.csv")
print(f"Cascade: {len(casc)} (level, k) entries", file=sys.stderr)

# Build a r-bin set that covers the cascade scales. Cascade lags are at
# r_tree = k * 2^(L_MAX - level), L_MAX = 8, so r_tree ∈ {1, 2, 4, ..., 128}.
# We use thin log-spaced bins around each unique r value, so DD pair counts
# are localized to the cascade lag.
# Cascade r values include up to r=128 = L/2, but Corrfunc requires r_max < L/2.
# Restrict to r_tree <= 64 for the comparison.
casc = casc[casc['r_tree'] <= 64].reset_index(drop=True)
print(f"After r_tree<=64 cut: {len(casc)} cascade entries", file=sys.stderr)
unique_rs = sorted(casc['r_tree'].unique())
print(f"Cascade r values: {unique_rs}", file=sys.stderr)

# Place bin edges to bracket each r value with a thin annulus.
# Use width factor 1.05 (well below the smallest r-spacing factor of 2 in cascade lags)
# so each bin is sharply localized near the cascade r.
log2_rs = np.log2(unique_rs)
edges = []
for lr in log2_rs:
    edges.extend([2**(lr - 0.025), 2**(lr + 0.025)])  # ±2.5% in log2 -> ±1.7% in r
edges = sorted(set(edges))
edges = np.array(edges)
print(f"Bin edges (narrow): {edges}", file=sys.stderr)

# Run Corrfunc DD
print("Running Corrfunc DD (autocorr=1, nthreads=1)...", file=sys.stderr)
t0 = time.time()
dd = DD(autocorr=1, nthreads=1, binfile=edges,
        X1=X, Y1=Y, Z1=Z, periodic=True, boxsize=BOXSIZE,
        verbose=False)
t_corrfunc = time.time() - t0
print(f"Corrfunc time: {t_corrfunc*1000:.2f} ms", file=sys.stderr)

# Convert DD pairs -> xi via analytic 2D RR.
# For 2D periodic Poisson on a box of side L with N points, the expected pair count
# in an annulus [rmin, rmax] is N*(N-1)/L^2 * pi * (rmax^2 - rmin^2) for separations
# small compared to L (no wrap correction). Corrfunc's DD counts each pair twice for
# autocorr (i,j) and (j,i) -- check against Poisson test above.
# In our small Poisson test above, DD gave 928 pairs in [1,2] and prediction was 941.5.
# So Corrfunc returns the unique-pair count (i<j), N*(N-1)/2 / V * 2*pi*r*dr.
# Wait: 941.5 was N*(N-1)/V * pi*(rmax^2 - rmin^2). That's N*(N-1)/V * shell area.
# DD reported 928, very close to 941.5 -> Corrfunc reports COUNT OF ORDERED PAIRS / 2,
# but the convention depends on autocorr. Let's just compute RR_predict the same way.

L = BOXSIZE
RR_pred = []
xi_corrfunc = []
r_mid = []
for d in dd:
    rmin, rmax = d['rmin'], d['rmax']
    rmid = 0.5 * (rmin + rmax)
    # For Corrfunc autocorr=1, npairs = #{(i,j): i<j, |x_i-x_j| in [rmin, rmax]}
    # Random expectation in 2D: same formula as our Poisson test confirmed.
    rr = N * (N - 1) / 2.0 / (L*L) * np.pi * (rmax**2 - rmin**2) * 2.0
    # Hmm, need to double-check the factor. From Poisson test:
    # N=1000, L=100, bin [1,2]: predict = 1000*999/100^2 * pi * (4-1) = 941.6 ✓
    # So formula is N*(N-1)/L^2 * pi * (rmax^2 - rmin^2). Let's recompute.
    rr = N * (N - 1) / (L*L) * np.pi * (rmax**2 - rmin**2)
    xi = d['npairs'] / rr - 1.0
    RR_pred.append(rr)
    xi_corrfunc.append(xi)
    r_mid.append(rmid)

cf_df = pd.DataFrame({
    'rmin': [d['rmin'] for d in dd],
    'rmax': [d['rmax'] for d in dd],
    'r_mid': r_mid,
    'npairs': [d['npairs'] for d in dd],
    'rr_pred': RR_pred,
    'xi_corrfunc': xi_corrfunc,
})
print()
print("Corrfunc DD -> xi:")
print(cf_df.to_string(index=False))

# Now match cascade xi to corrfunc xi at the closest r.
# For each cascade (level, k, r_tree), find the corrfunc bin centered nearest to r_tree.
print()
print("=" * 100)
print("Cascade vs Corrfunc comparison:")
print("=" * 100)
print(f"{'lvl':>5} {'k':>5} {'r_tree':>8} {'h_l':>8} {'xi(casc)':>14} {'xi(corrfunc)':>14} {'rel diff':>12} {'corrfunc r':>12}")

for _, row in casc.iterrows():
    r_t = row['r_tree']
    # Find corrfunc bin whose midpoint is closest in log to r_t
    idx = (np.abs(np.log(np.array(r_mid)) - np.log(r_t))).argmin()
    xi_cf = xi_corrfunc[idx]
    xi_c  = row['xi_measured']
    rel = (xi_c - xi_cf) / xi_cf if abs(xi_cf) > 1e-12 else float('nan')
    print(f"{int(row['level']):>5} {int(row['k']):>5} {r_t:>8.1f} {row['smoothing_h_fine']:>8.1f} "
          f"{xi_c:>14.5e} {xi_cf:>14.5e} {rel:>12.4f} {r_mid[idx]:>12.3f}")

print()
print(f"Corrfunc time: {t_corrfunc*1000:.2f} ms")
print(f"(Cascade time: see Rust output ~ 100 ms for the same N)")
