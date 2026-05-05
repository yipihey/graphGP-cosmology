#!/usr/bin/env python3
"""
Diagnose the cascade-vs-Corrfunc disagreement at r=64 by computing
axis-aligned vs angle-averaged xi from the realized Cox field.

We don't have the lambda field exposed from Rust, but we can use Corrfunc
in a controlled way to measure axis-aligned xi: limit pair counts to thin
angular wedges around the axes vs all angles.

Actually simpler: measure xi via direct cell binning of the points,
both axis-aligned and angle-averaged, in Python.
"""
import numpy as np
import time

BOXSIZE = 256.0

# Load one realization
raw = np.fromfile("/tmp/cascade_xi_run/points_000.bin", dtype=np.float64)
pts = raw.reshape(-1, 2)
N = pts.shape[0]
print(f"Loaded {N} points")

# Bin into a coarse grid for cross-correlation analysis
G = 256
hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1],
                             bins=[np.linspace(0, BOXSIZE, G+1)] * 2)
# hist is GxG count grid. Cell side = BOXSIZE/G = 1 tree-coord unit (matches level-8 cell size).
mean = hist.mean()
print(f"Mean count per level-8 cell: {mean:.4f}")

# Axis-aligned xi at lag k cells
def axis_xi(grid, k):
    """Mean of grid * shifted(grid, axis k), averaged over both axes."""
    s_x = np.roll(grid, k, axis=1)
    s_y = np.roll(grid, k, axis=0)
    avg_x = np.mean(grid * s_x)
    avg_y = np.mean(grid * s_y)
    m = grid.mean()
    avg = 0.5 * (avg_x + avg_y)
    return avg / (m * m) - 1.0

# Diagonal xi: shift by (k, k) cells
def diag_xi(grid, k):
    s_xy = np.roll(np.roll(grid, k, axis=0), k, axis=1)
    s_yx = np.roll(np.roll(grid, k, axis=0), -k, axis=1)
    avg = 0.5 * (np.mean(grid * s_xy) + np.mean(grid * s_yx))
    m = grid.mean()
    return avg / (m * m) - 1.0

# Angle-averaged xi at radius r (sum over annulus)
def annulus_xi(grid, r_target, ring_width=1.0):
    """Average xi over an annulus of pixels at radius r_target."""
    # Use FFT to compute the 2D autocorrelation in one shot
    # autocorr(x) = IFFT(|FFT(x)|^2) / N
    F = np.fft.fft2(grid - grid.mean())
    ac = np.real(np.fft.ifft2(np.abs(F)**2))
    ac /= grid.size  # normalize: ac[0,0] = var
    # Now extract the annulus
    G = grid.shape[0]
    iy, ix = np.indices((G, G))
    # Use minimum-image distance
    dx = np.minimum(ix, G - ix)
    dy = np.minimum(iy, G - iy)
    rr = np.sqrt(dx*dx + dy*dy)
    mask = (rr >= r_target - ring_width/2) & (rr < r_target + ring_width/2)
    if mask.sum() == 0:
        return float('nan'), 0
    avg = ac[mask].mean()
    m2 = grid.mean()**2
    # ac is the centered autocovariance; need to convert: full ac = E[xy] - mean^2
    # so ac/mean^2 = xi
    return avg / m2, mask.sum()

print()
print("Cell-cell xi(r) comparison:")
print(f"{'r':>6} {'axis':>14} {'diagonal':>14} {'angle avg':>14}  {'#pix in ring':>14}")
for r in [1, 2, 4, 8, 16, 32, 64]:
    a = axis_xi(hist, r)
    d = diag_xi(hist, int(r / np.sqrt(2))) if r > 1 else float('nan')
    ang, npix = annulus_xi(hist, r)
    print(f"{r:>6} {a:>14.5e} {d:>14.5e} {ang:>14.5e} {npix:>14}")

# What about Corrfunc's r=64 = L/4? In a periodic G=256 box with min-image distance,
# the maximum radial distance is G/2 = 128. r=64 is well inside.
# Let's also see angle-binned results
print()
print("Angle decomposition at r=64 (10 degree bins):")
F = np.fft.fft2(hist - hist.mean())
ac = np.real(np.fft.ifft2(np.abs(F)**2)) / hist.size
m2 = hist.mean()**2
G = hist.shape[0]
iy, ix = np.indices((G, G))
dx = np.where(ix < G/2, ix, ix - G)  # signed minimum-image
dy = np.where(iy < G/2, iy, iy - G)
rr = np.sqrt(dx*dx + dy*dy)
theta = np.arctan2(dy, dx) * 180 / np.pi  # in degrees
ring = (rr >= 63.5) & (rr < 64.5)
for t_lo in range(0, 180, 30):
    t_hi = t_lo + 30
    # Wrap theta to [0, 180) (since pair is symmetric)
    th = np.where(theta < 0, theta + 180, theta)
    th = np.where(th >= 180, th - 180, th)
    in_wedge = ring & (th >= t_lo) & (th < t_hi)
    if in_wedge.sum() > 0:
        xi_w = ac[in_wedge].mean() / m2
        print(f"  theta in [{t_lo:>3}, {t_hi:>3}): xi = {xi_w:>12.5e}, n_pix = {in_wedge.sum()}")
