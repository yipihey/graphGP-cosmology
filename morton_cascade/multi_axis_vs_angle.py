#!/usr/bin/env python3
"""
Multi-realization: average axis_xi across realizations and compare to Corrfunc's
angle-averaged xi across the same realizations. For an isotropic field these
should agree in the realization mean.
"""
import numpy as np
import glob

BOXSIZE = 256.0
G = 256

files = sorted(glob.glob("/tmp/cascade_xi_run/points_*.bin"))
print(f"Found {len(files)} realizations")

axis_acc = {r: [] for r in [1, 2, 4, 8, 16, 32, 64]}
angle_acc = {r: [] for r in [1, 2, 4, 8, 16, 32, 64]}

for fp in files:
    raw = np.fromfile(fp, dtype=np.float64)
    pts = raw.reshape(-1, 2)
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1],
                                 bins=[np.linspace(0, BOXSIZE, G+1)] * 2)
    m = hist.mean()

    # Axis-aligned xi
    for r in axis_acc.keys():
        sx = np.roll(hist, r, axis=1)
        sy = np.roll(hist, r, axis=0)
        avg = 0.5 * (np.mean(hist * sx) + np.mean(hist * sy))
        axis_acc[r].append(avg / (m * m) - 1.0)

    # Angle-averaged via FFT-based autocorrelation
    F = np.fft.fft2(hist - m)
    ac = np.real(np.fft.ifft2(np.abs(F)**2)) / hist.size
    iy, ix = np.indices((G, G))
    dx = np.minimum(ix, G - ix)
    dy = np.minimum(iy, G - iy)
    rr = np.sqrt(dx*dx + dy*dy)
    for r in angle_acc.keys():
        mask = (rr >= r - 0.5) & (rr < r + 0.5)
        if mask.sum() > 0:
            angle_acc[r].append(ac[mask].mean() / (m * m))

print()
print(f"Averaged over {len(files)} realizations:")
print(f"{'r':>4} {'<axis>':>14} {'sd(axis)':>14} {'<angle>':>14} {'sd(angle)':>14} {'rel diff':>12}")
for r in sorted(axis_acc.keys()):
    a = np.array(axis_acc[r])
    g = np.array(angle_acc[r])
    ma, sa = a.mean(), a.std() / np.sqrt(len(a))
    mg, sg = g.mean(), g.std() / np.sqrt(len(g))
    rel = (ma - mg) / mg if abs(mg) > 1e-12 else float('nan')
    print(f"{r:>4} {ma:>14.5e} {sa:>14.5e} {mg:>14.5e} {sg:>14.5e} {rel:>12.4f}")
