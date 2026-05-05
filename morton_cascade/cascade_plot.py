#!/usr/bin/env python3
"""
Cascade fingerprint plot — single multi-panel summary of one realization.

Layout (3x3 grid, last row spans full width for numerical summary):
  (a) sigma^2(R)              (b) PMF cascade            (c) xi(r) at multiple smoothings
  (d) Spatial map at level 5  (e) Reduced cumulants S_3, S_4
  (f) Schur excess (clustering signal above Poisson floor 13/16)
  Bottom row: numerical summary table
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

DIR = "/tmp/cascade_summary"

ls = pd.read_csv(f"{DIR}/level_stats.csv")
pmf_long = pd.read_csv(f"{DIR}/pmfs.csv")
tpcf = pd.read_csv(f"{DIR}/tpcf.csv")
spatial = pd.read_csv(f"{DIR}/spatial_map_l5.csv")

# Drop level 0 (only 1 cell -> degenerate)
ls = ls[ls['level'] >= 1].reset_index(drop=True)

n_lev = len(ls)
colors = cm.viridis(np.linspace(0.05, 0.95, n_lev))
level_to_color = {int(row['level']): colors[i] for i, (_, row) in enumerate(ls.iterrows())}

fig = plt.figure(figsize=(16, 10.5))
gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.45, height_ratios=[1, 1, 0.7])

# ---------- (a) sigma^2(R) ----------
ax_a = fig.add_subplot(gs[0, 0])
R = ls['R_tree'].values
sigma2 = ls['sigma2_field'].values
ax_a.loglog(R, sigma2, 'o-', color='C0', lw=2, ms=7)
ax_a.set_xlabel(r'smoothing scale $R$ [tree-coord]')
ax_a.set_ylabel(r'$\sigma^2_{\rm field}(R)$')
ax_a.set_title(r'(a) Variance cascade $\sigma^2(R)$', fontsize=11)
ax_a.grid(True, alpha=0.3)
ax_a.invert_xaxis()

# ---------- (b) PMF cascade — rendered as discrete bars at integer N ----------
# P_l(N) is a probability mass function over non-negative integers, NOT a continuous
# density. Render as bars/stems at integer N values.
#
# x-axis: dimensionless centered count.  We use (N - <N>) / sigma to share an axis
# across levels, where sigma here is sqrt(Var(N)) -- a real number used only for the
# x-axis rescaling. The integer structure of N is preserved by drawing one bar per
# integer N value at its rescaled position.
#
# Each level annotated with <N>, sigma (in counts).
ax_b = fig.add_subplot(gs[0, 1])
levels_to_show = sorted(pmf_long['level'].unique())
y_offset = 0
y_step = 1.05
xlim_lo, xlim_hi = -3, 6
n_levels_drawn = 0
for lvl in levels_to_show:
    if lvl == 0: continue
    sub = pmf_long[pmf_long['level'] == lvl]
    n_total = sub['n_total'].iloc[0]
    counts = sub['count'].values.astype(int)
    freqs = sub['frequency'].values / n_total
    mean_l = float(ls.loc[ls['level'] == lvl, 'mean'].iloc[0])
    var_l = float(ls.loc[ls['level'] == lvl, 'var'].iloc[0])
    if var_l <= 0 or mean_l < 1e-9: continue
    std_l = np.sqrt(var_l)
    delta = (counts - mean_l) / std_l   # rescaled positions; bars stay at integer N values
    in_range = (delta >= xlim_lo) & (delta <= xlim_hi)
    if not in_range.any(): continue
    color = level_to_color[lvl]
    p = freqs[in_range]
    p_norm = p / max(p.max(), 1e-30)   # normalize so each level has unit max for visual

    # Bar width: each integer N is sigma_l apart in rescaled units, so width = 1/sigma_l
    # but we use 0.8/sigma_l to leave a small gap.
    bar_w = 0.8 / std_l if std_l > 0.5 else 0.6
    ax_b.bar(delta[in_range], p_norm, width=bar_w,
             bottom=y_offset, color=color, alpha=0.7, edgecolor=color, lw=0.3,
             align='center')

    R_l = float(ls.loc[ls['level'] == lvl, 'R_tree'].iloc[0])
    annot = f'R={R_l:.0f}, $\\langle N\\rangle$={mean_l:.1f}, $\\sigma$={std_l:.1f}'
    # Place annotation in the upper right of each ridgeline, inside the axes
    ax_b.text(xlim_hi - 0.1, y_offset + 0.85, annot,
              fontsize=7, color=color, va='top', ha='right',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax_b.axhline(y_offset, color='gray', lw=0.3, alpha=0.4)
    y_offset += y_step
    n_levels_drawn += 1

# Reference: standard normal density (continuous, for comparison)
xg = np.linspace(xlim_lo, xlim_hi, 200)
g = np.exp(-xg**2/2) / np.sqrt(2*np.pi)
g /= g.max()
ax_b.plot(xg, y_offset + g, color='red', lw=1, alpha=0.7)
ax_b.fill_between(xg, y_offset, y_offset + g, color='red', alpha=0.15)
ax_b.text(xlim_hi - 0.1, y_offset + 0.85, 'N(0,1) ref', fontsize=7, color='red', va='top', ha='right',
          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax_b.set_xlabel(r'$(N - \langle N\rangle) / \sigma$    (bars at integer N)')
ax_b.set_ylabel('PMF (offset by level)')
ax_b.set_title('(b) PMF cascade (discrete)', fontsize=11)
ax_b.set_yticks([])
ax_b.set_xlim(xlim_lo, xlim_hi)
ax_b.axvline(0, color='gray', lw=0.5, alpha=0.5)

# ---------- (c) xi(r) cross-level ----------
ax_c = fig.add_subplot(gs[0, 2])
for lvl in sorted(tpcf['level'].unique()):
    if lvl == 0: continue
    sub = tpcf[tpcf['level'] == lvl].sort_values('r_tree')
    if len(sub) < 2: continue
    color = level_to_color[lvl]
    pos = sub[sub['xi'] > 0]
    R_smooth = float(ls.loc[ls['level'] == lvl, 'R_tree'].iloc[0]) if lvl in ls['level'].values else 0
    ax_c.loglog(pos['r_tree'], pos['xi'], 'o-', color=color, lw=1.0, ms=4, alpha=0.85,
                label=f'$R_s$={R_smooth:.0f}')
ax_c.set_xlabel(r'separation $r$ [tree-coord]')
ax_c.set_ylabel(r'$\xi(r)$')
ax_c.set_title(r'(c) $\xi(r)$ at multiple smoothings', fontsize=11)
ax_c.legend(loc='lower left', fontsize=7, ncol=2, title='smoothing $R_s$')
ax_c.grid(True, alpha=0.3)

# ---------- (d) Spatial map ----------
ax_d = fig.add_subplot(gs[1, 0])
n_per_axis = int(spatial['cy'].max()) + 1
grid = np.zeros((n_per_axis, n_per_axis))
for _, row in spatial.iterrows():
    grid[int(row['cy']), int(row['cx'])] = row['count']
im = ax_d.imshow(grid, origin='lower', cmap='magma', aspect='equal',
                 extent=[0, 256, 0, 256])
ax_d.set_xlabel('x [tree-coord]')
ax_d.set_ylabel('y [tree-coord]')
ax_d.set_title(f'(d) Spatial map at level 5 ({n_per_axis}$\\times${n_per_axis})', fontsize=11)
cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cbar.set_label('count per cell', fontsize=8)

# ---------- (e) Reduced cumulants ----------
ax_e = fig.add_subplot(gs[1, 1])
mean_arr = ls['mean'].values
var_arr  = ls['var'].values
skew_arr = ls['skew'].values
kurt_arr = ls['kurt'].values

# delta = (N - <N>)/<N>; sigma2_delta = var/<N>^2; S_p = <delta^p>_c / <delta^2>^{p-1}
sigma2_delta = var_arr / np.maximum(mean_arr, 1e-12)**2
sigma_delta = np.sqrt(np.maximum(sigma2_delta, 1e-30))
S3 = skew_arr / np.maximum(sigma_delta, 1e-12)
S4 = (kurt_arr - 3) / np.maximum(sigma2_delta, 1e-12)

ax_e.semilogx(R, S3, 'o-', color='C2', lw=2, label=r'$S_3$')
ax_e2 = ax_e.twinx()
ax_e2.semilogx(R, S4, 's--', color='C3', lw=2, label=r'$S_4 - 3$')
ax_e.set_xlabel(r'smoothing scale $R$')
ax_e.set_ylabel(r'$S_3$', color='C2')
ax_e2.set_ylabel(r'$S_4 - 3$', color='C3')
ax_e.set_title(r'(e) Reduced cumulants $S_p$ vs scale', fontsize=11)
ax_e.tick_params(axis='y', labelcolor='C2')
ax_e2.tick_params(axis='y', labelcolor='C3')
ax_e.grid(True, alpha=0.3)
ax_e.invert_xaxis()
ax_e2.invert_xaxis()

# ---------- (f) Variance increment per scale (Wilsonian decomposition) ----------
# For each level transition l-1 -> l (going to finer scales), the variance
# contribution is Δσ²(l) = σ²_field(R_l) - σ²_field(R_{l-1}).
# By construction Σ_l Δσ²(l) = σ²_field(R_min) - σ²_field(R_max), the total
# field variance gained going from coarsest to finest cell.
#
# This is the meaningful "scale-localized variance" — comparable across scales
# because each Δσ² is dimensionless variance per cell, normalized by ⟨N⟩².
ax_f = fig.add_subplot(gs[1, 2])
sigma2_arr = ls['sigma2_field'].values   # in order from coarsest (l=1) to finest (l=L_max)
R_arr = ls['R_tree'].values
# Increments: σ²(R_l) - σ²(R_{l-1}); element 0 is coarsest, no prior level.
# We plot increments for l=2..L_max (against the finer scale R_l).
incrs = np.diff(sigma2_arr)   # length n_lev - 1
R_for_incr = R_arr[1:]        # the finer scale of each pair
level_for_incr = ls['level'].values[1:]
total_field_var = sigma2_arr[-1] - sigma2_arr[0]

# Bar chart at the FINER scale of each pair, x is decreasing R (going right -> finer)
# Use level numbers as x positions for clarity
xpos = np.arange(len(incrs))
bars = ax_f.bar(xpos, incrs,
                color=[level_to_color[int(l)] for l in level_for_incr],
                edgecolor='black', lw=0.4, alpha=0.85)
# Annotate fraction of total
for j, (xp, val) in enumerate(zip(xpos, incrs)):
    frac = val / max(total_field_var, 1e-30)
    if frac > 0.02:
        ax_f.text(xp, val * 1.02, f'{frac*100:.0f}%',
                  ha='center', va='bottom', fontsize=8, color='black')
ax_f.set_xticks(xpos)
ax_f.set_xticklabels([f'{R_arr[i+1]:.0f}\n→{R_arr[i]:.0f}' for i in range(len(incrs))],
                     fontsize=8)
ax_f.set_xlabel(r'scale transition $R_l \to R_{l-1}$')
ax_f.set_ylabel(r'$\Delta\sigma^2_{\rm field}(l)$')
ax_f.set_title(r'(f) Variance increment per scale (RG-style)', fontsize=11)
ax_f.grid(True, alpha=0.3, axis='y')
ax_f.text(0.98, 0.95,
          f'Total: $\\sigma^2$ from\n{sigma2_arr[0]:.3f} → {sigma2_arr[-1]:.3f}\n'
          f'$\\Delta_{{\\rm tot}} = {total_field_var:.3f}$',
          transform=ax_f.transAxes, ha='right', va='top', fontsize=9,
          bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

# ---------- numerical summary table ----------
ax_g = fig.add_subplot(gs[2, :])
ax_g.axis('off')
total_pts = int(np.round(ls.iloc[-1]['mean'] * (1 << (2*int(ls.iloc[-1]['level'])))))
header = f"Cascade summary — N = {total_pts:,} points, 2D periodic box L = 256 (tree-coord)"
lines = [header, ""]
lines.append(f"{'l':>3}  {'R':>5}  {'cells':>7}  {'<N>':>10}  {'sigma2':>9}  "
             f"{'S_3':>8}  {'S_4-3':>8}  {'Schur exc.':>11}  {'xi(r=R)':>9}")
lines.append("-" * 90)
for i, (_, row) in enumerate(ls.iterrows()):
    if row['mean'] < 1e-9: continue
    lvl = int(row['level'])
    rs_excess = (row['dvar']/max(row['mean'], 1e-12)) - 13/16
    R_l = row['R_tree']
    sub = tpcf[(tpcf['level'] == lvl) & (np.isclose(tpcf['r_tree'], R_l, atol=0.5))]
    xi_at_R = float(sub['xi'].iloc[0]) if len(sub) > 0 else float('nan')
    lines.append(f"{lvl:>3}  {row['R_tree']:>5.0f}  {int(row['n_cells']):>7d}  "
                 f"{row['mean']:>10.2f}  {row['sigma2_field']:>9.4f}  "
                 f"{S3[i]:>8.3f}  {S4[i]:>8.3f}  {rs_excess:>11.3f}  {xi_at_R:>9.4f}")
ax_g.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9.5,
          verticalalignment='top', transform=ax_g.transAxes)

fig.suptitle('Cascade fingerprint — 2D Cox process, single realization',
             y=0.995, fontsize=13, weight='bold')

out_path = '/home/claude/morton_cascade/cascade_fingerprint.png'
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"Saved: {out_path}")
