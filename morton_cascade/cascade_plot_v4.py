#!/usr/bin/env python3
"""
Cascade fingerprint v4 — single realization, intra-realization errors.

Design constraints:
  - Only one realization is available (typical observational scenario)
  - Coarse levels have intrinsically few distinct cells -> high moment errors
  - Fine levels are statistically clean
  - Plot must communicate per-level trust without averaging

Strategy:
  - Error bars from analytic Gaussian approximation: sigma_var ~ var * sqrt(2/N_eff),
    sigma_skew ~ sqrt(6/N_eff), sigma_kurt ~ sqrt(24/N_eff), where N_eff is the
    number of distinct cells at that level.
  - For panels showing higher moments: drop or fade levels with N_eff < 100.
  - PMF panel: only show levels with enough distinct cells for a meaningful histogram.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.patches import Rectangle

DIR = "/tmp/cascade_summary"

ls = pd.read_csv(f"{DIR}/level_stats.csv")
pmf_long = pd.read_csv(f"{DIR}/pmfs.csv")
tpcf = pd.read_csv(f"{DIR}/tpcf.csv")
spatial = pd.read_csv(f"{DIR}/spatial_map_l5.csv")

# Drop level 0 (single cell -> degenerate)
ls = ls[ls['level'] >= 1].reset_index(drop=True)

# Trust threshold: levels with < 100 distinct cells get faded/dropped from moment plots
TRUST_MIN_CELLS = 100   # below this, moments are unreliable
PMF_MIN_CELLS = 32      # below this, PMF histogram has too few bins

n_lev = len(ls)
colors = cm.viridis(np.linspace(0.05, 0.95, n_lev))
level_to_color = {int(row['level']): colors[i] for i, (_, row) in enumerate(ls.iterrows())}

# Helper: which levels are trusted for moment plots, which for PMF
trust_mask = ls['n_cells'] >= TRUST_MIN_CELLS
pmf_mask   = ls['n_cells'] >= PMF_MIN_CELLS
print(f"Levels: {n_lev}; trusted for moments (≥{TRUST_MIN_CELLS} cells): {trust_mask.sum()}; "
      f"trusted for PMF (≥{PMF_MIN_CELLS} cells): {pmf_mask.sum()}")

fig = plt.figure(figsize=(15, 10.5))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32, height_ratios=[1, 1, 0.7])

# ---------- (a) sigma^2(R) with error bars ----------
ax = fig.add_subplot(gs[0, 0])
R = ls['R_tree'].values
sigma2 = ls['sigma2_field'].values
sigma2_err = ls['sigma2_err'].values
ax.errorbar(R, sigma2, yerr=sigma2_err, fmt='o-', color='C0', lw=1.8, ms=6, capsize=3)
# Shade untrusted region (coarse levels with few cells)
untrusted_R = ls.loc[~trust_mask, 'R_tree'].values
if len(untrusted_R) > 0:
    R_min_untrusted = untrusted_R.min()
    R_max_untrusted = untrusted_R.max()
    ax.axvspan(R_min_untrusted * 0.7, R_max_untrusted * 1.4, alpha=0.1, color='red',
               label=f'few cells ($N_\\text{{eff}}<{TRUST_MIN_CELLS}$)')
    ax.legend(fontsize=8, loc='lower right')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'smoothing scale $R$ [tree-coord]')
ax.set_ylabel(r'$\sigma^2_{\rm field}(R)$')
ax.set_title(r'(a) Variance cascade $\sigma^2(R)$', fontsize=11)
ax.grid(True, alpha=0.3); ax.invert_xaxis()

# ---------- (b) PMF cascade — only well-sampled levels ----------
ax = fig.add_subplot(gs[0, 1])
levels_for_pmf = sorted(ls.loc[pmf_mask, 'level'].astype(int).tolist())
n_show = len(levels_for_pmf)
y_offset = 0
y_step = 1.05
edges = np.linspace(-3, 6, 80)
centers = 0.5 * (edges[:-1] + edges[1:])

for lvl in levels_for_pmf:
    color = level_to_color[lvl]
    sub = pmf_long[pmf_long['level'] == lvl]
    if len(sub) == 0: continue
    n_total = sub['n_total'].iloc[0]
    counts = sub['count'].values.astype(float)
    freqs = sub['frequency'].values / n_total
    mean_l = float(ls.loc[ls['level'] == lvl, 'mean'].iloc[0])
    var_l = float(ls.loc[ls['level'] == lvl, 'var'].iloc[0])
    if var_l <= 0 or mean_l < 1e-9: continue
    std_l = np.sqrt(var_l)
    delta = (counts - mean_l) / std_l
    rebinned = np.zeros_like(centers)
    for d, weight in zip(delta, freqs):
        j = np.searchsorted(edges, d) - 1
        if 0 <= j < len(centers): rebinned[j] += weight
    rebinned /= max(rebinned.max(), 1e-30)
    ax.fill_between(centers, y_offset, y_offset + rebinned, color=color, alpha=0.5, lw=0)
    ax.plot(centers, y_offset + rebinned, color=color, lw=1.0)
    R_l = float(ls.loc[ls['level'] == lvl, 'R_tree'].iloc[0])
    n_cells_l = int(ls.loc[ls['level'] == lvl, 'n_cells'].iloc[0])
    ax.text(6.0, y_offset + 0.5, f'R={R_l:.0f}\n({n_cells_l} cells)',
            fontsize=7.5, color=color, va='center')
    y_offset += y_step

# Standard normal reference at top
xg = np.linspace(-3, 6, 200)
g = np.exp(-xg**2/2) / np.sqrt(2*np.pi)
g /= g.max()
ax.plot(xg, y_offset + g, color='red', lw=1.2, alpha=0.8)
ax.fill_between(xg, y_offset, y_offset + g, color='red', alpha=0.15)
ax.text(6.0, y_offset + 0.5, 'N(0,1)', fontsize=8, color='red', va='center')

ax.set_xlabel(r'$(N - \langle N\rangle) / \sigma$')
ax.set_ylabel('PMF (offset by level)')
ax.set_title(f'(b) PMF cascade — {n_show} well-sampled levels', fontsize=11)
ax.set_yticks([])
ax.set_xlim(-3, 7.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)

# ---------- (c) xi(r) ----------
ax = fig.add_subplot(gs[0, 2])
for lvl in sorted(tpcf['level'].unique()):
    if lvl == 0: continue
    sub = tpcf[tpcf['level'] == lvl].sort_values('r_tree')
    if len(sub) < 2: continue
    color = level_to_color[lvl]
    pos = sub[sub['xi'] > 0]
    R_smooth = float(ls.loc[ls['level'] == lvl, 'R_tree'].iloc[0]) if lvl in ls['level'].values else 0
    # sqrt(n_pairs) error on xi
    xi_err = pos['xi'] / np.sqrt(np.maximum(pos['n_pairs'].values, 1))
    ax.errorbar(pos['r_tree'], pos['xi'], yerr=xi_err, fmt='o-', color=color, lw=1.0, ms=4,
                alpha=0.85, label=f'$R_s$={R_smooth:.0f}', capsize=2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'separation $r$ [tree-coord]')
ax.set_ylabel(r'$\xi(r)$')
ax.set_title(r'(c) $\xi(r)$ at multiple smoothings', fontsize=11)
ax.legend(loc='lower left', fontsize=7, ncol=2, title='smoothing $R_s$')
ax.grid(True, alpha=0.3)

# ---------- (d) Spatial map ----------
ax = fig.add_subplot(gs[1, 0])
n_per_axis = int(spatial['cy'].max()) + 1
grid = np.zeros((n_per_axis, n_per_axis))
for _, row in spatial.iterrows():
    grid[int(row['cy']), int(row['cx'])] = row['count']
im = ax.imshow(grid, origin='lower', cmap='magma', aspect='equal', extent=[0, 256, 0, 256])
ax.set_xlabel('x [tree-coord]'); ax.set_ylabel('y [tree-coord]')
ax.set_title(f'(d) Spatial map at level 5 ({n_per_axis}$\\times${n_per_axis})', fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('count per cell', fontsize=8)

# ---------- (e) Reduced cumulants — drop untrusted, show error bars ----------
ax = fig.add_subplot(gs[1, 1])
trusted = ls[trust_mask].copy()
mean_arr = trusted['mean'].values
var_arr  = trusted['var'].values
skew_arr = trusted['skew'].values
kurt_arr = trusted['kurt'].values
sigma2_delta = var_arr / np.maximum(mean_arr, 1e-12)**2
sigma_delta = np.sqrt(np.maximum(sigma2_delta, 1e-30))
S3 = skew_arr / np.maximum(sigma_delta, 1e-12)
S4 = (kurt_arr - 3) / np.maximum(sigma2_delta, 1e-12)
S3_err = trusted['skew_err'].values / np.maximum(sigma_delta, 1e-12)
S4_err = trusted['kurt_err'].values / np.maximum(sigma2_delta, 1e-12)
R_trust = trusted['R_tree'].values

ax.errorbar(R_trust, S3, yerr=S3_err, fmt='o-', color='C2', lw=1.8, ms=6,
            label=r'$S_3$', capsize=3)
ax2 = ax.twinx()
ax2.errorbar(R_trust, S4, yerr=S4_err, fmt='s--', color='C3', lw=1.8, ms=6,
             label=r'$S_4-3$', capsize=3)
# Reference: 0 (Gaussian)
ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.7)
ax2.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel(r'smoothing scale $R$')
ax.set_ylabel(r'$S_3$', color='C2')
ax2.set_ylabel(r'$S_4-3$', color='C3')
ax.set_title(f'(e) Reduced cumulants ($N_\\text{{eff}} \\geq {TRUST_MIN_CELLS}$ only)', fontsize=11)
ax.tick_params(axis='y', labelcolor='C2')
ax2.tick_params(axis='y', labelcolor='C3')
ax.grid(True, alpha=0.3)
ax.invert_xaxis(); ax2.invert_xaxis()

# ---------- (f) Schur additive decomposition ----------
ax = fig.add_subplot(gs[1, 2])
finest_var = ls.iloc[-1]['var']
contributions = ls['dvar'].values / max(finest_var, 1e-12)
# Use sigma2_err as a proxy for dvar_err (rough)
contributions_err = (ls['sigma2_err'].values * ls['mean'].values**2) / max(finest_var, 1e-12)

bar_colors = [colors[i] for i in range(len(R))]
ax.bar(np.arange(len(R)), contributions, color=bar_colors,
       edgecolor='black', lw=0.5, alpha=0.85)
ax.errorbar(np.arange(len(R)), contributions, yerr=contributions_err,
            fmt='none', ecolor='black', capsize=2)
# Mark untrusted bars with hatching
for i, trustme in enumerate(trust_mask):
    if not trustme:
        ax.bar(i, contributions[i], color='none', edgecolor='red', lw=1.5,
               hatch='///', alpha=1.0)
ax.set_xticks(np.arange(len(R)))
ax.set_xticklabels([f'{r:.0f}' for r in R], fontsize=8)
ax.set_xlabel(r'level / smoothing scale $R$')
ax.set_ylabel(r'$\Delta V_l / {\rm Var}(N)_{\rm finest}$')
ax.set_title('(f) Schur variance decomposition', fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# ---------- (g) Numerical summary ----------
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
total_pts = int(np.round(ls.iloc[-1]['mean'] * (1 << (2*int(ls.iloc[-1]['level'])))))
header = f"Cascade fingerprint — N = {total_pts:,} points (one realization), 2D periodic L=256 (tree-coord)"
lines = [header, ""]
lines.append(f"{'l':>3}  {'R':>5}  {'cells':>6}  {'<N>':>10}  "
             f"{'sigma^2':>16}  {'S_3':>13}  {'S_4-3':>13}  {'trusted?':>10}")
lines.append("-" * 95)
sigma2_arr = ls['sigma2_field'].values
sigma2_err_arr = ls['sigma2_err'].values

# Build rows for ALL levels (not just trusted) but mark trust status
mean_full = ls['mean'].values
var_full = ls['var'].values
skew_full = ls['skew'].values
kurt_full = ls['kurt'].values
sigma2_delta_full = var_full / np.maximum(mean_full, 1e-12)**2
sigma_delta_full = np.sqrt(np.maximum(sigma2_delta_full, 1e-30))
S3_full = skew_full / np.maximum(sigma_delta_full, 1e-12)
S4_full = (kurt_full - 3) / np.maximum(sigma2_delta_full, 1e-12)
S3_err_full = ls['skew_err'].values / np.maximum(sigma_delta_full, 1e-12)
S4_err_full = ls['kurt_err'].values / np.maximum(sigma2_delta_full, 1e-12)

for i, (_, row) in enumerate(ls.iterrows()):
    if row['mean'] < 1e-9: continue
    s2_str = f"{sigma2_arr[i]:.4f}±{sigma2_err_arr[i]:.4f}"
    s3_str = f"{S3_full[i]:.3f}±{S3_err_full[i]:.3f}"
    s4_str = f"{S4_full[i]:.3f}±{S4_err_full[i]:.3f}"
    trust_str = "yes" if trust_mask.iloc[i] else "NO (low N)"
    lines.append(f"{int(row['level']):>3}  {row['R_tree']:>5.0f}  {int(row['n_cells']):>6d}  "
                 f"{row['mean']:>10.2f}  {s2_str:>16}  {s3_str:>13}  {s4_str:>13}  {trust_str:>10}")
ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9,
        verticalalignment='top', transform=ax.transAxes)

fig.suptitle('Cascade fingerprint — single realization, intra-realization errors',
             y=0.995, fontsize=13, weight='bold')

out_path = '/home/claude/morton_cascade/cascade_fingerprint_v4.png'
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"Saved: {out_path}")
