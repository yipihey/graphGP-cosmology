#!/usr/bin/env python3
"""
Cascade fingerprint v3: multi-realization with Poisson reference.

For each panel, show:
  - Cox field: solid line/curve, with shaded band = ±1σ across realizations
  - Poisson reference: dashed line/curve, with shaded band

The deviation between the two is the cosmological signal at every scale and
every observable. This is the diagnostic plot for "is this field clustered,
non-Gaussian, scale-dependent?" — answers all three at a glance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

DIR = "/tmp/cascade_summary"

ls_cox = pd.read_csv(f"{DIR}/level_stats_cox.csv")
ls_ref = pd.read_csv(f"{DIR}/level_stats_ref.csv")
pmf_cox = pd.read_csv(f"{DIR}/pmfs_cox.csv")
pmf_ref = pd.read_csv(f"{DIR}/pmfs_ref.csv")
tp_cox = pd.read_csv(f"{DIR}/tpcf_cox.csv")
tp_ref = pd.read_csv(f"{DIR}/tpcf_ref.csv")
spatial = pd.read_csv(f"{DIR}/spatial_map_l5.csv")

n_real = ls_cox['realization'].nunique()
print(f"Loaded {n_real} realizations of Cox + Poisson reference")

# Drop level 0
ls_cox = ls_cox[ls_cox['level'] >= 1].copy()
ls_ref = ls_ref[ls_ref['level'] >= 1].copy()

def aggregate(df, group_keys, value_keys):
    """Mean and std over realizations."""
    g = df.groupby(group_keys)[value_keys].agg(['mean', 'std']).reset_index()
    return g

# Aggregate stats by level (across realizations)
agg_cox = ls_cox.groupby('level').agg(
    R_tree=('R_tree', 'first'),
    mean_avg=('mean', 'mean'), mean_std=('mean', 'std'),
    var_avg=('var', 'mean'), var_std=('var', 'std'),
    dvar_avg=('dvar', 'mean'), dvar_std=('dvar', 'std'),
    sigma2_avg=('sigma2_field', 'mean'), sigma2_std=('sigma2_field', 'std'),
    skew_avg=('skew', 'mean'), skew_std=('skew', 'std'),
    kurt_avg=('kurt', 'mean'), kurt_std=('kurt', 'std'),
).reset_index()

agg_ref = ls_ref.groupby('level').agg(
    R_tree=('R_tree', 'first'),
    mean_avg=('mean', 'mean'), mean_std=('mean', 'std'),
    var_avg=('var', 'mean'), var_std=('var', 'std'),
    dvar_avg=('dvar', 'mean'), dvar_std=('dvar', 'std'),
    sigma2_avg=('sigma2_field', 'mean'), sigma2_std=('sigma2_field', 'std'),
    skew_avg=('skew', 'mean'), skew_std=('skew', 'std'),
    kurt_avg=('kurt', 'mean'), kurt_std=('kurt', 'std'),
).reset_index()

n_lev = len(agg_cox)
colors = cm.viridis(np.linspace(0.05, 0.95, n_lev))

fig = plt.figure(figsize=(15, 10.5))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32, height_ratios=[1, 1, 0.7])

def plot_band(ax, x, y_avg, y_std, color, label, marker='o', ls='-'):
    ax.plot(x, y_avg, marker=marker, color=color, lw=1.8, ms=6, label=label, ls=ls)
    ax.fill_between(x, y_avg - y_std, y_avg + y_std, color=color, alpha=0.2)

# ---------- (a) sigma^2(R) ----------
ax = fig.add_subplot(gs[0, 0])
R_c = agg_cox['R_tree'].values
R_r = agg_ref['R_tree'].values
plot_band(ax, R_c, agg_cox['sigma2_avg'].values, agg_cox['sigma2_std'].values,
          'C0', 'Cox field', marker='o', ls='-')
plot_band(ax, R_r, np.maximum(agg_ref['sigma2_avg'].values, 1e-8),
          np.maximum(agg_ref['sigma2_std'].values, 1e-9),
          'gray', 'Poisson ref', marker='s', ls='--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'smoothing scale $R$ [tree-coord]')
ax.set_ylabel(r'$\sigma^2_{\rm field}(R)$')
ax.set_title(r'(a) Variance cascade', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.invert_xaxis()

# ---------- (b) PMF cascade ----------
# Stack PMFs of Cox vs Poisson at each level. Use standardized (N - <N>)/sigma.
ax = fig.add_subplot(gs[0, 1])
levels_to_show = sorted(pmf_cox['level'].unique())
y_offset = 0
y_step = 1.05
edges = np.linspace(-3, 6, 80)
centers = 0.5 * (edges[:-1] + edges[1:])

for lvl in levels_to_show:
    if lvl == 0: continue
    color = colors[lvl - 1]
    # Cox: aggregate frequency over realizations
    sub_c = pmf_cox[pmf_cox['level'] == lvl]
    sub_r = pmf_ref[pmf_ref['level'] == lvl]
    if len(sub_c) == 0 or len(sub_r) == 0: continue

    # Sum frequencies across realizations, divide by total entries
    n_total_c = sub_c.groupby('realization')['n_total'].first().sum()
    n_total_r = sub_r.groupby('realization')['n_total'].first().sum()

    # Cox mean+std for standardization
    mean_c = float(agg_cox.loc[agg_cox['level']==lvl, 'mean_avg'].iloc[0])
    var_c = float(agg_cox.loc[agg_cox['level']==lvl, 'var_avg'].iloc[0])
    if var_c <= 0 or mean_c < 1e-9: continue
    std_c = np.sqrt(var_c)
    mean_r = float(agg_ref.loc[agg_ref['level']==lvl, 'mean_avg'].iloc[0])
    var_r = float(agg_ref.loc[agg_ref['level']==lvl, 'var_avg'].iloc[0])
    std_r = np.sqrt(max(var_r, 1e-30))

    # Bin Cox PMF
    rebinned_c = np.zeros_like(centers)
    for _, row in sub_c.iterrows():
        d = (row['count'] - mean_c) / std_c
        j = np.searchsorted(edges, d) - 1
        if 0 <= j < len(centers): rebinned_c[j] += row['frequency']
    rebinned_c /= max(n_total_c, 1)
    # Normalize peak to 1 for visual stacking
    if rebinned_c.max() > 0: rebinned_c /= rebinned_c.max()

    # Bin Poisson PMF (using its own mean/std)
    rebinned_r = np.zeros_like(centers)
    for _, row in sub_r.iterrows():
        d = (row['count'] - mean_r) / std_r
        j = np.searchsorted(edges, d) - 1
        if 0 <= j < len(centers): rebinned_r[j] += row['frequency']
    rebinned_r /= max(n_total_r, 1)
    if rebinned_r.max() > 0: rebinned_r /= rebinned_r.max()

    # Cox: filled
    ax.fill_between(centers, y_offset, y_offset + rebinned_c, color=color, alpha=0.5, lw=0)
    ax.plot(centers, y_offset + rebinned_c, color=color, lw=1.0)
    # Poisson reference: dashed outline only
    ax.plot(centers, y_offset + rebinned_r, color='black', lw=1.2, ls=(0, (3, 2)), alpha=0.85)

    R_l = float(agg_cox.loc[agg_cox['level']==lvl, 'R_tree'].iloc[0])
    ax.text(6.0, y_offset + 0.5, f'R={R_l:.0f}', fontsize=8, color=color, va='center')
    y_offset += y_step

ax.set_xlabel(r'$(N - \langle N\rangle) / \sigma$')
ax.set_ylabel('PMF (offset by level)')
ax.set_title('(b) PMF cascade — Cox (filled) vs Poisson (dashed)', fontsize=11)
ax.set_yticks([])
ax.set_xlim(-3, 7.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)

# ---------- (c) xi(r) ----------
ax = fig.add_subplot(gs[0, 2])
agg_tp_cox = tp_cox.groupby(['level', 'k']).agg(
    r_tree=('r_tree', 'first'),
    smoothing_h_fine=('smoothing_h_fine', 'first'),
    xi_avg=('xi', 'mean'), xi_std=('xi', 'std'),
).reset_index()
agg_tp_ref = tp_ref.groupby(['level', 'k']).agg(
    r_tree=('r_tree', 'first'),
    smoothing_h_fine=('smoothing_h_fine', 'first'),
    xi_avg=('xi', 'mean'), xi_std=('xi', 'std'),
).reset_index()

for lvl in sorted(agg_tp_cox['level'].unique()):
    if lvl == 0: continue
    sub = agg_tp_cox[agg_tp_cox['level'] == lvl].sort_values('r_tree')
    pos = sub[sub['xi_avg'] > 0]
    if len(pos) < 2: continue
    color = colors[lvl - 1]
    R_smooth = float(agg_cox.loc[agg_cox['level']==lvl, 'R_tree'].iloc[0])
    ax.plot(pos['r_tree'], pos['xi_avg'], 'o-', color=color, lw=1.0, ms=4, alpha=0.85,
            label=f'$R_s$={R_smooth:.0f}')
    ax.fill_between(pos['r_tree'], pos['xi_avg'] - pos['xi_std'], pos['xi_avg'] + pos['xi_std'],
                    color=color, alpha=0.15)

# Poisson reference: |xi| should be ~0; show its 1-sigma envelope as gray band
# (this is the noise floor below which Cox xi cannot be measured)
ref_spread = []
for lvl in sorted(agg_tp_ref['level'].unique()):
    if lvl == 0: continue
    sub = agg_tp_ref[agg_tp_ref['level'] == lvl].sort_values('r_tree')
    ref_spread.append(sub[['r_tree', 'xi_std']].values)
if ref_spread:
    all_r = np.concatenate([a[:, 0] for a in ref_spread])
    all_s = np.concatenate([a[:, 1] for a in ref_spread])
    # Bin by r and take max sigma across smoothings
    r_unique = sorted(set(all_r))
    s_max = [max(all_s[all_r == r]) for r in r_unique]
    ax.plot(r_unique, s_max, 'k:', lw=1, label='Poisson noise floor (1σ)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'separation $r$ [tree-coord]')
ax.set_ylabel(r'$\xi(r)$')
ax.set_title(r'(c) $\xi(r)$ — Cox signal vs Poisson floor', fontsize=11)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# ---------- (d) Spatial map ----------
ax = fig.add_subplot(gs[1, 0])
n_per_axis = int(spatial['cy'].max()) + 1
grid = np.zeros((n_per_axis, n_per_axis))
for _, row in spatial.iterrows():
    grid[int(row['cy']), int(row['cx'])] = row['count']
im = ax.imshow(grid, origin='lower', cmap='magma', aspect='equal',
               extent=[0, 256, 0, 256])
ax.set_xlabel('x [tree-coord]'); ax.set_ylabel('y [tree-coord]')
ax.set_title(f'(d) Spatial map at level 5 ({n_per_axis}$\\times${n_per_axis})', fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('count per cell', fontsize=8)

# ---------- (e) Reduced cumulants ----------
ax = fig.add_subplot(gs[1, 1])
mean_c = agg_cox['mean_avg'].values
var_c = agg_cox['var_avg'].values
sigma2_delta_c = var_c / np.maximum(mean_c, 1e-12)**2
sigma_delta_c = np.sqrt(np.maximum(sigma2_delta_c, 1e-30))
S3_c = agg_cox['skew_avg'].values / np.maximum(sigma_delta_c, 1e-12)
S4_c = (agg_cox['kurt_avg'].values - 3) / np.maximum(sigma2_delta_c, 1e-12)
# error propagation: rough
S3_err = agg_cox['skew_std'].values / np.maximum(sigma_delta_c, 1e-12)
S4_err = agg_cox['kurt_std'].values / np.maximum(sigma2_delta_c, 1e-12)

# Reference (Poisson): S_3, S_4 should be ~0 (after Poisson subtraction)
mean_r = agg_ref['mean_avg'].values
var_r = agg_ref['var_avg'].values
sigma2_delta_r = var_r / np.maximum(mean_r, 1e-12)**2
sigma_delta_r = np.sqrt(np.maximum(sigma2_delta_r, 1e-30))
S3_r = agg_ref['skew_avg'].values / np.maximum(sigma_delta_r, 1e-12)
S3_r_err = agg_ref['skew_std'].values / np.maximum(sigma_delta_r, 1e-12)

ax.errorbar(R_c, S3_c, yerr=S3_err, fmt='o-', color='C2', lw=1.8, ms=5,
            label=r'$S_3$ (Cox)', capsize=2)
ax.errorbar(R_r, S3_r, yerr=S3_r_err, fmt='s--', color='gray', lw=1, ms=4, alpha=0.7,
            label=r'$S_3$ (Poisson)', capsize=2)
ax2 = ax.twinx()
ax2.errorbar(R_c, S4_c, yerr=S4_err, fmt='^-', color='C3', lw=1.8, ms=5,
             label=r'$S_4-3$ (Cox)', capsize=2)
ax.set_xscale('log')
ax.set_xlabel(r'smoothing scale $R$')
ax.set_ylabel(r'$S_3$', color='C2')
ax2.set_ylabel(r'$S_4-3$', color='C3')
ax.set_title(r'(e) Reduced cumulants vs scale', fontsize=11)
ax.tick_params(axis='y', labelcolor='C2')
ax2.tick_params(axis='y', labelcolor='C3')
# Clip outliers — S_4 at coarsest level is unreliable from low cell count
ax.set_ylim(-0.5, 5)
ax2.set_ylim(-5, 7)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)
ax.invert_xaxis(); ax2.invert_xaxis()

# ---------- (f) Schur additive decomposition ----------
ax = fig.add_subplot(gs[1, 2])
finest_var_c = agg_cox['var_avg'].iloc[-1]
finest_var_r = agg_ref['var_avg'].iloc[-1]
contrib_c = agg_cox['dvar_avg'].values / max(finest_var_c, 1e-12)
contrib_r = agg_ref['dvar_avg'].values / max(finest_var_r, 1e-12)
contrib_c_err = agg_cox['dvar_std'].values / max(finest_var_c, 1e-12)

x = np.arange(len(R_c))
width = 0.4
ax.bar(x - width/2, contrib_c, width, color=[colors[i] for i in range(len(R_c))],
       edgecolor='black', lw=0.5, alpha=0.85, label='Cox')
ax.bar(x + width/2, contrib_r, width, color='gray',
       edgecolor='black', lw=0.5, alpha=0.5, label='Poisson')
ax.errorbar(x - width/2, contrib_c, yerr=contrib_c_err, fmt='none', ecolor='black', capsize=2)
ax.set_xticks(x)
ax.set_xticklabels([f'{r:.0f}' for r in R_c], fontsize=8)
ax.set_xlabel(r'level / smoothing scale $R$')
ax.set_ylabel(r'$\Delta V_l / {\rm Var}(N)_{\rm finest}$')
ax.set_title(r'(f) Schur variance decomposition', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# ---------- (g) Summary table ----------
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
total_pts = int(np.round(agg_cox.iloc[-1]['mean_avg'] * (1 << (2*int(agg_cox.iloc[-1]['level'])))))
header = (f"Cascade fingerprint — N = {total_pts:,} points/realization, "
          f"{n_real} realizations, 2D periodic L=256 (tree-coord)")
lines = [header, ""]
lines.append(f"{'l':>3}  {'R':>5}  {'<N>':>10}  "
             f"{'σ²(Cox)':>15}   {'σ²(Pois)':>15}   "
             f"{'S_3(Cox)':>14}   {'S_4-3(Cox)':>14}   "
             f"{'ΔV/Var(fine)':>14}")
lines.append("-" * 110)
for i, (_, row) in enumerate(agg_cox.iterrows()):
    if row['mean_avg'] < 1e-9: continue
    ref_row = agg_ref.iloc[i]
    s2_c = f"{row['sigma2_avg']:.4f}±{row['sigma2_std']:.4f}"
    s2_r = f"{ref_row['sigma2_avg']:.5f}±{ref_row['sigma2_std']:.5f}"
    S3str = f"{S3_c[i]:.3f}±{S3_err[i]:.3f}"
    S4str = f"{S4_c[i]:.3f}±{S4_err[i]:.3f}"
    contrib = f"{contrib_c[i]:.4f}±{contrib_c_err[i]:.4f}"
    lines.append(f"{int(row['level']):>3}  {row['R_tree']:>5.0f}  "
                 f"{row['mean_avg']:>10.2f}  {s2_c:>15}   {s2_r:>15}   "
                 f"{S3str:>14}   {S4str:>14}   {contrib:>14}")
ax.text(0.0, 1.0, "\n".join(lines), family='monospace', fontsize=9,
        verticalalignment='top', transform=ax.transAxes)

fig.suptitle(f'Cascade fingerprint with Poisson reference — {n_real} realizations',
             y=0.995, fontsize=13, weight='bold')

out_path = '/home/claude/morton_cascade/cascade_fingerprint_v3.png'
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"Saved: {out_path}")
