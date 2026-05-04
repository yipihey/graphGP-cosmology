#!/usr/bin/env python3
"""
P_N(V) for the Cox process: probability of a window of volume V containing
exactly N points. Compare to a matched Poisson reference.

Four panels:
  (a) PMFs at six log-spaced V (Cox filled, Poisson dashed)
  (b) P_0(V) — empty-window probability vs window mean count nu
  (c) Mean/variance growth with V (Cox vs Poisson)
  (d) Skewness/excess kurtosis of P_N as a function of V

Run after: cargo run --release --example pmf_cox_demo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

DIR = "/tmp/pmf_cox_demo"

mom_cox = pd.read_csv(f"{DIR}/moments_cox.csv")
mom_poi = pd.read_csv(f"{DIR}/moments_poi.csv")
pmf_cox = pd.read_csv(f"{DIR}/pmf_cox.csv")
pmf_poi = pd.read_csv(f"{DIR}/pmf_poi.csv")

print(f"Cox moments: {len(mom_cox)} window sizes")
print(f"Cox PMF rows: {len(pmf_cox)}")
print(f"Poisson PMF rows: {len(pmf_poi)}")

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32, width_ratios=[1.6, 1, 1])

# ---------- (a) PMFs at six log-spaced V ----------
# Pick 6 window sides spanning the range; show Cox as filled bars/stems
# and Poisson reference as dashed step.
ax = fig.add_subplot(gs[:, 0])

target_sides = [1, 4, 10, 25, 63, 158]  # spans nu from ~0.76 to ~1.9e4
colors = cm.viridis(np.linspace(0.05, 0.95, len(target_sides)))

y_offset = 0.0
y_step = 1.15  # vertical offset between PMF panels in standardized-N units

# Common standardized-N bins so we can smooth spiky tails uniformly
edges = np.linspace(-3, 7, 80)
centers = 0.5 * (edges[:-1] + edges[1:])

for i, k_target in enumerate(target_sides):
    # Find closest available window side
    avail = sorted(pmf_cox['window_side'].unique())
    k = min(avail, key=lambda kk: abs(kk - k_target))

    sub_c = pmf_cox[pmf_cox['window_side'] == k].sort_values('count')
    sub_p = pmf_poi[pmf_poi['window_side'] == k].sort_values('count')
    if len(sub_c) == 0:
        continue

    n_total_c = sub_c['frequency'].sum()
    n_total_p = sub_p['frequency'].sum()

    mean_c = mom_cox.loc[mom_cox['window_side'] == k, 'mean'].iloc[0]
    var_c = mom_cox.loc[mom_cox['window_side'] == k, 'var'].iloc[0]
    nu = mom_cox.loc[mom_cox['window_side'] == k, 'nu_expected'].iloc[0]
    std_c = np.sqrt(max(var_c, 1e-30))

    if std_c < 1e-12:
        continue

    # Bin Cox PMF in standardized units
    rebinned_c = np.zeros_like(centers)
    for _, row in sub_c.iterrows():
        x = (row['count'] - mean_c) / std_c
        j = np.searchsorted(edges, x) - 1
        if 0 <= j < len(centers):
            rebinned_c[j] += row['frequency']
    rebinned_c /= max(n_total_c, 1)

    # Bin Poisson PMF in same standardized units (using Cox mean/std for visual alignment)
    rebinned_p = np.zeros_like(centers)
    for _, row in sub_p.iterrows():
        x = (row['count'] - mean_c) / std_c
        j = np.searchsorted(edges, x) - 1
        if 0 <= j < len(centers):
            rebinned_p[j] += row['frequency']
    rebinned_p /= max(n_total_p, 1)

    # Normalize each to peak=1 for stacking
    if rebinned_c.max() > 0:
        rebinned_c_n = rebinned_c / rebinned_c.max()
    else:
        rebinned_c_n = rebinned_c
    if rebinned_p.max() > 0:
        rebinned_p_n = rebinned_p / rebinned_p.max()
    else:
        rebinned_p_n = rebinned_p

    # Plot Cox: filled area
    ax.fill_between(centers, y_offset, y_offset + rebinned_c_n, color=colors[i], alpha=0.55, lw=0)
    ax.plot(centers, y_offset + rebinned_c_n, color=colors[i], lw=1.2)
    # Plot Poisson: dashed line
    ax.plot(centers, y_offset + rebinned_p_n, color='black', lw=1.0, ls=(0, (3, 2)), alpha=0.85)

    # Label
    ax.text(5.5, y_offset + 0.5,
            f"k={k}\n$\\nu$={nu:.2g}",
            fontsize=8.5, color=colors[i], va='center')

    y_offset += y_step

ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlim(-3, 7)
ax.set_xlabel(r'standardized count  $(N - \langle N\rangle) / \sigma$')
ax.set_ylabel('P (offset by window size)')
ax.set_yticks([])
ax.set_title(r'(a) $P_N(V)$ — Cox (filled) vs Poisson (dashed)', fontsize=11)

# ---------- (b) P_0(V) ----------
ax = fig.add_subplot(gs[0, 1])
mc = mom_cox[mom_cox['P0_measured'] > 0]
mp = mom_poi[mom_poi['P0_measured'] > 0]
ax.semilogy(mc['nu_expected'], mc['P0_measured'], 'o-', color='C0',
            lw=1.6, ms=5, label='Cox')
ax.semilogy(mp['nu_expected'], mp['P0_measured'], 's-', color='gray',
            lw=1.0, ms=4, label='Poisson (measured)')
nu_grid_b = np.logspace(np.log10(mom_cox['nu_expected'].min()),
                        np.log10(mom_cox['nu_expected'].max()), 200)
ax.semilogy(nu_grid_b, np.exp(-nu_grid_b), 'k--', lw=0.8,
            label=r'$e^{-\nu}$ (analytic Poisson)')
ax.set_xscale('log')
ax.set_xlabel(r'mean count per window  $\nu = \bar n V$')
ax.set_ylabel(r'$P_0(V)$  (empty-window probability)')
ax.set_title(r'(b) Empty-window probability', fontsize=11)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(1e-7, 2.0)

# ---------- (c) variance growth with nu ----------
ax = fig.add_subplot(gs[0, 2])
# For Poisson, var = mean = nu. For Cox, var = mean + (sigma_field^2) * mean^2.
ax.loglog(mom_cox['nu_expected'], mom_cox['var'], 'o-', color='C0',
          lw=1.6, ms=5, label='Cox')
ax.loglog(mom_poi['nu_expected'], mom_poi['var'], 's-', color='gray',
          lw=1.0, ms=4, label='Poisson')
ax.loglog(mom_cox['nu_expected'], mom_cox['nu_expected'], 'k--', lw=0.8,
          label=r'$\nu$ (Poisson floor)')
ax.set_xlabel(r'mean count $\nu$')
ax.set_ylabel(r'variance  $\langle N^2 \rangle - \langle N\rangle^2$')
ax.set_title(r'(c) Variance growth', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# ---------- (d) Skewness and excess kurtosis ----------
ax = fig.add_subplot(gs[1, 1])
ax.semilogx(mom_cox['nu_expected'], mom_cox['skew'], 'o-', color='C2',
            lw=1.6, ms=5, label='Cox  skew')
ax.semilogx(mom_poi['nu_expected'], mom_poi['skew'], 's-', color='gray',
            lw=1.0, ms=4, label='Poisson  skew')
# Poisson analytic skew = 1/sqrt(nu); plot as dashed
nu_grid = np.logspace(np.log10(mom_cox['nu_expected'].min()),
                      np.log10(mom_cox['nu_expected'].max()), 100)
ax.semilogx(nu_grid, 1.0/np.sqrt(nu_grid), 'k--', lw=0.7,
            label=r'$1/\sqrt{\nu}$  (Poisson)')
ax.set_xlabel(r'mean count $\nu$')
ax.set_ylabel('skewness')
ax.set_title(r'(d) Skewness of $P_N(V)$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

ax = fig.add_subplot(gs[1, 2])
ax.semilogx(mom_cox['nu_expected'], mom_cox['kurt'] - 3, 'o-', color='C3',
            lw=1.6, ms=5, label='Cox  excess kurt')
ax.semilogx(mom_poi['nu_expected'], mom_poi['kurt'] - 3, 's-', color='gray',
            lw=1.0, ms=4, label='Poisson  excess kurt')
ax.semilogx(nu_grid, 1.0/nu_grid, 'k--', lw=0.7,
            label=r'$1/\nu$  (Poisson)')
ax.set_xlabel(r'mean count $\nu$')
ax.set_ylabel(r'excess kurtosis  $\kappa_4/\sigma^4 - 3$')
ax.set_title(r'(e) Excess kurtosis of $P_N(V)$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle(
    r'$P_N(V)$ for a 2D log-normal Cox process vs matched Poisson  '
    r'($N=2\times10^5$, 23 cube-window volumes, 5 PPD-V)',
    fontsize=12, weight='bold', y=0.995)

out = '/home/claude/morton_cascade/pmf_cox_demo.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print(f"Saved: {out}")
