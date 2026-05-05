"""sigma^2(R) on Quaia G < 20 -- parallel pipeline to wp(rp).

Computes the sphere-counts variance two-point statistic from the
same Quaia data + same analytic-RR window machinery used by the
wp(rp) BAO matched filter, demonstrating equivalence and
small-scale-systematics robustness.

  - sigma^2(R) from MC LS pair counts (DD, DR, RR) projected through
    the top-hat sphere kernel
  - sigma^2(R) from the same DD plus the analytic RR/DR (no MC RR)
  - sigma^2(R) from integrating the LS xi(s) measurement against
    the kernel (cross-check)
  - timing comparison (MC vs analytic) at multiple R

Output:
  demos/figures/quaia_sigma2.png
"""

from __future__ import annotations

import os
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.analytic_rr import dr_analytic, rr_analytic
from twopt_density.distance import DistanceCosmo
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2 import (
    kernel_TH_3d, sigma2_from_rp_pi_pairs, sigma2_from_xi,
)


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env(n, d, cast=str):
    v = os.environ.get(n)
    return cast(v) if v else d


def main():
    n_data = _env("PAPER_N_DATA", 60_000, int)
    n_random = _env("PAPER_N_RANDOM", 180_000, int)
    rp_max = _env("PAPER_RP_MAX", 200.0, float)
    pi_max = _env("PAPER_PI_MAX", 200.0, float)

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=3, rng_seed=0,
    )
    mask, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    mr = (cat.z_random >= 0.8) & (cat.z_random <= 2.5)

    rng = np.random.default_rng(0)
    idx_d = rng.choice(int(md.sum()), min(n_data, int(md.sum())),
                          replace=False)
    idx_r = rng.choice(int(mr.sum()), min(n_random, int(mr.sum())),
                          replace=False)
    where_d = np.where(md)[0][idx_d]
    where_r = np.where(mr)[0][idx_r]
    xyz_d = cat.xyz_data[where_d]
    xyz_r = cat.xyz_random[where_r]
    z_d = cat.z_data[where_d]
    print(f"  N_data = {len(xyz_d):,}, N_random = {len(xyz_r):,}")

    all_xyz = np.vstack([xyz_d, xyz_r])
    shift = -all_xyz.min(axis=0) + 100.0
    pos_d = xyz_d + shift
    pos_r = xyz_r + shift

    rp_edges = np.concatenate([
        np.logspace(np.log10(2.0), np.log10(40.0), 12),
        np.linspace(50.0, rp_max, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, pi_max, 41)
    rp_centres = 0.5 * (rp_edges[1:] + rp_edges[:-1])
    pi_centres = 0.5 * (pi_edges[1:] + pi_edges[:-1])
    R_grid = np.linspace(8.0, 100.0, 24)

    # --- 1. MC LS pair counts (DD, DR, RR) ---
    print("\nMC LS pair counts (DD, DR, RR in (rp, pi)) ...")
    t0 = time.perf_counter()
    DD = _count_pairs_rp_pi(pos_d, pos_d, rp_edges, pi_edges,
                              auto=True, chunk=4000)
    DR = _count_pairs_rp_pi(pos_d, pos_r, rp_edges, pi_edges, auto=False,
                              chunk=4000)
    RR_mc = _count_pairs_rp_pi(pos_r, pos_r, rp_edges, pi_edges,
                                 auto=True, chunk=4000)
    t_mc = time.perf_counter() - t0
    print(f"  done ({t_mc:.0f}s)")

    # --- 2. analytic RR + DR ---
    print("analytic RR / DR ...")
    t0 = time.perf_counter()
    res = rr_analytic(rp_edges, pi_edges, mask, nside, z_d, fid,
                        N_r=10 * len(pos_d))
    # quick MC calibration on a small subsample
    cal_rng = np.random.default_rng(11)
    n_cal = 12_000
    i_cd = cal_rng.choice(len(pos_d), n_cal, replace=False)
    i_cr = cal_rng.choice(len(pos_r), 3 * n_cal, replace=False)
    meas_cal = wp_landy_szalay(pos_d[i_cd], pos_r[i_cr],
                                 rp_edges, pi_max=pi_max, n_pi=40)
    ana_cal = rr_analytic(rp_edges, meas_cal.pi_edges, mask, nside,
                            z_d[i_cd], fid, N_r=3 * n_cal)
    calib = float(np.median(meas_cal.RR[meas_cal.RR > 0]
                             / np.maximum(ana_cal.RR[meas_cal.RR > 0], 1e-30)))
    print(f"  calib = {calib:.3f}")
    RR_ana = calib * res.RR
    DR_ana = dr_analytic(len(pos_d), 10 * len(pos_d), RR_ana)
    t_ana = time.perf_counter() - t0
    print(f"  done ({t_ana:.0f}s)")

    # --- 3. project onto sigma^2(R) -- three ways ---
    print("\nsigma^2(R) projections ...")
    s2_mc = sigma2_from_rp_pi_pairs(
        rp_centres, pi_centres, DD, RR_mc, R_grid,
        N_d=len(pos_d), N_r=len(pos_r), DR2=DR, kernel="tophat",
    )
    s2_ana = sigma2_from_rp_pi_pairs(
        rp_centres, pi_centres, DD, RR_ana, R_grid,
        N_d=len(pos_d), N_r=10 * len(pos_d), DR2=DR_ana,
        kernel="tophat",
    )

    # also recompute via xi(s) -> integrated kernel projection
    s2d = np.sqrt(rp_centres[:, None] ** 2 + pi_centres[None, :] ** 2)
    s_edges = np.linspace(2.0, 200.0, 32)
    s_centres = 0.5 * (s_edges[1:] + s_edges[:-1])
    Nd_pairs = len(pos_d) * (len(pos_d) - 1) / 2.0
    Nr_pairs = len(pos_r) * (len(pos_r) - 1) / 2.0
    DD_n = DD / Nd_pairs
    RR_n = RR_mc / Nr_pairs
    DR_n = DR / (len(pos_d) * len(pos_r))

    xi_per_s = np.zeros(len(s_centres))
    counts_per_s = np.zeros(len(s_centres))
    for ks in range(len(s_centres)):
        m = (s2d >= s_edges[ks]) & (s2d < s_edges[ks + 1])
        if not m.any():
            continue
        SDD = float(np.sum(DD_n[m]))
        SDR = float(np.sum(DR_n[m]))
        SRR = float(np.sum(RR_n[m]))
        if SRR > 0:
            xi_per_s[ks] = (SDD - 2 * SDR + SRR) / SRR
        counts_per_s[ks] = m.sum()
    s2_xi = sigma2_from_xi(s_centres, xi_per_s, R_grid, kernel="tophat")
    print(f"  three-way sigma^2(R) computed for R in "
          f"[{R_grid[0]:.0f}, {R_grid[-1]:.0f}] Mpc/h")

    # --- 4. timing summary ---
    speedup = t_mc / max(t_ana, 0.01)
    print(f"\ntiming: MC LS {t_mc:.0f}s vs analytic {t_ana:.0f}s "
          f"-> {speedup:.0f}x speedup")

    # --- figure ---
    fig, axs = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1.4, 1]})
    ax_s2, ax_kernel = axs

    ax_s2.plot(R_grid, s2_mc, "ko-", ms=5, label=r"MC LS pair counts")
    ax_s2.plot(R_grid, s2_ana, "C0s-", ms=5, lw=1.6,
                 label=r"analytic-RR LS (no MC RR)")
    ax_s2.plot(R_grid, s2_xi, "C3^-", ms=5, lw=1.4,
                 label=r"$\int \xi(s) K_{\rm TH}(s; R)\,dV$")
    ax_s2.set_xscale("log"); ax_s2.set_yscale("symlog", linthresh=1e-3)
    ax_s2.axhline(0, color="k", lw=0.5)
    ax_s2.set_xlabel(r"$R$ [Mpc/h]")
    ax_s2.set_ylabel(r"$\sigma^2_{\rm TH}(R)$")
    ax_s2.set_title(rf"Quaia G$<$20 $\sigma^2_{{\rm TH}}(R)$ "
                       rf"($N_d={len(pos_d):,}$)")
    ax_s2.legend(fontsize=9); ax_s2.grid(alpha=0.3, which="both")

    # right panel: top-hat kernel shape
    r_show = np.linspace(0, 200, 600)
    for R in [10, 30, 60]:
        K = kernel_TH_3d(r_show, R)
        ax_kernel.plot(r_show, K, lw=1.5, label=fr"$R={R}$ Mpc/h")
    ax_kernel.set_xlabel(r"$r$ [Mpc/h]")
    ax_kernel.set_ylabel(r"$K_{\rm TH}(r; R)$")
    ax_kernel.set_title(r"top-hat sphere kernel $K_{\rm TH}(r; R)$")
    ax_kernel.set_yscale("log"); ax_kernel.set_ylim(1e-9, 1e-2)
    ax_kernel.legend(fontsize=9); ax_kernel.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_sigma2.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
