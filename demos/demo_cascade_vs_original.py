"""Head-to-head benchmark: cascade xi vs original wp/xi pipeline on Quaia.

Compares two ways of computing the Quaia 2-point clustering signal:

  A.  Original pipeline (twopt_density.projected_xi + analytic_rr):
      - wp_landy_szalay: MC LS pair counts on (rp, pi) bins, scipy
        cKDTree, then projected to wp(rp). Optionally with analytic
        RR/DR replacing the MC random pair counts.

  B.  Cascade pipeline (twopt_density.cascade):
      - xi_landy_szalay_from_window: window-synthesised randoms +
        cascade O(N log N) pair counts at every dyadic shell.

Reports:
  - wall-time scaling vs N_d
  - output coverage (number of distinct r scales)
  - LS xi at matched scales

The two pipelines compute different observables (wp vs 3D xi at
dyadic shells), so the *output* comparison is qualitative; the
quantitative claim is that the cascade replaces O(N_d^2)-leaning
MC random-pair counts with O(N log N) and produces all dyadic
scales in one pass.
"""

from __future__ import annotations

import os
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.analytic_rr import (
    calibrate_norm_to_mc, dr_analytic, rr_analytic,
)
from twopt_density.cascade import xi_landy_szalay_from_window
from twopt_density.distance import DistanceCosmo
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=3, rng_seed=0,
    )
    sel_map, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    mr = (cat.z_random >= 0.8) & (cat.z_random <= 2.5)
    rng = np.random.default_rng(0)

    n_d_grid = [10_000, 30_000, 60_000, 120_000]
    times_orig_mc = []      # original MC LS wp(rp)
    times_orig_ana = []     # original analytic-RR wp(rp), DD only
    times_cascade = []      # cascade xi from window

    for n_d in n_d_grid:
        if n_d > int(md.sum()):
            n_d_grid = n_d_grid[:n_d_grid.index(n_d)]
            break
        iD = rng.choice(int(md.sum()), n_d, replace=False)
        iR = rng.choice(int(mr.sum()),
                          min(3 * n_d, int(mr.sum())), replace=False)
        pos_d = cat.xyz_data[np.where(md)[0][iD]]
        pos_r = cat.xyz_random[np.where(mr)[0][iR]]
        ra_d = cat.ra_data[np.where(md)[0][iD]]
        dec_d = cat.dec_data[np.where(md)[0][iD]]
        z_d = cat.z_data[np.where(md)[0][iD]]
        shift = -np.vstack([pos_d, pos_r]).min(axis=0) + 100.0
        pos_d_s = pos_d + shift; pos_r_s = pos_r + shift

        rp_edges = np.concatenate([
            np.logspace(np.log10(5.0), np.log10(50.0), 8),
            np.linspace(60.0, 200.0, 14)[1:],
        ])
        pi_edges = np.linspace(0.0, 200.0, 41)

        # A1. Original MC LS
        t0 = time.perf_counter()
        meas_mc = wp_landy_szalay(pos_d_s, pos_r_s, rp_edges,
                                       pi_max=200.0, n_pi=40)
        t_mc = time.perf_counter() - t0
        times_orig_mc.append(t_mc)

        # A2. Original analytic-RR LS (DD pair counts only + analytic RR/DR)
        t0 = time.perf_counter()
        DD = _count_pairs_rp_pi(pos_d_s, pos_d_s, rp_edges, pi_edges,
                                  auto=True, chunk=4000)
        res = rr_analytic(rp_edges, pi_edges, sel_map, nside, z_d, fid,
                            N_r=10 * n_d)
        n_cal = min(8000, n_d)
        i_cd = rng.choice(n_d, n_cal, replace=False)
        i_cr = rng.choice(len(pos_r_s), 3 * n_cal, replace=False)
        meas_cal = wp_landy_szalay(pos_d_s[i_cd], pos_r_s[i_cr],
                                        rp_edges, pi_max=200.0, n_pi=40)
        cal_a = rr_analytic(rp_edges, meas_cal.pi_edges, sel_map, nside,
                              z_d[i_cd], fid, N_r=3 * n_cal)
        calib = float(np.median(meas_cal.RR[meas_cal.RR > 0]
                                  / np.maximum(cal_a.RR[meas_cal.RR > 0], 1e-30)))
        RR = calib * res.RR
        t_ana = time.perf_counter() - t0
        times_orig_ana.append(t_ana)

        # B. Cascade xi from window-synthesised randoms
        t0 = time.perf_counter()
        arr = xi_landy_szalay_from_window(
            pos_d, ra_d, dec_d, z_d, sel_map, nside, fid,
            n_random_factor=3, rng_seed=42,
        )
        t_cas = time.perf_counter() - t0
        times_cascade.append(t_cas)

        print(f"  N_d = {n_d:>6,}  MC LS = {t_mc:5.1f}s  "
              f"analytic-RR LS = {t_ana:5.1f}s  cascade xi = {t_cas:5.1f}s")

    n_d_arr = np.array(n_d_grid[: len(times_cascade)])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n_d_arr, times_orig_mc, "ko-", ms=6,
                label="original MC LS wp(rp)")
    ax.loglog(n_d_arr, times_orig_ana, "C0s-", ms=6,
                label="original analytic-RR LS wp(rp)")
    ax.loglog(n_d_arr, times_cascade, "C3D-", ms=6,
                label="cascade xi(r) at every dyadic shell")
    n_ext = np.array([n_d_arr[0], 5e5, 5e6])
    ax.loglog(n_ext, times_orig_mc[0] * (n_ext / n_d_arr[0]) ** 2,
                "k:", lw=1, alpha=0.6, label=r"$\propto N^2$ guide")
    ax.loglog(n_ext, times_cascade[0] * (n_ext / n_d_arr[0]) * np.log2(n_ext / n_d_arr[0] + 1),
                "C3:", lw=1, alpha=0.6, label=r"$\propto N \log N$ guide")
    ax.set_xlabel(r"$N_{\rm data}$"); ax.set_ylabel(r"wall time [s]")
    ax.set_title(r"Quaia 2-point pipeline: original vs cascade scaling")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cascade_vs_original_timing.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
