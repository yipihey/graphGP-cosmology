"""Quaia BAO matched filter via sigma^2(R) and dsigma^2/dR.

Mirrors ``demos/demo_quaia_bao_filter.py`` but with sigma^2 and
its R-derivative as the fundamental two-point statistic instead
of wp(rp). The matched filter projects the residual

    r(R) = sigma^2_obs(R) - b^2 * sigma^2_smooth(R)

onto the BAO template ``T(R) = sigma^2_full - sigma^2_nowiggle``,
and the same for the derivative. With the analytic-RR backend
the entire pipeline produces sigma^2 noise-free from a single
DD pass.

Output: demos/figures/quaia_sigma2_bao.png
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.analytic_rr import dr_analytic, rr_analytic
from twopt_density.bao_filter import matched_filter_amplitude
from twopt_density.distance import DistanceCosmo
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2 import (
    dsigma2_dR_predicted, sigma2_bao_template, sigma2_from_rp_pi_pairs,
    sigma2_predicted,
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
    z_eff = _env("PAPER_Z_EFF", 1.5, float)
    bias = _env("PAPER_BIAS", 2.6, float)
    sigma8 = _env("PAPER_SIGMA8", 0.81, float)

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
    iD = rng.choice(int(md.sum()), n_data, replace=False)
    iR = rng.choice(int(mr.sum()), n_random, replace=False)
    pos_d = cat.xyz_data[np.where(md)[0][iD]]
    pos_r = cat.xyz_random[np.where(mr)[0][iR]]
    z_d = cat.z_data[np.where(md)[0][iD]]
    shift = -np.vstack([pos_d, pos_r]).min(axis=0) + 100.0
    pos_d = pos_d + shift; pos_r = pos_r + shift
    print(f"  N_data = {len(pos_d):,}, N_random = {len(pos_r):,}")

    rp_edges = np.concatenate([
        np.logspace(np.log10(2.0), np.log10(40.0), 12),
        np.linspace(50.0, rp_max, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, pi_max, 41)
    rp_c = 0.5 * (rp_edges[1:] + rp_edges[:-1])
    pi_c = 0.5 * (pi_edges[1:] + pi_edges[:-1])

    # --- 1. DD + analytic RR + DR ---
    print("\nDD pair counts ...")
    t0 = time.perf_counter()
    DD = _count_pairs_rp_pi(pos_d, pos_d, rp_edges, pi_edges, auto=True,
                              chunk=4000)
    print(f"  {time.perf_counter()-t0:.0f}s")

    print("analytic RR/DR + scalar calibration ...")
    t0 = time.perf_counter()
    res = rr_analytic(rp_edges, pi_edges, mask, nside, z_d, fid,
                        N_r=10 * len(pos_d))
    cal_rng = np.random.default_rng(11)
    n_cal = 12_000
    icd = cal_rng.choice(len(pos_d), n_cal, replace=False)
    icr = cal_rng.choice(len(pos_r), 3 * n_cal, replace=False)
    cal = wp_landy_szalay(pos_d[icd], pos_r[icr], rp_edges,
                            pi_max=pi_max, n_pi=40)
    cal_a = rr_analytic(rp_edges, cal.pi_edges, mask, nside,
                          z_d[icd], fid, N_r=3 * n_cal)
    calib = float(np.median(cal.RR[cal.RR > 0]
                              / np.maximum(cal_a.RR[cal.RR > 0], 1e-30)))
    RR = calib * res.RR
    DR = dr_analytic(len(pos_d), 10 * len(pos_d), RR)
    print(f"  {time.perf_counter()-t0:.0f}s, calib = {calib:.3f}")

    # --- 2. project DD/DR/RR onto sigma^2(R) ---
    R_grid = np.linspace(20.0, 150.0, 32)
    print(f"\nsigma^2(R) projection over R = {R_grid[0]}-{R_grid[-1]} Mpc/h ...")
    s2_obs = sigma2_from_rp_pi_pairs(rp_c, pi_c, DD, RR, R_grid,
                                          N_d=len(pos_d), N_r=10 * len(pos_d),
                                          DR2=DR, kernel="tophat")
    # finite-difference dsigma2/dR
    R_grid_p = R_grid + 1.0
    R_grid_m = R_grid - 1.0
    s2_p = sigma2_from_rp_pi_pairs(rp_c, pi_c, DD, RR, R_grid_p,
                                        N_d=len(pos_d), N_r=10 * len(pos_d),
                                        DR2=DR, kernel="tophat")
    s2_m = sigma2_from_rp_pi_pairs(rp_c, pi_c, DD, RR, R_grid_m,
                                        N_d=len(pos_d), N_r=10 * len(pos_d),
                                        DR2=DR, kernel="tophat")
    ds2_obs = (s2_p - s2_m) / (R_grid_p - R_grid_m)

    # --- 3. JAX-pure forward model: full + nowiggle, sigma^2 and dsigma^2/dR ---
    print(f"\nforward model: sigma^2_predicted at z_eff={z_eff}, b={bias}, "
          f"sigma_8={sigma8} ...")
    s2_full = np.asarray(sigma2_predicted(jnp.asarray(R_grid),
                                                z_eff=z_eff, cosmo=fid,
                                                bias=bias, sigma8=sigma8))
    s2_smooth = np.asarray(sigma2_predicted(jnp.asarray(R_grid),
                                                  z_eff=z_eff, cosmo=fid,
                                                  bias=bias, sigma8=sigma8,
                                                  nowiggle=True))
    ds2_full = np.asarray(dsigma2_dR_predicted(jnp.asarray(R_grid),
                                                     z_eff=z_eff, cosmo=fid,
                                                     bias=bias, sigma8=sigma8))
    ds2_smooth = np.asarray(dsigma2_dR_predicted(jnp.asarray(R_grid),
                                                       z_eff=z_eff, cosmo=fid,
                                                       bias=bias, sigma8=sigma8,
                                                       nowiggle=True))
    T_s2 = s2_full - s2_smooth
    T_ds2 = ds2_full - ds2_smooth
    print(f"  template peak |T_sigma2| = {np.max(np.abs(T_s2)):.3e}, "
          f"|T_dsigma2| = {np.max(np.abs(T_ds2)):.3e}")

    # --- 4. matched filter: sigma^2 and dsigma^2/dR ---
    print("\nmatched filter ...")
    use = (R_grid > 30.0) & (R_grid < 130.0)
    # Self-calibrated per-bin noise: rms scatter of (data - smooth) in a
    # rolling window, capped above the BAO template amplitude so the
    # matched filter doesn't explode at quiet bins. For Quaia photo-z
    # data the 3D sigma^2 measurement is dominated by LOS smear, so
    # this noise is generous; on DESI spectroscopic data the per-bin
    # cosmic-variance dominates.
    res_s2_for_sigma = s2_obs - s2_smooth
    res_ds2_for_sigma = ds2_obs - ds2_smooth
    sigma_s2 = np.maximum(
        np.std(res_s2_for_sigma[use]) * np.ones_like(R_grid),
        0.5 * np.max(np.abs(T_s2[use])),
    )
    sigma_ds2 = np.maximum(
        np.std(res_ds2_for_sigma[use]) * np.ones_like(R_grid),
        0.5 * np.max(np.abs(T_ds2[use])),
    )

    res_s2 = s2_obs - s2_smooth
    res_ds2 = ds2_obs - ds2_smooth
    A_s2, sA_s2, snr_s2, _, _ = matched_filter_amplitude(
        s2_obs[use], s2_smooth[use], T_s2[use], sigma_s2[use],
    )
    A_ds2, sA_ds2, snr_ds2, _, _ = matched_filter_amplitude(
        ds2_obs[use], ds2_smooth[use], T_ds2[use], sigma_ds2[use],
    )
    print(f"  sigma^2 BAO:        A = {A_s2:.3f} +/- {sA_s2:.3f}, "
          f"SNR = {snr_s2:.2f}")
    print(f"  dsigma^2/dR BAO:    A = {A_ds2:.3f} +/- {sA_ds2:.3f}, "
          f"SNR = {snr_ds2:.2f}")

    # --- 5. figure ---
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ax_s2, ax_ds2 = axs[0]
    ax_res_s2, ax_res_ds2 = axs[1]

    ax_s2.plot(R_grid, s2_obs, "ko", ms=4, label="data (analytic-RR LS)")
    ax_s2.plot(R_grid, s2_full, "C0-", lw=1.6, label="full BAO model")
    ax_s2.plot(R_grid, s2_smooth, "C2--", lw=1.4, label="no-wiggle smooth")
    ax_s2.set_xlabel(r"$R$ [Mpc/h]"); ax_s2.set_ylabel(r"$\sigma^2(R)$")
    ax_s2.set_yscale("symlog", linthresh=1e-3); ax_s2.axhline(0, color="k", lw=0.5)
    ax_s2.set_title(rf"$\sigma^2(R)$, $z_{{\rm eff}}={z_eff}$, $b={bias}$")
    ax_s2.legend(fontsize=8); ax_s2.grid(alpha=0.3, which="both")

    ax_ds2.plot(R_grid, ds2_obs, "ko", ms=4, label="data (finite-diff)")
    ax_ds2.plot(R_grid, ds2_full, "C0-", lw=1.6, label="full BAO model")
    ax_ds2.plot(R_grid, ds2_smooth, "C2--", lw=1.4, label="no-wiggle smooth")
    ax_ds2.set_xlabel(r"$R$ [Mpc/h]"); ax_ds2.set_ylabel(r"$d\sigma^2/dR$")
    ax_ds2.axhline(0, color="k", lw=0.5)
    ax_ds2.set_title(r"$d\sigma^2/dR$ vs predicted")
    ax_ds2.legend(fontsize=8); ax_ds2.grid(alpha=0.3)

    ax_res_s2.errorbar(R_grid[use], res_s2[use], yerr=sigma_s2[use],
                          fmt="ok", ms=4, capsize=2)
    ax_res_s2.plot(R_grid, T_s2, "C0-", lw=1.5,
                      label=fr"BAO template ($A=1$)")
    ax_res_s2.plot(R_grid, A_s2 * T_s2, "C3--", lw=1.5,
                      label=fr"best fit $A={A_s2:.2f}\pm{sA_s2:.2f}$ "
                             fr"(SNR$={snr_s2:.1f}$)")
    ax_res_s2.axhline(0, color="k", lw=0.5)
    ax_res_s2.set_xlabel(r"$R$ [Mpc/h]")
    ax_res_s2.set_ylabel(r"$\sigma^2 - b^2 \sigma^2_{\rm smooth}$")
    ax_res_s2.set_title(r"$\sigma^2$ BAO matched filter")
    ax_res_s2.legend(fontsize=8); ax_res_s2.grid(alpha=0.3)

    ax_res_ds2.errorbar(R_grid[use], res_ds2[use], yerr=sigma_ds2[use],
                            fmt="ok", ms=4, capsize=2)
    ax_res_ds2.plot(R_grid, T_ds2, "C0-", lw=1.5,
                        label=fr"BAO template ($A=1$)")
    ax_res_ds2.plot(R_grid, A_ds2 * T_ds2, "C3--", lw=1.5,
                        label=fr"best fit $A={A_ds2:.2f}\pm{sA_ds2:.2f}$ "
                               fr"(SNR$={snr_ds2:.1f}$)")
    ax_res_ds2.axhline(0, color="k", lw=0.5)
    ax_res_ds2.set_xlabel(r"$R$ [Mpc/h]")
    ax_res_ds2.set_ylabel(r"$d\sigma^2/dR - $ smooth")
    ax_res_ds2.set_title(r"$d\sigma^2/dR$ BAO matched filter")
    ax_res_ds2.legend(fontsize=8); ax_res_ds2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_sigma2_bao.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
