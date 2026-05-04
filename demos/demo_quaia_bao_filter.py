"""BAO matched-filter on Quaia wp(rp) + jackknife covariance.

Builds on ``demo_quaia_bao_analytic.py`` with two SNR-enhancement
techniques discussed earlier:

  (1) Matched filter: project the wp(rp) residual onto the BAO
      template T(rp) = b^2 (wp_full - wp_nowiggle). This combines
      coherently across all rp bins and gives a single amplitude
      A_hat with uncertainty -- the optimal one-number summary of
      "is BAO present in the data?"

  (2) BAO scaling alpha: extend the matched filter to a 2-parameter
      fit (alpha, A) where alpha shifts the BAO peak position.
      The chi^2(alpha) curve recovers alpha_hat with uncertainty
      via a parabolic fit -- this is the cosmology-relevant BAO
      observable.

  (3) Jackknife covariance: replace the diagonal sigma_wp(rp) with
      the full jackknife covariance C_ij. Captures off-diagonal
      correlations from cosmic variance / window function modes,
      changes the effective number of independent rp bins, and
      typically reduces the matched-filter SNR (because rp bins
      aren't truly independent).

Output: ``quaia_bao_filter.png`` -- four-panel plot:
  top-left:    wp_data residual + best-fit A * BAO template
  top-right:   chi^2(alpha) and SNR(alpha) scan
  bottom-left: jackknife covariance matrix (correlation form)
  bottom-right: matched-filter SNR vs rp_min cut (test of BAO
                signal vs systematics scale).
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

from twopt_density.analytic_rr import (
    calibrate_norm_to_mc, dr_analytic, rr_analytic,
)
from twopt_density.bao_filter import (
    bao_alpha_scan, bao_template, matched_filter_amplitude,
)
from twopt_density.distance import DistanceCosmo
from twopt_density.jackknife import wp_jackknife
from twopt_density.limber import (
    make_wp_fft, sigma_chi_from_sigma_z, wp_observed, wp_observed_nowiggle,
)
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.systematics import data_residual_weights


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_int(n, d):
    v = os.environ.get(n)
    return int(v) if v else d


def _env_float(n, d):
    v = os.environ.get(n)
    return float(v) if v else d


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8 = _env_float("QUAIA_SIGMA8", 0.81)
    n_data_max = _env_int("QUAIA_N_DATA", 200000)
    pi_max = _env_float("QUAIA_PI_MAX", 200.0)
    rp_max = _env_float("QUAIA_RP_MAX", 150.0)
    n_jk = _env_int("QUAIA_N_JK", 25)
    smooth_deg = _env_float("QUAIA_DEPROJ_FWHM_DEG", 60.0)

    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=2, rng_seed=0,
    )
    mask, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    mr = (cat.z_random >= 0.8) & (cat.z_random <= 2.5)
    rng = np.random.default_rng(0)
    xyz_d_all = np.asarray(cat.xyz_data[md])
    xyz_r_all = np.asarray(cat.xyz_random[mr])
    z_d_all = cat.z_data[md]; sig_z_all = cat.z_data_err[md]
    ra_d_all = cat.ra_data[md]; dec_d_all = cat.dec_data[md]
    if len(xyz_d_all) > n_data_max:
        idx = rng.choice(len(xyz_d_all), n_data_max, replace=False)
        xyz_d_all = xyz_d_all[idx]; z_d_all = z_d_all[idx]
        sig_z_all = sig_z_all[idx]; ra_d_all = ra_d_all[idx]
        dec_d_all = dec_d_all[idx]
    print(f"  N_data = {len(xyz_d_all):,}")

    # ---- 1. calibration MC RR ----
    print("\ncalibration MC RR ...")
    cal_rng = np.random.default_rng(1)
    n_cal_d, n_cal_r = 30_000, 90_000
    i_cd = cal_rng.choice(len(xyz_d_all), n_cal_d, replace=False)
    i_cr = cal_rng.choice(len(xyz_r_all), n_cal_r, replace=False)
    rp_edges = np.concatenate([
        np.logspace(np.log10(5.0), np.log10(50.0), 8),
        np.linspace(60.0, rp_max, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, pi_max, 41)
    t = time.perf_counter()
    meas_cal = wp_landy_szalay(xyz_d_all[i_cd], xyz_r_all[i_cr],
                                 rp_edges, pi_max=pi_max, n_pi=40)
    ana_cal = rr_analytic(rp_edges, meas_cal.pi_edges, mask, nside,
                            z_d_all[i_cd], fid, N_r=n_cal_r)
    calib = calibrate_norm_to_mc(ana_cal.RR, meas_cal.RR)
    print(f"  {time.perf_counter()-t:.0f}s, calib factor = {calib:.4f}")

    # ---- 2. deprojection weights ----
    print(f"\nbootstrap data-residual deprojection (FWHM={smooth_deg:.0f}deg) ...")
    w_data, _ = data_residual_weights(
        ra_d_all, dec_d_all, mask, nside,
        smoothing_fwhm_deg=smooth_deg,
    )
    print(f"  weights: mean={w_data.mean():.3f}, std={w_data.std():.3f}")

    # ---- 3. jackknife covariance ----
    print(f"\njackknife covariance ({n_jk} regions) ...")
    t = time.perf_counter()
    N_r_eff = 10 * len(xyz_d_all)
    wp_full, wp_jk, wp_cov = wp_jackknife(
        xyz_d_all, z_d_all, ra_d_all, dec_d_all,
        mask, nside, fid, rp_edges, pi_max=pi_max, n_pi=40,
        n_regions=n_jk, nside_jack=4,
        rr_norm_factor=calib, N_r_effective=N_r_eff,
        w_data=w_data,
    )
    print(f"  {time.perf_counter()-t:.0f}s")
    rp_centres = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    sigma_wp_diag = np.sqrt(np.diag(wp_cov))
    DD_per_rp_proxy = (200_000 / 4 + 1) ** 0.0   # (just for diagnostic)
    print(f"  jackknife sigma_wp(rp~100) = {np.interp(100.0, rp_centres, sigma_wp_diag):.3f}")

    # ---- 4. forward models ----
    sig_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d_all, sig_z_all, fid))
    sigma_chi_eff = float(np.sqrt(2.0) * np.median(sig_chi_per_obj))
    z_eff = float(np.median(z_d_all))
    fft, k_np = make_wp_fft()
    k_grid_jax = jnp.asarray(k_np)
    rp_pred_fine = np.linspace(5.0, rp_max, 200)
    wp_full_b1 = np.asarray(wp_observed(jnp.asarray(rp_pred_fine), z_eff=z_eff,
                                          sigma_chi_eff=sigma_chi_eff, cosmo=fid,
                                          bias=1.0, pi_max=pi_max, sigma8=sigma8,
                                          fft=fft, k_grid=k_grid_jax))
    wp_smooth_b1 = np.asarray(wp_observed_nowiggle(jnp.asarray(rp_pred_fine), z_eff=z_eff,
                                                     sigma_chi_eff=sigma_chi_eff, cosmo=fid,
                                                     bias=1.0, pi_max=pi_max, sigma8=sigma8,
                                                     fft=fft, k_grid=k_grid_jax))

    # bias fit on small-rp where BAO is small and signal is strong, using
    # the JK-diagonal sigma
    use = (rp_centres > 8.0) & (rp_centres < 50.0)
    wp_smooth_at = np.interp(rp_centres, rp_pred_fine, wp_smooth_b1)
    invsig2 = 1.0 / np.maximum(sigma_wp_diag[use], 1e-3) ** 2
    num = float(np.sum(wp_full[use] * wp_smooth_at[use] * invsig2))
    den = float(np.sum(wp_smooth_at[use] ** 2 * invsig2))
    b_fit = float(np.sqrt(max(num / den, 0.01)))
    print(f"\nbias fit (8 < rp < 50, JK-diag sigma): b = {b_fit:.3f}")

    # ---- 5. matched filter (diagonal AND full covariance) ----
    print("\nmatched filter ...")
    T = bao_template(rp_centres, b=b_fit, z_eff=z_eff,
                       sigma_chi_eff=sigma_chi_eff, pi_max=pi_max,
                       cosmo=fid, alpha=1.0, sigma8=sigma8)
    wp_smooth_at_meas = b_fit ** 2 * np.interp(
        rp_centres, rp_pred_fine, wp_smooth_b1
    )
    A_d, sdA_d, snr_d, c2null_d, c2best_d = matched_filter_amplitude(
        wp_full, wp_smooth_at_meas, T, sigma_wp_diag,
    )
    print(f"  diagonal sigma:  A = {A_d:.2f} +/- {sdA_d:.2f}, "
          f"SNR = {snr_d:.2f}")
    print(f"    chi2_null = {c2null_d:.1f}, chi2_best = {c2best_d:.1f}, "
          f"delta_chi2 = {c2null_d - c2best_d:.2f}")
    n_bins = len(rp_centres)
    A_c, sdA_c, snr_c, c2null_c, c2best_c = matched_filter_amplitude(
        wp_full, wp_smooth_at_meas, T, wp_cov, n_samples=n_jk,
    )
    hartlap = max((n_jk - n_bins - 2) / (n_jk - 1), 0.0)
    print(f"  full JK cov (Hartlap, N_jk={n_jk}, N_bins={n_bins}, "
          f"factor={hartlap:.2f}):")
    print(f"    A = {A_c:.2f} +/- {sdA_c:.2f}, SNR = {snr_c:.2f}")
    print(f"    chi2_null = {c2null_c:.1f}, chi2_best = {c2best_c:.1f}, "
          f"delta_chi2 = {c2null_c - c2best_c:.2f}")
    if c2null_c / max(n_bins, 1) > 3.0:
        print(f"  WARNING: reduced chi2_null = {c2null_c/n_bins:.1f} -- JK "
              "covariance likely under-estimated; full-cov SNR is unreliable.")

    # ---- 6. alpha scan ----
    print("\nalpha scan (BAO scaling parameter) ...")
    alpha_grid = np.linspace(0.85, 1.15, 31)
    out_d = bao_alpha_scan(rp_centres, wp_full, wp_smooth_at_meas, b_fit,
                             z_eff, sigma_chi_eff, pi_max, fid,
                             sigma_wp_diag, alpha_grid=alpha_grid,
                             sigma8=sigma8)
    out_c = bao_alpha_scan(rp_centres, wp_full, wp_smooth_at_meas, b_fit,
                             z_eff, sigma_chi_eff, pi_max, fid,
                             wp_cov, alpha_grid=alpha_grid, sigma8=sigma8,
                             n_samples=n_jk)
    print(f"  diagonal: alpha_hat = {out_d['alpha_hat']:.4f} +/- "
          f"{out_d['sigma_alpha']:.4f}, SNR = {out_d['SNR_at_best']:.2f}")
    print(f"  full cov: alpha_hat = {out_c['alpha_hat']:.4f} +/- "
          f"{out_c['sigma_alpha']:.4f}, SNR = {out_c['SNR_at_best']:.2f}")

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_res, ax_alpha, ax_cov, ax_rmin = axs.flatten()

    res = wp_full - wp_smooth_at_meas
    ax_res.errorbar(rp_centres, res, yerr=sigma_wp_diag, fmt="ok", ms=4,
                     capsize=3, label="$w_p^{data} - b^2 w_p^{smooth}$")
    ax_res.plot(rp_centres, A_d * T, "C0-", lw=2,
                 label=fr"$A \cdot T(r_p)$, $A_{{\rm diag}}={A_d:.2f}\pm{sdA_d:.2f}$")
    ax_res.plot(rp_centres, A_c * T, "C3--", lw=2,
                 label=fr"$A \cdot T(r_p)$, $A_{{\rm full\,cov}}={A_c:.2f}\pm{sdA_c:.2f}$")
    ax_res.set_xscale("log"); ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_xlabel(r"$r_p$ [Mpc/h]"); ax_res.set_ylabel(r"$\Delta w_p$ [Mpc/h]")
    ax_res.set_title(r"BAO matched filter: residual $w_p^{data} - "
                      r"b^2 w_p^{smooth}$ vs template $T(r_p)$")
    ax_res.axvspan(80, 130, alpha=0.10, color="C1")
    ax_res.legend(fontsize=9); ax_res.grid(alpha=0.3, which="both")

    ax_alpha.plot(out_d["alpha_grid"], out_d["chi2"] - out_d["chi2"].min(),
                    "C0-o", ms=3, label="diagonal sigma")
    ax_alpha.plot(out_c["alpha_grid"], out_c["chi2"] - out_c["chi2"].min(),
                    "C3-s", ms=3, label="full JK cov")
    ax_alpha.axvline(1.0, color="k", lw=0.5, ls=":")
    ax_alpha.set_xlabel(r"BAO scaling $\alpha$")
    ax_alpha.set_ylabel(r"$\chi^2 - \chi^2_{\min}$")
    ax_alpha.set_title(r"$\chi^2(\alpha)$ scan")
    ax_alpha.legend(fontsize=9); ax_alpha.grid(alpha=0.3)

    # correlation matrix
    diag = np.sqrt(np.maximum(np.diag(wp_cov), 1e-30))
    corr = wp_cov / np.outer(diag, diag)
    im = ax_cov.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r",
                         extent=[rp_centres[0], rp_centres[-1],
                                 rp_centres[-1], rp_centres[0]])
    plt.colorbar(im, ax=ax_cov, label=r"$\rho_{ij}$")
    ax_cov.set_xlabel(r"$r_p$ [Mpc/h]"); ax_cov.set_ylabel(r"$r_p$ [Mpc/h]")
    ax_cov.set_title(r"jackknife correlation matrix")

    # SNR vs rp_min cut (matched filter applied with rp > rp_min cut)
    rmin_grid = np.linspace(20.0, 80.0, 12)
    snr_d_arr = np.zeros_like(rmin_grid)
    snr_c_arr = np.zeros_like(rmin_grid)
    for i, rmin in enumerate(rmin_grid):
        mask_r = rp_centres > rmin
        if mask_r.sum() < 3:
            continue
        T_m = T[mask_r]
        res_m = res[mask_r]
        # diagonal
        invsig2 = 1.0 / np.maximum(sigma_wp_diag[mask_r], 1e-30) ** 2
        TtCT = float(np.sum(T_m ** 2 * invsig2))
        TtCr = float(np.sum(T_m * res_m * invsig2))
        if TtCT > 0:
            A_m = TtCr / TtCT; snr_d_arr[i] = A_m * np.sqrt(TtCT)
        # cov
        Cinv = np.linalg.inv(wp_cov[np.ix_(mask_r, mask_r)])
        TtCT_c = float(T_m @ Cinv @ T_m)
        TtCr_c = float(T_m @ Cinv @ res_m)
        if TtCT_c > 0:
            A_m_c = TtCr_c / TtCT_c
            snr_c_arr[i] = A_m_c * np.sqrt(TtCT_c)
    ax_rmin.plot(rmin_grid, snr_d_arr, "C0-o", ms=4, label="diagonal sigma")
    ax_rmin.plot(rmin_grid, snr_c_arr, "C3-s", ms=4, label="full JK cov")
    ax_rmin.axhline(0, color="k", lw=0.5)
    ax_rmin.set_xlabel(r"$r_p^{\min}$ for matched filter [Mpc/h]")
    ax_rmin.set_ylabel("BAO SNR")
    ax_rmin.set_title(r"matched-filter SNR vs $r_p^{\min}$")
    ax_rmin.legend(fontsize=9); ax_rmin.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_bao_filter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
