"""DESI DR1 QSO BAO -- analytic-RR + matched filter on spectro-z data.

Same pipeline as ``demo_quaia_bao_filter.py`` but on DESI DR1 QSOs.
The big difference is that DESI redshifts are *spectroscopic*
(sigma_chi ~ 1 Mpc/h) rather than photometric (sigma_chi ~ 170 Mpc/h
for Quaia G < 20), so the BAO bump in wp(rp) is essentially un-smeared
along the LOS. Combined with ~3x the QSO count, DESI Y1 has reported
a ~3-4 sigma BAO detection on this exact tracer (Adame et al. 2024).

Pipeline:
  1. Read QSO_NGC + QSO_SGC clustering catalogs and (one) random
     realization. Apply 0.8 <= z <= 2.1 (the standard DESI Y1 QSO
     BAO range).
  2. Build the angular completeness mask at NSIDE=256 by binning
     the random catalog -- DESI randoms Poisson-sample the survey
     footprint with completeness already imprinted, so the binned
     density IS the mask.
  3. Analytic RR + DR from (mask, n(z)) -- skip MC pair-counting on
     the random catalog (~1000x faster than computing RR from the
     full DESI random with N_r ~ 50M).
  4. Matched filter on wp(rp) residual = wp_data - b^2 wp_smooth
     against the BAO template T(rp) = b^2 (wp_full - wp_nowiggle).
  5. alpha-scan with parabolic interpolation around the minimum to
     get alpha_hat +/- sigma_alpha.

Files required (drop in ``data/desi/``):
  QSO_NGC_clustering.dat.fits      (data, north galactic cap)
  QSO_SGC_clustering.dat.fits      (data, south galactic cap)
  QSO_NGC_0_clustering.ran.fits    (one of 18 randoms, NGC)
  QSO_SGC_0_clustering.ran.fits    (one of 18 randoms, SGC)

These live at ``data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/
iron/LSScats/v1.5/`` (or v1.2). See the DESI DR1 release page.

Output: ``demos/figures/desi_qso_bao_filter.png``
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
from twopt_density.desi import (
    angular_completeness_from_randoms, load_desi_qso,
)
from twopt_density.distance import DistanceCosmo
from twopt_density.limber import (
    make_wp_fft, wp_observed, wp_observed_nowiggle,
)
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "desi")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env(n, d, cast=str):
    v = os.environ.get(n)
    return cast(v) if v else d


def _check_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        msg = (
            "missing DESI DR1 LSS files (sandbox firewall blocks "
            "data.desi.lbl.gov; download these manually):\n  "
            + "\n  ".join(missing)
            + "\nThey live at "
            "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/"
            "LSS/iron/LSScats/v1.5/"
        )
        raise FileNotFoundError(msg)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8 = _env("DESI_SIGMA8", 0.81, float)
    pi_max = _env("DESI_PI_MAX", 200.0, float)
    rp_max = _env("DESI_RP_MAX", 200.0, float)
    z_min = _env("DESI_Z_MIN", 0.8, float)
    z_max = _env("DESI_Z_MAX", 2.1, float)
    nside = _env("DESI_NSIDE", 256, int)
    n_data_max = _env("DESI_N_DATA", 0, int)        # 0 = use all
    n_random_for_mask = _env("DESI_N_RANDOM_MASK", 2_000_000, int)

    cat_paths = [
        os.path.join(DATA_DIR, "QSO_NGC_clustering.dat.fits"),
        os.path.join(DATA_DIR, "QSO_SGC_clustering.dat.fits"),
    ]
    ran_paths = [
        os.path.join(DATA_DIR, "QSO_NGC_0_clustering.ran.fits"),
        os.path.join(DATA_DIR, "QSO_SGC_0_clustering.ran.fits"),
    ]
    _check_files(cat_paths + ran_paths)

    print("loading DESI DR1 QSO ...")
    t = time.perf_counter()
    cat = load_desi_qso(
        catalog_paths=cat_paths,
        randoms_paths=ran_paths,
        fid_cosmo=fid, z_min=z_min, z_max=z_max,
        n_random_max=n_random_for_mask, rng_seed=0,
    )
    print(f"  {time.perf_counter()-t:.0f}s, "
          f"N_data = {cat.N_data:,}, "
          f"N_random_subsample = {cat.N_random:,}")

    rng = np.random.default_rng(0)
    if n_data_max and cat.N_data > n_data_max:
        idx = rng.choice(cat.N_data, n_data_max, replace=False)
        xyz_d = cat.xyz_data[idx]; z_d = cat.z_data[idx]
        w_d = cat.w_data[idx]
    else:
        xyz_d = cat.xyz_data; z_d = cat.z_data; w_d = cat.w_data
    print(f"  using N_data = {len(xyz_d):,} for pair counting")
    # mean-1 normalise weights so the analytic-RR / DR formulas work
    w_d = w_d / w_d.mean()

    print(f"\nbuilding angular completeness mask at NSIDE = {nside} ...")
    t = time.perf_counter()
    mask = angular_completeness_from_randoms(
        cat.ra_random, cat.dec_random, nside=nside, w_random=cat.w_random,
    )
    print(f"  {time.perf_counter()-t:.1f}s, "
          f"f_sky = {(mask > 0).mean():.3f}, "
          f"<mask> = {mask.mean():.3f}")

    rp_edges = np.concatenate([
        np.logspace(np.log10(5.0), np.log10(50.0), 8),
        np.linspace(60.0, rp_max, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, pi_max, 41)
    rp_centres = 0.5 * (rp_edges[:-1] + rp_edges[1:])

    # ---- 1. analytic RR + DR + DD-only pair count ----
    print("\nDD pair counts (FKP-weighted) ...")
    t = time.perf_counter()
    DD = _count_pairs_rp_pi(xyz_d, xyz_d, rp_edges, pi_edges,
                              auto=True, chunk=4000,
                              w1=w_d, w2=w_d)
    print(f"  {time.perf_counter()-t:.0f}s")

    print("analytic RR from (mask, n(z)) ...")
    t = time.perf_counter()
    N_d = len(xyz_d)
    N_r_eff = 10 * N_d
    res = rr_analytic(rp_edges, pi_edges, mask, nside, z_d, fid, N_r=N_r_eff)
    print(f"  {time.perf_counter()-t:.0f}s")

    # ---- 2. small-N MC RR for calibration ----
    print("calibration MC RR ...")
    cal_rng = np.random.default_rng(1)
    n_cal_d = min(20_000, N_d)
    n_cal_r = min(60_000, cat.N_random)
    i_cd = cal_rng.choice(N_d, n_cal_d, replace=False)
    i_cr = cal_rng.choice(cat.N_random, n_cal_r, replace=False)
    t = time.perf_counter()
    meas_cal = wp_landy_szalay(xyz_d[i_cd], cat.xyz_random[i_cr],
                                 rp_edges, pi_max=pi_max, n_pi=40)
    ana_cal = rr_analytic(rp_edges, meas_cal.pi_edges, mask, nside,
                            z_d[i_cd], fid, N_r=n_cal_r)
    calib = calibrate_norm_to_mc(ana_cal.RR, meas_cal.RR)
    print(f"  {time.perf_counter()-t:.0f}s, calib factor = {calib:.4f}")

    RR = calib * res.RR
    DR = dr_analytic(N_d, N_r_eff, RR)

    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r_eff * (N_r_eff - 1) / 2.0
    DD_n = DD / Nd_pairs
    DR_n = DR / (N_d * N_r_eff)
    RR_n = RR / Nr_pairs
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_2d = (DD_n - 2 * DR_n + RR_n) / RR_n
        xi_2d[~np.isfinite(xi_2d)] = 0.0
    pi_centres = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    dpi = np.diff(pi_edges)
    wp_full = 2.0 * np.sum(xi_2d * dpi[None, :], axis=1)
    sigma_wp_diag = np.sqrt(np.maximum(2.0 / RR.sum(axis=1), 1e-30)) * 50.0
    print(f"\nwp(rp~100) = {np.interp(100.0, rp_centres, wp_full):.3f}")

    # ---- 3. forward models ----
    z_eff = float(np.median(z_d))
    sigma_chi_eff = 1.0     # DESI is spectroscopic; LOS smear << BAO
    fft, k_np = make_wp_fft()
    k_grid_jax = jnp.asarray(k_np)
    rp_pred_fine = np.linspace(5.0, rp_max, 200)
    wp_full_b1 = np.asarray(wp_observed(
        jnp.asarray(rp_pred_fine), z_eff=z_eff,
        sigma_chi_eff=sigma_chi_eff, cosmo=fid,
        bias=1.0, pi_max=pi_max, sigma8=sigma8,
        fft=fft, k_grid=k_grid_jax))
    wp_smooth_b1 = np.asarray(wp_observed_nowiggle(
        jnp.asarray(rp_pred_fine), z_eff=z_eff,
        sigma_chi_eff=sigma_chi_eff, cosmo=fid,
        bias=1.0, pi_max=pi_max, sigma8=sigma8,
        fft=fft, k_grid=k_grid_jax))

    use_b = (rp_centres > 8.0) & (rp_centres < 50.0)
    wp_smooth_at = np.interp(rp_centres, rp_pred_fine, wp_smooth_b1)
    invsig2 = 1.0 / np.maximum(sigma_wp_diag[use_b], 1e-3) ** 2
    num = float(np.sum(wp_full[use_b] * wp_smooth_at[use_b] * invsig2))
    den = float(np.sum(wp_smooth_at[use_b] ** 2 * invsig2))
    b_fit = float(np.sqrt(max(num / den, 0.01)))
    print(f"\nbias fit (8 < rp < 50, diag sigma): b = {b_fit:.3f}")

    # ---- 4. matched filter ----
    print("\nmatched filter ...")
    T = bao_template(rp_centres, b=b_fit, z_eff=z_eff,
                       sigma_chi_eff=sigma_chi_eff, pi_max=pi_max,
                       cosmo=fid, alpha=1.0, sigma8=sigma8)
    wp_smooth_at_meas = b_fit ** 2 * np.interp(
        rp_centres, rp_pred_fine, wp_smooth_b1)
    A_d, sdA_d, snr_d, c2null_d, c2best_d = matched_filter_amplitude(
        wp_full, wp_smooth_at_meas, T, sigma_wp_diag,
    )
    print(f"  diagonal sigma:  A = {A_d:.2f} +/- {sdA_d:.2f}, "
          f"SNR = {snr_d:.2f}")
    print(f"    chi2_null = {c2null_d:.1f}, chi2_best = {c2best_d:.1f}, "
          f"delta_chi2 = {c2null_d - c2best_d:.2f}")

    # ---- 5. alpha scan ----
    print("\nalpha scan (BAO scaling parameter) ...")
    alpha_grid = np.linspace(0.85, 1.15, 31)
    out_d = bao_alpha_scan(rp_centres, wp_full, wp_smooth_at_meas, b_fit,
                             z_eff, sigma_chi_eff, pi_max, fid,
                             sigma_wp_diag, alpha_grid=alpha_grid,
                             sigma8=sigma8)
    print(f"  diagonal: alpha_hat = {out_d['alpha_hat']:.4f} +/- "
          f"{out_d['sigma_alpha']:.4f}, SNR = {out_d['SNR_at_best']:.2f}")

    # ---- figure ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax_res, ax_alpha = axs

    res_arr = wp_full - wp_smooth_at_meas
    ax_res.errorbar(rp_centres, res_arr, yerr=sigma_wp_diag, fmt="ok",
                     ms=4, capsize=3,
                     label=r"$w_p^{data} - b^2 w_p^{smooth}$")
    ax_res.plot(rp_centres, A_d * T, "C3-", lw=2,
                 label=fr"$A \cdot T(r_p)$, "
                        fr"$A={A_d:.2f}\pm{sdA_d:.2f}$ "
                        fr"(SNR$={snr_d:.1f}$)")
    ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_xscale("log")
    ax_res.axvspan(80, 130, alpha=0.10, color="C1")
    ax_res.set_xlabel(r"$r_p$ [Mpc/h]"); ax_res.set_ylabel(r"$\Delta w_p$")
    ax_res.set_title(r"DESI DR1 QSO BAO matched filter "
                       fr"(N={N_d:,}, $b={b_fit:.2f}$)")
    ax_res.legend(fontsize=9); ax_res.grid(alpha=0.3, which="both")

    ax_alpha.plot(out_d["alpha_grid"],
                    out_d["chi2"] - out_d["chi2"].min(),
                    "C0-o", ms=4)
    ax_alpha.axvline(1.0, color="k", lw=0.5, ls=":")
    ax_alpha.axvline(out_d["alpha_hat"], color="C3", lw=2,
                       label=fr"$\alpha={out_d['alpha_hat']:.4f}\pm"
                              fr"{out_d['sigma_alpha']:.4f}$, "
                              fr"SNR$={out_d['SNR_at_best']:.1f}$")
    ax_alpha.set_xlabel(r"BAO scaling $\alpha$")
    ax_alpha.set_ylabel(r"$\chi^2 - \chi^2_{\min}$")
    ax_alpha.set_title(r"$\chi^2(\alpha)$ scan, DESI Y1 QSO")
    ax_alpha.legend(fontsize=9); ax_alpha.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "desi_qso_bao_filter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
