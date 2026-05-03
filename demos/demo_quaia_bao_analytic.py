"""BAO search in Quaia wp(rp) using analytic RR + DR (no MC random).

Builds on ``demo_quaia_bao.py`` (which used a 3x MC random) by
replacing the random catalogue with the analytic pair-density
expressions in ``twopt_density.analytic_rr``. With the full Quaia
G<20 z-cut sample (~545k data) the limiting step is the DD pair
count (~5-10 min); the random side is now ~30 s analytic + a tiny
MC calibration run.

Pipeline:
  1. Calibration -- small MC RR on a 30k subsample to fix the
     analytic-vs-MC normalisation factor (~ 1.08 for Quaia,
     constant in (rp, pi); see ``calibrate_norm_to_mc``).
  2. Full-sample DD pair counts at extended (rp_max=150 Mpc/h,
     pi_max=300 Mpc/h).
  3. Analytic RR and DR from the healpix mask + n(z), scaled by the
     calibration factor.
  4. Landy-Szalay xi(rp, pi); wp(rp) via pi integration.
  5. Compare to halofit (with and without BAO wiggles) and plot the
     residual + ratio.

Outputs::

  quaia_bao_analytic.png   wp(rp) + halofit BAO and smooth + residual
                            + ratio panels.
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
    angular_corr_from_mask, calibrate_norm_to_mc, dr_analytic, rr_analytic,
)
from twopt_density.distance import DistanceCosmo
from twopt_density.limber import (
    make_wp_fft, sigma_chi_from_sigma_z, wp_observed, wp_observed_nowiggle,
)
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function


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


def panel_bao(rp_centres, wp_data, sigma_wp, rp_pred, wp_full, wp_smooth,
                b_fit, sigma_chi_eff, pi_max, calib, n_d, out_path):
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)
    ax_top, ax_res, ax_rat = axs

    ax_top.errorbar(rp_centres, wp_data, yerr=sigma_wp, fmt="ok", markersize=4,
                     capsize=3,
                     label=fr"Quaia G$<$20 (N$_d$={n_d:,}; analytic RR, "
                            fr"calib={calib:.3f})")
    ax_top.plot(rp_pred, b_fit ** 2 * wp_full, "C0-", lw=2,
                 label=fr"halofit + BAO, photo-z aware, $b={b_fit:.2f}$")
    ax_top.plot(rp_pred, b_fit ** 2 * wp_smooth, "C7--", lw=2, alpha=0.8,
                 label=fr"halofit no-wiggle (EH zero-baryon), $b={b_fit:.2f}$")
    ax_top.set_xscale("log"); ax_top.set_yscale("symlog", linthresh=0.1)
    ax_top.set_ylabel(r"$w_p(r_p)$ [Mpc/h]")
    ax_top.set_title(r"Quaia G$<$20: full-sample BAO search with"
                      r" analytic RR/DR -- no MC random")
    ax_top.legend(fontsize=9); ax_top.grid(alpha=0.3, which="both")
    ax_top.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_top.axvspan(80, 130, alpha=0.10, color="C1")

    wp_smooth_at = np.interp(rp_centres, rp_pred, b_fit ** 2 * wp_smooth)
    res = wp_data - wp_smooth_at
    ax_res.errorbar(rp_centres, res, yerr=sigma_wp, fmt="ok", markersize=4,
                     capsize=3, label=r"$w_p^{\rm data} - b^2 w_p^{\rm nowiggle}$")
    ax_res.plot(rp_pred, b_fit ** 2 * (wp_full - wp_smooth), "C0-", lw=2,
                 label=r"halofit BAO contribution")
    ax_res.set_xscale("log"); ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_ylabel(r"$\Delta w_p$ [Mpc/h]")
    ax_res.legend(fontsize=9); ax_res.grid(alpha=0.3, which="both")
    ax_res.axvspan(80, 130, alpha=0.10, color="C1",
                    label="BAO sound horizon")

    # ratio
    rat = wp_data / np.where(np.abs(wp_smooth_at) > 0.01, wp_smooth_at, np.nan)
    rat_err = sigma_wp / np.maximum(np.abs(wp_smooth_at), 0.01)
    ax_rat.errorbar(rp_centres, rat, yerr=rat_err, fmt="ok", markersize=4,
                     capsize=3,
                     label=r"$w_p^{\rm data} / (b^2 w_p^{\rm nowiggle})$")
    rat_pred = wp_full / np.maximum(np.abs(wp_smooth), 0.001) * np.sign(wp_smooth)
    ax_rat.plot(rp_pred, rat_pred, "C0-", lw=2,
                 label=r"halofit BAO/no-wiggle ratio")
    ax_rat.axhline(1, color="k", lw=0.5)
    ax_rat.set_xscale("log")
    ax_rat.set_xlabel(r"$r_p$ [Mpc/h]")
    ax_rat.set_ylabel(r"$w_p / w_p^{\rm nowiggle}$")
    ax_rat.set_ylim(-1, 4); ax_rat.legend(fontsize=9)
    ax_rat.grid(alpha=0.3, which="both")
    ax_rat.axvspan(80, 130, alpha=0.10, color="C1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8_fixed = _env_float("QUAIA_SIGMA8", 0.81)
    n_data_max = _env_int("QUAIA_N_DATA", 545000)        # default: full sample
    pi_max = _env_float("QUAIA_PI_MAX", 300.0)
    rp_max = _env_float("QUAIA_RP_MAX", 150.0)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)
    n_calib_data = _env_int("QUAIA_N_CALIB", 30000)
    n_calib_random = _env_int("QUAIA_N_CALIB_RANDOM", 90000)

    print("load Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=2, rng_seed=0,
    )
    mask, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    mr = (cat.z_random >= z_min) & (cat.z_random <= z_max)
    rng = np.random.default_rng(0)
    xyz_d_all = np.asarray(cat.xyz_data[md])
    z_d_all = cat.z_data[md]
    sig_z_all = cat.z_data_err[md]
    if len(xyz_d_all) > n_data_max:
        idx = rng.choice(len(xyz_d_all), n_data_max, replace=False)
        xyz_d_all, z_d_all, sig_z_all = xyz_d_all[idx], z_d_all[idx], sig_z_all[idx]
    print(f"data sample: N_d = {len(xyz_d_all):,}, "
          f"f_sky_eff = {mask.mean():.3f}, "
          f"<chi^2_p>(KDE) computed analytically")

    # rp grid: log-spaced small + linear around BAO scale for resolution
    rp_edges = np.concatenate([
        np.logspace(np.log10(5.0), np.log10(50.0), 8),
        np.linspace(60.0, rp_max, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, pi_max, 60 + 1)

    # ---- 1. calibration: small MC RR + analytic RR -> norm factor ----
    print()
    print(f"=== calibration: MC RR on {n_calib_data:,} x {n_calib_random:,} "
          "random subsample ===")
    xyz_r_all = np.asarray(cat.xyz_random[mr])
    cal_rng = np.random.default_rng(1)
    i_d = cal_rng.choice(len(xyz_d_all), n_calib_data, replace=False)
    i_r = cal_rng.choice(len(xyz_r_all), n_calib_random, replace=False)
    xyz_d_cal = xyz_d_all[i_d]; xyz_r_cal = xyz_r_all[i_r]
    z_d_cal = z_d_all[i_d]
    t = time.perf_counter()
    meas_cal = wp_landy_szalay(xyz_d_cal, xyz_r_cal, rp_edges,
                                 pi_max=pi_max, n_pi=60)
    print(f"  MC pair counts: {time.perf_counter()-t:.0f}s")
    t = time.perf_counter()
    ana_cal = rr_analytic(rp_edges, meas_cal.pi_edges, mask, nside, z_d_cal,
                            fid, N_r=n_calib_random)
    calib = calibrate_norm_to_mc(ana_cal.RR, meas_cal.RR)
    print(f"  analytic RR: {time.perf_counter()-t:.1f}s")
    print(f"  calibration factor MC/analytic = {calib:.4f}")

    # ---- 2. DD pair counts on the full data sample ----
    print()
    print(f"=== DD pair counts on {len(xyz_d_all):,} data points ===")
    t = time.perf_counter()
    DD = _count_pairs_rp_pi(xyz_d_all, xyz_d_all, rp_edges, pi_edges,
                              auto=True, chunk=4000)
    print(f"  {time.perf_counter()-t:.0f}s, DD total = {DD.sum():.0f}")

    # ---- 3. analytic RR (calibrated) and DR ----
    N_r_eff = 10 * len(xyz_d_all)        # standard 10x random equivalent
    print()
    print(f"=== analytic RR + DR (N_r_effective = {N_r_eff:,}) ===")
    t = time.perf_counter()
    ana_full = rr_analytic(rp_edges, pi_edges, mask, nside, z_d_all,
                             fid, N_r=N_r_eff)
    RR = calib * ana_full.RR
    print(f"  analytic RR: {time.perf_counter()-t:.1f}s, "
          f"RR total = {RR.sum():.0f}")
    DR = dr_analytic(len(xyz_d_all), N_r_eff, RR)

    # ---- 4. LS estimator -> wp(rp) ----
    N_d = len(xyz_d_all)
    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r_eff * (N_r_eff - 1) / 2.0
    DD_n = DD / Nd_pairs
    DR_n = DR / (N_d * N_r_eff)
    RR_n = RR / Nr_pairs
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(RR_n > 0, (DD_n - 2 * DR_n + RR_n) / RR_n, 0.0)
    d_pi = np.diff(pi_edges)
    wp_data = 2.0 * np.sum(xi * d_pi[None, :], axis=1)
    rp_centres = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    DD_per_rp = DD.sum(axis=1) + 1.0
    sigma_wp = pi_max / np.sqrt(DD_per_rp)

    # ---- 5. halofit forward models ----
    sig_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d_all, sig_z_all, fid))
    sigma_chi_eff = float(np.sqrt(2.0) * np.median(sig_chi_per_obj))
    z_eff = float(np.median(z_d_all))
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    rp_pred = np.linspace(5.0, rp_max, 200)
    print(f"\nforward models (z_eff={z_eff:.2f}, sigma_chi_eff={sigma_chi_eff:.0f} Mpc/h) ...")
    t = time.perf_counter()
    wp_full = np.asarray(wp_observed(jnp.asarray(rp_pred), z_eff=z_eff,
                                       sigma_chi_eff=sigma_chi_eff,
                                       cosmo=fid, bias=1.0, pi_max=pi_max,
                                       sigma8=sigma8_fixed, fft=fft,
                                       k_grid=k_grid))
    wp_smooth = np.asarray(wp_observed_nowiggle(jnp.asarray(rp_pred), z_eff=z_eff,
                                                  sigma_chi_eff=sigma_chi_eff,
                                                  cosmo=fid, bias=1.0,
                                                  pi_max=pi_max,
                                                  sigma8=sigma8_fixed,
                                                  fft=fft, k_grid=k_grid))
    print(f"  {time.perf_counter()-t:.1f}s")
    bao_amp = wp_full - wp_smooth
    bao_peak = float(rp_pred[np.argmax(bao_amp)])
    print(f"  BAO bump in wp(rp) at rp = {bao_peak:.0f} Mpc/h, "
          f"amp = {bao_amp.max():.3f} Mpc/h (b=1)")

    # bias fit on the smooth model where the BAO contribution is small
    use = (rp_centres > 8.0) & (rp_centres < 50.0)
    wp_smooth_at = np.interp(rp_centres, rp_pred, wp_smooth)
    num = float(np.sum(wp_data[use] * wp_smooth_at[use] / sigma_wp[use] ** 2))
    den = float(np.sum(wp_smooth_at[use] ** 2 / sigma_wp[use] ** 2))
    b2 = max(num / den, 0.01)
    b_fit = float(np.sqrt(b2))
    print(f"  bias fit (8 < rp < 50, away from BAO): b = {b_fit:.3f}")

    panel_bao(rp_centres, wp_data, sigma_wp, rp_pred, wp_full, wp_smooth,
                b_fit, sigma_chi_eff, pi_max, calib, len(xyz_d_all),
                os.path.join(FIG_DIR, "quaia_bao_analytic.png"))
    print("  wrote quaia_bao_analytic.png")

    print()
    print("=== BAO band data summary ===")
    print(f"  rp        wp_data    sigma     b^2 wp_smooth   b^2 wp_full   "
          f"residual")
    for i, rc in enumerate(rp_centres):
        if 70.0 < rc < 130.0:
            wp_full_at = np.interp(rc, rp_pred, b_fit ** 2 * wp_full)
            wp_sm_at = np.interp(rc, rp_pred, b_fit ** 2 * wp_smooth)
            print(f"  {rc:6.2f}  {wp_data[i]:8.4f}  {sigma_wp[i]:7.4f}   "
                  f"{wp_sm_at:8.4f}        {wp_full_at:8.4f}    "
                  f"{wp_data[i] - wp_sm_at:+.4f}")


if __name__ == "__main__":
    main()
