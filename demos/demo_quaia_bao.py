"""Search for the BAO peak in Quaia wp(rp) with the photo-z-aware
forward model.

Hypothesis: Quaia's spectro-photometric redshift errors smear
xi(s_parallel) over ~ 100 Mpc/h LOS scales, comparable to the BAO
sound horizon. But wp(rp) integrates over LOS up to pi_max -- as long
as pi_max is wider than the photo-z smearing, *all* the scattered
pairs are recovered, and the BAO peak survives in wp(rp) at
rp ~ 100 Mpc/h.

Pipeline:
  1. wp(rp) on the full Quaia G < 20 z-cut sample with extended
     rp out to 150 Mpc/h and pi_max = 300 Mpc/h.
  2. Two photo-z-aware Limber predictions at the data n(z) using:
       - ``wp_observed``           halofit + BAO wiggles (full)
       - ``wp_observed_nowiggle``  halofit + EH zero-baryon (smooth)
  3. Bias fit on small-rp wp_smooth (rp < 60 Mpc/h, away from BAO).
  4. Plot wp(rp), the two model curves, and the residual
     wp_meas - b^2 wp_nowiggle. The BAO peak should appear at
     rp ~ 100 Mpc/h.

Defaults sized for ~10 minute laptop run. ``QUAIA_N_DATA`` /
``QUAIA_N_RANDOM`` env vars scale up to the full sample for tighter
errors. ``QUAIA_PI_MAX`` controls the LOS integration window
(default 300 Mpc/h, well past Quaia's ~ 100 Mpc/h photo-z smearing).
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

from twopt_density.distance import DistanceCosmo
from twopt_density.limber import (
    make_wp_fft, sample_pair_sigma_chi, sigma_chi_from_sigma_z,
    wp_observed, wp_observed_nowiggle,
)
from twopt_density.projected_xi import wp_landy_szalay
from twopt_density.quaia import load_quaia


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


def panel_bao(meas, rp_pred, wp_full, wp_smooth, b_fit, sigma_chi_eff,
               pi_max_val, out_path):
    """3-row diagnostic plot:
      top:    wp(rp) measurement + halofit-with-BAO and -no-wiggle
      middle: residual wp_meas - b^2 wp_nowiggle vs rp;
              the BAO bump should peak at rp ~ 100 Mpc/h.
      bottom: ratio wp_meas / wp_nowiggle.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    ax_top, ax_res, ax_rat = axs
    rp_c = meas.rp_centres
    wp_meas = meas.wp
    DD_per_rp = meas.DD.sum(axis=1) + 1.0
    sigma_wp = pi_max_val / np.sqrt(DD_per_rp)
    yerr = sigma_wp

    ax_top.errorbar(rp_c, wp_meas, yerr=yerr, fmt="ok", markersize=4,
                     capsize=3, label=fr"Quaia $w_p$ (LS, $\pi_{{\max}}=$"
                                      f"{pi_max_val:.0f} Mpc/h)")
    ax_top.plot(rp_pred, b_fit ** 2 * wp_full, "C0-", lw=2,
                 label=fr"halofit + BAO, photo-z aware, $b={b_fit:.2f}$")
    ax_top.plot(rp_pred, b_fit ** 2 * wp_smooth, "C7--", lw=2, alpha=0.8,
                 label=fr"halofit no-wiggle (EH zero-baryon), $b={b_fit:.2f}$")
    ax_top.set_xscale("log"); ax_top.set_yscale("symlog", linthresh=0.1)
    ax_top.set_ylabel(r"$w_p(r_p)$ [Mpc/h]")
    ax_top.set_title(r"Quaia G$<$20: BAO search in $w_p(r_p)$"
                      r" -- photo-z aware forward model")
    ax_top.legend(fontsize=9); ax_top.grid(alpha=0.3, which="both")
    ax_top.axhline(0, color="k", lw=0.5, alpha=0.3)

    # residual
    wp_smooth_at_meas = np.interp(rp_c, rp_pred, b_fit ** 2 * wp_smooth)
    wp_full_at_meas = np.interp(rp_c, rp_pred, b_fit ** 2 * wp_full)
    res = wp_meas - wp_smooth_at_meas
    ax_res.errorbar(rp_c, res, yerr=yerr, fmt="ok", markersize=4, capsize=3,
                     label=r"$w_p^{\rm data} - b^2 w_p^{\rm nowiggle}$")
    ax_res.plot(rp_pred, b_fit ** 2 * (wp_full - wp_smooth), "C0-", lw=2,
                 label=r"halofit BAO contribution")
    ax_res.set_xscale("log"); ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_ylabel(r"$\Delta w_p$ [Mpc/h]")
    ax_res.legend(fontsize=9); ax_res.grid(alpha=0.3, which="both")
    ax_res.axvspan(80, 130, alpha=0.10, color="C1",
                    label="BAO sound horizon")

    # ratio
    ratio = wp_meas / np.maximum(np.abs(wp_smooth_at_meas), 1e-3)
    ratio = np.sign(wp_smooth_at_meas) * ratio
    ratio_err = yerr / np.maximum(np.abs(wp_smooth_at_meas), 1e-3)
    ax_rat.errorbar(rp_c, ratio, yerr=ratio_err, fmt="ok", markersize=4,
                     capsize=3, label=r"$w_p^{\rm data} / (b^2 w_p^{\rm nowiggle})$")
    ax_rat.plot(rp_pred, wp_full / np.maximum(np.abs(wp_smooth), 1e-3)
                 * np.sign(wp_smooth),
                 "C0-", lw=2, label=r"halofit BAO/no-wiggle ratio")
    ax_rat.set_xscale("log"); ax_rat.axhline(1, color="k", lw=0.5)
    ax_rat.set_xlabel(r"$r_p$ [Mpc/h]")
    ax_rat.set_ylabel(r"$w_p / w_p^{\rm nowiggle}$")
    ax_rat.legend(fontsize=9); ax_rat.grid(alpha=0.3, which="both")
    ax_rat.axvspan(80, 130, alpha=0.10, color="C1")
    ax_rat.set_ylim(-2, 4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8_fixed = _env_float("QUAIA_SIGMA8", 0.81)
    n_data_max = _env_int("QUAIA_N_DATA", 250000)
    n_random_max = _env_int("QUAIA_N_RANDOM", 750000)
    pi_max = _env_float("QUAIA_PI_MAX", 300.0)
    rp_max = _env_float("QUAIA_RP_MAX", 150.0)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)

    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=3, rng_seed=0,
    )
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    mr = (cat.z_random >= z_min) & (cat.z_random <= z_max)
    rng = np.random.default_rng(0)
    xyz_d = np.asarray(cat.xyz_data[md])
    xyz_r = np.asarray(cat.xyz_random[mr])
    z_d = cat.z_data[md]
    sig_z = cat.z_data_err[md]
    if len(xyz_d) > n_data_max:
        i = rng.choice(len(xyz_d), n_data_max, replace=False)
        xyz_d, z_d, sig_z = xyz_d[i], z_d[i], sig_z[i]
    if len(xyz_r) > n_random_max:
        i = rng.choice(len(xyz_r), n_random_max, replace=False)
        xyz_r = xyz_r[i]
    print(f"subsample: N_d={len(xyz_d):,}, N_r={len(xyz_r):,}, "
          f"pi_max={pi_max:.0f} Mpc/h, rp_max={rp_max:.0f} Mpc/h")

    # finer rp binning around BAO scale
    rp_edges = np.concatenate([
        np.logspace(np.log10(5.0), np.log10(50.0), 8),
        np.linspace(60.0, 150.0, 14)[1:],
    ])

    print(f"\npair counts (s_max ~ {np.sqrt(rp_max ** 2 + pi_max ** 2):.0f} Mpc/h) ...")
    t0 = time.perf_counter()
    meas = wp_landy_szalay(xyz_d, xyz_r, rp_edges, pi_max=pi_max, n_pi=60)
    print(f"  {time.perf_counter()-t0:.0f}s, DD={meas.DD.sum():.0f}, "
          f"DR={meas.DR.sum():.0f}, RR={meas.RR.sum():.0f}")

    # photo-z smearing
    sig_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d, sig_z, fid))
    sigma_chi_eff = float(np.sqrt(2.0) * np.median(sig_chi_per_obj))
    z_eff = float(np.median(z_d))
    print(f"  z_eff = {z_eff:.3f}, sigma_chi_eff = {sigma_chi_eff:.0f} Mpc/h")

    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    rp_pred = np.linspace(5.0, rp_max, 200)
    print("forward models: halofit-with-BAO and halofit-no-wiggle ...")
    t0 = time.perf_counter()
    wp_full = np.asarray(wp_observed(
        jnp.asarray(rp_pred), z_eff=z_eff, sigma_chi_eff=sigma_chi_eff,
        cosmo=fid, bias=1.0, pi_max=pi_max, sigma8=sigma8_fixed,
        fft=fft, k_grid=k_grid,
    ))
    wp_smooth = np.asarray(wp_observed_nowiggle(
        jnp.asarray(rp_pred), z_eff=z_eff, sigma_chi_eff=sigma_chi_eff,
        cosmo=fid, bias=1.0, pi_max=pi_max, sigma8=sigma8_fixed,
        fft=fft, k_grid=k_grid,
    ))
    print(f"  {time.perf_counter()-t0:.1f}s")
    bao_amp = wp_full - wp_smooth
    bao_peak_rp = float(rp_pred[np.argmax(bao_amp)])
    print(f"  predicted BAO peak in wp(rp) at rp = {bao_peak_rp:.0f} Mpc/h, "
          f"amplitude {bao_amp.max():.3f} Mpc/h (at b=1)")

    # bias fit on rp < 50 Mpc/h, where BAO is small and signal is strong
    use = (meas.rp_centres > 10.0) & (meas.rp_centres < 50.0)
    DD_per_rp = meas.DD.sum(axis=1) + 1.0
    sigma_wp = pi_max / np.sqrt(DD_per_rp)
    wp_smooth_at = np.interp(meas.rp_centres, rp_pred, wp_smooth)
    num = float(np.sum(meas.wp[use] * wp_smooth_at[use]
                        / sigma_wp[use] ** 2))
    den = float(np.sum(wp_smooth_at[use] ** 2 / sigma_wp[use] ** 2))
    b2 = max(num / den, 0.01)
    b_fit = float(np.sqrt(b2))
    print(f"  bias fit on smooth model (10 < rp < 50 Mpc/h): b = {b_fit:.3f}")

    panel_bao(meas, rp_pred, wp_full, wp_smooth, b_fit, sigma_chi_eff,
               pi_max, os.path.join(FIG_DIR, "quaia_bao_search.png"))
    print("  wrote quaia_bao_search.png")
    print()
    print("=== BAO-band wp(rp) summary (rp = 70..130 Mpc/h) ===")
    print(f"  rp    wp_meas   sigma   b^2 wp_smooth   b^2 wp_full   "
          f"data - smooth")
    for j, rc in enumerate(meas.rp_centres):
        if 70.0 < rc < 130.0:
            wp_full_at_j = np.interp(rc, rp_pred, b_fit ** 2 * wp_full)
            wp_smooth_at_j = np.interp(rc, rp_pred, b_fit ** 2 * wp_smooth)
            print(f"  {rc:5.1f} {meas.wp[j]:7.3f} {sigma_wp[j]:6.3f}   "
                  f"{wp_smooth_at_j:7.3f}        {wp_full_at_j:7.3f}    "
                  f"{meas.wp[j] - wp_smooth_at_j:+.3f}")


if __name__ == "__main__":
    main()
