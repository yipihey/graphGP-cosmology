"""Quaia x CMB-lensing forecast + ingestion path for a real measurement.

Builds the Limber forward model for ``C_ell^{g-kappa}`` (CMB lensing
cross galaxy-density) and forecasts the SNR on Quaia G < 20 against
Planck PR3 lensing reconstruction noise. Layers a joint
``wp(rp) + C_ell^{g-kappa}`` likelihood so cosmology gradients flow
through both probes simultaneously -- this is the channel that breaks
the sigma_8-b degeneracy that wp and Cl^{gg} can't do alone (lensing
scales as b sigma_8 D, auto-clustering as b^2 sigma_8^2 D^2).

Outputs:

  quaia_lensing_forecast.png  -- four panels:
    (a) lensing kernel W_kappa(z) overlaid on Quaia n(z)
    (b) C_ell predictions: gg, kk_signal, kk_noise, and gk
    (c) per-ell SNR^2 contribution and cumulative SNR
    (d) (sigma_8, b) constraint contours from wp alone, Cl^{gk} alone,
        and joint -- demonstrating the degeneracy break.

Once a real Planck kappa map is on disk (NSIDE = 64 or 1024 ud_grade'd
to NSIDE = 64), the existing NaMaster pseudo-Cl machinery in
``twopt_density.angular`` does the measurement directly via
``compute_cl_gg`` with delta_g and kappa as the two NmtFields.
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
from twopt_density.lensing import (
    cl_gkappa_limber, cl_kappa_kappa_planck_pr3,
    lensing_kernel_W_kappa, planck_pr3_lensing_noise,
    quaia_gkappa_snr_forecast,
)
from twopt_density.limber import cl_gg_limber
from twopt_density.quaia import load_quaia


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_float(n, d):
    v = os.environ.get(n)
    return float(v) if v else d


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8 = _env_float("QUAIA_SIGMA8", 0.81)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)
    f_sky = _env_float("QUAIA_FSKY", 0.66)
    N_kk_pivot = _env_float("PLANCK_NKK", 8e-8)

    print("loading Quaia n(z) ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    z_d = cat.z_data[md]
    print(f"  N = {len(z_d):,}, "
          f"z range [{z_d.min():.2f}, {z_d.max():.2f}], "
          f"median = {np.median(z_d):.3f}")

    z_grid = np.linspace(z_min + 0.05, z_max - 0.05, 60)
    nz, _ = np.histogram(z_d, bins=z_grid, density=True)
    z_centres = 0.5 * (z_grid[:-1] + z_grid[1:])

    # 1) lensing kernel
    z_fine = np.linspace(0.05, 5.0, 200)
    W_kappa_fine = np.asarray(lensing_kernel_W_kappa(jnp.asarray(z_fine), fid))

    # 2) C_ell forecasts at Planck-resolved scales
    print("\nforward models (Cl^gg, Cl^gk, Cl^kk) ...")
    ell = np.logspace(np.log10(8), np.log10(600), 40)
    b_z = np.full_like(z_centres, 2.6)
    t = time.perf_counter()
    cl_gk = np.asarray(cl_gkappa_limber(ell, z_centres, nz, b_z, fid,
                                          sigma8=sigma8))
    cl_gg = np.asarray(cl_gg_limber(ell, z_centres, nz, fid, bias=2.6,
                                      sigma8=sigma8))
    cl_kk = cl_kappa_kappa_planck_pr3(ell, fid, sigma8=sigma8)
    print(f"  {time.perf_counter()-t:.1f}s")
    N_kk = planck_pr3_lensing_noise(ell, N0=N_kk_pivot)
    N_gg = 4 * np.pi * f_sky / len(z_d)
    print(f"  N_gg shot noise = {N_gg:.3e}, N_kk PR3 (flat) = {N_kk[0]:.3e}")

    snr_total, snr2_per = quaia_gkappa_snr_forecast(
        ell, cl_gk, cl_gg + N_gg, cl_kk + N_kk, f_sky=f_sky,
    )
    print(f"\nSNR forecast Quaia G<20 x Planck PR3: {snr_total:.1f}")
    for emin, emax in [(8, 50), (50, 200), (200, 600)]:
        m = (ell > emin) & (ell <= emax)
        snr_b = float(np.sqrt(snr2_per[m].sum()))
        print(f"  ell in ({emin}, {emax}]: SNR = {snr_b:.2f}")

    # 3) Fisher: (sigma_8, b) constraint contours from wp, Cl^gk, joint
    # Treat sigma_8 and b as nuisances; build a Fisher matrix via
    # numerical derivatives of the predictions w.r.t. each parameter.
    print("\nFisher analysis (sigma_8, b) ...")
    rp_grid = np.logspace(np.log10(8), np.log10(80), 14)

    from twopt_density.limber import wp_observed
    sigma_chi_eff = 170.0; pi_max = 200.0
    z_eff = float(np.median(z_d))
    sigma_wp_diag = 0.20 * np.ones_like(rp_grid)         # representative

    def wp_pred(s8, b):
        return np.asarray(wp_observed(
            jnp.asarray(rp_grid), z_eff=z_eff,
            sigma_chi_eff=sigma_chi_eff, cosmo=fid, bias=b,
            pi_max=pi_max, sigma8=s8,
        ))
    def cl_pred(s8, b):
        return np.asarray(cl_gkappa_limber(
            ell, z_centres, nz, np.full_like(z_centres, b), fid,
            sigma8=s8,
        ))

    # numerical derivatives at fid (sigma_8, b) = (0.81, 2.6)
    s0, b0 = 0.81, 2.6
    eps = 0.01
    dwp_ds8 = (wp_pred(s0 + eps, b0) - wp_pred(s0 - eps, b0)) / (2 * eps)
    dwp_db = (wp_pred(s0, b0 + eps) - wp_pred(s0, b0 - eps)) / (2 * eps)
    dcl_ds8 = (cl_pred(s0 + eps, b0) - cl_pred(s0 - eps, b0)) / (2 * eps)
    dcl_db = (cl_pred(s0, b0 + eps) - cl_pred(s0, b0 - eps)) / (2 * eps)

    # Fisher matrices
    inv_sig_wp2 = 1.0 / sigma_wp_diag ** 2
    F_wp = np.zeros((2, 2))
    F_wp[0, 0] = np.sum(dwp_ds8 ** 2 * inv_sig_wp2)
    F_wp[0, 1] = F_wp[1, 0] = np.sum(dwp_ds8 * dwp_db * inv_sig_wp2)
    F_wp[1, 1] = np.sum(dwp_db ** 2 * inv_sig_wp2)

    sigma_cl_diag = np.sqrt((cl_gg + N_gg) * (cl_kk + N_kk)
                              / ((2 * ell + 1) * f_sky))
    inv_sig_cl2 = 1.0 / sigma_cl_diag ** 2
    F_cl = np.zeros((2, 2))
    F_cl[0, 0] = np.sum(dcl_ds8 ** 2 * inv_sig_cl2)
    F_cl[0, 1] = F_cl[1, 0] = np.sum(dcl_ds8 * dcl_db * inv_sig_cl2)
    F_cl[1, 1] = np.sum(dcl_db ** 2 * inv_sig_cl2)

    F_joint = F_wp + F_cl
    cov_wp = np.linalg.inv(F_wp + 1e-12 * np.eye(2))
    cov_cl = np.linalg.inv(F_cl + 1e-12 * np.eye(2))
    cov_joint = np.linalg.inv(F_joint + 1e-12 * np.eye(2))
    sd_wp = np.sqrt(np.diag(cov_wp))
    sd_cl = np.sqrt(np.diag(cov_cl))
    sd_j = np.sqrt(np.diag(cov_joint))
    rho_wp = cov_wp[0, 1] / max(sd_wp[0] * sd_wp[1], 1e-30)
    rho_cl = cov_cl[0, 1] / max(sd_cl[0] * sd_cl[1], 1e-30)
    rho_j = cov_joint[0, 1] / max(sd_j[0] * sd_j[1], 1e-30)
    print(f"  wp:    sigma_8 = {s0:.2f} +/- {sd_wp[0]:.3f}, "
          f"b = {b0:.2f} +/- {sd_wp[1]:.3f}, corr = {rho_wp:+.3f}")
    print(f"  Cl_gk: sigma_8 = {s0:.2f} +/- {sd_cl[0]:.3f}, "
          f"b = {b0:.2f} +/- {sd_cl[1]:.3f}, corr = {rho_cl:+.3f}")
    print(f"  joint: sigma_8 = {s0:.2f} +/- {sd_j[0]:.3f}, "
          f"b = {b0:.2f} +/- {sd_j[1]:.3f}, corr = {rho_j:+.3f}")

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax_kernel, ax_cl, ax_snr, ax_contour = axs.flatten()

    # n(z) and W_kappa
    ax_kernel.plot(z_centres, nz / nz.max(), "C0-", lw=2,
                    label="Quaia n(z) [normalised]")
    ax_kernel.plot(z_fine, W_kappa_fine / W_kappa_fine.max(), "C3-", lw=2,
                    label=r"CMB lensing $W_\kappa(z)$ [normalised]")
    overlap = np.minimum(np.interp(z_fine, z_centres, nz / nz.max()),
                          W_kappa_fine / W_kappa_fine.max())
    ax_kernel.fill_between(z_fine, 0, overlap, color="C2", alpha=0.25,
                            label="overlap (lensing efficiency)")
    ax_kernel.set_xlabel("z"); ax_kernel.set_ylabel("normalised kernel")
    ax_kernel.set_xlim(0, 5)
    ax_kernel.set_title(r"Quaia n(z) overlaps the CMB lensing kernel "
                         r"($z_{\rm peak} \sim 2$)")
    ax_kernel.legend(fontsize=9); ax_kernel.grid(alpha=0.3)

    # C_ell predictions
    ax_cl.loglog(ell, cl_gg, "C0-", lw=2, label=r"$C_\ell^{gg}$ (b=2.6)")
    ax_cl.loglog(ell, [N_gg] * len(ell), "C0:", lw=1,
                  label=r"$N_\ell^{gg}$ (Quaia shot)")
    ax_cl.loglog(ell, cl_kk, "C3-", lw=2, label=r"$C_\ell^{\kappa\kappa}$")
    ax_cl.loglog(ell, N_kk, "C3:", lw=1, label=r"$N_\ell^{\kappa\kappa}$ (Planck PR3)")
    ax_cl.loglog(ell, cl_gk, "C2-", lw=2,
                   label=r"$C_\ell^{g\kappa}$ -- the cross we want")
    ax_cl.set_xlabel(r"$\ell$"); ax_cl.set_ylabel(r"$C_\ell$")
    ax_cl.set_title("C_ell predictions and noise levels")
    ax_cl.legend(fontsize=8.5); ax_cl.grid(alpha=0.3, which="both")

    # SNR
    snr_cum = np.sqrt(np.cumsum(snr2_per))
    ax_snr.semilogx(ell, np.sqrt(snr2_per), "C0-o", ms=3,
                     label=r"per-$\ell$ SNR")
    ax_snr.semilogx(ell, snr_cum, "C3-", lw=2,
                     label=fr"cumulative SNR (total={snr_total:.1f})")
    ax_snr.set_xlabel(r"$\ell$"); ax_snr.set_ylabel("SNR")
    ax_snr.set_title(r"SNR forecast: Quaia $\times$ Planck PR3")
    ax_snr.legend(fontsize=9); ax_snr.grid(alpha=0.3, which="both")

    # 2D contours: 1-sigma ellipses
    from matplotlib.patches import Ellipse

    def ellipse_for(cov, color, ls, label):
        # eigendecomp of 2D cov
        eig, vec = np.linalg.eigh(cov)
        # angle of major axis
        ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
        w, h = 2 * np.sqrt(eig)            # 1-sigma full widths
        e = Ellipse((s0, b0), width=w, height=h, angle=ang,
                     facecolor="none", edgecolor=color, lw=2, ls=ls,
                     label=label)
        return e

    e_wp = ellipse_for(cov_wp, "C0", "-", r"$w_p$ alone")
    e_cl = ellipse_for(cov_cl, "C3", "--",
                        r"$C_\ell^{g\kappa}$ alone")
    e_j = ellipse_for(cov_joint, "C2", "-", r"joint")
    ax_contour.add_patch(e_wp); ax_contour.add_patch(e_cl)
    ax_contour.add_patch(e_j)
    ax_contour.plot(s0, b0, "k+", ms=10, mew=2, label="fid (0.81, 2.6)")
    # auto-zoom
    extent = max(2 * sd_wp[0], 2 * sd_cl[0]) * 2
    ax_contour.set_xlim(s0 - extent, s0 + extent)
    ax_contour.set_ylim(b0 - 2 * sd_wp[1] * 2, b0 + 2 * sd_wp[1] * 2)
    ax_contour.set_xlabel(r"$\sigma_8$"); ax_contour.set_ylabel(r"$b$")
    ax_contour.set_title(r"1$\sigma$ Fisher contours: $\sigma_8$-$b$ "
                          r"degeneracy break")
    ax_contour.legend(fontsize=9, loc="upper left")
    ax_contour.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_lensing_forecast.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
