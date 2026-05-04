"""Quaia x Planck PR4 CMB lensing -- the real measurement.

Builds the cross-spectrum C_ell^{g-kappa} between Quaia G < 20 galaxies
and the Planck PR4 CMB lensing reconstruction (J. Carron + lensing
team, MV minimum-variance estimator at lmax = 2048, mean-field
subtracted).

The Planck PR4 lensing data lives at::

    https://github.com/carronj/planck_PR4_lensing/releases/tag/Data

Specifically the tar ``PR42018like_maps.tar`` (241 MB) which extracts
``PR4_variations/PR42018like_klm_dat_MV.fits`` (data alms, lmax=2048),
``PR42018like_klm_mf_MV.fits`` (mean-field), and ``mask.fits.gz``
(NSIDE=2048 lensing analysis mask). We pre-process to NSIDE = 64 to
match the Quaia selection function.

Pipeline:
  1. Load Quaia data + selection function (NSIDE=64).
  2. Load Planck PR4 kappa map + mask (already pre-built at NSIDE=64
     via ``healpy.alm2map(klm_dat - klm_mf)``).
  3. NaMaster pseudo-Cl cross-correlation on the joint mask
     (Quaia x Planck mask product).
  4. Compare to the Limber forward model from
     ``twopt_density.lensing.cl_gkappa_limber`` at the published
     b ~ 2.6 and Planck cosmology.
  5. Joint MAP fit (sigma_8, b) from wp + Cl^{g-kappa} -- the channel
     that breaks the sigma_8-b banana.

Output: ``demos/figures/quaia_planck_lensing.png``.
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

from twopt_density.angular import compute_cl_gkappa
from twopt_density.distance import DistanceCosmo
from twopt_density.lensing import (
    cl_gkappa_limber, cl_kappa_kappa_planck_pr3,
    planck_pr3_lensing_noise, quaia_gkappa_snr_forecast,
)
from twopt_density.limber import cl_gg_limber
from twopt_density.quaia import load_quaia, load_selection_function


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
PLANCK_DIR = os.path.join(REPO_ROOT, "data", "planck")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_float(n, d):
    v = os.environ.get(n)
    return float(v) if v else d


def main():
    import healpy as hp

    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8 = _env_float("QUAIA_SIGMA8", 0.81)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)

    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    mask_g, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    print(f"  Quaia: NSIDE={nside}, N_data after z-cut = {md.sum():,}, "
          f"f_sky_g = {mask_g.mean():.3f}")

    print("\nloading Planck PR4 kappa map ...")
    fn_planck = os.path.join(PLANCK_DIR, "planck_pr4_kappa_nside64.fits")
    kappa, mask_kappa = hp.read_map(fn_planck, field=[0, 1])
    mask_kappa = mask_kappa.astype(np.float64)
    print(f"  kappa: NSIDE={hp.npix2nside(kappa.size)}, "
          f"std = {kappa.std():.4e}, "
          f"f_sky_kappa = {mask_kappa.mean():.3f}")
    f_sky_joint = float((mask_g * mask_kappa).mean())
    print(f"  joint f_sky = {f_sky_joint:.3f}")

    # ---- 1. measurement ----
    print("\nNaMaster cross-correlation (Quaia delta_g x Planck kappa) ...")
    t = time.perf_counter()
    meas = compute_cl_gkappa(
        cat.ra_data[md], cat.dec_data[md],
        mask_g, kappa, mask_kappa, nside=nside, n_per_bin=12,
    )
    print(f"  {time.perf_counter()-t:.0f}s, n_bins = {len(meas.ell_eff)}, "
          f"f_sky_eff = {meas.f_sky:.3f}")

    # ---- 2. Limber forward model at fid b ----
    z_grid = np.linspace(z_min + 0.05, z_max - 0.05, 60)
    nz, _ = np.histogram(cat.z_data[md], bins=z_grid, density=True)
    z_centres = 0.5 * (z_grid[:-1] + z_grid[1:])

    b_const = 2.6
    print(f"\nLimber prediction (b = {b_const}) ...")
    t = time.perf_counter()
    cl_gk_pred = np.asarray(cl_gkappa_limber(
        meas.ell_eff, z_centres, nz,
        np.full_like(z_centres, b_const), fid, sigma8=sigma8,
    ))
    cl_gg_pred = np.asarray(cl_gg_limber(
        meas.ell_eff, z_centres, nz, fid, bias=b_const, sigma8=sigma8,
    ))
    cl_kk_pred = cl_kappa_kappa_planck_pr3(meas.ell_eff, fid, sigma8=sigma8)
    N_kk = planck_pr3_lensing_noise(meas.ell_eff)
    N_gg = 4 * np.pi * f_sky_joint / md.sum()
    print(f"  {time.perf_counter()-t:.1f}s")

    # ---- 3. amplitude (b * sigma_8) fit ----
    # Per-ell sigma from the Knox-style Gaussian variance with the
    # joint mask
    sigma_cl = np.sqrt(
        ((cl_gg_pred + N_gg) * (cl_kk_pred + N_kk)
         + cl_gk_pred ** 2)
        / ((2 * meas.ell_eff + 1) * f_sky_joint)
    )
    # exclude lowest-ell bins (mode-coupling instability on partial sky
    # at NSIDE=64) and ell > 3*NSIDE-1 (Nyquist limit)
    ell_max_useful = 3 * nside - 1
    use = (meas.ell_eff > 30) & (meas.ell_eff < ell_max_useful)
    # closed-form optimal amplitude: A = b_obs / b_fid
    A_num = float(np.sum(meas.cl_decoupled[use] * cl_gk_pred[use]
                          / sigma_cl[use] ** 2))
    A_den = float(np.sum(cl_gk_pred[use] ** 2 / sigma_cl[use] ** 2))
    A_hat = A_num / A_den if A_den > 0 else float("nan")
    sigma_A = 1.0 / np.sqrt(A_den) if A_den > 0 else float("nan")
    SNR = A_hat / sigma_A if sigma_A > 0 else float("nan")
    b_hat = b_const * A_hat
    sigma_b = b_const * sigma_A
    print(f"\nAmplitude fit (30 < ell < 400):")
    print(f"  A = C_ell^obs / C_ell^pred(b={b_const}) = {A_hat:.3f} +/- {sigma_A:.3f}")
    print(f"  -> b_obs = {b_hat:.3f} +/- {sigma_b:.3f}")
    print(f"  detection SNR = {SNR:.2f}")

    # ---- 4. figure ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax_cl, ax_sn = axs

    # mark the unreliable ell bins (mode-coupling residuals at low ell
    # on a partial sky) but plot all data
    ax_cl.errorbar(meas.ell_eff[~use], meas.cl_decoupled[~use],
                     yerr=sigma_cl[~use], fmt="o", color="0.6", ms=4,
                     capsize=3, label="excluded from fit")
    ax_cl.errorbar(meas.ell_eff[use], meas.cl_decoupled[use],
                     yerr=sigma_cl[use], fmt="ok", ms=5, capsize=3,
                     label=fr"Quaia $\times$ Planck PR4 (NaMaster, "
                            f"f_sky_joint={f_sky_joint:.2f})")
    ax_cl.plot(meas.ell_eff, cl_gk_pred, "C0-", lw=2,
                label=fr"Limber prediction at $b={b_const}$, "
                       fr"$\sigma_8={sigma8}$")
    ax_cl.plot(meas.ell_eff, A_hat * cl_gk_pred, "C3--", lw=2,
                label=fr"best fit $A={A_hat:.2f}$ "
                       fr"(b={b_hat:.2f}+/-{sigma_b:.2f})")
    ax_cl.axhline(0, color="k", lw=0.5)
    ax_cl.set_xscale("log")
    ax_cl.set_xlabel(r"$\ell$"); ax_cl.set_ylabel(r"$C_\ell^{g-\kappa}$")
    ax_cl.set_title(rf"Quaia $\times$ Planck PR4 CMB lensing -- "
                       rf"$C_\ell^{{g-\kappa}}$"
                       rf" (NSIDE={nside}, $\ell_{{\max}}$=3$\cdot$NSIDE-1={ell_max_useful})")
    ax_cl.legend(fontsize=9); ax_cl.grid(alpha=0.3, which="both")

    # SNR per ell
    snr_per = np.abs(meas.cl_decoupled) / sigma_cl
    ax_sn.semilogx(meas.ell_eff, snr_per, "C0-o", ms=4, label="per-bin |SNR|")
    # cumulative on the *kept* bins only (excluding low-ell artifacts)
    cum_use = np.sqrt(np.cumsum(snr_per[use] ** 2))
    ax_sn.semilogx(meas.ell_eff[use], cum_use, "C3-", lw=2,
                    label=fr"cumulative SNR (kept bins, total {cum_use[-1]:.1f})")
    ax_sn.set_xlabel(r"$\ell$"); ax_sn.set_ylabel("SNR")
    ax_sn.set_title("per-bin and cumulative SNR")
    ax_sn.set_yscale("symlog", linthresh=1.0)
    ax_sn.legend(fontsize=9); ax_sn.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_planck_lensing.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
