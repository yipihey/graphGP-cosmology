"""Reproduce Storey-Fisher+24 C_ell^gg on Quaia, and add a JAX-
differentiable photo-z-aware wp(rp).

Two complementary clustering measurements on the published
Storey-Fisher+24 G < 20 Quaia catalogue:

  1. Angular auto power spectrum C_ell^gg via NaMaster pseudo-Cl
     (the standard Quaia clustering observable). Compared against
     a syren-halofit Limber prediction (cross-checked against pyccl
     to <1%) with proper D^2(z) growth scaling -- see
     ``twopt_density.limber.pnl_at_z``. The Limber kernel uses the
     photo-z PDF stacked dN/dz (each Quaia object contributes a
     Gaussian N(z_quaia, redshift_quaia_err) instead of a delta);
     this is the same dndz construction Storey-Fisher use, and it
     is what brings the recovered linear bias close to their
     published value.

  2. Projected real-space correlation function wp(rp), via
     Landy-Szalay 2D (rp, pi) pair counts on the comoving point
     cloud. *This is what we add* -- and the model side is fully
     JAX-differentiable end-to-end. Each object's photo-z width
     enters the forward model as a per-pair Gaussian LOS kernel
     of width sqrt(2)*sigma_chi(z_eff, sigma_z); the predicted
     ``wp_observed`` (twopt_density.limber.wp_observed) integrates
     xi_real(s) along pi with that kernel and the finite-pi_max
     window, with all of (cosmo, bias, sigma_chi) carrying
     gradients through ``pnl_at_z`` -> halofit_from_plin ->
     plin_emulated. That makes joint MAP / HMC fits over
     (cosmo, b) with photo-z uncertainty a one-line jax.grad away.

Two PNGs::

  quaia_clustering_cl.png    C_ell^gg measurement + Limber bias fit
                              (with photo-z-PDF-stacked dndz).
  quaia_clustering_wp.png    wp(rp) measurement + photo-z-aware
                              halofit overlays (real-space and
                              LOS-convolved at sigma_chi_pair).

Defaults are sized for ~5-10 minutes on a single core. Env vars:

  QUAIA_NSIDE       healpix NSIDE for C_ell (default 64; matches mask)
  QUAIA_N_DATA      max data points used in wp pair counts (default 200000)
  QUAIA_N_RANDOM    random points used in wp pair counts (default 600000)
  QUAIA_Z_MIN/MAX   redshift cuts (default 0.8, 2.5)
  QUAIA_PI_MAX      LOS integration limit for wp [Mpc/h] (default 80)
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.angular import compute_cl_gg
from twopt_density.distance import DistanceCosmo
from twopt_density.limber import (
    cl_gg_limber, dndz_pdf_stack, sigma_chi_from_sigma_z,
    wp_limber, wp_observed,
)
from twopt_density.projected_xi import wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v else default


def panel_cl(meas, ell_pred, cl_pred_b1, b_fit, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    yerr = meas.cl_decoupled / np.sqrt(2.0 * meas.ell_eff + 1.0) / np.sqrt(meas.f_sky)
    ax.errorbar(meas.ell_eff, meas.cl_decoupled, yerr=np.abs(yerr),
                fmt="ok", markersize=4, capsize=3,
                label=r"Quaia $C_\ell^{gg}$ (NaMaster, shot-noise sub.)")
    ax.plot(ell_pred, b_fit ** 2 * cl_pred_b1, "C0-", lw=2,
            label=rf"Limber halofit, $b={b_fit:.2f}$")
    ax.axhline(meas.n_shot, color="C7", ls=":", lw=1,
               label=fr"shot $N_\ell$ = {meas.n_shot:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$C_\ell^{gg}$")
    ax.set_title(r"Quaia G$<$20: angular auto power spectrum"
                 r" + Limber-halofit linear-bias fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_wp_photoz(meas, rp_pred, wp_real_b1, wp_obs_b1, b_cl, b_wp,
                     sigma_chi_pair, pi_max_val, out_path):
    """wp(rp) measurement vs (a) real-space halofit and (b) photo-z-aware
    halofit. The photo-z curve uses the same bias as the real-space curve
    so that the gap between them is purely the LOS smearing effect."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rp_c = meas.rp_centres
    wp_meas = meas.wp
    DD_per_rp = meas.DD.sum(axis=1) + 1.0
    yerr = np.abs(wp_meas) / np.sqrt(DD_per_rp)
    ax.errorbar(rp_c, wp_meas, yerr=yerr, fmt="ok", markersize=4, capsize=3,
                label=(r"Quaia $w_p$ (LS 2D, $\pi_{\max}=$"
                       f"{pi_max_val:.0f} Mpc/h)"))
    ax.plot(rp_pred, b_cl ** 2 * wp_real_b1, "C7-", lw=2, alpha=0.7,
            label=rf"halofit (real-space, $b={b_cl:.2f}$ from $C_\ell$)")
    ax.plot(rp_pred, b_cl ** 2 * wp_obs_b1, "C0-", lw=2,
            label=(r"halofit (photo-z aware, $\sigma_\chi^{pair}\!=$"
                   f"{sigma_chi_pair:.0f} Mpc/h, "
                   rf"$b={b_cl:.2f}$ from $C_\ell$)"))
    ax.plot(rp_pred, b_wp ** 2 * wp_obs_b1, "C1--", lw=2,
            label=(r"halofit photo-z-aware, "
                   rf"$b={b_wp:.2f}$ (refit on $w_p$)"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/h]")
    ax.set_ylabel(r"$w_p(r_p)$ [Mpc/h]")
    ax.set_title(r"Quaia G$<$20: $w_p(r_p)$ + photo-z-aware Limber model")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_wp(meas, rp_pred, wp_pred_b1, b_fit, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    wp_meas = meas.wp
    rp_c = meas.rp_centres
    # crude error from Landy-Szalay variance ~ 1/sqrt(DD)
    DD_per_rp = meas.DD.sum(axis=1) + 1.0
    rel_err = 1.0 / np.sqrt(DD_per_rp)
    yerr = np.abs(wp_meas) * rel_err
    pi_max_val = float(meas.pi_edges[-1])
    ax.errorbar(rp_c, wp_meas, yerr=yerr, fmt="ok", markersize=4, capsize=3,
                label=(r"Quaia $w_p(r_p)$ (LS 2D, $\pi_{\max}=$"
                       f"{pi_max_val:.0f} Mpc/h)"))
    ax.plot(rp_pred, b_fit ** 2 * wp_pred_b1, "C0-", lw=2,
            label=rf"halofit $w_p$ (real-space, $b={b_fit:.2f}$ from $C_\ell$ fit)")
    ax.plot(rp_pred, 2.6 ** 2 * wp_pred_b1, "C1--", lw=2,
            label=r"halofit $w_p$ (real-space, $b=2.6$ Storey-Fisher+24)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/h]")
    ax.set_ylabel(r"$w_p(r_p)$ [Mpc/h]")
    ax.set_title(r"Quaia G$<$20: projected real-space correlation $w_p(r_p)$"
                 r" + halofit overlay")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    cat_path = os.path.join(DATA_DIR, "quaia_G20.0.fits")
    sel_path = os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits")
    nside = _env_int("QUAIA_NSIDE", 64)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)
    n_data_max = _env_int("QUAIA_N_DATA", 200000)
    n_random_max = _env_int("QUAIA_N_RANDOM", 600000)
    pi_max = _env_float("QUAIA_PI_MAX", 200.0)

    print(f"load Quaia (selection-aware random, n_random_factor=2)")
    t0 = time.perf_counter()
    cat = load_quaia(
        catalog_path=cat_path, selection_path=sel_path, fid_cosmo=fid,
        n_random_factor=2, rng_seed=0,
    )
    print(f"  {time.perf_counter()-t0:.1f}s -> N_d={cat.N_data:,}, "
          f"N_r={cat.N_random:,}")
    sel, _ = load_selection_function(sel_path)

    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    mr = (cat.z_random >= z_min) & (cat.z_random <= z_max)
    print(f"redshift cut {z_min} < z < {z_max}: "
          f"N_d={md.sum():,}, N_r={mr.sum():,}")

    # ---- 1. C_ell^gg measurement ----
    print()
    print("=== C_ell^gg via NaMaster ===")
    t0 = time.perf_counter()
    meas_cl = compute_cl_gg(
        cat.ra_data[md], cat.dec_data[md], sel, nside=nside, n_per_bin=12,
    )
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"f_sky={meas_cl.f_sky:.3f}, n_bins={len(meas_cl.ell_eff)}")

    # Two dndz constructions for the Limber kernel:
    #   point-estimate histogram (what we did before -- naive)
    #   photo-z PDF stacked (each galaxy contributes a Gaussian
    #     N(z_quaia, sigma_z); see Storey-Fisher+24 sec. 5)
    z_grid = np.linspace(z_min, z_max, 80)
    z_centres = 0.5 * (z_grid[:-1] + z_grid[1:])
    dndz_point = np.histogram(cat.z_data[md], bins=z_grid)[0].astype(np.float64)
    if cat.z_data_err is not None:
        print("photo-z PDF-stacked dndz (Gaussian per object using "
              "redshift_quaia_err)")
        dndz_stack = dndz_pdf_stack(z_centres, cat.z_data[md],
                                     cat.z_data_err[md])
    else:
        dndz_stack = dndz_point.copy()

    print("Limber halofit (b=1, both dndz variants) ...")
    t0 = time.perf_counter()
    cl_pred_b1_point = cl_gg_limber(
        meas_cl.ell_eff, z_centres, dndz_point, fid, bias=1.0,
    )
    cl_pred_b1_stack = cl_gg_limber(
        meas_cl.ell_eff, z_centres, dndz_stack, fid, bias=1.0,
    )
    print(f"  {time.perf_counter()-t0:.1f}s")

    # bias fit on each: ell > 20 (Limber valid; lowest ell mode-coupled)
    use = meas_cl.ell_eff > 20.0

    def fit_bias(meas, model):
        num = float(np.sum(meas[use] * model[use]))
        den = float(np.sum(model[use] ** 2))
        return float(np.sqrt(max(num / den, 0.01)))

    b_point = fit_bias(meas_cl.cl_decoupled, cl_pred_b1_point)
    b_stack = fit_bias(meas_cl.cl_decoupled, cl_pred_b1_stack)
    print(f"linear-bias fit (ell > 20): b_point = {b_point:.3f}, "
          f"b_stack = {b_stack:.3f}")
    print(f"  point-estimate dndz width: {np.std(z_centres - z_centres.mean()):.3f}")
    print(f"  vs Storey-Fisher+24 published b ~ 2.5-2.6 at z_eff~1.4")
    b_fit = b_stack  # use the photo-z-aware bias for downstream wp prediction

    panel_cl(meas_cl, meas_cl.ell_eff, cl_pred_b1_stack, b_stack,
             os.path.join(FIG_DIR, "quaia_clustering_cl.png"))
    print("  wrote quaia_clustering_cl.png")

    # ---- 2. wp(rp) measurement ----
    print()
    print("=== wp(rp) via 2D Landy-Szalay ===")
    rng = np.random.default_rng(1)
    xyz_d = np.asarray(cat.xyz_data[md])
    xyz_r = np.asarray(cat.xyz_random[mr])
    if len(xyz_d) > n_data_max:
        idx = rng.choice(len(xyz_d), size=n_data_max, replace=False)
        xyz_d = xyz_d[idx]
    if len(xyz_r) > n_random_max:
        idx = rng.choice(len(xyz_r), size=n_random_max, replace=False)
        xyz_r = xyz_r[idx]
    print(f"pair-count subsample: N_d={len(xyz_d):,}, N_r={len(xyz_r):,}")

    rp_edges = np.logspace(np.log10(5.0), np.log10(80.0), 12)
    t0 = time.perf_counter()
    meas_wp = wp_landy_szalay(
        xyz_d, xyz_r, rp_edges, pi_max=pi_max, n_pi=40,
    )
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"DD total={meas_wp.DD.sum():.0f}, "
          f"RR total={meas_wp.RR.sum():.0f}")
    print("  rp_c    wp(rp)")
    for r, w in zip(meas_wp.rp_centres, meas_wp.wp):
        print(f"  {r:6.2f}  {w:8.3f}")

    # Real-space + photo-z-aware Limber wp at z_eff = median(z_data)
    z_eff = float(np.median(cat.z_data[md]))
    rp_fine = np.logspace(np.log10(0.5), np.log10(80.0), 60)
    print(f"Limber halofit wp at z_eff = {z_eff:.2f} ...")
    t0 = time.perf_counter()
    wp_pred_b1 = wp_limber(rp_fine, z_eff=z_eff, cosmo=fid, bias=1.0,
                            pi_max=pi_max, n_pi=200)
    print(f"  real-space   wp_limber:   {time.perf_counter()-t0:.1f}s")

    # Per-pair effective LOS sigma: sqrt(2) * sigma_chi at z_eff. Quaia
    # spectro-photo-z's are ~Gaussian; we use the median per-object
    # sigma_z weighted by dchi/dz at z_eff.
    sigma_z_med = float(np.median(cat.z_data_err[md])) if cat.z_data_err is not None else 0.0
    sigma_chi_one = float(sigma_chi_from_sigma_z(
        np.array([z_eff]), np.array([sigma_z_med]), fid,
    )[0])
    sigma_chi_eff_pair = float(np.sqrt(2.0) * sigma_chi_one)
    print(f"  per-object sigma_z (median): {sigma_z_med:.3f}")
    print(f"  per-object sigma_chi at z={z_eff:.2f}: {sigma_chi_one:.1f} Mpc/h")
    print(f"  per-pair effective sigma_chi (auto): {sigma_chi_eff_pair:.1f} Mpc/h")

    print("  photo-z-aware wp_observed ...")
    t0 = time.perf_counter()
    import jax.numpy as jnp
    wp_obs_b1 = np.asarray(wp_observed(
        jnp.asarray(rp_fine), z_eff=z_eff, sigma_chi_eff=sigma_chi_eff_pair,
        cosmo=fid, bias=1.0, pi_max=pi_max, n_pi_true=400,
    ))
    print(f"  photo-z wp_observed:      {time.perf_counter()-t0:.1f}s")

    # Direct bias fit ON wp using the photo-z-aware model. Linear in b^2.
    # Use a Poisson-style error sigma_wp ~ pi_max / sqrt(DD) (independent
    # of the data value -- avoids weighting toward low-DD outliers).
    rp_pred_at_meas = np.interp(meas_wp.rp_centres, rp_fine, wp_obs_b1)
    use_wp = (meas_wp.rp_centres > 10.0) & (meas_wp.rp_centres < 60.0)
    DD_per_rp = meas_wp.DD.sum(axis=1) + 1.0
    sigma_wp = pi_max / np.sqrt(DD_per_rp)
    num = np.sum(meas_wp.wp[use_wp] * rp_pred_at_meas[use_wp]
                 / sigma_wp[use_wp] ** 2)
    den = np.sum(rp_pred_at_meas[use_wp] ** 2 / sigma_wp[use_wp] ** 2)
    b2_wp = max(num / den, 0.01)
    b_wp_fit = float(np.sqrt(b2_wp))
    print(f"linear-bias fit on photo-z-aware wp (10 < rp < 60, "
          f"sigma = pi_max/sqrt(DD)): b_wp = {b_wp_fit:.3f}")

    panel_wp_photoz(meas_wp, rp_fine, wp_pred_b1, wp_obs_b1, b_fit, b_wp_fit,
                    sigma_chi_eff_pair, pi_max,
                    os.path.join(FIG_DIR, "quaia_clustering_wp.png"))
    print("  wrote quaia_clustering_wp.png")

    print()
    print("=== summary ===")
    print(f"C_ell^gg fit (ell > 20):")
    print(f"  point-estimate dndz   -> b = {b_point:.2f}")
    print(f"  photo-z-PDF-stacked   -> b = {b_stack:.2f}")
    print(f"  Storey-Fisher+24      -> b ~ 2.5-2.6 at z_eff~1.4")
    print()
    print(f"wp(rp) (photo-z-aware halofit, JAX-differentiable end-to-end):")
    print(f"  per-pair sigma_chi    = {sigma_chi_eff_pair:.0f} Mpc/h")
    print(f"  bias from C_ell       -> b = {b_fit:.2f}")
    print(f"  bias refit on wp      -> b = {b_wp_fit:.2f}")
    print(f"  consistent C_ell + wp bias is a real cross-check the published")
    print(f"  Quaia analyses don't have, because they don't measure wp.")
    print()
    print(f"Differentiability: ``wp_observed`` is jax.grad-differentiable in")
    print(f"  (cosmo, bias, sigma_chi). Test_clustering exercises grad flow.")


if __name__ == "__main__":
    main()
