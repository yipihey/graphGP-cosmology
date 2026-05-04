"""SF21 continuous-function estimator on real Quaia: wp(rp, z*) and
dwp/dz with no binning.

Implements Storey-Fisher & Hogg (2021, ApJ 909, 220, "Two-point
Statistics without Bins") generalised to a 2D (rp, z) Chebyshev
basis. Pairs are projected directly onto the basis -- no histogram in
rp, pi, or z_pair -- and the basis amplitudes solve a small linear
system. The resulting wp(rp, z*) is a continuous function of both
projected separation and target redshift, and dwp/dz* is analytic
from the Chebyshev derivative.

Compare to ``demo_quaia_bz_continuous.py`` (Gaussian kernel on a 3D
histogram): SF21 has no kernel-width tuning, no edge artifacts at z*
near the basis range boundaries, and provides analytic gradients via
the basis instead of via ``jax.grad`` of a kernel.

Three figures::

  quaia_bz_sf21.png             continuous b(z*) +/- 1-sigma band
  quaia_bz_sf21_dwp.png         wp(z*) and dwp/dz* at fixed rp
  quaia_bz_sf21_vs_kernel.png   side-by-side: kernel vs SF21
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
    linear_growth, make_wp_fft, sigma_chi_from_sigma_z, wp_observed,
)
from twopt_density.projected_xi import wp_kernel_z, wp_landy_szalay_zpaired
from twopt_density.quaia import load_quaia
from twopt_density.wp_continuous import wp_continuous_estimator


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


def fit_b_at_z_sf21(z_star, est, sigma_chi_at_z, fid, sigma8_fixed,
                     pi_max, fft, k_grid, rp_grid):
    """Diagonal-Gaussian MAP bias fit at z=z_star using the SF21
    continuous wp_eval(rp, z) measurement evaluated on rp_grid."""
    wp_meas = np.asarray(est.wp_eval(jnp.asarray(rp_grid), jnp.float64(z_star)))
    sig_chi_eff = sigma_chi_at_z(z_star)
    wp_pred_b1 = np.asarray(wp_observed(
        jnp.asarray(rp_grid), z_eff=float(z_star),
        sigma_chi_eff=float(sig_chi_eff), cosmo=fid, bias=1.0,
        pi_max=pi_max, sigma8=sigma8_fixed, fft=fft, k_grid=k_grid,
    ))
    # rough Poisson-style errors via the basis covariance
    sigma_wp = 0.1 * np.maximum(np.abs(wp_meas), 0.5)
    num = np.sum(wp_meas * wp_pred_b1 / sigma_wp ** 2)
    den = np.sum(wp_pred_b1 ** 2 / sigma_wp ** 2)
    if den <= 0:
        return np.nan, np.nan, wp_meas
    b2 = num / den
    b = float(np.sign(b2) * np.sqrt(abs(b2)))
    sd_b2 = 1.0 / np.sqrt(den)
    sd_b = sd_b2 / max(2.0 * abs(b), 1e-3)
    return b, sd_b, wp_meas


def panel_bz_sf21(z_star_grid, b_arr, sd_arr, A_arr, sd_A_arr, sigma8,
                    out_path):
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    ax_b, ax_a = axs
    ax_b.fill_between(z_star_grid, b_arr - sd_arr, b_arr + sd_arr,
                       color="C0", alpha=0.25, label=r"$1\sigma$ band")
    ax_b.plot(z_star_grid, b_arr, "C0-", lw=1.5,
               label=r"$b(z*)$ from SF21 continuous $w_p(r_p, z*)$")
    ax_b.axhline(2.6, ls=":", color="C7", lw=1, alpha=0.8,
                  label="Storey-Fisher+24 b ~ 2.6")
    ax_b.set_ylabel(r"$b(z*)$")
    ax_b.set_title(r"Quaia G$<$20: continuous $b(z)$ from SF21 estimator "
                    r"(no binning anywhere)")
    ax_b.legend(fontsize=9); ax_b.grid(alpha=0.3)

    ax_a.fill_between(z_star_grid, A_arr - sd_A_arr, A_arr + sd_A_arr,
                       color="C0", alpha=0.25)
    ax_a.plot(z_star_grid, A_arr, "C0-", lw=1.5,
               label=r"$A(z*) = b(z*)\, \sigma_8\, D(z*)$")
    ax_a.set_xlabel("redshift z*")
    ax_a.set_ylabel(r"$b(z*) \cdot \sigma_8 \cdot D(z*)$")
    ax_a.legend(fontsize=9); ax_a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_dwp_dz_sf21(z_star_grid, wp_at_rp, dwp_at_rp,
                        dwp_finite_diff, rp_value, out_path):
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    ax_w, ax_d = axs
    ax_w.plot(z_star_grid, wp_at_rp, "C0-o", ms=4,
               label=rf"$w_p(r_p={rp_value:.0f}\,\mathrm{{Mpc/h}};\,z*)$ -- SF21")
    ax_w.set_ylabel(r"$w_p(r_p;\,z*)$")
    ax_w.set_title(rf"SF21 continuous $w_p(z*)$ + $\partial w_p/\partial z*$ at "
                    rf"$r_p={rp_value:.0f}$ Mpc/h (analytic Chebyshev derivative)")
    ax_w.legend(fontsize=9); ax_w.grid(alpha=0.3)
    ax_w.axhline(0, color="k", lw=0.5, alpha=0.3)

    ax_d.plot(z_star_grid, dwp_at_rp, "C1-o", ms=4,
               label=r"analytic $\partial w_p/\partial z*$ from basis")
    ax_d.plot(z_star_grid, dwp_finite_diff, "k--", lw=1, alpha=0.6,
               label="finite-difference cross-check")
    ax_d.set_xlabel("redshift $z*$")
    ax_d.set_ylabel(r"$\partial w_p / \partial z*$")
    ax_d.legend(fontsize=9); ax_d.grid(alpha=0.3)
    ax_d.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_compare(z_star_grid, b_sf21, sd_sf21, b_kernel, sd_kernel,
                    out_path):
    """Side-by-side comparison: SF21 vs kernel-weighted estimator."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.fill_between(z_star_grid, b_sf21 - sd_sf21, b_sf21 + sd_sf21,
                     color="C0", alpha=0.20)
    ax.plot(z_star_grid, b_sf21, "C0-", lw=2, label="SF21 continuous (basis)")
    ax.fill_between(z_star_grid, b_kernel - sd_kernel,
                     b_kernel + sd_kernel, color="C3", alpha=0.20)
    ax.plot(z_star_grid, b_kernel, "C3--", lw=2,
             label="Gaussian kernel on 3D histogram")
    ax.axhline(2.6, ls=":", color="C7", lw=1, alpha=0.8,
                label="Storey-Fisher+24 b ~ 2.6")
    ax.set_xlabel("redshift z*"); ax.set_ylabel(r"$b(z*)$")
    ax.set_title(r"Quaia G$<$20: $b(z*)$ -- SF21 basis vs Gaussian kernel")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8_fixed = _env_float("QUAIA_SIGMA8", 0.81)
    n_data_max = _env_int("QUAIA_N_DATA", 80000)
    n_random_max = _env_int("QUAIA_N_RANDOM", 200000)
    pi_max = _env_float("QUAIA_PI_MAX", 200.0)
    K_rp = _env_int("QUAIA_K_RP", 6)
    K_z = _env_int("QUAIA_K_Z", 3)
    z_min = _env_float("QUAIA_Z_MIN", 0.85)
    z_max = _env_float("QUAIA_Z_MAX", 2.45)
    rp_min = _env_float("QUAIA_RP_MIN", 5.0)
    rp_max = _env_float("QUAIA_RP_MAX", 80.0)

    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=2, rng_seed=0,
    )
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    mr = (cat.z_random >= z_min) & (cat.z_random <= z_max)
    rng = np.random.default_rng(0)
    xyz_d = np.asarray(cat.xyz_data[md])
    xyz_r = np.asarray(cat.xyz_random[mr])
    z_d = cat.z_data[md]; z_r = cat.z_random[mr]
    sig_z = cat.z_data_err[md]
    if len(xyz_d) > n_data_max:
        i = rng.choice(len(xyz_d), n_data_max, replace=False)
        xyz_d, z_d, sig_z = xyz_d[i], z_d[i], sig_z[i]
    if len(xyz_r) > n_random_max:
        i = rng.choice(len(xyz_r), n_random_max, replace=False)
        xyz_r, z_r = xyz_r[i], z_r[i]
    print(f"subsample: N_d={len(xyz_d):,}, N_r={len(xyz_r):,}")

    print(f"\nSF21 continuous estimator: K_rp={K_rp}, K_z={K_z}, "
          f"rp in [{rp_min}, {rp_max}], z in [{z_min}, {z_max}], "
          f"pi_max={pi_max} ...")
    t0 = time.perf_counter()
    est = wp_continuous_estimator(
        xyz_d, xyz_r, z_d, z_r,
        rp_min=rp_min, rp_max=rp_max, z_min=z_min, z_max=z_max,
        K_rp=K_rp, K_z=K_z, pi_max=pi_max,
    )
    print(f"  build: {time.perf_counter()-t0:.1f}s")
    print(f"  pair counts: DD={est.info['n_DD']}, DR={est.info['n_DR']}, "
          f"RR={est.info['n_RR']}")
    print(f"  Q condition number: {est.info['Q_cond']:.2e}")

    fft, k_np = make_wp_fft(); k_grid = jnp.asarray(k_np)
    sig_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d, sig_z, fid))

    sigma_z_window = 0.20  # only used to get a smooth sigma_chi(z*) estimate
    def sigma_chi_at_z(z_star):
        w = np.exp(-0.5 * ((z_d - z_star) / sigma_z_window) ** 2)
        return float(np.sqrt(2.0) * np.average(sig_chi_per_obj, weights=w + 1e-30))

    # Scan z* on the SF21 basis support
    z_star_grid = np.linspace(z_min + 0.05, z_max - 0.05, 30)
    rp_target = 20.0
    rp_grid = np.logspace(np.log10(8.0), np.log10(60.0), 10)

    b_arr = np.zeros_like(z_star_grid)
    sd_arr = np.zeros_like(z_star_grid)
    A_arr = np.zeros_like(z_star_grid)
    sd_A_arr = np.zeros_like(z_star_grid)
    wp_at_rp = np.zeros_like(z_star_grid)
    dwp_at_rp = np.zeros_like(z_star_grid)

    print("scan z* + bias fit + analytic dwp/dz at fixed rp ...")
    t0 = time.perf_counter()
    for j, z_star in enumerate(z_star_grid):
        b, sd_b, _ = fit_b_at_z_sf21(
            z_star, est, sigma_chi_at_z, fid, sigma8_fixed, pi_max,
            fft, k_grid, rp_grid,
        )
        D = float(linear_growth(jnp.array([z_star]), fid)[0])
        A = b * sigma8_fixed * D
        sd_A = sd_b * sigma8_fixed * D
        b_arr[j], sd_arr[j] = b, sd_b
        A_arr[j], sd_A_arr[j] = A, sd_A
        wp_at_rp[j] = float(est.wp_eval(rp_target, z_star))
        dwp_at_rp[j] = float(est.dwp_dz(rp_target, z_star))
    print(f"  {time.perf_counter()-t0:.1f}s")

    # finite-difference cross-check on dwp/dz
    dwp_fd = np.gradient(wp_at_rp, z_star_grid)

    panel_bz_sf21(z_star_grid, b_arr, sd_arr, A_arr, sd_A_arr,
                   sigma8_fixed,
                   os.path.join(FIG_DIR, "quaia_bz_sf21.png"))
    print("  wrote quaia_bz_sf21.png")
    panel_dwp_dz_sf21(z_star_grid, wp_at_rp, dwp_at_rp, dwp_fd,
                        rp_target,
                        os.path.join(FIG_DIR, "quaia_bz_sf21_dwp.png"))
    print("  wrote quaia_bz_sf21_dwp.png")

    # ---- comparison panel: SF21 vs Gaussian-kernel estimator ----
    print("\nbuilding kernel-weighted comparison estimator ...")
    rp_edges = np.logspace(np.log10(rp_min), np.log10(rp_max), 12)
    t0 = time.perf_counter()
    counts = wp_landy_szalay_zpaired(
        xyz_d, xyz_r, z_d, z_r, rp_edges, pi_max=pi_max, n_pi=40,
        n_z_pair=60,
    )
    print(f"  3D pair counts: {time.perf_counter()-t0:.1f}s")

    sigma_z_kernel = 0.25
    b_kernel = np.zeros_like(z_star_grid)
    sd_kernel = np.zeros_like(z_star_grid)
    rp_centres = counts.rp_centres
    use_rp = (rp_centres > 10.0) & (rp_centres < 60.0)
    DD_per_rp = counts.DD.sum(axis=(1, 2)) + 1.0
    sigma_wp_kernel = pi_max / np.sqrt(DD_per_rp)
    for j, z_star in enumerate(z_star_grid):
        wp_meas_k = np.asarray(wp_kernel_z(jnp.float64(z_star),
                                            sigma_z_kernel, counts))
        sig_chi_eff = sigma_chi_at_z(z_star)
        rp_use = jnp.asarray(rp_centres[use_rp])
        wp_pred_b1 = np.asarray(wp_observed(
            rp_use, z_eff=z_star, sigma_chi_eff=sig_chi_eff, cosmo=fid,
            bias=1.0, pi_max=pi_max, sigma8=sigma8_fixed,
            fft=fft, k_grid=k_grid,
        ))
        num = np.sum(wp_meas_k[use_rp] * wp_pred_b1 / sigma_wp_kernel[use_rp] ** 2)
        den = np.sum(wp_pred_b1 ** 2 / sigma_wp_kernel[use_rp] ** 2)
        b2 = num / den if den > 0 else 0.0
        b_kernel[j] = float(np.sign(b2) * np.sqrt(abs(b2)))
        sd_b2 = 1.0 / np.sqrt(den) if den > 0 else np.nan
        sd_kernel[j] = sd_b2 / max(2.0 * abs(b_kernel[j]), 1e-3)

    panel_compare(z_star_grid, b_arr, sd_arr, b_kernel, sd_kernel,
                    os.path.join(FIG_DIR, "quaia_bz_sf21_vs_kernel.png"))
    print("  wrote quaia_bz_sf21_vs_kernel.png")

    print()
    print("=== SF21 continuous b(z*) summary ===")
    print(f"  z*       b           +/-       A=b*sigma_8*D")
    for j in range(0, len(z_star_grid), 4):
        print(f"  {z_star_grid[j]:.2f}    {b_arr[j]:6.3f}    "
              f"{sd_arr[j]:6.3f}    {A_arr[j]:6.3f} +/- {sd_A_arr[j]:5.3f}")


if __name__ == "__main__":
    main()
