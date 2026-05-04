"""Continuous wp(rp; z) and dwp/dz from kernel-weighted pair counts.

Vecchia-style "no-binning" idea: each pair (i, j) carries its own
z_pair = (z_i + z_j)/2, so we can build a continuous-in-z* estimator

    wp(rp; z*) = LS estimator on DD/DR/RR weighted by Gaussian
                  kernel G(z_pair; z*, sigma_z) instead of indicators
                  on hard z-bins.

Pre-computing the (rp, pi, z_pair) 3D pair-count histogram once and
threading kernel weights through the LS combination makes wp(rp; z*)
JAX-pure and differentiable in z*. ``dwp/dz*`` falls out of
``jax.grad(wp_kernel_z)``.

The same machinery feeds a continuous b(z) MAP fit:
  - at each z* on a fine grid, fit b(z*) at fixed (Om, sigma_8)
    using the wp(rp; z*) estimate, with proper Hessian-based errors;
  - this is an "infinite-bin" version of demo_quaia_bz_evolution.

Three figures::

  quaia_bz_continuous_wp_surface.png  -- wp(rp; z*) curves at multiple z*
                                          + dwp/dz* via jax.grad.
  quaia_bz_continuous.png             -- b(z*) continuous + 1-sigma band.
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
from twopt_density.projected_xi import (
    wp_kernel_z, wp_landy_szalay_zpaired,
)
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


def fit_b_at_z(z_star, sigma_z_kernel, counts, sigma_chi_eff_z, fid,
                sigma8_fixed, pi_max, fft, k_grid):
    """Diagonal-Gaussian MAP fit of bias at z=z_star using the kernel-
    weighted wp(rp; z_star). Returns (b, sd_b, wp_meas, wp_pred_b1)."""
    wp_meas = np.asarray(wp_kernel_z(jnp.float64(z_star), sigma_z_kernel, counts))
    rp_centres = counts.rp_centres
    use = (rp_centres > 10.0) & (rp_centres < 60.0)
    if not use.any():
        return np.nan, np.nan, wp_meas, np.zeros_like(wp_meas)

    # Poisson-style sigma per rp: pi_max / sqrt(eff_DD)
    weights = np.exp(-0.5 * ((counts.z_pair_centres - z_star) / sigma_z_kernel) ** 2)
    weights /= weights.sum()
    DD_w = (counts.DD * weights[None, None, :]).sum(axis=(1, 2))
    sigma_wp = pi_max / np.sqrt(DD_w + 1.0)

    # closed-form b^2 = sum(wp_meas * wp_pred_b1 / sigma^2) / sum(wp_pred_b1^2 / sigma^2)
    rp_use = jnp.asarray(rp_centres[use])
    wp_pred_b1 = np.asarray(wp_observed(
        rp_use, z_eff=z_star, sigma_chi_eff=sigma_chi_eff_z, cosmo=fid,
        bias=1.0, pi_max=pi_max, sigma8=sigma8_fixed,
        fft=fft, k_grid=k_grid,
    ))
    num = float(np.sum(wp_meas[use] * wp_pred_b1 / sigma_wp[use] ** 2))
    den = float(np.sum(wp_pred_b1 ** 2 / sigma_wp[use] ** 2))
    if den <= 0:
        return np.nan, np.nan, wp_meas, np.zeros_like(wp_meas)
    b2 = num / den
    b = float(np.sign(b2) * np.sqrt(abs(b2)))
    sd_b2 = 1.0 / np.sqrt(den) if den > 0 else np.nan
    sd_b = sd_b2 / max(2.0 * abs(b), 1e-3)        # propagate sigma(b^2) -> sigma(b)
    return b, sd_b, wp_meas, wp_pred_b1


def panel_wp_dwp_at_rp(z_star_grid, wp_at_rp, dwp_dz_at_rp, rp_value,
                          sigma_z, out_path):
    """Show wp(z*; rp_fixed) and dwp/dz* as 1D curves -- the direct
    redshift-evolution observable extracted from kernel-weighted pair
    counts, with the derivative computed via ``jax.grad`` of the
    estimator (no finite differences)."""
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    ax_w, ax_d = axs

    ax_w.plot(z_star_grid, wp_at_rp, "C0-o", ms=4,
               label=rf"$w_p(r_p={rp_value:.0f}\,\mathrm{{Mpc/h}};\,z*)$")
    ax_w.set_ylabel(r"$w_p(r_p;\,z*)$")
    ax_w.set_title(rf"Kernel-weighted $w_p(z*)$ + $\partial w_p/\partial z*$"
                    rf" at fixed $r_p = {rp_value:.0f}$ Mpc/h "
                    rf"(kernel $\sigma_z = {sigma_z}$)")
    ax_w.legend(fontsize=9); ax_w.grid(alpha=0.3)
    ax_w.axhline(0, color="k", lw=0.5, alpha=0.3)

    ax_d.plot(z_star_grid, dwp_dz_at_rp, "C1-o", ms=4,
               label=r"$\partial w_p / \partial z*$ via ``jax.grad``")
    # finite-difference cross-check
    fd = np.gradient(wp_at_rp, z_star_grid)
    ax_d.plot(z_star_grid, fd, "k--", lw=1, alpha=0.6,
               label="finite-difference cross-check")
    ax_d.set_xlabel("redshift $z*$")
    ax_d.set_ylabel(r"$\partial w_p / \partial z*$")
    ax_d.legend(fontsize=9); ax_d.grid(alpha=0.3)
    ax_d.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_bz_continuous(z_star_grid, b_arr, sd_arr, A_arr, sd_A_arr,
                         sigma_z_kernel, out_path):
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
    ax_b, ax_a = axs
    ax_b.fill_between(z_star_grid, b_arr - sd_arr, b_arr + sd_arr,
                       color="C0", alpha=0.25,
                       label=r"$1\sigma$ band")
    ax_b.plot(z_star_grid, b_arr, "C0-", lw=1.5,
               label=(r"$b(z*)$ from kernel-$w_p(r_p; z*)$, $\sigma_z=$"
                      f"{sigma_z_kernel}"))
    ax_b.axhline(2.6, ls=":", color="C7", lw=1, alpha=0.8,
                  label="Storey-Fisher+24 b ~ 2.6")
    ax_b.set_ylabel(r"$b(z*)$")
    ax_b.set_title(r"Quaia G$<$20: continuous $b(z)$ from kernel-weighted "
                    "pair counts (no z-bins)")
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


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8_fixed = _env_float("QUAIA_SIGMA8", 0.81)
    n_data_max = _env_int("QUAIA_N_DATA", 80000)
    n_random_max = _env_int("QUAIA_N_RANDOM", 200000)
    pi_max = _env_float("QUAIA_PI_MAX", 200.0)
    sigma_z_kernel = _env_float("QUAIA_SIGMA_Z", 0.25)
    z_min = _env_float("QUAIA_Z_MIN", 0.85)
    z_max = _env_float("QUAIA_Z_MAX", 2.45)

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
    z_d = cat.z_data[md]
    z_r = cat.z_random[mr]
    sig_z = cat.z_data_err[md]
    if len(xyz_d) > n_data_max:
        i = rng.choice(len(xyz_d), n_data_max, replace=False)
        xyz_d, z_d, sig_z = xyz_d[i], z_d[i], sig_z[i]
    if len(xyz_r) > n_random_max:
        i = rng.choice(len(xyz_r), n_random_max, replace=False)
        xyz_r, z_r = xyz_r[i], z_r[i]
    print(f"subsample: N_d={len(xyz_d):,}, N_r={len(xyz_r):,}")

    rp_edges = np.logspace(np.log10(5.0), np.log10(80.0), 12)

    print(f"3D pair counts (rp, pi, z_pair) ...")
    t0 = time.perf_counter()
    counts = wp_landy_szalay_zpaired(
        xyz_d, xyz_r, z_d, z_r, rp_edges, pi_max=pi_max, n_pi=40,
        n_z_pair=60,
    )
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"DD totals: {counts.DD.sum():.0f}, RR: {counts.RR.sum():.0f}")
    print(f"  z_pair grid: {len(counts.z_pair_centres)} bins from "
          f"{counts.z_pair_edges[0]:.2f} to {counts.z_pair_edges[-1]:.2f}")

    # Pre-compute sigma_chi(z*) from the data (kernel-weighted average
    # over per-galaxy sigma_chi). Cleaner than assuming sigma_z(z) is a
    # smooth analytic function.
    sig_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d, sig_z, fid))
    def sigma_chi_at_z(z_star):
        w = np.exp(-0.5 * ((z_d - z_star) / sigma_z_kernel) ** 2)
        return float(np.sqrt(2.0) * np.average(sig_chi_per_obj, weights=w + 1e-30))

    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)

    # keep z* away from kernel-truncation boundaries (jax.grad of the
    # kernel-weighted wp blows up where the kernel hits the histogram
    # edge).
    z_star_grid = np.linspace(z_min + 1.5 * sigma_z_kernel,
                                z_max - 1.5 * sigma_z_kernel, 25)
    b_arr = np.zeros_like(z_star_grid)
    sd_arr = np.zeros_like(z_star_grid)
    A_arr = np.zeros_like(z_star_grid)
    sd_A_arr = np.zeros_like(z_star_grid)
    # rp index closest to a representative scale of 20 Mpc/h
    rp_target = 20.0
    rp_idx = int(np.argmin(np.abs(counts.rp_centres - rp_target)))
    rp_value = float(counts.rp_centres[rp_idx])
    wp_at_rp = np.zeros_like(z_star_grid)
    dwp_dz_at_rp = np.zeros_like(z_star_grid)

    print(f"scan z* on a {len(z_star_grid)}-point grid (sigma_z={sigma_z_kernel}) "
          "with continuous wp + bias fit at each z* ...")
    t0 = time.perf_counter()

    # JAX function we'll grad: wp at fixed rp_idx, function of z*
    def wp_at_rp_jax(z_star):
        return wp_kernel_z(z_star, sigma_z_kernel, counts)[rp_idx]
    dwp_jax = jax.grad(wp_at_rp_jax)

    for j, z_star in enumerate(z_star_grid):
        sig_chi_eff = sigma_chi_at_z(z_star)
        b, sd_b, wp_meas, _ = fit_b_at_z(
            z_star, sigma_z_kernel, counts, sig_chi_eff, fid,
            sigma8_fixed, pi_max, fft, k_grid,
        )
        D = float(linear_growth(jnp.array([z_star]), fid)[0])
        A = b * sigma8_fixed * D
        sd_A = sd_b * sigma8_fixed * D
        b_arr[j], sd_arr[j] = b, sd_b
        A_arr[j], sd_A_arr[j] = A, sd_A
        wp_at_rp[j] = float(wp_meas[rp_idx])
        dwp_dz_at_rp[j] = float(dwp_jax(jnp.float64(z_star)))
    print(f"  {time.perf_counter()-t0:.1f}s for full scan")
    print(f"  reporting wp and dwp/dz at fixed rp = {rp_value:.1f} Mpc/h")

    panel_wp_dwp_at_rp(
        z_star_grid, wp_at_rp, dwp_dz_at_rp, rp_value, sigma_z_kernel,
        os.path.join(FIG_DIR, "quaia_bz_continuous_dwp.png"),
    )
    print("  wrote quaia_bz_continuous_dwp.png")

    panel_bz_continuous(z_star_grid, b_arr, sd_arr, A_arr, sd_A_arr,
                         sigma_z_kernel,
                         os.path.join(FIG_DIR, "quaia_bz_continuous.png"))
    print("  wrote quaia_bz_continuous.png")
    print()
    print("=== continuous b(z*) summary ===")
    print(f"  z*       b           +/-       A=b*sigma_8*D")
    for j in range(0, len(z_star_grid), 3):
        print(f"  {z_star_grid[j]:.2f}    {b_arr[j]:6.3f}    "
              f"{sd_arr[j]:6.3f}    {A_arr[j]:6.3f} +/- {sd_A_arr[j]:5.3f}")


if __name__ == "__main__":
    main()
