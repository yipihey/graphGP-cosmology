"""Angular two-point function w(θ) for 10 GP-posterior catalog samples.

Measures the 3D Landy-Szalay xi(r) via the morton_cascade Rust binary for
each GP sample and the original BOSS CMASS catalog, then Limber-projects
xi(r) → w(θ) and overplots all curves on one figure.

Usage::

    python demos/plot_angular_wtheta.py [--n-samples 10] [--n-rand-sub 300000]
                                        [--out output/boss_wtheta_samples.png]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

from twopt_density.boss import load_boss
from twopt_density.density_field import sample_posterior_density_field
from twopt_density.distance import comoving_distance, radec_z_to_cartesian
from twopt_density.cascade import xi_landy_szalay


# ── Limber projection xi(r) → w(θ) ───────────────────────────────────────────

def limber_wtheta(r_cen, xi_vals, theta_deg_arr, chi_bar,
                  pi_max=300.0, n_pi=400):
    """Project xi(r) → w(θ) via the flat-sky Limber integral.

    w(θ) = 2 ∫₀^{pi_max} ξ(√(π² + (χ̄ sinθ)²)) dπ

    Parameters
    ----------
    r_cen, xi_vals : 1-D arrays of (r [Mpc/h], xi(r)) from the cascade.
    theta_deg_arr  : array of θ values in degrees.
    chi_bar        : effective mean comoving distance [Mpc/h].
    pi_max         : LOS integration limit [Mpc/h].
    n_pi           : number of π quadrature points.
    """
    # Sort ascending in r (cascade outputs large→small), then log-space interpolate
    order = np.argsort(r_cen)
    log_r_sorted = np.log(np.clip(r_cen[order], 1e-3, None))
    xi_sorted = xi_vals[order]

    def xi_interp(r):
        lr = np.log(np.clip(r, 1e-3, None))
        return np.interp(lr, log_r_sorted, xi_sorted, left=0.0, right=0.0)

    pi_arr = np.linspace(0.0, pi_max, n_pi)
    w_arr = np.empty(len(theta_deg_arr))
    for i, theta_deg in enumerate(theta_deg_arr):
        chi_perp = chi_bar * np.sin(np.radians(theta_deg))
        r_arr = np.sqrt(pi_arr**2 + chi_perp**2)
        xi_arr = xi_interp(r_arr)
        w_arr[i] = 2.0 * np.trapz(xi_arr, pi_arr)
    return w_arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms",    default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-samples",  type=int,   default=10)
    p.add_argument("--n-rand-sub", type=int,   default=300_000,
                   help="Randoms subsample for xi (default 300k)")
    p.add_argument("--theta-min",  type=float, default=0.05)
    p.add_argument("--theta-max",  type=float, default=8.0)
    p.add_argument("--n-theta",    type=int,   default=20)
    p.add_argument("--pi-max",     type=float, default=300.0,
                   help="LOS integration limit for Limber projection [Mpc/h]")
    p.add_argument("--out",        default="output/boss_wtheta_samples.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    theta_edges = np.logspace(np.log10(args.theta_min),
                              np.log10(args.theta_max), args.n_theta + 1)
    theta_cen = np.sqrt(theta_edges[:-1] * theta_edges[1:])

    # ── 1. Load catalog ───────────────────────────────────────────────────
    print("Loading BOSS CMASS-SGC ...")
    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    print(f"  N_data={cat.N_data:,}  N_random={len(cat.ra_random):,}")

    # Effective comoving distance for Limber projection
    chi_vals = np.array(comoving_distance(jnp.asarray(cat.z_data), cat.fid_cosmo))
    chi_bar  = float(np.mean(chi_vals))
    print(f"  χ̄ = {chi_bar:.0f} Mpc/h  (z_eff = {float(np.mean(cat.z_data)):.3f})")

    # ── 2. Posterior density field ────────────────────────────────────────
    print("Running posterior sampler ...")
    t0 = time.time()
    result = sample_posterior_density_field(
        cat, n_samples=args.n_samples, n_z_bins=32, nside=64, verbose=False,
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── 3. GP catalog samples ─────────────────────────────────────────────
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"Generating {args.n_samples} GP samples ...")
    t0 = time.time()
    catalogs = result.sample_catalogs_gp(cat, seed=42, w_completeness=w_comp)
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── 4. Shared random subsample ────────────────────────────────────────
    rng = np.random.default_rng(999)
    N_r_full = len(cat.ra_random)
    n_sub    = min(args.n_rand_sub, N_r_full)
    idx_sub  = rng.choice(N_r_full, size=n_sub, replace=False)

    xyz_r_sub = cat.xyz_random[idx_sub]

    # Shift everything into non-negative coords for the cascade
    all_xyz = np.vstack([cat.xyz_data, xyz_r_sub])
    shift   = -all_xyz.min(axis=0) + 100.0
    box_size = float(np.max(all_xyz + shift)) + 200.0
    print(f"  box_size = {box_size:.0f} Mpc/h")

    xyz_d_s = cat.xyz_data + shift
    xyz_r_s = xyz_r_sub   + shift

    # ── 5. xi(r) for original catalog ────────────────────────────────────
    print("cascade xi(r) for original catalog ...")
    t0 = time.time()
    xi_orig_arr = xi_landy_szalay(xyz_d_s, xyz_r_s, box_size=box_size,
                                   dim=3, periodic=False)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"({len(xi_orig_arr)} shells)")

    # ── 6. xi(r) for each GP sample ──────────────────────────────────────
    xi_samples = []
    for i, c in enumerate(catalogs):
        xyz_gp = np.array(radec_z_to_cartesian(
            c["ra"], c["dec"], c["z"], cat.fid_cosmo)) + shift
        t0 = time.time()
        xi_i = xi_landy_szalay(xyz_gp, xyz_r_s, box_size=box_size,
                                dim=3, periodic=False)
        xi_samples.append(xi_i)
        print(f"  Sample {i+1:2d}/{args.n_samples}: {time.time()-t0:.1f}s  "
              f"N_gal={c['N_galaxies']:,}")

    # ── 7. Limber projection ──────────────────────────────────────────────
    def _to_wtheta(xi_arr):
        """Extract finite xi shells and Limber-project to w(θ)."""
        r_cen = 0.5 * (xi_arr["r_inner_phys"] + xi_arr["r_outer_phys"])
        xi_v  = xi_arr["xi_ls"]
        # keep shells with positive width and finite xi
        ok = (xi_arr["r_outer_phys"] > xi_arr["r_inner_phys"]) & np.isfinite(xi_v)
        return limber_wtheta(r_cen[ok], xi_v[ok], theta_cen, chi_bar,
                             pi_max=args.pi_max)

    print("Limber projecting xi(r) → w(θ) ...")
    w_orig    = _to_wtheta(xi_orig_arr)
    w_samples = np.array([_to_wtheta(xi_i) for xi_i in xi_samples])

    # ── 8. Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # GP sample envelope (fill 16–84%)
    w_med = np.nanmedian(w_samples, axis=0)
    w_lo  = np.nanpercentile(w_samples, 16, axis=0)
    w_hi  = np.nanpercentile(w_samples, 84, axis=0)

    ax.fill_between(theta_cen, w_lo * theta_cen, w_hi * theta_cen,
                    color="#4a90d9", alpha=0.25, label="GP samples 16–84%")

    # Individual GP sample lines
    colors = plt.cm.cool(np.linspace(0.1, 0.9, args.n_samples))
    for i, (w_i, col) in enumerate(zip(w_samples, colors)):
        label = "GP posterior samples" if i == 0 else None
        ax.plot(theta_cen, w_i * theta_cen, color=col, lw=1.0,
                alpha=0.7, label=label)

    # Original catalog
    ax.plot(theta_cen, w_orig * theta_cen, color="white", lw=2.5,
            zorder=10, label="BOSS CMASS-SGC (observed)")
    ax.plot(theta_cen, w_orig * theta_cen, color="#f5a623", lw=1.8,
            zorder=11)

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [degrees]", fontsize=13)
    ax.set_ylabel(r"$\theta \, w(\theta)$", fontsize=13)
    ax.set_title(
        "BOSS CMASS-SGC — angular two-point function\n"
        r"10 GP posterior catalog samples  [cascade xi(r) + Limber projection]",
        fontsize=12,
    )
    ax.set_xlim(theta_edges[0], theta_edges[-1])
    ax.legend(fontsize=10, framealpha=0.3)

    # BAO scale reference (~6.5° at z~0.5)
    ax.axvline(6.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.text(6.5 * 1.06, ax.get_ylim()[0] * 1.4, "BAO ≈ 6.5°",
            color="gray", fontsize=8, va="bottom")

    ax.set_facecolor("#0a0a12")
    fig.patch.set_facecolor("#0a0a12")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, which="both", alpha=0.12, color="white")
    ax.legend(fontsize=10, framealpha=0.2, labelcolor="white",
              facecolor="#111")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {args.out}")

    # Print summary statistics
    print(f"\nw(θ) summary  [cascade + Limber, χ̄={chi_bar:.0f} Mpc/h]:")
    print(f"  {'θ[°]':>8}  {'w_orig':>9}  {'w_med':>9}  {'σ_GP':>9}  {'σ/w':>6}")
    for i, tc in enumerate(theta_cen):
        rms = np.nanstd(w_samples[:, i])
        print(f"  {tc:8.3f}  {w_orig[i]:9.5f}  {w_med[i]:9.5f}"
              f"  {rms:9.5f}  {rms/abs(w_orig[i]+1e-6):6.2f}")


if __name__ == "__main__":
    main()
