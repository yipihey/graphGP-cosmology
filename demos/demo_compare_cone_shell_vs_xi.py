"""Side-by-side comparison: cone-shell ``sigma^2(theta; z)`` vs
3D dyadic-shell ``xi(r; z_shell)`` from morton_cascade, on the same
Quaia subsample sliced into the same z-shells.

The two pipelines differ in geometry:
  - cone-shell   : spherical caps in (ra, dec) x redshift slice; never
                   converts to comoving Cartesian. Pure observable space.
  - cascade xi   : converts (ra, dec, z) to comoving xyz under a
                   fiducial cosmology, then runs Cartesian dyadic-cell
                   pair counts. Standard 3D pipeline.

This demo lets the user check the paper's information-equivalence claim
empirically: per shell, the cone-shell sigma^2 SNR should sit at the
same order as the cascade xi SNR.

Output figures (under demos/figures/cone_shell_vs_xi/):
  panels.png   -- sigma^2(theta; z) vs xi_LS(r = theta * chi_eff; z) per shell
  timing.png   -- wall time: cap counting vs cascade pair counting
  snr.png      -- per-bin signal-to-Poisson-noise per shell

Tunables via env vars:
  PAPER_N_DATA            (default 60_000)
  PAPER_N_RANDOM_FACTOR   (default 3)
  PAPER_NSIDE_CENTRES     (default 64)
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from twopt_density.cascade import xi_landy_szalay_from_window
from twopt_density.distance import (
    DistanceCosmo, comoving_distance, radec_z_to_cartesian,
)
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2_cone_shell_estimator import (
    cap_centre_grid, cone_shell_counts, sigma2_estimate_cone_shell,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures", "cone_shell_vs_xi")
os.makedirs(FIG_DIR, exist_ok=True)


def _per_shell_cascade_xi(
    ra, dec, z, sel_map, nside_mask, fid, n_random_factor, z_edges,
):
    """Run cascade xi_LS per redshift shell. Returns
    (per_shell_arr, t_per_shell)."""
    out = []
    timings = []
    for k in range(z_edges.size - 1):
        z_lo, z_hi = float(z_edges[k]), float(z_edges[k + 1])
        msk = (z >= z_lo) & (z < z_hi)
        if msk.sum() < 200:
            print(f"  shell {k} z=[{z_lo:.2f},{z_hi:.2f}] -- skipped "
                  f"(only {msk.sum()} galaxies)")
            out.append(None)
            timings.append(0.0)
            continue
        ra_s = ra[msk]; dec_s = dec[msk]; z_s = z[msk]
        pos_d = np.asarray(radec_z_to_cartesian(
            jnp.asarray(ra_s), jnp.asarray(dec_s), jnp.asarray(z_s),
            fid,
        ))
        t0 = time.perf_counter()
        arr = xi_landy_szalay_from_window(
            pos_d, ra_s, dec_s, z_s, sel_map, nside_mask, fid,
            n_random_factor=n_random_factor, rng_seed=42 + k,
        )
        t = time.perf_counter() - t0
        timings.append(t)
        out.append(arr)
        print(f"  shell {k} z=[{z_lo:.2f},{z_hi:.2f}] -- "
              f"N={int(msk.sum()):,} cascade={t:.1f}s")
    return out, np.array(timings, dtype=np.float64)


def main():
    n_data = int(os.environ.get("PAPER_N_DATA", 60_000))
    n_random_factor = int(os.environ.get("PAPER_N_RANDOM_FACTOR", 3))
    nside_centres = int(os.environ.get("PAPER_NSIDE_CENTRES", 64))

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia G < 20.0 ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    sel_map, nside_mask = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))

    # restrict to z in [0.6, 2.6] -- where Quaia clustering signal sits
    md = (cat.z_data >= 0.6) & (cat.z_data <= 2.6)
    rng = np.random.default_rng(0)
    n_avail = int(md.sum())
    take = min(n_data, n_avail)
    iD = rng.choice(n_avail, take, replace=False)
    where = np.where(md)[0][iD]
    ra_d = cat.ra_data[where]
    dec_d = cat.dec_data[where]
    z_d = cat.z_data[where]
    print(f"  N_data = {len(ra_d):,}")

    # 4 redshift shells (coarser than the cone-shell-only demo so that
    # cascade has enough galaxies per shell)
    z_edges = np.array([0.6, 1.1, 1.6, 2.1, 2.6])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    print(f"  z-shells: {z_edges}")

    # 8 theta bins log-spaced from 0.1 to 4 deg
    theta_deg = np.exp(np.linspace(np.log(0.1), np.log(4.0), 8))
    theta_rad = np.deg2rad(theta_deg)

    # ---- 2D side: cone-shell sigma^2 -----------------------------------
    print(f"\n2D cone-shell side (cap counts at NSIDE={nside_centres}) ...")
    t0 = time.perf_counter()
    ra_c, dec_c, _ = cap_centre_grid(
        sel_map, nside_centres=nside_centres,
        theta_max_rad=float(theta_rad.max()),
        edge_buffer_frac=1.0, mask_threshold=0.5,
    )
    print(f"  n_centres = {ra_c.size}")
    N_cap, A_cap = cone_shell_counts(
        ra_d, dec_d, z_d, theta_rad, z_edges, ra_c, dec_c,
        nside_lookup=512,
    )
    s2_obs = sigma2_estimate_cone_shell(N_cap)
    t_cone = time.perf_counter() - t0
    print(f"  cone-shell sigma^2 done in {t_cone:.1f}s "
            f"(mean N range {N_cap.mean(axis=0).min():.2g}"
            f" ... {N_cap.mean(axis=0).max():.2g})")
    # SE per (theta, z) from the per-cap variance: SE(sigma^2) ~
    # sqrt(2/N_centres) * Var(N) / mu^2 (large-mu Gaussian limit)
    mu_cap = N_cap.mean(axis=0)
    var_cap = N_cap.var(axis=0, ddof=1)
    se_s2_obs = np.sqrt(2.0 / N_cap.shape[0]) * var_cap / np.maximum(mu_cap, 1e-9) ** 2

    # ---- 3D side: cascade xi_LS per shell -------------------------------
    print("\n3D cascade xi_LS per shell ...")
    cascade_arrs, t_cascade_per_shell = _per_shell_cascade_xi(
        ra_d, dec_d, z_d, sel_map, nside_mask, fid, n_random_factor, z_edges,
    )
    t_cascade_total = float(t_cascade_per_shell.sum())
    print(f"  cascade per-shell total: {t_cascade_total:.1f}s")

    # convert (theta, z_shell) -> r = theta * chi(z_centre) for overlay
    chi_centres = np.asarray(comoving_distance(
        jnp.asarray(z_centres, dtype=jnp.float64), fid
    ), dtype=np.float64)
    # r grid per shell from theta_rad * chi
    r_per_shell = theta_rad[None, :] * chi_centres[:, None]   # (n_z, n_theta)

    # ---- panels.png -----------------------------------------------------
    fig, axes = plt.subplots(1, z_edges.size - 1,
                                  figsize=(3.7 * (z_edges.size - 1), 4.3),
                                  sharey=False)
    if z_edges.size == 2:
        axes = [axes]
    for k, ax in enumerate(axes):
        z_lo, z_hi = z_edges[k], z_edges[k + 1]
        ax.errorbar(
            r_per_shell[k], s2_obs[:, k],
            yerr=np.maximum(se_s2_obs[:, k], 1e-10),
            color="C0", marker="o", ms=4, lw=0,
            elinewidth=1.0, capsize=2,
            label=r"$\sigma^2_{\rm obs}(\theta;z)$",
        )
        # overlay cascade xi (on the right axis)
        ax2 = ax.twinx()
        arr = cascade_arrs[k]
        if arr is not None:
            r_in = arr["r_inner_phys"]
            r_out = arr["r_outer_phys"]
            xi = arr["xi_ls"]
            r_centre = 0.5 * (r_in + r_out)
            keep = (r_centre > 1.0) & np.isfinite(xi)
            ax2.plot(r_centre[keep], xi[keep], "C3-", lw=1.0, marker="s",
                       ms=3, label=r"$\xi_{\rm LS}(r;z)$ (cascade)")
            ax2.set_yscale("log")
            ax2.set_ylabel(r"$\xi_{\rm LS}$", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$r = \theta \cdot \chi(z_{\rm eff})$ [Mpc/h]")
        ax.set_ylabel(r"$\sigma^2_{\rm obs}$", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.set_title(rf"$z\in[{z_lo:.2f},{z_hi:.2f}]$")
    plt.tight_layout()
    out_panels = os.path.join(FIG_DIR, "panels.png")
    plt.savefig(out_panels, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out_panels}")

    # ---- timing.png -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(
        ["cone-shell\n(cap counting)", "cascade\n(pair counting per shell)"],
        [t_cone, t_cascade_total],
        color=["C0", "C3"], edgecolor="k",
    )
    for b, v in zip(bars, [t_cone, t_cascade_total]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                  f"{v:.1f}s", ha="center", fontsize=10)
    ax.set_ylabel("wall time [s]")
    ax.set_title(rf"head-to-head: same Quaia subsample, "
                    rf"$N_d$={len(ra_d):,}, {z_edges.size-1} shells")
    plt.tight_layout()
    out_timing = os.path.join(FIG_DIR, "timing.png")
    plt.savefig(out_timing, dpi=140)
    plt.close(fig)
    print(f"wrote {out_timing}")

    # ---- snr.png --------------------------------------------------------
    snr_cone = s2_obs / np.maximum(se_s2_obs, 1e-30)        # (n_theta, n_z)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("viridis")
    for k in range(z_edges.size - 1):
        col = cmap(k / max(z_edges.size - 2, 1))
        ax.plot(theta_deg, snr_cone[:, k], color=col, marker="o", ms=4, lw=1,
                  label=rf"$z\in[{z_edges[k]:.2f},{z_edges[k+1]:.2f}]$")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$\sigma^2_{\rm obs} / {\rm SE}$")
    ax.set_title(r"per-bin SNR of cone-shell $\sigma^2(\theta;z)$")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    out_snr = os.path.join(FIG_DIR, "snr.png")
    plt.savefig(out_snr, dpi=140)
    plt.close(fig)
    print(f"wrote {out_snr}")

    # ---- bao_fisher.png -------------------------------------------------
    # simple per-shell BAO matched-filter Fisher for the cone-shell side
    # alone, F = T^T C^-1 T with T the BAO template (full - nowiggle).
    # Diagonal Poisson covariance proxy ((SE)^2). Cascade-side BAO Fisher
    # would mirror this construction with cascade's xi templates -- left
    # for a dedicated BAO study.
    print("\nper-shell BAO matched-filter Fisher (cone-shell, diagonal cov) ...")
    from scipy.ndimage import gaussian_filter1d
    from twopt_density.sigma2_cone_shell import sigma2_cone_shell_bao_template

    z_grid = np.linspace(0.01, 4.0, 400)
    hist, _ = np.histogram(z_d, bins=z_grid)
    nz_centres = 0.5 * (z_grid[:-1] + z_grid[1:])
    dndz_smooth = gaussian_filter1d(hist.astype(np.float64), sigma=4)
    dndz = np.interp(z_grid, nz_centres, dndz_smooth)

    fisher = np.zeros(z_edges.size - 1)
    for k in range(z_edges.size - 1):
        b_k = 0.55 * (1.0 + z_centres[k])
        T = sigma2_cone_shell_bao_template(
            theta_rad, z_edges[k], z_edges[k + 1], z_grid, dndz, fid,
            bias=float(b_k), sigma8=0.81,
            ell_min=2.0, ell_max=5e4, n_ell=600,
        )
        cov_diag = np.maximum(se_s2_obs[:, k], 1e-30) ** 2
        fisher[k] = float(np.sum(T ** 2 / cov_diag))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(
        [rf"$z\in[{z_edges[k]:.2f},{z_edges[k+1]:.2f}]$"
         for k in range(z_edges.size - 1)],
        fisher,
        color="C0", edgecolor="k",
    )
    for k, v in enumerate(fisher):
        ax.text(k, v * 1.02, f"{v:.2g}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel(r"BAO Fisher $F_\alpha = T^\top C^{-1} T$")
    ax.set_title("per-shell cone-shell BAO information")
    plt.tight_layout()
    out_fisher = os.path.join(FIG_DIR, "bao_fisher.png")
    plt.savefig(out_fisher, dpi=140)
    plt.close(fig)
    print(f"wrote {out_fisher}")

    # ---- summary table to stdout ---------------------------------------
    print("\n" + "=" * 64)
    print("Summary: cone-shell vs cascade on Quaia G<20")
    print("=" * 64)
    print(f"{'shell':>14s}   {'N_gal':>8s}   {'mean N_cap':>11s}   "
            f"{'cascade [s]':>11s}   {'cone-shell SNR (sum, sqrt(F))':>30s}")
    for k in range(z_edges.size - 1):
        N_in_shell = int(((z_d >= z_edges[k]) & (z_d < z_edges[k + 1])).sum())
        avg_mu = float(N_cap[:, :, k].mean())
        snr_total = float(np.sqrt(fisher[k]))
        print(
            f"  z=[{z_edges[k]:4.2f},{z_edges[k+1]:4.2f}]   "
            f"{N_in_shell:8d}   {avg_mu:11.3g}   "
            f"{t_cascade_per_shell[k]:11.1f}   {snr_total:30.3g}"
        )
    print(f"\ncone-shell wall time: {t_cone:.1f}s "
            f"(cap counting on {ra_c.size:,} centres)")
    print(f"cascade wall time:    {t_cascade_total:.1f}s "
            f"({z_edges.size - 1} per-shell pair-count runs)")
    print(f"speedup: {t_cascade_total / max(t_cone, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
