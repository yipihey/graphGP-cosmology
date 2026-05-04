"""Per-galaxy density weights tied to sigma^2(R), on Quaia G < 20.

Builds the smooth, kernel-weighted per-particle overdensity
delta_i(R) for a top-hat sphere of radius R.  The mean
<delta_i> is the DD-only sigma^2(R) estimate -- a single sum
over data points, no random pair counting at the final step.

Comparison vs. the xi-based weights:
  - xi-aggregated weights (twopt_density.weights_pair_counts):
    integer bin counts inside [r_lo, r_hi] -> per-bin overdensity
    -> aggregated to one weight per particle.
  - sigma^2 weights: smooth kernel K_TH(r; R) integrates over the
    same volume but weighted by the spherical-overlap profile.

We plot:
  - on-sky map coloured by w_i (sigma^2 version) -- much smoother
    than the xi version since each w_i averages over many partners
    weighted by the kernel
  - PDF of w_i vs the xi-based PDF
  - <delta_i>_i (the DD-only sigma^2) vs the LS sigma^2 (with
    full random catalogue)

Output: demos/figures/quaia_sigma2_weights.png
"""

from __future__ import annotations

import os
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2 import (
    density_weights_sigma2, kernel_TH_3d, per_particle_kernel_counts,
    sigma2_from_rp_pi_pairs,
)
from twopt_density.weights_pair_counts import (
    aggregate_weights, per_particle_overdensity_windowed,
    per_particle_pair_counts, per_particle_cross_counts,
)


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env(n, d, cast=str):
    v = os.environ.get(n)
    return cast(v) if v else d


def main():
    n_data = _env("PAPER_N_DATA", 50_000, int)
    n_random = _env("PAPER_N_RANDOM", 150_000, int)
    R = _env("SIGMA2_R", 30.0, float)         # Mpc/h, sphere radius

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=3, rng_seed=0,
    )
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    mr = (cat.z_random >= 0.8) & (cat.z_random <= 2.5)
    rng = np.random.default_rng(0)
    idx_d = rng.choice(int(md.sum()), min(n_data, int(md.sum())),
                          replace=False)
    idx_r = rng.choice(int(mr.sum()), min(n_random, int(mr.sum())),
                          replace=False)
    where_d = np.where(md)[0][idx_d]
    where_r = np.where(mr)[0][idx_r]
    xyz_d = cat.xyz_data[where_d]
    xyz_r = cat.xyz_random[where_r]
    ra_d = cat.ra_data[where_d]; dec_d = cat.dec_data[where_d]
    z_d = cat.z_data[where_d]
    print(f"  N_data = {len(xyz_d):,}, N_random = {len(xyz_r):,}, "
          f"R = {R} Mpc/h")

    # shift to non-negative for cKDTree
    all_xyz = np.vstack([xyz_d, xyz_r])
    shift = -all_xyz.min(axis=0) + 100.0
    pos_d = xyz_d + shift
    pos_r = xyz_r + shift

    # --- 1. sigma^2 per-galaxy weights ---
    print("\nbuilding sigma^2 per-galaxy weights (kernel-weighted DP) ...")
    t0 = time.perf_counter()
    w_s2, delta_s2, aux_s2 = density_weights_sigma2(
        pos_d, pos_r, R=R, kernel="tophat",
    )
    t_s2 = time.perf_counter() - t0
    sigma2_from_w = float(np.mean(delta_s2))
    print(f"  done ({t_s2:.0f}s); mean partners = "
          f"{aux_s2['b_DD_K_mean']:.2f} (DD), {aux_s2['b_DR_K_mean']:.2f} (DR)")
    print(f"  <delta_i> -> sigma^2(R={R}) = {sigma2_from_w:.4f}")

    # --- 2. xi-based per-galaxy weights for comparison ---
    print("\nbuilding xi-based per-galaxy weights "
          "(window-aware DP, same R range) ...")
    t0 = time.perf_counter()
    r_edges = np.linspace(2.0, 2.0 * R, 14)
    b_DD_xi = per_particle_pair_counts(pos_d, r_edges)
    b_DR_xi = per_particle_cross_counts(pos_d, pos_r, r_edges)
    delta_xi_perbin = per_particle_overdensity_windowed(
        b_DD_xi.astype(np.float64), b_DR_xi.astype(np.float64),
        len(pos_d), len(pos_r),
    )
    delta_xi_perbin = np.where(np.isfinite(delta_xi_perbin),
                                  delta_xi_perbin, 0.0)
    # uniform "RR" weighting across bins (just for the comparison)
    a_j = np.ones(len(r_edges) - 1)
    delta_xi = (delta_xi_perbin * a_j[None, :]).sum(axis=1) / a_j.sum()
    w_xi = 1.0 + delta_xi
    t_xi = time.perf_counter() - t0
    print(f"  done ({t_xi:.0f}s)")

    # --- 3. statistics summary ---
    print(f"\nstatistics:")
    print(f"  sigma^2 weights: mean = {w_s2.mean():.3f}, "
          f"std = {w_s2.std():.3f}, "
          f"5/95% = [{np.percentile(w_s2, 5):.3f}, "
          f"{np.percentile(w_s2, 95):.3f}]")
    print(f"  xi weights:      mean = {w_xi.mean():.3f}, "
          f"std = {w_xi.std():.3f}, "
          f"5/95% = [{np.percentile(w_xi, 5):.3f}, "
          f"{np.percentile(w_xi, 95):.3f}]")
    ratio_std = float(w_s2.std() / max(w_xi.std(), 1e-12))
    print(f"  std ratio (sigma^2 / xi) = {ratio_std:.2f}")

    # --- figure ---
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    ax_sky_s2, ax_sky_xi = axs[0]
    ax_pdf, ax_scatter = axs[1]

    # downsample for the sky scatter
    n_plot = min(40_000, len(ra_d))
    ip = rng.choice(len(ra_d), n_plot, replace=False)
    vmin = float(np.percentile(np.r_[w_s2[ip], w_xi[ip]], 2))
    vmax = float(np.percentile(np.r_[w_s2[ip], w_xi[ip]], 98))
    sc = ax_sky_s2.scatter(ra_d[ip], dec_d[ip], c=w_s2[ip], s=1.5,
                              cmap="coolwarm", vmin=vmin, vmax=vmax,
                              rasterized=True)
    ax_sky_s2.set_xlim(0, 360); ax_sky_s2.set_ylim(-90, 90)
    ax_sky_s2.set_xlabel("RA [deg]"); ax_sky_s2.set_ylabel("Dec [deg]")
    ax_sky_s2.set_title(rf"$\sigma^2$ weights: $w_i = 1 + \delta_i(R={R:.0f})$")
    fig.colorbar(sc, ax=ax_sky_s2, shrink=0.85, label=r"$w_i^{\sigma^2}$")

    sc_xi = ax_sky_xi.scatter(ra_d[ip], dec_d[ip], c=w_xi[ip], s=1.5,
                                 cmap="coolwarm", vmin=vmin, vmax=vmax,
                                 rasterized=True)
    ax_sky_xi.set_xlim(0, 360); ax_sky_xi.set_ylim(-90, 90)
    ax_sky_xi.set_xlabel("RA [deg]"); ax_sky_xi.set_ylabel("Dec [deg]")
    ax_sky_xi.set_title(r"$\xi$ weights (DP per-particle)")
    fig.colorbar(sc_xi, ax=ax_sky_xi, shrink=0.85, label=r"$w_i^{\xi}$")

    # PDF comparison
    bins = np.linspace(min(w_s2.min(), w_xi.min()) - 0.1,
                          max(w_s2.max(), w_xi.max()) + 0.1, 80)
    ax_pdf.hist(w_s2, bins=bins, density=True, alpha=0.55,
                  color="C0", label=fr"$\sigma^2$ weight, std={w_s2.std():.3f}")
    ax_pdf.hist(w_xi, bins=bins, density=True, alpha=0.55,
                  color="C3", label=fr"$\xi$ weight, std={w_xi.std():.3f}")
    ax_pdf.axvline(1.0, color="g", ls="--", lw=1.5, label="uniform $w=1$")
    ax_pdf.set_xlabel(r"$w_i$"); ax_pdf.set_ylabel(r"$p(w_i)$")
    ax_pdf.set_title(r"PDF of per-galaxy weights "
                       fr"($N={len(w_s2):,}$, ratio $\sigma^2/\xi = "
                       fr"{ratio_std:.2f}$)")
    ax_pdf.legend(fontsize=9); ax_pdf.grid(alpha=0.3)

    # scatter w_s2 vs w_xi -- same particles
    n_sc = min(20_000, len(w_s2))
    isc = rng.choice(len(w_s2), n_sc, replace=False)
    ax_scatter.plot(w_xi[isc], w_s2[isc], "o", ms=1.0, alpha=0.3, color="C2",
                       rasterized=True)
    lim = (vmin, vmax)
    ax_scatter.plot(lim, lim, "k:", lw=0.7, label="$y=x$")
    ax_scatter.set_xlabel(r"$\xi$ weight $w_i^\xi$")
    ax_scatter.set_ylabel(r"$\sigma^2$ weight $w_i^{\sigma^2}$")
    ax_scatter.set_xlim(*lim); ax_scatter.set_ylim(*lim)
    pearson = float(np.corrcoef(w_xi[isc], w_s2[isc])[0, 1])
    ax_scatter.set_title(rf"per-galaxy weight comparison "
                            rf"(Pearson $\rho={pearson:.3f}$)")
    ax_scatter.legend(fontsize=9); ax_scatter.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_sigma2_weights.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
