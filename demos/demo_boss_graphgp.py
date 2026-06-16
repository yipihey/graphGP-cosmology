"""End-to-end BOSS DR12 (simBIG SGC) → posterior graphGP density field demo.

Pipeline:
  1. Load BOSS DR12 CMASS-SGC and/or LOWZ-SGC catalogs (or run on mock)
  2. Apply simBIG clean subsample cuts
  3. Measure ξ(r) via Landy-Szalay
  4. Sample posterior density field (Matheron rule, Vecchia GP)
  5. Save lightcone HDF5 + optional 3D Cartesian grid
  6. Plot: sky map, n(z), ξ(r), example shell

Usage::

    # Run on mock (no data download needed)
    python demos/demo_boss_graphgp.py --mock

    # Run on real BOSS CMASS-SGC data (after running fetch_boss.py)
    python demos/demo_boss_graphgp.py \\
        --data data/boss/galaxy_DR12v5_CMASS_South.fits.gz \\
        --randoms data/boss/random0_DR12v5_CMASS_South.fits.gz \\
        --sample CMASS

    # LOWZ
    python demos/demo_boss_graphgp.py \\
        --data data/boss/galaxy_DR12v5_LOWZ_South.fits.gz \\
        --randoms data/boss/random0_DR12v5_LOWZ_South.fits.gz \\
        --sample LOWZ

    # Full run: 20 samples, save HDF5
    python demos/demo_boss_graphgp.py \\
        --data data/boss/galaxy_DR12v5_CMASS_South.fits.gz \\
        --randoms data/boss/random0_DR12v5_CMASS_South.fits.gz \\
        --sample CMASS --n-samples 20 --out output/boss_cmass_density_field.h5

Environment variables (override defaults):
    BOSS_DATA_CMASS     path to galaxy_DR12v5_CMASS_South.fits.gz
    BOSS_RAND_CMASS     path to random0_DR12v5_CMASS_South.fits.gz
    BOSS_DATA_LOWZ      path to galaxy_DR12v5_LOWZ_South.fits.gz
    BOSS_RAND_LOWZ      path to random0_DR12v5_LOWZ_South.fits.gz
    BOSS_SAMPLE         CMASS or LOWZ (default CMASS)
    BOSS_N_SAMPLES      number of posterior samples (default 5)
    BOSS_NSIDE          HealPIX NSIDE (default 256)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twopt_density.boss import load_boss, make_mock_boss, SIMBIG_CUTS
from twopt_density.density_field import sample_posterior_density_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="Use a mock BOSS catalog (no data file needed)")
    p.add_argument("--sample", default=os.environ.get("BOSS_SAMPLE", "CMASS"),
                   choices=["CMASS", "LOWZ"],
                   help="simBIG subsample (CMASS or LOWZ)")
    p.add_argument("--data", default="",
                   help="Path to galaxy FITS (overrides env; auto-detected if empty)")
    p.add_argument("--randoms", default="", nargs="+",
                   help="Path(s) to random FITS (overrides env; auto-detected if empty)")
    p.add_argument("--n-samples", type=int,
                   default=int(os.environ.get("BOSS_N_SAMPLES", "5")))
    p.add_argument("--nside", type=int,
                   default=int(os.environ.get("BOSS_NSIDE", "256")))
    p.add_argument("--n-z-bins", type=int, default=32,
                   help="Redshift shells in lightcone output")
    p.add_argument("--out", default="")
    p.add_argument("--fig-dir", default="demos/figures")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _auto_detect_paths(args):
    """Fill in data/randoms paths from environment or conventional filenames."""
    sample = args.sample
    if not args.data:
        env_key = f"BOSS_DATA_{sample}"
        args.data = os.environ.get(env_key,
            f"data/boss/galaxy_DR12v5_{sample}_South.fits.gz")
    if not args.randoms or args.randoms == [""]:
        env_key = f"BOSS_RAND_{sample}"
        default = f"data/boss/random0_DR12v5_{sample}_South.fits.gz"
        args.randoms = [os.environ.get(env_key, default)]
    if not args.out:
        args.out = f"output/boss_{sample.lower()}_density_field.h5"
    return args


def main():
    args = parse_args()
    args = _auto_detect_paths(args)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sample = args.sample
    cuts = SIMBIG_CUTS[sample]
    label = f"boss_{sample.lower()}_sgc"

    # ── Load or mock catalog ─────────────────────────────────────────
    if args.mock or not os.path.exists(args.data):
        if not args.mock:
            print(f"  Data file not found: {args.data} — using mock instead")
        print(f"=== Using mock BOSS {sample}-SGC catalog ===")
        cat = make_mock_boss(sample=sample, seed=args.seed)
        label = f"mock_{label}"
    else:
        print(f"=== Loading real BOSS {sample}-SGC: {args.data} ===")
        existing_randoms = [r for r in args.randoms if os.path.exists(r)]
        cat = load_boss(
            data_paths=[args.data],
            randoms_paths=existing_randoms or None,
            sample=sample,
            nside=args.nside,
            simbig_sgc_cuts=True,
        )

    print(f"  N_data={cat.N_data:,}  N_random={cat.N_random:,}")
    print(f"  z range: [{cat.z_data.min():.4f}, {cat.z_data.max():.4f}]  "
          f"(simBIG cut: {cuts['z_min']:.2f}–{cuts['z_max']:.2f})")

    # ── Sky map ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": "mollweide"})
    ra_plot = np.where(cat.ra_data > 180, cat.ra_data - 360, cat.ra_data)
    sc = ax.scatter(np.radians(ra_plot[::5]), np.radians(cat.dec_data[::5]),
                    s=0.3, alpha=0.4, c=cat.z_data[::5], cmap="viridis",
                    rasterized=True, vmin=cuts["z_min"], vmax=cuts["z_max"])
    plt.colorbar(sc, ax=ax, label="z", fraction=0.02, pad=0.02)
    ax.set_title(f"BOSS {sample}-SGC (simBIG cuts)  N={cat.N_data:,}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_skymap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {label}_skymap.png")

    # ── n(z) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(cuts["z_min"], cuts["z_max"], 40)
    ax.hist(cat.z_data, bins=bins, density=True, alpha=0.7, label="Data")
    ax.hist(cat.z_random, bins=bins, density=True, alpha=0.4, label="Randoms")
    ax.set_xlabel("z"); ax.set_ylabel("n(z) [normalized]")
    ax.set_title(f"BOSS {sample}-SGC n(z)"); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_nz.png", dpi=150)
    plt.close(fig)

    # ── FKP weight distribution ───────────────────────────────────────
    if hasattr(cat, 'w_data') and cat.w_data is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cat.w_data, bins=60, density=True, alpha=0.7)
        ax.set_xlabel("w_total (sys × noz × cp × fkp)")
        ax.set_ylabel("n [normalized]")
        ax.set_title(f"BOSS {sample}-SGC combined weights")
        ax.text(0.98, 0.95, f"mean={cat.w_data.mean():.3f}  "
                f"std={cat.w_data.std():.3f}",
                transform=ax.transAxes, ha="right", va="top")
        plt.tight_layout()
        fig.savefig(f"{args.fig_dir}/{label}_wdist.png", dpi=150)
        plt.close(fig)

    # ── Posterior density field ───────────────────────────────────────
    print(f"\n=== Sampling {args.n_samples} posterior density field(s) ===")
    result = sample_posterior_density_field(
        cat,
        n_samples=args.n_samples,
        n_z_bins=args.n_z_bins,
        nside=args.nside,
        seed=args.seed,
        verbose=True,
    )

    # ── ξ(r) plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    r = result.r_centers
    xi = result.xi_j
    ax.plot(r, xi, "ko-", ms=4, label="Measured ξ(r)")
    A, r0, al = result.kernel_fit
    r_fit = np.logspace(np.log10(r[0]), np.log10(r[-1]), 200)
    xi_fit = A * np.exp(-((r_fit / r0) ** al))
    ax.plot(r_fit, xi_fit, "r--", label=f"Fit A={A:.2f} r₀={r0:.1f} α={al:.2f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel("ξ(r)")
    ax.set_title(f"BOSS {sample}-SGC two-point correlation function"); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_xi.png", dpi=150)
    plt.close(fig)
    print(f"  Kernel fit: A={A:.3f}  r0={r0:.2f} Mpc/h  alpha={al:.2f}")

    # ── Example shell plot ────────────────────────────────────────────
    try:
        import healpy as hp
        iz_mid = args.n_z_bins // 2
        z_lo = result.z_edges[iz_mid]
        z_hi = result.z_edges[iz_mid + 1]
        shell_mean = result.shell_mean(iz_mid)
        shell_std = result.shell_std(iz_mid)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        hp.mollview(shell_mean, fig=fig.number, sub=121,
                    title=f"Posterior mean 1+δ  z=[{z_lo:.3f},{z_hi:.3f}]",
                    min=0, max=3, cmap="RdBu_r", hold=True)
        hp.mollview(shell_std, fig=fig.number, sub=122,
                    title=f"Posterior std  z=[{z_lo:.3f},{z_hi:.3f}]",
                    min=0, cmap="viridis", hold=True)
        plt.tight_layout()
        fig.savefig(f"{args.fig_dir}/{label}_shell_z{iz_mid}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {label}_shell_z{iz_mid}.png  "
              f"(z=[{z_lo:.3f},{z_hi:.3f}])")
    except ImportError:
        print("  (healpy not importable for shell plot — skipping mollview)")

    # ── Data-point weights ────────────────────────────────────────────
    weights = result.data_weights()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(weights, bins=50, alpha=0.7, label="Posterior mean w = 1+δ")
    ax.set_xlabel("Weight 1+δ"); ax.set_ylabel("Count")
    ax.set_title(f"Per-galaxy density weights  BOSS {sample}-SGC")
    ax.axvline(1.0, color="r", ls="--", label="Mean field = 1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_weights.png", dpi=150)
    plt.close(fig)
    print(f"  Data weights: mean={weights.mean():.3f}  "
          f"std={weights.std():.3f}  "
          f"range=[{weights.min():.3f}, {weights.max():.3f}]")

    # ── Angular completeness map ──────────────────────────────────────
    if hasattr(result, 'sel_map') and result.sel_map is not None:
        try:
            import healpy as hp
            fig = plt.figure(figsize=(8, 4))
            hp.mollview(result.sel_map, fig=fig.number,
                        title=f"BOSS {sample}-SGC angular completeness",
                        min=0, max=1, cmap="Blues", hold=True)
            plt.tight_layout()
            fig.savefig(f"{args.fig_dir}/{label}_completeness.png", dpi=150)
            plt.close(fig)
            covered = (result.sel_map > 0.5).mean()
            print(f"  Angular completeness: {covered*100:.1f}% sky covered (>50%)")
        except Exception:
            pass

    # ── Save HDF5 ─────────────────────────────────────────────────────
    result.to_hdf5(args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n  Saved: {args.out}  ({size_mb:.1f} MB)")
    print(f"  delta_lightcone shape: {result.delta_lightcone.shape}")
    print(f"  (n_samples={args.n_samples}, "
          f"n_z_bins={args.n_z_bins}, N_pix=12×{args.nside}²={12*args.nside**2})")
    print(f"\n  Read back with h5py:")
    print(f"    import h5py")
    print(f"    f = h5py.File('{args.out}', 'r')")
    print(f"    delta_lc = f['lightcone'][:]  # {result.delta_lightcone.shape}")


if __name__ == "__main__":
    main()
