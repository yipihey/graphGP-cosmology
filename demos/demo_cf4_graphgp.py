"""End-to-end CF4 → posterior graphGP density field demo.

Pipeline:
  1. Load CF4 catalog (or run on mock if file not present)
  2. Optional: toggle distance-modulus positions vs z_CMB positions
  3. Measure ξ(r) via Landy-Szalay
  4. Sample posterior density field (Matheron rule, Vecchia GP)
  5. Save lightcone HDF5 + optional 3D Cartesian grid
  6. Plot: sky map, n(z), ξ(r), example shell, velocity field overlay

Usage::

    # Run on mock (no data download needed)
    python demos/demo_cf4_graphgp.py --mock

    # Real CF4 galaxies (z_CMB positions)
    python demos/demo_cf4_graphgp.py --data data/cf4/kallcf4.fits

    # Use distance-modulus positions (removes peculiar velocity bias)
    python demos/demo_cf4_graphgp.py --data data/cf4/kallcf4.fits --use-dm

    # Full run: 20 samples, save HDF5
    python demos/demo_cf4_graphgp.py --data data/cf4/kallcf4.fits \\
        --n-samples 20 --out output/cf4_density_field.h5

Environment variables (override defaults):
    CF4_DATA            path to kallcf4.fits
    CF4_N_SAMPLES       number of posterior samples (default 5)
    CF4_Z_MAX           max redshift cut (default 0.05)
    CF4_NSIDE           HealPIX NSIDE (default 64)
    CF4_USE_DM          set to "1" to use distance-modulus positions
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

from twopt_density.cf4 import load_cf4, make_mock_cf4
from twopt_density.density_field import sample_posterior_density_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="Use a mock CF4 catalog (no data file needed)")
    p.add_argument("--data", default=os.environ.get("CF4_DATA", ""),
                   help="Path to kallcf4.fits")
    p.add_argument("--use-dm", action="store_true",
                   default=bool(int(os.environ.get("CF4_USE_DM", "0"))),
                   help="Use distance-modulus positions (removes peculiar velocity bias)")
    p.add_argument("--n-samples", type=int,
                   default=int(os.environ.get("CF4_N_SAMPLES", "5")))
    p.add_argument("--z-max", type=float,
                   default=float(os.environ.get("CF4_Z_MAX", "0.05")))
    p.add_argument("--nside", type=int,
                   default=int(os.environ.get("CF4_NSIDE", "64")))
    p.add_argument("--n-z-bins", type=int, default=32,
                   help="Redshift shells in lightcone output")
    p.add_argument("--out", default="output/cf4_density_field.h5")
    p.add_argument("--fig-dir", default="demos/figures")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pos_mode = "distance_modulus" if args.use_dm else "z_CMB"
    label = f"cf4_{pos_mode.replace('_', '')}"

    # ── Load or mock catalog ─────────────────────────────────────────
    if args.mock or not args.data:
        print("=== Using mock CF4 catalog ===")
        cat = make_mock_cf4(n_data=10000, n_random=50000, z_max=args.z_max,
                            seed=args.seed)
        label = f"mock_{label}"
    else:
        print(f"=== Loading real CF4: {args.data}  "
              f"(positions: {pos_mode}) ===")
        cat = load_cf4(args.data, z_max=args.z_max,
                       use_distance_modulus=args.use_dm,
                       n_random_factor=8)

    print(f"  N_data={cat.N_data:,}  N_random={cat.N_random:,}")
    print(f"  z range: [{cat.z_data.min():.4f}, {cat.z_data.max():.4f}]")
    if hasattr(cat, 'mu_data') and cat.mu_data is not None:
        print(f"  μ range: [{cat.mu_data.min():.2f}, {cat.mu_data.max():.2f}]  "
              f"σ_μ mean: {cat.sigma_mu_data.mean():.3f}")

    # ── Sky map ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": "mollweide"})
    ra_plot = np.where(cat.ra_data > 180, cat.ra_data - 360, cat.ra_data)
    sc = ax.scatter(np.radians(ra_plot[::3]), np.radians(cat.dec_data[::3]),
                    s=0.5, alpha=0.4, c=cat.z_data[::3], cmap="plasma",
                    rasterized=True, vmin=0, vmax=args.z_max)
    plt.colorbar(sc, ax=ax, label="z", fraction=0.02, pad=0.02)
    ax.set_title(f"CF4 sky distribution  (N={cat.N_data:,}, positions: {pos_mode})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_skymap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {label}_skymap.png")

    # ── n(z) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, args.z_max, 40)
    ax.hist(cat.z_data, bins=bins, density=True, alpha=0.7, label="Data")
    ax.hist(cat.z_random, bins=bins, density=True, alpha=0.4, label="Randoms")
    ax.set_xlabel("z"); ax.set_ylabel("n(z) [normalized]")
    ax.set_title(f"CF4 n(z)  (positions: {pos_mode})"); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_nz.png", dpi=150)
    plt.close(fig)

    # ── Peculiar velocity distribution (if available) ─────────────────
    if not args.use_dm and hasattr(cat, 'peculiar_velocities_km_s'):
        try:
            vpec = cat.peculiar_velocities_km_s()
            if vpec is None:
                raise ValueError("mu_data not available (mock catalog has no μ)")

            fig, ax = plt.subplots(figsize=(7, 4))
            vclip = np.clip(vpec, -1500, 1500)
            ax.hist(vclip, bins=60, density=True, alpha=0.7, color="steelblue")
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.set_xlabel("v_pec [km/s]"); ax.set_ylabel("n [normalized]")
            ax.set_title("CF4 peculiar velocity distribution")
            sigma = np.std(vpec[np.abs(vpec) < 1500])
            ax.text(0.98, 0.95, f"σ = {sigma:.0f} km/s",
                    transform=ax.transAxes, ha="right", va="top")
            plt.tight_layout()
            fig.savefig(f"{args.fig_dir}/{label}_vpec.png", dpi=150)
            plt.close(fig)
            print(f"  v_pec: mean={vpec.mean():.0f}  σ={sigma:.0f} km/s")
        except Exception as e:
            print(f"  (peculiar velocity plot skipped: {e})")

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
    ax.set_title(f"CF4 two-point correlation function  ({pos_mode})"); ax.legend()
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
    ax.set_title(f"Per-galaxy density weights (posterior mean)  [{pos_mode}]")
    ax.axvline(1.0, color="r", ls="--", label="Mean field = 1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_weights.png", dpi=150)
    plt.close(fig)
    print(f"  Data weights: mean={weights.mean():.3f}  "
          f"std={weights.std():.3f}  "
          f"range=[{weights.min():.3f}, {weights.max():.3f}]")

    # ── Comparison: z_CMB vs distance-modulus shells (if possible) ────
    # This plot is only generated from the --use-dm run to avoid running twice
    if args.use_dm and not (args.mock or not args.data):
        print("  (Tip: run without --use-dm to compare z_CMB positions)")

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
