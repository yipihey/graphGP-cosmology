"""End-to-end 2MRS → posterior graphGP density field demo.

Pipeline:
  1. Load 2MRS catalog (or run on mock if file not present)
  2. Measure ξ(r) via Landy-Szalay
  3. Sample posterior density field (Matheron rule, Vecchia GP)
  4. Save lightcone HDF5 + optional 3D Cartesian grid
  5. Plot: sky map, n(z), ξ(r), example shell

Usage::

    # Run on mock (no data download needed)
    python demos/demo_2mrs_graphgp.py --mock

    # Run on real 2MRS data (after running fetch_2mrs.py)
    python demos/demo_2mrs_graphgp.py --data data/2mrs/2mrs_1175_done.fits

    # Full run: 20 samples, save HDF5
    python demos/demo_2mrs_graphgp.py --data data/2mrs/2mrs_1175_done.fits \\
        --n-samples 20 --out output/2mrs_density_field.h5

Environment variables (override defaults):
    TWOMRS_DATA         path to 2mrs_1175_done.fits
    TWOMRS_N_SAMPLES    number of posterior samples (default 5)
    TWOMRS_Z_MAX        max redshift cut (default 0.05)
    TWOMRS_NSIDE        HealPIX NSIDE (default 64)
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

# ── path setup ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twopt_density.twoMRS import load_2mrs, make_mock_2mrs
from twopt_density.density_field import sample_posterior_density_field


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="Use a mock catalog (no data file needed)")
    p.add_argument("--data", default=os.environ.get("TWOMRS_DATA", ""),
                   help="Path to 2mrs_1175_done.fits")
    p.add_argument("--n-samples", type=int,
                   default=int(os.environ.get("TWOMRS_N_SAMPLES", "5")))
    p.add_argument("--z-max", type=float,
                   default=float(os.environ.get("TWOMRS_Z_MAX", "0.05")))
    p.add_argument("--nside", type=int,
                   default=int(os.environ.get("TWOMRS_NSIDE", "64")))
    p.add_argument("--n-z-bins", type=int, default=32,
                   help="Redshift shells in lightcone output")
    p.add_argument("--out", default="output/2mrs_density_field.h5")
    p.add_argument("--fig-dir", default="demos/figures")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Load or mock catalog ─────────────────────────────────────────
    if args.mock or not args.data:
        print("=== Using mock 2MRS catalog ===")
        cat = make_mock_2mrs(n_data=15000, n_random=75000, z_max=args.z_max,
                             seed=args.seed)
        label = "mock_2mrs"
    else:
        print(f"=== Loading real 2MRS: {args.data} ===")
        cat = load_2mrs(args.data, z_max=args.z_max, n_random_factor=8)
        label = "2mrs"

    print(f"  N_data={cat.N_data:,}  N_random={cat.N_random:,}")
    print(f"  z range: [{cat.z_data.min():.4f}, {cat.z_data.max():.4f}]")

    # ── Sky map ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={"projection": "mollweide"})
    ra_plot = np.where(cat.ra_data > 180, cat.ra_data - 360, cat.ra_data)
    ax.scatter(np.radians(ra_plot[::5]), np.radians(cat.dec_data[::5]),
               s=0.5, alpha=0.3, c=cat.z_data[::5], cmap="plasma", rasterized=True)
    ax.set_title(f"2MRS sky distribution  (N={cat.N_data:,})")
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
    ax.set_title("2MRS n(z)"); ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_nz.png", dpi=150)
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
    # Kernel fit overlay
    A, r0, al = result.kernel_fit
    r_fit = np.logspace(np.log10(r[0]), np.log10(r[-1]), 200)
    xi_fit = A * np.exp(-((r_fit / r0) ** al))
    ax.plot(r_fit, xi_fit, "r--", label=f"Fit A={A:.2f} r₀={r0:.1f} α={al:.2f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel("ξ(r)")
    ax.set_title("2MRS two-point correlation function"); ax.legend()
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
    weights_std = result.data_weights_std()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(weights, bins=50, alpha=0.7, label="Posterior mean w = 1+δ")
    ax.set_xlabel("Weight 1+δ"); ax.set_ylabel("Count")
    ax.set_title("Per-galaxy density weights (posterior mean)")
    ax.axvline(1.0, color="r", ls="--", label="Mean field = 1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{args.fig_dir}/{label}_weights.png", dpi=150)
    plt.close(fig)
    print(f"  Data weights: mean={weights.mean():.3f}  "
          f"std={weights.std():.3f}  "
          f"range=[{weights.min():.3f}, {weights.max():.3f}]")

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
