"""Blend movie of 10 GP-posterior galaxy catalog samples — BOSS CMASS SGC.

Renders a classic cone/wedge diagram (RA vs comoving distance) for each
posterior sample, then writes an MP4 that smoothly crossfades between them.
The morphing of filaments, voids, and walls between samples visualises the
observational uncertainty in the reconstructed large-scale structure.

Usage::

    python demos/movie_boss_gp_samples.py [--n-samples 10] [--fps 30]
                                          [--blend-frames 90]
                                          [--out output/boss_gp_blend.mp4]
"""

import argparse
import os
import sys
import time

# Ensure the package root is on the path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

from twopt_density.boss import load_boss
from twopt_density.density_field import sample_posterior_density_field
from twopt_density.distance import comoving_distance


# ── Cosmology helper ──────────────────────────────────────────────────────────

def chi_of_z(z_arr, fid_cosmo):
    """Comoving distance (Mpc/h) for an array of redshifts."""
    import jax.numpy as jnp
    return np.array(comoving_distance(jnp.asarray(z_arr), fid_cosmo))


# ── Projection ────────────────────────────────────────────────────────────────

def to_wedge_xy(ra_deg, z_arr, fid_cosmo, ra_centre=361.5):
    """RA (deg) × z → 2D Cartesian wedge coordinates.

    The SGC footprint wraps around RA=0: [335°, 360°] ∪ [0°, 28°].
    We unwrap to [335°, 388°] and project onto a fan centred at ra_centre.
    """
    ra_uw = np.where(ra_deg < 100.0, ra_deg + 360.0, ra_deg)
    chi   = chi_of_z(z_arr, fid_cosmo)
    angle = np.radians(ra_uw - ra_centre)
    x = chi * np.cos(angle)
    y = chi * np.sin(angle)
    return x, y


def density_map(x, y, bins, x_range, y_range, sigma_pix=2.5):
    """2D log-density map on a regular grid, Gaussian-smoothed."""
    H, _, _ = np.histogram2d(x, y, bins=bins,
                              range=[x_range, y_range])
    H = gaussian_filter(H.astype(np.float64), sigma=sigma_pix)
    return H.T   # (ny, nx)


# ── Colormap ──────────────────────────────────────────────────────────────────

def make_cmap():
    """Black → deep blue → cyan → white: shows voids-to-filaments."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.00, "#000000"),
        (0.20, "#08103a"),
        (0.40, "#0d3060"),
        (0.60, "#1a7aad"),
        (0.78, "#48c0e0"),
        (0.90, "#a8e8f8"),
        (1.00, "#ffffff"),
    ]
    return LinearSegmentedColormap.from_list(
        "boss_lss", [(v, c) for v, c in colors]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms", default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-samples",    type=int, default=10)
    p.add_argument("--fps",          type=int, default=30)
    p.add_argument("--blend-frames", type=int, default=90,
                   help="Frames per crossfade (default 90 → 3 s at 30 fps)")
    p.add_argument("--hold-frames",  type=int, default=30,
                   help="Frames to hold each sample before blending (default 30 → 1 s)")
    p.add_argument("--grid",    type=int, default=800,
                   help="Density map resolution (pixels per side, default 800)")
    p.add_argument("--sigma",   type=float, default=2.5,
                   help="Gaussian smoothing in pixels (default 2.5)")
    p.add_argument("--dec-cut", type=float, default=12.0,
                   help="Half-width of Dec slice for cone diagram (deg, default 12)")
    p.add_argument("--out",     default="output/boss_gp_blend.mp4")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── 1. Load catalog ───────────────────────────────────────────────────
    print("Loading BOSS CMASS-SGC ...")
    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    print(f"  N_data={cat.N_data:,}  N_random={len(cat.ra_random):,}")

    # ── 2. Posterior density field ────────────────────────────────────────
    print("Running posterior sampler ...")
    t0 = time.time()
    result = sample_posterior_density_field(
        cat, n_samples=args.n_samples, n_z_bins=32, nside=64, verbose=True,
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── 3. GP-native catalog samples ──────────────────────────────────────
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"Generating {args.n_samples} GP catalog samples ...")
    t0 = time.time()
    catalogs = result.sample_catalogs_gp(cat, seed=42, w_completeness=w_comp)
    print(f"  Done in {time.time()-t0:.0f}s  "
          f"(N_gal ≈ {np.mean([c['N_galaxies'] for c in catalogs]):.0f})")

    # ── 4. Compute wedge density maps ─────────────────────────────────────
    print("Computing density maps ...")
    fid_cosmo = cat.fid_cosmo
    ra_centre = 361.5   # centre of SGC RA range [335, 388]
    dec_cut   = args.dec_cut

    # Survey boundary in wedge coordinates (for reference lines)
    chi_min = float(chi_of_z(np.array([0.450]), fid_cosmo)[0])
    chi_max = float(chi_of_z(np.array([0.600]), fid_cosmo)[0])
    ra_range_uw = [335.0, 388.0]
    angle_lo = np.radians(ra_range_uw[0] - ra_centre)
    angle_hi = np.radians(ra_range_uw[1] - ra_centre)

    # Grid extent: slightly larger than survey
    pad = 80.0
    x_lo = chi_min * np.cos(max(abs(angle_lo), abs(angle_hi))) - pad
    x_hi = chi_max + pad
    y_lo = chi_max * np.sin(angle_lo) - pad
    y_hi = chi_max * np.sin(angle_hi) + pad

    NX = args.grid
    NY = int(NX * (y_hi - y_lo) / (x_hi - x_lo))
    print(f"  Grid: {NX}×{NY}  Dec cut: |dec| < {dec_cut}°")

    maps = []
    for i, c in enumerate(catalogs):
        ra  = c["ra"]
        dec = c["dec"]
        z   = c["z"]

        # Thin Dec slice for the cone projection
        m = np.abs(dec) < dec_cut
        x, y = to_wedge_xy(ra[m], z[m], fid_cosmo, ra_centre=ra_centre)

        H = density_map(x, y, bins=(NX, NY),
                        x_range=(x_lo, x_hi), y_range=(y_lo, y_hi),
                        sigma_pix=args.sigma)
        maps.append(H)
        print(f"  Sample {i+1}/{args.n_samples}: {m.sum():,} galaxies in Dec slice")

    # Normalise all maps to [0, 1] against the global max for consistent colour
    global_max = max(H.max() for H in maps)
    maps = [H / (global_max + 1e-30) for H in maps]

    # ── 5. Draw survey boundary arc ───────────────────────────────────────
    def boundary_arcs():
        """Return (x, y) arrays for inner and outer arcs + two radial edges."""
        theta = np.linspace(angle_lo, angle_hi, 400)
        arcs = []
        for chi in (chi_min, chi_max):
            arcs.append((chi * np.cos(theta), chi * np.sin(theta)))
        for ang in (angle_lo, angle_hi):
            arcs.append(([chi_min * np.cos(ang), chi_max * np.cos(ang)],
                         [chi_min * np.sin(ang), chi_max * np.sin(ang)]))
        return arcs

    # ── 6. Build frame sequence ───────────────────────────────────────────
    # Each pair (i, i+1 mod N): hold_frames at sample i, then blend_frames
    # crossfade, then hold_frames at sample i+1 (shared with next pair).
    BF = args.blend_frames
    HF = args.hold_frames
    n_s = args.n_samples

    def frame_map(frame_idx):
        """Return blended density map for a given global frame index."""
        cycle = BF + HF
        pos   = frame_idx % (n_s * cycle)
        si    = pos // cycle        # which sample
        fi    = pos %  cycle        # position within that sample's window
        sj    = (si + 1) % n_s
        if fi < HF:
            t = 0.0
        else:
            t = (fi - HF) / BF
        return (1.0 - t) * maps[si] + t * maps[sj], si, sj, t

    total_frames = n_s * (BF + HF)
    duration_s   = total_frames / args.fps
    print(f"\nMovie: {total_frames} frames  {duration_s:.0f}s  @ {args.fps} fps")

    # ── 7. Render ─────────────────────────────────────────────────────────
    cmap   = make_cmap()
    arcs   = boundary_arcs()

    fig = plt.figure(figsize=(16, 9), facecolor="black")
    ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")

    # Seed image
    img_data, si, sj, t = frame_map(0)
    im = ax.imshow(
        img_data, origin="lower", cmap=cmap, vmin=0, vmax=1,
        extent=[x_lo, x_hi, y_lo, y_hi],
        interpolation="bilinear",
        aspect="auto",
    )

    # Survey boundary lines
    arc_lines = []
    for xv, yv in arcs:
        ln, = ax.plot(xv, yv, color="white", lw=0.6, alpha=0.35)
        arc_lines.append(ln)

    # Redshift tick arcs
    for z_tick in np.arange(0.46, 0.60, 0.02):
        chi_t = float(chi_of_z(np.array([z_tick]), fid_cosmo)[0])
        theta = np.linspace(angle_lo, angle_hi, 200)
        ax.plot(chi_t * np.cos(theta), chi_t * np.sin(theta),
                color="white", lw=0.3, alpha=0.18, zorder=2)
        # Label at midpoint
        ang_mid = 0.5 * (angle_lo + angle_hi)
        ax.text(chi_t * np.cos(ang_mid), chi_t * np.sin(ang_mid) - 15,
                f"z={z_tick:.2f}", color="white", fontsize=7, alpha=0.5,
                ha="center", va="top")

    # Title / sample counter
    title_txt = ax.text(
        0.50, 0.97, "", transform=ax.transAxes,
        color="white", fontsize=14, ha="center", va="top",
        fontfamily="monospace",
    )
    info_txt = ax.text(
        0.50, 0.03,
        "BOSS CMASS-SGC  |  GP posterior samples  |  Dec slice ±{:.0f}°".format(dec_cut),
        transform=ax.transAxes,
        color="white", fontsize=9, alpha=0.6, ha="center", va="bottom",
    )

    def update(frame_idx):
        blended, si, sj, t = frame_map(frame_idx)
        im.set_data(blended)
        if t < 0.05:
            label = f"Sample {si+1:02d} / {n_s}"
        elif t > 0.95:
            label = f"Sample {sj+1:02d} / {n_s}"
        else:
            label = f"Sample {si+1:02d} → {sj+1:02d} / {n_s}  ({t:.0%})"
        title_txt.set_text(label)
        return [im, title_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / args.fps, blit=True,
    )

    print(f"Writing {args.out} ...")
    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p",
                    "-profile:v", "main",
                    "-level", "4.1",
                    "-crf", "18",
                    "-movflags", "+faststart"],
    )
    t0 = time.time()
    ani.save(args.out, writer=writer, dpi=120)
    print(f"Done in {time.time()-t0:.0f}s  →  {args.out}")


if __name__ == "__main__":
    main()
