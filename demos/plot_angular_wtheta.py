"""Angular two-point function w(θ) for analytic-window posterior catalogs.

Catalog realizations are drawn from the **analytic survey window** — no MC
random catalog is instantiated for the density estimate; the expected random
density is ρ̂_W ∝ S_ang·n(z)/χ² evaluated directly at each point, and the
overdensity uses an adaptive-bandwidth FKP-KDE (twopt_density.window +
density_field.sample_catalogs_analytic_window).

w(θ) is measured with Corrfunc's ``DDtheta_mocks`` — the exact Landy-Szalay
angular estimator from (RA, Dec) pair counts, no Limber approximation and no
cascade cell-binning (the morton_cascade dyadic-cell ξ is a *different*
observable; see demos/validate_cascade_vs_corrfunc.py):

    w(θ) = (DD − 2 DR + RR) / RR        (Landy-Szalay)

The observed catalog is completeness-weighted (w_sys·w_noz·w_cp); the
realizations encode completeness through the analytic-window thinning.

Usage::

    python demos/plot_angular_wtheta.py [--n-samples 10] [--n-rand-rr 150000]
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

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

from twopt_density.boss import load_boss
from twopt_density.window import build_survey_window
from twopt_density.density_field import sample_catalogs_lgcp


# ── Landy-Szalay angular estimator via Corrfunc ──────────────────────────────

def _theta_counts(ra1, dec1, theta_bins, nthreads, ra2=None, dec2=None,
                  w1=None, w2=None):
    """Weighted angular pair counts in θ bins (degrees). Returns weighted
    pair sums per bin (npairs × weightavg)."""
    autocorr = 1 if ra2 is None else 0
    # DDtheta_mocks(autocorr, nthreads, binfile, RA1, DEC1, **kwargs)
    args = (autocorr, nthreads, theta_bins, ra1, dec1)
    extra = {}
    if ra2 is not None:
        extra.update(RA2=ra2, DEC2=dec2)
    if w1 is not None:
        extra.update(weights1=w1, weight_type="pair_product")
    if w2 is not None:
        extra.update(weights2=w2, weight_type="pair_product")
    res = DDtheta_mocks(*args, **extra)
    npairs = res["npairs"].astype(np.float64)
    if w1 is not None:
        wsum = npairs * res["weightavg"].astype(np.float64)
        return wsum
    return npairs


def wtheta_ls(ra_d, dec_d, ra_r, dec_r, theta_bins, nthreads=16,
              w_d=None, RR=None, sumw_r=None):
    """Landy-Szalay w(θ). RR (weighted-normalised) may be precomputed and
    reused across catalogs that share the same random subsample."""
    nd_w = float(w_d.sum()) if w_d is not None else float(len(ra_d))
    nr_w = sumw_r if sumw_r is not None else float(len(ra_r))

    DD = _theta_counts(ra_d, dec_d, theta_bins, nthreads, w1=w_d, w2=w_d)
    DR = _theta_counts(ra_d, dec_d, theta_bins, nthreads,
                       ra2=ra_r, dec2=dec_r, w1=w_d,
                       w2=np.ones(len(ra_r)))
    if RR is None:
        RR = _theta_counts(ra_r, dec_r, theta_bins, nthreads,
                           w1=np.ones(len(ra_r)), w2=np.ones(len(ra_r)))

    dd = DD / (nd_w * nd_w)
    dr = DR / (nd_w * nr_w)
    rr = RR / (nr_w * nr_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(rr > 0, (dd - 2.0 * dr + rr) / rr, np.nan)
    return w, RR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms",    default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-samples",  type=int,   default=10)
    p.add_argument("--n-rand-rr",  type=int,   default=150_000,
                   help="Randoms subsample for RR/DR (default 150k)")
    p.add_argument("--theta-min",  type=float, default=0.05)
    p.add_argument("--theta-max",  type=float, default=8.0)
    p.add_argument("--n-theta",    type=int,   default=16)
    p.add_argument("--nthreads",   type=int,   default=16)
    p.add_argument("--out",        default="output/boss_wtheta_samples.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    theta_bins = np.logspace(np.log10(args.theta_min),
                             np.log10(args.theta_max), args.n_theta + 1)
    theta_cen = np.sqrt(theta_bins[:-1] * theta_bins[1:])

    # ── 1. Load catalog ───────────────────────────────────────────────────
    print("Loading BOSS CMASS-SGC ...")
    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    print(f"  N_data={cat.N_data:,}  N_random={len(cat.ra_random):,}")

    # ── 2. graphGP LGCP posterior-predictive catalogs ─────────────────────
    # Log-Gaussian Cox process: Gaussian field with covariance ln(1+ξ),
    # log-normal intensity, drawn on millions of window candidates via the
    # chunked-refinement graphGP fork (GPU, memory-bounded), Poisson-thinned.
    # Reproduces the observed ξ(r)/w(θ) across measured scales by construction.
    print("Drawing graphGP-LGCP realizations ...")
    t0 = time.time()
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    window = build_survey_window(cat, kde_bandwidth=0.02)
    catalogs = sample_catalogs_lgcp(
        cat, window, n_samples=args.n_samples, seed=42,
        w_completeness=w_comp, n_cand_factor=20, chunk_size=50_000,
        nthreads=args.nthreads, verbose=True,
    )
    print(f"  {args.n_samples} LGCP realizations in {time.time()-t0:.0f}s")

    # ── 3. Shared random subsample for RR / DR ────────────────────────────
    rng = np.random.default_rng(999)
    Nr = len(cat.ra_random)
    nsub = min(args.n_rand_rr, Nr)
    isub = rng.choice(Nr, nsub, replace=False)
    ra_r = np.ascontiguousarray(np.asarray(cat.ra_random)[isub], dtype=np.float64)
    dec_r = np.ascontiguousarray(np.asarray(cat.dec_random)[isub], dtype=np.float64)
    sumw_r = float(nsub)
    print(f"  RR/DR randoms: {nsub:,}")

    # ── 4. w(θ) for observed catalog (completeness-weighted) ──────────────
    print("Measuring w(θ): observed catalog ...")
    t0 = time.time()
    ra_d = np.ascontiguousarray(np.asarray(cat.ra_data), dtype=np.float64)
    dec_d = np.ascontiguousarray(np.asarray(cat.dec_data), dtype=np.float64)
    w_d = np.ascontiguousarray(np.asarray(w_comp), dtype=np.float64)
    w_orig, RR = wtheta_ls(ra_d, dec_d, ra_r, dec_r, theta_bins,
                           nthreads=args.nthreads, w_d=w_d, sumw_r=sumw_r)
    print(f"    {time.time()-t0:.1f}s  (RR cached for reuse)")

    # ── 5. w(θ) for each GP sample ────────────────────────────────────────
    w_samples = []
    for i, c in enumerate(catalogs):
        t0 = time.time()
        ra_s = np.ascontiguousarray(np.asarray(c["ra"]), dtype=np.float64)
        dec_s = np.ascontiguousarray(np.asarray(c["dec"]), dtype=np.float64)
        w_i, _ = wtheta_ls(ra_s, dec_s, ra_r, dec_r, theta_bins,
                           nthreads=args.nthreads, w_d=None, RR=RR,
                           sumw_r=sumw_r)
        w_samples.append(w_i)
        print(f"    sample {i+1:2d}/{args.n_samples}: {time.time()-t0:.1f}s  "
              f"N_gal={c['N_galaxies']:,}")
    w_samples = np.array(w_samples)

    # ── 6. Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    w_med = np.nanmedian(w_samples, axis=0)
    w_lo  = np.nanpercentile(w_samples, 16, axis=0)
    w_hi  = np.nanpercentile(w_samples, 84, axis=0)
    ax.fill_between(theta_cen, w_lo, w_hi, color="#4a90d9", alpha=0.25,
                    label="realizations 16–84%")

    colors = plt.cm.cool(np.linspace(0.1, 0.9, args.n_samples))
    for i, (w_i, col) in enumerate(zip(w_samples, colors)):
        ax.plot(theta_cen, w_i, color=col, lw=1.0, alpha=0.7,
                label="graphGP-LGCP realizations" if i == 0 else None)

    ax.plot(theta_cen, w_orig, color="white", lw=2.6, zorder=10)
    ax.plot(theta_cen, w_orig, color="#f5a623", lw=1.8, zorder=11,
            label="BOSS CMASS-SGC (observed, LS)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [degrees]", fontsize=13)
    ax.set_ylabel(r"$w(\theta)$", fontsize=13)
    ax.set_title("BOSS CMASS-SGC — Landy-Szalay angular two-point function\n"
                 r"Corrfunc DDtheta_mocks · graphGP-LGCP posterior realizations",
                 fontsize=12)
    ax.set_xlim(theta_bins[0], theta_bins[-1])

    ax.set_facecolor("#0a0a12")
    fig.patch.set_facecolor("#0a0a12")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, which="both", alpha=0.12, color="white")
    ax.legend(fontsize=10, framealpha=0.2, labelcolor="white", facecolor="#111")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {args.out}")

    # ── 7. Summary ────────────────────────────────────────────────────────
    print(f"\nLandy-Szalay w(θ)  [Corrfunc DDtheta_mocks]:")
    print(f"  {'θ[°]':>8}  {'w_obs':>9}  {'w_med_GP':>9}  {'GP/obs':>7}")
    for i, tc in enumerate(theta_cen):
        ratio = w_med[i] / w_orig[i] if w_orig[i] != 0 else np.nan
        print(f"  {tc:8.3f}  {w_orig[i]:9.5f}  {w_med[i]:9.5f}  {ratio:7.3f}")


if __name__ == "__main__":
    main()
