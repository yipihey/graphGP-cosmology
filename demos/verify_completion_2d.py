"""Deeper verification of the photo-z completion: the FULL 2-D ξ(Δθ,Δz) plane
and per-redshift-slice angular clustering (not just the Δz=0 column).

Equal-weight completed (real CMASS targets + photo-z) vs the w_c-weighted
observed, across the whole (Δθ,Δz) plane and in per-z slices — the AP signal
lives in the 2-D anisotropy, so this is the test that the completion preserves
the cosmological geometry.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/verify_completion_2d.py \
        --targets data/boss/cmass_targets_South.fits
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import (measure_K2d, compute_rr, complete_catalog_photoz,
                                       measure_close_pair_dz)
from twopt_density import perf

COLL = 62.0 / 3600.0


def xi_grid(ra_d, dec_d, z_d, w_d, ra_r, dec_r, z_r, te, ze, md=None, mr=None, precomp_rr=None):
    if md is not None:
        ra_d, dec_d, z_d, w_d = ra_d[md], dec_d[md], z_d[md], w_d[md]
        ra_r, dec_r, z_r = ra_r[mr], dec_r[mr], z_r[mr]
    return measure_K2d(ra_d, dec_d, z_d, w_d, ra_r, dec_r, z_r, np.ones(len(ra_r)),
                       theta_edges=te, z_edges=ze, precomp_rr=precomp_rr)[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/boss/cmass_targets_South.fits")
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--n-rand-factor", type=int, default=2)
    p.add_argument("--out", default="output/completion_2d_verify.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)

    feat = photoz_features(cat.colors_data, cat.mags_data)
    tr = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[tr], z_d[tr])
    dz_pool = measure_close_pair_dz(cat, COLL)
    targets = load_cmass_targets(cat, path=args.targets, seed=0)

    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 16)]); ze = np.linspace(0.0, 0.03, 9)
    tcen = np.empty(len(te) - 1); tcen[0] = 0.5 * te[1]; tcen[1:] = np.sqrt(te[1:-1] * te[2:])
    zcen = 0.5 * (ze[1:] + ze[:-1])

    rng = np.random.default_rng(0)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_factor * cat.N_data,
        z_data=z_d, nside=cat.nside, rng=rng)

    # generate completions ONCE; reuse for full-plane and every per-z slice below
    cats = [complete_catalog_photoz(cat, targets, pz, seed=s, dz_pool=dz_pool)
            for s in range(args.n_real)]
    rr_full = compute_rr(rar, decr, zr, np.ones(len(rar)), theta_edges=te, z_edges=ze)
    xw = xi_grid(ra_d, dec_d, z_d, w_c, rar, decr, zr, te, ze, precomp_rr=rr_full)   # (nθ,nz)
    Xc = [xi_grid(c["ra"], c["dec"], c["z"], np.ones(c["N"]), rar, decr, zr, te, ze, precomp_rr=rr_full)
          for c in cats]
    xc = np.mean(Xc, 0)

    # full-plane ratio where the signal is measurable
    ratio = np.where(xw > 0.02, xc / xw, np.nan)
    res = tcen > COLL
    print("2-D ξ(Δθ,Δz) closure  completed/weighted (median over resolved Δθ, per Δz):")
    for j in range(len(zcen)):
        col = ratio[res, j]
        print(f"  Δz={zcen[j]:.4f}: median={np.nanmedian(col):.3f}  "
              f"(n={np.sum(np.isfinite(col))})")

    # per-z-slice angular closure
    zedges = np.quantile(z_d, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("\nper-z-slice angular ξ(Δθ,0) closure (median over resolved Δθ):")
    slice_curves = []
    for a, b in zip(zedges[:-1], zedges[1:]):
        md = (z_d >= a) & (z_d < b); mr = (zr >= a) & (zr < b)
        rr_sl = compute_rr(rar[mr], decr[mr], zr[mr], np.ones(int(mr.sum())),
                           theta_edges=te, z_edges=ze)
        xw_s = xi_grid(ra_d, dec_d, z_d, w_c, rar, decr, zr, te, ze, md, mr, precomp_rr=rr_sl)[:, 0]
        # measure per realization restricting to the slice (cached completions)
        xcs = []
        for c in cats:
            mc = (c["z"] >= a) & (c["z"] < b)
            xcs.append(xi_grid(c["ra"], c["dec"], c["z"], np.ones(c["N"]), rar, decr, zr, te, ze, mc, mr, precomp_rr=rr_sl)[:, 0])
        rr = np.mean(xcs, 0) / xw_s
        slice_curves.append((a, b, rr))
        print(f"  z∈[{a:.2f},{b:.2f}): median={np.nanmedian(rr[res]):.3f}  N={md.sum():,}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    im = a1.pcolormesh(zcen, tcen, ratio, vmin=0.8, vmax=1.05, cmap="RdBu_r", shading="nearest")
    a1.set_yscale("log"); a1.set_ylabel(r"$\Delta\theta$ [deg]"); a1.set_xlabel(r"$\Delta z$")
    a1.set_title("completed/weighted  ξ(Δθ,Δz)"); plt.colorbar(im, ax=a1, fraction=0.046)
    a1.axhline(COLL, color="k", ls=":")
    for a, b, rr in slice_curves:
        a2.semilogx(tcen, rr, "o-", ms=3, label=f"z∈[{a:.2f},{b:.2f})")
    a2.axhline(1, color="gray", ls="--"); a2.axvline(COLL, color="gray", ls=":")
    a2.fill_between(tcen, 0.95, 1.05, color="green", alpha=0.12)
    a2.set_ylim(0.8, 1.15); a2.set_xlabel(r"$\Delta\theta$ [deg]")
    a2.set_ylabel("completed/weighted"); a2.legend(fontsize=8); a2.set_title("per-z-slice angular")
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    perf.report("verify_completion_2d")


if __name__ == "__main__":
    main()
