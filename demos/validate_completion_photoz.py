"""Validate the photo-z-informed completion (placeholder targets).

End-to-end: load BOSS with photometry, train the k-NN photo-z on the good-spec
sample, build the (placeholder) missing-target set, run complete_catalog_photoz
over several realizations, and check the equal-weight completed ξ(Δθ,Δz=0)
against the w_c-weighted observed clustering + the realization scatter.

NOTE: the placeholder gives missing targets the HOST's colours, so the photo-z
cannot yet discriminate true pairs from projections — this run tests the plumbing
(positions at known sites, z sampled from photo-z×clustering prior, w_systot
Poisson). The real fetched targets (real colours) are needed for the science.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/validate_completion_photoz.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import (measure_K2d, compute_rr, complete_catalog_photoz,
                                       measure_close_pair_dz)
from twopt_density import perf

COLL = 62.0 / 3600.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--n-rand-factor", type=int, default=2)
    p.add_argument("--targets", default=None,
                   help="fetched CMASS target FITS (real loader); default = placeholder")
    args = p.parse_args()

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    print(f"N_obs={cat.N_data:,}  <w_c>={w_c.mean():.4f}")

    # train photo-z on good-spec galaxies with reliable photometry
    feat = photoz_features(cat.colors_data, cat.mags_data)
    train = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[train], np.asarray(cat.z_data)[train])
    dz_pool = measure_close_pair_dz(cat, COLL)
    targets = load_cmass_targets(cat, path=args.targets, seed=0)
    wcp = np.asarray(cat.w_cp_data); wnoz = np.asarray(cat.w_noz_data)
    print(f"{'REAL' if args.targets else 'placeholder'} missing targets: {targets.N:,} "
          f"(collided {np.sum(targets.miss_kind=='collided'):,} "
          f"[w_cp implies {(wcp-1).sum():.0f}], "
          f"zfail {np.sum(targets.miss_kind=='zfail'):,} "
          f"[w_noz implies {(wnoz-1).sum():.0f}])")

    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 18)]); ze = np.linspace(0.0, 0.03, 11)
    tcen = np.empty(len(te) - 1); tcen[0] = 0.5 * te[1]; tcen[1:] = np.sqrt(te[1:-1] * te[2:])

    rng = np.random.default_rng(0)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_factor * cat.N_data,
        z_data=np.asarray(cat.z_data), nside=cat.nside, rng=rng)
    wr = np.ones(len(rar))

    rr_full = compute_rr(rar, decr, zr, wr, theta_edges=te, z_edges=ze)  # randoms fixed
    _, _, xw = measure_K2d(cat.ra_data, cat.dec_data, cat.z_data, w_c,
                           rar, decr, zr, wr, theta_edges=te, z_edges=ze, precomp_rr=rr_full)
    X = []
    for s in range(args.n_real):
        c = complete_catalog_photoz(cat, targets, pz, seed=s, dz_pool=dz_pool,
                                    verbose=(s == 0))
        _, _, xe = measure_K2d(c["ra"], c["dec"], c["z"], np.ones(c["N"]),
                               rar, decr, zr, wr, theta_edges=te, z_edges=ze, precomp_rr=rr_full)
        X.append(xe)
    X = np.array(X); xm = X.mean(0)[:, 0]; xs = X.std(0)[:, 0]; xw0 = xw[:, 0]

    print(f"\nequal-weight (photo-z completed) ξ(Δθ,0) / w_c-weighted "
          f"(* = below {COLL:.3f}° collision scale):")
    print(f"{'theta':>8}{'xi_wgt':>10}{'xi_eq':>10}{'eq/wgt':>9}{'scat%':>7}")
    for i in range(len(tcen)):
        f = "*" if tcen[i] < COLL else " "
        print(f"{tcen[i]:8.4f}{xw0[i]:10.4f}{xm[i]:10.4f}"
              f"{xm[i]/xw0[i] if xw0[i] else np.nan:9.3f}"
              f"{100*xs[i]/xm[i] if xm[i] else np.nan:7.1f} {f}")

    # --- imaging-consistency checks (one realization) ---
    import healpy as hp
    c = complete_catalog_photoz(cat, targets, pz, seed=0, dz_pool=dz_pool)
    print("\nn(z): completed vs observed (fraction per z-bin)")
    zb = np.linspace(0.43, 0.62, 8)
    ho, _ = np.histogram(np.asarray(cat.z_data), zb, weights=w_c)  # weighted observed
    hc, _ = np.histogram(c["z"], zb)
    for a, b, fo, fc in zip(zb[:-1], zb[1:], ho / ho.sum(), hc / hc.sum()):
        print(f"  z[{a:.2f},{b:.2f}): wobs={fo:.3f}  completed={fc:.3f}")
    # angular density per HEALPix pixel: completed vs (observed+missing targets)
    ns = 32
    pix_c = hp.ang2pix(ns, c["ra"], c["dec"], lonlat=True)
    dens_c = np.bincount(pix_c, minlength=12 * ns**2).astype(float)
    occ = dens_c > 0
    pix_t = hp.ang2pix(ns, np.r_[np.asarray(cat.ra_data), np.asarray(targets.ra)],
                       np.r_[np.asarray(cat.dec_data), np.asarray(targets.dec)], lonlat=True)
    dens_t = np.bincount(pix_t, minlength=12 * ns**2).astype(float)
    r = np.corrcoef(dens_c[occ], dens_t[occ])[0, 1]
    print(f"\nangular density per nside={ns} pixel: corr(completed, observed+targets)={r:.3f}")
    perf.report("validate_completion_photoz")


if __name__ == "__main__":
    main()
