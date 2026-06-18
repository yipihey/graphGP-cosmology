"""Phase 1 — sensitivity of the completed clustering to the pipeline's choices.

Each completion choice is varied around the default and the impact on the
clustering is reported, so every heuristic is either justified (result is robust)
or flagged. We measure the completed-ensemble wp(rp) (Corrfunc) and w(θ), and
report the fractional deviation from the default configuration. Varied:
  * z_mode: field (default) / nn / photoz
  * count:  round (default) / poisson
  * photo-z k:  50 / 100 / 150
  * close-pair / collision scale used for the Δz prior:  40 / 62 / 90 arcsec

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/audit_sensitivity.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import wp_rp

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=4)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)
    rng = np.random.default_rng(7)
    rar, decr, zr = make_random_from_selection_function(sel_map=cat.sel_map, n_random=3*cat.N_data,
                                                        z_data=z, nside=cat.nside, rng=rng)
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    RR = [None]

    def measure(z_mode="field", count="round", k=100, coll=62.0):
        pz = PhotoZKNN(k=k).fit(feat[good], z[good])
        dz = measure_close_pair_dz(cat, coll/3600.)
        W = []
        for s in range(args.n_real):
            c = complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz, z_mode=z_mode, count=count)
            wp = wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]), rar, decr, zr,
                       rp_edges=rp_edges, pimax=40., nthreads=32, precomp_RR=RR[0], return_RR=(RR[0] is None))
            if RR[0] is None:
                wp, RR[0] = wp
            W.append(wp)
        return np.mean(W, 0)

    base = measure()
    print(f"baseline wp(rp): {np.round(base, 1)}")
    variants = [("z_mode=nn", dict(z_mode="nn")), ("z_mode=photoz", dict(z_mode="photoz")),
                ("count=poisson", dict(count="poisson")), ("k=50", dict(k=50)), ("k=150", dict(k=150)),
                ("coll=40\"", dict(coll=40.)), ("coll=90\"", dict(coll=90.))]
    print(f"\n{'variant':16}{'med|Δwp/wp|':>14}{'max|Δwp/wp|':>14}  (deviation from default)")
    for name, kw in variants:
        wp = measure(**kw); d = np.abs(wp/base - 1)
        print(f"{name:16}{np.median(d):14.3f}{np.max(d):14.3f}")
    print("\n(small deviations => the default is robust to that choice; large => the choice matters)")


if __name__ == "__main__":
    main()
