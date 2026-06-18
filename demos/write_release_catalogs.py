"""Write the distributable completion catalogue release (ensemble + summary).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/write_release_catalogs.py \
        --n-real 20 --out output/release
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.catalog_io import write_release

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=20)
    p.add_argument("--n-rand-factor", type=int, default=3)
    p.add_argument("--out", default="output/release")
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)
    host = np.asarray(tg.host_index)
    wsys = np.asarray(cat.w_sys_data)
    w_systot_prefix = np.concatenate([wsys, wsys[np.clip(host, 0, len(z)-1)]])   # observed + missing

    print(f"generating {args.n_real} realizations ...", flush=True)
    cats = [complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz, verbose=(s == 0))
            for s in range(args.n_real)]

    rng = np.random.default_rng(7)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_factor*cat.N_data, z_data=z, nside=cat.nside, rng=rng)

    meta = dict(ZRANGE="0.43<z<0.7", ZMODE="field (GP/local-density LOS)",
                SYSTOT="analog (keep-all + excess)", PIMAXOK="angular+3D wp recovered to ~1-2%")
    write_release(cats, args.out, w_systot=w_systot_prefix, randoms=(rar, decr, zr), meta=meta)
    print(f"\nwrote release to {args.out}/ (ensemble/ + summary.fits + randoms.fits + PROVENANCE.txt)")


if __name__ == "__main__":
    main()
