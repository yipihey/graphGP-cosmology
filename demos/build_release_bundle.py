"""Build the shareable release bundle into docs/data/ (small enough for GitHub +
served by GitHub Pages; the report HTML links to these files for download).

Writes:
  docs/data/cmass_south_posterior.npz   the compact posterior (observed base once +
                                        per-missing inverse-CDF); draw unlimited samples
  docs/data/cmass_south_randoms.npz     uniform-footprint randoms (downsampled survey
                                        randoms; CMASS COMP~0.99 so they are uniform to ~1%)
  docs/data/draw_samples.py             standalone numpy-only sampler (shipped separately)
  docs/data/README.md                   quickstart

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/build_release_bundle.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import measure_close_pair_dz
from twopt_density import posterior_sampler as PS

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
OUT = "docs/data"
N_RAND_MULT = 4          # randoms = 4x N_data (uniform-footprint; users can draw more)


def main():
    os.makedirs(OUT, exist_ok=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    # ---- posterior package ----
    pkg = PS.build_package(cat, tg, pz, dz_pool=dz, verbose=True)
    ppath = os.path.join(OUT, "cmass_south_posterior.npz")
    PS.write_package(pkg, ppath)
    psz = os.path.getsize(ppath)

    # ---- uniform-footprint randoms (downsampled survey randoms; COMP~0.99 => uniform) ----
    rng = np.random.default_rng(0)
    nr = min(N_RAND_MULT * cat.N_data, len(cat.ra_random))
    idx = rng.choice(len(cat.ra_random), nr, replace=False)
    rpath = os.path.join(OUT, "cmass_south_randoms.npz")
    np.savez_compressed(rpath,
                        ra=np.asarray(cat.ra_random)[idx].astype(np.float32),
                        dec=np.asarray(cat.dec_random)[idx].astype(np.float32),
                        z=np.asarray(cat.z_random)[idx].astype(np.float32))
    rsz = os.path.getsize(rpath)

    # ---- README ----
    readme = f"""# BOSS CMASS-South completed-catalog posterior (release bundle)

Equal-weight, cosmology-free completed catalogs of BOSS DR12 CMASS-South. Every
realization keeps the {pkg['n_obs']:,} observed galaxies (fixed) and adds the
{pkg['n_miss']:,} spectroscopically-missing galaxies (fiber collisions + redshift
failures) at their real imaging positions with a GP/local-density redshift. The
posterior is stored compactly so you draw as many samples as you like locally.

## Files
- `cmass_south_posterior.npz`  ({psz/1e6:.2f} MB) the posterior (observed base once +
  each missing galaxy's redshift inverse-CDF). 1 file = the whole ensemble.
- `cmass_south_randoms.npz`     ({rsz/1e6:.2f} MB) uniform-footprint randoms (RA, DEC, Z),
  {nr:,} points. CMASS-South is ~99% complete (COMP~0.99) so these are uniform to ~1%.
- `draw_samples.py`             standalone numpy-only sampler.

## Quickstart
```bash
pip install numpy astropy
python draw_samples.py --seed 0 --out catalog_0.fits        # one realization
python draw_samples.py --seed 0 --n 100 --out-prefix cat_   # 100 realizations
```
```python
from draw_samples import load_package, draw
pkg = load_package("cmass_south_posterior.npz")
cat = draw(pkg, seed=0)            # dict(ra, dec, z, prov, N); ~120k galaxies
```
A fixed, reproducible ensemble of K catalogs is just K seeds (0..K-1) — no need to
store K copies (the observed galaxies are shared). Pair the catalogs with
`cmass_south_randoms.npz` and use equal weights (no completeness weights needed):
the completion reproduces the official w_c-weighted clustering to ~1-2%.

## Columns
RA, DEC [deg]; Z (redshift); PROV: 0 observed-specz, 1 fiber-collided,
2 redshift-failure, 3 systot-analog, 4 zhost-fallback.

See DATA_MODEL.md (repository root) for full conventions, scope, and the
systematics budget.
"""
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(readme)

    total = psz + rsz + os.path.getsize(os.path.join(OUT, "draw_samples.py"))
    print(f"\nwrote release bundle to {OUT}/:")
    print(f"  cmass_south_posterior.npz  {psz/1e6:6.2f} MB")
    print(f"  cmass_south_randoms.npz    {rsz/1e6:6.2f} MB  ({nr:,} randoms)")
    print(f"  draw_samples.py + README.md")
    print(f"  total ~ {total/1e6:.2f} MB  (fits in the repo + GitHub Pages; <100 MB/file, <1 GB/repo)")


if __name__ == "__main__":
    main()
