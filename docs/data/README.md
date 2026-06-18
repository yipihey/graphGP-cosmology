# BOSS CMASS-South completed-catalog posterior (release bundle)

Equal-weight, cosmology-free completed catalogs of BOSS DR12 CMASS-South. Every
realization keeps the 109,636 observed galaxies (fixed) and adds the
6,777 spectroscopically-missing galaxies (fiber collisions + redshift
failures) at their real imaging positions with a GP/local-density redshift. The
posterior is stored compactly so you draw as many samples as you like locally.

## Files
- `cmass_south_posterior.npz`  (2.04 MB) the posterior (observed base once +
  each missing galaxy's redshift inverse-CDF). 1 file = the whole ensemble.
- `cmass_south_randoms.npz`     (4.63 MB) uniform-footprint randoms (RA, DEC, Z),
  438,544 points. CMASS-South is ~99% complete (COMP~0.99) so these are uniform to ~1%.
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
