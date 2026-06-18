"""Quick diagnostic: does a finer binary footprint make the OFFICIAL w_c-weighted
w(theta) agree with a uniform-footprint random? Official data is unbiased, so any
deviation of off/uniform from off/survey-random is a window-construction artifact.
If it shrinks with nside -> it's pixel/geometry resolution; if it plateaus -> the
limitation is the true selection window (sector completeness), not resolvable from
a binary mask."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import healpy as hp
from twopt_density.boss import load_boss
from twopt_density.inpaint import fine_completeness_map, find_interior_holes
from demos.validate_dropin_uniform_randoms import wtheta, make_uniform_window

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"

cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
wc = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
rar_full = np.asarray(cat.ra_random); decr_full = np.asarray(cat.dec_random)
rng = np.random.default_rng(1)

tb = np.logspace(np.log10(0.05), np.log10(2.5), 13); tc = np.sqrt(tb[1:]*tb[:-1])
ro = rng.choice(len(rar_full), 4*cat.N_data, replace=False)
w_srv, rr_srv = wtheta(ra, dec, rar_full[ro], decr_full[ro], tb, w_d=wc, return_rr=True)

print(f"{'theta':>8}", end="")
for ns in [256, 512, 1024, 2048]:
    print(f"{('off/unif n'+str(ns)):>16}", end="")
print()

ratios = {}
for ns in [256, 512, 1024, 2048]:
    counts, _ = fine_completeness_map(rar_full, decr_full, nside=ns)
    holes = find_interior_holes(counts, ns, empty_count=0.0, min_neighbour_frac=0.75)
    fp = np.where(counts > 0)[0]
    if holes:
        fp = np.union1d(fp, np.concatenate([h.pixels for h in holes]))
    wr, dr, _ = make_uniform_window(fp, ns, 8*cat.N_data, z, rng)
    rr_u = wtheta(wr, dr, wr, dr, tb, return_rr=True)[1]
    w_u = wtheta(ra, dec, wr, dr, tb, w_d=wc, rr=rr_u)
    ratios[ns] = w_u / w_srv
    area = len(fp) * hp.nside2pixarea(ns, degrees=True)
    print(f"[n={ns}] footprint {area:.0f} deg^2, median off/unif = {np.median(ratios[ns]):.3f}")

print(f"\n{'theta':>8}" + "".join(f"{('n'+str(ns)):>10}" for ns in [256,512,1024,2048]))
for i in range(len(tc)):
    print(f"{tc[i]:8.3f}" + "".join(f"{ratios[ns][i]:10.3f}" for ns in [256,512,1024,2048]))
