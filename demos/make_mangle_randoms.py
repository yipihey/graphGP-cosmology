"""Generate uniform-geometric randoms from the BOSS DR12 mangle mask (one-time).

Run with the Anaconda Python (which has Python.h so pymangle builds):
    ~/.local/share/anaconda3/bin/python3 -m pip install pymangle
    ~/.local/share/anaconda3/bin/python3 demos/make_mangle_randoms.py

Mask (18 MB, 33072 polygons):
    https://data.sdss.org/sas/dr12/boss/lss/mask_DR12v5_CMASS_South.ply

What we learned using it (documented here so the result is not re-litigated):
  * The polygon "weight" IS the spectroscopic completeness COMP. At the galaxies
    COMP = 0.987 (5-95%: 0.95-1.0) -- CMASS-South is ~99% complete, so there is
    essentially NO angular completeness window to remove: the survey random is
    already a uniform-footprint random to ~1%.
  * genrand is area-uniform over ALL polygons including the large weight==0
    masked-out regions; keep weight>0 for the surveyed area. The weight>0 area is
    99.9% at completeness>0.9.
  * IMPORTANT: mask_DR12v5 is the GEOMETRY mask and is ~40% larger (in area) than
    the LSS clustering footprint (it includes chunks/regions excluded from the LSS
    sample). Uniform randoms over the raw mask spill ~830 deg^2 outside the data
    and inflate w(theta) by up to ~50x. Clip to the LSS footprint (proximity to the
    survey random catalogue) before use -- see validate_dropin_uniform_randoms.py.

Conclusion: the equal-weight completed catalogue reproduces the official weighted
survey to ~1.5% in w(theta) and ~1-2% in wp(rp)/xi0 USING THE SURVEY RANDOM, which
for CMASS (COMP~0.99) IS the uniform-footprint window. A separately-constructed
uniform random matches only to the precision of its footprint-boundary fit.
"""
import os
import numpy as np
import pymangle

PLY = "data/boss/mask_DR12v5_CMASS_South.ply"
OUT = "data/boss/mangle_uniform_radec.npy"
N_GEN = 3_000_000


def main():
    m = pymangle.Mangle(PLY)
    print(f"polygons: {m.npoly}")
    ra, dec = m.genrand(N_GEN)                 # area-uniform over the mask
    w = m.weight(ra, dec)
    keep = w > 0                               # the surveyed (completeness>0) area
    ra, dec = ra[keep], dec[keep]
    np.save(OUT, np.column_stack([ra, dec]).astype("f4"))
    raw = ((ra + 180) % 360) - 180
    print(f"kept {keep.sum():,} weight>0 of {N_GEN:,}; "
          f"RA(wrapped) [{raw.min():.1f},{raw.max():.1f}] Dec [{dec.min():.1f},{dec.max():.1f}]")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
