"""Build the compact shareable posterior package + validate the fast sampler.

Builds the posterior once, writes the heavily-compressed package, then:
  * reports the on-disk size (and the "naive ensemble" size it replaces),
  * times how many samples/sec the sampler draws,
  * checks the sampler reproduces complete_catalog_photoz(z_mode='field') in n(z)
    and angular w(theta) (the sampler is a fast re-implementation of the same
    posterior, so the ensembles must agree).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/build_posterior_package.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density import posterior_sampler as PS

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
PKG = "output/release/cmass_south_posterior.npz"


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, rr=None):
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    f8 = lambda a: np.ascontiguousarray(a, "f8"); nd, nr = len(ra_d), len(ra_r)
    if rr is None:
        rr = DDtheta_mocks(1, 32, tb, f8(ra_r), f8(dec_r))["npairs"].astype(float)
    DD = DDtheta_mocks(1, 32, tb, f8(ra_d), f8(dec_d))["npairs"].astype(float) / (nd*(nd-1.))
    dr = DDtheta_mocks(0, 32, tb, f8(ra_d), f8(dec_d), RA2=f8(ra_r), DEC2=f8(dec_r))["npairs"].astype(float)
    RR = rr/(nr*(nr-1.))
    return np.where(RR>0, (DD - 2*dr/(nd*nr) + RR)/RR, np.nan), rr


def main():
    os.makedirs(os.path.dirname(PKG), exist_ok=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    print("=== build package (once) ===")
    t = time.time()
    pkg = PS.build_package(cat, tg, pz, dz_pool=dz, verbose=True)
    print(f"build: {time.time()-t:.1f}s")
    PS.write_package(pkg, PKG)
    sz = os.path.getsize(PKG if PKG.endswith(".npz") else PKG + ".npz")
    print(f"package on disk: {sz/1e6:.2f} MB  ({pkg['n_obs']:,} obs + {pkg['n_miss']:,} missing)")

    pk = PS.load_package(PKG)

    # ---- speed: many draws ----
    print("\n=== sampler speed ===")
    t = time.time(); ND = 50
    for s in range(ND):
        c = PS.draw(pk, seed=s)
    dt = time.time() - t
    print(f"{ND} draws in {dt:.2f}s -> {ND/dt:.1f} samples/s  (N≈{c['N']:,} galaxies each)")
    # naive ensemble size this replaces (float32 RA,Dec,Z per realization)
    naive_per = c["N"] * 3 * 4
    print(f"naive stored ensemble: {naive_per/1e6:.2f} MB/realization; "
          f"1000 realizations = {1000*naive_per/1e9:.2f} GB  vs  package {sz/1e6:.2f} MB + seeds")

    # ---- fidelity: sampler vs complete_catalog_photoz ----
    print("\n=== fidelity vs complete_catalog_photoz(z_mode='field') ===")
    nreal = 8
    samp = [PS.draw(pk, seed=s) for s in range(nreal)]
    ref = [complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz) for s in range(nreal)]
    zb = np.linspace(z.min(), z.max(), 40)
    ns_s = np.mean([np.histogram(np.asarray(c["z"]), zb, density=True)[0] for c in samp], 0)
    ns_r = np.mean([np.histogram(np.asarray(c["z"]), zb, density=True)[0] for c in ref], 0)
    print(f"  n(z): max |sampler-ref|/ref = {np.nanmax(np.abs(ns_s/ns_r - 1)):.3f}")
    print(f"  mean N: sampler {np.mean([c['N'] for c in samp]):.0f}  ref {np.mean([c['N'] for c in ref]):.0f}")
    from twopt_density.quaia import make_random_from_selection_function
    rng = np.random.default_rng(1)
    rar, decr, _ = make_random_from_selection_function(sel_map=cat.sel_map, n_random=2*cat.N_data,
                                                       z_data=z, nside=cat.nside, rng=rng)
    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:]*tb[:-1])
    rr = None; Ws, Wr = [], []
    for c in samp:
        w, rr = wtheta(np.asarray(c["ra"]), np.asarray(c["dec"]), rar, decr, tb, rr); Ws.append(w)
    for c in ref:
        w, rr = wtheta(np.asarray(c["ra"]), np.asarray(c["dec"]), rar, decr, tb, rr); Wr.append(w)
    ratio = np.mean(Ws, 0) / np.mean(Wr, 0)
    print(f"  w(θ): sampler/ref median {np.median(ratio):.3f}, range {np.nanmin(ratio):.3f}-{np.nanmax(ratio):.3f}")
    print("\n(small => the compact sampler reproduces the full completion; ship the package + sampler)")


if __name__ == "__main__":
    main()
