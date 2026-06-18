"""Is the graphGP density-field redshift assignment cosmology-dependent, or is the
fiducial cosmology just a unit/gauge choice?

The conditional-field engine (density_field.py) converts z->comoving r with a
fiducial cosmology to (a) build the Vecchia neighbour graph and (b) measure the
kernel xi(r). The kernel is MEASURED from the data (no LCDM/BAO/growth assumed),
and the output redshift grid is in observed z. Claim under test (Abel): the
fiducial cosmology is a monotonic z->distance reparametrisation that the measured
kernel absorbs, so it injects NO cosmology prior — the result is gauge-invariant.

We test it head-on: assign the SAME missing galaxies' redshifts via the GP field
built under TWO very different fiducial cosmologies — Planck (Om=0.315) and
Einstein-de Sitter (Om=1.0) — with identical galaxies, targets, photo-z, RNG. If
the per-object redshifts and the recovered wp(rp)/xi0 are invariant, the fiducial
is a gauge choice and we can state the method is fully data-driven, no cosmology
prior.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/graphgp_cosmology_invariance.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.distance import DistanceCosmo
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import measure_close_pair_dz
from twopt_density.density_field import sample_posterior_density_field
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import wp_rp, xi_smu_multipoles
from demos.graphgp_vs_knn_zfield import graphgp_catalogs

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=3)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    args = p.parse_args()

    COSMO = {"Planck(Om=0.315)": DistanceCosmo(Om=0.315, h=0.674, w0=-1.0, wa=0.0),
             "EdS(Om=1.0)":      DistanceCosmo(Om=1.000, h=0.674, w0=-1.0, wa=0.0)}

    fields = {}; z_assign = {}; cats = {}
    base = None
    for name, cosmo in COSMO.items():
        cat = load_boss([DATA], [RAND], sample="CMASS", fid_cosmo=cosmo, nside=256, with_photometry=True)
        if base is None:   # galaxies/targets/photo-z are cosmology-INDEPENDENT (built once)
            z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
            good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
            pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
            tg = load_cmass_targets(cat, path=TARGETS, seed=0)
            base = cat
        print(f"\n[{name}] building GP field ...")
        res = sample_posterior_density_field(cat, n_samples=args.n_real, nside=args.nside, n_z_bins=args.nz,
                                             r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28),
                                             seed=0, verbose=False)
        rng = np.random.default_rng(3)                 # SAME rng for both cosmologies
        cc = graphgp_catalogs(res, cat, tg, pz, dz, args.n_real, rng)
        cats[name] = cc
        # store realization-0 missing-z (last n_miss entries)
        M = tg.N
        z_assign[name] = np.array([c["z"][-M:] for c in cc])

    names = list(COSMO)
    zA, zB = z_assign[names[0]], z_assign[names[1]]
    dz_obj = (zA - zB).ravel()
    scatter = np.std([c["z"][-tg.N:] for c in cats[names[0]]], axis=0).mean()  # per-object assignment scatter
    print("\n" + "="*70)
    print(f"per-object missing-z: RMS(z_{names[0]} - z_{names[1]}) = {np.sqrt(np.mean(dz_obj**2)):.5f}")
    print(f"  (vs typical per-object assignment scatter across realizations {scatter:.5f}; "
          f"vs photo-z sigma_z~0.03)")
    print(f"  correlation(z_A, z_B) = {np.corrcoef(zA.ravel(), zB.ravel())[0,1]:.4f}")

    # recovered clustering under each cosmology
    z = np.asarray(base.z_data); ra = np.asarray(base.ra_data); dec = np.asarray(base.dec_data)
    rar, decr, zr = make_random_from_selection_function(sel_map=base.sel_map, n_random=3*base.N_data,
                                                        z_data=z, nside=base.nside, rng=np.random.default_rng(7))
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    RR = [None]
    def wp_of(cc):
        W = []
        for c in cc:
            out = wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]), rar, decr, zr,
                       rp_edges=rp_edges, nthreads=16, precomp_RR=RR[0], return_RR=(RR[0] is None))
            if RR[0] is None: out, RR[0] = out
            W.append(out)
        return np.mean(W, 0)
    wpA, wpB = wp_of(cats[names[0]]), wp_of(cats[names[1]])
    print(f"\nwp(rp) ratio {names[0]} / {names[1]}: median {np.median(wpA/wpB):.4f}, "
          f"max|dev| {np.max(np.abs(wpA/wpB-1)):.4f}")
    for i in range(len(rpc)):
        print(f"  rp={rpc[i]:6.2f}  {names[0]}={wpA[i]:7.2f}  {names[1]}={wpB[i]:7.2f}  ratio={wpA[i]/wpB[i]:.4f}")
    print("\n(RMS(z_A-z_B) << scatter and wp ratio ~1 => fiducial cosmology is a gauge/unit "
          "choice; the measured kernel absorbs it; NO cosmology prior — fully data-driven.)")


if __name__ == "__main__":
    main()
