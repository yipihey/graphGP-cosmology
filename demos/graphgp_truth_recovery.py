"""Phase 2: which redshift engine is more FAITHFUL TO TRUTH — KNN-KDE or graphGP?

Inject-and-recover on real-BOSS-truth. The full real CMASS-South is TRUTH. We inject
extra fiber collisions + redshift failures + imaging thinning, then complete the
mock-observed catalog with BOTH redshift engines and compare the recovered wp(rp) to
the TRUE wp (and to an oracle that places the missing galaxies at their true z). The
real-data head-to-head showed graphGP matches the official WEIGHTED clustering more
closely than KNN; this says which one matches TRUTH.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/graphgp_truth_recovery.py
"""
import argparse, dataclasses, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz, PROV
from twopt_density.mock_systematics import apply_survey_systematics
from twopt_density.density_field import sample_posterior_density_field
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import wp_rp
from demos.graphgp_vs_knn_zfield import graphgp_catalogs

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def strip_systot(c):
    m = np.asarray(c["prov"]) != PROV["systot"]
    return {"ra": np.asarray(c["ra"])[m], "dec": np.asarray(c["dec"])[m], "z": np.asarray(c["z"])[m]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=4)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)

    # ---- inject systematics: real CMASS = TRUTH, observe an incomplete mock ----
    obs, tg, kept, true_z = apply_survey_systematics(ra, dec, z, colors, mags, wsys,
                                                     coll_frac=0.6, zfail_frac=0.014, seed=0)
    print(f"truth N={len(ra):,}  observed N={obs.N_data:,}  missing(targets) N={tg.N:,}")

    # photo-z + close-pair prior measured on the MOCK-OBSERVED (what an analyst has)
    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62/3600.)

    # mock-observed as a BOSSCatalog (subset) so the GP engine can condition on it
    km = kept
    mock_cat = dataclasses.replace(cat, ra_data=ra[km], dec_data=dec[km], z_data=z[km],
                                   xyz_data=np.asarray(cat.xyz_data)[km],
                                   w_data=np.ones(int(km.sum())),
                                   # neutralize completeness components (mock-observed IS the
                                   # conditioning set; density_field would otherwise read the
                                   # full-length real-cat weights -> shape mismatch)
                                   w_sys_data=None, w_cp_data=None, w_noz_data=None, w_fkp_data=None,
                                   colors_data=None, mags_data=None, colors_finite=None,
                                   imatch_data=None, icollided_data=None)

    # ---- KNN-KDE ensemble ----
    print("[knn] completing ...")
    knn = [strip_systot(complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz, z_mode="field"))
           for s in range(args.n_real)]

    # ---- graphGP conditional-field ensemble ----
    print(f"[graphgp] building conditional field (n={args.n_real}, nside={args.nside}, nz={args.nz}) ...")
    res = sample_posterior_density_field(mock_cat, n_samples=args.n_real, nside=args.nside, n_z_bins=args.nz,
                                         r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28),
                                         seed=0, verbose=False)
    ggp = graphgp_catalogs(res, mock_cat, tg, pz, dz, args.n_real, np.random.default_rng(3))

    # ---- truth / observed / oracle references ----
    truth = {"ra": ra, "dec": dec, "z": z}
    observed = {"ra": np.asarray(obs.ra_data), "dec": np.asarray(obs.dec_data), "z": np.asarray(obs.z_data)}
    oracle = {"ra": np.concatenate([observed["ra"], np.asarray(tg.ra)]),
              "dec": np.concatenate([observed["dec"], np.asarray(tg.dec)]),
              "z": np.concatenate([observed["z"], true_z])}

    rar, decr, zr = make_random_from_selection_function(sel_map=cat.sel_map, n_random=3*cat.N_data,
                                                        z_data=z, nside=cat.nside, rng=np.random.default_rng(7))
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    wp_t, RR = wp_rp(truth["ra"], truth["dec"], truth["z"], rar, decr, zr, rp_edges=rp_edges, nthreads=16, return_RR=True)
    def wp1(c): return wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]), rar, decr, zr,
                            rp_edges=rp_edges, nthreads=16, precomp_RR=RR)
    def wpe(cc): return np.mean([wp1(c) for c in cc], 0)
    wp_o = wp1(observed); wp_or = wp1(oracle); wp_k = wpe(knn); wp_g = wpe(ggp)

    def summ(name, w):
        r = w / wp_t
        print(f"  {name:22s} median {np.median(r):.3f}  range {r.min():.3f}-{r.max():.3f}")
    print("\n=== wp(rp) recovered / TRUTH ===")
    summ("observed (incomplete)", wp_o)
    summ("oracle (true z)", wp_or)
    summ("KNN-KDE completion", wp_k)
    summ("graphGP completion", wp_g)
    print("\n  rp      truth    obs/t   oracle/t   KNN/t    GP/t")
    for i in range(len(rpc)):
        print(f"  {rpc[i]:6.2f}  {wp_t[i]:7.2f}  {wp_o[i]/wp_t[i]:6.3f}  {wp_or[i]/wp_t[i]:7.3f}  "
              f"{wp_k[i]/wp_t[i]:6.3f}  {wp_g[i]/wp_t[i]:6.3f}")
    print("\n(closest-to-1 across rp = more faithful to truth; oracle is the floor set by z-assignment.)")


if __name__ == "__main__":
    main()
