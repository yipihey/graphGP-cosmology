"""Phase 3 — higher-order clustering recovery of the completion (beyond 2-point).

Two-point closure can hide higher-order errors. Using the real-BOSS-truth
inject-and-recover setup (real 1-halo clustering; Patchy is unreliable sub-Mpc),
we test that the completed ensemble recovers the TRUTH for higher-order,
coincidence-sensitive statistics:
  * kNN-CDF — the CDF of the distance from random query points to the k-th nearest
    galaxy (k=1,2,4), a full-hierarchy clustering probe (Banerjee & Abel 2021);
    also directly sensitive to the Δθ=0 duplicate artifact (a 1-NN spike at 0).
  * counts-in-cells PDF — mean, var/mean, skew in fixed apertures.
3-D distances use a fiducial cosmology (measurement-time only).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/validate_completion_highorder.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import comoving_mpc_h
from twopt_density.mock_systematics import apply_survey_systematics

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def xyz(ra, dec, z):
    d = comoving_mpc_h(z); r = np.radians(ra); dd = np.radians(dec)
    return np.column_stack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)])


def knn_cdf(gal_xyz, q_xyz, ks, redges):
    tree = cKDTree(gal_xyz)
    dist, _ = tree.query(q_xyz, k=max(ks), workers=-1)
    return {k: np.searchsorted(np.sort(dist[:, k-1]), redges) / len(q_xyz) for k in ks}


def cic(gal_xyz, cen_xyz, radius):
    return cKDTree(gal_xyz).query_ball_point(cen_xyz, radius, return_length=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--out", default="output/completion_highorder.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, _ = apply_survey_systematics(ra, dec, z, colors, mags, wsys,
                                                coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=0)
    dz = measure_close_pair_dz(obs, 62/3600.)
    cats = [complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz) for s in range(args.n_real)]

    # query points + cell centres from randoms (footprint-uniform)
    rng = np.random.default_rng(3)
    rar, decr, zr = make_random_from_selection_function(sel_map=cat.sel_map, n_random=2*len(ra),
                                                        z_data=z, nside=cat.nside, rng=rng)
    qsel = rng.choice(len(rar), 60000, replace=False)
    q_xyz = xyz(rar[qsel], decr[qsel], zr[qsel])
    csel = rng.choice(len(rar), 8000, replace=False)
    c_xyz = xyz(rar[csel], decr[csel], zr[csel])

    ks = [1, 2, 4]; redges = np.logspace(np.log10(2.0), np.log10(40.0), 30)
    tru_xyz = xyz(ra, dec, z); obs_xyz = xyz(obs.ra_data, obs.dec_data, obs.z_data)
    knn_t = knn_cdf(tru_xyz, q_xyz, ks, redges)
    knn_o = knn_cdf(obs_xyz, q_xyz, ks, redges)
    KNN_c = [knn_cdf(xyz(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"])), q_xyz, ks, redges) for c in cats]

    Rcic = 8.0
    m_t = cic(tru_xyz, c_xyz, Rcic)
    m_c = np.mean([cic(xyz(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"])), c_xyz, Rcic) for c in cats], 0)
    def mom(x): x=np.asarray(x,float); return x.mean(), x.var()/max(x.mean(),1e-9), ((x-x.mean())**3).mean()/max(x.var(),1e-9)**1.5
    print(f"counts-in-cells (R={Rcic} Mpc/h)  mean, var/mean, skew:")
    print(f"  truth:     {tuple(np.round(mom(m_t),3))}")
    print(f"  completed: {tuple(np.round(mom(m_c),3))}")
    # report kNN-CDF recovery as max ratio deviation per k
    rc = np.sqrt(redges[1:]*redges[:-1]); rmid = 0.5*(redges[1:]+redges[:-1])
    knn_c_mean = {k: np.mean([K[k] for K in KNN_c], 0) for k in ks}
    print("\nkNN-CDF recovery (max |completed-truth| over r, per k):")
    for k in ks:
        print(f"  k={k}: max|ΔCDF|={np.max(np.abs(knn_c_mean[k]-knn_t[k])):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    for k, col in zip(ks, ["#3a6ea8", "#e8853a", "#7b3ff2"]):
        a.semilogx(redges, knn_t[k], "-", color=col, lw=2, label=f"truth k={k}")
        a.semilogx(redges, knn_c_mean[k], "o", color=col, ms=3, label=f"completed k={k}")
    a.set_xlabel("r [Mpc/h]"); a.set_ylabel("kNN-CDF P(<r)"); a.legend(fontsize=7); a.set_title("kNN-CDF recovery")
    a = ax[1]
    mx = int(max(np.max(m_t), np.max(m_c))); bins = np.arange(0, mx+2)
    a.hist(m_t, bins=bins, density=True, histtype="step", color="k", lw=2, label="truth")
    a.hist(m_c, bins=bins, density=True, histtype="step", color="#3a6ea8", lw=2, label="completed")
    a.set_xlabel(f"galaxies in R={Rcic} Mpc/h sphere"); a.set_ylabel("PDF"); a.legend(); a.set_title("counts-in-cells")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
