"""Systematics scan over the completed-catalog ensemble — the inference use case.

The completed catalogs are the unbiased, equal-weight inputs a cosmology analysis
would run on. This demo shows what that analysis consumes: for a clustering
summary statistic w(θ), the ENSEMBLE of completion realizations gives a data
vector + covariance, and scanning the completion's redshift-assignment prior
(photo-z × close-pair clustering  vs  photo-z only) quantifies the
*observational-systematic budget* — how much the missing-galaxy treatment moves
the summary, relative to the realization scatter. That budget is exactly what
"scan over all realizations to characterize systematics" means.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/completion_systematics_scan.py \
        --targets data/boss/cmass_targets_South.fits
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density import perf

COLL = 62.0 / 3600.0
NTH = 16


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, rr=None):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    if rr is None:                                    # RR depends only on the (fixed) randoms
        rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/boss/cmass_targets_South.fits")
    p.add_argument("--n-real", type=int, default=10)
    p.add_argument("--out", default="output/completion_systematics.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    z_d = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    tr = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[tr], z_d[tr])
    dz_pool = measure_close_pair_dz(cat, COLL)
    targets = load_cmass_targets(cat, path=args.targets, seed=0)

    rng = np.random.default_rng(7)
    rar, decr, _ = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=300_000, z_data=z_d, nside=cat.nside, rng=rng)
    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:]*tb[:-1])

    rr_w = DDtheta_mocks(1, NTH, tb, rar.astype("f8"), decr.astype("f8"))["npairs"].astype(float)
    configs = {"photo-z × clustering": "data", "photo-z only": "none"}
    res = {}
    for label, prior in configs.items():
        W = []
        for s in range(args.n_real):
            c = complete_catalog_photoz(cat, targets, pz, seed=s, clustering_prior=prior,
                                        dz_pool=dz_pool)
            W.append(wtheta(c["ra"], c["dec"], rar, decr, tb, rr=rr_w))
        W = np.array(W)
        res[label] = (W.mean(0), W.std(0))

    (mA, sA), (mB, sB) = res["photo-z × clustering"], res["photo-z only"]
    print(f"completion-systematic budget (clustering-prior choice) over {args.n_real} realizations:")
    print(f"{'theta':>7}{'w[clust]':>10}{'w[pzonly]':>11}{'Δ_sys':>9}{'σ_stat':>9}{'Δ/σ':>7}")
    for i in range(len(tc)):
        d = abs(mA[i] - mB[i]); s = 0.5 * (sA[i] + sB[i])
        print(f"{tc[i]:7.3f}{mA[i]:10.4f}{mB[i]:11.4f}{d:9.4f}{s:9.4f}{d/s if s else np.nan:7.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.fill_between(tc, mA - sA, mA + sA, color="#4a90d9", alpha=0.25)
    a1.plot(tc, mA, "o-", color="#4a90d9", label="photo-z × clustering (mean ± realization σ)")
    a1.fill_between(tc, mB - sB, mB + sB, color="#d0021b", alpha=0.2)
    a1.plot(tc, mB, "s--", color="#d0021b", label="photo-z only")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel(r"$\theta$ [deg]")
    a1.set_ylabel(r"$w(\theta)$"); a1.legend(); a1.set_title("ensemble w(θ): two completion priors")
    a1.grid(True, which="both", alpha=0.2)
    a2.semilogx(tc, np.abs(mA - mB) / (0.5 * (sA + sB)), "o-", color="#333")
    a2.axhline(1, color="r", ls="--", label="systematic = statistical")
    a2.set_xlabel(r"$\theta$ [deg]"); a2.set_ylabel(r"$\Delta_{\rm sys}/\sigma_{\rm stat}$")
    a2.set_title("prior-systematic budget"); a2.legend(); a2.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    perf.report("completion_systematics_scan")


if __name__ == "__main__":
    main()
