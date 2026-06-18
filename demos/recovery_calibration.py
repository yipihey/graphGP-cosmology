"""Phase 4 — is the completion ENSEMBLE a calibrated posterior?

Truth recovery (Phase 2) showed the ensemble MEAN matches truth. Here we test the
ensemble SCATTER: across many MultiDark-Patchy mocks we inject the systematics,
build the completion ensemble, and ask whether the truth falls inside the
ensemble's credible interval at the nominal rate (coverage / PIT) for each
statistic and scale. A trustworthy posterior must be neither over- nor
under-confident. We use wp(rp) (Corrfunc, parallel; randoms fixed so RR is shared)
and report:
  * coverage: fraction of (mock, rp-bin) with truth inside the 68% ensemble band,
  * PIT: rank of truth within the per-bin ensemble (uniform if calibrated),
  * the completion (within-mock) scatter vs the mock-to-mock (cosmic-variance)
    scatter — the completion uncertainty is the *added* uncertainty from the
    unobserved galaxies, and should be the smaller, sub-dominant term.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/recovery_calibration.py
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.clustering_corrfunc import wp_rp
from twopt_density.mock_systematics import (apply_survey_systematics, load_patchy_truth,
                                            load_patchy_randoms)

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
MOCKDIR = "data/boss/mocks"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mocks", type=int, default=6)
    p.add_argument("--n-real", type=int, default=10)
    p.add_argument("--out", default="output/recovery_calibration.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])           # colour-z relation (shared)

    rar, decr, zr = load_patchy_randoms(f"{MOCKDIR}/Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x10.dat",
                                        z_min=0.43, z_max=0.7, max_n=450_000)
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    RRwp = None

    mocks = sorted(glob.glob(f"{MOCKDIR}/Patchy-Mocks-DR12SGC-COMPSAM_V6C_*.dat"))[:args.n_mocks]
    wp_truth, wp_ens = [], []                                # per mock: truth, (n_real, nrp)
    for mi, mf in enumerate(mocks):
        ra, dec, zz, colors, mags, wsys = load_patchy_truth(mf, cat, z_min=0.43, z_max=0.7)
        obs, tg, kept, _ = apply_survey_systematics(ra, dec, zz, colors, mags, wsys,
                                                    coll_frac=0.6, zfail_frac=0.014,
                                                    zfail_faint_bias=1.5, seed=mi)
        dz = measure_close_pair_dz(obs, 62/3600.)
        wt, RRwp = wp_rp(ra, dec, zz, rar, decr, zr, rp_edges=rp_edges, pimax=40., nthreads=32,
                         precomp_RR=RRwp, return_RR=True)
        wp_truth.append(wt)
        W = []
        for s in range(args.n_real):
            c = complete_catalog_photoz(obs, tg, pz, seed=100*mi+s, dz_pool=dz)
            W.append(wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]),
                           rar, decr, zr, rp_edges=rp_edges, pimax=40., nthreads=32, precomp_RR=RRwp))
        wp_ens.append(np.array(W))
        print(f"  mock {mi+1}/{len(mocks)} done", flush=True)
    wp_truth = np.array(wp_truth)                            # (Nm, nrp)
    wp_ens = np.array(wp_ens)                                # (Nm, nreal, nrp)

    # coverage: truth inside the per-mock 16-84 ensemble band
    lo = np.percentile(wp_ens, 16, axis=1); hi = np.percentile(wp_ens, 84, axis=1)   # (Nm,nrp)
    inside = (wp_truth >= lo) & (wp_truth <= hi)
    cov = inside.mean()
    # PIT: rank of truth within the ensemble per (mock, bin)
    pit = (wp_ens < wp_truth[:, None, :]).mean(axis=1).ravel()
    # within-mock completion scatter vs mock-to-mock (cosmic) scatter, fractional
    cmp_std = wp_ens.std(axis=1).mean(0) / wp_truth.mean(0)          # completion uncertainty
    mock_std = wp_truth.std(0) / wp_truth.mean(0)                    # cosmic variance
    print(f"\ncoverage (68% band, target 0.68): {cov:.2f}  over {inside.size} (mock,bin) cells")
    print(f"PIT mean {pit.mean():.2f} (0.5 ideal), std {pit.std():.2f} (0.29 ideal)")
    print(f"{'rp':>8}{'cmp_unc%':>10}{'cosmic%':>10}{'ratio':>8}")
    for i in range(len(rpc)):
        print(f"{rpc[i]:8.2f}{100*cmp_std[i]:10.2f}{100*mock_std[i]:10.2f}{cmp_std[i]/max(mock_std[i],1e-9):8.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    for mi in range(len(mocks)):
        a.semilogx(rpc, wp_ens[mi].mean(0)/wp_truth[mi], color="#3a6ea8", alpha=0.5, lw=1)
    a.axhline(1, color="k", ls=":"); a.fill_between(rpc, 0.97, 1.03, color="green", alpha=0.1)
    a.set_ylim(0.9, 1.1); a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("completed/truth (per mock)")
    a.set_title(f"wp recovery across {len(mocks)} mocks")
    a = ax[1]
    a.hist(pit, bins=10, range=(0, 1), color="#3a6ea8", alpha=0.8, edgecolor="white")
    a.axhline(len(pit)/10, color="r", ls="--", label="uniform (calibrated)")
    a.set_xlabel("PIT: rank of truth in ensemble"); a.set_ylabel("count"); a.legend()
    a.set_title(f"calibration  (coverage {cov:.2f}, target 0.68)")
    a = ax[2]
    a.loglog(rpc, 100*cmp_std, "o-", color="#3a6ea8", label="completion uncertainty")
    a.loglog(rpc, 100*mock_std, "s--", color="#c0392b", label="cosmic variance (mock-to-mock)")
    a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("fractional scatter [%]"); a.legend()
    a.set_title("completion uncertainty is sub-dominant")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
