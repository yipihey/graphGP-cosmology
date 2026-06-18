"""Controlled truth-recovery test for the completion (inject-and-recover).

Take the real BOSS CMASS-South galaxies as the TRUTH (a fair sample of the true
field, with realistic clustering, colours and n(z)); inject a known, realistic
systematics model (imaging-systematic thinning + fiber collisions + redshift
failures, twopt_density.mock_systematics); run the completion on the degraded
"observed" catalogue; and check that the completed ENSEMBLE recovers the TRUTH
statistics — not by construction (the completion never sees the truth), but as a
genuine inject-and-recover test. We report three curves for each statistic:
TRUTH, OBSERVED (degraded, uncorrected), COMPLETED (ensemble mean ± scatter).
Recovery = completed≈truth while observed deviates.

Statistics: w(θ), ξ(Δθ,Δz=0), n(z), counts-in-cells (higher-order).
Cosmology-free (observed RA, Dec, z).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/mock_truth_recovery.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.observed import _radec_to_nhat
from twopt_density.observed_ls import (complete_catalog_photoz, measure_close_pair_dz,
                                       measure_K2d, compute_rr)
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.mock_systematics import apply_survey_systematics

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
NTH = 16


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, rr=None):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    if rr is None:
        rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan), rr


def cic(ra_g, dec_g, cen_nhat, radius_deg):
    t = cKDTree(_radec_to_nhat(ra_g, dec_g))
    return np.array(t.query_ball_point(cen_nhat, np.radians(radius_deg), return_length=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=8)
    p.add_argument("--coll-frac", type=float, default=0.6)
    p.add_argument("--zfail-frac", type=float, default=0.014)
    p.add_argument("--out", default="output/mock_truth_recovery.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data)
    wsys = np.asarray(cat.w_sys_data)              # realistic w_systot amplitude template
    print(f"TRUTH = {len(ra):,} real CMASS-South galaxies")

    # ---- inject known systematics ----
    obs, tg, kept = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=args.coll_frac,
        zfail_frac=args.zfail_frac, zfail_faint_bias=1.5, seed=0)
    print(f"OBSERVED = {obs.N_data:,} ({100*obs.N_data/len(ra):.1f}% of truth); "
          f"missing: {int((tg.miss_kind=='collided').sum()):,} collided + "
          f"{int((tg.miss_kind=='zfail').sum()):,} zfail")

    # photo-z trained on the OBSERVED survivors (which have z), applied to missing
    feat = photoz_features(obs.colors_data, obs.mags_data)
    g = np.isfinite(feat).all(1)
    pz = PhotoZKNN(k=100).fit(feat[g], obs.z_data[g])
    dz = measure_close_pair_dz(obs, 62/3600.)

    # ---- randoms (shared) ----
    rng = np.random.default_rng(7)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=2*len(ra), z_data=z, nside=cat.nside, rng=rng)
    one = np.ones(len(rar))

    # ---- completion ensemble ----
    cats = [complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz) for s in range(args.n_real)]

    # ---- w(theta): truth / observed / completed ensemble ----
    tb = np.logspace(np.log10(0.02), np.log10(2.5), 13); tc = np.sqrt(tb[1:]*tb[:-1])
    nsub = min(400_000, len(rar)); ri = rng.choice(len(rar), nsub, False)
    rsa, rsd = rar[ri], decr[ri]
    w_truth, rrw = wtheta(ra, dec, rsa, rsd, tb)
    w_obs, _ = wtheta(obs.ra_data, obs.dec_data, rsa, rsd, tb, rr=rrw)
    W = np.array([wtheta(np.asarray(c["ra"]), np.asarray(c["dec"]), rsa, rsd, tb, rr=rrw)[0] for c in cats])
    w_cmp, w_cmp_s = W.mean(0), W.std(0)
    print("\nw(θ) recovery  (completed/truth, observed/truth):")
    for i in range(len(tc)):
        print(f"  θ={tc[i]:.3f}: truth={w_truth[i]:.4f} obs={w_obs[i]:.4f} cmp={w_cmp[i]:.4f}"
              f"  cmp/tru={w_cmp[i]/w_truth[i]:.3f}  obs/tru={w_obs[i]/w_truth[i]:.3f}")

    # ---- xi(dtheta, dz=0) recovery ----
    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 14)]); ze = np.linspace(0, 0.03, 9)
    tcen = np.empty(len(te)-1); tcen[0] = 0.5*te[1]; tcen[1:] = np.sqrt(te[1:-1]*te[2:])
    rrc = compute_rr(rar, decr, zr, one, theta_edges=te, z_edges=ze)
    xi_tru = measure_K2d(ra, dec, z, np.ones(len(ra)), rar, decr, zr, one, theta_edges=te, z_edges=ze, precomp_rr=rrc)[2][:,0]
    Xi = np.array([measure_K2d(np.asarray(c["ra"]),np.asarray(c["dec"]),np.asarray(c["z"]),np.ones(c["N"]),
                               rar,decr,zr,one,theta_edges=te,z_edges=ze,precomp_rr=rrc)[2][:,0] for c in cats])
    xi_cmp = Xi.mean(0)

    # ---- n(z) recovery ----
    zb = np.linspace(0.43, 0.62, 30); zc = 0.5*(zb[1:]+zb[:-1])
    nz_tru,_ = np.histogram(z, zb); nz_obs,_ = np.histogram(obs.z_data, zb)
    nz_cmp = np.mean([np.histogram(np.asarray(c["z"]), zb)[0] for c in cats], 0)

    # ---- counts-in-cells (higher-order) at random footprint centres ----
    ci = rng.choice(len(rar), 4000, False); cen = _radec_to_nhat(rar[ci], decr[ci]); R=0.3
    def moms(x): return x.mean(), x.var()/max(x.mean(),1e-9), ((x-x.mean())**3).mean()/max(x.var(),1e-9)**1.5
    m_tru = moms(cic(ra,dec,cen,R)); m_obs = moms(cic(obs.ra_data,obs.dec_data,cen,R))
    m_cmp = np.mean([moms(cic(np.asarray(c["ra"]),np.asarray(c["dec"]),cen,R)) for c in cats], 0)
    print(f"\ncounts-in-cells (r={R}°)  mean, var/mean, skew:")
    print(f"  truth:     {tuple(np.round(m_tru,3))}")
    print(f"  observed:  {tuple(np.round(m_obs,3))}")
    print(f"  completed: {tuple(np.round(m_cmp,3))}")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0,0]
    a.loglog(tc, w_truth, "k-", lw=2, label="truth"); a.loglog(tc, w_obs, "s--", color="#c0392b", label="observed (degraded)")
    a.fill_between(tc, w_cmp-w_cmp_s, w_cmp+w_cmp_s, color="#3a6ea8", alpha=0.3)
    a.loglog(tc, w_cmp, "o-", color="#3a6ea8", label="completed (ens. mean±σ)")
    a.set_xlabel("θ [deg]"); a.set_ylabel("w(θ)"); a.legend(); a.set_title("w(θ) recovery")
    a = ax[0,1]
    a.axhline(1, color="gray", ls=":"); a.fill_between(tc, 0.95, 1.05, color="green", alpha=0.1)
    a.semilogx(tc, w_obs/w_truth, "s--", color="#c0392b", label="observed/truth")
    a.semilogx(tc, w_cmp/w_truth, "o-", color="#3a6ea8", label="completed/truth")
    a.set_ylim(0.7, 1.2); a.set_xlabel("θ [deg]"); a.set_ylabel("ratio to truth"); a.legend(); a.set_title("w(θ) ratio")
    a = ax[1,0]
    a.semilogx(tcen, np.where(xi_tru>0, xi_cmp/xi_tru, np.nan), "o-", color="#3a6ea8")
    a.axhline(1, color="gray", ls=":"); a.fill_between(tcen, 0.95, 1.05, color="green", alpha=0.1)
    a.set_ylim(0.7,1.2); a.set_xlabel("Δθ [deg]"); a.set_ylabel("ξ(Δθ,0) completed/truth"); a.set_title("2-D ξ recovery")
    a = ax[1,1]
    a.plot(zc, nz_tru, "k-", lw=2, label="truth"); a.plot(zc, nz_obs, "s--", color="#c0392b", label="observed")
    a.plot(zc, nz_cmp, "o-", color="#3a6ea8", label="completed"); a.set_xlabel("z"); a.set_ylabel("N/bin")
    a.legend(); a.set_title("n(z) recovery")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
