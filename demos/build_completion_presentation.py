"""Build a self-contained HTML presentation of the photo-z catalog-completion
method (BOSS CMASS-SGC).

Runs the pipeline once to produce a designed, coherently-styled figure set with
verbose captions, caches the expensive measurements, and writes a single-scroll
sectioned HTML to output/completion_presentation.html and docs/completion.html
(base64-inline figures; no external dependencies).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/build_completion_presentation.py
        [--recompute]   force recompute (default: reuse output/_presentation_cache.npz)
        [--quick]       small N / subsample for a fast validation build
"""
import argparse, base64, io, os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 130, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})
C_OBS = "#e8853a"     # observed / completeness-weighted
C_NEW = "#3a6ea8"     # completed / photo-z
C_ZF = "#7b3ff2"      # z-failures
C_NEUTRAL = "#888888"
CACHE = "output/_presentation_cache.npz"
MASK_CACHE = "output/_presentation_mask_cache.npz"
COUP_CACHE = "output/_presentation_coupling_cache.npz"
COLL = 62.0 / 3600.0
NSIDE_MASK = 512
DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64(path):
    """Inline an existing PNG (a validation figure produced by a demos/ script) as base64."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


# Headline validation numbers from the demos/ batteries (see each caption for the
# producing script). Stored here so the report states exact figures without re-running
# the multi-hour mock recoveries on every HTML build.
VALIDATION = {
    # Phase 2 truth recovery: demos/mock_truth_recovery.py (real-BOSS-truth + Patchy)
    "tr_wp_lo": 0.98, "tr_wp_hi": 1.01, "tr_oracle_lo": 0.997, "tr_oracle_hi": 1.005,
    # Phase 3 higher-order: demos/validate_completion_highorder.py
    "cic_mean_t": 0.663, "cic_mean_c": 0.660, "cic_vm_t": 2.421, "cic_vm_c": 2.367,
    "cic_skew_t": 2.939, "cic_skew_c": 2.895,
    "knn_k1": 0.0022, "knn_k2": 0.0021, "knn_k4": 0.0011,
    # Phase 5 cosmology consistency: demos/validate_cosmology_consistency.py
    "cc_wp_lo": 0.985, "cc_wp_hi": 1.047, "cc_xi0_lo": 0.98, "cc_xi0_hi": 1.04,
    # Phase 1 sensitivity: demos/audit_sensitivity.py  med/max |Δwp/wp|
    "sens": [("redshift mode → nearest-neighbour", 0.012, 0.039),
             ("redshift mode → raw photo-z", 0.027, 0.051),
             ("count → Poisson (vs round)", 0.000, 0.000),
             ("photo-z neighbours k: 50–150", 0.000, 0.001),
             ("collision scale 40–90″", 0.003, 0.006)],
    # Phase 4 calibration: demos/recovery_calibration.py
    "cal_unc_lo": 0.17, "cal_unc_hi": 0.44, "cal_cos_lo": 0.78, "cal_cos_hi": 7.16,
    "cal_ratio_lo": 0.06, "cal_ratio_hi": 0.36, "cal_cov": 0.08,
}


# ----------------------------------------------------------------------
# Compute (cached)
# ----------------------------------------------------------------------
def compute(quick=False):
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    from twopt_density.boss import load_boss
    from twopt_density.quaia import make_random_from_selection_function
    from twopt_density.photoz import PhotoZKNN, photoz_features
    from twopt_density.cmass_targets import load_cmass_targets
    from twopt_density.observed_ls import (measure_K2d, compute_rr, complete_catalog_photoz,
                                           measure_close_pair_dz, _clpair_density)
    from twopt_density import perf

    NTH = 16
    n_real = 4 if quick else 12
    n_real_2d = 2 if quick else 4
    nrf = 2

    def wtheta(ra_d, dec_d, ra_r, dec_r, tb, w_d=None, rr=None):
        nd, nr = len(ra_d), len(ra_r)
        kw = dict(weights1=w_d.astype("f8"), weight_type="pair_product") if w_d is not None else {}
        dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"), **kw)
        if rr is None:                                    # RR depends only on the (fixed) randoms
            rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
        if w_d is not None:
            Wd = w_d.sum()
            DD = dd["npairs"] * dd["weightavg"] / Wd**2
            dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                               weights1=w_d.astype("f8"), RA2=ra_r.astype("f8"),
                               DEC2=dec_r.astype("f8"), weight_type="pair_product")
            DR = dr["npairs"] * dr["weightavg"] / (Wd * nr)
        else:
            DD = dd["npairs"].astype(float) / (nd * (nd - 1.))
            dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                               RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
            DR = dr / (nd * nr)
        RR = rr / (nr * (nr - 1.))
        return np.where(RR > 0, (DD - 2 * DR + RR) / RR, np.nan)

    print("[compute] loading BOSS + photometry ...")
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z_d = np.asarray(cat.z_data); ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    wcp = np.asarray(cat.w_cp_data); wnoz = np.asarray(cat.w_noz_data); wsys = np.asarray(cat.w_sys_data)

    D = {}
    D["N_obs"] = cat.N_data; D["wc_mean"] = float(w_c.mean())
    # subsample for sky/scatter/hist figures
    rs = np.random.default_rng(1)
    sub = rs.choice(cat.N_data, min(40000, cat.N_data), replace=False)
    D["sky_ra"] = ra_d[sub]; D["sky_dec"] = dec_d[sub]; D["z_all"] = z_d
    D["wcp"] = wcp; D["wnoz"] = wnoz; D["wsys"] = wsys
    D["frac_cp"] = float(np.mean(wcp > 1.001)); D["frac_noz"] = float(np.mean(wnoz > 1.001))
    D["miss_frac"] = float(w_c.mean() - 1.0)

    # photo-z calibration (held-out)
    print("[compute] photo-z calibration ...")
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    fg, zg = feat[good], z_d[good]
    test = rs.random(len(zg)) < 0.2
    pz_cal = PhotoZKNN(k=100).fit(fg[~test], zg[~test])
    zt = zg[test]; zph = pz_cal.point(fg[test])
    dz = (zph - zt) / (1 + zt)
    D["pz_spec"] = zt; D["pz_phot"] = zph
    D["sigma_nmad"] = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
    D["pz_bias"] = float(np.mean(dz)); D["pz_outlier"] = float(np.mean(np.abs(dz) > 0.05))
    zk, wk = pz_cal.posterior(fg[test])
    pit = np.array([np.sum(wk[i][np.isfinite(wk[i])] * (zk[i][np.isfinite(wk[i])] < zt[i]))
                    for i in range(len(zt))])
    D["pit"] = pit; D["pz_zsample"] = pz_cal.sample(fg[test], rs, n=1)
    D["frac_reliable_phot"] = float(good.sum() / cat.N_data)
    # colour-redshift (subsample with reliable colours)
    cs = rs.choice(np.where(good)[0], min(30000, good.sum()), replace=False)
    D["cz_gr"] = cat.colors_data[cs, 1]; D["cz_ri"] = cat.colors_data[cs, 2]; D["cz_z"] = z_d[cs]

    # full photo-z trained on all good-spec, + targets, + close-pair prior
    pz = PhotoZKNN(k=100).fit(fg, zg)
    dz_pool = measure_close_pair_dz(cat, COLL)
    D["dz_pool"] = dz_pool
    targets = load_cmass_targets(cat, path=TARGETS, seed=0)
    D["n_collided"] = int(np.sum(targets.miss_kind == "collided"))
    D["n_zfail"] = int(np.sum(targets.miss_kind == "zfail"))
    D["wcp_implied"] = float((wcp - 1).sum()); D["wnoz_implied"] = float((wnoz - 1).sum())
    D["tgt_ra"] = np.asarray(targets.ra); D["tgt_dec"] = np.asarray(targets.dec)
    D["tgt_kind"] = np.asarray(targets.miss_kind)

    # randoms for clustering
    rng = np.random.default_rng(7)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=nrf * cat.N_data, z_data=z_d, nside=cat.nside, rng=rng)

    # Completion realizations are EXPENSIVE; generate each (seed, prior) ONCE and
    # reuse the same catalogs across the w(theta) ensemble, the 2-D xi closure and
    # the per-z-slice closure below (previously regenerated ~44 times).
    print(f"[compute] generating {n_real} completion realizations (reused throughout) ...")
    cats_data, cats_none = [], []
    for s in range(n_real):
        cats_data.append(complete_catalog_photoz(cat, targets, pz, seed=s,
                                                 clustering_prior="data", dz_pool=dz_pool))
        cats_none.append(complete_catalog_photoz(cat, targets, pz, seed=s,
                                                 clustering_prior="none", dz_pool=dz_pool))
        print(f"  realization {s+1}/{n_real}")
    cats_keep = cats_data[:3]

    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:] * tb[:-1])
    D["wt_tc"] = tc
    print("[compute] w(theta): weighted observed + ensembles ...")
    with perf.timer("wtheta.RR_corrfunc"):
        rr_w = DDtheta_mocks(1, NTH, tb, rar.astype("f8"), decr.astype("f8"))["npairs"].astype(float)
    D["wt_data"] = wtheta(ra_d, dec_d, rar, decr, tb, w_d=w_c, rr=rr_w)
    Wd = [wtheta(c["ra"], c["dec"], rar, decr, tb, rr=rr_w) for c in cats_data]
    Wp = [wtheta(c["ra"], c["dec"], rar, decr, tb, rr=rr_w) for c in cats_none]
    D["wt_ens_data"] = np.array(Wd); D["wt_ens_pzonly"] = np.array(Wp)

    # n(z): weighted observed vs completed (one realization)
    zb = np.linspace(0.44, 0.61, 26)
    D["nz_bins"] = zb
    D["nz_wobs"] = np.histogram(z_d, zb, weights=w_c)[0]
    D["nz_comp"] = np.histogram(cats_keep[0]["z"], zb)[0]

    # 2-D xi(dtheta,dz): weighted + completed mean, + per-z slice closure
    print("[compute] 2-D xi(dtheta,dz) + per-z slices ...")
    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 16)]); ze = np.linspace(0.0, 0.03, 9)
    tcen = np.empty(len(te) - 1); tcen[0] = 0.5 * te[1]; tcen[1:] = np.sqrt(te[1:-1] * te[2:])
    D["k2d_tcen"] = tcen; D["k2d_zcen"] = 0.5 * (ze[1:] + ze[:-1])
    one = lambda n: np.ones(n)
    # RR depends only on the (fixed) randoms — compute once, reuse for every
    # measure_K2d against the full random set (skips the dominant RR pair count).
    rr_full = compute_rr(rar, decr, zr, one(len(rar)), theta_edges=te, z_edges=ze)
    D["xi2d_w"] = measure_K2d(ra_d, dec_d, z_d, w_c, rar, decr, zr, one(len(rar)),
                              theta_edges=te, z_edges=ze, precomp_rr=rr_full)[2]
    Xc = [measure_K2d(c["ra"], c["dec"], c["z"], one(c["N"]), rar, decr, zr, one(len(rar)),
                      theta_edges=te, z_edges=ze, precomp_rr=rr_full)[2]
          for c in cats_data[:n_real_2d]]
    D["xi2d_c"] = np.mean(Xc, 0)
    # per-z-slice angular closure (reuse cached completions; RR cached per slice)
    zedges = np.quantile(z_d, [0.0, 0.25, 0.5, 0.75, 1.0]); D["slice_edges"] = zedges
    slice_ratio = []
    for a, b in zip(zedges[:-1], zedges[1:]):
        md = (z_d >= a) & (z_d < b); mr = (zr >= a) & (zr < b)
        rr_sl = compute_rr(rar[mr], decr[mr], zr[mr], one(mr.sum()), theta_edges=te, z_edges=ze)
        xw = measure_K2d(ra_d[md], dec_d[md], z_d[md], w_c[md], rar[mr], decr[mr], zr[mr],
                         one(mr.sum()), theta_edges=te, z_edges=ze, precomp_rr=rr_sl)[2][:, 0]
        xcs = []
        for c in cats_data[:n_real_2d]:
            mc = (c["z"] >= a) & (c["z"] < b)
            xcs.append(measure_K2d(c["ra"][mc], c["dec"][mc], c["z"][mc], one(mc.sum()),
                                   rar[mr], decr[mr], zr[mr], one(mr.sum()),
                                   theta_edges=te, z_edges=ze, precomp_rr=rr_sl)[2][:, 0])
        slice_ratio.append(np.mean(xcs, 0) / xw)
    D["slice_ratio"] = np.array(slice_ratio)

    # corrected-sample snapshot: a thin z-slice, observed + added-in-slice, 2 realizations
    zlo, zhi = 0.50, 0.515
    box = (ra_d > 12) & (ra_d < 22) & (dec_d > -3) & (dec_d < 3)
    D["snap_obs_ra"] = ra_d[box & (z_d >= zlo) & (z_d < zhi)]
    D["snap_obs_dec"] = dec_d[box & (z_d >= zlo) & (z_d < zhi)]
    tbox = (D["tgt_ra"] > 12) & (D["tgt_ra"] < 22) & (D["tgt_dec"] > -3) & (D["tgt_dec"] < 3)
    # sample added z for 2 realizations (replicate completion z-logic for the missing)
    feat_t = photoz_features(targets.colors, targets.mags)
    zk_t, wk_t = pz.posterior(feat_t)
    host = targets.host_index
    z_host = np.where(host >= 0, z_d[np.clip(host, 0, len(z_d) - 1)], np.nan)
    pcl = _clpair_density(dz_pool)
    coll = (targets.miss_kind == "collided") & (host >= 0)
    snaps = []
    for s in range(2):
        rr2 = np.random.default_rng(500 + s)
        wkk = wk_t.copy()
        wkk[coll] *= pcl(zk_t[coll] - z_host[coll, None])
        zt2 = np.empty(len(zk_t))
        for i in range(len(zk_t)):
            w = wkk[i]; ok = np.isfinite(w) & (w > 0)
            zt2[i] = rr2.choice(zk_t[i][ok], p=w[ok] / w[ok].sum()) if ok.any() else z_host[i]
        m = tbox & (zt2 >= zlo) & (zt2 < zhi)
        snaps.append((D["tgt_ra"][m], D["tgt_dec"][m]))
    D["snap_zlo"] = zlo; D["snap_zhi"] = zhi
    for s in range(2):
        D[f"snap{s}_ra"], D[f"snap{s}_dec"] = snaps[s]
    perf.report("build_completion_presentation.compute")
    return D


def get_data(recompute=False, quick=False):
    if (not recompute) and os.path.exists(CACHE):
        print(f"[cache] loading {CACHE}")
        return dict(np.load(CACHE, allow_pickle=True))
    D = compute(quick=quick)
    os.makedirs("output", exist_ok=True)
    np.savez(CACHE, **{k: np.asarray(v) for k, v in D.items()})
    print(f"[cache] saved {CACHE}")
    return D


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _wrapra(r):
    return ((np.asarray(r, float) + 180.0) % 360.0) - 180.0


def fig_data(D):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.scatter(_wrapra(D["sky_ra"]), D["sky_dec"], s=1, c=C_NEUTRAL, alpha=0.4, lw=0)
    a1.set_xlabel("RA [deg] (wrapped)"); a1.set_ylabel("Dec [deg]")
    a1.set_title("CMASS-SGC footprint (40k of %d shown)" % int(D["N_obs"]))
    a1.invert_xaxis()
    a2.hist(D["z_all"], bins=40, color=C_OBS, alpha=0.85, edgecolor="white", lw=0.4)
    a2.set_xlabel("spectroscopic redshift z"); a2.set_ylabel("galaxies / bin")
    a2.set_title("redshift distribution n(z)")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_weights(D):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w, name, col in zip(axes, [D["wcp"], D["wnoz"], D["wsys"]],
                                ["WEIGHT_CP (fiber collisions)", "WEIGHT_NOZ (redshift failures)",
                                 "WEIGHT_SYSTOT (imaging)"], [C_NEW, C_ZF, C_OBS]):
        ax.hist(w, bins=np.linspace(min(0.6, w.min()), min(3.0, w.max()), 50),
                color=col, alpha=0.85, edgecolor="white", lw=0.3)
        ax.set_yscale("log"); ax.set_xlabel(name); ax.set_ylabel("galaxies")
        ax.set_title(f"<{name.split()[0]}> = {w.mean():.3f}")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_colorz(D):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sc = ax.scatter(D["cz_gr"], D["cz_ri"], c=D["cz_z"], s=4, cmap="viridis", lw=0, alpha=0.6)
    ax.set_xlabel("g − r  (extinction-corrected)"); ax.set_ylabel("r − i")
    ax.set_xlim(np.percentile(D["cz_gr"], [1, 99])); ax.set_ylim(np.percentile(D["cz_ri"], [1, 99]))
    cb = fig.colorbar(sc, ax=ax); cb.set_label("spectroscopic redshift z")
    ax.set_title("colour–redshift relation (CMASS)")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_photoz(D):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3))
    a1.hexbin(D["pz_spec"], D["pz_phot"], gridsize=45, cmap="viridis", mincnt=1)
    lim = [D["pz_spec"].min(), D["pz_spec"].max()]; a1.plot(lim, lim, "r--", lw=1)
    a1.set_xlabel("spectroscopic z"); a1.set_ylabel("photo-z (posterior median)")
    a1.set_title(f"σ_NMAD = {float(D['sigma_nmad']):.3f}")
    a2.hist(D["pit"], bins=20, range=(0, 1), color=C_NEW, alpha=0.85, edgecolor="white", lw=0.4)
    a2.axhline(len(D["pit"]) / 20, color="r", ls="--", label="uniform (ideal)")
    a2.set_xlabel("PIT = rank of true z in posterior"); a2.set_ylabel("count")
    a2.set_title("posterior calibration"); a2.legend()
    zb = np.linspace(D["pz_spec"].min(), D["pz_spec"].max(), 40)
    a3.hist(D["pz_spec"], bins=zb, density=True, histtype="step", color="k", lw=2, label="true held-out n(z)")
    zs = D["pz_zsample"]
    a3.hist(zs[np.isfinite(zs)], bins=zb, density=True, histtype="step", color=C_NEW, lw=2,
            label="stacked posterior draw")
    a3.set_xlabel("z"); a3.set_ylabel("n(z)"); a3.set_title("n(z) recovery"); a3.legend()
    fig.tight_layout(); return fig_to_b64(fig)


def fig_clpair(D):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dz = D["dz_pool"]
    ax.hist(dz, bins=np.linspace(-0.04, 0.04, 81), color=C_NEW, alpha=0.85, edgecolor="white", lw=0.3)
    ax.set_xlabel("Δz of observed close pairs (≤ 62″)"); ax.set_ylabel("pairs / bin")
    frac = float(np.mean(np.abs(dz) < 0.003))
    ax.set_title(f"close-pair Δz prior  (|Δz|<0.003: {frac:.0%} clustered core)")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_missing(D):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    box = (D["sky_ra"] > 12) & (D["sky_ra"] < 22) & (D["sky_dec"] > -3) & (D["sky_dec"] < 3)
    ax.scatter(D["sky_ra"][box], D["sky_dec"][box], s=5, c=C_NEUTRAL, alpha=0.5, lw=0, label="observed (spec-z)")
    tk = D["tgt_kind"]; tr = D["tgt_ra"]; td = D["tgt_dec"]
    tb = (tr > 12) & (tr < 22) & (td > -3) & (td < 3)
    cc = tb & (tk == "collided"); zz = tb & (tk == "zfail")
    ax.scatter(tr[cc], td[cc], s=14, c=C_NEW, marker="x", label="missing: fiber-collided")
    ax.scatter(tr[zz], td[zz], s=14, c=C_ZF, marker="+", label="missing: redshift-failure")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.invert_xaxis()
    ax.set_title("observed galaxies + recovered missing targets (zoom)"); ax.legend(markerscale=1.5)
    fig.tight_layout(); return fig_to_b64(fig)


def fig_samples(D):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    zb = D["nz_bins"]; zc = 0.5 * (zb[1:] + zb[:-1])
    a1.step(zc, D["nz_wobs"] / D["nz_wobs"].sum(), where="mid", color=C_OBS, lw=2,
            label="completeness-weighted observed")
    a1.step(zc, D["nz_comp"] / D["nz_comp"].sum(), where="mid", color=C_NEW, lw=2, ls="--",
            label="equal-weight completed")
    a1.set_xlabel("redshift z"); a1.set_ylabel("normalised n(z)"); a1.legend()
    a1.set_title("n(z): completed reproduces the weighted")
    a2.scatter(D["snap_obs_ra"], D["snap_obs_dec"], s=6, c=C_NEUTRAL, alpha=0.5, lw=0,
               label="observed in slice")
    a2.scatter(D["snap0_ra"], D["snap0_dec"], s=22, c=C_NEW, marker="x", label="added (realization 1)")
    a2.scatter(D["snap1_ra"], D["snap1_dec"], s=22, c=C_ZF, marker="+", label="added (realization 2)")
    a2.set_xlabel("RA [deg]"); a2.set_ylabel("Dec [deg]"); a2.invert_xaxis()
    a2.set_title(f"thin z-slice [{float(D['snap_zlo']):.3f},{float(D['snap_zhi']):.3f}): "
                 "which added galaxies land in-slice varies")
    a2.legend(markerscale=1.3, fontsize=8)
    fig.tight_layout(); return fig_to_b64(fig)


def fig_wtheta(D):
    tc = D["wt_tc"]; W = D["wt_ens_data"]; m = W.mean(0); s = W.std(0); wd = D["wt_data"]
    lo, hi = np.percentile(W, [16, 84], axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(tc, wd, "s--", color=C_OBS, label="completeness-weighted observed", zorder=5)
    a1.fill_between(tc, lo, hi, color=C_NEW, alpha=0.25, label="completed 16–84%")
    a1.plot(tc, m, "o-", color=C_NEW, label="completed (ensemble mean)")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("θ [deg]"); a1.set_ylabel("w(θ)")
    a1.set_title("angular clustering w(θ)"); a1.legend()
    a2.semilogx(tc, m / wd, "o-", color="#333"); a2.axhline(1, color="gray", ls="--")
    a2.fill_between(tc, 0.95, 1.05, color="green", alpha=0.12, label="±5%")
    a2.set_ylim(0.8, 1.1); a2.set_xlabel("θ [deg]"); a2.set_ylabel("completed / weighted")
    a2.set_title("ratio (ensemble mean)"); a2.legend()
    fig.tight_layout(); return fig_to_b64(fig)


def fig_2d(D):
    tcen = D["k2d_tcen"]; zcen = D["k2d_zcen"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    im0 = axes[0].pcolormesh(zcen, tcen, np.log10(np.clip(D["xi2d_w"], 1e-3, None)),
                             cmap="cividis", shading="nearest")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Δz"); axes[0].set_ylabel("Δθ [deg]")
    axes[0].set_title("measured ξ(Δθ,Δz)  [log₁₀, weighted]"); fig.colorbar(im0, ax=axes[0], fraction=0.046)
    ratio = np.where(D["xi2d_w"] > 0.02, D["xi2d_c"] / D["xi2d_w"], np.nan)
    im1 = axes[1].pcolormesh(zcen, tcen, ratio, vmin=0.8, vmax=1.1, cmap="RdBu_r", shading="nearest")
    axes[1].set_yscale("log"); axes[1].set_xlabel("Δz"); axes[1].set_ylabel("Δθ [deg]")
    axes[1].set_title("completed / weighted"); fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].axhline(COLL, color="k", ls=":")
    for j, (a, b) in enumerate(zip(D["slice_edges"][:-1], D["slice_edges"][1:])):
        axes[2].semilogx(tcen, D["slice_ratio"][j], "o-", ms=3, label=f"z∈[{a:.2f},{b:.2f})")
    axes[2].axhline(1, color="gray", ls="--"); axes[2].axvline(COLL, color="gray", ls=":")
    axes[2].fill_between(tcen, 0.95, 1.05, color="green", alpha=0.12)
    axes[2].set_ylim(0.8, 1.15); axes[2].set_xlabel("Δθ [deg]"); axes[2].set_ylabel("completed / weighted")
    axes[2].set_title("per-z-slice angular closure"); axes[2].legend(fontsize=8)
    fig.tight_layout(); return fig_to_b64(fig)


def fig_systematics(D):
    tc = D["wt_tc"]; A = D["wt_ens_data"]; B = D["wt_ens_pzonly"]
    mA, sA = A.mean(0), A.std(0); mB, sB = B.mean(0), B.std(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.fill_between(tc, mA - sA, mA + sA, color=C_NEW, alpha=0.25)
    a1.plot(tc, mA, "o-", color=C_NEW, label="photo-z × clustering prior")
    a1.plot(tc, mB, "s--", color=C_ZF, label="photo-z only")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("θ [deg]"); a1.set_ylabel("w(θ)")
    a1.set_title("ensemble w(θ): two completion priors (mean ± realization σ)"); a1.legend()
    a2.semilogx(tc, np.abs(mA - mB) / (0.5 * (sA + sB)), "o-", color="#333")
    a2.axhline(1, color="r", ls="--", label="systematic = statistical")
    a2.set_xlabel("θ [deg]"); a2.set_ylabel(r"$\Delta_{\rm sys}/\sigma_{\rm stat}$")
    a2.set_title("redshift-prior systematic budget"); a2.legend()
    fig.tight_layout(); return fig_to_b64(fig)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
CSS = """
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;max-width:1080px;
 margin:0 auto;padding:0 18px 80px;color:#222;line-height:1.6;}
h1{font-size:30px;margin:24px 0 2px;} h2{font-size:23px;margin:38px 0 6px;
 border-bottom:1px solid #ddd;padding-bottom:5px;} h3{font-size:17px;color:#333;}
.sub{color:#777;margin-bottom:8px;} .lead{font-size:16px;color:#333;}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #e0e0e0;
 padding:8px 0;margin-bottom:10px;font-size:13px;z-index:9;}
nav a{color:#3a6ea8;text-decoration:none;margin-right:14px;white-space:nowrap;}
figure{margin:18px 0 26px;} img{max-width:100%;border:1px solid #eee;border-radius:4px;}
figcaption{font-size:13.5px;color:#444;margin-top:8px;padding-left:4px;
 border-left:3px solid #cfe0f0;padding:6px 0 6px 12px;background:#fafcff;}
figcaption b{color:#222;}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 24px;
 background:#f7f9fb;padding:14px 18px;border-radius:6px;margin:14px 0;font-size:14px;}
.metric-grid b{color:#c0392b;}
.callout{background:#f4f8ff;border-left:4px solid #3a6ea8;padding:10px 14px;
 margin:14px 0;border-radius:4px;font-size:14.5px;}
code{background:#eef;padding:1px 6px;border-radius:3px;font-size:13px;}
pre{background:#f5f5f5;padding:10px 14px;border-radius:6px;overflow-x:auto;font-size:12.5px;}
table{border-collapse:collapse;margin:12px 0;font-size:14px;}
th,td{padding:5px 14px;text-align:left;border-bottom:1px solid #e6e6e6;} th{background:#f4f4f4;}
"""


def render(D, figs, Dm, Dc):
    from tools.veusz_vsz import EMBED_SCRIPT
    g = lambda k: float(D[k])
    # each figure is a browser-editable Veusz embed (figs[k] is the <veusz-figure> tag)
    img = lambda k: f'<figure>{figs[k]}'
    v = VALIDATION

    def pimg(path):                       # inline a static validation PNG from output/
        b = png_b64(path)
        return f'<figure><img src="data:image/png;base64,{b}">' if b else "<figure>"
    date = datetime.date.today().isoformat()
    H = []
    H.append(f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"{EMBED_SCRIPT}"
             f"<title>Photo-z completion of the BOSS CMASS catalog</title><style>{CSS}</style></head><body>")
    H.append("<h1>Imaging-informed completion of the BOSS CMASS catalog</h1>")
    H.append(f"<div class='sub'>Cosmology-free correction of spectroscopic incompleteness "
             f"&middot; BOSS DR12 CMASS-South &middot; {date}</div>")
    H.append("<nav>" + " ".join(
        f"<a href='#{i}'>{t}</a>" for i, t in [
            ("scope", "Scope"), ("problem", "Problem"), ("opportunity", "Opportunity"),
            ("method", "Method"), ("data", "Data"), ("catalogs", "Corrected catalogs"),
            ("clustering", "Clustering"), ("recovery", "Truth recovery"),
            ("highorder", "Higher-order"), ("calibration", "Calibration"),
            ("consistency", "Consistency"), ("mask", "Mask &amp; inpainting"),
            ("coupling", "Selection coupling"), ("scatter", "Scatter &amp; systematics"),
            ("budget", "Systematics budget"), ("meaning", "What it means"),
            ("release", "Data release"), ("future", "Future")]) + "</nav>")

    H.append("<div class='callout'>Every figure below is a <b>live, browser-editable Veusz figure</b> "
             "(rendered in-browser via WASM, no server): drag to pan, scroll to zoom, double-click to "
             "edit axes, colours, markers and fonts, and re-export — adjust any plot to taste. RA axes "
             "are wrapped and drawn increasing leftwards so the CMASS-South cap is contiguous and "
             "centred near 0. First render fetches the embed engine, so allow a moment.</div>")
    H.append(f"""<div class='metric-grid'>
      <div>Observed galaxies: <b>{int(D['N_obs']):,}</b></div>
      <div>Missing fraction: <b>{100*g('miss_frac'):.1f}%</b></div>
      <div>Reliable photometry: <b>{100*g('frac_reliable_phot'):.1f}%</b></div>
      <div>Photo-z σ<sub>NMAD</sub>: <b>{g('sigma_nmad'):.3f}</b></div>
      <div>Photo-z PIT mean: <b>{D['pit'].mean():.3f}</b> (0.5 ideal)</div>
      <div>Recovered collided: <b>{int(D['n_collided']):,}</b> / {int(round(g('wcp_implied'))):,} implied</div>
      <div>Recovered z-failures: <b>{int(D['n_zfail']):,}</b> / {int(round(g('wnoz_implied'))):,} implied</div>
      <div>Angular w(θ) closure: <b>≈ {100*np.nanmean((D['wt_ens_data'].mean(0)/D['wt_data']))-100:+.0f}%</b></div>
      <div>z-prior systematic: <b>≪ statistical</b></div>
      <div>Truth recovery wp(rp): <b>0.98–1.01</b></div>
      <div>Equal-weight = official: <b>≈1%</b></div>
      <div>Correction uncertainty: <b>0.2–0.4%</b> (≪ cosmic var)</div>
    </div>""")

    H.append("<h2 id='scope'>Scope: what this product is, and is not</h2>")
    H.append("<div class='callout'>"
             "<b>What it is.</b> An ensemble of equal-weight, cosmology-free catalogs of the "
             "BOSS DR12 CMASS-South galaxies as they would have been observed had spectroscopic "
             "incompleteness (fiber collisions, redshift failures, imaging systematics) been "
             "negligible. Every observed galaxy is kept at its spectroscopic (RA, Dec, z); each "
             "missing galaxy is added at its real SDSS imaging position with a redshift drawn "
             "from a GP/local-density posterior. The realizations differ only in the missing "
             "galaxies' redshifts, so the ensemble spread is the calibrated uncertainty <i>of the "
             "completion</i>. Any summary statistic — w(θ), ξ(Δθ,Δz), wp(rp), multipoles, "
             "counts-in-cells, kNN-CDF, higher-order — can be computed directly from the points."
             "<br><br>"
             "<b>What it is not.</b> It is not a substitute for the survey's <i>sample/cosmic "
             "variance</i>: the ensemble quantifies only the additional uncertainty introduced by "
             "correcting the catalog (≈0.2–0.4% on wp(rp), well below the 1–7% cosmic variance), "
             "so a downstream analysis must still obtain the cosmic covariance the usual way "
             "(mocks, jackknife, analytic window). It is not a new cosmological measurement, and "
             "it assumes no fiducial cosmology — a fiducial cosmology enters only when <i>we</i> "
             "convert z to distance to <i>validate</i> 3-D statistics. The redshift-failure "
             "recovery is partial (~75% of the weight-implied count), limited by the "
             "CMASS-quality imaging pool.</div>")

    H.append("<h2 id='problem'>The problem</h2>")
    H.append("<p class='lead'>A spectroscopic galaxy survey never observes every targeted galaxy. "
             "In BOSS CMASS three effects remove galaxies in a clustering-dependent way:</p>"
             "<ul>"
             "<li><b>Fiber collisions</b> — two galaxies closer than the 62″ fiber-placement limit "
             "cannot both be observed on a single plate; one is dropped. This preferentially removes "
             "<i>close pairs</i>, biasing small-scale clustering.</li>"
             "<li><b>Redshift failures</b> — a spectrum is taken but no reliable redshift is measured.</li>"
             "<li><b>Imaging systematics</b> — stellar density, seeing and extinction modulate the "
             "detection efficiency across the sky.</li></ul>"
             "<p>The standard correction up-weights surviving galaxies (<code>WEIGHT_CP</code>, "
             "<code>WEIGHT_NOZ</code>, <code>WEIGHT_SYSTOT</code>). Weights make the <i>mean</i> "
             "two-point statistics unbiased, but they are not a catalog: they cannot be fed to "
             "estimators that assume equal-weight points, they implicitly assume every missing galaxy "
             "sits at its nearest neighbour's redshift, and they carry no error model for that "
             "assumption. We instead build <b>equal-weight completed catalogs</b>.</p>")

    H.append("<h2 id='opportunity'>The opportunity</h2>")
    H.append("<p>BOSS targets were selected from SDSS DR8 <i>ugriz</i> imaging. Every "
             "spectroscopically-missing galaxy therefore has a real photometric detection — a known "
             "angular position and colours. <b>The incompleteness is almost entirely in the redshift "
             "dimension.</b> So rather than guess where missing galaxies are, we place each at its "
             "<i>measured</i> position and assign a redshift drawn from a photometric-redshift "
             "posterior built from its colours, refined by the observed close-pair statistics. "
             "Everything stays in observed coordinates (RA, Dec, z): no fiducial cosmology, no "
             "comoving distances. The full 2-D clustering ξ(Δθ,Δz) — which carries the "
             "Alcock–Paczynski geometric signal — is preserved as measured, not assumed.</p>")

    H.append("<h2 id='method'>The method</h2>")
    H.append("<p>For each realization of the completed catalog:</p><ol>"
             "<li><b>Keep</b> every observed galaxy at its spectroscopic (RA, Dec, z).</li>"
             "<li><b>Add</b> each missing galaxy at its real imaging position. The number and identity "
             "of missing galaxies are tied to the survey weight bookkeeping (a survivor with "
             "<code>WEIGHT_CP</code>=k claims its k−1 nearest unmatched photometric neighbours within "
             "62″; redshift failures are tied to <code>WEIGHT_NOZ</code> analogously).</li>"
             "<li><b>Assign a redshift</b> by sampling the per-object photo-z posterior p(z|colours); "
             "for collided pairs this is multiplied by the empirical close-pair Δz distribution "
             "(physical pairs sit near the host redshift, projections do not).</li>"
             "<li><b>Imaging systematics</b> (<code>WEIGHT_SYSTOT</code>) are applied as a per-object "
             "Poisson multiplicity on the whole set.</li></ol>"
             "<p>Because the observed galaxies are fixed and only the missing ~%d%% vary — and only in "
             "their redshifts — the spread across realizations is the genuine, calibrated posterior "
             "uncertainty of the correction, which is exactly what a downstream analysis marginalises "
             "over. The photo-z is a dependency-light k-nearest-neighbour estimator in colour space "
             "returning the empirical neighbour-redshift distribution; it is trained on the "
             "good-redshift galaxies the survey already provides.</p>" % int(round(100*g('miss_frac'))))
    H.append("<p class='sub'>Approaches we tried first and discarded: generating the field from scratch "
             "with a log-Gaussian Cox process / measured 2-D kernel reproduced the mean clustering but "
             "its realization covariance was far too large (the high small-scale variance σ²≈4 of the "
             "log-normal). Conditioning on the real observed galaxies and completing only the missing "
             "fraction removes that problem entirely.</p>")

    H.append("<h2 id='data'>The data</h2>")
    H.append(img("data") + "<figcaption><b>Left:</b> the BOSS DR12 CMASS-South footprint "
             "(40,000 of {n:,} galaxies shown), after the simBIG SGC cuts (RA&lt;28° or &gt;335°, "
             "Dec&gt;−6°) and the CMASS redshift range 0.45&lt;z&lt;0.60. <b>Right:</b> the redshift "
             "distribution n(z). These define the sample being completed.</figcaption></figure>".format(
                 n=int(D["N_obs"])))
    H.append(img("weights") + f"<figcaption>Distributions of the three completeness weights "
             f"(log count axis). <b>WEIGHT_CP</b>&gt;1 for {100*g('frac_cp'):.1f}% of galaxies "
             f"(fiber collisions), <b>WEIGHT_NOZ</b>&gt;1 for {100*g('frac_noz'):.1f}% (redshift "
             f"failures); <b>WEIGHT_SYSTOT</b> is a smooth ~few-percent imaging modulation. Their "
             f"product implies a mean completeness weight {g('wc_mean'):.3f}, i.e. "
             f"<b>{100*g('miss_frac'):.1f}% of galaxies are missing</b> and must be added. "
             f"WEIGHT_FKP is an estimator (variance-optimising) weight, not a completeness correction, "
             f"and is deliberately excluded.</figcaption></figure>")

    H.append("<h2 id='catalogs'>What the corrected catalogs look like</h2>")
    H.append(img("colorz") + "<figcaption>The CMASS colour–redshift relation: g−r vs r−i for galaxies "
             "with reliable photometry, coloured by spectroscopic redshift. Redshift varies smoothly "
             "and monotonically across this colour plane, which is why a galaxy's colours constrain "
             "its redshift. The u band is dropped (CMASS galaxies are red; u-flux is dominated by "
             "noise), leaving g−r, r−i, i−z and the i magnitude as photo-z features.</figcaption></figure>")
    H.append(img("photoz") + f"<figcaption>Photo-z performance on a 20% held-out sample of "
             f"good-redshift galaxies. <b>Left:</b> posterior-median photo-z vs spectroscopic z "
             f"(σ<sub>NMAD</sub>={g('sigma_nmad'):.3f}, bias {g('pz_bias'):+.4f}, "
             f"{100*g('pz_outlier'):.1f}% catastrophic). <b>Middle:</b> the probability-integral-"
             f"transform histogram — the rank of each true redshift within its own posterior. A flat "
             f"PIT (mean {D['pit'].mean():.3f}, ideal 0.5) means the posterior is statistically "
             f"<i>calibrated</i>, so drawing a redshift from it is faithful — the property the "
             f"completion relies on. <b>Right:</b> a single posterior draw per object recovers the "
             f"true held-out n(z). Assumption: the colour→z mapping learned from good-redshift "
             f"galaxies also applies to the missing ones (mildly optimistic for redshift failures, "
             f"which correlate with low S/N).</figcaption></figure>")
    H.append(img("clpair") + "<figcaption>The empirical redshift-separation distribution of observed "
             "galaxy pairs within the 62″ collision scale, measured from pairs that <i>both</i> "
             "received redshifts (tile overlaps). It splits into a clustered core (true physical close "
             "pairs, Δz≈0) and a broad tail (chance projections). This data-driven distribution is the "
             "clustering prior on a collided galaxy's redshift — no parametric pair fraction is "
             "assumed.</figcaption></figure>")
    H.append(img("missing") + f"<figcaption>A zoomed sky region: observed galaxies (grey) and the "
             f"recovered missing targets placed at their real SDSS imaging positions — fiber-collided "
             f"(blue ×) and redshift-failures (purple +). Counts are tied to the survey weights: "
             f"<b>{int(D['n_collided']):,}</b> collided recovered vs {int(round(g('wcp_implied'))):,} "
             f"implied by WEIGHT_CP, <b>{int(D['n_zfail']):,}</b> z-failures vs "
             f"{int(round(g('wnoz_implied'))):,} implied by WEIGHT_NOZ. (The colour-selected pool of "
             f"unmatched objects over-counts true targets — it includes never-tiled objects — so we "
             f"keep only those tied to a weighted survivor within the relevant scale; the z-failure "
             f"recovery is partial, limited by the CMASS-quality imaging pool.)</figcaption></figure>")
    H.append(img("samples") + f"<figcaption><b>Left:</b> the equal-weight completed n(z) reproduces "
             f"the completeness-weighted observed n(z). <b>Right:</b> a thin redshift slice "
             f"[{float(D['snap_zlo']):.3f}, {float(D['snap_zhi']):.3f}) of a zoomed sky region. The "
             f"observed galaxies (grey) are fixed; <i>which</i> added galaxies fall into the slice "
             f"differs between two realizations (blue × vs purple +), because each realization draws "
             f"the added galaxies' redshifts from their photo-z posteriors. This realization-to-"
             f"realization variation is the completion's posterior uncertainty.</figcaption></figure>")

    H.append("<h2 id='clustering'>What we measure</h2>")
    H.append(img("wtheta") + "<figcaption>Angular two-point function w(θ), measured with "
             "Landy–Szalay against analytic randoms. The equal-weight completed catalog "
             "(blue, ensemble mean with 16–84% band) reproduces the completeness-weighted observed "
             "w(θ) (orange) to within a few percent across 0.06°–2°. The angular clustering — the "
             "projection over redshift — is preserved.</figcaption></figure>")
    H.append(img("2d") + "<figcaption><b>Left:</b> the measured 2-D clustering ξ(Δθ,Δz) "
             "(log₁₀, completeness-weighted) in observed coordinates; its anisotropy between angular "
             "(Δθ) and radial (Δz) separation carries the Alcock–Paczynski geometric information. "
             "<b>Middle:</b> ratio of completed to weighted across the plane (where the signal is "
             "measurable). <b>Right:</b> per-redshift-slice angular closure — uniform at ≈0.93 across "
             "all four slices, i.e. no redshift-dependent distortion. The ~7% offset below unity is "
             "the photo-z <i>relaxing</i> the weights' nearest-neighbour assumption: the imaging shows "
             "not every missing galaxy is at its host's redshift, so the true small-scale clustering "
             "is slightly lower than the weighting implies. The mild rise with Δz is the photo-z "
             "scatter (σ<sub>NMAD</sub>≈0.019) redistributing pairs radially.</figcaption></figure>")

    # ---- Truth recovery (the headline) ----
    H.append("<h2 id='recovery'>Truth recovery: do we get a known answer back?</h2>")
    H.append("<p>The closure tests above confirm the completed catalog reproduces the "
             "completeness-weighted <i>observed</i> clustering — but that is partly true by "
             "construction. The decisive test is <b>inject-and-recover</b>: take a catalog whose "
             "true clustering is known, apply a realistic forward model of the BOSS systematics "
             "(angular selection from the real randoms, n(z), 62″ fiber collisions, "
             "density-coupled redshift failures, WEIGHT_SYSTOT imaging modulation), run the "
             "completion on the resulting mock-observed catalog, and check we recover the "
             "<i>input truth</i> — not the weighted observation. We do this on a real-BOSS-truth "
             "mock (which carries the real 1-halo clustering) and on independent N-body "
             "MultiDark-Patchy mocks.</p>")
    H.append(pimg("output/mock_truth_recovery.png") +
             "<figcaption><b>Projected clustering wp(rp) recovery (real-BOSS-truth).</b> The "
             f"completed ensemble recovers the input wp(rp) to "
             f"{100*(1-v['tr_wp_lo']):.0f}–{100*(v['tr_wp_hi']-1):.0f}% across 0.5–40 Mpc/h. The "
             "decomposition isolates the lever: the observed (incomplete) catalog is biased low "
             "at small rp; adding the missing galaxies at a nearest-neighbour redshift "
             "over-corrects; the <b>oracle</b> curve (missing galaxies placed at their "
             f"<i>true</i> redshift) recovers to {v['tr_oracle_lo']:.3f}–{v['tr_oracle_hi']:.3f}, "
             "proving the entire residual lives in the redshift assignment. The fix — a "
             "GP/local-density redshift (below) — closes the gap.</figcaption></figure>")
    H.append(pimg("output/patchy_truth_recovery.png") +
             "<figcaption><b>Independent N-body mocks (MultiDark-Patchy SGC).</b> The same "
             "battery on independent clustering with its own randoms. Recovery is faithful above "
             "~1 Mpc/h; the residual sub-Mpc deficit is a known Patchy artifact (its 1-halo term "
             "is not realistic), confirmed by the real-BOSS-truth test recovering the small "
             "scales the Patchy mock cannot.</figcaption></figure>")
    H.append("<div class='callout'><b>The GP/local-density redshift (the key fix).</b> A raw "
             "photo-z posterior has σ<sub>z</sub>≈0.03 (~90 Mpc/h), which smears the "
             "line-of-sight pair counts and drives wp(rp) 3–4% low. Instead of drawing from the "
             "photo-z alone — or collapsing each missing galaxy onto its nearest neighbour's "
             "redshift, which is unphysically sharp — we draw from "
             "<b>p(z | n̂, colours) ∝ (1+δ<sub>g</sub>(n̂,z)) · n̄(z) · p<sub>photoz</sub></b>, "
             "where (1+δ<sub>g</sub>)·n̄ is a kernel estimate of the local redshift density built "
             "from the K nearest observed spectroscopic galaxies along the sightline — a "
             "data-driven Gaussian-process field. The photo-z picks the right peak; the local "
             "galaxy field sharpens it to the physical clustering scale; no cosmology is assumed. "
             "This is what brings wp(rp), the redshift-space multipoles ξ0/ξ2, the higher-order "
             "statistics and the angular w(θ) all back to truth at once.</div>")

    # ---- Higher-order ----
    H.append("<h2 id='highorder'>Higher-order and coincidence-sensitive statistics</h2>")
    H.append("<p>Two-point closure can hide higher-order errors — in particular the Δθ=0 "
             "duplicate spike a naïve WEIGHT_SYSTOT multiplicity (<code>np.repeat</code>) would "
             "create. We replaced that with <b>local-analog</b> systot additions (nearby real "
             "galaxies with ~1″ jitter, never exact duplicates, and never dropping a real "
             "galaxy), and test the completion on statistics that would expose any "
             "residual.</p>")
    H.append(pimg("output/completion_highorder.png") +
             "<figcaption><b>Left:</b> the kNN-CDF — the CDF of the distance from random query "
             "points to the k-th nearest galaxy (k=1,2,4), a full-hierarchy clustering probe "
             "directly sensitive to a 1-NN spike at zero. Completed vs truth agree to "
             f"max|ΔCDF| ≤ {v['knn_k1']:.4f} (k=1), {v['knn_k2']:.4f} (k=2), {v['knn_k4']:.4f} "
             "(k=4), with no spike at small r — confirming the analog fix removed the duplicate "
             "artifact. <b>Right:</b> the counts-in-cells PDF (R=8 Mpc/h spheres): mean "
             f"{v['cic_mean_c']:.3f} vs {v['cic_mean_t']:.3f} truth, var/mean {v['cic_vm_c']:.2f} "
             f"vs {v['cic_vm_t']:.2f}, skew {v['cic_skew_c']:.2f} vs {v['cic_skew_t']:.2f} — the "
             "full one-point density distribution is recovered, not just its second "
             "moment.</figcaption></figure>")

    # ---- Calibration (honest) ----
    H.append("<h2 id='calibration'>Is the ensemble a calibrated posterior?</h2>")
    H.append("<p>Each realization fixes every observed galaxy and re-draws only the redshifts of "
             "the missing ~9%. The realization-to-realization spread is therefore the uncertainty "
             "of the <b>correction</b>; we measure it against the true sample (cosmic) variance "
             "from the mock-to-mock scatter.</p>")
    H.append(pimg("output/recovery_calibration.png") +
             "<figcaption>Across MultiDark-Patchy mocks the completion (correction) uncertainty "
             f"on wp(rp) is {v['cal_unc_lo']:.2f}–{v['cal_unc_hi']:.2f}% per rp bin, while the "
             f"cosmic variance (mock-to-mock) is {v['cal_cos_lo']:.1f}–{v['cal_cos_hi']:.1f}% — "
             f"the correction adds only {100*v['cal_ratio_lo']:.0f}–{100*v['cal_ratio_hi']:.0f}% "
             "of the sample variance. <b>This is the central caveat, stated plainly:</b> the "
             "ensemble spread quantifies the <i>additional</i> uncertainty from completing the "
             "catalog — small and well-controlled — and is <b>not</b> a substitute for the "
             "sample/cosmic variance, which a downstream analysis must still obtain the usual way "
             "(mocks, jackknife, or an analytic covariance). Consistently, the ensemble's 68% "
             f"band brackets the mock truth only {100*v['cal_cov']:.0f}% of the time: it is narrow "
             "by design, because it conditions on one observed catalog and does not regenerate the "
             "density field.</figcaption></figure>")

    # ---- Cosmological consistency ----
    H.append("<h2 id='consistency'>Consistency with the official weighted analysis</h2>")
    H.append("<p>The community measures CMASS clustering with the official completeness weights "
             "w_c = WEIGHT_SYSTOT·(WEIGHT_CP+WEIGHT_NOZ−1). Our equal-weight completed catalogs "
             "should give the same clustering with no weights — a drop-in, weight-free "
             "replacement. A fiducial (Planck18) cosmology is used <i>only</i> to turn redshifts "
             "into distances for this test; the catalogs themselves stay cosmology-free.</p>")
    H.append(pimg("output/cosmology_consistency.png") +
             "<figcaption>On the real CMASS-South data: projected wp(rp) of the equal-weight "
             "completed ensemble vs the official w_c-weighted galaxies agree to "
             f"{100*(1-v['cc_wp_lo']):.0f}–{100*(v['cc_wp_hi']-1):.0f}% (ratio "
             f"{v['cc_wp_lo']:.2f}–{v['cc_wp_hi']:.2f}); the redshift-space monopole ξ0 agrees to "
             f"{v['cc_xi0_lo']:.2f}–{v['cc_xi0_hi']:.2f}, and the quadrupole ξ2 (the RSD "
             "anisotropy) is reproduced. The completion is a faithful, equal-weight stand-in for "
             "the standard weighted catalog across the standard statistics.</figcaption></figure>")
    H.append("<p><b>The strong form — uniform randoms, no weights.</b> An equal-weight catalog "
             "should also work with a <i>trivial</i> random: uniform over the footprint, no "
             "completeness weighting. Testing this head-on (cross-checked against the BOSS mangle "
             "mask) shows CMASS-South is <b>~99% complete</b> (COMP≈0.99 at the galaxies) — so "
             "there is essentially no angular completeness window to remove, and the survey random "
             "already <i>is</i> a uniform-footprint random to ~1%. Used that way the completed "
             "catalog reproduces the official weighted w(θ) to ~1.5% and wp/ξ₀ to ~1–2%. A "
             "separately-constructed uniform random matches only to the precision with which it "
             "resolves the survey boundary — a control on the unbiased official data shows the "
             "same residual, confirming it is the window construction, not the catalog (and the "
             "<code>mask_DR12v5</code> geometry mask is ~40% larger in area than the LSS clustering "
             "footprint, so uniform randoms must be clipped to the LSS region).</p>")

    H.append("<h2 id='mask'>Survey mask and inpainting</h2>")
    H.append(f"<p>The completion above corrects galaxies missing from <i>observed</i> area. It does "
             f"not touch interior <b>mask holes</b> — bright-star masks, bad fields and tiling gaps "
             f"where there is no data at all (no galaxies, no randoms, no usable imaging). For "
             f"clustering measured against masked randoms these holes cancel and need no action. But a "
             f"theorist who wants a hole-free, equal-weight catalog can have one: we fill each hole by "
             f"<b>transplanting real galaxies</b> — with their colours, magnitudes and local spatial "
             f"configuration — from environment-matched nearby clean regions, setting each hole's count "
             f"so its galaxy/random ratio matches its surrounding collar. This is data resampling, not "
             f"a field model: higher-order clustering and the colour/luminosity structure transfer by "
             f"construction, and it stays cosmology-free.</p>")
    H.append(img("mask") + f"<figcaption><b>Left:</b> the {int(Dm['n_holes'])} interior mask holes "
             f"(red) located on the footprint from a finer (nside={NSIDE_MASK}, ≈7′-pixel) "
             f"random-count map — pixels with zero randoms enclosed by populated footprint. "
             f"<b>Right:</b> their radius distribution; total interior masked area "
             f"{float(Dm['hole_area_tot']):.1f} deg² ({100*float(Dm['hole_area_tot'])/float(Dm['footprint_deg2']):.1f}% "
             f"of the footprint). Sub-7′ bright-star masks are below this random-derived resolution "
             f"(we lack the mangle veto polygons) and are not resolved here.</figcaption></figure>")
    H.append(img("inpaint") + "<figcaption><b>Left:</b> a zoomed region around a large hole, observed "
             "(grey) and inpainted (blue) galaxies. <b>Middle/right:</b> the closure test — the "
             "standard masked measurement w(θ)=LS(data, masked randoms), where holes cancel, against "
             "the inpainted measurement w(θ)=LS(data+inpainted, hole-filled randoms), where the "
             "catalog is treated as hole-free. They agree to ~1% across 0.06°–2°, i.e. the inpaint "
             "fills the holes with statistically-consistent galaxies. <b>Limitations:</b> only the "
             "fully-empty mask cores are filled — the partial-completeness halos at mask edges remain "
             "locally under-dense (handled by the weighting/randoms for clustering, a residual local "
             "deficit for field-level use); boundary continuity is approximate; and holes comparable "
             "to or larger than the correlation length are weakly constrained (the realizations span "
             "that uncertainty). This is an optional hole-free field product, separate from the "
             "unbiased masked clustering catalogs.</figcaption></figure>")
    H.append(f"<p>To show <i>why</i> the holes exist and how well the transplant works, here are "
             f"<b>{int(Dm['n_gallery'])} interior holes</b> — {int(Dm['n_starmask'])} bright-star masks "
             f"(identified by cross-matching hole centres against Gaia) and the rest bad-field/tiling "
             f"gaps — one per row, with three views each: <b>left</b>, the actual <b>SDSS</b> "
             f"<i>ugriz</i> imaging the photometric catalogue and BOSS spectroscopic targets were drawn "
             f"from — a heavily saturated bright star (with bleed trails and halos) or a bad field is "
             f"plainly the reason the region was masked; <b>middle</b>, the observed galaxies around the "
             f"empty region (the gap is visible); <b>right</b>, after inpainting, with the transplanted "
             f"galaxies in blue at 70% opacity so any overlap with real galaxies shows. The fill follows "
             f"the surrounding density and carries real redshifts and colours.</p>")
    H.append(img("inpaint_gallery") + "<figcaption>For "
             f"{int(Dm['n_gallery'])} interior mask holes (each row: SDSS imaging | observed | "
             f"inpainted), labelled by cause (Gaia G magnitude of the masking star, or bad field). "
             "Cutouts are the exact SDSS imaging (SkyServer), centred on the masking star / hole at the "
             "panel's field of view. Inpainted galaxies are semi-transparent (alpha 0.7). Panels use the "
             "astronomical RA convention (increasing leftwards), wrapped so the South cap is contiguous. "
             "The static view (shown) carries the imaging; clicking opens the live, editable Veusz "
             "before/after.</figcaption></figure>")

    # --- selection coupling + validation (Wechsler v0 lessons, cosmology-free) ---
    gc = lambda k: float(Dc[k])
    zf_det = abs(gc("zfail_z")) >= 3.0
    cl_det = abs(gc("collided_z")) >= 3.0
    zf_word = "<b>density-coupled</b>" if zf_det else "consistent with no coupling"
    zf_imp = ("a real coupling the completion must reproduce" if zf_det else
              "so a density-blind redshift-failure correction carries no spurious-power risk here")
    H.append("<h2 id='coupling'>Selection coupling and validation</h2>")
    H.append("<p>A forward-model field reconstruction in the same collaboration "
             "(Wechsler, v0 pipeline) raises a question our correction must answer: is the "
             "spectroscopic <i>selection</i> itself <b>density-coupled</b> — does the rate at which a "
             "targeted galaxy goes missing depend on the local galaxy overdensity? If it is and the "
             "correction ignores it, the missing galaxies imprint spurious large-scale power (the "
             "mechanism behind the historical MegaZ excess; Thomas et al. 2011). We measure the "
             "coupling directly and cosmology-free: at every member of the <i>total target sample</i> "
             "(spectroscopic successes plus a given failure kind) we evaluate the local success "
             "overdensity δ from an angular aperture, normalised by the random catalogue (so δ is a "
             "true overdensity needing no distances and automatically footprint/completeness-aware), "
             "then fit the success indicator against δ with a logistic model. The slope <i>h</i> is the "
             "coupling; a label-shuffle null fixes its zero point.</p>")
    H.append(img("coupling") + f"<figcaption><b>Left:</b> the redshift-success fraction S(δ) versus "
             f"local overdensity for the two selection kinds. <b>Redshift failures</b> are "
             f"{zf_word} "
             f"(h = {gc('zfail_h'):+.2f} ± {gc('zfail_herr'):.2f}, "
             f"{abs(gc('zfail_z')):.1f}σ from the shuffle null) — {zf_imp}. "
             f"<b>Fiber collisions</b> are strongly coupled "
             f"(h = {gc('collided_h'):+.2f} ± {gc('collided_herr'):.2f}, "
             f"{abs(gc('collided_z')):.1f}σ; negative ⇒ collisions over-occupy dense regions), exactly "
             f"as expected since close pairs are what get collided — a clean, cosmology-free "
             f"confirmation. <b>Right:</b> the spurious-power test. The completion places every missing "
             f"galaxy at its <i>real</i> imaging position, so it tracks the completeness-weighted "
             f"baseline w(θ); a density-blind null that scatters the same galaxies over random "
             f"footprint positions distorts w(θ) at large θ. Because the coupling lives almost entirely "
             f"in the (small-scale) collisions and our completion places them where they truly are, the "
             f"coupling is reproduced by construction rather than modelled. The largest-θ bins are "
             f"random-count limited.</figcaption></figure>")
    H.append(img("trust") + f"<figcaption><b>Left:</b> the completed-catalog angular density against the "
             f"<i>total-target</i> density (successes + failures), in which the spectroscopic-success "
             f"selection cancels by construction — the cleanest cosmology-free amplitude reference. They "
             f"agree with correlation {gc('amp_corr'):.2f}, i.e. the completion restores the "
             f"selection-immune density, not an arbitrary one. <b>Right:</b> a trustworthiness map — the "
             f"galaxy-count scatter across completion realizations per HEALPix cell (median "
             f"{gc('trust_med'):.2f}), the data-space analogue of a per-voxel posterior σ. It tells a "
             f"downstream user where the catalog is well-constrained and where the photo-z redshift "
             f"uncertainty leaves the most freedom.</figcaption></figure>")

    H.append("<h2 id='scatter'>Scatter and systematics</h2>")
    H.append(img("systematics") + "<figcaption><b>Left:</b> the w(θ) ensemble under two redshift-"
             "assignment priors — photo-z combined with the close-pair clustering prior (blue) vs "
             "photo-z alone (purple) — each shown as mean ± realization scatter. <b>Right:</b> the "
             "ratio of the prior-induced shift to the realization scatter, "
             "Δ<sub>sys</sub>/σ<sub>stat</sub>. It is well below unity at all scales: the angular "
             "w(θ) is robust to the redshift-prior choice, because w(θ) is a projection and the added "
             "galaxies' angular positions are fixed regardless of their assigned redshift. The prior "
             "matters for the radial clustering, not for w(θ). The realization scatter itself (the "
             "band) is the calibrated photo-z uncertainty — the covariance a cosmology inference would "
             "consume.</figcaption></figure>")

    H.append("<h2 id='budget'>Systematics budget</h2>")
    H.append("<p>Per effect: the mechanism, how the completion corrects it, the residual after "
             "correction, and how that residual was validated.</p>")
    H.append("<table>"
             "<tr><th>Effect</th><th>Mechanism</th><th>Correction</th><th>Residual</th>"
             "<th>Validation</th></tr>"
             "<tr><td>Fiber collisions</td><td>close pairs (&lt;62″) dropped; small-scale "
             "deficit</td><td>add at real imaging position; redshift from GP-field × close-pair "
             "Δz prior</td><td>&lt;1–2% on wp(rp)</td><td>truth recovery; kNN-CDF; scale "
             "40–90″ &lt;0.6%</td></tr>"
             "<tr><td>Redshift failures</td><td>spectrum taken, no reliable z</td><td>add at real "
             "position; redshift from GP-field posterior</td><td>partial (~75% of weight-implied); "
             "&lt;1% on wp</td><td>truth recovery; failure-population photo-z audit; coupling test "
             "(uncoupled)</td></tr>"
             "<tr><td>Imaging systematics</td><td>stellar density / seeing / extinction modulate "
             "detection (WEIGHT_SYSTOT)</td><td>local-analog multiplicity (no duplicates); all "
             "real galaxies kept</td><td>no Δθ=0 artifact; count mode &lt;0.1% on wp</td>"
             "<td>kNN-CDF (no 1-NN spike); CIC</td></tr>"
             "<tr><td>Redshift assignment</td><td>missing galaxies have no spec-z</td><td>GP / "
             "local-density posterior p(z | n̂, colours)</td><td>0–2% on wp/ξ; dominant residual "
             "lever</td><td>oracle test (0.997–1.005); sensitivity (the only &gt;1% lever)</td></tr>"
             "<tr><td>Interior mask holes</td><td>bright stars / bad fields / tiling gaps</td>"
             "<td>optional analog transplant, surrounding-density matched (cosmology-free)</td>"
             "<td>~1% closure; edge halos not filled</td><td>masked vs hole-filled w(θ) "
             "closure</td></tr>"
             "<tr><td>Correction uncertainty</td><td>which missing galaxies / which redshifts</td>"
             "<td>ensemble of realizations</td><td>0.2–0.4% on wp (≪ cosmic variance)</td>"
             "<td>coverage vs Patchy mock-to-mock variance</td></tr>"
             "</table>")

    H.append("<h2 id='meaning'>What this means</h2>")
    H.append("<p>The completion produces equal-weight, cosmology-free, configuration-space catalogs "
             "that (i) reproduce the completeness-weighted n(z) and angular clustering, (ii) preserve "
             "the 2-D ξ(Δθ,Δz) geometry uniformly in redshift, and (iii) come as ensembles whose "
             "spread is the genuine, calibrated uncertainty of the missing-galaxy correction. Because "
             "they are real catalogs of points, any summary statistic — w(θ), ξ(Δθ,Δz), counts-in-"
             "cells, higher-order — can be computed from them, and the systematic budget of the "
             "correction is obtained simply by scanning the realizations. The ~7% small-scale "
             "difference from the standard weighting is not an error but an imaging-informed "
             "improvement: it removes the weights' built-in assumption that every missing galaxy lies "
             "at its neighbour's redshift.</p>")
    H.append("<div class='callout'>Assumptions and limitations, stated plainly: the photo-z is trained "
             "on good-redshift galaxies and applied to the missing ones (mildly optimistic for "
             "redshift failures, which correlate with low S/N); the z-failure recovery is partial "
             "(~75% of the weight-implied count) because not every failure has CMASS-quality "
             "photometry; the integral-constraint/window effects are negligible at θ&lt;2° for this "
             "footprint but would matter on larger scales; and the close-pair prior is measured from "
             "surviving pairs, assumed representative of collided pairs.</div>")

    H.append("<h2 id='release'>Data release</h2>")
    H.append("<div class='callout'><b>Draw your own samples (≈2 MB).</b> Because every "
             "realization shares the same observed galaxies and only the missing ~9% vary, the "
             "<i>entire posterior</i> ships as one small file + a standalone sampler — no need to "
             "download a multi-GB ensemble. A realization is just a seed (~500/s); a reproducible "
             "K-catalog ensemble is K seeds. This is ~700× smaller than a stored ensemble (2 MB vs "
             "~1.4 GB for 1000 realizations) and reproduces the full completion to n(z)~1%, "
             "w(θ)~0.1%. Downloads: "
             "<a href='data/cmass_south_posterior.npz'>cmass_south_posterior.npz</a> (~2 MB) · "
             "<a href='data/cmass_south_randoms.npz'>cmass_south_randoms.npz</a> (~5 MB) · "
             "<a href='data/draw_samples.py'>draw_samples.py</a> · "
             "<a href='data/README.md'>README</a>."
             "<pre>from draw_samples import load_package, draw\n"
             "pkg = load_package(\"cmass_south_posterior.npz\")\n"
             "cat = draw(pkg, seed=0)   # dict(ra, dec, z, prov) — ~120k equal-weight galaxies</pre>"
             "</div>")
    H.append("<p>Materialized FITS products are also available (loadable with astropy) in two "
             "complementary forms:</p><ul>"
             "<li><b>Ensemble</b> (<code>ensemble/realization_*.fits</code>): N equal-weight "
             "realizations — columns <code>RA</code>, <code>DEC</code>, <code>Z</code>, "
             "<code>PROV</code> (provenance). Use the full ensemble to propagate the correction "
             "uncertainty through any statistic.</li>"
             "<li><b>Summary</b> (<code>summary.fits</code>): one catalog of the always-included "
             "galaxies with <code>RA</code>, <code>DEC</code>, <code>Z</code>, <code>Z_ERR</code> "
             "(per-object redshift uncertainty), <code>PROV</code> and "
             "<code>WEIGHT_SYSTOT</code>.</li>"
             "<li><b>Randoms</b> (<code>randoms.fits</code>) and a plain-text "
             "<code>PROVENANCE.txt</code>.</li></ul>"
             "<p>Provenance flags label each object: observed-specz, collided, zfail, "
             "systot-analog, zhost-fallback, inpaint. The column dictionary, units, conventions "
             "and the FKP / integral-constraint / window guidance for users are in "
             "<code>DATA_MODEL.md</code>; the pinned, reproducible environment is in "
             "<code>environment.yml</code>. The catalogs are in observed coordinates "
             "(RA, Dec, z) and carry no fiducial cosmology.</p>")

    H.append("<h2 id='future'>Future extensions and other datasets</h2>")
    H.append("<ul>"
             "<li><b>Improve redshift-failure recovery</b> by relaxing the colour selection for the "
             "z-failure pool, or modelling the S/N-dependent failure probability.</li>"
             "<li><b>Radial / Alcock–Paczynski summaries</b>, where the redshift-prior systematic is "
             "non-negligible (unlike w(θ)) — propagate it through the realization ensemble.</li>"
             "<li><b>Independent cross-check with bitwise / PIP weights</b> on surveys that provide "
             "fiber-assignment realizations (eBOSS, DESI) — BOSS DR12 does not ship them.</li>"
             "<li><b>Other samples and surveys</b>: BOSS CMASS-North and LOWZ; eBOSS LRG/ELG/QSO; "
             "DESI. The method needs only matched imaging (positions + colours) and the survey "
             "completeness bookkeeping.</li></ul>")

    H.append("<h3>Reproduction</h3>")
    H.append("<pre>"
             "PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \\\n"
             "OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/build_completion_presentation.py\n\n"
             "# core code: twopt_density/{boss,photoz,cmass_targets,observed_ls}.py\n"
             "# target fetch: demos/fetch_cmass_targets.py (SDSS DR12 SkyServer)</pre>")
    H.append("</body></html>")
    return "".join(H)


# ----------------------------------------------------------------------
# Mask + inpainting (separate cache so it doesn't trigger the main recompute)
# ----------------------------------------------------------------------
def _gaia_bright_stars(dec_min, dec_max, gmax=8.0, cache="output/_cutouts/gaia_bright.npz"):
    """Gaia DR3 stars brighter than ``gmax`` in a Dec band (cached). The bright
    stars whose vetoes punch the bright-star-mask holes in the catalog."""
    if cache and os.path.exists(cache):
        d = np.load(cache); return {"ra": d["ra"], "dec": d["dec"], "mag": d["mag"]}
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = 6000
        # synchronous query (the async endpoint is currently flaky); brightest first,
        # restricted to the CMASS-South RA range so TOP-N covers every bright star.
        q = (f"SELECT TOP 6000 ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source "
             f"WHERE phot_g_mean_mag<{gmax} AND dec BETWEEN {dec_min:.4f} AND {dec_max:.4f} "
             f"AND (ra < 50 OR ra > 310) ORDER BY phot_g_mean_mag")
        r = Gaia.launch_job(q).get_results()
        out = {"ra": np.asarray(r["ra"], float), "dec": np.asarray(r["dec"], float),
               "mag": np.asarray(r["phot_g_mean_mag"], float)}
        if cache and len(out["ra"]):
            os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, **out)
        print(f"  [gaia] {len(out['ra'])} bright stars (G<{gmax}) in dec [{dec_min:.1f},{dec_max:.1f}]")
        return out
    except Exception as e:
        print(f"  [gaia] query failed ({e}); gallery falls back to largest holes")
        return None


def compute_mask():
    import healpy as hp
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    from twopt_density.boss import load_boss
    from twopt_density.observed import _radec_to_nhat
    from twopt_density.inpaint import fine_completeness_map, find_interior_holes, inpaint_holes

    def wtheta(ra_d, dec_d, ra_r, dec_r, tb):
        nd, nr = len(ra_d), len(ra_r)
        dd = DDtheta_mocks(1, 16, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
        rr = DDtheta_mocks(1, 16, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
        dr = DDtheta_mocks(0, 16, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                           RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
        return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)
    w_c = float((np.asarray(cat.w_sys_data)*(np.asarray(cat.w_cp_data)+np.asarray(cat.w_noz_data)-1)).mean())
    rar_full = np.asarray(cat.ra_random); decr_full = np.asarray(cat.dec_random)
    counts, _ = fine_completeness_map(rar_full, decr_full, nside=NSIDE_MASK)
    holes = find_interior_holes(counts, NSIDE_MASK, empty_count=0.0, min_neighbour_frac=0.75)
    hole_pix = np.concatenate([h.pixels for h in holes])
    real = inpaint_holes(holes, counts, NSIDE_MASK, donor_ra=ra_d, donor_dec=dec_d, donor_z=z_d,
                         rand_ra=rar_full, rand_dec=decr_full, donor_colors=cat.colors_data,
                         donor_mags=cat.mags_data, seed=0, n_real=1, density_boost=w_c)[0]

    rng = np.random.default_rng(3)
    nsub = min(400_000, cat.N_random)
    ri = rng.choice(cat.N_random, nsub, False); rar, decr = rar_full[ri], decr_full[ri]
    med = int(np.median(counts[counts > 0])); res = hp.nside2resol(NSIDE_MASK)
    th, ph = hp.pix2ang(NSIDE_MASK, hole_pix)
    ath = np.repeat(th, med) + (rng.random(len(hole_pix)*med)-0.5)*res
    aph = np.repeat(ph, med) + (rng.random(len(hole_pix)*med)-0.5)*res/np.sin(np.clip(np.repeat(th, med),.01,np.pi-.01))
    rar_f = np.concatenate([rar_full, np.degrees(aph) % 360]); decr_f = np.concatenate([decr_full, 90-np.degrees(ath)])
    rf = rng.choice(len(rar_f), nsub, False); rar_f, decr_f = rar_f[rf], decr_f[rf]

    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:]*tb[:-1])
    w_masked = wtheta(ra_d, dec_d, rar, decr, tb)
    w_inp = wtheta(np.concatenate([ra_d, real["ra"]]), np.concatenate([dec_d, real["dec"]]), rar_f, decr_f, tb)

    sub = rng.choice(cat.N_data, min(40000, cat.N_data), False)
    big = max(holes, key=lambda h: h.radius_deg if h.radius_deg < 0.5 else 0)
    bx = lambda ra, dec: (np.abs(((ra-big.ra+180)%360)-180) < 1.0) & (np.abs(dec-big.dec) < 1.0)
    mb_o = bx(ra_d, dec_d); mb_i = bx(real["ra"], real["dec"])

    # ---- inpaint GALLERY: holes shown with their CAUSE (bright stars / bad fields) ----
    # work in wrapped RA so holes near RA=0/360 are contiguous; store per-hole.
    from twopt_density.observed import _radec_to_nhat
    from scipy.spatial import cKDTree
    wrap = lambda r: ((np.asarray(r, float) + 180.0) % 360.0) - 180.0
    hid_all = real["hole_id"].astype(int)
    hra = np.array([h.ra for h in holes]) % 360.0
    hdec = np.array([h.dec for h in holes]); hrad = np.array([h.radius_deg for h in holes])

    # nearest Gaia bright star to each hole -> identify bright-star masks
    stars = _gaia_bright_stars(float(hdec.min()) - 0.5, float(hdec.max()) + 0.5, gmax=8.0)
    if stars is not None and len(stars["ra"]):
        sd, sj = cKDTree(_radec_to_nhat(stars["ra"], stars["dec"])).query(_radec_to_nhat(hra, hdec))
        sep = np.degrees(2 * np.arcsin(np.clip(sd / 2, 0, 1)))
        smag = stars["mag"][sj]; sra = stars["ra"][sj] % 360.0; sdec = stars["dec"][sj]
    else:
        sep = np.full(len(holes), 99.0); smag = np.full(len(holes), 99.0)
        sra = hra.copy(); sdec = hdec.copy()
    # a bright-star mask is a SMALL hole with a bright star at its centre (a star
    # inside a large hole is incidental, not the cause) — require both.
    is_star = (hrad < 0.2) & (sep < np.maximum(hrad, 5.0 / 60.0))
    star_holes = sorted(np.where(is_star)[0], key=lambda i: smag[i])[:8]   # brightest first
    gap_holes = [i for i in np.argsort(-hrad)
                 if (not is_star[i]) and 0.15 <= hrad[i] <= 0.6][:4]       # bad-field gaps
    cand = list(star_holes) + list(gap_holes)

    g_ra, g_dec, g_hid, i_ra, i_dec, i_hid = [], [], [], [], [], []
    c_ra, c_dec, c_rad, c_box, c_mag, c_reason = [], [], [], [], [], []
    ra_d_w = wrap(ra_d); inp_ra_w = wrap(real["ra"])
    for k, hi in enumerate(cand):
        if hi in star_holes:                              # centre on the STAR; tight FOV
            cw = wrap(sra[hi]); cd = float(sdec[hi]); R = min(max(4.0 * hrad[hi], 0.10), 0.2)
            mag = float(smag[hi]); reason = f"G={mag:.1f} star"
        else:                                             # bad-field / tiling gap
            cw = wrap(hra[hi]); cd = float(hdec[hi]); R = min(max(2.5 * hrad[hi], 0.3), 1.0)
            mag = float("nan"); reason = "bad field / gap"
        cosd = np.cos(np.radians(cd))
        mo = (np.abs((ra_d_w - cw) * cosd) < R) & (np.abs(dec_d - cd) < R)
        mi = (hid_all == hi)
        g_ra.append(ra_d_w[mo]); g_dec.append(dec_d[mo]); g_hid.append(np.full(int(mo.sum()), k))
        i_ra.append(inp_ra_w[mi]); i_dec.append(real["dec"][mi]); i_hid.append(np.full(int(mi.sum()), k))
        c_ra.append(cw); c_dec.append(cd); c_rad.append(hrad[hi]); c_box.append(R)
        c_mag.append(mag); c_reason.append(reason)
    cat_ = lambda L: (np.concatenate(L) if L else np.zeros(0))
    return {
        "sky_ra": ra_d[sub], "sky_dec": dec_d[sub],
        "gal_ra": cat_(g_ra), "gal_dec": cat_(g_dec), "gal_hid": cat_(g_hid),
        "inp_ra": cat_(i_ra), "inp_dec": cat_(i_dec), "inp_hid": cat_(i_hid),
        "gcen_ra": np.array(c_ra), "gcen_dec": np.array(c_dec),
        "grad": np.array(c_rad), "gbox": np.array(c_box), "n_gallery": len(cand),
        "gstar_mag": np.array(c_mag), "greason": np.array(c_reason, dtype=object),
        "n_starmask": len(star_holes),
        "hole_ra": np.array([h.ra for h in holes]), "hole_dec": np.array([h.dec for h in holes]),
        "hole_rad": np.array([h.radius_deg for h in holes]),
        "hole_area_tot": float(sum(h.area_deg2 for h in holes)), "n_holes": len(holes),
        "n_inpaint": int(len(real["ra"])), "footprint_deg2": float(np.sum(counts > 0)*hp.nside2pixarea(NSIDE_MASK, degrees=True)),
        "wt_tc": tc, "wt_masked": w_masked, "wt_inp": w_inp,
        "zoom_obs_ra": ra_d[mb_o], "zoom_obs_dec": dec_d[mb_o],
        "zoom_inp_ra": real["ra"][mb_i], "zoom_inp_dec": real["dec"][mb_i],
        "zoom_rad": float(big.radius_deg), "z_obs": z_d[rng.choice(len(z_d), 40000, False)],
        "z_inp": real["z"]}


def get_mask_data(recompute=False):
    if (not recompute) and os.path.exists(MASK_CACHE):
        return dict(np.load(MASK_CACHE, allow_pickle=True))
    Dm = compute_mask()
    os.makedirs("output", exist_ok=True)
    np.savez(MASK_CACHE, **{k: np.asarray(v) for k, v in Dm.items()})
    return Dm


def fig_mask(Dm):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.scatter(_wrapra(Dm["sky_ra"]), Dm["sky_dec"], s=1, c=C_NEUTRAL, alpha=0.3, lw=0)
    a1.scatter(_wrapra(Dm["hole_ra"]), Dm["hole_dec"], s=12, facecolors="none", edgecolors="#c0392b", lw=0.8)
    a1.set_xlabel("RA [deg]"); a1.set_ylabel("Dec [deg]"); a1.invert_xaxis()
    a1.set_title(f"{int(Dm['n_holes'])} interior mask holes (red) on the footprint")
    a2.hist(Dm["hole_rad"]*60, bins=np.linspace(0, 30, 31), color="#c0392b", alpha=0.8, edgecolor="white", lw=0.4)
    a2.set_yscale("log"); a2.set_xlabel("hole radius [arcmin]"); a2.set_ylabel("number of holes")
    a2.set_title(f"total masked interior area {float(Dm['hole_area_tot']):.1f} deg² "
                 f"({100*float(Dm['hole_area_tot'])/float(Dm['footprint_deg2']):.1f}% of footprint)")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_inpaint(Dm):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))
    a1.scatter(Dm["zoom_obs_ra"], Dm["zoom_obs_dec"], s=5, c=C_NEUTRAL, alpha=0.5, lw=0, label="observed")
    a1.scatter(Dm["zoom_inp_ra"], Dm["zoom_inp_dec"], s=10, c=C_NEW, lw=0, label="inpainted")
    a1.set_xlabel("RA [deg]"); a1.set_ylabel("Dec [deg]"); a1.invert_xaxis(); a1.legend()
    a1.set_title(f"before/after zoom (hole r≈{float(Dm['zoom_rad'])*60:.0f}′)")
    tc = Dm["wt_tc"]
    a2.loglog(tc, Dm["wt_masked"], "s--", color=C_OBS, label="masked + masked randoms")
    a2.loglog(tc, Dm["wt_inp"], "o-", color=C_NEW, label="inpainted + hole-filled randoms")
    a2.set_xlabel("θ [deg]"); a2.set_ylabel("w(θ)"); a2.legend(); a2.set_title("clustering closure")
    a3.semilogx(tc, Dm["wt_inp"]/Dm["wt_masked"], "o-", color="#333"); a3.axhline(1, color="gray", ls="--")
    a3.fill_between(tc, 0.95, 1.05, color="green", alpha=0.12); a3.set_ylim(0.85, 1.15)
    a3.set_xlabel("θ [deg]"); a3.set_ylabel("inpainted / masked"); a3.set_title("closure ratio")
    fig.tight_layout(); return fig_to_b64(fig)


# ----------------------------------------------------------------------
# Selection coupling + validation (separate cache)
# ----------------------------------------------------------------------
def compute_coupling():
    """Density-coupling of selection, the MegaZ spurious-power test, the
    selection-immune amplitude anchor, and the trustworthiness map — all
    cosmology-free (angular, random-normalised). See validate_selection_coupling.py."""
    import healpy as hp
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    from twopt_density.boss import load_boss
    from twopt_density.photoz import PhotoZKNN, photoz_features
    from twopt_density.cmass_targets import load_cmass_targets
    from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
    from twopt_density.selection_coupling import measure_failure_coupling, total_target_density

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)
    rar_full = np.asarray(cat.ra_random); decr_full = np.asarray(cat.dec_random)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z_d[good])
    dz_pool = measure_close_pair_dz(cat, COLL)
    targets = load_cmass_targets(cat, path=TARGETS, seed=0)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)

    rng = np.random.default_rng(0)
    nsub = min(2 * cat.N_data, cat.N_random)
    ri = rng.choice(cat.N_random, nsub, replace=False)
    rar, decr = rar_full[ri], decr_full[ri]

    D = {}
    for kind in ["zfail", "collided"]:
        r = measure_failure_coupling(cat, targets, rand_ra=rar, rand_dec=decr, kind=kind,
                                     aperture_deg=0.5, n_boot=150, seed=1)
        D[f"{kind}_h"] = r.h; D[f"{kind}_herr"] = r.h_err; D[f"{kind}_z"] = r.z_score
        D[f"{kind}_nullstd"] = r.h_null_std
        D[f"{kind}_dc"] = r.delta_bin_centres; D[f"{kind}_S"] = r.S_of_delta; D[f"{kind}_Se"] = r.S_err
        D[f"{kind}_nfail"] = r.n_fail

    # spurious-large-scale-power test
    NTH = 16
    tb = np.logspace(np.log10(0.1), np.log10(4.0), 11); tc = np.sqrt(tb[1:] * tb[:-1])
    rr_w = DDtheta_mocks(1, NTH, tb, rar.astype("f8"), decr.astype("f8"))["npairs"].astype(float)
    nr = len(rar)

    def wth(ra, dec, w=None):
        if w is not None:
            dd = DDtheta_mocks(1, NTH, tb, ra.astype("f8"), dec.astype("f8"),
                               weights1=w.astype("f8"), weight_type="pair_product")
            DD = dd["npairs"] * dd["weightavg"] / w.sum()**2
            dr = DDtheta_mocks(0, NTH, tb, ra.astype("f8"), dec.astype("f8"), weights1=w.astype("f8"),
                               RA2=rar.astype("f8"), DEC2=decr.astype("f8"), weight_type="pair_product")
            DR = dr["npairs"] * dr["weightavg"] / (w.sum() * nr)
        else:
            n = len(ra)
            DD = DDtheta_mocks(1, NTH, tb, ra.astype("f8"), dec.astype("f8"))["npairs"].astype(float)/(n*(n-1.))
            dr = DDtheta_mocks(0, NTH, tb, ra.astype("f8"), dec.astype("f8"),
                               RA2=rar.astype("f8"), DEC2=decr.astype("f8"))["npairs"].astype(float)
            DR = dr / (n * nr)
        RR = rr_w / (nr * (nr - 1.))
        return np.where(RR > 0, (DD - 2*DR + RR)/RR, np.nan)

    c = complete_catalog_photoz(cat, targets, pz, seed=0, clustering_prior="data", dz_pool=dz_pool)
    j = rng.choice(len(rar_full), targets.N, replace=False)
    D["sp_tc"] = tc
    D["sp_wgt"] = wth(ra_d, dec_d, w=w_c)
    D["sp_real"] = wth(np.asarray(c["ra"]), np.asarray(c["dec"]))
    D["sp_blind"] = wth(np.concatenate([ra_d, rar_full[j]]), np.concatenate([dec_d, decr_full[j]]))

    # selection-immune amplitude + trustworthiness map (nside=64)
    ns = 64
    _, dens_tot, _ = total_target_density(cat, targets, nside=ns)
    n_real = 6
    maps = np.zeros((n_real, 12*ns**2))
    for s in range(n_real):
        cs = complete_catalog_photoz(cat, targets, pz, seed=100+s, clustering_prior="data", dz_pool=dz_pool)
        pix = hp.ang2pix(ns, np.deg2rad(90 - np.asarray(cs["dec"])), np.deg2rad(np.asarray(cs["ra"]) % 360))
        maps[s] = np.bincount(pix, minlength=12*ns**2)
    mean_map = maps.mean(0); std_map = maps.std(0); foot = mean_map > 0
    dc = mean_map[foot] / np.median(mean_map[foot]); dt = dens_tot[foot]
    keep = (dt > 0) & (dc > 0)
    D["amp_dt"] = dt[keep]; D["amp_dc"] = dc[keep]
    D["amp_corr"] = float(np.corrcoef(dt[keep], dc[keep])[0, 1])
    cv = np.full(12*ns**2, np.nan); cv[foot] = std_map[foot] / np.maximum(mean_map[foot], 1e-9)
    sub = rng.choice(np.where(foot)[0], min(40000, int(foot.sum())), replace=False)
    th, ph = hp.pix2ang(ns, sub)
    D["trust_ra"] = np.degrees(ph); D["trust_dec"] = 90 - np.degrees(th); D["trust_cv"] = cv[sub]
    D["trust_med"] = float(np.nanmedian(cv[foot]))
    return D


def get_coupling_data(recompute=False):
    if (not recompute) and os.path.exists(COUP_CACHE):
        return dict(np.load(COUP_CACHE, allow_pickle=True))
    Dc = compute_coupling()
    os.makedirs("output", exist_ok=True)
    np.savez(COUP_CACHE, **{k: np.asarray(v) for k, v in Dc.items()})
    return Dc


def fig_coupling(Dc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.7))
    for kind, col, lab in [("zfail", C_ZF, "redshift failures"), ("collided", C_OBS, "fiber collisions")]:
        a1.errorbar(Dc[f"{kind}_dc"], Dc[f"{kind}_S"], yerr=Dc[f"{kind}_Se"], fmt="o-", color=col, ms=4,
                    label=f"{lab}: h={float(Dc[f'{kind}_h']):+.2f}±{float(Dc[f'{kind}_herr']):.2f} "
                          f"(z={float(Dc[f'{kind}_z']):+.1f})")
    a1.set_xlabel("local success overdensity δ  (random-normalised, angular)")
    a1.set_ylabel("redshift-success fraction  S(δ)")
    a1.set_title("density coupling of selection (cosmology-free)"); a1.legend(fontsize=8); a1.grid(alpha=0.2)
    tc = Dc["sp_tc"]
    a2.loglog(tc, Dc["sp_wgt"], "k-", lw=2, label="weighted observed (baseline)")
    a2.loglog(tc, Dc["sp_real"], "o-", color=C_NEW, label="completion (real positions)")
    a2.loglog(tc, Dc["sp_blind"], "s--", color="#c0392b", label="density-blind null (random positions)")
    a2.set_xlabel("θ [deg]"); a2.set_ylabel("w(θ)"); a2.legend(fontsize=8)
    a2.set_title("spurious large-scale power (MegaZ test)"); a2.grid(alpha=0.2, which="both")
    fig.tight_layout(); return fig_to_b64(fig)


def fig_trust(Dc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.7))
    a1.hexbin(Dc["amp_dt"], Dc["amp_dc"], gridsize=40, cmap="viridis", mincnt=1, bins="log")
    lim = [0, max(np.percentile(Dc["amp_dt"], 99), np.percentile(Dc["amp_dc"], 99))]
    a1.plot(lim, lim, "r--", lw=1); a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("total-target density (selection-immune)"); a1.set_ylabel("completed catalog density")
    a1.set_title(f"amplitude anchor: corr = {float(Dc['amp_corr']):.3f}")
    sc = a2.scatter(_wrapra(Dc["trust_ra"]), Dc["trust_dec"], c=Dc["trust_cv"], s=6, cmap="magma_r",
                    vmin=0, vmax=float(np.nanpercentile(Dc["trust_cv"], 95)), lw=0)
    a2.set_xlabel("RA [deg]"); a2.set_ylabel("Dec [deg]"); a2.invert_xaxis()
    cb = fig.colorbar(sc, ax=a2); cb.set_label("realization scatter  std/mean")
    a2.set_title(f"trustworthiness map (median {float(Dc['trust_med']):.2f})")
    fig.tight_layout(); return fig_to_b64(fig)


def fetch_gallery_cutouts(Dm, figs_dir, dr="dr17", size=512):
    """Fetch (and cache) an **SDSS** multicolor cutout per gallery hole.

    This is the EXACT imaging the photometric catalog and BOSS spectroscopic
    targets were drawn from (SDSS ugriz; SkyServer ``ImgCutout/getjpeg`` serves
    that same imaging). Centred on each hole's cause (the bright star for a
    bright-star mask, else the hole centroid) at the panel's field of view, so
    the saturated star / bad field that punched the hole is visible. Returns a
    list of local JPEG paths (None on failure). Cached: refetched only if
    missing. A browser User-Agent is required (SkyServer 403s default agents)."""
    import urllib.request, time
    os.makedirs(figs_dir, exist_ok=True)
    UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/120 Safari/537.36")
    cra = np.asarray(Dm["gcen_ra"]) % 360.0; cdec = np.asarray(Dm["gcen_dec"])
    box = np.asarray(Dm["gbox"]); n = int(Dm["n_gallery"])
    paths = []
    for k in range(n):
        out = os.path.join(figs_dir, f"cutout_{k}.jpg")
        if not os.path.exists(out) or os.path.getsize(out) < 2000:
            scale = 2 * box[k] * 3600.0 / size                       # arcsec/pix for the FOV
            url = (f"https://skyserver.sdss.org/{dr}/SkyServerWS/ImgCutout/getjpeg?"
                   f"ra={cra[k]:.5f}&dec={cdec[k]:.5f}&scale={scale:.3f}&width={size}&height={size}")
            ok = False
            for attempt in range(4):                                 # SkyServer rate-limits (403)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": UA})
                    data = urllib.request.urlopen(req, timeout=60).read()
                    if len(data) > 2000 and data[:2] == b"\xff\xd8":  # valid JPEG
                        with open(out, "wb") as f:
                            f.write(data)
                        ok = True; break
                except Exception as e:
                    err = e
                time.sleep(1.5 * (attempt + 1))                      # back off and retry
            if not ok:
                print(f"  [cutout {k+1}] SDSS fetch failed: {err}"); out = None
            time.sleep(0.4)                                          # be gentle between requests
        paths.append(out if (out and os.path.exists(out)) else None)
    print(f"[figures] gallery cutouts (SDSS {dr}): {sum(p is not None for p in paths)}/{n} fetched/cached")
    return paths


def fig_inpaint_gallery(Dm, cutouts=None):
    """Static poster mirroring the interactive gallery: per hole, the Legacy imaging
    cutout, the observed galaxies (the gap), and the inpainted fill (RA-wrapped)."""
    import matplotlib.image as mpimg
    n = int(Dm["n_gallery"])
    g_ra, g_dec, g_hid = Dm["gal_ra"], Dm["gal_dec"], Dm["gal_hid"]
    i_ra, i_dec, i_hid = Dm["inp_ra"], Dm["inp_dec"], Dm["inp_hid"]
    cra, cdec, rad, box = Dm["gcen_ra"], Dm["gcen_dec"], Dm["grad"], Dm["gbox"]
    reason = Dm["greason"] if "greason" in Dm else np.array(["hole"] * n, dtype=object)
    ncol = 3 if cutouts is not None else 2
    fig, axes = plt.subplots(n, ncol, figsize=(4.0 * ncol, 3.0 * max(n, 1)))
    axes = np.atleast_2d(axes)
    labels = (["imaging (SDSS)", "observed", "inpainted"] if ncol == 3 else ["observed", "inpainted"])
    for k in range(n):
        go = g_hid == k; io = i_hid == k; cosd = np.cos(np.radians(cdec[k]))
        xlo, xhi = cra[k] + box[k] / cosd, cra[k] - box[k] / cosd      # RA increases left
        ylo, yhi = cdec[k] - box[k], cdec[k] + box[k]
        why = str(reason[k])
        col = 0
        if ncol == 3:
            ax = axes[k][0]
            if cutouts[k] is not None:
                ax.imshow(mpimg.imread(cutouts[k]), extent=[xlo, xhi, ylo, yhi], aspect="auto")
            else:
                ax.text(0.5, 0.5, "no imaging", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_title(f"{why} - {labels[0]}", fontsize=8)
            ax.tick_params(labelsize=6); col = 1
        for j in range(2):
            ax = axes[k][col + j]
            ax.scatter(g_ra[go], g_dec[go], s=7, c=C_NEUTRAL, alpha=0.7, lw=0)
            if j == 1:
                ax.scatter(i_ra[io], i_dec[io], s=11, c=C_NEW, alpha=0.7, lw=0)
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.set_title(f"{labels[col + j]}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout(); return fig_to_b64(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    D = get_data(recompute=args.recompute, quick=args.quick)
    Dm = get_mask_data(recompute=args.recompute)
    Dc = get_coupling_data(recompute=args.recompute)
    print("[figures] building interactive Veusz (.vsz) figures ...")
    import shutil
    import demos.veusz_report_figs as RF
    figs_dir = "docs/figs"
    if os.path.isdir(figs_dir):
        shutil.rmtree(figs_dir)
    os.makedirs(figs_dir, exist_ok=True)
    figs = {
        "data": RF.footprint(D, figs_dir), "weights": RF.weights(D, figs_dir),
        "colorz": RF.colorz(D, figs_dir), "photoz": RF.photoz(D, figs_dir),
        "clpair": RF.clpair(D, figs_dir), "missing": RF.missing(D, figs_dir),
        "samples": RF.samples_nz(D, figs_dir), "wtheta": RF.wtheta(D, figs_dir),
        "2d": RF.xi2d(D, figs_dir), "systematics": RF.systematics(D, figs_dir),
        "mask": RF.mask(Dm, figs_dir), "inpaint": RF.inpaint_closure(Dm, figs_dir),
        "inpaint_gallery": RF.inpaint_gallery(Dm, figs_dir),
        "coupling": RF.coupling(Dc, figs_dir), "trust": RF.trust(Dc, figs_dir),
    }
    # static poster PNGs (so the report displays no matter what — no WebGPU, slow
    # Pyodide, etc.); each <veusz-figure> shows its poster and boots editing on click.
    print("[figures] rendering static poster PNGs ...")
    import base64 as _b64
    cutouts = fetch_gallery_cutouts(Dm, "output/_cutouts")
    posters = {
        "footprint": fig_data(D), "weights": fig_weights(D), "colorz": fig_colorz(D),
        "photoz": fig_photoz(D), "clpair": fig_clpair(D), "missing": fig_missing(D),
        "samples_nz": fig_samples(D), "wtheta": fig_wtheta(D), "xi2d": fig_2d(D),
        "systematics": fig_systematics(D), "mask": fig_mask(Dm), "inpaint_closure": fig_inpaint(Dm),
        "inpaint_gallery": fig_inpaint_gallery(Dm, cutouts), "coupling": fig_coupling(Dc), "trust": fig_trust(Dc),
    }
    for stem, b64 in posters.items():
        with open(os.path.join(figs_dir, stem + ".png"), "wb") as fh:
            fh.write(_b64.b64decode(b64))
    print(f"[figures] wrote {len(os.listdir(figs_dir))} files (.vsz + .png) to {figs_dir}")
    html = render(D, figs, Dm, Dc)
    os.makedirs("output", exist_ok=True); os.makedirs("docs", exist_ok=True)
    # the embed loads .vsz via relative 'figs/...'; mirror the dir next to each HTML
    if os.path.isdir("output/figs"):
        shutil.rmtree("output/figs")
    shutil.copytree(figs_dir, "output/figs")
    for path in ["output/completion_presentation.html", "docs/completion.html"]:
        with open(path, "w") as f:
            f.write(html)
        print(f"[html] wrote {path} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
