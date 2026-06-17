"""Validate the density-coupling of spectroscopic selection — cosmology-free.

Integrates the lessons of Risa Wechsler's v0 forward-model document into our
data-space completion pipeline, all in observed coordinates (no distances, no
fiducial cosmology):

  (1) COUPLING (headline): measure the density coupling h of redshift failures
      (the S_zsucc analogue) and fiber collisions against the local success
      overdensity δ, with a label-shuffle null. A detection (|h| ≫ null) means
      the selection is density-correlated — the MegaZ spurious-power risk.
  (2) SPURIOUS LARGE-SCALE POWER (the MegaZ test): w(θ) of (a) the weighted
      observed baseline, (b) our completion (missing galaxies at their REAL
      imaging positions), and (c) a density-blind null that places the same
      missing galaxies at RANDOM footprint positions. If the selection is
      coupled, the density-blind null distorts large-scale w(θ) while our
      real-position completion tracks the weighted baseline — i.e. our method
      reproduces the coupling and does not imprint spurious power.
  (3) SELECTION-IMMUNE AMPLITUDE: the completed angular density vs the
      total-target (successes+failures) density, in which selection cancels.
  (4) TRUSTWORTHINESS MAP: per-region scatter of the completion realizations —
      where the catalog is least constrained (Risa's per-voxel σ analogue).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/validate_selection_coupling.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.selection_coupling import (measure_failure_coupling, local_overdensity,
                                              total_target_density)

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
COLL = 62.0 / 3600.0
NTH = 16


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, rr=None):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    if rr is None:
        rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan), rr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/boss/cmass_targets_South.fits")
    p.add_argument("--aperture", type=float, default=0.5)
    p.add_argument("--n-real", type=int, default=8)
    p.add_argument("--n-rand-factor", type=int, default=2)
    p.add_argument("--out", default="output/selection_coupling.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)
    rar_full = np.asarray(cat.ra_random); decr_full = np.asarray(cat.dec_random)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z_d[good])
    dz_pool = measure_close_pair_dz(cat, COLL)
    targets = load_cmass_targets(cat, path=args.targets, seed=0)

    # subsample randoms once for δ and w(θ)
    rng = np.random.default_rng(0)
    nsub = min(args.n_rand_factor * cat.N_data, cat.N_random)
    ri = rng.choice(cat.N_random, nsub, replace=False)
    rar, decr = rar_full[ri], decr_full[ri]

    # ---- (1) coupling h ----
    print("=== density coupling of spectroscopic selection (cosmology-free) ===")
    res = {}
    for kind in ["zfail", "collided"]:
        if not np.any(targets.miss_kind == kind):
            continue
        r = measure_failure_coupling(cat, targets, rand_ra=rar, rand_dec=decr, kind=kind,
                                     aperture_deg=args.aperture, n_boot=150, seed=1)
        res[kind] = r
        print(f"  {kind:9s}: h = {r.h:+.3f} ± {r.h_err:.3f}  "
              f"null = {r.h_null_mean:+.3f} ± {r.h_null_std:.3f}  "
              f"z = {r.z_score:+.1f}  pearson_r = {r.pearson_r:+.3f}  "
              f"(N_succ={r.n_success:,}, N_{kind}={r.n_fail:,})  "
              f"{'DETECTED' if r.detected else 'consistent with 0'}")

    # ---- (2) spurious-large-scale-power test ----
    print("\n=== spurious large-scale power test (MegaZ failure mode) ===")
    tb = np.logspace(np.log10(0.1), np.log10(6.0), 12); tc = np.sqrt(tb[1:] * tb[:-1])
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    _, rr_w = wtheta(ra_d, dec_d, rar, decr, tb)                 # cache RR (fixed randoms)
    # weighted baseline with completeness weights w_c via Corrfunc pair_product
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       weights1=w_c.astype("f8"), weight_type="pair_product")
    nrr = len(rar)
    DDw = dd["npairs"] * dd["weightavg"] / w_c.sum()**2
    drw = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                        weights1=w_c.astype("f8"), RA2=rar.astype("f8"), DEC2=decr.astype("f8"),
                        weight_type="pair_product")
    DRw = drw["npairs"] * drw["weightavg"] / (w_c.sum() * nrr)
    RRw = rr_w / (nrr * (nrr - 1.))
    w_wgt = np.where(RRw > 0, (DDw - 2*DRw + RRw)/RRw, np.nan)

    c = complete_catalog_photoz(cat, targets, pz, seed=0, clustering_prior="data", dz_pool=dz_pool)
    w_real, _ = wtheta(c["ra"], c["dec"], rar, decr, tb, rr=rr_w)

    # density-blind null: same missing galaxies, RANDOM footprint positions
    nmiss = targets.N
    j = rng.choice(len(rar_full), nmiss, replace=False)
    ra_blind = np.concatenate([ra_d, rar_full[j]]); dec_blind = np.concatenate([dec_d, decr_full[j]])
    w_blind, _ = wtheta(ra_blind, dec_blind, rar, decr, tb, rr=rr_w)
    print(f"{'theta':>8}{'w_wgt':>10}{'w_real':>10}{'w_blind':>10}{'real/wgt':>10}{'blind/wgt':>10}")
    for i in range(len(tc)):
        print(f"{tc[i]:8.3f}{w_wgt[i]:10.4f}{w_real[i]:10.4f}{w_blind[i]:10.4f}"
              f"{w_real[i]/w_wgt[i]:10.3f}{w_blind[i]/w_wgt[i]:10.3f}")

    # ---- (3) selection-immune amplitude: completed vs total-target density ----
    ns = 64
    _, dens_tot, _ = total_target_density(cat, targets, nside=ns)
    pix_c = hp.ang2pix(ns, np.deg2rad(90 - c["dec"]), np.deg2rad(c["ra"] % 360))
    dens_c = np.bincount(pix_c, minlength=12*ns**2).astype(float)
    occ = (dens_tot > 0) & (dens_c > 0)
    dc = dens_c[occ] / np.median(dens_c[occ]); dt = dens_tot[occ]
    r_amp = float(np.corrcoef(dc, dt)[0, 1])
    print(f"\nselection-immune amplitude: corr(completed, total-target density) = {r_amp:.3f} "
          f"(nside={ns}, {occ.sum():,} pixels)")

    # ---- (4) trustworthiness map: per-pixel realization scatter ----
    print(f"\ntrustworthiness map: {args.n_real} completion realizations, nside={ns} ...")
    maps = np.zeros((args.n_real, 12*ns**2))
    for s in range(args.n_real):
        cs = complete_catalog_photoz(cat, targets, pz, seed=100+s, clustering_prior="data", dz_pool=dz_pool)
        pix = hp.ang2pix(ns, np.deg2rad(90 - cs["dec"]), np.deg2rad(cs["ra"] % 360))
        maps[s] = np.bincount(pix, minlength=12*ns**2)
    mean_map = maps.mean(0); std_map = maps.std(0)
    foot = mean_map > 0
    cv = np.full(12*ns**2, np.nan); cv[foot] = std_map[foot] / np.maximum(mean_map[foot], 1e-9)
    print(f"  median realization scatter (std/mean) over footprint: {np.nanmedian(cv[foot]):.3f}")

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9))
    a1 = fig.add_subplot(2, 2, 1)
    for kind, col in [("zfail", "#3a6ea8"), ("collided", "#e8853a")]:
        if kind not in res:
            continue
        r = res[kind]
        a1.errorbar(r.delta_bin_centres, r.S_of_delta, yerr=r.S_err, fmt="o-", color=col, ms=4,
                    label=f"{kind}: h={r.h:+.2f}±{r.h_err:.2f} (z={r.z_score:+.1f})")
    a1.set_xlabel("local success overdensity δ (random-normalised, angular)")
    a1.set_ylabel("redshift-success fraction  S(δ)")
    a1.set_title("density coupling of selection (cosmology-free)"); a1.legend(fontsize=8)
    a1.grid(alpha=0.2)

    a2 = fig.add_subplot(2, 2, 2)
    a2.loglog(tc, w_wgt, "k-", lw=2, label="weighted observed (baseline)")
    a2.loglog(tc, w_real, "o-", color="#3a6ea8", label="completion (real positions)")
    a2.loglog(tc, w_blind, "s--", color="#c0392b", label="density-blind null (random positions)")
    a2.set_xlabel("θ [deg]"); a2.set_ylabel("w(θ)")
    a2.set_title("spurious large-scale power (MegaZ test)"); a2.legend(fontsize=8); a2.grid(alpha=0.2, which="both")

    a3 = fig.add_subplot(2, 2, 3)
    a3.hexbin(dt, dc, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    lim = [min(dt.min(), dc.min()), max(np.percentile(dt, 99), np.percentile(dc, 99))]
    a3.plot(lim, lim, "r--", lw=1)
    a3.set_xlabel("total-target density (selection-immune)")
    a3.set_ylabel("completed catalog density")
    a3.set_title(f"amplitude anchor: corr = {r_amp:.3f}")

    a4 = fig.add_subplot(2, 2, 4)
    sub = rng.choice(np.where(foot)[0], min(40000, foot.sum()), replace=False)
    th, ph = hp.pix2ang(ns, sub)
    sky_ra = np.degrees(ph); sky_dec = 90 - np.degrees(th)
    sc = a4.scatter(sky_ra, sky_dec, c=cv[sub], s=6, cmap="magma_r", vmin=0,
                    vmax=np.nanpercentile(cv[foot], 95), lw=0)
    a4.set_xlabel("RA [deg]"); a4.set_ylabel("Dec [deg]"); a4.invert_xaxis()
    cb = fig.colorbar(sc, ax=a4); cb.set_label("realization scatter std/mean")
    a4.set_title("trustworthiness map (per-region uncertainty)")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
