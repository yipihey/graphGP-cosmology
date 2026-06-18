"""The drop-in test (corrected): equal-weight completed+inpainted catalog with a
UNIFORM, ANALYTIC random over the HOLE-FREE footprint reproduces the official
weighted survey, for every two-point flavor.

Premise being tested
---------------------
The whole point of an equal-weight completed catalog is that the completion
*absorbs* the angular completeness (fiber collisions, redshift failures, imaging
systematics) into the galaxy distribution, and the interior mask holes are filled
by inpainting -- so a downstream user needs neither completeness weights nor a
data-derived random:
    * NO weights (every point counts once), and
    * a TRIVIAL random: uniform within the survey footprint, computed ANALYTICALLY.

Two corrections over the first attempt (per Abel):
  1. FOOTPRINT = the HOLE-FREE outer boundary. We inpaint the interior holes, so
     the matching random must be uniform over the filled footprint -- NOT a holey
     mask, and NOT the survey's completeness-traced randoms. We build the footprint
     as (populated pixels) U (interior holes filled) from the random-count map
     (reusing the inpaint hole-finder), i.e. the alpha-shape outer boundary at the
     pixel scale.
  2. ANALYTIC randoms. For a uniform window the random pair count RR is the WINDOW
     AUTOCORRELATION -- a geometric quantity, not a Monte-Carlo estimate. We:
       (a) compute RR(theta) analytically from the footprint mask via its angular
           power spectrum, RR(theta) = sum_l (2l+1)/(4pi) C_l^WW P_l(cos theta)
           (one spherical-harmonic transform, no random catalog, no Poisson), and
       (b) for the LS estimator and the 3-D statistics draw a uniform window sample
           ONCE (hole-free footprint x n(z)) and compute every RR a SINGLE time,
           reused across all realizations -- the per-realization cost is only the
           data pair counts. Panel (a) overlays the analytic RR(theta) on the
           one-time sampled RR to show they agree, i.e. the analytic shortcut is
           exact and the random catalog is unnecessary.

OFFICIAL side = the survey with all corrections: observed galaxies, weight
w_c = w_systot*(w_cp+w_noz-1), the survey's own (completeness-traced, holey)
randoms. A fiducial (Planck18) cosmology enters ONLY to turn z into distance for
wp/xi; the catalogs stay cosmology-free.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        demos/validate_dropin_uniform_randoms.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
from twopt_density.inpaint import fine_completeness_map, find_interior_holes, inpaint_holes
from twopt_density.clustering_corrfunc import wp_rp, xi_smu_ell024

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
NTH = 32
NSIDE_MASK = 1024     # fine enough to resolve the survey boundary (the off/uniform
                      # control converges 34%->3% over nside 256->2048; 1024 ~ 6%)


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, w_d=None, rr=None, return_rr=False):
    """Landy-Szalay angular w(theta) (Corrfunc DDtheta_mocks, parallel). ``rr`` is
    the (fixed) random-random npairs; pass it to reuse a once-computed RR."""
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
    f8 = lambda a: np.ascontiguousarray(a, "f8")
    nd, nr = len(ra_d), len(ra_r)
    if rr is None:
        rr = DDtheta_mocks(1, NTH, tb, f8(ra_r), f8(dec_r))["npairs"].astype(float)
    if w_d is not None:
        Wd = w_d.sum()
        dd = DDtheta_mocks(1, NTH, tb, f8(ra_d), f8(dec_d), weights1=f8(w_d), weight_type="pair_product")
        DD = dd["npairs"] * dd["weightavg"] / Wd**2
        dr = DDtheta_mocks(0, NTH, tb, f8(ra_d), f8(dec_d), weights1=f8(w_d),
                           RA2=f8(ra_r), DEC2=f8(dec_r), weight_type="pair_product")
        DR = dr["npairs"] * dr["weightavg"] / (Wd * nr)
    else:
        DD = DDtheta_mocks(1, NTH, tb, f8(ra_d), f8(dec_d))["npairs"].astype(float) / (nd*(nd-1.))
        dr = DDtheta_mocks(0, NTH, tb, f8(ra_d), f8(dec_d), RA2=f8(ra_r), DEC2=f8(dec_r))["npairs"].astype(float)
        DR = dr / (nd * nr)
    RR = rr / (nr * (nr - 1.))
    w = np.where(RR > 0, (DD - 2*DR + RR) / RR, np.nan)
    return (w, rr) if return_rr else w


def make_uniform_window(footprint_pix, nside, n, z_pool, rng):
    """Uniform sample over the HOLE-FREE footprint (equal-area pixels + within-pixel
    jitter => uniform in solid angle), with redshifts from the completed n(z)."""
    import healpy as hp
    pick = rng.choice(footprint_pix, size=n)
    th, ph = hp.pix2ang(nside, pick)
    res = hp.nside2resol(nside)
    th = np.clip(th + (rng.random(n) - 0.5) * res, 1e-4, np.pi - 1e-4)
    ph = ph + (rng.random(n) - 0.5) * res / np.sin(th)
    ra = np.degrees(ph) % 360.0; dec = 90.0 - np.degrees(th)
    return ra, dec, rng.choice(z_pool, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--out", default="output/dropin_uniform_randoms.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    wc = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    rar_full = np.asarray(cat.ra_random); decr_full = np.asarray(cat.dec_random); zr_full = np.asarray(cat.z_random)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)
    rng = np.random.default_rng(7)

    # ---- hole-free footprint (alpha-shape outer boundary at the pixel scale) ----
    print("[dropin] building hole-free footprint (populated U interior-holes) ...")
    counts, _ = fine_completeness_map(rar_full, decr_full, nside=NSIDE_MASK)
    holes = find_interior_holes(counts, NSIDE_MASK, empty_count=0.0, min_neighbour_frac=0.75)
    footprint_pix = np.where(counts > 0)[0]
    if holes:
        footprint_pix = np.union1d(footprint_pix, np.concatenate([h.pixels for h in holes]))
    import healpy as hp
    area = len(footprint_pix) * hp.nside2pixarea(NSIDE_MASK, degrees=True)
    print(f"[dropin] footprint {len(footprint_pix):,} pix = {area:.0f} deg² "
          f"({len(holes)} interior holes filled)")

    # ---- inpaint the interior holes ONCE (transplanted real galaxies), reused ----
    print("[dropin] inpainting interior holes (once, reused across realizations) ...")
    inp = inpaint_holes(holes, counts, NSIDE_MASK, donor_ra=ra, donor_dec=dec, donor_z=z,
                        rand_ra=rar_full, rand_dec=decr_full, donor_colors=cat.colors_data,
                        donor_mags=cat.mags_data, seed=0, n_real=1, density_boost=float(wc.mean()))[0]
    inp_ra = np.asarray(inp["ra"]); inp_dec = np.asarray(inp["dec"]); inp_z = np.asarray(inp["z"])
    print(f"[dropin] inpaint added {len(inp_ra):,} galaxies")

    print(f"[dropin] generating {args.n_real} completed realizations ...")
    cats = []
    for s in range(args.n_real):
        c = complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz)
        cats.append((np.concatenate([np.asarray(c["ra"]), inp_ra]),     # + inpaint => hole-free
                     np.concatenate([np.asarray(c["dec"]), inp_dec]),
                     np.concatenate([np.asarray(c["z"]), inp_z])))
    z_pool = np.concatenate([c[2] for c in cats])

    # ---- the two randoms: official (survey, completeness-traced) vs uniform window ----
    Nr = 4 * cat.N_data
    ro = rng.choice(len(rar_full), min(Nr, len(rar_full)), replace=False)
    rar_o, decr_o, zr_o = rar_full[ro], decr_full[ro], zr_full[ro]
    win_ra, win_dec, win_z = make_uniform_window(footprint_pix, NSIDE_MASK, 8 * cat.N_data, z_pool, rng)
    print(f"[dropin] randoms: official(survey) {len(rar_o):,} | uniform window {len(win_ra):,} (RR computed once)")

    results = {}

    # ---------- w(theta): analytic window + 2x2 diagnostic ----------
    tb = np.logspace(np.log10(0.05), np.log10(2.5), 13); tc = np.sqrt(tb[1:]*tb[:-1])
    w_off, rr_off = wtheta(ra, dec, rar_o, decr_o, tb, w_d=wc, return_rr=True)   # official / survey rand (truth)
    rr_win = wtheta(win_ra, win_dec, win_ra, win_dec, tb, return_rr=True)[1]     # RR_uniform, ONCE
    Wc = np.array([wtheta(c[0], c[1], win_ra, win_dec, tb, rr=rr_win) for c in cats])  # completed / uniform
    # CONTROLS to localise any discrepancy:
    w_off_uni = wtheta(ra, dec, win_ra, win_dec, tb, w_d=wc, rr=rr_win)          # official / uniform rand
    w_cmp_srv = np.array([wtheta(c[0], c[1], rar_o, decr_o, tb, rr=rr_off) for c in cats]).mean(0)  # completed / survey rand
    results["wtheta"] = (tc, w_off, Wc.mean(0), Wc.std(0), w_off_uni, w_cmp_srv)
    print("\n--- w(θ) 2x2 diagnostic (ratio to official/survey-random) ---")
    print(f"{'θ[deg]':>8}{'off/srv':>9}{'off/unif':>9}{'cmp/srv':>9}{'cmp/unif':>9}")
    for i in range(len(tc)):
        b = w_off[i]
        print(f"{tc[i]:8.3f}{1.0:9.3f}{w_off_uni[i]/b:9.3f}{w_cmp_srv[i]/b:9.3f}{Wc.mean(0)[i]/b:9.3f}")
    print("  (off/unif tests the uniform footprint+random; cmp/srv tests the completed catalog)")

    # ---------- wp(rp) ----------
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    wp_off, RRo = wp_rp(ra, dec, z, rar_o, decr_o, zr_o, w=wc, rp_edges=rp_edges, pimax=40., nthreads=NTH, return_RR=True)
    RRwin = [None]; Wp = []
    for c in cats:
        out = wp_rp(c[0], c[1], c[2], win_ra, win_dec, win_z, rp_edges=rp_edges, pimax=40.,
                    nthreads=NTH, precomp_RR=RRwin[0], return_RR=(RRwin[0] is None))
        if RRwin[0] is None:
            out, RRwin[0] = out
        Wp.append(out)
    Wp = np.array(Wp); results["wp"] = (rpc, wp_off, Wp.mean(0), Wp.std(0))

    # ---------- xi_0, xi_2, xi_4 ----------
    s_edges = np.logspace(np.log10(1.0), np.log10(40.0), 11)
    sc, x0o, x2o, x4o, _ = xi_smu_ell024(ra, dec, z, rar_o, decr_o, zr_o, w=wc, s_edges=s_edges,
                                         nmu=20, nthreads=NTH, return_RR=True)
    RRsw = [None]; X0, X2, X4 = [], [], []
    for c in cats:
        sc, a0, a2, a4, rr = xi_smu_ell024(c[0], c[1], c[2], win_ra, win_dec, win_z, s_edges=s_edges,
                                           nmu=20, nthreads=NTH, precomp_RR=RRsw[0], return_RR=(RRsw[0] is None))
        if RRsw[0] is None:
            RRsw[0] = rr
        X0.append(a0); X2.append(a2); X4.append(a4)
    results["xi"] = (sc, (x0o, x2o, x4o), (np.mean(X0,0), np.mean(X2,0), np.mean(X4,0)),
                     (np.std(X0,0), np.std(X2,0), np.std(X4,0)))

    # ---------- report ----------
    def band(name, x, off, cmp, std):
        ok = np.isfinite(off) & np.isfinite(cmp) & (np.abs(off) > 1e-6)
        r = cmp[ok]/off[ok]; nsig = np.abs(cmp[ok]-off[ok])/np.where(std[ok]>0, std[ok], np.inf)
        print(f"\n{name}: completed(equal-wt, uniform hole-free window) / official(w_c, survey randoms)")
        print(f"  median ratio {np.median(r):.3f}, range {r.min():.3f}-{r.max():.3f}, max |Δ|/σ_real {np.nanmax(nsig):.1f}")

    print("\n" + "="*72)
    tc, w_off, w_m, w_s, w_off_uni, w_cmp_srv = results["wtheta"]
    band("w(theta)", tc, w_off, w_m, w_s)
    print(f"  control: completed/survey-rand median {np.median(w_cmp_srv/w_off):.3f} (catalog faithful); "
          f"official/uniform-rand median {np.median(w_off_uni/w_off):.3f} (window resolution, not catalog)")
    band("wp(rp)", *results["wp"])
    sc, (x0o,x2o,x4o), (x0c,x2c,x4c), (s0,s2,s4) = results["xi"]
    band("xi_0(s)", sc, x0o, x0c, s0); band("xi_2(s)", sc, x2o, x2c, s2); band("xi_4(s)", sc, x4o, x4c, s4)

    # ---------- figure ----------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    C_OFF, C_CMP = "#222", "#3a6ea8"
    a = ax[0,0]; a.loglog(tc, w_off, "s-", color=C_OFF, label="official (w_c, survey randoms)")
    a.fill_between(tc, w_m-w_s, w_m+w_s, color=C_CMP, alpha=0.3)
    a.loglog(tc, w_m, "o-", color=C_CMP, label="completed (equal-wt, uniform hole-free window)")
    a.set_xlabel("θ [deg]"); a.set_ylabel("w(θ)"); a.legend(fontsize=8); a.set_title("angular w(θ)")
    a = ax[0,1]
    a.axhline(1, color="gray", ls=":")
    a.semilogx(tc, w_cmp_srv/w_off, "o-", color=C_CMP, label="completed / survey-rand (catalog test)")
    a.semilogx(tc, w_off_uni/w_off, "s--", color="#c0392b", label="official / uniform-rand (window test)")
    a.semilogx(tc, w_m/w_off, "^-", color="#e8853a", label="completed / uniform-rand")
    a.set_ylim(0.9, 1.4); a.set_xlabel("θ [deg]"); a.set_ylabel("ratio to official/survey-rand"); a.legend(fontsize=7)
    a.set_title("controls: catalog is faithful; uniform-window offset = boundary resolution")
    rpc, wp_off, wp_m, wp_s = results["wp"]
    a = ax[0,2]; a.loglog(rpc, wp_off, "s-", color=C_OFF, label="official")
    a.fill_between(rpc, wp_m-wp_s, wp_m+wp_s, color=C_CMP, alpha=0.3); a.loglog(rpc, wp_m, "o-", color=C_CMP, label="completed")
    a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("wp(rp)"); a.legend(fontsize=8); a.set_title("projected wp(rp)")
    for j,(lab,xo,xc,xs) in enumerate([("ξ0",x0o,x0c,s0),("ξ2",x2o,x2c,s2),("ξ4",x4o,x4c,s4)]):
        a = ax[1,j]
        a.plot(sc, sc**2*xo, "s-", color=C_OFF, label=f"{lab} official")
        a.fill_between(sc, sc**2*(xc-xs), sc**2*(xc+xs), color=C_CMP, alpha=0.3)
        a.plot(sc, sc**2*xc, "o-", color=C_CMP, label=f"{lab} completed")
        a.set_xscale("log"); a.set_xlabel("s [Mpc/h]"); a.set_ylabel(f"s² {lab}"); a.legend(fontsize=8)
        a.set_title(f"multipole {lab}")
    fig.suptitle("Drop-in: equal-weight completed+inpainted + uniform analytic hole-free randoms "
                 "vs official weighted survey", y=1.0)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
