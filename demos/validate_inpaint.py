"""Validate the analog-transplant inpainting of survey-mask holes.

Checks (cosmology-free, observed coordinates):
  (1) CLUSTERING CLOSURE (headline): the standard masked measurement
      w(θ) = LS(data, masked-randoms) — where holes cancel — must agree with the
      inpainted measurement w(θ) = LS(data+inpaint, hole-filled-randoms), where
      the catalog is now treated as hole-free. Agreement ⇒ the inpaint filled the
      holes with statistically-correct galaxies.
  (2) HIGHER-ORDER (counts-in-cells): the galaxy-count distribution in apertures
      centred on inpainted holes vs matched clean control apertures.
  (3) PROPERTIES: inpainted vs observed colour and n(z) distributions.
  (4) before/after sky zoom on a hole.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/validate_inpaint.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from twopt_density.boss import load_boss
from twopt_density.observed import _radec_to_nhat
from twopt_density.inpaint import fine_completeness_map, find_interior_holes, inpaint_holes

NTH = 16


def wtheta(ra_d, dec_d, ra_r, dec_r, tb):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan)


def fill_random_holes(ra_r, dec_r, counts, nside, hole_pixels, rng):
    """Add uniform randoms inside the hole pixels at the median populated density,
    making the random window hole-free (for the closure test)."""
    med = int(np.median(counts[counts > 0]))
    res = hp.nside2resol(nside)
    th, ph = hp.pix2ang(nside, hole_pixels)
    add_th = np.repeat(th, med); add_ph = np.repeat(ph, med)
    # uniform jitter within ~one pixel
    add_th = add_th + (rng.random(len(add_th)) - 0.5) * res
    add_ph = add_ph + (rng.random(len(add_ph)) - 0.5) * res / np.sin(np.clip(add_th, 0.01, np.pi - 0.01))
    ra_add = np.degrees(add_ph) % 360; dec_add = 90 - np.degrees(add_th)
    return np.concatenate([ra_r, ra_add]), np.concatenate([dec_r, dec_add])


def cic(ra_g, dec_g, centers_ra, centers_dec, radius_deg):
    """Counts-in-cells: galaxies within radius of each centre."""
    gt = cKDTree(_radec_to_nhat(ra_g, dec_g))
    cv = _radec_to_nhat(centers_ra, centers_dec)
    return np.array([len(gt.query_ball_point(cv[i], np.radians(radius_deg)))
                     for i in range(len(cv))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--out", default="output/inpaint_validation.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)
    w_c = float((np.asarray(cat.w_sys_data)*(np.asarray(cat.w_cp_data)+np.asarray(cat.w_noz_data)-1)).mean())
    NS = args.nside
    counts, _ = fine_completeness_map(cat.ra_random, cat.dec_random, nside=NS)
    holes = find_interior_holes(counts, NS, empty_count=0.0, min_neighbour_frac=0.75)
    hole_pix = np.concatenate([h.pixels for h in holes])
    print(f"{len(holes)} holes, {sum(h.area_deg2 for h in holes):.1f} deg^2")

    ra_r_full = np.asarray(cat.ra_random); dec_r_full = np.asarray(cat.dec_random)
    real = inpaint_holes(holes, counts, NS, donor_ra=ra_d, donor_dec=dec_d, donor_z=z_d,
                         rand_ra=ra_r_full, rand_dec=dec_r_full,
                         donor_colors=cat.colors_data, donor_mags=cat.mags_data,
                         seed=0, n_real=1, density_boost=w_c)[0]
    print(f"inpainted galaxies: {len(real['ra']):,}")

    # randoms at ONE consistent scale: subsample the full set for the masked
    # measurement; fill the FULL set in the holes then subsample for the inpainted
    # one (so the hole window density matches the rest of the footprint).
    rng = np.random.default_rng(3)
    nsub = min(400_000, cat.N_random)
    ri = rng.choice(cat.N_random, nsub, replace=False)
    rar, decr = ra_r_full[ri], dec_r_full[ri]
    rar_full_f, decr_full_f = fill_random_holes(ra_r_full, dec_r_full, counts, NS, hole_pix, rng)
    rf = rng.choice(len(rar_full_f), nsub, replace=False)
    rar_f, decr_f = rar_full_f[rf], decr_full_f[rf]

    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:]*tb[:-1])
    w_masked = wtheta(ra_d, dec_d, rar, decr, tb)                                  # holes cancel
    ra_inp = np.concatenate([ra_d, real["ra"]]); dec_inp = np.concatenate([dec_d, real["dec"]])
    w_inp = wtheta(ra_inp, dec_inp, rar_f, decr_f, tb)                             # hole-free
    print("\nclustering closure  w_inpainted / w_masked:")
    for i in range(len(tc)):
        print(f"  θ={tc[i]:.3f}: masked={w_masked[i]:.4f} inpaint={w_inp[i]:.4f} ratio={w_inp[i]/w_masked[i]:.3f}")

    # counts-in-cells: apertures centred on holes vs control apertures centred on
    # RANDOM footprint positions (not galaxies — centring on galaxies would bias
    # the control high via clustering). Controls rejected if within 0.4° of a hole.
    small = [h for h in holes if 0.05 < h.radius_deg < 0.2]
    rad = 0.25
    hole_ctr_ra = np.array([h.ra for h in small]); hole_ctr_dec = np.array([h.dec for h in small])
    hole_nhat = _radec_to_nhat(np.array([h.ra for h in holes]), np.array([h.dec for h in holes]))
    htree = cKDTree(hole_nhat)
    ci = rng.choice(len(rar), 4 * len(small), replace=False)
    cand_ra, cand_dec = rar[ci], decr[ci]
    cnh = _radec_to_nhat(cand_ra, cand_dec)
    far = np.array([len(htree.query_ball_point(cnh[i], np.radians(0.4))) == 0 for i in range(len(cnh))])
    ctrl_ra, ctrl_dec = cand_ra[far][:len(small)], cand_dec[far][:len(small)]
    cic_hole = cic(ra_inp, dec_inp, hole_ctr_ra, hole_ctr_dec, rad)
    cic_ctrl = cic(ra_inp, dec_inp, ctrl_ra, ctrl_dec, rad)
    def moms(x): return x.mean(), x.var()/max(x.mean(),1), (((x-x.mean())**3).mean()/max(x.var(),1)**1.5)
    print(f"\ncounts-in-cells (r={rad}°)  mean, var/mean, skew:")
    print(f"  hole apertures:  {moms(cic_hole)}")
    print(f"  clean controls:  {moms(cic_ctrl)}")

    # figure
    fig = plt.figure(figsize=(15, 9))
    a1 = fig.add_subplot(2, 3, 1)
    a1.loglog(tc, w_masked, "s--", color="#e8853a", label="masked data + masked randoms")
    a1.loglog(tc, w_inp, "o-", color="#3a6ea8", label="inpainted + hole-filled randoms")
    a1.set_xlabel("θ [deg]"); a1.set_ylabel("w(θ)"); a1.legend(); a1.set_title("clustering closure")
    a2 = fig.add_subplot(2, 3, 2)
    a2.semilogx(tc, w_inp/w_masked, "o-", color="#333"); a2.axhline(1, color="gray", ls="--")
    a2.fill_between(tc, 0.95, 1.05, color="green", alpha=0.12); a2.set_ylim(0.85, 1.15)
    a2.set_xlabel("θ [deg]"); a2.set_ylabel("inpaint / masked"); a2.set_title("closure ratio")
    a3 = fig.add_subplot(2, 3, 3)
    mx = max(cic_hole.max(), cic_ctrl.max())
    bins = np.arange(0, mx+2)
    a3.hist(cic_ctrl, bins=bins, density=True, histtype="step", color="#e8853a", lw=2, label="clean controls")
    a3.hist(cic_hole, bins=bins, density=True, histtype="step", color="#3a6ea8", lw=2, label="hole apertures")
    a3.set_xlabel(f"galaxies in r={rad}° aperture"); a3.set_ylabel("PDF"); a3.legend()
    a3.set_title("counts-in-cells (higher-order)")
    a4 = fig.add_subplot(2, 3, 4)
    a4.hist(z_d, bins=40, density=True, histtype="step", color="#e8853a", lw=2, label="observed")
    a4.hist(real["z"], bins=40, density=True, histtype="step", color="#3a6ea8", lw=2, label="inpainted")
    a4.set_xlabel("z"); a4.set_ylabel("n(z)"); a4.legend(); a4.set_title("inpainted vs observed n(z)")
    a5 = fig.add_subplot(2, 3, 5)
    fin = cat.colors_finite
    a5.hist(cat.colors_data[fin, 1], bins=40, range=(1,2.2), density=True, histtype="step",
            color="#e8853a", lw=2, label="observed g−r")
    a5.hist(real["colors"][:, 1], bins=40, range=(1,2.2), density=True, histtype="step",
            color="#3a6ea8", lw=2, label="inpainted g−r")
    a5.set_xlabel("g − r"); a5.set_ylabel("PDF"); a5.legend(); a5.set_title("inpainted vs observed colour")
    a6 = fig.add_subplot(2, 3, 6)
    big = max(holes, key=lambda h: h.radius_deg if h.radius_deg < 0.5 else 0)
    box = (np.abs(((ra_d-big.ra+180)%360)-180) < 1.0) & (np.abs(dec_d-big.dec) < 1.0)
    mb = (np.abs(((real["ra"]-big.ra+180)%360)-180) < 1.0) & (np.abs(real["dec"]-big.dec) < 1.0)
    a6.scatter(ra_d[box], dec_d[box], s=4, c="#888", lw=0, label="observed")
    a6.scatter(real["ra"][mb], real["dec"][mb], s=6, c="#3a6ea8", lw=0, label="inpainted")
    a6.set_xlabel("RA [deg]"); a6.set_ylabel("Dec [deg]"); a6.invert_xaxis(); a6.legend()
    a6.set_title(f"before/after zoom (hole r={big.radius_deg*60:.0f}′)")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
