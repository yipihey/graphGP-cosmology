"""Validate the systematics completion against the w_c-weighted observed catalog.

Three checks (cosmology-free, observed coordinates):
  (1) global ξ(Δθ, Δz=0): equal-weight completed vs w_c-weighted observed, and
      the data-driven vs host redshift assignment (the small-scale prior);
  (2) the proper systematic/collision split (background vs clustered z);
  (3) per-redshift-slice angular ξ(Δθ|z̄) closure — "identical at that redshift".

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/validate_completion.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.observed_ls import (measure_K2d, complete_catalog,
                                       measure_close_pair_dz)

COLL = 62.0 / 3600.0


def xi0(ra_d, dec_d, z_d, w_d, ra_r, dec_r, z_r, te, ze, mask_d=None, mask_r=None):
    if mask_d is not None:
        ra_d, dec_d, z_d, w_d = ra_d[mask_d], dec_d[mask_d], z_d[mask_d], w_d[mask_d]
        ra_r, dec_r, z_r = ra_r[mask_r], dec_r[mask_r], z_r[mask_r]
    _, _, xi = measure_K2d(ra_d, dec_d, z_d, w_d, ra_r, dec_r, z_r, np.ones(len(ra_r)),
                           theta_edges=te, z_edges=ze)
    return xi[:, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--n-rand-factor", type=int, default=2)
    p.add_argument("--out", default="output/completion_validation.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"], sample="CMASS", nside=256)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); z_d = np.asarray(cat.z_data)
    print(f"N_obs={cat.N_data:,}  <w_c>={w_c.mean():.4f}")

    dz_pool = measure_close_pair_dz(cat, COLL)
    print(f"close pairs ≤{COLL*3600:.0f}\": {len(dz_pool)//2:,}  "
          f"frac |Δz|<0.003 = {np.mean(np.abs(dz_pool) < 0.003):.2f} (clustered core)")

    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 18)]); ze = np.linspace(0.0, 0.03, 11)
    tcen = np.empty(len(te) - 1); tcen[0] = 0.5 * te[1]; tcen[1:] = np.sqrt(te[1:-1] * te[2:])

    rng = np.random.default_rng(0)
    nr = args.n_rand_factor * cat.N_data
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=nr, z_data=z_d, nside=cat.nside, rng=rng)

    xw = xi0(ra_d, dec_d, z_d, w_c, rar, decr, zr, te, ze)

    # (1)+(2) global closure for data vs host z-assignment
    res = {}
    for za in ("data", "host"):
        X = []
        for s in range(args.n_real):
            c = complete_catalog(cat, seed=s, z_assign=za, dz_pool=dz_pool, verbose=(s == 0))
            X.append(xi0(c["ra"], c["dec"], c["z"], np.ones(c["N"]), rar, decr, zr, te, ze))
        res[za] = (np.mean(X, 0), np.std(X, 0))

    print(f"\n(1) global ξ(Δθ,0): equal-weight / w_c-weighted")
    print(f"{'theta':>8}{'xi_wgt':>10}{'data':>8}{'host':>8}{'scat%':>7}")
    for i in range(len(tcen)):
        f = "*" if tcen[i] < COLL else " "
        print(f"{tcen[i]:8.4f}{xw[i]:10.4f}{res['data'][0][i]/xw[i]:8.3f}"
              f"{res['host'][0][i]/xw[i]:8.3f}{100*res['data'][1][i]/res['data'][0][i]:7.1f} {f}")

    # (3) per-redshift-slice angular closure (z_assign='data')
    zedges = np.quantile(z_d, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n(3) per-z-slice ξ(Δθ|z̄) eq/weighted (z_assign=data, resolved θ≳{COLL:.3f}°):")
    cats = [complete_catalog(cat, seed=s, z_assign="data", dz_pool=dz_pool) for s in range(args.n_real)]
    slice_rows = []
    for a, b in zip(zedges[:-1], zedges[1:]):
        md = (z_d >= a) & (z_d < b); mr = (zr >= a) & (zr < b)
        xw_s = xi0(ra_d, dec_d, z_d, w_c, rar, decr, zr, te, ze, md, mr)
        Xs = []
        for c in cats:
            zc = c["z"]; mc = (zc >= a) & (zc < b)
            Xs.append(xi0(c["ra"], c["dec"], c["z"], np.ones(c["N"]), rar, decr, zr, te, ze, mc, mr))
        xm = np.mean(Xs, 0)
        ratio = xm / xw_s
        resolved = tcen > COLL
        med = np.nanmedian(ratio[resolved])
        slice_rows.append((a, b, xw_s, xm))
        print(f"  z∈[{a:.2f},{b:.2f}): median eq/wgt (resolved) = {med:.3f}  "
              f"N_gal={md.sum():,}")

    # plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(tcen, xw, "ks--", label="w_c-weighted observed", zorder=5)
    a1.errorbar(tcen, res["data"][0], yerr=res["data"][1], fmt="o-", color="#4a90d9",
                label="completed (z=data)")
    a1.plot(tcen, res["host"][0], "^:", color="#d0021b", label="completed (z=host)")
    a1.axvline(COLL, color="gray", ls=":")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel(r"$\Delta\theta$ [deg]")
    a1.set_ylabel(r"$\xi(\Delta\theta,0)$"); a1.legend(); a1.grid(True, which="both", alpha=0.2)
    a1.set_title("global closure + z-assignment prior")
    for (a, b, xw_s, xm) in slice_rows:
        l, = a2.plot(tcen, xm / xw_s, "o-", label=f"z∈[{a:.2f},{b:.2f})")
    a2.axhline(1, color="gray", ls="--"); a2.axvline(COLL, color="gray", ls=":")
    a2.fill_between(tcen, 0.97, 1.03, color="green", alpha=0.12)
    a2.set_xscale("log"); a2.set_ylim(0.8, 1.2); a2.set_xlabel(r"$\Delta\theta$ [deg]")
    a2.set_ylabel("eq / weighted"); a2.legend(fontsize=8); a2.grid(True, which="both", alpha=0.2)
    a2.set_title("per-redshift-slice closure")
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
