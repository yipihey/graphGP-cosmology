"""Validate the systematics completion: equal-weight completed catalog vs the
w_c-weighted observed catalog.

The requirement (cosmology-free, observed coordinates): the equal-weight
completed catalog's clustering reproduces the completeness-weighted observed
clustering at resolved separations, and many realizations span the posterior of
where the missing ~8% of galaxies are. Compares ξ(Δθ, Δz=0) of the weighted
observed vs the mean (and scatter) of the completed realizations.

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
from twopt_density.observed_ls import measure_K2d, complete_catalog


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--n-rand-factor", type=int, default=2)
    p.add_argument("--z-assign", default="host", choices=["host", "nz", "mix"])
    p.add_argument("--out", default="output/completion_xi.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"], sample="CMASS", nside=256)
    w_c = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    print(f"N_obs={cat.N_data:,}  <w_c>={w_c.mean():.4f}")

    te = np.concatenate([[0.0], np.geomspace(0.01, 2.5, 18)]); ze = np.linspace(0.0, 0.03, 11)
    tcen = np.empty(len(te) - 1); tcen[0] = 0.5 * te[1]; tcen[1:] = np.sqrt(te[1:-1] * te[2:])

    # shared randoms from the observed n(z) window
    rng = np.random.default_rng(0)
    nr = args.n_rand_factor * cat.N_data
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=nr, z_data=np.asarray(cat.z_data), nside=cat.nside, rng=rng)
    wr = np.ones(len(rar))

    # weighted observed clustering (the target)
    _, _, xi_w = measure_K2d(cat.ra_data, cat.dec_data, cat.z_data, w_c,
                             rar, decr, zr, wr, theta_edges=te, z_edges=ze)

    # equal-weight completed realizations
    Xi = []
    for s in range(args.n_real):
        c = complete_catalog(cat, seed=s, z_assign=args.z_assign, verbose=(s == 0))
        _, _, xi_e = measure_K2d(c["ra"], c["dec"], c["z"], np.ones(c["N"]),
                                 rar, decr, zr, wr, theta_edges=te, z_edges=ze)
        Xi.append(xi_e)
    Xi = np.array(Xi)
    xi_m = Xi.mean(0)[:, 0]; xi_s = Xi.std(0)[:, 0]; xw0 = xi_w[:, 0]

    coll = 62.0 / 3600.0
    print(f"\ncollision scale = {coll:.4f}° (completion valid above this)")
    print(f"{'theta':>8}{'xi_wgt':>10}{'xi_eq':>10}{'eq/wgt':>9}{'scatter%':>9}")
    for i in range(len(tcen)):
        flag = " " if tcen[i] > coll else " *"  # * = below collision scale
        r = xi_m[i] / xw0[i] if xw0[i] else np.nan
        sc = 100 * xi_s[i] / xi_m[i] if xi_m[i] else np.nan
        print(f"{tcen[i]:8.4f}{xw0[i]:10.4f}{xi_m[i]:10.4f}{r:9.3f}{sc:9.1f}{flag}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.errorbar(tcen, xi_m, yerr=xi_s, fmt="o-", color="#4a90d9", label="equal-weight completed (mean±scatter)")
    a1.plot(tcen, xw0, "s--", color="#f5a623", label="w_c-weighted observed")
    a1.axvline(coll, color="gray", ls=":", label="collision scale")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel(r"$\Delta\theta$ [deg]")
    a1.set_ylabel(r"$\xi(\Delta\theta, \Delta z{=}0)$"); a1.legend(); a1.grid(True, which="both", alpha=0.2)
    a1.set_title(f"completion closure (z_assign={args.z_assign})")
    a2.semilogx(tcen, xi_m / xw0, "o-", color="#333"); a2.axhline(1, color="gray", ls="--")
    a2.axvline(coll, color="gray", ls=":"); a2.fill_between(tcen, 0.98, 1.02, color="green", alpha=0.15, label="±2%")
    a2.set_ylim(0.7, 1.3); a2.set_xlabel(r"$\Delta\theta$ [deg]"); a2.set_ylabel("eq / weighted")
    a2.legend(); a2.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
