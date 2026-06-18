"""Phase 5 — do the equal-weight completed catalogues reproduce the OFFICIAL
weighted BOSS clustering? (referee consistency check)

The community measures clustering with the official completeness weights
w_c = w_systot·(w_cp+w_noz-1) (the FKP weight is an estimator weight, applied on
top). Our equal-weight completed catalogues should give the SAME clustering with
no weights. We compare, on the real CMASS-South data, the standard statistics:
projected wp(rp) and the redshift-space multipoles ξ0, ξ2 (Corrfunc, parallel),
for (a) the w_c-weighted observed galaxies and (b) the equal-weight completed
ensemble. Agreement = the completion is a drop-in, weight-free replacement.

A fiducial cosmology (Planck18) is used ONLY to measure these statistics; the
catalogues remain cosmology-free.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/validate_cosmology_consistency.py
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
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import wp_rp, xi_smu_multipoles

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=8)
    p.add_argument("--out", default="output/cosmology_consistency.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    wc = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    rng = np.random.default_rng(7)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=3*cat.N_data, z_data=z, nside=cat.nside, rng=rng)

    cats = [complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz) for s in range(args.n_real)]

    # ---- wp(rp): official weighted vs equal-weight completed ----
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    wp_off, RRwp = wp_rp(ra, dec, z, rar, decr, zr, w=wc, rp_edges=rp_edges, pimax=40., nthreads=32, return_RR=True)
    Wp = np.array([wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]), rar, decr, zr,
                         rp_edges=rp_edges, pimax=40., nthreads=32, precomp_RR=RRwp) for c in cats])
    wp_cmp, wp_cmp_s = Wp.mean(0), Wp.std(0)
    print("wp(rp): completed(equal-weight) / official(w_c-weighted)")
    for i in range(len(rpc)):
        print(f"  rp={rpc[i]:6.2f}: official={wp_off[i]:7.2f} completed={wp_cmp[i]:7.2f} "
              f"ratio={wp_cmp[i]/wp_off[i]:.3f} (±{wp_cmp_s[i]/wp_off[i]:.3f})")

    # ---- xi(s,mu) multipoles ----
    s_edges = np.logspace(np.log10(1.0), np.log10(40.0), 11)
    sc, xi0_off, xi2_off, RRsmu = xi_smu_multipoles(ra, dec, z, rar, decr, zr, w=wc,
                                                    s_edges=s_edges, nmu=20, nthreads=32, return_RR=True)
    X0, X2 = [], []
    for c in cats:
        _, x0, x2, _ = xi_smu_multipoles(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]),
                                         rar, decr, zr, s_edges=s_edges, nmu=20, nthreads=32, precomp_RR=RRsmu)
        X0.append(x0); X2.append(x2)
    xi0_cmp, xi2_cmp = np.mean(X0, 0), np.mean(X2, 0)
    print("\nmonopole s^2 ξ0: completed/official")
    for i in range(len(sc)):
        print(f"  s={sc[i]:6.2f}: off={xi0_off[i]:.4f} cmp={xi0_cmp[i]:.4f} ratio={xi0_cmp[i]/xi0_off[i] if xi0_off[i] else np.nan:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    a.loglog(rpc, wp_off, "k-", lw=2, label="official (w_c-weighted)")
    a.fill_between(rpc, wp_cmp-wp_cmp_s, wp_cmp+wp_cmp_s, color="#3a6ea8", alpha=0.3)
    a.loglog(rpc, wp_cmp, "o-", color="#3a6ea8", label="completed (equal-weight)")
    a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("wp(rp)"); a.legend(); a.set_title("projected wp(rp)")
    a = ax[1]
    a.axhline(1, color="gray", ls=":"); a.fill_between(rpc, 0.97, 1.03, color="green", alpha=0.1)
    a.semilogx(rpc, wp_cmp/wp_off, "o-", color="#3a6ea8")
    a.set_ylim(0.9, 1.1); a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("completed/official"); a.set_title("wp ratio")
    a = ax[2]
    a.semilogx(sc, sc**2*xi0_off, "k-", lw=2, label="ξ0 official")
    a.semilogx(sc, sc**2*xi0_cmp, "o-", color="#3a6ea8", label="ξ0 completed")
    a.semilogx(sc, sc**2*xi2_off, "k--", lw=2, label="ξ2 official")
    a.semilogx(sc, sc**2*xi2_cmp, "s--", color="#c0392b", label="ξ2 completed")
    a.set_xlabel("s [Mpc/h]"); a.set_ylabel("s² ξ_ℓ"); a.legend(fontsize=8); a.set_title("multipoles")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
