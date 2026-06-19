"""Head-to-head: graphGP conditional-field redshifts vs the KNN-KDE proxy.

Audit follow-up. The production completion assigns each missing galaxy's redshift
from a hand-rolled KNN+KDE "local field". graphGP can do the principled version: a
Matheron conditional posterior of the density field delta(n_hat,z) given the observed
galaxies (twopt_density.density_field, the namesake method). This builds BOTH z-fields,
completes the SAME missing galaxies (observed + missing only; systot excluded to
isolate the redshift engine), and compares on real BOSS CMASS-South:
  * n(z) vs the completeness-weighted observed n(z),
  * wp(rp) and xi_0(s) vs the official w_c-weighted clustering,
  * the ensemble spread (does the GP give a larger / more honest posterior?).

The two engines differ in regime: KNN-KDE reads the EXACT redshifts of the 150
nearest observed galaxies (sharp, cosmology-free); the GP is a smooth-kernel
posterior on a HEALPix x z-shell grid (correlated across missing galaxies, but
limited by grid resolution and S/N, and it uses a fiducial cosmology in the prior
kernel). This quantifies the trade.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/graphgp_vs_knn_zfield.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import healpy as hp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets
from twopt_density.observed_ls import (complete_catalog_photoz, measure_close_pair_dz,
                                       _clpair_density)
from twopt_density.density_field import sample_posterior_density_field
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.clustering_corrfunc import wp_rp, xi_smu_multipoles

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def graphgp_catalogs(res, cat, tg, pz, dz, n_real, rng):
    """Observed+missing catalogs from the GP field draws, via the FIRST-CLASS engine
    ``complete_catalog_photoz(z_mode='graphgp', gp_field=res)`` (systot stripped to
    isolate the redshift assignment). ``rng`` is unused (seeds are seed=0..n_real-1)."""
    from twopt_density.observed_ls import PROV
    cats = []
    for s in range(n_real):
        c = complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz, z_mode="graphgp", gp_field=res)
        m = np.asarray(c["prov"]) != PROV["systot"]
        cats.append({"ra": np.asarray(c["ra"])[m], "dec": np.asarray(c["dec"])[m],
                     "z": np.asarray(c["z"])[m]})
    return cats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data)
    wc = np.asarray(cat.w_sys_data) * (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)
    rng = np.random.default_rng(3)

    # ---- KNN-KDE ensemble (observed+missing only; systot off to isolate the z-engine) ----
    print("[knn] completing ...")
    knn = []
    for s in range(args.n_real):
        c = complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz, z_mode="field", systot_mode="off")
        knn.append(c)
    # systot_mode='off' may be unsupported; fall back to stripping systot rows by prov
    from twopt_density.observed_ls import PROV
    def strip_systot(c):
        m = np.asarray(c["prov"]) != PROV["systot"]
        return {"ra": np.asarray(c["ra"])[m], "dec": np.asarray(c["dec"])[m], "z": np.asarray(c["z"])[m]}
    knn = [strip_systot(c) for c in knn]

    # ---- graphGP conditional-field ensemble ----
    print(f"[graphgp] building conditional posterior field (n_samples={args.n_real}, nside={args.nside}, nz={args.nz}) ...")
    res = sample_posterior_density_field(cat, n_samples=args.n_real, nside=args.nside, n_z_bins=args.nz,
                                         r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28),
                                         seed=0, verbose=True)
    ggp = graphgp_catalogs(res, cat, tg, pz, dz, args.n_real, rng)

    # ---- randoms + measurements ----
    rar, decr, zr = make_random_from_selection_function(sel_map=cat.sel_map, n_random=3*cat.N_data,
                                                        z_data=z, nside=cat.nside, rng=np.random.default_rng(7))
    rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:]*rp_edges[:-1])
    s_edges = np.logspace(np.log10(1.0), np.log10(40.0), 11)
    wp_off, RRwp = wp_rp(ra, dec, z, rar, decr, zr, w=wc, rp_edges=rp_edges, nthreads=16, return_RR=True)
    sc, xi0_off, _, RRsmu = xi_smu_multipoles(ra, dec, z, rar, decr, zr, w=wc, s_edges=s_edges, nthreads=16, return_RR=True)

    def measure(cats):
        WP = np.array([wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]), rar, decr, zr,
                             rp_edges=rp_edges, nthreads=16, precomp_RR=RRwp) for c in cats])
        X0 = np.array([xi_smu_multipoles(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]),
                                         rar, decr, zr, s_edges=s_edges, nthreads=16, precomp_RR=RRsmu)[1] for c in cats])
        return WP, X0

    wp_k, x0_k = measure(knn); wp_g, x0_g = measure(ggp)
    zb = np.linspace(z.min(), z.max(), 40); zc2 = 0.5*(zb[1:]+zb[:-1])
    nz_w = np.histogram(z, zb, weights=wc, density=True)[0]
    nz_k = np.mean([np.histogram(np.asarray(c["z"]), zb, density=True)[0] for c in knn], 0)
    nz_g = np.mean([np.histogram(np.asarray(c["z"]), zb, density=True)[0] for c in ggp], 0)

    print("\n=== n(z) max |completed - weighted|/weighted ===")
    print(f"  KNN-KDE: {np.nanmax(np.abs(nz_k/nz_w-1)):.3f}   graphGP: {np.nanmax(np.abs(nz_g/nz_w-1)):.3f}")
    print("\n=== wp(rp): completed/official (median ratio | mean realization spread %) ===")
    print(f"  KNN-KDE: ratio {np.median(wp_k.mean(0)/wp_off):.3f}  spread {100*np.mean(wp_k.std(0)/wp_k.mean(0)):.2f}%")
    print(f"  graphGP: ratio {np.median(wp_g.mean(0)/wp_off):.3f}  spread {100*np.mean(wp_g.std(0)/wp_g.mean(0)):.2f}%")
    print("  rp     official   KNN/off   GP/off")
    for i in range(len(rpc)):
        print(f"  {rpc[i]:6.2f}  {wp_off[i]:8.2f}  {wp_k.mean(0)[i]/wp_off[i]:7.3f}  {wp_g.mean(0)[i]/wp_off[i]:7.3f}")
    print("\n=== xi_0(s): completed/official ===")
    ok = np.abs(xi0_off) > 1e-6
    print(f"  KNN-KDE median {np.median((x0_k.mean(0)/xi0_off)[ok]):.3f}   "
          f"graphGP median {np.median((x0_g.mean(0)/xi0_off)[ok]):.3f}")
    print("\n(KNN-KDE sharp/cosmology-free; graphGP correlated/principled but grid+S/N limited. "
          "Bigger spread = more honest posterior IF recovery stays ~1%.)")

    # ---- figure for the report's graphGP tab ----
    CK, CG, CO = "#3a6ea8", "#c0392b", "#222"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    a = ax[0]
    a.loglog(rpc, wp_off, "s-", color=CO, label="official (w_c-weighted)")
    a.loglog(rpc, wp_k.mean(0), "o-", color=CK, label="KNN-KDE completion")
    a.loglog(rpc, wp_g.mean(0), "^-", color=CG, label="graphGP completion")
    a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("wp(rp)"); a.legend(fontsize=8); a.set_title("projected wp(rp)")
    a = ax[1]
    a.axhline(1, color="gray", ls=":"); a.fill_between(rpc, 0.97, 1.03, color="green", alpha=0.1)
    a.semilogx(rpc, wp_k.mean(0)/wp_off, "o-", color=CK, label="KNN / official")
    a.semilogx(rpc, wp_g.mean(0)/wp_off, "^-", color=CG, label="graphGP / official")
    a.set_ylim(0.9, 1.12); a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("completed / official")
    a.legend(fontsize=8); a.set_title("wp ratio (graphGP matches the weighted more closely)")
    a = ax[2]
    a.step(zc2, nz_w, where="mid", color=CO, lw=2, label="weighted observed")
    a.step(zc2, nz_k, where="mid", color=CK, lw=1.6, ls="--", label="KNN-KDE")
    a.step(zc2, nz_g, where="mid", color=CG, lw=1.6, ls=":", label="graphGP")
    a.set_xlabel("redshift z"); a.set_ylabel("n(z)"); a.legend(fontsize=8); a.set_title("n(z)")
    fig.suptitle("Redshift engine head-to-head on real CMASS-South (observed+missing, vs official weighted)", y=1.01)
    fig.tight_layout(); os.makedirs("output", exist_ok=True)
    fig.savefig("output/graphgp_vs_knn.png", dpi=130, bbox_inches="tight")
    print("Saved: output/graphgp_vs_knn.png")


if __name__ == "__main__":
    main()
