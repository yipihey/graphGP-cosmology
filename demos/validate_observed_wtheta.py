"""Validate the cosmology-free observed-space anisotropic LGCP against BOSS w(θ).

Generates posterior-predictive catalogs entirely in observed coordinates
(n̂, z) — no fiducial cosmology — and compares their Landy-Szalay angular
two-point function w(θ) to the observed BOSS CMASS-SGC catalog, measured with
Corrfunc's DDtheta_mocks (the validated reference estimator).

This is the reproduction harness for the open task: drive w(θ)/data to the
percent level (see docs/OBSERVED_WTHETA_HANDOFF.md).

Run (needs the chunked+anisotropic graphGP fork on PYTHONPATH and the A6000):

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=16 \
    ~/.venv/k3d/bin/python3 demos/validate_observed_wtheta.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

from twopt_density.boss import load_boss
from twopt_density.observed import sample_catalogs_lgcp_observed
from twopt_density.quaia import make_random_from_selection_function


def wtheta(ra_d, dec_d, ra_r, dec_r, theta_bins, nthreads=16):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, nthreads, theta_bins, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    rr = DDtheta_mocks(1, nthreads, theta_bins, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, nthreads, theta_bins, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd / (nd * (nd - 1.)) - 2 * dr / (nd * nr)
                            + rr / (nr * (nr - 1.))) / (rr / (nr * (nr - 1.))), np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms", default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-samples",    type=int, default=5)
    p.add_argument("--n-cand-factor", type=int, default=20)
    p.add_argument("--n-rand-rr",    type=int, default=200_000)
    p.add_argument("--nthreads",     type=int, default=16)
    p.add_argument("--out",          default="output/observed_wtheta.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"N_data={cat.N_data:,}")

    cats, te, ze, xi = sample_catalogs_lgcp_observed(
        cat, n_samples=args.n_samples, seed=1, w_completeness=w_comp,
        n_cand_factor=args.n_cand_factor, chunk_size=50_000, verbose=True)

    rng = np.random.default_rng(7)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_rr,
        z_data=np.asarray(cat.z_data), nside=cat.nside, rng=rng)

    theta_bins = np.logspace(np.log10(0.05), np.log10(2.5), 11)
    tc = np.sqrt(theta_bins[1:] * theta_bins[:-1])
    wd = wtheta(np.asarray(cat.ra_data), np.asarray(cat.dec_data), ra_r, dec_r,
                theta_bins, args.nthreads)
    ws = np.array([wtheta(c["ra"], c["dec"], ra_r, dec_r, theta_bins, args.nthreads)
                   for c in cats])
    wm, wlo, whi = (np.nanmedian(ws, 0), np.nanpercentile(ws, 16, 0),
                    np.nanpercentile(ws, 84, 0))

    print(f"\n{'theta':>7} {'w_data':>9} {'w_LGCP':>9} {'LGCP/data':>10}")
    for i in range(len(tc)):
        print(f"{tc[i]:7.3f} {wd[i]:9.4f} {wm[i]:9.4f} "
              f"{wm[i]/wd[i] if wd[i] else np.nan:10.3f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.fill_between(tc, wlo, whi, color="#4a90d9", alpha=0.25, label="LGCP 16–84%")
    ax1.plot(tc, wm, "o-", color="#4a90d9", label="observed-space LGCP")
    ax1.plot(tc, wd, "s--", color="#f5a623", label="BOSS CMASS-SGC (observed)")
    ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_ylabel(r"$w(\theta)$")
    ax1.set_title("Cosmology-free observed-space anisotropic LGCP vs BOSS w(θ)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.2)
    ax2.semilogx(tc, wm / wd, "o-", color="#333"); ax2.axhline(1, color="gray", ls="--")
    ax2.fill_between(tc, 0.99, 1.01, color="green", alpha=0.15, label="±1%")
    ax2.set_ylim(0.4, 1.6); ax2.set_ylabel("LGCP/data"); ax2.set_xlabel(r"$\theta$ [deg]")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
