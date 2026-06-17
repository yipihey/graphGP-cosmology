"""Mean + covariance validation of the measurement-first LGCP mocks.

The per-realization ratio is the wrong target at the wings (the signal there is
w~0.008 and the cosmic+shot scatter is tens of percent). The right test of the
mocks as a statistical model is:

  - the MEAN w(θ) of a large mock ensemble is unbiased vs BOSS, and
  - the mock realization COVARIANCE matches the survey's own covariance,
    estimated by a delete-one jackknife over equal-area healpix regions.

Compares diagonal errors σ(θ), the bin-bin correlation matrices, and the χ² of
(mock mean − data) under the jackknife covariance (Hartlap-corrected).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=16 \
    ~/.venv/k3d/bin/python3 demos/validate_observed_K2d_covariance.py --n-mock 32
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.observed_ls import (measure_K2d_data, deconvolve_window,
                                       kernel_from_K2d, generate_catalogs_from_kernel)

NTH = 16


def wtheta(ra_d, dec_d, w_d, ra_r, dec_r, w_r, tb):
    """Weighted Landy-Szalay w(θ)."""
    Wd, Wr = w_d.sum(), w_r.sum()
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       weights1=w_d.astype("f8"), weight_type="pair_product")
    rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"),
                       weights1=w_r.astype("f8"), weight_type="pair_product")
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       weights1=w_d.astype("f8"), RA2=ra_r.astype("f8"),
                       DEC2=dec_r.astype("f8"), weights2=w_r.astype("f8"),
                       weight_type="pair_product")
    DD = dd["npairs"] * dd["weightavg"] / Wd**2
    RR = rr["npairs"] * rr["weightavg"] / Wr**2
    DR = dr["npairs"] * dr["weightavg"] / (Wd * Wr)
    return np.where(RR > 0, (DD - 2*DR + RR) / RR, np.nan)


def corr_from_cov(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mock", type=int, default=32)
    p.add_argument("--nside-jk", type=int, default=8)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--out", default="output/observed_K2d_cov.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"N_data={cat.N_data:,}")

    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)]); ze = np.linspace(0.0, 0.03, 11)
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze,
                                       n_data=80_000, n_rand_factor=3, seed=0, return_counts=True)
    xi_in, _ = deconvolve_window(xi_w, cnt["rr"])
    cov, sigma2 = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)

    cats = generate_catalogs_from_kernel(cat, cov, sigma2, alpha=args.alpha,
                                         n_samples=args.n_mock, seed=1,
                                         w_completeness=w_comp, verbose=True)

    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tc = np.sqrt(tb[1:]*tb[:-1])

    # --- comparison randoms (shared) ---
    rng = np.random.default_rng(7)
    rar, decr, _ = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=400_000, z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=rng)
    wr = np.ones(len(rar))

    # --- mock ensemble: mean and covariance ---
    Wm = np.array([wtheta(c["ra"], c["dec"], np.ones(len(c["ra"])), rar, decr, wr, tb)
                   for c in cats])
    w_mock = Wm.mean(0)
    C_mock = np.cov(Wm, rowvar=False)

    # --- data w(θ), weighted ---
    ra_d = np.asarray(cat.ra_data); dec_d = np.asarray(cat.dec_data); w_d = np.asarray(cat.w_data)
    w_data = wtheta(ra_d, dec_d, w_d, rar, decr, wr, tb)

    # --- delete-one jackknife over occupied healpix regions ---
    pix_d = hp.ang2pix(args.nside_jk, ra_d, dec_d, lonlat=True)
    pix_r = hp.ang2pix(args.nside_jk, rar, decr, lonlat=True)
    regions = np.intersect1d(np.unique(pix_d), np.unique(pix_r))
    njk = len(regions)
    Wjk = []
    for k in regions:
        dm = pix_d != k; rm = pix_r != k
        Wjk.append(wtheta(ra_d[dm], dec_d[dm], w_d[dm], rar[rm], decr[rm], wr[rm], tb))
    Wjk = np.array(Wjk)
    w_jk = Wjk.mean(0)
    C_jk = (njk - 1) / njk * np.cov(Wjk, rowvar=False, bias=True) * njk  # delete-one jackknife
    print(f"\njackknife regions: {njk} (nside={args.nside_jk})")

    sig_mock = np.sqrt(np.diag(C_mock)); sig_jk = np.sqrt(np.diag(C_jk))
    print(f"\n{'theta':>7}{'w_data':>9}{'w_mock':>9}{'mean/data':>10}"
          f"{'sig_jk%':>9}{'sig_mock%':>10}{'(m-d)/sig_jk':>13}")
    for i in range(len(tc)):
        print(f"{tc[i]:7.3f}{w_data[i]:9.4f}{w_mock[i]:9.4f}{w_mock[i]/w_data[i]:10.3f}"
              f"{100*sig_jk[i]/w_data[i]:9.1f}{100*sig_mock[i]/w_data[i]:10.1f}"
              f"{(w_mock[i]-w_data[i])/sig_jk[i]:13.2f}")

    # chi^2 of (mock mean - data) under jackknife covariance (Hartlap on C_jk)
    nb = len(tc)
    hart = (njk - nb - 2) / (njk - 1)
    Cinv = hart * np.linalg.inv(C_jk)
    d = w_mock - w_data
    chi2 = float(d @ Cinv @ d)
    print(f"\nχ²(mock_mean − data | C_jk) = {chi2:.1f} for {nb} bins  (χ²/dof={chi2/nb:.2f})")
    # also with combined covariance (jk + mock-mean error)
    Ccomb = C_jk + C_mock / args.n_mock
    chi2c = float(d @ np.linalg.inv(Ccomb) @ d) * (njk - nb - 2) / (njk - 1)
    print(f"χ²(mock_mean − data | C_jk + C_mock/N) = {chi2c:.1f}  (χ²/dof={chi2c/nb:.2f})")

    # --- plots ---
    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.errorbar(tc, w_data, yerr=sig_jk, fmt="s", color="#f5a623",
                 label="BOSS (jackknife err)", capsize=3)
    lo, hi = np.percentile(Wm, [16, 84], axis=0)
    ax1.fill_between(tc, lo, hi, color="#4a90d9", alpha=0.25, label="mock 16–84%")
    ax1.plot(tc, w_mock, "o-", color="#4a90d9", label="mock mean")
    ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlabel(r"$\theta$ [deg]")
    ax1.set_ylabel(r"$w(\theta)$"); ax1.legend(); ax1.set_title("mean + scatter")
    ax1.grid(True, which="both", alpha=0.2)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.loglog(tc, sig_jk, "s-", color="#f5a623", label="data jackknife σ")
    ax2.loglog(tc, sig_mock, "o-", color="#4a90d9", label="mock σ")
    ax2.set_xlabel(r"$\theta$ [deg]"); ax2.set_ylabel(r"$\sigma_{w}$")
    ax2.legend(); ax2.set_title("diagonal errors"); ax2.grid(True, which="both", alpha=0.2)

    ax3 = fig.add_subplot(1, 3, 3)
    R = corr_from_cov(C_mock); Rj = corr_from_cov(C_jk)
    tri = np.tril(Rj) + np.triu(R, 1)   # lower=jackknife, upper=mock
    im = ax3.imshow(tri, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    ax3.set_title("corr: lower=data-jk, upper=mock"); plt.colorbar(im, ax=ax3, fraction=0.046)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
