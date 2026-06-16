"""Closure test for the measurement-first observed-space pipeline.

Measure the window/weight-corrected 2D kernel K_in(Δθ,Δz) from BOSS by weighted
Landy-Szalay pair counting against the analytic randoms (FKP×completeness,
integral-constraint deconvolved); reuse K_in directly as the GraphGP generation
covariance; then re-measure K_out(Δθ,Δz) from each generated catalog with the
*identical* estimator. Success = K_out ≈ K_in across the (Δθ,Δz) plane, and the
w(θ) projection matches BOSS.

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=16 \
    ~/.venv/k3d/bin/python3 demos/validate_observed_K2d.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

from twopt_density.boss import load_boss
from twopt_density.quaia import make_random_from_selection_function
from twopt_density.observed_ls import (measure_K2d, measure_K2d_data,
                                       deconvolve_window, kernel_from_K2d,
                                       generate_catalogs_from_kernel)


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, nthreads=16):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, nthreads, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    rr = DDtheta_mocks(1, nthreads, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, nthreads, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms", default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-samples",     type=int, default=4)
    p.add_argument("--n-data-meas",   type=int, default=80_000)
    p.add_argument("--n-rand-meas",   type=int, default=3)
    p.add_argument("--n-cand-factor", type=int, default=20)
    p.add_argument("--alpha",         type=float, default=2.0)
    p.add_argument("--out", default="output/observed_K2d.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"N_data={cat.N_data:,}")

    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)])
    ze = np.linspace(0.0, 0.03, 11)
    theta_c = np.empty(len(te) - 1); theta_c[0] = 0.5 * te[1]
    theta_c[1:] = np.sqrt(te[1:-1] * te[2:])

    # --- 1. measure K_in (weighted, window-deconvolved) ---
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze,
                                       n_data=args.n_data_meas, n_rand_factor=args.n_rand_meas,
                                       seed=0, return_counts=True)
    xi_in, ic = deconvolve_window(xi_w, cnt["rr"])
    print(f"K_in measured (IC={ic:.5f}). K_in(Δθ,0) core={np.log1p(xi_in[0,0]):.3f}")

    # --- 2. reuse K_in directly as the generation kernel ---
    cov, sigma2 = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)
    print(f"kernel σ²={sigma2:.3f}")

    # --- 3. generate catalogs ---
    cats = generate_catalogs_from_kernel(
        cat, cov, sigma2, alpha=args.alpha, n_samples=args.n_samples, seed=1,
        w_completeness=w_comp, n_cand_factor=args.n_cand_factor, verbose=True)

    # --- 4. re-measure K_out (identical estimator, unweighted mock) ---
    xi_outs = []
    for s, c in enumerate(cats):
        rng = np.random.default_rng(100 + s)
        nrr = args.n_rand_meas * len(c["ra"])
        ra_r, dec_r, z_r = make_random_from_selection_function(
            sel_map=cat.sel_map, n_random=nrr, z_data=np.asarray(cat.z_data),
            nside=cat.nside, rng=rng)
        _, _, xi_o = measure_K2d(c["ra"], c["dec"], c["z"], np.ones(len(c["ra"])),
                                 ra_r, dec_r, z_r, np.ones(len(ra_r)),
                                 theta_edges=te, z_edges=ze)
        xi_outs.append(xi_o)
    xi_out = np.median(xi_outs, axis=0)

    # --- 5. w(θ) projection vs BOSS (Corrfunc, consistent randoms) ---
    rng = np.random.default_rng(7)
    ra_r, dec_r, _ = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=200_000, z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=rng)
    tb = np.logspace(np.log10(0.05), np.log10(2.5), 11); tcw = np.sqrt(tb[1:]*tb[:-1])
    wd = wtheta(np.asarray(cat.ra_data), np.asarray(cat.dec_data), ra_r, dec_r, tb)
    ws = np.median([wtheta(c["ra"], c["dec"], ra_r, dec_r, tb) for c in cats], axis=0)

    # --- report ---
    Kin0 = np.log1p(np.clip(xi_in[:, 0], 0, None))
    Kout0 = np.log1p(np.clip(xi_out[:, 0], 0, None))
    print(f"\n2D kernel closure  K(Δθ, Δz=0):")
    print(f"{'theta':>8}{'K_in':>9}{'K_out':>9}{'out/in':>9}")
    for i in range(len(theta_c)):
        r = Kout0[i]/Kin0[i] if Kin0[i] > 1e-6 else np.nan
        print(f"{theta_c[i]:8.3f}{Kin0[i]:9.4f}{Kout0[i]:9.4f}{r:9.3f}")
    print(f"\nw(θ) projection vs BOSS:")
    print(f"{'theta':>8}{'w_data':>9}{'w_LGCP':>9}{'ratio':>8}")
    for i in range(len(tcw)):
        print(f"{tcw[i]:8.3f}{wd[i]:9.4f}{ws[i]:9.4f}{ws[i]/wd[i] if wd[i] else np.nan:8.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.loglog(theta_c, Kin0, "s-", label="K_in (BOSS, weighted+deconv)")
    a1.loglog(theta_c, Kout0, "o-", label="K_out (LGCP, same estimator)")
    a1.set_xlabel(r"$\Delta\theta$ [deg]"); a1.set_ylabel(r"$K(\Delta\theta,0)=\ln(1+\xi)$")
    a1.set_title("2D kernel closure"); a1.legend(); a1.grid(True, which="both", alpha=0.2)
    a2.loglog(tcw, wd, "s--", label="BOSS CMASS-SGC")
    a2.loglog(tcw, ws, "o-", label="observed-space LGCP")
    a2.set_xlabel(r"$\theta$ [deg]"); a2.set_ylabel(r"$w(\theta)$")
    a2.set_title("w(θ) projection"); a2.legend(); a2.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
