"""Lightcone-native angular cone-shell variance on Quaia G < 20.

Computes ``sigma^2_obs(theta; z, dz)`` via cap counts + redshift slices
(no comoving conversion -- pure observable space), and the adjacent-
shell finite-difference ``d sigma^2 / d ln(1+z)``. Overlays the
predicted ``sigma2_cone_shell_predicted`` forward model with a fiducial
power-law bias.

Output: demos/figures/quaia_sigma2_cone_shell.png

Three panels:
    (a) sigma^2_obs(theta; z) per shell with predicted curves
    (b) d sigma^2 / d ln(1+z) per (theta, z) pivot
    (c) bias / growth / geometry decomposition trace at one theta

Tunables via env vars:
    PAPER_N_DATA         (default 80_000)
    PAPER_NSIDE_CENTRES  (default 64)
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2_cone_shell import (
    dsigma2_dz_cone_shell_predicted,
    sigma2_cone_shell_decomposition,
    sigma2_cone_shell_predicted_stack,
)
from twopt_density.sigma2_cone_shell_estimator import (
    cap_centre_grid,
    cone_shell_counts,
    dsigma2_dz_estimate,
    sigma2_cone_shell_jackknife,
    sigma2_estimate_cone_shell,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    n_data = int(os.environ.get("PAPER_N_DATA", 80_000))
    nside_centres = int(os.environ.get("PAPER_NSIDE_CENTRES", 64))
    do_jackknife = bool(int(os.environ.get("PAPER_JACKKNIFE", 1)))
    centres_subsample = int(os.environ.get("PAPER_CENTRES_SUBSAMPLE", 1))

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia G < 20.0 ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    sel_map, nside_mask = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))

    md = (cat.z_data >= 0.5) & (cat.z_data <= 3.0)
    rng = np.random.default_rng(0)
    n_avail = int(md.sum())
    take = min(n_data, n_avail)
    iD = rng.choice(n_avail, take, replace=False)
    where = np.where(md)[0][iD]
    ra_d = cat.ra_data[where]
    dec_d = cat.dec_data[where]
    z_d = cat.z_data[where]
    print(f"  N_data = {len(ra_d):,} (Quaia G<20 in z in [0.5, 3.0])")

    # 8 shells log-spaced in (1+z) from 0.6 to 2.6
    z_edges = np.exp(np.linspace(np.log(1.6), np.log(3.6), 9)) - 1.0
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    print(f"  z-shells: {z_edges}")

    # 8 theta bins log-spaced from 6 arcmin to 4 deg
    theta_deg = np.exp(np.linspace(np.log(0.1), np.log(4.0), 8))
    theta_rad = np.deg2rad(theta_deg)
    print(f"  theta [deg]: {theta_deg.round(3)}")

    # cap centres on a HEALPix-pixel-centre grid
    print(f"\nbuilding cap-centre grid at NSIDE={nside_centres} ...")
    theta_max = float(theta_rad.max())
    ra_c, dec_c, _ = cap_centre_grid(
        sel_map, nside_centres=nside_centres,
        theta_max_rad=theta_max,
        edge_buffer_frac=1.0,
        mask_threshold=0.5,
    )
    if centres_subsample > 1:
        ra_c = ra_c[::centres_subsample]
        dec_c = dec_c[::centres_subsample]
        print(f"  subsampling cap centres by {centres_subsample}x")
    print(f"  n_centres = {ra_c.size}")

    # 1) per-shell counts and sigma^2_obs
    print("\ncap counts + per-shell sigma^2 ...")
    t0 = time.perf_counter()
    N, A_cap = cone_shell_counts(
        ra_d, dec_d, z_d, theta_rad, z_edges, ra_c, dec_c,
        nside_lookup=512,
    )
    s2_obs = sigma2_estimate_cone_shell(N)
    t_est = time.perf_counter() - t0
    print(f"  done in {t_est:.1f}s; mean N range "
            f"{N.mean(axis=0).min():.2g} ... {N.mean(axis=0).max():.2g}")

    # 2) jackknife covariance (optional)
    s2_jk_err = None
    if do_jackknife:
        print("\njackknife covariance ...")
        t0 = time.perf_counter()
        s2_jk_mean, s2_jk_samples, s2_jk_cov = sigma2_cone_shell_jackknife(
            ra_d, dec_d, z_d, theta_rad, z_edges, ra_c, dec_c,
            n_regions=25, nside_jack=4, nside_lookup=512,
        )
        t_jk = time.perf_counter() - t0
        # diagonal -> standard errors per (theta, z)
        diag = np.diag(s2_jk_cov).reshape(s2_obs.shape)
        s2_jk_err = np.sqrt(np.maximum(diag, 0.0))
        print(f"  done in {t_jk:.1f}s; max relative error "
                f"{np.nanmax(s2_jk_err / np.abs(s2_obs + 1e-30)):.2g}")

    # 3) predicted forward model with a fiducial Quaia b(z)
    #    (Storey-Fisher+24 power-law approximation: b ~ 0.55 (1+z))
    z_grid = np.linspace(0.01, 4.0, 400)
    # data-driven dN/dz via histogram (mock-quality smoothing)
    hist, _ = np.histogram(z_d, bins=z_grid)
    nz_centres = 0.5 * (z_grid[:-1] + z_grid[1:])
    # Gaussian-smoothed histogram
    from scipy.ndimage import gaussian_filter1d
    dndz_smooth = gaussian_filter1d(hist.astype(np.float64), sigma=4)
    # interpolate back onto z_grid
    dndz = np.interp(z_grid, nz_centres, dndz_smooth)

    def b_of_z(z):
        return 0.55 * (1.0 + np.asarray(z))

    bias_per_shell = np.array([float(b_of_z(z)) for z in z_centres])
    print(f"\nfiducial b(z) per shell: {bias_per_shell.round(2)}")

    print("computing sigma^2_predicted across shells ...")
    t0 = time.perf_counter()
    s2_pred = np.asarray(sigma2_cone_shell_predicted_stack(
        theta_rad, z_edges, z_grid, dndz, fid,
        bias=bias_per_shell, sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=600,
    )).T  # (n_theta, n_zshell)
    t_pred = time.perf_counter() - t0
    print(f"  done in {t_pred:.1f}s")

    # 4) finite-difference d sigma^2 / d ln(1+z), measurement and prediction
    z_pivots, ds2_obs = dsigma2_dz_estimate(s2_obs, z_centres, log1pz=True)
    z_pivots_pred, ds2_pred = dsigma2_dz_cone_shell_predicted(
        theta_rad, z_edges, z_grid, dndz, fid,
        bias=bias_per_shell, sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=600, log1pz=True,
    )
    ds2_pred = np.asarray(ds2_pred).T  # (n_theta, n_pivots)

    # 5) bias/growth/geometry decomposition at the median theta
    j_med = theta_rad.size // 2
    theta_med = theta_rad[j_med]
    z_dec, lhs, c_b, c_g, c_geom = sigma2_cone_shell_decomposition(
        np.array([theta_med]), z_edges, z_grid, dndz, fid,
        bias_z=lambda z: float(b_of_z(z)),
        sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=600,
    )

    # ------ plotting --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # panel a: sigma^2(theta; z)
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for k in range(z_edges.size - 1):
        col = cmap(k / (z_edges.size - 1))
        ax.plot(theta_deg, s2_pred[:, k], color=col, lw=1.0)
        if s2_jk_err is not None:
            ax.errorbar(theta_deg, s2_obs[:, k],
                          yerr=s2_jk_err[:, k],
                          color=col, marker="o", ms=4, lw=0,
                          elinewidth=1.2, capsize=2,
                          label=f"z in [{z_edges[k]:.2f},{z_edges[k+1]:.2f}]")
        else:
            ax.plot(theta_deg, s2_obs[:, k],
                       color=col, marker="o", ms=4, lw=0,
                       label=f"z in [{z_edges[k]:.2f},{z_edges[k+1]:.2f}]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$\sigma^2_{\rm obs}(\theta; z, \Delta z)$")
    ax.set_title("(a) cap-shell variance vs. theta per shell")
    ax.legend(fontsize=6, ncol=1, loc="lower left")

    # panel b: d sigma^2 / d ln(1+z)
    ax = axes[1]
    for j in range(theta_rad.size):
        col = cmap(j / max(theta_rad.size - 1, 1))
        ax.plot(z_pivots_pred, ds2_pred[j],
                  color=col, lw=1.0)
        ax.plot(z_pivots, ds2_obs[j, :],
                  color=col, marker="o", ms=4, lw=0,
                  label=f"theta={theta_deg[j]:.2f} deg")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$d\sigma^2 / d\ln(1+z)$")
    ax.set_title("(b) lightcone derivative")
    ax.legend(fontsize=6, ncol=1, loc="best")

    # panel c: decomposition at theta_med
    ax = axes[2]
    ax.plot(z_dec, lhs[:, 0], "k-", lw=1.5,
              label=r"$d\ln\sigma^2/d\ln(1+z)$ (LHS)")
    ax.plot(z_dec, c_b, "C0--", lw=1.2,
              label=r"$2\,d\ln b/d\ln(1+z)$ (bias)")
    ax.plot(z_dec, c_g, "C1--", lw=1.2,
              label=r"$2\,d\ln D/d\ln(1+z)$ (growth)")
    ax.plot(z_dec, c_geom[:, 0], "C2--", lw=1.2,
              label=r"$n_{\rm eff}\,d\ln\chi/d\ln(1+z)$ (geometry)")
    ax.plot(z_dec, c_b + c_g + c_geom[:, 0], "k:", lw=1.0,
              label="sum of three terms")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"contribution to $d\ln\sigma^2/d\ln(1+z)$")
    ax.set_title(rf"(c) decomposition at $\theta$={np.rad2deg(theta_med):.2f} deg")
    ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    suffix = f"_sub{centres_subsample}" if centres_subsample > 1 else ""
    if n_data >= 200_000:
        suffix += f"_N{n_data // 1000}k"
    if nside_centres != 64:
        suffix += f"_nside{nside_centres}"
    out_path = os.path.join(FIG_DIR, f"quaia_sigma2_cone_shell{suffix}.png")
    plt.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
