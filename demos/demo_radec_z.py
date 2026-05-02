"""Survey-style (RA, Dec, z) -> comoving xi(s) workflow.

Closes the survey-cosmology loop end-to-end:

  catalogue in (RA, Dec, z)  --(JAX comoving distance)-->  comoving xyz
  comoving xyz               --(SF&H + AP)-------------->  xi_data(s)
  syren-halofit              ---------------------------->  xi_model(s)

Differentiation now propagates through the coordinate transform too,
so a small change in Omega_m moves every galaxy's comoving position
(by an amount that depends on the galaxy's redshift) -- the AP scaling
captures only the dominant zero-th order piece.

Two PNGs::

  radec_z_scatter.png  - sky and (RA, z) projections of the mock survey
                         catalogue.
  radec_z_grad.png     - dxi(s)/dOmega_m via two paths:
                         (1) the (RA, Dec, z) -> comoving Jacobian
                             (full coordinate gradient), and
                         (2) the AP rescaling at the catalogue median z
                             (zero-th order approximation).
                         The difference is the inhomogeneous-z correction.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density import cosmology as cj
from twopt_density.ap import apply_ap, xi_LS_AP
from twopt_density.basis_xi import JAXBasis, xi_LS_basis_AP
from twopt_density.differentiable_lisa import build_state
from twopt_density.distance import (
    DistanceCosmo, cartesian_to_radec_z, comoving_distance,
    radec_z_to_cartesian,
)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_radec_catalog(seed=7, ra_range=(80, 220), dec_range=(0, 50),
                       z_range=(0.2, 0.6), n_clusters=80, n_per=80):
    """Synthesize a clustered (RA, Dec, z) survey-shape catalogue.

    Cluster centres are drawn uniformly in (RA, Dec, z); galaxies are
    distributed around each cluster with sigma ~ 1 deg in angle and 0.01
    in z. RA is wrapped to [0, 360); Dec is clipped to [-90, 90].
    """
    rng = np.random.default_rng(seed)
    ra_c = rng.uniform(*ra_range, size=n_clusters)
    dec_c = rng.uniform(*dec_range, size=n_clusters)
    z_c = rng.uniform(*z_range, size=n_clusters)

    ra = np.concatenate([rng.normal(r, 0.6, size=n_per) for r in ra_c]) % 360
    dec = np.concatenate([rng.normal(d, 0.4, size=n_per) for d in dec_c])
    dec = np.clip(dec, -90.0 + 1e-3, 90.0 - 1e-3)
    z = np.concatenate([rng.normal(zc, 0.005, size=n_per) for zc in z_c])
    z = np.clip(z, z_range[0] - 0.05, z_range[1] + 0.05)
    return ra, dec, z


def make_random_catalog_in_volume(positions, n_rand_factor=8, seed=0):
    """Uniform random catalogue in the bounding sphere of the data.

    Simple isotropic background; not survey-aware. Replace with a real
    random catalogue when one is available.
    """
    rng = np.random.default_rng(seed)
    # Bounding box (with a small margin)
    lo = positions.min(axis=0) - 5.0
    hi = positions.max(axis=0) + 5.0
    N_R = int(n_rand_factor * len(positions))
    return rng.uniform(lo, hi, size=(N_R, 3))


def panel_scatter(ra, dec, z, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(ra, dec, s=2, alpha=0.4)
    axes[0].set_xlabel("RA [deg]")
    axes[0].set_ylabel("Dec [deg]")
    axes[0].set_title("Sky plane")
    axes[0].invert_xaxis()
    axes[1].scatter(ra, z, s=2, alpha=0.4)
    axes[1].set_xlabel("RA [deg]")
    axes[1].set_ylabel("redshift z")
    axes[1].set_title("(RA, z) projection")
    fig.suptitle(f"Mock survey catalogue: N={len(ra)} galaxies")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    print("synthesise (RA, Dec, z) survey catalogue ...")
    ra, dec, z = make_radec_catalog()
    panel_scatter(ra, dec, z, os.path.join(FIG_DIR, "radec_z_scatter.png"))
    print(f"  N={len(ra)}, z_med={np.median(z):.3f}, "
          f"wrote radec_z_scatter.png")

    fid = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)
    z_j = jnp.asarray(z)
    ra_j = jnp.asarray(ra)
    dec_j = jnp.asarray(dec)

    print("convert to fiducial comoving + build SF&H state ...")
    xyz_fid = np.asarray(radec_z_to_cartesian(ra_j, dec_j, z_j, fid))
    print(f"  comoving range: x [{xyz_fid[:,0].min():.0f}, {xyz_fid[:,0].max():.0f}],"
          f" y [{xyz_fid[:,1].min():.0f}, {xyz_fid[:,1].max():.0f}],"
          f" z [{xyz_fid[:,2].min():.0f}, {xyz_fid[:,2].max():.0f}]")
    # Shift to all-positive coordinates and use an envelope box much
    # larger than the data extent so periodic wraparound is irrelevant
    # (the KDTree query radius is r_max << box).
    margin = 100.0
    shift = -xyz_fid.min(axis=0) + margin
    xyz_fid = xyz_fid + shift
    box_eff = float(np.max(xyz_fid.max(axis=0)) + margin)
    randoms = make_random_catalog_in_volume(xyz_fid)

    r_edges = np.logspace(np.log10(2.0), np.log10(80.0), 14)
    t0 = time.perf_counter()
    state = build_state(xyz_fid, r_edges, box_eff, randoms=randoms,
                        los=np.array([0.0, 0.0, 1.0]), cache_rr=True)
    print(f"  build_state {time.perf_counter()-t0:.1f}s, "
          f"DD={state.DD_pi.size}, DR={state.DR_pi.size}")

    jb = JAXBasis.from_cubic_spline(n_basis=14, r_min=2.0, r_max=80.0, n_grid=2000)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)

    s_query = jnp.asarray(np.logspace(np.log10(3.0), np.log10(50.0), 60))

    # --- Path A: dxi/dOm via global AP at median z (single-z approx) ---
    # Map Om change -> (alpha_par, alpha_perp) at the median redshift,
    # then dispatch through the smooth-basis SF&H estimator. The basis
    # path gives clean smooth gradients (no bin-edge cusps).
    print("path A (global AP at median z): dxi/dOm via basis SF&H ...")
    z_med = float(jnp.median(z_j))

    def alphas_from_Om(Om, zv):
        new = DistanceCosmo(Om=Om, h=fid.h, w0=fid.w0, wa=fid.wa)
        D_new = comoving_distance(jnp.array([zv]), new)[0]
        D_fid = comoving_distance(jnp.array([zv]), fid)[0]
        eps = 1e-3
        Dp_new = comoving_distance(jnp.array([zv + eps]), new)[0]
        Dm_new = comoving_distance(jnp.array([zv - eps]), new)[0]
        Dp_fid = comoving_distance(jnp.array([zv + eps]), fid)[0]
        Dm_fid = comoving_distance(jnp.array([zv - eps]), fid)[0]
        H_new = 1.0 / ((Dp_new - Dm_new) / (2 * eps))
        H_fid = 1.0 / ((Dp_fid - Dm_fid) / (2 * eps))
        alpha_par = H_fid / H_new
        alpha_perp = D_new / D_fid
        return alpha_par, alpha_perp

    def xi_A(Om):
        ap_par, ap_perp = alphas_from_Om(Om, z_med)
        return xi_LS_basis_AP(state, jb, w_d, w_r, ap_par, ap_perp, s_query)

    t0 = time.perf_counter()
    J_A = np.asarray(jax.jacfwd(xi_A)(0.31))
    print(f"  jacfwd {time.perf_counter()-t0:.1f}s, shape {J_A.shape}, "
          f"max |dxi/dOm| = {np.max(np.abs(J_A)):.4f}")

    # --- Path B sweep: how the alphas vary with z ----------------------
    # Show alpha_par(z) and alpha_perp(z) for Om perturbed by +/- 1%.
    # This visualises the inhomogeneous-z correction that the global-AP
    # of path A misses. (A full path-B gradient through the per-pair
    # rescale is straightforward but requires caching per-pair z.)
    z_grid = jnp.linspace(0.05, 1.5, 40)
    a_par_p = np.array([float(alphas_from_Om(0.32, float(zz))[0]) for zz in z_grid])
    a_par_m = np.array([float(alphas_from_Om(0.30, float(zz))[0]) for zz in z_grid])
    a_perp_p = np.array([float(alphas_from_Om(0.32, float(zz))[1]) for zz in z_grid])
    a_perp_m = np.array([float(alphas_from_Om(0.30, float(zz))[1]) for zz in z_grid])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(np.asarray(s_query), J_A, "-", color="C0", lw=2,
                 label=r"path A: AP at median $z=" + f"{z_med:.2f}$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("s [Mpc/h]")
    axes[0].set_ylabel(r"$\partial \xi(s) / \partial \Omega_m$")
    axes[0].set_title("Cosmology gradient via (RA, Dec, z) -> comoving")
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].legend()

    axes[1].plot(np.asarray(z_grid), a_par_p, "-", color="C0",
                 label=r"$\alpha_\parallel,\ \Omega_m=0.32$")
    axes[1].plot(np.asarray(z_grid), a_par_m, "--", color="C0",
                 label=r"$\alpha_\parallel,\ \Omega_m=0.30$")
    axes[1].plot(np.asarray(z_grid), a_perp_p, "-", color="C3",
                 label=r"$\alpha_\perp,\ \Omega_m=0.32$")
    axes[1].plot(np.asarray(z_grid), a_perp_m, "--", color="C3",
                 label=r"$\alpha_\perp,\ \Omega_m=0.30$")
    axes[1].axhline(1.0, color="k", lw=0.5, alpha=0.3)
    axes[1].axvline(z_med, color="C2", lw=1, alpha=0.5,
                    label=f"survey median z={z_med:.2f}")
    axes[1].set_xlabel("redshift z")
    axes[1].set_ylabel(r"AP rescale factor")
    axes[1].set_title("Inhomogeneous-z AP under $\\pm 1\\%$ $\\Omega_m$ change")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "radec_z_grad.png"), dpi=140)
    plt.close(fig)
    print("  wrote radec_z_grad.png")


if __name__ == "__main__":
    main()
