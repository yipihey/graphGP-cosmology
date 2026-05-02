"""Cosmology -> xi(r) forward model overlaid on basis-projected xi_data.

Closes the loop on the differentiable cosmology pipeline:

    syren-halofit / syren-new  ->  P_NL(k, theta)  ->  xi_model(s, theta)
                                                            ||
    galaxies, randoms          ->  basis-projected xi_data(s)

Two PNGs in ``demos/figures/``:

  cosmology_pk.png    - syren-halofit & syren-new linear + nonlinear P(k);
                        sweep one parameter (Om) for each variant.
  cosmology_xi.png    - xi_model(s) under an Om sweep, with the
                        basis-projected xi_data from the existing AP demo
                        catalog overlaid as markers.
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
from twopt_density.basis_xi import JAXBasis, xi_LS_basis
from twopt_density.differentiable_lisa import build_state
from twopt_density.spectra import make_log_k_grid, xi_from_Pk


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_catalog(seed=7, box=400.0, n_centers=40, n_per=200):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([rng.normal(c, 8.0, size=(n_per, 3)) for c in centers])
    pts = np.mod(pts, box).astype(np.float64)
    randoms = rng.uniform(0, box, size=(4 * len(pts), 3)).astype(np.float64)
    return pts, randoms, box


def panel_pk(out_path):
    """Compare syren-halofit and syren-new under an Om sweep."""
    k = make_log_k_grid(1e-3, 1e1, 1500)
    base = dict(Ob=0.049, h=0.68, ns=0.965, a=1.0)
    Om_values = [0.27, 0.31, 0.35]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ax_h, ax_n = axes
    for Om, c in zip(Om_values, ["C0", "C1", "C2"]):
        # syren-halofit
        P_lin = cj.plin_emulated(k, sigma8=0.8, Om=Om, **base)
        P_NL = cj.run_halofit(k, sigma8=0.8, Om=Om, **base)
        ax_h.loglog(np.asarray(k), np.asarray(P_lin), "--", color=c, alpha=0.6,
                    label=rf"linear, $\Omega_m={Om}$")
        ax_h.loglog(np.asarray(k), np.asarray(P_NL), "-", color=c, lw=2,
                    label=rf"nonlinear, $\Omega_m={Om}$")
        # syren-new
        P_lin_n = cj.plin_new_emulated(k, As=2.1, Om=Om, mnu=0.06, w0=-1.0, wa=0.0, **base)
        P_NL_n = cj.pnl_new_emulated(k, As=2.1, Om=Om, mnu=0.06, w0=-1.0, wa=0.0, **base)
        ax_n.loglog(np.asarray(k), np.asarray(P_lin_n), "--", color=c, alpha=0.6,
                    label=rf"linear, $\Omega_m={Om}$")
        ax_n.loglog(np.asarray(k), np.asarray(P_NL_n), "-", color=c, lw=2,
                    label=rf"nonlinear, $\Omega_m={Om}$")
    ax_h.set_title(r"syren-halofit ($\sigma_8=0.8$, $\Lambda$CDM)")
    ax_n.set_title(r"syren-new ($A_s=2.1$, $m_\nu=0.06\,\mathrm{eV}$, $w_0=-1$)")
    for ax in axes:
        ax.set_xlabel("k [h/Mpc]")
        ax.legend(fontsize=8)
    ax_h.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_xi(out_path):
    """Cosmology xi(s) under Om sweep + data xi from basis SF&H, fit linear bias."""
    pts, randoms, box = make_catalog()
    r_edges = np.logspace(np.log10(2.0), np.log10(80.0), 14)
    los = np.array([0.0, 0.0, 1.0])

    print(f"  build_state with N_D={len(pts)}, N_R={len(randoms)}...")
    t0 = time.perf_counter()
    state = build_state(pts, r_edges, box, randoms=randoms, los=los, cache_rr=True)
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    jb = JAXBasis.from_cubic_spline(n_basis=18, r_min=2.0, r_max=80.0, n_grid=4000)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s_data = jnp.asarray(np.logspace(np.log10(2.5), np.log10(60.0), 80))

    print("  basis xi_data ...")
    xi_data = np.asarray(xi_LS_basis(state, jb, w_d, w_r, s_data))

    k = make_log_k_grid(1e-4, 1e2, 2000)
    base = dict(Ob=0.049, h=0.68, ns=0.965, a=1.0)
    s_model = jnp.asarray(np.logspace(np.log10(2.0), np.log10(80.0), 60))

    # --- Fit linear bias b^2 so the toy catalog matches a chosen cosmology
    # --- on intermediate scales (5 < s < 30 Mpc/h). The toy catalog has
    # --- Gaussian blobs that dwarf any reasonable matter clustering, so a
    # --- single linear amplitude rescaling is enough to compare shapes.
    s_fit = jnp.asarray(np.logspace(np.log10(5.0), np.log10(30.0), 30))
    P_fid = cj.run_halofit(k, sigma8=0.8, Om=0.31, **base)
    xi_fid = np.asarray(xi_from_Pk(s_fit, k, P_fid))
    xi_data_fit = np.asarray(xi_LS_basis(state, jb, w_d, w_r, s_fit))
    b2 = float(np.dot(xi_data_fit, xi_fid) / np.dot(xi_fid, xi_fid))
    print(f"  fitted linear b^2 = {b2:.2f}")

    Om_values = [0.27, 0.31, 0.35]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(np.asarray(s_data), xi_data, "ok", markersize=4, alpha=0.7,
            label="basis $\\xi_{data}$ (clustered toy catalog)")
    for Om, c in zip(Om_values, ["C0", "C1", "C2"]):
        P_NL = cj.run_halofit(k, sigma8=0.8, Om=Om, **base)
        xi_m = b2 * np.asarray(xi_from_Pk(s_model, k, P_NL))
        ax.plot(np.asarray(s_model), xi_m, "-", color=c, lw=2,
                label=rf"$b^2\, \xi_{{model}}(s|\Omega_m={Om})$, $b^2={b2:.1f}$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title("syren-halofit forward model vs basis-projected data (linear-bias fit)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_grad(out_path):
    """jax.grad of xi_model wrt cosmological parameters on a fine s-grid."""
    k = make_log_k_grid(1e-4, 1e2, 2000)
    r = jnp.asarray(np.logspace(np.log10(2.0), np.log10(80.0), 80))
    base = dict(sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)

    def xi_of(theta):
        # theta = (Om, sigma8, h)
        P = cj.run_halofit(k, sigma8=theta[1], Om=theta[0], Ob=base["Ob"],
                           h=theta[2], ns=base["ns"], a=base["a"])
        return xi_from_Pk(r, k, P)

    print("  jacfwd cosmology gradient ...")
    t0 = time.perf_counter()
    J = jax.jacfwd(xi_of)(jnp.array([base["Om"], base["sigma8"], base["h"]]))
    J = np.asarray(J)
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(np.asarray(r), J[:, 0], "-", color="C0", lw=2, label=r"$\partial \xi / \partial \Omega_m$")
    ax.plot(np.asarray(r), J[:, 1], "-", color="C1", lw=2, label=r"$\partial \xi / \partial \sigma_8$")
    ax.plot(np.asarray(r), J[:, 2], "-", color="C2", lw=2, label=r"$\partial \xi / \partial h$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\partial \xi / \partial \theta$")
    ax.set_title("Cosmology sensitivity of $\\xi(s)$ via jax.jacfwd through syren-halofit")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    print("panel: cosmology P(k)")
    panel_pk(os.path.join(FIG_DIR, "cosmology_pk.png"))
    print("  wrote cosmology_pk.png")

    print("panel: cosmology xi(s) vs basis xi_data")
    panel_xi(os.path.join(FIG_DIR, "cosmology_xi.png"))
    print("  wrote cosmology_xi.png")

    print("panel: cosmology gradient")
    panel_grad(os.path.join(FIG_DIR, "cosmology_grad.png"))
    print("  wrote cosmology_grad.png")


if __name__ == "__main__":
    main()
