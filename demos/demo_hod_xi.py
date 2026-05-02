"""HOD-as-weight demo: FakeSim halos -> Zheng07 weights -> xi(s) via SF&H.

The HOD's expected per-halo galaxy occupation <N_gal>(M_h | theta_HOD)
is treated as a differentiable per-halo weight feeding straight into
the existing SF&H + AP estimator. No Monte-Carlo galaxy sampling: we
compute xi at the population mean directly.

Three PNGs::

  hod_occupation.png  - <N_cen>, <N_sat>, <N_gal> vs log10(M_h) for two
                        HODs (sweep alpha to show effect).
  hod_xi.png          - basis-projected xi(s) for the two HODs (host
                        halos as the underlying tracers, weighted by
                        <N_gal>) overlaid on syren-halofit cosmology
                        forward model.
  hod_grad.png        - jax.jacfwd of xi(s) w.r.t. (logMmin, alpha) on a
                        fine s-grid -- continuous HOD-parameter
                        sensitivity at every separation.
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

from halotools.sim_manager import FakeSim

from twopt_density import cosmology as cj
from twopt_density.basis_xi import JAXBasis, xi_LS_basis_AP
from twopt_density.differentiable_lisa import build_state
from twopt_density.halo_loader import halocat_to_state_inputs
from twopt_density.hod import (
    Zheng07Params,
    mean_ncen_zheng07,
    mean_ngal_zheng07,
    mean_nsat_zheng07,
)
from twopt_density.spectra import (
    FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog,
)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def panel_occupation(out_path):
    M = jnp.logspace(11, 15, 200)
    p_lo = Zheng07Params(logMmin=12.5, sigma_logM=0.4, logM0=12.0, logM1=13.6, alpha=0.95)
    p_hi = Zheng07Params(logMmin=12.5, sigma_logM=0.4, logM0=12.0, logM1=13.6, alpha=1.30)

    fig, ax = plt.subplots(figsize=(8, 5))
    for p, ls, lab in [(p_lo, "-", r"$\alpha=0.95$"), (p_hi, "--", r"$\alpha=1.30$")]:
        ncen = np.asarray(mean_ncen_zheng07(M, p))
        nsat = np.asarray(mean_nsat_zheng07(M, p, modulate_with_ncen=True))
        ntot = np.asarray(mean_ngal_zheng07(M, p))
        ax.loglog(np.asarray(M), ncen, "C0" + ls, lw=2, alpha=0.6,
                  label=rf"$\langle N_{{cen}}\rangle$, {lab}")
        ax.loglog(np.asarray(M), nsat, "C1" + ls, lw=2, alpha=0.6,
                  label=rf"$\langle N_{{sat}}\rangle$, {lab}")
        ax.loglog(np.asarray(M), ntot, "C2" + ls, lw=2,
                  label=rf"$\langle N_{{gal}}\rangle$, {lab}")
    ax.set_xlabel(r"$M_h \ [M_\odot]$")
    ax.set_ylabel(r"mean occupation per halo")
    ax.set_ylim(1e-2, 1e3)
    ax.set_title("Zheng+07 HOD: alpha sweep")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_xi(out_path, state, jb, w_lo, w_hi, w_rand, label_lo, label_hi):
    s_data = jnp.asarray(np.logspace(np.log10(2.5), np.log10(60.0), 80))
    xi_lo = np.asarray(xi_LS_basis_AP(state, jb, w_lo, w_rand, 1.0, 1.0, s_data))
    xi_hi = np.asarray(xi_LS_basis_AP(state, jb, w_hi, w_rand, 1.0, 1.0, s_data))

    # Cosmology overlay (b^2 fit on mid-range to xi_lo for reference)
    k = make_log_k_grid(1e-4, 1e2, 2048)
    fft = FFTLogP2xi(k, l=0)
    s_fit = jnp.asarray(np.logspace(np.log10(5.0), np.log10(30.0), 30))
    P = cj.run_halofit(k, sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)
    xi_fid = np.asarray(xi_from_Pk_fftlog(s_fit, fft, P))
    xi_lo_fit = np.asarray(xi_LS_basis_AP(state, jb, w_lo, w_rand, 1.0, 1.0, s_fit))
    b2 = float(np.dot(xi_lo_fit, xi_fid) / np.dot(xi_fid, xi_fid))
    s_model = jnp.asarray(np.logspace(np.log10(2.5), np.log10(60.0), 100))
    xi_m = b2 * np.asarray(xi_from_Pk_fftlog(s_model, fft, P))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(np.asarray(s_data), xi_lo, "o", color="C0", markersize=4,
            label=label_lo)
    ax.plot(np.asarray(s_data), xi_hi, "s", color="C1", markersize=4,
            label=label_hi)
    ax.plot(np.asarray(s_model), xi_m, "-", color="k", lw=1.5, alpha=0.7,
            label=rf"syren-halofit, $b^2={b2:.1f}$ (fit to {label_lo})")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title("HOD weights -> SF&H xi(s) on FakeSim host halos")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_grad(out_path, state, jb, M_h, w_rand):
    """jax.jacfwd of xi(s) w.r.t. (logMmin, alpha)."""
    s = jnp.asarray(np.logspace(np.log10(2.5), np.log10(50.0), 60))
    fixed = dict(sigma_logM=0.4, logM0=12.0, logM1=13.6)

    def xi_of(theta):
        p = Zheng07Params(
            logMmin=theta[0], sigma_logM=fixed["sigma_logM"],
            logM0=fixed["logM0"], logM1=fixed["logM1"], alpha=theta[1],
        )
        w = mean_ngal_zheng07(M_h, p)
        return xi_LS_basis_AP(state, jb, w, w_rand, 1.0, 1.0, s)

    print("  jacfwd HOD gradient ...")
    t0 = time.perf_counter()
    J = jax.jacfwd(xi_of)(jnp.array([12.5, 1.10]))   # (n_s, 2)
    J = np.asarray(J)
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(np.asarray(s), J[:, 0], "-", color="C0", lw=2,
            label=r"$\partial \xi / \partial \log M_{min}$")
    ax.plot(np.asarray(s), J[:, 1], "-", color="C1", lw=2,
            label=r"$\partial \xi / \partial \alpha$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\partial \xi / \partial \theta_{HOD}$")
    ax.set_title("HOD parameter sensitivity via jax.jacfwd")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_clustered_halocat(seed=7, box=400.0, n_centers=80, n_per=120,
                            log_mh_mean=13.0, log_mh_sigma=0.7):
    """Toy clustered halo catalog: Gaussian-blob positions + log-normal masses.

    FakeSim's host halos are uniformly distributed (subhalos cluster, but
    the HOD operates on hosts), so the SF&H xi(s) shows no clustering
    signal. For a representative xi(s) overlay we use a synthetic
    clustered halo catalog with log-normal mvir distribution.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([rng.normal(c, 8.0, size=(n_per, 3)) for c in centers])
    pts = np.mod(pts, box).astype(np.float64)
    M_h = 10.0 ** rng.normal(log_mh_mean, log_mh_sigma, size=len(pts))
    return pts, M_h, box


def main():
    print("FakeSim halocat (API smoke test) ...")
    fake = FakeSim(num_massive_hosts=2000, num_subs_per_massive_host=4, redshift=0)
    _, M_fake, _, w_fake = halocat_to_state_inputs(fake)
    print(f"  N_host={len(M_fake)}, sum(w_fake)={float(w_fake.sum()):.0f}")

    print("Clustered toy halocat (Gaussian blobs + log-normal mvir) ...")
    p_lo = Zheng07Params(logMmin=12.5, sigma_logM=0.4, logM0=12.0, logM1=13.6, alpha=0.95)
    p_hi = Zheng07Params(logMmin=12.5, sigma_logM=0.4, logM0=12.0, logM1=13.6, alpha=1.30)

    positions, M_h, Lbox = make_clustered_halocat()
    M_h = jnp.asarray(M_h)
    from twopt_density.hod import mean_ngal_zheng07 as _ngal
    w_lo = _ngal(M_h, p_lo)
    w_hi = _ngal(M_h, p_hi)
    print(f"  N_halo={len(M_h)}, Lbox={Lbox}, "
          f"sum(w_lo)={float(w_lo.sum()):.0f}, sum(w_hi)={float(w_hi.sum()):.0f}")

    print("panel: HOD occupation curves")
    panel_occupation(os.path.join(FIG_DIR, "hod_occupation.png"))
    print("  wrote hod_occupation.png")

    print("build_state on host halos + matched random catalog ...")
    rng = np.random.default_rng(42)
    randoms = rng.uniform(0, Lbox, size=(8 * len(positions), 3)).astype(np.float64)
    r_edges = np.logspace(np.log10(2.0), np.log10(80.0), 14)
    t0 = time.perf_counter()
    state = build_state(positions, r_edges, Lbox, randoms=randoms,
                        los=np.array([0.0, 0.0, 1.0]), cache_rr=True)
    print(f"  ({time.perf_counter() - t0:.1f}s, "
          f"DD={state.DD_pi.size}, DR={state.DR_pi.size})")

    jb = JAXBasis.from_cubic_spline(n_basis=18, r_min=2.0, r_max=80.0, n_grid=4000)
    w_rand = jnp.ones(state.N_R)

    print("panel: HOD-weighted xi(s) + cosmology overlay")
    panel_xi(
        os.path.join(FIG_DIR, "hod_xi.png"),
        state, jb, w_lo, w_hi, w_rand,
        label_lo=r"$\alpha=0.95$", label_hi=r"$\alpha=1.30$",
    )
    print("  wrote hod_xi.png")

    print("panel: HOD parameter gradient")
    panel_grad(
        os.path.join(FIG_DIR, "hod_grad.png"),
        state, jb, jnp.asarray(M_h), w_rand,
    )
    print("  wrote hod_grad.png")


if __name__ == "__main__":
    main()
