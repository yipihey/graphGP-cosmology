"""SF&H basis-projected xi(s) under AP -- smooth, differentiable.

Replaces the binning step with a cubic B-spline basis projection over
log s. ``xi(s)`` is a continuous function of s and of (alpha_par,
alpha_perp), so ``jax.grad`` flows without bin-edge cusps. Compared
side-by-side with the binned AP estimator from
``demo_ap_sensitivity.py``.

Two PNGs in ``demos/figures/``:

  ap_basis_xi.png    - binned LS xi(s) markers + smooth basis xi(s)
                       under three AP choices.
  ap_basis_grad.png  - jax.jacobian d xi / d alpha at fiducial,
                       evaluated on a fine continuous s-grid.
"""

from __future__ import annotations

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twopt_density.differentiable_lisa import build_state
from twopt_density.ap import apply_ap, xi_LS_AP
from twopt_density.basis_xi import JAXBasis, xi_LS_basis_AP


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_catalog(seed=7, box=400.0, n_centers=40, n_per=200):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([rng.normal(c, 8.0, size=(n_per, 3)) for c in centers])
    pts = np.mod(pts, box).astype(np.float64)
    randoms = rng.uniform(0, box, size=(4 * len(pts), 3)).astype(np.float64)
    return pts, randoms, box


def main():
    pts, randoms, box = make_catalog()
    r_edges = np.logspace(np.log10(2.0), np.log10(80.0), 14)
    los = np.array([0.0, 0.0, 1.0])

    print(f"N_D={len(pts)}, N_R={len(randoms)}, box={box}")
    t0 = time.perf_counter()
    state = build_state(pts, r_edges, box, randoms=randoms, los=los, cache_rr=True)
    print(f"build_state: {time.perf_counter() - t0:.2f} s")

    jb = JAXBasis.from_cubic_spline(
        n_basis=18, r_min=2.0, r_max=80.0, n_grid=4000,
    )
    print(f"basis: cubic B-spline, n_basis={jb.n_basis}")

    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    re = jnp.asarray(r_edges)
    s_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    s_fine = jnp.asarray(np.logspace(np.log10(2.5), np.log10(75.0), 200))

    ap_points = [(0.85, 1.15), (1.0, 1.0), (1.15, 0.85)]
    colors = ["C0", "C1", "C2"]

    # --- Panel: binned vs basis xi(s) under AP -----------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for (apar, aperp), c in zip(ap_points, colors):
        # Binned (markers)
        ap = apply_ap(state, re, apar, aperp)
        xi_b = np.asarray(xi_LS_AP(state, ap, w_d, w_r))
        ax.plot(
            s_centres, xi_b, "o", color=c, markersize=6, alpha=0.7,
            label=rf"binned, $\alpha_\parallel={apar},\ \alpha_\perp={aperp}$",
        )
        # Basis-projected (smooth)
        xi_f = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, apar, aperp, s_fine))
        ax.plot(np.asarray(s_fine), xi_f, "-", color=c, lw=2, alpha=0.9,
                label=rf"basis,    $\alpha_\parallel={apar},\ \alpha_\perp={aperp}$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title("SF&H basis-projected $\\xi(s)$ vs binned $\\xi$ under AP")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    out_xi = os.path.join(FIG_DIR, "ap_basis_xi.png")
    fig.savefig(out_xi, dpi=140)
    plt.close(fig)
    print(f"wrote {out_xi}")

    # --- Panel: smooth d xi / d alpha along the fine s-grid ----------
    def xi_fn(alpha):
        return xi_LS_basis_AP(state, jb, w_d, w_r, alpha[0], alpha[1], s_fine)

    t0 = time.perf_counter()
    # Forward-mode: 2 inputs, ~200 outputs -> 2 forward passes.
    J = jax.jacfwd(xi_fn)(jnp.array([1.0, 1.0]))  # (n_query, 2)
    J = np.asarray(J)
    print(f"jacobian: {time.perf_counter() - t0:.2f} s, shape {J.shape}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(np.asarray(s_fine), J[:, 0], "-", color="C0", lw=2,
            label=r"$\partial \xi(s) / \partial \alpha_\parallel$")
    ax.plot(np.asarray(s_fine), J[:, 1], "-", color="C3", lw=2,
            label=r"$\partial \xi(s) / \partial \alpha_\perp$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\partial \xi / \partial \alpha$")
    ax.set_title("AP sensitivity (continuous s) via jax.jacobian on basis $\\xi$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_g = os.path.join(FIG_DIR, "ap_basis_grad.png")
    fig.savefig(out_g, dpi=140)
    plt.close(fig)
    print(f"wrote {out_g}")


if __name__ == "__main__":
    main()
