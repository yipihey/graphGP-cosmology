"""Per-galaxy Alcock-Paczynski sensitivity demo.

Run a clustered catalog through ``differentiable_lisa`` + ``ap``, sweep
``(alpha_par, alpha_perp)`` away from (1,1), and visualise the response:

  Panel 1 - projected scatter at three AP points: shows visible
            stretching/compression of the catalog along the LOS.
  Panel 2 - per-particle delta_i histogram at the same three AP points:
            tracks how the density PDF skews under AP.
  Panel 3 - xi(s) curves under the AP sweep + jax.grad overlay
            (d xi / d alpha) computed via the soft-binned variant.

Outputs three PNGs in ``demos/figures/``.
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

from twopt_density.differentiable_lisa import (
    build_state,
    per_particle_overdensity,
)
from twopt_density.ap import (
    apply_ap,
    xi_LS_AP,
    xi_LS_AP_soft,
    per_particle_overdensity_AP,
)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _ap_distort_positions(pts, los, alpha_par, alpha_perp):
    """Apply AP to data points (used only for the scatter visualisation)."""
    par = pts @ los
    perp = pts - par[:, None] * los[None, :]
    return alpha_par * par[:, None] * los[None, :] + alpha_perp * perp


def make_catalog(seed=7, box=400.0, n_centers=40, n_per=200):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([rng.normal(c, 8.0, size=(n_per, 3)) for c in centers])
    pts = np.mod(pts, box).astype(np.float64)
    randoms = rng.uniform(0, box, size=(4 * len(pts), 3)).astype(np.float64)
    return pts, randoms, box


def panel_scatter(pts, box, los, ap_points, out_path):
    """Scatter projection on x-z plane under three AP choices."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    # Centre on the box for a clean visual.
    centred = pts - box / 2
    for ax, (apar, aperp) in zip(axes, ap_points):
        warped = _ap_distort_positions(centred, los, apar, aperp)
        ax.scatter(warped[:, 0], warped[:, 2], s=1.5, alpha=0.4, c="C0")
        ax.set_aspect("equal")
        ax.set_title(rf"$\alpha_\parallel={apar},\ \alpha_\perp={aperp}$")
        ax.set_xlim(-box / 2 - 30, box / 2 + 30)
        ax.set_ylim(-box / 2 - 30, box / 2 + 30)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_ylabel("z (LOS) [Mpc/h]")
    for ax in axes:
        ax.set_xlabel("x [Mpc/h]")
    fig.suptitle("Catalog projection under AP distortion (LOS = $\\hat z$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_pdf(state, w_d, w_r, r_edges, ap_points, out_path):
    """delta_i PDF under three AP choices."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.linspace(-1.0, 4.0, 70)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for apar, aperp in ap_points:
        ap = apply_ap(state, jnp.asarray(r_edges), apar, aperp)
        delta = np.asarray(per_particle_overdensity_AP(
            state, ap, w_d, w_r, aggregation="RR",
        ))
        h, _ = np.histogram(delta, bins=edges, density=True)
        ax.plot(
            centres, h, lw=2,
            label=rf"$\alpha_\parallel={apar},\ \alpha_\perp={aperp}$",
        )
    ax.set_xlabel(r"$\delta_i$ (per-galaxy overdensity)")
    ax.set_ylabel("PDF")
    ax.set_title("Per-galaxy density PDF under AP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_xi(state, w_d, w_r, r_edges, ap_points, out_path):
    """xi(s) curves and dxi/dalpha at fiducial."""
    re = jnp.asarray(r_edges)
    s_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_xi, ax_g = axes

    for apar, aperp in ap_points:
        ap = apply_ap(state, re, apar, aperp)
        xi = np.asarray(xi_LS_AP(state, ap, w_d, w_r))
        ax_xi.plot(
            s_centres, xi, marker="o", lw=2,
            label=rf"$\alpha_\parallel={apar},\ \alpha_\perp={aperp}$",
        )
    ax_xi.set_xscale("log")
    ax_xi.set_xlabel("s [Mpc/h]")
    ax_xi.set_ylabel(r"$\xi(s)$")
    ax_xi.set_title("Two-point function under AP sweep")
    ax_xi.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_xi.legend()

    # Per-bin dxi/d alpha at fiducial via jax.jacobian on the soft path.
    def xi_fn(apar_aperp):
        return xi_LS_AP_soft(
            state, re, w_d, w_r, apar_aperp[0], apar_aperp[1],
        )

    J = jax.jacobian(xi_fn)(jnp.array([1.0, 1.0]))
    J = np.asarray(J)  # (n_bins, 2)
    ax_g.plot(s_centres, J[:, 0], "o-", lw=2, label=r"$\partial \xi / \partial \alpha_\parallel$")
    ax_g.plot(s_centres, J[:, 1], "s-", lw=2, label=r"$\partial \xi / \partial \alpha_\perp$")
    ax_g.set_xscale("log")
    ax_g.set_xlabel("s [Mpc/h]")
    ax_g.set_ylabel(r"$\partial \xi / \partial \alpha$")
    ax_g.set_title("AP sensitivity (jax.grad, soft-binned)")
    ax_g.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_g.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    pts, randoms, box = make_catalog()
    r_edges = np.logspace(np.log10(2.0), np.log10(80.0), 14)
    los = np.array([0.0, 0.0, 1.0])

    print(f"N_D={len(pts)}, N_R={len(randoms)}, box={box}")
    t0 = time.perf_counter()
    state = build_state(pts, r_edges, box, randoms=randoms, los=los, cache_rr=True)
    print(f"build_state: {time.perf_counter() - t0:.2f} s "
          f"({state.DD_pi.size} DD pairs, {state.DR_pi.size} DR pairs)")

    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    ap_points = [(0.85, 1.15), (1.0, 1.0), (1.15, 0.85)]

    panel_scatter(pts, box, los, ap_points, os.path.join(FIG_DIR, "ap_scatter.png"))
    print("wrote ap_scatter.png")

    panel_pdf(state, w_d, w_r, r_edges, ap_points,
              os.path.join(FIG_DIR, "ap_pdf.png"))
    print("wrote ap_pdf.png")

    panel_xi(state, w_d, w_r, r_edges, ap_points,
             os.path.join(FIG_DIR, "ap_xi.png"))
    print("wrote ap_xi.png")


if __name__ == "__main__":
    main()
