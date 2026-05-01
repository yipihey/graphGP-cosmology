"""Scan particle count, track weighted-xi offset, and identify the role
of the average weight ``<w>`` in setting the recovered amplitude.

Question
--------
If we *don't* mean-subtract the KDE input, the weights have ``<w> > 1``
because the KDE samples preferentially over-dense regions. How does the
average weight set the offset and amplitude of the recovered
``xi_w(r) = DD_w(r)/RR(r) - 1``?

Analytical prediction
---------------------
Write ``w_i = w_bar + delta_w_i`` with ``<delta_w> = 0``. The weighted
pair count in bin ``j`` is

    DD_w_j = sum_{i<k in bin j} w_i w_k
           = sum_{i<k} (w_bar + delta_w_i)(w_bar + delta_w_k) 1[r_ik in B_j]
           = w_bar^2 DD_j  +  w_bar [sum delta_w_i b_i^j]
                            +  sum_{i<k} delta_w_i delta_w_k 1[r in B_j]

The middle term has zero mean by construction. Taking expectations
(treating the field as an ensemble):

    E[DD_w_j] = w_bar^2 DD_j + xi_ww(r_j) DD_j
              = (w_bar^2 + xi_ww(r_j)) DD_j

where ``xi_ww(r) = <delta_w_i delta_w_k | r_ik = r>`` is the weight-weight
correlation function.  Since ``DD_j / RR_j = 1 + xi_LS(r_j)``,

    xi_w(r) = DD_w(r)/RR(r) - 1
            = (1 + xi_LS(r))(w_bar^2 + xi_ww(r)) - 1     (Eq. *)

Two limiting cases:

1. **Uncorrelated weights** (``xi_ww = 0``): ``xi_w = w_bar^2(1+xi_LS) - 1``.
   For ``w_bar = 1`` this collapses to ``xi_w = xi_LS``; for ``w_bar > 1``
   the recovered xi has a *constant* offset ``w_bar^2 - 1`` plus an
   amplitude scaling ``w_bar^2``.

2. **Weights track the field** (``xi_ww = xi_LS``): ``xi_w = (1+xi_LS)
   (w_bar^2 + xi_LS) - 1``. For strong clustering ``xi_LS >> 1`` the
   leading term is ``xi_LS^2`` -- the recovered xi grows quadratically
   with the input.

What we measure
---------------
For each ``N``:
    * ``xi_LS(r)`` from Corrfunc.
    * Two flavors of weights:
        a. ``mean_centered``  -- the doc's intended construction,
                                  ``<w>`` near 1.
        b. ``raw``            -- no mean subtraction, ``<w> > 1`` reflects
                                  the fact that data points sit in
                                  over-dense regions.
    * ``xi_w(r)`` for both, and the implied predicted curve from Eq. (*).

Run with::
    PYTHONPATH=. python demos/scan_N_dependence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twopt_density.ls_corrfunc import xi_landy_szalay, local_mean_density
from twopt_density.weights_binned import (
    compute_binned_weights, kde_overdensity, default_kernel_radius,
)
from twopt_density.validate import weighted_xi


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "plots"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_clustered(box, n_centers, n_per, sigma, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([
        rng.normal(c, sigma, size=(n_per, 3)) for c in centers
    ])
    return np.mod(pts, box).astype(np.float64)


def run_one(N, box, sigma, seed):
    """Generate a clustered catalog of N points and return diagnostics."""
    n_centers = 20
    n_per = max(1, N // n_centers)
    pts = make_clustered(box, n_centers, n_per, sigma, seed=seed)
    actual_N = len(pts)

    r_edges = np.logspace(np.log10(1.0), np.log10(0.49 * box), 22)
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=4
    )
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    R = default_kernel_radius(nbar)

    out = {"N": actual_N, "r_c": r_c, "xi_LS": xi, "r_kernel": R}
    for label, sub in (("mean_centered", True), ("raw", False)):
        w = compute_binned_weights(
            pts, r_c, xi, nbar,
            box_size=box, mode="mean", subtract_mean=sub,
        )
        _, xi_w = weighted_xi(pts, w, r_edges, box_size=box)
        out[label] = {
            "w_mean": float(w.mean()),
            "w_std": float(w.std()),
            "xi_w": xi_w,
        }
    return out


def main():
    box = 200.0
    sigma = 8.0  # blob size in Mpc
    Ns = [600, 1200, 2500, 5000]

    results = [run_one(N, box, sigma, seed=42) for N in Ns]

    # ---- Diagnostic plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(Ns) - 1)) for i in range(len(Ns))]

    # (a) xi_LS for each N (the input curve)
    ax = axes[0, 0]
    for r, color in zip(results, colors):
        ax.plot(r["r_c"], np.maximum(r["xi_LS"], 1e-3), "o-",
                color=color, label=f"N = {r['N']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel(r"$\hat\xi_{LS}(r)$")
    ax.set_title("(a) Input LS estimator")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) xi_w (raw, no mean subtraction) and predicted curve
    ax = axes[0, 1]
    for r, color in zip(results, colors):
        w_bar = r["raw"]["w_mean"]
        xi_pred = (1 + r["xi_LS"]) * w_bar**2 - 1.0
        ax.plot(r["r_c"], np.maximum(r["raw"]["xi_w"], 1e-3), "o-",
                color=color, label=f"N={r['N']}, "
                rf"$\langle w\rangle$={w_bar:.2f}")
        ax.plot(r["r_c"], np.maximum(xi_pred, 1e-3), "--",
                color=color, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel(r"$\hat\xi_w(r)$ (raw, no mean sub)")
    ax.set_title(r"(b) raw weights:  solid = measured, "
                 r"dashed = $\langle w\rangle^2(1+\xi_{LS}) - 1$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (c) ratio (1+xi_w)/(1+xi_LS) vs <w>^2 -- the calibration test
    ax = axes[1, 0]
    for r, color in zip(results, colors):
        w_bar = r["raw"]["w_mean"]
        ratio = (1.0 + r["raw"]["xi_w"]) / (1.0 + r["xi_LS"])
        ax.plot(r["r_c"], ratio, "o-", color=color,
                label=f"N={r['N']}")
        ax.axhline(w_bar**2, color=color, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel(r"$(1+\hat\xi_w)/(1+\hat\xi_{LS})$")
    ax.set_title(r"(c) calibration: should equal $\langle w\rangle^2$  "
                 r"(dashed)  if $\xi_{ww} \approx \xi_{LS}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) <w>^2 vs N, plus the offset that needs to be calibrated out
    ax = axes[1, 1]
    Ns_arr = np.array([r["N"] for r in results])
    wbar_raw = np.array([r["raw"]["w_mean"] for r in results])
    wbar_cen = np.array([r["mean_centered"]["w_mean"] for r in results])
    # The "asymptotic" offset is the geometric mean of the calibration
    # ratio over the strongly-clustered bins.
    offset = []
    for r in results:
        mask = r["xi_LS"] > 1.0
        ratio = (1 + r["raw"]["xi_w"][mask]) / (1 + r["xi_LS"][mask])
        offset.append(np.exp(np.log(ratio).mean()))
    offset = np.array(offset)

    ax.plot(Ns_arr, wbar_raw**2, "o-", color="#d62728",
            label=r"$\langle w\rangle^2$ (raw)")
    ax.plot(Ns_arr, offset, "s--", color="#1f77b4",
            label=r"measured $(1+\xi_w)/(1+\xi_{LS})$ "
                  r"(geom. mean over $\xi_{LS}>1$)")
    ax.plot(Ns_arr, wbar_cen**2, "x:", color="#2ca02c",
            label=r"$\langle w\rangle^2$ mean-centered (~1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"calibration / $\langle w\rangle^2$")
    ax.set_title(r"(d) $\langle w\rangle^2$ vs measured offset across N")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "07_N_scan_calibration.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Summary table
    print("\nN    <w>_raw   <w>^2     measured offset   xi_LS_max  xi_w_max(raw)")
    print("-" * 72)
    for r, off in zip(results, offset):
        print(f"{r['N']:5d}  {r['raw']['w_mean']:6.3f}    "
              f"{r['raw']['w_mean']**2:6.2f}    "
              f"{off:6.2f}             "
              f"{r['xi_LS'].max():6.2f}   {r['raw']['xi_w'].max():6.2f}")


if __name__ == "__main__":
    main()
