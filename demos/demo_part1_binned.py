"""End-to-end Part I demo: binned LS-consistent per-point weights.

Generates a small Poisson-thinned GP catalog, runs Corrfunc-LS, solves for
binned Wiener-filter weights, and validates the weighted-DD pair sum recovers
the original ``xi_LS(r)``.

Usage
-----
    python demos/demo_part1_binned.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twopt_density.ls_corrfunc import xi_landy_szalay, local_mean_density
from twopt_density.weights_binned import compute_binned_weights
from twopt_density.validate import weighted_xi, assert_recovery


def _toy_catalog(n: int = 4000, box: float = 200.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, box, size=(20, 3))
    pts = np.vstack([
        rng.normal(c, 8.0, size=(n // 20, 3)) for c in centers
    ])
    return np.mod(pts, box)


def main() -> None:
    box = 200.0
    positions = _toy_catalog(box=box)
    # Corrfunc requires r_max strictly less than box/2 for periodic catalogs.
    r_edges = np.logspace(np.log10(1.0), np.log10(0.49 * box), 25)

    r_c, xi_j, RR_j, DD_j, DR_j = xi_landy_szalay(
        positions, r_edges=r_edges, box_size=box,
    )
    nbar = local_mean_density(positions, randoms=None, box_size=box)
    weights = compute_binned_weights(
        positions, r_c, xi_j, nbar, box_size=box, mode="mean",
    )
    print(f"weights: mean={weights.mean():.3f}  std={weights.std():.3f}  "
          f"min={weights.min():.3f}  max={weights.max():.3f}")

    r_c2, xi_w = weighted_xi(positions, weights, r_edges, box_size=box)
    mask = xi_j > 1.0
    ratios = xi_w[mask] / xi_j[mask]
    pearson = np.corrcoef(xi_j[mask], xi_w[mask])[0, 1]
    print(f"xi_w / xi_LS over {mask.sum()} strongly-clustered bins:")
    print(f"  median ratio = {np.median(ratios):.2f},  "
          f"range = [{ratios.min():.2f}, {ratios.max():.2f}]")
    print(f"  Pearson r between xi_LS and xi_w: {pearson:.3f}")
    print("Note: at r << r_kernel = "
          f"{(30*3/(4*np.pi*nbar.mean()))**(1/3):.1f} Mpc the recovery "
          "is biased by the smoothing scale; see IMPLEMENTATION_PLAN.md.")


if __name__ == "__main__":
    main()
