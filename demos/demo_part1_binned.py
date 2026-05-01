"""End-to-end Part I demo: binned LS-consistent per-point weights.

Generates a small Poisson-thinned GP catalog, runs Corrfunc-LS, solves for
binned Wiener-filter weights, and validates the weighted-DD pair sum recovers
the original ``xi_LS(r)``.

Usage
-----
    python demos/demo_part1_binned.py
"""

from __future__ import annotations

import numpy as np

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
    r_edges = np.logspace(np.log10(1.0), np.log10(box / 2), 25)

    r_c, xi_j, RR_j, DD_j, DR_j = xi_landy_szalay(
        positions, r_edges=r_edges, box_size=box,
    )
    nbar = local_mean_density(positions, randoms=None, box_size=box)
    weights = compute_binned_weights(positions, r_c, xi_j, nbar)

    r_c2, xi_w = weighted_xi(positions, weights, r_edges, box_size=box)
    assert_recovery(xi_j, xi_w, rtol=0.2)
    print("Part I OK: |xi_w - xi_LS|/xi < 0.2 over",
          len(r_c), "bins")


if __name__ == "__main__":
    main()
