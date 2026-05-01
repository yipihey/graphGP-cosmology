"""Part III: GP / Vecchia weights via graphgp.

Wraps Sec. 4.4 of ``twopt_density.pdf``:

    1. Tabulate the SFH kernel xi_hat(r) on a log r-grid.
    2. Build a Vecchia neighbor graph.
    3. Apply the inverse Cholesky factor to the centered overdensity.
"""

from __future__ import annotations

import numpy as np

from .basis import Basis
from .basis_projection import xi_from_basis


def tabulate_kernel(
    theta_hat: np.ndarray,
    basis: Basis,
    n_grid: int = 1000,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    r_grid = np.logspace(np.log10(basis.r_min), np.log10(basis.r_max), n_grid)
    xi_grid = xi_from_basis(theta_hat, basis, r_grid)
    xi_grid[0] += jitter
    return r_grid, xi_grid


def compute_2pt_weights(
    positions: np.ndarray,
    theta_hat: np.ndarray,
    basis: Basis,
    nbar: np.ndarray,
    n0: int = 100,
    k: int = 30,
):
    """Run the full Part III pipeline and return ``w_i = 1 + delta_hat_i``.

    Stub: imports graphgp lazily so the rest of the package is usable
    without it. Replace the inner block with the actual graphgp calls
    once the API is fixed for this repo (see graphGP_cosmo.py for the
    existing wiring).
    """
    import graphgp as gp  # noqa: F401  (imported lazily, see docstring)

    r_grid, xi_grid = tabulate_kernel(theta_hat, basis)
    covariance = (r_grid, xi_grid)

    graph = gp.build_graph(positions, n0=n0, k=k)
    delta_data = (1.0 - nbar / nbar.mean())
    delta_hat = gp.apply_inverse_cholesky(graph, covariance, delta_data)
    return 1.0 + np.asarray(delta_hat)
