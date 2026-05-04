"""Storey-Fisher & Hogg basis-projected, AP-differentiable LS estimator.

Replaces histogram bins with a smooth basis ``{phi_k(r)}`` (cubic B-splines
on log-r by default). Each pair contributes ``phi_k(r_pair)`` to coefficient
``c_k`` instead of dropping into one bin::

    c_DD_k = 2 sum_{DD pairs} phi_k(d)        (factor 2: unordered)
    c_DR_k =   sum_{DR pairs} phi_k(d)
    c_RR_k = 2 sum_{RR pairs} phi_k(d)

then evaluate

    DD(s) = sum_k c_DD_k phi_k(s)             # similarly DR(s), RR(s)
    xi(s) = (DD Nr^2 - 2 DR Nr Nd + RR Nd^2) / (RR Nd^2)

``xi(s)`` is now a smooth function of s and (under AP via ``_ap_rescale``
on the per-pair distances) a smooth function of ``(alpha_par, alpha_perp)``
-- so ``jax.grad`` and ``jax.jacobian`` flow without bin-edge cusps.

JAX compatibility: a scipy cubic B-spline is precomputed on a fine log-r
grid and looked up via ``jnp.interp``. ``jax.lax.scan`` over basis index
keeps transient memory at ``O(n_pairs)``; total compute stays
``O(n_basis * n_pairs)``.

Required: ``state`` was built with ``cache_rr=True`` (RR per-pair distances
needed for AP-rebinning of RR(s)).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .basis import CubicSplineBasis
from .differentiable_lisa import FrozenPairGraph
from .ap import _ap_rescale


@dataclass
class JAXBasis:
    """Tabulated basis values on a log-r grid, ready for JAX evaluation."""

    log_r_grid: jnp.ndarray         # (n_grid,)
    F_grid: jnp.ndarray             # (n_basis, n_grid)
    r_min: float
    r_max: float
    n_basis: int

    @classmethod
    def from_cubic_spline(
        cls, n_basis: int = 12, r_min: float = 1.0, r_max: float = 100.0,
        n_grid: int = 4000,
    ) -> "JAXBasis":
        basis = CubicSplineBasis(n_basis=n_basis, r_min=r_min, r_max=r_max)
        r_grid = np.logspace(np.log10(r_min), np.log10(r_max), n_grid)
        F = basis.evaluate(r_grid)  # (n_basis, n_grid) numpy
        return cls(
            log_r_grid=jnp.asarray(np.log(r_grid)),
            F_grid=jnp.asarray(F),
            r_min=r_min, r_max=r_max, n_basis=n_basis,
        )


def _interp_indices_fracs(log_d, log_r_grid):
    """Linear-interp index and fraction for ``log_d`` against ``log_r_grid``."""
    n = log_r_grid.shape[0]
    idx = jnp.clip(
        jnp.searchsorted(log_r_grid, log_d, side="right") - 1,
        0, n - 2,
    )
    frac = (log_d - log_r_grid[idx]) / (log_r_grid[idx + 1] - log_r_grid[idx])
    frac = jnp.clip(frac, 0.0, 1.0)
    return idx, frac


def _basis_coeffs(
    d: jnp.ndarray,
    weights: jnp.ndarray,
    jb: JAXBasis,
) -> jnp.ndarray:
    """Sum_pairs weights * phi_k(d) for each k, returned as (n_basis,)."""
    log_d = jnp.log(jnp.clip(d, jb.r_min * 1.0001, jb.r_max * 0.9999))
    idx, frac = _interp_indices_fracs(log_d, jb.log_r_grid)
    in_support = ((d >= jb.r_min) & (d < jb.r_max)).astype(d.dtype)
    w = weights * in_support

    def step(carry, F_k):
        F_k_pair = (1.0 - frac) * F_k[idx] + frac * F_k[idx + 1]
        c_k = jnp.sum(w * F_k_pair)
        return None, c_k

    _, c = jax.lax.scan(step, None, jb.F_grid)
    return c


def _eval_basis_at(query_r: jnp.ndarray, jb: JAXBasis) -> jnp.ndarray:
    """Evaluate basis at query_r. Returns (n_basis, n_query)."""
    log_q = jnp.log(jnp.clip(query_r, jb.r_min * 1.0001, jb.r_max * 0.9999))
    idx, frac = _interp_indices_fracs(log_q, jb.log_r_grid)
    F_left = jb.F_grid[:, idx]
    F_right = jb.F_grid[:, idx + 1]
    return (1.0 - frac) * F_left + frac * F_right


def xi_LS_basis_AP(
    state: FrozenPairGraph,
    jb: JAXBasis,
    w_data: jnp.ndarray,
    w_rand: jnp.ndarray,
    alpha_par: float,
    alpha_perp: float,
    query_r: jnp.ndarray,
) -> jnp.ndarray:
    """SF&H basis-projected LS xi(s) under AP, evaluated at ``query_r``."""
    if state.RR_d is None:
        raise ValueError(
            "xi_LS_basis_AP needs state.RR_d; rebuild with cache_rr=True"
        )
    dd_d, _ = _ap_rescale(state.DD_d, state.DD_mu, alpha_par, alpha_perp)
    dr_d, _ = _ap_rescale(state.DR_d, state.DR_mu, alpha_par, alpha_perp)
    rr_d, _ = _ap_rescale(state.RR_d, state.RR_mu, alpha_par, alpha_perp)

    w_dd = w_data[state.DD_pi] * w_data[state.DD_pk]
    w_dr = w_data[state.DR_pi] * w_rand[state.DR_pk]
    w_rr = jnp.ones_like(rr_d)

    c_DD = 2.0 * _basis_coeffs(dd_d, w_dd, jb)
    c_DR = _basis_coeffs(dr_d, w_dr, jb)
    c_RR = 2.0 * _basis_coeffs(rr_d, w_rr, jb)

    F_q = _eval_basis_at(query_r, jb)            # (n_basis, n_query)
    DD_s = F_q.T @ c_DD                          # (n_query,)
    DR_s = F_q.T @ c_DR
    RR_s = F_q.T @ c_RR

    Nd = jnp.sum(w_data)
    Nr = jnp.sum(w_rand)
    num = DD_s * Nr ** 2 - 2.0 * DR_s * Nr * Nd + RR_s * Nd ** 2
    den = RR_s * Nd ** 2 + 1e-30
    return num / den


def xi_LS_basis(
    state: FrozenPairGraph,
    jb: JAXBasis,
    w_data: jnp.ndarray,
    w_rand: jnp.ndarray,
    query_r: jnp.ndarray,
) -> jnp.ndarray:
    """No-AP convenience: alpha=(1,1)."""
    return xi_LS_basis_AP(state, jb, w_data, w_rand, 1.0, 1.0, query_r)
