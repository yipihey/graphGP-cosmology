"""Alcock-Paczynski (AP) distortion as a JAX-differentiable rescaling.

Once the frozen pair graph is built at a fiducial cosmology, AP is just
two scalar parameters that linearly rescale the LOS and transverse
components of every pair separation::

    d_par'  = alpha_par  * (mu * d)                  # LOS component
    d_perp' = alpha_perp * (d * sqrt(1 - mu^2))      # transverse component
    d'      = sqrt(d_par'^2 + d_perp'^2)
    mu'     = d_par' / d'

Pairs get rebinned under the new metric with no rebuild of the KDTree.
``jax.grad`` of any downstream observable (xi, per-particle weights) flows
through the rescaling and through the JAX ``searchsorted`` re-bin. Bin-
edge crossings produce step changes under finite AP and zero-measure
(missing) gradient contributions under infinitesimal AP — the standard
behavior of histograms.

Mapping cosmology -> (alpha_par, alpha_perp)::

    alpha_par  = H_fid(z) / H_true(z)
    alpha_perp = D_A_true(z) / D_A_fid(z)

so a sweep over (alpha_par, alpha_perp) is equivalent to a sweep over
(H, D_A) -- or any cosmological parameter that maps to those.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from .differentiable_lisa import FrozenPairGraph, _bin_sum, _legendre_jax


class APState(NamedTuple):
    """Distortion-rebinned view of a FrozenPairGraph for a given AP."""
    DD_d: jnp.ndarray
    DD_mu: jnp.ndarray
    DD_bin: jnp.ndarray
    DD_valid: jnp.ndarray
    DR_d: Optional[jnp.ndarray]
    DR_mu: Optional[jnp.ndarray]
    DR_bin: Optional[jnp.ndarray]
    DR_valid: Optional[jnp.ndarray]
    RR_per_bin: jnp.ndarray
    n_bins: int
    N_D: int
    N_R: int


def _ap_rescale(d, mu, alpha_par, alpha_perp):
    d_par = mu * d
    d_perp_sq = jnp.maximum(d * d - d_par * d_par, 0.0)
    d_perp = jnp.sqrt(d_perp_sq)
    d_par_new = alpha_par * d_par
    d_perp_new = alpha_perp * d_perp
    d_new = jnp.sqrt(d_par_new ** 2 + d_perp_new ** 2)
    mu_new = d_par_new / (d_new + 1e-30)
    return d_new, mu_new


def _rebin(d_new, r_edges, n_bins):
    bin_idx = jnp.searchsorted(r_edges, d_new, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    return jnp.where(valid, bin_idx, 0), valid.astype(d_new.dtype)


def _soft_bin(d_new, r_edges, n_bins):
    """Linear interpolation between adjacent log-bin centers.

    Each pair contributes ``(1-frac)`` to bin ``k`` and ``frac`` to bin
    ``k+1`` where ``k`` is the left-center index. Makes the binned sum a
    smooth function of ``d_new`` so ``jax.grad`` flows through edge
    crossings (the discrete histogram limit recovers as the bin widths
    shrink). Pairs outside ``[r_edges[0], r_edges[-1]]`` get zero weight.
    """
    log_d = jnp.log(d_new + 1e-30)
    log_centers = 0.5 * (jnp.log(r_edges[:-1]) + jnp.log(r_edges[1:]))
    k = jnp.clip(
        jnp.searchsorted(log_centers, log_d, side="right") - 1,
        0, n_bins - 2,
    )
    dlc = log_centers[k + 1] - log_centers[k]
    frac = jnp.clip((log_d - log_centers[k]) / dlc, 0.0, 1.0)
    valid = ((d_new >= r_edges[0]) & (d_new < r_edges[-1])).astype(d_new.dtype)
    return k, frac, valid


def _soft_bin_sum(values, k, frac, valid, n_bins):
    v = values * valid
    s_left = jax.ops.segment_sum(v * (1.0 - frac), k, num_segments=n_bins)
    s_right = jax.ops.segment_sum(v * frac, k + 1, num_segments=n_bins + 1)
    return s_left + s_right[:n_bins]


def apply_ap(
    state: FrozenPairGraph,
    r_edges: jnp.ndarray,
    alpha_par: float,
    alpha_perp: float,
) -> APState:
    """Re-bin the frozen graph under AP. Returns an APState.

    If ``state.RR_d`` and ``state.RR_mu`` were cached at build time
    (``cache_rr=True`` in ``build_state``), RR is re-binned exactly under
    AP. Otherwise it falls back to a Jacobian rescale
    ``RR_per_bin / (alpha_par * alpha_perp**2)`` -- correct only at first
    order in (alpha - 1) for Poisson-uniform randoms in a periodic box.
    """
    DD_d, DD_mu = _ap_rescale(state.DD_d, state.DD_mu, alpha_par, alpha_perp)
    DD_bin, DD_valid = _rebin(DD_d, r_edges, state.n_bins)
    if state.DR_d is not None:
        DR_d, DR_mu = _ap_rescale(state.DR_d, state.DR_mu, alpha_par, alpha_perp)
        DR_bin, DR_valid = _rebin(DR_d, r_edges, state.n_bins)
    else:
        DR_d = DR_mu = DR_bin = DR_valid = None

    if state.RR_d is not None and state.RR_mu is not None:
        RR_d_new, _mu_new = _ap_rescale(
            state.RR_d, state.RR_mu, alpha_par, alpha_perp,
        )
        RR_bin, RR_valid = _rebin(RR_d_new, r_edges, state.n_bins)
        RR_per_bin = 2.0 * _bin_sum(RR_valid, RR_bin, state.n_bins)
    else:
        sigma = alpha_par * alpha_perp ** 2
        RR_per_bin = state.RR_per_bin / sigma

    return APState(
        DD_d=DD_d, DD_mu=DD_mu, DD_bin=DD_bin, DD_valid=DD_valid,
        DR_d=DR_d, DR_mu=DR_mu, DR_bin=DR_bin, DR_valid=DR_valid,
        RR_per_bin=RR_per_bin,
        n_bins=state.n_bins, N_D=state.N_D, N_R=state.N_R,
    )


def DD_AP(state: FrozenPairGraph, ap: APState, w_data: jnp.ndarray) -> jnp.ndarray:
    weights = w_data[state.DD_pi] * w_data[state.DD_pk] * ap.DD_valid
    return 2.0 * _bin_sum(weights, ap.DD_bin, ap.n_bins)


def DR_AP(state: FrozenPairGraph, ap: APState, w_data: jnp.ndarray, w_rand: jnp.ndarray) -> jnp.ndarray:
    weights = w_data[state.DR_pi] * w_rand[state.DR_pk] * ap.DR_valid
    return _bin_sum(weights, ap.DR_bin, ap.n_bins)


def xi_LS_AP(
    state: FrozenPairGraph,
    ap: APState,
    w_data: jnp.ndarray,
    w_rand: jnp.ndarray,
) -> jnp.ndarray:
    """Landy-Szalay xi(s) under AP-rebinned pair counts."""
    dd = DD_AP(state, ap, w_data)
    dr = DR_AP(state, ap, w_data, w_rand)
    rr = ap.RR_per_bin
    Nd = jnp.sum(w_data)
    Nr = jnp.sum(w_rand)
    num = dd * Nr ** 2 - 2.0 * dr * Nr * Nd + rr * Nd ** 2
    den = rr * Nd ** 2 + 1e-30
    return num / den


def xi_simple_AP(
    state: FrozenPairGraph,
    ap: APState,
    w_data: jnp.ndarray,
) -> jnp.ndarray:
    """Periodic-uniform xi(s) under AP."""
    return DD_AP(state, ap, w_data) / (ap.RR_per_bin + 1e-30) - 1.0


def xi_multipole_AP(
    state: FrozenPairGraph,
    ap: APState,
    w_data: jnp.ndarray,
    w_rand: Optional[jnp.ndarray] = None,
    L: int = 2,
) -> jnp.ndarray:
    """Pair-count multipole xi_L(s) under AP.

    For uniform randoms RR_L(s) = 0 for L > 0, so the LS formula reduces
    to ``(DD_L Nr^2 - 2 DR_L Nr Nd) / (RR_0 Nd^2)`` with the (2L+1) prefactor.
    L=0 falls back to the standard LS expression.
    """
    if L == 0:
        if w_rand is None:
            return xi_simple_AP(state, ap, w_data)
        return xi_LS_AP(state, ap, w_data, w_rand)
    P_dd = _legendre_jax(L, ap.DD_mu)
    weights_dd = (
        w_data[state.DD_pi] * w_data[state.DD_pk] * ap.DD_valid * P_dd
    )
    DD_L = 2.0 * (2 * L + 1) * _bin_sum(weights_dd, ap.DD_bin, ap.n_bins)
    rr = ap.RR_per_bin
    Nd = jnp.sum(w_data)
    if w_rand is None:
        return DD_L / (rr + 1e-30)
    P_dr = _legendre_jax(L, ap.DR_mu)
    weights_dr = (
        w_data[state.DR_pi] * w_rand[state.DR_pk] * ap.DR_valid * P_dr
    )
    DR_L = (2 * L + 1) * _bin_sum(weights_dr, ap.DR_bin, ap.n_bins)
    Nr = jnp.sum(w_rand)
    num = DD_L * Nr ** 2 - 2.0 * DR_L * Nr * Nd
    return num / (rr * Nd ** 2 + 1e-30)


def xi_LS_AP_soft(
    state: FrozenPairGraph,
    r_edges: jnp.ndarray,
    w_data: jnp.ndarray,
    w_rand: jnp.ndarray,
    alpha_par: float,
    alpha_perp: float,
) -> jnp.ndarray:
    """Soft-binned LS xi(s) under AP.

    Uses linear interpolation between log-bin centers so ``jax.grad``
    captures edge-crossing contributions smoothly. Forward values are
    slightly smoothed compared with the hard-binned ``xi_LS_AP``; use
    this variant when you want gradients of the monopole w.r.t.
    ``(alpha_par, alpha_perp)``.
    """
    n_bins = state.n_bins
    DD_d, _ = _ap_rescale(state.DD_d, state.DD_mu, alpha_par, alpha_perp)
    k_dd, f_dd, v_dd = _soft_bin(DD_d, r_edges, n_bins)
    w_dd = w_data[state.DD_pi] * w_data[state.DD_pk]
    DD_b = 2.0 * _soft_bin_sum(w_dd, k_dd, f_dd, v_dd, n_bins)

    DR_d, _ = _ap_rescale(state.DR_d, state.DR_mu, alpha_par, alpha_perp)
    k_dr, f_dr, v_dr = _soft_bin(DR_d, r_edges, n_bins)
    w_dr = w_data[state.DR_pi] * w_rand[state.DR_pk]
    DR_b = _soft_bin_sum(w_dr, k_dr, f_dr, v_dr, n_bins)

    if state.RR_d is not None:
        RR_d_new, _ = _ap_rescale(state.RR_d, state.RR_mu, alpha_par, alpha_perp)
        k_rr, f_rr, v_rr = _soft_bin(RR_d_new, r_edges, n_bins)
        ones = jnp.ones_like(RR_d_new)
        RR_b = 2.0 * _soft_bin_sum(ones, k_rr, f_rr, v_rr, n_bins)
    else:
        sigma = alpha_par * alpha_perp ** 2
        RR_b = state.RR_per_bin / sigma

    Nd = jnp.sum(w_data)
    Nr = jnp.sum(w_rand)
    num = DD_b * Nr ** 2 - 2.0 * DR_b * Nr * Nd + RR_b * Nd ** 2
    den = RR_b * Nd ** 2 + 1e-30
    return num / den


def per_particle_overdensity_AP(
    state: FrozenPairGraph,
    ap: APState,
    w_data: jnp.ndarray,
    w_rand: Optional[jnp.ndarray] = None,
    aggregation: str = "RR",
    L: int = 0,
) -> jnp.ndarray:
    """Per-particle delta_i under AP rebinning.

    Mirrors ``differentiable_lisa.per_particle_overdensity`` but uses the
    AP-distorted bin assignments and mu values so that gradients w.r.t.
    ``alpha_par, alpha_perp`` flow through to each particle's weight.
    """
    if w_rand is None:
        xi = xi_simple_AP(state, ap, w_data)
    else:
        xi = xi_LS_AP(state, ap, w_data, w_rand)

    if aggregation == "RR":
        a = jnp.maximum(ap.RR_per_bin, 0.0)
    elif aggregation == "RR_xi":
        a = jnp.where(xi > 0, ap.RR_per_bin * jnp.abs(xi), 0.0)
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    a_norm = a / (jnp.sum(a) + 1e-30)

    P_L = _legendre_jax(L, ap.DD_mu)

    if w_rand is None:
        E_per_bin = ap.RR_per_bin * (jnp.sum(w_data) - 1) / (state.N_D * (state.N_D - 1)) * 2
        c = a_norm / (E_per_bin + 1e-30)
        weights_pair = c[ap.DD_bin] * P_L * ap.DD_valid
        out = (
            jax.ops.segment_sum(weights_pair, state.DD_pi, num_segments=state.N_D)
            + jax.ops.segment_sum(weights_pair, state.DD_pk, num_segments=state.N_D)
        )
        return out - (1.0 if L == 0 else 0.0)

    c_DD = a_norm * (jnp.sum(w_rand) / jnp.sum(w_data))
    c_DR = a_norm
    P_L_DR = _legendre_jax(L, ap.DR_mu)
    inc_DD = (
        c_DD[ap.DD_bin] * P_L * ap.DD_valid
        * (w_data[state.DD_pi] * w_data[state.DD_pk])
    )
    num = (
        jax.ops.segment_sum(inc_DD, state.DD_pi, num_segments=state.N_D)
        + jax.ops.segment_sum(inc_DD, state.DD_pk, num_segments=state.N_D)
    )
    inc_DR = (
        c_DR[ap.DR_bin] * P_L_DR * ap.DR_valid
        * (w_data[state.DR_pi] * w_rand[state.DR_pk])
    )
    den = jax.ops.segment_sum(inc_DR, state.DR_pi, num_segments=state.N_D)
    return num / (den + 1e-30) - (1.0 if L == 0 else 0.0)
