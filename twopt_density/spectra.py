"""Hankel transform xi(r) and sigma^2(R) from a P(k) callable.

Direct trapezoid integration on a log-k grid::

    xi(r)      = (1 / (2 pi^2)) integral k^2 P(k) j_0(k r) dk
    sigma^2(R) = (1 / (2 pi^2)) integral k^2 P(k) W^2(k R) dk

with the top-hat window ``W(x) = 3 (sin x - x cos x) / x^3``. Both are
written as JAX functions so gradients flow through the cosmological
parameters that drive ``P_k`` (typically a partial of
``twopt_density.cosmology.run_halofit`` or ``pnl_new_emulated``).

A log-k grid with ~2000 points covering ``[1e-4, 100] h/Mpc`` is
sufficient for sub-percent accuracy on the relevant scales (r in
[1, 200] Mpc/h). The FFTLog approach would be faster for very dense
r-grids; this trapezoid implementation is JAX-clean and adequate for the
typical ~50-200 r-points used in inference.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def _j0(x):
    """Spherical Bessel j_0(x) = sin(x)/x, with the limit at x=0."""
    return jnp.where(jnp.abs(x) < 1e-8, 1.0 - x ** 2 / 6.0, jnp.sin(x) / x)


def _W_tophat(x):
    """Top-hat window W(x) = 3 (sin x - x cos x) / x^3, with the limit at 0."""
    safe = jnp.where(jnp.abs(x) < 1e-3, 1.0, x)
    W_smooth = 1.0 - x ** 2 / 10.0  # leading expansion at small x
    W_full = 3.0 * (jnp.sin(safe) - safe * jnp.cos(safe)) / safe ** 3
    return jnp.where(jnp.abs(x) < 1e-3, W_smooth, W_full)


def make_log_k_grid(k_min: float = 1e-4, k_max: float = 1e2, n: int = 2000):
    """Standard log-k integration grid."""
    return jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n)


def xi_from_Pk(
    r: jnp.ndarray,
    k: jnp.ndarray,
    P_k: jnp.ndarray,
) -> jnp.ndarray:
    """Hankel transform: ``xi(r) = (1/2pi^2) int k^2 P(k) j_0(kr) dk``.

    ``r``: (n_r,) radii. ``k``, ``P_k``: (n_k,) on the same log-k grid.
    Uses ``jnp.trapezoid`` over ``log k`` with the substitution
    ``dk = k d(log k)``, so the integrand becomes ``k^3 P(k) j_0(kr)``.
    """
    log_k = jnp.log(k)
    kr = jnp.outer(r, k)                     # (n_r, n_k)
    integrand = (k ** 3 * P_k)[None, :] * _j0(kr)
    return jnp.trapezoid(integrand, log_k, axis=1) / (2.0 * jnp.pi ** 2)


def sigma2_from_Pk(
    R: jnp.ndarray,
    k: jnp.ndarray,
    P_k: jnp.ndarray,
) -> jnp.ndarray:
    """Top-hat ``sigma^2(R) = (1/2pi^2) int k^2 P(k) W^2(kR) dk``."""
    log_k = jnp.log(k)
    kR = jnp.outer(R, k)
    W2 = _W_tophat(kR) ** 2
    integrand = (k ** 3 * P_k)[None, :] * W2
    return jnp.trapezoid(integrand, log_k, axis=1) / (2.0 * jnp.pi ** 2)


def xi_model(
    r: jnp.ndarray,
    P_k_fn: Callable[[jnp.ndarray], jnp.ndarray],
    k_grid: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Convenience: evaluate ``P_k_fn(k_grid)`` then transform to xi(r)."""
    if k_grid is None:
        k_grid = make_log_k_grid()
    P_k = P_k_fn(k_grid)
    return xi_from_Pk(r, k_grid, P_k)
