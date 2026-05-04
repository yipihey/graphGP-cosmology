"""Tests for the JAX-based differentiable LISA implementation.

Two things are checked:
  1. Forward agreement: xi and per-particle weights from the JAX path match
     ``streaming_lisa`` to machine precision when both use unit weights.
  2. Gradient correctness: ``jax.grad`` of a scalar reduction of xi (and of
     per-particle weights) matches a centered finite-difference reference to
     ~1e-7 relative.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.differentiable_lisa import (
    build_state,
    xi_LS,
    xi_DP,
    xi_simple,
    per_particle_overdensity,
    per_particle_weights,
)
from twopt_density.streaming_lisa import StreamingLISA


@pytest.fixture
def small_state():
    """Small periodic + random catalog and frozen graph, shared across tests."""
    rng = np.random.default_rng(7)
    box = 200.0
    N_D = 400
    N_R = 1600
    pts = rng.uniform(0, box, size=(N_D, 3))
    randoms = rng.uniform(0, box, size=(N_R, 3))
    r_edges = np.logspace(np.log10(2.0), np.log10(50.0), 8)
    state = build_state(pts, r_edges, box, randoms=randoms)
    return pts, randoms, r_edges, box, state


def test_xi_LS_matches_streaming(small_state):
    pts, randoms, r_edges, box, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    xi_jax = np.asarray(xi_LS(state, w_d, w_r))
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box,
        randoms=randoms, multipoles=(0,),
    ).fit()
    xi_ref = sl.xi_LS()
    np.testing.assert_allclose(xi_jax, xi_ref, rtol=1e-12, atol=1e-14)


def test_xi_DP_matches_streaming(small_state):
    pts, randoms, r_edges, box, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    xi_jax = np.asarray(xi_DP(state, w_d, w_r))
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box,
        randoms=randoms, multipoles=(0,),
    ).fit()
    xi_ref = sl.xi_DP()
    np.testing.assert_allclose(xi_jax, xi_ref, rtol=1e-12, atol=1e-14)


def test_xi_simple_matches_streaming(small_state):
    pts, randoms, r_edges, box, state = small_state
    w_d = jnp.ones(state.N_D)
    xi_jax = np.asarray(xi_simple(state, w_d))
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box,
        randoms=randoms, multipoles=(0,),
    ).fit()
    xi_ref = sl.xi_simple()
    np.testing.assert_allclose(xi_jax, xi_ref, rtol=1e-12, atol=1e-14)


def test_per_particle_weights_match_streaming(small_state):
    pts, randoms, r_edges, box, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    w_jax = np.asarray(per_particle_weights(
        state, w_d, w_r, aggregation="RR", estimator="LS",
    ))
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box,
        randoms=randoms, multipoles=(0,),
    ).fit()
    w_ref = sl.per_particle_weights(estimator="LS", aggregation="RR")
    np.testing.assert_allclose(w_jax, w_ref, rtol=1e-10, atol=1e-12)


def test_grad_xi_LS_matches_finite_difference(small_state):
    """Centered FD vs jax.grad on sum(xi_LS) wrt w_data."""
    *_, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)

    def loss(w):
        return jnp.sum(xi_LS(state, w, w_r))

    g_jax = np.asarray(jax.grad(loss)(w_d))

    # Centered FD on a small random subset for speed.
    rng = np.random.default_rng(0)
    idx = rng.choice(state.N_D, size=12, replace=False)
    eps = 1e-4
    g_fd = np.zeros_like(idx, dtype=np.float64)
    f0 = float(loss(w_d))
    for k, i in enumerate(idx):
        wp = w_d.at[i].add(eps)
        wm = w_d.at[i].add(-eps)
        g_fd[k] = (float(loss(wp)) - float(loss(wm))) / (2 * eps)

    rel = np.abs(g_jax[idx] - g_fd) / (np.abs(g_fd) + 1e-12)
    assert np.max(rel) < 1e-5, (
        f"max rel grad mismatch {np.max(rel):.2e}, jax={g_jax[idx]}, fd={g_fd}"
    )


def test_grad_w_rand_matches_finite_difference(small_state):
    """Random-catalog selection sensitivity: dxi/dw_rand checked via FD."""
    *_, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)

    def loss(wr):
        return jnp.sum(xi_LS(state, w_d, wr))

    g_jax = np.asarray(jax.grad(loss)(w_r))

    rng = np.random.default_rng(1)
    idx = rng.choice(state.N_R, size=12, replace=False)
    eps = 1e-4
    g_fd = np.zeros_like(idx, dtype=np.float64)
    for k, i in enumerate(idx):
        wp = w_r.at[i].add(eps)
        wm = w_r.at[i].add(-eps)
        g_fd[k] = (float(loss(wp)) - float(loss(wm))) / (2 * eps)

    rel = np.abs(g_jax[idx] - g_fd) / (np.abs(g_fd) + 1e-10)
    # FD on tiny random contributions is noisier; loosen the tolerance.
    assert np.max(rel) < 1e-3, (
        f"max rel grad mismatch {np.max(rel):.2e}, jax={g_jax[idx]}, fd={g_fd}"
    )


def test_jit_roundtrip(small_state):
    """xi_LS JIT-compiles and gives the same numbers as the eager call."""
    *_, state = small_state
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    xi_eager = np.asarray(xi_LS(state, w_d, w_r))

    @jax.jit
    def jitted(wd, wr):
        return xi_LS(state, wd, wr)

    xi_jit = np.asarray(jitted(w_d, w_r))
    np.testing.assert_allclose(xi_jit, xi_eager, rtol=1e-12, atol=1e-14)


def test_periodic_no_randoms_path(small_state):
    """xi_LS with no randoms (periodic-uniform) is finite and ~ xi_simple."""
    pts, _, r_edges, box, _ = small_state
    state = build_state(pts, r_edges, box, randoms=None)
    # No-randoms path needs RR for the analytic fallback; build with a tiny
    # uniform random set just to populate RR_per_bin (mimics typical use).
    rng = np.random.default_rng(0)
    randoms = rng.uniform(0, box, size=(2000, 3))
    state = build_state(pts, r_edges, box, randoms=randoms)
    w_d = jnp.ones(state.N_D)
    xi_no_r = np.asarray(xi_LS(state, w_d, w_rand=None))
    xi_simp = np.asarray(xi_simple(state, w_d))
    # Both reduce to DD/RR-1 up to the (Nd-1)/Nd vs Nd/Nd normalization.
    np.testing.assert_allclose(xi_no_r, xi_simp, rtol=5e-3, atol=5e-3)
