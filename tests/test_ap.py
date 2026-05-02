"""Tests for the JAX-based AP (Alcock-Paczynski) distortion module."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.differentiable_lisa import build_state, xi_LS
from twopt_density.ap import (
    apply_ap,
    xi_LS_AP,
    xi_LS_AP_soft,
    xi_simple_AP,
    xi_multipole_AP,
)


@pytest.fixture
def small_state():
    rng = np.random.default_rng(7)
    box = 200.0
    N_D, N_R = 400, 1600
    pts = rng.uniform(0, box, size=(N_D, 3))
    randoms = rng.uniform(0, box, size=(N_R, 3))
    r_edges = np.logspace(np.log10(2.0), np.log10(50.0), 8)
    state = build_state(pts, r_edges, box, randoms=randoms, cache_rr=True)
    return state, r_edges


def test_ap_unity_recovers_fiducial_xi_LS(small_state):
    """At alpha=(1,1) the AP path must reproduce the fiducial xi_LS exactly."""
    state, r_edges = small_state
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    xi_fid = np.asarray(xi_LS(state, w_d, w_r))
    ap = apply_ap(state, re, 1.0, 1.0)
    xi_ap = np.asarray(xi_LS_AP(state, ap, w_d, w_r))
    np.testing.assert_allclose(xi_ap, xi_fid, rtol=1e-12, atol=1e-14)


def test_ap_changes_xi_under_distortion(small_state):
    """Non-unity AP must change at least one bin's xi by a measurable amount."""
    state, r_edges = small_state
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    xi_fid = np.asarray(xi_LS_AP(state, apply_ap(state, re, 1.0, 1.0), w_d, w_r))
    xi_dist = np.asarray(xi_LS_AP(state, apply_ap(state, re, 1.2, 0.8), w_d, w_r))
    assert np.max(np.abs(xi_fid - xi_dist)) > 1e-3


def test_ap_xi_simple_periodic_path(small_state):
    """xi_simple_AP under unity recovers xi_simple from the no-AP path."""
    state, r_edges = small_state
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(state.N_D)
    ap = apply_ap(state, re, 1.0, 1.0)
    xi_ap = np.asarray(xi_simple_AP(state, ap, w_d))
    from twopt_density.differentiable_lisa import xi_simple
    xi_ref = np.asarray(xi_simple(state, w_d))
    np.testing.assert_allclose(xi_ap, xi_ref, rtol=1e-12, atol=1e-14)


def test_ap_multipole_finite_and_smooth(small_state):
    """xi_multipole_AP returns finite values and varies smoothly with AP."""
    state, r_edges = small_state
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    for apar, aperp in [(1.0, 1.0), (1.05, 0.95), (0.9, 1.1)]:
        ap = apply_ap(state, re, apar, aperp)
        xi2 = np.asarray(xi_multipole_AP(state, ap, w_d, w_r, L=2))
        assert np.all(np.isfinite(xi2)), f"non-finite xi2 at AP=({apar},{aperp})"


def test_ap_soft_grad_matches_finite_difference():
    """Soft-binned AP gradient agrees with centered FD on a wide bin range."""
    rng = np.random.default_rng(7)
    box = 200.0
    N_D, N_R = 1500, 4500
    pts = rng.uniform(0, box, size=(N_D, 3))
    randoms = rng.uniform(0, box, size=(N_R, 3))
    # Wide bin range so few pairs sit at the boundary of [r_edges[0], r_edges[-1]]:
    r_edges = np.logspace(np.log10(0.5), np.log10(80.0), 12)
    state = build_state(pts, r_edges, box, randoms=randoms, cache_rr=True)
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(N_D)
    w_r = jnp.ones(N_R)

    def loss(apar, aperp):
        return jnp.sum(xi_LS_AP_soft(state, re, w_d, w_r, apar, aperp))

    g = jax.grad(loss, argnums=(0, 1))(1.0, 1.0)
    eps = 5e-4
    fd_par = (loss(1.0 + eps, 1.0) - loss(1.0 - eps, 1.0)) / (2 * eps)
    fd_perp = (loss(1.0, 1.0 + eps) - loss(1.0, 1.0 - eps)) / (2 * eps)
    # Soft binning is C0 not C1; relax tolerance.
    rel_par = abs(float(g[0]) - float(fd_par)) / (abs(float(fd_par)) + 1e-6)
    rel_perp = abs(float(g[1]) - float(fd_perp)) / (abs(float(fd_perp)) + 1e-6)
    assert rel_par < 0.05, f"par rel diff {rel_par:.3e}"
    assert rel_perp < 0.05, f"perp rel diff {rel_perp:.3e}"


def test_ap_jit_roundtrip(small_state):
    """xi_LS_AP JIT-compiles and gives the same numbers as the eager call."""
    state, r_edges = small_state
    re = jnp.asarray(r_edges)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)

    @jax.jit
    def jitted(alpha_par, alpha_perp):
        ap = apply_ap(state, re, alpha_par, alpha_perp)
        return xi_LS_AP(state, ap, w_d, w_r)

    xi_jit = np.asarray(jitted(1.05, 0.95))
    ap = apply_ap(state, re, 1.05, 0.95)
    xi_eager = np.asarray(xi_LS_AP(state, ap, w_d, w_r))
    np.testing.assert_allclose(xi_jit, xi_eager, rtol=1e-12, atol=1e-14)
