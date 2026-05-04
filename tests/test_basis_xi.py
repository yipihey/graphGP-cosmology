"""Tests for the JAX-native SF&H basis-projected, AP-differentiable LS xi."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.differentiable_lisa import build_state
from twopt_density.basis_xi import JAXBasis, xi_LS_basis, xi_LS_basis_AP


@pytest.fixture
def small_setup():
    rng = np.random.default_rng(7)
    box = 200.0
    N_D, N_R = 1500, 4500
    pts = rng.uniform(0, box, size=(N_D, 3))
    randoms = rng.uniform(0, box, size=(N_R, 3))
    r_edges = np.logspace(np.log10(0.5), np.log10(80.0), 8)
    state = build_state(pts, r_edges, box, randoms=randoms, cache_rr=True)
    jb = JAXBasis.from_cubic_spline(
        n_basis=14, r_min=0.5, r_max=80.0, n_grid=2000,
    )
    return state, jb


def test_basis_xi_returns_finite_smooth_values(small_setup):
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.linspace(2.0, 50.0, 80)
    xi = np.asarray(xi_LS_basis(state, jb, w_d, w_r, s))
    assert np.all(np.isfinite(xi)), "non-finite basis xi"
    # Cubic-spline interpolation should give a C2-smooth curve. Probe
    # by comparing xi at a query and at a slightly-shifted query: the
    # difference should be O(ds), not jumpy.
    s2 = s + 0.01
    xi2 = np.asarray(xi_LS_basis(state, jb, w_d, w_r, s2))
    assert np.max(np.abs(xi - xi2)) < 0.05  # smooth, no step changes


def test_basis_xi_AP_grad_matches_finite_difference(small_setup):
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.asarray(np.logspace(np.log10(2.0), np.log10(40.0), 8))

    def loss(apar, aperp):
        return jnp.sum(xi_LS_basis_AP(state, jb, w_d, w_r, apar, aperp, s))

    g = jax.grad(loss, argnums=(0, 1))(1.0, 1.0)
    eps = 1e-3
    fd_par = (loss(1.0 + eps, 1.0) - loss(1.0 - eps, 1.0)) / (2 * eps)
    fd_perp = (loss(1.0, 1.0 + eps) - loss(1.0, 1.0 - eps)) / (2 * eps)
    rel_par = abs(float(g[0]) - float(fd_par)) / (abs(float(fd_par)) + 1e-12)
    rel_perp = abs(float(g[1]) - float(fd_perp)) / (abs(float(fd_perp)) + 1e-12)
    # SF&H ξ is C2-smooth in α via the basis -- expect FD agreement to ~1e-3.
    assert rel_par < 5e-3, f"par rel {rel_par:.3e}"
    assert rel_perp < 5e-3, f"perp rel {rel_perp:.3e}"


def test_basis_xi_changes_under_AP(small_setup):
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.asarray(np.logspace(np.log10(3.0), np.log10(30.0), 10))
    xi_fid = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, 1.0, 1.0, s))
    xi_dist = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, 1.2, 0.8, s))
    assert np.max(np.abs(xi_fid - xi_dist)) > 1e-3


def test_basis_xi_AP_jit_roundtrip(small_setup):
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.linspace(3.0, 30.0, 25)

    @jax.jit
    def jitted(apar, aperp):
        return xi_LS_basis_AP(state, jb, w_d, w_r, apar, aperp, s)

    xi_jit = np.asarray(jitted(1.05, 0.95))
    xi_eager = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, 1.05, 0.95, s))
    np.testing.assert_allclose(xi_jit, xi_eager, rtol=1e-12, atol=1e-14)


def test_basis_xi_AP_unity_consistent_with_no_AP(small_setup):
    """xi_LS_basis_AP at alpha=(1,1) equals xi_LS_basis."""
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.linspace(3.0, 40.0, 12)
    xi_a = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, 1.0, 1.0, s))
    xi_b = np.asarray(xi_LS_basis(state, jb, w_d, w_r, s))
    np.testing.assert_allclose(xi_a, xi_b, rtol=1e-12, atol=1e-14)


def test_basis_xi_grad_w_data_finite_difference(small_setup):
    """jax.grad w.r.t. data weights matches centered FD for the basis path."""
    state, jb = small_setup
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)
    s = jnp.asarray(np.logspace(np.log10(3.0), np.log10(30.0), 6))

    def loss(w):
        return jnp.sum(xi_LS_basis_AP(state, jb, w, w_r, 1.0, 1.0, s))

    g = jax.grad(loss)(w_d)
    g = np.asarray(g)
    rng = np.random.default_rng(0)
    idx = rng.choice(state.N_D, size=8, replace=False)
    eps = 1e-4
    g_fd = np.zeros(len(idx))
    for k, i in enumerate(idx):
        wp = w_d.at[i].add(eps)
        wm = w_d.at[i].add(-eps)
        g_fd[k] = (float(loss(wp)) - float(loss(wm))) / (2 * eps)
    rel = np.abs(g[idx] - g_fd) / (np.abs(g_fd) + 1e-12)
    assert np.max(rel) < 1e-4, f"max rel {np.max(rel):.3e}"
