"""Tests for the JAX Hankel transform xi(r) and sigma^2(R)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density import cosmology as cj
from twopt_density.spectra import (
    make_log_k_grid,
    sigma2_from_Pk,
    xi_from_Pk,
    xi_model,
)


@pytest.fixture
def lcdm():
    return dict(sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)


def test_sigma8_self_consistency(lcdm):
    """integrating P_lin with top-hat at R=8 returns the input sigma8."""
    k = make_log_k_grid(1e-4, 1e2, 2000)
    P_lin = cj.plin_emulated(k, **lcdm)
    sig8 = float(jnp.sqrt(sigma2_from_Pk(jnp.array([8.0]), k, P_lin))[0])
    assert abs(sig8 - lcdm["sigma8"]) < 5e-3


def test_xi_finite_and_decaying(lcdm):
    """xi(r) is finite and decays from large positive at small r."""
    k = make_log_k_grid(1e-4, 1e2, 2000)
    P_NL = cj.run_halofit(k, **lcdm)
    r = jnp.logspace(np.log10(0.5), np.log10(150.0), 25)
    xi = np.asarray(xi_from_Pk(r, k, P_NL))
    assert np.all(np.isfinite(xi))
    # First (smallest r) should be much larger than last (largest r).
    assert xi[0] > xi[-1]


def test_xi_model_grad_om_finite_difference(lcdm):
    """jax.grad on a cosmology -> xi pipeline matches centered FD."""
    k = make_log_k_grid(5e-4, 5e1, 1500)
    r = jnp.asarray(np.logspace(0, 1.5, 12))
    fixed = {kk: lcdm[kk] for kk in ("sigma8", "Ob", "h", "ns", "a")}

    def loss(Om):
        P = cj.run_halofit(k, Om=Om, **fixed)
        return jnp.sum(xi_from_Pk(r, k, P))

    g = float(jax.grad(loss)(lcdm["Om"]))
    eps = 1e-4
    fd = (float(loss(lcdm["Om"] + eps)) - float(loss(lcdm["Om"] - eps))) / (2 * eps)
    rel = abs(g - fd) / (abs(fd) + 1e-12)
    assert rel < 1e-3, f"d xi/d Om rel diff {rel:.3e}"


def test_xi_model_convenience(lcdm):
    """xi_model wraps Pk_fn and the manual call."""
    k = make_log_k_grid(1e-4, 1e2, 2000)
    r = jnp.linspace(2.0, 100.0, 30)
    xi_a = xi_model(r, lambda kk: cj.run_halofit(kk, **lcdm), k_grid=k)
    P_NL = cj.run_halofit(k, **lcdm)
    xi_b = xi_from_Pk(r, k, P_NL)
    np.testing.assert_allclose(np.asarray(xi_a), np.asarray(xi_b),
                               rtol=1e-12, atol=1e-14)


def test_xi_jit_roundtrip(lcdm):
    k = make_log_k_grid(1e-4, 1e2, 1500)
    r = jnp.linspace(2.0, 60.0, 25)

    @jax.jit
    def jitted(Om):
        P = cj.run_halofit(k, sigma8=lcdm["sigma8"], Om=Om,
                           Ob=lcdm["Ob"], h=lcdm["h"], ns=lcdm["ns"], a=lcdm["a"])
        return xi_from_Pk(r, k, P)

    xi_jit = np.asarray(jitted(0.31))
    P = cj.run_halofit(k, **lcdm)
    xi_eager = np.asarray(xi_from_Pk(r, k, P))
    np.testing.assert_allclose(xi_jit, xi_eager, rtol=1e-12, atol=1e-14)
