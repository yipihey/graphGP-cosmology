"""Tests for the small scipy + jax MAP-fit helper."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.fit import map_fit


def test_recovers_quadratic_minimum():
    """Trivial quadratic: argmin of (theta - mu)^2 is mu."""
    mu = jnp.array([1.0, -2.0, 0.5])

    def loss(theta):
        return jnp.sum((theta - mu) ** 2)

    res = map_fit(loss, theta0=jnp.zeros(3))
    assert res.success
    np.testing.assert_allclose(res.theta, np.asarray(mu), atol=1e-6)


def test_bounds_enforced():
    """L-BFGS-B respects box constraints."""
    def loss(theta):
        return -jnp.sum(theta)  # tries to push to +infinity

    res = map_fit(loss, theta0=[0.5, 0.5], bounds=[(0.0, 1.0), (0.0, 1.0)])
    assert res.success
    np.testing.assert_allclose(res.theta, [1.0, 1.0], atol=1e-6)


def test_recovers_cosmology_chi2():
    """Recovers (Om, sigma8, b^2) from a noiseless mock."""
    pytest.importorskip("mcfit")
    from twopt_density import cosmology as cj
    from twopt_density.spectra import (
        FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog,
    )
    Om_t, s8_t, b2_t = 0.30, 0.78, 2.0
    k = make_log_k_grid(1e-4, 1e2, 2048)
    fft = FFTLogP2xi(k, l=0)
    s = jnp.asarray(np.logspace(np.log10(3.0), np.log10(40.0), 12))

    def model(theta):
        Om, sigma8, b2 = theta
        P = cj.run_halofit(k, sigma8=sigma8, Om=Om, Ob=0.049, h=0.68,
                           ns=0.965, a=1.0)
        return b2 * xi_from_Pk_fftlog(s, fft, P)

    xi_data = model(jnp.array([Om_t, s8_t, b2_t]))

    def loss(theta):
        return jnp.sum((xi_data - model(theta)) ** 2)

    res = map_fit(
        loss, theta0=[0.27, 0.75, 1.5],
        bounds=[(0.20, 0.45), (0.6, 1.0), (0.5, 5.0)],
        tol=1e-10,
    )
    assert res.success
    np.testing.assert_allclose(res.theta, [Om_t, s8_t, b2_t], rtol=1e-3)
