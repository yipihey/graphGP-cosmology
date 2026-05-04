"""Tests for the JAX cosmology distance module."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.distance import (
    DistanceCosmo,
    E_of_z,
    cartesian_to_radec_z,
    comoving_distance,
    radec_z_to_cartesian,
)


@pytest.fixture
def cosmo():
    return DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)


def test_E_of_z_at_zero(cosmo):
    assert abs(float(E_of_z(jnp.array([0.0]), cosmo)[0]) - 1.0) < 1e-12


def test_comoving_distance_matches_astropy(cosmo):
    astropy = pytest.importorskip("astropy.cosmology")
    ap = astropy.FlatLambdaCDM(H0=68, Om0=0.31, Tcmb0=0)
    z_test = jnp.array([0.05, 0.1, 0.5, 1.0, 2.0])
    d_jax = np.asarray(comoving_distance(z_test, cosmo))
    d_ap = np.array([ap.comoving_distance(float(z)).value * 0.68
                     for z in z_test])
    rel = np.max(np.abs(d_jax - d_ap) / d_ap)
    assert rel < 1e-6, f"max rel {rel:.3e}"


def test_radec_z_roundtrip(cosmo):
    rng = np.random.default_rng(0)
    ra = jnp.asarray(rng.uniform(0.0, 360.0, size=64))
    dec = jnp.asarray(rng.uniform(-60.0, 60.0, size=64))
    z = jnp.asarray(rng.uniform(0.05, 1.5, size=64))
    xyz = radec_z_to_cartesian(ra, dec, z, cosmo)
    ra_b, dec_b, z_b = cartesian_to_radec_z(xyz, cosmo)
    np.testing.assert_allclose(np.asarray(ra), np.asarray(ra_b), atol=1e-9)
    np.testing.assert_allclose(np.asarray(dec), np.asarray(dec_b), atol=1e-9)
    np.testing.assert_allclose(np.asarray(z), np.asarray(z_b), atol=1e-6)


def test_grad_d_d_Om_finite_difference(cosmo):
    """jax.grad of D_C(z=0.5) w.r.t. Om matches centered FD."""
    def loss(Om):
        c = DistanceCosmo(Om=Om, h=cosmo.h, w0=cosmo.w0, wa=cosmo.wa)
        return jnp.sum(comoving_distance(jnp.array([0.5]), c))
    g = float(jax.grad(loss)(0.31))
    eps = 1e-4
    fd = (float(loss(0.31 + eps)) - float(loss(0.31 - eps))) / (2 * eps)
    rel = abs(g - fd) / abs(fd)
    assert rel < 1e-6, f"d/dOm rel {rel:.3e}"


def test_grad_through_xyz_pair_distance(cosmo):
    """Gradient through (RA, Dec, z) -> xyz of a pair separation, vs FD."""
    ra = jnp.array([12.0, 13.0])
    dec = jnp.array([-5.0, -4.5])
    z = jnp.array([0.4, 0.42])

    def loss(Om):
        c = DistanceCosmo(Om=Om, h=cosmo.h, w0=cosmo.w0, wa=cosmo.wa)
        xyz = radec_z_to_cartesian(ra, dec, z, c)
        return jnp.sqrt(jnp.sum((xyz[0] - xyz[1]) ** 2))

    g = float(jax.grad(loss)(0.31))
    eps = 1e-4
    fd = (float(loss(0.31 + eps)) - float(loss(0.31 - eps))) / (2 * eps)
    rel = abs(g - fd) / abs(fd)
    assert rel < 1e-5, f"d|sep|/dOm rel {rel:.3e}"


def test_h_independence_of_Mpch_distance(cosmo):
    """Mpc/h distances are independent of h by construction."""
    c1 = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)
    c2 = DistanceCosmo(Om=0.31, h=0.5, w0=-1.0, wa=0.0)
    z = jnp.array([0.1, 0.5, 1.0])
    d1 = np.asarray(comoving_distance(z, c1))
    d2 = np.asarray(comoving_distance(z, c2))
    np.testing.assert_allclose(d1, d2, atol=1e-9)


def test_w0_dependence(cosmo):
    """Quintessence w0 != -1 changes D_C measurably."""
    c1 = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)
    c2 = DistanceCosmo(Om=0.31, h=0.68, w0=-0.9, wa=0.0)
    z = jnp.array([0.5, 1.0])
    d1 = np.asarray(comoving_distance(z, c1))
    d2 = np.asarray(comoving_distance(z, c2))
    assert np.max(np.abs(d1 - d2)) > 5.0  # Mpc/h shift is well above noise
