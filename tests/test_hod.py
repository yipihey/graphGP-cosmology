"""Tests for the JAX-native Zheng07 HOD and halo_loader adapter."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.hod import (
    Zheng07Params,
    mean_ncen_zheng07,
    mean_nsat_zheng07,
    mean_ngal_zheng07,
)
from twopt_density.halo_loader import (
    apply_hod_zheng07,
    halocat_to_state_inputs,
    halos_to_positions_and_mass,
)


@pytest.fixture
def mass_grid():
    return jnp.logspace(10, 15, 64)


def test_ncen_matches_halotools(mass_grid):
    halotools = pytest.importorskip("halotools")
    from halotools.empirical_models import zheng07_components
    p = Zheng07Params()
    M = np.asarray(mass_grid)
    ref = zheng07_components.Zheng07Cens(threshold=-21).mean_occupation(prim_haloprop=M)
    ours = np.asarray(mean_ncen_zheng07(mass_grid, p))
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-14)


def test_nsat_matches_halotools_no_modulation(mass_grid):
    halotools = pytest.importorskip("halotools")
    from halotools.empirical_models import zheng07_components
    p = Zheng07Params()
    M = np.asarray(mass_grid)
    ref = zheng07_components.Zheng07Sats(threshold=-21).mean_occupation(prim_haloprop=M)
    ours = np.asarray(mean_nsat_zheng07(mass_grid, p, modulate_with_ncen=False))
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-14)


def test_nsat_modulation_multiplies_by_ncen(mass_grid):
    p = Zheng07Params()
    nsat_pure = mean_nsat_zheng07(mass_grid, p, modulate_with_ncen=False)
    nsat_mod = mean_nsat_zheng07(mass_grid, p, modulate_with_ncen=True)
    ncen = mean_ncen_zheng07(mass_grid, p)
    np.testing.assert_allclose(
        np.asarray(nsat_mod), np.asarray(nsat_pure * ncen),
        rtol=1e-12, atol=1e-14,
    )


def test_ngal_grad_finite_difference(mass_grid):
    """jax.grad over Zheng07 parameters matches centered FD."""
    M = mass_grid
    theta0 = jnp.array([12.79, 0.39, 11.92, 13.94, 1.15])

    def loss(theta):
        p = Zheng07Params(theta[0], theta[1], theta[2], theta[3], theta[4])
        return jnp.sum(mean_ngal_zheng07(M, p))

    g = np.asarray(jax.grad(loss)(theta0))
    eps = 1e-4
    fd = np.zeros_like(g)
    for i in range(5):
        e = jnp.zeros(5).at[i].set(eps)
        fd[i] = (float(loss(theta0 + e)) - float(loss(theta0 - e))) / (2 * eps)
    rel = np.abs(g - fd) / (np.abs(fd) + 1e-12)
    assert np.max(rel) < 1e-4, f"max rel {np.max(rel):.3e}"


def test_low_mass_limit_zero(mass_grid):
    """Far below logMmin, <Ncen> -> 0 and <Nsat> -> 0."""
    p = Zheng07Params(logMmin=13.0, sigma_logM=0.3)
    M = jnp.array([1e9])
    assert float(mean_ncen_zheng07(M, p)[0]) < 1e-30
    assert float(mean_nsat_zheng07(M, p)[0]) < 1e-30


def test_high_mass_ncen_saturates():
    """Far above logMmin, <Ncen> -> 1."""
    p = Zheng07Params()
    M = jnp.array([1e16])
    assert abs(float(mean_ncen_zheng07(M, p)[0]) - 1.0) < 1e-10


def test_apply_hod_zheng07_returns_jax_array():
    M = jnp.logspace(11, 14, 20)
    w = apply_hod_zheng07(M)
    assert hasattr(w, "shape")
    assert w.shape == (20,)
    assert np.all(np.asarray(w) >= 0)


def test_halocat_to_state_inputs_with_fakesim():
    halotools = pytest.importorskip("halotools")
    from halotools.sim_manager import FakeSim
    fake = FakeSim(num_massive_hosts=200, num_subs_per_massive_host=4, redshift=0)
    positions, M_h, Lbox, w = halocat_to_state_inputs(fake)
    assert positions.shape[1] == 3
    assert positions.shape[0] == M_h.shape[0] == w.shape[0]
    assert Lbox > 0
    # host_only=True means halo_upid == -1; should match halotools count
    expected = (np.asarray(fake.halo_table["halo_upid"]) == -1).sum()
    assert positions.shape[0] == expected


def test_halocat_to_state_inputs_with_dict():
    """Plain numpy/dict input also works."""
    rng = np.random.default_rng(0)
    N = 100
    table = {
        "halo_x": rng.uniform(0, 100, size=N),
        "halo_y": rng.uniform(0, 100, size=N),
        "halo_z": rng.uniform(0, 100, size=N),
        "halo_mvir": 10.0 ** rng.normal(13.0, 0.5, size=N),
        "halo_upid": np.full(N, -1, dtype=np.int64),
    }
    positions, M_h, Lbox, w = halocat_to_state_inputs(table, host_only=False)
    assert positions.shape == (N, 3) and M_h.shape == (N,) and w.shape == (N,)
    assert Lbox == 0.0  # not deducible from a dict
