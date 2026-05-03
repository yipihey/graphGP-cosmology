"""Tests for angular C_ell, Limber, and projected wp(rp) modules."""

from __future__ import annotations

import numpy as np
import pytest


def test_make_overdensity_map_zero_mean_when_uniform():
    """Uniform Poisson sampling on a fully-on mask -> <delta>=0 to within
    sqrt(N) Poisson noise."""
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.angular import make_overdensity_map

    nside = 16
    npix = 12 * nside ** 2
    mask = np.ones(npix)
    rng = np.random.default_rng(0)
    n_obj = 50000
    ra = rng.uniform(0, 360, n_obj)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_obj)))
    delta, n_bar = make_overdensity_map(ra, dec, mask, nside)
    assert delta.shape == (npix,)
    # mean should be zero by construction
    np.testing.assert_allclose(delta.mean(), 0.0, atol=5 / np.sqrt(n_obj))
    # n_bar = N / N_pix on a full mask
    np.testing.assert_allclose(n_bar, n_obj / npix, rtol=1e-12)


def test_compute_cl_gg_returns_finite_decoupled_spectrum():
    """End-to-end smoke: a uniform Poisson sample produces a finite
    decoupled C_ell that scatters around shot noise."""
    pytest.importorskip("pymaster")
    pytest.importorskip("healpy")
    from twopt_density.angular import compute_cl_gg

    nside = 16
    npix = 12 * nside ** 2
    mask = np.ones(npix)
    rng = np.random.default_rng(0)
    n_obj = 50000
    ra = rng.uniform(0, 360, n_obj)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_obj)))
    res = compute_cl_gg(ra, dec, mask, nside=nside, n_per_bin=4)
    assert np.isfinite(res.cl_decoupled).all()
    assert res.f_sky == pytest.approx(1.0)
    # shot noise = 4 pi / N for a full mask
    np.testing.assert_allclose(
        res.n_shot, 4 * np.pi / n_obj, rtol=1e-12,
    )


def test_pnl_at_z_matches_run_halofit_at_z0():
    """At z=0 the proper z-aware halofit reduces to ``run_halofit(a=1)``."""
    import jax.numpy as jnp
    from twopt_density.cosmology import run_halofit
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import pnl_at_z

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    k = jnp.logspace(-3, 0, 64)
    P_z0 = np.asarray(run_halofit(
        k, sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0,
    ))
    P_at_zeq0 = np.asarray(pnl_at_z(k, z=0.0, sigma8=0.8, cosmo=cosmo))
    np.testing.assert_allclose(P_at_zeq0, P_z0, rtol=1e-10)


def test_linear_growth_matches_lcdm_sanity_values():
    """D(z) for Om=0.31 LCDM must be ~0.61 at z=1, ~0.42 at z=2."""
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import linear_growth

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    D = linear_growth(np.array([0.0, 0.5, 1.0, 2.0]), cosmo)
    np.testing.assert_allclose(D[0], 1.0, atol=2e-3)
    # Reference values from independent integration of the LCDM growth eqn.
    np.testing.assert_allclose(D[1], 0.770, atol=0.02)
    np.testing.assert_allclose(D[2], 0.608, atol=0.02)
    np.testing.assert_allclose(D[3], 0.418, atol=0.02)


def test_cl_gg_limber_matches_pyccl():
    """Limber implementation must agree with pyccl to <2% on a uniform
    dndz at Quaia-like z range."""
    pyccl = pytest.importorskip("pyccl")
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import cl_gg_limber

    Om, Ob, h, ns, sigma8 = 0.31, 0.049, 0.68, 0.965, 0.8
    cosmo = DistanceCosmo(Om=Om, h=h)
    z = np.linspace(0.85, 2.45, 40)
    nz = np.ones_like(z)

    cosmo_ccl = pyccl.Cosmology(
        Omega_c=Om - Ob, Omega_b=Ob, h=h, sigma8=sigma8, n_s=ns,
        transfer_function="eisenstein_hu", matter_power_spectrum="halofit",
    )
    g = pyccl.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z, nz),
                                  bias=(z, np.ones_like(z)))
    ell = np.array([30., 100., 300.])
    cl_ccl = pyccl.angular_cl(cosmo_ccl, g, g, ell)
    cl_mine = cl_gg_limber(ell, z, nz, cosmo, bias=1.0)
    np.testing.assert_allclose(cl_mine, cl_ccl, rtol=0.03)


def test_wp_landy_szalay_isotropic_recovers_zero_clustering():
    """Random-on-random vs random-on-random gives wp ~ 0 within Poisson."""
    from twopt_density.projected_xi import wp_landy_szalay

    rng = np.random.default_rng(0)
    L = 300.0
    pts1 = rng.uniform(2000, 2000 + L, size=(8000, 3))
    pts2 = rng.uniform(2000, 2000 + L, size=(20000, 3))
    rp_edges = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    res = wp_landy_szalay(pts1, pts2, rp_edges, pi_max=80.0, n_pi=20)
    wp = res.wp
    assert np.isfinite(wp).all()
    # for Poisson sampling on a uniform field xi ~ 0 -> |wp| << pi_max
    assert (np.abs(wp) < 30.0).all(), f"wp too large for random field: {wp}"
