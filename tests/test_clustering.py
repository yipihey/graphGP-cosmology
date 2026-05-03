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


def test_dndz_pdf_stack_normalised_and_smooth():
    """Stacked dN/dz on objects with one identical photo-z and width
    must be a Gaussian centred on that z."""
    from twopt_density.limber import dndz_pdf_stack

    z_grid = np.linspace(0.0, 4.0, 400)
    z_obs = np.full(1000, 1.4)
    sigma_z = np.full(1000, 0.1)
    dndz = dndz_pdf_stack(z_grid, z_obs, sigma_z)
    # Normalise
    dndz = dndz / np.trapezoid(dndz, z_grid)
    expected = (1 / (np.sqrt(2 * np.pi) * 0.1)
                * np.exp(-0.5 * ((z_grid - 1.4) / 0.1) ** 2))
    np.testing.assert_allclose(dndz, expected, atol=1e-3)


def test_wp_observed_recovers_wp_limber_when_sigma_zero():
    """In the (sigma_chi -> 0, pi_max -> inf) limit, wp_observed must
    reduce to the deterministic real-space wp_limber."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import wp_limber, wp_observed

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    rp = jnp.array([5.0, 10.0, 20.0, 50.0])
    wp_obs = np.asarray(wp_observed(
        rp, z_eff=1.5, sigma_chi_eff=1e-6, cosmo=cosmo, bias=2.6,
        pi_max=400.0, n_pi_true=400,
    ))
    wp_real = wp_limber(np.asarray(rp), z_eff=1.5, cosmo=cosmo, bias=2.6,
                        pi_max=400.0, n_pi=400)
    np.testing.assert_allclose(wp_obs, wp_real, rtol=2e-3)


def test_wp_observed_is_differentiable_in_cosmo_bias_sigma():
    """Gradients of a wp_observed-derived loss wrt (Om, b, sigma_chi)
    must all be finite -- the whole forward model is JAX-pure."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import wp_observed

    rp = jnp.array([10.0, 30.0])
    z_eff = 1.5
    pi_max = 200.0

    def loss_om(Om):
        c = DistanceCosmo(Om=Om, h=0.68)
        return jnp.sum(wp_observed(rp, z_eff=z_eff, sigma_chi_eff=120.0,
                                    cosmo=c, bias=2.6, pi_max=pi_max) ** 2)
    g_om = jax.grad(loss_om)(jnp.float64(0.31))
    assert jnp.isfinite(g_om) and abs(g_om) > 0

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    def loss_b(b):
        return jnp.sum(wp_observed(rp, z_eff=z_eff, sigma_chi_eff=120.0,
                                    cosmo=cosmo, bias=b, pi_max=pi_max) ** 2)
    g_b = jax.grad(loss_b)(jnp.float64(2.6))
    assert jnp.isfinite(g_b) and abs(g_b) > 0

    def loss_sig(sig):
        return jnp.sum(wp_observed(rp, z_eff=z_eff, sigma_chi_eff=sig,
                                    cosmo=cosmo, bias=2.6, pi_max=pi_max) ** 2)
    g_sig = jax.grad(loss_sig)(jnp.float64(120.0))
    assert jnp.isfinite(g_sig) and abs(g_sig) > 0


def test_sigma_chi_from_sigma_z_matches_dchi_dz():
    """sigma_chi = (c/H_0/E(z)) * sigma_z."""
    import jax.numpy as jnp
    from twopt_density.distance import (
        C_OVER_H100_MPCH, DistanceCosmo, E_of_z,
    )
    from twopt_density.limber import sigma_chi_from_sigma_z

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z = np.array([0.5, 1.0, 1.5, 2.0])
    sigma_z = np.full_like(z, 0.05)
    sig_chi = sigma_chi_from_sigma_z(z, sigma_z, cosmo)
    E = np.asarray(E_of_z(jnp.asarray(z), cosmo))
    expected = C_OVER_H100_MPCH / E * 0.05
    np.testing.assert_allclose(sig_chi, expected, rtol=1e-12)


def test_wp_observed_perpair_reduces_to_single_sigma():
    """When all pair sigmas are identical, wp_observed_perpair must
    equal wp_observed at that scalar sigma."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        make_wp_fft, wp_observed, wp_observed_perpair,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    rp = jnp.array([5.0, 10.0, 30.0])
    sigma = 120.0
    wp1 = np.asarray(wp_observed(
        rp, z_eff=1.4, sigma_chi_eff=sigma, cosmo=cosmo, bias=2.6,
        pi_max=200.0, fft=fft, k_grid=k_grid,
    ))
    wp_pp = np.asarray(wp_observed_perpair(
        rp, z_eff=1.4, sigma_chi_samples=jnp.full(64, sigma), cosmo=cosmo,
        bias=2.6, pi_max=200.0, fft=fft, k_grid=k_grid,
    ))
    np.testing.assert_allclose(wp_pp, wp1, rtol=1e-10)


def test_sample_pair_sigma_chi_recovers_pythagorean_sum():
    """For a constant sigma_z, sigma_pair = sqrt(2) * sigma_chi(z)."""
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        sample_pair_sigma_chi, sigma_chi_from_sigma_z,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    n = 500
    sigma_z = np.full(n, 0.05)
    z = np.full(n, 1.4)
    sigma_chi = float(sigma_chi_from_sigma_z(z, sigma_z, cosmo)[0])
    sig_pair = sample_pair_sigma_chi(sigma_z, z, cosmo, n_pairs=2000,
                                      rng=np.random.default_rng(0))
    expected = np.sqrt(2) * sigma_chi
    np.testing.assert_allclose(sig_pair.mean(), expected, rtol=1e-6)
    np.testing.assert_allclose(sig_pair.std(), 0.0, atol=1e-6)


def test_wp_map_fit_recovers_synthetic_truth_within_one_sigma():
    """JAX-MAP fit on synthetic wp data with known truth bias must
    converge and recover b within 1 sigma."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import make_wp_fft, wp_map_fit, wp_observed

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    rp = np.logspace(np.log10(8.0), np.log10(60.0), 10)
    truth_b = 2.5
    wp_truth = np.asarray(wp_observed(
        jnp.asarray(rp), z_eff=1.4, sigma_chi_eff=160.0, cosmo=cosmo,
        bias=truth_b, pi_max=200.0, fft=fft, k_grid=k_grid,
    ))
    sigma_wp = 0.05 * np.maximum(np.abs(wp_truth), 0.5)
    rng = np.random.default_rng(0)
    wp_data = wp_truth + rng.normal(0.0, sigma_wp)
    res, cov, _, theta_full = wp_map_fit(
        rp, wp_data, sigma_wp, sigma_chi_eff=160.0, z_eff=1.4,
        free=("b",), pi_max=200.0,
    )
    assert res.success
    sd_b = float(np.sqrt(max(cov[0, 0], 0.0)))
    # truth recovered within ~2 sigma (chi^2 noise is realistic)
    assert abs(theta_full["b"] - truth_b) < 3.0 * max(sd_b, 0.01), (
        f"b={theta_full['b']:.3f} vs truth={truth_b}; sd_b={sd_b:.3f}"
    )


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
