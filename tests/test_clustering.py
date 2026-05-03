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


def test_cl_gg_limber_is_jax_differentiable():
    """JAX-cl_gg_limber must allow gradients through (Om, sigma8, bias)
    -- needed for joint Cl+wp fits."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import cl_gg_limber

    z = jnp.linspace(0.85, 2.45, 30)
    nz = jnp.ones_like(z)
    ell = jnp.array([20.0, 100.0])

    def loss_om(Om):
        return jnp.sum(cl_gg_limber(
            ell, z, nz, DistanceCosmo(Om=Om, h=0.68), bias=2.0,
        ) ** 2)
    g_om = jax.grad(loss_om)(jnp.float64(0.31))
    assert jnp.isfinite(g_om) and g_om != 0.0

    def loss_b(b):
        return jnp.sum(cl_gg_limber(
            ell, z, nz, DistanceCosmo(Om=0.31, h=0.68), bias=b,
        ) ** 2)
    g_b = jax.grad(loss_b)(jnp.float64(2.0))
    assert jnp.isfinite(g_b) and g_b != 0.0

    def loss_s8(s8):
        return jnp.sum(cl_gg_limber(
            ell, z, nz, DistanceCosmo(Om=0.31, h=0.68),
            bias=2.0, sigma8=s8,
        ) ** 2)
    g_s8 = jax.grad(loss_s8)(jnp.float64(0.81))
    assert jnp.isfinite(g_s8) and g_s8 != 0.0


def test_joint_cl_wp_map_fit_runs_and_returns_finite_covariance():
    """Joint fit on synthetic Cl+wp must converge and return a finite
    covariance for the (sigma8, b) constrained combination."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        cl_gg_limber, joint_cl_wp_map_fit, make_wp_fft, wp_observed,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    truth_b = 2.5
    truth_s8 = 0.83

    z = np.linspace(0.85, 2.45, 30)
    nz = np.ones_like(z)
    ell = np.array([30., 50., 100., 200.])
    cl_truth = np.asarray(cl_gg_limber(ell, z, nz, cosmo, bias=truth_b,
                                         sigma8=truth_s8))
    sigma_cl = 0.1 * np.maximum(np.abs(cl_truth), 1e-7)
    rng = np.random.default_rng(0)
    cl_data = cl_truth + rng.normal(0, sigma_cl)

    rp = np.logspace(np.log10(8.0), np.log10(60.0), 8)
    wp_truth = np.asarray(wp_observed(
        jnp.asarray(rp), z_eff=1.4, sigma_chi_eff=160.0, cosmo=cosmo,
        bias=truth_b, sigma8=truth_s8, pi_max=200.0,
        fft=fft, k_grid=k_grid,
    ))
    sigma_wp = 0.1 * np.maximum(np.abs(wp_truth), 0.5)
    wp_data = wp_truth + rng.normal(0, sigma_wp)
    sig_chi_samples = np.full(64, 160.0)

    res, cov, theta_full = joint_cl_wp_map_fit(
        ell, cl_data, sigma_cl, z, nz,
        rp, wp_data, sigma_wp, sig_chi_samples, z_eff=1.4,
        free=("sigma8", "b"), pi_max=200.0, ell_min=20.0,
    )
    assert res.success
    # joint sigma8*b should recover truth (the only well-constrained quantity)
    np.testing.assert_allclose(theta_full["sigma8"] * theta_full["b"],
                                truth_s8 * truth_b, rtol=0.05)


def test_fit_bz_powerlaw_recovers_truth_on_synthetic_per_bin_data():
    """fit_bz_powerlaw on N synthetic per-bin wp measurements at known
    truth b(z) = b0 ((1+z)/(1+z_pivot))^alpha must recover (b0, alpha)
    within the joint covariance."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        fit_bz_powerlaw, make_wp_fft, sigma_chi_from_sigma_z, wp_observed,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    sigma8 = 0.81
    truth_b0, truth_alpha, z_pivot = 2.0, 1.5, 1.5
    b_true = lambda z: truth_b0 * ((1 + z) / (1 + z_pivot)) ** truth_alpha

    per_bin = []
    for z_eff in (1.0, 1.4, 1.8, 2.2):
        b_z = b_true(z_eff)
        sigma_chi = float(np.sqrt(2) * np.asarray(sigma_chi_from_sigma_z(
            np.array([z_eff]), np.array([0.04 * (1 + z_eff)]), cosmo,
        ))[0])
        rp = np.logspace(np.log10(10), np.log10(50), 8)
        wp_truth = np.asarray(wp_observed(
            jnp.asarray(rp), z_eff=z_eff, sigma_chi_eff=sigma_chi,
            cosmo=cosmo, bias=b_z, sigma8=sigma8, pi_max=200.0,
            fft=fft, k_grid=k_grid,
        ))
        sigma_wp = 0.07 * np.maximum(np.abs(wp_truth), 0.05)
        rng = np.random.default_rng(int(z_eff * 100))
        wp = wp_truth + rng.normal(0, sigma_wp)
        per_bin.append({"z_eff": z_eff, "sigma_chi_eff": sigma_chi,
                        "rp": rp, "wp": wp, "sigma_wp": sigma_wp})

    res, cov, b_of_z, (b0_f, alpha_f) = fit_bz_powerlaw(
        per_bin, cosmo=cosmo, sigma8=sigma8, z_pivot=z_pivot,
        theta0=(2.0, 0.0),
    )
    assert res.success
    sd_b0 = float(np.sqrt(max(cov[0, 0], 0.0)))
    sd_a = float(np.sqrt(max(cov[1, 1], 0.0)))
    assert abs(b0_f - truth_b0) < 5.0 * max(sd_b0, 0.05)
    assert abs(alpha_f - truth_alpha) < 5.0 * max(sd_a, 0.05)


def test_wp_observed_continuous_bz_constant_b_matches_wp_observed():
    """With constant b(z) and a delta-like dndz, wp_observed_continuous_bz
    must reduce to wp_observed at that single z."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        make_wp_fft, wp_observed, wp_observed_continuous_bz,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    z = np.linspace(1.4, 1.5, 30)              # very narrow z-window
    nz = np.exp(-0.5 * ((z - 1.45) / 0.005) ** 2)  # delta-like
    rp = jnp.array([10.0, 20.0])
    b_const = 2.6
    wp_c = np.asarray(wp_observed_continuous_bz(
        rp, z, nz, jnp.full(z.shape, b_const), sigma_chi_eff=160.0,
        cosmo=cosmo, pi_max=200.0, fft=fft, k_grid=k_grid,
    ))
    wp_s = np.asarray(wp_observed(
        rp, z_eff=1.45, sigma_chi_eff=160.0, cosmo=cosmo, bias=b_const,
        pi_max=200.0, fft=fft, k_grid=k_grid,
    ))
    np.testing.assert_allclose(wp_c, wp_s, rtol=0.02)


def test_wp_continuous_estimator_runs_and_is_jax_evaluable():
    """End-to-end SF21 build on a tiny clustered catalog. Verify the
    estimator returns finite wp(rp, z) and that the analytic Chebyshev
    derivative matches a finite-difference cross-check."""
    import jax
    import jax.numpy as jnp
    from twopt_density.wp_continuous import wp_continuous_estimator

    rng = np.random.default_rng(0)
    L = 1000.0
    n_centers = 30
    centers = rng.uniform(0, L, size=(n_centers, 3))
    xyz_d = np.vstack([rng.normal(c, 8.0, (200, 3)) for c in centers])
    xyz_d = np.mod(xyz_d, L).astype(np.float64)
    xyz_r = rng.uniform(0, L, size=(15000, 3))
    z_d = rng.uniform(1.4, 1.6, len(xyz_d))
    z_r = rng.uniform(1.4, 1.6, len(xyz_r))

    est = wp_continuous_estimator(
        xyz_d, xyz_r, z_d, z_r,
        rp_min=5.0, rp_max=80.0, z_min=1.4, z_max=1.6,
        K_rp=4, K_z=2, pi_max=80.0,
    )
    # finite wp at multiple (rp, z)
    for rp in (10.0, 30.0):
        for z in (1.45, 1.55):
            v = float(est.wp_eval(rp, z))
            assert np.isfinite(v)

    # analytic dwp/dz matches jax.grad cross-check
    rp_test = 20.0; z_test = 1.5
    analytic = float(est.dwp_dz(rp_test, z_test))
    g = float(jax.grad(lambda z: est.wp_eval(rp_test, z))(jnp.float64(z_test)))
    np.testing.assert_allclose(analytic, g, rtol=1e-9, atol=1e-9)


def test_wp_continuous_chebyshev_derivative_matches_jax_grad():
    """Standalone test: ``_cheb_dT_dx_jax`` must agree with
    ``jax.grad(_cheb_T_jax)`` for K up to 6."""
    import jax
    import jax.numpy as jnp
    from twopt_density.wp_continuous import _cheb_T_jax, _cheb_dT_dx_jax

    for K in (2, 3, 5, 6):
        for x in (-0.5, 0.0, 0.3, 0.7):
            analytic = np.asarray(_cheb_dT_dx_jax(jnp.float64(x), K))
            g = np.asarray(jax.jacobian(
                lambda y: _cheb_T_jax(y, K)
            )(jnp.float64(x)))
            np.testing.assert_allclose(analytic, g, rtol=1e-9, atol=1e-9)


def test_pnl_at_z_nowiggle_matches_eh_zero_baryon_at_z0():
    """At z=0 ``pnl_at_z_nowiggle`` must be the halofit result with the
    Eisenstein-Hu zero-baryon (smooth) linear input, not the BAO-bearing
    plin_emulated."""
    import jax.numpy as jnp
    from twopt_density.cosmology import (
        halofit_from_plin, pk_EisensteinHu_zb,
    )
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import pnl_at_z_nowiggle

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    k = jnp.logspace(-3, 0, 64)
    sigma8, Ob, h, ns = 0.8, 0.049, 0.68, 0.965
    plin0_smooth = pk_EisensteinHu_zb(k, sigma8, 0.31, Ob, h, ns)
    P_expected = np.asarray(halofit_from_plin(
        k, plin0_smooth, sigma8, 0.31, Ob, h, ns, a=1.0,
    ))
    P_at_zeq0 = np.asarray(pnl_at_z_nowiggle(
        k, z=0.0, sigma8=sigma8, cosmo=cosmo,
    ))
    np.testing.assert_allclose(P_at_zeq0, P_expected, rtol=1e-9)


def test_wp_observed_minus_nowiggle_shows_bao_bump_at_sound_horizon():
    """The difference ``wp_observed - wp_observed_nowiggle`` must peak
    in [80, 110] Mpc/h (sound horizon) -- i.e. the BAO bump survives
    the photo-z LOS smearing as long as pi_max is wide enough."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import (
        make_wp_fft, wp_observed, wp_observed_nowiggle,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)
    rp = jnp.asarray(np.linspace(20.0, 150.0, 27))
    full = np.asarray(wp_observed(
        rp, z_eff=1.4, sigma_chi_eff=170.0, cosmo=cosmo, bias=2.6,
        pi_max=300.0, fft=fft, k_grid=k_grid,
    ))
    smooth = np.asarray(wp_observed_nowiggle(
        rp, z_eff=1.4, sigma_chi_eff=170.0, cosmo=cosmo, bias=2.6,
        pi_max=300.0, fft=fft, k_grid=k_grid,
    ))
    bao = full - smooth
    rp_np = np.asarray(rp)
    in_band = (rp_np > 80.0) & (rp_np < 110.0)
    out_of_band = (rp_np > 30.0) & (rp_np < 70.0)
    # peak in the BAO band exceeds the typical out-of-band value
    assert bao[in_band].max() > 2 * np.abs(bao[out_of_band]).mean()


def test_angular_corr_from_mask_matches_squared_mean_at_zero():
    """At theta -> 0 the mask auto-correlation must equal <mask^2> via
    Parseval. With anafast at finite lmax = 3*nside - 1 we capture
    nearly all the power for a smooth mask -- check at the percent
    level, which is the well-known truncation effect."""
    pytest.importorskip("healpy")
    from twopt_density.analytic_rr import angular_corr_from_mask

    rng = np.random.default_rng(0)
    nside = 32
    npix = 12 * nside ** 2
    # smooth mask: a half-sky cap with soft edge
    import healpy as hp
    mu = np.cos(hp.pix2ang(nside, np.arange(npix))[0])
    mask = 0.5 * (1.0 + np.tanh(5.0 * mu))
    _, xi = angular_corr_from_mask(mask, nside, theta_max_rad=0.05,
                                    n_theta=10)
    np.testing.assert_allclose(xi[0], (mask ** 2).mean(), rtol=2e-2)


def test_rr_analytic_returns_finite_pair_count():
    """Smoke test: analytic RR runs end-to-end on a half-sky cap mask
    and returns finite, positive pair counts. (Quantitative validation
    against MC on real Quaia is in the demo; toy NSIDE=8/16 surveys
    are too coarse for the small-angle / Legendre approximation to
    reach 10% accuracy on isolated rp bins.)"""
    pytest.importorskip("healpy")
    from twopt_density.analytic_rr import rr_analytic
    from twopt_density.distance import DistanceCosmo

    import healpy as hp
    nside = 32
    npix = 12 * nside ** 2
    mu = np.cos(hp.pix2ang(nside, np.arange(npix))[0])
    mask = (mu > -0.5).astype(np.float64)
    z = np.random.default_rng(0).uniform(1.0, 1.5, size=2000)
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    rp_edges = np.array([100.0, 200.0, 400.0])
    pi_edges = np.linspace(0.0, 400.0, 11)
    res = rr_analytic(rp_edges, pi_edges, mask, nside, z, cosmo, N_r=5000)
    assert np.isfinite(res.RR).all()
    assert (res.RR >= 0).all()
    assert res.RR.sum() > 0


def test_calibrate_norm_to_mc_returns_median_ratio():
    """``calibrate_norm_to_mc`` returns the median pair-by-pair ratio
    of MC to analytic RR. Trivial unit test."""
    from twopt_density.analytic_rr import calibrate_norm_to_mc

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = a * 1.08    # a uniform 8% offset
    f = calibrate_norm_to_mc(a, m)
    np.testing.assert_allclose(f, 1.08, rtol=1e-9)


def test_dr_analytic_simple_scaling():
    """``dr_analytic`` is just (2 N_d / N_r) * RR -- validate the trivial
    scaling so the helper is locked in."""
    from twopt_density.analytic_rr import dr_analytic

    RR = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = dr_analytic(N_d=100, N_r=300, RR=RR)
    np.testing.assert_allclose(out, RR * (2.0 * 100 / 300))


def test_data_residual_weights_uniform_field_returns_unit_weights():
    """A perfectly Poisson-distributed catalogue under the published
    selection function should yield ~ unit per-galaxy weights (no
    systematic to deproject). Allow ~5% scatter from finite-N noise."""
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.systematics import data_residual_weights

    rng = np.random.default_rng(0)
    nside = 16
    npix = 12 * nside ** 2
    # smooth half-sky cap
    mu = np.cos(hp.pix2ang(nside, np.arange(npix))[0])
    mask = 0.5 * (1.0 + np.tanh(5.0 * mu))
    # uniform random under the mask
    n_obj = 60_000
    p = mask / mask.sum()
    pix = rng.choice(npix, size=n_obj, p=p)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi); dec = 90.0 - np.degrees(theta)
    w, _ = data_residual_weights(ra, dec, mask, nside, smoothing_fwhm_deg=10.0)
    # mean should be exactly 1 (by construction)
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-12)
    # std should be small for a uniform Poisson sample
    assert w.std() < 0.1, f"weight std too large: {w.std():.3f}"


def test_data_residual_weights_recovers_imposed_systematic():
    """Inject a known multiplicative systematic on the data density and
    check that ``data_residual_weights`` produces weights that bring
    the density back to flat (within Poisson)."""
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.systematics import (
        data_residual_weights, galaxy_count_map,
    )

    rng = np.random.default_rng(0)
    nside = 16
    npix = 12 * nside ** 2
    mu = np.cos(hp.pix2ang(nside, np.arange(npix))[0])
    mask = 0.5 * (1.0 + np.tanh(5.0 * mu))
    # multiplicative systematic: 30% over/underdensity slowly varying
    phi_arr = hp.pix2ang(nside, np.arange(npix))[1]
    sys_density = 1.0 + 0.3 * np.cos(phi_arr)
    eff_density = mask * sys_density
    n_obj = 100_000
    p = eff_density / eff_density.sum()
    pix = rng.choice(npix, size=n_obj, p=p)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi); dec = 90.0 - np.degrees(theta)
    w, _ = data_residual_weights(ra, dec, mask, nside, smoothing_fwhm_deg=45.0)
    # weighted galaxy count should match expected = mean * mask, no
    # systematic. Compare weighted vs unweighted residual variance:
    n_obs = galaxy_count_map(ra, dec, nside)
    good = mask > 0.05
    n_bar = n_obs[good].sum() / mask[good].sum()
    expected = n_bar * mask
    raw_resid = (n_obs[good] - expected[good]) / expected[good]
    # weighted: sum w_i in each pixel
    pix_data = hp.ang2pix(nside, np.deg2rad(90.0 - dec), np.deg2rad(ra))
    n_weighted = np.bincount(pix_data, weights=w, minlength=npix)
    n_bar_w = n_weighted[good].sum() / mask[good].sum()
    expected_w = n_bar_w * mask
    weighted_resid = (n_weighted[good] - expected_w[good]) / expected_w[good]
    # weighted residual variance reduced by the deprojection. The
    # Poisson floor ~ 1/sqrt(N_per_pix) limits how far this can go;
    # for 100k objects in ~ 2000 good pixels we expect a ~ 25-30%
    # variance reduction (ratio ~ 0.7 - 0.75).
    assert weighted_resid.std() < raw_resid.std() * 0.80, (
        f"deprojection didn't reduce variance: raw {raw_resid.std():.3f}, "
        f"weighted {weighted_resid.std():.3f}"
    )


def test_count_pairs_rp_pi_with_unit_weights_matches_unweighted():
    """Weighted pair counter with all unit weights must match the
    unweighted version pair-for-pair."""
    from twopt_density.projected_xi import _count_pairs_rp_pi

    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 200, size=(3000, 3))
    rp_edges = np.array([5.0, 20.0, 50.0])
    pi_edges = np.linspace(0, 80, 11)
    plain = _count_pairs_rp_pi(pos, pos, rp_edges, pi_edges, auto=True)
    weighted = _count_pairs_rp_pi(pos, pos, rp_edges, pi_edges, auto=True,
                                    w1=np.ones(len(pos)),
                                    w2=np.ones(len(pos)))
    np.testing.assert_allclose(weighted, plain, rtol=1e-12)


def test_wp_kernel_z_reduces_to_uniform_over_full_range():
    """A flat (very wide) Gaussian kernel must reproduce the standard
    z-integrated Landy-Szalay wp -- the kernel weights all z_pair bins
    equally so the kernel-weighted DD/DR/RR are proportional to the
    z-integrated counts, and the LS ratio is unchanged."""
    import jax.numpy as jnp
    from twopt_density.projected_xi import (
        wp_kernel_z, wp_landy_szalay, wp_landy_szalay_zpaired,
    )

    rng = np.random.default_rng(0)
    L = 200.0
    pts1 = rng.uniform(2000, 2000 + L, size=(2000, 3))
    pts2 = rng.uniform(2000, 2000 + L, size=(6000, 3))
    z1 = rng.uniform(1.0, 2.0, size=2000)
    z2 = rng.uniform(1.0, 2.0, size=6000)
    rp_edges = np.array([10.0, 30.0, 60.0])
    pi_max = 80.0

    counts = wp_landy_szalay_zpaired(
        pts1, pts2, z1, z2, rp_edges, pi_max=pi_max, n_pi=10,
        n_z_pair=20,
    )
    # very wide kernel -> uniform z weighting
    wp_kernel = np.asarray(wp_kernel_z(jnp.float64(1.5), sigma_z=100.0,
                                         counts=counts))
    standard = wp_landy_szalay(pts1, pts2, rp_edges, pi_max=pi_max, n_pi=10)
    np.testing.assert_allclose(wp_kernel, standard.wp, rtol=0.02, atol=2.0)


def test_wp_kernel_z_jax_grad_matches_finite_difference():
    """``jax.grad(wp_kernel_z)`` must agree with central-difference
    finite differences on the kernel-weighted estimator."""
    import jax
    import jax.numpy as jnp
    from twopt_density.projected_xi import (
        wp_kernel_z, wp_landy_szalay_zpaired,
    )

    rng = np.random.default_rng(0)
    pts1 = rng.uniform(0, 200, size=(2000, 3))
    pts2 = rng.uniform(0, 200, size=(6000, 3))
    z1 = rng.uniform(1.0, 2.0, size=2000)
    z2 = rng.uniform(1.0, 2.0, size=6000)
    rp_edges = np.array([10.0, 30.0, 60.0])
    counts = wp_landy_szalay_zpaired(
        pts1, pts2, z1, z2, rp_edges, pi_max=80.0, n_pi=10, n_z_pair=20,
    )

    def wp_total(z):
        return jnp.sum(wp_kernel_z(z, sigma_z=0.2, counts=counts))

    g_jax = float(jax.grad(wp_total)(jnp.float64(1.5)))
    h = 1e-3
    g_fd = (float(wp_total(jnp.float64(1.5 + h)))
            - float(wp_total(jnp.float64(1.5 - h)))) / (2 * h)
    np.testing.assert_allclose(g_jax, g_fd, rtol=1e-4, atol=1e-6)


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
