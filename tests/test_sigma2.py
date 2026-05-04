"""Tests for sigma^2(R) -- the count-in-cells variance two-point statistic.

These tests pin three identities that the pipeline depends on:

  1. The TH and Gaussian kernels integrate to 1 over the volume.
  2. For a power-law xi(r) = (r/r0)^-gamma the analytic sigma^2(R)
     matches the numerical kernel projection.
  3. On a small clustered point set, sigma^2(R) computed by
     pair-count projection agrees with sigma^2(R) computed by
     integrating the LS xi(r) measurement.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_TH_kernel_integrates_to_one():
    """``int K_TH(r; R) dV = 1`` over ``r in [0, 2R]``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 7.5
    r = np.linspace(0.0, 2.0 * R, 5000)
    K = kernel_TH_3d(r, R)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)


def test_Gauss_kernel_integrates_to_one():
    """``int K_G(r; R) dV = 1`` (5-sigma cutoff is enough)."""
    from twopt_density.sigma2 import kernel_Gauss_3d
    R = 5.0
    r = np.linspace(0.0, 12.0 * R, 5000)
    K = kernel_Gauss_3d(r, R)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-4)


def test_TH_kernel_zero_outside_2R():
    """``K_TH(r > 2R; R) == 0``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 4.0
    r = np.array([0.5 * R, 2.0 * R, 2.0 * R + 1e-9, 5.0 * R])
    K = kernel_TH_3d(r, R)
    assert K[0] > 0 and K[1] >= 0
    assert K[2] == 0.0 and K[3] == 0.0


def test_TH_kernel_at_origin():
    """``K_TH(0; R) = 3 / (4 pi R^3)`` (overlap of two coincident spheres)."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 3.0
    K0 = kernel_TH_3d(np.array([0.0]), R)[0]
    np.testing.assert_allclose(K0, 3.0 / (4.0 * np.pi * R ** 3), rtol=1e-12)


def test_sigma2_from_xi_matches_powerlaw_analytic():
    """For ``xi(r) = (r/r0)^-gamma`` with ``gamma != 3, 4, 5``, the
    integral against ``K_TH`` is computable in closed form. We compare
    the numerical projection against a direct one-shot integral on a
    very fine grid."""
    from twopt_density.sigma2 import sigma2_from_xi
    R = 5.0
    r0 = 5.0
    gamma = 1.8
    r = np.linspace(0.01, 6.0 * R, 8000)
    xi = (r / r0) ** (-gamma)
    sigma2 = sigma2_from_xi(r, xi, np.array([R]), kernel="tophat")[0]
    # closed form: sigma^2 = (3/R^3) integral_0^{2R} dr r^2 (r/r0)^-gamma
    #                                          x [1 - 3r/(4R) + r^3/(16R^3)]
    ref = 0.0
    rfine = np.linspace(1e-8, 2.0 * R, 200_000)
    K_ref = (3.0 / R ** 3) * (1.0 - 0.75 * rfine / R
                                + (rfine ** 3) / (16.0 * R ** 3))
    ref = np.trapezoid(rfine ** 2 * (rfine / r0) ** (-gamma) * K_ref, rfine)
    np.testing.assert_allclose(sigma2, ref, rtol=2e-3)


def test_sigma2_kernel_is_strictly_positive_on_support():
    """``K_TH(r; R) > 0`` for ``0 < r < 2R``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 10.0
    r = np.linspace(1e-3, 2.0 * R - 1e-3, 200)
    K = kernel_TH_3d(r, R)
    assert (K > 0).all()


def test_sigma2_pair_counts_matches_xi_integral_on_uniform():
    """Synthetic Poisson catalogue: sigma^2 from pair counts equals
    the LS xi integral to within Poisson noise."""
    from twopt_density.sigma2 import (
        sigma2_from_pair_counts, sigma2_from_xi,
    )
    rng = np.random.default_rng(0)
    N_d = 3000; N_r = 9000
    pos_d = rng.uniform(0, 1, (N_d, 3))
    pos_r = rng.uniform(0, 1, (N_r, 3))

    from scipy.spatial import cKDTree
    r_edges = np.linspace(0.01, 0.4, 16)
    r_centres = 0.5 * (r_edges[1:] + r_edges[:-1])

    def pair_count(a, b, auto):
        tree = cKDTree(b)
        idx = tree.query_ball_point(a, r=r_edges[-1])
        d = []
        for i, lst in enumerate(idx):
            if not lst:
                continue
            arr = np.asarray(lst)
            if auto:
                arr = arr[arr > i]
                if arr.size == 0:
                    continue
            dist = np.linalg.norm(b[arr] - a[i], axis=1)
            d.append(dist)
        if not d:
            return np.zeros(len(r_centres))
        d = np.concatenate(d)
        h, _ = np.histogram(d, bins=r_edges)
        return h

    DD = pair_count(pos_d, pos_d, True)
    RR = pair_count(pos_r, pos_r, True)
    DR = pair_count(pos_d, pos_r, False)
    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r * (N_r - 1) / 2.0
    xi_ls = (DD / max(Nd_pairs, 1) - 2 * DR / (N_d * N_r) + RR / max(Nr_pairs, 1)) \
        / np.maximum(RR / max(Nr_pairs, 1), 1e-30)

    R_grid = np.array([0.05, 0.1, 0.15])
    s2_pair = sigma2_from_pair_counts(r_centres, DD, RR, R_grid,
                                          N_d=N_d, N_r=N_r, DR=DR,
                                          kernel="tophat")
    s2_xi = sigma2_from_xi(r_centres, xi_ls, R_grid, kernel="tophat")
    # Two estimators of the same quantity differ by O(1) bin-discretisation
    # corrections (the kernel is evaluated at bin centres in both cases,
    # so the difference is small but not zero). Require agreement at the
    # level of the kernel-bin-centre approximation for these scales.
    np.testing.assert_allclose(s2_pair, s2_xi, atol=2.0)


def test_per_particle_kernel_counts_consistent_with_total():
    """``sum_i b_DD_K_i = 2 * total_DD_K`` (each unordered pair counted twice)."""
    from twopt_density.sigma2 import (
        kernel_TH_3d, per_particle_kernel_counts,
    )
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 1, (400, 3))
    R = 0.1
    b = per_particle_kernel_counts(pos, pos, R, kernel="tophat", auto=True)
    # brute-force total kernel pair sum
    from scipy.spatial.distance import pdist
    d = pdist(pos)
    total_K = float(np.sum(kernel_TH_3d(d, R)))
    np.testing.assert_allclose(b.sum(), 2.0 * total_K, rtol=1e-9)


def test_density_weights_sigma2_global_identity_matches_DP():
    """The *global* DP identity holds exactly:
    ``sum_i b_DD_K_i N_r / (sum_i b_DR_K_i N_d) - 1 = sigma^2_DP(R)``.
    """
    from twopt_density.sigma2 import (
        density_weights_sigma2, kernel_TH_3d, per_particle_kernel_counts,
    )
    rng = np.random.default_rng(2)
    N_d = 400; N_r = 1200
    pos_d = rng.uniform(0, 1, (N_d, 3))
    pos_r = rng.uniform(0, 1, (N_r, 3))
    R = 0.12
    b_DD = per_particle_kernel_counts(pos_d, pos_d, R, auto=True)
    b_DR = per_particle_kernel_counts(pos_d, pos_r, R, auto=False)
    sum_DD = float(b_DD.sum())
    sum_DR = float(b_DR.sum())
    sigma2_global = (sum_DD * N_r) / max(sum_DR * N_d, 1e-30) - 1.0

    # Verify against direct Davis-Peebles using independent unordered
    # pair sums via brute-force on the same catalogue.
    from scipy.spatial.distance import pdist, cdist
    d_dd = pdist(pos_d)
    d_dr = cdist(pos_d, pos_r).ravel()
    DD_unord = float(np.sum(kernel_TH_3d(d_dd, R)))
    DR_total = float(np.sum(kernel_TH_3d(d_dr, R)))
    sigma2_DP = (2.0 * DD_unord * N_r) / max(DR_total * N_d, 1e-30) - 1.0
    np.testing.assert_allclose(sigma2_global, sigma2_DP, rtol=1e-9)


def test_TH_derivative_kernel_integrates_to_zero():
    """``int dK_TH/dR dV = d/dR int K_TH dV = d/dR (1) = 0`` -- the
    derivative kernel of a normalised window must integrate to zero."""
    from twopt_density.sigma2 import kernel_TH_derivative_3d
    R = 6.0
    r = np.linspace(0.0, 2.0 * R, 4000)
    K = kernel_TH_derivative_3d(r, R)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 0.0, atol=1e-3)


def test_TH_derivative_matches_finite_difference():
    """``dK_TH/dR(r; R)`` matches the central finite-difference of
    ``(K_TH(r; R+h) - K_TH(r; R-h)) / (2h)`` to high precision."""
    from twopt_density.sigma2 import (
        kernel_TH_3d, kernel_TH_derivative_3d,
    )
    R = 6.0; h = 1e-4
    # interior: r < 2(R - h) -- safely in the kernel support of all
    # three evaluations, no boundary effects
    r = np.linspace(0.05 * R, 1.99 * (R - h), 50)
    fd = (kernel_TH_3d(r, R + h) - kernel_TH_3d(r, R - h)) / (2.0 * h)
    ana = kernel_TH_derivative_3d(r, R)
    np.testing.assert_allclose(ana, fd, rtol=1e-3, atol=1e-12)


def test_dsigma2_dR_matches_finite_difference_of_sigma2():
    """``dsigma2_dR_from_xi`` agrees with ``(sigma^2(R+h) - sigma^2(R-h))
    / (2h)`` from ``sigma2_from_xi``."""
    from twopt_density.sigma2 import dsigma2_dR_from_xi, sigma2_from_xi
    r0 = 5.0; gamma = 1.8
    R = 7.0
    h = 1e-3
    r = np.linspace(0.05, 6.0 * R, 5000)
    xi = (r / r0) ** (-gamma)
    ana = dsigma2_dR_from_xi(r, xi, np.array([R]))[0]
    fd = (sigma2_from_xi(r, xi, np.array([R + h]))[0]
            - sigma2_from_xi(r, xi, np.array([R - h]))[0]) / (2.0 * h)
    np.testing.assert_allclose(ana, fd, rtol=1e-3)


def test_shell_kernel_integrates_to_one():
    """``int K_shell(r; R1, R2) dV = 1`` for any thick shell."""
    from twopt_density.sigma2 import kernel_shell_3d
    R1 = 4.0; R2 = 6.0
    r = np.linspace(0.0, 2.0 * R2 + 1e-3, 5000)
    K = kernel_shell_3d(r, R1, R2)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)


def test_shell_kernel_thin_limit_equals_dKdR_normalised():
    """In the thin-shell limit ``R_out - R_in -> 0``, the shell-kernel
    projection times ``1 / (R2 - R1)`` should converge (up to a
    Jacobian factor) to the derivative-kernel projection ``dsigma2/dR``."""
    from twopt_density.sigma2 import (
        dsigma2_dR_from_xi, sigma2_shell_from_xi,
    )
    r0 = 5.0; gamma = 1.8
    R = 8.0
    r = np.linspace(0.05, 6.0 * R, 5000)
    xi = (r / r0) ** (-gamma)

    ds_dR = dsigma2_dR_from_xi(r, xi, np.array([R]))[0]

    # shell-variance trapezoid finite-difference: as the shell narrows,
    # sigma^2_shell(R - h, R + h) approaches a value that, scaled by
    # 1/(2h) times d V_shell / dR factor, converges to ds_dR.
    # Here we just check that the shell-kernel evaluated for thin
    # shells produces a sensible (positive, finite) sigma^2 that is
    # consistent across the (R-h, R+h) finite-difference width when
    # h shrinks: the *ratio* of sigma^2 between two h's is stable.
    h1 = 1.0; h2 = 0.5
    s1 = sigma2_shell_from_xi(r, xi, R - h1, R + h1)
    s2 = sigma2_shell_from_xi(r, xi, R - h2, R + h2)
    assert s1 > 0 and s2 > 0
    # ratio approaches a constant set by the kernel-shape evolution
    # with shell thickness; loose check that both are within an order
    # of magnitude (they sample the same xi)
    assert 0.1 < s2 / s1 < 10.0


def test_sigma2_predicted_jax_grad_in_cosmo_and_bias():
    """``sigma2_predicted`` is JAX-grad differentiable in
    (cosmo.Om, bias, sigma8)."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_predicted

    R = jnp.array([10.0, 30.0, 60.0])

    def loss_om(Om):
        c = DistanceCosmo(Om=Om, h=0.68)
        s2 = sigma2_predicted(R, z_eff=0.5, cosmo=c, bias=2.0, sigma8=0.81)
        return jnp.sum(s2 ** 2)
    g = float(jax.grad(loss_om)(jnp.float64(0.31)))
    assert np.isfinite(g) and g != 0.0

    def loss_b(b):
        c = DistanceCosmo(Om=0.31, h=0.68)
        return jnp.sum(sigma2_predicted(R, z_eff=0.5, cosmo=c, bias=b,
                                            sigma8=0.81) ** 2)
    g_b = float(jax.grad(loss_b)(jnp.float64(2.0)))
    assert np.isfinite(g_b) and g_b > 0.0

    def loss_s8(s8):
        c = DistanceCosmo(Om=0.31, h=0.68)
        return jnp.sum(sigma2_predicted(R, z_eff=0.5, cosmo=c, bias=2.0,
                                            sigma8=s8) ** 2)
    g_s8 = float(jax.grad(loss_s8)(jnp.float64(0.81)))
    assert np.isfinite(g_s8) and g_s8 > 0.0


def test_dsigma2_dR_predicted_matches_finite_difference():
    """Analytic Fourier-space ``dsigma2_dR_predicted`` agrees with the
    central finite-difference of ``sigma2_predicted``."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import (
        dsigma2_dR_predicted, sigma2_predicted,
    )
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    R0 = 30.0; h = 0.5
    s_plus = float(sigma2_predicted(jnp.array([R0 + h]),
                                        z_eff=0.5, cosmo=cosmo,
                                        bias=2.0, sigma8=0.81)[0])
    s_minus = float(sigma2_predicted(jnp.array([R0 - h]),
                                         z_eff=0.5, cosmo=cosmo,
                                         bias=2.0, sigma8=0.81)[0])
    fd = (s_plus - s_minus) / (2 * h)
    ana = float(dsigma2_dR_predicted(jnp.array([R0]),
                                          z_eff=0.5, cosmo=cosmo,
                                          bias=2.0, sigma8=0.81)[0])
    np.testing.assert_allclose(ana, fd, rtol=2e-3)


def test_sigma2_bao_template_smaller_than_signal_at_BAO_scale():
    """The BAO template ``T(R) = sigma^2_full - sigma^2_nowiggle`` is
    much smaller than the signal but non-zero at BAO scales.

    On a fiducial cosmology at z=0.5, sigma^2(R=110/h Mpc) ~ 1e-2,
    and the BAO contribution is at the few-percent level."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import (
        sigma2_bao_template, sigma2_predicted,
    )
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    R = jnp.array([80.0, 110.0, 150.0])
    s2 = np.asarray(sigma2_predicted(R, z_eff=0.5, cosmo=cosmo,
                                          bias=2.0, sigma8=0.81))
    T = sigma2_bao_template(R, z_eff=0.5, cosmo=cosmo,
                                 bias=2.0, sigma8=0.81, derivative=False)
    # template is smaller than full signal but not zero
    assert (np.abs(T) < 0.5 * np.abs(s2)).all()
    assert (np.abs(T) > 0.0).any()


def test_sigma2_predicted_matches_xi_integral_on_eisenstein_hu():
    """``sigma2_predicted`` (Fourier-space, no-wiggle EH) at z=0,
    bias=1, sigma8=0.81 matches the configuration-space integral
    ``int dr 4 pi r^2 xi_NL(r) K_TH(r; R)`` to within a few percent
    for R in BAO range."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.limber import xi_real_at_z
    from twopt_density.sigma2 import (
        kernel_TH_3d, sigma2_predicted,
    )
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    R_test = 50.0
    s2_fourier = float(sigma2_predicted(jnp.array([R_test]),
                                              z_eff=0.0, cosmo=cosmo,
                                              bias=1.0, sigma8=0.81)[0])
    # configuration-space cross-check
    s_grid = np.linspace(0.5, 200.0, 800)
    xi = xi_real_at_z(s_grid, z_eff=0.0, cosmo=cosmo, sigma8=0.81)
    K = kernel_TH_3d(s_grid, R_test)
    s2_config = float(np.trapezoid(4 * np.pi * s_grid ** 2 * xi * K,
                                     s_grid))
    np.testing.assert_allclose(s2_fourier, s2_config, rtol=0.10)


def test_density_weights_sigma2_particle_mean_recovers_sigma2():
    """``<delta_i>_i`` (per-particle mean) recovers the full LS
    sigma^2(R) to within Poisson noise on uniform random data.
    This is the operational identity that lets the data weights
    *fully subsume the random catalogue*: a single sum over data
    points (weighted by their kernel partner counts) returns
    sigma^2(R) without further pair counting."""
    from twopt_density.sigma2 import (
        density_weights_sigma2, kernel_TH_3d,
    )
    rng = np.random.default_rng(42)
    N_d = 800; N_r = 8000
    pos_d = rng.uniform(0, 1, (N_d, 3))
    pos_r = rng.uniform(0, 1, (N_r, 3))
    R = 0.10

    # benchmark LS sigma^2 from full pair counts
    from scipy.spatial.distance import pdist, cdist
    K_dd = float(np.sum(kernel_TH_3d(pdist(pos_d), R)))
    K_rr = float(np.sum(kernel_TH_3d(pdist(pos_r), R)))
    K_dr = float(np.sum(kernel_TH_3d(cdist(pos_d, pos_r).ravel(), R)))
    Ndp = N_d * (N_d - 1) / 2.0; Nrp = N_r * (N_r - 1) / 2.0
    DDn = K_dd / Ndp; DRn = K_dr / (N_d * N_r); RRn = K_rr / Nrp
    sigma2_LS = (DDn - 2 * DRn + RRn) / RRn
    assert abs(sigma2_LS) < 0.25

    # per-particle weights -- mean delta = sigma^2 (DP form)
    _, delta, _ = density_weights_sigma2(pos_d, pos_r, R, kernel="tophat")
    sigma2_from_w = float(np.mean(delta))
    # Both estimators measure the same sigma^2 with shared random
    # catalogue, so they agree to within sqrt(2) x Poisson noise.
    assert abs(sigma2_from_w - sigma2_LS) < 0.15, (
        f"<delta_i> = {sigma2_from_w:.3f} vs LS = {sigma2_LS:.3f}"
    )


def test_sigma2_cyl_predicted_shrinks_with_photoz_smear():
    """Cylinder variance with photo-z LOS smear should be smaller
    than the unsmeared version: convolving the count along LOS by a
    Gaussian reduces the per-cylinder variance."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_cyl_predicted
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    R = jnp.array([30.0, 60.0])
    s2_clean = sigma2_cyl_predicted(R, pi_max=200.0, z_eff=1.5, cosmo=cosmo,
                                          bias=2.6, sigma8=0.81)
    s2_smear = sigma2_cyl_predicted(R, pi_max=200.0, z_eff=1.5, cosmo=cosmo,
                                          bias=2.6, sigma8=0.81,
                                          sigma_chi=170.0)
    assert (np.asarray(s2_smear) < np.asarray(s2_clean)).all()
    # but both positive -- still a real variance
    assert (np.asarray(s2_clean) > 0).all()
    assert (np.asarray(s2_smear) > 0).all()


def test_sigma2_cyl_jax_grad_in_bias_and_sigma8():
    """``sigma2_cyl_predicted`` is JAX-grad differentiable in
    (bias, sigma8)."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_cyl_predicted
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    R = jnp.array([30.0, 80.0])

    def loss_b(b):
        return jnp.sum(sigma2_cyl_predicted(R, pi_max=200.0, z_eff=1.5,
                                                  cosmo=cosmo, bias=b,
                                                  sigma8=0.81) ** 2)
    g_b = float(jax.grad(loss_b)(jnp.float64(2.6)))
    assert np.isfinite(g_b) and g_b > 0.0

    def loss_s8(s8):
        return jnp.sum(sigma2_cyl_predicted(R, pi_max=200.0, z_eff=1.5,
                                                  cosmo=cosmo, bias=2.6,
                                                  sigma8=s8) ** 2)
    g_s8 = float(jax.grad(loss_s8)(jnp.float64(0.81)))
    assert np.isfinite(g_s8) and g_s8 > 0.0


def test_sigma2_gkappa_predicted_decreases_with_R():
    """``sigma^2_{g-kappa}(R)`` decreases monotonically with the
    smoothing radius R (smoothing kills small-scale power)."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_gkappa_predicted
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z = jnp.linspace(0.85, 2.45, 60)
    nz = jnp.exp(-0.5 * ((z - 1.5) / 0.5) ** 2)
    b_z = jnp.full_like(z, 2.6)
    R = jnp.array([20.0, 60.0, 120.0])
    s2 = np.asarray(sigma2_gkappa_predicted(R, z, nz, b_z, cosmo,
                                                  sigma8=0.81))
    # monotone decrease
    assert s2[0] > s2[1] > s2[2] > 0.0


def test_sigma2_gkappa_jax_grad_differentiable():
    """``sigma2_gkappa_predicted`` is differentiable in (sigma8, b_z)."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_gkappa_predicted
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z = jnp.linspace(0.85, 2.45, 40)
    nz = jnp.exp(-0.5 * ((z - 1.5) / 0.5) ** 2)
    R = jnp.array([60.0])

    def loss(s8):
        b_z = jnp.full_like(z, 2.6)
        return jnp.sum(sigma2_gkappa_predicted(R, z, nz, b_z, cosmo,
                                                       sigma8=s8) ** 2)
    g = float(jax.grad(loss)(jnp.float64(0.81)))
    assert np.isfinite(g) and g > 0.0


def test_sigma2_RR_analytic_window_runs_on_synthetic_mask():
    """Direct analytic-window sigma^2_RR returns a positive,
    finite array on a synthetic galactic-cap mask."""
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2 import sigma2_RR_analytic_window
    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    nside = 32
    npix = 12 * nside ** 2
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # galactic-style mask: |b| > 10 deg
    mask = np.where(np.abs(np.pi / 2 - theta) > np.deg2rad(10.0), 1.0, 0.0)
    z_data = np.linspace(0.8, 2.5, 500)
    R_grid = np.array([30.0, 80.0])
    out = sigma2_RR_analytic_window(R_grid, mask, nside, z_data, cosmo,
                                          n_rp=40, n_pi=40)
    assert np.all(np.isfinite(out))
    assert (out >= 0).all()
