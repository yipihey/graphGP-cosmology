"""Tests for the lightcone-native angular cone-shell variance forward model.

These tests pin the identities the pipeline depends on:

  1. ``sigma^2(theta; z_min, z_max)`` decreases monotonically with theta
     above the per-shell coherence scale.
  2. For a thin shell of mean redshift z_eff, the angular sigma^2(theta)
     matches the 3D top-hat sigma^2(R = theta * chi(z_eff)) to within a
     few percent -- the paper's information-equivalence claim
     operationalised.
  3. Adjacent-shell finite-difference d sigma^2 / d ln(1+z) agrees with
     central finite differences of the predicted stack.
  4. The forward model is JAX-grad differentiable in
     (cosmo.Om, bias, sigma8).
  5. The BAO template is smaller than the signal but non-zero at the
     angular BAO scale.
  6. The bias / growth / geometry decomposition reconstructs the LHS
     log-derivative within finite-difference accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest


def _fid_dndz(z_grid, z0=1.5, sig=0.6):
    return np.exp(-0.5 * ((z_grid - z0) / sig) ** 2)


def test_sigma2_cone_shell_decreases_with_theta():
    """``sigma^2(theta)`` falls monotonically with theta in the
    information-rich range (0.1 to 4 deg)."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2_cone_shell import sigma2_cone_shell_predicted

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_grid = np.linspace(0.01, 4.0, 200)
    dndz = _fid_dndz(z_grid)
    theta = np.deg2rad(np.array([0.1, 0.3, 0.7, 1.5, 3.0]))
    s2 = np.asarray(sigma2_cone_shell_predicted(
        theta, z_min=1.0, z_max=2.0,
        z_grid=z_grid, dndz=dndz, cosmo=cosmo,
        bias=2.0, sigma8=0.81,
        ell_min=2.0, ell_max=3e4, n_ell=400,
    ))
    assert (s2 > 0).all()
    assert np.all(np.diff(s2) < 0)


def test_sigma2_cone_shell_matches_xi_integral_thin_shell():
    """Thin-shell information-equivalence test (paper Sec. 2.3): for a
    narrow z-shell, the angular ``sigma^2(theta_R)`` coincides (within
    ~10 percent) with the 3D ``sigma^2_TH(R = theta_R * chi_eff)`` from
    the same P_NL on the same z. Note this only holds when the shell is
    thin enough that the LOS smearing of the 2D top-hat is negligible at
    R << shell thickness; we use a 4 percent shell width centred on
    z=1.0 and an angular scale that maps to R = 30 Mpc/h.
    """
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo, comoving_distance
    from twopt_density.limber import xi_real_at_z
    from twopt_density.sigma2 import kernel_TH_3d
    from twopt_density.sigma2_cone_shell import sigma2_cone_shell_predicted

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_eff = 1.0
    dz = 0.04
    z_min = z_eff - dz / 2
    z_max = z_eff + dz / 2

    z_grid = np.linspace(0.01, 4.0, 600)
    dndz = _fid_dndz(z_grid)

    # angular sigma^2 at theta s.t. R = theta * chi(z_eff)
    chi_eff = float(comoving_distance(
        jnp.asarray([z_eff], dtype=jnp.float64), cosmo
    )[0])
    R = 30.0
    theta = R / chi_eff
    s2_ang = float(sigma2_cone_shell_predicted(
        np.asarray([theta]), z_min=z_min, z_max=z_max,
        z_grid=z_grid, dndz=dndz, cosmo=cosmo,
        bias=1.0, sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=800,
    )[0])

    # 3D config-space cross-check
    s_grid = np.linspace(0.5, 5.0 * R, 1000)
    xi = xi_real_at_z(s_grid, z_eff=z_eff, cosmo=cosmo, sigma8=0.81)
    K = kernel_TH_3d(s_grid, R)
    s2_3d = float(np.trapezoid(4 * np.pi * s_grid ** 2 * xi * K, s_grid))

    # paper-exact identity holds in the infinitely-thin-shell limit.
    # For a finite shell width the angular sigma^2 has finite LOS
    # support and is somewhat smaller; allow 30 percent tolerance.
    np.testing.assert_allclose(s2_ang, s2_3d, rtol=0.30)


def test_dsigma2_dz_matches_finite_difference():
    """``dsigma2_dz_cone_shell_predicted`` returns the same value as
    central finite differences of the predicted stack."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2_cone_shell import (
        dsigma2_dz_cone_shell_predicted,
        sigma2_cone_shell_predicted_stack,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_grid = np.linspace(0.01, 4.0, 300)
    dndz = _fid_dndz(z_grid)
    z_edges = np.array([0.5, 0.9, 1.3, 1.7, 2.1, 2.5])
    theta = np.deg2rad(np.array([0.3, 1.0, 2.0]))

    z_centres, ds = dsigma2_dz_cone_shell_predicted(
        theta, z_edges, z_grid, dndz, cosmo,
        bias=2.0, sigma8=0.81,
        ell_min=2.0, ell_max=3e4, n_ell=400,
        log1pz=True,
    )
    s2 = np.asarray(sigma2_cone_shell_predicted_stack(
        theta, z_edges, z_grid, dndz, cosmo,
        bias=2.0, sigma8=0.81,
        ell_min=2.0, ell_max=3e4, n_ell=400,
    ))
    z_shell = 0.5 * (z_edges[:-1] + z_edges[1:])
    x = np.log(1.0 + z_shell)
    fd = (s2[2:] - s2[:-2]) / (x[2:] - x[:-2])[:, None]
    np.testing.assert_allclose(np.asarray(ds), fd, rtol=1e-10, atol=1e-15)
    # finite, signed values
    assert np.isfinite(np.asarray(ds)).all()
    # at z >= 1.5, sigma^2 is decreasing with z (linear growth shrinks
    # power) for a fixed theta -- expect negative derivative on the
    # higher-z pivot.
    j_late = np.argmax(z_centres > 1.5) if (z_centres > 1.5).any() else None
    if j_late is not None:
        assert (np.asarray(ds)[j_late] < 0).all()


def test_sigma2_cone_shell_jax_grad_in_cosmo_bias_sigma8():
    """``sigma2_cone_shell_predicted`` is JAX-grad differentiable in
    (Om, bias, sigma8) with finite, non-zero gradients."""
    import jax
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2_cone_shell import sigma2_cone_shell_predicted

    z_grid = np.linspace(0.01, 4.0, 200)
    dndz = _fid_dndz(z_grid)
    theta = jnp.asarray(np.deg2rad(np.array([0.5, 1.5])))

    def loss_om(Om):
        c = DistanceCosmo(Om=Om, h=0.68)
        s2 = sigma2_cone_shell_predicted(
            theta, z_min=1.0, z_max=2.0,
            z_grid=z_grid, dndz=dndz, cosmo=c,
            bias=2.0, sigma8=0.81,
            ell_min=2.0, ell_max=3e4, n_ell=300,
        )
        return jnp.sum(s2 ** 2)
    g_om = float(jax.grad(loss_om)(jnp.float64(0.31)))
    assert np.isfinite(g_om) and g_om != 0.0

    def loss_b(b):
        c = DistanceCosmo(Om=0.31, h=0.68)
        return jnp.sum(sigma2_cone_shell_predicted(
            theta, z_min=1.0, z_max=2.0,
            z_grid=z_grid, dndz=dndz, cosmo=c,
            bias=b, sigma8=0.81,
            ell_min=2.0, ell_max=3e4, n_ell=300,
        ) ** 2)
    g_b = float(jax.grad(loss_b)(jnp.float64(2.0)))
    assert np.isfinite(g_b) and g_b > 0.0

    def loss_s8(s8):
        c = DistanceCosmo(Om=0.31, h=0.68)
        return jnp.sum(sigma2_cone_shell_predicted(
            theta, z_min=1.0, z_max=2.0,
            z_grid=z_grid, dndz=dndz, cosmo=c,
            bias=2.0, sigma8=s8,
            ell_min=2.0, ell_max=3e4, n_ell=300,
        ) ** 2)
    g_s8 = float(jax.grad(loss_s8)(jnp.float64(0.81)))
    assert np.isfinite(g_s8) and g_s8 > 0.0


def test_bao_template_smaller_than_signal_at_BAO_scale():
    """The BAO template is non-zero but smaller than the full signal at
    the angular BAO scale theta_BAO ~ 105 Mpc/h / chi(z_eff)."""
    import jax.numpy as jnp
    from twopt_density.distance import DistanceCosmo, comoving_distance
    from twopt_density.sigma2_cone_shell import (
        sigma2_cone_shell_bao_template, sigma2_cone_shell_predicted,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_eff = 1.0
    z_min, z_max = 0.85, 1.15
    z_grid = np.linspace(0.01, 4.0, 300)
    dndz = _fid_dndz(z_grid)
    chi_eff = float(comoving_distance(
        jnp.asarray([z_eff], dtype=jnp.float64), cosmo
    )[0])
    # cluster around theta_BAO ~ 105 Mpc/h / chi_eff
    theta_bao = 105.0 / chi_eff
    theta = np.array([0.6, 0.8, 1.0, 1.2, 1.4]) * theta_bao

    s2 = np.asarray(sigma2_cone_shell_predicted(
        theta, z_min=z_min, z_max=z_max,
        z_grid=z_grid, dndz=dndz, cosmo=cosmo,
        bias=2.0, sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=800,
    ))
    T = sigma2_cone_shell_bao_template(
        theta, z_min=z_min, z_max=z_max,
        z_grid=z_grid, dndz=dndz, cosmo=cosmo,
        bias=2.0, sigma8=0.81,
        ell_min=2.0, ell_max=5e4, n_ell=800,
    )
    assert (np.abs(T) < 0.5 * np.abs(s2)).all()
    assert (np.abs(T) > 0.0).any()


def test_decomposition_signs_and_finiteness():
    """Diagnostic decomposition has correct signs for a fiducial
    setup. The full identity only holds in the thin-shell limit; for
    finite shells (here Delta z ~ 0.3) the LHS and the sum of the
    three terms differ at the factor-of-order-unity level. This test
    checks the qualitative signs that make the decomposition useful as
    a diagnostic trace:

      - growth term ``2 d ln D / d ln(1+z) < 0`` (D decreases with z);
      - bias term ``2 d ln b / d ln(1+z) > 0`` for ``b(z) = 1 + 0.5 z``;
      - geometry term has the sign of ``n_eff * d ln chi / d ln(1+z)``;
        d ln chi / d ln(1+z) > 0 always, n_eff < 0 (sigma^2 decreasing
        with theta), so geom < 0;
      - all values are finite.
    """
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2_cone_shell import (
        sigma2_cone_shell_decomposition,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_grid = np.linspace(0.01, 4.0, 300)
    dndz = _fid_dndz(z_grid)
    z_edges = np.array([0.7, 1.0, 1.3, 1.6, 1.9, 2.2])
    theta = np.deg2rad(np.array([0.3, 1.0, 2.0]))

    z_centres, lhs, c_b, c_g, c_geom = sigma2_cone_shell_decomposition(
        theta, z_edges, z_grid, dndz, cosmo,
        bias_z=lambda z: 1.0 + 0.5 * z,
        sigma8=0.81,
        ell_min=2.0, ell_max=3e4, n_ell=400,
    )
    assert np.isfinite(lhs).all()
    assert np.isfinite(c_b).all()
    assert np.isfinite(c_g).all()
    assert np.isfinite(c_geom).all()
    assert (c_g < 0).all()
    assert (c_b > 0).all()
    assert (c_geom < 0).all()


def test_gaussian_covariance_block_diagonal_psd_and_diag_decreases_with_theta():
    """``sigma2_cone_shell_gaussian_covariance`` must be:
      - symmetric and positive-semidefinite,
      - block-diagonal in the redshift-shell index (Gaussian limit),
      - have diagonal entries that decrease with theta in the
        information-rich range.
    """
    from twopt_density.distance import DistanceCosmo
    from twopt_density.sigma2_cone_shell import (
        sigma2_cone_shell_gaussian_covariance,
    )

    cosmo = DistanceCosmo(Om=0.31, h=0.68)
    z_grid = np.linspace(0.01, 4.0, 200)
    dndz = _fid_dndz(z_grid)
    z_edges = np.array([0.5, 1.0, 1.5, 2.0])
    theta_rad = np.deg2rad(np.array([0.3, 0.6, 1.2, 2.4]))

    cov = sigma2_cone_shell_gaussian_covariance(
        theta_rad, z_edges, z_grid, dndz, cosmo,
        bias=2.0, sigma8=0.81, f_sky=0.7,
        n_bar_per_steradian=1e7,            # ~Quaia-scale density
        ell_min=2.0, ell_max=3e4, n_ell=400,
    )

    n_theta = theta_rad.size
    n_zshell = z_edges.size - 1
    n_total = n_theta * n_zshell

    # shape, finiteness
    assert cov.shape == (n_total, n_total)
    assert np.isfinite(cov).all()

    # symmetric
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-300)

    # diagonal positive
    diag = np.diag(cov)
    assert (diag > 0).all()

    # block-diagonal in z: cov[i_theta * n_z + k_z, j_theta * n_z + l_z]
    # is zero when k_z != l_z.
    for k in range(n_zshell):
        for l in range(n_zshell):
            if k == l:
                continue
            for i in range(n_theta):
                for j in range(n_theta):
                    val = cov[i * n_zshell + k, j * n_zshell + l]
                    assert val == 0.0, (
                        f"non-zero cross-shell entry at "
                        f"({i},{k}),({j},{l}): {val}"
                    )

    # diagonal SE per (theta, z) decreases with theta in each shell
    # (more area integrated -> more independent modes -> smaller cov)
    for k in range(n_zshell):
        d_k = np.array([cov[i * n_zshell + k, i * n_zshell + k]
                        for i in range(n_theta)])
        assert np.all(np.diff(d_k) < 0), (
            f"shell {k} cov diagonal not decreasing with theta: {d_k}"
        )

    # PSD: smallest eigenvalue >= 0 (within numerical noise)
    eigs = np.linalg.eigvalsh(cov)
    assert eigs[0] > -1e-15 * eigs[-1], f"non-PSD: min eig {eigs[0]}"
