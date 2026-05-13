"""Tests for the kNN-CDF derived-statistics layer.

These tests pin:

  1. ``sigma2_clust`` from the knn_cdf primitive (with each query at a
     single cap centre) matches ``sigma2_estimate_cone_shell`` from
     the legacy estimator on the same caps to floating-point
     precision.
  2. ``mean_count`` matches ``cic_moments(p=1)``.
  3. ``cic_pmf`` integrates to 1 along the ``k`` axis (with the
     ``k_max`` tail accounted for).
  4. ``xi_dp`` and ``xi_ls`` reduce to constants on a uniform Poisson
     mock with matched DD/RD/RR construction.
"""

from __future__ import annotations

import numpy as np
import pytest


def _full_sky_poisson(n_g: int, z_min: float = 0.5, z_max: float = 2.0,
                      seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n_g)
    sin_dec = rng.uniform(-1, 1, n_g)
    ra = np.degrees(phi)
    dec = np.degrees(np.arcsin(sin_dec))
    z = rng.uniform(z_min, z_max, n_g)
    return ra, dec, z


def test_sigma2_clust_matches_legacy_estimator():
    """knn_cdf-derived sigma^2_clust must equal the legacy
    ``sigma^2_estimate_cone_shell`` on the same set of cap centres,
    same neighbor catalog, same theta and z bins. ``ddof`` differs
    (legacy uses ddof=1; the moment-based form uses N-divisor) so we
    require agreement up to the (n_centres-1)/n_centres factor."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import sigma2_clust
    from twopt_density.sigma2_cone_shell_estimator import (
        cone_shell_counts, sigma2_estimate_cone_shell,
    )

    rng = np.random.default_rng(7)
    n_g = 4000
    ra, dec, z = _full_sky_poisson(n_g, seed=11)

    # cluster injection so sigma^2 is non-trivial
    n_c = 500
    ra_cc = rng.uniform(0, 360, 30)
    dec_cc = np.degrees(np.arcsin(rng.uniform(-1, 1, 30)))
    a = rng.integers(0, 30, n_c)
    ra = np.concatenate([ra, (ra_cc[a] + 1.0 * rng.standard_normal(n_c)) % 360])
    dec = np.concatenate([dec, np.clip(dec_cc[a]
                                       + 1.0 * rng.standard_normal(n_c),
                                       -89, 89)])
    z = np.concatenate([z, rng.uniform(1.0, 1.5, n_c)])

    nside = 16
    mask = np.ones(12 * nside ** 2, dtype=np.float64)
    theta_max = np.deg2rad(3.0)
    from twopt_density.sigma2_cone_shell_estimator import cap_centre_grid
    ra_centres, dec_centres, _ = cap_centre_grid(
        mask, nside_centres=nside, theta_max_rad=theta_max,
        edge_buffer_frac=1.0,
    )

    theta_radii = np.deg2rad(np.array([1.5, 2.5, 3.0]))
    z_n_edges = np.array([0.5, 1.0, 1.5, 2.0])

    # Legacy: per-centre cube + Var/<N>^2 - 1/<N>
    N, _ = cone_shell_counts(
        ra, dec, z, theta_radii, z_n_edges, ra_centres, dec_centres,
        nside_lookup=128, n_threads=1,
    )
    s2_legacy = sigma2_estimate_cone_shell(N)        # (n_theta, n_z_n) ddof=1

    # New: knn_cdf with queries = cap centres (z_q dummy collapsed to 1 shell)
    z_q_dummy = np.full(ra_centres.size, 1.0)
    z_q_edges = np.array([0.5, 2.0])
    res = joint_knn_cdf(
        ra_centres, dec_centres, z_q_dummy, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=0, flavor="RD",  # different catalogs, no self-exclusion
        nside_lookup=128, n_threads=1,
    )
    s2_new = sigma2_clust(res)[:, 0, :]              # squeeze z_q axis

    # Legacy uses ddof=1 (Bessel); the moment form uses N-divisor.
    # Convert: var_ddof1 = var_N * N / (N-1)
    n_centres = N.shape[0]
    s2_new_corrected = s2_new + (
        s2_new + 1.0 / N.mean(axis=0)
    ) * (1.0 / (n_centres - 1))
    # The correction above expands Var_N -> Var_N * N/(N-1):
    #   sigma^2 = Var/<N>^2 - 1/<N>
    #   delta = (Var * 1/(N-1)) / <N>^2
    #         = (sigma^2 + 1/<N>) / (N-1)
    np.testing.assert_allclose(s2_new_corrected, s2_legacy, rtol=1e-10,
                               atol=1e-12)


def test_mean_count_equals_p1_moment():
    """``mean_count`` and ``cic_moments(p=1)`` agree by construction
    when the kNN ladder is populated past the per-cap support."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import mean_count, cic_moments

    ra, dec, z = _full_sky_poisson(80, seed=2)
    theta_radii = np.deg2rad(np.array([2.0, 5.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 2.0])

    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=200, flavor="DD",  # large enough to fully resolve PMF
        nside_lookup=64, n_threads=1,
    )
    np.testing.assert_allclose(mean_count(res), cic_moments(res, 1),
                               rtol=1e-12, atol=1e-14)


def test_cic_pmf_sums_to_one():
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import cic_pmf

    ra, dec, z = _full_sky_poisson(80, seed=4)
    theta_radii = np.deg2rad(np.array([2.0, 5.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 2.0])
    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=200, flavor="DD",
        nside_lookup=64, n_threads=1,
    )
    pmf = cic_pmf(res)
    np.testing.assert_allclose(pmf.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-14)


def test_xi_dp_zero_on_matched_poisson():
    """On Poisson data with matched random *queries* (same density),
    xi_DP = nbar^DD/nbar^RD - 1 ~ 0 within sqrt(N_pair) shot noise."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import xi_dp

    n_d = 2500
    ra_d, dec_d, z_d = _full_sky_poisson(n_d, seed=21)
    n_r = 5 * n_d
    ra_r, dec_r, z_r = _full_sky_poisson(n_r, seed=22)

    theta_radii = np.deg2rad(np.array([5.0, 10.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 2.0])

    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=0, nside_lookup=128, n_threads=1,
    )
    # DD: data queries on data neighbors (self-excluded)
    res_dd = joint_knn_cdf(ra_d, dec_d, z_d, ra_d, dec_d, z_d,
                            flavor="DD", **common)
    # RD: random queries on data neighbors
    res_rd = joint_knn_cdf(ra_r, dec_r, z_r, ra_d, dec_d, z_d,
                            flavor="RD", **common)
    xi = xi_dp(res_dd, res_rd)
    # On Poisson data, xi should be consistent with 0 within Poisson
    # noise. Use the mean count to bound the tolerance: SE ~ 1/sqrt(<N>*N_q)
    nbar_dd = res_dd.sum_n / np.maximum(res_dd.N_q[None, :, None], 1).astype(float)
    se = 1.0 / np.sqrt(np.maximum(nbar_dd * res_dd.N_q[None, :, None], 1))
    assert (np.abs(xi) < 8.0 * se).all(), (
        f"|xi_dp|={np.abs(xi).max():.3g} exceeds 8 SE={8.0*se.max():.3g}"
    )


def test_xi_ls_cross_reduces_to_xi_ls_in_auto_limit():
    """When the cross-catalog x = y, the asymmetric LS cross estimator
    must reduce to the symmetric auto-correlation LS (because
    DR_xy == RD_xy by construction)."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import xi_ls, xi_ls_cross

    n_d = 1500
    ra_d, dec_d, z_d = _full_sky_poisson(n_d, seed=31)
    n_r = n_d
    ra_r, dec_r, z_r = _full_sky_poisson(n_r, seed=32)

    theta_radii = np.deg2rad(np.array([3.0, 6.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 2.0])

    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=0, nside_lookup=64, n_threads=1,
    )
    res_dd = joint_knn_cdf(ra_d, dec_d, z_d, ra_d, dec_d, z_d,
                            flavor="DD", **common)
    res_dr = joint_knn_cdf(ra_d, dec_d, z_d, ra_r, dec_r, z_r,
                            flavor="RD", **common)
    res_rd = joint_knn_cdf(ra_r, dec_r, z_r, ra_d, dec_d, z_d,
                            flavor="RD", **common)
    res_rr = joint_knn_cdf(ra_r, dec_r, z_r, ra_r, dec_r, z_r,
                            flavor="DD", **common)

    n_d_per_zn = np.array([n_d], dtype=np.int64)
    n_r_per_zn = np.array([n_r], dtype=np.int64)

    # auto: feed RD cube twice (the auto path expects mu_DR == mu_RD)
    xi_auto = xi_ls(res_dd, res_rd, res_rr,
                    n_d_per_zn, n_d_per_zn, n_r_per_zn)
    # cross: feed both DR (data x random) and RD (random x data) explicitly
    xi_cross = xi_ls_cross(res_dd, res_dr, res_rd, res_rr,
                           n_d_per_zn, n_r_per_zn)
    # NOTE: xi_auto uses res_rd as the DR proxy (assuming DR==RD). The
    # cross-form computes (DD - DR - RD + RR)/RR explicitly. They agree
    # exactly when DR == RD per query, which holds here only in the
    # limit of large N -- so we test agreement to within Poisson scatter.
    diff = np.abs(xi_cross - xi_auto)
    # crude noise floor: Poisson SE on per-cap counts, scaled to xi
    nbar_rr = res_rr.sum_n / np.maximum(res_rr.N_q[None, :, None], 1).astype(float)
    se_floor = 5.0 / np.sqrt(np.maximum(nbar_rr * res_rr.N_q[None, :, None], 1))
    assert (diff < se_floor + 1e-2).all(), (
        f"xi_ls_cross deviates from xi_ls by {diff.max():.3g} > "
        f"{se_floor.max():.3g}"
    )


def test_jackknife_cov_smoke():
    """Round-trip: build a (n_regions, n_theta, n_z_n) sample stack,
    feed to jackknife_cov, get back a positive-semidefinite matrix."""
    pytest.importorskip("numba")
    from twopt_density.knn_derived import jackknife_cov

    rng = np.random.default_rng(0)
    n_regions = 12
    samples = rng.normal(size=(n_regions, 3, 2))
    mean, cov = jackknife_cov(samples)
    assert mean.shape == (3, 2)
    assert cov.shape == (6, 6)
    # PSD: smallest eigenvalue >= 0 (allow numerical slop)
    assert np.linalg.eigvalsh(cov).min() > -1e-12
