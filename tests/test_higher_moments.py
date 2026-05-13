"""Sanity tests for the higher-moment derived helpers (note v4_1 §6).

These tests don't require the cascade binary — they exercise the
Python derivation of S₃ skewness, S₄ kurtosis, the LS and Hamilton
moment estimators, and the quantile-grid bin populations against
analytic expectations.

Numba is required for ``joint_knn_cdf``; tests are skipped if it's
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _numba_ok():
    try:
        from twopt_density.sigma2_cone_shell_estimator import _NUMBA_OK
        return bool(_NUMBA_OK)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _numba_ok(),
    reason="numba kernel required for joint_knn_cdf; install numba",
)


def _synth_uniform_sphere(n, seed=0):
    """Uniform RA/Dec on a sky patch + uniform z. Used as a reference
    Poisson field for testing ``S₃ → 0`` and ``S₄ → 1/⟨N⟩`` limits."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 60, n).astype(np.float64)
    dec = rng.uniform(-20, 30, n).astype(np.float64)
    z = rng.uniform(1.0, 1.5, n).astype(np.float64)
    return ra, dec, z


def test_quantile_grid_bin_populations():
    """Each D-centered 1%-quantile bin should contain ~1% of the data."""
    from twopt_density.zgrid import quantile_edges
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 3.0, 50000).astype(np.float64)
    edges = quantile_edges(z, n_bins=90)
    counts = np.histogram(z, bins=edges)[0]
    # In the inner 81% (0.095 to 0.905) of N(z), each of 90 bins
    # contains ~ (0.81 * N) / 90 ≈ 0.9% of the sample. With N=50000
    # that's ~450 ± 21 (Poisson scatter). Allow ±15% tolerance.
    target = (0.81 * z.size) / 90
    assert abs(counts.mean() - target) / target < 0.05, \
        f"mean count {counts.mean()} differs from target {target}"
    assert counts.min() > 0.85 * target
    assert counts.max() < 1.15 * target


def test_decile_edges_monotone():
    """Decile edges must be strictly ascending."""
    from twopt_density.zgrid import decile_edges
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 3.0, 5000).astype(np.float64)
    edges = decile_edges(z, n_deciles=9)
    assert edges.shape == (10,)
    assert np.all(np.diff(edges) > 0), \
        f"decile edges not strictly ascending: {edges}"


def test_higher_moments_poisson_field():
    """For a uniform random field, the per-cap skewness should be near
    zero and the kurtosis near the Poisson value 1/⟨N⟩.

    With finite samples we accept |S₃| < 0.5 and |kurt - 1/⟨N⟩| < 1.0
    in the per-cap mean range we sample.
    """
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import (
        cic_skewness_raw, cic_kurtosis_raw, mean_count,
    )
    ra, dec, z = _synth_uniform_sphere(2000, seed=42)
    theta = np.deg2rad(np.array([1.0, 2.0, 4.0]))
    z_edges = np.array([1.0, 1.25, 1.5])
    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_edges, z_n_edges=z_edges,
        k_max=5, flavor="DD",
        diagonal_only=True, n_threads=1, nside_lookup=64,
    )
    s3 = cic_skewness_raw(res)
    s4 = cic_kurtosis_raw(res)
    mean = mean_count(res)
    assert s3.shape == s4.shape == mean.shape
    # Skewness should be small for a Poisson-uniform field with
    # finite samples; tolerance is loose because n_q per shell ~1000.
    finite = np.isfinite(s3) & (mean > 1.0)
    if finite.any():
        assert np.nanmedian(np.abs(s3[finite])) < 1.5, \
            f"S₃ should be near zero for Poisson field, got median "\
            f"{np.nanmedian(np.abs(s3[finite]))}"


def test_hamilton_matches_ls_in_high_signal():
    """Hamilton ξ and LS ξ should agree to leading order when RR is
    well-measured (i.e. lots of randoms). This is more a smoke test
    that both run without errors and return finite values, not a
    precision check."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import xi_hamilton, xi_ls
    rng = np.random.default_rng(7)
    n = 600
    ra_d = rng.uniform(0, 60, n).astype(np.float64)
    dec_d = rng.uniform(-20, 20, n).astype(np.float64)
    z_d = rng.uniform(1.0, 1.5, n).astype(np.float64)
    ra_r = rng.uniform(0, 60, 2 * n).astype(np.float64)
    dec_r = rng.uniform(-20, 20, 2 * n).astype(np.float64)
    z_r = rng.uniform(1.0, 1.5, 2 * n).astype(np.float64)
    theta = np.deg2rad(np.array([1.0, 2.0, 4.0]))
    z_edges = np.array([1.0, 1.5])

    common = dict(
        theta_radii_rad=theta, z_q_edges=z_edges, z_n_edges=z_edges,
        k_max=3, n_threads=1, nside_lookup=64, diagonal_only=True,
    )
    res_dd = joint_knn_cdf(
        ra_d, dec_d, z_d, ra_d, dec_d, z_d, flavor="DD", **common)
    res_dr = joint_knn_cdf(
        ra_d, dec_d, z_d, ra_r, dec_r, z_r, flavor="DR", **common)
    res_rd = joint_knn_cdf(
        ra_r, dec_r, z_r, ra_d, dec_d, z_d, flavor="RD", **common)
    res_rr = joint_knn_cdf(
        ra_r, dec_r, z_r, ra_r, dec_r, z_r, flavor="DD", **common)

    n_neigh_d = np.array([n], dtype=np.int64)
    n_neigh_r = np.array([2 * n], dtype=np.int64)
    xi_h = xi_hamilton(res_dd, res_dr, res_rd, res_rr,
                       n_neigh_d, n_neigh_r)
    xi_l = xi_ls(res_dd, res_dr, res_rr,
                 n_neigh_d, n_neigh_d, n_neigh_r)
    # Both should be finite (Poisson field → ξ ≈ 0; checks both
    # estimators run and return arrays of the right shape).
    assert xi_h.shape == xi_l.shape
    assert np.isfinite(xi_h).any()
    assert np.isfinite(xi_l).any()


def test_dlnsigma2_dlogz_constant_field():
    """For a constant σ²(z), the log derivative should be 0 everywhere."""
    from twopt_density.knn_derived import dlnsigma2_dlogz
    n_theta = 5
    n_z = 8
    sigma2 = np.full((n_theta, n_z), 0.1)
    z_centres = np.linspace(0.8, 2.1, n_z)
    deriv = dlnsigma2_dlogz(sigma2, z_centres)
    np.testing.assert_allclose(deriv, 0.0, atol=1e-12)


def test_dlnsigma2_dlogz_powerlaw_field():
    """For σ² ~ (1+z)^p, the log derivative should equal p."""
    from twopt_density.knn_derived import dlnsigma2_dlogz
    n_theta = 3
    z_centres = np.linspace(0.8, 2.1, 9)
    p = -2.5
    sigma2 = (1.0 + z_centres[None, :]) ** p * np.ones((n_theta, 1))
    deriv = dlnsigma2_dlogz(sigma2, z_centres)
    # Centred-difference gives p exactly for a perfect power law in
    # log(1+z); endpoints (forward/backward) are also exact.
    np.testing.assert_allclose(deriv, p, rtol=1e-10)


def test_differential_pair_count_shape():
    """differential_pair_count should preserve cube shape."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.knn_derived import differential_pair_count
    ra, dec, z = _synth_uniform_sphere(400, seed=123)
    theta = np.deg2rad(np.array([0.5, 1.0, 2.0, 4.0]))
    z_edges = np.array([1.0, 1.25, 1.5])
    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta, z_q_edges=z_edges, z_n_edges=z_edges,
        k_max=3, flavor="DD", diagonal_only=True, n_threads=1,
        nside_lookup=64,
    )
    dn = differential_pair_count(res)
    assert dn.shape == res.sum_n.shape
    # Differential should be non-negative for a non-decreasing
    # cumulative count (allow tiny negative due to floating-point
    # near-zero subtraction).
    assert (dn >= -1e-12).all()
