"""Tests for the joint angular kNN-CDF primitive (twopt_density.knn_cdf).

The primitive is the data vector of ``lightcone_native_v3.pdf``. These
tests pin:

  1. ``H_geq_k`` matches a brute-force reference on a small toy
     catalog (exact integer counts).
  2. ``sum_n`` matches the per-cap counts produced by
     ``cone_shell_counts`` to floating-point precision (same kernel,
     same input).
  3. DD-flavor self-exclusion removes exactly N_q from the smallest
     theta bin's accumulator (one self-pair per query).
  4. The kNN ladder is monotonically non-increasing in k.
  5. Single-pass jackknife: per-region cubes sum to the full cubes.
  6. ``k_max=0`` short-circuits the ladder aggregation but still
     populates the moment cubes.
"""

from __future__ import annotations

import numpy as np
import pytest


def _toy_catalog(seed: int = 0, n: int = 80, full_sky: bool = True):
    rng = np.random.default_rng(seed)
    if full_sky:
        phi = rng.uniform(0, 2 * np.pi, n)
        sin_dec = rng.uniform(-1, 1, n)
        ra = np.degrees(phi)
        dec = np.degrees(np.arcsin(sin_dec))
    else:
        ra = rng.uniform(0, 90, n)
        dec = rng.uniform(0, 60, n)
    z = rng.uniform(0.5, 2.0, n)
    return ra, dec, z


def _brute_h_geq_k(ra_q, dec_q, z_q, ra_n, dec_n, z_n,
                   theta_radii_rad, z_q_edges, z_n_edges, k_max,
                   exclude_self=False):
    """Pure-Python brute-force reference: O(N_q * N_n) with no tricks."""
    n_theta = theta_radii_rad.size
    n_z_q = z_q_edges.size - 1
    n_z_n = z_n_edges.size - 1
    H = np.zeros((n_theta, n_z_q, n_z_n, k_max), dtype=np.int64)
    sum_n = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
    N_q = np.zeros(n_z_q, dtype=np.int64)

    theta_q = np.deg2rad(90.0 - dec_q)
    phi_q = np.deg2rad(ra_q)
    theta_n = np.deg2rad(90.0 - dec_n)
    phi_n = np.deg2rad(ra_n)

    for qi in range(ra_q.size):
        i_z_q = np.searchsorted(z_q_edges, z_q[qi], side="right") - 1
        if i_z_q < 0 or i_z_q >= n_z_q:
            continue
        N_q[i_z_q] += 1
        cs = (np.sin(theta_q[qi]) * np.sin(theta_n)
              * np.cos(phi_q[qi] - phi_n)
              + np.cos(theta_q[qi]) * np.cos(theta_n))
        cs = np.clip(cs, -1.0, 1.0)
        sep = np.arccos(cs)
        i_z_n = np.searchsorted(z_n_edges, z_n, side="right") - 1
        valid = (i_z_n >= 0) & (i_z_n < n_z_n)
        if exclude_self:
            valid = valid & (np.arange(ra_n.size) != qi)
        for t in range(n_theta):
            in_cap = sep <= theta_radii_rad[t]
            mask = in_cap & valid
            if not mask.any():
                continue
            counts_per_zn = np.bincount(i_z_n[mask], minlength=n_z_n)
            sum_n[t, i_z_q] += counts_per_zn
            for jn in range(n_z_n):
                n = int(counts_per_zn[jn])
                top = min(n, k_max)
                for k in range(top):
                    H[t, i_z_q, jn, k] += 1
    return H, sum_n, N_q


def test_h_geq_k_matches_brute_force_dd():
    """End-to-end: knn_cdf primitive on a 80-galaxy DD-flavor mock vs
    the brute-force reference — exact integer match."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(seed=0, n=80)

    theta_radii = np.deg2rad(np.array([1.0, 2.5, 5.0, 10.0, 20.0]))
    z_q_edges = np.array([0.5, 1.0, 1.5, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])
    k_max = 4

    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=k_max, flavor="DD",
        nside_lookup=64, n_threads=1,
    )

    H_ref, sum_ref, Nq_ref = _brute_h_geq_k(
        ra, dec, z, ra, dec, z,
        theta_radii, z_q_edges, z_n_edges, k_max,
        exclude_self=True,
    )

    np.testing.assert_array_equal(res.N_q, Nq_ref)
    np.testing.assert_array_equal(res.H_geq_k, H_ref)
    np.testing.assert_allclose(res.sum_n, sum_ref, rtol=0, atol=0)


def test_rd_flavor_no_self_exclusion():
    """RD-flavor: query and neighbor catalogs are different objects, so
    no self-exclusion is applied even if their *contents* overlap."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra_q, dec_q, z_q = _toy_catalog(seed=42, n=40)
    ra_n, dec_n, z_n = _toy_catalog(seed=123, n=80)

    theta_radii = np.deg2rad(np.array([2.0, 5.0, 10.0]))
    z_q_edges = np.array([0.5, 1.25, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])
    k_max = 3

    res = joint_knn_cdf(
        ra_q, dec_q, z_q, ra_n, dec_n, z_n,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=k_max, flavor="RD",
        nside_lookup=64, n_threads=1,
    )

    H_ref, sum_ref, Nq_ref = _brute_h_geq_k(
        ra_q, dec_q, z_q, ra_n, dec_n, z_n,
        theta_radii, z_q_edges, z_n_edges, k_max,
        exclude_self=False,
    )

    np.testing.assert_array_equal(res.N_q, Nq_ref)
    np.testing.assert_array_equal(res.H_geq_k, H_ref)
    np.testing.assert_allclose(res.sum_n, sum_ref)


def test_k_ladder_monotonicity():
    """H_>=k must be non-increasing in k for every (theta, z_q, z_n)."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(seed=2, n=200)
    theta_radii = np.deg2rad(np.array([2.0, 5.0, 10.0, 20.0]))
    z_q_edges = np.array([0.5, 1.25, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])

    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=8, flavor="DD",
        nside_lookup=64, n_threads=1,
    )
    diffs = np.diff(res.H_geq_k, axis=-1)
    assert (diffs <= 0).all(), "H_geq_k must be non-increasing in k"


def test_sum_n_matches_cone_shell_counts_aggregate():
    """With z_q collapsed to one shell, sum_n[t, 0, jn] must equal the
    total over centres of cone_shell_counts(N)[i_centre, t, jn]."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.sigma2_cone_shell_estimator import cone_shell_counts

    rng = np.random.default_rng(5)
    n_g = 500
    ra = rng.uniform(0, 360, n_g)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_g)))
    z = rng.uniform(0.5, 2.0, n_g)

    # Use a few HEALPix centres as queries; treat them as "queries
    # without redshift" by collapsing z_q to one shell covering the
    # full range. Note: cone_shell_counts has cap centres without z;
    # for the comparison we synthesise z_q from the centres' indices
    # (any value in the single shell will do).
    ra_c = np.array([10.0, 50.0, 100.0, 200.0, 300.0])
    dec_c = np.array([0.0, 30.0, -20.0, 45.0, -45.0])
    z_q_dummy = np.full(ra_c.size, 1.0)  # any value in the single shell

    theta_radii = np.deg2rad(np.array([2.0, 5.0, 10.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])

    N, _ = cone_shell_counts(
        ra, dec, z, theta_radii, z_n_edges, ra_c, dec_c,
        nside_lookup=128, n_threads=1,
    )
    res = joint_knn_cdf(
        ra_c, dec_c, z_q_dummy, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=0, flavor="RD",  # no self-exclusion: different catalogs
        nside_lookup=128, n_threads=1,
    )
    # sum over centres axis of cone_shell_counts equals sum_n[t,0,jn]
    np.testing.assert_allclose(N.sum(axis=0), res.sum_n[:, 0, :])


def test_jackknife_per_region_sums_to_full():
    """Per-region cubes must sum (over the region axis) to the full
    cubes — single-pass jackknife consistency."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(seed=11, n=120)
    rng = np.random.default_rng(99)
    n_regions = 5
    region_labels = rng.integers(0, n_regions, size=ra.size)

    theta_radii = np.deg2rad(np.array([2.0, 5.0, 10.0]))
    z_q_edges = np.array([0.5, 1.25, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])

    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=4, flavor="DD",
        region_labels_query=region_labels, n_regions=n_regions,
        nside_lookup=64, n_threads=1,
    )
    np.testing.assert_allclose(
        res.sum_n_per_region.sum(axis=-1), res.sum_n,
    )
    np.testing.assert_allclose(
        res.sum_n2_per_region.sum(axis=-1), res.sum_n2,
    )
    np.testing.assert_array_equal(
        res.N_q_per_region.sum(axis=-1), res.N_q,
    )
    np.testing.assert_array_equal(
        res.H_geq_k_per_region.sum(axis=-1), res.H_geq_k,
    )


def test_kmax_zero_skips_ladder_but_keeps_moments():
    """k_max=0 must populate sum_n / sum_n2 but leave H_geq_k empty."""
    healpy = pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(seed=3, n=60)
    theta_radii = np.deg2rad(np.array([5.0, 10.0]))
    z_q_edges = np.array([0.5, 2.0])
    z_n_edges = np.array([0.5, 2.0])

    res = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=0, flavor="DD",
        nside_lookup=64, n_threads=1,
    )
    assert (res.H_geq_k == 0).all()
    assert res.sum_n.sum() > 0
    assert res.sum_n2.sum() > 0
