"""Test that chunked query processing in joint_knn_cdf gives bit-exact
results vs unchunked (single-chunk) processing."""

from __future__ import annotations

import numpy as np
import pytest


def _toy_catalog(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_dec = rng.uniform(-1, 1, n)
    ra = np.degrees(phi)
    dec = np.degrees(np.arcsin(sin_dec))
    z = rng.uniform(0.5, 2.0, n)
    return ra, dec, z


@pytest.mark.parametrize("chunk_size", [200, 500, 1000])
def test_chunked_equals_single_pass_dd(chunk_size):
    pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(n=1200, seed=3)
    theta_radii = np.deg2rad(np.array([1.0, 2.5, 5.0, 10.0]))
    z_q_edges = np.array([0.5, 1.25, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])

    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=5, flavor="DD",
        nside_lookup=64, n_threads=1,
    )

    res_full = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        query_chunk_size=10_000, **common,  # one chunk
    )
    res_chunked = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        query_chunk_size=chunk_size, **common,
    )

    np.testing.assert_array_equal(res_full.H_geq_k, res_chunked.H_geq_k)
    np.testing.assert_allclose(res_full.sum_n, res_chunked.sum_n,
                               rtol=0, atol=0)
    np.testing.assert_allclose(res_full.sum_n2, res_chunked.sum_n2,
                               rtol=0, atol=0)
    np.testing.assert_array_equal(res_full.N_q, res_chunked.N_q)


def test_chunked_equals_single_pass_jackknife():
    pytest.importorskip("healpy")
    pytest.importorskip("numba")
    from twopt_density.knn_cdf import joint_knn_cdf

    ra, dec, z = _toy_catalog(n=1500, seed=7)
    rng = np.random.default_rng(99)
    n_regions = 7
    region_labels = rng.integers(0, n_regions, size=ra.size)

    theta_radii = np.deg2rad(np.array([2.0, 5.0]))
    z_q_edges = np.array([0.5, 1.25, 2.0])
    z_n_edges = np.array([0.5, 1.25, 2.0])

    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=4, flavor="DD",
        region_labels_query=region_labels, n_regions=n_regions,
        nside_lookup=64, n_threads=2,
    )
    res_full = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        query_chunk_size=10_000, **common,
    )
    res_chunked = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        query_chunk_size=300, **common,
    )

    np.testing.assert_array_equal(
        res_full.H_geq_k_per_region, res_chunked.H_geq_k_per_region)
    np.testing.assert_allclose(
        res_full.sum_n_per_region, res_chunked.sum_n_per_region,
        rtol=0, atol=0)
    np.testing.assert_array_equal(
        res_full.N_q_per_region, res_chunked.N_q_per_region)
