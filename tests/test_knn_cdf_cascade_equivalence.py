"""Bit-equivalence tests: cascade backend vs numba backend.

For each scenario, run both ``joint_knn_cdf`` (numba/HEALPix) and
``joint_knn_cdf_cascade`` (Rust morton-cascade CLI) on identical
inputs and assert:

- ``H_geq_k`` integers match exactly (np.array_equal).
- ``sum_n`` / ``sum_n2`` match to machine precision
  (~1e-10 absolute on cap counts that are ≲ catalog size).
- Per-region cubes match when jackknife is enabled.

The cascade binary must be built first:
``cd morton_cascade && cargo build --release``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _synth_sphere(n, seed=0, ra_lo=0.0, ra_hi=60.0, dec_lo=-20.0, dec_hi=30.0,
                   z_lo=0.8, z_hi=2.1):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(ra_lo, ra_hi, n)
    dec = rng.uniform(dec_lo, dec_hi, n)
    z = rng.uniform(z_lo, z_hi, n)
    return ra.astype(np.float64), dec.astype(np.float64), z.astype(np.float64)


def _binary_available():
    """Skip if the cascade binary isn't built."""
    from twopt_density.morton_backend import _resolve_binary
    try:
        _resolve_binary()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _binary_available(),
    reason="morton-cascade binary not built; run `cargo build --release` "
           "in morton_cascade/",
)


def test_tiny_unweighted():
    """200 data, no weights, small cube."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.morton_knn_cdf import joint_knn_cdf_cascade

    ra, dec, z = _synth_sphere(200, seed=42)
    theta = np.deg2rad(np.array([0.5, 1.0, 2.0, 4.0, 8.0]))
    z_q_edges = np.array([0.8, 1.5, 2.1])
    z_n_edges = z_q_edges.copy()

    # Note: same array objects → same_catalog=True path.
    res_numba = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=5, flavor="DD", n_threads=2,
    )
    res_cas = joint_knn_cdf_cascade(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=5, flavor="DD",
    )

    assert np.array_equal(res_numba.N_q, res_cas.N_q), \
        f"N_q mismatch: {res_numba.N_q} vs {res_cas.N_q}"
    assert np.array_equal(res_numba.H_geq_k, res_cas.H_geq_k), \
        "H_geq_k integers must be bit-identical"
    np.testing.assert_allclose(res_numba.sum_n, res_cas.sum_n, atol=1e-10)
    np.testing.assert_allclose(res_numba.sum_n2, res_cas.sum_n2, atol=1e-10)
    # Higher moments p=3,4 must also match bit-identically (note v4_1 §6).
    assert res_numba.sum_n3 is not None and res_cas.sum_n3 is not None
    assert res_numba.sum_n4 is not None and res_cas.sum_n4 is not None
    np.testing.assert_allclose(res_numba.sum_n3, res_cas.sum_n3, atol=1e-9)
    np.testing.assert_allclose(res_numba.sum_n4, res_cas.sum_n4, atol=1e-9)


def test_weighted():
    """1000 data with random weights."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.morton_knn_cdf import joint_knn_cdf_cascade

    ra, dec, z = _synth_sphere(1000, seed=7)
    rng = np.random.default_rng(11)
    w = rng.uniform(0.7, 1.3, ra.size).astype(np.float64)
    theta = np.deg2rad(np.array([0.3, 0.7, 1.5, 3.0, 6.0]))
    z_q_edges = np.array([0.8, 1.2, 1.7, 2.1])
    z_n_edges = z_q_edges.copy()

    # NB: weighted same-catalogue self-exclusion is the numba "subtract
    # 1.0" rule, which assumes self-weight = 1.0 (a known quirk for
    # weighted DD). The cascade backend mirrors it exactly.
    res_numba = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=20, flavor="DD",
        weights_neigh=w, n_threads=2,
    )
    res_cas = joint_knn_cdf_cascade(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=20, flavor="DD",
        weights_neigh=w,
    )

    assert np.array_equal(res_numba.N_q, res_cas.N_q)
    assert np.array_equal(res_numba.H_geq_k, res_cas.H_geq_k), \
        "Weighted H_geq_k must be bit-identical"
    np.testing.assert_allclose(res_numba.sum_n, res_cas.sum_n, atol=1e-9)
    np.testing.assert_allclose(res_numba.sum_n2, res_cas.sum_n2, atol=1e-9)
    np.testing.assert_allclose(res_numba.sum_n3, res_cas.sum_n3, atol=1e-8)
    np.testing.assert_allclose(res_numba.sum_n4, res_cas.sum_n4, atol=1e-8)


def test_jackknife():
    """5K data with 10 jackknife regions."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.morton_knn_cdf import joint_knn_cdf_cascade

    ra, dec, z = _synth_sphere(5000, seed=3)
    rng = np.random.default_rng(5)
    n_regions = 10
    regions = rng.integers(0, n_regions, ra.size).astype(np.int64)
    theta = np.deg2rad(np.array([0.2, 0.5, 1.0, 2.5, 5.0]))
    z_q_edges = np.array([0.8, 1.5, 2.1])
    z_n_edges = z_q_edges.copy()

    res_numba = joint_knn_cdf(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=15, flavor="DD",
        region_labels_query=regions, n_regions=n_regions, n_threads=2,
    )
    res_cas = joint_knn_cdf_cascade(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=15, flavor="DD",
        region_labels_query=regions, n_regions=n_regions,
    )

    assert np.array_equal(res_numba.N_q, res_cas.N_q)
    assert np.array_equal(res_numba.H_geq_k, res_cas.H_geq_k)
    assert np.array_equal(res_numba.N_q_per_region, res_cas.N_q_per_region)
    assert np.array_equal(res_numba.H_geq_k_per_region,
                           res_cas.H_geq_k_per_region), \
        "Per-region H_geq_k must be bit-identical"
    np.testing.assert_allclose(res_numba.sum_n_per_region,
                                res_cas.sum_n_per_region, atol=1e-9)
    np.testing.assert_allclose(res_numba.sum_n2_per_region,
                                res_cas.sum_n2_per_region, atol=1e-9)
    np.testing.assert_allclose(res_numba.sum_n3_per_region,
                                res_cas.sum_n3_per_region, atol=1e-8)
    np.testing.assert_allclose(res_numba.sum_n4_per_region,
                                res_cas.sum_n4_per_region, atol=1e-8)


def test_rd_flavor_no_self_exclusion():
    """RD: query=random, neighbour=data (different arrays). No self-pair."""
    from twopt_density.knn_cdf import joint_knn_cdf
    from twopt_density.morton_knn_cdf import joint_knn_cdf_cascade

    ra_q, dec_q, z_q_arr = _synth_sphere(500, seed=1)
    ra_n, dec_n, z_n_arr = _synth_sphere(800, seed=2)
    theta = np.deg2rad(np.array([0.5, 1.5, 4.0]))
    z_q_edges = np.array([0.8, 1.5, 2.1])
    z_n_edges = z_q_edges.copy()

    res_numba = joint_knn_cdf(
        ra_q, dec_q, z_q_arr, ra_n, dec_n, z_n_arr,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=10, flavor="RD", n_threads=2,
    )
    res_cas = joint_knn_cdf_cascade(
        ra_q, dec_q, z_q_arr, ra_n, dec_n, z_n_arr,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=10, flavor="RD",
    )

    assert np.array_equal(res_numba.N_q, res_cas.N_q)
    assert np.array_equal(res_numba.H_geq_k, res_cas.H_geq_k)
    np.testing.assert_allclose(res_numba.sum_n, res_cas.sum_n, atol=1e-9)


def test_h_geq_k_self_consistency():
    """Cascade output should obey monotone-non-increasing in k AND
    per-region cubes should sum to global cube."""
    from twopt_density.morton_knn_cdf import joint_knn_cdf_cascade

    ra, dec, z = _synth_sphere(500, seed=99)
    rng = np.random.default_rng(100)
    n_regions = 4
    regions = rng.integers(0, n_regions, ra.size).astype(np.int64)
    theta = np.deg2rad(np.array([0.3, 1.0, 3.0]))
    z_q_edges = np.array([0.8, 1.5, 2.1])
    z_n_edges = z_q_edges.copy()

    res = joint_knn_cdf_cascade(
        ra, dec, z, ra, dec, z,
        theta_radii_rad=theta,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        k_max=8, flavor="DD",
        region_labels_query=regions, n_regions=n_regions,
    )

    # Monotone non-increasing in k.
    diff = np.diff(res.H_geq_k, axis=-1)
    assert (diff <= 0).all(), "H_geq_k should be monotone non-increasing in k"

    # Per-region sums to global.
    assert np.array_equal(
        res.H_geq_k, res.H_geq_k_per_region.sum(axis=-1)
    ), "H_geq_k_per_region sum must equal global H_geq_k"
    np.testing.assert_allclose(
        res.sum_n, res.sum_n_per_region.sum(axis=-1), atol=1e-12
    )
    assert np.array_equal(
        res.N_q, res.N_q_per_region.sum(axis=-1)
    )
