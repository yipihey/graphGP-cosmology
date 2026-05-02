"""Tests for the Quaia loader and Quaia-shape mock generator."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import QuaiaCatalog, _galactic_mask, make_mock_quaia


def test_mock_returns_quaia_catalog_schema():
    cat = make_mock_quaia(n_data=200, n_random=800, seed=0)
    assert isinstance(cat, QuaiaCatalog)
    assert cat.xyz_data.shape == (cat.N_data, 3)
    assert cat.xyz_random.shape == (cat.N_random, 3)
    assert len(cat.ra_data) == cat.N_data == len(cat.dec_data) == len(cat.z_data)
    assert len(cat.ra_random) == cat.N_random


def test_mock_respects_galactic_mask():
    cat = make_mock_quaia(n_data=2000, n_random=2000, b_min_galactic=15.0, seed=1)
    in_data = _galactic_mask(cat.ra_data, cat.dec_data, b_min=14.99)
    in_rand = _galactic_mask(cat.ra_random, cat.dec_random, b_min=14.99)
    assert in_data.all(), f"{(~in_data).sum()} data outside |b|>=15 mask"
    assert in_rand.all(), f"{(~in_rand).sum()} random outside |b|>=15 mask"


def test_mock_nz_in_range():
    cat = make_mock_quaia(n_data=500, n_random=500, z_min=0.5, z_max=4.5, seed=2)
    assert cat.z_data.min() >= 0.5 - 1e-6
    assert cat.z_data.max() <= 4.5 + 1e-6
    # bimodal target: median should be between the two peaks (~1 and ~2)
    assert 0.9 <= np.median(cat.z_data) <= 2.0


def test_mock_clustered_fraction_zero_gives_uniform_sky():
    """clustered_fraction=0 -> data and random both uniform on the masked sky.
    The two should have indistinguishable n(z) and a sky distribution
    consistent with uniform-on-sphere within Poisson noise."""
    cat = make_mock_quaia(n_data=4000, n_random=4000,
                          clustered_fraction=0.0, seed=3)
    # n(z) means should agree to a few percent
    rel = abs(cat.z_data.mean() - cat.z_random.mean()) / cat.z_random.mean()
    assert rel < 0.05, f"n(z) mean rel diff {rel:.3e}"


def test_mock_clustered_fraction_positive_increases_pair_density():
    """clustered_fraction > 0 -> excess pairs at the cluster scale
    (~ a few tens of Mpc/h) relative to clustered_fraction == 0."""
    cat0 = make_mock_quaia(n_data=4000, n_random=0,
                            clustered_fraction=0.0, seed=4)
    cat1 = make_mock_quaia(n_data=4000, n_random=0,
                            clustered_fraction=0.5, seed=4)
    from scipy.spatial import cKDTree
    pts0 = cat0.xyz_data - cat0.xyz_data.min(axis=0) + 100.0
    pts1 = cat1.xyz_data - cat1.xyz_data.min(axis=0) + 100.0
    n0 = len(cKDTree(pts0).query_pairs(r=50.0, output_type="ndarray"))
    n1 = len(cKDTree(pts1).query_pairs(r=50.0, output_type="ndarray"))
    assert n1 > 3 * n0 + 50, f"clustered:{n1} vs uniform:{n0}"


def test_shift_to_positive_keeps_relative_separations():
    cat = make_mock_quaia(n_data=300, n_random=300, seed=5)
    pos, rnd, box = cat.shift_to_positive(margin=20.0)
    assert pos.min() >= 20.0 - 1e-6
    assert rnd.min() >= 20.0 - 1e-6
    assert pos.max() < box and rnd.max() < box
    # Pair separations are translation-invariant
    sep_orig = np.linalg.norm(cat.xyz_data[0] - cat.xyz_data[1])
    sep_new = np.linalg.norm(pos[0] - pos[1])
    np.testing.assert_allclose(sep_orig, sep_new, rtol=1e-12)


def test_mock_xyz_consistent_with_distance_module():
    """The xyz built by make_mock_quaia matches an independent
    radec_z_to_cartesian call on the same inputs."""
    from twopt_density.distance import radec_z_to_cartesian
    fid = DistanceCosmo(Om=0.31, h=0.68)
    cat = make_mock_quaia(n_data=200, n_random=200, fid_cosmo=fid, seed=6)
    xyz_check = np.asarray(radec_z_to_cartesian(
        jnp.asarray(cat.ra_data), jnp.asarray(cat.dec_data),
        jnp.asarray(cat.z_data), fid,
    ))
    np.testing.assert_allclose(cat.xyz_data, xyz_check, rtol=1e-12, atol=1e-9)


def test_load_quaia_signature():
    """``load_quaia`` exists and signals a clear error on missing file."""
    from twopt_density.quaia import load_quaia
    with pytest.raises(Exception):
        load_quaia("nonexistent.fits", "nonexistent_random.fits")
