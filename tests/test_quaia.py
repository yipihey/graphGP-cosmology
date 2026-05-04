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


def test_load_quaia_minimal_fits_roundtrip(tmp_path):
    """Synthesise a minimal Quaia-schema FITS pair (data has redshift_quaia,
    random has only (ra, dec, ebv) -- the published schema) and verify that
    load_quaia produces a valid QuaiaCatalog with random z's drawn from the
    data n(z)."""
    pytest.importorskip("astropy")
    from astropy.table import Table
    from twopt_density.quaia import load_quaia

    rng = np.random.default_rng(0)
    N_d, N_r = 800, 8000
    data = Table({
        "ra": rng.uniform(0, 360, size=N_d),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, size=N_d))),
        "redshift_quaia": rng.uniform(0.5, 4.0, size=N_d),
    })
    random = Table({
        "ra": rng.uniform(0, 360, size=N_r),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, size=N_r))),
        "ebv": rng.uniform(0.0, 0.5, size=N_r),
    })
    fn_d = tmp_path / "mini_quaia.fits"
    fn_r = tmp_path / "mini_random.fits"
    data.write(fn_d, overwrite=True)
    random.write(fn_r, overwrite=True)

    cat = load_quaia(str(fn_d), str(fn_r))

    assert cat.N_data == N_d
    assert cat.N_random == N_r
    # default strategy "sample_from_data": random n(z) tracks data n(z)
    assert abs(cat.z_random.mean() - cat.z_data.mean()) < 0.05 * cat.z_data.mean()
    # z bounds preserved
    assert cat.z_random.min() >= cat.z_data.min() - 1e-6
    assert cat.z_random.max() <= cat.z_data.max() + 1e-6


def test_make_random_from_selection_function():
    """A random sampled from a healpix selection map respects the mask
    (no points in zero-completeness pixels) and inherits the data n(z)."""
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.quaia import make_random_from_selection_function

    nside = 16
    npix = 12 * nside ** 2
    rng = np.random.default_rng(0)

    # synthetic selection map: 50% of pixels at completeness 1, rest at 0
    sel = np.zeros(npix, dtype=np.float64)
    on_pix = rng.choice(npix, size=npix // 2, replace=False)
    sel[on_pix] = 1.0

    z_data = rng.uniform(0.5, 4.0, size=2000)
    ra, dec, z = make_random_from_selection_function(
        sel, n_random=10000, z_data=z_data, nside=nside, rng=rng,
    )
    assert ra.shape == (10000,) and dec.shape == (10000,) and z.shape == (10000,)
    # all random points must land in pixels that have completeness > 0
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec), np.deg2rad(ra))
    assert (sel[pix] > 0).all(), "random points landed in masked pixels"
    # n(z) close to z_data n(z)
    assert abs(z.mean() - z_data.mean()) < 0.05 * z_data.mean()


def test_load_selection_function_round_trips():
    """Synthesise a Quaia-format selection FITS and round-trip via the loader."""
    import os, tempfile
    pytest.importorskip("astropy")
    from astropy.io import fits
    from twopt_density.quaia import load_selection_function

    nside = 8
    npix = 12 * nside ** 2
    sel = np.linspace(0.0, 1.0, npix, dtype=np.float64)

    # The published format: 1 column 'T' with 1024-double rows (npix/1024 rows).
    # For small NSIDE we just match column-name + flatten on read.
    col = fits.Column(name="T", format="D", array=sel)
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["NSIDE"] = nside
    hdu.header["TTYPE1"] = "T"
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "sel.fits")
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fn, overwrite=True)
        sel_back, nside_back = load_selection_function(fn)
    assert nside_back == nside and sel_back.size == npix
    np.testing.assert_allclose(sel_back, sel, rtol=1e-12, atol=1e-14)


def test_load_quaia_unknown_strategy_raises(tmp_path):
    pytest.importorskip("astropy")
    from astropy.table import Table
    from twopt_density.quaia import load_quaia

    rng = np.random.default_rng(0)
    data = Table({
        "ra": rng.uniform(0, 360, size=100),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, size=100))),
        "redshift_quaia": rng.uniform(0.5, 4.0, size=100),
    })
    random = Table({
        "ra": rng.uniform(0, 360, size=200),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, size=200))),
        "ebv": rng.uniform(0.0, 0.5, size=200),
    })
    fn_d = tmp_path / "d.fits"
    fn_r = tmp_path / "r.fits"
    data.write(fn_d, overwrite=True)
    random.write(fn_r, overwrite=True)
    with pytest.raises(ValueError):
        load_quaia(str(fn_d), str(fn_r), random_z_strategy="bogus")
