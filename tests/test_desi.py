"""Regression tests for the DESI DR1 QSO loader (twopt_density.desi).

The real DESI DR1 LSS catalogs are external (see
``demos/demo_desi_qso_bao.py``). These tests build tiny synthetic
DESI-style FITS files in tmp_path so the loader logic is verified
end-to-end without the ~500 MB downloads.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_synthetic_desi_fits(path: str, n: int, ra_centre: float = 180.0,
                                dec_centre: float = 0.0, ra_span: float = 30.0,
                                dec_span: float = 30.0, z_min: float = 0.8,
                                z_max: float = 2.1, seed: int = 0,
                                with_full_weights: bool = True):
    """Write a tiny DESI-style clustering FITS at ``path``."""
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(seed)

    ra = ra_centre + ra_span * (rng.uniform(size=n) - 0.5)
    dec = dec_centre + dec_span * (rng.uniform(size=n) - 0.5)
    z = rng.uniform(z_min, z_max, n)
    cols = [
        fits.Column(name="RA", format="D", array=ra),
        fits.Column(name="DEC", format="D", array=dec),
        fits.Column(name="Z", format="D", array=z),
        fits.Column(name="WEIGHT_FKP", format="D",
                    array=1.0 / (1.0 + 1e-4 * z)),
    ]
    if with_full_weights:
        cols.extend([
            fits.Column(name="WEIGHT_SYS", format="D",
                        array=np.ones(n) + 0.05 * rng.uniform(size=n)),
            fits.Column(name="WEIGHT_NOZ", format="D",
                        array=np.ones(n) + 0.02 * rng.uniform(size=n)),
            fits.Column(name="WEIGHT_COMP_TILE", format="D",
                        array=np.ones(n) + 0.10 * rng.uniform(size=n)),
        ])
    else:
        cols.append(fits.Column(name="WEIGHT", format="D",
                                  array=np.ones(n)))
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)


def test_desi_loader_reads_basic_columns(tmp_path):
    """Loader reads RA/DEC/Z and combined WEIGHT*WEIGHT_FKP from
    a synthetic DESI FITS, applies z cut, and returns a DESICatalog."""
    pytest.importorskip("astropy.io.fits")
    from twopt_density.desi import load_desi_qso
    from twopt_density.distance import DistanceCosmo

    fid = DistanceCosmo(Om=0.31, h=0.68)
    n = 5000
    p = str(tmp_path / "QSO_NGC_clustering.dat.fits")
    _make_synthetic_desi_fits(p, n=n, z_min=0.5, z_max=2.5, seed=1)
    cat = load_desi_qso(catalog_paths=[p], fid_cosmo=fid,
                          z_min=0.8, z_max=2.1)
    # at least the z cut applied
    assert cat.N_data > 0 and cat.N_data <= n
    assert ((cat.z_data >= 0.8) & (cat.z_data <= 2.1)).all()
    # weights must be positive and finite
    assert (cat.w_data > 0).all() and np.isfinite(cat.w_data).all()
    # xyz_data has the right shape
    assert cat.xyz_data.shape == (cat.N_data, 3)
    # no randoms loaded -> empty arrays
    assert cat.N_random == 0


def test_desi_loader_combines_weight_components(tmp_path):
    """When WEIGHT is not provided, the loader builds it as
    WEIGHT_SYS * WEIGHT_NOZ * WEIGHT_COMP_TILE, then * WEIGHT_FKP."""
    pytest.importorskip("astropy.io.fits")
    from astropy.io import fits
    from twopt_density.desi import load_desi_qso
    from twopt_density.distance import DistanceCosmo

    fid = DistanceCosmo(Om=0.31, h=0.68)
    p = str(tmp_path / "QSO_NGC_clustering.dat.fits")
    n = 100
    rng = np.random.default_rng(0)
    ra = 180.0 + 5 * (rng.uniform(size=n) - 0.5)
    dec = 0.0 + 5 * (rng.uniform(size=n) - 0.5)
    z = rng.uniform(1.0, 1.5, n)
    ws = np.full(n, 1.2); wn = np.full(n, 1.1); wc = np.full(n, 0.9)
    wfkp = np.full(n, 0.5)
    cols = [
        fits.Column(name="RA", format="D", array=ra),
        fits.Column(name="DEC", format="D", array=dec),
        fits.Column(name="Z", format="D", array=z),
        fits.Column(name="WEIGHT_SYS", format="D", array=ws),
        fits.Column(name="WEIGHT_NOZ", format="D", array=wn),
        fits.Column(name="WEIGHT_COMP_TILE", format="D", array=wc),
        fits.Column(name="WEIGHT_FKP", format="D", array=wfkp),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(p, overwrite=True)
    cat = load_desi_qso(catalog_paths=[p], fid_cosmo=fid,
                          z_min=0.8, z_max=2.1)
    expected = 1.2 * 1.1 * 0.9 * 0.5
    np.testing.assert_allclose(cat.w_data, expected, rtol=1e-10)


def test_angular_completeness_normalised_to_unity(tmp_path):
    """``angular_completeness_from_randoms`` returns a HEALPix map
    in [0, 1] with the populated-pixel median at 1."""
    pytest.importorskip("astropy.io.fits")
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.desi import angular_completeness_from_randoms

    rng = np.random.default_rng(1)
    n = 200_000
    # uniform on a 30deg x 30deg patch
    ra = 180.0 + 30.0 * (rng.uniform(size=n) - 0.5)
    dec = 0.0 + 30.0 * (rng.uniform(size=n) - 0.5)
    nside = 64
    mask = angular_completeness_from_randoms(ra, dec, nside=nside)
    assert mask.shape == (12 * nside ** 2,)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    pop = mask[mask > 0]
    np.testing.assert_allclose(np.median(pop), 1.0, atol=0.5)
    # only a small fraction of the sky should be lit
    assert mask.mean() < 0.05


def test_desi_loader_with_randoms_roundtrip(tmp_path):
    """Loader accepts both data and random paths and returns a coherent
    DESICatalog with both populated."""
    pytest.importorskip("astropy.io.fits")
    from twopt_density.desi import (
        angular_completeness_from_randoms, load_desi_qso,
    )
    from twopt_density.distance import DistanceCosmo

    fid = DistanceCosmo(Om=0.31, h=0.68)
    pdat = str(tmp_path / "QSO_NGC_clustering.dat.fits")
    pran = str(tmp_path / "QSO_NGC_0_clustering.ran.fits")
    _make_synthetic_desi_fits(pdat, n=3_000, seed=11)
    _make_synthetic_desi_fits(pran, n=30_000, seed=12)
    cat = load_desi_qso(catalog_paths=[pdat], randoms_paths=[pran],
                          fid_cosmo=fid, z_min=0.8, z_max=2.1,
                          n_random_max=20_000)
    assert cat.N_data > 0 and cat.N_random > 0 and cat.N_random <= 20_000
    # subsample never exceeds the requested cap
    mask = angular_completeness_from_randoms(cat.ra_random, cat.dec_random,
                                               nside=64)
    # coarse sanity: footprint covers ~ (30 deg)^2 / 41253 deg^2 ~ 0.022
    assert 0.005 < mask.mean() < 0.05
