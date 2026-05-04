"""DESI DR1 LSS catalog loader for QSO clustering analysis.

The DESI DR1 LSS catalogs (Ross et al. 2024, arXiv:2405.16593) carry
matched data and random files following the BOSS-style convention.
For QSOs the published files are::

    QSO_NGC_clustering.dat.fits
    QSO_SGC_clustering.dat.fits
    QSO_NGC_{0..17}_clustering.ran.fits      # 18 random realizations
    QSO_SGC_{0..17}_clustering.ran.fits

at ``data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/``
(or v1.2). Standard column conventions:

  RA, DEC          [deg]
  Z                spectroscopic redshift
  WEIGHT           combined weight = WEIGHT_SYS * WEIGHT_NOZ * WEIGHT_COMP_TILE
  WEIGHT_FKP       FKP weight (Anderson+12 eq 17, optimised for BAO)
  NZ               n(z) value at object's z (used in FKP weight)

For pair-counting BAO analysis the recommended per-object multiplicative
weight is ``WEIGHT * WEIGHT_FKP``.

Ports cleanly into our analytic-RR pipeline because:
  1. Randoms are Poisson-uniform over the survey footprint, then
     sub-sampled by completeness -- the random density on the sky IS
     the angular completeness function. We can derive a HEALPix
     completeness map by binning randoms at NSIDE=256.
  2. Random redshifts are assigned by *shuffling* from the data n(z),
     so the radial distribution is just the histogram of data z's --
     same as our Quaia analytic-RR ``n(z)``.
  3. Our analytic-RR with smooth histogrammed n(z) sidesteps the
     Radial Integral Constraint (de Mattia & Ruhlmann-Kleider 2019)
     that DESI's shuffled-z method introduces.

This module exposes::

    load_desi_qso(catalog_paths, randoms_paths, fid_cosmo,
                  z_min, z_max, ...) -> DESICatalog
        Read N+S galactic caps (concatenate), apply z cut, build comoving
        positions, optionally load randoms.

    angular_completeness_from_randoms(ra_r, dec_r, nside=256) -> mask
        HEALPix completeness map normalised to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian


@dataclass
class DESICatalog:
    """Self-contained DESI LSS catalog.

    Mirrors ``QuaiaCatalog`` with two additions: per-object
    multiplicative weights ``w_data`` (combined WEIGHT * WEIGHT_FKP)
    and the radial counterpart ``w_random`` for the randoms.
    """
    ra_data: np.ndarray              # (N_d,) deg
    dec_data: np.ndarray             # (N_d,) deg
    z_data: np.ndarray               # (N_d,) spectroscopic z
    xyz_data: np.ndarray             # (N_d, 3) comoving Mpc/h
    w_data: np.ndarray               # (N_d,) per-object weight

    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    xyz_random: np.ndarray
    w_random: np.ndarray

    fid_cosmo: DistanceCosmo

    @property
    def N_data(self) -> int:
        return len(self.ra_data)

    @property
    def N_random(self) -> int:
        return len(self.ra_random)


def _read_clustering_fits(path: str, with_weight_fkp: bool = True):
    """Read RA/DEC/Z and combined weight from a DESI clustering FITS.

    Tries the canonical column names in order:
      WEIGHT * WEIGHT_FKP (preferred)
      WEIGHT_SYS * WEIGHT_NOZ * WEIGHT_COMP_TILE * WEIGHT_FKP
      fall back to 1.0
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        cols = [c.upper() for c in t.columns.names]

        def col(name):
            return np.asarray(t[name], dtype=np.float64) if name.upper() in cols else None

        ra = col("RA"); dec = col("DEC")
        z = col("Z") if "Z" in cols else col("Z_QSO_HP")
        if ra is None or dec is None or z is None:
            raise ValueError(f"{path}: required columns RA/DEC/Z missing")

        w = col("WEIGHT")
        if w is None:
            ws = col("WEIGHT_SYS"); wn = col("WEIGHT_NOZ"); wc = col("WEIGHT_COMP_TILE")
            w = (ws if ws is not None else 1.0) * \
                (wn if wn is not None else 1.0) * \
                (wc if wc is not None else 1.0)
            if isinstance(w, float):
                w = np.full(len(ra), w)
        if with_weight_fkp:
            w_fkp = col("WEIGHT_FKP")
            if w_fkp is not None:
                w = w * w_fkp
        return ra, dec, z, np.asarray(w, dtype=np.float64)


def load_desi_qso(
    catalog_paths: Iterable[str],
    fid_cosmo: DistanceCosmo,
    randoms_paths: Optional[Iterable[str]] = None,
    z_min: float = 0.8,
    z_max: float = 2.1,
    n_random_max: Optional[int] = None,
    rng_seed: int = 0,
    with_weight_fkp: bool = True,
) -> DESICatalog:
    """Read DESI DR1 QSO clustering catalogs (combined N+S galactic caps).

    Parameters
    ----------
    catalog_paths   : iterable of paths, e.g. ['QSO_NGC_clustering.dat.fits',
                                                'QSO_SGC_clustering.dat.fits']
    randoms_paths   : iterable of random FITS paths; can be None to skip
                       loading randoms (we don't need them for analytic RR
                       but they're useful for sanity checks).
    fid_cosmo       : ``DistanceCosmo`` to map (RA, Dec, z) -> comoving xyz.
    z_min, z_max    : DESI QSO BAO uses 0.8 < z < 2.1 for DR1.
    n_random_max    : sub-sample the loaded randoms if too large.

    Returns
    -------
    ``DESICatalog`` with combined N+S sample.
    """
    rng = np.random.default_rng(rng_seed)

    ra_d_l = []; dec_d_l = []; z_d_l = []; w_d_l = []
    for p in catalog_paths:
        ra, dec, z, w = _read_clustering_fits(p, with_weight_fkp=with_weight_fkp)
        m = (z >= z_min) & (z <= z_max) & np.isfinite(w) & (w > 0)
        ra_d_l.append(ra[m]); dec_d_l.append(dec[m])
        z_d_l.append(z[m]); w_d_l.append(w[m])
    ra_data = np.concatenate(ra_d_l); dec_data = np.concatenate(dec_d_l)
    z_data = np.concatenate(z_d_l); w_data = np.concatenate(w_d_l)

    xyz_data = radec_z_to_cartesian(ra_data, dec_data, z_data, fid_cosmo)

    if randoms_paths is not None:
        ra_r_l = []; dec_r_l = []; z_r_l = []; w_r_l = []
        for p in randoms_paths:
            ra, dec, z, w = _read_clustering_fits(p, with_weight_fkp=with_weight_fkp)
            m = (z >= z_min) & (z <= z_max) & np.isfinite(w) & (w > 0)
            ra_r_l.append(ra[m]); dec_r_l.append(dec[m])
            z_r_l.append(z[m]); w_r_l.append(w[m])
        ra_random = np.concatenate(ra_r_l); dec_random = np.concatenate(dec_r_l)
        z_random = np.concatenate(z_r_l); w_random = np.concatenate(w_r_l)
        if n_random_max is not None and len(ra_random) > n_random_max:
            idx = rng.choice(len(ra_random), n_random_max, replace=False)
            ra_random = ra_random[idx]; dec_random = dec_random[idx]
            z_random = z_random[idx]; w_random = w_random[idx]
        xyz_random = radec_z_to_cartesian(ra_random, dec_random, z_random, fid_cosmo)
    else:
        ra_random = np.zeros(0); dec_random = np.zeros(0)
        z_random = np.zeros(0); xyz_random = np.zeros((0, 3))
        w_random = np.zeros(0)

    return DESICatalog(
        ra_data=ra_data, dec_data=dec_data, z_data=z_data,
        xyz_data=xyz_data, w_data=w_data,
        ra_random=ra_random, dec_random=dec_random, z_random=z_random,
        xyz_random=xyz_random, w_random=w_random,
        fid_cosmo=fid_cosmo,
    )


def angular_completeness_from_randoms(
    ra_random: np.ndarray, dec_random: np.ndarray,
    nside: int = 256, w_random: Optional[np.ndarray] = None,
):
    """Build a HEALPix angular completeness map by binning the random
    catalog at the given NSIDE.

    DESI randoms Poisson-sample the angular survey footprint with
    completeness corrections already imprinted (the random density on
    the sky IS the completeness function). Histogramming randoms at
    HEALPix NSIDE gives a continuous mask normalised to peak = 1.

    If per-random weights are provided, the binned counts are weighted
    sums (this captures z-dependent FKP weight, but for the angular
    mask the WEIGHT product is what matters).
    """
    import healpy as hp

    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_random),
                       np.deg2rad(ra_random))
    if w_random is None:
        counts = np.bincount(pix, minlength=npix).astype(np.float64)
    else:
        counts = np.bincount(pix, weights=np.asarray(w_random, dtype=np.float64),
                              minlength=npix).astype(np.float64)
    # normalise to [0, 1] by the median of the populated pixels
    populated = counts[counts > 0]
    if populated.size == 0:
        raise ValueError("no random objects in any pixel")
    rho = np.median(populated)
    mask = np.clip(counts / rho, 0.0, 1.0)
    return mask
