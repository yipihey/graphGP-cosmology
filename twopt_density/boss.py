"""BOSS DR12 catalog loader — simBIG clean subsamples.

Implements the exact LOWZ-SGC and CMASS-SGC selection used in:

    simBIG (Hahn et al. 2023, arXiv:2310.15256):
        LOWZ-South   0.20 < z < 0.37   RA < 28° or RA > 335°, Dec > −6°
        CMASS-South  0.45 < z < 0.60   same RA / Dec boundary

Data files from SDSS DR12 (download via ``demos/fetch_boss.py``):

    galaxy_DR12v5_LOWZ_South.fits.gz
    galaxy_DR12v5_CMASS_South.fits.gz
    random0_DR12v5_LOWZ_South.fits.gz   (or random0..17)
    random0_DR12v5_CMASS_South.fits.gz

Standard BOSS column conventions:
    RA, DEC, Z                spectroscopic redshift
    WEIGHT_SYSTOT             imaging systematics weight
    WEIGHT_NOZ                redshift-failure upweighting
    WEIGHT_CP                 close-pair (fiber collision) upweighting
    WEIGHT_FKP                FKP weight (P₀ = 10^4 h⁻³ Mpc³)
    NZ                        n(z) at object's redshift

Combined per-object weight follows Ross et al. 2017:
    WEIGHT_TOTAL = (WEIGHT_SYSTOT × WEIGHT_NOZ × WEIGHT_CP) × WEIGHT_FKP

Main interface::

    cat = load_boss(data_paths, randoms_paths, sample="CMASS", ...)
    pos, rand, box = cat.shift_to_positive()

``BOSSCatalog`` is a drop-in replacement for ``DESICatalog`` in all
downstream code — same field names, same weight columns, same regionalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian
from .desi import (
    DESICatalog,
    _read_clustering_fits,
    angular_completeness_from_randoms,
)
from .quaia import _sample_z_from_data

_DEFAULT_COSMO = DistanceCosmo(Om=0.315, h=0.674, w0=-1.0, wa=0.0)


# ──────────────────────────────────────────────────────────────────────
# simBIG clean-subsample specifications (Hahn et al. 2023, App. B)
# ──────────────────────────────────────────────────────────────────────
SIMBIG_CUTS = {
    "LOWZ": dict(
        z_min=0.20, z_max=0.37,
        ra_lo=None, ra_hi=None,   # applied via _in_sgc_footprint
        dec_min=-6.0,
    ),
    "CMASS": dict(
        z_min=0.45, z_max=0.60,
        ra_lo=None, ra_hi=None,
        dec_min=-6.0,
    ),
}

# SGC RA boundary: (RA < 28°) OR (RA > 335°)
_SGC_RA_HI = 28.0
_SGC_RA_LO = 335.0


def _in_sgc_footprint(ra: np.ndarray, dec: np.ndarray,
                      dec_min: float = -6.0) -> np.ndarray:
    """Boolean mask for the simBIG South Galactic Cap footprint."""
    ra_ok = (ra < _SGC_RA_HI) | (ra > _SGC_RA_LO)
    dec_ok = dec > dec_min
    return ra_ok & dec_ok


@dataclass
class BOSSCatalog:
    """BOSS DR12 survey catalog — mirrors DESICatalog for drop-in use.

    ``sample`` is 'LOWZ', 'CMASS', or 'LOWZ+CMASS' to label which
    simBIG subsample was loaded.  ``photsys_data`` is kept as a stub
    (always 'S' for SGC) so code that splits by region still works.
    """
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray
    xyz_data: np.ndarray
    w_data: np.ndarray            # WEIGHT_TOTAL = (SYS×NOZ×CP) × FKP

    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    xyz_random: np.ndarray
    w_random: np.ndarray

    fid_cosmo: DistanceCosmo
    sel_map: np.ndarray           # (12*nside²,) completeness from randoms
    nside: int

    sample: str = "CMASS"         # 'LOWZ', 'CMASS', or 'LOWZ+CMASS'
    photsys_data: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="U1"))
    photsys_random: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype="U1"))

    # Individual BOSS weight components (None for mock catalogs).
    # Stored so weight_budget() can decompose the total correction per galaxy.
    w_sys_data:  Optional[np.ndarray] = None  # WEIGHT_SYSTOT (dust+imaging systematics)
    w_noz_data:  Optional[np.ndarray] = None  # WEIGHT_NOZ    (redshift failure)
    w_cp_data:   Optional[np.ndarray] = None  # WEIGHT_CP     (fiber collision close pairs)
    w_fkp_data:  Optional[np.ndarray] = None  # WEIGHT_FKP    (FKP statistical weight)

    @property
    def N_data(self) -> int:
        return len(self.ra_data)

    @property
    def N_random(self) -> int:
        return len(self.ra_random)

    def shift_to_positive(self, margin: float = 100.0):
        all_xyz = np.concatenate([self.xyz_data, self.xyz_random], axis=0)
        shift = -all_xyz.min(axis=0) + margin
        positions = self.xyz_data + shift
        randoms = self.xyz_random + shift
        box_size = float(
            np.max(np.concatenate([positions, randoms]).max(axis=0)) + margin
        )
        return positions, randoms, box_size


def _read_boss_fits(
    path: str,
    with_weight_fkp: bool = True,
) -> tuple:
    """Read RA/DEC/Z and all weight components from a BOSS LSS FITS file.

    Returns (ra, dec, z, w_total, w_sys, w_noz, w_cp, w_fkp) so callers
    can preserve individual components for weight-budget decomposition.
    Falls back gracefully when any column is absent (treats as 1).
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        cols = {c.upper() for c in t.columns.names}

        def _get(name, default=1.0):
            if name.upper() in cols:
                return np.asarray(t[name], dtype=np.float64)
            return np.full(len(t), default, dtype=np.float64)

        ra = np.asarray(t["RA"], dtype=np.float64)
        dec = np.asarray(t["DEC"], dtype=np.float64)
        z = np.asarray(t["Z"], dtype=np.float64)

        w_sys = _get("WEIGHT_SYSTOT")
        w_noz = _get("WEIGHT_NOZ")
        w_cp  = _get("WEIGHT_CP")
        w_fkp = _get("WEIGHT_FKP") if with_weight_fkp else np.ones(len(t))
        w = w_sys * w_noz * w_cp * (w_fkp if with_weight_fkp else 1.0)

    return ra, dec, z, w, w_sys, w_noz, w_cp, w_fkp


def load_boss(
    data_paths: Iterable[str],
    randoms_paths: Optional[Iterable[str]] = None,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    sample: str = "CMASS",
    nside: int = 256,
    n_random_max: Optional[int] = None,
    with_weight_fkp: bool = True,
    simbig_sgc_cuts: bool = True,
    rng_seed: int = 0,
) -> BOSSCatalog:
    """Load BOSS DR12 LOWZ or CMASS South catalog (simBIG subsamples).

    Parameters
    ----------
    data_paths
        Iterable of galaxy FITS paths, e.g.
        ``['galaxy_DR12v5_CMASS_South.fits.gz']``.
    randoms_paths
        Iterable of random FITS paths (one or more realizations).
        If None, random catalog is empty (you can still compute xi(r)
        analytically with the selection function).
    fid_cosmo
        Fiducial cosmology for comoving distances.
    sample
        ``'LOWZ'`` or ``'CMASS'``.  Controls which simBIG redshift and
        footprint cuts are applied.
    nside
        HealPIX NSIDE for the angular completeness map built from randoms.
    n_random_max
        Sub-sample randoms to this many objects if provided.
    with_weight_fkp
        Whether to multiply WEIGHT_FKP into the per-object weight.
    simbig_sgc_cuts
        If True, apply the exact simBIG SGC footprint cuts (RA, Dec, z-range
        from ``SIMBIG_CUTS``).  Set False to load the full file contents.
    rng_seed
        Random seed for any sub-sampling.

    Returns
    -------
    BOSSCatalog
    """
    rng = np.random.default_rng(rng_seed)
    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    cuts = SIMBIG_CUTS.get(sample.upper(), {})
    z_min = cuts.get("z_min", 0.0)
    z_max = cuts.get("z_max", 1.0)
    dec_min = cuts.get("dec_min", -90.0)

    ra_l, dec_l, z_l, w_l = [], [], [], []
    ws_l, wn_l, wc_l, wf_l = [], [], [], []   # individual components
    for p in data_paths:
        ra, dec, z, w, w_sys, w_noz, w_cp, w_fkp = _read_boss_fits(
            p, with_weight_fkp=with_weight_fkp)
        m = np.isfinite(w) & (w > 0) & (z >= z_min) & (z <= z_max)
        if simbig_sgc_cuts:
            m = m & _in_sgc_footprint(ra, dec, dec_min=dec_min)
        ra_l.append(ra[m]); dec_l.append(dec[m])
        z_l.append(z[m]); w_l.append(w[m])
        ws_l.append(w_sys[m]); wn_l.append(w_noz[m])
        wc_l.append(w_cp[m]);  wf_l.append(w_fkp[m])

    ra_data = np.concatenate(ra_l)
    dec_data = np.concatenate(dec_l)
    z_data = np.concatenate(z_l)
    w_data = np.concatenate(w_l)
    w_sys_data = np.concatenate(ws_l)
    w_noz_data = np.concatenate(wn_l)
    w_cp_data  = np.concatenate(wc_l)
    w_fkp_data = np.concatenate(wf_l)
    xyz_data = radec_z_to_cartesian(ra_data, dec_data, z_data, fid_cosmo)

    if randoms_paths is not None:
        ra_rl, dec_rl, z_rl, w_rl = [], [], [], []
        for p in randoms_paths:
            ra, dec, z, w, *_ = _read_boss_fits(p, with_weight_fkp=with_weight_fkp)
            m = np.isfinite(w) & (w > 0) & (z >= z_min) & (z <= z_max)
            if simbig_sgc_cuts:
                m = m & _in_sgc_footprint(ra, dec, dec_min=dec_min)
            ra_rl.append(ra[m]); dec_rl.append(dec[m])
            z_rl.append(z[m]); w_rl.append(w[m])

        ra_random = np.concatenate(ra_rl)
        dec_random = np.concatenate(dec_rl)
        z_random = np.concatenate(z_rl)
        w_random = np.concatenate(w_rl)

        if n_random_max is not None and len(ra_random) > n_random_max:
            idx = rng.choice(len(ra_random), n_random_max, replace=False)
            ra_random = ra_random[idx]; dec_random = dec_random[idx]
            z_random = z_random[idx]; w_random = w_random[idx]

        xyz_random = radec_z_to_cartesian(ra_random, dec_random, z_random,
                                          fid_cosmo)
        sel_map = angular_completeness_from_randoms(
            ra_random, dec_random, nside=nside, w_random=w_random,
        )
    else:
        ra_random = dec_random = z_random = w_random = np.zeros(0)
        xyz_random = np.zeros((0, 3))
        sel_map = np.zeros(12 * nside ** 2)

    # Label all objects as South (SGC)
    photsys_d = np.full(len(ra_data), "S", dtype="U1")
    photsys_r = np.full(len(ra_random), "S", dtype="U1")

    return BOSSCatalog(
        ra_data=ra_data, dec_data=dec_data, z_data=z_data,
        xyz_data=np.asarray(xyz_data), w_data=w_data,
        ra_random=ra_random, dec_random=dec_random,
        z_random=z_random, xyz_random=np.asarray(xyz_random),
        w_random=w_random,
        fid_cosmo=fid_cosmo,
        sel_map=sel_map, nside=nside,
        sample=sample,
        photsys_data=photsys_d, photsys_random=photsys_r,
        w_sys_data=w_sys_data, w_noz_data=w_noz_data,
        w_cp_data=w_cp_data,   w_fkp_data=w_fkp_data,
    )


def make_mock_boss(
    sample: str = "CMASS",
    n_data: int = 5000,
    n_random: int = 25000,
    fid_cosmo: Optional[DistanceCosmo] = None,
    nside: int = 64,
    seed: int = 0,
) -> BOSSCatalog:
    """Build a mock BOSS catalog for testing without real data.

    Generates galaxies and randoms uniformly within the simBIG SGC footprint
    (RA < 28° or RA > 335°, Dec > −6°) and the sample's z-range.
    """
    rng = np.random.default_rng(seed)
    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    cuts = SIMBIG_CUTS[sample]
    z_min, z_max, dec_min = cuts["z_min"], cuts["z_max"], cuts["dec_min"]

    def _sample_sgc(n: int) -> tuple:
        ra_out, dec_out, z_out = [], [], []
        while sum(len(r) for r in ra_out) < n:
            batch = n * 4
            # RA uniform [0, 360), Dec with cos-weighted sampling
            ra_b = rng.uniform(0, 360, batch)
            sin_dec_min = np.sin(np.radians(dec_min))
            sin_dec_hi = 1.0
            sin_dec_b = rng.uniform(sin_dec_min, sin_dec_hi, batch)
            dec_b = np.degrees(np.arcsin(sin_dec_b))
            z_b = rng.uniform(z_min, z_max, batch)
            mask = _in_sgc_footprint(ra_b, dec_b, dec_min=dec_min)
            ra_out.append(ra_b[mask]); dec_out.append(dec_b[mask])
            z_out.append(z_b[mask])

        ra_all = np.concatenate(ra_out)[:n]
        dec_all = np.concatenate(dec_out)[:n]
        z_all = np.concatenate(z_out)[:n]
        return ra_all, dec_all, z_all

    ra_data, dec_data, z_data = _sample_sgc(n_data)
    ra_random, dec_random, z_random = _sample_sgc(n_random)

    # Uniform weights for mock
    w_data = np.ones(n_data)
    w_random = np.ones(n_random)

    xyz_data = np.asarray(radec_z_to_cartesian(ra_data, dec_data, z_data, fid_cosmo))
    xyz_random = np.asarray(radec_z_to_cartesian(ra_random, dec_random, z_random, fid_cosmo))

    sel_map = angular_completeness_from_randoms(ra_random, dec_random,
                                                nside=nside, w_random=w_random)

    photsys_d = np.full(n_data, "S", dtype="U1")
    photsys_r = np.full(n_random, "S", dtype="U1")

    return BOSSCatalog(
        ra_data=ra_data, dec_data=dec_data, z_data=z_data,
        xyz_data=xyz_data, w_data=w_data,
        ra_random=ra_random, dec_random=dec_random,
        z_random=z_random, xyz_random=xyz_random,
        w_random=w_random,
        fid_cosmo=fid_cosmo,
        sel_map=sel_map, nside=nside,
        sample=sample,
        photsys_data=photsys_d, photsys_random=photsys_r,
    )
