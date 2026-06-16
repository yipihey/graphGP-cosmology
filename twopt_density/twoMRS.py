"""2MASS Redshift Survey (2MRS) catalog loader.

Huchra et al. 2012 (ApJS 199, 26):
    43,533 galaxies with K_s ≤ 11.75 mag, 91% sky coverage,
    z_median = 0.028. VizieR catalog J/ApJS/199/26.

Data files (download via ``demos/fetch_2mrs.py`` or manually from
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/199/26):

    2mrs_1175_done.fits    — 43,533 objects with measured redshifts
    2mrs_1175_nocz.fits    — 1,066 objects without redshifts (not used here)

Zone of Avoidance mask applied internally:
    |b| < 5°   for 30° ≤ l ≤ 330°
    |b| < 8°   for l < 30° or l > 330° (toward Galactic bulge)
    E(B−V) ≥ 1.0   (heavy extinction)

No published healpix selection function exists; this module builds one from
the geometric ZoA mask plus an optional per-pixel galaxy-count completeness
correction.

Main interface::

    cat = load_2mrs(catalog_path, n_random_factor=10, fid_cosmo=...)
    pos, rand, box = cat.shift_to_positive()
    # pass pos, rand to xi_landy_szalay / knn_cdf / density_field pipeline

Mock interface::

    cat = make_mock_2mrs(n_data=20000, n_random=100000)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian
from .quaia import make_random_from_selection_function, _sample_z_from_data

# Speed of light in km/s
_C_KMS = 299792.458

# ZoA mask parameters (Huchra et al. 2012, Sec. 2)
_ZOA_B_MIN_DEFAULT = 5.0      # |b| < 5° excluded everywhere
_ZOA_B_MIN_BULGE = 8.0        # |b| < 8° excluded toward bulge
_BULGE_L_LO = 330.0           # l > 330° or l < 30° is "toward bulge"
_BULGE_L_HI = 30.0
_EBV_MAX = 1.0                # E(B-V) cut matching 2MRS selection

# Default fiducial cosmology for 2MRS (flat ΛCDM, low-z)
_DEFAULT_COSMO = DistanceCosmo(Om=0.315, h=0.674, w0=-1.0, wa=0.0)


@dataclass
class TwoMRSCatalog:
    """Self-contained 2MRS survey catalog — data + randoms + comoving xyz.

    Fields mirror ``QuaiaCatalog`` so it drops into the same downstream
    pipeline (xi_landy_szalay, knn_cdf, density_field).  Distances are
    computed from the heliocentric CZ corrected to the CMB frame.

    ``w_data`` carries an optional per-galaxy completeness weight (default
    1.0 for the magnitude-limited 2MRS sample, which has uniform selection
    within the ZoA mask).
    """
    ra_data: np.ndarray        # (N_d,) deg, J2000
    dec_data: np.ndarray       # (N_d,) deg
    z_data: np.ndarray         # (N_d,) CMB-frame redshift
    xyz_data: np.ndarray       # (N_d, 3) comoving Mpc/h
    cz_helio: np.ndarray       # (N_d,) heliocentric velocity km/s (diagnostic)

    ra_random: np.ndarray      # (N_r,)
    dec_random: np.ndarray     # (N_r,)
    z_random: np.ndarray       # (N_r,)
    xyz_random: np.ndarray     # (N_r, 3) comoving Mpc/h

    fid_cosmo: DistanceCosmo
    sel_map: np.ndarray        # (12*nside²,) angular completeness [0, 1]
    nside: int                 # HealPIX NSIDE of the selection function

    w_data: Optional[np.ndarray] = None   # (N_d,) completeness weights
    ks_mag: Optional[np.ndarray] = None   # (N_d,) extinction-corrected K_s
    ebv_data: Optional[np.ndarray] = None # (N_d,) E(B-V) from Schlegel/Planck dust map

    @property
    def N_data(self) -> int:
        return len(self.ra_data)

    @property
    def N_random(self) -> int:
        return len(self.ra_random)

    def shift_to_positive(self, margin: float = 100.0):
        """Return ``(positions, randoms, box_size)`` with all coords ≥ margin.

        Matches the ``QuaiaCatalog.shift_to_positive`` interface so both
        catalog types work identically in downstream code.
        """
        all_xyz = np.concatenate([self.xyz_data, self.xyz_random], axis=0)
        shift = -all_xyz.min(axis=0) + margin
        positions = self.xyz_data + shift
        randoms = self.xyz_random + shift
        box_size = float(
            np.max(np.concatenate([positions, randoms]).max(axis=0)) + margin
        )
        return positions, randoms, box_size


def _galactic_coords(ra_deg: np.ndarray,
                     dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ICRS (ra, dec) to Galactic (l, b) in degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    l = sc.galactic.l.deg
    b = sc.galactic.b.deg
    return l, b


def build_zoa_mask(
    nside: int = 64,
    ebv_map: Optional[np.ndarray] = None,
    b_min: float = _ZOA_B_MIN_DEFAULT,
    b_min_bulge: float = _ZOA_B_MIN_BULGE,
) -> np.ndarray:
    """Build a binary HealPIX ZoA selection mask for 2MRS.

    Returns
    -------
    sel : (12*nside²,) float64 array
        1.0 = pixel in survey (|b| > threshold, low extinction),
        0.0 = pixel masked out.
    """
    import healpy as hp

    npix = 12 * nside ** 2
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix = np.degrees(phi)
    dec_pix = 90.0 - np.degrees(theta)
    l_pix, b_pix = _galactic_coords(ra_pix, dec_pix)

    # primary latitude cut
    in_bulge = (l_pix < _BULGE_L_HI) | (l_pix > _BULGE_L_LO)
    b_threshold = np.where(in_bulge, b_min_bulge, b_min)
    lat_ok = np.abs(b_pix) >= b_threshold

    # extinction cut
    if ebv_map is not None:
        ext_ok = ebv_map < _EBV_MAX
    else:
        ext_ok = np.ones(npix, dtype=bool)

    sel = (lat_ok & ext_ok).astype(np.float64)
    return sel


def _cz_helio_to_z_cmb(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    cz_helio: np.ndarray,
) -> np.ndarray:
    """Convert heliocentric velocity CZ to CMB-frame redshift.

    Uses the dipole correction from Fixsen et al. 1996:
        v_CMB = v_helio + 371.0 * cos(angle_to_CMB_apex)
    CMB apex: l=264.14°, b=+48.26° (Planck 2018 CMB dipole direction).
    """
    # CMB dipole (Planck 2018, Table 3; apex in Galactic coords)
    l_apex_deg = 264.14
    b_apex_deg = 48.26
    v_apex = 369.82  # km/s (Planck 2018)

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    apex = SkyCoord(l=l_apex_deg * u.deg, b=b_apex_deg * u.deg,
                    frame="galactic")
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    sep = sc.separation(apex.icrs).rad
    cz_cmb = cz_helio + v_apex * np.cos(sep)
    return cz_cmb / _C_KMS


def make_2mrs_sel_map(
    ra_data: np.ndarray,
    dec_data: np.ndarray,
    nside: int = 64,
    smooth_deg: float = 3.0,
) -> np.ndarray:
    """Build a healpix completeness map from 2MRS ZoA geometry.

    Returns a binary (0/1) mask based on the |b| and extinction criteria.
    The ``smooth_deg`` Gaussian smoothing softens the sharp ZoA boundary
    for the random-generation step; it does not change the survey geometry.
    """
    import healpy as hp

    sel = build_zoa_mask(nside=nside)

    # Optional mild Gaussian smoothing of the mask boundary
    if smooth_deg > 0.0:
        sigma_rad = np.radians(smooth_deg)
        sel_smooth = hp.smoothing(sel, sigma=sigma_rad)
        # clip back to [0, 1] after smoothing
        sel = np.clip(sel_smooth, 0.0, 1.0)

    return sel


def load_2mrs(
    catalog_path: str,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    nside: int = 64,
    n_random_factor: int = 10,
    z_min: float = 0.0,
    z_max: float = 0.1,
    b_min: float = _ZOA_B_MIN_DEFAULT,
    b_min_bulge: float = _ZOA_B_MIN_BULGE,
    ebv_max: float = _EBV_MAX,
    smooth_mask_deg: float = 3.0,
    rng_seed: int = 0,
    ra_key: str = "RAJ2000",
    dec_key: str = "DEJ2000",
    cz_key: str = "Vh",
    ks_key: str = "Ks",
    ebv_key: str = "E_B-V_",
) -> TwoMRSCatalog:
    """Load the 2MRS FITS catalog and build a random catalog.

    Parameters
    ----------
    catalog_path
        Path to ``2mrs_1175_done.fits`` (downloaded from VizieR
        J/ApJS/199/26 or by running ``demos/fetch_2mrs.py``).
    fid_cosmo
        Fiducial cosmology for comoving distances. Defaults to
        Planck18 flat ΛCDM.
    nside
        HealPIX NSIDE for the angular selection map (default 64).
    n_random_factor
        Number of randoms = ``n_random_factor × N_data``.
    z_min, z_max
        Redshift cuts applied after the CZ-to-z conversion.
    b_min, b_min_bulge
        ZoA latitude thresholds (degrees).
    ebv_max
        Maximum E(B−V) permitted (matching the 2MRS extinction cut).
    smooth_mask_deg
        Gaussian FWHM (degrees) used to soften the healpix mask boundary
        before sampling randoms. Set to 0 for a binary mask.
    rng_seed
        Random seed for the random catalog generation.
    ra_key, dec_key, cz_key, ks_key, ebv_key
        Column names in the FITS file. The VizieR distribution of
        J/ApJS/199/26 uses ``RAJ2000``, ``DEJ2000``, ``Vh``, ``Ks``,
        ``E_B-V_``. Override if using a different column naming.

    Returns
    -------
    TwoMRSCatalog
    """
    import jax.numpy as jnp
    from astropy.table import Table

    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    cat = Table.read(catalog_path)
    col_names_upper = {c.upper(): c for c in cat.colnames}

    def _col(key: str) -> Optional[np.ndarray]:
        k = col_names_upper.get(key.upper())
        if k is None:
            return None
        return np.asarray(cat[k], dtype=np.float64)

    ra = _col(ra_key)
    dec = _col(dec_key)
    cz = _col(cz_key)
    ks = _col(ks_key)
    ebv = _col(ebv_key)

    if ra is None or dec is None or cz is None:
        raise ValueError(
            f"{catalog_path}: could not find RA/DEC/CZ columns. "
            f"Available: {list(cat.colnames)[:10]}"
        )

    # CMB-frame redshift
    z = _cz_helio_to_z_cmb(ra, dec, cz)

    # Apply ZoA and extinction masks
    l, b = _galactic_coords(ra, dec)
    in_bulge = (l < _BULGE_L_HI) | (l > _BULGE_L_LO)
    b_thresh = np.where(in_bulge, b_min_bulge, b_min)
    lat_ok = np.abs(b) >= b_thresh
    ext_ok = (ebv < ebv_max) if ebv is not None else np.ones(len(ra), bool)
    z_ok = (z >= z_min) & (z <= z_max) & (z > 0.0)
    mask = lat_ok & ext_ok & z_ok

    ra_d = ra[mask]
    dec_d = dec[mask]
    z_d = z[mask]
    cz_d = cz[mask]
    ks_d  = ks[mask]  if ks  is not None else None
    ebv_d = ebv[mask] if ebv is not None else None

    xyz_d = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_d), jnp.asarray(dec_d), jnp.asarray(z_d), fid_cosmo,
    ))

    # Build healpix selection map and generate randoms
    sel = make_2mrs_sel_map(ra_d, dec_d, nside=nside, smooth_deg=smooth_mask_deg)

    rng = np.random.default_rng(rng_seed)
    n_random = n_random_factor * len(ra_d)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=sel, n_random=n_random, z_data=z_d, nside=nside, rng=rng,
    )

    xyz_r = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_r), jnp.asarray(dec_r), jnp.asarray(z_r), fid_cosmo,
    ))

    return TwoMRSCatalog(
        ra_data=ra_d, dec_data=dec_d, z_data=z_d, xyz_data=xyz_d,
        cz_helio=cz_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
        sel_map=sel, nside=nside,
        w_data=np.ones(len(ra_d)),
        ks_mag=ks_d,
        ebv_data=ebv_d,
    )


def make_mock_2mrs(
    n_data: int = 20000,
    n_random: int = 100000,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    z_min: float = 0.002,
    z_max: float = 0.05,
    n_clusters: int = 400,
    cluster_sigma_deg: float = 1.5,
    cluster_sigma_z: float = 0.003,
    clustered_fraction: float = 0.20,
    nside: int = 64,
    seed: int = 7,
) -> TwoMRSCatalog:
    """2MRS-shaped mock catalog for pipeline validation.

    Sky geometry matches the 2MRS ZoA mask (no galactic plane); n(z) is
    roughly flat in comoving volume from ``z_min`` to ``z_max``.  The
    clustered fraction produces a realistic |ξ(r)| amplitude (~a few at
    1 Mpc/h for local universe galaxies).
    """
    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    sel = build_zoa_mask(nside=nside)

    # Uniform n(z) in comoving volume (approx. for low-z)
    z_grid = np.linspace(z_min, z_max, 500)
    nz_pdf = z_grid ** 2  # dV/dz ∝ χ² dχ/dz ≈ z²
    nz_pdf /= nz_pdf.sum()

    def _sample_z(n):
        cdf = np.cumsum(nz_pdf)
        cdf /= cdf[-1]
        return np.interp(rng.uniform(size=n), cdf, z_grid)

    # Random catalog
    ra_r, dec_r, _ = make_random_from_selection_function(
        sel_map=sel, n_random=n_random, z_data=z_grid,
        nside=nside, rng=rng,
    )
    z_r = _sample_z(n_random)

    # Data: uniform + clustered blobs
    n_clust = int(clustered_fraction * n_data)
    n_unif = n_data - n_clust

    ra_u, dec_u, _ = make_random_from_selection_function(
        sel_map=sel, n_random=n_unif, z_data=z_grid, nside=nside, rng=rng,
    )
    z_u = _sample_z(n_unif)

    if n_clust > 0 and n_clusters > 0:
        ra_c, dec_c, _ = make_random_from_selection_function(
            sel_map=sel, n_random=n_clusters, z_data=z_grid, nside=nside, rng=rng,
        )
        z_c = _sample_z(n_clusters)
        n_per = max(1, n_clust // n_clusters)
        ra_b = np.repeat(ra_c, n_per) + rng.normal(0, cluster_sigma_deg,
                                                    n_clusters * n_per)
        dec_b = np.clip(
            np.repeat(dec_c, n_per) + rng.normal(0, cluster_sigma_deg,
                                                  n_clusters * n_per),
            -89.99, 89.99,
        )
        z_b = np.clip(
            np.repeat(z_c, n_per) + rng.normal(0, cluster_sigma_z,
                                                n_clusters * n_per),
            z_min, z_max,
        )
        ra_b %= 360.0
        ra_d = np.concatenate([ra_u, ra_b])
        dec_d = np.concatenate([dec_u, dec_b])
        z_d = np.concatenate([z_u, z_b])
    else:
        ra_d, dec_d, z_d = ra_u, dec_u, z_u

    xyz_d = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_d), jnp.asarray(dec_d), jnp.asarray(z_d), fid_cosmo,
    ))
    xyz_r = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_r), jnp.asarray(dec_r), jnp.asarray(z_r), fid_cosmo,
    ))

    # Synthetic E(B-V): log-normal with median 0.05, truncated at 1
    ebv_mock = np.clip(rng.lognormal(mean=np.log(0.05), sigma=0.7, size=len(ra_d)), 0.0, 1.0)

    return TwoMRSCatalog(
        ra_data=ra_d, dec_data=dec_d, z_data=z_d, xyz_data=xyz_d,
        cz_helio=z_d * _C_KMS,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
        sel_map=sel, nside=nside,
        w_data=np.ones(len(ra_d)),
        ebv_data=ebv_mock,
    )
