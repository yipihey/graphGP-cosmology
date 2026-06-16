"""Cosmicflows-4 (CF4) catalog loader.

Tully et al. 2023 (ApJ 944, 94):
    55,877 individual galaxy distances; 38,065 groups.
    z ≤ 0.1 (cz ≤ 30,000 km/s).  VizieR catalog J/ApJ/944/94.

Eight distance indicators: Tully-Fisher (TFR), Fundamental Plane (FP),
Type Ia supernovae (SNe Ia), Surface Brightness Fluctuations (SBF), etc.

Two position modes (``use_distance_modulus`` flag):

    False (default)
        Use CMB-frame redshift z_CMB → comoving distance.
        Consistent with 2MRS / BOSS pipeline; mixes real-space density
        with radial peculiar velocity.

    True (constrained simulation mode)
        Use distance modulus μ → physical distance d = 10^((μ-25)/5) Mpc.
        Removes peculiar-velocity bias; preferred for SIBELIUS/CLUES-style
        constrained initial conditions.

Main interface::

    cat = load_cf4(catalog_path, use_distance_modulus=False, ...)
    pos, rand, box = cat.shift_to_positive()

Data files (download via ``demos/fetch_cf4.py`` or from VizieR J/ApJ/944/94):

    kallcf4.fits          — individual galaxy distances
    kallcf4_groups.fits   — group-averaged catalog (preferred for density field)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian, comoving_distance
from .quaia import make_random_from_selection_function, _sample_z_from_data

_C_KMS = 299792.458
_DEFAULT_COSMO = DistanceCosmo(Om=0.315, h=0.674, w0=-1.0, wa=0.0)


@dataclass
class CF4Catalog:
    """Cosmicflows-4 survey catalog — data + randoms + comoving xyz.

    Mirrors ``QuaiaCatalog`` for compatibility with all downstream code.

    ``use_distance_modulus`` records which position mode was used so
    downstream diagnostics can label plots correctly.

    ``mu_data`` and ``sigma_mu_data`` carry distance moduli and their
    uncertainties when available — useful for peculiar-velocity analysis
    or constrained-simulation weight computation.
    """
    ra_data: np.ndarray        # (N_d,) deg J2000
    dec_data: np.ndarray       # (N_d,) deg
    z_data: np.ndarray         # (N_d,) effective redshift (CMB or from μ)
    xyz_data: np.ndarray       # (N_d, 3) comoving Mpc/h
    z_cmb: np.ndarray          # (N_d,) CMB-frame redshift (always stored)

    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    xyz_random: np.ndarray

    fid_cosmo: DistanceCosmo
    sel_map: np.ndarray        # (12*nside²,) completeness [0, 1]
    nside: int

    use_distance_modulus: bool = False
    mu_data: Optional[np.ndarray] = None       # (N_d,) distance modulus
    sigma_mu_data: Optional[np.ndarray] = None # (N_d,) uncertainty in μ
    n_galaxies_in_group: Optional[np.ndarray] = None  # (N_d,) if group catalog

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

    def peculiar_velocities_km_s(self, H0: float = 67.4) -> np.ndarray:
        """Estimate peculiar velocities v_pec = cz_CMB − H0 × d_Mpc.

        Requires ``mu_data`` to be set.  Returns None otherwise.
        """
        if self.mu_data is None:
            return None
        d_mpc = 10.0 ** ((self.mu_data - 25.0) / 5.0)  # physical Mpc
        cz_cmb = self.z_cmb * _C_KMS
        v_pec = cz_cmb - H0 * d_mpc
        return v_pec


def _build_cf4_sel_map(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int = 64,
    smooth_deg: float = 5.0,
    min_count_frac: float = 0.05,
) -> np.ndarray:
    """Build a data-driven angular completeness map from CF4 positions.

    CF4 has no published healpix selection function.  We estimate completeness
    from the galaxy/group distribution on the sky: pixels are marked as
    'in survey' if their galaxy count is above ``min_count_frac`` times the
    median over populated pixels, after Gaussian smoothing.
    """
    import healpy as hp

    npix = 12 * nside ** 2
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)
    counts = np.bincount(pix, minlength=npix).astype(np.float64)

    # Smooth to fill small gaps
    sigma_rad = np.radians(smooth_deg)
    counts_smooth = hp.smoothing(counts, sigma=sigma_rad)
    counts_smooth = np.clip(counts_smooth, 0.0, None)

    populated = counts_smooth[counts_smooth > 0]
    if populated.size == 0:
        return np.ones(npix)
    threshold = min_count_frac * np.median(populated)
    sel = np.where(counts_smooth > threshold, counts_smooth, 0.0)
    peak = sel.max()
    if peak > 0:
        sel /= peak
    return sel.astype(np.float64)


def _mu_to_z_eff(
    mu: np.ndarray,
    fid_cosmo: DistanceCosmo,
    h: float = 0.674,
    z_max: float = 0.5,
    n_grid: int = 2000,
) -> np.ndarray:
    """Convert distance modulus μ to an effective redshift via D_C(z).

    Inverts D_L(z) = (1+z) D_C(z) = 10^((μ-25)/5) Mpc numerically.
    """
    import jax.numpy as jnp

    d_lum_mpc = 10.0 ** ((mu - 25.0) / 5.0)           # Mpc (physical)
    d_lum_mpch = d_lum_mpc * h                          # Mpc/h

    z_grid = np.linspace(1e-4, z_max, n_grid)
    dc_grid = np.asarray(comoving_distance(jnp.asarray(z_grid), fid_cosmo))
    dl_grid = dc_grid * (1.0 + z_grid)                  # D_L in Mpc/h

    # Invert: find z such that D_L(z) = d_lum_mpch
    z_eff = np.interp(d_lum_mpch, dl_grid, z_grid)
    return z_eff


def load_cf4(
    catalog_path: str,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    use_distance_modulus: bool = False,
    use_groups: bool = False,
    nside: int = 64,
    n_random_factor: int = 10,
    z_min: float = 0.0,
    z_max: float = 0.1,
    smooth_mask_deg: float = 5.0,
    rng_seed: int = 0,
    ra_key: str = "RAJ2000",
    dec_key: str = "DEJ2000",
    z_cmb_key: str = "zCMB",
    mu_key: str = "DM",
    sigma_mu_key: str = "e_DM",
    ngal_key: str = "Ng",
) -> CF4Catalog:
    """Load CF4 FITS catalog and build a random catalog.

    Parameters
    ----------
    catalog_path
        Path to ``kallcf4.fits`` or ``kallcf4_groups.fits`` (VizieR
        J/ApJ/944/94 or from ``demos/fetch_cf4.py``).
    fid_cosmo
        Fiducial cosmology for comoving distances.
    use_distance_modulus
        If True, use distance modulus μ to set comoving positions (removes
        peculiar velocity bias — preferred for constrained simulations).
        If False, use CMB-frame redshift z_CMB.
    use_groups
        If True, expect group-catalog column naming conventions.
    nside
        HealPIX NSIDE for the angular selection map.
    n_random_factor
        Randoms = ``n_random_factor × N_data``.
    z_min, z_max
        Redshift cuts (applied to z_CMB regardless of position mode).
    smooth_mask_deg
        Gaussian smoothing of the data-driven mask (degrees).
    rng_seed
        Random seed.
    ra_key, dec_key, z_cmb_key, mu_key, sigma_mu_key, ngal_key
        Column names in the FITS file.  VizieR defaults above; override
        if using a different table.

    Returns
    -------
    CF4Catalog
    """
    import jax.numpy as jnp
    from astropy.table import Table

    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    cat = Table.read(catalog_path)
    col_up = {c.upper(): c for c in cat.colnames}

    def _col(key: str) -> Optional[np.ndarray]:
        k = col_up.get(key.upper())
        return np.asarray(cat[k], dtype=np.float64) if k else None

    ra = _col(ra_key)
    dec = _col(dec_key)
    z_cmb_raw = _col(z_cmb_key)
    mu = _col(mu_key)
    sigma_mu = _col(sigma_mu_key)
    ngal = _col(ngal_key)

    if ra is None or dec is None:
        raise ValueError(
            f"{catalog_path}: RA/DEC columns not found. "
            f"Available: {list(cat.colnames)[:10]}"
        )

    # CMB-frame redshift (divide by c if stored as velocity in km/s)
    if z_cmb_raw is None:
        raise ValueError(f"{catalog_path}: z_CMB column '{z_cmb_key}' not found.")
    if z_cmb_raw.max() > 10.0:
        z_cmb = z_cmb_raw / _C_KMS
    else:
        z_cmb = z_cmb_raw

    # Filter by z_CMB range and valid distances
    z_ok = (z_cmb >= z_min) & (z_cmb <= z_max) & (z_cmb > 0.0)
    if mu is not None:
        z_ok = z_ok & np.isfinite(mu) & (mu > 0.0)

    ra = ra[z_ok]; dec = dec[z_ok]; z_cmb = z_cmb[z_ok]
    mu = mu[z_ok] if mu is not None else None
    sigma_mu = sigma_mu[z_ok] if sigma_mu is not None else None
    ngal = ngal[z_ok] if ngal is not None else None

    # Choose position mode
    if use_distance_modulus and mu is not None:
        z_pos = _mu_to_z_eff(mu, fid_cosmo)
    else:
        z_pos = z_cmb

    xyz_d = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(z_pos), fid_cosmo,
    ))

    # Build selection map from data distribution
    sel = _build_cf4_sel_map(ra, dec, nside=nside, smooth_deg=smooth_mask_deg)

    rng = np.random.default_rng(rng_seed)
    n_random = n_random_factor * len(ra)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=sel, n_random=n_random, z_data=z_cmb, nside=nside, rng=rng,
    )
    if use_distance_modulus and mu is not None:
        z_r_pos = _mu_to_z_eff(
            np.interp(z_r, z_cmb, mu),   # approximate μ for randoms via n(z) matching
            fid_cosmo,
        )
    else:
        z_r_pos = z_r

    xyz_r = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_r), jnp.asarray(dec_r), jnp.asarray(z_r_pos), fid_cosmo,
    ))

    return CF4Catalog(
        ra_data=ra, dec_data=dec, z_data=z_pos, xyz_data=xyz_d,
        z_cmb=z_cmb,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
        sel_map=sel, nside=nside,
        use_distance_modulus=use_distance_modulus,
        mu_data=mu, sigma_mu_data=sigma_mu,
        n_galaxies_in_group=ngal,
    )


def make_mock_cf4(
    n_data: int = 20000,
    n_random: int = 100000,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    z_min: float = 0.002,
    z_max: float = 0.08,
    n_clusters: int = 300,
    cluster_sigma_deg: float = 2.0,
    cluster_sigma_z: float = 0.005,
    clustered_fraction: float = 0.25,
    nside: int = 64,
    seed: int = 13,
) -> CF4Catalog:
    """CF4-shaped mock with approximate n(z) ∝ z²."""
    if fid_cosmo is None:
        fid_cosmo = _DEFAULT_COSMO

    import jax.numpy as jnp
    from .twoMRS import build_zoa_mask

    rng = np.random.default_rng(seed)

    # Use the 2MRS ZoA mask as a proxy for CF4's incomplete sky
    sel = build_zoa_mask(nside=nside)

    z_grid = np.linspace(z_min, z_max, 500)
    nz = z_grid ** 2
    nz /= nz.sum()

    def _sz(n):
        cdf = np.cumsum(nz); cdf /= cdf[-1]
        return np.interp(rng.uniform(size=n), cdf, z_grid)

    ra_r, dec_r, _ = make_random_from_selection_function(
        sel, n_random, z_data=z_grid, nside=nside, rng=rng,
    )
    z_r = _sz(n_random)

    n_clust = int(clustered_fraction * n_data)
    n_unif = n_data - n_clust
    ra_u, dec_u, _ = make_random_from_selection_function(
        sel, n_unif, z_data=z_grid, nside=nside, rng=rng,
    )
    z_u = _sz(n_unif)

    if n_clust > 0:
        ra_c, dec_c, _ = make_random_from_selection_function(
            sel, n_clusters, z_data=z_grid, nside=nside, rng=rng,
        )
        z_c = _sz(n_clusters)
        n_per = max(1, n_clust // n_clusters)
        ra_b = (np.repeat(ra_c, n_per) +
                rng.normal(0, cluster_sigma_deg, n_clusters * n_per)) % 360
        dec_b = np.clip(np.repeat(dec_c, n_per) +
                        rng.normal(0, cluster_sigma_deg, n_clusters * n_per),
                        -89.99, 89.99)
        z_b = np.clip(np.repeat(z_c, n_per) +
                      rng.normal(0, cluster_sigma_z, n_clusters * n_per),
                      z_min, z_max)
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

    # Synthetic distance moduli and uncertainties (Tully-Fisher / FP typical values)
    from .distance import comoving_distance
    import jax.numpy as jnp
    fid_chi = np.array(comoving_distance(jnp.asarray(z_d), fid_cosmo))
    mu_mock = 5.0 * np.log10(fid_chi * (1.0 + z_d) * 1e6 / 10.0)  # distance modulus
    sigma_mu_mock = np.clip(
        rng.lognormal(mean=np.log(0.40), sigma=0.3, size=len(ra_d)), 0.05, 2.0
    )

    return CF4Catalog(
        ra_data=ra_d, dec_data=dec_d, z_data=z_d, xyz_data=xyz_d,
        z_cmb=z_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
        sel_map=sel, nside=nside,
        mu_data=mu_mock,
        sigma_mu_data=sigma_mu_mock,
    )
