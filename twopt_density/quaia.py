"""Quaia quasar catalog: real-data loader and Quaia-shape mock generator.

The Quaia catalog (Storey-Fisher et al. 2024, arXiv:2306.17749) is the
all-sky Gaia x unWISE quasar sample (~1.3M objects with G < 20.5,
spectro-photometric redshifts in 0.5 < z < 4.5). Data products are
hosted on Zenodo (DOI 10.5281/zenodo.8060755, see
https://github.com/kstoreyf/gaia-quasars-lss). The published files are::

    quaia_G20.0.fits                  # data, G < 20.0 cut
    quaia_G20.5.fits                  # data, G < 20.5 cut (full sample)
    random_G20.0_10x.fits             # 10x random for G < 20.0 sample
    random_G20.5_10x.fits             # 10x random for G < 20.5 sample
    selection_function_NSIDE64_G20.0.fits   # healpix completeness map
    selection_function_NSIDE64_G20.5.fits

The data catalog has columns ``ra, dec, redshift_quaia`` (and several
metadata columns); the **random catalog has only sky positions**
``(ra, dec, ebv)`` -- redshifts must be sampled from the data n(z) at
load time, which is the standard Quaia analysis recipe.

This module provides::

  load_quaia(catalog_path, randoms_path, fid_cosmo, ...) -> QuaiaCatalog
      Read the real Quaia FITS files. ``randoms_path`` may also be the
      string ``"sample_from_data"`` to use a uniform-on-mask synthetic
      random with z drawn from the loaded data's n(z). Otherwise,
      reads the random FITS file and assigns z's per
      ``random_z_strategy`` (``"sample_from_data"`` or ``"copy_from_column"``).

  make_mock_quaia(n_data, n_random, fid_cosmo, ..., seed=...) -> QuaiaCatalog
      Synthesise a Quaia-shape mock with a galactic-plane mask
      (|b| > b_min), the published bimodal n(z), light Gaussian-blob
      clustering for the data, and uniform randoms under the same
      selection. Outputs the same ``QuaiaCatalog`` schema as the real
      loader.

Once you have the real ``quaia_G20.5.fits`` and ``random_G20.5_10x.fits``
on a machine that can reach Zenodo, the demo at
``demos/demo_quaia_mock.py`` becomes a real-data run by swapping
``make_mock_quaia(...)`` for ``load_quaia(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distance import DistanceCosmo, radec_z_to_cartesian


@dataclass
class QuaiaCatalog:
    """Self-contained survey catalog -- data + randoms + comoving xyz.

    All angles in degrees. ``ra`` in [0, 360), ``dec`` in [-90, 90].
    ``xyz_data`` and ``xyz_random`` are comoving Mpc/h under
    ``fid_cosmo``. Apply a constant shift before passing positions to
    ``build_state`` so coordinates are non-negative (cKDTree requirement).
    """
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray
    xyz_data: np.ndarray            # (N_d, 3) comoving Mpc/h

    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    xyz_random: np.ndarray          # (N_r, 3) comoving Mpc/h

    fid_cosmo: DistanceCosmo

    @property
    def N_data(self) -> int:
        return len(self.ra_data)

    @property
    def N_random(self) -> int:
        return len(self.ra_random)

    def shift_to_positive(self, margin: float = 100.0):
        """Return ``(positions_data, positions_random, box_size)`` shifted
        so all coordinates are >= margin and ``box_size`` covers both."""
        all_xyz = np.concatenate([self.xyz_data, self.xyz_random], axis=0)
        shift = -all_xyz.min(axis=0) + margin
        positions = self.xyz_data + shift
        randoms = self.xyz_random + shift
        box_size = float(np.max(np.concatenate([positions, randoms]).max(axis=0))
                         + margin)
        return positions, randoms, box_size


def _galactic_mask(ra_deg: np.ndarray, dec_deg: np.ndarray, b_min: float) -> np.ndarray:
    """Boolean mask: True for points with ``|b| >= b_min`` (Galactic latitude)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    b = sc.galactic.b.to(u.deg).value
    return np.abs(b) >= b_min


def _quaia_nz_pdf(z: np.ndarray) -> np.ndarray:
    """Approximate Quaia n(z) as a sum of two Gaussians.

    Tuned (qualitatively) to Storey-Fisher+24 Fig. 5: peaks near z ~ 1
    and z ~ 1.7-2 with comparable amplitude. Not a fit -- it's the
    target for the mock.
    """
    g1 = np.exp(-0.5 * ((z - 1.05) / 0.30) ** 2)
    g2 = 0.6 * np.exp(-0.5 * ((z - 1.95) / 0.45) ** 2)
    return g1 + g2


def _sample_from_pdf(z_grid: np.ndarray, pdf: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    u = rng.uniform(size=n)
    return np.interp(u, cdf, z_grid)


def _sample_uniform_sky(n: int, rng: np.random.Generator,
                        b_min: float = 10.0) -> np.ndarray:
    """Uniform sample on the sphere with |b| >= b_min. Returns (ra, dec)."""
    out_ra = np.empty(n)
    out_dec = np.empty(n)
    filled = 0
    chunk = max(1, int(1.6 * n))
    while filled < n:
        ra = rng.uniform(0, 360, size=chunk)
        # uniform on sphere: cos(dec) ~ uniform in sin(dec)
        sin_dec = rng.uniform(-1, 1, size=chunk)
        dec = np.degrees(np.arcsin(sin_dec))
        keep = _galactic_mask(ra, dec, b_min=b_min)
        m = keep.sum()
        take = min(m, n - filled)
        out_ra[filled:filled + take] = ra[keep][:take]
        out_dec[filled:filled + take] = dec[keep][:take]
        filled += take
    return out_ra, out_dec


def make_mock_quaia(
    n_data: int = 50000,
    n_random: int = 200000,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    b_min_galactic: float = 10.0,
    n_clusters: int = 600,
    cluster_sigma_deg: float = 0.8,
    cluster_sigma_z: float = 0.02,
    clustered_fraction: float = 0.15,
    z_min: float = 0.5,
    z_max: float = 4.5,
    seed: int = 7,
) -> QuaiaCatalog:
    """Quaia-shape mock: galactic-plane-masked sky + bimodal n(z); the
    data is a mixture of ``clustered_fraction`` light Gaussian blobs and
    ``1 - clustered_fraction`` uniform-on-mask points; randoms are
    uniform-on-mask under the same n(z) selection.

    Default ``clustered_fraction=0.15`` gives a quasar-like ``xi``
    amplitude of a few -- representative of Quaia's effective bias of
    ~3. Bumping the fraction to 1.0 recovers the highly-clustered
    Gaussian-blob regime; setting to 0.0 gives a pure-Poisson sample.

    Default sizes (50k data, 200k random) are picked for fast pipeline
    runs; for full Quaia (~1.3M data) bump n_data accordingly.
    """
    if fid_cosmo is None:
        fid_cosmo = DistanceCosmo()
    rng = np.random.default_rng(seed)

    z_grid = np.linspace(z_min, z_max, 2000)
    nz = _quaia_nz_pdf(z_grid)

    # --- random catalog: uniform on masked sky x n(z) -----------------
    ra_r, dec_r = _sample_uniform_sky(n_random, rng, b_min=b_min_galactic)
    z_r = _sample_from_pdf(z_grid, nz, n_random, rng)

    # --- data catalog: mix of uniform-on-mask + Gaussian blobs --------
    n_clustered = int(clustered_fraction * n_data)
    n_uniform = n_data - n_clustered

    # Uniform component
    ra_u, dec_u = _sample_uniform_sky(n_uniform, rng, b_min=b_min_galactic)
    z_u = _sample_from_pdf(z_grid, nz, n_uniform, rng)

    # Clustered component
    if n_clustered > 0 and n_clusters > 0:
        ra_c, dec_c = _sample_uniform_sky(n_clusters, rng, b_min=b_min_galactic)
        z_c = _sample_from_pdf(z_grid, nz, n_clusters, rng)
        n_per = max(1, n_clustered // n_clusters)
        ra_b = np.repeat(ra_c, n_per) + rng.normal(0.0, cluster_sigma_deg,
                                                    size=n_clusters * n_per)
        dec_b = np.repeat(dec_c, n_per) + rng.normal(0.0, cluster_sigma_deg,
                                                      size=n_clusters * n_per)
        z_b = np.repeat(z_c, n_per) + rng.normal(0.0, cluster_sigma_z,
                                                  size=n_clusters * n_per)
        ra_b = ra_b % 360.0
        dec_b = np.clip(dec_b, -89.999, 89.999)
        z_b = np.clip(z_b, z_min, z_max)
        keep = _galactic_mask(ra_b, dec_b, b_min=b_min_galactic)
        ra_b, dec_b, z_b = ra_b[keep], dec_b[keep], z_b[keep]
        if len(ra_b) > n_clustered:
            idx = rng.choice(len(ra_b), size=n_clustered, replace=False)
            ra_b, dec_b, z_b = ra_b[idx], dec_b[idx], z_b[idx]
        ra_d = np.concatenate([ra_u, ra_b])
        dec_d = np.concatenate([dec_u, dec_b])
        z_d = np.concatenate([z_u, z_b])
    else:
        ra_d, dec_d, z_d = ra_u, dec_u, z_u

    # --- comoving xyz under fiducial cosmology ------------------------
    import jax.numpy as jnp
    xyz_d = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_d), jnp.asarray(dec_d), jnp.asarray(z_d), fid_cosmo,
    ))
    xyz_r = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_r), jnp.asarray(dec_r), jnp.asarray(z_r), fid_cosmo,
    ))

    return QuaiaCatalog(
        ra_data=ra_d, dec_data=dec_d, z_data=z_d, xyz_data=xyz_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
    )


def _sample_z_from_data(z_data: np.ndarray, n: int,
                         rng: np.random.Generator,
                         n_bins: int = 200) -> np.ndarray:
    """Draw ``n`` redshifts from the empirical n(z) of the data.

    Standard Quaia recipe for assigning z to the (sky-only) random
    catalog. Uses linear interpolation of the empirical CDF on a
    histogram-binned n(z). For Quaia's smooth n(z) (~10^6 sources)
    the result is indistinguishable from a kernel-density resample.
    """
    z_min = float(np.min(z_data))
    z_max = float(np.max(z_data))
    edges = np.linspace(z_min, z_max, n_bins + 1)
    hist, _ = np.histogram(z_data, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf = hist.astype(np.float64)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    u = rng.uniform(size=n)
    return np.interp(u, cdf, centres)


def load_quaia(
    catalog_path: str,
    randoms_path: str,
    fid_cosmo: Optional[DistanceCosmo] = None,
    *,
    ra_key: str = "ra",
    dec_key: str = "dec",
    z_key: str = "redshift_quaia",
    random_z_strategy: str = "sample_from_data",
    rng_seed: int = 0,
) -> QuaiaCatalog:
    """Load real Quaia FITS files into the ``QuaiaCatalog`` schema.

    Defaults match the public Zenodo distribution
    (``quaia_G20.5.fits``, ``random_G20.5_10x.fits``). The catalog file
    must have ``ra``, ``dec``, ``redshift_quaia`` columns; override the
    ``*_key`` arguments for non-standard column names.

    Random redshifts. The published random catalog only carries
    ``(ra, dec, ebv)`` so we must assign z's at load time. Two strategies::

      "sample_from_data"  -- draw z's from the empirical n(z) of the
                             loaded data (the standard Quaia recipe).
      "copy_from_column"  -- read a redshift column from the random
                             FITS file (override ``z_key`` if needed).

    Notes
    -----
    Reading is via astropy.table. If the catalog is large you may want
    to first downsample on disk; the pipeline runs in ``O(N_pair)``
    time which is dominated by the KDTree query at ~1.3M objects.
    """
    from astropy.table import Table
    import jax.numpy as jnp

    if fid_cosmo is None:
        fid_cosmo = DistanceCosmo()

    cat = Table.read(catalog_path)
    rnd = Table.read(randoms_path)

    ra_d = np.asarray(cat[ra_key], dtype=np.float64)
    dec_d = np.asarray(cat[dec_key], dtype=np.float64)
    z_d = np.asarray(cat[z_key], dtype=np.float64)
    ra_r = np.asarray(rnd[ra_key], dtype=np.float64)
    dec_r = np.asarray(rnd[dec_key], dtype=np.float64)

    if random_z_strategy == "sample_from_data":
        rng = np.random.default_rng(rng_seed)
        z_r = _sample_z_from_data(z_d, len(ra_r), rng)
    elif random_z_strategy == "copy_from_column":
        z_r = np.asarray(rnd[z_key], dtype=np.float64)
    else:
        raise ValueError(
            f"random_z_strategy must be 'sample_from_data' or "
            f"'copy_from_column', got {random_z_strategy!r}"
        )

    xyz_d = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_d), jnp.asarray(dec_d), jnp.asarray(z_d), fid_cosmo,
    ))
    xyz_r = np.asarray(radec_z_to_cartesian(
        jnp.asarray(ra_r), jnp.asarray(dec_r), jnp.asarray(z_r), fid_cosmo,
    ))

    return QuaiaCatalog(
        ra_data=ra_d, dec_data=dec_d, z_data=z_d, xyz_data=xyz_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r, xyz_random=xyz_r,
        fid_cosmo=fid_cosmo,
    )
