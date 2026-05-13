"""Build per-PHOTSYS-region HEALPix angular completeness maps for DESI Y1 QSO.

DESI builds its random catalogue independently per photometric region
(BASS/MzLS = "N", DECaLS = "S" — and DES as a third region in DR1, but
DES isn't present in our z-cut Y1 sample). Each region has a different
target-selection function, different imaging-systematics weights, and
its own data n(z) used as the redshift-shuffling pool. To match this
on the random side we need a separate angular completeness map per
region; the per-region random factorises exactly as
``W_R(alpha, dec) * n_R(z)``.

Outputs:

- ``data/desi/desi_qso_y1_completeness_N_NSIDE64.fits``  (BASS/MzLS)
- ``data/desi/desi_qso_y1_completeness_S_NSIDE64.fits``  (DECaLS)
- ``data/desi/desi_qso_y1_completeness_NSIDE64.fits``    (legacy global,
                                                          DEPRECATED)

The legacy global map is still written for back-compat with old
artifacts; it should be removed in a follow-up after all consumers
have migrated.

The recipe (per region) is unchanged from the legacy single-region
build: histogram data positions at NSIDE, smooth at half-pixel scale
to fill survey-edge pixels, threshold at 5% of the in-survey median,
normalise to [0, 1].
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from twopt_density.distance import DistanceCosmo
from twopt_density.desi import load_desi_qso


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "desi")
ARTIFACT_N = os.path.join(DATA_DIR, "desi_qso_y1_completeness_N_NSIDE64.fits")
ARTIFACT_S = os.path.join(DATA_DIR, "desi_qso_y1_completeness_S_NSIDE64.fits")
ARTIFACT_GLOBAL = os.path.join(DATA_DIR, "desi_qso_y1_completeness_NSIDE64.fits")


PHOTSYS_DEC_BOUNDARY = 32.375  # BASS/MzLS (N) above; DECaLS (S) at or below


def _build_region_map(ra_deg: np.ndarray, dec_deg: np.ndarray,
                      nside: int,
                      dec_cut: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Histogram + smooth + threshold + normalise one region's data
    positions into a HEALPix completeness map.

    Parameters
    ----------
    dec_cut
        Optional ``(dec_lo, dec_hi)`` in degrees. After smoothing,
        zero out pixels whose centre Dec falls outside ``[dec_lo,
        dec_hi]``. Used to enforce the PHOTSYS boundary at +32.375 deg
        as a hard mask, so per-region maps don't overlap from the
        smoothing kernel bleeding across.
    """
    import healpy as hp

    npix = 12 * nside ** 2
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    ipix = hp.ang2pix(nside, theta, phi)
    counts = np.bincount(ipix, minlength=npix).astype(np.float64)

    # Smooth at half a pixel scale so randoms aren't quantised to
    # pixels with zero data inside an otherwise-filled neighbourhood
    # (typical of survey edges).
    sigma_rad = float(
        np.deg2rad(np.sqrt(4 * np.pi / npix) * 180.0 / np.pi * 0.5))
    smoothed = hp.smoothing(counts, sigma=sigma_rad, verbose=False)
    smoothed = np.maximum(smoothed, 0.0)

    in_survey = smoothed > 0.05 * np.median(smoothed[counts > 0])

    if dec_cut is not None:
        dec_lo, dec_hi = dec_cut
        theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
        dec_pix = 90.0 - np.degrees(theta_pix)
        in_survey &= (dec_pix >= dec_lo) & (dec_pix <= dec_hi)

    sel = np.zeros(npix, dtype=np.float64)
    if smoothed[in_survey].max() > 0:
        sel[in_survey] = smoothed[in_survey] / smoothed[in_survey].max()
    return sel


def main():
    nside = int(os.environ.get("PAPER_NSIDE_MASK", 64))
    npix = 12 * nside ** 2

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading DESI Y1 QSO ...")
    cat = load_desi_qso(
        catalog_paths=[
            os.path.join(DATA_DIR, "QSO_NGC_clustering.dat.fits"),
            os.path.join(DATA_DIR, "QSO_SGC_clustering.dat.fits"),
        ],
        randoms_paths=None,
        fid_cosmo=fid, z_min=0.8, z_max=2.1, with_weight_fkp=True,
        with_photsys=True,
    )
    print(f"  N_data total = {cat.ra_data.size}")
    if cat.photsys_data.size != cat.ra_data.size:
        raise SystemExit(
            "PHOTSYS column not present in catalog FITS — cannot build "
            "per-region completeness maps. Confirm the loader read it."
        )

    import healpy as hp

    region_dec_cuts = {
        "N": (PHOTSYS_DEC_BOUNDARY, 90.0),
        "S": (-90.0, PHOTSYS_DEC_BOUNDARY),
    }
    sel_per_region: dict[str, np.ndarray] = {}
    for region in ("N", "S"):
        m = cat.photsys_data == region
        n_r = int(m.sum())
        if n_r == 0:
            print(f"  region {region!r}: 0 data — skipping")
            continue
        ra_r = cat.ra_data[m]; dec_r = cat.dec_data[m]
        sel = _build_region_map(ra_r, dec_r, nside=nside,
                                dec_cut=region_dec_cuts[region])
        sel_per_region[region] = sel
        n_in = int((sel > 0).sum())
        f_in = n_in / npix
        print(f"  region {region!r}: N_data={n_r}, "
              f"in-survey pixels {n_in}/{npix} ({f_in*100:.1f}% of sky), "
              f"area ~ {f_in * 41253:.0f} deg^2, "
              f"Dec range [{dec_r.min():.2f}, {dec_r.max():.2f}]")

    if "N" in sel_per_region and "S" in sel_per_region:
        overlap = int(((sel_per_region["N"] > 0)
                       & (sel_per_region["S"] > 0)).sum())
        print(f"  overlap pixels (N & S): {overlap}  "
              "(expected 0 — regions are Dec-disjoint at +32.375 deg)")

    if "N" in sel_per_region:
        hp.write_map(ARTIFACT_N, sel_per_region["N"], overwrite=True,
                     dtype=np.float64)
        print(f"  wrote {ARTIFACT_N} ({os.path.getsize(ARTIFACT_N)/1024:.0f} KB)")
    if "S" in sel_per_region:
        hp.write_map(ARTIFACT_S, sel_per_region["S"], overwrite=True,
                     dtype=np.float64)
        print(f"  wrote {ARTIFACT_S} ({os.path.getsize(ARTIFACT_S)/1024:.0f} KB)")

    # Legacy global map (DEPRECATED): rebuild from full catalog so the
    # old artifact stays consistent if any consumer still reads it.
    print("\nbuilding legacy global map (DEPRECATED; drop in a follow-up)...")
    sel_global = _build_region_map(cat.ra_data, cat.dec_data, nside=nside)
    n_in = int((sel_global > 0).sum())
    print(f"  global: in-survey pixels {n_in}/{npix} "
          f"({n_in/npix*100:.1f}% of sky)")
    hp.write_map(ARTIFACT_GLOBAL, sel_global, overwrite=True,
                 dtype=np.float64)
    print(f"  wrote {ARTIFACT_GLOBAL} "
          f"({os.path.getsize(ARTIFACT_GLOBAL)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
