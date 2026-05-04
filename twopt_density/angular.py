"""Angular galaxy auto-power spectrum C_ell^gg with NaMaster.

The Storey-Fisher+24 (and Alonso+24, Piccirilli+24) Quaia-clustering
recipe is the standard masked pseudo-Cl estimator:

    1. Bin data into a healpix pixelisation; weight by the selection
       function to form the overdensity map
       delta_g(p) = n_obs(p) / (n_bar * w(p)) - 1.
    2. Use the selection function w(p) as the *mask* in
       ``pymaster.NmtField``.
    3. Decouple the mode-coupling matrix with ``NmtWorkspace`` /
       ``compute_master`` to get the unbiased C_ell.
    4. Subtract the analytic shot noise N_l = <w^2> Omega_pix /
       <n_obs> = Omega_unmasked / N_total.

This module exposes that pipeline at ~80 lines + tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CellMeasurement:
    """Container for a galaxy auto power-spectrum measurement."""
    ell_eff: np.ndarray          # (N_bin,) effective multipole per bin
    cl_decoupled: np.ndarray     # (N_bin,) shot-noise-subtracted C_ell^gg
    cl_raw: np.ndarray           # (N_bin,) decoupled C_ell *before* shot-noise sub
    n_shot: float                # 1 / n_bar (shot-noise level, per-multipole)
    nside: int
    f_sky: float                 # effective sky fraction = <mask>


def make_overdensity_map(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    mask: np.ndarray,
    nside: int,
    mask_threshold: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Build the Quaia-style galaxy overdensity healpix map.

    ``mask`` is the angular selection function (completeness in [0, 1])
    on the same NSIDE. Pixels with ``mask < mask_threshold`` are
    excluded from the mean and set to zero in the returned map.

    Returns
    -------
    delta_g : (NPIX,) float64 healpix RING-ordered map
    n_bar   : mean galaxy count per pixel inside the unmasked area
              (number per pixel area, used to compute shot noise)
    """
    import healpy as hp

    if mask.size != 12 * nside ** 2:
        raise ValueError(
            f"mask has {mask.size} pixels, expected NSIDE={nside} "
            f"-> {12 * nside ** 2}"
        )
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_deg), np.deg2rad(ra_deg))
    n_obs = np.bincount(pix, minlength=12 * nside ** 2).astype(np.float64)

    good = mask >= mask_threshold
    n_bar = n_obs[good].sum() / mask[good].sum()
    delta = np.zeros_like(n_obs)
    delta[good] = n_obs[good] / (n_bar * mask[good]) - 1.0
    return delta, float(n_bar)


def compute_cl_gg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    mask: np.ndarray,
    nside: int,
    ell_bins: Optional[np.ndarray] = None,
    n_per_bin: int = 16,
    mask_threshold: float = 1e-6,
) -> CellMeasurement:
    """Pseudo-C_ell^gg via NaMaster on the Quaia overdensity map.

    Parameters
    ----------
    ra_deg, dec_deg : object positions in degrees.
    mask : selection function on healpix RING ordering at ``nside``.
    ell_bins : log-spaced multipole bin edges; if None uses
        ``n_per_bin`` linear bins from ell=2 to 3*nside-1.
    """
    import healpy as hp
    import pymaster as nmt

    delta, n_bar = make_overdensity_map(
        ra_deg, dec_deg, mask, nside, mask_threshold=mask_threshold,
    )

    if ell_bins is None:
        b = nmt.NmtBin.from_nside_linear(nside, n_per_bin)
    else:
        ell_bins = np.asarray(ell_bins, dtype=np.int32)
        b = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:], is_Dell=False)
    ell_eff = b.get_effective_ells()

    f = nmt.NmtField(mask, [delta], spin=0)
    cl_coupled = nmt.compute_coupled_cell(f, f)              # (1, n_ell_full)
    wsp = nmt.NmtWorkspace.from_fields(f, f, b)
    cl_decoupled_full = wsp.decouple_cell(cl_coupled)        # (1, N_bin)
    cl_raw = np.asarray(cl_decoupled_full[0], dtype=np.float64)

    # shot noise (Alonso+24 eq 3.6): N_l = Omega_pix * <w^2>_sphere / N_g
    omega_pix = 4.0 * np.pi / (12 * nside ** 2)
    n_total = float(len(ra_deg))
    n_shot = omega_pix * float(np.sum(mask ** 2)) / n_total
    cl_decoupled = cl_raw - n_shot

    f_sky = float(mask.mean())
    return CellMeasurement(
        ell_eff=np.asarray(ell_eff, dtype=np.float64),
        cl_decoupled=cl_decoupled,
        cl_raw=cl_raw,
        n_shot=n_shot,
        nside=nside,
        f_sky=f_sky,
    )


def compute_cl_gkappa(
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    mask_g: np.ndarray, kappa_map: np.ndarray, mask_kappa: np.ndarray,
    nside: int,
    ell_bins: Optional[np.ndarray] = None,
    n_per_bin: int = 16,
    mask_threshold: float = 1e-6,
):
    """Pseudo-Cl cross-correlation of galaxy density and CMB lensing kappa.

    Build the Quaia overdensity map ``delta_g``, then compute the
    NaMaster pseudo-Cl cross-correlation with the input kappa map. The
    combined mask is the product ``mask_g * mask_kappa`` so only pixels
    in the intersection of both surveys contribute.

    Parameters
    ----------
    ra_deg, dec_deg : galaxy positions [deg]
    mask_g          : Quaia angular selection function (NPIX,)
    kappa_map       : Planck-like CMB convergence map (NPIX,)
    mask_kappa      : Planck lensing analysis mask (NPIX,)
    nside           : healpix NSIDE for everything; all maps must agree.
    ell_bins        : multipole bin edges; default = linear binning
                       n_per_bin per bin from ell = 2 to 3 * nside - 1.

    Returns
    -------
    ``CellMeasurement`` with ``cl_decoupled`` = pseudo-Cl^{g-kappa}
    deconvolved against the joint mask. Shot noise on the cross is
    zero so ``n_shot = 0``.
    """
    import healpy as hp
    import pymaster as nmt

    if not (mask_g.size == kappa_map.size == mask_kappa.size == 12 * nside ** 2):
        raise ValueError("all maps must have the same NSIDE")
    delta, _ = make_overdensity_map(
        ra_deg, dec_deg, mask_g, nside, mask_threshold=mask_threshold,
    )
    # joint mask -- pixels where both surveys are valid
    mask_joint = mask_g * mask_kappa

    if ell_bins is None:
        b = nmt.NmtBin.from_nside_linear(nside, n_per_bin)
    else:
        ell_bins = np.asarray(ell_bins, dtype=np.int32)
        b = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:], is_Dell=False)
    ell_eff = b.get_effective_ells()

    f_g = nmt.NmtField(mask_joint, [delta], spin=0)
    f_k = nmt.NmtField(mask_joint, [kappa_map], spin=0)
    cl_coupled = nmt.compute_coupled_cell(f_g, f_k)
    wsp = nmt.NmtWorkspace.from_fields(f_g, f_k, b)
    cl_dec = np.asarray(wsp.decouple_cell(cl_coupled)[0], dtype=np.float64)
    return CellMeasurement(
        ell_eff=np.asarray(ell_eff, dtype=np.float64),
        cl_decoupled=cl_dec,
        cl_raw=cl_dec,
        n_shot=0.0,
        nside=nside,
        f_sky=float(np.mean(mask_joint)),
    )
