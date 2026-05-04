"""Systematics deprojection for galaxy clustering.

The published Storey-Fisher / Alonso / Piccirilli Quaia analyses build
a multi-template linear model of the per-pixel galaxy density:

    n_obs(p) = n_bar * mask(p) * [1 + sum_k c_k * t_k(p)]

with template maps t_k for stellar density, E(B-V) extinction, ecliptic
latitude, etc. Per-galaxy weights w_i = 1 / [1 + sum_k c_k t_k(Omega_i)]
absorb the systematic over-/under-density and drop into the LS
estimator as DD weighting.

Two paths supported:

  - ``fit_template_weights(...)`` : you supply pre-built template
    maps (e.g. SFD dust at NSIDE=64, Gaia star counts, ...). Linear-
    least-squares fit of c_k against the residual ``n_obs/(n_bar*mask) - 1``,
    then per-galaxy weights from the fitted model.

  - ``data_residual_weights(...)`` : when you don't have external
    templates, build one self-consistently from the data: smooth the
    observed angular residual at scales >> BAO sound horizon (default
    FWHM = 10 deg ~= 500 Mpc/h at z = 1.5), use the smoothed residual
    itself as the "systematic template". Captures dust- / stellar-
    density-like patterns visible in the data without absorbing real
    clustering. Cleanly reproducible without external maps, and the
    same machinery accepts user-supplied templates when available.

Per-galaxy weights then plug into the weighted Landy-Szalay pair
counting in ``twopt_density.projected_xi`` and the analytic-RR wp
estimator in ``twopt_density.analytic_rr``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def galaxy_count_map(ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int):
    """Healpix RING-ordered count of galaxies per pixel."""
    import healpy as hp
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_deg), np.deg2rad(ra_deg))
    return np.bincount(pix, minlength=12 * nside ** 2).astype(np.float64)


def data_residual_weights(
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    mask: np.ndarray, nside: int,
    smoothing_fwhm_deg: float = 10.0,
    mask_threshold: float = 1e-6,
):
    """Bootstrap per-galaxy systematic weights from a smoothed data
    residual map (no external templates needed).

    The recipe:
      1. Build the observed galaxy count map ``n_obs``;
      2. Form the residual ``r(p) = n_obs(p) / (n_bar * mask(p))`` over
         the unmasked pixels (1 if the data perfectly tracks the
         selection function);
      3. Smooth ``r`` with a Gaussian beam of width ``smoothing_fwhm_deg``
         (default 10 deg ~ 500 Mpc/h at z=1.5, well outside the BAO
         sound horizon);
      4. Per-galaxy weight ``w_i = 1 / r_smooth(Omega_i)``, normalised
         so ``mean(w_i) = 1`` over the data.

    The smoothing scale is the key knob: larger than BAO so real
    clustering isn't absorbed, smaller than the survey for the
    correction to do work. Returns the per-galaxy weights and the
    fitted residual map for diagnostics.
    """
    import healpy as hp

    n_obs = galaxy_count_map(ra_deg, dec_deg, nside)
    good = mask >= mask_threshold
    n_bar = n_obs[good].sum() / mask[good].sum()
    expected = n_bar * mask
    residual = np.ones_like(mask)
    residual[good] = n_obs[good] / np.maximum(expected[good], 1e-30)

    # smooth: convert FWHM in deg to sigma in rad
    fwhm_rad = np.deg2rad(smoothing_fwhm_deg)
    smoothed = hp.smoothing(residual * good, fwhm=fwhm_rad)
    # divide by smoothed mask so we recover a smoothed *ratio*, not
    # smoothed numerator / true mask
    smoothed_mask = hp.smoothing(good.astype(np.float64), fwhm=fwhm_rad)
    safe = smoothed_mask > 0.05
    r_smooth = np.ones_like(mask)
    r_smooth[safe] = smoothed[safe] / smoothed_mask[safe]
    r_smooth = np.maximum(r_smooth, 0.1)        # floor to avoid blow-up

    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_deg), np.deg2rad(ra_deg))
    w = 1.0 / r_smooth[pix]
    w = w / w.mean()                              # mean = 1
    return w, r_smooth


def coordinate_templates(nside: int, kinds: Sequence[str] = ("gal", "ecl")):
    """Geometric systematic templates derivable from healpix coordinates
    alone -- no external data required.

    The dominant Quaia angular systematics are stellar density (peaks
    on the galactic plane) and Gaia scanning law (peaks on the ecliptic).
    Both are well captured by smooth functions of galactic / ecliptic
    latitude. These templates regress out the leading-order spatial
    pattern without needing the Gaia source catalog or SFD dust maps.

    ``kinds`` selects which template families to include:
        'gal' : sin(|b_gal|) and sin^2(|b_gal|)
        'ecl' : sin(|b_ecl|) and sin^2(|b_ecl|)
        'gal_cos_l' : cos(l_gal), to pick up galactic-longitude
                       asymmetries (e.g. Magellanic clouds).

    Returns
    -------
    list of (NPIX,) healpix maps, ready to pass to
    ``fit_template_weights(templates=...)``.
    """
    import healpy as hp
    from astropy.coordinates import Galactic, ICRS, BarycentricMeanEcliptic
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    npix = 12 * nside ** 2
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    out = []
    if "gal" in kinds:
        b_gal = sc.galactic.b.to(u.rad).value
        out.append(np.sin(np.abs(b_gal)))
        out.append(np.sin(np.abs(b_gal)) ** 2)
    if "ecl" in kinds:
        b_ecl = sc.transform_to(BarycentricMeanEcliptic()).lat.to(u.rad).value
        out.append(np.sin(np.abs(b_ecl)))
        out.append(np.sin(np.abs(b_ecl)) ** 2)
    if "gal_cos_l" in kinds:
        l_gal = sc.galactic.l.to(u.rad).value
        out.append(np.cos(l_gal))
        out.append(np.sin(l_gal))
    return out


def low_ell_templates(nside: int, lmax: int = 4):
    """Spherical-harmonic Y_ell^m basis for ell = 1..lmax (skipping
    monopole l=0, which is absorbed into the constant).

    These are the smooth, large-scale modes that systematics (dust,
    stars, scanning) tend to dominate. Projecting against them is a
    principled way to remove "low-l contamination" without explicit
    physical templates -- equivalent to the leading terms of any
    smooth real-space template.
    """
    import healpy as hp

    npix = 12 * nside ** 2
    out = []
    for ell in range(1, lmax + 1):
        for m in range(0, ell + 1):
            # one alm with this (l, m) coefficient set to 1
            n_alm = hp.Alm.getsize(lmax)
            alm_real = np.zeros(n_alm, dtype=np.complex128)
            alm_imag = np.zeros(n_alm, dtype=np.complex128)
            idx = hp.Alm.getidx(lmax, ell, m)
            alm_real[idx] = 1.0 + 0j
            real_map = hp.alm2map(alm_real, nside, lmax=lmax)
            out.append(real_map)
            if m > 0:
                alm_imag[idx] = 0 + 1j
                imag_map = hp.alm2map(alm_imag, nside, lmax=lmax)
                out.append(imag_map)
    return out


def fit_template_weights(
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    mask: np.ndarray, nside: int,
    templates: Sequence[np.ndarray],
    mask_threshold: float = 1e-6,
):
    """Linear-least-squares deprojection against user-supplied template
    maps t_k(p) (one per systematic).

    Fits ``residual(p) = 1 + sum_k c_k t_k(p)`` over the unmasked
    pixels, then evaluates the model at each galaxy's pixel and returns
    weights ``w_i = 1 / residual_model(Omega_i)``.

    ``templates`` is a sequence of healpix maps at the same NSIDE as
    ``mask``. Standard Quaia choices: stellar density, E(B-V), ecliptic
    latitude. We mean-centre each template so the c_k fit isolates the
    spatial dependence.
    """
    import healpy as hp

    n_obs = galaxy_count_map(ra_deg, dec_deg, nside)
    good = mask >= mask_threshold
    n_bar = n_obs[good].sum() / mask[good].sum()
    expected = n_bar * mask
    residual = (n_obs - expected) / np.maximum(expected, 1e-30)

    T = np.column_stack([
        (t[good] - t[good].mean()) for t in templates
    ])                                              # (N_good, K)
    y = residual[good]
    # least-squares
    c, *_ = np.linalg.lstsq(T, y, rcond=None)

    # build the model map
    model = np.zeros_like(mask)
    for ck, t in zip(c, templates):
        model = model + ck * (t - t[good].mean())

    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_deg), np.deg2rad(ra_deg))
    w = 1.0 / np.maximum(1.0 + model[pix], 0.1)
    w = w / w.mean()
    return w, c, model
