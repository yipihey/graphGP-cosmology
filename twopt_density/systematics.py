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
