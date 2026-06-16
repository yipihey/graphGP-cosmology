"""Analytic survey window W(n̂, z) — selection function without MC randoms.

The survey selection is modelled as a separable window

    W(n̂, z) = S_ang(n̂) · S_rad(z)

where ``S_ang`` is the HEALPix angular completeness map (the published
mask, or the completeness reconstructed from the survey randoms) and
``S_rad`` is the smooth radial selection n(z).  Small-scale completeness
effects — fibre collisions (``S_fib``) and redshift failures
(``S_zsucc``) — are *not* baked into the window; they enter as per-galaxy
completeness weights on the data side (``w_cp``, ``w_noz``, ``w_sys``),
exactly as in the BOSS "selection-as-weights" methodology (Ross et al.
2011/2012) and matching Risa's S = S_ang·S_rad·S_fib·S_zsucc decomposition.

The key object is the 3-D expected *random* number density at a comoving
position x = (n̂, χ):

    ρ_W(n̂, χ) ∝ S_ang(n̂) · n_χ(χ) / χ²

(``n_χ`` is the radial number density per unit comoving distance,
∫ n_χ dχ = 1; the 1/χ² converts the radial line density to a volume
density).  This is evaluated **analytically** at any galaxy, grid voxel,
or query point — the survey randoms never have to be instantiated.

Used by the analytic-window FKP-KDE in ``density_field`` to replace the
k-NN-over-randoms denominator, which removes the MC shot noise, the
self-pair singularity, and any radial data/random mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distance import DistanceCosmo, comoving_distance


@dataclass
class SurveyWindow:
    """Separable survey window W(n̂, z) = S_ang(n̂) · S_rad(z).

    Attributes
    ----------
    sel_map : (12·nside²,) HEALPix angular completeness in [0, 1].
    nside : HEALPix NSIDE of ``sel_map``.
    chi_grid, n_chi : radial number density n_χ(χ) tabulated on a grid,
        normalised so ∫ n_χ dχ = 1.
    cosmo : fiducial cosmology for z ↔ χ.
    omega_eff : effective solid angle ∫ S_ang dΩ = mean(sel_map)·4π [sr].
    chi_min, chi_max : radial support of the data.
    """
    sel_map: np.ndarray
    nside: int
    chi_grid: np.ndarray
    n_chi: np.ndarray
    cosmo: DistanceCosmo
    omega_eff: float
    chi_min: float
    chi_max: float

    # ── angular ──────────────────────────────────────────────────────────
    def angular(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """S_ang(n̂) = sel_map at the given sky positions."""
        import healpy as hp
        theta = np.radians(90.0 - np.asarray(dec_deg))
        phi = np.radians(np.asarray(ra_deg))
        ipix = hp.ang2pix(self.nside, theta, phi)
        return np.asarray(self.sel_map)[ipix]

    # ── radial ───────────────────────────────────────────────────────────
    def radial_chi(self, chi: np.ndarray) -> np.ndarray:
        """n_χ(χ) — radial number density per unit comoving distance."""
        return np.interp(np.asarray(chi), self.chi_grid, self.n_chi,
                         left=0.0, right=0.0)

    # ── 3-D expected random density ──────────────────────────────────────
    def density(self, ra_deg, dec_deg, z=None, chi=None) -> np.ndarray:
        """Expected random number density ρ_W ∝ S_ang(n̂)·n_χ(χ)/χ².

        Returns the 3-D density up to the overall constant N_random; the
        FKP-KDE uses it only as a *relative* denominator and renormalises
        the resulting field, so the constant cancels.  Provide either ``z``
        or ``chi`` (comoving distance, Mpc/h).
        """
        import jax.numpy as jnp
        if chi is None:
            if z is None:
                raise ValueError("provide z or chi")
            chi = np.asarray(comoving_distance(
                jnp.asarray(np.asarray(z), dtype=jnp.float64), self.cosmo))
        chi = np.asarray(chi, dtype=np.float64)
        s_ang = self.angular(ra_deg, dec_deg)
        n_r = self.radial_chi(chi)
        chi2 = np.maximum(chi ** 2, 1e-12)
        return s_ang * n_r / chi2


def build_survey_window(
    catalog,
    *,
    n_chi_bins: int = 100,
    kde_bandwidth: float = 0.02,
    sel_map: Optional[np.ndarray] = None,
    nside: Optional[int] = None,
) -> SurveyWindow:
    """Construct the analytic ``SurveyWindow`` for a survey catalog.

    ``S_ang`` is taken from ``catalog.sel_map`` (the published mask or the
    completeness reconstructed from randoms).  ``S_rad`` = n(z) is a small
    Gaussian-KDE smoothing of the *data* redshift histogram, converted to a
    radial number density n_χ(χ).  Completeness weights are handled
    separately by the caller (weighted FKP-KDE numerator).

    Parameters
    ----------
    catalog
        Survey catalog with ``sel_map``, ``nside``, ``z_data``, ``fid_cosmo``.
    n_chi_bins
        Number of radial bins for n_χ(χ).
    kde_bandwidth
        Gaussian KDE bandwidth in z for n(z) (default 0.02 — tighter than
        the analytic-RR default since the field wants n(z) resolved).
    sel_map, nside
        Override the catalog's angular completeness if given.
    """
    import jax.numpy as jnp

    sm = sel_map if sel_map is not None else np.asarray(catalog.sel_map)
    ns = nside if nside is not None else int(catalog.nside)
    cosmo = catalog.fid_cosmo
    z_data = np.asarray(catalog.z_data, dtype=np.float64)

    # Radial number density n_χ(χ) (∫ n_χ dχ = 1) — Gaussian KDE in z.
    z_lo, z_hi = float(z_data.min()), float(z_data.max())
    z_grid = np.linspace(z_lo, z_hi, n_chi_bins + 1)
    z_cen = 0.5 * (z_grid[:-1] + z_grid[1:])
    d = z_cen[:, None] - z_data[None, :]
    nz = np.exp(-0.5 * (d / kde_bandwidth) ** 2).sum(axis=1)
    nz = nz / np.trapezoid(nz, z_cen)
    chi_cen = np.asarray(comoving_distance(
        jnp.asarray(z_cen, dtype=jnp.float64), cosmo))
    dchi_dz = np.gradient(chi_cen, z_cen)
    n_chi = nz / np.maximum(dchi_dz, 1e-12)

    omega_eff = float(np.mean(sm)) * 4.0 * np.pi

    return SurveyWindow(
        sel_map=sm, nside=ns,
        chi_grid=chi_cen, n_chi=n_chi,
        cosmo=cosmo, omega_eff=omega_eff,
        chi_min=float(chi_cen[0]), chi_max=float(chi_cen[-1]),
    )
