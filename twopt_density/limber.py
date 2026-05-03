"""Limber-projected angular power spectrum C_ell^gg and wp(rp).

For a galaxy auto-correlation in the small-angle / Limber limit::

    C_ell^gg = int dz [W_g(z)]^2 / [chi^2(z) * dchi/dz] *
               b^2(z) * P_NL(k = (ell + 1/2) / chi(z), z)

with W_g(z) = (1/N_total) dN/dz the unit-normalised redshift kernel.

The projected real-space correlation function is

    wp(rp) = 2 int_0^pi_max xi_real(s = sqrt(rp^2 + pi^2), z_eff) d(pi)

with xi_real evaluated at an effective redshift via FFTLog of P_NL.

Both are implemented against syren-halofit P_NL via
``twopt_density.cosmology.run_halofit`` and the comoving-distance
machinery in ``twopt_density.distance``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .cosmology import halofit_from_plin, plin_emulated, run_halofit
from .distance import C_OVER_H100_MPCH, DistanceCosmo, E_of_z, comoving_distance


def _scale_factor(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + z)


def pnl_at_z(k, z: float, sigma8: float, cosmo: DistanceCosmo,
             Ob: float = 0.049, ns: float = 0.965,
             add_correction: bool = True):
    """syren-halofit P_NL(k, z) with the proper D^2(z) growth scaling.

    Calls ``cosmology.halofit_from_plin`` with::

        plin(k, z) = D^2(z) * plin_emulated(k, sigma8, ..., a=1)
        a = 1 / (1 + z)
        sigma8 unchanged (the z=0 normalisation, halofit's training input)

    This is the correct way to evaluate the syren-halofit emulator at
    z > 0 -- ``run_halofit(..., a=1/(1+z))`` does NOT scale plin and so
    only differs from the a=1 result via the NL emulators.
    """
    import jax.numpy as jnp

    a_z = 1.0 / (1.0 + float(z))
    D = float(linear_growth(np.array([z]), cosmo)[0])
    plin0 = plin_emulated(k, sigma8, cosmo.Om, Ob, cosmo.h, ns, a=1.0)
    plin_z = (D ** 2) * plin0
    return halofit_from_plin(
        k, plin_z, sigma8, cosmo.Om, Ob, cosmo.h, ns,
        a=a_z, add_correction=add_correction,
    )


def linear_growth(z, cosmo: DistanceCosmo, n_grid: int = 1024,
                   z_max_table: float = 100.0) -> np.ndarray:
    """Normalised linear growth D(z)/D(0) for a flat (w0, wa) cosmology.

    Solves the integral form D(a) ∝ H(a) ∫_0^a da' / [a' H(a')]^3
    on a tabulated a-grid and divides by D(a=1).
    """
    import jax.numpy as jnp
    a = 1.0 / (1.0 + z_max_table)
    a_grid = np.linspace(a, 1.0, n_grid)
    z_grid = 1.0 / a_grid - 1.0
    E = np.asarray(E_of_z(jnp.asarray(z_grid), cosmo))
    integrand = 1.0 / (a_grid * E) ** 3
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * np.diff(a_grid)
    )])
    D_unnorm = E * cum
    D_today = float(D_unnorm[-1])
    D = D_unnorm / D_today
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    a_q = 1.0 / (1.0 + z_arr)
    return np.interp(a_q, a_grid, D)


def cl_gg_limber(
    ell: np.ndarray,
    z_grid: np.ndarray,
    dndz: np.ndarray,
    cosmo: DistanceCosmo,
    bias: float = 1.0,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    k_min: float = 1e-4,
    k_max: float = 1e2,
    n_k: int = 1024,
) -> np.ndarray:
    """Limber C_ell^gg via syren-halofit P_NL on a (z, k) grid.

    ``dndz`` is sampled on ``z_grid`` and is normalised internally to
    integrate to 1. ``bias`` is a constant linear bias applied to the
    galaxy kernel; for a redshift-dependent bias call this in batches.
    """
    import jax.numpy as jnp

    ell = np.asarray(ell, dtype=np.float64)
    z = np.asarray(z_grid, dtype=np.float64)
    nz = np.asarray(dndz, dtype=np.float64)
    if not (z[1:] > z[:-1]).all():
        raise ValueError("z_grid must be strictly increasing")
    # normalise dN/dz to integrate to 1
    nz = nz / np.trapezoid(nz, z)

    chi = np.asarray(comoving_distance(jnp.asarray(z), cosmo))
    E = np.asarray(E_of_z(jnp.asarray(z), cosmo))
    dchi_dz = C_OVER_H100_MPCH / E

    # Build a single k-grid for halofit; we'll sample at k = (ell + 1/2)/chi
    # for each z separately by interpolation.
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_jax = jnp.asarray(k_grid)

    # Evaluate P_NL(k, z) row by row -- run_halofit takes a single scale factor.
    # For ~30 z bins this is cheap (a few seconds).
    # halofit-with-growth: D^2(z)-scaled plin fed into halofit_from_plin
    # at the proper a=1/(1+z) (NL emulators on their training convention).
    P_kz = np.empty((len(z), len(k_grid)), dtype=np.float64)
    for i, zi in enumerate(z):
        P_kz[i, :] = np.asarray(pnl_at_z(
            k_jax, z=float(zi), sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns,
        ))

    # Limber kernel evaluated at each (ell, z)
    cl = np.zeros_like(ell)
    chi_safe = np.where(chi > 0, chi, np.inf)
    weight = nz ** 2 / (chi_safe ** 2 * dchi_dz)   # (n_z,)
    for j, l in enumerate(ell):
        k_at_z = (l + 0.5) / chi_safe              # (n_z,)
        # row-wise interpolation of P_NL(k, z) at k_at_z
        Pl = np.array([
            np.interp(k_at_z[i], k_grid, P_kz[i, :]) if k_at_z[i] < k_max
            else 0.0
            for i in range(len(z))
        ])
        integrand = weight * Pl
        cl[j] = bias ** 2 * np.trapezoid(integrand, z)
    return cl


def xi_real_at_z(
    s: np.ndarray,
    z_eff: float,
    cosmo: DistanceCosmo,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    k_min: float = 1e-4,
    k_max: float = 1e2,
    n_k: int = 4096,
) -> np.ndarray:
    """Real-space correlation function xi(s) at z=z_eff via FFTLog of P_NL."""
    import jax.numpy as jnp
    from .spectra import FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog

    k = make_log_k_grid(k_min, k_max, n_k)
    P = pnl_at_z(k, z=z_eff, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns)
    fft = FFTLogP2xi(k, l=0)
    s_jax = jnp.asarray(np.asarray(s, dtype=np.float64))
    return np.asarray(xi_from_Pk_fftlog(s_jax, fft, P), dtype=np.float64)


def wp_limber(
    rp: np.ndarray,
    z_eff: float,
    cosmo: DistanceCosmo,
    bias: float = 1.0,
    pi_max: float = 100.0,
    n_pi: int = 200,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
) -> np.ndarray:
    """Real-space projected correlation wp(rp) at z_eff.

        wp(rp) = 2 * int_0^pi_max b^2 xi(sqrt(rp^2 + pi^2), z_eff) d(pi)

    Real-space prediction; no Kaiser RSD correction (Quaia photo-z's
    suppress LOS clustering anyway, so the real-space wp is the
    relevant target after pi_max projects out small-scale RSD).
    """
    rp = np.asarray(rp, dtype=np.float64)
    pi_grid = np.linspace(0.0, pi_max, n_pi)
    s_grid = np.sqrt(rp[:, None] ** 2 + pi_grid[None, :] ** 2)
    s_unique = np.unique(s_grid.ravel())
    xi_unique = xi_real_at_z(
        s_unique, z_eff=z_eff, cosmo=cosmo,
        sigma8=sigma8, Ob=Ob, ns=ns,
    )
    xi_grid = np.interp(s_grid, s_unique, xi_unique)
    wp = 2.0 * np.trapezoid(xi_grid, pi_grid, axis=1)
    return bias ** 2 * wp
