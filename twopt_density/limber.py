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

    a_z = 1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64))
    D = linear_growth(jnp.asarray([z], dtype=jnp.float64), cosmo)[0]
    plin0 = plin_emulated(k, sigma8, cosmo.Om, Ob, cosmo.h, ns, a=1.0)
    plin_z = (D ** 2) * plin0
    return halofit_from_plin(
        k, plin_z, sigma8, cosmo.Om, Ob, cosmo.h, ns,
        a=a_z, add_correction=add_correction,
    )


def dndz_pdf_stack(
    z_grid: np.ndarray,
    z_obs: np.ndarray,
    sigma_z: np.ndarray,
    sigma_z_floor: float = 1e-3,
) -> np.ndarray:
    """Photo-z PDF-stacked dN/dz on ``z_grid``.

    Each object contributes a Gaussian ``N(z_obs_i, sigma_z_i^2)``
    instead of a delta at its point estimate. The output is summed over
    all objects (un-normalised; the Limber kernel renormalises).

    For Quaia this should be the published spectro-photometric PDFs;
    until those are loaded we approximate each PDF as a Gaussian with
    width ``redshift_quaia_err``. ``sigma_z_floor`` clamps the kernel
    width to avoid 0/0 when a few objects have zero error.
    """
    z_obs = np.asarray(z_obs, dtype=np.float64)
    sig = np.maximum(np.asarray(sigma_z, dtype=np.float64), sigma_z_floor)
    # vectorised Gaussian sum, summed in O(N_obj * N_z)
    dz = z_grid[:, None] - z_obs[None, :]                  # (N_z, N_obj)
    pdfs = np.exp(-0.5 * (dz / sig[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sig[None, :])
    return pdfs.sum(axis=1)                                # (N_z,)


def linear_growth(z, cosmo: DistanceCosmo, n_grid: int = 1024,
                   z_max_table: float = 100.0):
    """Normalised linear growth D(z)/D(0) for a flat (w0, wa) cosmology.

    Solves the integral form D(a) ∝ H(a) ∫_0^a da' / [a' H(a')]^3
    on a tabulated a-grid and divides by D(a=1).

    JAX-pure: differentiable in ``cosmo`` and accepts traced ``z``.
    """
    import jax.numpy as jnp
    a_min = 1.0 / (1.0 + z_max_table)
    a_grid = jnp.linspace(a_min, 1.0, n_grid)
    z_grid = 1.0 / a_grid - 1.0
    E = E_of_z(z_grid, cosmo)
    integrand = 1.0 / (a_grid * E) ** 3
    da = a_grid[1] - a_grid[0]
    cum = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * da),
    ])
    D_unnorm = E * cum
    D_today = D_unnorm[-1]
    D = D_unnorm / D_today
    z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))
    a_q = 1.0 / (1.0 + z_arr)
    # interp expects increasing xp -> a_grid is increasing
    return jnp.interp(a_q, a_grid, D)


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


def sigma_chi_from_sigma_z(z_obs: np.ndarray, sigma_z: np.ndarray,
                            cosmo: DistanceCosmo) -> np.ndarray:
    """Per-object comoving LOS error sigma_chi = (dchi/dz) * sigma_z."""
    import jax.numpy as jnp
    z_jax = jnp.asarray(z_obs, dtype=np.float64)
    E = np.asarray(E_of_z(z_jax, cosmo))
    dchi_dz = C_OVER_H100_MPCH / E
    return dchi_dz * np.asarray(sigma_z, dtype=np.float64)


def wp_observed(
    rp,
    z_eff: float,
    sigma_chi_eff,
    cosmo: DistanceCosmo,
    bias=1.0,
    pi_max: float = 200.0,
    pi_int_range: float = 800.0,
    n_pi_true: int = 400,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    k_min: float = 1e-4,
    k_max: float = 1e2,
    n_k: int = 4096,
):
    """JAX-differentiable photo-z-aware projected correlation wp_obs(rp).

    Forward model for the observed wp(rp) accounting for per-pair
    Gaussian photo-z scatter:

        xi_obs(rp, pi_obs) = int dpi_true xi_real(sqrt(rp^2 + pi_true^2))
                              * G(pi_obs - pi_true; sigma_pair)
        wp_obs(rp; pi_max) = int_{-pi_max}^{pi_max} xi_obs(rp, pi_obs) dpi_obs

    Switching the order of integration gives a single 1D integral over
    pi_true with an analytic erf window:

        wp_obs(rp) = int dpi_true xi_real(...) * (1/2)
            * [erf((pi_max - pi_true) / (sigma_pair sqrt(2)))
             + erf((pi_max + pi_true) / (sigma_pair sqrt(2)))]

    where ``sigma_chi_eff`` is the effective per-pair LOS sigma:
    ``sqrt(2) * <sigma_chi>`` for an auto-correlation. This function is
    fully JAX-differentiable in ``(bias, sigma_chi_eff, cosmo)`` --
    ``cosmo`` enters via ``pnl_at_z`` (P_NL(k, z) -> xi_real(s) FFTLog).

    The deterministic ``wp_limber`` (real-space, no photo-z kernel) is
    recovered in the limit ``sigma_chi_eff -> 0`` and ``pi_max -> inf``.
    """
    import jax.numpy as jnp
    from jax.scipy.special import erf
    from .spectra import FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog

    rp = jnp.atleast_1d(jnp.asarray(rp, dtype=jnp.float64))
    sigma = jnp.asarray(sigma_chi_eff, dtype=jnp.float64)
    bias = jnp.asarray(bias, dtype=jnp.float64)
    sigma_safe = jnp.maximum(sigma, 1e-8)

    # 1) xi_real(s, z_eff) via FFTLog of P_NL(k, z_eff).
    k = make_log_k_grid(k_min, k_max, n_k)
    P = pnl_at_z(k, z=z_eff, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns)
    fft = FFTLogP2xi(k, l=0)
    # xi_real evaluated on a dense s-grid spanning the static integration
    # range. ``pi_int_range`` should be >> sqrt(pi_max^2 + (5*sigma_chi)^2)
    # so the erf window has decayed at the boundary; default 800 Mpc/h
    # covers Quaia (sigma_chi <~ 200 Mpc/h, pi_max <~ 200 Mpc/h).
    rp_max = float(jnp.max(rp))
    s_max = float(jnp.sqrt(rp_max ** 2 + pi_int_range ** 2) + 1.0)
    s_grid = jnp.geomspace(0.5, s_max, 2048)
    xi_grid = xi_from_Pk_fftlog(s_grid, fft, P)

    # 2) integrate over pi_true with the erf window.
    pi_true = jnp.linspace(-pi_int_range, pi_int_range, n_pi_true)
    s_eval = jnp.sqrt(rp[:, None] ** 2 + pi_true[None, :] ** 2)   # (n_rp, n_pi)
    xi_eval = jnp.interp(s_eval.ravel(), s_grid, xi_grid).reshape(s_eval.shape)

    sigma_sqrt2 = sigma_safe * jnp.sqrt(2.0)
    window = 0.5 * (
        erf((pi_max - pi_true) / sigma_sqrt2)
        + erf((pi_max + pi_true) / sigma_sqrt2)
    )                                                              # (n_pi,)
    integrand = xi_eval * window[None, :]
    wp = jnp.trapezoid(integrand, pi_true, axis=1)
    return bias ** 2 * wp


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
