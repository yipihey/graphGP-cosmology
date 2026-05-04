"""CMB lensing -- galaxy density cross-correlation.

The Limber-projected angular cross spectrum of galaxy density and CMB
lensing convergence is::

    C_ell^{g-kappa} = int dz [b(z) dN/dz] * W_kappa(z) / chi^2(z)
                       * P_NL(k = (ell+1/2)/chi(z), z)

with the standard CMB lensing kernel::

    W_kappa(z) = (3/2) Om_m (c/H0)^-2 * (1+z) * chi(z) (chi_* - chi(z)) / chi_*

In ``Mpc/h``-based comoving distances, ``c/H0 = 2997.92 Mpc/h``, so
the prefactor is ``1/2997.92^2`` in ``(Mpc/h)^-2``. ``chi_*`` is the
comoving distance to the surface of last scattering (z* ~ 1090).

For Quaia x Planck CMB lensing, this is the channel that boosts the
clustering-amplitude SNR by ~ 10x over the auto wp/C_ell -- the lensing
kernel peaks at z ~ 2 and overlaps strongly with Quaia's 0.8 < z < 2.5
range. Alonso et al. 2024 measure SNR ~ 28 on Planck PR3 x Quaia G<20.5.

This module exposes:

  ``lensing_kernel_W_kappa(z, cosmo, z_star)`` -> W_kappa(z)
  ``cl_gkappa_limber(ell, z_grid, dndz, b_z, cosmo, ...)`` -> C_ell^{g-kappa}
  ``quaia_lensing_snr_forecast(...)``  -> SNR estimate for an ell-range
                                          and assumed noise spectrum.

The forecast assumes the standard Planck PR3 lensing reconstruction
noise (``N_ell^{kappa-kappa}``) and a Quaia-G<20-shaped n(z); both
overridable. Once you drop in a real Planck kappa healpix map, the
NaMaster pseudo-Cl machinery in ``twopt_density.angular`` does the
measurement directly.
"""

from __future__ import annotations

import numpy as np


def lensing_kernel_W_kappa(
    z, cosmo, z_star: float = 1090.0, n_grid_chi: int = 1024,
):
    """CMB lensing convergence kernel ``W_kappa(z)``.

    Returns the kernel evaluated at the input ``z`` in units of
    ``(Mpc/h)^-1`` so that ``C_ell = int dz W_g W_kappa / chi^2 P``
    is dimensionless.

    The form is

        W_kappa(z) = (3/2) Om_m (H0/c)^2 (1+z) chi (chi_* - chi)/chi_*

    with ``chi_*`` the comoving distance to the surface of last
    scattering (z_star = 1090 by default).
    """
    import jax.numpy as jnp
    from .distance import C_OVER_H100_MPCH, comoving_distance

    z_j = jnp.asarray(z, dtype=jnp.float64)
    # comoving_distance defaults to z_max=4; lensing needs out to z_star.
    # z_max_int is static (doesn't depend on traced z) so jax.grad works.
    z_max_int = float(z_star) + 50.0
    chi = comoving_distance(z_j, cosmo, z_max=z_max_int, n_grid=4096)
    chi_star = comoving_distance(
        jnp.asarray([z_star]), cosmo, z_max=z_max_int, n_grid=4096
    )[0]
    prefactor = 1.5 * cosmo.Om / C_OVER_H100_MPCH ** 2
    W = prefactor * (1.0 + z_j) * chi * (chi_star - chi) / chi_star
    return jnp.where(chi < chi_star, W, 0.0)


def cl_gkappa_limber(
    ell, z_grid, dndz, b_z, cosmo,
    sigma8: float = 0.81, Ob: float = 0.049, ns: float = 0.965,
    z_star: float = 1090.0,
    k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 1024,
):
    """Limber-projected ``C_ell^{g-kappa}``.

    Same vmap'd machinery as ``cl_gg_limber`` -- one factor of
    ``b(z) dN/dz`` for the galaxy side, one factor of ``W_kappa(z)``
    for the lensing side. JAX-pure: differentiable in
    ``(cosmo, b_z, sigma8)``.
    """
    import jax
    import jax.numpy as jnp
    from .distance import C_OVER_H100_MPCH, E_of_z, comoving_distance
    from .limber import pnl_at_z

    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    z = jnp.asarray(z_grid, dtype=jnp.float64)
    nz = jnp.asarray(dndz, dtype=jnp.float64)
    b_z_j = jnp.asarray(b_z, dtype=jnp.float64)
    nz = nz / jnp.trapezoid(nz, z)

    chi = comoving_distance(z, cosmo)
    E = E_of_z(z, cosmo)
    dchi_dz = C_OVER_H100_MPCH / E
    W_kappa_z = lensing_kernel_W_kappa(z, cosmo, z_star=z_star)

    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_grid = jnp.asarray(k_np)

    pnl_at_zi = lambda zi: pnl_at_z(
        k_grid, z=zi, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns,
    )
    P_kz = jax.vmap(pnl_at_zi)(z)                    # (n_z, n_k)

    chi_safe = jnp.where(chi > 0, chi, jnp.inf)

    # Galaxy and lensing kernels (per dz):
    #   W_g(z) = b(z) dN/dz
    #   integrand = W_g(z) W_kappa(z) / chi^2 * P / dchi/dz... wait:
    #
    # The Limber integral in z (substituting dchi = (dchi/dz) dz):
    #   C_ell = int dchi W_g(chi) W_kappa(chi) P / chi^2
    #        = int dz (dchi/dz) (W_g_per_dz / (dchi/dz))
    #                  (W_kappa_per_dchi) P / chi^2
    # Standard form (with W_g per dz, W_kappa per dchi):
    #   C_ell = int dz [b nz / 1] W_kappa P / chi^2
    # since dN/dz / (dchi/dz) is "per dchi" and we already have
    # dN/dz form for W_g.
    weight = b_z_j * nz * W_kappa_z / chi_safe ** 2

    def Pl_at(l, zi_idx):
        k_at = (l + 0.5) / chi_safe[zi_idx]
        return jnp.interp(k_at, k_grid, P_kz[zi_idx])
    z_idx = jnp.arange(z.shape[0])
    Pl_grid = jax.vmap(jax.vmap(Pl_at, in_axes=(None, 0)),
                         in_axes=(0, None))(ell_j, z_idx)   # (n_ell, n_z)

    integrand = weight[None, :] * Pl_grid
    return jnp.trapezoid(integrand, z, axis=1)


def cl_gkappa_limber_nowiggle(
    ell, z_grid, dndz, b_z, cosmo,
    sigma8: float = 0.81, Ob: float = 0.049, ns: float = 0.965,
    z_star: float = 1090.0,
    k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 1024,
):
    """No-BAO-wiggles companion to ``cl_gkappa_limber``.

    Identical projection but uses ``pnl_at_z_nowiggle`` (Eisenstein-Hu
    smooth) so the BAO contribution is absent. Subtract from
    ``cl_gkappa_limber`` to isolate the angular BAO template in the
    galaxy x CMB-lensing cross spectrum.
    """
    import jax
    import jax.numpy as jnp
    from .distance import C_OVER_H100_MPCH, E_of_z, comoving_distance
    from .limber import pnl_at_z_nowiggle

    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    z = jnp.asarray(z_grid, dtype=jnp.float64)
    nz = jnp.asarray(dndz, dtype=jnp.float64)
    b_z_j = jnp.asarray(b_z, dtype=jnp.float64)
    nz = nz / jnp.trapezoid(nz, z)

    chi = comoving_distance(z, cosmo)
    W_kappa_z = lensing_kernel_W_kappa(z, cosmo, z_star=z_star)

    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_grid = jnp.asarray(k_np)

    pnl_at_zi = lambda zi: pnl_at_z_nowiggle(
        k_grid, z=zi, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns,
    )
    P_kz = jax.vmap(pnl_at_zi)(z)

    chi_safe = jnp.where(chi > 0, chi, jnp.inf)
    weight = b_z_j * nz * W_kappa_z / chi_safe ** 2

    def Pl_at(l, zi_idx):
        k_at = (l + 0.5) / chi_safe[zi_idx]
        return jnp.interp(k_at, k_grid, P_kz[zi_idx])
    z_idx = jnp.arange(z.shape[0])
    Pl_grid = jax.vmap(jax.vmap(Pl_at, in_axes=(None, 0)),
                         in_axes=(0, None))(ell_j, z_idx)

    integrand = weight[None, :] * Pl_grid
    return jnp.trapezoid(integrand, z, axis=1)


def planck_pr3_lensing_noise(ell, ell_pivot: float = 100.0,
                               N0: float = 8e-8):
    """Crude approximation to the Planck PR3 lensing reconstruction
    noise N_ell^{kappa-kappa}.

    Real Planck noise is well-fit by N_ell ~ 8e-8 over ell = 50-300
    with growth at low ell (mean-field uncertainty) and high ell
    (mode-coupling). This default is a flat power-law approximation
    suitable for SNR forecasts; for a real measurement use the
    delivered Planck noise spectrum.
    """
    return N0 * np.ones_like(ell, dtype=np.float64)


def cl_kappa_kappa_planck_pr3(
    ell, cosmo, sigma8: float = 0.81, Ob: float = 0.049, ns: float = 0.965,
    z_star: float = 1090.0, n_z_grid: int = 80,
):
    """Theoretical CMB-lensing auto C_ell^{kappa-kappa} signal under
    the same Limber + halofit machinery used for cl_gkappa_limber.

    For SNR forecasts. Use ``cl_kappa_kappa_planck_pr3 +
    planck_pr3_lensing_noise`` for the ``Sigma_kk = signal + noise``
    that goes into the Cramer-Rao bound on the cross-correlation.
    """
    import jax
    import jax.numpy as jnp
    from .distance import C_OVER_H100_MPCH, E_of_z, comoving_distance
    from .limber import pnl_at_z

    # ``pnl_at_z`` (halofit) breaks at z >> ~ 10; the kappa-kappa
    # integrand W_kappa^2/chi^2 P_NL falls off >> 100x by z = 10
    # (W_kappa rises ~ 3x, chi^2 ~ 5x, but D^2 drops ~ 10x and P
    # at fixed k ~ ell/chi shifts to higher k where amplitudes
    # are very small). Capping at z = 10 misses < 1 % of the
    # full integral while keeping the forward model finite.
    z = jnp.linspace(0.01, 10.0, n_z_grid)
    z_max_int = float(z_star) + 50.0
    chi = comoving_distance(z, cosmo, z_max=z_max_int, n_grid=4096)
    E = E_of_z(z, cosmo)
    dchi_dz = C_OVER_H100_MPCH / E
    W_kappa_z = lensing_kernel_W_kappa(z, cosmo, z_star=z_star)
    chi_safe = jnp.where(chi > 0, chi, jnp.inf)

    k_grid = jnp.asarray(np.logspace(-4, 2, 1024))
    pnl_at_zi = lambda zi: pnl_at_z(
        k_grid, z=zi, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns,
    )
    P_kz = jax.vmap(pnl_at_zi)(z)

    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    z_idx = jnp.arange(z.shape[0])
    def Pl_at(l, zi_idx):
        k_at = (l + 0.5) / chi_safe[zi_idx]
        return jnp.interp(k_at, k_grid, P_kz[zi_idx])
    Pl_grid = jax.vmap(jax.vmap(Pl_at, in_axes=(None, 0)),
                         in_axes=(0, None))(ell_j, z_idx)
    # change of variable: int dchi -> int dz dchi/dz
    weight = dchi_dz * W_kappa_z ** 2 / chi_safe ** 2
    integrand = weight[None, :] * Pl_grid
    return np.asarray(jnp.trapezoid(integrand, z, axis=1))


def quaia_gkappa_snr_forecast(
    ell_grid, cl_gk, cl_gg, cl_kk_signal_plus_noise, f_sky: float = 0.66,
):
    """Cramer-Rao SNR forecast for a galaxy x kappa cross-correlation.

    Per multipole the Gaussian variance of the cross-spectrum estimator
    is
        Var(C_ell^{g-kappa}) = (1 / (2 ell + 1) f_sky)
                                * [(C_gg + N_gg)(C_kk + N_kk) + C_gk^2]

    Total SNR^2 = sum_ell C_gk^2 / Var(C_gk).

    We pass ``cl_gg`` already including its shot noise N_gg, and
    ``cl_kk_signal_plus_noise`` already as ``C_kk + N_kk``.
    """
    ell = np.asarray(ell_grid, dtype=np.float64)
    cl_gk = np.asarray(cl_gk, dtype=np.float64)
    cl_gg = np.asarray(cl_gg, dtype=np.float64)
    cl_kk = np.asarray(cl_kk_signal_plus_noise, dtype=np.float64)
    var = (cl_gg * cl_kk + cl_gk ** 2) / ((2 * ell + 1) * f_sky)
    snr2_per = cl_gk ** 2 / np.maximum(var, 1e-50)
    return float(np.sqrt(snr2_per.sum())), snr2_per


def joint_wp_clkappa_map_fit(
    rp, wp_data, sigma_chi_eff_pair, wp_sigma_or_cov,
    ell, cl_gk_data, cl_gk_sigma,
    z_grid_b, dndz, cosmo,
    z_eff: float = 1.5, pi_max: float = 200.0, sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    free=("sigma8", "b"), fix=None,
    z_pivot_b: float = 1.5, z_star: float = 1090.0,
    pi_int_range: float = 800.0, n_pi_true: int = 400,
):
    """JAX-MAP joint fit on (wp, C_ell^{g-kappa}).

    The lensing cross-correlation has a different sigma_8-b degeneracy
    direction than wp/Cl^{gg}: it scales as b * sigma_8 * D(z) at the
    galaxy z_eff (one factor of bias, one factor of growth, vs the
    auto's two factors of bias). Combining the two breaks the
    sigma_8-b banana that wp+Cl_gg can't.

    ``free`` selects which of (Om, sigma8, b) are fit. ``fix={...}``
    overrides defaults (0.31, 0.81, 2.6).
    """
    import jax
    import jax.numpy as jnp
    from .fit import map_fit
    from .limber import wp_observed
    from .limber import _PARAM_BOUNDS, _PARAM_DEFAULTS, _PARAM_NAMES

    fix = dict(fix or {})
    free = tuple(free)
    for name in free:
        if name not in _PARAM_NAMES:
            raise ValueError(f"unknown parameter {name}")
        fix.pop(name, None)
    fixed = {**{k: _PARAM_DEFAULTS[k] for k in _PARAM_NAMES if k not in free},
             **fix}

    rp_j = jnp.asarray(rp, dtype=jnp.float64)
    wp_j = jnp.asarray(wp_data, dtype=jnp.float64)
    sw = jnp.asarray(wp_sigma_or_cov, dtype=jnp.float64)
    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    cl_j = jnp.asarray(cl_gk_data, dtype=jnp.float64)
    sl = jnp.asarray(cl_gk_sigma, dtype=jnp.float64)
    zb = jnp.asarray(z_grid_b, dtype=jnp.float64)
    nz = jnp.asarray(dndz, dtype=jnp.float64)

    def expand(theta):
        out = {**fixed}
        for i, name in enumerate(free):
            out[name] = theta[i]
        return out

    def loss(theta):
        p = expand(theta)
        from .distance import DistanceCosmo
        cosmo_p = DistanceCosmo(Om=p["Om"], h=cosmo.h)
        b_const = p["b"]
        wp_pred = wp_observed(
            rp_j, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff_pair,
            cosmo=cosmo_p, bias=b_const, pi_max=pi_max,
            pi_int_range=pi_int_range, n_pi_true=n_pi_true,
            sigma8=p["sigma8"], Ob=Ob, ns=ns,
        )
        # diagonal wp loss (caller may pass a sigma vector or full cov)
        if sw.ndim == 1:
            chi2_wp = jnp.sum(((wp_j - wp_pred) / sw) ** 2)
        else:
            d = wp_j - wp_pred
            Cinv = jnp.linalg.inv(sw)
            chi2_wp = d @ Cinv @ d

        b_z = jnp.full_like(zb, b_const)
        cl_pred = cl_gkappa_limber(
            ell_j, zb, nz, b_z, cosmo_p,
            sigma8=p["sigma8"], Ob=Ob, ns=ns, z_star=z_star,
        )
        if sl.ndim == 1:
            chi2_cl = jnp.sum(((cl_j - cl_pred) / sl) ** 2)
        else:
            d = cl_j - cl_pred
            Cinv = jnp.linalg.inv(sl)
            chi2_cl = d @ Cinv @ d

        return chi2_wp + chi2_cl

    theta0 = tuple(_PARAM_DEFAULTS[k] for k in free)
    bounds = tuple(_PARAM_BOUNDS[k] for k in free)
    result = map_fit(loss, theta0, bounds=bounds)
    H = np.asarray(jax.hessian(loss)(jnp.asarray(result.theta)))
    try:
        cov = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((len(free), len(free)), np.nan)
    theta_full = expand(jnp.asarray(result.theta))
    theta_full = {k: float(v) for k, v in theta_full.items()}
    return result, cov, theta_full
