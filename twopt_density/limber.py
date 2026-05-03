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
    ell,
    z_grid,
    dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    k_min: float = 1e-4,
    k_max: float = 1e2,
    n_k: int = 1024,
):
    """Limber C_ell^gg via syren-halofit P_NL on a (z, k) grid.

    ``dndz`` is sampled on ``z_grid`` and normalised internally to
    integrate to 1. ``bias`` is a constant linear bias on the galaxy
    kernel.

    JAX-pure: differentiable in ``(cosmo, bias, sigma8)``. Numpy inputs
    are coerced to jax arrays; the returned value is a jax array but
    interoperates with numpy via ``np.asarray``.
    """
    import jax
    import jax.numpy as jnp

    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    z = jnp.asarray(z_grid, dtype=jnp.float64)
    nz = jnp.asarray(dndz, dtype=jnp.float64)
    nz = nz / jnp.trapezoid(nz, z)

    chi = comoving_distance(z, cosmo)
    E = E_of_z(z, cosmo)
    dchi_dz = C_OVER_H100_MPCH / E

    # static k-grid (must be Python-built so the vmap below has a fixed shape)
    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_grid = jnp.asarray(k_np)

    # vmap pnl_at_z over z -> (n_z, n_k) tensor
    pnl_at_zi = lambda zi: pnl_at_z(
        k_grid, z=zi, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns,
    )
    P_kz = jax.vmap(pnl_at_zi)(z)

    chi_safe = jnp.where(chi > 0, chi, jnp.inf)
    weight = nz ** 2 / (chi_safe ** 2 * dchi_dz)         # (n_z,)

    # For each (ell, z) interpolate P_NL row in k at k = (ell+1/2)/chi(z).
    def Pl_at(l, zi_idx):
        k_at = (l + 0.5) / chi_safe[zi_idx]
        return jnp.interp(k_at, k_grid, P_kz[zi_idx])

    z_idx = jnp.arange(z.shape[0])
    Pl_grid = jax.vmap(
        jax.vmap(Pl_at, in_axes=(None, 0)),
        in_axes=(0, None),
    )(ell_j, z_idx)                                       # (n_ell, n_z)

    integrand = weight[None, :] * Pl_grid
    cl = jnp.trapezoid(integrand, z, axis=1)
    cl = bias ** 2 * cl
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


def make_wp_fft(k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 4096):
    """Pre-build the FFTLog kernel + its k-grid used by ``wp_observed``.

    ``mcfit.P2xi`` cannot be constructed inside a JIT trace, so the
    fitter precomputes it once at non-traced time and threads it
    through; the jitted loss then only depends on traced ``cosmo`` /
    ``bias`` / ``sigma_chi`` values, not on the FFT setup.
    """
    from .spectra import FFTLogP2xi

    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    fft = FFTLogP2xi(k_np, l=0)
    return fft, k_np


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
    fft=None,
    k_grid=None,
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
    from .spectra import xi_from_Pk_fftlog

    rp = jnp.atleast_1d(jnp.asarray(rp, dtype=jnp.float64))
    sigma = jnp.asarray(sigma_chi_eff, dtype=jnp.float64)
    bias = jnp.asarray(bias, dtype=jnp.float64)
    sigma_safe = jnp.maximum(sigma, 1e-8)

    # 1) xi_real(s, z_eff) via FFTLog of P_NL(k, z_eff).
    # ``mcfit`` cannot be JIT-traced, so when this function runs under
    # jax.grad/jit the caller must pre-build (fft, k_grid) via
    # ``make_wp_fft`` and pass them in. Default path: build once for
    # eager use.
    if fft is None or k_grid is None:
        fft, k_np = make_wp_fft(k_min, k_max, n_k)
        k_grid = jnp.asarray(k_np)
    P = pnl_at_z(k_grid, z=z_eff, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns)
    # xi_real evaluated on a dense, *static* s-grid that spans more than
    # any rp + pi_true we'll need; using a static range keeps the
    # function jit-traceable. 1500 Mpc/h covers rp <~ 200 + pi_int_range
    # 800; bump if you need wider rp.
    s_grid = jnp.geomspace(0.5, 1500.0, 2048)
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


_PARAM_NAMES = ("Om", "sigma8", "b")
_PARAM_DEFAULTS = {"Om": 0.31, "sigma8": 0.81, "b": 2.6}
_PARAM_BOUNDS = {"Om": (0.15, 0.55), "sigma8": (0.5, 1.1), "b": (0.3, 6.0)}


def wp_map_fit(
    rp_meas,
    wp_meas,
    sigma_wp,
    sigma_chi_eff,
    z_eff: float,
    free=("Om", "sigma8", "b"),
    fix=None,
    pi_max: float = 200.0,
    h: float = 0.68,
    Ob: float = 0.049,
    ns: float = 0.965,
    n_pi_true: int = 400,
):
    """JAX MAP fit of subset of (Om, sigma8, b) on a wp(rp) measurement.

    Diagonal-Gaussian chi^2 loss with the photo-z-aware ``wp_observed``
    forward model. Any subset of (Om, sigma8, b) can be free; the rest
    are fixed via ``fix={...}`` (defaults: Planck 0.31 / 0.81 / 2.6).

    Returns
    -------
    result : FitResult on the free parameters only.
    cov : (n_free, n_free) covariance from inverse Hessian.
    wp_pred : (N_rp,) wp at the best fit.
    theta_full : dict {name: value} including fixed parameters.

    Notes
    -----
    On wp(rp) alone, (sigma8, b) are heavily degenerate -- the data
    constrain only ``sigma8 * b``. To get a well-defined covariance fix
    one of them (or fold in C_ell to break the degeneracy).
    """
    import jax
    import jax.numpy as jnp
    from .fit import map_fit

    fix = dict(fix or {})
    free = tuple(free)
    for name in free:
        if name not in _PARAM_NAMES:
            raise ValueError(f"unknown parameter {name}; expected one of "
                              f"{_PARAM_NAMES}")
        fix.pop(name, None)
    fixed = {**{k: _PARAM_DEFAULTS[k] for k in _PARAM_NAMES if k not in free},
             **fix}

    rp_jax = jnp.asarray(rp_meas, dtype=jnp.float64)
    wp_jax = jnp.asarray(wp_meas, dtype=jnp.float64)
    sig_jax = jnp.asarray(sigma_wp, dtype=jnp.float64)

    fft, k_np = make_wp_fft()
    k_grid = jnp.asarray(k_np)

    def expand(theta):
        out = {**fixed}
        for i, name in enumerate(free):
            out[name] = theta[i]
        return out

    def loss(theta):
        p = expand(theta)
        cosmo = DistanceCosmo(Om=p["Om"], h=h)
        wp_pred = wp_observed(
            rp_jax, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff, cosmo=cosmo,
            bias=p["b"], pi_max=pi_max, n_pi_true=n_pi_true,
            sigma8=p["sigma8"], Ob=Ob, ns=ns, fft=fft, k_grid=k_grid,
        )
        return jnp.sum(((wp_jax - wp_pred) / sig_jax) ** 2)

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
    cosmo_best = DistanceCosmo(Om=theta_full["Om"], h=h)
    wp_pred = np.asarray(wp_observed(
        rp_jax, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff, cosmo=cosmo_best,
        bias=theta_full["b"], pi_max=pi_max, n_pi_true=n_pi_true,
        sigma8=theta_full["sigma8"], Ob=Ob, ns=ns,
        fft=fft, k_grid=k_grid,
    ))
    return result, cov, wp_pred, theta_full


def sample_pair_sigma_chi(
    sigma_z_array: np.ndarray,
    z_array: np.ndarray,
    cosmo: DistanceCosmo,
    n_pairs: int = 20000,
    rng=None,
) -> np.ndarray:
    """Empirical per-pair LOS sigma sampled from the catalogue.

    For an auto-correlation, each pair (i, j) has Gaussian LOS variance
    sigma_chi_i^2 + sigma_chi_j^2 (treating photo-z's as independent
    Gaussians on each side). Sample ``n_pairs`` random index pairs from
    the catalogue and return the array of sqrt(sum-of-squares) so that
    ``wp_observed_perpair`` can average the LOS window across the actual
    pair-sigma distribution.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(sigma_z_array)
    if n < 2:
        raise ValueError("need >=2 objects for pair sampling")
    sig_chi = sigma_chi_from_sigma_z(z_array, sigma_z_array, cosmo)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    # avoid self-pairs (rare but messy)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % n
    return np.sqrt(sig_chi[i] ** 2 + sig_chi[j] ** 2)


def wp_observed_perpair(
    rp,
    z_eff: float,
    sigma_chi_samples,
    cosmo: DistanceCosmo,
    bias=1.0,
    pi_max: float = 200.0,
    pi_int_range: float = 800.0,
    n_pi_true: int = 400,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    fft=None,
    k_grid=None,
):
    """Photo-z-aware wp_observed averaged over an empirical per-pair
    sigma_chi distribution.

    Generalisation of ``wp_observed``: instead of assuming a single
    effective LOS sigma, average the analytic erf window over the
    empirical distribution of per-pair sigma_chi sampled from the
    catalogue (see ``sample_pair_sigma_chi``). Reduces to ``wp_observed``
    when all samples have the same value.

    Fully JAX-differentiable in (cosmo, bias, sigma_chi_samples).
    """
    import jax.numpy as jnp
    from jax.scipy.special import erf
    from .spectra import xi_from_Pk_fftlog

    rp = jnp.atleast_1d(jnp.asarray(rp, dtype=jnp.float64))
    sigma_samples = jnp.asarray(sigma_chi_samples, dtype=jnp.float64)
    sigma_safe = jnp.maximum(sigma_samples, 1e-8)
    bias = jnp.asarray(bias, dtype=jnp.float64)

    if fft is None or k_grid is None:
        fft, k_np = make_wp_fft()
        k_grid = jnp.asarray(k_np)

    P = pnl_at_z(k_grid, z=z_eff, sigma8=sigma8, cosmo=cosmo, Ob=Ob, ns=ns)
    s_grid = jnp.geomspace(0.5, 1500.0, 2048)
    xi_grid = xi_from_Pk_fftlog(s_grid, fft, P)

    pi_true = jnp.linspace(-pi_int_range, pi_int_range, n_pi_true)
    s_eval = jnp.sqrt(rp[:, None] ** 2 + pi_true[None, :] ** 2)   # (n_rp, n_pi)
    xi_eval = jnp.interp(s_eval.ravel(), s_grid, xi_grid).reshape(s_eval.shape)

    # Per-sample erf window, then average over samples.
    # shape: window[i, p] for sample i and pi-bin p
    sigma_sqrt2 = sigma_safe[:, None] * jnp.sqrt(2.0)
    window = 0.5 * (
        erf((pi_max - pi_true[None, :]) / sigma_sqrt2)
        + erf((pi_max + pi_true[None, :]) / sigma_sqrt2)
    )                                                              # (n_samp, n_pi)
    window_avg = jnp.mean(window, axis=0)                          # (n_pi,)

    integrand = xi_eval * window_avg[None, :]
    wp = jnp.trapezoid(integrand, pi_true, axis=1)
    return bias ** 2 * wp


def joint_cl_wp_map_fit(
    ell, cl_meas, sigma_cl,
    z_grid_dndz, dndz,
    rp, wp_meas, sigma_wp, sigma_chi_samples, z_eff,
    free=("Om", "sigma8", "b"),
    fix=None,
    pi_max: float = 200.0,
    h: float = 0.68,
    Ob: float = 0.049,
    ns: float = 0.965,
    ell_min: float = 20.0,
    n_pi_true: int = 400,
):
    """Joint MAP fit on (C_ell, wp(rp)) -- breaks the wp-only sigma8-b
    degeneracy.

    The two probes have different sigma8/bias degeneracy directions:
    wp(rp) ~ b^2 sigma8^2 P_NL(k, z=z_eff) at small scales; C_ell^gg
    integrates over the dndz with chi(z) entering geometrically. The
    joint chi^2 = chi^2(C_ell, ell > ell_min) + chi^2(wp) breaks the
    degeneracy through the two probes' different sensitivity to
    (Om, chi(z), P_NL shape).

    All gradients flow through ``cl_gg_limber`` (JAX-vmap'd halofit) and
    ``wp_observed_perpair`` (analytic erf window over the empirical
    pair-sigma distribution).
    """
    import jax
    import jax.numpy as jnp
    from .fit import map_fit

    fix = dict(fix or {})
    free = tuple(free)
    for name in free:
        if name not in _PARAM_NAMES:
            raise ValueError(f"unknown parameter {name}")
        fix.pop(name, None)
    fixed = {**{k: _PARAM_DEFAULTS[k] for k in _PARAM_NAMES if k not in free},
             **fix}

    ell_j = jnp.asarray(ell, dtype=jnp.float64)
    cl_j = jnp.asarray(cl_meas, dtype=jnp.float64)
    sig_cl_j = jnp.asarray(sigma_cl, dtype=jnp.float64)
    z_j = jnp.asarray(z_grid_dndz, dtype=jnp.float64)
    nz_j = jnp.asarray(dndz, dtype=jnp.float64)
    rp_j = jnp.asarray(rp, dtype=jnp.float64)
    wp_j = jnp.asarray(wp_meas, dtype=jnp.float64)
    sig_wp_j = jnp.asarray(sigma_wp, dtype=jnp.float64)
    sig_pair_j = jnp.asarray(sigma_chi_samples, dtype=jnp.float64)
    cl_use = ell_j > ell_min                                # mask low-ell

    fft, k_np = make_wp_fft()
    k_grid_wp = jnp.asarray(k_np)

    def expand(theta):
        out = {**fixed}
        for i, name in enumerate(free):
            out[name] = theta[i]
        return out

    def loss(theta):
        p = expand(theta)
        cosmo = DistanceCosmo(Om=p["Om"], h=h)
        cl_pred = cl_gg_limber(
            ell_j, z_j, nz_j, cosmo, bias=p["b"], sigma8=p["sigma8"],
            Ob=Ob, ns=ns,
        )
        wp_pred = wp_observed_perpair(
            rp_j, z_eff=z_eff, sigma_chi_samples=sig_pair_j,
            cosmo=cosmo, bias=p["b"], pi_max=pi_max,
            n_pi_true=n_pi_true, sigma8=p["sigma8"],
            Ob=Ob, ns=ns, fft=fft, k_grid=k_grid_wp,
        )
        chi2_cl = jnp.sum(jnp.where(
            cl_use, ((cl_j - cl_pred) / sig_cl_j) ** 2, 0.0,
        ))
        chi2_wp = jnp.sum(((wp_j - wp_pred) / sig_wp_j) ** 2)
        return chi2_cl + chi2_wp

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
