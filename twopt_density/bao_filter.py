"""BAO matched-filter analysis on wp(rp) residuals.

The maximum-likelihood BAO amplitude on a wp(rp) measurement at fixed
cosmology is obtained by projecting the data onto the BAO template
``T(rp) = b^2 (wp_full - wp_nowiggle)``::

    A_hat = T^T C^-1 (wp_data - wp_smooth) / (T^T C^-1 T)
    sigma_A = 1 / sqrt(T^T C^-1 T)
    SNR    = A_hat / sigma_A

with ``C`` either diagonal (Poisson) or the full data covariance
(jackknife, mocks, or analytic). Combining BAO information across
all rp bins coherently gives a single SNR that beats per-bin
significance scaling roughly as ``sqrt(N_band_bins)``.

Adding a BAO scaling parameter ``alpha`` (sound-horizon-shift; alpha=1
is the fiducial cosmology) generalises this to a 2-parameter fit
``(alpha, A)``: scan alpha over a grid, evaluate the matched filter
at each, recover ``alpha_hat`` from the SNR maximum and ``sigma_alpha``
from the curvature.

The infrastructure is JAX-compatible -- ``A_hat(alpha, b)`` is
differentiable in cosmology so a joint cosmology + BAO MAP fit is
one extra LBFGS call away.
"""

from __future__ import annotations

import numpy as np


def bao_template(
    rp_grid,
    b: float,
    z_eff: float,
    sigma_chi_eff: float,
    pi_max: float,
    cosmo,
    alpha: float = 1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049,
    ns: float = 0.965,
    fft=None,
    k_grid=None,
):
    """BAO template ``T(rp) = b^2 (wp_full - wp_nowiggle)`` at z_eff,
    with optional scaling ``alpha`` so the template is evaluated at
    ``rp/alpha`` (BAO sound-horizon-shift parameterisation).

    Returns ``T`` evaluated on ``rp_grid``.
    """
    import jax.numpy as jnp
    from .limber import (
        make_wp_fft, wp_observed, wp_observed_nowiggle,
    )

    rp = np.asarray(rp_grid, dtype=np.float64) / float(alpha)
    if fft is None or k_grid is None:
        fft, k_np = make_wp_fft()
        k_grid = jnp.asarray(k_np)
    rp_j = jnp.asarray(rp)
    wp_full = np.asarray(wp_observed(
        rp_j, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff, cosmo=cosmo,
        bias=1.0, pi_max=pi_max, sigma8=sigma8, Ob=Ob, ns=ns,
        fft=fft, k_grid=k_grid,
    ))
    wp_smooth = np.asarray(wp_observed_nowiggle(
        rp_j, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff, cosmo=cosmo,
        bias=1.0, pi_max=pi_max, sigma8=sigma8, Ob=Ob, ns=ns,
        fft=fft, k_grid=k_grid,
    ))
    return b ** 2 * (wp_full - wp_smooth)


def matched_filter_amplitude(
    wp_data: np.ndarray,
    wp_smooth: np.ndarray,
    T: np.ndarray,
    sigma_or_cov,
):
    """Maximum-likelihood BAO amplitude with either diagonal or full
    data covariance.

    Parameters
    ----------
    wp_data    : (N,) measured wp(rp_i)
    wp_smooth  : (N,) no-wiggle model evaluated at the same rp_i and
                 fitted bias
    T          : (N,) BAO template evaluated at the same rp_i
    sigma_or_cov : (N,) diagonal sigmas, or (N, N) full covariance

    Returns
    -------
    A_hat      : best-fit amplitude (1.0 = fiducial halofit BAO)
    sigma_A    : 1-sigma uncertainty
    SNR        : A_hat / sigma_A
    chi2_null  : chi^2 with A=0 (no-BAO null)
    chi2_best  : chi^2 at the best-fit A
    """
    wp_data = np.asarray(wp_data, dtype=np.float64)
    wp_smooth = np.asarray(wp_smooth, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    res = wp_data - wp_smooth                            # (N,)
    sc = np.asarray(sigma_or_cov, dtype=np.float64)
    if sc.ndim == 1:
        # diagonal noise
        invsig2 = 1.0 / np.maximum(sc, 1e-30) ** 2
        TtCinvT = float(np.sum(T ** 2 * invsig2))
        TtCinvr = float(np.sum(T * res * invsig2))
        rtCinvr = float(np.sum(res ** 2 * invsig2))
    else:
        # full covariance
        Cinv = np.linalg.inv(sc)
        TtCinvT = float(T @ Cinv @ T)
        TtCinvr = float(T @ Cinv @ res)
        rtCinvr = float(res @ Cinv @ res)
    if TtCinvT <= 0:
        return float("nan"), float("inf"), 0.0, rtCinvr, rtCinvr
    A_hat = TtCinvr / TtCinvT
    sigma_A = 1.0 / np.sqrt(TtCinvT)
    SNR = A_hat / sigma_A
    chi2_null = rtCinvr
    chi2_best = rtCinvr - A_hat ** 2 * TtCinvT
    return A_hat, sigma_A, SNR, chi2_null, chi2_best


def bao_alpha_scan(
    rp_grid,
    wp_data,
    wp_smooth,
    b: float,
    z_eff: float,
    sigma_chi_eff: float,
    pi_max: float,
    cosmo,
    sigma_or_cov,
    alpha_grid=None,
    sigma8: float = 0.81,
):
    """Scan ``alpha`` (BAO scaling parameter) and return best-fit
    amplitude + significance.

    For each alpha, the BAO template is evaluated at rp/alpha and the
    matched-filter amplitude is computed. The chi^2(alpha) curve from
    a parabolic fit around the minimum gives ``alpha_hat`` and
    ``sigma_alpha``.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.85, 1.15, 31)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    A_arr = np.zeros_like(alpha_grid)
    sd_arr = np.zeros_like(alpha_grid)
    SNR_arr = np.zeros_like(alpha_grid)
    chi2_arr = np.zeros_like(alpha_grid)
    for i, a in enumerate(alpha_grid):
        T_a = bao_template(
            rp_grid, b=b, z_eff=z_eff, sigma_chi_eff=sigma_chi_eff,
            pi_max=pi_max, cosmo=cosmo, alpha=float(a),
            sigma8=sigma8,
        )
        A, sd, snr, _, c2 = matched_filter_amplitude(
            wp_data, wp_smooth, T_a, sigma_or_cov,
        )
        A_arr[i] = A; sd_arr[i] = sd; SNR_arr[i] = snr; chi2_arr[i] = c2

    # parabolic fit around the chi^2 minimum -> alpha_hat, sigma_alpha
    i_min = int(np.argmin(chi2_arr))
    if 0 < i_min < len(alpha_grid) - 1:
        a0, a1, a2 = alpha_grid[i_min - 1: i_min + 2]
        c0, c1, c2 = chi2_arr[i_min - 1: i_min + 2]
        # quadratic chi^2(alpha) = c0 + 2 b' (alpha - a_hat) + a' (alpha - a_hat)^2
        denom = (a0 - a1) * (a0 - a2) * (a1 - a2)
        A_q = (a2 * (c1 - c0) + a1 * (c0 - c2) + a0 * (c2 - c1)) / denom
        B_q = (a2 ** 2 * (c0 - c1) + a1 ** 2 * (c2 - c0) + a0 ** 2 * (c1 - c2)) / denom
        if A_q > 0:
            alpha_hat = -B_q / (2 * A_q)
            sigma_alpha = 1.0 / np.sqrt(A_q)
        else:
            alpha_hat = float(alpha_grid[i_min])
            sigma_alpha = float("nan")
    else:
        alpha_hat = float(alpha_grid[i_min])
        sigma_alpha = float("nan")
    return {
        "alpha_grid": alpha_grid,
        "A": A_arr, "sigma_A": sd_arr, "SNR": SNR_arr,
        "chi2": chi2_arr,
        "alpha_hat": float(alpha_hat),
        "sigma_alpha": float(sigma_alpha),
        "A_hat_at_best": float(A_arr[i_min]),
        "SNR_at_best": float(SNR_arr[i_min]),
    }
