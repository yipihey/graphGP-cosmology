"""Evolution-aware clustering: pair-z forward model + optimal weights.

For a wide-redshift sample, the LS estimator measures::

    wp_LS(rp) = <b(z_1) b(z_2) xi(s; z_pair)>_pair (after pi-integration)

For pairs that survive the LOS pi_max cut: |z_1 - z_2| is small
(< 0.12 for Quaia at pi_max = 200 Mpc/h), so b(z_1) b(z_2) ~ b^2(z_pair),
and the estimator collapses to::

    wp_LS(rp) ~ <b^2(z_pair) D^2(z_pair)>_pair * wp_real(rp; z=0)

The averaging weight is the *empirical* pair-z PDF, which is the
auto-convolution of the data n(z) restricted to |Delta_chi| < pi_max.
This module provides:

  pair_z_distribution(z_data, cosmo, pi_max, n_pairs)
      -> (z_grid, p_pair_pdf): the empirical pair-redshift PDF for
      pairs surviving the pi_max LOS cut. Substitutes for the
      n^2(z) approximation in ``wp_observed_continuous_bz`` for
      a more rigorous evolution-aware forward model.

  wp_pair_evolved(rp, z_grid, p_pair_pdf, b_z, cosmo, ...)
      -> wp_pred(rp): JAX-pure forward model that integrates over the
      empirical pair-z distribution with explicit b^2(z) weighting and
      the proper photo-z LOS smearing per z. Differentiable in
      (cosmo, b_z, sigma_chi) so it drops into MAP / HMC fits.

  effective_amplitude_under_evolution(z_grid, p_pair_pdf, b_z, cosmo)
      -> sqrt(<b^2(z) D^2(z)>_pair / D^2(z_eff)): the effective
      multiplicative bias the standard "single-z_eff" estimator
      recovers, divided by the "true" b(z_eff). Quantifies the bias
      from using a single z_eff approximation on a wide-z sample.

  optimal_clustering_weights(z_data, b_z_grid, z_grid_b, cosmo, ...)
      -> per-galaxy weights w_i ~ b(z_i) D(z_i): Modi & White-style
      optimal weighting for measuring the clustering amplitude on a
      wide-z sample. Reduces the variance of the bias / amplitude
      estimator at no SNR cost (Poisson-dominated regime).
"""

from __future__ import annotations

import numpy as np


def pair_z_distribution(
    z_data: np.ndarray,
    cosmo,
    pi_max: float = 200.0,
    n_pairs: int = 50_000,
    n_bins: int = 80,
    rng=None,
):
    """Empirical pair-redshift PDF for pairs surviving the pi_max cut.

    Samples ``n_pairs`` random index pairs from the data, computes
    ``z_pair = (z_i + z_j)/2`` and ``Delta_chi = chi(z_i) - chi(z_j)``,
    and histograms z_pair for pairs with ``|Delta_chi| < pi_max``.
    This is the proper weighting in the evolution-aware wp forward
    model -- replaces the n^2(z) approximation with the actual pair
    distribution.

    Returns
    -------
    z_centres : (n_bins,) z-grid
    p_pair    : (n_bins,) PDF normalised so int p dz = 1
    n_kept    : number of pairs that passed the LOS cut
    """
    import jax.numpy as jnp
    from .distance import comoving_distance

    if rng is None:
        rng = np.random.default_rng(0)
    z_data = np.asarray(z_data, dtype=np.float64)
    n = len(z_data)
    chi_d = np.asarray(comoving_distance(jnp.asarray(z_data), cosmo))

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % n
    z_pair = 0.5 * (z_data[i] + z_data[j])
    delta_chi = np.abs(chi_d[i] - chi_d[j])
    keep = delta_chi < pi_max
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError("no pairs survive the pi_max cut")
    z_lo, z_hi = z_data.min(), z_data.max()
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(z_pair[keep], bins=edges)
    pdf = counts.astype(np.float64)
    pdf = pdf / np.trapezoid(pdf, centres)
    return centres, pdf, n_kept


def wp_pair_evolved(
    rp,
    z_grid,
    p_pair_pdf,
    b_z,
    sigma_chi_z,
    cosmo,
    pi_max: float = 200.0,
    pi_int_range: float = 800.0,
    n_pi_true: int = 400,
    sigma8: float = 0.8,
    Ob: float = 0.049,
    ns: float = 0.965,
    fft=None,
    k_grid=None,
):
    """Pair-evolution-aware wp(rp) forward model.

    Integrates the per-z wp_observed prediction over the empirical
    pair-z PDF, with per-z bias b(z) and per-z photo-z LOS smearing
    sigma_chi(z) entering inside the integral:

        wp_pred(rp) = int dz p_pair(z) * b^2(z)
                       * wp_observed(rp; z, sigma_chi(z))

    Reduces to ``wp_observed`` at a delta-function p_pair (single-z
    sample). For Quaia's z = 0.8 - 2.5 sample this differs from the
    single-z approximation by ~ 5-15% in the recovered amplitude
    (depending on b(z) shape).

    JAX-pure: differentiable in (cosmo, b_z, sigma_chi_z, sigma8).
    """
    import jax
    import jax.numpy as jnp
    from .limber import make_wp_fft, wp_observed

    rp_j = jnp.atleast_1d(jnp.asarray(rp, dtype=jnp.float64))
    z = jnp.asarray(z_grid, dtype=jnp.float64)
    pdf = jnp.asarray(p_pair_pdf, dtype=jnp.float64)
    b_z = jnp.asarray(b_z, dtype=jnp.float64)
    sigma_chi_z = jnp.asarray(sigma_chi_z, dtype=jnp.float64)

    if fft is None or k_grid is None:
        fft, k_np = make_wp_fft()
        k_grid = jnp.asarray(k_np)

    def wp_at_z(zi, sig_chi_i):
        return wp_observed(
            rp_j, z_eff=zi, sigma_chi_eff=sig_chi_i, cosmo=cosmo,
            bias=1.0, pi_max=pi_max, pi_int_range=pi_int_range,
            n_pi_true=n_pi_true, sigma8=sigma8, Ob=Ob, ns=ns,
            fft=fft, k_grid=k_grid,
        )

    wp_grid = jax.vmap(wp_at_z)(z, sigma_chi_z)        # (n_z, n_rp)
    weight = pdf * b_z ** 2                             # (n_z,)
    integrand = weight[:, None] * wp_grid               # (n_z, n_rp)
    norm = jnp.trapezoid(pdf, z)                        # = 1 by construction
    return jnp.trapezoid(integrand, z, axis=0) / norm


def effective_amplitude_under_evolution(
    z_grid: np.ndarray,
    p_pair_pdf: np.ndarray,
    b_z,
    cosmo,
    z_eff: float = None,
):
    """Effective multiplicative bias of the single-z_eff approximation.

    Returns ``A_eff = sqrt(<b^2(z) D^2(z)>_pair) / [b(z_eff) D(z_eff)]``,
    i.e. the ratio between the amplitude the standard "fit a single
    bias at z_eff" estimator recovers and the value you'd want at
    the chosen z_eff. ``A_eff != 1`` is the "wide-z bias".

    For Quaia spanning [0.8, 2.5], with median z = 1.5 and the
    published b(z) ~ (1+z)^0.7, this is typically 1.05 - 1.15
    -- i.e. the recovered single-z bias is ~ 5-15 % too high
    compared to b(z_eff = z_median).
    """
    import jax.numpy as jnp
    from .limber import linear_growth

    z = jnp.asarray(z_grid, dtype=jnp.float64)
    pdf = jnp.asarray(p_pair_pdf, dtype=jnp.float64)
    b_z = jnp.asarray(b_z, dtype=jnp.float64)
    D_z = linear_growth(z, cosmo)

    integrand = pdf * b_z ** 2 * D_z ** 2
    avg_b2_D2 = float(jnp.trapezoid(integrand, z))

    if z_eff is None:
        # mean z under the pair PDF
        z_eff = float(jnp.trapezoid(pdf * z, z))
    z_eff_j = jnp.asarray([z_eff], dtype=jnp.float64)
    b_at_z = float(jnp.interp(z_eff_j, z, b_z)[0])
    D_at_z = float(linear_growth(z_eff_j, cosmo)[0])
    return float(np.sqrt(avg_b2_D2) / max(b_at_z * D_at_z, 1e-12))


def optimal_clustering_weights(
    z_data: np.ndarray,
    b_z_grid: np.ndarray,
    z_grid_b: np.ndarray,
    cosmo,
    P_target_mpc3h3: float = 0.0,
):
    """Per-galaxy FKP-like weights for measuring the clustering amplitude
    on a wide-z sample.

    Modi & White (2023)-style optimal weight for a target observable
    that scales as b(z) D(z) (e.g. b * sigma8(z)):

        w(z_i) ~ b(z_i) D(z_i) / [1 + n(z_i) P_target]

    For Quaia (n_data ~ 1e-6 / Mpc^3, P_target ~ 1e4 Mpc^3 at BAO scale)
    n*P ~ 0.01 << 1, so the FKP correction is negligible and the weight
    reduces to ``w(z) ~ b(z) D(z)``. This concentrates the SNR on
    epochs where the clustering signal is strongest.

    Weights are normalised so mean = 1 across the input galaxies.
    """
    import jax.numpy as jnp
    from .limber import linear_growth

    z_data = np.asarray(z_data, dtype=np.float64)
    z_grid_b = np.asarray(z_grid_b, dtype=np.float64)
    b_z_grid = np.asarray(b_z_grid, dtype=np.float64)
    b_at = np.interp(z_data, z_grid_b, b_z_grid)
    D_at = np.asarray(linear_growth(jnp.asarray(z_data), cosmo))
    if P_target_mpc3h3 > 0:
        # estimate n(z) from the data via histogram
        n_bins = 80
        z_lo, z_hi = z_data.min(), z_data.max()
        edges = np.linspace(z_lo, z_hi, n_bins + 1)
        counts, _ = np.histogram(z_data, bins=edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # comoving volume per dz
        from .distance import comoving_distance
        chi_centres = np.asarray(comoving_distance(jnp.asarray(centres),
                                                     cosmo))
        # crude: n(z) = counts / (volume slice). Using chi^2 dchi/dz dOmega.
        # for Quaia f_sky ~ 0.66; we use 4 pi to keep nP dimensionless,
        # the absolute normalization rarely matters since nP << 1.
        from .distance import C_OVER_H100_MPCH, E_of_z
        E = np.asarray(E_of_z(jnp.asarray(centres), cosmo))
        dchi_dz = C_OVER_H100_MPCH / E
        vol_per_dz = 4 * np.pi * 0.66 * chi_centres ** 2 * dchi_dz
        n_z = counts.astype(np.float64) / np.maximum(vol_per_dz, 1e-20)
        n_at = np.interp(z_data, centres, n_z)
        denom = 1.0 + n_at * P_target_mpc3h3
    else:
        denom = 1.0
    w = b_at * D_at / denom
    w = w / w.mean()
    return w
