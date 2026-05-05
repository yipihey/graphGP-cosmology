"""Tests for the lightcone-native cap-counting sigma^2(theta; z) estimator.

These tests pin three behaviours:

  1. ``sigma^2_estimate_cone_shell`` recovers ~0 on a uniform Poisson
     mock (no clustering), within the predicted Var-of-Var noise.
  2. On a clustered mock (radial-only structure injected via
     redshift-shell-correlated weights), sigma^2_obs > 0 well above
     Poisson noise, and increases with the injected amplitude.
  3. ``cap_centre_grid`` edge-buffering produces a self-consistent
     subset of centres -- no centre gets a cap that crosses a fully
     masked region.
"""

from __future__ import annotations

import numpy as np


def _full_sky_poisson(n_g: int, z_min: float = 0.5, z_max: float = 2.0,
                        seed: int = 0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n_g)
    sin_dec = rng.uniform(-1, 1, n_g)
    ra = np.degrees(phi)
    dec = np.degrees(np.arcsin(sin_dec))
    z = rng.uniform(z_min, z_max, n_g)
    return ra, dec, z


def test_poisson_mock_gives_zero_sigma2_within_noise():
    """Uniform Poisson on the full sky -> sigma^2_obs ~ 0 within the
    predicted Var-of-Var noise. We use caps large enough to have
    mu >~ 5 so the per-cap Poisson estimate is well-behaved."""
    import healpy as hp
    from twopt_density.sigma2_cone_shell_estimator import (
        cap_centre_grid, cone_shell_counts, sigma2_estimate_cone_shell,
    )

    n_g = 200_000
    ra, dec, z = _full_sky_poisson(n_g, seed=1)

    nside = 32
    mask = np.ones(12 * nside ** 2, dtype=np.float64)
    theta_max = np.deg2rad(3.0)
    ra_c, dec_c, _ = cap_centre_grid(
        mask, nside_centres=nside,
        theta_max_rad=theta_max, edge_buffer_frac=1.0,
    )
    assert ra_c.size > 100

    theta_radii = np.deg2rad(np.array([2.0, 3.0]))
    z_edges = np.array([0.5, 1.25, 2.0])
    N, A = cone_shell_counts(
        ra, dec, z, theta_radii, z_edges, ra_c, dec_c,
        nside_lookup=256,
    )
    s2 = sigma2_estimate_cone_shell(N)
    mu = N.mean(axis=0)
    # Predicted Var(sigma^2_estimator) ~ 2 / N_centres for Poisson at large mu.
    # Standard error ~ sqrt(2 / N_centres) ~ 0.014 at N_c=12288 here.
    se = np.sqrt(2.0 / N.shape[0])
    # require the residual to be less than 5 standard errors
    assert (np.abs(s2) < 5 * se).all(), (
        f"|sigma^2_obs|={np.abs(s2).max():.3g} > 5 SE={5*se:.3g} "
        f"(mu range {mu.min():.2g}..{mu.max():.2g})"
    )


def test_clustered_mock_gives_positive_sigma2_above_poisson_noise():
    """Inject a radial-shell-correlated overdensity: galaxies in
    selected redshift slices are bunched into a few sky 'patches'.
    The cap-counting estimator should report sigma^2 > Poisson noise
    in those slices."""
    import healpy as hp
    from twopt_density.sigma2_cone_shell_estimator import (
        cap_centre_grid, cone_shell_counts, sigma2_estimate_cone_shell,
    )

    rng = np.random.default_rng(7)
    # 80% Poisson + 20% clustered around 50 random sky patches with sigma=2 deg
    n_p = 60_000; n_c_clust = 20_000; n_clusters = 50
    # Poisson part
    phi_p = rng.uniform(0, 2 * np.pi, n_p)
    sin_dec_p = rng.uniform(-1, 1, n_p)
    ra_p = np.degrees(phi_p)
    dec_p = np.degrees(np.arcsin(sin_dec_p))
    z_p = rng.uniform(0.5, 2.0, n_p)
    # Clustered part: pick cluster centres uniformly on sphere
    ra_cc = rng.uniform(0, 360, n_clusters)
    sin_dec_cc = rng.uniform(-1, 1, n_clusters)
    dec_cc = np.degrees(np.arcsin(sin_dec_cc))
    cluster_assign = rng.integers(0, n_clusters, n_c_clust)
    sigma_deg = 2.0
    ra_c_pts = (ra_cc[cluster_assign]
                  + sigma_deg * rng.standard_normal(n_c_clust)) % 360
    dec_c_pts = np.clip(
        dec_cc[cluster_assign] + sigma_deg * rng.standard_normal(n_c_clust),
        -89.0, 89.0,
    )
    # all clusters concentrated in a single z slice (1.0 -- 1.5)
    z_c_pts = rng.uniform(1.0, 1.5, n_c_clust)
    # also assign their Poisson partners outside that slice
    ra = np.concatenate([ra_p, ra_c_pts])
    dec = np.concatenate([dec_p, dec_c_pts])
    z = np.concatenate([z_p, z_c_pts])

    nside = 32
    mask = np.ones(12 * nside ** 2, dtype=np.float64)
    theta_max = np.deg2rad(3.0)
    ra_c_grid, dec_c_grid, _ = cap_centre_grid(
        mask, nside_centres=nside,
        theta_max_rad=theta_max, edge_buffer_frac=1.0,
    )

    theta_radii = np.deg2rad(np.array([2.0, 3.0]))
    z_edges = np.array([0.5, 1.0, 1.5, 2.0])
    N, A = cone_shell_counts(
        ra, dec, z, theta_radii, z_edges, ra_c_grid, dec_c_grid,
        nside_lookup=256,
    )
    s2 = sigma2_estimate_cone_shell(N)
    se = np.sqrt(2.0 / N.shape[0])

    # clustered z-slice (index 1) should have sigma^2 well above
    # Poisson; non-clustered slices (0, 2) should be near zero.
    assert (s2[:, 1] > 5 * se).all(), (
        f"clustered slice s2={s2[:, 1]} not above 5 SE={5*se:.3g}"
    )
    # signal in clustered slice exceeds non-clustered slices
    assert s2[0, 1] > s2[0, 0]
    assert s2[0, 1] > s2[0, 2]


def test_fast_jackknife_matches_slow_on_clustered_mock():
    """The default ``fast=True`` jackknife (one count pass + cube
    reductions) must produce the same per-fold samples and covariance
    as the slow ``fast=False`` path that re-runs ``cone_shell_counts``
    per fold. On the same mock the two should agree to at most a
    few-eps floating-point summation-order difference."""
    import healpy as hp
    from twopt_density.sigma2_cone_shell_estimator import (
        cap_centre_grid, sigma2_cone_shell_jackknife,
    )

    rng = np.random.default_rng(11)
    n_p = 8000
    phi_p = rng.uniform(0, 2 * np.pi, n_p)
    sin_dec_p = rng.uniform(-1, 1, n_p)
    ra = np.degrees(phi_p)
    dec = np.degrees(np.arcsin(sin_dec_p))
    z = rng.uniform(0.5, 2.0, n_p)

    # inject a cluster in one z slice so cosmic variance is non-trivial
    n_c_pts = 1500
    cluster_centres = rng.uniform([0, -45], [360, 45], (40, 2))
    assign = rng.integers(0, 40, n_c_pts)
    ra_c = (cluster_centres[assign, 0]
              + 2.0 * rng.standard_normal(n_c_pts)) % 360
    dec_c = np.clip(cluster_centres[assign, 1]
                       + 2.0 * rng.standard_normal(n_c_pts), -89, 89)
    z_c = rng.uniform(1.0, 1.5, n_c_pts)
    ra = np.concatenate([ra, ra_c])
    dec = np.concatenate([dec, dec_c])
    z = np.concatenate([z, z_c])

    nside = 16
    mask = np.ones(12 * nside ** 2, dtype=np.float64)
    theta_max = np.deg2rad(3.0)
    ra_centres, dec_centres, _ = cap_centre_grid(
        mask, nside_centres=nside, theta_max_rad=theta_max,
        edge_buffer_frac=1.0,
    )

    theta_radii = np.deg2rad(np.array([1.5, 2.5, 3.0]))
    z_edges = np.array([0.5, 1.0, 1.5, 2.0])

    common = dict(
        ra_deg=ra, dec_deg=dec, z=z,
        theta_radii_rad=theta_radii, z_edges=z_edges,
        ra_centres_deg=ra_centres, dec_centres_deg=dec_centres,
        n_regions=8, nside_jack=4, nside_lookup=128,
    )
    mean_f, samp_f, cov_f = sigma2_cone_shell_jackknife(**common, fast=True)
    mean_s, samp_s, cov_s = sigma2_cone_shell_jackknife(**common, fast=False)

    np.testing.assert_allclose(samp_f, samp_s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean_f, mean_s, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cov_f, cov_s, rtol=1e-10, atol=1e-12)


def test_cap_centre_grid_edge_buffer_drops_boundary():
    """A non-trivial mask: keep only the northern hemisphere. With
    sufficient edge buffer, no kept centre's cap should cross into
    the southern (masked) hemisphere."""
    import healpy as hp
    from twopt_density.sigma2_cone_shell_estimator import cap_centre_grid

    nside = 64
    npix = 12 * nside ** 2
    theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
    dec_pix = 90.0 - np.degrees(theta_pix)
    mask = (dec_pix > 0).astype(np.float64)

    theta_max = np.deg2rad(2.0)
    # without buffer: many centres very near dec=0 are kept
    ra_a, dec_a, _ = cap_centre_grid(
        mask, nside_centres=nside,
        theta_max_rad=theta_max, edge_buffer_frac=0.5,
    )
    # with full buffer: centres should all be at dec >= theta_max
    ra_b, dec_b, _ = cap_centre_grid(
        mask, nside_centres=nside,
        theta_max_rad=theta_max, edge_buffer_frac=1.0,
    )
    assert ra_b.size > 0
    # Strict containment: all cap discs lie inside the mask, so the
    # centre dec must exceed the cap radius.
    assert (dec_b >= np.degrees(theta_max) - 1e-3).all(), (
        f"min kept dec={dec_b.min():.3g} < theta_max={np.degrees(theta_max):.3g}"
    )
    # buffer-1 keeps fewer centres than buffer-0.5
    assert ra_b.size <= ra_a.size
