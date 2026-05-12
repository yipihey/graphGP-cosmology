"""Analytic and Monte-Carlo random-query adapters for the joint kNN-CDF.

For Landy-Szalay angular xi (paper Eq. 12) we need three KnnCdfResult
cubes — DD, DR, RR — sharing the same theta and z grids. Two paths are
provided:

- ``random_queries_from_selection_function``: draws random (RA, Dec, z)
  matched to the survey's ``mask(Omega) * n(z)``. Plug into
  ``joint_knn_cdf`` to get DR/RR via Monte Carlo. Per the paper §5.1
  the noise scales as ``1/sqrt(N_q)`` — ``N_R = 5 N_D`` is the sweet
  spot.

- ``analytic_rr_cube``: for separable windows ``W(Omega, z) =
  mask(Omega) n(z)``, the expected RR mean count per query has a
  closed form involving the angular auto-correlation of the mask and
  the data n(z). Wraps ``analytic_rr.angular_corr_from_mask`` and
  packages the prediction into a ``KnnCdfResult`` shape so it drops
  into ``knn_derived.xi_ls`` directly.

The MC path is the production default; the analytic path is for very
large surveys where even ``5 N_D`` randoms become a performance issue
(or for diagnostic comparison against the MC randoms).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .knn_cdf import KnnCdfResult


def random_queries_from_selection_function(
    sel_map: np.ndarray,
    z_data: np.ndarray,
    n_random: int,
    nside: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper around ``quaia.make_random_from_selection_function``.

    Returns ``(ra_deg, dec_deg, z)`` arrays of length ``n_random`` drawn
    from ``mask(Omega) * n(z)``. Use the same array as both the
    *query* and the *neighbor* catalog (with ``flavor="DD"`` and
    object-identity preserved) to get an RR cube; pair with the data
    catalog to get a DR cube.
    """
    from .quaia import make_random_from_selection_function

    return make_random_from_selection_function(
        sel_map=sel_map, n_random=n_random, z_data=z_data,
        nside=nside, rng=rng,
    )


def _mean_windowed_cap_area(
    sel_map: np.ndarray, nside: int,
    query_ra_deg: np.ndarray, query_dec_deg: np.ndarray,
    theta_radii_rad: np.ndarray,
    n_subsample: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """⟨A_w(q,θ)⟩ averaged over (subsampled) query positions, where
    A_w(q,θ) = ∫_cap(q,θ) sel(Ω) dΩ is the windowed cap area at cap
    centre q. Returns a length-n_theta array.

    For binary masks A_w ≈ Omega_cap and ⟨A_w⟩/Omega_cap ≈ 1; for
    continuous masks (Quaia, DESI completeness maps) the ratio is < 1
    and θ-dependent — exactly the "c_mask" correction the analytic-RR
    formula needs to be window-correct.
    """
    import healpy as hp
    rng = np.random.default_rng(seed)
    if query_ra_deg.size > n_subsample:
        take = rng.choice(query_ra_deg.size, n_subsample, replace=False)
    else:
        take = np.arange(query_ra_deg.size)
    ra_s = query_ra_deg[take]; dec_s = query_dec_deg[take]
    pix_area = hp.nside2pixarea(nside)
    pix_radius = float(np.sqrt(pix_area / np.pi))   # effective pixel radius
    theta_q = np.deg2rad(90.0 - dec_s)
    phi_q = np.deg2rad(ra_s % 360.0)
    vecs_q = hp.ang2vec(theta_q, phi_q)
    theta_max = float(theta_radii_rad.max())
    n_t = theta_radii_rad.size
    Omega_cap_per_t = 2.0 * np.pi * (1.0 - np.cos(theta_radii_rad))
    # Per-query pixel sel value at the query position — used for the
    # small-θ regime where the cap is entirely inside the query's pixel
    # and A_w = sel(q_pix) * Omega_cap. The pixel-cumulative algorithm
    # below otherwise rounds A_w to 0 at θ < pix_radius (the closest
    # pixel center can be > θ from the query).
    qpix_idx = hp.ang2pix(nside, theta_q, phi_q)
    sel_at_q = sel_map[qpix_idx]
    A_w = np.zeros((take.size, n_t), dtype=np.float64)
    for i in range(take.size):
        ipix = hp.query_disc(nside, vecs_q[i], theta_max, inclusive=True)
        if ipix.size == 0:
            continue
        theta_pix, phi_pix = hp.pix2ang(nside, ipix)
        cos_sep = (np.sin(theta_q[i]) * np.sin(theta_pix)
                   * np.cos(phi_q[i] - phi_pix)
                   + np.cos(theta_q[i]) * np.cos(theta_pix))
        sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
        w_pix = sel_map[ipix]
        order = np.argsort(sep)
        sep_s = sep[order]; w_s = w_pix[order]
        cum = np.cumsum(w_s) * pix_area
        idx = np.searchsorted(sep_s, theta_radii_rad, side="right") - 1
        for t in range(n_t):
            if theta_radii_rad[t] < pix_radius:
                # Cap inside one pixel: use exact small-θ form.
                A_w[i, t] = sel_at_q[i] * Omega_cap_per_t[t]
            else:
                A_w[i, t] = cum[idx[t]] if idx[t] >= 0 else 0.0
    return A_w.mean(axis=0)


def analytic_rr_cube(
    sel_map: np.ndarray,
    z_data: np.ndarray,
    theta_radii_rad: np.ndarray,
    z_q_edges: np.ndarray, z_n_edges: np.ndarray,
    n_q_per_shell: np.ndarray,
    n_random_total: int,
    nside: Optional[int] = None,
    lmax: Optional[int] = None,
    taper: str = "hann",
    query_ra_deg: Optional[np.ndarray] = None,
    query_dec_deg: Optional[np.ndarray] = None,
    n_subsample_for_window: int = 5000,
    window_query_source: str = "random",
) -> KnnCdfResult:
    """Analytic prediction of the RR cube for a separable window.

    The prediction populates ``sum_n[t, iq, jn]`` with the expected
    Sum-over-queries cap count of randoms in shell ``z_n``. Other cube
    slots (``sum_n2``, ``H_geq_k``, ``N_q_per_region`` …) are left
    zero — the analytic cube is intended for ``xi_ls`` and
    ``mean_count`` consumers only.

    Parameters
    ----------
    sel_map
        Healpix completeness ``[0, 1]`` of the survey footprint.
    z_data
        Data redshifts (used as the empirical ``n(z)`` for the
        analytic prediction). ``random_queries_from_selection_function``
        uses the same recipe.
    theta_radii_rad, z_q_edges, z_n_edges
        Same axes as the matching DD/DR cubes.
    n_q_per_shell
        ``(n_z_q,)`` integer array — the number of random queries in
        each ``z_q`` shell. Set this to the per-shell counts of the
        actual MC random query catalog used for DD/DR, so all three
        cubes share the same denominator scale.
    n_random_total
        Total number of randoms (matches the MC path's ``N_r``).
        Together with ``n_q_per_shell`` this determines the random
        density on the sky in each shell.
    nside, lmax, taper
        Forwarded to ``analytic_rr.angular_corr_from_mask``.

    Notes
    -----
    The construction:

      Per random query at angular position q, the expected count of
      randoms in shell ``jn`` within an angular cap of opening angle
      ``θ`` (under the separable window ``W(Ω, z) = sel(Ω) n(z)``) is

          E[N_jn | cap at q, θ] = N_jn · A_w(q, θ) / Ω_mask_w

      where ``A_w(q, θ) = ∫_cap(q,θ) sel(Ω) dΩ`` is the windowed cap
      area at q and ``Ω_mask_w = ∫ sel dΩ`` is the effective mask
      area. Averaging over query positions q (drawn from sel),

          ⟨E[N_jn | cap, θ]⟩ = N_jn · ⟨A_w(q, θ)⟩_q / Ω_mask_w.

      For binary masks, ⟨A_w⟩_q ≈ Ω_cap (the nominal cap area), and
      this reduces to the simple ``Ω_cap / Ω_mask`` form. For
      continuous masks (Quaia, DESI completeness), ⟨A_w⟩_q < Ω_cap
      because the cap intersects regions where ``sel < 1``.

      This implementation supports two regimes for ⟨A_w⟩:
        - ``window_query_source="random"`` (default, recommended):
          draws ``n_subsample_for_window`` queries from ``sel_map``
          itself, exactly matching the MC random query distribution.
        - ``window_query_source="data"``: uses the supplied
          ``query_ra_deg``/``query_dec_deg`` as proxy positions.
          Carries a small clustering bias because data positions
          oversample higher-density (typically also higher-sel)
          pixels — at θ ≲ 1° the bias is ~3% in mean_count for DESI.

      If neither source is usable, falls back to the binary-mask
      shortcut ⟨A_w⟩ ≈ Ω_cap.

      The ``lmax`` and ``taper`` parameters are reserved for a future
      Legendre/spherical-harmonic implementation of ⟨A_w⟩ and are
      currently unused.
    """
    import healpy as hp

    if nside is None:
        nside = int(np.sqrt(sel_map.size / 12))
    if lmax is None:
        lmax = 3 * nside - 1

    n_theta = theta_radii_rad.size
    n_z_q = z_q_edges.size - 1
    n_z_n = z_n_edges.size - 1

    Omega_cap = 2.0 * np.pi * (1.0 - np.cos(theta_radii_rad))   # (n_theta,)
    # Effective windowed footprint area: ∫ sel dΩ.
    Omega_mask_w = float(sel_map.sum()) * (4.0 * np.pi / sel_map.size)

    # Predicted per-shell neighbor count: total randoms times the
    # fraction of the data n(z) PDF that falls in each shell.
    z_data = np.asarray(z_data)
    counts_per_shell, _ = np.histogram(z_data, bins=z_n_edges)
    if counts_per_shell.sum() == 0:
        raise ValueError("z_data falls entirely outside z_n_edges")
    f_per_shell = counts_per_shell.astype(np.float64) / float(z_data.size)
    N_n_per_shell = n_random_total * f_per_shell                # (n_z_n,)

    # Window-corrected mean cap area: ⟨A_w(q, θ)⟩_q. The query
    # distribution must match the MC random catalog (≡ sel_map),
    # otherwise clustering bias slips into the prediction.
    if window_query_source == "random":
        # Draw a fresh random subsample from sel_map (matches MC).
        rng = np.random.default_rng(0)
        ra_sub, dec_sub, _ = random_queries_from_selection_function(
            sel_map=sel_map, z_data=np.array([1.0]),
            n_random=n_subsample_for_window, nside=nside, rng=rng,
        )
        mean_A_w = _mean_windowed_cap_area(
            sel_map=sel_map, nside=nside,
            query_ra_deg=ra_sub, query_dec_deg=dec_sub,
            theta_radii_rad=theta_radii_rad,
            n_subsample=n_subsample_for_window, seed=0,
        )                                                       # (n_theta,)
    elif (window_query_source == "data"
          and query_ra_deg is not None and query_dec_deg is not None):
        mean_A_w = _mean_windowed_cap_area(
            sel_map=sel_map, nside=nside,
            query_ra_deg=query_ra_deg, query_dec_deg=query_dec_deg,
            theta_radii_rad=theta_radii_rad,
            n_subsample=n_subsample_for_window, seed=0,
        )                                                       # (n_theta,)
    else:
        mean_A_w = Omega_cap                                    # (n_theta,)

    # Predicted per-query expected neighbors per (theta, z_n).
    expected_per_query = (
        N_n_per_shell[None, :] * (mean_A_w / Omega_mask_w)[:, None]
    )                                                            # (n_theta, n_z_n)

    # sum_n[t, iq, jn] = N_q_per_shell[iq] * expected_per_query[t, jn]
    Nq = np.asarray(n_q_per_shell, dtype=np.int64)
    sum_n = (Nq[None, :, None].astype(np.float64)
             * expected_per_query[:, None, :])

    return KnnCdfResult(
        H_geq_k=np.zeros((n_theta, n_z_q, n_z_n, 1), dtype=np.int64),
        sum_n=sum_n,
        sum_n2=np.zeros_like(sum_n),
        N_q=Nq,
        theta_radii_rad=np.asarray(theta_radii_rad, dtype=np.float64),
        z_q_edges=np.asarray(z_q_edges, dtype=np.float64),
        z_n_edges=np.asarray(z_n_edges, dtype=np.float64),
        flavor="RR",
        backend_used="analytic",
        area_per_cap=Omega_cap,
    )
