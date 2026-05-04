"""Jackknife covariance for wp(rp) measurements.

Splits the survey sphere into ``n_regions`` ~equal-area patches via
low-NSIDE healpix indexing (each region is one or more low-NSIDE
pixels). For each region k = 0..N-1 we drop all data inside region k,
recompute wp(rp) (analytic RR + DD on the reduced sample + analytic DR
that scales as N_d^2/N_d_orig^2), and accumulate jackknife samples.

The covariance is the standard jackknife formula::

    C_ij = ((N - 1) / N) * sum_k (wp_k_i - <wp>_i) (wp_k_j - <wp>_j)

This is the dominant source of off-diagonal correlation between rp
bins on a wide-area survey -- captures cosmic variance from the
spatial structure of the data, which the diagonal Poisson estimate
misses.

For Quaia at NSIDE_jack = 4 (192 pixels, ~ 215 sq deg each),
``n_regions ~ 25-50`` covers the unmasked area robustly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def jackknife_region_labels(
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    n_regions: int = 25,
    nside_jack: int = 4,
):
    """Assign each (ra, dec) to a jackknife region label 0..n_regions-1.

    Uses ``hp.ang2pix`` at ``nside_jack`` to bin the data into healpix
    super-pixels, then groups them into ``n_regions`` chunks of ~equal
    galaxy count. Returns per-galaxy labels and per-label region masks
    (a per-region boolean array of size N_data).
    """
    import healpy as hp

    pix = hp.ang2pix(
        nside_jack,
        np.deg2rad(90.0 - dec_deg), np.deg2rad(ra_deg),
    )
    # only keep pixels actually populated
    unique_pix, counts = np.unique(pix, return_counts=True)
    # sort pixels by descending count, greedy assign to N regions
    order = np.argsort(-counts)
    region_of_pixel = -np.ones(unique_pix.max() + 1, dtype=np.int64)
    counts_per_region = np.zeros(n_regions, dtype=np.int64)
    for idx in order:
        # assign this pixel to the lightest-loaded region so far
        target = int(np.argmin(counts_per_region))
        region_of_pixel[unique_pix[idx]] = target
        counts_per_region[target] += counts[idx]
    labels = region_of_pixel[pix]
    return labels, counts_per_region


def wp_jackknife(
    pos_data: np.ndarray, z_data: np.ndarray,
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    mask: np.ndarray, nside: int, cosmo,
    rp_edges: np.ndarray, pi_max: float = 200.0, n_pi: int = 40,
    n_regions: int = 25, nside_jack: int = 4,
    rr_norm_factor: float = 1.0,
    N_r_effective: int = 1_000_000,
    chunk: int = 4000,
    w_data: Optional[np.ndarray] = None,
):
    """Jackknife mean and covariance of wp(rp) for an analytic-RR run.

    For each region k, drop the data in that region, recompute DD and
    rebuild wp(rp) using the analytic RR / DR (scaled to the reduced
    N_d). The DD pair count dominates the per-jackknife runtime; for
    100k data and 25 regions, each jackknife is ~ 25 s, total ~ 10
    minutes.

    Returns
    -------
    wp_full   : (N_rp,) wp on the *full* (un-jackknifed) sample
    wp_jk     : (n_regions, N_rp) array of jackknife wp samples
    wp_cov    : (N_rp, N_rp) jackknife covariance matrix
    """
    from .analytic_rr import dr_analytic, rr_analytic
    from .projected_xi import _count_pairs_rp_pi

    pi_edges = np.linspace(0.0, pi_max, n_pi + 1)
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    n_rp = len(rp_edges) - 1
    labels, _ = jackknife_region_labels(ra_deg, dec_deg, n_regions=n_regions,
                                          nside_jack=nside_jack)
    n_total = len(pos_data)

    # --- full-sample wp ---
    DD_full = _count_pairs_rp_pi(
        pos_data, pos_data, rp_edges, pi_edges, auto=True, chunk=chunk,
        w1=w_data, w2=w_data,
    )
    res_full = rr_analytic(rp_edges, pi_edges, mask, nside, z_data,
                             cosmo, N_r=N_r_effective)
    RR_full = rr_norm_factor * res_full.RR
    DR_full = dr_analytic(n_total, N_r_effective, RR_full)

    def _wp_from_counts(DD, DR, RR, N_d):
        Nd_pairs = N_d * (N_d - 1) / 2.0
        Nr_pairs = N_r_effective * (N_r_effective - 1) / 2.0
        DD_n = DD / Nd_pairs
        DR_n = DR / (N_d * N_r_effective)
        RR_n = RR / Nr_pairs
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.where(RR_n > 0, (DD_n - 2 * DR_n + RR_n) / RR_n, 0.0)
        d_pi = np.diff(pi_edges)
        return 2.0 * np.sum(xi * d_pi[None, :], axis=1)

    wp_full = _wp_from_counts(DD_full, DR_full, RR_full, n_total)

    # --- per-region jackknife: drop region k, recompute ---
    wp_jk = np.zeros((n_regions, n_rp))
    for k in range(n_regions):
        keep = labels != k
        n_k = int(keep.sum())
        if n_k < 100:
            wp_jk[k] = wp_full
            continue
        DD_k = _count_pairs_rp_pi(
            pos_data[keep], pos_data[keep], rp_edges, pi_edges,
            auto=True, chunk=chunk,
            w1=(w_data[keep] if w_data is not None else None),
            w2=(w_data[keep] if w_data is not None else None),
        )
        # RR is angular + radial separable; for jackknife we keep the
        # full-survey RR (the dropped region only changes data; the
        # window function treatment of the random is unchanged).
        # This is the standard approach -- jackknife only DD.
        DR_k = dr_analytic(n_k, N_r_effective, RR_full)
        wp_jk[k] = _wp_from_counts(DD_k, DR_k, RR_full, n_k)

    # standard jackknife covariance
    wp_mean = wp_jk.mean(axis=0)
    diff = wp_jk - wp_mean[None, :]
    wp_cov = ((n_regions - 1) / n_regions) * (diff.T @ diff)
    return wp_full, wp_jk, wp_cov
