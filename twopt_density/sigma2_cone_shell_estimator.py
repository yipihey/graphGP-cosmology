"""Count-in-cap estimator for ``sigma^2(theta; z, dz)`` and ``d sigma^2 / dz``.

Operates directly on ``(ra, dec, z)`` -- never converts to comoving
Cartesian, so the estimator is genuinely cosmology-free in the sense of
Tom Abel's draft (Sec. 2.1).

Pipeline:

  1. Choose cap centres on a HEALPix-pixel-centre grid covering the
     survey footprint, dropping centres whose disc of radius
     ``theta_max`` extends past the mask edge.
  2. For each ``(centre, theta_radius, z_shell)`` triple, count
     galaxies in the cap and the redshift slice via
     ``healpy.query_disc`` against a high-NSIDE galaxy lookup.
  3. Estimate ``sigma^2_obs(theta; z) = Var(N) / <N>^2 - 1/<N>``
     across centres for each ``(theta, z)``.
  4. Adjacent-shell finite differences in log(1+z) give
     ``d sigma^2 / d ln(1+z)``.

The hot loop in ``cone_shell_counts`` -- per-cap angular separation,
theta/z binning, cumulative sum -- is implemented as a
``@numba.njit(nogil=True)`` kernel. Numba is an optional dependency;
when unavailable we fall back to a pure-NumPy path. Multiple cap
centres are processed in parallel via ``concurrent.futures
.ThreadPoolExecutor`` because the kernel releases the GIL.

Functions:

  cap_centre_grid(mask, nside_centres, theta_max_rad, edge_buffer_frac)
      -> (ra_c, dec_c) HEALPix-pixel-centre array, edge-buffered.

  cone_shell_counts(ra, dec, z, theta_radii, z_edges,
                      ra_centres, dec_centres, nside_lookup=512,
                      n_threads=None)
      -> N[i_centre, i_theta, i_zshell], A_eff[i_centre, i_theta]

  sigma2_estimate_cone_shell(N) -> sigma^2_obs[i_theta, i_zshell].

  dsigma2_dz_estimate(sigma2, z_centres) -> central differences in ln(1+z).

  sigma2_cone_shell_jackknife(...) -> jackknife covariance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _per_cap_count_kernel(
        theta_c, phi_c,
        ipix_disc,
        pix_starts,
        theta_g_sorted, phi_g_sorted, z_g_sorted, w_g_sorted,
        theta_radii, z_edges,
    ):
        """Numba-JIT per-cap kernel.

        For a single cap centre ``(theta_c, phi_c)`` and the set of
        ``ipix_disc`` pixels returned by ``hp.query_disc`` at the
        largest theta, accumulate weighted counts into a (n_theta, n_z)
        matrix where the (t, k) entry is the count of galaxies with
        angular separation <= ``theta_radii[t]`` and redshift in shell
        ``k``.

        Releases the GIL (``nogil=True``) so the outer loop can run
        across many caps in a thread pool.
        """
        n_theta = theta_radii.shape[0]
        n_z = z_edges.shape[0] - 1
        out = np.zeros((n_theta, n_z))
        sin_tc = np.sin(theta_c)
        cos_tc = np.cos(theta_c)
        z_lo_total = z_edges[0]
        z_hi_total = z_edges[n_z]
        for ip_idx in range(ipix_disc.shape[0]):
            ip = ipix_disc[ip_idx]
            s = pix_starts[ip]
            e = pix_starts[ip + 1]
            for j in range(s, e):
                tg = theta_g_sorted[j]
                pg = phi_g_sorted[j]
                cs = (sin_tc * np.sin(tg) * np.cos(phi_c - pg)
                        + cos_tc * np.cos(tg))
                if cs > 1.0:
                    cs = 1.0
                elif cs < -1.0:
                    cs = -1.0
                sep = np.arccos(cs)
                # binary search for smallest t with theta_radii[t] >= sep
                lo = 0
                hi = n_theta
                while lo < hi:
                    mid = (lo + hi) // 2
                    if theta_radii[mid] < sep:
                        lo = mid + 1
                    else:
                        hi = mid
                t_bin = lo
                if t_bin >= n_theta:
                    continue
                # binary search for z bin: smallest k with z_edges[k+1] > zj
                zj = z_g_sorted[j]
                if zj < z_lo_total or zj >= z_hi_total:
                    continue
                lo = 0
                hi = n_z
                while lo < hi:
                    mid = (lo + hi) // 2
                    if z_edges[mid + 1] <= zj:
                        lo = mid + 1
                    else:
                        hi = mid
                z_bin = lo
                out[t_bin, z_bin] += w_g_sorted[j]
        # cumulative sum across the theta axis -- a galaxy at
        # angular separation sep contributes to all caps t with
        # theta_radii[t] >= sep, so we accumulate in-place.
        for k in range(n_z):
            acc = 0.0
            for t in range(n_theta):
                acc += out[t, k]
                out[t, k] = acc
        return out

    _NUMBA_OK = True
except ImportError:                                            # pragma: no cover
    _NUMBA_OK = False
    _per_cap_count_kernel = None


def cap_centre_grid(
    mask: np.ndarray,
    nside_centres: int = 64,
    theta_max_rad: float = 0.07,        # ~4 deg by default
    edge_buffer_frac: float = 1.0,
    mask_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HEALPix-pixel-centre grid of cap centres on the survey footprint.

    For each pixel of the supplied ``mask`` (any NSIDE), if the pixel
    centre is inside the footprint and a disc of radius
    ``edge_buffer_frac * theta_max_rad`` around it does not extend
    outside the mask (where 'outside' means ``mask < mask_threshold``),
    the pixel centre is kept as a cap centre.

    Parameters
    ----------
    mask : (NPIX,) array, the survey completeness map (in [0, 1]) or
        a boolean footprint mask. NSIDE inferred from size.
    nside_centres : NSIDE of the centre grid. Larger -> denser sampling
        but smaller per-centre independence (overlap between caps grows).
        Default 64.
    theta_max_rad : maximum cap radius the centres must accommodate.
    edge_buffer_frac : multiplier on ``theta_max_rad`` for edge
        rejection. ``1.0`` keeps centres whose ``theta_max`` disc fits;
        ``2.0`` is conservative for caps that may double in size.
    mask_threshold : ``mask >= mask_threshold`` defines the footprint.

    Returns
    -------
    ra_c : (n_centres,) array, RA of accepted centres [deg].
    dec_c : (n_centres,) array, Dec of accepted centres [deg].
    a_eff_frac : (n_centres,) array, fraction of the
        ``theta_max_rad``-disc that is inside the mask (always >= 1 -
        edge_buffer_frac mask coverage; useful for weighting).
    """
    import healpy as hp

    mask = np.asarray(mask, dtype=np.float64)
    npix_mask = mask.size
    nside_mask = int(round(np.sqrt(npix_mask / 12)))
    if 12 * nside_mask ** 2 != npix_mask:
        raise ValueError(f"mask size {npix_mask} not a valid HEALPix NPIX")

    # iterate over candidate centres at the requested NSIDE
    npix_c = 12 * nside_centres ** 2
    theta_c, phi_c = hp.pix2ang(nside_centres, np.arange(npix_c))
    ra = np.degrees(phi_c)
    dec = 90.0 - np.degrees(theta_c)

    # 1) on-footprint test: centre pixel of mask >= threshold
    pix_in_mask = hp.ang2pix(nside_mask, theta_c, phi_c)
    on_foot = mask[pix_in_mask] >= mask_threshold

    # 2) edge-buffer test: query_disc at radius edge_buffer_frac * theta_max
    #    must contain only pixels with mask >= threshold.
    rad = float(edge_buffer_frac) * float(theta_max_rad)

    keep = np.zeros(npix_c, dtype=bool)
    a_eff_frac = np.zeros(npix_c, dtype=np.float64)
    cand_idx = np.flatnonzero(on_foot)
    cand_vec = hp.ang2vec(theta_c[cand_idx], phi_c[cand_idx])
    for j, vec in enumerate(cand_vec):
        ipix = hp.query_disc(nside_mask, vec, rad, inclusive=False)
        if ipix.size == 0:
            continue
        m = mask[ipix]
        # fraction of the disc inside the footprint, weighted by
        # completeness
        frac = float(m.mean())
        a_eff_frac[cand_idx[j]] = frac
        if (m >= mask_threshold).all():
            keep[cand_idx[j]] = True

    ra_c = ra[keep]
    dec_c = dec[keep]
    return ra_c, dec_c, a_eff_frac[keep]


def cone_shell_counts(
    ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray,
    theta_radii_rad: np.ndarray,
    z_edges: np.ndarray,
    ra_centres_deg: np.ndarray, dec_centres_deg: np.ndarray,
    nside_lookup: int = 512,
    weights: Optional[np.ndarray] = None,
    n_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Count galaxies in spherical caps of half-angle ``theta`` and
    redshift slices ``[z_edges[k], z_edges[k+1]]`` for every cap centre.

    Pre-builds a HEALPix pixel -> galaxy-index lookup at
    ``nside_lookup`` (RING ordering). For each centre, walks
    ``query_disc`` outward to ``max(theta_radii)``; the resulting set
    of pixels is then handed to a Numba kernel that bins each galaxy
    by its exact (sep, z) and accumulates the (n_theta, n_z) count
    matrix in a single pass.

    Parameters
    ----------
    ra_deg, dec_deg, z : galaxy positions / redshifts.
    theta_radii_rad : (n_theta,) ascending cap half-angles [rad].
    z_edges : (n_zshell + 1,) shell edges, ascending.
    ra_centres_deg, dec_centres_deg : (n_centres,) cap centres.
    nside_lookup : NSIDE of the galaxy lookup. Choose so each pixel is
        much smaller than the smallest cap. For NSIDE=512 the pixel
        side is ~7 arcmin, which is finer than typical cap radii.
    weights : optional (N_gal,) per-galaxy weights (e.g.
        completeness reciprocal). Default uniform.
    n_threads : if Numba is available, run the per-cap kernel in a
        thread pool with this many workers. ``None`` (default)
        auto-selects ``os.cpu_count()``; ``1`` forces single-threaded.

    Returns
    -------
    N : (n_centres, n_theta, n_zshell) float array of weighted counts.
    A_cap : (n_theta,) cap solid angles ``2 pi (1 - cos theta)`` [sr].
    """
    import healpy as hp

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    theta_radii_rad = np.asarray(theta_radii_rad, dtype=np.float64)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    ra_c = np.asarray(ra_centres_deg, dtype=np.float64)
    dec_c = np.asarray(dec_centres_deg, dtype=np.float64)

    n_theta = theta_radii_rad.size
    n_z = z_edges.size - 1
    n_c = ra_c.size

    if weights is None:
        w = np.ones_like(z)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # 1) galaxy -> pixel lookup at nside_lookup; sort all per-galaxy
    #    arrays by pixel so the kernel can iterate contiguous slices.
    theta_g = np.deg2rad(90.0 - dec_deg)
    phi_g = np.deg2rad(ra_deg)
    ipix_g = hp.ang2pix(nside_lookup, theta_g, phi_g)
    order = np.argsort(ipix_g, kind="stable")
    ipix_g_sorted = ipix_g[order]
    theta_g_sorted = theta_g[order]
    phi_g_sorted = phi_g[order]
    z_sorted = z[order]
    w_sorted = w[order]
    npix_lookup = 12 * nside_lookup ** 2
    pix_starts = np.searchsorted(
        ipix_g_sorted, np.arange(npix_lookup + 1), side="left",
    ).astype(np.int64)

    # 2) per-centre cap walk
    theta_c_rad = np.deg2rad(90.0 - dec_c)
    phi_c_rad = np.deg2rad(ra_c)
    vecs_c = hp.ang2vec(theta_c_rad, phi_c_rad)
    theta_max = float(theta_radii_rad.max())

    N = np.zeros((n_c, n_theta, n_z), dtype=np.float64)

    if _NUMBA_OK:
        # Pre-fetch every cap's disc-pixel list in the main thread
        # (healpy.query_disc isn't reliably GIL-releasing). Then the
        # per-cap kernels run in a thread pool because they release
        # the GIL.
        ipix_per_cap = [
            hp.query_disc(nside_lookup, vecs_c[i], theta_max, inclusive=True)
                .astype(np.int64)
            for i in range(n_c)
        ]
        if n_threads is None:
            import os
            n_threads = os.cpu_count() or 1

        def _one(i):
            ipix = ipix_per_cap[i]
            if ipix.size == 0:
                return i, np.zeros((n_theta, n_z), dtype=np.float64)
            return i, _per_cap_count_kernel(
                theta_c_rad[i], phi_c_rad[i],
                ipix, pix_starts,
                theta_g_sorted, phi_g_sorted, z_sorted, w_sorted,
                theta_radii_rad, z_edges,
            )

        if n_threads > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for i, mat in pool.map(_one, range(n_c)):
                    N[i] = mat
        else:
            for i in range(n_c):
                _, mat = _one(i)
                N[i] = mat
    else:                                                       # pragma: no cover
        # Pure-NumPy fallback (slow; only triggered if numba is missing)
        for i in range(n_c):
            ipix_disc = hp.query_disc(
                nside_lookup, vecs_c[i], theta_max, inclusive=True,
            )
            if ipix_disc.size == 0:
                continue
            offs = []
            for ip in ipix_disc:
                s, e = pix_starts[ip], pix_starts[ip + 1]
                if s < e:
                    offs.append((s, e))
            if not offs:
                continue
            n_total = int(sum(e - s for s, e in offs))
            if n_total == 0:
                continue
            idx = np.empty(n_total, dtype=np.int64)
            cur = 0
            for s, e in offs:
                d = e - s
                idx[cur:cur + d] = np.arange(s, e)
                cur += d
            z_in = z_sorted[idx]
            w_in = w_sorted[idx]
            cos_sep = (
                np.sin(theta_c_rad[i]) * np.sin(theta_g_sorted[idx])
                * np.cos(phi_c_rad[i] - phi_g_sorted[idx])
                + np.cos(theta_c_rad[i]) * np.cos(theta_g_sorted[idx])
            )
            cos_sep = np.clip(cos_sep, -1.0, 1.0)
            sep = np.arccos(cos_sep)
            z_bin = np.searchsorted(z_edges, z_in, side="right") - 1
            z_in_range = (z_bin >= 0) & (z_bin < n_z)
            if not z_in_range.any():
                continue
            sep_e = sep[z_in_range]
            w_e = w_in[z_in_range]
            z_bin_e = z_bin[z_in_range]
            s_order = np.argsort(sep_e, kind="stable")
            sep_sorted = sep_e[s_order]
            w_sorted_e = w_e[s_order]
            z_bin_sorted = z_bin_e[s_order]
            cum = np.zeros((n_z, sep_sorted.size + 1), dtype=np.float64)
            for k in range(n_z):
                mk = (z_bin_sorted == k)
                cw = np.zeros(sep_sorted.size + 1)
                cw[1:] = np.cumsum(w_sorted_e * mk)
                cum[k] = cw
            idx_theta = np.searchsorted(
                sep_sorted, theta_radii_rad, side="right",
            )
            for t in range(n_theta):
                it = idx_theta[t]
                for k in range(n_z):
                    N[i, t, k] = cum[k, it]

    A_cap = 2.0 * np.pi * (1.0 - np.cos(theta_radii_rad))
    return N, A_cap


def sigma2_estimate_cone_shell(N: np.ndarray) -> np.ndarray:
    """Estimate ``sigma^2_obs(theta, z) = Var(N) / <N>^2 - 1 / <N>``
    across the centre axis (axis 0).

    The 1/<N> shot-noise subtraction is the same as paper Eq. (3); for
    Poisson-distributed N with ``<N> = mu`` (no clustering) it gives
    sigma^2_obs = 0 in expectation.

    Parameters
    ----------
    N : (n_centres, n_theta, n_zshell) array of cap counts.

    Returns
    -------
    sigma2 : (n_theta, n_zshell) numpy array.
    """
    N = np.asarray(N, dtype=np.float64)
    mu = N.mean(axis=0)
    var = N.var(axis=0, ddof=1) if N.shape[0] > 1 else np.zeros_like(mu)
    safe = np.where(mu > 0, mu, np.inf)
    return var / safe ** 2 - 1.0 / safe


def dsigma2_dz_estimate(
    sigma2_grid: np.ndarray,
    z_centres: np.ndarray,
    log1pz: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference derivative of ``sigma^2(theta; z)`` across
    the redshift axis (axis -1).

    Parameters
    ----------
    sigma2_grid : (..., n_z) array; the leading axes are theta, etc.
    z_centres : (n_z,) shell-centre redshifts.
    log1pz : if True (default), differentiate w.r.t. ``ln(1+z)``;
        else ``z``.

    Returns
    -------
    z_pivots : (n_z - 2,) numpy array of central pivots.
    dsigma2  : (..., n_z - 2) array.
    """
    s = np.asarray(sigma2_grid, dtype=np.float64)
    zc = np.asarray(z_centres, dtype=np.float64)
    if log1pz:
        x = np.log(1.0 + zc)
    else:
        x = zc
    dx = x[2:] - x[:-2]
    ds = (s[..., 2:] - s[..., :-2]) / dx
    return zc[1:-1], ds


def sigma2_cone_shell_jackknife(
    ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray,
    theta_radii_rad: np.ndarray, z_edges: np.ndarray,
    ra_centres_deg: np.ndarray, dec_centres_deg: np.ndarray,
    n_regions: int = 25, nside_jack: int = 4,
    nside_lookup: int = 512,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jackknife mean and covariance of ``sigma^2(theta, z)``.

    Splits the sky into ``n_regions`` low-NSIDE-pixel super-regions via
    ``jackknife_region_labels``. For each region k:

      - drop all galaxies with that region label
      - drop all cap centres that fall inside that region
      - re-estimate sigma^2 across the remaining centres + galaxies

    Returns the jackknife mean, the per-region samples, and the
    standard jackknife covariance::

        C = ((N - 1) / N) sum_k (s_k - <s>) (s_k - <s>)^T

    Parameters / returns
    --------------------
    same naming as ``cone_shell_counts``; covariance is shape
    ``(n_theta * n_zshell, n_theta * n_zshell)``.
    """
    from .jackknife import jackknife_region_labels

    labels_g, _ = jackknife_region_labels(ra_deg, dec_deg,
                                              n_regions=n_regions,
                                              nside_jack=nside_jack)
    labels_c, _ = jackknife_region_labels(ra_centres_deg, dec_centres_deg,
                                              n_regions=n_regions,
                                              nside_jack=nside_jack)
    n_theta = theta_radii_rad.size
    n_zshell = z_edges.size - 1

    samples = np.zeros((n_regions, n_theta, n_zshell), dtype=np.float64)
    for k in range(n_regions):
        keep_g = labels_g != k
        keep_c = labels_c != k
        if keep_c.sum() == 0:
            samples[k] = np.nan
            continue
        N_k, _ = cone_shell_counts(
            ra_deg[keep_g], dec_deg[keep_g], z[keep_g],
            theta_radii_rad, z_edges,
            ra_centres_deg[keep_c], dec_centres_deg[keep_c],
            nside_lookup=nside_lookup,
            weights=None if weights is None else weights[keep_g],
        )
        samples[k] = sigma2_estimate_cone_shell(N_k)

    mean = np.nanmean(samples, axis=0)
    diff = samples - mean[None]
    n_used = np.isfinite(samples[:, 0, 0]).sum()
    flat = diff.reshape(samples.shape[0], -1)
    flat = np.nan_to_num(flat, nan=0.0)
    cov = ((n_used - 1) / n_used) * (flat.T @ flat)
    return mean, samples, cov
