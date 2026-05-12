"""PIP-weighted angular kNN-CDF / pair-counting estimator for DESI.

Pairwise inverse-probability (PIP) weighting corrects DESI fibre-assignment
incompleteness in the data-data pair count. Each object carries an
``(n_words,)`` int64 bitmask whose set bits indicate the assignment-
realisations in which it was observed. For a pair ``(i, j)``:

  popcount_ij = popcount(b_i & b_j)
  w_PIP(i, j) = N_realizations / popcount_ij    if popcount_ij > 0
                                                (else the pair is dropped)

with ``N_realizations = 64 * n_words`` (128 for DESI EDR/Y1).

This module provides a standalone DD-flavour kNN-CDF kernel that
applies PIP weights to pair counts. DR and RR remain unweighted (the
random catalogue is independent of fibre assignment), so the LS
combination is

  xi_PIP_LS(theta) = (DD_PIP - 2 DR + RR) / RR.

Used as a complement to ``twopt_density.knn_cdf.joint_knn_cdf`` —
which handles the unweighted (or scalar-weighted) DD/RD/RR cubes — by
producing a single PIP-weighted DD cube. The scalar-weighting path in
``joint_knn_cdf`` (``weights_neigh``) is preserved separately because
PIP is a *pair* weight, not an object weight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .knn_cdf import KnnCdfResult


try:
    import numba

    @numba.njit(cache=True, inline="always")
    def _popcount64(x):
        """Software popcount of a (signed) int64; treat the value as
        an unsigned 64-bit pattern. Numba's ``np.bit_count`` exists in
        recent versions but isn't universally available — this manual
        SWAR popcount is portable and fast enough for our pair-count
        loop (one popcount per pair candidate).
        """
        # Reinterpret as unsigned by masking to 64 bits.
        x = x & 0xFFFFFFFFFFFFFFFF
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        return (x * 0x0101010101010101) >> 56

    @numba.njit(cache=True, nogil=True)
    def _per_cap_pip_count_kernel(
        theta_c, phi_c,
        bw_q,                                              # (n_words,) int64
        ipix_disc,
        pix_starts,
        theta_g_sorted, phi_g_sorted, z_g_sorted,
        bw_g_sorted,                                       # (N_n, n_words)
        is_self_g_sorted,                                  # (N_n,) int8 (1 = same TARGETID)
        theta_radii, z_edges,
        n_realizations,
    ):
        """Per-cap PIP-weighted neighbor count.

        For each candidate neighbor ``g`` whose pixel is in
        ``ipix_disc`` and whose angular separation from the cap centre
        is ≤ ``theta_radii[t]``, accumulate

          w = N_realizations / popcount(bw_q & bw_g)         if popcount > 0
              0                                              otherwise.

        ``is_self_g_sorted`` is an int8 flag (per-row) to drop the
        self-pair (q == g by TARGETID), which the caller pre-computes
        outside the kernel.
        """
        n_theta = theta_radii.shape[0]
        n_z = z_edges.shape[0] - 1
        n_words = bw_q.shape[0]
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
                if is_self_g_sorted[j] != 0:
                    continue
                tg = theta_g_sorted[j]
                pg = phi_g_sorted[j]
                cs = (sin_tc * np.sin(tg) * np.cos(phi_c - pg)
                        + cos_tc * np.cos(tg))
                if cs > 1.0:
                    cs = 1.0
                elif cs < -1.0:
                    cs = -1.0
                sep = np.arccos(cs)
                # binary search smallest t with theta_radii[t] >= sep
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
                # PIP weight: popcount over all words
                pop = 0
                for w in range(n_words):
                    pop += _popcount64(bw_q[w] & bw_g_sorted[j, w])
                if pop > 0:
                    out[t_bin, z_bin] += n_realizations / pop
                # else: pair is dropped (no realisation observed both)
        # cumulative-sum along theta axis
        for k in range(n_z):
            acc = 0.0
            for t in range(n_theta):
                acc += out[t, k]
                out[t, k] = acc
        return out

    _NUMBA_OK = True
except ImportError:                                            # pragma: no cover
    _NUMBA_OK = False
    _popcount64 = None
    _per_cap_pip_count_kernel = None


def joint_knn_cdf_pip_dd(
    ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray,
    bitweights: np.ndarray,                                # (N, n_words) int64
    targetid: np.ndarray,                                   # (N,) int64 — for self-pair removal
    theta_radii_rad: np.ndarray,
    z_q_edges: np.ndarray, z_n_edges: np.ndarray,
    nside_lookup: int = 512,
    n_threads: Optional[int] = None,
    query_chunk_size: int = 5000,
    progress: bool = False,
) -> KnnCdfResult:
    """PIP-weighted DD kNN-CDF cube.

    The query and neighbor catalogs are the same data catalog (this is
    the DD pass). For each query q at separation ≤ θ from neighbor n,
    accumulate the PIP weight ``N_realizations / popcount(bw_q & bw_n)``
    in the (theta, z_q, z_n) cube. Self-pairs (q == n by TARGETID) are
    excluded.

    Returns a ``KnnCdfResult`` with the same axes as
    ``joint_knn_cdf``; ``sum_n`` carries the PIP-weighted neighbor
    *sum* per query (not an integer count). ``H_geq_k`` is left zero
    because the PIP-weighted "at least k neighbors" semantics doesn't
    have a clean integer interpretation; for the LS panel we only need
    ``sum_n`` which feeds ``mean_count`` and the LS combination.

    Notes
    -----
    The PIP weight ``N_realizations / popcount(b_q & b_n)`` is the
    *inverse* probability that the pair is jointly observed in any
    given realisation. Sum over realisations = N_realizations, so the
    expected weighted pair count equals the true pair count if
    ``b_q & b_n`` is unbiased — which is the IIP construction
    (Bianchi & Percival 2017).
    """
    if not _NUMBA_OK:
        raise RuntimeError("joint_knn_cdf_pip_dd requires numba.")
    import healpy as hp

    ra_deg = np.ascontiguousarray(ra_deg, dtype=np.float64)
    dec_deg = np.ascontiguousarray(dec_deg, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    bitweights = np.ascontiguousarray(bitweights, dtype=np.int64)
    targetid = np.ascontiguousarray(targetid, dtype=np.int64)
    theta_radii_rad = np.ascontiguousarray(theta_radii_rad, dtype=np.float64)
    z_q_edges = np.ascontiguousarray(z_q_edges, dtype=np.float64)
    z_n_edges = np.ascontiguousarray(z_n_edges, dtype=np.float64)

    n_words = int(bitweights.shape[1])
    n_realizations = 64 * n_words
    n_q = ra_deg.size
    n_theta = theta_radii_rad.size
    n_z_q = z_q_edges.size - 1
    n_z_n = z_n_edges.size - 1

    # Build pixel-lookup grid on the neighbor catalog (== data).
    theta_g = np.deg2rad(90.0 - dec_deg)
    phi_g = np.deg2rad(ra_deg)
    ipix_g = hp.ang2pix(nside_lookup, theta_g, phi_g)
    order = np.argsort(ipix_g, kind="stable")
    ipix_g_sorted = ipix_g[order]
    theta_g_sorted = theta_g[order]
    phi_g_sorted = phi_g[order]
    z_g_sorted = z[order]
    bw_g_sorted = bitweights[order]
    targetid_g_sorted = targetid[order]
    npix_lookup = 12 * nside_lookup ** 2
    pix_starts = np.searchsorted(
        ipix_g_sorted, np.arange(npix_lookup + 1), side="left",
    ).astype(np.int64)

    # Bin queries by z_q.
    i_z_q_per_q = np.searchsorted(z_q_edges, z, side="right") - 1
    in_range_q = (i_z_q_per_q >= 0) & (i_z_q_per_q < n_z_q)
    i_z_q_per_q = i_z_q_per_q.astype(np.int64)

    theta_q = np.deg2rad(90.0 - dec_deg)
    phi_q = np.deg2rad(ra_deg)
    theta_max = float(theta_radii_rad.max())
    vecs_q = hp.ang2vec(theta_q, phi_q)

    # Output cubes.
    sum_n = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
    sum_n2 = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
    N_q = np.zeros(n_z_q, dtype=np.int64)

    valid_q = np.flatnonzero(in_range_q)
    np.add.at(N_q, i_z_q_per_q[valid_q], 1)

    # Self-pair flag per neighbor row, in sorted order. For each query we
    # rebuild a boolean strip of which sorted-rows correspond to that
    # query's TARGETID. Cheaper alternative: build a TARGETID -> sorted-
    # row map. Since TARGETID is unique, the map has size N_q. Use it
    # to set a single int8 flag inside the kernel.
    tid_to_sorted_row = {}
    for row, tid in enumerate(targetid_g_sorted):
        tid_to_sorted_row[int(tid)] = row

    is_self_strip = np.zeros(targetid_g_sorted.size, dtype=np.int8)

    if n_threads is None:
        import os
        n_threads = os.cpu_count() or 1

    n_chunks = (valid_q.size + query_chunk_size - 1) // query_chunk_size
    import time as _t
    for ic in range(n_chunks):
        chunk_start = ic * query_chunk_size
        chunk_end = min(chunk_start + query_chunk_size, valid_q.size)
        chunk = valid_q[chunk_start:chunk_end]
        t0 = _t.time()

        # Pre-fetch each query's disc.
        discs = [hp.query_disc(nside_lookup, vecs_q[q], theta_max,
                                inclusive=True).astype(np.int64) for q in chunk]

        for k_idx, q in enumerate(chunk):
            ipix = discs[k_idx]
            tid_q = int(targetid[q])
            row_self = tid_to_sorted_row.get(tid_q, -1)
            if 0 <= row_self < is_self_strip.size:
                is_self_strip[row_self] = 1
            n_cap = _per_cap_pip_count_kernel(
                theta_q[q], phi_q[q],
                bitweights[q],
                ipix, pix_starts,
                theta_g_sorted, phi_g_sorted, z_g_sorted,
                bw_g_sorted, is_self_strip,
                theta_radii_rad, z_n_edges,
                n_realizations,
            )
            if 0 <= row_self < is_self_strip.size:
                is_self_strip[row_self] = 0
            iq = i_z_q_per_q[q]
            for t in range(n_theta):
                for jn in range(n_z_n):
                    v = n_cap[t, jn]
                    sum_n[t, iq, jn] += v
                    sum_n2[t, iq, jn] += v * v

        if progress:
            t = _t.time() - t0
            print(f"  [chunk {ic+1}/{n_chunks}] queries "
                  f"{chunk_start}..{chunk_end}, t={t:.1f}s")

    return KnnCdfResult(
        H_geq_k=np.zeros((n_theta, n_z_q, n_z_n, 1), dtype=np.int64),
        sum_n=sum_n, sum_n2=sum_n2, N_q=N_q,
        theta_radii_rad=theta_radii_rad,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        flavor="DD_PIP", backend_used="numba_pip",
        area_per_cap=2.0 * np.pi * (1.0 - np.cos(theta_radii_rad)),
    )
