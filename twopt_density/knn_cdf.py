"""Joint angular kNN-CDF primitive on the lightcone.

Implements the data primitive described in ``lightcone_native_v3.pdf``
(T. Abel, May 2026):

    P_{>=k}^{(z_n)}(theta; z_q) = Prob[ N^{(z_n)}(theta; Omega_q, z_q) >= k ]

— the probability that a query in redshift shell ``z_q`` has at least
``k`` data neighbors with redshift in shell ``z_n`` within an angular cap
of radius ``theta``. Pure observables ``(theta, log(1+z_q),
log(1+z_n), k)``; no fiducial cosmology, no comoving distance, no
implicit redshift averaging in the data vector.

A single hierarchical pass yields, as derived quantities, all standard
clustering observables. See ``twopt_density.knn_derived`` for the
reductions.

Architecture
------------
The per-cap angular-binning + z-shell-payload kernel
``_per_cap_count_kernel`` already exists in
``sigma2_cone_shell_estimator`` (extensively tested, Numba/nogil,
threadpool-parallelizable). This module is a thin sibling orchestrator
that (a) treats each query point as a cap centre, (b) bins queries by
``z_q`` shell, and (c) aggregates the per-cap ``(n_theta, n_z_n)``
output into the joint kNN-CDF cube + moment accumulators. The existing
``cone_shell_counts`` orchestrator is unchanged.

Backends
--------
``backend="numba"`` runs the Numba kernel + Python+ThreadPoolExecutor
orchestration. ``backend="rust"`` is a future hook for the
``morton_cascade.angular_knn_cdf`` PyO3 binding (Phase 2). ``backend
="auto"`` picks Rust when available, Numba otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# Re-use the per-cap kernel from the sigma^2 estimator. This is the only
# Numba kernel in the cone-shell pipeline; both orchestrators share it.
from .sigma2_cone_shell_estimator import (
    _per_cap_count_kernel,
    _per_cap_count_kernel_per_region,
    _NUMBA_OK,
)


try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _aggregate_into_cube(
        n_cap, i_z_q, H_geq_k, sum_n, sum_n2, k_max,
    ):
        """In-place aggregate one cap's (n_theta, n_z_n) count matrix
        into the (n_theta, n_z_q, n_z_n, k_max) kNN-CDF cube and the
        first/second-moment accumulators.

        Called once per query. ``n_cap`` is the cumulative-on-theta
        count matrix produced by ``_per_cap_count_kernel``: entry
        ``[t, jn]`` is the count of shell-``jn`` neighbors at angular
        separation ``<= theta_radii[t]``.

        Releases the GIL (``nogil=True``) so the outer query loop can
        run in a thread pool.

        NOTE: This is the LEGACY aggregation that increments
        ``H_geq_k[..., 0..n_int-1]`` directly — O(n_int) per cell.
        Production now uses ``_aggregate_query_global`` (and
        ``_aggregate_query_jackknife``) which increment a single
        slot ``H_geq_k[..., n_int - 1]`` per cell and rely on a
        suffix sum at the end of the run. Kept for unit tests that
        compare the two.
        """
        n_theta, n_z_n = n_cap.shape
        for t in range(n_theta):
            for jn in range(n_z_n):
                v = n_cap[t, jn]
                sum_n[t, i_z_q, jn] += v
                sum_n2[t, i_z_q, jn] += v * v
                if k_max > 0:
                    n_int = int(v)
                    if n_int > k_max:
                        n_int = k_max
                    for k in range(n_int):
                        H_geq_k[t, i_z_q, jn, k] += 1

    @numba.njit(cache=True, nogil=True)
    def _aggregate_query_global(
        n_cap, iq, jn_self, has_self,
        sum_n, sum_n2, sum_n3, sum_n4, H_geq_k, k_max,
    ):
        """Optimised per-query aggregation into global cubes.

        Compared to ``_aggregate_into_cube`` this kernel:

        - Performs DD self-pair exclusion inline (subtract 1 from
          ``n_cap[t, jn_self]`` when ``has_self != 0``), eliminating
          the ``n_cap.copy()`` previously done in Python.
        - Increments ``H_geq_k[..., n_int - 1]`` ONCE per cell
          (O(1)) instead of looping ``for k in range(n_int)``
          (O(n_int)). The cube is then converted from "exactly k+1"
          to "at least k+1" semantics by a single suffix-sum on the
          k-axis at end of run (``_finalize_h_suffix_sum``).

        For ``k_max=302`` and DESI's 860K data queries with average
        n_int ~ 50, this drops aggregation cost by ~50× per query.

        Accumulates raw moments p=1..4 (sum_n .. sum_n4). Higher
        moments enable skewness S₃ and kurtosis S₄ via Eq. (13–14)
        of the v4_1 note.
        """
        n_theta, n_z_n = n_cap.shape
        for t in range(n_theta):
            for jn in range(n_z_n):
                v = n_cap[t, jn]
                if has_self != 0 and jn == jn_self:
                    v -= 1.0
                    if v < 0.0:
                        v = 0.0
                v2 = v * v
                sum_n[t, iq, jn] += v
                sum_n2[t, iq, jn] += v2
                sum_n3[t, iq, jn] += v2 * v
                sum_n4[t, iq, jn] += v2 * v2
                if k_max > 0:
                    n_int = int(v)
                    if n_int > k_max:
                        n_int = k_max
                    if n_int >= 1:
                        H_geq_k[t, iq, jn, n_int - 1] += 1

    @numba.njit(cache=True, nogil=True)
    def _aggregate_query_global_diag(
        n_cap, iq, jn_self, has_self,
        sum_n, sum_n2, sum_n3, sum_n4, H_geq_k, k_max,
    ):
        """Diagonal-only variant of ``_aggregate_query_global``: only
        the z_q == z_n cell (jn == iq) is written into the cube. The
        diagonal cube has shape ``(n_theta, n_z, k_max)``  — the
        single z axis stands in for both z_q and z_n. Off-diagonal
        cap counts are silently dropped (they are never accumulated
        in the per-query loop because the inner pair test already
        skips wrong-shell neighbours when ``diagonal_only`` is on).
        """
        n_theta, n_z_n = n_cap.shape
        # In diagonal-only mode the only column populated by the
        # per-query inner loop is jn == iq; iterate just that one.
        if iq < 0 or iq >= n_z_n:
            return
        jn = iq
        for t in range(n_theta):
            v = n_cap[t, jn]
            if has_self != 0 and jn == jn_self:
                v -= 1.0
                if v < 0.0:
                    v = 0.0
            v2 = v * v
            sum_n[t, iq] += v
            sum_n2[t, iq] += v2
            sum_n3[t, iq] += v2 * v
            sum_n4[t, iq] += v2 * v2
            if k_max > 0:
                n_int = int(v)
                if n_int > k_max:
                    n_int = k_max
                if n_int >= 1:
                    H_geq_k[t, iq, n_int - 1] += 1

    @numba.njit(cache=True, nogil=True)
    def _aggregate_query_jackknife_diag(
        n_cap, iq, jn_self, has_self, ir,
        sum_n, sum_n2, sum_n3, sum_n4, H_geq_k,
        sum_n_pr, sum_n2_pr, sum_n3_pr, sum_n4_pr, H_pr, k_max,
    ):
        """Diagonal-only jackknife variant. Cube shapes:
            sum_n,2,3,4:    (n_theta, n_z)
            H_geq_k:        (n_theta, n_z, k_max)
            sum_n*_pr:      (n_theta, n_z, n_regions)
            H_pr:           (n_theta, n_z, k_max, n_regions)
        """
        n_theta, n_z_n = n_cap.shape
        if iq < 0 or iq >= n_z_n:
            return
        jn = iq
        for t in range(n_theta):
            v = n_cap[t, jn]
            if has_self != 0 and jn == jn_self:
                v -= 1.0
                if v < 0.0:
                    v = 0.0
            v2 = v * v
            v3 = v2 * v
            v4 = v2 * v2
            sum_n[t, iq] += v
            sum_n2[t, iq] += v2
            sum_n3[t, iq] += v3
            sum_n4[t, iq] += v4
            sum_n_pr[t, iq, ir] += v
            sum_n2_pr[t, iq, ir] += v2
            sum_n3_pr[t, iq, ir] += v3
            sum_n4_pr[t, iq, ir] += v4
            if k_max > 0:
                n_int = int(v)
                if n_int > k_max:
                    n_int = k_max
                if n_int >= 1:
                    H_geq_k[t, iq, n_int - 1] += 1
                    H_pr[t, iq, n_int - 1, ir] += 1

    @numba.njit(cache=True, nogil=True)
    def _aggregate_query_jackknife(
        n_cap, iq, jn_self, has_self, ir,
        sum_n, sum_n2, sum_n3, sum_n4, H_geq_k,
        sum_n_pr, sum_n2_pr, sum_n3_pr, sum_n4_pr, H_pr, k_max,
    ):
        """Optimised per-query aggregation: global cubes + per-region
        jackknife cubes in one fused pass.

        Eliminates the previous Python ``for k in range(k_max)`` loop
        with ``np.add(H_per_region[:, iq, :, k, ir], (n_int > k)...)``,
        which allocated ``(n_theta, n_z_n)`` int arrays
        ``k_max`` times per query. With ``k_max=302`` and ~860K
        queries this loop accounted for the majority of DD wall time.
        New: O(1) per (t, jn) cell, no allocations.
        """
        n_theta, n_z_n = n_cap.shape
        for t in range(n_theta):
            for jn in range(n_z_n):
                v = n_cap[t, jn]
                if has_self != 0 and jn == jn_self:
                    v -= 1.0
                    if v < 0.0:
                        v = 0.0
                v2 = v * v
                v3 = v2 * v
                v4 = v2 * v2
                sum_n[t, iq, jn] += v
                sum_n2[t, iq, jn] += v2
                sum_n3[t, iq, jn] += v3
                sum_n4[t, iq, jn] += v4
                sum_n_pr[t, iq, jn, ir] += v
                sum_n2_pr[t, iq, jn, ir] += v2
                sum_n3_pr[t, iq, jn, ir] += v3
                sum_n4_pr[t, iq, jn, ir] += v4
                if k_max > 0:
                    n_int = int(v)
                    if n_int > k_max:
                        n_int = k_max
                    if n_int >= 1:
                        H_geq_k[t, iq, jn, n_int - 1] += 1
                        H_pr[t, iq, jn, n_int - 1, ir] += 1

    @numba.njit(cache=True, nogil=True)
    def _aggregate_into_cube_per_region(
        n_cap, i_z_q, i_region,
        H_geq_k_region, sum_n_region, sum_n2_region, k_max,
    ):
        """Per-region variant: one extra axis at the end of every cube.

        Used by the single-pass jackknife path. Note that for the
        per-region jackknife, the kNN-ladder aggregation is on the
        FULL count (sum over regions) rather than per region — this
        function is only useful for the moment cubes. The H_geq_k
        per-region cube is reduced after-the-fact in the orchestrator
        (see ``joint_knn_cdf`` for the jackknife pattern).
        """
        n_theta, n_z_n, n_regions = n_cap.shape
        for t in range(n_theta):
            for jn in range(n_z_n):
                for r in range(n_regions):
                    v = n_cap[t, jn, r]
                    sum_n_region[t, i_z_q, jn, r] += v
                    sum_n2_region[t, i_z_q, jn, r] += v * v
                # H_geq_k aggregation uses the FULL (region-summed)
                # cap count, since "at least k neighbors in cap" is
                # defined on the full sample (jackknife folds drop
                # one region at the consumer level).
                if k_max > 0:
                    n_full = 0.0
                    for r in range(n_regions):
                        n_full += n_cap[t, jn, r]
                    n_int = int(n_full)
                    if n_int > k_max:
                        n_int = k_max
                    for k in range(n_int):
                        H_geq_k_region[t, i_z_q, jn, k, i_region] += 1

except ImportError:                                            # pragma: no cover
    _aggregate_into_cube = None
    _aggregate_into_cube_per_region = None
    _aggregate_query_global_diag = None
    _aggregate_query_jackknife_diag = None


@dataclass
class KnnCdfResult:
    """Result of ``joint_knn_cdf``.

    Cubes (all ``float64`` for moment quantities, ``int64`` for ``H_geq_k``):

    - ``H_geq_k``: ``(n_theta, n_z_q, n_z_n, k_max)``. Entry
      ``[t, iq, jn, k-1]`` = number of queries in z_q shell ``iq`` whose
      angular cap of radius ``theta_radii[t]`` contains at least ``k``
      neighbors in z_n shell ``jn``. Paper Eq. (7).
    - ``sum_n``, ``sum_n2``: ``(n_theta, n_z_q, n_z_n)`` accumulators
      over queries: ``Sum_q n_q^{(z_n)}(theta)`` and ``Sum_q
      [n_q^{(z_n)}(theta)]^2``. ``mean_count = sum_n / N_q`` is paper
      Eq. (3); ``var = sum_n2 / N_q - (sum_n / N_q)^2`` enters
      ``sigma^2_clust`` (paper Eq. 13).
    - ``N_q``: ``(n_z_q,)`` query-count per shell — denominator for the
      CDF normalization ``P_{>=k} = H_geq_k / N_q``.

    Per-region cubes (single-pass jackknife; populated when
    ``region_labels_query`` is provided):

    - ``sum_n_per_region``, ``sum_n2_per_region``:
      ``(n_theta, n_z_q, n_z_n, n_regions)``.
    - ``H_geq_k_per_region``:
      ``(n_theta, n_z_q, n_z_n, k_max, n_regions)``. Region axis is
      indexed by the *query's* region label.
    - ``N_q_per_region``: ``(n_z_q, n_regions)``.

    Metadata:

    - ``theta_radii_rad``, ``z_q_edges``, ``z_n_edges``, ``flavor``,
      ``backend_used``, ``area_per_cap``.
    """

    H_geq_k: np.ndarray
    sum_n: np.ndarray
    sum_n2: np.ndarray
    N_q: np.ndarray
    theta_radii_rad: np.ndarray
    z_q_edges: np.ndarray
    z_n_edges: np.ndarray
    flavor: str
    backend_used: str
    area_per_cap: np.ndarray
    H_geq_k_per_region: Optional[np.ndarray] = None
    sum_n_per_region: Optional[np.ndarray] = None
    sum_n2_per_region: Optional[np.ndarray] = None
    N_q_per_region: Optional[np.ndarray] = None
    is_diagonal: bool = False
    # Higher-order raw moments (note v4_1 §6): ⟨N³⟩, ⟨N⁴⟩ enable
    # skewness S₃ and kurtosis S₄ at both raw and LS-corrected
    # levels. Optional for back-compat with older artifacts; populated
    # by the production aggregation kernels and the cascade backend
    # when the meta.json flag has_higher_moments is True.
    sum_n3: Optional[np.ndarray] = None
    sum_n4: Optional[np.ndarray] = None
    sum_n3_per_region: Optional[np.ndarray] = None
    sum_n4_per_region: Optional[np.ndarray] = None
    """When True, cubes have shape ``(n_theta, n_z, k_max)`` and
    only the z_q == z_n diagonal is populated. The single z axis
    stands in for both z_q and z_n. Storage scales as ``n_z`` not
    ``n_z²``, enabling much finer z-resolution at the same cube
    size. ``z_q_edges == z_n_edges`` in this mode."""

    @property
    def k_max(self) -> int:
        return self.H_geq_k.shape[-1]

    @property
    def n_theta(self) -> int:
        return self.theta_radii_rad.size

    @property
    def n_z_q(self) -> int:
        return self.z_q_edges.size - 1

    @property
    def n_z_n(self) -> int:
        return self.z_n_edges.size - 1


def joint_knn_cdf(
    query_ra_deg: np.ndarray, query_dec_deg: np.ndarray, query_z: np.ndarray,
    neigh_ra_deg: np.ndarray, neigh_dec_deg: np.ndarray, neigh_z: np.ndarray,
    theta_radii_rad: np.ndarray,
    z_q_edges: np.ndarray, z_n_edges: np.ndarray,
    k_max: int = 10,
    weights_neigh: Optional[np.ndarray] = None,
    region_labels_query: Optional[np.ndarray] = None,
    n_regions: int = 0,
    flavor: str = "DD",
    backend: str = "auto",
    nside_lookup: int = 512,
    n_threads: Optional[int] = None,
    query_chunk_size: int = 5000,
    progress: bool = False,
    diagonal_only: bool = False,
) -> KnnCdfResult:
    """Compute the joint angular kNN-CDF and moment cubes in a single
    hierarchical pass.

    Parameters
    ----------
    query_ra_deg, query_dec_deg, query_z
        ``(N_q,)`` query catalog. For DD-flavor, this is the data;
        for RD-flavor, the random-query catalog.
    neigh_ra_deg, neigh_dec_deg, neigh_z
        ``(N_n,)`` neighbor catalog. For DD- and RD-flavor, this is the
        data; for RR-flavor (purely-random pair counts), the random
        catalog.
    theta_radii_rad
        ``(n_theta,)`` ascending angular cap half-angles [rad]. These
        define the cumulative-θ axis of the cube.
    z_q_edges, z_n_edges
        Shell edges for the query and neighbor redshift dimensions.
        Strictly ascending; ``log(1+z)`` is recommended (paper §2.3).
    k_max
        Top of the kNN ladder. Set to 0 to skip the ``H_geq_k``
        aggregation entirely (only ``sum_n`` and ``sum_n2`` are
        populated). Default 10.
    weights_neigh
        Optional ``(N_n,)`` per-neighbor weights (e.g. completeness).
        Default uniform.
    region_labels_query
        Optional ``(N_q,)`` integer region labels for single-pass
        jackknife. When given, the result populates the per-region
        cubes. ``n_regions`` must also be > 0.
    n_regions
        Number of jackknife regions. Required if
        ``region_labels_query`` is given.
    flavor
        ``"DD"``, ``"RD"``, or ``"RR"``. ``"DD"`` (and only ``"DD"``)
        applies self-exclusion when the query and neighbor catalogs
        are object-identical or have identical RA arrays — the
        ``i==j`` pair is removed from each query's count.
    backend
        ``"numba"`` (default if Rust unavailable), ``"rust"`` (Phase 2;
        not yet implemented), or ``"auto"``.
    nside_lookup
        HEALPix NSIDE for the neighbor pixel-lookup grid. Choose so
        each pixel is much smaller than the smallest cap.
    n_threads
        Threads for the per-query parallel loop. ``None`` uses
        ``os.cpu_count()``.
    query_chunk_size
        Process queries in batches of this size, so the per-query disc
        pre-fetch (``healpy.query_disc``) does not blow up memory at
        catalog scale. At ``theta_max = 8 deg`` and
        ``nside_lookup = 512`` each disc is ~15k int64 (~120 KB), so
        the default 5000 keeps the per-batch peak under ~1 GB. Output
        cubes are accumulated across batches and are tiny by comparison.
    progress
        If True, print one line per chunk: ``[chunk i/N] queries
        x..y, t=...s``. Useful for the multi-hour full-Quaia runs.
    diagonal_only
        If True, only the z_q == z_n diagonal of the cube is
        populated. Cubes are allocated with shape
        ``(n_theta, n_z, k_max)`` instead of
        ``(n_theta, n_z_q, n_z_n, k_max)``, reducing storage from
        O(n_z²) to O(n_z) and enabling much finer z-resolution at
        the same cube budget. Requires ``z_q_edges == z_n_edges``.
        Cross-shell pair counts are silently dropped (the inner
        per-pair loop short-circuits when neigh-z-bin ≠ query-z-bin).

    Returns
    -------
    KnnCdfResult
    """
    if backend in ("cascade", "rust"):
        # Dispatch to the morton_cascade Rust backend. The CLI binding
        # produces bit-identical H_geq_k integers and machine-precision
        # sum_n / sum_n2 (verified by tests/test_knn_cdf_cascade_equivalence.py).
        from .morton_knn_cdf import joint_knn_cdf_cascade
        return joint_knn_cdf_cascade(
            query_ra_deg, query_dec_deg, query_z,
            neigh_ra_deg, neigh_dec_deg, neigh_z,
            theta_radii_rad=theta_radii_rad,
            z_q_edges=z_q_edges, z_n_edges=z_n_edges,
            k_max=k_max,
            weights_neigh=weights_neigh,
            region_labels_query=region_labels_query,
            n_regions=n_regions,
            flavor=flavor,
            diagonal_only=diagonal_only,
        )
    if backend not in ("numba", "auto"):
        raise ValueError(f"unknown backend {backend!r}")

    if not _NUMBA_OK:                                          # pragma: no cover
        raise RuntimeError(
            "joint_knn_cdf requires numba; install numba or use the "
            "fallback path in cone_shell_counts."
        )

    flavor = flavor.upper()
    if flavor not in ("DD", "DR", "RD", "RR"):
        raise ValueError(
            f"unknown flavor {flavor!r}; expected DD, DR, RD or RR")

    import healpy as hp

    query_ra_deg = np.ascontiguousarray(query_ra_deg, dtype=np.float64)
    query_dec_deg = np.ascontiguousarray(query_dec_deg, dtype=np.float64)
    query_z = np.ascontiguousarray(query_z, dtype=np.float64)
    neigh_ra_deg = np.ascontiguousarray(neigh_ra_deg, dtype=np.float64)
    neigh_dec_deg = np.ascontiguousarray(neigh_dec_deg, dtype=np.float64)
    neigh_z = np.ascontiguousarray(neigh_z, dtype=np.float64)
    theta_radii_rad = np.ascontiguousarray(theta_radii_rad, dtype=np.float64)
    z_q_edges = np.ascontiguousarray(z_q_edges, dtype=np.float64)
    z_n_edges = np.ascontiguousarray(z_n_edges, dtype=np.float64)
    k_max = int(k_max)

    if weights_neigh is None:
        w_n = np.ones_like(neigh_z)
    else:
        w_n = np.ascontiguousarray(weights_neigh, dtype=np.float64)

    n_q = query_ra_deg.size
    n_theta = theta_radii_rad.size
    n_z_q = z_q_edges.size - 1
    n_z_n = z_n_edges.size - 1
    if k_max < 0:
        raise ValueError("k_max must be >= 0")

    do_jack = region_labels_query is not None and n_regions > 0
    if do_jack:
        region_labels_query = np.ascontiguousarray(
            region_labels_query, dtype=np.int64,
        )
        if region_labels_query.size != n_q:
            raise ValueError("region_labels_query size must match N_q")

    # Detect "same catalog" for DD self-exclusion. Object identity is the
    # cheap and unambiguous check; we don't try to detect equal-content
    # arrays because that would be expensive and ambiguous (a random
    # catalog drawn the same way is not the same catalog).
    same_catalog = (
        flavor == "DD"
        and (query_ra_deg is neigh_ra_deg)
        and (query_dec_deg is neigh_dec_deg)
        and (query_z is neigh_z)
    )

    # 1) build the neighbor pixel-lookup grid (mirrors cone_shell_counts).
    theta_g = np.deg2rad(90.0 - neigh_dec_deg)
    phi_g = np.deg2rad(neigh_ra_deg)
    ipix_g = hp.ang2pix(nside_lookup, theta_g, phi_g)
    order = np.argsort(ipix_g, kind="stable")
    ipix_g_sorted = ipix_g[order]
    theta_g_sorted = theta_g[order]
    phi_g_sorted = phi_g[order]
    z_g_sorted = neigh_z[order]
    w_g_sorted = w_n[order]
    npix_lookup = 12 * nside_lookup ** 2
    pix_starts = np.searchsorted(
        ipix_g_sorted, np.arange(npix_lookup + 1), side="left",
    ).astype(np.int64)

    # 2) bin queries by z_q shell. Out-of-range queries are dropped.
    i_z_q_per_q = np.searchsorted(z_q_edges, query_z, side="right") - 1
    in_range_q = (i_z_q_per_q >= 0) & (i_z_q_per_q < n_z_q)
    if not in_range_q.any():
        # Empty result.
        H_shape = (n_theta, n_z_q, n_z_n, max(k_max, 1))
        zero_cube = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
        return KnnCdfResult(
            H_geq_k=np.zeros(H_shape, dtype=np.int64),
            sum_n=zero_cube.copy(),
            sum_n2=zero_cube.copy(),
            N_q=np.zeros(n_z_q, dtype=np.int64),
            theta_radii_rad=theta_radii_rad,
            z_q_edges=z_q_edges,
            z_n_edges=z_n_edges,
            flavor=flavor,
            backend_used="numba",
            area_per_cap=2.0 * np.pi * (1.0 - np.cos(theta_radii_rad)),
            sum_n3=zero_cube.copy(),
            sum_n4=zero_cube.copy(),
        )
    i_z_q_per_q = i_z_q_per_q.astype(np.int64)

    # Precompute query angular coordinates and pixel-disc lists.
    theta_q = np.deg2rad(90.0 - query_dec_deg)
    phi_q = np.deg2rad(query_ra_deg)
    theta_max = float(theta_radii_rad.max())
    vecs_q = hp.ang2vec(theta_q, phi_q)
    # For DD-flavor with same catalog, also need each query's z_n bin
    # to subtract the self-pair contribution (the query is at sep=0
    # against itself; it lands in shell ``j_z_n_self``).
    if same_catalog:
        i_z_n_self_per_q = (
            np.searchsorted(z_n_edges, query_z, side="right") - 1
        ).astype(np.int64)
    else:
        i_z_n_self_per_q = None

    # 3) allocate output cubes. Diagonal-only mode collapses the
    # (z_q, z_n) plane to a single z axis, dropping cube storage from
    # O(n_z²) to O(n_z) — enables much finer z-resolution at the
    # same budget. Requires z_q_edges == z_n_edges.
    if diagonal_only:
        if not np.array_equal(z_q_edges, z_n_edges):
            raise ValueError(
                "diagonal_only=True requires z_q_edges == z_n_edges; "
                "got distinct edge arrays.")
        if n_z_q != n_z_n:
            raise ValueError(
                "diagonal_only=True requires equal n_z_q and n_z_n.")
    H_max = max(k_max, 1)  # dummy slot when k_max==0 to keep shapes uniform
    if diagonal_only:
        n_z = n_z_q
        H_geq_k = np.zeros((n_theta, n_z, H_max), dtype=np.int64)
        sum_n = np.zeros((n_theta, n_z), dtype=np.float64)
        sum_n2 = np.zeros((n_theta, n_z), dtype=np.float64)
        sum_n3 = np.zeros((n_theta, n_z), dtype=np.float64)
        sum_n4 = np.zeros((n_theta, n_z), dtype=np.float64)
    else:
        H_geq_k = np.zeros((n_theta, n_z_q, n_z_n, H_max), dtype=np.int64)
        sum_n = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
        sum_n2 = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
        sum_n3 = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
        sum_n4 = np.zeros((n_theta, n_z_q, n_z_n), dtype=np.float64)
    N_q = np.zeros(n_z_q, dtype=np.int64)

    if do_jack:
        if diagonal_only:
            H_per_region = np.zeros(
                (n_theta, n_z, H_max, n_regions), dtype=np.int64,
            )
            sum_n_per_region = np.zeros(
                (n_theta, n_z, n_regions), dtype=np.float64,
            )
            sum_n2_per_region = np.zeros(
                (n_theta, n_z, n_regions), dtype=np.float64,
            )
            sum_n3_per_region = np.zeros(
                (n_theta, n_z, n_regions), dtype=np.float64,
            )
            sum_n4_per_region = np.zeros(
                (n_theta, n_z, n_regions), dtype=np.float64,
            )
        else:
            H_per_region = np.zeros(
                (n_theta, n_z_q, n_z_n, H_max, n_regions), dtype=np.int64,
            )
            sum_n_per_region = np.zeros(
                (n_theta, n_z_q, n_z_n, n_regions), dtype=np.float64,
            )
            sum_n2_per_region = np.zeros(
                (n_theta, n_z_q, n_z_n, n_regions), dtype=np.float64,
            )
            sum_n3_per_region = np.zeros(
                (n_theta, n_z_q, n_z_n, n_regions), dtype=np.float64,
            )
            sum_n4_per_region = np.zeros(
                (n_theta, n_z_q, n_z_n, n_regions), dtype=np.float64,
            )
        N_q_per_region = np.zeros((n_z_q, n_regions), dtype=np.int64)
    else:
        H_per_region = None
        sum_n_per_region = None
        sum_n2_per_region = None
        sum_n3_per_region = None
        sum_n4_per_region = None
        N_q_per_region = None

    # 4) per-query: kernel + aggregation. Threadpool parallelism over
    # queries because both kernels release the GIL.
    if n_threads is None:
        import os
        n_threads = os.cpu_count() or 1

    valid_q = np.flatnonzero(in_range_q)

    # Update per-shell query counts (cube-axis denominators).
    np.add.at(N_q, i_z_q_per_q[valid_q], 1)
    if do_jack:
        np.add.at(
            N_q_per_region,
            (i_z_q_per_q[valid_q], region_labels_query[valid_q]),
            1,
        )

    # Chunked per-query loop. The disc pre-fetch (healpy.query_disc) is
    # the memory hot spot — pixels-per-disc * Nq * 8 bytes can OOM at
    # catalog scale. We pre-fetch only one chunk at a time. Each
    # chunk's per-cap kernels run in the threadpool; aggregation is
    # serial in the main thread so the global cubes are write-safe.
    if n_threads > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=n_threads)
    else:
        pool = None

    chunk_size = max(1, int(query_chunk_size))
    n_chunks = (valid_q.size + chunk_size - 1) // chunk_size

    import time as _time
    for ic in range(n_chunks):
        t0 = _time.time()
        lo = ic * chunk_size
        hi = min(lo + chunk_size, valid_q.size)
        qi_block = valid_q[lo:hi]
        # Slice per-query coords for this chunk so the worker can index
        # by local qi without touching the full arrays.
        theta_q_chunk = theta_q[qi_block]
        phi_q_chunk = phi_q[qi_block]
        vecs_q_chunk = vecs_q[qi_block]

        def _one_full(qi_local: int):
            """Pre-fetch this query's HEALPix disc + run the per-cap
            kernel. Both healpy.query_disc (C extension) and the numba
            kernel release the GIL, so each worker thread is fully
            independent. Returning the disc to a per-thread local
            keeps peak memory at n_threads discs, not chunk_size discs.
            """
            ipix = hp.query_disc(
                nside_lookup, vecs_q_chunk[qi_local], theta_max,
                inclusive=True,
            ).astype(np.int64)
            if ipix.size == 0:
                return qi_local, np.zeros((n_theta, n_z_n),
                                          dtype=np.float64)
            return qi_local, _per_cap_count_kernel(
                theta_q_chunk[qi_local], phi_q_chunk[qi_local],
                ipix, pix_starts,
                theta_g_sorted, phi_g_sorted, z_g_sorted, w_g_sorted,
                theta_radii_rad, z_n_edges,
            )

        if pool is not None and qi_block.size > 1:
            results = pool.map(_one_full, range(qi_block.size))
        else:
            results = (_one_full(j) for j in range(qi_block.size))

        for qi_local, n_cap in results:
            qi = int(qi_block[qi_local])
            _aggregate_one(
                n_cap, qi,
                i_z_q_per_q, i_z_n_self_per_q,
                same_catalog,
                region_labels_query, do_jack,
                H_geq_k, sum_n, sum_n2, sum_n3, sum_n4,
                H_per_region,
                sum_n_per_region, sum_n2_per_region,
                sum_n3_per_region, sum_n4_per_region,
                k_max,
                diagonal_only=diagonal_only,
            )

        if progress:
            print(f"  [chunk {ic + 1}/{n_chunks}] queries {lo}..{hi} "
                  f"in {_time.time() - t0:.1f}s")

    if pool is not None:
        pool.shutdown(wait=True)

    # Convert H cubes from per-query "exactly k+1" semantic (the
    # optimised aggregation form) to the canonical "at least k+1"
    # semantic that all consumers expect (cic_pmf, VPF, etc.).
    if k_max > 0:
        H_geq_k = _finalize_h_suffix_sum(H_geq_k)
        if do_jack:
            if diagonal_only:
                # Diag region cube is (n_theta, n_z, k_max, n_regions);
                # the generic helper picks the wrong axis on ndim==4.
                H_per_region = _finalize_h_suffix_sum_diag_region(
                    H_per_region)
            else:
                H_per_region = _finalize_h_suffix_sum(H_per_region)

    return KnnCdfResult(
        H_geq_k=H_geq_k if k_max > 0 else np.zeros_like(H_geq_k),
        sum_n=sum_n,
        sum_n2=sum_n2,
        N_q=N_q,
        theta_radii_rad=theta_radii_rad,
        z_q_edges=z_q_edges,
        z_n_edges=z_n_edges,
        flavor=flavor,
        backend_used="numba",
        area_per_cap=2.0 * np.pi * (1.0 - np.cos(theta_radii_rad)),
        H_geq_k_per_region=H_per_region if k_max > 0 else None,
        sum_n_per_region=sum_n_per_region,
        sum_n2_per_region=sum_n2_per_region,
        N_q_per_region=N_q_per_region,
        is_diagonal=diagonal_only,
        sum_n3=sum_n3,
        sum_n4=sum_n4,
        sum_n3_per_region=sum_n3_per_region,
        sum_n4_per_region=sum_n4_per_region,
    )


def _aggregate_one(
    n_cap, qi,
    i_z_q_per_q, i_z_n_self_per_q, same_catalog,
    region_labels_query, do_jack,
    H_geq_k, sum_n, sum_n2, sum_n3, sum_n4,
    H_per_region,
    sum_n_per_region, sum_n2_per_region,
    sum_n3_per_region, sum_n4_per_region,
    k_max,
    diagonal_only=False,
):
    """Dispatch one query's aggregation to the appropriate fused
    Numba kernel. DD self-exclusion is handled inline inside the
    kernel (no n_cap.copy()). The H_geq_k cube is filled with
    "exactly k+1" semantics; ``_finalize_h_suffix_sum`` converts to
    "at least k+1" semantics at the end of the run.
    """
    iq = int(i_z_q_per_q[qi])
    if same_catalog:
        j_self = int(i_z_n_self_per_q[qi])
        if 0 <= j_self < n_cap.shape[1]:
            jn_self = j_self
            has_self = 1
        else:
            jn_self = 0
            has_self = 0
    else:
        jn_self = 0
        has_self = 0

    # Pick the right kernel pair based on cube shape (full vs diagonal).
    if diagonal_only:
        agg_global = _aggregate_query_global_diag
        agg_jack = _aggregate_query_jackknife_diag
    else:
        agg_global = _aggregate_query_global
        agg_jack = _aggregate_query_jackknife

    if do_jack:
        ir = int(region_labels_query[qi])
        if ir < 0 or ir >= sum_n_per_region.shape[-1]:
            # Out-of-region query — fall back to global only.
            agg_global(
                n_cap, iq, jn_self, has_self,
                sum_n, sum_n2, sum_n3, sum_n4, H_geq_k, k_max,
            )
            return
        agg_jack(
            n_cap, iq, jn_self, has_self, ir,
            sum_n, sum_n2, sum_n3, sum_n4, H_geq_k,
            sum_n_per_region, sum_n2_per_region,
            sum_n3_per_region, sum_n4_per_region, H_per_region, k_max,
        )
    else:
        agg_global(
            n_cap, iq, jn_self, has_self,
            sum_n, sum_n2, sum_n3, sum_n4, H_geq_k, k_max,
        )


def _finalize_h_suffix_sum(H_geq_k):
    """Convert the kNN-CDF cube from "exactly k+1" to "at least k+1"
    semantics by a reverse cumulative sum on the k-axis. Supports both
    full and diagonal-only cube layouts:

    - Full global:   (n_theta, n_z_q, n_z_n, k_max)            → axis -1
    - Full region:   (n_theta, n_z_q, n_z_n, k_max, n_regions) → axis -2
    - Diag  global:  (n_theta, n_z, k_max)                     → axis -1
    - Diag  region:  (n_theta, n_z, k_max, n_regions)          → axis -2

    One pass over the cube, O(N_total) — done once per
    ``joint_knn_cdf`` call.
    """
    if H_geq_k.ndim == 3:
        # Diagonal global cube: k axis is last.
        return np.flip(np.cumsum(np.flip(H_geq_k, axis=-1), axis=-1),
                        axis=-1)
    if H_geq_k.ndim == 4:
        # Full global OR diagonal-region. Heuristic: full global has
        # last axis = k_max; diag-region has second-to-last = k_max.
        # Disambiguate by checking whether the last axis is "small"
        # (regions, typically ≤100) vs "large" (k_max, typically
        # 100s). To avoid heuristic guessing we rely on the caller's
        # knowledge of which cube it is — here we treat ndim==4 as
        # FULL global cube (the legacy case) and use axis -1. The
        # diagonal-region cube routes through the ndim==4 branch by
        # callers that pass a region-shaped cube; we add an explicit
        # dispatch keyed on layout via a separate function below.
        return np.flip(np.cumsum(np.flip(H_geq_k, axis=-1), axis=-1),
                        axis=-1)
    if H_geq_k.ndim == 5:
        # Full per-region cube: k axis is second-to-last.
        return np.flip(np.cumsum(np.flip(H_geq_k, axis=-2), axis=-2),
                        axis=-2)
    raise ValueError(f"unexpected H_geq_k ndim {H_geq_k.ndim}")


def _finalize_h_suffix_sum_diag_region(H_geq_k):
    """Suffix-sum for the diagonal per-region cube of shape
    ``(n_theta, n_z, k_max, n_regions)``. The k axis is at index
    -2; regions stay last. Separate from the generic helper because
    the ndim==4 case is ambiguous between full-global and
    diag-region without extra context.
    """
    return np.flip(np.cumsum(np.flip(H_geq_k, axis=-2), axis=-2),
                    axis=-2)
