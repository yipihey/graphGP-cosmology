//! # angular_knn_cdf
//!
//! Per-query angular kNN-CDF on the celestial sphere via chord-length
//! metric, **bit-identical** in observable semantics to the Python
//! `joint_knn_cdf` numba backend at `twopt_density/knn_cdf.py:192`.
//!
//! ## What this produces
//!
//! For each (RA, Dec) query (mapped to an S² unit vector), counts
//! neighbours within angular separation θ for each of `n_theta`
//! ascending θ values, binned by neighbour redshift shell `z_n`,
//! aggregated per query redshift shell `z_q`. The output cubes are:
//!
//! - `H_geq_k[t, iq, jn, k]` — count of queries in shell `iq` whose
//!   cap of radius θ_t contains AT LEAST `k+1` shell-`jn` neighbours
//!   (post-suffix-sum semantics, matching `_finalize_h_suffix_sum`).
//! - `sum_n[t, iq, jn]` — Σ over queries q in shell iq of the
//!   weighted neighbour count in q's cap.
//! - `sum_n2[t, iq, jn]` — Σ over queries q of (weighted count)².
//! - `N_q[iq]` — number of queries in shell iq (denominator).
//!
//! Plus per-region jackknife versions when `region_labels_query`
//! is supplied.
//!
//! ## Algorithm
//!
//! Per-query depth-first traversal of an axis-aligned bounding box
//! (AABB) median-split tree built once over the neighbour catalog.
//! At each tree node, compute the squared minimum chord distance
//! from the query to the node's AABB; if it exceeds the largest θ
//! shell, prune the entire subtree. At leaves, iterate the small
//! point list with the exact chord test. Outer loop is parallelised
//! over queries with rayon. Cumulative-on-θ aggregation matches the
//! numba kernel; bit-identity is preserved by applying the SAME
//! chord and ladder rules at the leaf, only with fewer points
//! tested per query.
//!
//! ## Self-pair handling
//!
//! For DD-flavour with same catalogue (object identity), the
//! self-pair is removed by EXACTLY mirroring the numba behaviour:
//! count the self-pair with its full weight in `n_cap[:, jn_self]`,
//! then subtract 1.0 from those cells before aggregation, clipped
//! at 0. (For weighted catalogues with non-unit self-weight the
//! resulting cube preserves the numba quirk; the alternative
//! "skip self by TARGETID" behaviour is NOT used here for
//! bit-equivalence.) Detection is via TARGETID; pass
//! `same_catalog=true` and identical targetid arrays.

use rayon::prelude::*;

/// Output cubes of [`angular_knn_cdf_3d`]. Field shapes mirror the
/// Python `KnnCdfResult` dataclass; per-region cubes are populated
/// only when `region_labels_query.is_some()` and `n_regions > 0`.
pub struct AngularKnnCdfCubes {
    pub n_theta: usize,
    pub n_z_q: usize,
    pub n_z_n: usize,
    pub k_max: usize,
    pub n_regions: usize,
    /// True ⇒ cubes are stored in diagonal-only layout (z_q = z_n
    /// is collapsed to a single z axis). Shapes change accordingly:
    /// `h_geq_k` is `(n_theta, n_z, k_max)` instead of
    /// `(n_theta, n_z_q, n_z_n, k_max)`, `sum_n` is `(n_theta, n_z)`
    /// instead of `(n_theta, n_z_q, n_z_n)`, etc. Off-diagonal
    /// (z_q ≠ z_n) pair counts are silently dropped at the source
    /// (the inner loop short-circuits when neighbour-z-bin ≠
    /// query-z-bin), so the diagonal cube is bit-identical to the
    /// diagonal slice of the equivalent full cube.
    pub is_diagonal: bool,
    /// Flat row-major. Full: (n_theta, n_z_q, n_z_n, k_max). Diagonal:
    /// (n_theta, n_z, k_max).
    pub h_geq_k: Vec<i64>,
    /// Flat row-major. Full: (n_theta, n_z_q, n_z_n). Diagonal:
    /// (n_theta, n_z).
    pub sum_n: Vec<f64>,
    pub sum_n2: Vec<f64>,
    /// Higher-order raw moments (note v4_1 §6 — feed S₃ skewness and
    /// S₄ kurtosis at both raw and LS-corrected levels). Same shape
    /// as `sum_n` / `sum_n2`.
    pub sum_n3: Vec<f64>,
    pub sum_n4: Vec<f64>,
    /// Length n_z_q.
    pub n_q_per_zq: Vec<i64>,
    /// Flat row-major. Full:
    /// (n_theta, n_z_q, n_z_n, k_max, n_regions). Diagonal:
    /// (n_theta, n_z, k_max, n_regions).
    pub h_geq_k_per_region: Option<Vec<i64>>,
    /// Flat row-major. Full: (n_theta, n_z_q, n_z_n, n_regions).
    /// Diagonal: (n_theta, n_z, n_regions).
    pub sum_n_per_region: Option<Vec<f64>>,
    pub sum_n2_per_region: Option<Vec<f64>>,
    pub sum_n3_per_region: Option<Vec<f64>>,
    pub sum_n4_per_region: Option<Vec<f64>>,
    /// Flat row-major (n_z_q, n_regions).
    pub n_q_per_region: Option<Vec<i64>>,
}

/// Per-query angular kNN-CDF aggregation.
///
/// All array shapes are documented in [`AngularKnnCdfCubes`].
/// `chord_radii` must be ascending; `z_q_edges` and `z_n_edges` are
/// strictly ascending bin edges.
///
/// Returns cubes whose `h_geq_k` field has the post-suffix-sum
/// semantic ("at least k+1 neighbours") — bit-identical to what
/// Python `_finalize_h_suffix_sum` produces from the numba kernel.
pub fn angular_knn_cdf_3d(
    query_pts: &[[f64; 3]],
    query_z: &[f64],
    query_targetid: Option<&[i64]>,
    neigh_pts: &[[f64; 3]],
    neigh_z: &[f64],
    neigh_targetid: Option<&[i64]>,
    neigh_weight: Option<&[f64]>,
    chord_radii: &[f64],
    z_q_edges: &[f64],
    z_n_edges: &[f64],
    region_labels_query: Option<&[i64]>,
    n_regions: usize,
    k_max: usize,
    self_exclude: bool,
    diagonal_only: bool,
) -> AngularKnnCdfCubes {
    let n_q = query_pts.len();
    let n_n = neigh_pts.len();
    let n_theta = chord_radii.len();
    let n_z_q = z_q_edges.len() - 1;
    let n_z_n = z_n_edges.len() - 1;

    assert_eq!(query_pts.len(), query_z.len(), "query_pts/query_z size mismatch");
    assert_eq!(neigh_pts.len(), neigh_z.len(), "neigh_pts/neigh_z size mismatch");
    if let Some(w) = neigh_weight { assert_eq!(w.len(), n_n); }
    if let Some(r) = region_labels_query { assert_eq!(r.len(), n_q); }
    if let Some(t) = query_targetid { assert_eq!(t.len(), n_q); }
    if let Some(t) = neigh_targetid { assert_eq!(t.len(), n_n); }

    let do_jack = region_labels_query.is_some() && n_regions > 0;
    // Self-exclusion as per the numba pipeline: same_catalog is the
    // upstream Python flag (object identity). We require both
    // targetid arrays to be Some for the kernel-level matching, but
    // fall back to the position-equality detection inside the
    // Python wrapper (which can synthesise targetids if needed).
    let do_self = self_exclude
        && query_targetid.is_some()
        && neigh_targetid.is_some();

    // Precompute neighbour z_n bin (-1 for out-of-range).
    let neigh_zn_bin: Vec<i32> = neigh_z
        .iter()
        .map(|&z| zbin(z, z_n_edges))
        .collect();

    // Precompute query z_q bin (-1 for out-of-range).
    let query_zq_bin: Vec<i32> = query_z
        .iter()
        .map(|&z| zbin(z, z_q_edges))
        .collect();

    // For DD same-catalog self-exclusion we also need each query's
    // OWN z_n bin (the j_self z bin from numba's
    // i_z_n_self_per_q array).
    let query_zn_bin: Vec<i32> = query_z
        .iter()
        .map(|&z| zbin(z, z_n_edges))
        .collect();

    // chord_radii squared (avoids sqrt in inner loop).
    let chord_radii_sq: Vec<f64> = chord_radii.iter().map(|&r| r * r).collect();
    let chord_max_sq = chord_radii_sq[n_theta - 1];

    // Build a single AABB median-split tree on the neighbour catalog.
    // Cost: O(N log N) once, amortised across all N_q queries. The
    // tree is read-only, automatically Sync, and shared across the
    // rayon worker pool.
    let kd = KdTree::build(neigh_pts);

    // Validate diagonal-only mode requirements.
    if diagonal_only {
        if n_z_q != n_z_n {
            panic!("diagonal_only=true requires n_z_q == n_z_n; \
                    got n_z_q={} n_z_n={}", n_z_q, n_z_n);
        }
        if z_q_edges != z_n_edges {
            panic!("diagonal_only=true requires identical z_q_edges \
                    and z_n_edges arrays.");
        }
    }

    // Output cube sizes. Diagonal mode collapses (n_z_q, n_z_n) → n_z.
    let h_max = k_max.max(1);
    let cube_size: usize = if diagonal_only {
        n_theta * n_z_q
    } else {
        n_theta * n_z_q * n_z_n
    };
    let h_size = cube_size * h_max;
    let region_cube_size = if do_jack { cube_size * n_regions } else { 0 };
    let region_h_size = if do_jack { h_size * n_regions } else { 0 };
    let n_q_pr_size = if do_jack { n_z_q * n_regions } else { 0 };

    // Stride helpers for cube indexing.
    // FULL:
    //   sum_n[t, iq, jn] -> (t * n_z_q + iq) * n_z_n + jn
    //   h_geq_k[t, iq, jn, k] -> ((t * n_z_q + iq) * n_z_n + jn) * h_max + k
    //   sum_n_pr[t, iq, jn, ir] -> ((t * n_z_q + iq) * n_z_n + jn) * n_regions + ir
    //   h_pr[t, iq, jn, k, ir] -> (((t * n_z_q + iq) * n_z_n + jn) * h_max + k) * n_regions + ir
    // DIAGONAL (n_z = n_z_q = n_z_n; only iq == jn cells stored):
    //   sum_n[t, iq] -> t * n_z + iq
    //   h_geq_k[t, iq, k] -> (t * n_z + iq) * h_max + k
    //   sum_n_pr[t, iq, ir] -> (t * n_z + iq) * n_regions + ir
    //   h_pr[t, iq, k, ir] -> ((t * n_z + iq) * h_max + k) * n_regions + ir

    // Per-query parallel fold + reduce. Each thread accumulates
    // local cubes; the reduce combines them. Integer cube
    // accumulation is order-independent; f64 sum order may differ
    // from a single-thread numba run by O(eps × N), within "machine
    // precision".
    // Per-thread fold accumulators. Tuple fields, in order:
    //   h, s1, s2, s3, s4, n_q_arr,
    //   h_pr, s1_pr, s2_pr, s3_pr, s4_pr, n_q_pr_arr
    let local_init = || (
        vec![0i64; h_size],
        vec![0.0f64; cube_size],
        vec![0.0f64; cube_size],
        vec![0.0f64; cube_size],
        vec![0.0f64; cube_size],
        vec![0i64; n_z_q],
        vec![0i64; region_h_size],
        vec![0.0f64; region_cube_size],
        vec![0.0f64; region_cube_size],
        vec![0.0f64; region_cube_size],
        vec![0.0f64; region_cube_size],
        vec![0i64; n_q_pr_size],
    );

    let folded = (0..n_q).into_par_iter().fold(
        local_init,
        |(mut h, mut s1, mut s2, mut s3, mut s4, mut n_q_arr,
          mut h_pr, mut s1_pr, mut s2_pr, mut s3_pr, mut s4_pr, mut n_q_pr_arr),
         iq| {
            let iz_q = query_zq_bin[iq];
            if iz_q < 0 {
                return (h, s1, s2, s3, s4, n_q_arr,
                        h_pr, s1_pr, s2_pr, s3_pr, s4_pr, n_q_pr_arr);
            }
            let iz_q = iz_q as usize;
            n_q_arr[iz_q] += 1;

            // Region for this query (used for both N_q_per_region and the
            // jackknife cube increments). `ir < 0` ⇒ skip per-region path.
            let ir: i32 = if do_jack {
                let r = region_labels_query.unwrap()[iq];
                if r < 0 || (r as usize) >= n_regions { -1 } else { r as i32 }
            } else { -1 };
            if ir >= 0 {
                n_q_pr_arr[iz_q * n_regions + ir as usize] += 1;
            }

            // Per-query n_cap (n_theta × n_z_n) — flat row-major:
            //   n_cap[t, jn] = n_cap_flat[t * n_z_n + jn]
            let mut n_cap = vec![0.0f64; n_theta * n_z_n];

            let q = query_pts[iq];
            let q_tid = query_targetid.map(|t| t[iq]).unwrap_or(-1);

            // Inner loop: kd-tree-pruned traversal of the neighbour
            // catalog. Tree nodes whose AABB-min-distance to q
            // exceeds chord_max_sq are skipped wholesale; at leaves
            // we apply the same exact chord and ladder rules as the
            // brute-force version, so per-pair semantics are bit-
            // identical.
            kd.for_each_within(
                neigh_pts, q, chord_max_sq,
                |j, chord_sq| {
                    let j = j as usize;
                    let iz_n = neigh_zn_bin[j];
                    if iz_n < 0 { return; }
                    let iz_n = iz_n as usize;

                    // Find smallest t with chord_radii_sq[t] >= chord_sq.
                    let mut lo = 0usize;
                    let mut hi = n_theta;
                    while lo < hi {
                        let mid = (lo + hi) >> 1;
                        if chord_radii_sq[mid] < chord_sq { lo = mid + 1; }
                        else { hi = mid; }
                    }
                    let t_bin = lo;
                    if t_bin >= n_theta { return; }

                    let w = neigh_weight.map(|wv| wv[j]).unwrap_or(1.0);
                    n_cap[t_bin * n_z_n + iz_n] += w;
                },
            );

            // Cumulative sum along theta axis (mirror
            // _per_cap_count_kernel:124–128).
            for jn in 0..n_z_n {
                let mut acc = 0.0;
                for t in 0..n_theta {
                    acc += n_cap[t * n_z_n + jn];
                    n_cap[t * n_z_n + jn] = acc;
                }
            }

            // DD same-catalog self-pair exclusion (mirror numba
            // _aggregate_query_global / _aggregate_query_jackknife
            // lines 119–120 / 152–153). Subtract 1.0 from
            // n_cap[t, jn_self_q] then clip to 0.
            if do_self {
                let q_zn = query_zn_bin[iq];
                if q_zn >= 0 {
                    let jn_self = q_zn as usize;
                    // Verify self-pair was actually counted: the
                    // self contribution at sep=0 lives in t_bin=0
                    // and propagates through cumulative-sum to all
                    // theta bins. Numba subtracts unconditionally
                    // (assumes self-weight = 1.0).
                    //
                    // Note: if the upstream caller asked for self-
                    // exclusion via TARGETID matching but the query
                    // doesn't actually appear in the neighbour
                    // catalogue (mismatched arrays), the subtract
                    // can produce -1 values that get clipped to 0.
                    for t in 0..n_theta {
                        let v = &mut n_cap[t * n_z_n + jn_self];
                        *v -= 1.0;
                        if *v < 0.0 { *v = 0.0; }
                    }
                }
                // Touch q_tid to suppress unused-var lint when
                // the actual TARGETID check below isn't reached
                // (do_self gates the whole block).
                let _ = q_tid;
            }

            // Aggregate n_cap into global + per-region cubes.
            // Diagonal-only: only the jn == iz_q cell is written;
            // the cube layout collapses (n_z_q, n_z_n) → n_z so the
            // index stride changes accordingly.
            if diagonal_only {
                let jn = iz_q;
                if jn < n_z_n {
                    for t in 0..n_theta {
                        let v = n_cap[t * n_z_n + jn];
                        let v2 = v * v;
                        let v3 = v2 * v;
                        let v4 = v2 * v2;
                        let cube_idx = t * n_z_q + iz_q;  // n_z_q == n_z
                        s1[cube_idx] += v;
                        s2[cube_idx] += v2;
                        s3[cube_idx] += v3;
                        s4[cube_idx] += v4;
                        if ir >= 0 {
                            let pr_idx = cube_idx * n_regions + ir as usize;
                            s1_pr[pr_idx] += v;
                            s2_pr[pr_idx] += v2;
                            s3_pr[pr_idx] += v3;
                            s4_pr[pr_idx] += v4;
                        }
                        if k_max > 0 {
                            let mut n_int = v as i64;
                            if n_int > k_max as i64 { n_int = k_max as i64; }
                            if n_int >= 1 {
                                let h_idx = cube_idx * h_max + (n_int as usize - 1);
                                h[h_idx] += 1;
                                if ir >= 0 {
                                    let h_pr_idx = h_idx * n_regions + ir as usize;
                                    h_pr[h_pr_idx] += 1;
                                }
                            }
                        }
                    }
                }
            } else {
                for t in 0..n_theta {
                    for jn in 0..n_z_n {
                        let v = n_cap[t * n_z_n + jn];
                        let v2 = v * v;
                        let v3 = v2 * v;
                        let v4 = v2 * v2;
                        let cube_idx = (t * n_z_q + iz_q) * n_z_n + jn;
                        s1[cube_idx] += v;
                        s2[cube_idx] += v2;
                        s3[cube_idx] += v3;
                        s4[cube_idx] += v4;
                        if ir >= 0 {
                            let pr_idx = cube_idx * n_regions + ir as usize;
                            s1_pr[pr_idx] += v;
                            s2_pr[pr_idx] += v2;
                            s3_pr[pr_idx] += v3;
                            s4_pr[pr_idx] += v4;
                        }
                        if k_max > 0 {
                            // Mirror the optimised aggregation: write
                            // h[..., n_int - 1] only (later: suffix-sum).
                            let mut n_int = v as i64;
                            if n_int > k_max as i64 { n_int = k_max as i64; }
                            if n_int >= 1 {
                                let h_idx = cube_idx * h_max + (n_int as usize - 1);
                                h[h_idx] += 1;
                                if ir >= 0 {
                                    let h_pr_idx = h_idx * n_regions + ir as usize;
                                    h_pr[h_pr_idx] += 1;
                                }
                            }
                        }
                    }
                }
            }

            (h, s1, s2, s3, s4, n_q_arr,
             h_pr, s1_pr, s2_pr, s3_pr, s4_pr, n_q_pr_arr)
        },
    ).reduce(
        local_init,
        |(mut a_h, mut a_s1, mut a_s2, mut a_s3, mut a_s4, mut a_nq,
          mut a_hpr, mut a_s1pr, mut a_s2pr, mut a_s3pr, mut a_s4pr, mut a_nqpr),
         (b_h, b_s1, b_s2, b_s3, b_s4, b_nq,
          b_hpr, b_s1pr, b_s2pr, b_s3pr, b_s4pr, b_nqpr)| {
            for i in 0..a_h.len() { a_h[i] += b_h[i]; }
            for i in 0..a_s1.len() { a_s1[i] += b_s1[i]; }
            for i in 0..a_s2.len() { a_s2[i] += b_s2[i]; }
            for i in 0..a_s3.len() { a_s3[i] += b_s3[i]; }
            for i in 0..a_s4.len() { a_s4[i] += b_s4[i]; }
            for i in 0..a_nq.len() { a_nq[i] += b_nq[i]; }
            for i in 0..a_hpr.len() { a_hpr[i] += b_hpr[i]; }
            for i in 0..a_s1pr.len() { a_s1pr[i] += b_s1pr[i]; }
            for i in 0..a_s2pr.len() { a_s2pr[i] += b_s2pr[i]; }
            for i in 0..a_s3pr.len() { a_s3pr[i] += b_s3pr[i]; }
            for i in 0..a_s4pr.len() { a_s4pr[i] += b_s4pr[i]; }
            for i in 0..a_nqpr.len() { a_nqpr[i] += b_nqpr[i]; }
            (a_h, a_s1, a_s2, a_s3, a_s4, a_nq,
             a_hpr, a_s1pr, a_s2pr, a_s3pr, a_s4pr, a_nqpr)
        },
    );

    let (mut h_geq_k, sum_n, sum_n2, sum_n3, sum_n4, n_q_per_zq,
         mut h_geq_k_pr,
         sum_n_pr, sum_n2_pr, sum_n3_pr, sum_n4_pr,
         n_q_pr_arr) = folded;

    // Suffix-sum on k axis (mirror _finalize_h_suffix_sum).
    if k_max > 0 {
        // Global cube: shape (n_theta, n_z_q, n_z_n, k_max).
        // Loop over (t, iq, jn) and for each contiguous k-strip of
        // length k_max, do an in-place reverse cumulative sum.
        for cube_idx in 0..cube_size {
            let base = cube_idx * h_max;
            for k in (0..h_max - 1).rev() {
                h_geq_k[base + k] += h_geq_k[base + k + 1];
            }
        }
        if do_jack {
            // Per-region cube: shape (n_theta, n_z_q, n_z_n, k_max,
            // n_regions). For each (cube_idx, ir) the k-axis stride
            // is n_regions; loop over (cube_idx, ir) and do reverse
            // cumulative sum on the k slice.
            for cube_idx in 0..cube_size {
                for ir in 0..n_regions {
                    let base = cube_idx * h_max * n_regions + ir;
                    for k in (0..h_max - 1).rev() {
                        h_geq_k_pr[base + k * n_regions]
                            += h_geq_k_pr[base + (k + 1) * n_regions];
                    }
                }
            }
        }
    }

    AngularKnnCdfCubes {
        n_theta, n_z_q, n_z_n, k_max,
        n_regions: if do_jack { n_regions } else { 0 },
        is_diagonal: diagonal_only,
        h_geq_k,
        sum_n, sum_n2, sum_n3, sum_n4, n_q_per_zq,
        h_geq_k_per_region: if do_jack { Some(h_geq_k_pr) } else { None },
        sum_n_per_region: if do_jack { Some(sum_n_pr) } else { None },
        sum_n2_per_region: if do_jack { Some(sum_n2_pr) } else { None },
        sum_n3_per_region: if do_jack { Some(sum_n3_pr) } else { None },
        sum_n4_per_region: if do_jack { Some(sum_n4_pr) } else { None },
        n_q_per_region: if do_jack { Some(n_q_pr_arr) } else { None },
    }
}

/// Find the largest `i` such that `edges[i] <= z` and `z < edges[i+1]`.
/// Returns -1 if `z` is out of range `[edges[0], edges[n])`.
#[inline]
fn zbin(z: f64, edges: &[f64]) -> i32 {
    let n = edges.len() - 1;
    if z < edges[0] || z >= edges[n] { return -1; }
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if edges[mid + 1] <= z { lo = mid + 1; }
        else { hi = mid; }
    }
    lo as i32
}

// ============================================================================
// Spatial index: AABB median-split tree
// ============================================================================
//
// Built once per `angular_knn_cdf_3d` call on the neighbour catalog.
// Per-query traversal prunes whole subtrees whose minimum-chord-
// distance to the query exceeds `chord_max`. Gives O(log N)
// expected work per query against ~constant work per neighbour at
// leaf cells, replacing the O(N_n) brute-force inner loop.
//
// We use a binary tree (kd-tree-like, splitting on the widest axis
// at the geometric midpoint) rather than the project's bit-vector
// cascade because:
//   - The cascade operates on integer-quantised coords; mapping cell
//     ids back to f64 AABBs for distance pruning is non-trivial.
//   - For unit vectors on S² confined to a small box, a simple
//     median-split on f64 coords is competitive in practice.
//
// The struct is `Send + Sync` (auto-derived) so it can be shared
// read-only across rayon workers.

const KDT_LEAF_MAX: usize = 32;

struct KdNode {
    aabb_min: [f64; 3],
    aabb_max: [f64; 3],
    /// Index range into [`KdTree::perm`] for this subtree's points.
    point_first: u32,
    point_count: u32,
    /// `u32::MAX` for leaf, otherwise child index in
    /// [`KdTree::nodes`]. Children are guaranteed to be present in
    /// pairs (left, right).
    left: u32,
    right: u32,
}

struct KdTree {
    /// Permutation: `position-in-tree` → original neighbour index.
    /// Length = `n_n`. Tree leaves point into contiguous ranges of
    /// this array.
    perm: Vec<u32>,
    nodes: Vec<KdNode>,
}

impl KdTree {
    fn build(points: &[[f64; 3]]) -> Self {
        let n = points.len();
        let mut perm: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<KdNode> = Vec::with_capacity(
            (2 * n).max(1) / KDT_LEAF_MAX.max(1),
        );
        if n > 0 {
            Self::build_recursive(points, &mut perm, 0, n, &mut nodes);
        }
        Self { perm, nodes }
    }

    fn build_recursive(
        points: &[[f64; 3]],
        perm: &mut [u32],
        start: usize,
        end: usize,
        nodes: &mut Vec<KdNode>,
    ) -> u32 {
        let n = end - start;
        // Compute AABB of points in perm[start..end].
        let mut amin = [f64::INFINITY; 3];
        let mut amax = [f64::NEG_INFINITY; 3];
        for k in start..end {
            let i = perm[k] as usize;
            let p = points[i];
            for d in 0..3 {
                if p[d] < amin[d] { amin[d] = p[d]; }
                if p[d] > amax[d] { amax[d] = p[d]; }
            }
        }
        let node_idx = nodes.len() as u32;
        nodes.push(KdNode {
            aabb_min: amin,
            aabb_max: amax,
            point_first: start as u32,
            point_count: n as u32,
            left: u32::MAX,
            right: u32::MAX,
        });
        if n <= KDT_LEAF_MAX {
            return node_idx;
        }
        // Pick widest axis.
        let widths = [amax[0] - amin[0], amax[1] - amin[1], amax[2] - amin[2]];
        let split_axis = if widths[0] >= widths[1] && widths[0] >= widths[2] { 0 }
                         else if widths[1] >= widths[2] { 1 } else { 2 };
        let split_value = 0.5 * (amin[split_axis] + amax[split_axis]);

        // Hoare-style partition of perm[start..end] by points[i][axis] < split.
        let mut lo = start;
        let mut hi = end;
        while lo < hi {
            let i = perm[lo] as usize;
            if points[i][split_axis] < split_value {
                lo += 1;
            } else {
                hi -= 1;
                perm.swap(lo, hi);
            }
        }
        let mut mid = lo;
        // If the split degenerates (all points on one side, e.g.
        // because the geometric midpoint falls outside the actual
        // point distribution), force a balanced index split.
        if mid <= start || mid >= end {
            mid = start + n / 2;
        }
        let left = Self::build_recursive(points, perm, start, mid, nodes);
        let right = Self::build_recursive(points, perm, mid, end, nodes);
        nodes[node_idx as usize].left = left;
        nodes[node_idx as usize].right = right;
        node_idx
    }

    /// Visit every neighbour point whose chord distance from `query`
    /// is ≤ √`chord_max_sq`, calling `callback(orig_index,
    /// chord_sq)`. Tree pruning short-circuits whole subtrees that
    /// cannot contain such points.
    fn for_each_within<F: FnMut(u32, f64)>(
        &self,
        points: &[[f64; 3]],
        query: [f64; 3],
        chord_max_sq: f64,
        mut callback: F,
    ) {
        if self.nodes.is_empty() { return; }
        // Iterative DFS via an explicit stack (avoids recursion
        // overhead and lets us bail early without unwinding).
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0u32);
        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];

            // Squared minimum chord distance from query to AABB.
            let mut dmin_sq = 0.0f64;
            for d in 0..3 {
                if query[d] < node.aabb_min[d] {
                    let dd = node.aabb_min[d] - query[d];
                    dmin_sq += dd * dd;
                } else if query[d] > node.aabb_max[d] {
                    let dd = query[d] - node.aabb_max[d];
                    dmin_sq += dd * dd;
                }
            }
            if dmin_sq > chord_max_sq {
                continue;
            }

            if node.left == u32::MAX {
                // Leaf — exact distance test on each point.
                let lo = node.point_first as usize;
                let hi = lo + node.point_count as usize;
                for k in lo..hi {
                    let i = self.perm[k];
                    let p = points[i as usize];
                    let dx = query[0] - p[0];
                    let dy = query[1] - p[1];
                    let dz = query[2] - p[2];
                    let cs = dx * dx + dy * dy + dz * dz;
                    if cs <= chord_max_sq {
                        callback(i, cs);
                    }
                }
            } else {
                // Internal — push both children.
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: O(N_q × N_n) over all pairs, no
    /// parallelism, no rayon. Same observable as
    /// `angular_knn_cdf_3d` but written in the simplest possible
    /// form. Used to sanity-check the parallel implementation.
    fn brute_reference(
        query_pts: &[[f64; 3]],
        query_z: &[f64],
        neigh_pts: &[[f64; 3]],
        neigh_z: &[f64],
        chord_radii: &[f64],
        z_q_edges: &[f64],
        z_n_edges: &[f64],
        k_max: usize,
    ) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>) {
        let n_theta = chord_radii.len();
        let n_z_q = z_q_edges.len() - 1;
        let n_z_n = z_n_edges.len() - 1;
        let h_max = k_max.max(1);
        let cube_size = n_theta * n_z_q * n_z_n;
        let mut h = vec![0i64; cube_size * h_max];
        let mut s1 = vec![0.0f64; cube_size];
        let mut s2 = vec![0.0f64; cube_size];
        let mut nq = vec![0i64; n_z_q];

        let chord_max_sq = chord_radii[n_theta - 1] * chord_radii[n_theta - 1];

        for iq in 0..query_pts.len() {
            let iz_q = zbin(query_z[iq], z_q_edges);
            if iz_q < 0 { continue; }
            let iz_q = iz_q as usize;
            nq[iz_q] += 1;

            let mut n_cap = vec![0.0f64; n_theta * n_z_n];
            let q = query_pts[iq];

            for j in 0..neigh_pts.len() {
                let iz_n = zbin(neigh_z[j], z_n_edges);
                if iz_n < 0 { continue; }
                let iz_n = iz_n as usize;
                let p = neigh_pts[j];
                let dx = q[0] - p[0];
                let dy = q[1] - p[1];
                let dz = q[2] - p[2];
                let cs = dx * dx + dy * dy + dz * dz;
                if cs > chord_max_sq { continue; }
                let mut t_bin = n_theta;
                for t in 0..n_theta {
                    if cs <= chord_radii[t] * chord_radii[t] {
                        t_bin = t;
                        break;
                    }
                }
                if t_bin >= n_theta { continue; }
                n_cap[t_bin * n_z_n + iz_n] += 1.0;
            }

            // Cumulative on theta.
            for jn in 0..n_z_n {
                let mut acc = 0.0;
                for t in 0..n_theta {
                    acc += n_cap[t * n_z_n + jn];
                    n_cap[t * n_z_n + jn] = acc;
                }
            }

            for t in 0..n_theta {
                for jn in 0..n_z_n {
                    let v = n_cap[t * n_z_n + jn];
                    let ci = (t * n_z_q + iz_q) * n_z_n + jn;
                    s1[ci] += v;
                    s2[ci] += v * v;
                    if k_max > 0 {
                        let mut n_int = v as i64;
                        if n_int > k_max as i64 { n_int = k_max as i64; }
                        if n_int >= 1 {
                            h[ci * h_max + (n_int as usize - 1)] += 1;
                        }
                    }
                }
            }
        }

        // Suffix sum.
        if k_max > 0 {
            for ci in 0..cube_size {
                let base = ci * h_max;
                for k in (0..h_max - 1).rev() {
                    h[base + k] += h[base + k + 1];
                }
            }
        }
        (h, s1, s2, nq)
    }

    #[test]
    fn matches_brute_force_unweighted() {
        // 50-point synthetic catalog on a unit sphere patch.
        let n = 50;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                // Quasi-random scatter on a small sphere patch.
                let t = (i as f64) * 0.7124 + 0.3;
                let phi = (i as f64) * 1.193 + 0.1;
                let dec = 0.1 + 0.05 * t.sin();
                let ra = 0.05 * phi.cos();
                let theta_polar = std::f64::consts::FRAC_PI_2 - dec;
                [theta_polar.sin() * ra.cos(),
                 theta_polar.sin() * ra.sin(),
                 theta_polar.cos()]
            })
            .collect();
        let z: Vec<f64> = (0..n).map(|i| 1.0 + 0.001 * (i as f64)).collect();
        let chord = vec![0.005, 0.01, 0.02, 0.05];
        let z_q = vec![0.9, 1.025, 1.1];
        let z_n = z_q.clone();

        let res = angular_knn_cdf_3d(
            &pts, &z, None, &pts, &z, None, None,
            &chord, &z_q, &z_n, None, 0, 5, false, false,
        );
        let (h_ref, s1_ref, s2_ref, nq_ref) = brute_reference(
            &pts, &z, &pts, &z, &chord, &z_q, &z_n, 5);

        assert_eq!(res.h_geq_k, h_ref, "H_geq_k mismatch vs brute reference");
        assert_eq!(res.n_q_per_zq, nq_ref, "N_q mismatch");
        for i in 0..res.sum_n.len() {
            assert!((res.sum_n[i] - s1_ref[i]).abs() < 1e-12,
                    "sum_n[{}] {} vs {}", i, res.sum_n[i], s1_ref[i]);
            assert!((res.sum_n2[i] - s2_ref[i]).abs() < 1e-12,
                    "sum_n2[{}] {} vs {}", i, res.sum_n2[i], s2_ref[i]);
        }
    }

    #[test]
    fn h_geq_k_monotone_after_suffix_sum() {
        // Random catalog; H_geq_k must be monotone non-increasing in k.
        let n = 200;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let f = i as f64;
                let theta_polar = 1.5 + 0.001 * (f * 0.97).sin();
                let phi = 0.001 * f;
                [theta_polar.sin() * phi.cos(),
                 theta_polar.sin() * phi.sin(),
                 theta_polar.cos()]
            }).collect();
        let z: Vec<f64> = (0..n).map(|i| 0.9 + 0.002 * (i as f64)).collect();
        let chord = vec![0.001, 0.005, 0.02, 0.1];
        let z_q = vec![0.9, 1.1, 1.3];
        let z_n = z_q.clone();

        let res = angular_knn_cdf_3d(
            &pts, &z, None, &pts, &z, None, None,
            &chord, &z_q, &z_n, None, 0, 8, false, false,
        );

        let n_theta = chord.len();
        let n_z_q = z_q.len() - 1;
        let n_z_n = z_n.len() - 1;
        let h_max = 8;
        for t in 0..n_theta {
            for iq in 0..n_z_q {
                for jn in 0..n_z_n {
                    let base = ((t * n_z_q + iq) * n_z_n + jn) * h_max;
                    for k in 0..h_max - 1 {
                        assert!(res.h_geq_k[base + k] >= res.h_geq_k[base + k + 1],
                                "H_geq_k not monotone at t={} iq={} jn={} k={}: \
                                 {} vs {}",
                                t, iq, jn, k,
                                res.h_geq_k[base + k], res.h_geq_k[base + k + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn jackknife_sums_to_global() {
        let n = 100;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let f = i as f64;
                let theta_polar = 1.4 + 0.01 * (f * 0.71).sin();
                let phi = 0.05 * (f * 0.31).cos();
                [theta_polar.sin() * phi.cos(),
                 theta_polar.sin() * phi.sin(),
                 theta_polar.cos()]
            }).collect();
        let z: Vec<f64> = (0..n).map(|i| 0.9 + 0.003 * (i as f64)).collect();
        let regions: Vec<i64> = (0..n).map(|i| (i as i64) % 5).collect();
        let chord = vec![0.005, 0.02, 0.1];
        let z_q = vec![0.9, 1.1, 1.3];
        let z_n = z_q.clone();

        let res = angular_knn_cdf_3d(
            &pts, &z, None, &pts, &z, None, None,
            &chord, &z_q, &z_n, Some(&regions), 5, 5, false, false,
        );

        // Per-region sum_n must equal global sum_n.
        let pr = res.sum_n_per_region.unwrap();
        let cube_size = res.n_theta * res.n_z_q * res.n_z_n;
        for ci in 0..cube_size {
            let pr_sum: f64 = (0..5)
                .map(|ir| pr[ci * 5 + ir]).sum();
            assert!((pr_sum - res.sum_n[ci]).abs() < 1e-10,
                    "per-region sum_n at ci={} doesn't match global: {} vs {}",
                    ci, pr_sum, res.sum_n[ci]);
        }
        // N_q_per_region rows must sum to global N_q.
        let nqpr = res.n_q_per_region.unwrap();
        for iq in 0..res.n_z_q {
            let nq_sum: i64 = (0..5).map(|ir| nqpr[iq * 5 + ir]).sum();
            assert_eq!(nq_sum, res.n_q_per_zq[iq],
                       "N_q_per_region row iq={} doesn't sum to global", iq);
        }
    }
}
