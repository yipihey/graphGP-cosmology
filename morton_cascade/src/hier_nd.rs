// Generic D-dimensional hierarchical cascade.
//
// Const-generic over the dimension D. The compiler monomorphizes each D you
// instantiate (e.g. cascade_nd::<2>, cascade_nd::<3>, cascade_nd::<5>), so the
// 2^D children loop is fully unrolled and there is no per-call branching on D.
//
// Layout convention: flat index in row-major-by-axis order
//     i = a_{D-1} * M^{D-1} + ... + a_1 * M + a_0
// where each a_d in [0, M) decomposes as
//     a_d = c_d * h + s_d,   c_d in [0, M/h),  s_d in [0, h).
//
// Reduction l+1 -> l (h_cur -> h_par = 2 h_cur): each parent (c_p, s_p) sums
// 2^D children. Per axis:
//     s_c[d] = s_p[d] mod h_cur                      (= a_p[d] mod h_cur)
//     off[d] = s_p[d] / h_cur                        (high bit of s_p in [0, h_par))
//     c_c[d] = (2 * c_p[d] + off[d] + bit_d) mod n_cur,   bit_d = (child >> d) & 1
//     a_c[d] = c_c[d] * h_cur + s_c[d]
// then the child's flat index is sum_d a_c[d] * M^d.
//
// Limits: M = 2^(L + s_sub) and the buffer is M^D entries. For D=5, M=8 is the
// largest practical (32k entries; M=16 gives 1M; M=32 gives 33M which is borderline).

#[derive(Clone, Debug)]
pub struct LevelStatsND {
    pub n_cells_total: usize,
    pub mean: f64,
    pub var: f64,
    pub dvar: f64,
}

/// Bin u16-coords into an M^D grid. Coords are tuples of [u16; D] passed as a slice.
pub fn bin_nd<const D: usize>(pts: &[[u16; D]], l_max: usize, s_subshift: usize) -> (Vec<u32>, usize) {
    assert!(l_max + s_subshift <= 16, "fine grid would exceed u16 resolution");
    assert!(D >= 1 && D <= 8, "D out of supported range");
    let m_bits = l_max + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;
    let total = m.pow(D as u32);
    let mut c = vec![0u32; total];
    for p in pts {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            let a = (p[d] >> shift) as usize;
            idx += a * stride;
            stride *= m;
        }
        c[idx] += 1;
    }
    (c, m)
}

/// Generic D-dim hierarchical cascade. Returns (level_stats, n_distinct).
///
/// `l_max` chooses the deepest reported tree level; the tree-coord box is 2^l_max per side.
/// `s_subshift` adds extra refinement levels for shift averaging (s=1 is usually best).
///
/// The buffer size is M^D where M = 2^(l_max + s_subshift). At D=5 with l_max=3, s_sub=1
/// gives M=16 -> 1M cell buffer (8 MB as u64). At l_max=4 -> M=32 -> 33M cells (256 MB);
/// borderline on a workstation.
pub fn cascade_nd<const D: usize>(
    pts: &[[u16; D]],
    l_max: usize,
    s_subshift: usize,
    periodic: bool,
) -> (Vec<LevelStatsND>, Vec<usize>) {
    let n_levels = l_max + 1;
    let n_children = 1usize << D;
    let total_pts = pts.len() as u64;

    let (c_grid, m) = bin_nd::<D>(pts, l_max, s_subshift);
    let buf_len = m.pow(D as u32);

    let mut cur: Vec<u64> = c_grid.iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; buf_len];

    let mut out: Vec<Option<LevelStatsND>> = (0..n_levels).map(|_| None).collect();
    let mut n_distinct: Vec<usize> = vec![0; n_levels];

    let l_max_eff = l_max + s_subshift;

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;

        for v in nxt.iter_mut() { *v = 0; }

        let mut sum_n: u128 = 0;
        let mut sum_n2: u128 = 0;
        let mut sum_sib_var: f64 = 0.0;
        let mut n_valid: u64 = 0;

        // Iterate over all parent flat indices i_p in [0, M^D).
        // Decompose into per-axis (c_d, s_d) on the fly.
        for i_p in 0..buf_len {
            // Decompose i_p into per-axis a_d
            let mut a_p = [0usize; 8];   // up to D=8
            let mut tmp = i_p;
            for d in 0..D {
                a_p[d] = tmp % m;
                tmp /= m;
            }

            // Per-axis: c_p[d], s_p[d], s_c[d], off[d], validity for non-periodic
            let mut c_p = [0usize; 8];
            let mut s_c = [0usize; 8];
            let mut off = [0usize; 8];
            let mut valid_axis = [true; 8];
            let mut all_valid = true;
            for d in 0..D {
                let c_d = a_p[d] / h_par;
                let s_d = a_p[d] % h_par;
                c_p[d] = c_d;
                s_c[d] = s_d % h_cur;
                off[d] = s_d / h_cur;
                if !periodic {
                    valid_axis[d] = (s_d + (c_d + 1) * h_par <= m)
                                 && (2 * c_d + off[d] + 1 < n_per_axis_cur);
                    if !valid_axis[d] { all_valid = false; }
                }
            }

            // Now sum 2^D children. Each child has bit_d = (child >> d) & 1.
            let mut s = 0u64;
            let mut s2 = 0u128;
            for child in 0..n_children {
                // Compute child flat index by accumulating per-axis contributions
                let mut idx_c = 0usize;
                let mut stride = 1usize;
                for d in 0..D {
                    let bit_d = (child >> d) & 1;
                    let c_c_d = (2 * c_p[d] + off[d] + bit_d) % n_per_axis_cur;
                    let a_c_d = c_c_d * h_cur + s_c[d];
                    idx_c += a_c_d * stride;
                    stride *= m;
                }
                let n_v = cur[idx_c];
                s += n_v;
                s2 += (n_v as u128) * (n_v as u128);
            }
            nxt[i_p] = s;

            if all_valid {
                sum_n += s as u128;
                sum_n2 += (s as u128) * (s as u128);
                let mean_c = (s as f64) / n_children as f64;
                let v_sib = (s2 as f64) / n_children as f64 - mean_c * mean_c;
                sum_sib_var += v_sib;
                n_valid += 1;
            }
        }

        if par_level <= l_max {
            let l_rep = par_level;
            let n_used = if periodic { buf_len as f64 } else { n_valid as f64 };
            if n_used > 0.0 {
                let mean = (sum_n as f64) / n_used;
                let var  = ((sum_n2 as f64) / n_used - mean * mean).max(0.0);
                let avg_sib_var = sum_sib_var / n_used;
                let dvar = var - avg_sib_var;
                let n_cells_total = 1usize << (D * l_rep);
                let n_dist = if periodic {
                    h_par.pow(D as u32).min(buf_len)
                } else {
                    n_valid as usize
                };
                out[l_rep] = Some(LevelStatsND { n_cells_total, mean, var, dvar });
                n_distinct[l_rep] = n_dist;
            }
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    // s_subshift == 0: deepest level comes from the binned grid itself
    if s_subshift == 0 && out[l_max].is_none() {
        let mut sum_n: u128 = 0;
        let mut sum_n2: u128 = 0;
        for &v in c_grid.iter() {
            sum_n += v as u128;
            sum_n2 += (v as u128) * (v as u128);
        }
        let n_total = buf_len as f64;
        let mean = (sum_n as f64) / n_total;
        let var = ((sum_n2 as f64) / n_total - mean * mean).max(0.0);
        out[l_max] = Some(LevelStatsND {
            n_cells_total: 1 << (D * l_max),
            mean, var, dvar: var,
        });
        n_distinct[l_max] = 1;
    }

    // Conservation check (periodic only)
    if periodic {
        for l in 0..n_levels {
            if let Some(s) = &out[l] {
                let cells_per_shift = (1u64 << ((D as u64) * l as u64)) as f64;
                let n_dist = n_distinct[l] as f64;
                let actual = (s.mean * cells_per_shift * n_dist).round() as u128;
                let expected = (total_pts as u128) * (n_dist as u128);
                let diff = if actual > expected { actual - expected } else { expected - actual };
                assert!(diff <= (1 << D) as u128,
                    "ND level {} conservation: actual={}, expected={}, D={}",
                    l, actual, expected, D);
            }
        }
    }

    let final_out: Vec<LevelStatsND> = out.into_iter()
        .enumerate()
        .map(|(l, o)| o.unwrap_or(LevelStatsND {
            n_cells_total: 1 << (D * l),
            mean: 0.0, var: 0.0, dvar: 0.0,
        }))
        .collect();
    (final_out, n_distinct)
}

// ============================================================================
// Generic D-dim windowed PMF: P_N(V) at arbitrary integer cube-window sides
// ============================================================================
//
// Given a binned grid c_grid[M^D] and a list of window sides {k_1, ..., k_n},
// for each k compute the count in every M^D cube of side k via D sequential
// 1D periodic sliding-sum passes, then histogram into a PMF.
//
// Cost per window: D * M^D adds (sliding) + M^D histogram updates. For D=3,
// M=256, single window is ~16M arithmetic ops + 16M histogram increments,
// roughly 200-400 ms depending on memory bandwidth.
//
// Memory: 2 * M^D ping-pong buffers (u64), reused across windows.

#[derive(Clone, Debug)]
pub struct PmfWindow {
    /// Cube side in fine-grid units.
    pub window_side: usize,
    /// Volume in tree-coord^D units (= window_side^D, with the convention that
    /// the fine grid has cell side 1 in tree-coord at L_max; rescale externally
    /// for physical units).
    pub volume_tree: f64,
    /// Total (cell, shift) entries contributing.
    pub n_total: u64,
    pub mean: f64,
    pub var: f64,
    /// Standardized 3rd moment (mu_3 / sigma^3).
    pub skew: f64,
    /// Standardized 4th moment (mu_4 / sigma^4). Excess = kurt - 3.
    pub kurt: f64,
    /// `histogram[k]` = number of windows containing exactly k points.
    pub histogram: Vec<u64>,
}

/// Generate a deduplicated, sorted list of integer window sides spanning
/// [side_min, side_max] (inclusive) with approximately `points_per_decade_V`
/// points per decade in volume V = side^D.
///
/// `points_per_decade_V` is in V-decades; the corresponding spacing in side
/// is 10^(1 / (D * points_per_decade_V)).
pub fn log_spaced_window_sides(
    side_min: usize,
    side_max: usize,
    dim: usize,
    points_per_decade_v: f64,
) -> Vec<usize> {
    assert!(side_min >= 1 && side_max >= side_min);
    assert!(dim >= 1);
    assert!(points_per_decade_v > 0.0);
    let log_step_side = 1.0 / (dim as f64 * points_per_decade_v);
    let mut sides: Vec<usize> = Vec::new();
    let log_min = (side_min as f64).log10();
    let log_max = (side_max as f64).log10();
    let n_steps = ((log_max - log_min) / log_step_side).ceil() as usize + 1;
    for i in 0..=n_steps {
        let log_v = log_min + i as f64 * log_step_side;
        let s = 10f64.powf(log_v).round() as usize;
        let s = s.clamp(side_min, side_max);
        sides.push(s);
    }
    sides.sort_unstable();
    sides.dedup();
    sides
}

/// Compute per-window cell-count PMFs for arbitrary integer cube-window sides.
///
/// `pts` are u16 coords in the unit periodic box (mapped to fine-grid 2^L_max
/// per side). `s_subshift` adds extra refinement to the fine grid; `periodic`
/// controls boundary conditions. `window_sides` lists the cube sides (in fine-
/// grid units) for which to compute PMFs; the function deduplicates and sorts
/// them internally.
///
/// For periodic=false, windows that wrap around the box edge are excluded
/// from the histogram for that window size (n_total decreases accordingly).
pub fn cascade_with_pmf_windows<const D: usize>(
    pts: &[[u16; D]],
    l_max: usize,
    s_subshift: usize,
    periodic: bool,
    window_sides: &[usize],
) -> Vec<PmfWindow> {
    let (c_grid_u32, m) = bin_nd::<D>(pts, l_max, s_subshift);
    let total_cells = m.pow(D as u32);
    // Promote to u64 to avoid overflow during sliding sums for large windows.
    let c_grid: Vec<u64> = c_grid_u32.iter().map(|&x| x as u64).collect();

    // Deduplicate and sort window sides; clamp to valid range.
    let mut sides: Vec<usize> = window_sides.iter().copied()
        .filter(|&k| k >= 1 && k <= m)
        .collect();
    sides.sort_unstable();
    sides.dedup();

    // Two ping-pong buffers reused across windows.
    let mut buf_a: Vec<u64> = vec![0; total_cells];
    let mut buf_b: Vec<u64> = vec![0; total_cells];

    let mut out: Vec<PmfWindow> = Vec::with_capacity(sides.len());

    for &k in &sides {
        // Special case k==1: window count IS the binned grid value.
        if k == 1 {
            let pmf = build_pmf_from_buffer::<D>(&c_grid, m, k, periodic);
            out.push(pmf);
            continue;
        }

        // Three sequential 1D sliding-sum passes, one per axis.
        // Pass d: input has the windowed sum along axes 0..d already applied;
        // output applies the window along axis d.
        // First pass reads c_grid -> buf_a; subsequent passes alternate.
        sliding_sum_1d_axis::<D>(&c_grid, &mut buf_a, m, k, 0, periodic);
        let mut src_is_a = true;
        for axis in 1..D {
            if src_is_a {
                sliding_sum_1d_axis::<D>(&buf_a, &mut buf_b, m, k, axis, periodic);
            } else {
                sliding_sum_1d_axis::<D>(&buf_b, &mut buf_a, m, k, axis, periodic);
            }
            src_is_a = !src_is_a;
        }
        let final_buf = if src_is_a { &buf_a } else { &buf_b };

        let pmf = build_pmf_from_buffer::<D>(final_buf, m, k, periodic);
        out.push(pmf);
    }

    out
}

/// Apply a 1D periodic sliding-sum of width `k` along the given `axis` of an
/// M^D grid. Input and output are flat arrays of size M^D, axis-major layout
/// (axis 0 contiguous, axis 1 stride M, axis d stride M^d).
///
/// For periodic=false, output entries at positions where the window would
/// wrap are set to a sentinel value (u64::MAX) to be filtered out downstream.
fn sliding_sum_1d_axis<const D: usize>(
    src: &[u64],
    dst: &mut [u64],
    m: usize,
    k: usize,
    axis: usize,
    periodic: bool,
) {
    // Stride along the chosen axis is m^axis.
    let stride: usize = m.pow(axis as u32);
    let total = m.pow(D as u32);

    // We need to sweep all "rows" along this axis. A row is characterized by
    // fixing all axes != axis. There are total / m such rows.
    //
    // For axis = 0: rows are contiguous chunks of m elements -> simple.
    // For axis > 0: rows are strided by `stride`, with m elements within each row.
    //
    // We iterate over the index of the row in a flat enumeration.
    let n_rows = total / m;
    for row_idx in 0..n_rows {
        // Decompose row_idx into multi-axis coordinates excluding `axis`,
        // then build the base flat index using flat-array strides (m^d).
        let mut base = 0usize;
        let mut tmp = row_idx;
        for d in 0..D {
            if d == axis { continue; }
            let coord = tmp % m;
            tmp /= m;
            base += coord * m.pow(d as u32);
        }

        // Sliding sum along axis. Position i along the axis means flat index
        // base + i * stride.
        // Initial sum over [0, k):
        let mut acc: u64 = 0;
        for j in 0..k {
            acc += src[base + j * stride];
        }
        // Position 0:
        dst[base] = acc;
        // Position i for i = 1..m: subtract element at i-1, add at (i+k-1) mod m.
        for i in 1..m {
            let leaving = src[base + (i - 1) * stride];
            let entering_pos = i + k - 1;
            if entering_pos < m {
                acc = acc + src[base + entering_pos * stride] - leaving;
            } else if periodic {
                let wrap = entering_pos - m;
                acc = acc + src[base + wrap * stride] - leaving;
            } else {
                // Non-periodic: window would wrap, mark as sentinel.
                // Use u64::MAX to flag; downstream filters these out.
                acc = u64::MAX;
            }
            dst[base + i * stride] = acc;
        }
    }
}

fn build_pmf_from_buffer<const D: usize>(
    buf: &[u64],
    m: usize,
    window_side: usize,
    periodic: bool,
) -> PmfWindow {
    let total = m.pow(D as u32);
    let mut max_n: u64 = 0;
    let mut n_valid: u64 = 0;
    for &v in buf {
        if v != u64::MAX {
            if v > max_n { max_n = v; }
            n_valid += 1;
        }
    }
    // Cap histogram size to avoid pathological cases (e.g., all points in one
    // cell at the coarsest level).
    let n_bins = (max_n + 1).min(2_000_000) as usize;
    let mut hist = vec![0u64; n_bins];
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let mut sum4 = 0.0f64;
    for &v in buf {
        if v == u64::MAX { continue; }
        let idx = (v as usize).min(n_bins - 1);
        hist[idx] += 1;
        let x = v as f64;
        sum += x; sum2 += x*x; sum3 += x*x*x; sum4 += x*x*x*x;
    }
    let n = n_valid as f64;
    let mean = if n > 0.0 { sum / n } else { 0.0 };
    let var = if n > 0.0 { (sum2 / n - mean*mean).max(0.0) } else { 0.0 };
    let std = var.sqrt();
    let m3 = if n > 0.0 { sum3 / n - 3.0 * mean * sum2 / n + 2.0 * mean.powi(3) } else { 0.0 };
    let m4 = if n > 0.0 { sum4 / n - 4.0 * mean * sum3 / n + 6.0 * mean.powi(2) * sum2 / n - 3.0 * mean.powi(4) } else { 0.0 };
    let skew = if std > 1e-12 { m3 / std.powi(3) } else { 0.0 };
    let kurt = if std > 1e-12 { m4 / std.powi(4) } else { 0.0 };

    // For periodic boundaries every cell is valid (n_valid == total).
    // For non-periodic we report only the windows that fit entirely in the box.
    let _ = (periodic, total);   // silence unused

    let volume_tree = (window_side as f64).powi(D as i32);
    PmfWindow {
        window_side,
        volume_tree,
        n_total: n_valid,
        mean, var, skew, kurt,
        histogram: hist,
    }
}

// =============================================================================
// Paired PMF windows: data + randoms via cube-window sliding sums.
//
// Same machinery as cascade_with_pmf_windows but with TWO grids — one for
// the data catalog (with optional weights), one for the randoms catalog.
// For each cube window of integer side k, computes the W_d and W_r summed
// inside each window position, then per cell forms the density contrast
// δ(c) = W_d(c) / (α · W_r(c)) − 1 and accumulates W_r-weighted moments
// and an optional log-spaced PDF of (1+δ).
//
// This is the survey-aware version of P_N(V) at non-dyadic cube sides:
// the randoms encode the survey selection function (footprint, fiber
// completeness, dust corrections, etc.), and the moments / PDF are reported
// natively in terms of δ rather than raw counts.
//
// Cells with W_r ≤ w_r_min are excluded as outside-footprint (their W_d
// contributes to a separate diagnostic — sum_w_d_outside per window).

/// Per-window density-field statistics, parallel to DensityFieldStats but
/// keyed by integer window side rather than dyadic level.
#[derive(Clone, Debug)]
pub struct PairedPmfWindow {
    /// Cube side in fine-grid units.
    pub window_side: usize,
    /// Volume in fine-grid^D units.
    pub volume_fine: f64,
    /// Total number of valid (non-wrapping) window positions of this size.
    /// For periodic=true this is m^D. For periodic=false it is the number
    /// of windows that fit entirely inside the box.
    pub n_windows_total: u64,
    /// Number of windows whose W_r > w_r_min (active = inside footprint).
    pub n_windows_active: u64,
    /// Total W_r across active windows (= effective surveyed volume in
    /// W_r units, summed over all valid window positions of this size).
    pub sum_w_r_active: f64,
    /// W_r-weighted central moments of δ.
    pub mean_delta: f64,
    pub var_delta: f64,
    pub m3_delta: f64,
    pub m4_delta: f64,
    pub min_delta: f64,
    pub max_delta: f64,
    /// W_r-weighted reduced skewness S_3 = m3 / var^2.
    pub s3_delta: f64,
    /// Diagnostic: number of windows with n_d > 0 but W_r ≤ w_r_min
    /// (data outside the random catalog's footprint at this scale).
    pub n_windows_data_outside: u64,
    /// Sum of W_d across those outside-footprint windows.
    pub sum_w_d_outside: f64,
    /// Histogram bin edges in log10(1+δ). `hist_density[k]` is the
    /// W_r-fraction in bin k. Empty if `hist_bins == 0`.
    pub hist_bin_edges: Vec<f64>,
    pub hist_density: Vec<f64>,
    pub hist_underflow_w_r: f64,
    pub hist_overflow_w_r: f64,
}

/// Configuration shared with BitVecCascadePair::analyze_field_stats.
#[derive(Clone, Debug)]
pub struct PairedPmfConfig {
    pub w_r_min: f64,
    pub hist_bins: usize,
    pub hist_log_min: f64,
    pub hist_log_max: f64,
}

impl Default for PairedPmfConfig {
    fn default() -> Self {
        Self { w_r_min: 0.0, hist_bins: 50, hist_log_min: -3.0, hist_log_max: 3.0 }
    }
}

/// Bin (points, optional weights) into an M^D f64 grid.
/// `weights` must be None or have the same length as `pts`.
pub fn bin_nd_weighted_f64<const D: usize>(
    pts: &[[u16; D]],
    weights: Option<&[f64]>,
    l_max: usize,
    s_subshift: usize,
) -> (Vec<f64>, usize) {
    assert!(l_max + s_subshift <= 16, "fine grid would exceed u16 resolution");
    if let Some(w) = weights {
        assert_eq!(w.len(), pts.len(), "weights length must match pts");
    }
    let m_bits = l_max + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;
    let total = m.pow(D as u32);
    let mut c = vec![0.0f64; total];
    for (i, p) in pts.iter().enumerate() {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            let a = (p[d] >> shift) as usize;
            idx += a * stride;
            stride *= m;
        }
        let w = weights.map(|wv| wv[i]).unwrap_or(1.0);
        c[idx] += w;
    }
    (c, m)
}

/// f64 1D sliding-sum, structurally identical to sliding_sum_1d_axis but
/// for f64 grids. For non-periodic boundaries, NaN marks invalid windows.
fn sliding_sum_1d_axis_f64<const D: usize>(
    src: &[f64],
    dst: &mut [f64],
    m: usize,
    k: usize,
    axis: usize,
    periodic: bool,
) {
    let stride: usize = m.pow(axis as u32);
    let total = m.pow(D as u32);
    let n_rows = total / m;
    for row_idx in 0..n_rows {
        let mut base = 0usize;
        let mut tmp = row_idx;
        for d in 0..D {
            if d == axis { continue; }
            let coord = tmp % m;
            tmp /= m;
            base += coord * m.pow(d as u32);
        }
        let mut acc: f64 = 0.0;
        for j in 0..k {
            acc += src[base + j * stride];
        }
        dst[base] = acc;
        for i in 1..m {
            let leaving = src[base + (i - 1) * stride];
            let entering_pos = i + k - 1;
            if entering_pos < m {
                acc = acc + src[base + entering_pos * stride] - leaving;
            } else if periodic {
                let wrap = entering_pos - m;
                acc = acc + src[base + wrap * stride] - leaving;
            } else {
                acc = f64::NAN;
            }
            dst[base + i * stride] = acc;
        }
    }
}

/// Compute paired (data + randoms) per-cube-window density-field statistics
/// for arbitrary integer window sides.
///
/// `pts_d`, `pts_r` are u16 coords on the unit periodic box, mapped to the
/// fine grid M = 2^(l_max + s_subshift).
///
/// `weights_d`, `weights_r` are optional per-point weights (DESI-style
/// FKP/completeness/imaging-systematics weights).
///
/// `window_sides` lists the cube sides (in fine-grid units) to compute
/// statistics for. Deduplicated and sorted internally. Note: with `s_subshift > 0`,
/// integer fine-grid sides correspond to non-dyadic fractions of the box.
///
/// `cfg.w_r_min` excludes windows with summed random weight ≤ this from the
/// moments (outside-footprint diagnostic counts them separately).
///
/// For `periodic=false`, windows that would wrap around the box are excluded
/// from both the moments and the diagnostic.
pub fn cascade_pmf_windows_with_randoms<const D: usize>(
    pts_d: &[[u16; D]],
    weights_d: Option<&[f64]>,
    pts_r: &[[u16; D]],
    weights_r: Option<&[f64]>,
    l_max: usize,
    s_subshift: usize,
    periodic: bool,
    window_sides: &[usize],
    cfg: &PairedPmfConfig,
) -> Vec<PairedPmfWindow> {
    let (cw_d, m) = bin_nd_weighted_f64::<D>(pts_d, weights_d, l_max, s_subshift);
    let (cw_r, m_r) = bin_nd_weighted_f64::<D>(pts_r, weights_r, l_max, s_subshift);
    assert_eq!(m, m_r, "data and randoms must use the same grid size");
    let total_cells = m.pow(D as u32);

    // Global α = ΣW_d / ΣW_r. If the random catalog is empty, return empty
    // results rather than NaN.
    let total_w_d: f64 = cw_d.iter().sum();
    let total_w_r: f64 = cw_r.iter().sum();
    if total_w_r <= 0.0 {
        // Build empty windows for each requested side
        return window_sides.iter().copied().filter(|&k| k >= 1 && k <= m)
            .map(|k| empty_paired_window::<D>(k, cfg)).collect();
    }
    let alpha = total_w_d / total_w_r;

    // Deduplicate and sort window sides
    let mut sides: Vec<usize> = window_sides.iter().copied()
        .filter(|&k| k >= 1 && k <= m)
        .collect();
    sides.sort_unstable();
    sides.dedup();

    // Two ping-pong buffer pairs reused across windows (one pair per catalog).
    let mut buf_d_a: Vec<f64> = vec![0.0; total_cells];
    let mut buf_d_b: Vec<f64> = vec![0.0; total_cells];
    let mut buf_r_a: Vec<f64> = vec![0.0; total_cells];
    let mut buf_r_b: Vec<f64> = vec![0.0; total_cells];

    let mut out: Vec<PairedPmfWindow> = Vec::with_capacity(sides.len());

    for &k in &sides {
        // Compute windowed grids of W_d and W_r at side k.
        // Special case k==1: window count IS the bin value.
        let (final_d, final_r): (&[f64], &[f64]) = if k == 1 {
            (&cw_d, &cw_r)
        } else {
            sliding_sum_1d_axis_f64::<D>(&cw_d, &mut buf_d_a, m, k, 0, periodic);
            sliding_sum_1d_axis_f64::<D>(&cw_r, &mut buf_r_a, m, k, 0, periodic);
            let mut src_is_a = true;
            for axis in 1..D {
                if src_is_a {
                    sliding_sum_1d_axis_f64::<D>(&buf_d_a, &mut buf_d_b, m, k, axis, periodic);
                    sliding_sum_1d_axis_f64::<D>(&buf_r_a, &mut buf_r_b, m, k, axis, periodic);
                } else {
                    sliding_sum_1d_axis_f64::<D>(&buf_d_b, &mut buf_d_a, m, k, axis, periodic);
                    sliding_sum_1d_axis_f64::<D>(&buf_r_b, &mut buf_r_a, m, k, axis, periodic);
                }
                src_is_a = !src_is_a;
            }
            if src_is_a { (&buf_d_a[..], &buf_r_a[..]) } else { (&buf_d_b[..], &buf_r_b[..]) }
        };

        out.push(build_paired_pmf_window::<D>(final_d, final_r, m, k, alpha, cfg));
    }

    out
}

/// Build per-window stats from windowed W_d, W_r grids.
fn build_paired_pmf_window<const D: usize>(
    win_d: &[f64],
    win_r: &[f64],
    m: usize,
    k: usize,
    alpha: f64,
    cfg: &PairedPmfConfig,
) -> PairedPmfWindow {
    let total = m.pow(D as u32);
    debug_assert_eq!(win_d.len(), total);
    debug_assert_eq!(win_r.len(), total);

    let mut sum_w = 0.0f64;
    let mut sum_w_d1 = 0.0f64;
    let mut sum_w_d2 = 0.0f64;
    let mut sum_w_d3 = 0.0f64;
    let mut sum_w_d4 = 0.0f64;
    let mut n_total: u64 = 0;
    let mut n_active: u64 = 0;
    let mut min_delta = f64::INFINITY;
    let mut max_delta = f64::NEG_INFINITY;
    let mut n_outside: u64 = 0;
    let mut sum_w_d_outside = 0.0f64;

    let mut hist = vec![0.0f64; cfg.hist_bins];
    let mut hist_under = 0.0f64;
    let mut hist_over = 0.0f64;

    for i in 0..total {
        let w_d = win_d[i];
        let w_r = win_r[i];
        // Skip non-periodic invalid windows (NaN marker from sliding_sum).
        if !w_d.is_finite() || !w_r.is_finite() { continue; }
        n_total += 1;
        if w_r > cfg.w_r_min {
            let delta = if w_r > 0.0 { w_d / (alpha * w_r) - 1.0 } else { -1.0 };
            sum_w += w_r;
            sum_w_d1 += w_r * delta;
            let d2 = delta * delta;
            sum_w_d2 += w_r * d2;
            sum_w_d3 += w_r * d2 * delta;
            sum_w_d4 += w_r * d2 * d2;
            n_active += 1;
            if delta < min_delta { min_delta = delta; }
            if delta > max_delta { max_delta = delta; }
            if cfg.hist_bins > 0 {
                let one_plus = 1.0 + delta;
                if one_plus > 0.0 {
                    let log_v = one_plus.log10();
                    if log_v < cfg.hist_log_min { hist_under += w_r; }
                    else if log_v >= cfg.hist_log_max { hist_over += w_r; }
                    else {
                        let frac = (log_v - cfg.hist_log_min)
                            / (cfg.hist_log_max - cfg.hist_log_min);
                        let bin = (frac * cfg.hist_bins as f64) as usize;
                        let bin = bin.min(cfg.hist_bins - 1);
                        hist[bin] += w_r;
                    }
                } else { hist_under += w_r; }
            }
        } else if w_d > 0.0 {
            n_outside += 1;
            sum_w_d_outside += w_d;
        }
    }

    let (mean, var, m3c, m4c) = if sum_w > 0.0 {
        let m1 = sum_w_d1 / sum_w;
        let m2_raw = sum_w_d2 / sum_w;
        let m3_raw = sum_w_d3 / sum_w;
        let m4_raw = sum_w_d4 / sum_w;
        let var = (m2_raw - m1 * m1).max(0.0);
        let m3c = m3_raw - 3.0 * m1 * m2_raw + 2.0 * m1.powi(3);
        let m4c = m4_raw - 4.0 * m1 * m3_raw + 6.0 * m1.powi(2) * m2_raw - 3.0 * m1.powi(4);
        (m1, var, m3c, m4c)
    } else { (0.0, 0.0, 0.0, 0.0) };
    let s3 = if var > 0.0 { m3c / (var * var) } else { 0.0 };

    let edges: Vec<f64> = if cfg.hist_bins > 0 {
        (0..=cfg.hist_bins).map(|i| {
            cfg.hist_log_min
                + (cfg.hist_log_max - cfg.hist_log_min) * (i as f64) / (cfg.hist_bins as f64)
        }).collect()
    } else { Vec::new() };
    let hist_density: Vec<f64> = if sum_w > 0.0 && cfg.hist_bins > 0 {
        hist.iter().map(|&c| c / sum_w).collect()
    } else { vec![0.0; cfg.hist_bins] };
    let hist_under_n = if sum_w > 0.0 { hist_under / sum_w } else { 0.0 };
    let hist_over_n = if sum_w > 0.0 { hist_over / sum_w } else { 0.0 };

    PairedPmfWindow {
        window_side: k,
        volume_fine: (k as f64).powi(D as i32),
        n_windows_total: n_total,
        n_windows_active: n_active,
        sum_w_r_active: sum_w,
        mean_delta: mean,
        var_delta: var,
        m3_delta: m3c,
        m4_delta: m4c,
        min_delta: if n_active > 0 { min_delta } else { 0.0 },
        max_delta: if n_active > 0 { max_delta } else { 0.0 },
        s3_delta: s3,
        n_windows_data_outside: n_outside,
        sum_w_d_outside,
        hist_bin_edges: edges,
        hist_density,
        hist_underflow_w_r: hist_under_n,
        hist_overflow_w_r: hist_over_n,
    }
}

fn empty_paired_window<const D: usize>(k: usize, cfg: &PairedPmfConfig) -> PairedPmfWindow {
    let edges: Vec<f64> = if cfg.hist_bins > 0 {
        (0..=cfg.hist_bins).map(|i| {
            cfg.hist_log_min
                + (cfg.hist_log_max - cfg.hist_log_min) * (i as f64) / (cfg.hist_bins as f64)
        }).collect()
    } else { Vec::new() };
    PairedPmfWindow {
        window_side: k,
        volume_fine: (k as f64).powi(D as i32),
        n_windows_total: 0,
        n_windows_active: 0, sum_w_r_active: 0.0,
        mean_delta: 0.0, var_delta: 0.0, m3_delta: 0.0, m4_delta: 0.0,
        min_delta: 0.0, max_delta: 0.0, s3_delta: 0.0,
        n_windows_data_outside: 0, sum_w_d_outside: 0.0,
        hist_bin_edges: edges,
        hist_density: vec![0.0; cfg.hist_bins],
        hist_underflow_w_r: 0.0, hist_overflow_w_r: 0.0,
    }
}

#[cfg(test)]
mod paired_pmf_tests {
    use super::*;

    fn sm64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn make_pts_3d(n: usize, seed: u64) -> Vec<[u16; 3]> {
        let mut s = seed;
        (0..n).map(|_| {
            [(sm64(&mut s) >> 48) as u16,
             (sm64(&mut s) >> 48) as u16,
             (sm64(&mut s) >> 48) as u16]
        }).collect()
    }

    #[test]
    fn paired_pmf_data_equals_randoms_zero_delta() {
        // If data == randoms, δ = 0 everywhere, all moments zero.
        let pts = make_pts_3d(500, 1234);
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts, None, &pts, None, 5, 0, true, &[1, 2, 3, 5, 7], &cfg);
        for s in &stats {
            assert!(s.mean_delta.abs() < 1e-12,
                "side {}: mean_delta = {}", s.window_side, s.mean_delta);
            assert!(s.var_delta < 1e-20,
                "side {}: var_delta = {}", s.window_side, s.var_delta);
        }
    }

    #[test]
    fn paired_pmf_periodic_full_box_window_zero_delta() {
        // A window equal to the whole box (k = m) covers everything.
        // For periodic=true, every "window" sees ΣW_d and ΣW_r, so δ ≡ 0.
        let pts_d = make_pts_3d(200, 11);
        let pts_r = make_pts_3d(800, 22);
        let l_max = 4;
        let s_subshift = 0;
        let m = 1usize << (l_max + s_subshift);
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts_d, None, &pts_r, None, l_max, s_subshift, true, &[m], &cfg);
        let s = &stats[0];
        assert!(s.mean_delta.abs() < 1e-12,
            "full-box window mean_delta = {}", s.mean_delta);
        assert!(s.var_delta < 1e-20,
            "full-box window var_delta = {}", s.var_delta);
    }

    #[test]
    fn paired_pmf_non_dyadic_sides_run() {
        // Confirm we can request non-power-of-2 sides without crashing,
        // and that the moments are sensible.
        let pts_d = make_pts_3d(800, 3);
        let pts_r = make_pts_3d(2400, 4);
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let sides = vec![1, 2, 3, 5, 7, 11, 13];  // various integer sides
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts_d, None, &pts_r, None, 4, 0, true, &sides, &cfg);
        assert_eq!(stats.len(), sides.len());
        for (s, &k) in stats.iter().zip(sides.iter()) {
            assert_eq!(s.window_side, k);
            assert!(s.var_delta.is_finite(), "side {}: var = {}", k, s.var_delta);
            assert!(s.var_delta >= 0.0);
        }
    }

    #[test]
    fn paired_pmf_subshift_gives_finer_sides() {
        // With s_subshift = 1, fine grid is 2x finer, so integer sides give
        // non-dyadic fractions of the box.
        let pts_d = make_pts_3d(500, 7);
        let pts_r = make_pts_3d(1500, 8);
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        // l_max = 3, s_sub = 1 -> m = 16
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts_d, None, &pts_r, None, 3, 1, true, &[1, 2, 3, 4, 5, 8, 16], &cfg);
        // m=16, last entry k=16 = full box, δ = 0
        assert!(stats[stats.len() - 1].var_delta < 1e-20);
        // All other windows should give finite var
        for s in &stats[..stats.len() - 1] {
            assert!(s.var_delta.is_finite());
            assert!(s.var_delta >= 0.0);
        }
    }

    #[test]
    fn paired_pmf_data_outside_footprint_diagnostic() {
        // Construct: randoms only in lower half-cube, data has a small
        // cluster in upper half. At a moderate window size (k=4) the
        // random density is high enough that footprint cells reliably have
        // randoms, so the diagnostic cleanly counts only the contamination.
        let mut s = 12345u64;
        let mut sm = || sm64(&mut s);
        let randoms: Vec<[u16; 3]> = (0..30000).map(|_| [
            (sm() >> 49) as u16,                 // top bit always 0 -> lower half
            (sm() >> 48) as u16,
            (sm() >> 48) as u16,
        ]).collect();
        let mut data: Vec<[u16; 3]> = (0..200).map(|_| [
            (sm() >> 49) as u16,                 // also lower half (in footprint)
            (sm() >> 48) as u16,
            (sm() >> 48) as u16,
        ]).collect();
        // Add 50 contamination points in upper half
        let upper_half_offset = 1u16 << 15;
        for _ in 0..50 {
            data.push([
                upper_half_offset | ((sm() >> 49) as u16),
                (sm() >> 48) as u16,
                (sm() >> 48) as u16,
            ]);
        }
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &data, None, &randoms, None, 4, 0, true, &[1, 2, 4, 8], &cfg);
        // At least one window size should report data outside footprint.
        let any_outside: bool = stats.iter().any(|s| s.n_windows_data_outside > 0);
        assert!(any_outside, "expected some outside-footprint windows");
        // Each contamination point appears in some subset of the 4^3 = 64
        // overlapping windows (the actual count varies per point because
        // points near the footprint boundary get partial overlap). What we
        // CAN verify: the diagnostic counts the contamination cleanly,
        // without spurious in-footprint contributions.
        //
        // Lower bound: each contamination point appears in at least 1 window
        //   that's entirely in the upper half cube (the (x_d, y_d, z_d)-anchored
        //   one with origin (x_d, y_d, z_d)). So sum_w_d_outside ≥ 50.
        // Upper bound: each contamination point appears in at most 4^3 = 64
        //   windows total, all of which could be outside if the point is
        //   deep enough in the upper half. So sum_w_d_outside ≤ 50 * 64.
        // In-footprint data should NOT contribute: their windows always have
        //   randoms because the lower-half catalog is dense.
        let k4 = stats.iter().find(|s| s.window_side == 4).unwrap();
        let lower = 50.0;
        let upper = 50.0 * 4.0_f64.powi(3);
        assert!(k4.sum_w_d_outside >= lower && k4.sum_w_d_outside <= upper,
            "k=4 sum_w_d_outside = {} (expected in [{}, {}])",
            k4.sum_w_d_outside, lower, upper);
        // It should be MUCH more than 50 (sliding-window multiplicity).
        assert!(k4.sum_w_d_outside > 200.0,
            "k=4 sum_w_d_outside = {} (expected >200 from sliding-window multiplicity)",
            k4.sum_w_d_outside);
    }

    #[test]
    fn paired_pmf_with_weights() {
        // Per-point weights propagate through the windowed sums.
        let pts_d = make_pts_3d(300, 100);
        let pts_r = make_pts_3d(1200, 200);
        // Half the data gets weight 2, half weight 1. Total W_d = 1.5 * N_d.
        let n_d = pts_d.len();
        let w_d: Vec<f64> = (0..n_d).map(|i| if i % 2 == 0 { 2.0 } else { 1.0 }).collect();
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts_d, Some(&w_d), &pts_r, None, 4, 0, true, &[1, 2, 4, 16], &cfg);
        // At full-box window, mean_delta = 0 by α normalization.
        let full_box = stats.iter().find(|s| s.window_side == 16).unwrap();
        assert!(full_box.mean_delta.abs() < 1e-12,
            "weighted full-box mean_delta = {}", full_box.mean_delta);
    }

    #[test]
    fn paired_pmf_matches_bitvec_field_stats_at_dyadic_scales() {
        // The sliding-window paired PMF and the bit-vector pair cascade's
        // analyze_field_stats compute related but DIFFERENT observables:
        //
        //   - sliding-window version visits ALL m^D possible window origins
        //     (modulo wrap or not), so totals like sum_w_r_active are
        //     summed over m^D windows of side k.
        //   - cascade reports per-lattice-aligned-cell moments — at
        //     level l there are 2^(D*l) cells, none overlapping.
        //
        // For a full-box window (k=m, periodic), every sliding origin
        // gives the same window, so PER-WINDOW moments must agree:
        // mean and var are intensive quantities (averages weighted by W_r);
        // they should be IDENTICAL to the cascade's level-0 result.
        use crate::coord_range::{CoordRange, TrimmedPoints};
        use crate::hier_bitvec_pair::{BitVecCascadePair, FieldStatsConfig};

        let pts_d_u16 = make_pts_3d(500, 333);
        let pts_r_u16 = make_pts_3d(1500, 444);

        // Full-box window via sliding-window: side = m (= 16 with l_max=4).
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let l_max = 4;
        let m = 1usize << l_max;
        let stats_pmf = cascade_pmf_windows_with_randoms::<3>(
            &pts_d_u16, None, &pts_r_u16, None, l_max, 0, true, &[m], &cfg);

        // Same data via bit-vector pair cascade -> level 0 (root).
        let pts_d_u64: Vec<[u64; 3]> = pts_d_u16.iter().map(|p|
            [p[0] as u64, p[1] as u64, p[2] as u64]).collect();
        let pts_r_u64: Vec<[u64; 3]> = pts_r_u16.iter().map(|p|
            [p[0] as u64, p[1] as u64, p[2] as u64]).collect();
        let range = CoordRange::analyze_pair(&pts_d_u64, &pts_r_u64);
        let td = TrimmedPoints::from_points_with_range(pts_d_u64, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r_u64, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let bvcfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats_bv = pair.analyze_field_stats(&bvcfg);

        // Intensive quantities (per-window mean/var) must agree.
        let pmf = &stats_pmf[0];
        let bv0 = &stats_bv[0];
        assert!(pmf.mean_delta.abs() < 1e-12, "pmf root mean = {}", pmf.mean_delta);
        assert!(bv0.mean_delta.abs() < 1e-12, "bv root mean = {}", bv0.mean_delta);
        assert!(pmf.var_delta < 1e-20, "pmf root var = {}", pmf.var_delta);
        assert!(bv0.var_delta < 1e-20, "bv root var = {}", bv0.var_delta);
        // Number of windows: pmf is shift-averaged so visits m^D origins,
        // cascade is lattice-aligned so visits 1 cell at level 0.
        assert_eq!(pmf.n_windows_active as usize, m.pow(3));
        assert_eq!(bv0.n_cells_active, 1);
        // Extensive quantity sum_w_r_active scales accordingly:
        // pmf visits m^D windows each with full ΣW_r = 1500 → m^D * 1500.
        // cascade visits 1 cell with full ΣW_r = 1500.
        assert_eq!(pmf.sum_w_r_active as u64, m.pow(3) as u64 * 1500);
        assert_eq!(bv0.sum_w_r_active as u64, 1500);
    }

    #[test]
    fn paired_pmf_non_periodic_excludes_wrapping_windows() {
        // For periodic=false, windows that wrap should be excluded entirely
        // (sentinel NaN in the sliding sum, dropped from the moment loop).
        let pts_d = make_pts_3d(200, 1);
        let pts_r = make_pts_3d(800, 2);
        let cfg = PairedPmfConfig { hist_bins: 0, ..Default::default() };
        let stats = cascade_pmf_windows_with_randoms::<3>(
            &pts_d, None, &pts_r, None, 4, 0, false, &[1, 4, 8, 16], &cfg);
        // For k=1, every cell is a valid (non-wrapping) window: count = m^D.
        let k1 = &stats[0];
        let m: usize = 16;
        assert_eq!(k1.n_windows_total as usize, m.pow(3),
            "k=1 should have m^D total valid windows; got {}", k1.n_windows_total);
        // For k=m, only the (0,0,0) origin yields a non-wrapping window.
        let km = stats.iter().find(|s| s.window_side == m).unwrap();
        assert_eq!(km.n_windows_total, 1,
            "k=m non-periodic should have exactly 1 valid window");
        // For k=8 (half the box), 9^3 = 729 valid window origins.
        let k8 = stats.iter().find(|s| s.window_side == 8).unwrap();
        assert_eq!(k8.n_windows_total, 9u64.pow(3),
            "k=8 non-periodic: expected 9^3 = 729 valid windows; got {}", k8.n_windows_total);
    }
}
