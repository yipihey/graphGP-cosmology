// Hierarchical all-shifts cascade.
//
// Replaces the SAT approach with a level-by-level 2x2 reduction. At each
// level we maintain a buffer D_l of size M^2 indexed by (shift, cell):
//
//     D_l[s_y, s_x, c_y, c_x]  with  s in [0, h_l)^2,  c in [0, M/h_l)^2.
//
// where h_l = M / 2^l. Going from level l+1 to level l: each level-l cell at
// shift (s_y, s_x) is the sum of four level-(l+1) cells at shift
// (s_y mod h_{l+1}, s_x mod h_{l+1}) and adjacent cell indices that depend on
// the high bit of s_y / s_x (whether the level-l shift "crosses" a
// level-(l+1) boundary).
//
// Concretely, with h' = h_{l+1} = h_l / 2:
//   if s_y < h':   first child y-index = 2 c_y     (no boundary cross)
//   if s_y >= h':  first child y-index = 2 c_y + 1 (cross by one h')
// and second child y-index = first + 1, modulo n_per_axis_{l+1}.
// Same for x.
//
// Per level: O(M^2) adds. Total: O(M^2 * L_MAX). No SAT, no wrap splits, no
// multiplications. We compute moments (sum, sum^2, sibling sum-of-squared
// deviations) on the fly while building D_l, so output is obtained in one pass.

use crate::{LevelStats, L_MAX, N_LEVELS};

/// Bin points into an M x M grid where M = 2^(L_MAX + s_subshift).
/// This is the level-L_MAX shifted-count buffer with shift = (0, 0) per shift cell;
/// equivalently, it is D_(L_MAX + s_subshift) at the very finest virtual level.
pub fn bin_to_fine_grid(pts: &[(u16, u16)], s_subshift: usize) -> (Vec<u32>, usize) {
    assert!(L_MAX + s_subshift <= 16, "fine grid would exceed u16 resolution");
    let m_bits = L_MAX + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;
    let mut c = vec![0u32; m * m];
    for &(x, y) in pts {
        let ix = (x >> shift) as usize;
        let iy = (y >> shift) as usize;
        c[iy * m + ix] += 1;
    }
    (c, m)
}

/// Hierarchical all-shifts cascade.
///
/// Returns per-level (mean, var, dvar) averaged over ALL distinct shifts at
/// each level, and the per-level n_distinct count.
///
/// Layout convention for D_l (buffer of length M^2):
///   index = (c_y * h_l + s_y) * M + (c_x * h_l + s_x)
/// where 0 <= s_y, s_x < h_l (the period at level l)
///   and 0 <= c_y, c_x < n_per_axis_l = M / h_l.
/// Then every (i, j) in [0, M)^2 corresponds to a unique (cell, shift) pair at
/// level l.
///
/// Reduction l+1 -> l: each parent (cell c_p, shift s_p) sums 4 children at
///   child shift s_c = s_p mod h_cur          (low log2(h_cur) bits of s_p)
///   off            = s_p / h_cur             (high bit of s_p in [0, h_par))
///   child cells    = (2 c_p + off) mod n_per_axis_cur,
///                    (2 c_p + off + 1) mod n_per_axis_cur
/// per axis. From a parent index i_p in [0, M):
///   c_y_p = i_p / h_par
///   s_y_p = i_p mod h_par
///   s_y_c = s_y_p mod h_cur = i_p mod h_cur
///   off_y = s_y_p / h_cur
///   c_y_c = (2 c_y_p + off_y) mod n_per_axis_cur
///   i_cur = c_y_c * h_cur + s_y_c
/// Hierarchical all-shifts cascade.
///
/// `periodic = true` wraps cells across the box boundary (default behavior).
/// `periodic = false` discards any (shift, cell) pair whose cell would extend past
/// the box edge. With non-periodic BCs the per-level effective sample count is
/// reduced near the box boundary, but the cascade is honest about a non-periodic
/// underlying field.
pub fn cascade_hierarchical(pts: &[(u16, u16)], s_subshift: usize) -> (Vec<LevelStats>, Vec<usize>) {
    cascade_hierarchical_bc(pts, s_subshift, true)
}

pub fn cascade_hierarchical_bc(pts: &[(u16, u16)], s_subshift: usize, periodic: bool) -> (Vec<LevelStats>, Vec<usize>) {
    let (c_grid, m) = bin_to_fine_grid(pts, s_subshift);
    let total_pts: u64 = c_grid.iter().map(|&x| x as u64).sum();

    // The binned grid C has cells of side 1 (= h at the very finest level
    // l_max_eff = L_MAX + s_subshift). With h = 1 there are no shifts to
    // distinguish, so the layout collapses: i = c_y * 1 + 0 = c_y. So C is
    // already in our layout at the finest level.
    let mut cur: Vec<u64> = c_grid.iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; m * m];

    let mut out: Vec<Option<LevelStats>> = (0..N_LEVELS).map(|_| None).collect();
    let mut n_distinct: Vec<usize> = vec![0; N_LEVELS];

    let l_max_eff = L_MAX + s_subshift;

    // First, collect stats for the finest *reported* level (= L_MAX), which
    // corresponds to par_level = l_max_eff - 0... wait: cur starts at level
    // l_max_eff (the binned grid). We need stats for reported level L_MAX, which
    // is par_level = L_MAX + s_subshift.... no.
    //
    // Reported level convention: l_rep in 0..=L_MAX, with l_rep = 0 being one cell
    // (the box) and l_rep = L_MAX being the finest tree level (cells of side
    // M_tree / 2^L_MAX = 1 in tree-coord units, = 2^s_subshift in fine-grid units).
    //
    // Mapping: at internal level l_eff (counted from the M-grid), cells have side
    // h_eff = M / 2^l_eff. The corresponding tree level is the one with the same
    // physical cell size: M_tree / 2^l_rep = M / 2^l_eff * (M_tree/M) = h_eff /
    // 2^s_subshift, so 2^l_rep = M_tree * 2^s_subshift / h_eff = (M_tree * 2^s_subshift) / (M/2^l_eff) =
    // 2^l_eff (since M = M_tree * 2^s_subshift). So l_rep = l_eff. (!)
    //
    // Wait that's not right either. Let's just enumerate. M_tree = 2^L_MAX.
    // M = M_tree * 2^s_subshift. At l_eff = 0, h_eff = M, one cell -- this is the box,
    // = reported level 0. At l_eff = L_MAX + s_subshift, h_eff = 1 -- this is finer than
    // any reported level. The reported finest level has cell side 2^s_subshift in
    // fine-grid units, i.e. h_eff = 2^s_subshift, i.e. l_eff = L_MAX. So:
    //
    //   l_rep = l_eff   for l_eff in 0..=L_MAX (the reported levels)
    //
    // Good. Levels l_eff in (L_MAX, l_max_eff] are "subshift" levels we use only as
    // intermediate accumulators; we don't report them.

    // Collect stats for the *finest reported level* (l_rep = L_MAX, l_eff = L_MAX).
    // That requires reducing from cur_level = l_max_eff down to l_eff = L_MAX without
    // emitting stats, then on the L_MAX level emit. We'll emit during the reduction
    // loop whenever par_level <= L_MAX.

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;
        let n_per_axis_par = m / h_par;
        let _ = n_per_axis_par;

        // Reset nxt
        for v in nxt.iter_mut() { *v = 0; }

        let mut sum_n: u128 = 0;
        let mut sum_n2: u128 = 0;
        let mut sum_sib_var: f64 = 0.0;
        let mut n_valid: u64 = 0;          // count of valid (parent shift, parent cell) pairs in non-periodic mode
        let n_parent_entries: u64 = (m * m) as u64;

        for i_p in 0..m {
            let c_y_p = i_p / h_par;
            let s_y_p = i_p % h_par;
            let s_y_c = s_y_p % h_cur;
            let off_y = s_y_p / h_cur;
            let c_y_c_first  = (2 * c_y_p + off_y) % n_per_axis_cur;
            let c_y_c_second_unwrapped = 2 * c_y_p + off_y + 1;
            let c_y_c_second = c_y_c_second_unwrapped % n_per_axis_cur;
            let i_cur_first  = c_y_c_first  * h_cur + s_y_c;
            let i_cur_second = c_y_c_second * h_cur + s_y_c;
            // In non-periodic mode, parent cell extends from (s_y_p + c_y_p*h_par) to
            // (s_y_p + (c_y_p+1)*h_par) in fine units; valid iff the latter <= m.
            let valid_y = if periodic { true } else {
                s_y_p + (c_y_p + 1) * h_par <= m && c_y_c_second_unwrapped < n_per_axis_cur
            };

            for j_p in 0..m {
                let c_x_p = j_p / h_par;
                let s_x_p = j_p % h_par;
                let s_x_c = s_x_p % h_cur;
                let off_x = s_x_p / h_cur;
                let c_x_c_first  = (2 * c_x_p + off_x) % n_per_axis_cur;
                let c_x_c_second_unwrapped = 2 * c_x_p + off_x + 1;
                let c_x_c_second = c_x_c_second_unwrapped % n_per_axis_cur;
                let j_cur_first  = c_x_c_first  * h_cur + s_x_c;
                let j_cur_second = c_x_c_second * h_cur + s_x_c;
                let valid_x = if periodic { true } else {
                    s_x_p + (c_x_p + 1) * h_par <= m && c_x_c_second_unwrapped < n_per_axis_cur
                };
                let valid = valid_y && valid_x;

                let n_a = cur[i_cur_first  * m + j_cur_first];
                let n_b = cur[i_cur_first  * m + j_cur_second];
                let n_c = cur[i_cur_second * m + j_cur_first];
                let n_d = cur[i_cur_second * m + j_cur_second];

                let s = n_a + n_b + n_c + n_d;
                nxt[i_p * m + j_p] = s;

                if valid {
                    sum_n  += s as u128;
                    sum_n2 += (s as u128) * (s as u128);
                    let s2 = (n_a as u128) * (n_a as u128)
                           + (n_b as u128) * (n_b as u128)
                           + (n_c as u128) * (n_c as u128)
                           + (n_d as u128) * (n_d as u128);
                    let mean_c = (s as f64) / 4.0;
                    let v_sib = (s2 as f64) / 4.0 - mean_c * mean_c;
                    sum_sib_var += v_sib;
                    n_valid += 1;
                }
            }
        }

        if par_level <= L_MAX {
            let l_rep = par_level;
            let n_total_used = if periodic { n_parent_entries as f64 } else { n_valid as f64 };
            if n_total_used > 0.0 {
                let mean = (sum_n as f64) / n_total_used;
                let var  = ((sum_n2 as f64) / n_total_used - mean * mean).max(0.0);
                let avg_sib_var = sum_sib_var / n_total_used;
                let dvar = var - avg_sib_var;
                let n_cells_total = 1usize << (2 * l_rep);
                // n_distinct: in periodic mode = h_par^2; in non-periodic, this is the
                // number of valid parent (shift, cell) entries.
                let n_dist = if periodic {
                    (h_par * h_par).min(m * m)
                } else {
                    n_valid as usize
                };
                out[l_rep] = Some(LevelStats { n_cells_total, mean, var, dvar });
                n_distinct[l_rep] = n_dist;
            }
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    // For s_subshift == 0 the loop produces par_level in [0, L_MAX-1] and never hits
    // l_rep = L_MAX (which corresponds to par_level = L_MAX = l_max_eff, which is the
    // initial buffer not produced as a "par"). In that case we compute L_MAX stats from
    // the initial binned grid C directly.
    if s_subshift == 0 {
        let l_rep = L_MAX;
        let mut sum_n: u128 = 0;
        let mut sum_n2: u128 = 0;
        for &v in c_grid.iter() {
            sum_n  += v as u128;
            sum_n2 += (v as u128) * (v as u128);
        }
        let n_total = (m * m) as f64;
        let mean = (sum_n as f64) / n_total;
        let var  = ((sum_n2 as f64) / n_total - mean * mean).max(0.0);
        out[l_rep] = Some(LevelStats {
            n_cells_total: 1 << (2 * l_rep),
            mean, var,
            dvar: var,  // no children in reported range
        });
        n_distinct[l_rep] = 1;  // no shifts at this level when s_subshift = 0
    }

    // Sanity check: total mass conservation at every reported level (periodic only)
    if periodic {
        for l in 0..N_LEVELS {
            if let Some(s) = &out[l] {
                let cells_per_shift = (1u64 << (2 * l as u64)) as f64;
                let n_dist = n_distinct[l] as f64;
                let actual = (s.mean * cells_per_shift * n_dist).round() as u128;
                let expected = (total_pts as u128) * (n_dist as u128);
                let diff = if actual > expected { actual - expected } else { expected - actual };
                assert!(diff <= 2,
                    "level {} conservation failed: actual={}, expected={}", l, actual, expected);
            }
        }
    }

    let final_out: Vec<LevelStats> = out.into_iter()
        .enumerate()
        .map(|(l, o)| o.unwrap_or(LevelStats {
            n_cells_total: 1 << (2 * l),
            mean: 0.0, var: 0.0, dvar: 0.0,
        }))
        .collect();
    (final_out, n_distinct)
}

// ===================== Two-point correlation function (TPCF) =====================
//
// At each tree level l, the (cell, shift) buffer D_l holds counts for cells of side
// h_l in fine-grid units. Two cells with the same shift and cell-coords differing by
// (dy, dx) have centers separated by (dy * h_l, dx * h_l) in fine-grid units. Doing
// the cross-product
//
//     S(l, dy, dx) = sum_{shift} sum_{cy, cx} D_l[shift, cy, cx] * D_l[shift, (cy+dy) mod n, (cx+dx) mod n]
//
// gives an unbiased estimator of n_pairs * <N(x) N(x + (dy, dx)*h_l)>, and from
// it we recover the cell-cell correlation
//
//     xi_cell(l, sep) = S(l, dy, dx) / (n_pairs * <N>^2) - 1
//
// We sample axis-aligned lags k = 1, 2, 4, 8, ..., n_per_axis_l/2 on each axis and
// average x- and y-axis estimates (assumes statistical isotropy). That gives us a
// log-spaced ξ(r) at scales r = k * h_l (in fine-grid units) for each level.

#[derive(Clone, Debug)]
pub struct TpcfPoint {
    pub level: usize,
    pub k: usize,            // lag in cell units (separation = k cells at this level)
    pub r_fine: f64,         // physical separation in fine-grid units = k * h_l
    pub r_tree: f64,         // separation in tree-coord units (1 = h_L_MAX)
    pub xi: f64,             // measured correlation
    pub n_pairs: u64,        // number of (cell pairs, shifts) contributing
    pub smoothing_h_fine: f64,// cell side length h_l in fine-grid units
}

/// Hierarchical cascade that ALSO computes the cell-cell two-point correlation
/// function at log-spaced lags within each level.
///
/// Parameters:
///   - `pts`, `s_subshift`, `periodic`: same as cascade_hierarchical_bc
///   - `lag_levels`: only compute TPCF at these reported tree levels (e.g. [3, 4, 5, 6, 7, 8])
///                   to save work; pass an empty slice to skip TPCF entirely.
///
/// Returns: (level_stats, n_distinct, tpcf_points) where tpcf_points is a flat
/// vector spanning all (level, k) combinations requested.
pub fn cascade_hierarchical_with_tpcf(
    pts: &[(u16, u16)],
    s_subshift: usize,
    periodic: bool,
    lag_levels: &[usize],
) -> (Vec<LevelStats>, Vec<usize>, Vec<TpcfPoint>) {
    // First run the standard cascade to get level stats and means.
    let (level_stats, n_distinct) = cascade_hierarchical_bc(pts, s_subshift, periodic);

    if lag_levels.is_empty() {
        return (level_stats, n_distinct, Vec::new());
    }

    // We need the (cell, shift) buffers at the requested levels. Re-run the reduction
    // and snapshot the buffer at each requested level. This costs an extra pass equal
    // in cost to the cascade itself (i.e. doubles the cascade time, but binning is
    // not duplicated since we save the binned grid).
    let (c_grid, m) = bin_to_fine_grid(pts, s_subshift);
    let mut cur: Vec<u64> = c_grid.iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; m * m];
    let l_max_eff = L_MAX + s_subshift;

    // Storage for snapshots at requested levels.
    let mut snapshots: Vec<(usize, Vec<u64>)> = Vec::new();   // (l_rep, buffer)

    // For s_subshift == 0, level L_MAX is the initial binned grid. Capture it now.
    if s_subshift == 0 && lag_levels.contains(&L_MAX) {
        snapshots.push((L_MAX, cur.clone()));
    }

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;

        for v in nxt.iter_mut() { *v = 0; }

        for i_p in 0..m {
            let c_y_p = i_p / h_par;
            let s_y_p = i_p % h_par;
            let s_y_c = s_y_p % h_cur;
            let off_y = s_y_p / h_cur;
            let c_y_c_first = (2 * c_y_p + off_y) % n_per_axis_cur;
            let c_y_c_second = (2 * c_y_p + off_y + 1) % n_per_axis_cur;
            let i_cur_first = c_y_c_first * h_cur + s_y_c;
            let i_cur_second = c_y_c_second * h_cur + s_y_c;

            for j_p in 0..m {
                let c_x_p = j_p / h_par;
                let s_x_p = j_p % h_par;
                let s_x_c = s_x_p % h_cur;
                let off_x = s_x_p / h_cur;
                let c_x_c_first = (2 * c_x_p + off_x) % n_per_axis_cur;
                let c_x_c_second = (2 * c_x_p + off_x + 1) % n_per_axis_cur;
                let j_cur_first = c_x_c_first * h_cur + s_x_c;
                let j_cur_second = c_x_c_second * h_cur + s_x_c;

                let s = cur[i_cur_first  * m + j_cur_first]
                      + cur[i_cur_first  * m + j_cur_second]
                      + cur[i_cur_second * m + j_cur_first]
                      + cur[i_cur_second * m + j_cur_second];
                nxt[i_p * m + j_p] = s;
            }
        }

        if par_level <= L_MAX && lag_levels.contains(&par_level) {
            snapshots.push((par_level, nxt.clone()));
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    // Now compute TPCF for each snapshot.
    let mut tpcf_points: Vec<TpcfPoint> = Vec::new();
    for (l_rep, buf) in &snapshots {
        let l_eff = *l_rep;       // l_eff == l_rep with our convention
        let h_l = m >> l_eff;
        let n_per_axis = m / h_l;
        if n_per_axis < 2 { continue; }   // l_rep = 0 (one cell): no separations

        let mean = level_stats[*l_rep].mean;
        if mean <= 0.0 { continue; }

        // Generate log-spaced lags: k = 1, 2, 4, ..., up to n_per_axis/2 (or n_per_axis-1 for non-periodic)
        let max_k = if periodic { n_per_axis / 2 } else { n_per_axis - 1 };
        let mut k_list: Vec<usize> = Vec::new();
        let mut k = 1usize;
        while k <= max_k {
            k_list.push(k);
            k *= 2;
        }

        for k in k_list {
            // Compute S = sum over (shift, cell) of D[shift, cy, cx] * D[shift, (cy+dy) mod n, (cx+dx) mod n]
            // for two configurations: dy=k,dx=0 (y-axis lag) and dy=0,dx=k (x-axis lag).
            // Average them as the isotropic estimate. We use the (cell, shift) layout:
            //   index = (c_y * h_l + s_y) * M + (c_x * h_l + s_x)
            let mut sum_xy_y: u128 = 0;     // y-axis lag
            let mut sum_xy_x: u128 = 0;     // x-axis lag
            let mut n_pairs: u64 = 0;

            for c_y in 0..n_per_axis {
                let c_y_lag_unwrapped = c_y + k;
                let c_y_lag = c_y_lag_unwrapped % n_per_axis;
                let valid_y = if periodic { true } else { c_y_lag_unwrapped < n_per_axis };

                for s_y in 0..h_l {
                    let i = c_y * h_l + s_y;
                    let i_lag = c_y_lag * h_l + s_y;

                    for c_x in 0..n_per_axis {
                        let c_x_lag_unwrapped = c_x + k;
                        let c_x_lag = c_x_lag_unwrapped % n_per_axis;
                        let valid_x = if periodic { true } else { c_x_lag_unwrapped < n_per_axis };

                        for s_x in 0..h_l {
                            let j = c_x * h_l + s_x;
                            let j_lag = c_x_lag * h_l + s_x;

                            let center = buf[i * m + j];

                            if valid_y {
                                let neigh_y = buf[i_lag * m + j];
                                sum_xy_y += (center as u128) * (neigh_y as u128);
                            }
                            if valid_x {
                                let neigh_x = buf[i * m + j_lag];
                                sum_xy_x += (center as u128) * (neigh_x as u128);
                            }
                            // n_pairs only counts pairs that contributed (when both axes are
                            // valid). For a cleaner accounting, treat each axis's contribution
                            // as half a pair so the average works out; we sum and average below.
                            let v = (if valid_y {1u64} else {0}) + (if valid_x {1u64} else {0});
                            n_pairs += v;
                        }
                    }
                }
            }

            // Average over the two axes
            let total_xy_sum = (sum_xy_y as f64) + (sum_xy_x as f64);
            let avg_pair_value = total_xy_sum / (n_pairs as f64);
            let xi = avg_pair_value / (mean * mean) - 1.0;

            tpcf_points.push(TpcfPoint {
                level: *l_rep,
                k,
                r_fine: (k * h_l) as f64,
                r_tree: (k as f64) * ((1usize << (L_MAX - l_rep)) as f64),
                xi,
                n_pairs,
                smoothing_h_fine: h_l as f64,
            });
        }
    }

    (level_stats, n_distinct, tpcf_points)
}

// ===================== PMF cascade =====================
//
// For each tree level l, build the histogram (PMF) of cell counts N_c across
// all (cell, shift) pairs. Returns Vec<Vec<u64>> indexed by [level][count],
// along with the running mean, var, skewness, kurtosis at each level.

#[derive(Clone, Debug)]
pub struct PmfLevel {
    pub level: usize,
    /// Cell side in tree-coord units (= 2^(L_MAX - level)).
    pub r_tree: f64,
    pub mean: f64,
    pub var: f64,
    /// Standardized third moment: mu_3 / sigma^3.
    pub skew: f64,
    /// Standardized fourth moment: mu_4 / sigma^4. Excess kurtosis = kurt - 3.
    pub kurt: f64,
    /// Total (cell, shift) entries contributing to the histogram.
    pub n_total: u64,
    /// `hist[k]` = number of cells with count exactly `k`.
    pub histogram: Vec<u64>,
}

pub fn cascade_with_pmf(pts: &[(u16, u16)], s_subshift: usize, periodic: bool)
    -> (Vec<LevelStats>, Vec<usize>, Vec<PmfLevel>)
{
    use crate::L_MAX;
    let (level_stats, n_distinct) = cascade_hierarchical_bc(pts, s_subshift, periodic);

    // Re-run the reduction to snapshot per-level (cell, shift) buffers.
    // (Same trick as cascade_hierarchical_with_tpcf.)
    let (c_grid, m) = bin_to_fine_grid(pts, s_subshift);
    let mut cur: Vec<u64> = c_grid.iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; m * m];
    let l_max_eff = L_MAX + s_subshift;

    let mut pmfs: Vec<PmfLevel> = Vec::new();

    // Helper to compute a PmfLevel from a buffer
    fn buffer_to_pmf(buf: &[u64], level: usize, _m: usize) -> PmfLevel {
        let r_tree = (1usize << (L_MAX - level)) as f64;
        // Find max count to size the histogram
        let mut max_n = 0u64;
        for &v in buf { if v > max_n { max_n = v; } }
        let n_bins = (max_n + 1).min(1_000_000) as usize;   // safety cap
        let mut hist = vec![0u64; n_bins];
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut sum4 = 0.0f64;
        let n_total = buf.len() as u64;
        for &v in buf {
            let idx = (v as usize).min(n_bins - 1);
            hist[idx] += 1;
            let x = v as f64;
            sum += x; sum2 += x*x; sum3 += x*x*x; sum4 += x*x*x*x;
        }
        let n = n_total as f64;
        let mean = sum / n;
        let var = (sum2 / n - mean*mean).max(0.0);
        let std = var.sqrt();
        let m3 = sum3 / n - 3.0 * mean * sum2 / n + 2.0 * mean.powi(3);
        let m4 = sum4 / n - 4.0 * mean * sum3 / n + 6.0 * mean.powi(2) * sum2 / n - 3.0 * mean.powi(4);
        let skew = if std > 1e-12 { m3 / std.powi(3) } else { 0.0 };
        let kurt = if std > 1e-12 { m4 / std.powi(4) } else { 0.0 };
        PmfLevel { level, r_tree, mean, var, skew, kurt, n_total, histogram: hist }
    }

    // Capture L_MAX from the binned grid if s_subshift==0, otherwise from the loop.
    if s_subshift == 0 {
        pmfs.push(buffer_to_pmf(&cur, L_MAX, m));
    }

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;

        for v in nxt.iter_mut() { *v = 0; }

        for i_p in 0..m {
            let c_y_p = i_p / h_par;
            let s_y_p = i_p % h_par;
            let s_y_c = s_y_p % h_cur;
            let off_y = s_y_p / h_cur;
            let c_y_c_first = (2 * c_y_p + off_y) % n_per_axis_cur;
            let c_y_c_second = (2 * c_y_p + off_y + 1) % n_per_axis_cur;
            let i_cur_first = c_y_c_first * h_cur + s_y_c;
            let i_cur_second = c_y_c_second * h_cur + s_y_c;

            for j_p in 0..m {
                let c_x_p = j_p / h_par;
                let s_x_p = j_p % h_par;
                let s_x_c = s_x_p % h_cur;
                let off_x = s_x_p / h_cur;
                let c_x_c_first = (2 * c_x_p + off_x) % n_per_axis_cur;
                let c_x_c_second = (2 * c_x_p + off_x + 1) % n_per_axis_cur;
                let j_cur_first = c_x_c_first * h_cur + s_x_c;
                let j_cur_second = c_x_c_second * h_cur + s_x_c;

                let s = cur[i_cur_first  * m + j_cur_first]
                      + cur[i_cur_first  * m + j_cur_second]
                      + cur[i_cur_second * m + j_cur_first]
                      + cur[i_cur_second * m + j_cur_second];
                nxt[i_p * m + j_p] = s;
            }
        }

        if par_level <= L_MAX {
            pmfs.push(buffer_to_pmf(&nxt, par_level, m));
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    pmfs.sort_by_key(|p| p.level);
    (level_stats, n_distinct, pmfs)
}
