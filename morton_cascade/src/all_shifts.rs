// All-shifts cascade via summed-area table.
//
// Given a finest auxiliary grid C of size M x M (M = 2^(L_MAX + s)) into which
// we bin all points, every distinct origin shift of every tree level is a
// different tiling of C into h_l x h_l blocks (h_l = M / 2^l). With a 2D
// summed-area table built once, each block sum is O(1); the total cost per
// level is O(M^2), independent of the number of shifts. Turning ALL distinct
// shifts on at every level (h_l^2 per level, capped by M^2) is therefore
// the same cost as turning on just one.
//
// We accumulate, per level, sufficient statistics summed over all distinct
// shifts: Sum1 = sum_{shift, cell} N, Sum2 = sum_{shift, cell} N^2, and for
// the Schur residual SumSibSS = sum_{shift, parent} sum_{i in 4 children}
// (N_i - N_parent/4)^2.

use crate::{LevelStats, L_MAX, N_LEVELS};

/// Bin points into an M x M finest grid where M = 2^(L_MAX + s).
/// s = "subshift levels"; s = 0 means no subshifting (one cell per finest tree cell).
pub fn bin_to_fine_grid(pts: &[(u16, u16)], s: usize) -> (Vec<u32>, usize) {
    assert!(L_MAX + s <= 16, "fine grid would exceed u16 resolution");
    let m_bits = L_MAX + s;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;     // bits to drop from each u16 coord
    let mut c = vec![0u32; m * m];
    for &(x, y) in pts {
        let ix = (x >> shift) as usize;
        let iy = (y >> shift) as usize;
        c[iy * m + ix] += 1;
    }
    (c, m)
}

/// Build 2D summed-area table (inclusive). `S[i,j] = sum_{i'<=i, j'<=j} c[i',j']`.
/// Stored as u64 to avoid overflow at large M with many points.
/// Layout: row-major, length M*M.
pub fn build_sat(c: &[u32], m: usize) -> Vec<u64> {
    let mut s = vec![0u64; m * m];
    // First row
    let mut row = 0u64;
    for j in 0..m {
        row += c[j] as u64;
        s[j] = row;
    }
    // Subsequent rows: S[i,j] = S[i-1,j] + row_i_inclusive[j]
    for i in 1..m {
        let mut row = 0u64;
        for j in 0..m {
            row += c[i * m + j] as u64;
            s[i * m + j] = s[(i - 1) * m + j] + row;
        }
    }
    s
}

/// Generalized rectangle sum on a non-wrapping rectangle of size hi x hj (rows x cols).
///
/// At level l: cell side h_l = m / 2^l (in fine units). Number of distinct
/// shifts per axis is h_l (cells live on the M-grid, period h_l). So
/// n_distinct_per_axis = h_l, capped at m. n_distinct = h_l^2.
///
/// We use periodic wrap-around in the (s_x, s_y) tilings so that *every* shift
/// produces (M/h_l)^2 cells -> total cell evaluations per shift is m^2/h_l^2,
/// and across all h_l^2 shifts that is m^2 cell evaluations per level.
pub fn cascade_all_shifts(pts: &[(u16, u16)], s_subshift: usize) -> (Vec<LevelStats>, Vec<usize>) {
    let (c, m) = bin_to_fine_grid(pts, s_subshift);
    let sat = build_sat(&c, m);
    let total_pts = c.iter().map(|&x| x as u64).sum::<u64>();

    // Build a fast sum-over-(possibly-wrapping)-rectangle closure.
    // The rectangle [i0..i0+h) x [j0..j0+h) (mod m) splits into at most 4 non-wrapping
    // sub-rectangles. We just enumerate them in place.
    let split_rect = |i0: usize, j0: usize, h: usize| -> u64 {
        let h_i_first = (m - i0).min(h);
        let h_i_second = h - h_i_first;
        let h_j_first = (m - j0).min(h);
        let h_j_second = h - h_j_first;
        let mut total = 0u64;
        // (i0, j0) corner -- always present
        total += rect_sum_general(&sat, m, i0, j0, h_i_first, h_j_first);
        if h_j_second > 0 {
            total += rect_sum_general(&sat, m, i0, 0, h_i_first, h_j_second);
        }
        if h_i_second > 0 {
            total += rect_sum_general(&sat, m, 0, j0, h_i_second, h_j_first);
        }
        if h_i_second > 0 && h_j_second > 0 {
            total += rect_sum_general(&sat, m, 0, 0, h_i_second, h_j_second);
        }
        total
    };

    let mut out = Vec::with_capacity(N_LEVELS);
    let mut n_distinct = Vec::with_capacity(N_LEVELS);

    // We need per-level (mean, var, dvar). dvar requires knowledge of the 4 children.
    // We'll compute each level's basic stats here, then in a second pass compute dvar
    // using the per-level "sum of within-sibling squared deviations" formula evaluated
    // in fine-grid units.
    //
    // Pass 1: per-level mean and var (averaged over all distinct shifts at that level).
    //
    // For level l, h = m / 2^l. (At the very coarsest l=0, h = m, only 1 cell per shift,
    // so var = 0 within a shift; across shifts there's also only 1 distinct shift.)

    // Storage of per-level (mean, var) so we can subtract child siblings later.
    let mut basic = Vec::with_capacity(N_LEVELS);
    for l in 0..N_LEVELS {
        let n_per_axis = 1usize << l;             // cells per axis at level l (per shift)
        let h = m / n_per_axis;                   // cell side in fine units
        let n_dist_per_axis = h.min(m);           // distinct shifts per axis = period h
        let n_dist = n_dist_per_axis * n_dist_per_axis;
        let cells_per_shift = n_per_axis * n_per_axis;
        let total_evals = (n_dist * cells_per_shift) as u64;

        let mut sum_n = 0u64;
        let mut sum_n2 = 0u128;  // u128 to be safe with large total_pts

        // Loop over all distinct shifts and all cells per shift, summing the cell counts.
        // Uses periodic wrap so every shift gets the same number of cells.
        for sy in 0..n_dist_per_axis {
            for sx in 0..n_dist_per_axis {
                for cy in 0..n_per_axis {
                    for cx in 0..n_per_axis {
                        let i0 = (sy + cy * h) % m;
                        let j0 = (sx + cx * h) % m;
                        let n = split_rect(i0, j0, h);
                        sum_n += n;
                        sum_n2 += (n as u128) * (n as u128);
                    }
                }
            }
        }

        let mean = (sum_n as f64) / (total_evals as f64);
        let var  = ((sum_n2 as f64) / (total_evals as f64) - mean * mean).max(0.0);

        basic.push((mean, var));
        out.push(LevelStats { n_cells_total: cells_per_shift, mean, var, dvar: 0.0 });
        n_distinct.push(n_dist);
    }

    // Pass 2: Schur residual.
    // dvar_l = var_l - <within-sibling variance>_{children at l+1}
    // The within-sibling variance for one parent's 4 children {N_1..N_4} is
    //   V_sib = (1/4) * sum (N_i - N_p / 4)^2
    // Equivalent algebraic form: V_sib = (1/4) * sum N_i^2 - (N_p / 4)^2
    // Averaging over all parents at level l, all shifts at level l (= all distinct shifts at l):
    //   <V_sib>_l = (1/4)*<sum N_i^2 over 4 children>_(parent,shift) - <(N_p/4)^2>_(parent,shift)
    //
    // Note that "all distinct shifts at level l" projected onto level l+1 give h_{l+1} distinct
    // shifts per axis (out of h_l = 2 * h_{l+1}), each appearing twice. That's fine: the average
    // over level-l shifts of the children's stats equals the average over level-(l+1) shifts of
    // children's stats.
    //
    // We just compute <sum_{4 children} N^2>_(parent, shift_l) directly from the SAT.
    for l in 0..L_MAX {
        let n_per_axis = 1usize << l;
        let h = m / n_per_axis;          // parent cell side in fine units
        let h_child = h / 2;              // child cell side
        let n_dist_per_axis = h.min(m);   // shifts at parent level

        let mut sum_sib_var_total = 0.0f64;  // sum over (shift, parent) of V_sib
        let mut count = 0u64;

        for sy in 0..n_dist_per_axis {
            for sx in 0..n_dist_per_axis {
                for py in 0..n_per_axis {
                    for px in 0..n_per_axis {
                        // Parent rect origin
                        let pi = (sy + py * h) % m;
                        let pj = (sx + px * h) % m;
                        // Sum over 4 children: their origins are (pi + dy*h_child, pj + dx*h_child)
                        let mut sum_n = 0u64;
                        let mut sum_n2 = 0u128;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let ci = (pi + dy * h_child) % m;
                                let cj = (pj + dx * h_child) % m;
                                let n = split_rect(ci, cj, h_child);
                                sum_n += n;
                                sum_n2 += (n as u128) * (n as u128);
                            }
                        }
                        // V_sib = (1/4) * sum N_i^2 - (sum N_i / 4)^2
                        let mean_c = sum_n as f64 / 4.0;
                        let v_sib = (sum_n2 as f64) / 4.0 - mean_c * mean_c;
                        sum_sib_var_total += v_sib;
                        count += 1;
                    }
                }
            }
        }
        let avg_sib_var = sum_sib_var_total / count as f64;
        out[l].dvar = out[l].var - avg_sib_var;
    }
    // Finest level: no children
    out[L_MAX].dvar = out[L_MAX].var;

    // Sanity: total points conserved at every level
    for l in 0..N_LEVELS {
        let n_per_axis = 1usize << l;
        let cells_per_shift = (n_per_axis * n_per_axis) as u64;
        let n_dist = n_distinct[l] as u64;
        let expected_sum = total_pts * n_dist;        // each point counted once per shift
        let actual_sum = (out[l].mean * (cells_per_shift * n_dist) as f64).round() as u64;
        assert!(
            ((expected_sum as i128) - (actual_sum as i128)).abs() <= 1,
            "level {} mass conservation: expected {}, got {}", l, expected_sum, actual_sum
        );
    }
    (out, n_distinct)
}

/// Generalized rectangle sum on a non-wrapping rectangle of size hi x hj (rows x cols).
#[inline]
fn rect_sum_general(s: &[u64], m: usize, i0: usize, j0: usize, hi: usize, hj: usize) -> u64 {
    let i1 = i0 + hi - 1;
    let j1 = j0 + hj - 1;
    let a = s[i1 * m + j1];
    let b = if i0 == 0 { 0 } else { s[(i0 - 1) * m + j1] };
    let c = if j0 == 0 { 0 } else { s[i1 * m + (j0 - 1)] };
    let d = if i0 == 0 || j0 == 0 { 0 } else { s[(i0 - 1) * m + (j0 - 1)] };
    a + d - b - c
}
