// Threaded variants of the hierarchical cascade and binning.
//
// Strategy:
//   bin_to_fine_grid_par: split points into chunks, each thread bins into a
//     private M^2 grid, then we element-wise sum the per-thread grids.
//   cascade_hierarchical_par: per level, parallel-for over rows i_p in [0, M),
//     with each thread accumulating private (sum_n, sum_n2, sum_sib_var)
//     partials that are reduced at the end of the level.

use crate::{LevelStats, L_MAX, N_LEVELS};
use rayon::prelude::*;

pub fn bin_to_fine_grid_par(pts: &[(u16, u16)], s_subshift: usize) -> (Vec<u32>, usize) {
    assert!(L_MAX + s_subshift <= 16);
    let m_bits = L_MAX + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;

    // Choose chunk count = number of rayon threads (or use a sensible default).
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (pts.len() + n_threads - 1) / n_threads;
    if chunk_size == 0 {
        return (vec![0u32; m * m], m);
    }

    // Each thread bins into a private M^2 grid.
    let partials: Vec<Vec<u32>> = pts
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut g = vec![0u32; m * m];
            for &(x, y) in chunk {
                let ix = (x >> shift) as usize;
                let iy = (y >> shift) as usize;
                g[iy * m + ix] += 1;
            }
            g
        })
        .collect();

    // Reduce: element-wise sum into a single grid, in parallel by row.
    let mut out = vec![0u32; m * m];
    let row_stride = m;
    out.par_chunks_mut(row_stride).enumerate().for_each(|(i, row)| {
        for partial in &partials {
            let src = &partial[i * row_stride..(i + 1) * row_stride];
            for j in 0..row_stride {
                row[j] += src[j];
            }
        }
    });
    (out, m)
}

/// Threaded all-shifts cascade. Same outputs as `hier::cascade_hierarchical_bc`.
pub fn cascade_hierarchical_par(
    pts: &[(u16, u16)],
    s_subshift: usize,
    periodic: bool,
) -> (Vec<LevelStats>, Vec<usize>) {
    let (c_grid, m) = bin_to_fine_grid_par(pts, s_subshift);
    let total_pts: u64 = c_grid.par_iter().map(|&x| x as u64).sum();

    let mut cur: Vec<u64> = c_grid.par_iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; m * m];

    let mut out: Vec<Option<LevelStats>> = (0..N_LEVELS).map(|_| None).collect();
    let mut n_distinct: Vec<usize> = vec![0; N_LEVELS];

    let l_max_eff = L_MAX + s_subshift;

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;

        // Reset nxt — parallel
        nxt.par_iter_mut().for_each(|v| *v = 0);

        // Borrow cur immutably so we can use it in the parallel closure
        let cur_ref: &[u64] = &cur;

        // Compute per-row partials for both nxt and the moment accumulators.
        // Each row of length M is independent: reads from cur_ref, writes nxt[i_p*M..i_p*M+M].
        // We use par_chunks_mut on nxt rows and pass i_p via enumerate.
        let row_partials: Vec<(u128, u128, f64, u64)> = nxt
            .par_chunks_mut(m)
            .enumerate()
            .map(|(i_p, nxt_row)| {
                let c_y_p = i_p / h_par;
                let s_y_p = i_p % h_par;
                let s_y_c = s_y_p % h_cur;
                let off_y = s_y_p / h_cur;
                let c_y_c_first = (2 * c_y_p + off_y) % n_per_axis_cur;
                let c_y_c_second_unwrapped = 2 * c_y_p + off_y + 1;
                let c_y_c_second = c_y_c_second_unwrapped % n_per_axis_cur;
                let i_cur_first = c_y_c_first * h_cur + s_y_c;
                let i_cur_second = c_y_c_second * h_cur + s_y_c;
                let valid_y = if periodic {
                    true
                } else {
                    s_y_p + (c_y_p + 1) * h_par <= m && c_y_c_second_unwrapped < n_per_axis_cur
                };

                let mut sum_n: u128 = 0;
                let mut sum_n2: u128 = 0;
                let mut sum_sib_var: f64 = 0.0;
                let mut n_valid: u64 = 0;

                for j_p in 0..m {
                    let c_x_p = j_p / h_par;
                    let s_x_p = j_p % h_par;
                    let s_x_c = s_x_p % h_cur;
                    let off_x = s_x_p / h_cur;
                    let c_x_c_first = (2 * c_x_p + off_x) % n_per_axis_cur;
                    let c_x_c_second_unwrapped = 2 * c_x_p + off_x + 1;
                    let c_x_c_second = c_x_c_second_unwrapped % n_per_axis_cur;
                    let j_cur_first = c_x_c_first * h_cur + s_x_c;
                    let j_cur_second = c_x_c_second * h_cur + s_x_c;
                    let valid_x = if periodic {
                        true
                    } else {
                        s_x_p + (c_x_p + 1) * h_par <= m && c_x_c_second_unwrapped < n_per_axis_cur
                    };
                    let valid = valid_y && valid_x;

                    let n_a = cur_ref[i_cur_first * m + j_cur_first];
                    let n_b = cur_ref[i_cur_first * m + j_cur_second];
                    let n_c = cur_ref[i_cur_second * m + j_cur_first];
                    let n_d = cur_ref[i_cur_second * m + j_cur_second];

                    let s = n_a + n_b + n_c + n_d;
                    nxt_row[j_p] = s;

                    if valid {
                        sum_n += s as u128;
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
                (sum_n, sum_n2, sum_sib_var, n_valid)
            })
            .collect();

        // Reduce row partials into totals
        let mut sum_n_tot: u128 = 0;
        let mut sum_n2_tot: u128 = 0;
        let mut sum_sib_var_tot: f64 = 0.0;
        let mut n_valid_tot: u64 = 0;
        for &(s1, s2, sv, nv) in &row_partials {
            sum_n_tot += s1;
            sum_n2_tot += s2;
            sum_sib_var_tot += sv;
            n_valid_tot += nv;
        }
        let n_parent_entries: u64 = (m * m) as u64;

        if par_level <= L_MAX {
            let l_rep = par_level;
            let n_used = if periodic {
                n_parent_entries as f64
            } else {
                n_valid_tot as f64
            };
            if n_used > 0.0 {
                let mean = (sum_n_tot as f64) / n_used;
                let var = ((sum_n2_tot as f64) / n_used - mean * mean).max(0.0);
                let avg_sib_var = sum_sib_var_tot / n_used;
                let dvar = var - avg_sib_var;
                let n_cells_total = 1usize << (2 * l_rep);
                let n_dist = if periodic {
                    (h_par * h_par).min(m * m)
                } else {
                    n_valid_tot as usize
                };
                out[l_rep] = Some(LevelStats {
                    n_cells_total,
                    mean,
                    var,
                    dvar,
                });
                n_distinct[l_rep] = n_dist;
            }
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    // s_subshift == 0 special case (matches serial version)
    if s_subshift == 0 {
        let l_rep = L_MAX;
        let (sum_n, sum_n2): (u128, u128) = c_grid
            .par_iter()
            .map(|&v| (v as u128, (v as u128) * (v as u128)))
            .reduce(|| (0u128, 0u128), |a, b| (a.0 + b.0, a.1 + b.1));
        let n_total = (m * m) as f64;
        let mean = (sum_n as f64) / n_total;
        let var = ((sum_n2 as f64) / n_total - mean * mean).max(0.0);
        out[l_rep] = Some(LevelStats {
            n_cells_total: 1 << (2 * l_rep),
            mean,
            var,
            dvar: var,
        });
        n_distinct[l_rep] = 1;
    }

    // Conservation check (periodic only)
    if periodic {
        for l in 0..N_LEVELS {
            if let Some(s) = &out[l] {
                let cells_per_shift = (1u64 << (2 * l as u64)) as f64;
                let n_dist = n_distinct[l] as f64;
                let actual = (s.mean * cells_per_shift * n_dist).round() as u128;
                let expected = (total_pts as u128) * (n_dist as u128);
                let diff = if actual > expected {
                    actual - expected
                } else {
                    expected - actual
                };
                assert!(
                    diff <= 2,
                    "level {} conservation failed: actual={}, expected={}",
                    l,
                    actual,
                    expected
                );
            }
        }
    }

    let final_out: Vec<LevelStats> = out
        .into_iter()
        .enumerate()
        .map(|(l, o)| {
            o.unwrap_or(LevelStats {
                n_cells_total: 1 << (2 * l),
                mean: 0.0,
                var: 0.0,
                dvar: 0.0,
            })
        })
        .collect();
    (final_out, n_distinct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hier::cascade_hierarchical_bc;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn make_pts(n: usize, seed: u64) -> Vec<(u16, u16)> {
        let mut s = seed;
        (0..n).map(|_| (
            (splitmix64(&mut s) & 0xFFFF) as u16,
            (splitmix64(&mut s) & 0xFFFF) as u16,
        )).collect()
    }

    /// Equivalence test: the threaded `cascade_hierarchical_par` must
    /// produce results bit-equivalent (or numerically very close, for
    /// the f64 dvar field) to the serial `cascade_hierarchical_bc`
    /// reference.
    ///
    /// Per the parallelism audit (docs/parallelism.md §1), `hier_par.rs`
    /// is currently dead code (not used or tested elsewhere). This test
    /// exists so that future modifications can't silently break it.
    #[test]
    fn par_matches_serial_periodic_unweighted() {
        let pts = make_pts(2000, 12345);
        let (serial_stats, serial_dist) = cascade_hierarchical_bc(&pts, 0, true);
        let (par_stats, par_dist) = cascade_hierarchical_par(&pts, 0, true);
        assert_eq!(serial_stats.len(), par_stats.len());
        assert_eq!(serial_dist, par_dist);
        for (l, (a, b)) in serial_stats.iter().zip(par_stats.iter()).enumerate() {
            assert_eq!(a.n_cells_total, b.n_cells_total,
                "level {}: n_cells_total", l);
            // Means and variances are derived from u128 sums divided by
            // f64 totals. Both implementations use the same math but
            // different summation orders — for u128 inputs the order
            // doesn't matter, so equality should be bit-exact in
            // mean/var. dvar involves an f64 sum_sib_var that's
            // accumulated per-row deterministically and then summed
            // serially — also bit-exact.
            assert_eq!(a.mean.to_bits(), b.mean.to_bits(),
                "level {}: mean (serial {} vs par {})", l, a.mean, b.mean);
            assert_eq!(a.var.to_bits(), b.var.to_bits(),
                "level {}: var (serial {} vs par {})", l, a.var, b.var);
            assert_eq!(a.dvar.to_bits(), b.dvar.to_bits(),
                "level {}: dvar (serial {} vs par {})", l, a.dvar, b.dvar);
        }
    }
}
