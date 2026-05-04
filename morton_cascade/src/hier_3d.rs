// 3D hierarchical cascade.
//
// Same algorithm as 2D hier.rs but with M^3 buffers and 2x2x2 children.
// Coordinates are u16 in each axis. Internal buffer side M = 2^(L_MAX_3D + s_subshift).
// Memory per buffer = M^3 * 8 bytes. With L_MAX_3D = 7, s_subshift = 1 -> M = 256 ->
// 128 MB per buffer (256 MB ping-pong). With s_subshift = 0 -> M = 128 -> 16 MB.
//
// Per level: O(M^3) adds. Total: O(M^3 * L_MAX_3D).

use std::time::Instant;

pub const L_MAX_3D: usize = 7;             // tree-coord box = 128 per side in 3D
pub const N_LEVELS_3D: usize = L_MAX_3D + 1;

#[derive(Clone, Debug)]
pub struct LevelStats3D {
    pub n_cells_total: usize,    // 8^l
    pub mean: f64,
    pub var: f64,
    pub dvar: f64,                // Schur residual
}

/// Bin (u16, u16, u16) points into an M^3 grid.
pub fn bin_to_fine_grid_3d(pts: &[(u16, u16, u16)], s_subshift: usize) -> (Vec<u32>, usize) {
    assert!(L_MAX_3D + s_subshift <= 16, "fine grid would exceed u16 resolution");
    let m_bits = L_MAX_3D + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;
    let mut c = vec![0u32; m * m * m];
    for &(x, y, z) in pts {
        let ix = (x >> shift) as usize;
        let iy = (y >> shift) as usize;
        let iz = (z >> shift) as usize;
        c[(iz * m + iy) * m + ix] += 1;
    }
    (c, m)
}

#[derive(Clone, Debug)]
pub struct TpcfPoint3D {
    pub level: usize,
    pub k: usize,
    pub r_fine: f64,
    pub r_tree: f64,
    pub xi: f64,
    pub n_pairs: u64,
    pub smoothing_h_fine: f64,
}

/// 3D hierarchical cascade with optional TPCF computation.
///
/// Layout convention for a buffer at internal level l_eff (cell side h = M / 2^l_eff):
///     idx = ((c_z * h + s_z) * M + (c_y * h + s_y)) * M + (c_x * h + s_x)
/// where 0 <= s_* < h and 0 <= c_* < M / h.
pub fn cascade_3d_with_tpcf(
    pts: &[(u16, u16, u16)],
    s_subshift: usize,
    periodic: bool,
    lag_levels: &[usize],
) -> (Vec<LevelStats3D>, Vec<usize>, Vec<TpcfPoint3D>) {
    let (c_grid, m) = bin_to_fine_grid_3d(pts, s_subshift);
    let total_pts: u64 = c_grid.iter().map(|&x| x as u64).sum();

    let mut cur: Vec<u64> = c_grid.iter().map(|&x| x as u64).collect();
    let mut nxt: Vec<u64> = vec![0u64; m * m * m];

    let mut out: Vec<Option<LevelStats3D>> = (0..N_LEVELS_3D).map(|_| None).collect();
    let mut n_distinct: Vec<usize> = vec![0; N_LEVELS_3D];

    let l_max_eff = L_MAX_3D + s_subshift;
    let mut snapshots: Vec<(usize, Vec<u64>)> = Vec::new();

    if s_subshift == 0 && lag_levels.contains(&L_MAX_3D) {
        snapshots.push((L_MAX_3D, cur.clone()));
    }

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
        let n_parent_entries: u64 = (m * m * m) as u64;

        // Loop over parent (c, s) pairs encoded as (i_p, j_p, k_p) in [0, M)^3.
        for k_p in 0..m {
            let c_z_p = k_p / h_par;
            let s_z_p = k_p % h_par;
            let s_z_c = s_z_p % h_cur;
            let off_z = s_z_p / h_cur;
            let c_z_c_first  = (2 * c_z_p + off_z) % n_per_axis_cur;
            let c_z_c_second_unwrapped = 2 * c_z_p + off_z + 1;
            let c_z_c_second = c_z_c_second_unwrapped % n_per_axis_cur;
            let k_cur_first  = c_z_c_first  * h_cur + s_z_c;
            let k_cur_second = c_z_c_second * h_cur + s_z_c;
            let valid_z = if periodic { true } else {
                s_z_p + (c_z_p + 1) * h_par <= m && c_z_c_second_unwrapped < n_per_axis_cur
            };

            for j_p in 0..m {
                let c_y_p = j_p / h_par;
                let s_y_p = j_p % h_par;
                let s_y_c = s_y_p % h_cur;
                let off_y = s_y_p / h_cur;
                let c_y_c_first  = (2 * c_y_p + off_y) % n_per_axis_cur;
                let c_y_c_second_unwrapped = 2 * c_y_p + off_y + 1;
                let c_y_c_second = c_y_c_second_unwrapped % n_per_axis_cur;
                let j_cur_first  = c_y_c_first  * h_cur + s_y_c;
                let j_cur_second = c_y_c_second * h_cur + s_y_c;
                let valid_y = if periodic { true } else {
                    s_y_p + (c_y_p + 1) * h_par <= m && c_y_c_second_unwrapped < n_per_axis_cur
                };

                for i_p in 0..m {
                    let c_x_p = i_p / h_par;
                    let s_x_p = i_p % h_par;
                    let s_x_c = s_x_p % h_cur;
                    let off_x = s_x_p / h_cur;
                    let c_x_c_first  = (2 * c_x_p + off_x) % n_per_axis_cur;
                    let c_x_c_second_unwrapped = 2 * c_x_p + off_x + 1;
                    let c_x_c_second = c_x_c_second_unwrapped % n_per_axis_cur;
                    let i_cur_first  = c_x_c_first  * h_cur + s_x_c;
                    let i_cur_second = c_x_c_second * h_cur + s_x_c;
                    let valid_x = if periodic { true } else {
                        s_x_p + (c_x_p + 1) * h_par <= m && c_x_c_second_unwrapped < n_per_axis_cur
                    };
                    let valid = valid_z && valid_y && valid_x;

                    // 8 children
                    let n0 = cur[(k_cur_first  * m + j_cur_first ) * m + i_cur_first ];
                    let n1 = cur[(k_cur_first  * m + j_cur_first ) * m + i_cur_second];
                    let n2 = cur[(k_cur_first  * m + j_cur_second) * m + i_cur_first ];
                    let n3 = cur[(k_cur_first  * m + j_cur_second) * m + i_cur_second];
                    let n4 = cur[(k_cur_second * m + j_cur_first ) * m + i_cur_first ];
                    let n5 = cur[(k_cur_second * m + j_cur_first ) * m + i_cur_second];
                    let n6 = cur[(k_cur_second * m + j_cur_second) * m + i_cur_first ];
                    let n7 = cur[(k_cur_second * m + j_cur_second) * m + i_cur_second];

                    let s = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7;
                    nxt[(k_p * m + j_p) * m + i_p] = s;

                    if valid {
                        sum_n  += s as u128;
                        sum_n2 += (s as u128) * (s as u128);
                        let s2_sum = (n0 as u128).pow(2) + (n1 as u128).pow(2)
                                   + (n2 as u128).pow(2) + (n3 as u128).pow(2)
                                   + (n4 as u128).pow(2) + (n5 as u128).pow(2)
                                   + (n6 as u128).pow(2) + (n7 as u128).pow(2);
                        let mean_c = (s as f64) / 8.0;
                        let v_sib = (s2_sum as f64) / 8.0 - mean_c * mean_c;
                        sum_sib_var += v_sib;
                        n_valid += 1;
                    }
                }
            }
        }

        if par_level <= L_MAX_3D {
            let l_rep = par_level;
            let n_used = if periodic { n_parent_entries as f64 } else { n_valid as f64 };
            if n_used > 0.0 {
                let mean = (sum_n as f64) / n_used;
                let var  = ((sum_n2 as f64) / n_used - mean * mean).max(0.0);
                let avg_sib_var = sum_sib_var / n_used;
                let dvar = var - avg_sib_var;
                let n_cells_total = 1usize << (3 * l_rep);
                let n_dist = if periodic {
                    (h_par * h_par * h_par).min(m * m * m)
                } else {
                    n_valid as usize
                };
                out[l_rep] = Some(LevelStats3D { n_cells_total, mean, var, dvar });
                n_distinct[l_rep] = n_dist;

                if lag_levels.contains(&par_level) {
                    snapshots.push((par_level, nxt.clone()));
                }
            }
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    if s_subshift == 0 {
        let l_rep = L_MAX_3D;
        if out[l_rep].is_none() {
            let mut sum_n: u128 = 0;
            let mut sum_n2: u128 = 0;
            for &v in c_grid.iter() {
                sum_n  += v as u128;
                sum_n2 += (v as u128) * (v as u128);
            }
            let n_total = (m * m * m) as f64;
            let mean = (sum_n as f64) / n_total;
            let var  = ((sum_n2 as f64) / n_total - mean * mean).max(0.0);
            out[l_rep] = Some(LevelStats3D {
                n_cells_total: 1 << (3 * l_rep),
                mean, var, dvar: var,
            });
            n_distinct[l_rep] = 1;
        }
    }

    // Conservation check
    if periodic {
        for l in 0..N_LEVELS_3D {
            if let Some(s) = &out[l] {
                let cells_per_shift = (1u64 << (3 * l as u64)) as f64;
                let n_dist = n_distinct[l] as f64;
                let actual = (s.mean * cells_per_shift * n_dist).round() as u128;
                let expected = (total_pts as u128) * (n_dist as u128);
                let diff = if actual > expected { actual - expected } else { expected - actual };
                assert!(diff <= 4, "3D level {} conservation: actual={}, expected={}",
                    l, actual, expected);
            }
        }
    }

    let final_out: Vec<LevelStats3D> = out.into_iter()
        .enumerate()
        .map(|(l, o)| o.unwrap_or(LevelStats3D {
            n_cells_total: 1 << (3 * l),
            mean: 0.0, var: 0.0, dvar: 0.0,
        }))
        .collect();

    // ---------------- TPCF ----------------
    let mut tpcf_points: Vec<TpcfPoint3D> = Vec::new();
    for (l_rep, buf) in &snapshots {
        let l_eff = *l_rep;
        let h_l = m >> l_eff;
        let n_per_axis = m / h_l;
        if n_per_axis < 2 { continue; }
        let mean = final_out[*l_rep].mean;
        if mean <= 0.0 { continue; }

        let max_k = if periodic { n_per_axis / 2 } else { n_per_axis - 1 };
        let mut k_list: Vec<usize> = Vec::new();
        let mut k = 1usize;
        while k <= max_k { k_list.push(k); k *= 2; }

        for k in k_list {
            let mut sum_xy_x: u128 = 0;
            let mut sum_xy_y: u128 = 0;
            let mut sum_xy_z: u128 = 0;
            let mut n_pairs_x: u64 = 0;
            let mut n_pairs_y: u64 = 0;
            let mut n_pairs_z: u64 = 0;

            // Iterate over (cell, shift) combinations indexed in our layout.
            // For each (c_z, c_y, c_x, s_z, s_y, s_x), compute lagged neighbours
            // along x, y, z separately.
            for c_z in 0..n_per_axis {
                let c_z_lag_unwrapped = c_z + k;
                let c_z_lag = c_z_lag_unwrapped % n_per_axis;
                let valid_z = if periodic { true } else { c_z_lag_unwrapped < n_per_axis };
                for s_z in 0..h_l {
                    let kk = c_z * h_l + s_z;
                    let kk_lag = c_z_lag * h_l + s_z;
                    for c_y in 0..n_per_axis {
                        let c_y_lag_unwrapped = c_y + k;
                        let c_y_lag = c_y_lag_unwrapped % n_per_axis;
                        let valid_y = if periodic { true } else { c_y_lag_unwrapped < n_per_axis };
                        for s_y in 0..h_l {
                            let jj = c_y * h_l + s_y;
                            let jj_lag = c_y_lag * h_l + s_y;
                            for c_x in 0..n_per_axis {
                                let c_x_lag_unwrapped = c_x + k;
                                let c_x_lag = c_x_lag_unwrapped % n_per_axis;
                                let valid_x = if periodic { true } else { c_x_lag_unwrapped < n_per_axis };
                                for s_x in 0..h_l {
                                    let ii = c_x * h_l + s_x;
                                    let ii_lag = c_x_lag * h_l + s_x;
                                    let center = buf[(kk * m + jj) * m + ii];
                                    if valid_x {
                                        let n_x = buf[(kk * m + jj) * m + ii_lag];
                                        sum_xy_x += (center as u128) * (n_x as u128);
                                        n_pairs_x += 1;
                                    }
                                    if valid_y {
                                        let n_y = buf[(kk * m + jj_lag) * m + ii];
                                        sum_xy_y += (center as u128) * (n_y as u128);
                                        n_pairs_y += 1;
                                    }
                                    if valid_z {
                                        let n_z = buf[(kk_lag * m + jj) * m + ii];
                                        sum_xy_z += (center as u128) * (n_z as u128);
                                        n_pairs_z += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let total_xy = (sum_xy_x as f64) + (sum_xy_y as f64) + (sum_xy_z as f64);
            let total_pairs = n_pairs_x + n_pairs_y + n_pairs_z;
            if total_pairs == 0 { continue; }
            let avg = total_xy / (total_pairs as f64);
            let xi = avg / (mean * mean) - 1.0;

            tpcf_points.push(TpcfPoint3D {
                level: *l_rep,
                k,
                r_fine: (k * h_l) as f64,
                r_tree: (k as f64) * ((1usize << (L_MAX_3D - l_rep)) as f64),
                xi,
                n_pairs: total_pairs,
                smoothing_h_fine: h_l as f64,
            });
        }
    }

    let _ = Instant::now();
    (final_out, n_distinct, tpcf_points)
}
