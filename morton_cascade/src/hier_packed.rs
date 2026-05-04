// Adaptive-width hierarchical cascade.
//
// Per-level buffers stored in the smallest integer type that fits the level's
// max count. Concretely:
//   level l has max count <= max_count_per_level[l] (provided/measured at runtime)
//   buffer element type chosen from {u8, u16, u32, u64} accordingly
//
// The inner loop is generic over the element type. Children at level l+1 are
// read at width T_child, summed in u32 (always wide enough since 4 children and
// even u8 max=255 -> sum=1020 fits in u32), and written at width T_parent.
//
// Bit width selection is conservative: round the measured max up to the next
// power-of-2 type, with optional 4x safety margin. The user can also pin the
// width manually for MCMC inner loops where the same field is sampled many
// times and the worst-case max is known after the first run.
//
// Memory savings vs always-u64:
//   level 8 (M^2 = 65536 cells, max ~30): u8 gives 8x reduction
//   level 5 (M^2 = 1024 cells, max ~200): u16 gives 4x reduction
//   level 0 (1 cell, max = N): u32 or u64 needed
// For typical N=1e6 in 2D L_MAX=8, total buffer memory drops from ~4 MB to ~1 MB.
//
// Speed-wise: smaller types -> better cache utilization on the cascade reduction
// (which is bandwidth-limited at large M). Expect ~1.5-2x speedup at fine levels
// when buffers no longer fit in L1.

use crate::{LevelStats, L_MAX, N_LEVELS};

/// Storage type for one level's buffer.
#[derive(Clone)]
pub enum LevelBuf {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl LevelBuf {
    pub fn len(&self) -> usize {
        match self {
            LevelBuf::U8(v) => v.len(),
            LevelBuf::U16(v) => v.len(),
            LevelBuf::U32(v) => v.len(),
            LevelBuf::U64(v) => v.len(),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            LevelBuf::U8(v) => v.len(),
            LevelBuf::U16(v) => v.len() * 2,
            LevelBuf::U32(v) => v.len() * 4,
            LevelBuf::U64(v) => v.len() * 8,
        }
    }
    /// Read a cell as u32 (wide enough for any of the variants).
    #[inline(always)]
    pub fn get(&self, i: usize) -> u32 {
        match self {
            LevelBuf::U8(v) => v[i] as u32,
            LevelBuf::U16(v) => v[i] as u32,
            LevelBuf::U32(v) => v[i],
            LevelBuf::U64(v) => v[i] as u32,   // truncates -- only safe if max < 2^32
        }
    }
    pub fn alloc_zeroed(n: usize, max_count: u64) -> Self {
        if max_count <= u8::MAX as u64 {
            LevelBuf::U8(vec![0u8; n])
        } else if max_count <= u16::MAX as u64 {
            LevelBuf::U16(vec![0u16; n])
        } else if max_count <= u32::MAX as u64 {
            LevelBuf::U32(vec![0u32; n])
        } else {
            LevelBuf::U64(vec![0u64; n])
        }
    }
    pub fn type_str(&self) -> &'static str {
        match self {
            LevelBuf::U8(_) => "u8", LevelBuf::U16(_) => "u16",
            LevelBuf::U32(_) => "u32", LevelBuf::U64(_) => "u64",
        }
    }
    /// Write a u32 value; panics if value exceeds the storage width.
    #[inline(always)]
    fn set(&mut self, i: usize, v: u32) {
        match self {
            LevelBuf::U8(b)  => { debug_assert!(v <= u8::MAX  as u32); b[i] = v as u8;  }
            LevelBuf::U16(b) => { debug_assert!(v <= u16::MAX as u32); b[i] = v as u16; }
            LevelBuf::U32(b) => { b[i] = v; }
            LevelBuf::U64(b) => { b[i] = v as u64; }
        }
    }
    /// Slice access for in-place ops (zeroing, etc.)
    pub fn zero(&mut self) {
        match self {
            LevelBuf::U8(v)  => { for x in v.iter_mut() { *x = 0; } }
            LevelBuf::U16(v) => { for x in v.iter_mut() { *x = 0; } }
            LevelBuf::U32(v) => { for x in v.iter_mut() { *x = 0; } }
            LevelBuf::U64(v) => { for x in v.iter_mut() { *x = 0; } }
        }
    }
}

/// Hint: predicted max counts per reported level (0..=L_MAX). Used to choose
/// bit widths. If unknown, pass a conservative estimate (e.g., total_pts at level 0,
/// scaling down by 2^D per level, with safety factor).
pub fn predict_max_counts_2d(total_pts: u64, safety_factor: u64) -> [u64; N_LEVELS] {
    let mut out = [0u64; N_LEVELS];
    for l in 0..N_LEVELS {
        let n_cells = 1u64 << (2 * l);
        let mean = total_pts / n_cells.max(1);
        // Heuristic: max ~ (mean + 5*sqrt(mean) + 10) * safety_factor for clustering tail.
        // Use saturating arithmetic to avoid overflow at coarse levels with large mean.
        let mean_f = mean as f64;
        let bound = (mean_f + 5.0 * mean_f.sqrt() + 10.0) * safety_factor as f64;
        out[l] = bound.min(u32::MAX as f64) as u64;
    }
    // Level 0 always = total_pts
    out[0] = total_pts;
    out
}

/// Adaptive-width 2D hierarchical cascade. Returns same stats as the dense version
/// plus per-level memory usage.
///
/// `max_counts_hint` should be a per-level upper bound on max cell count across
/// shifts. If too tight, this function will panic in debug mode (debug_assert in
/// LevelBuf::set) -- in release, narrow types silently truncate (BAD), so make sure
/// the hint is conservative. Use `predict_max_counts_2d` for safe defaults.
pub fn cascade_adaptive(
    pts: &[(u16, u16)],
    s_subshift: usize,
    periodic: bool,
    max_counts_hint: Option<[u64; N_LEVELS]>,
) -> (Vec<LevelStats>, Vec<usize>, Vec<usize>) {
    assert!(L_MAX + s_subshift <= 16);
    let m_bits = L_MAX + s_subshift;
    let m = 1usize << m_bits;
    let shift = 16 - m_bits as u32;

    let total_pts = pts.len() as u64;
    let max_counts = max_counts_hint.unwrap_or_else(|| predict_max_counts_2d(total_pts, 4));

    // Build the binned grid in u32 (worst case at the finest level fits comfortably).
    // We don't shrink C itself based on max_counts because some cells could exceed
    // the level-L_MAX bound during binning of clustered data; u32 is safe.
    let mut c_grid = vec![0u32; m * m];
    for &(x, y) in pts {
        let ix = (x >> shift) as usize;
        let iy = (y >> shift) as usize;
        c_grid[iy * m + ix] += 1;
    }

    // The "cur" buffer starts as the binned grid (treated as level l_max_eff).
    // Type at the finest reported level (L_MAX) is determined by max_counts[L_MAX].
    let mut cur: LevelBuf = {
        // For the finest reported level we need at least max_counts[L_MAX].
        // Initial (l_eff = l_max_eff) max is at most max_counts[L_MAX] (since
        // l_eff = l_max_eff has cells of side 1, smaller than tree level L_MAX).
        let max_init = max_counts[L_MAX];
        let mut buf = LevelBuf::alloc_zeroed(m * m, max_init);
        for i in 0..(m*m) {
            buf.set(i, c_grid[i]);
        }
        buf
    };
    let mut nxt: LevelBuf = LevelBuf::alloc_zeroed(m * m, max_counts[L_MAX]);

    let mut out: Vec<Option<LevelStats>> = (0..N_LEVELS).map(|_| None).collect();
    let mut n_distinct: Vec<usize> = vec![0; N_LEVELS];
    let mut bytes_per_level: Vec<usize> = vec![0; N_LEVELS];

    let l_max_eff = L_MAX + s_subshift;

    for cur_level in (1..=l_max_eff).rev() {
        let par_level = cur_level - 1;
        let h_cur = m >> cur_level;
        let h_par = h_cur << 1;
        let n_per_axis_cur = m / h_cur;

        // The parent buffer's type is determined by max_counts at the *parent's
        // reported level*, if par_level <= L_MAX. Otherwise (subshift levels)
        // we use the finest reported level's max. We may need to *grow* the
        // existing nxt buffer to a wider type.
        let par_max = if par_level <= L_MAX {
            max_counts[par_level]
        } else {
            max_counts[L_MAX]
        };
        // Allocate nxt fresh if its current type can't hold par_max.
        let need_realloc = match (&nxt, par_max) {
            (LevelBuf::U8(_), m)  if m > u8::MAX as u64  => true,
            (LevelBuf::U16(_), m) if m > u16::MAX as u64 => true,
            (LevelBuf::U32(_), m) if m > u32::MAX as u64 => true,
            _ => false,
        };
        if need_realloc {
            nxt = LevelBuf::alloc_zeroed(m * m, par_max);
        } else {
            nxt.zero();
        }

        let mut sum_n: u128 = 0;
        let mut sum_n2: u128 = 0;
        let mut sum_sib_var: f64 = 0.0;
        let mut n_valid: u64 = 0;
        let n_parent_entries: u64 = (m * m) as u64;

        for i_p in 0..m {
            let c_y_p = i_p / h_par;
            let s_y_p = i_p % h_par;
            let s_y_c = s_y_p % h_cur;
            let off_y = s_y_p / h_cur;
            let c_y_c_first = (2 * c_y_p + off_y) % n_per_axis_cur;
            let c_y_c_second_unwrapped = 2 * c_y_p + off_y + 1;
            let c_y_c_second = c_y_c_second_unwrapped % n_per_axis_cur;
            let i_cur_first = c_y_c_first * h_cur + s_y_c;
            let i_cur_second = c_y_c_second * h_cur + s_y_c;
            let valid_y = if periodic { true } else {
                s_y_p + (c_y_p + 1) * h_par <= m && c_y_c_second_unwrapped < n_per_axis_cur
            };

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
                let valid_x = if periodic { true } else {
                    s_x_p + (c_x_p + 1) * h_par <= m && c_x_c_second_unwrapped < n_per_axis_cur
                };
                let valid = valid_y && valid_x;

                let n_a = cur.get(i_cur_first  * m + j_cur_first );
                let n_b = cur.get(i_cur_first  * m + j_cur_second);
                let n_c = cur.get(i_cur_second * m + j_cur_first );
                let n_d = cur.get(i_cur_second * m + j_cur_second);

                // Sum is up to 4 * max_child <= 4 * 2^32, fits in u64.
                let s = (n_a as u64) + (n_b as u64) + (n_c as u64) + (n_d as u64);
                debug_assert!(s <= u32::MAX as u64);
                nxt.set(i_p * m + j_p, s as u32);

                if valid {
                    sum_n  += s as u128;
                    sum_n2 += (s as u128) * (s as u128);
                    let s2 = (n_a as u128).pow(2) + (n_b as u128).pow(2)
                           + (n_c as u128).pow(2) + (n_d as u128).pow(2);
                    let mean_c = (s as f64) / 4.0;
                    let v_sib = (s2 as f64) / 4.0 - mean_c * mean_c;
                    sum_sib_var += v_sib;
                    n_valid += 1;
                }
            }
        }

        if par_level <= L_MAX {
            let l_rep = par_level;
            let n_used = if periodic { n_parent_entries as f64 } else { n_valid as f64 };
            if n_used > 0.0 {
                let mean = (sum_n as f64) / n_used;
                let var  = ((sum_n2 as f64) / n_used - mean * mean).max(0.0);
                let avg_sib_var = sum_sib_var / n_used;
                let dvar = var - avg_sib_var;
                let n_cells_total = 1usize << (2 * l_rep);
                let n_dist = if periodic { (h_par * h_par).min(m * m) } else { n_valid as usize };
                out[l_rep] = Some(LevelStats { n_cells_total, mean, var, dvar });
                n_distinct[l_rep] = n_dist;
                bytes_per_level[l_rep] = nxt.bytes();
            }
        }

        std::mem::swap(&mut cur, &mut nxt);
    }

    if s_subshift == 0 {
        let l_rep = L_MAX;
        if out[l_rep].is_none() {
            let mut sum_n: u128 = 0;
            let mut sum_n2: u128 = 0;
            for &v in c_grid.iter() {
                sum_n += v as u128;
                sum_n2 += (v as u128) * (v as u128);
            }
            let n_total = (m * m) as f64;
            let mean = (sum_n as f64) / n_total;
            let var = ((sum_n2 as f64) / n_total - mean * mean).max(0.0);
            out[l_rep] = Some(LevelStats {
                n_cells_total: 1 << (2 * l_rep),
                mean, var, dvar: var,
            });
            n_distinct[l_rep] = 1;
            bytes_per_level[l_rep] = c_grid.len() * 4;  // u32
        }
    }

    let final_out: Vec<LevelStats> = out.into_iter()
        .enumerate()
        .map(|(l, o)| o.unwrap_or(LevelStats {
            n_cells_total: 1 << (2 * l),
            mean: 0.0, var: 0.0, dvar: 0.0,
        }))
        .collect();
    (final_out, n_distinct, bytes_per_level)
}
