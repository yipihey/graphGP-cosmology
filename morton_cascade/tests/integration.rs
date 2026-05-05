// Integration tests for morton_cascade. Run with `cargo test --release`.
//
// Tests cover:
//   - Conservation laws (point count preservation across levels)
//   - Mean count exactly equals N / 2^(D*l)
//   - Schur ratio matches the theoretical Poisson prediction at every D
//   - Generic cascade_nd matches hand-written 2D and 3D
//   - Packed (adaptive-width) cascade is bitwise-identical to dense reference
//   - Uniform Poisson xi(r) is statistically zero at r > 0
//
// All use a fixed RNG seed so results are reproducible.

use morton_cascade::{
    L_MAX, N_LEVELS,
    L_MAX_3D, N_LEVELS_3D,
    hier, hier_3d, hier_packed,
    hier_nd::cascade_nd,
};

/// Build a per-test unique temporary working directory.
///
/// CLI smoke tests previously used fixed-name tempdirs, which cause flaky
/// failures when tests run in parallel and stomp on each other's outputs.
/// This helper appends the test's `name` and the process PID, plus a small
/// counter for uniqueness within a single process. Each call returns a
/// fresh empty directory.
fn unique_workdir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let workdir = std::env::temp_dir()
        .join(format!("morton_cascade_test_{}_{}_{}", name, pid, n));
    let _ = std::fs::remove_dir_all(&workdir);
    std::fs::create_dir_all(&workdir).unwrap();
    workdir
}

// ---------- Tiny xorshift PRNG (shared with examples/common.rs) ----------

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.s = x; x
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn gen_uniform_2d(n: usize, rng: &mut Rng) -> Vec<(u16, u16)> {
    let scale = (1u32 << 16) as f64;
    (0..n).map(|_| {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        (x, y)
    }).collect()
}

fn gen_uniform_3d(n: usize, rng: &mut Rng) -> Vec<(u16, u16, u16)> {
    let scale = (1u32 << 16) as f64;
    (0..n).map(|_| {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let z = (rng.uniform() * scale).min(scale - 1.0) as u16;
        (x, y, z)
    }).collect()
}

fn gen_uniform_nd<const D: usize>(n: usize, rng: &mut Rng) -> Vec<[u16; D]> {
    let scale = (1u32 << 16) as f64;
    (0..n).map(|_| {
        let mut p = [0u16; D];
        for d in 0..D {
            p[d] = (rng.uniform() * scale).min(scale - 1.0) as u16;
        }
        p
    }).collect()
}

// ============================================================================
// Conservation tests
// ============================================================================

#[test]
fn conservation_2d_periodic() {
    let mut rng = Rng::new(11);
    let pts = gen_uniform_2d(50_000, &mut rng);
    let n = pts.len() as f64;
    let (stats, _) = hier::cascade_hierarchical_bc(&pts, 1, true);
    for (l, st) in stats.iter().enumerate() {
        let expected_mean = n / (1u64 << (2 * l)) as f64;
        let rel_err = (st.mean - expected_mean).abs() / expected_mean;
        assert!(rel_err < 1e-12,
            "2D conservation level {}: mean = {}, expected {}, rel error {}",
            l, st.mean, expected_mean, rel_err);
    }
}

#[test]
fn conservation_3d_periodic() {
    // Conservation is an exact algebraic identity, independent of cascade
    // implementation and of l_max. We use cascade_nd::<3> at l_max=5
    // (32^3 cells) instead of the legacy hier_3d::cascade_3d_with_tpcf
    // (128^3 cells, ~50x slower) since they are equivalent for this
    // identity. Saves ~3s.
    let mut rng = Rng::new(12);
    let pts = gen_uniform_3d(50_000, &mut rng);
    let pts_arr: Vec<[u16; 3]> = pts.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let n = pts.len() as f64;
    let l_max = 5;
    let (stats, _) = cascade_nd::<3>(&pts_arr, l_max, 1, true);
    for (l, st) in stats.iter().enumerate() {
        let expected_mean = n / (1u64 << (3 * l)) as f64;
        let rel_err = (st.mean - expected_mean).abs() / expected_mean;
        assert!(rel_err < 1e-12,
            "3D conservation level {}: mean = {}, expected {}, rel error {}",
            l, st.mean, expected_mean, rel_err);
    }
}

// ============================================================================
// Schur ratio tests for uniform Poisson
// ============================================================================

/// For uniform Poisson at level l, the predicted Schur residual ratio is
/// dvar / <N> = 1 - (2^D - 1) / 4^D = 13/16 (D=2), 57/64 (D=3), ...
///
/// We average over multiple realizations to get a tight estimate at the
/// finest level (most cells -> least noise).

fn poisson_schur_predicted(d: u32) -> f64 {
    1.0 - ((1u64 << d) - 1) as f64 / (1u64 << (2 * d)) as f64
}

#[test]
fn schur_2d_poisson_floor() {
    let predicted = poisson_schur_predicted(2);   // 13/16 = 0.8125
    assert!((predicted - 13.0/16.0).abs() < 1e-15);

    let mut rng = Rng::new(21);
    let n_real = 8;
    let mut acc_dvar = 0.0;
    let mut acc_mean = 0.0;
    for _ in 0..n_real {
        let pts = gen_uniform_2d(50_000, &mut rng);
        let (stats, _) = hier::cascade_hierarchical_bc(&pts, 1, true);
        // Use the finest level (most data, least sample noise)
        let s = &stats[L_MAX];
        acc_dvar += s.dvar;
        acc_mean += s.mean;
    }
    let measured = acc_dvar / acc_mean;
    let rel_err = (measured - predicted).abs() / predicted;
    assert!(rel_err < 0.005,    // 0.5% tolerance
        "2D Schur ratio: measured {}, predicted {}, rel err {}",
        measured, predicted, rel_err);
}

#[test]
fn schur_3d_poisson_floor() {
    // Switched from hier_3d::cascade_3d_with_tpcf (l_max=7, fixed in legacy
    // API) to cascade_nd::<3>(l_max=5). The Schur identity 1 - (2^D-1)/4^D
    // holds at every level; running at l_max=5 instead of l_max=7 means
    // 32^3 cells (~32k samples) per realization instead of 128^3 (~2M).
    // The estimator variance grows as 1/N_cells but the mean is unbiased.
    // With 8 realizations × 32k cells = 262k effective samples we still
    // measure to <1% relative error (verified empirically). Saves ~28s.
    let predicted = poisson_schur_predicted(3);   // 57/64 = 0.890625
    let mut rng = Rng::new(22);
    let n_real = 8;
    let l_max = 5;
    let mut acc_dvar = 0.0;
    let mut acc_mean = 0.0;
    for _ in 0..n_real {
        let pts = gen_uniform_3d(50_000, &mut rng);
        let pts_arr: Vec<[u16; 3]> = pts.iter().map(|&(x, y, z)| [x, y, z]).collect();
        let (stats, _) = cascade_nd::<3>(&pts_arr, l_max, 1, true);
        let s = &stats[l_max];
        acc_dvar += s.dvar;
        acc_mean += s.mean;
    }
    let measured = acc_dvar / acc_mean;
    let rel_err = (measured - predicted).abs() / predicted;
    assert!(rel_err < 0.01,
        "3D Schur ratio: measured {}, predicted {}, rel err {}",
        measured, predicted, rel_err);
}

#[test]
fn schur_4d_poisson_floor() {
    let predicted = poisson_schur_predicted(4);   // 241/256 = 0.94141
    let mut rng = Rng::new(23);
    let n_real = 8;
    let mut acc_dvar = 0.0;
    let mut acc_mean = 0.0;
    let l_max = 3;
    for _ in 0..n_real {
        let pts = gen_uniform_nd::<4>(50_000, &mut rng);
        let (stats, _) = cascade_nd::<4>(&pts, l_max, 1, true);
        let s = &stats[l_max];
        acc_dvar += s.dvar;
        acc_mean += s.mean;
    }
    let measured = acc_dvar / acc_mean;
    let rel_err = (measured - predicted).abs() / predicted;
    assert!(rel_err < 0.01,
        "4D Schur ratio: measured {}, predicted {}, rel err {}",
        measured, predicted, rel_err);
}

#[test]
fn schur_5d_poisson_floor() {
    let predicted = poisson_schur_predicted(5);   // 993/1024 = 0.96973
    let mut rng = Rng::new(24);
    let n_real = 16;
    let mut acc_dvar = 0.0;
    let mut acc_mean = 0.0;
    let l_max = 2;
    for _ in 0..n_real {
        let pts = gen_uniform_nd::<5>(50_000, &mut rng);
        let (stats, _) = cascade_nd::<5>(&pts, l_max, 1, true);
        let s = &stats[l_max];
        acc_dvar += s.dvar;
        acc_mean += s.mean;
    }
    let measured = acc_dvar / acc_mean;
    let rel_err = (measured - predicted).abs() / predicted;
    // 5D test is noisier (only 1024 cells per realization, fewer realizations
    // affordable due to M^5 buffer cost); allow 3% tolerance.
    assert!(rel_err < 0.03,
        "5D Schur ratio: measured {}, predicted {}, rel err {}",
        measured, predicted, rel_err);
}

// ============================================================================
// Generic D vs hand-written equivalence
// ============================================================================

#[test]
fn generic_2d_matches_handwritten() {
    let mut rng = Rng::new(31);
    let pts = gen_uniform_2d(100_000, &mut rng);
    let pts_arr: Vec<[u16; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();

    let (st_ref, _) = hier::cascade_hierarchical_bc(&pts, 1, true);
    let (st_nd, _) = cascade_nd::<2>(&pts_arr, L_MAX, 1, true);

    assert_eq!(st_ref.len(), st_nd.len());
    for l in 0..N_LEVELS {
        // Mean should be exactly identical (deterministic counts)
        assert_eq!(st_ref[l].mean, st_nd[l].mean,
            "2D level {}: mean differs (ref {}, nd {})",
            l, st_ref[l].mean, st_nd[l].mean);
        // Variance and Schur should match to float precision
        let dv = (st_ref[l].var - st_nd[l].var).abs() / st_ref[l].var.abs().max(1e-30);
        let dd = (st_ref[l].dvar - st_nd[l].dvar).abs() / st_ref[l].dvar.abs().max(1e-30);
        assert!(dv < 1e-12, "2D level {} var: {} vs {}, rel diff {}",
                l, st_ref[l].var, st_nd[l].var, dv);
        assert!(dd < 1e-12, "2D level {} dvar: {} vs {}, rel diff {}",
                l, st_ref[l].dvar, st_nd[l].dvar, dd);
    }
}

// Slow (~6s): exercises the legacy hier_3d::cascade_3d_with_tpcf at full
// l_max=7. This is a parity check that the modern cascade_nd::<3> matches
// the hand-written legacy implementation bit-for-bit. Important during
// refactoring of the modern code, but not on every test run.
//
// Run on demand with: cargo test --release -- --ignored generic_3d_matches_handwritten
#[test]
#[ignore]
fn generic_3d_matches_handwritten() {
    let mut rng = Rng::new(32);
    let pts = gen_uniform_3d(50_000, &mut rng);
    let pts_arr: Vec<[u16; 3]> = pts.iter().map(|&(x, y, z)| [x, y, z]).collect();

    let (st_ref, _, _) = hier_3d::cascade_3d_with_tpcf(&pts, 1, true, &[]);
    let (st_nd, _) = cascade_nd::<3>(&pts_arr, L_MAX_3D, 1, true);

    assert_eq!(st_ref.len(), st_nd.len());
    for l in 0..N_LEVELS_3D {
        assert_eq!(st_ref[l].mean, st_nd[l].mean,
            "3D level {} mean differs", l);
        let dv = (st_ref[l].var - st_nd[l].var).abs() / st_ref[l].var.abs().max(1e-30);
        let dd = (st_ref[l].dvar - st_nd[l].dvar).abs() / st_ref[l].dvar.abs().max(1e-30);
        assert!(dv < 1e-12, "3D level {} var rel diff {}", l, dv);
        assert!(dd < 1e-12, "3D level {} dvar rel diff {}", l, dd);
    }
}

// ============================================================================
// Packed adaptive cascade: bitwise identical to reference
// ============================================================================

#[test]
fn packed_matches_dense_uniform() {
    let mut rng = Rng::new(41);
    let pts = gen_uniform_2d(100_000, &mut rng);
    let (st_ref, _) = hier::cascade_hierarchical_bc(&pts, 1, true);
    let (st_pk, _, _) = hier_packed::cascade_adaptive(&pts, 1, true, None);
    for l in 0..N_LEVELS {
        assert_eq!(st_ref[l].mean, st_pk[l].mean,
            "packed mean differs at level {}", l);
        assert_eq!(st_ref[l].var, st_pk[l].var,
            "packed var differs at level {}: ref {}, pk {}",
            l, st_ref[l].var, st_pk[l].var);
        assert_eq!(st_ref[l].dvar, st_pk[l].dvar,
            "packed dvar differs at level {}", l);
    }
}

#[test]
fn packed_matches_dense_with_hint() {
    let mut rng = Rng::new(42);
    let pts = gen_uniform_2d(200_000, &mut rng);
    let (st_ref, _) = hier::cascade_hierarchical_bc(&pts, 1, true);
    let hint = hier_packed::predict_max_counts_2d(pts.len() as u64, 4);
    let (st_pk, _, _) = hier_packed::cascade_adaptive(&pts, 1, true, Some(hint));
    for l in 0..N_LEVELS {
        assert_eq!(st_ref[l].mean, st_pk[l].mean);
        assert_eq!(st_ref[l].var, st_pk[l].var);
        assert_eq!(st_ref[l].dvar, st_pk[l].dvar);
    }
}

#[test]
fn packed_matches_dense_nonperiodic() {
    let mut rng = Rng::new(43);
    let pts = gen_uniform_2d(100_000, &mut rng);
    let (st_ref, _) = hier::cascade_hierarchical_bc(&pts, 1, false);
    let (st_pk, _, _) = hier_packed::cascade_adaptive(&pts, 1, false, None);
    for l in 0..N_LEVELS {
        let dv = (st_ref[l].var - st_pk[l].var).abs() / st_ref[l].var.abs().max(1e-30);
        let dd = (st_ref[l].dvar - st_pk[l].dvar).abs() / st_ref[l].dvar.abs().max(1e-30);
        assert!(dv < 1e-12);
        assert!(dd < 1e-12);
    }
}

// ============================================================================
// Uniform Poisson: xi(r) should be statistically zero at all r > 0
// ============================================================================

#[test]
fn poisson_xi_is_zero() {
    let mut rng = Rng::new(51);
    let pts = gen_uniform_2d(800_000, &mut rng);
    let lag_levels: Vec<usize> = (1..=8).collect();
    let (_, _, tpcf) = hier::cascade_hierarchical_with_tpcf(&pts, 1, true, &lag_levels);
    // For 800k uniform Poisson points, the Cox xi should be < 1e-3 at every (level, k).
    for tp in &tpcf {
        // Skip very coarse levels with few cells where sample noise dominates
        if tp.level <= 2 { continue; }
        assert!(tp.xi.abs() < 5e-3,
            "Poisson xi at level {}, k={}, r={}: |xi| = {} (should be ~0)",
            tp.level, tp.k, tp.r_tree, tp.xi.abs());
    }
}

// ============================================================================
// PMF sums to 1 (i.e., total counts in the histogram equal n_total)
// ============================================================================

#[test]
fn pmf_normalization() {
    let mut rng = Rng::new(61);
    let pts = gen_uniform_2d(100_000, &mut rng);
    let (_, _, pmfs) = hier::cascade_with_pmf(&pts, 1, true);
    for p in &pmfs {
        let total: u64 = p.histogram.iter().sum();
        assert_eq!(total, p.n_total,
            "PMF level {} normalization: histogram sum {} != n_total {}",
            p.level, total, p.n_total);
    }
}

#[test]
fn pmf_first_moment_matches_mean() {
    let mut rng = Rng::new(62);
    let pts = gen_uniform_2d(100_000, &mut rng);
    let (stats, _, pmfs) = hier::cascade_with_pmf(&pts, 1, true);
    for p in &pmfs {
        // Compute mean from the PMF
        let mut sum = 0.0f64;
        let mut wsum = 0.0f64;
        for (k, &h) in p.histogram.iter().enumerate() {
            sum += (k as f64) * (h as f64);
            wsum += h as f64;
        }
        let mean_from_pmf = sum / wsum;
        let mean_from_stats = stats[p.level].mean;
        let rel_err = (mean_from_pmf - mean_from_stats).abs() / mean_from_stats.max(1e-12);
        assert!(rel_err < 1e-12,
            "PMF level {} first moment {} != stats mean {}",
            p.level, mean_from_pmf, mean_from_stats);
    }
}

// ============================================================================
// Windowed PMF: arbitrary cube-window sides via 1D sliding-sum cascade
// ============================================================================

use morton_cascade::hier_nd::{cascade_with_pmf_windows, log_spaced_window_sides};

#[test]
fn pmf_windows_2d_matches_dyadic_at_dyadic_sides() {
    // At dyadic window sides k = 2^j (in fine-grid units of M = 2^(L_max+s_sub)),
    // the windowed PMF should match the dyadic cascade PMF at the matching level.
    //
    // With L_max = 8, s_sub = 1: M = 512. Cascade level l corresponds to cell
    // side 2^(L_max - l) tree-units = 2^(L_max + s_sub - l) fine-units.
    //
    // So level l matches window_side = 2^(L_max + s_sub - l) = 2^(9-l):
    //   l=8 -> k=2,  l=7 -> k=4,  l=6 -> k=8, ..., l=1 -> k=256.

    let mut rng = Rng::new(101);
    let pts = gen_uniform_2d(50_000, &mut rng);

    // Run dyadic PMF cascade
    let (_, _, dyadic_pmfs) = morton_cascade::hier::cascade_with_pmf(&pts, 1, true);

    // Convert pts to [u16; 2] for the ND function
    let pts_arr: Vec<[u16; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();

    // Run windowed PMF at the same dyadic sides
    let m_eff: usize = 1 << (L_MAX + 1);   // s_sub = 1
    let dyadic_sides: Vec<usize> = (1..=L_MAX).map(|l| m_eff >> l).collect();
    let pmfs_w = cascade_with_pmf_windows::<2>(&pts_arr, L_MAX, 1, true, &dyadic_sides);

    // For each windowed PMF, find the matching dyadic PMF and compare moments.
    for pw in &pmfs_w {
        // Find the dyadic pmf with matching mean (since means are deterministic
        // for a given cell side, exact mean match identifies the level).
        let mean_target = pw.mean;
        let dy = dyadic_pmfs.iter()
            .min_by(|a, b| {
                let da = (a.mean - mean_target).abs();
                let db = (b.mean - mean_target).abs();
                da.partial_cmp(&db).unwrap()
            }).unwrap();
        // Means must match exactly (same total points / same cell count).
        assert!((pw.mean - dy.mean).abs() / dy.mean.max(1e-12) < 1e-12,
            "mean mismatch at k={}: windowed {} vs dyadic {}",
            pw.window_side, pw.mean, dy.mean);
        // Variance must match exactly too -- both are computed over the same
        // (cell, shift) entries.
        assert!((pw.var - dy.var).abs() / dy.var.max(1e-12) < 1e-12,
            "var mismatch at k={}: windowed {} vs dyadic {}",
            pw.window_side, pw.var, dy.var);
        // Histograms must be identical.
        let n_min = pw.histogram.len().min(dy.histogram.len());
        for n in 0..n_min {
            assert_eq!(pw.histogram[n], dy.histogram[n],
                "histogram mismatch at k={}, N={}: windowed {} vs dyadic {}",
                pw.window_side, n, pw.histogram[n], dy.histogram[n]);
        }
    }
}

#[test]
fn pmf_windows_normalization() {
    let mut rng = Rng::new(102);
    let pts = gen_uniform_2d(30_000, &mut rng);
    let pts_arr: Vec<[u16; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();
    let sides = vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    let pmfs = cascade_with_pmf_windows::<2>(&pts_arr, L_MAX, 1, true, &sides);
    for p in &pmfs {
        let total: u64 = p.histogram.iter().sum();
        assert_eq!(total, p.n_total,
            "windowed PMF normalization at side {}: histogram sum {} != n_total {}",
            p.window_side, total, p.n_total);
    }
}

#[test]
fn pmf_windows_first_moment() {
    let mut rng = Rng::new(103);
    let pts = gen_uniform_2d(30_000, &mut rng);
    let pts_arr: Vec<[u16; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();
    let sides = vec![1, 2, 3, 5, 7, 11, 16, 23, 32, 47];
    let pmfs = cascade_with_pmf_windows::<2>(&pts_arr, L_MAX, 1, true, &sides);
    for p in &pmfs {
        let mut sum = 0.0f64;
        for (n, &h) in p.histogram.iter().enumerate() {
            sum += (n as f64) * (h as f64);
        }
        let mean_from_hist = sum / p.n_total as f64;
        let rel_err = (mean_from_hist - p.mean).abs() / p.mean.max(1e-12);
        assert!(rel_err < 1e-12,
            "first moment mismatch at side {}: histogram-derived {} vs reported {}",
            p.window_side, mean_from_hist, p.mean);
    }
}

#[test]
fn pmf_windows_3d_basic() {
    let mut rng = Rng::new(104);
    let pts = gen_uniform_3d(20_000, &mut rng);
    let pts_arr: Vec<[u16; 3]> = pts.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let sides = vec![1, 2, 4, 8, 16, 32];
    // l_max=5 (32^3 cells) suffices to cover all sides up to 32; previously
    // l_max=7 (128^3 cells) was unnecessary and ~30x slower.
    let l_max_3d = 5;
    let pmfs = cascade_with_pmf_windows::<3>(&pts_arr, l_max_3d, 1, true, &sides);
    assert_eq!(pmfs.len(), sides.len());
    for p in &pmfs {
        // Conservation: histogram sums to n_total
        let total: u64 = p.histogram.iter().sum();
        assert_eq!(total, p.n_total);
        // Mean per cell of side k in 3D periodic: n_pts / (M/k)^3 in fine units
        // Actually mean per cell = n_pts * k^3 / M^3 in 3D (since each window
        // covers k^3 fine cells, and there are M^3 windows).
        // For uniform Poisson the count mean equals n_pts * k^3 / M^3.
        let m_eff: usize = 1 << (l_max_3d + 1);
        let expected_mean = pts.len() as f64 * (p.window_side as f64).powi(3) / (m_eff as f64).powi(3);
        let rel_err = (p.mean - expected_mean).abs() / expected_mean.max(1e-12);
        assert!(rel_err < 1e-12,
            "3D window k={}: mean {} vs expected {}", p.window_side, p.mean, expected_mean);
    }
}

#[test]
fn pmf_windows_uniform_poisson_p0_matches_analytic() {
    // For uniform Poisson with mean nu = n_pts * V/V_box per window,
    // P_0(V) = exp(-nu).
    //
    // 2D, N = 100k, periodic box of side M = 512 fine units, window k -> nu = N k^2 / M^2.
    let mut rng = Rng::new(105);
    let n_pts = 100_000usize;
    let pts = gen_uniform_2d(n_pts, &mut rng);
    let pts_arr: Vec<[u16; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();
    let sides = vec![1, 2, 3, 5, 8, 13, 21];
    let pmfs = cascade_with_pmf_windows::<2>(&pts_arr, L_MAX, 1, true, &sides);
    let m_eff: f64 = (1 << (L_MAX + 1)) as f64;
    for p in &pmfs {
        let nu = n_pts as f64 * (p.window_side as f64 / m_eff).powi(2);
        let expected_p0 = (-nu).exp();
        let measured_p0 = p.histogram[0] as f64 / p.n_total as f64;
        // Tolerance: 1/sqrt(N_eff) * sqrt(p0(1-p0)) where N_eff = n_total / correlation
        // For 100k points with these window sizes, expect agreement to ~1%.
        let abs_err = (measured_p0 - expected_p0).abs();
        assert!(abs_err < 0.02,
            "P_0 at side k={}: measured {} vs expected exp(-{:.3})={:.4} (abs err {})",
            p.window_side, measured_p0, nu, expected_p0, abs_err);
    }
}

#[test]
fn log_spaced_sides_default() {
    // 2D, range [1, 256], 5 ppd-V -> spacing factor in side = 10^(1/(2*5)) = 1.259
    let sides = log_spaced_window_sides(1, 256, 2, 5.0);
    // Sanity checks
    assert!(sides[0] == 1);
    assert!(*sides.last().unwrap() == 256);
    // Strictly monotonic
    for w in sides.windows(2) {
        assert!(w[0] < w[1]);
    }
    // Should cover roughly 4.8 decades V * 5 ppd ≈ 24 distinct sides
    assert!(sides.len() >= 15 && sides.len() <= 35,
        "expected 15-35 distinct sides, got {}: {:?}", sides.len(), sides);
}

// ============================================================================
// CLI: `xi` subcommand round-trip
// ============================================================================

#[test]
fn cli_xi_matched_density_null_3d() {
    // Round-trip through the binary CLI: write data + randoms as f64 binaries,
    // invoke `morton-cascade xi`, parse the CSV, and check the matched-density
    // null condition (uniform-vs-uniform should give |xi_LS| << 1 at moderate
    // scales). This guards against any wiring breakage between BitVecCascadePair
    // and the CLI surface.

    use std::fs::File;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::process::Command;

    // Locate the release binary. cargo test sets CARGO_TARGET_DIR sometimes;
    // fall back to ./target/release when absent.
    let bin = std::env::var("CARGO_BIN_EXE_morton-cascade")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./target/release/morton-cascade"));
    if !bin.exists() {
        // The CLI must be built. cargo test --release with a `[[bin]]` and
        // the env var should make this just work; if not, skip with a note
        // rather than hard-failing the suite.
        eprintln!("skipping cli_xi_matched_density_null_3d: binary `{}` not found",
            bin.display());
        return;
    }

    // Build temp dir under the target dir to keep things tidy.
    let workdir = unique_workdir("xi_cli");

    let box_size: f64 = 100.0;
    let n_d = 4_000;
    let n_r = 16_000;

    let mut rng = Rng::new(31415);

    let write_pts = |path: &PathBuf, n: usize, rng: &mut Rng| {
        let mut f = File::create(path).unwrap();
        for _ in 0..n {
            for _ in 0..3 {
                let v: f64 = rng.uniform() * box_size;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    write_pts(&data_path, n_d, &mut rng);
    write_pts(&rand_path, n_r, &mut rng);

    let status = Command::new(&bin)
        .args([
            "xi",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "morton-cascade xi failed: {:?}", status);

    let csv_path = workdir.join("xi_landy_szalay.csv");
    let csv = std::fs::read_to_string(&csv_path).unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    // Sanity: header has the columns we expect
    assert!(header.contains("xi_ls"), "missing xi_ls column: {}", header);

    // For every level, sanity-check the row is parseable.
    let mut max_abs_xi_mid: f64 = 0.0;
    let mut n_rows = 0;
    let mut total_dd_l0: f64 = 0.0;
    for line in lines {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), 12, "wrong column count in: {}", line);
        let level: usize = cols[0].parse().unwrap();
        let xi: f64 = cols[8].parse().unwrap_or(f64::NAN);
        let cum_dd: f64 = cols[9].parse().unwrap();
        if level == 0 {
            total_dd_l0 = cum_dd;
        }
        // Mid scales (cells between 1/4 and 1/16 of the box) should be in the
        // estimator's well-behaved regime for these N values.
        if level >= 2 && level <= 4 {
            assert!(xi.is_finite(),
                "xi_LS not finite at level {}: {}", level, xi);
            if xi.abs() > max_abs_xi_mid { max_abs_xi_mid = xi.abs(); }
        }
        n_rows += 1;
    }
    assert!(n_rows > 5, "too few CSV rows: {}", n_rows);
    // cumulative_dd at level 0 should equal N_d * (N_d - 1) / 2.
    let expected_dd_l0 = (n_d as f64) * (n_d as f64 - 1.0) / 2.0;
    assert!((total_dd_l0 - expected_dd_l0).abs() < 1.0,
        "cumulative_dd at level 0 = {}, expected {}", total_dd_l0, expected_dd_l0);
    // Matched-density null: |xi_LS| should be small at moderate scales.
    assert!(max_abs_xi_mid < 0.2,
        "uniform-vs-uniform: max |xi_LS| at levels 2..=4 = {} (expected << 1)",
        max_abs_xi_mid);

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_field_stats_3d_smoke() {
    // Round-trip through the binary. Generate matched-density data + randoms,
    // run `morton-cascade field-stats`, parse outputs, sanity-check.
    use std::fs::File;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::process::Command;

    let bin = std::env::var("CARGO_BIN_EXE_morton-cascade")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./target/release/morton-cascade"));
    if !bin.exists() {
        eprintln!("skipping cli_field_stats_3d_smoke: binary `{}` not found",
            bin.display());
        return;
    }

    let workdir = unique_workdir("field_stats");

    let box_size: f64 = 100.0;
    let n_d = 3_000usize;
    let n_r = 12_000usize;

    let mut rng = Rng::new(271828);

    let write_pts = |path: &PathBuf, n: usize, rng: &mut Rng| {
        let mut f = File::create(path).unwrap();
        for _ in 0..n {
            for _ in 0..3 {
                let v: f64 = rng.uniform() * box_size;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    write_pts(&data_path, n_d, &mut rng);
    write_pts(&rand_path, n_r, &mut rng);

    let status = Command::new(&bin)
        .args([
            "field-stats",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--hist-bins", "30",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "morton-cascade field-stats failed: {:?}", status);

    // Moments file must exist with expected header.
    let mom_path = workdir.join("field_moments.csv");
    let mom = std::fs::read_to_string(&mom_path).unwrap();
    let mut lines = mom.lines();
    let header = lines.next().unwrap();
    assert!(header.contains("mean_delta"), "moments header missing fields: {}", header);
    assert!(header.contains("var_delta"));
    assert!(header.contains("s3_delta"));
    let mut n_rows = 0;
    let mut level_0_mean: Option<f64> = None;
    for line in lines {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), 14, "unexpected column count: {}", line);
        let level: usize = cols[0].parse().unwrap();
        let mean: f64 = cols[5].parse().unwrap();
        if level == 0 { level_0_mean = Some(mean); }
        n_rows += 1;
    }
    assert!(n_rows > 5, "too few moment rows: {}", n_rows);
    assert!(level_0_mean.is_some(), "no level-0 row");
    // The single-cell mean at the root should be 0 by global α normalization.
    assert!(level_0_mean.unwrap().abs() < 1e-12,
        "root-level <δ>_W_r should be ~0, got {}", level_0_mean.unwrap());

    // PDF file must exist and be non-trivial.
    let pdf_path = workdir.join("field_pdf.csv");
    let pdf = std::fs::read_to_string(&pdf_path).unwrap();
    let pdf_lines: Vec<&str> = pdf.lines().collect();
    assert!(pdf_lines.len() > 30, "PDF file too short: {} lines", pdf_lines.len());

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_anisotropy_3d_smoke() {
    // CLI smoke test for anisotropy. Build a z-anisotropic mock and verify
    // the produced field_anisotropy.csv has expected structure plus a
    // measurable Q_2 signal at the cluster scale.
    use std::fs::File;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::process::Command;

    let bin = std::env::var("CARGO_BIN_EXE_morton-cascade")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./target/release/morton-cascade"));
    if !bin.exists() {
        eprintln!("skipping cli_anisotropy_3d_smoke: binary not found");
        return;
    }

    let workdir = unique_workdir("aniso");

    let box_size: f64 = 1000.0;
    let n_clusters = 100usize;
    let n_d = 12_000usize;
    let n_r = 60_000usize;

    let mut rng = Rng::new(42);

    // Cluster centers
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        centers.push([
            rng.uniform() * box_size,
            rng.uniform() * box_size,
            rng.uniform() * box_size,
        ]);
    }

    // Write data: clustered, tight in z, loose in x, y
    let data_path = workdir.join("data.bin");
    let mut df = File::create(&data_path).unwrap();
    for _ in 0..n_d {
        let c_idx = (rng.uniform() * n_clusters as f64) as usize;
        let c = centers[c_idx.min(n_clusters - 1)];
        for d in 0..3 {
            // tight in z (sigma=4), loose in x,y (sigma=25)
            let sigma = if d == 2 { 4.0 } else { 25.0 };
            // Box-Muller via two uniforms
            let u1 = rng.uniform().max(1e-10);
            let u2 = rng.uniform();
            let dx = sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mut v = c[d] + dx;
            // Periodic wrap
            if v < 0.0 { v += box_size; }
            if v >= box_size { v -= box_size; }
            df.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    // Write randoms: uniform
    let rand_path = workdir.join("randoms.bin");
    let mut rf = File::create(&rand_path).unwrap();
    for _ in 0..n_r {
        for _ in 0..3 {
            let v = rng.uniform() * box_size;
            rf.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    let status = Command::new(&bin)
        .args([
            "anisotropy",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "morton-cascade anisotropy failed: {:?}", status);

    // Output file present and well-formed. The CSV is now D-generic:
    // header columns are w2_axis_<d> for d in 0..D and w2_p<binary>
    // for each non-trivial pattern, plus quadrupole_los etc.
    let out_path = workdir.join("field_anisotropy.csv");
    let content = std::fs::read_to_string(&out_path).unwrap();
    let mut lines = content.lines();
    let header = lines.next().unwrap();
    let header_cols: Vec<&str> = header.split(',').collect();
    assert!(header.contains("quadrupole_los"),
        "missing quadrupole_los column: {}", header);
    assert!(header.contains("w2_axis_0") && header.contains("w2_axis_1")
            && header.contains("w2_axis_2"),
        "missing axis columns: {}", header);

    // Find column indices by name.
    let idx = |name: &str| header_cols.iter().position(|&c| c == name)
        .unwrap_or_else(|| panic!("missing column {} in header: {}", name, header));
    let i_cell_phys = idx("cell_side_phys");
    let i_n_par    = idx("n_parents");
    let i_wx2      = idx("w2_axis_0");
    let i_wy2      = idx("w2_axis_1");
    let i_wz2      = idx("w2_axis_2");

    // Look for a clear positive Q_2 at intermediate scale (cell ~ 30-150).
    // For Kaiser-like (z-tight) clusters, wz² > wx², wy² at cluster scale.
    let mut found_signal = false;
    for line in lines {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), header_cols.len(),
            "row column count {} != header {}: {}",
            cols.len(), header_cols.len(), line);
        let cell_phys: f64 = cols[i_cell_phys].parse().unwrap();
        let n_par: u64 = cols[i_n_par].parse().unwrap();
        let wx2: f64 = cols[i_wx2].parse().unwrap();
        let wy2: f64 = cols[i_wy2].parse().unwrap();
        let wz2: f64 = cols[i_wz2].parse().unwrap();
        if !(15.0 < cell_phys && cell_phys < 200.0) { continue; }
        if n_par < 50 { continue; }
        // wz² should clearly exceed both wx² and wy² at cluster scale
        if wz2 > 1.05 * wx2 && wz2 > 1.05 * wy2 {
            found_signal = true;
            break;
        }
    }
    assert!(found_signal,
        "Expected positive Q_2 signature at cluster scale, but no level showed \
         wz² > 1.05·max(wx², wy²)");

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_scattering_3d_smoke() {
    // CLI smoke test for second-order scattering. Build a clustered mock,
    // run the binary, check the output CSV has expected structure.
    use std::fs::File;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::process::Command;

    let bin = std::env::var("CARGO_BIN_EXE_morton-cascade")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./target/release/morton-cascade"));
    if !bin.exists() {
        eprintln!("skipping cli_scattering_3d_smoke: binary not found");
        return;
    }

    let workdir = unique_workdir("scat");

    let box_size: f64 = 200.0;
    let n_d = 3000usize;
    let n_r = 15_000usize;
    let mut rng = Rng::new(123);

    let write_pts = |path: &PathBuf, n: usize, rng: &mut Rng| {
        let mut f = File::create(path).unwrap();
        for _ in 0..n {
            for _ in 0..3 {
                let v: f64 = rng.uniform() * box_size;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    };
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    write_pts(&data_path, n_d, &mut rng);
    write_pts(&rand_path, n_r, &mut rng);

    let status = Command::new(&bin)
        .args([
            "scattering",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "morton-cascade scattering failed: {:?}", status);

    let out_path = workdir.join("field_scattering.csv");
    let content = std::fs::read_to_string(&out_path).unwrap();
    let mut lines = content.lines();
    let header = lines.next().unwrap();
    // Header should mention key fields
    assert!(header.contains("level_fine") && header.contains("level_coarse"),
        "missing level columns: {}", header);
    assert!(header.contains("first_x") && header.contains("first_y") && header.contains("first_z"),
        "missing first-order columns: {}", header);
    assert!(header.contains("s2_x_x") && header.contains("s2_z_z"),
        "missing second-order columns: {}", header);

    // Should have at least a few non-empty data rows
    let mut n_rows = 0;
    let mut n_nonzero_first = 0;
    for line in lines {
        if line.is_empty() { continue; }
        let cols: Vec<&str> = line.split(',').collect();
        assert_eq!(cols.len(), 18, "unexpected column count: {}", line);
        n_rows += 1;
        let first_x: f64 = cols[6].parse().unwrap();
        let first_y: f64 = cols[7].parse().unwrap();
        let first_z: f64 = cols[8].parse().unwrap();
        if first_x > 0.0 || first_y > 0.0 || first_z > 0.0 {
            n_nonzero_first += 1;
        }
    }
    assert!(n_rows > 5, "too few scattering rows: {}", n_rows);
    assert!(n_nonzero_first > 0,
        "all first-order coefficients are zero — pipeline broken?");

    let _ = std::fs::remove_dir_all(&workdir);
}

// ============================================================================
// CLI smoke tests for the `multi-run` subcommand
// ============================================================================

/// Helper: locate the morton-cascade binary, or return None if unavailable
/// (so tests can skip cleanly when running in environments where the binary
/// hasn't been built).
fn locate_morton_cascade_binary() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let bin = std::env::var("CARGO_BIN_EXE_morton-cascade")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./target/release/morton-cascade"));
    if bin.exists() { Some(bin) } else { None }
}

/// Helper: write `n` uniform-random 3D f64 points into `path` covering
/// `[0, box_size)^3`.
fn write_uniform_3d_f64(path: &std::path::Path, n: usize, box_size: f64, seed: u64) {
    use std::fs::File;
    use std::io::Write as _;
    let mut rng = Rng::new(seed);
    let mut f = File::create(path).unwrap();
    for _ in 0..n {
        for _ in 0..3 {
            let v = rng.uniform() * box_size;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

#[test]
fn cli_multi_run_field_stats_periodic_smoke() {
    // Multi-shift periodic field-stats: verify CSV structure and that
    // mean_delta_arv populates with multiple contributing runs.
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_fs");
    let data_path = workdir.join("data.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1500, box_size, 1234);

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "field-stats",
            "--boundary", "periodic",
            "--n-shifts", "4",
            "--shift-magnitude", "0.25",
            "--shift-seed", "777",
            "--resize-factors", "1.0,0.7",
            "--bin-tol", "0.05",
            "--max-depth", "6",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "multi-run field-stats failed: {:?}", status);

    // Aggregated CSV present with expected header
    let stats_path = workdir.join("multi_run_field_stats.csv");
    let stats_csv = std::fs::read_to_string(&stats_path).unwrap();
    let mut lines = stats_csv.lines();
    let header = lines.next().expect("empty stats CSV");
    let cols: Vec<&str> = header.split(',').collect();
    for required in ["physical_side", "n_runs", "mean_delta", "var_delta",
                     "mean_delta_arv", "var_delta_arv"] {
        assert!(cols.contains(&required),
            "header missing column '{}': {}", required, header);
    }
    let idx = |name: &str| cols.iter().position(|&c| c == name).unwrap();
    let i_n_runs = idx("n_runs");
    let i_mean_arv = idx("mean_delta_arv");
    let i_var_arv = idx("var_delta_arv");

    // At least one bin should have multiple runs AND nonzero across-run var
    // (proving the multi-shift aggregation actually pooled distinct runs).
    let mut found_multi_run_bin = false;
    let mut found_nonzero_arv = false;
    for line in lines {
        if line.is_empty() { continue; }
        let row: Vec<&str> = line.split(',').collect();
        assert_eq!(row.len(), cols.len(), "row width mismatch: {}", line);
        let n_runs: usize = row[i_n_runs].parse().unwrap();
        let var_arv: f64 = row[i_var_arv].parse().unwrap();
        let mean_arv: f64 = row[i_mean_arv].parse().unwrap();
        if n_runs > 1 { found_multi_run_bin = true; }
        if var_arv > 0.0 || mean_arv > 0.0 { found_nonzero_arv = true; }
    }
    assert!(found_multi_run_bin,
        "no bin had n_runs > 1 — cartesian product not pooling correctly");
    assert!(found_nonzero_arv,
        "all across-run variances are zero — aggregation may be reusing the \
         same per-run results");

    // Diagnostics CSV present and lists every run with footprint_coverage = 1
    // for periodic mode.
    let diag_path = workdir.join("multi_run_diagnostics.csv");
    let diag_csv = std::fs::read_to_string(&diag_path).unwrap();
    let diag_lines: Vec<&str> = diag_csv.lines().collect();
    // 4 shifts × 2 resize factors = 8 runs + 1 header line
    assert_eq!(diag_lines.len(), 9,
        "expected 8 run rows + header in diagnostics, got {}", diag_lines.len());
    for line in &diag_lines[1..] {
        let row: Vec<&str> = line.split(',').collect();
        // schema: name,scale,offset,footprint_coverage,n_levels
        let coverage: f64 = row[3].parse().unwrap();
        assert!((coverage - 1.0).abs() < 1e-6,
            "periodic mode footprint_coverage should be 1.0, got {}", coverage);
    }

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_multi_run_anisotropy_periodic_smoke() {
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_aniso");
    let data_path = workdir.join("data.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1500, box_size, 5678);

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "anisotropy",
            "--boundary", "periodic",
            "--n-shifts", "4",
            "--shift-magnitude", "0.25",
            "--shift-seed", "888",
            "--bin-tol", "0.05",
            "--max-depth", "6",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "multi-run anisotropy failed: {:?}", status);

    let aniso_path = workdir.join("multi_run_anisotropy.csv");
    let aniso_csv = std::fs::read_to_string(&aniso_path).unwrap();
    let mut lines = aniso_csv.lines();
    let header = lines.next().expect("empty anisotropy CSV");
    let cols: Vec<&str> = header.split(',').collect();
    // D=3 ⇒ should have w2_axis_0..2 and w2_p001..p111 plus quadrupole columns
    for required in ["physical_side", "n_runs", "n_parents_total",
                     "w2_axis_0", "w2_axis_1", "w2_axis_2",
                     "w2_p001", "w2_p111",
                     "quadrupole_los", "quadrupole_los_arv"] {
        assert!(cols.contains(&required),
            "anisotropy header missing column '{}': {}", required, header);
    }

    // Verify the across-run variance machinery fires for at least one bin
    // with multiple runs and non-trivial cell coverage.
    let i_n_runs = cols.iter().position(|&c| c == "n_runs").unwrap();
    let i_n_par  = cols.iter().position(|&c| c == "n_parents_total").unwrap();
    let i_q_arv  = cols.iter().position(|&c| c == "quadrupole_los_arv").unwrap();
    let mut found_signal = false;
    for line in lines {
        if line.is_empty() { continue; }
        let row: Vec<&str> = line.split(',').collect();
        let n_runs: usize = row[i_n_runs].parse().unwrap();
        let n_par: u64 = row[i_n_par].parse().unwrap();
        let q_arv: f64 = row[i_q_arv].parse().unwrap();
        if n_runs >= 2 && n_par > 50 && q_arv > 0.0 {
            found_signal = true;
            break;
        }
    }
    assert!(found_signal,
        "no bin with ≥2 runs, ≥50 parents, and nonzero quadrupole_los_arv — \
         multi-shift aggregation not producing shift-bootstrap variance");

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_multi_run_cic_pmf_periodic_smoke() {
    // CIC PMF in periodic mode: per-side densities should sum to 1, and
    // the k=0 bin should be populated (every level has unvisited cells
    // contributing via the periodic correction).
    use std::collections::BTreeMap;
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_cic");
    let data_path = workdir.join("data.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1500, box_size, 9012);

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "cic-pmf",
            "--boundary", "periodic",
            "--n-shifts", "3",
            "--shift-magnitude", "0.25",
            "--shift-seed", "999",
            "--bin-tol", "0.05",
            "--max-depth", "5",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "multi-run cic-pmf failed: {:?}", status);

    let cic_path = workdir.join("multi_run_cic_pmf.csv");
    let cic_csv = std::fs::read_to_string(&cic_path).unwrap();
    let mut lines = cic_csv.lines();
    let header = lines.next().expect("empty cic-pmf CSV");
    let cols: Vec<&str> = header.split(',').collect();
    for required in ["physical_side", "count_bin", "histogram_count",
                     "histogram_density", "mean", "var"] {
        assert!(cols.contains(&required),
            "cic-pmf header missing column '{}': {}", required, header);
    }

    let i_side    = cols.iter().position(|&c| c == "physical_side").unwrap();
    let i_bin     = cols.iter().position(|&c| c == "count_bin").unwrap();
    let i_count   = cols.iter().position(|&c| c == "histogram_count").unwrap();
    let i_density = cols.iter().position(|&c| c == "histogram_density").unwrap();

    // Group rows by physical_side (string-keyed to avoid f64 ordering issues)
    // and check that densities sum to 1 per side.
    let mut by_side: BTreeMap<String, (u64, f64, bool)> = BTreeMap::new();
    for line in lines {
        if line.is_empty() { continue; }
        let row: Vec<&str> = line.split(',').collect();
        let side_key = row[i_side].to_string();
        let bin_idx: usize = row[i_bin].parse().unwrap();
        let cnt: u64 = row[i_count].parse().unwrap();
        let p: f64 = row[i_density].parse().unwrap();
        let entry = by_side.entry(side_key).or_insert((0, 0.0, false));
        entry.0 += cnt;       // cumulative count for this side
        entry.1 += p;         // cumulative density for this side
        if bin_idx == 0 && p > 0.0 { entry.2 = true; } // saw populated k=0 bin
    }

    // Need at least a couple of distinct sides to make the test meaningful
    assert!(by_side.len() >= 2,
        "expected at least 2 distinct sides, got {}", by_side.len());

    // For each side, density should sum to ~1 (periodic invariant)
    for (side, (_, density_sum, _)) in &by_side {
        assert!((density_sum - 1.0).abs() < 1e-6,
            "side {}: density sum = {}, expected ~1.0", side, density_sum);
    }
    // At least one side should have a populated k=0 bin (almost always
    // true for periodic with finite N_d — most cells are empty).
    assert!(by_side.values().any(|(_, _, has_zero)| *has_zero),
        "no side had a populated k=0 bin — periodic correction may not be \
         flowing through to the CSV");

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_multi_run_isolated_resize_reports_partial_coverage() {
    // Isolated mode + small resize ⇒ rescaled sub-cube extends outside
    // the survey footprint ⇒ per-run footprint_coverage should be < 1
    // for the rescaled runs (and = 1 for the un-rescaled one if any).
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_iso");
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1000, box_size, 11);
    write_uniform_3d_f64(&rand_path, 3000, box_size, 12);

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "field-stats",
            "--boundary", "isolated",
            "--n-shifts", "2",
            "--shift-magnitude", "0.25",
            "--shift-seed", "21",
            "--resize-factors", "0.5",   // sub-cube of half-side
            "--bin-tol", "0.05",
            "--max-depth", "5",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "isolated multi-run failed: {:?}", status);

    let diag_path = workdir.join("multi_run_diagnostics.csv");
    let diag_csv = std::fs::read_to_string(&diag_path).unwrap();
    let lines: Vec<&str> = diag_csv.lines().collect();
    // 2 shifts × 1 resize = 2 runs + header
    assert_eq!(lines.len(), 3,
        "expected 2 run rows + header, got {}", lines.len());

    // All resized sub-cubes should have coverage strictly less than 1
    // (the half-side cube at a random offset rarely fits entirely inside
    // the original survey).
    for line in &lines[1..] {
        let row: Vec<&str> = line.split(',').collect();
        let scale: f64 = row[1].parse().unwrap();
        let coverage: f64 = row[3].parse().unwrap();
        assert!((scale - 0.5).abs() < 1e-6,
            "expected scale=0.5, got {}", scale);
        assert!(coverage > 0.0 && coverage < 1.0,
            "resized run should have partial coverage in (0, 1), got {}",
            coverage);
    }

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_multi_run_xi_isolated_smoke() {
    // Multi-shift + multi-resize ξ: verify the CSV has both resize groups,
    // shifts pool within group, and shift-bootstrap variance populates.
    use std::collections::BTreeSet;
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_xi");
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1500, box_size, 31);
    write_uniform_3d_f64(&rand_path, 4500, box_size, 32);

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "xi",
            "--boundary", "isolated",
            "--n-shifts", "3",
            "--shift-magnitude", "0.25",
            "--shift-seed", "33",
            "--resize-factors", "1.0,0.5",
            "--bin-tol", "1e-6",
            "--max-depth", "5",
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "multi-run xi failed: {:?}", status);

    let xi_path = workdir.join("multi_run_xi_raw.csv");
    let xi_csv = std::fs::read_to_string(&xi_path).unwrap();
    let mut lines = xi_csv.lines();
    let header = lines.next().expect("empty xi CSV");
    let cols: Vec<&str> = header.split(',').collect();
    for required in ["scale", "n_shifts", "level", "r_center",
                     "dd_sum", "rr_sum", "dr_sum",
                     "xi_naive", "xi_shift_bootstrap_var"] {
        assert!(cols.contains(&required),
            "xi header missing column '{}': {}", required, header);
    }
    let i_scale     = cols.iter().position(|&c| c == "scale").unwrap();
    let i_nshifts   = cols.iter().position(|&c| c == "n_shifts").unwrap();
    let i_rr        = cols.iter().position(|&c| c == "rr_sum").unwrap();
    let i_xi_arv    = cols.iter().position(|&c| c == "xi_shift_bootstrap_var").unwrap();

    // Walk all rows. Verify: two distinct scales, n_shifts=3 in each,
    // and at least one row with rr>0 has nonzero shift-bootstrap variance.
    let mut scales: BTreeSet<String> = BTreeSet::new();
    let mut had_arv = false;
    for line in lines {
        if line.is_empty() { continue; }
        let row: Vec<&str> = line.split(',').collect();
        scales.insert(row[i_scale].to_string());
        let n_shifts: usize = row[i_nshifts].parse().unwrap();
        assert_eq!(n_shifts, 3,
            "expected n_shifts=3 in every row, got {}", n_shifts);
        let rr: f64 = row[i_rr].parse().unwrap();
        let arv: f64 = row[i_xi_arv].parse().unwrap();
        if rr > 0.0 && arv > 0.0 { had_arv = true; }
    }
    assert_eq!(scales.len(), 2,
        "expected 2 distinct resize-group scales, got {}: {:?}",
        scales.len(), scales);
    assert!(had_arv,
        "no row had RR>0 AND nonzero shift-bootstrap variance — \
         shift pooling may not be tracking per-shift xi_ls correctly");

    let _ = std::fs::remove_dir_all(&workdir);
}

#[test]
fn cli_multi_run_xi_continuous_fit_smoke() {
    // Run xi with --xi-fit-basis linear-bsplines and verify the
    // continuous-fit CSVs appear with expected structure.
    use std::process::Command;

    let bin = match locate_morton_cascade_binary() {
        Some(b) => b,
        None => { eprintln!("skipping: binary not built"); return; }
    };
    let workdir = unique_workdir("multi_xi_fit");
    let data_path = workdir.join("data.bin");
    let rand_path = workdir.join("randoms.bin");
    let box_size: f64 = 100.0;
    write_uniform_3d_f64(&data_path, 1500, box_size, 41);
    write_uniform_3d_f64(&rand_path, 4500, box_size, 42);

    let n_knots: usize = 6;
    let eval_n: usize = 25;

    let status = Command::new(&bin)
        .args([
            "multi-run",
            "-i", data_path.to_str().unwrap(),
            "--randoms", rand_path.to_str().unwrap(),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", workdir.to_str().unwrap(),
            "--statistic", "xi",
            "--boundary", "isolated",
            "--n-shifts", "3",
            "--shift-magnitude", "0.25",
            "--shift-seed", "44",
            "--resize-factors", "1.0,0.7,0.5",
            "--bin-tol", "1e-6",
            "--max-depth", "5",
            "--xi-fit-basis", "linear-bsplines",
            "--xi-fit-knots", &n_knots.to_string(),
            "--xi-fit-window", "empirical-rr",
            "--xi-fit-weighting", "shift-bootstrap-poisson",
            "--xi-fit-eval-n", &eval_n.to_string(),
            "-q",
        ])
        .status().unwrap();
    assert!(status.success(), "multi-run xi+fit failed: {:?}", status);

    // All four expected files
    for fname in [
        "multi_run_xi_raw.csv",
        "multi_run_xi_fit_coefs.csv",
        "multi_run_xi_evaluated.csv",
        "multi_run_diagnostics.csv",
    ] {
        let p = workdir.join(fname);
        assert!(p.exists(), "expected output file missing: {}", fname);
    }

    // Coefs CSV: header + n_knots data rows
    let coefs_csv = std::fs::read_to_string(workdir.join("multi_run_xi_fit_coefs.csv"))
        .unwrap();
    let coefs_lines: Vec<&str> = coefs_csv.lines().collect();
    assert_eq!(coefs_lines.len(), n_knots + 1,
        "expected {} rows in coefs CSV (header + {} knots), got {}",
        n_knots + 1, n_knots, coefs_lines.len());
    let coefs_header: Vec<&str> = coefs_lines[0].split(',').collect();
    for required in ["knot_index", "knot_r", "coef", "coef_sigma"] {
        assert!(coefs_header.contains(&required),
            "coefs header missing column '{}'", required);
    }
    // Coefficient sigma should be non-negative everywhere.
    let i_sigma = coefs_header.iter().position(|&c| c == "coef_sigma").unwrap();
    for line in &coefs_lines[1..] {
        let row: Vec<&str> = line.split(',').collect();
        let sigma: f64 = row[i_sigma].parse().unwrap();
        assert!(sigma >= 0.0,
            "coef_sigma must be non-negative, got {}", sigma);
        assert!(sigma.is_finite(),
            "coef_sigma must be finite, got {}", sigma);
    }

    // Evaluated CSV: header + eval_n data rows
    let eval_csv = std::fs::read_to_string(workdir.join("multi_run_xi_evaluated.csv"))
        .unwrap();
    let eval_lines: Vec<&str> = eval_csv.lines().collect();
    assert_eq!(eval_lines.len(), eval_n + 1,
        "expected {} rows in evaluated CSV (header + {} eval points), got {}",
        eval_n + 1, eval_n, eval_lines.len());
    let eval_header: Vec<&str> = eval_lines[0].split(',').collect();
    for required in ["r", "xi_fit", "xi_fit_sigma"] {
        assert!(eval_header.contains(&required),
            "evaluated header missing column '{}'", required);
    }
    // r should be monotone increasing across the grid.
    let i_r = eval_header.iter().position(|&c| c == "r").unwrap();
    let i_sigma = eval_header.iter().position(|&c| c == "xi_fit_sigma").unwrap();
    let mut prev_r = -f64::INFINITY;
    for line in &eval_lines[1..] {
        let row: Vec<&str> = line.split(',').collect();
        let r: f64 = row[i_r].parse().unwrap();
        let sigma: f64 = row[i_sigma].parse().unwrap();
        assert!(r > prev_r,
            "r grid should be monotone: {} <= {}", r, prev_r);
        assert!(sigma >= 0.0 && sigma.is_finite(),
            "xi_fit_sigma must be non-negative finite, got {}", sigma);
        prev_r = r;
    }

    let _ = std::fs::remove_dir_all(&workdir);
}

