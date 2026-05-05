// benchmark_packed.rs
//
// Benchmarks the adaptive-width packed cascade vs the dense u64 reference for
// MCMC inner loops. Reports memory savings and speedup at multiple N.
//
// Usage:
//     cargo run --release --example benchmark_packed

use morton_cascade::{hier, hier_packed, L_MAX, N_LEVELS};
use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::{LogNormalField, Rng};

fn main() {
    let mut rng = Rng::new(20260502);

    println!("================================================================================");
    println!("  ADAPTIVE-WIDTH PACKED CASCADE BENCHMARK");
    println!("================================================================================");

    // Use a Cox field for realistic max-count behavior
    let g_field = 128usize;
    println!("\nBuilding Cox field (G={}, alpha=1.5, sigma_g^2=0.4)...", g_field);
    let field = LogNormalField::new(g_field, 1.5, 0.4, 800, &mut rng);

    let n_pts = 800_000usize;
    let pts_pk = field.sample(n_pts, &mut rng);

    // Correctness
    println!("\n--- Correctness vs reference cascade (N={}) ---", n_pts);
    let (st_ref, _) = hier::cascade_hierarchical_bc(&pts_pk, 1, true);
    let (st_pk, _, bytes_pk) = hier_packed::cascade_adaptive(&pts_pk, 1, true, None);
    let mut max_diff = 0.0f64;
    for l in 0..N_LEVELS {
        let dv = (st_ref[l].var - st_pk[l].var).abs() / st_ref[l].var.abs().max(1e-30);
        let dd = (st_ref[l].dvar - st_pk[l].dvar).abs() / st_ref[l].dvar.abs().max(1e-30);
        max_diff = max_diff.max(dv).max(dd);
    }
    println!("  Max relative diff vs reference = {:.3e} (should be 0)", max_diff);

    // Per-level bytes
    println!("\n--- Per-level buffer storage (max-count safety = 4x) ---");
    println!("  {:>5} {:>14} {:>14} {:>10}", "lvl", "bytes", "vs all-u64", "type");
    let m_size = 1usize << (L_MAX + 1);
    let dense_bytes_per_level = m_size * m_size * 8;
    let mut max_bytes = 0;
    for l in 0..N_LEVELS {
        let bytes = bytes_pk[l];
        if bytes > max_bytes { max_bytes = bytes; }
        let ratio = dense_bytes_per_level as f64 / bytes.max(1) as f64;
        let bytes_per_elem = if (m_size * m_size) > 0 { bytes / (m_size * m_size) } else { 0 };
        let typ = match bytes_per_elem { 1 => "u8", 2 => "u16", 4 => "u32", 8 => "u64", _ => "?" };
        println!("  {:>5} {:>14} {:>14.2}x {:>10}", l, bytes, ratio, typ);
    }
    println!("  Peak working memory (1 buffer):  {} bytes", max_bytes);
    println!("  Reference (one u64 buffer):       {} bytes ({:.1}x more)",
             dense_bytes_per_level, dense_bytes_per_level as f64 / max_bytes.max(1) as f64);

    // MCMC-style timing
    let max_hint = hier_packed::predict_max_counts_2d(n_pts as u64, 4);
    let n_inner = 50;
    println!("\n--- MCMC-style: avg over {} runs (warm cache) ---", n_inner);
    for _ in 0..5 {
        let _ = hier::cascade_hierarchical_bc(&pts_pk, 1, true);
        let _ = hier_packed::cascade_adaptive(&pts_pk, 1, true, Some(max_hint));
    }

    let t = Instant::now();
    for _ in 0..n_inner {
        let _ = hier::cascade_hierarchical_bc(&pts_pk, 1, true);
    }
    let t_inner_ref = t.elapsed().as_secs_f64() * 1000.0 / n_inner as f64;

    let t = Instant::now();
    for _ in 0..n_inner {
        let _ = hier_packed::cascade_adaptive(&pts_pk, 1, true, Some(max_hint));
    }
    let t_inner_pk = t.elapsed().as_secs_f64() * 1000.0 / n_inner as f64;

    println!("  ref {:.2} ms, packed {:.2} ms, speedup {:.2}x",
             t_inner_ref, t_inner_pk, t_inner_ref / t_inner_pk);

    // Parameter sweep
    println!("\n--- Parameter sweep ---");
    println!("  {:>10} {:>5} {:>10} {:>10} {:>10} {:>10}",
             "N", "s_sub", "ref ms", "pkd ms", "speedup", "peak MB");
    for n_test in [100_000usize, 800_000, 3_200_000] {
        for s_sub in [0usize, 1, 2] {
            let pts = field.sample(n_test, &mut rng);
            let hint = hier_packed::predict_max_counts_2d(n_test as u64, 4);
            for _ in 0..2 {
                let _ = hier::cascade_hierarchical_bc(&pts, s_sub, true);
                let _ = hier_packed::cascade_adaptive(&pts, s_sub, true, Some(hint));
            }
            let n_runs = 5;
            let t = Instant::now();
            for _ in 0..n_runs { let _ = hier::cascade_hierarchical_bc(&pts, s_sub, true); }
            let t_ref = t.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

            let mut peak_mb = 0.0;
            let t = Instant::now();
            for _ in 0..n_runs {
                let (_, _, b) = hier_packed::cascade_adaptive(&pts, s_sub, true, Some(hint));
                let pk = b.iter().max().copied().unwrap_or(0);
                peak_mb = (pk as f64 / 1.0e6).max(peak_mb);
            }
            let t_pkd = t.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

            println!("  {:>10} {:>5} {:>10.2} {:>10.2} {:>10.2}x {:>10.2}",
                     pts.len(), s_sub, t_ref, t_pkd, t_ref / t_pkd, peak_mb);
        }
    }
}
