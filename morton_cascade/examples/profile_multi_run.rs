// profile_multi_run.rs
//
// Measures the speedup of multi-run analyses (analyze_field_stats,
// gradient_var_delta_data_pooled) under different thread counts.
// Runs each workload in dedicated rayon thread pools of size 1, 2, 4,
// 8 (or up to logical_cpu_count) and reports wall-clock time.
//
// Build & run:
//   cargo build --release --example profile_multi_run
//   ./target/release/examples/profile_multi_run
//
// Note: on single-core machines the higher thread counts will not show
// speedup (and may show small overhead). On a multi-core machine
// expect close-to-linear speedup for thread counts up to N_runs.
// See docs/parallelism.md for the audit and rationale.

use morton_cascade::hier_bitvec_pair::FieldStatsConfig;
use morton_cascade::multi_run::{CascadeRunner, CascadeRunPlan};
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn make_uniform(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| [
        splitmix64(&mut s) & mask,
        splitmix64(&mut s) & mask,
        splitmix64(&mut s) & mask,
    ]).collect()
}

fn main() {
    let n_d = 20_000;
    let n_r = 60_000;
    let n_runs = 8;
    let bits = [8u32; 3];

    println!("Generating catalog: {} data, {} randoms, {} runs",
        n_d, n_r, n_runs);
    let pts_d = make_uniform(n_d, 8, 11);
    let pts_r = make_uniform(n_r, 8, 22);
    let plan = CascadeRunPlan::random_offsets(n_runs, 0.5, 33);
    let cfg = FieldStatsConfig::default();
    let bin_tol = 1e-6;

    let max_threads = num_cpus_or(8).max(8);
    let thread_counts: Vec<usize> = (0..)
        .map(|i| 1usize << i)
        .take_while(|&n| n <= max_threads)
        .collect();

    println!("\nanalyze_field_stats (per-run cascade build + analysis + pooling):");
    println!("  threads   wall(ms)   speedup");
    let baseline_analyze = run_analyze(&pts_d, &pts_r, bits, &plan, &cfg, bin_tol, 1);
    for &nt in &thread_counts {
        let t = run_analyze(&pts_d, &pts_r, bits, &plan, &cfg, bin_tol, nt);
        println!("  {:>7}   {:>8.1}   {:>5.2}x", nt, t * 1000.0, baseline_analyze / t);
    }

    println!("\ngradient_var_delta_data_pooled (per-run cascade + gradient + lifting + pooling):");
    println!("  threads   wall(ms)   speedup");
    let baseline_grad = run_grad(&pts_d, &pts_r, bits, &plan, &cfg, bin_tol, 1);
    for &nt in &thread_counts {
        let t = run_grad(&pts_d, &pts_r, bits, &plan, &cfg, bin_tol, nt);
        println!("  {:>7}   {:>8.1}   {:>5.2}x", nt, t * 1000.0, baseline_grad / t);
    }
}

fn run_analyze(
    pts_d: &[[u64; 3]], pts_r: &[[u64; 3]], bits: [u32; 3],
    plan: &CascadeRunPlan<3>, cfg: &FieldStatsConfig, bin_tol: f64,
    nthreads: usize,
) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads).build().unwrap();
    let runner = CascadeRunner::new_isolated(
        pts_d.to_vec(), None, pts_r.to_vec(), None, bits, plan.clone());
    // Warm up.
    let _ = pool.install(|| runner.analyze_field_stats(cfg, bin_tol));
    let t0 = Instant::now();
    let n_iter = 3;
    for _ in 0..n_iter {
        let _ = pool.install(|| runner.analyze_field_stats(cfg, bin_tol));
    }
    t0.elapsed().as_secs_f64() / n_iter as f64
}

fn run_grad(
    pts_d: &[[u64; 3]], pts_r: &[[u64; 3]], bits: [u32; 3],
    plan: &CascadeRunPlan<3>, cfg: &FieldStatsConfig, bin_tol: f64,
    nthreads: usize,
) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads).build().unwrap();
    let runner = CascadeRunner::new_isolated(
        pts_d.to_vec(), None, pts_r.to_vec(), None, bits, plan.clone());
    let _ = pool.install(|| runner.gradient_var_delta_data_pooled(cfg, bin_tol));
    let t0 = Instant::now();
    let n_iter = 3;
    for _ in 0..n_iter {
        let _ = pool.install(|| runner.gradient_var_delta_data_pooled(cfg, bin_tol));
    }
    t0.elapsed().as_secs_f64() / n_iter as f64
}

fn num_cpus_or(default: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(default)
}
