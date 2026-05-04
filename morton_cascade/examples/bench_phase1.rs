// Phase 1 benchmark: measure the actual impact of point-list crossover.
//
// Differences from bench_bitvec:
//   - Uses splitmix64 (high-quality) instead of LCG-low-bits, so points span
//     the full coordinate range and the cascade visits realistic cell counts.
//   - Sweeps the crossover threshold so we can SEE the speedup (crossover=0
//     means immediate point-list switch; crossover=N+1 means pure bit-vector).
//   - Targets the regimes the session-state doc flagged as slow.
//
// Reads: BitVecCascade::build_with_threshold lets us dial the threshold.
// Writes: prints a table of {threshold, build_ms, analyze_ms, cells_visited}
// per benchmark case.

use morton_cascade::coord_range::TrimmedPoints;
use morton_cascade::hier_bitvec::BitVecCascade;
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn make_uniform<const D: usize>(n: usize, bits: u32, seed: u64) -> Vec<[u64; D]> {
    let mut s = seed;
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| {
        let mut p = [0u64; D];
        for d in 0..D {
            p[d] = splitmix64(&mut s) & mask;
        }
        p
    }).collect()
}

fn run_case<const D: usize>(label: &str, pts: Vec<[u64; D]>) {
    let n = pts.len();
    let trimmed = TrimmedPoints::from_points(pts);
    let l_max = trimmed.range.max_supported_l_max();
    let eff = trimmed.range.effective_bits;
    println!("\n--- {} : N={}  D={}  eff_bits={:?}  l_max={} ---",
        label, n, D, eff, l_max);
    println!("{:>10}  {:>10}  {:>10}  {:>14}", "thresh", "build_ms", "analyze_ms", "cells_visited");

    // Key thresholds to sweep:
    //   0       — pure point-list (cross over immediately)
    //   8, 64   — practical defaults
    //   N + 1   — pure bit-vector (never cross over)
    let thresholds = [0usize, 8, 32, 64, 256, 1024, n + 1];
    for &thresh in &thresholds {
        // Re-build trimmed each iteration — no Clone on TrimmedPoints, so
        // we use from_points_with_range with the same coord_range.
        let trimmed_iter = TrimmedPoints::<D>::from_points_with_range(
            trimmed.points.iter().map(|p| {
                let mut q = [0u64; D];
                for d in 0..D { q[d] = p[d] << trimmed.range.bit_min[d]; }
                q
            }).collect(),
            trimmed.range.clone(),
        );
        let t = Instant::now();
        let casc = BitVecCascade::<D>::build_with_threshold(trimmed_iter, None, thresh);
        let t_build = t.elapsed();
        let t = Instant::now();
        let stats = casc.analyze();
        let t_analyze = t.elapsed();
        let total_nonempty: u64 = stats.iter().map(|s| s.n_nonempty_cells).sum();
        let analyze_ms = t_analyze.as_secs_f64() * 1000.0;
        println!("{:>10}  {:>10.2}  {:>10.2}  {:>14}",
            thresh,
            t_build.as_secs_f64() * 1000.0,
            analyze_ms,
            total_nonempty);
        // If this configuration takes more than 60s, skip the rest of this case.
        if analyze_ms > 60_000.0 {
            println!("  (analyze > 60s; skipping remaining thresholds for this case)");
            break;
        }
    }
}

fn main() {
    println!("=== Phase 1 crossover benchmark — splitmix64 PRNG, full-range coords ===");

    // Cases the session-state doc flagged as slow before crossover:
    println!("\n## 2D u16 N=100k l_max=16 (was 8324 ms before crossover)");
    run_case("2D u16 N=100k", make_uniform::<2>(100_000, 16, 1001));

    println!("\n## 3D u16 N=50k l_max=16 (was 7753 ms before crossover)");
    run_case("3D u16 N=50k", make_uniform::<3>(50_000, 16, 1002));

    // Stress: deeper l_max with realistic distributions
    println!("\n## 3D large-N regimes");
    run_case("3D u16 N=100k", make_uniform::<3>(100_000, 16, 1003));

    // 2D depth scaling at fixed N — start small, ramp up only if previous fits.
    println!("\n## 2D depth scaling, N=100k");
    run_case("2D u14 N=100k", make_uniform::<2>(100_000, 14, 1007));
    run_case("2D u20 N=100k", make_uniform::<2>(100_000, 20, 1009));

    println!("\n=== done ===");
}
