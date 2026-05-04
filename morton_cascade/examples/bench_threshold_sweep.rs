// Find the optimal crossover threshold for the worst case from
// bench_threshold_scan: 1M points, D=2, l_max=16.

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

fn run_sweep<const D: usize>(n: usize, bits: u32) {
    println!("\n=== {}D, N={}, bits={} ===", D, n, bits);
    let pts = make_uniform::<D>(n, bits, 4242);
    let trimmed = TrimmedPoints::from_points(pts);
    println!("{:>10}  {:>10} {:>10}", "thresh", "build_ms", "analyze_ms");
    let mut best_thresh = 0usize;
    let mut best_t = f64::INFINITY;
    let candidates: Vec<usize> = vec![
        8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, n + 1,
    ];
    for &thresh in &candidates {
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
        let build = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let _ = casc.analyze();
        let analyze = t.elapsed().as_secs_f64() * 1000.0;
        let label = if thresh > n { "BV".to_string() } else { thresh.to_string() };
        println!("{:>10}  {:>10.2} {:>10.2}", label, build, analyze);
        if analyze < best_t { best_t = analyze; best_thresh = thresh; }
    }
    let label = if best_thresh > n { "BV".to_string() } else { best_thresh.to_string() };
    println!("Best: {:.2} ms at thresh = {}", best_t, label);
}

fn main() {
    // Small N: the regime where the original Phase 1 work was supposedly
    // motivated. Make sure the new default doesn't regress here.
    run_sweep::<2>(10_000, 16);
    run_sweep::<2>(50_000, 16);
    run_sweep::<2>(250_000, 16);
    run_sweep::<2>(1_000_000, 16);
    run_sweep::<3>(50_000, 16);
    run_sweep::<3>(250_000, 16);
}
