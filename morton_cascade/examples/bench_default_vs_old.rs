// Confirm the new adaptive default beats the old constant default.

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

fn time_one<const D: usize>(pts: Vec<[u64; D]>, thresh: usize) -> f64 {
    let trimmed = TrimmedPoints::from_points(pts);
    let casc = BitVecCascade::<D>::build_with_threshold(trimmed, None, thresh);
    let t = Instant::now();
    let _ = casc.analyze();
    t.elapsed().as_secs_f64() * 1000.0
}

fn run_scan<const D: usize>(label: &str, ns: &[usize], bits: u32) {
    println!("\n--- {} (D={}, bits={}) ---", D, label, bits);
    println!("{:>10}  {:>9}  {:>9}  {:>9}  {:>8}",
        "N", "old(64)", "new(adp)", "pure_BV", "speedup");
    for &n in ns {
        let new_thresh = BitVecCascade::<D>::default_crossover_threshold(n);
        let pts1 = make_uniform::<D>(n, bits, 4242);
        let t_old = time_one::<D>(pts1, 64);
        let pts2 = make_uniform::<D>(n, bits, 4242);
        let t_new = time_one::<D>(pts2, new_thresh);
        let pts3 = make_uniform::<D>(n, bits, 4242);
        let t_bv = time_one::<D>(pts3, n + 1);
        println!("{:>10}  {:>9.2}  {:>9.2}  {:>9.2}  {:>7.2}x  (new thresh={})",
            n, t_old, t_new, t_bv, t_old / t_new, new_thresh);
    }
}

fn main() {
    println!("=== Adaptive default vs old constant default ===");
    println!("'old(64)' = previous constant default of 64 points");
    println!("'new(adp)' = max(64, N/64) — the new adaptive default");
    println!("'pure_BV' = no crossover (lower bound on what's achievable)");

    run_scan::<2>("2D, l_max=16",
        &[1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], 16);

    run_scan::<3>("3D, l_max=16",
        &[1_000, 10_000, 50_000, 100_000, 250_000, 500_000], 16);
}
