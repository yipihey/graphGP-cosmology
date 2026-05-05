// Follow-up: scan N at fixed l_max=16, with default thresh=64 vs pure bit-vec.
// Tells us whether the marginal slowdown from crossover gets WORSE or BETTER
// as N grows.

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

fn time_one<const D: usize>(pts: Vec<[u64; D]>, thresh: usize) -> (f64, f64) {
    let trimmed = TrimmedPoints::from_points(pts);
    let t = Instant::now();
    let casc = BitVecCascade::<D>::build_with_threshold(trimmed, None, thresh);
    let build = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let _ = casc.analyze();
    let analyze = t.elapsed().as_secs_f64() * 1000.0;
    (build, analyze)
}

fn run_scan<const D: usize>(label: &str, ns: &[usize], bits: u32) {
    println!("\n--- {} (D={}, bits={}) ---", label, D, bits);
    println!("{:>10}  {:>10} {:>10}  {:>10} {:>10}  {:>8}",
        "N", "build64", "analyze64", "buildBV", "analyzeBV", "ratio");
    for &n in ns {
        let pts1 = make_uniform::<D>(n, bits, 4242);
        let (b64, a64) = time_one::<D>(pts1, 64);
        let pts2 = make_uniform::<D>(n, bits, 4242);
        let (bbv, abv) = time_one::<D>(pts2, n + 1);
        let ratio = a64 / abv;
        println!("{:>10}  {:>10.2} {:>10.2}  {:>10.2} {:>10.2}  {:>8.3}",
            n, b64, a64, bbv, abv, ratio);
    }
}

fn main() {
    println!("=== Crossover-threshold sensitivity vs N ===");
    println!("'BV' = pure bit-vector (thresh = N+1, never crosses over)");
    println!("'64' = default crossover threshold = 64 points");
    println!("ratio = analyze64 / analyzeBV   (>1 means default is slower)");

    run_scan::<2>("2D, l_max=16",
        &[1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], 16);

    run_scan::<3>("3D, l_max=16",
        &[1_000, 10_000, 50_000, 100_000, 250_000, 500_000], 16);

    println!("\n=== done ===");
}
