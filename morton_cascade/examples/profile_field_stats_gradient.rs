use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
use morton_cascade::hier_bitvec_pair::{BitVecCascadePair, FieldStatsConfig};
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
    (0..n)
        .map(|_| [splitmix64(&mut s) & mask, splitmix64(&mut s) & mask, splitmix64(&mut s) & mask])
        .collect()
}

fn main() {
    let cfg = FieldStatsConfig::default();
    println!("    N      forward(ms)   gradient(ms)   ratio");
    for &n in &[1000usize, 5000, 20000, 100000] {
        let pts_d = make_uniform(n, 7, 11);
        let pts_r = make_uniform(3 * n, 7, 22);
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        // Time forward (3x to amortize warmup)
        let t0 = Instant::now();
        let mut stats = vec![];
        for _ in 0..3 { stats = pair.analyze_field_stats(&cfg); }
        let fwd_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;

        // Time gradient (3x)
        let t0 = Instant::now();
        for _ in 0..3 { let _ = pair.gradient_var_delta_all_levels(&cfg, &stats); }
        let grad_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;

        println!("  {:>6}  {:>11.2}   {:>11.2}   {:>5.2}x", n, fwd_ms, grad_ms, grad_ms / fwd_ms);
    }
}
