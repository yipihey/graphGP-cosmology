// Confirm the PRNG used by make_uniform_3d (with the larger increment)
// gives essentially no duplicates at the (3000 points, 1024^3 box) scale.
use std::collections::HashMap;

fn main() {
    let bits = 10u32;
    let max = 1u64 << bits;
    let mask = max - 1;
    let n_r = 3000usize;

    // Same construction as make_uniform_3d in hier_bitvec_pair tests
    let mut s = 12345u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    };
    let randoms: Vec<[u64; 3]> = (0..n_r).map(|_| {
        [next() & mask, next() & mask, next() & mask]
    }).collect();

    let mut counts: HashMap<[u64; 3], usize> = HashMap::new();
    for p in &randoms {
        *counts.entry(*p).or_insert(0) += 1;
    }
    let n_unique = counts.len();
    let n_dup_pairs: usize = counts.values()
        .map(|&c| if c >= 2 { c * (c - 1) / 2 } else { 0 })
        .sum();
    println!("randoms (good PRNG): {} total, {} unique, {} dup pairs",
        n_r, n_unique, n_dup_pairs);
    // Expected for true random: ~ n*(n-1)/(2*max^3) ≈ 3000^2/(2*1e9) ≈ 0.004
}
