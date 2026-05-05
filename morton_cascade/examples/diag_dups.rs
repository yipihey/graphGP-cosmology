// How many random points have IDENTICAL coords?
use std::collections::HashMap;

fn main() {
    let bits = 10;
    let max = 1u64 << bits;
    let mut s = 9999u64;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); s };

    // Discard the first 1000 outputs (the data points in the failing test)
    // and the cluster generation, then take 3000 randoms.
    // Actually, the failing test path is:
    //   - 40 cluster centers (40*3 = 120 nexts)
    //   - 40 clusters * 25 pts/cluster * 3 nexts = 3000 nexts
    //   - 1000 randoms * 3 nexts = 3000 nexts
    // Replicate exactly:

    let n_clusters = 40;
    let pts_per_cluster = 25;
    let _centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();
    let mut _data: Vec<[u64; 3]> = Vec::new();
    for _c in 0..n_clusters {
        for _ in 0..pts_per_cluster {
            let _ = next() % (max / 32);
            let _ = next() % (max / 32);
            let _ = next() % (max / 32);
        }
    }
    let n_d = n_clusters * pts_per_cluster;
    let randoms: Vec<[u64; 3]> = (0..n_d * 3).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();

    // Count duplicates
    let mut counts: HashMap<[u64; 3], usize> = HashMap::new();
    for p in &randoms {
        *counts.entry(*p).or_insert(0) += 1;
    }
    let n_dup_pairs: usize = counts.values()
        .map(|&c| if c >= 2 { c * (c - 1) / 2 } else { 0 })
        .sum();
    let n_unique = counts.len();
    println!("randoms: {} total, {} unique coords, {} duplicate pairs",
        randoms.len(), n_unique, n_dup_pairs);

    // Repeat for the data
    let mut s2 = 9999u64;
    let mut next2 = || { s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1); s2 };
    let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| {
        [next2() % max, next2() % max, next2() % max]
    }).collect();
    let mut data: Vec<[u64; 3]> = Vec::new();
    for c in &centers {
        for _ in 0..pts_per_cluster {
            let dx = (next2() % (max / 32)) as i64 - (max / 64) as i64;
            let dy = (next2() % (max / 32)) as i64 - (max / 64) as i64;
            let dz = (next2() % (max / 32)) as i64 - (max / 64) as i64;
            let x = ((c[0] as i64 + dx).rem_euclid(max as i64)) as u64;
            let y = ((c[1] as i64 + dy).rem_euclid(max as i64)) as u64;
            let z = ((c[2] as i64 + dz).rem_euclid(max as i64)) as u64;
            data.push([x, y, z]);
        }
    }
    let mut counts: HashMap<[u64; 3], usize> = HashMap::new();
    for p in &data {
        *counts.entry(*p).or_insert(0) += 1;
    }
    let n_dup_pairs: usize = counts.values()
        .map(|&c| if c >= 2 { c * (c - 1) / 2 } else { 0 })
        .sum();
    let n_unique = counts.len();
    println!("data:    {} total, {} unique coords, {} duplicate pairs",
        data.len(), n_unique, n_dup_pairs);
}
