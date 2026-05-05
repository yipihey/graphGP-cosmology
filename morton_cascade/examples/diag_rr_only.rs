// Minimal test: only random points, no clustering. Compare cum_rr from
// pair cascade against single-catalog cascade run on the same randoms.
use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
use morton_cascade::hier_bitvec_pair::BitVecCascadePair;
use morton_cascade::hier_bitvec::BitVecCascade;

fn main() {
    let bits = 10;
    let max = 1u64 << bits;
    let mut s = 9999u64;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); s };

    // Same data + randoms recipe as the failing test, but DATA is also uniform
    let n_d = 1000;
    let n_r = 3000;
    let data: Vec<[u64; 3]> = (0..n_d).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();
    let randoms: Vec<[u64; 3]> = (0..n_r).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();

    let range = CoordRange::analyze_pair(&data, &randoms);
    let td = TrimmedPoints::from_points_with_range(data.clone(), range.clone());
    let tr = TrimmedPoints::from_points_with_range(randoms.clone(), range.clone());

    let pair = BitVecCascadePair::<3>::build(td, tr, None);
    let pstats = pair.analyze();

    let single_t = TrimmedPoints::from_points_with_range(randoms, range.clone());
    let single = BitVecCascade::<3>::build(single_t, Some(pair.l_max));
    let sstats = single.analyze();

    println!("level   pair_cum_RR   single_cum_pairs   diff");
    for l in 0..pstats.len() {
        let p = pstats[l].cumulative_rr as u64;
        let s = sstats[l].cumulative_pairs;
        let mark = if p as i64 - s as i64 != 0 { " <-- MISMATCH" } else { "" };
        println!("{:>5}  {:>11}  {:>16}  {:>5}{}",
            l, p, s, p as i64 - s as i64, mark);
    }
}
