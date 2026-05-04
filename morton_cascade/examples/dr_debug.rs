use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
use morton_cascade::hier_bitvec_pair::BitVecCascadePair;

fn make_uniform_3d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); s };
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| [next() & mask, next() & mask, next() & mask]).collect()
}

fn main() {
    let pts_d = make_uniform_3d(25, 7, 13);
    let pts_r = make_uniform_3d(40, 7, 17);
    let range = CoordRange::analyze_pair(&pts_d, &pts_r);
    println!("range bit_min={:?} effective_bits={:?}", range.bit_min, range.effective_bits);
    let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
    let tr = TrimmedPoints::from_points_with_range(pts_r, range);
    let pair = BitVecCascadePair::<3>::build(td, tr, None);
    println!("l_max={} n_words_d={} n_words_r={}", pair.l_max, pair.n_words_d, pair.n_words_r);
    // Print bit_planes_d[0][0] and bit_planes_r[0][0] in binary
    println!("bit_planes_d[0][0] = {:064b}", pair.bit_planes_d[0][0][0]);
    println!("bit_planes_r[0][0] = {:064b}", pair.bit_planes_r[0][0][0]);
    // The first word of bit_planes is N_d=25 or N_r=40 bits. Let's count bits in plane[0][0]:
    println!("popcount d[0][0] = {}", pair.bit_planes_d[0][0][0].count_ones());
    println!("popcount r[0][0] = {}", pair.bit_planes_r[0][0][0].count_ones());
    // analyze
    let stats = pair.analyze();
    for s in &stats {
        println!("level {}: dd={} rr={} dr={} cells_d={} cells_r={}",
            s.level, s.cumulative_dd, s.cumulative_rr, s.cumulative_dr,
            s.n_nonempty_d, s.n_nonempty_r);
    }
}
