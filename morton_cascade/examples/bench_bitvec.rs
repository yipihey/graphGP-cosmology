// Quick benchmark: bitvec cascade at moderate N.
use morton_cascade::coord_range::TrimmedPoints;
use morton_cascade::hier_bitvec::BitVecCascade;
use std::time::Instant;

fn make_uniform_2d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 2]> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    };
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| [next() & mask, next() & mask]).collect()
}

fn make_uniform_3d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s
    };
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| [next() & mask, next() & mask, next() & mask]).collect()
}

fn bench<const D: usize>(label: &str, pts: Vec<[u64; D]>) {
    let n = pts.len();
    let trimmed = TrimmedPoints::from_points(pts);
    let l_max_data = trimmed.range.max_supported_l_max();
    let t = Instant::now();
    let casc = BitVecCascade::<D>::build(trimmed, None);
    let t_build = t.elapsed();
    let t = Instant::now();
    let stats = casc.analyze();
    let t_analyze = t.elapsed();

    let mut total_nonempty: u64 = 0;
    for s in &stats { total_nonempty += s.n_nonempty_cells; }
    println!("{}  N={}  l_max={}  build {:.1} ms  analyze {:.1} ms  cells_visited={}",
        label, n, l_max_data,
        t_build.as_secs_f64() * 1000.0,
        t_analyze.as_secs_f64() * 1000.0,
        total_nonempty);
}

fn main() {
    println!("=== 2D, varying N ===");
    bench("2D u8 ", make_uniform_2d(  1_000,  8, 1));
    bench("2D u8 ", make_uniform_2d( 10_000,  8, 2));
    bench("2D u10", make_uniform_2d( 10_000, 10, 3));
    bench("2D u12", make_uniform_2d( 10_000, 12, 4));
    bench("2D u14", make_uniform_2d( 10_000, 14, 5));
    bench("2D u16", make_uniform_2d( 10_000, 16, 6));
    bench("2D u16", make_uniform_2d(100_000, 16, 7));

    println!("\n=== 3D, varying N ===");
    bench("3D u8 ", make_uniform_3d(  1_000,  8, 11));
    bench("3D u8 ", make_uniform_3d( 10_000,  8, 12));
    bench("3D u10", make_uniform_3d( 10_000, 10, 13));
    bench("3D u12", make_uniform_3d( 10_000, 12, 14));
    bench("3D u14", make_uniform_3d( 10_000, 14, 15));
    bench("3D u16", make_uniform_3d( 50_000, 16, 16));

    println!("\n=== 3D, large N ===");
    bench("3D u16", make_uniform_3d(  100_000, 16, 21));
    bench("3D u16", make_uniform_3d(  500_000, 16, 22));
    bench("3D u20", make_uniform_3d(  100_000, 20, 23));
    bench("3D u24", make_uniform_3d(  100_000, 24, 24));
    bench("3D u32", make_uniform_3d(  100_000, 32, 25));
}
