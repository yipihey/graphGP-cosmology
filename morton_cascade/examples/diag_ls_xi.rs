// Diagnose ls_xi_recovers_clustering_for_clustered_data: print every shell.
use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
use morton_cascade::hier_bitvec_pair::BitVecCascadePair;

fn main() {
    let n_clusters = 40;
    let pts_per_cluster = 25;
    let bits = 10;
    let max = 1u64 << bits;

    let mut s = 9999u64;
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); s };
    let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();

    let mut data: Vec<[u64; 3]> = Vec::new();
    for c in &centers {
        for _ in 0..pts_per_cluster {
            let dx = (next() % (max / 32)) as i64 - (max / 64) as i64;
            let dy = (next() % (max / 32)) as i64 - (max / 64) as i64;
            let dz = (next() % (max / 32)) as i64 - (max / 64) as i64;
            let x = ((c[0] as i64 + dx).rem_euclid(max as i64)) as u64;
            let y = ((c[1] as i64 + dy).rem_euclid(max as i64)) as u64;
            let z = ((c[2] as i64 + dz).rem_euclid(max as i64)) as u64;
            data.push([x, y, z]);
        }
    }
    let n_d = data.len();
    let randoms: Vec<[u64; 3]> = (0..n_d * 3).map(|_| {
        [next() % max, next() % max, next() % max]
    }).collect();
    let n_r = randoms.len();

    let range = CoordRange::analyze_pair(&data, &randoms);
    let td = TrimmedPoints::from_points_with_range(data, range.clone());
    let tr = TrimmedPoints::from_points_with_range(randoms, range);
    let pair = BitVecCascadePair::<3>::build(td, tr, None);
    let stats = pair.analyze();
    let shells = pair.xi_landy_szalay(&stats);

    println!("# l_max = {}, n stats levels = {}", pair.l_max, stats.len());
    println!("# eff_bits = {:?}", pair.data.range.effective_bits);
    println!("# cumulative arrays:");
    for (l, st) in stats.iter().enumerate() {
        println!("#   level {}: cum_DD={:.0} cum_RR={:.0} cum_DR={:.0}",
            l, st.cumulative_dd, st.cumulative_rr, st.cumulative_dr);
    }
    println!("# N_d = {}, N_r = {}, max = {}, cluster size ~{}", n_d, n_r, max, max/32);
    println!("# norms: DD = N_d*(N_d-1)/2 = {}, RR = N_r*(N_r-1)/2 = {}, DR = N_d*N_r = {}",
        n_d * (n_d-1) / 2, n_r * (n_r-1) / 2, n_d * n_r);
    println!("level  side       DD          RR          DR        xi_LS");
    for s in &shells {
        println!("{:>4}  {:>8.1}  {:>10.0}  {:>10.0}  {:>10.0}  {:>10.4}",
            s.level, s.cell_side_trimmed, s.dd, s.rr, s.dr, s.xi_ls);
    }
}
