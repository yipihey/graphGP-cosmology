// Per-particle attribution example via the cell-membership index.
//
// Given a cascade analysis result like CIC PMF (histogram of cell
// occupancies), each output bin counts the number of cells with k
// particles. Each particle sits in exactly one finest-level cell of
// some size n_cell — and that cell contributes 1 to bin n_cell.
// The natural per-particle attribution distributes that cell's bin
// contribution equally among its n_cell particles: each gets 1/n_cell.
//
// This demo:
//   1. Builds a small clustered catalog
//   2. Runs the cascade
//   3. Builds the cell membership index
//   4. Computes per-particle attribution to the CIC PMF
//   5. Verifies the attribution sums correctly
//   6. Identifies the highest- and lowest-influence particles
//      (those in the most- and least-populated cells)
//
// The same pattern works for any cascade statistic that admits a
// per-cell decomposition. Field-stats variance, anisotropy moments,
// and pair counts all do — the coefficient `1/n_cell` is replaced
// by whatever sensitivity the statistic has w.r.t. the cell's
// per-particle contribution.

use morton_cascade::cell_membership::{CellMembership, WhichCatalog};
use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
use morton_cascade::hier_bitvec_pair::BitVecCascadePair;
use std::collections::HashMap;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Generate a clustered 3D catalog: half the points uniform, half
/// in three Gaussian clumps. Returns (points in [0, box_side)^3, n).
fn make_clustered(n: usize, box_bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let box_side = 1u64 << box_bits;
    let mut pts = Vec::with_capacity(n);

    // Three clump centers; pick at random.
    let centers: [[f64; 3]; 3] = [
        [0.25, 0.25, 0.25],
        [0.70, 0.30, 0.50],
        [0.40, 0.65, 0.65],
    ];
    let sigma = 0.05;

    for i in 0..n {
        let (x, y, z) = if i % 2 == 0 {
            // Uniform background.
            (uniform(&mut s), uniform(&mut s), uniform(&mut s))
        } else {
            // Pick a clump and Gaussian around it (Box-Muller).
            let center = &centers[(splitmix64(&mut s) as usize) % 3];
            let mut sample = || -> f64 {
                let u1 = uniform(&mut s).max(1e-12);
                let u2 = uniform(&mut s);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            (
                (center[0] + sigma * sample()).rem_euclid(1.0),
                (center[1] + sigma * sample()).rem_euclid(1.0),
                (center[2] + sigma * sample()).rem_euclid(1.0),
            )
        };
        let to_u = |f: f64| ((f * box_side as f64) as u64).min(box_side - 1);
        pts.push([to_u(x), to_u(y), to_u(z)]);
    }
    pts
}

fn make_uniform(n: usize, box_bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let mask = (1u64 << box_bits) - 1;
    (0..n)
        .map(|_| {
            [
                splitmix64(&mut s) & mask,
                splitmix64(&mut s) & mask,
                splitmix64(&mut s) & mask,
            ]
        })
        .collect()
}

fn main() {
    println!("==========================================================");
    println!(" Per-particle attribution via the cell-membership index");
    println!("==========================================================\n");

    let n_d = 5_000;
    let n_r = 15_000;
    let bits = 7;

    println!("Step 1: build a clustered data catalog ({} points) and a", n_d);
    println!("        uniform random catalog ({} points), with {} cascade bits", n_r, bits);
    println!("        per axis (cell sizes from {}^3 down to 1).\n", 1u64 << bits);

    let pts_d = make_clustered(n_d, bits, 12345);
    let pts_r = make_uniform(n_r, bits, 67890);

    let range = CoordRange::analyze_pair(&pts_d, &pts_r);
    let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
    let tr = TrimmedPoints::from_points_with_range(pts_r, range);
    let pair = BitVecCascadePair::<3>::build(td, tr, None);
    println!(
        "Step 2: cascade built. depth={}, n_d={}, n_r={}\n",
        pair.l_max(),
        pair.n_d(),
        pair.n_r()
    );

    println!("Step 3: build the per-particle CellMembership index for");
    println!("        the data catalog. (Morton-sort, O(N log N).)\n");
    let mem = CellMembership::build(&pair, WhichCatalog::Data);

    let l = mem.l_max();
    println!(
        "Step 4: at the finest level (l={}), count cells of each size",
        l
    );
    println!("        and compute the CIC PMF directly from membership:\n");

    let mut bin_counts: HashMap<usize, usize> = HashMap::new();
    for (_, members) in mem.non_empty_cells_at(l) {
        *bin_counts.entry(members.len()).or_insert(0) += 1;
    }
    let mut bins_sorted: Vec<(usize, usize)> = bin_counts.iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    bins_sorted.sort();
    println!("        bin (cell size) | cell count");
    println!("        ----------------+-----------");
    for (k, count) in &bins_sorted {
        println!("        {:>15} | {:>10}", k, count);
    }
    let total_cells: usize = bins_sorted.iter().map(|(_, c)| c).sum();
    println!("        {:>15} | {:>10}", "Total", total_cells);
    println!();

    println!("Step 5: compute per-particle attribution. Each particle in a");
    println!("        cell of size n_cell contributes 1/n_cell to bin n_cell.");
    println!("        Sum over all particles of their contribution to bin k");
    println!("        must equal bin k's count. (Conservation invariant.)\n");

    let mut bin_attribution: HashMap<usize, f64> = HashMap::new();
    let mut per_particle_attr: Vec<f64> = vec![0.0; mem.n_particles()];
    for (_, members) in mem.non_empty_cells_at(l) {
        let n = members.len();
        let per_particle = 1.0 / n as f64;
        for &p in members {
            *bin_attribution.entry(n).or_insert(0.0) += per_particle;
            per_particle_attr[p as usize] = per_particle;
        }
    }
    println!("        bin (cell size) | direct count | attribution sum");
    println!("        ----------------+--------------+----------------");
    for (k, count) in &bins_sorted {
        let attr = bin_attribution.get(k).copied().unwrap_or(0.0);
        println!(
            "        {:>15} | {:>12} | {:>15.6}",
            k, count, attr
        );
    }
    println!();

    println!("Step 6: identify outliers in per-particle influence.");
    println!("        Particles in dense cells have LOW per-particle");
    println!("        contribution (their credit is split many ways);");
    println!("        particles in sparse cells have HIGH contribution.\n");

    let mut sorted_attr: Vec<(usize, f64)> = per_particle_attr
        .iter()
        .enumerate()
        .map(|(i, &a)| (i, a))
        .collect();
    sorted_attr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("        TOP 5 most-influential particles (singletons or near):");
    println!("        particle_idx | contribution | cell_size");
    println!("        -------------+--------------+----------");
    for (i, attr) in sorted_attr.iter().take(5) {
        let cell_size = (1.0 / attr).round() as usize;
        println!(
            "        {:>12} | {:>12.6} | {:>9}",
            i, attr, cell_size
        );
    }
    println!();

    println!("        BOTTOM 5 least-influential particles (in dense cells):");
    println!("        particle_idx | contribution | cell_size");
    println!("        -------------+--------------+----------");
    for (i, attr) in sorted_attr.iter().rev().take(5) {
        let cell_size = (1.0 / attr).round() as usize;
        println!(
            "        {:>12} | {:>12.6} | {:>9}",
            i, attr, cell_size
        );
    }
    println!();

    println!("==========================================================");
    println!(" The same pattern works for any cascade statistic. For");
    println!(" field-stats variance, the per-particle coefficient is");
    println!(" 2 * delta(c) * (W_r(c) / sum_W_r) / (alpha * W_r(c))");
    println!(" instead of 1/n_cell. The cell-membership index is the");
    println!(" structural primitive; per-statistic backward passes");
    println!(" (commit 8+) plug into it for efficient gradients.");
    println!("==========================================================");
}
