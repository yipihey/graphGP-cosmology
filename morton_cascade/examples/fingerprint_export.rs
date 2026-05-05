// fingerprint_export.rs
//
// Generates the cascade fingerprint data: many Cox realizations + matched
// Poisson references. Writes CSVs to /tmp/cascade_summary/ for downstream
// plotting (see cascade_plot_v3.py).
//
// Usage:
//     cargo run --release --example fingerprint_export

use morton_cascade::{hier, hier::PmfLevel, hier::TpcfPoint, LevelStats, L_MAX};
use std::io::Write;
use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::{LogNormalField, Rng};

fn main() {
    let mut rng = Rng::new(20260502);

    println!("================================================================================");
    println!("  CASCADE FINGERPRINT EXPORT (multi-realization, with Poisson reference)");
    println!("================================================================================\n");

    // Cox field
    let g_field = 128usize;
    println!("Building 2D log-normal Cox field (G={}, alpha=1.5, sigma_g^2 = 0.4)...", g_field);
    let field = LogNormalField::new(g_field, 1.5, 0.4, 800, &mut rng);

    let n_pts = 200_000usize;
    let n_real = 16usize;
    println!("Drawing {} Cox realizations of {} points each...", n_real, n_pts);

    let mut all_ls_cox: Vec<Vec<LevelStats>> = Vec::with_capacity(n_real);
    let mut all_pmfs_cox: Vec<Vec<PmfLevel>> = Vec::with_capacity(n_real);
    let mut all_tpcf_cox: Vec<Vec<TpcfPoint>> = Vec::with_capacity(n_real);
    let lag_levels: Vec<usize> = (1..=8).collect();

    let t = Instant::now();
    for i in 0..n_real {
        let pts = field.sample(n_pts, &mut rng);
        let (ls, _, pmfs) = hier::cascade_with_pmf(&pts, 1, true);
        let (_, _, tpcf) = hier::cascade_hierarchical_with_tpcf(&pts, 1, true, &lag_levels);
        all_ls_cox.push(ls);
        all_pmfs_cox.push(pmfs);
        all_tpcf_cox.push(tpcf);
        if (i + 1) % 4 == 0 { println!("  {} / {} done", i + 1, n_real); }
    }
    println!("Cox cascade total: {:?}\n", t.elapsed());

    // Poisson reference
    println!("Drawing {} uniform Poisson references of {} points each...", n_real, n_pts);
    let scale = (1u32 << 16) as f64;
    let mut all_ls_ref: Vec<Vec<LevelStats>> = Vec::with_capacity(n_real);
    let mut all_pmfs_ref: Vec<Vec<PmfLevel>> = Vec::with_capacity(n_real);
    let mut all_tpcf_ref: Vec<Vec<TpcfPoint>> = Vec::with_capacity(n_real);

    let t = Instant::now();
    for _ in 0..n_real {
        let mut pts: Vec<(u16, u16)> = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
            let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
            pts.push((x, y));
        }
        let (ls, _, pmfs) = hier::cascade_with_pmf(&pts, 1, true);
        let (_, _, tpcf) = hier::cascade_hierarchical_with_tpcf(&pts, 1, true, &lag_levels);
        all_ls_ref.push(ls);
        all_pmfs_ref.push(pmfs);
        all_tpcf_ref.push(tpcf);
    }
    println!("Poisson cascade total: {:?}\n", t.elapsed());

    // Output
    let dir = "/tmp/cascade_summary";
    std::fs::create_dir_all(dir).ok();
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() { std::fs::remove_file(p).ok(); }
    }

    write_stats(&format!("{}/level_stats_cox.csv", dir), &all_ls_cox, &all_pmfs_cox);
    write_stats(&format!("{}/level_stats_ref.csv", dir), &all_ls_ref, &all_pmfs_ref);
    write_pmfs(&format!("{}/pmfs_cox.csv", dir), &all_pmfs_cox);
    write_pmfs(&format!("{}/pmfs_ref.csv", dir), &all_pmfs_ref);
    write_tpcf(&format!("{}/tpcf_cox.csv", dir), &all_tpcf_cox);
    write_tpcf(&format!("{}/tpcf_ref.csv", dir), &all_tpcf_ref);

    // Spatial map at level 5 from a fresh realization
    {
        let l_map = 5usize;
        let n_per_axis = 1usize << l_map;
        let h_l = (1u32 << 16) / (n_per_axis as u32);
        let mut grid = vec![0u32; n_per_axis * n_per_axis];
        let pts0 = field.sample(n_pts, &mut rng);
        for &(x, y) in &pts0 {
            let cx = (x as u32 / h_l) as usize;
            let cy = (y as u32 / h_l) as usize;
            grid[cy * n_per_axis + cx] += 1;
        }
        let mut f = std::fs::File::create(format!("{}/spatial_map_l{}.csv", dir, l_map)).unwrap();
        writeln!(f, "cy,cx,count").unwrap();
        for cy in 0..n_per_axis {
            for cx in 0..n_per_axis {
                writeln!(f, "{},{},{}", cy, cx, grid[cy * n_per_axis + cx]).unwrap();
            }
        }
    }

    println!("Exported to {}/", dir);
    println!("  level_stats_cox.csv, level_stats_ref.csv");
    println!("  pmfs_cox.csv, pmfs_ref.csv");
    println!("  tpcf_cox.csv, tpcf_ref.csv");
    println!("  spatial_map_l5.csv");
    println!("\nNext: python3 cascade_plot_v3.py");
}

fn write_stats(path: &str, all_ls: &[Vec<LevelStats>], all_pmfs: &[Vec<PmfLevel>]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "realization,level,R_tree,n_cells,mean,var,dvar,sigma2_field,skew,kurt").unwrap();
    for (r, (ls_set, pmf_set)) in all_ls.iter().zip(all_pmfs.iter()).enumerate() {
        for (i, ls) in ls_set.iter().enumerate() {
            let r_tree = (1usize << (L_MAX - i)) as f64;
            let s2 = if ls.mean > 1e-12 { (ls.var - ls.mean) / (ls.mean * ls.mean) } else { 0.0 };
            let pmf = pmf_set.iter().find(|p| p.level == i);
            let (sk, ku) = pmf.map(|p| (p.skew, p.kurt)).unwrap_or((0.0, 0.0));
            writeln!(f, "{},{},{},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.6e},{:.6e}",
                r, i, r_tree, ls.n_cells_total, ls.mean, ls.var, ls.dvar, s2, sk, ku).unwrap();
        }
    }
}

fn write_pmfs(path: &str, all_pmfs: &[Vec<PmfLevel>]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "realization,level,R_tree,n_total,count,frequency").unwrap();
    for (r, pmf_set) in all_pmfs.iter().enumerate() {
        for p in pmf_set {
            for (k, &h) in p.histogram.iter().enumerate() {
                if h > 0 {
                    writeln!(f, "{},{},{},{},{},{}", r, p.level, p.r_tree, p.n_total, k, h).unwrap();
                }
            }
        }
    }
}

fn write_tpcf(path: &str, all_tpcf: &[Vec<TpcfPoint>]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "realization,level,k,r_tree,smoothing_h_fine,xi").unwrap();
    for (r, tpset) in all_tpcf.iter().enumerate() {
        for tp in tpset {
            writeln!(f, "{},{},{},{},{},{:.8e}",
                r, tp.level, tp.k, tp.r_tree, tp.smoothing_h_fine, tp.xi).unwrap();
        }
    }
}
