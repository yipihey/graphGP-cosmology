// pmf_cox_demo.rs
//
// Run the windowed P_N(V) measurement on a 2D Cox field, with a matched
// Poisson reference, and dump CSVs for plotting.
//
// Usage:  cargo run --release --example pmf_cox_demo

use morton_cascade::{
    hier_nd::{cascade_with_pmf_windows, log_spaced_window_sides, PmfWindow},
    L_MAX,
};
use std::io::Write;
use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::{LogNormalField, Rng};

fn main() {
    let mut rng = Rng::new(20260502);

    // 2D log-normal Cox field
    let g_field = 128usize;
    println!("Building 2D Cox field (G={}, alpha=1.5, sigma_g^2 = 0.4)...", g_field);
    let field = LogNormalField::new(g_field, 1.5, 0.4, 800, &mut rng);
    println!("  realized sigma_g^2 = {:.4}", field.sigma2_g);

    let n_pts = 200_000usize;
    println!("Sampling {} Cox points...", n_pts);
    let cox_pts = field.sample(n_pts, &mut rng);
    let cox_arr: Vec<[u16; 2]> = cox_pts.iter().map(|&(x, y)| [x, y]).collect();

    // Matched Poisson reference, same N
    println!("Sampling {} Poisson points (matched N)...", n_pts);
    let scale = (1u32 << 16) as f64;
    let mut poi_pts: Vec<(u16, u16)> = Vec::with_capacity(n_pts);
    for _ in 0..n_pts {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        poi_pts.push((x, y));
    }
    let poi_arr: Vec<[u16; 2]> = poi_pts.iter().map(|&(x, y)| [x, y]).collect();

    // Log-spaced cube-window sides: 5 points per decade in V (default)
    // 2D: M = 2^(L_max + s_sub) = 2^9 = 512 fine-grid units per axis.
    let m_eff: usize = 1 << (L_MAX + 1);
    let sides = log_spaced_window_sides(1, m_eff / 2, 2, 5.0);
    println!("Sides ({} distinct values): {:?}", sides.len(), sides);

    // Run windowed PMFs on both
    let t = Instant::now();
    let pmf_cox = cascade_with_pmf_windows::<2>(&cox_arr, L_MAX, 1, true, &sides);
    println!("Cox windowed PMF: {:?}", t.elapsed());

    let t = Instant::now();
    let pmf_poi = cascade_with_pmf_windows::<2>(&poi_arr, L_MAX, 1, true, &sides);
    println!("Poisson windowed PMF: {:?}", t.elapsed());

    // Dump CSVs
    let dir = "/tmp/pmf_cox_demo";
    std::fs::create_dir_all(dir).ok();
    write_pmfs(&format!("{}/pmf_cox.csv", dir), &pmf_cox).unwrap();
    write_pmfs(&format!("{}/pmf_poi.csv", dir), &pmf_poi).unwrap();
    write_moments(&format!("{}/moments_cox.csv", dir), &pmf_cox, n_pts, m_eff).unwrap();
    write_moments(&format!("{}/moments_poi.csv", dir), &pmf_poi, n_pts, m_eff).unwrap();

    println!("\nWrote PMFs and moments to {}/", dir);
    println!("  pmf_cox.csv, pmf_poi.csv: per-window histograms");
    println!("  moments_cox.csv, moments_poi.csv: per-window mean/var/skew/kurt + nu = mean count");
}

fn write_pmfs(path: &str, pmfs: &[PmfWindow]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "window_side,volume_tree,n_total,count,frequency")?;
    for p in pmfs {
        for (n, &h) in p.histogram.iter().enumerate() {
            if h > 0 {
                writeln!(f, "{},{},{},{},{}", p.window_side, p.volume_tree, p.n_total, n, h)?;
            }
        }
    }
    Ok(())
}

fn write_moments(path: &str, pmfs: &[PmfWindow], n_pts: usize, m_eff: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "window_side,volume_tree,nu_expected,mean,var,skew,kurt,P0_measured,P0_poisson")?;
    let m_vol = (m_eff as f64).powi(2);
    for p in pmfs {
        let nu = n_pts as f64 * p.volume_tree / m_vol;
        let p0_measured = p.histogram[0] as f64 / p.n_total as f64;
        let p0_poi = (-nu).exp();
        writeln!(f, "{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            p.window_side, p.volume_tree, nu, p.mean, p.var, p.skew, p.kurt, p0_measured, p0_poi)?;
    }
    Ok(())
}
