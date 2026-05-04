// test_3d.rs
//
// 3D cascade tests: Schur ratio validation on Poisson, Cox sample with TPCF,
// scaling check.
//
// Usage:
//     cargo run --release --example test_3d

use morton_cascade::{hier_3d, hier_3d::cascade_3d_with_tpcf, L_MAX_3D, N_LEVELS_3D};
use std::time::Instant;
use std::io::Write;

#[path = "common.rs"]
mod common;
use common::{LogNormalField3D, Rng};

fn main() {
    let mut rng = Rng::new(20260502);

    println!("================================================================================");
    println!("  3D CASCADE TESTS (L_MAX_3D = {}, tree-coord box = {} per side)",
             L_MAX_3D, 1usize << L_MAX_3D);
    println!("================================================================================");

    // ---------- Test 1: 3D Poisson Schur ratio (predicted: 57/64 = 0.8906) ----------
    println!("\n--- 3D Poisson: expected dvar/<N> = 57/64 = 0.8906 at every level ---");
    let n_real = 8;
    let n_per = 50_000usize;
    let scale = (1u32 << 16) as f64;
    let mut acc_dvar = vec![0.0f64; N_LEVELS_3D];
    let mut acc_mean = vec![0.0f64; N_LEVELS_3D];
    for _ in 0..n_real {
        let mut p: Vec<(u16,u16,u16)> = Vec::with_capacity(n_per);
        let n_actual = rng.poisson(n_per as f64) as usize;
        for _ in 0..n_actual {
            let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
            let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
            let z = (rng.uniform() * scale).min(scale - 1.0) as u16;
            p.push((x, y, z));
        }
        let (s, _, _) = cascade_3d_with_tpcf(&p, 1, true, &[]);
        for l in 0..N_LEVELS_3D {
            acc_dvar[l] += s[l].dvar;
            acc_mean[l] += s[l].mean;
        }
    }
    println!("  {:>3}  {:>10}  {:>10}  {:>10}", "l", "<N>", "dvar", "ratio");
    for l in 0..N_LEVELS_3D {
        let m = acc_mean[l] / n_real as f64;
        let d = acc_dvar[l] / n_real as f64;
        if m > 1e-9 {
            println!("  {:>3}  {:>10.4}  {:>10.4}  {:>10.4}", l, m, d, d/m);
        }
    }

    // ---------- Test 2: 3D Cox sample, write points + cascade output ----------
    println!("\n--- 3D Cox process: build field, sample points, run cascade ---");
    let g_field = 64usize;
    println!("  Building 3D log-normal field (G={}, alpha=1.5, sigma_g^2 ~ 0.4)...", g_field);
    let t = Instant::now();
    let field3d = LogNormalField3D::new(g_field, 1.5, 0.4, 600, &mut rng);
    println!("  built in {:?}; realized sigma_g^2 = {:.4}", t.elapsed(), field3d.sigma2_g);

    let n_3d = 200_000usize;
    let n_real_3d = 8usize;
    let dir = "/tmp/cascade_3d_run";
    std::fs::create_dir_all(dir).ok();
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() { std::fs::remove_file(p).ok(); }
    }
    println!("  Writing {} realizations of {} points each to {}", n_real_3d, n_3d, dir);

    let lag_levels: Vec<usize> = (3..=L_MAX_3D).collect();
    let mut total_casc_time = 0.0;
    for i in 0..n_real_3d {
        let p = field3d.sample(n_3d, &mut rng);
        let mut pf = std::fs::File::create(format!("{}/points_{:03}.bin", dir, i)).unwrap();
        for &(x, y, z) in &p {
            let xf = (x as f64) / 512.0;
            let yf = (y as f64) / 512.0;
            let zf = (z as f64) / 512.0;
            pf.write_all(&xf.to_le_bytes()).unwrap();
            pf.write_all(&yf.to_le_bytes()).unwrap();
            pf.write_all(&zf.to_le_bytes()).unwrap();
        }

        let t_c = Instant::now();
        let (st, _, tps) = cascade_3d_with_tpcf(&p, 1, true, &lag_levels);
        total_casc_time += t_c.elapsed().as_secs_f64();

        let mut tf = std::fs::File::create(format!("{}/tpcf_{:03}.csv", dir, i)).unwrap();
        writeln!(tf, "level,k,r_tree,r_fine,smoothing_h_fine,xi_measured,n_pairs").unwrap();
        for tp in &tps {
            writeln!(tf, "{},{},{},{},{},{:.8e},{}",
                tp.level, tp.k, tp.r_tree, tp.r_fine, tp.smoothing_h_fine, tp.xi, tp.n_pairs).unwrap();
        }

        let mut sf = std::fs::File::create(format!("{}/stats_{:03}.csv", dir, i)).unwrap();
        writeln!(sf, "level,R_tree,n_cells,mean,var,dvar,sigma2_field").unwrap();
        for l in 0..N_LEVELS_3D {
            let r_tree = (1usize << (L_MAX_3D - l)) as f64;
            let mean = st[l].mean;
            let var = st[l].var;
            let s2 = if mean > 1e-12 { (var - mean) / (mean * mean) } else { 0.0 };
            writeln!(sf, "{},{},{},{:.8e},{:.8e},{:.8e},{:.8e}",
                l, r_tree, st[l].n_cells_total, mean, var, st[l].dvar, s2).unwrap();
        }
        println!("    realization {}/{}: {} pts, cascade {:.0} ms",
                 i+1, n_real_3d, p.len(), t_c.elapsed().as_secs_f64() * 1000.0);
    }
    println!("  Mean cascade time per realization: {:.0} ms",
             total_casc_time / n_real_3d as f64 * 1000.0);

    // ---------- Test 3: 3D N-scaling ----------
    println!("\n--- 3D cascade N-scaling (uniform points) ---");
    println!("  {:>10} {:>10}", "N", "tot (ms)");
    for &n_test in &[200_000usize, 1_000_000] {
        let mut p_test: Vec<(u16,u16,u16)> = Vec::with_capacity(n_test);
        for _ in 0..n_test {
            let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
            let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
            let z = (rng.uniform() * scale).min(scale - 1.0) as u16;
            p_test.push((x, y, z));
        }
        let t = Instant::now();
        let _ = cascade_3d_with_tpcf(&p_test, 1, true, &[]);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        println!("  {:>10} {:>10.1}", n_test, dt);
    }

    // Suppress unused warning
    let _ = hier_3d::bin_to_fine_grid_3d;
}
