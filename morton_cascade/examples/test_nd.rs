// test_nd.rs
//
// Generic D-dimensional cascade tests. Validates correctness against the
// hand-written 2D and 3D codes, and exercises D=4 and D=5.
//
// Usage:
//     cargo run --release --example test_nd

use morton_cascade::{hier, hier_3d, hier_nd::{cascade_nd, LevelStatsND}};
use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::Rng;

fn main() {
    let mut rng = Rng::new(20260502);

    println!("================================================================================");
    println!("  GENERIC D-DIMENSIONAL CASCADE TESTS");
    println!("================================================================================");

    // ---------- D=2 generic vs hand-written ----------
    println!("\n--- D=2 generic vs hand-written 2D (correctness + speed) ---");
    let n_2d = 800_000usize;
    let scale = (1u32 << 16) as f64;
    let pts_2d: Vec<(u16, u16)> = (0..n_2d).map(|_| {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        (x, y)
    }).collect();
    let pts_2d_arr: Vec<[u16; 2]> = pts_2d.iter().map(|&(x, y)| [x, y]).collect();

    let t = Instant::now();
    let (st_2d_ref, _) = hier::cascade_hierarchical_bc(&pts_2d, 1, true);
    let t_ref = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (st_2d_nd, _) = cascade_nd::<2>(&pts_2d_arr, 8, 1, true);
    let t_nd = t.elapsed().as_secs_f64() * 1000.0;

    let mut max_diff = 0.0f64;
    for l in 0..st_2d_ref.len() {
        let dv = (st_2d_ref[l].var - st_2d_nd[l].var).abs() / st_2d_ref[l].var.abs().max(1e-30);
        let dd = (st_2d_ref[l].dvar - st_2d_nd[l].dvar).abs() / st_2d_ref[l].dvar.abs().max(1e-30);
        max_diff = max_diff.max(dv).max(dd);
    }
    println!("  N = {}, max relative diff (var, dvar) = {:.3e}", n_2d, max_diff);
    println!("  Hand-written 2D: {:.2} ms", t_ref);
    println!("  Generic D=2:     {:.2} ms", t_nd);
    println!("  Overhead:        {:.2}x", t_nd / t_ref);

    // ---------- D=3 generic vs hand-written ----------
    println!("\n--- D=3 generic vs hand-written 3D (correctness) ---");
    let n_3d = 100_000usize;
    let pts_3d: Vec<(u16, u16, u16)> = (0..n_3d).map(|_| {
        let x = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let y = (rng.uniform() * scale).min(scale - 1.0) as u16;
        let z = (rng.uniform() * scale).min(scale - 1.0) as u16;
        (x, y, z)
    }).collect();
    let pts_3d_arr: Vec<[u16; 3]> = pts_3d.iter().map(|&(x, y, z)| [x, y, z]).collect();

    let t = Instant::now();
    let (st_3d_ref, _, _) = hier_3d::cascade_3d_with_tpcf(&pts_3d, 1, true, &[]);
    let t_ref3 = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (st_3d_nd, _) = cascade_nd::<3>(&pts_3d_arr, 7, 1, true);
    let t_nd3 = t.elapsed().as_secs_f64() * 1000.0;

    let mut max_diff_3d = 0.0f64;
    for l in 0..st_3d_ref.len() {
        let dv = (st_3d_ref[l].var - st_3d_nd[l].var).abs() / st_3d_ref[l].var.abs().max(1e-30);
        let dd = (st_3d_ref[l].dvar - st_3d_nd[l].dvar).abs() / st_3d_ref[l].dvar.abs().max(1e-30);
        max_diff_3d = max_diff_3d.max(dv).max(dd);
    }
    println!("  N = {}, max relative diff (var, dvar) = {:.3e}", n_3d, max_diff_3d);
    println!("  Hand-written 3D: {:.2} ms", t_ref3);
    println!("  Generic D=3:     {:.2} ms", t_nd3);
    println!("  Overhead:        {:.2}x", t_nd3 / t_ref3);

    // ---------- Schur ratio at D = 2, 3, 4, 5 ----------
    println!("\n--- Poisson Schur dvar/<N> at multiple D (predicted: 1 - (2^D-1)/4^D) ---");

    fn run_poisson<const D: usize>(rng: &mut Rng, l_max: usize, n_real: usize, n_per: usize) -> Vec<LevelStatsND> {
        let mut acc: Vec<LevelStatsND> = (0..(l_max+1))
            .map(|l| LevelStatsND { n_cells_total: 1usize << (D*l), mean: 0.0, var: 0.0, dvar: 0.0 })
            .collect();
        let scale = (1u32 << 16) as f64;
        for _ in 0..n_real {
            let mut p: Vec<[u16; D]> = Vec::with_capacity(n_per);
            for _ in 0..n_per {
                let mut x = [0u16; D];
                for d in 0..D {
                    x[d] = (rng.uniform() * scale).min(scale - 1.0) as u16;
                }
                p.push(x);
            }
            let (st, _) = cascade_nd::<D>(&p, l_max, 1, true);
            for l in 0..(l_max+1) {
                acc[l].mean += st[l].mean;
                acc[l].var  += st[l].var;
                acc[l].dvar += st[l].dvar;
            }
        }
        for l in 0..(l_max+1) {
            acc[l].mean /= n_real as f64;
            acc[l].var  /= n_real as f64;
            acc[l].dvar /= n_real as f64;
        }
        acc
    }

    let cases: [(&str, usize); 4] = [("2", 8), ("3", 5), ("4", 3), ("5", 2)];
    for (d_str, l_max_test) in cases.iter() {
        let r: Vec<LevelStatsND> = match *d_str {
            "2" => run_poisson::<2>(&mut rng, *l_max_test, 4, 50_000),
            "3" => run_poisson::<3>(&mut rng, *l_max_test, 4, 50_000),
            "4" => run_poisson::<4>(&mut rng, *l_max_test, 4, 50_000),
            "5" => run_poisson::<5>(&mut rng, *l_max_test, 4, 50_000),
            _ => unreachable!(),
        };
        let d: f64 = d_str.parse().unwrap();
        let predicted = 1.0 - ((1u64 << (d as u64)) - 1) as f64 / (1u64 << ((2.0 * d) as u64)) as f64;
        println!("  D={}: predicted dvar/<N> = {:.6}", d_str, predicted);
        for (l, s) in r.iter().enumerate() {
            if s.mean > 1e-9 {
                println!("    l={}  <N>={:>11.4}  dvar={:>11.4}  ratio={:.4}",
                         l, s.mean, s.dvar, s.dvar / s.mean);
            }
        }
    }

    println!("\nAll D values use the same generic code (cascade_nd<const D: usize>).");
}
