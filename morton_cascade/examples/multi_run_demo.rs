// Demo: cosmological-box analysis using the multi-run cascade.
//
// Generates a uniform-random periodic box, then runs three flavors of
// the multi-run subcommand to illustrate the typical workflow:
//
//   1. Field-stats with shift-bootstrap error bars
//   2. Anisotropy with log-spaced resize factors (non-dyadic scales)
//   3. CIC PMF with both shifts and resizings combined
//
// For each, prints the first few rows of the output CSV.

use std::fs::File;
use std::io::Write;
use std::process::Command;

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

fn write_pts3(path: &str, pts: &[[f64; 3]]) {
    let mut f = File::create(path).unwrap();
    for p in pts {
        for v in p {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

fn print_csv_head(path: &str, max_rows: usize, max_cols: usize) {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s, Err(e) => { println!("(could not read {}: {})", path, e); return; }
    };
    for (i, line) in content.lines().take(max_rows + 1).enumerate() {
        let cols: Vec<&str> = line.split(',').take(max_cols).collect();
        let prefix = if i == 0 { "  HEADER  " } else { "  row     " };
        println!("{}: {}{}", prefix, cols.join(","),
            if line.split(',').count() > max_cols { ", ..." } else { "" });
    }
    println!();
}

fn run_cli(label: &str, args: &[&str]) {
    println!("=== {} ===", label);
    println!("$ morton-cascade {}", args.join(" "));
    let out = Command::new("./target/release/morton-cascade")
        .args(args).output()
        .expect("failed to spawn morton-cascade — did you `cargo build --release`?");
    if !out.status.success() {
        println!("STDERR:");
        println!("{}", String::from_utf8_lossy(&out.stderr));
        panic!("CLI invocation failed");
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.is_empty() {
        for line in stderr.lines() {
            println!("  cli> {}", line);
        }
    }
}

fn main() {
    let tmp = "/tmp/multi_run_demo";
    std::fs::create_dir_all(tmp).unwrap();
    let box_size = 100.0_f64;
    let n_d = 5_000usize;

    // Uniform-random data in a periodic box.
    let mut s = 271828u64;
    let data: Vec<[f64; 3]> = (0..n_d).map(|_| [
        uniform(&mut s) * box_size,
        uniform(&mut s) * box_size,
        uniform(&mut s) * box_size,
    ]).collect();
    let data_path = format!("{}/data.bin", tmp);
    write_pts3(&data_path, &data);

    println!("Generated {} uniform-random points in a {}^3 periodic box.\n",
        data.len(), box_size);

    // ----- Demo 1: field-stats with shift-bootstrap error bars -----
    //
    // 8 random shifts of the cell-grid origin give per-bin estimates
    // of the across-shift variance — cheap shift-bootstrap error bars.
    // For uniform fields the per-shift mean δ is 0 to floating-point
    // precision (periodic-mode invariant), so var-of-var dominates.
    run_cli("Demo 1: field-stats with shift-bootstrap error bars", &[
        "multi-run",
        "-i", &data_path,
        "-d", "3",
        "-L", &box_size.to_string(),
        "-o", tmp,
        "--statistic", "field-stats",
        "--boundary", "periodic",
        "--n-shifts", "8",
        "--shift-magnitude", "0.25",
        "--shift-seed", "1",
        "--bin-tol", "0.01",
        "--max-depth", "7",
        "-q",
    ]);
    println!("→ multi_run_field_stats.csv  (one row per dyadic level)");
    println!("  Note `var_delta_arv`: shift-bootstrap variance of var_delta");
    print_csv_head(&format!("{}/multi_run_field_stats.csv", tmp), 5, 8);

    // ----- Demo 2: anisotropy with log-spaced resize factors -----
    //
    // 5 points-per-decade in volume gives non-dyadic intermediate scales.
    // Combined with shifts, every (shift × scale) cartesian product is
    // a separate cascade run — 4 × ~6 = ~24 runs total. The aggregator
    // pools by physical cell side, producing a smooth 5-ppd-V curve.
    run_cli("Demo 2: anisotropy with log-spaced resize + shifts", &[
        "multi-run",
        "-i", &data_path,
        "-d", "3",
        "-L", &box_size.to_string(),
        "-o", tmp,
        "--statistic", "anisotropy",
        "--boundary", "periodic",
        "--n-shifts", "4",
        "--shift-magnitude", "0.25",
        "--shift-seed", "2",
        "--resize-points-per-decade", "5",
        "--resize-min-scale", "0.5",
        "--bin-tol", "0.01",
        "--max-depth", "6",
        "-q",
    ]);
    println!("→ multi_run_anisotropy.csv  (one row per physical-side bin,");
    println!("   D-generic columns: w2_axis_<d> + w2_p<binary> for each pattern)");
    print_csv_head(&format!("{}/multi_run_anisotropy.csv", tmp), 6, 9);

    // ----- Demo 3: CIC PMF with combined shifts + resizings -----
    //
    // CIC PMF output is denser: one row per (side, count_bin) pair. The
    // periodic-mode density correction puts unvisited zero-data cells
    // into the k=0 bin so the per-side density sums to 1.
    run_cli("Demo 3: CIC PMF with combined shifts + log-spaced resizes", &[
        "multi-run",
        "-i", &data_path,
        "-d", "3",
        "-L", &box_size.to_string(),
        "-o", tmp,
        "--statistic", "cic-pmf",
        "--boundary", "periodic",
        "--n-shifts", "3",
        "--shift-magnitude", "0.25",
        "--shift-seed", "3",
        "--resize-points-per-decade", "5",
        "--resize-min-scale", "0.5",
        "--bin-tol", "0.01",
        "--max-depth", "5",
        "-q",
    ]);
    println!("→ multi_run_cic_pmf.csv  (one row per (side, count_bin) pair)");
    print_csv_head(&format!("{}/multi_run_cic_pmf.csv", tmp), 5, 9);

    // ----- Demo 4: ξ(r) with shift pooling and resize groups -----
    //
    // ξ requires a random catalog (Landy-Szalay), so we run in isolated
    // mode. Shifts pool DD/RR/DR within each resize group; resize groups
    // stay separate in the output because each scale measures different
    // shell volumes. To combine across resize groups into a continuous
    // ξ(r), feed the multi_run_xi_raw.csv into the SFH-style continuous
    // fit (commit 2 / examples/xi_graphgp_fit.py downstream).
    let n_r = 15_000usize;
    let randoms: Vec<[f64; 3]> = (0..n_r).map(|_| [
        uniform(&mut s) * box_size,
        uniform(&mut s) * box_size,
        uniform(&mut s) * box_size,
    ]).collect();
    let randoms_path = format!("{}/randoms.bin", tmp);
    write_pts3(&randoms_path, &randoms);

    run_cli("Demo 4: ξ(r) — shift-pooled, resize-grouped (with randoms)", &[
        "multi-run",
        "-i", &data_path,
        "--randoms", &randoms_path,
        "-d", "3",
        "-L", &box_size.to_string(),
        "-o", tmp,
        "--statistic", "xi",
        "--boundary", "isolated",
        "--n-shifts", "4",
        "--shift-magnitude", "0.25",
        "--shift-seed", "4",
        "--resize-factors", "1.0,0.7",
        "--bin-tol", "1e-6",
        "--max-depth", "5",
        "-q",
    ]);
    println!("→ multi_run_xi_raw.csv  (one row per (resize_group, shell);");
    println!("   shifts pooled within each resize group, scales kept separate;");
    println!("   `xi_shift_bootstrap_var` is the cheap per-shell uncertainty)");
    print_csv_head(&format!("{}/multi_run_xi_raw.csv", tmp), 6, 9);

    // Diagnostics: same shape regardless of statistic
    println!("=== Per-run diagnostics ===");
    println!("→ multi_run_diagnostics.csv  (one row per cascade run; the");
    println!("   `footprint_coverage` column is = 1.0 in periodic mode and");
    println!("   < 1.0 for sub-cubes that extend past the survey edge in");
    println!("   isolated mode)");
    print_csv_head(&format!("{}/multi_run_diagnostics.csv", tmp), 6, 5);

    println!("All output files written to: {}", tmp);
    println!();
    println!("Next steps to try:");
    println!("  • Use --boundary isolated with --randoms <path> for surveys");
    println!("    (resizing then becomes sub-cube clipping; the");
    println!("    diagnostics file's footprint_coverage shows how much of");
    println!("    each rescaled cube actually overlaps the survey).");
    println!("  • Vary --shift-magnitude (default 0.25 box-fraction) to");
    println!("    tune decorrelation between shift realizations.");
    println!("  • Increase --n-shifts to tighten shift-bootstrap error bars");
    println!("    (cost: linear in n_shifts × n_resize_factors).");
}
