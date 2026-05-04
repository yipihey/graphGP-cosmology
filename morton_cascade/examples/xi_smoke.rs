// Smoke test: generate two 3D point catalogs as binary f64 files,
// run the `xi` CLI subcommand, then read back the CSV and check sanity.
//
// This is a stand-alone example, not a unit test (it shells out to a binary).

use std::fs::File;
use std::io::Write;
use std::process::Command;

fn write_f64_bin(path: &str, pts: &[[f64; 3]]) {
    let mut f = File::create(path).unwrap();
    for p in pts {
        for v in p {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

// SplitMix64 — high-quality PRNG that doesn't have the LCG low-bit pathology
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    let r = splitmix64(state);
    // Use top 53 bits for a uniform double in [0, 1)
    (r >> 11) as f64 / ((1u64 << 53) as f64)
}

fn main() {
    let tmpdir = "/tmp/xi_smoke";
    std::fs::create_dir_all(tmpdir).unwrap();

    let box_size = 100.0_f64;
    let n_d = 5_000;
    let n_r = 20_000;

    // Generate uniform data and uniform randoms — null test (xi should be ~0).
    let mut s = 42u64;
    let data: Vec<[f64; 3]> = (0..n_d).map(|_| {
        [uniform(&mut s) * box_size, uniform(&mut s) * box_size, uniform(&mut s) * box_size]
    }).collect();
    let randoms: Vec<[f64; 3]> = (0..n_r).map(|_| {
        [uniform(&mut s) * box_size, uniform(&mut s) * box_size, uniform(&mut s) * box_size]
    }).collect();
    write_f64_bin(&format!("{}/data.bin", tmpdir), &data);
    write_f64_bin(&format!("{}/randoms.bin", tmpdir), &randoms);

    // Run the CLI
    let status = Command::new("./target/release/morton-cascade")
        .args([
            "xi",
            "-i", &format!("{}/data.bin", tmpdir),
            "--randoms", &format!("{}/randoms.bin", tmpdir),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", tmpdir,
        ])
        .status().unwrap();
    assert!(status.success(), "CLI exited with {:?}", status);

    // Read back and print
    let csv = std::fs::read_to_string(&format!("{}/xi_landy_szalay.csv", tmpdir)).unwrap();
    println!("--- xi_landy_szalay.csv ---");
    for line in csv.lines() {
        println!("{}", line);
    }

    // Quickly check: in the matched-density case, |xi_LS| should be small at
    // moderate scales (not the very deepest where shot noise dominates).
    let mut max_abs_xi_mid = 0.0_f64;
    for line in csv.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        let level: usize = cols[0].parse().unwrap();
        let xi: f64 = cols[8].parse().unwrap_or(f64::NAN);
        // mid scales: skip level 0 (no inner edge) and skip the deepest 2
        if level >= 2 && level <= 4 && !xi.is_nan() {
            if xi.abs() > max_abs_xi_mid { max_abs_xi_mid = xi.abs(); }
        }
    }
    println!("max |xi| at levels 2..=4: {}", max_abs_xi_mid);
    assert!(max_abs_xi_mid < 0.5,
        "uniform-vs-uniform should give |xi| << 1 at moderate scales, got {}",
        max_abs_xi_mid);
    println!("OK: matched-density null test passed");
}
