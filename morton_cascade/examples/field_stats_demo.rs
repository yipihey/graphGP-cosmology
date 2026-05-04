// Demo: simulate clustered data inside a survey footprint, run field-stats,
// print moments and a histogram tail.

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

fn main() {
    let tmp = "/tmp/field_stats_demo";
    std::fs::create_dir_all(tmp).unwrap();
    let box_size = 100.0_f64;

    // Survey footprint: octant (x > 0.5L AND y > 0.5L). Implemented by
    // putting randoms only in that region. Data is also restricted to that
    // region but is *clustered* (Gaussian blobs).
    let mut s = 314159u64;
    let n_d = 5_000usize;
    let n_r = 50_000usize;

    // Cluster centers in the octant
    let n_clusters = 30;
    let centers: Vec<[f64; 3]> = (0..n_clusters).map(|_| [
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
    ]).collect();

    // Data: Gaussian-like blob around each center, with cluster size ~ box/64
    let mut data: Vec<[f64; 3]> = Vec::new();
    let blob_size = box_size / 64.0;
    while data.len() < n_d {
        let c = &centers[(splitmix64(&mut s) % n_clusters as u64) as usize];
        let dx = (uniform(&mut s) - 0.5) * 2.0 * blob_size;
        let dy = (uniform(&mut s) - 0.5) * 2.0 * blob_size;
        let dz = (uniform(&mut s) - 0.5) * 2.0 * blob_size;
        let p = [c[0] + dx, c[1] + dy, c[2] + dz];
        // Reject outside octant or outside box
        if p[0] >= 0.5*box_size && p[0] < box_size
            && p[1] >= 0.5*box_size && p[1] < box_size
            && p[2] >= 0.5*box_size && p[2] < box_size {
            data.push(p);
        }
    }

    // Randoms: uniform inside the octant
    let randoms: Vec<[f64; 3]> = (0..n_r).map(|_| [
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
        0.5 * box_size + 0.5 * box_size * uniform(&mut s),
    ]).collect();

    write_pts3(&format!("{}/data.bin", tmp), &data);
    write_pts3(&format!("{}/randoms.bin", tmp), &randoms);

    println!("Generated {} clustered data + {} uniform randoms inside upper octant",
        data.len(), randoms.len());
    println!("Cluster size ~ box/64 = {} units", blob_size);

    // Run the CLI
    let out = Command::new("./target/release/morton-cascade")
        .args(["field-stats",
            "-i", &format!("{}/data.bin", tmp),
            "--randoms", &format!("{}/randoms.bin", tmp),
            "-d", "3",
            "-L", &box_size.to_string(),
            "-o", tmp,
            "--hist-bins", "30",
            "--w-r-min", "0.5",  // require at least 1 random in cell to count
        ])
        .output().unwrap();
    if !out.status.success() {
        println!("CLI stderr:");
        println!("{}", String::from_utf8_lossy(&out.stderr));
        panic!("CLI failed");
    }
    println!("CLI stderr:");
    print!("{}", String::from_utf8_lossy(&out.stderr));

    // Print first 12 rows of moments
    println!("\n--- field_moments.csv (first 12 rows) ---");
    let mom = std::fs::read_to_string(&format!("{}/field_moments.csv", tmp)).unwrap();
    for line in mom.lines().take(12) {
        println!("{}", line);
    }
    println!("...");

    // Print PDF at one informative level (where we expect to see clustering)
    let pdf = std::fs::read_to_string(&format!("{}/field_pdf.csv", tmp)).unwrap();
    let target_level = 4usize;
    println!("\n--- field_pdf.csv at level {} (cells of side ~box/{}) ---",
        target_level, 1u64 << target_level);
    println!("delta_lo, delta_hi, density");
    for line in pdf.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols[0].parse::<usize>().unwrap_or(0) == target_level {
            let delta_lo: f64 = cols[3].parse().unwrap();
            let delta_hi: f64 = cols[4].parse().unwrap();
            let density: f64 = cols[7].parse().unwrap();
            if density > 1e-4 {
                println!("  [{:>7.3}, {:>7.3}]   density = {:.4}",
                    delta_lo, delta_hi, density);
            }
        }
    }
}
