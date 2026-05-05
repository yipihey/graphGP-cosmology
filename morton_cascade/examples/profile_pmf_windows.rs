// Profile each axis-pass separately so we know which dominates
use morton_cascade::hier_nd::{cascade_with_pmf_windows, log_spaced_window_sides};
use std::time::Instant;

fn main() {
    let m_eff: usize = 256;  // 2^8 (l_max=7, s_sub=1)
    let n_pts = 200_000;
    let scale = (1u32 << 16) as f64;

    let mut state = 0xdeadbeefu64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };
    let pts: Vec<[u16; 3]> = (0..n_pts).map(|_| {
        let x = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        let y = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        let z = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        [x, y, z]
    }).collect();

    // Profile a few representative window sizes
    for k in [2, 8, 32, 64, 128].iter() {
        let t = Instant::now();
        let _ = cascade_with_pmf_windows::<3>(&pts, 7, 1, true, &[*k]);
        println!("k={:>3}: {:.3} s for one window", k, t.elapsed().as_secs_f64());
    }

    // All log-spaced
    let sides = log_spaced_window_sides(1, 128, 3, 5.0);
    println!("\n{} log-spaced sides", sides.len());
    let t = Instant::now();
    let _ = cascade_with_pmf_windows::<3>(&pts, 7, 1, true, &sides);
    println!("Total: {:.3} s = {:.2} ms/window", 
             t.elapsed().as_secs_f64(),
             t.elapsed().as_secs_f64() * 1000.0 / sides.len() as f64);
}
