use morton_cascade::hier_nd::cascade_with_pmf_windows;
use std::time::Instant;

fn main() {
    let scale = (1u32 << 16) as f64;
    let mut state = 0xdeadbeefu64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };
    let pts: Vec<[u16; 3]> = (0..200_000).map(|_| {
        let x = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        let y = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        let z = ((next() as f64 / u64::MAX as f64) * scale) as u16;
        [x, y, z]
    }).collect();

    // Same window k=2 three times in a row in separate calls
    for i in 0..3 {
        let t = Instant::now();
        let _ = cascade_with_pmf_windows::<3>(&pts, 7, 1, true, &[2]);
        println!("k=2 call {}: {:.3} s", i+1, t.elapsed().as_secs_f64());
    }

    // Single call with same k three times
    let t = Instant::now();
    let _ = cascade_with_pmf_windows::<3>(&pts, 7, 1, true, &[2, 2, 2]);
    println!("[2,2,2] in one call: {:.3} s", t.elapsed().as_secs_f64());
}
