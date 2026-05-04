// Use 4 points with KNOWN coordinates and verify cumulative_pairs.
use morton_cascade::coord_range::TrimmedPoints;
use morton_cascade::hier_bitvec::BitVecCascade;

fn main() {
    // 4 points in a 2D u8 box, eff_bits should be 8.
    // Points: (0,0), (1,1), (128,128), (255,255).
    // At level 0: 1 cell of 4 pts -> 6 pairs.
    // At level 1: cells are 128-wide. (0,0),(1,1) in cell (0,0); (128,128),(255,255) in cell (1,1) -> 1+1 = 2 pairs.
    // At level 2: cells 64-wide. Same as level 1 (within-pair distance < 64). 2 pairs.
    // At level 3: cells 32-wide. (0,0)(1,1) still in (0,0); (128,128)(255,255) -> 128 in cell (4,4), 255 in cell (7,7), DIFFERENT. So 1 pair.
    // At level 4: cells 16-wide. (0,0)(1,1) still together in (0,0). (128,128)(255,255) split.
    // ...
    // At level 7: cells 2-wide. (0,0) in (0,0); (1,1) in (0,0). Still together, 1 pair.
    // At level 8: cells 1-wide. (0,0) in (0,0); (1,1) in (1,1). Separate. 0 pairs.

    let pts: Vec<[u64; 2]> = vec![
        [0, 0], [1, 1], [128, 128], [255, 255]
    ];
    let trimmed = TrimmedPoints::from_points(pts);
    println!("eff_bits = {:?}", trimmed.range.effective_bits);
    println!("trimmed points = {:?}", trimmed.points);

    let casc = BitVecCascade::<2>::build(trimmed, None);
    println!("l_max = {}", casc.l_max);
    let stats = casc.analyze();
    println!("level  side    n_nonempty  cum_pairs");
    for st in &stats {
        let side = if st.level <= 8 { 1u64 << (8 - st.level) } else { 1 };
        println!("{:>5}  {:>4}    {:>10}  {:>9}",
            st.level, side, st.n_nonempty_cells, st.cumulative_pairs);
    }
    println!("Expected (l=0): 6 pairs; (l=8): 0 pairs.");
}
