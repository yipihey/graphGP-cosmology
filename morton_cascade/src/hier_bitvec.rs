// hier_bitvec.rs
//
// Bit-vector cascade. Per-particle bit representations let us prune empty
// cells and traverse to the depth set by data resolution (effective coord
// bits per axis), not by available memory for an M^D buffer.
//
// Headline new capability: pair counts per cube-shell at every level,
// produced as a byproduct of the same single traversal that yields cell
// count statistics.
//
// Algorithm:
//   - Build per-axis per-bit-position bit-planes: D x L vectors of N bits.
//     bit_planes[d][l].bit(i) = 1 iff point i has bit (L-1-l) of coord d set,
//     i.e. point i is in the upper half along axis d at descent step l.
//   - Recursive depth-first traversal. At level l with parent membership M_par
//     of count n_par >= threshold:
//       * Compute per-axis upper-half vectors: H_d = M_par AND bit_planes[d][l]
//       * For each of 2^D children indexed by bit pattern (b_0,...,b_{D-1}):
//           M_child = AND over d of (H_d if b_d else (M_par AND NOT bit_planes[d][l]))
//           If popcount(M_child) > 0, recurse.
//   - At each visited non-empty cell of count n_c, contribute:
//       * per-level cell count moments: sum n_c, sum n_c^2, histogram[n_c]++
//       * per-level pair count: cumulative_pairs[l] += n_c*(n_c-1)/2
//
// The traversal naturally skips empty cells (the "prune"). Total cost scales
// with the number of non-empty cells, not 2^(D*L).
//
// This is the simplest-correct version; performance optimizations
// (count-cascade for shallow levels, point-list switchover for very small
// cells) are deferred.

use crate::coord_range::TrimmedPoints;

/// Per-level statistics produced by the bit-vector cascade.
#[derive(Clone, Debug)]
pub struct BitVecLevelStats {
    /// Tree level (0 = whole box, larger = finer).
    pub level: usize,
    /// Number of *non-empty* cells visited at this level (after pruning).
    pub n_nonempty_cells: u64,
    /// Total number of cells at this level (2^(D*level)). Includes empty.
    pub n_total_cells: u64,
    /// Mean cell count, averaged over **non-empty** cells.
    pub mean_nonempty: f64,
    /// Mean cell count, averaged over **all** cells (including empty).
    /// Equal to N / n_total_cells.
    pub mean_all: f64,
    /// Variance over non-empty cells.
    pub var_nonempty: f64,
    /// Variance over all cells (zero-padded for empty).
    pub var_all: f64,
    /// Cumulative pair count: number of point-pairs sharing a cell at this level
    /// (or any deeper level). Equivalently: pairs at separation <= this cell's diagonal.
    pub cumulative_pairs: u64,
    /// Cell-count histogram: histogram[k] = number of cells with exactly k points.
    /// Empty cells (k=0) are NOT included in this histogram (they were pruned).
    pub histogram: Vec<u64>,
}

/// The bit-vector cascade itself.
pub struct BitVecCascade<const D: usize> {
    /// Trimmed points (right-shifted to drop dead trailing bits).
    pub points: TrimmedPoints<D>,
    /// L = number of cascade levels (= max effective bits across axes).
    /// Tree depth is L; level 0 = whole box, level L = finest.
    pub l_max: usize,
    /// Number of u64 words per bit-vector (= ceil(N/64)).
    pub n_words: usize,
    /// Crossover threshold: when a cell's count drops to this many points or fewer,
    /// switch from bit-vector representation to point-list for the descendants.
    /// Default 64 (one u64 word's worth) is conservative but safe.
    pub crossover_threshold: usize,
    /// bit_planes[d][l] = N-bit vector where bit i = 1 iff
    /// point i has bit (L_axis_d - 1 - l) of its (trimmed) axis-d coord set.
    /// Padded with zero-bits at the end of the last word.
    /// Note: levels beyond effective_bits[d] all have value zero (no further
    /// subdivision possible along that axis); we still allocate them to keep
    /// indexing uniform.
    pub bit_planes: Vec<Vec<Vec<u64>>>,
}

impl<const D: usize> BitVecCascade<D> {
    /// Default crossover threshold for `build`.
    ///
    /// The bit-vector path costs `~N/64` u64 ANDs per cell visit; the
    /// point-list path costs `~count * D` bit reads. Bit-vec wins when
    /// `count > N/(64*D)`. So using a constant default like 64 makes the
    /// crossover trigger far too eagerly at large N — the point-list path is
    /// invoked at cells that still have hundreds of points, where bit-vec is
    /// cheaper. Empirically (see `examples/bench_threshold_sweep.rs`) the
    /// constant default of 64 ran 3.9× slower than pure bit-vec at N=1M (2D
    /// l_max=16): 2871 ms vs 728 ms.
    ///
    /// The N-aware default below tracks the bit-vec / point-list breakeven
    /// point and recovers near-optimal performance at every N tested:
    /// `max(64, N / 64)`. The lower bound of 64 keeps small-N behavior
    /// identical to the old default; the `N/64` term handles large N.
    /// (We use 64 not `64*D` to stay slightly conservative — pulling D in
    /// would push the threshold higher in 3D and might be marginally worse
    /// at small N.)
    pub fn default_crossover_threshold(n: usize) -> usize {
        (n / 64).max(64)
    }

    /// Build the cascade from trimmed points. `l_max` defaults to the
    /// data-supported maximum if `None`; otherwise capped at the supported max.
    /// Crossover threshold uses the N-aware default; pass an explicit value to
    /// `build_with_threshold` to override.
    pub fn build(points: TrimmedPoints<D>, l_max: Option<usize>) -> Self {
        let n = points.len();
        Self::build_with_threshold(points, l_max, Self::default_crossover_threshold(n))
    }

    /// Build with explicit crossover threshold for the bit-vector → point-list switch.
    pub fn build_with_threshold(
        points: TrimmedPoints<D>,
        l_max: Option<usize>,
        crossover_threshold: usize,
    ) -> Self {
        let n = points.len();
        let n_words = (n + 63) / 64;

        let supported = points.range.max_supported_l_max() as usize;
        let l_max = l_max.map(|l| l.min(supported)).unwrap_or(supported);

        // Build bit planes. For each axis d and each level l in [0, l_max),
        // the relevant bit of the coordinate is at position (eff_d - 1 - l)
        // where eff_d = effective_bits[d]. If l >= eff_d, this axis has no
        // further information at this level and the plane is all zero
        // (every point is in the "lower half" along this axis at this level
        // because no bits below the bit_min are set).
        //
        // Because all axes' effective ranges have been right-shifted to
        // start at bit 0, the relevant bit for level l on axis d is
        // bit (eff_d - 1 - l) of points.points[i][d], if l < eff_d.
        let mut bit_planes: Vec<Vec<Vec<u64>>> = vec![vec![vec![0u64; n_words]; l_max]; D];
        for (i, p) in points.points.iter().enumerate() {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let mask = 1u64 << bit_idx;
            for d in 0..D {
                let eff = points.range.effective_bits[d] as usize;
                for l in 0..l_max.min(eff) {
                    let coord_bit = eff - 1 - l;
                    if (p[d] >> coord_bit) & 1 == 1 {
                        bit_planes[d][l][word_idx] |= mask;
                    }
                }
            }
        }

        Self { points, l_max, n_words, crossover_threshold, bit_planes }
    }

    /// Run the cascade traversal and accumulate per-level statistics.
    pub fn analyze(&self) -> Vec<BitVecLevelStats> {
        let n = self.points.len();
        let n_words = self.n_words;

        // Allocate accumulators per level.
        // We have l_max + 1 levels: 0 (whole box) through l_max (finest).
        let n_levels = self.l_max + 1;
        let mut sum_n = vec![0u128; n_levels];
        let mut sum_n2 = vec![0u128; n_levels];
        let mut n_nonempty = vec![0u64; n_levels];
        let mut cum_pairs = vec![0u64; n_levels];
        let mut histograms: Vec<Vec<u64>> = vec![Vec::new(); n_levels];

        // Initial membership = all bits set up to bit n-1, then masked
        let mut root_mem = vec![u64::MAX; n_words];
        // Mask off the unused bits in the last word
        if n > 0 && n % 64 != 0 {
            let last_word_bits = n % 64;
            root_mem[n_words - 1] = (1u64 << last_word_bits) - 1;
        }
        if n == 0 {
            // Pathological: no points. Return empty stats per level.
            return (0..n_levels).map(|l| BitVecLevelStats {
                level: l,
                n_nonempty_cells: 0,
                n_total_cells: 1u64 << (D * l),
                mean_nonempty: 0.0, mean_all: 0.0,
                var_nonempty: 0.0, var_all: 0.0,
                cumulative_pairs: 0,
                histogram: vec![],
            }).collect();
        }

        // Recurse from root
        self.recurse(&root_mem, 0, &mut sum_n, &mut sum_n2,
                     &mut n_nonempty, &mut cum_pairs, &mut histograms);

        // Build per-level stats
        let mut out: Vec<BitVecLevelStats> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let n_total: u64 = 1u64 << (D * l);
            let n_ne = n_nonempty[l];
            let mean_ne = if n_ne > 0 { sum_n[l] as f64 / n_ne as f64 } else { 0.0 };
            let var_ne = if n_ne > 0 {
                (sum_n2[l] as f64 / n_ne as f64) - mean_ne * mean_ne
            } else { 0.0 };
            let mean_all = n as f64 / n_total as f64;
            let var_all = if n_total > 0 {
                (sum_n2[l] as f64 / n_total as f64) - mean_all * mean_all
            } else { 0.0 };
            out.push(BitVecLevelStats {
                level: l,
                n_nonempty_cells: n_ne,
                n_total_cells: n_total,
                mean_nonempty: mean_ne,
                mean_all,
                var_nonempty: var_ne.max(0.0),
                var_all: var_all.max(0.0),
                cumulative_pairs: cum_pairs[l],
                histogram: std::mem::take(&mut histograms[l]),
            });
        }
        out
    }

    /// Recursive descent. `mem` is the membership vector for the current cell;
    /// `level` is the cell's tree level.
    fn recurse(
        &self,
        mem: &[u64],
        level: usize,
        sum_n: &mut [u128],
        sum_n2: &mut [u128],
        n_nonempty: &mut [u64],
        cum_pairs: &mut [u64],
        histograms: &mut [Vec<u64>],
    ) {
        // Count points in this cell
        let count = popcount_vec(mem) as u64;
        if count == 0 {
            return;   // pruned
        }

        // Accumulate this cell's contribution at its level
        sum_n[level] += count as u128;
        sum_n2[level] += (count as u128) * (count as u128);
        n_nonempty[level] += 1;
        cum_pairs[level] += count * (count - 1) / 2;
        let h = &mut histograms[level];
        if h.len() <= count as usize { h.resize(count as usize + 1, 0); }
        h[count as usize] += 1;

        // If we've reached the finest level, stop
        if level >= self.l_max {
            return;
        }

        // Crossover decision: if count is at or below threshold, switch to
        // point-list representation for descendants. Faster because each
        // descent step is O(D * n_c) instead of O(D * n_words).
        if (count as usize) <= self.crossover_threshold {
            // Extract point indices from the bit-vector once.
            let mut indices: Vec<u32> = Vec::with_capacity(count as usize);
            for (w_idx, &word) in mem.iter().enumerate() {
                let mut w = word;
                let base = (w_idx * 64) as u32;
                while w != 0 {
                    indices.push(base + w.trailing_zeros());
                    w &= w - 1;
                }
            }
            // Partition into 2^D children using bit_planes[d][level], then
            // recurse on each non-empty bucket at level+1. This mirrors the
            // bit-vec path which calls recurse once per child at level+1.
            let n_children = 1usize << D;
            let mut buckets: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            for &idx in &indices {
                let word_idx = (idx / 64) as usize;
                let bit_idx = (idx % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 {
                        child_id |= 1 << d;
                    }
                }
                buckets[child_id].push(idx);
            }
            for bucket in &buckets {
                if !bucket.is_empty() {
                    self.recurse_pointlist(bucket, level + 1, sum_n, sum_n2,
                                            n_nonempty, cum_pairs, histograms);
                }
            }
            return;
        }

        // Compute upper-half vectors for each axis at this descent step
        let n_words = self.n_words;
        let mut upper: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; n_words]);
        for d in 0..D {
            for w in 0..n_words {
                upper[d][w] = mem[w] & self.bit_planes[d][level][w];
            }
        }
        // Lower-half = mem AND NOT bit_planes (= mem XOR upper, since upper is subset of mem)
        let mut lower: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; n_words]);
        for d in 0..D {
            for w in 0..n_words {
                lower[d][w] = mem[w] & !self.bit_planes[d][level][w];
            }
        }

        // Recurse into 2^D children
        let mut child_mem = vec![0u64; n_words];
        for child in 0..(1u32 << D) {
            // Initialize with the half-vector for axis 0
            let b0 = (child & 1) != 0;
            child_mem.copy_from_slice(if b0 { &upper[0] } else { &lower[0] });
            // AND in the half-vectors for axes 1..D
            for d in 1..D {
                let bd = (child >> d) & 1 != 0;
                let half = if bd { &upper[d] } else { &lower[d] };
                for w in 0..n_words {
                    child_mem[w] &= half[w];
                }
            }
            self.recurse(&child_mem, level + 1, sum_n, sum_n2,
                         n_nonempty, cum_pairs, histograms);
        }
    }

    /// Point-list recursion. Once a cell's count is at or below the crossover
    /// threshold, we represent the cell's contents as an explicit list of point
    /// indices and partition by reading bit-plane bits per point.
    ///
    /// `indices` is the list of point indices in this cell.
    /// `level` is the cell's tree level.
    fn recurse_pointlist(
        &self,
        indices: &[u32],
        level: usize,
        sum_n: &mut [u128],
        sum_n2: &mut [u128],
        n_nonempty: &mut [u64],
        cum_pairs: &mut [u64],
        histograms: &mut [Vec<u64>],
    ) {
        let count = indices.len() as u64;
        if count == 0 {
            return;
        }

        // Accumulate this cell's contribution at its level
        sum_n[level] += count as u128;
        sum_n2[level] += (count as u128) * (count as u128);
        n_nonempty[level] += 1;
        cum_pairs[level] += count * (count - 1) / 2;
        let h = &mut histograms[level];
        if h.len() <= count as usize { h.resize(count as usize + 1, 0); }
        h[count as usize] += 1;

        if level >= self.l_max {
            return;
        }

        // Partition into 2^D children. For each point, compute its child index
        // by reading D bit-plane bits at the current descent level.
        // descent_level = level - 1 ... wait. The bit-plane index is the
        // descent step that creates this level. So to descend from level
        // (level) to level (level+1), we use bit_planes[d][level].
        let descent_level = level;   // bit plane index for descending FROM level to level+1
        let n_children = 1usize << D;
        let mut buckets: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        for &idx in indices {
            let word_idx = (idx / 64) as usize;
            let bit_idx = (idx % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 {
                    child_id |= 1 << d;
                }
            }
            buckets[child_id].push(idx);
        }

        for bucket in &buckets {
            if !bucket.is_empty() {
                self.recurse_pointlist(bucket, level + 1, sum_n, sum_n2,
                                        n_nonempty, cum_pairs, histograms);
            }
        }
    }
}

/// popcount over a vector of u64 words.
#[inline]
fn popcount_vec(v: &[u64]) -> u64 {
    let mut c: u64 = 0;
    for &w in v { c += w.count_ones() as u64; }
    c
}

// ============================================================================
// Phase 2: pair counts as first-class output, per-particle gradients
// ============================================================================

/// One axis-aligned cube-shell pair-count bin.
///
/// At level `level`, cells have (isotropic-case) side `cell_side_trimmed` in
/// trimmed-coordinate units. Pairs in this shell have separation in
/// roughly the range (`r_inner`, `r_outer`] in the same units. The exact
/// pair-separation distribution within a cube-shell is not concentrated at one
/// value: two points "in the same cube of side R" can be anywhere from 0 to
/// √D · R apart. The shell label is the cube-side scale.
#[derive(Clone, Debug)]
pub struct PairShell {
    /// Tree level (level 0 = whole box; cube-side at level l is box_side / 2^l).
    pub level: usize,
    /// Cube side at this level, in trimmed-coordinate units (assumes isotropic
    /// effective_bits across axes — for anisotropic data this is the side
    /// along the axis with the largest effective_bits).
    pub cell_side_trimmed: f64,
    /// Outer scale of the shell (= cube side at the parent level).
    /// For level 0 (root), `r_outer` is the box diagonal cube side.
    pub r_outer_trimmed: f64,
    /// Inner scale of the shell (= cube side at this level).
    pub r_inner_trimmed: f64,
    /// Number of pairs with separation in the shell (between adjacent levels).
    /// = cumulative_pairs[level - 1] - cumulative_pairs[level]
    /// For the deepest level, this is just cumulative_pairs[deepest], i.e. all
    /// pairs that share the finest cell.
    pub n_pairs: u64,
}

impl<const D: usize> BitVecCascade<D> {
    /// Convert per-level cumulative pair counts into per-shell bins.
    ///
    /// `box_side_trimmed`: the side of the box at level 0 in trimmed-coord
    /// units. For uniformly-trimmed data this is `2^max_eff_bits`.
    ///
    /// Returns one PairShell per level. The first shell (level 0) holds pairs
    /// with separation between the box diagonal and the level-1 cube side.
    /// In practice, level 0 always contains all N(N-1)/2 pairs as cumulative,
    /// so the level-0 shell pairs = cumulative_pairs[0] - cumulative_pairs[1].
    pub fn pair_counts_per_shell(
        &self,
        stats: &[BitVecLevelStats],
    ) -> Vec<PairShell> {
        // Determine the box side from the cascade's effective bits.
        // We use the largest eff_bits across axes; cells at level l have
        // side 2^(max_eff - l) along that axis.
        let max_eff = self.points.range.max_supported_l_max() as usize;
        let n_levels = stats.len();
        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            // Cube side at this level
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else {
                1.0   // can't go finer
            };
            // Outer scale = side at level (l-1); for l=0 that's the box itself
            let r_outer = if l == 0 {
                (1u64 << max_eff) as f64
            } else if (l - 1) <= max_eff {
                (1u64 << (max_eff - (l - 1))) as f64
            } else {
                1.0
            };
            // Pairs in this shell = cumulative[l] - cumulative[l+1]
            // For the deepest level, cumulative[l+1] doesn't exist; pairs in
            // shell = cumulative[l] (these are all pairs that fit in finest cell)
            let n_pairs = if l + 1 < n_levels {
                stats[l].cumulative_pairs.saturating_sub(stats[l + 1].cumulative_pairs)
            } else {
                stats[l].cumulative_pairs
            };
            out.push(PairShell {
                level: l,
                cell_side_trimmed: side_l,
                r_outer_trimmed: r_outer,
                r_inner_trimmed: side_l,
                n_pairs,
            });
        }
        out
    }

    /// Per-particle pair gradient at a given level.
    ///
    /// Returns a vector of length N: `grad[i]` = number of *other* points that
    /// share a cell with point i at the given level. Sum of grad[i] over all i
    /// equals 2 × cumulative_pairs[level] (each pair counted from both sides).
    ///
    /// This is the discrete (delta-function) derivative of the pair count at
    /// the given level with respect to point identity: removing point i would
    /// reduce cumulative_pairs[level] by exactly grad[i].
    pub fn pair_gradient_per_particle(&self, level: usize) -> Vec<u64> {
        let n = self.points.len();
        let n_words = self.n_words;
        let mut grad = vec![0u64; n];

        // Walk the cascade and at each cell, every point in the cell gets
        // grad[i] += (cell_count - 1).
        // We re-traverse using bit-vec all the way (no crossover): straightforward
        // but potentially slow for large N. Optimization is left for later;
        // correctness first.
        //
        // For now, we recompute by descent. At level=0, cell is the whole box
        // and all points are in it. At each descent we partition.
        let mut root_mem = vec![u64::MAX; n_words];
        if n % 64 != 0 {
            let last_word_bits = n % 64;
            root_mem[n_words - 1] = (1u64 << last_word_bits) - 1;
        }
        if n == 0 { return grad; }

        self.gradient_recurse(&root_mem, 0, level, &mut grad);
        grad
    }

    fn gradient_recurse(&self, mem: &[u64], cur_level: usize, target_level: usize, grad: &mut [u64]) {
        let count = popcount_vec(mem);
        if count == 0 {
            return;
        }
        if cur_level == target_level {
            // Add (count-1) to every point in this cell.
            let bonus = count.saturating_sub(1);
            if bonus == 0 {
                return;
            }
            for (w_idx, &word) in mem.iter().enumerate() {
                let mut w = word;
                let base = w_idx * 64;
                while w != 0 {
                    let i = base + w.trailing_zeros() as usize;
                    grad[i] += bonus;
                    w &= w - 1;
                }
            }
            return;
        }
        if cur_level >= self.l_max {
            // Already past target_level (target was beyond what cascade supports)
            return;
        }

        // Descend one level
        let n_words = self.n_words;
        let mut upper: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; n_words]);
        for d in 0..D {
            for w in 0..n_words {
                upper[d][w] = mem[w] & self.bit_planes[d][cur_level][w];
            }
        }
        let mut lower: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; n_words]);
        for d in 0..D {
            for w in 0..n_words {
                lower[d][w] = mem[w] & !self.bit_planes[d][cur_level][w];
            }
        }
        let mut child_mem = vec![0u64; n_words];
        for child in 0..(1u32 << D) {
            let b0 = (child & 1) != 0;
            child_mem.copy_from_slice(if b0 { &upper[0] } else { &lower[0] });
            for d in 1..D {
                let bd = (child >> d) & 1 != 0;
                let half = if bd { &upper[d] } else { &lower[d] };
                for w in 0..n_words {
                    child_mem[w] &= half[w];
                }
            }
            self.gradient_recurse(&child_mem, cur_level + 1, target_level, grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_2d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 2]> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| [next() & mask, next() & mask]).collect()
    }

    #[test]
    fn empty_cells_pruned() {
        // Place 4 points in widely separated quadrants of an 8-bit box.
        // At the root (level 0), 1 cell with all 4 points.
        // At level 1, 4 cells each with 1 point (D=2, 4 quadrants).
        // No empty cells should be visited.
        let pts: Vec<[u64; 2]> = vec![
            [0x10, 0x10],
            [0x10, 0xF0],
            [0xF0, 0x10],
            [0xF0, 0xF0],
        ];
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();

        // Level 0: 1 non-empty cell with 4 points
        assert_eq!(stats[0].n_nonempty_cells, 1);
        assert_eq!(stats[0].mean_nonempty, 4.0);

        // Level 1: 4 non-empty cells each with 1 point
        assert_eq!(stats[1].n_nonempty_cells, 4);
        assert_eq!(stats[1].mean_nonempty, 1.0);

        // Pair counts: at level 0 there are C(4,2) = 6 pairs (all in same root cell).
        // At level 1, no two points share a cell, so cumulative_pairs[1] = 0.
        assert_eq!(stats[0].cumulative_pairs, 6);
        assert_eq!(stats[1].cumulative_pairs, 0);
    }

    #[test]
    fn pair_count_total_is_n_choose_2() {
        // Pair counts at the root level should always equal N(N-1)/2.
        let pts = make_uniform_2d(1000, 8, 42);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        assert_eq!(stats[0].cumulative_pairs, 1000 * 999 / 2);
    }

    #[test]
    fn pair_counts_monotonically_decrease_with_depth() {
        let pts = make_uniform_2d(500, 10, 7);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        for l in 1..stats.len() {
            assert!(stats[l].cumulative_pairs <= stats[l-1].cumulative_pairs,
                "level {}: pairs increased from {} to {}",
                l, stats[l-1].cumulative_pairs, stats[l].cumulative_pairs);
        }
    }

    #[test]
    fn level_total_cells_correct() {
        let pts = make_uniform_2d(100, 8, 1);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        for l in 0..stats.len() {
            assert_eq!(stats[l].n_total_cells, 1u64 << (2 * l));
        }
    }

    #[test]
    fn pair_counts_match_brute_force_2d() {
        // For a small N, enumerate all pairs by brute force and check that
        // the cumulative_pairs[l] matches "pairs sharing a cell at level l".
        let n = 50;
        let bits = 8;
        let pts = make_uniform_2d(n, bits, 100);
        let trimmed = TrimmedPoints::from_points(pts.clone());
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();

        let eff_bits = casc.points.range.effective_bits;
        let l_max = casc.l_max;

        // For each level l, brute force count pairs sharing a cell.
        for l in 0..=l_max {
            let mut bf = 0u64;
            for i in 0..n {
                for j in (i+1)..n {
                    let mut same_cell = true;
                    for d in 0..2 {
                        let eff = eff_bits[d] as usize;
                        // Cell index along axis d at level l = top l bits of trimmed coord.
                        // If l >= eff, the cell index uses all eff bits (coords themselves).
                        // Two points share a cell iff their cell indices match.
                        let l_use = l.min(eff);
                        let shift = if l_use >= eff { 0 } else { eff - l_use };
                        let cell_i = casc.points.points[i][d] >> shift;
                        let cell_j = casc.points.points[j][d] >> shift;
                        if cell_i != cell_j {
                            same_cell = false;
                            break;
                        }
                    }
                    if same_cell { bf += 1; }
                }
            }
            assert_eq!(stats[l].cumulative_pairs, bf,
                "level {} cumulative pairs: cascade {} vs brute force {}",
                l, stats[l].cumulative_pairs, bf);
        }
    }

    #[test]
    fn cell_count_sum_equals_n() {
        // At every level, summing n_c across non-empty cells equals N.
        let pts = make_uniform_2d(200, 9, 13);
        let n = pts.len();
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        for st in &stats {
            let sum_from_hist: u64 = st.histogram.iter().enumerate()
                .map(|(k, &h)| (k as u64) * h)
                .sum();
            assert_eq!(sum_from_hist, n as u64,
                "level {}: histogram sums to {}, expected {}",
                st.level, sum_from_hist, n);
        }
    }

    #[test]
    fn lmax_auto_set_from_data() {
        // Coordinates use only 6 bits even though stored as u64
        let pts: Vec<[u64; 2]> = (0..50)
            .map(|i| [i as u64 % 64, (i * 13) as u64 % 64])
            .collect();
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        // Should be 6 levels (0..=6 actually, since l_max is the deepest reached)
        assert_eq!(casc.l_max, 6);
    }

    #[test]
    fn lmax_capped_by_user() {
        let pts: Vec<[u64; 2]> = (0..100)
            .map(|i| [(i * 7) as u64, (i * 11) as u64])
            .collect();
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, Some(4));
        assert!(casc.l_max <= 4);
    }

    #[test]
    fn crossover_matches_pure_bitvec_2d() {
        // For the same dataset, the cascade with crossover threshold > N (i.e.
        // pure bit-vec, never crosses over) and with threshold = 0 (immediately
        // crosses over at root) must produce bitwise-identical statistics.
        let pts = make_uniform_2d(500, 12, 7);
        let n = pts.len();

        let trimmed_a = TrimmedPoints::from_points(pts.clone());
        let casc_a = BitVecCascade::<2>::build_with_threshold(trimmed_a, None, n + 1);
        let stats_a = casc_a.analyze();

        let trimmed_b = TrimmedPoints::from_points(pts);
        let casc_b = BitVecCascade::<2>::build_with_threshold(trimmed_b, None, 0);
        let stats_b = casc_b.analyze();

        assert_eq!(stats_a.len(), stats_b.len());
        for l in 0..stats_a.len() {
            let a = &stats_a[l];
            let b = &stats_b[l];
            assert_eq!(a.n_nonempty_cells, b.n_nonempty_cells, "level {} n_nonempty", l);
            assert_eq!(a.cumulative_pairs, b.cumulative_pairs, "level {} cum_pairs", l);
            assert!((a.mean_nonempty - b.mean_nonempty).abs() < 1e-12,
                "level {} mean_nonempty: {} vs {}", l, a.mean_nonempty, b.mean_nonempty);
            assert!((a.var_nonempty - b.var_nonempty).abs() < 1e-9,
                "level {} var_nonempty: {} vs {}", l, a.var_nonempty, b.var_nonempty);
            // Histograms must be identical
            let len = a.histogram.len().max(b.histogram.len());
            for k in 0..len {
                let ha = a.histogram.get(k).copied().unwrap_or(0);
                let hb = b.histogram.get(k).copied().unwrap_or(0);
                assert_eq!(ha, hb, "level {} histogram[{}]: {} vs {}", l, k, ha, hb);
            }
        }
    }

    #[test]
    fn crossover_matches_pure_bitvec_3d() {
        let pts = (0..300u64).map(|i| {
            // pseudo-random 3D points in u10
            let x = (i.wrapping_mul(2654435761)) & 0x3FF;
            let y = (i.wrapping_mul(40503)) & 0x3FF;
            let z = (i.wrapping_mul(2246822519)) & 0x3FF;
            [x, y, z]
        }).collect::<Vec<[u64; 3]>>();

        let trimmed_a = TrimmedPoints::from_points(pts.clone());
        let casc_a = BitVecCascade::<3>::build_with_threshold(trimmed_a, None, 10000);
        let stats_a = casc_a.analyze();

        for &thresh in &[0usize, 1, 2, 4, 8, 16, 64, 256] {
            let trimmed = TrimmedPoints::from_points(pts.clone());
            let casc = BitVecCascade::<3>::build_with_threshold(trimmed, None, thresh);
            let stats = casc.analyze();
            for l in 0..stats_a.len() {
                let a = &stats_a[l];
                let b = &stats[l];
                assert_eq!(a.n_nonempty_cells, b.n_nonempty_cells,
                    "thresh={} level {} n_nonempty differs", thresh, l);
                assert_eq!(a.cumulative_pairs, b.cumulative_pairs,
                    "thresh={} level {} cum_pairs differs", thresh, l);
            }
        }
    }

    // ---- Phase 2 tests: pair shells and per-particle gradients ----

    #[test]
    fn pair_shells_sum_to_total() {
        // Sum of n_pairs across all shells = N(N-1)/2.
        let n = 200;
        let pts = make_uniform_2d(n, 12, 99);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        let shells = casc.pair_counts_per_shell(&stats);
        let total: u64 = shells.iter().map(|s| s.n_pairs).sum();
        let expected = (n as u64) * (n as u64 - 1) / 2;
        assert_eq!(total, expected,
            "sum of shell pair counts = {}, expected {}", total, expected);
    }

    #[test]
    fn pair_shell_sides_decrease_with_level() {
        let n = 100;
        let pts = make_uniform_2d(n, 10, 11);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();
        let shells = casc.pair_counts_per_shell(&stats);
        for w in shells.windows(2) {
            assert!(w[0].cell_side_trimmed > w[1].cell_side_trimmed
                || (w[0].cell_side_trimmed == 1.0 && w[1].cell_side_trimmed == 1.0),
                "side at level {} ({}) <= side at level {} ({})",
                w[0].level, w[0].cell_side_trimmed,
                w[1].level, w[1].cell_side_trimmed);
        }
    }

    #[test]
    fn pair_gradient_sums_to_2x_pairs() {
        // For any level, sum of grad[i] over all i = 2 * cumulative_pairs[level].
        let n = 150;
        let pts = make_uniform_2d(n, 10, 13);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let stats = casc.analyze();

        for l in 0..stats.len() {
            let grad = casc.pair_gradient_per_particle(l);
            assert_eq!(grad.len(), n);
            let sum: u64 = grad.iter().sum();
            assert_eq!(sum, 2 * stats[l].cumulative_pairs,
                "level {}: sum(grad) = {}, expected 2*cumulative = {}",
                l, sum, 2 * stats[l].cumulative_pairs);
        }
    }

    #[test]
    fn pair_gradient_at_root_is_n_minus_1() {
        // At level 0, every point is in the same cell with all others.
        let n = 50;
        let pts = make_uniform_2d(n, 8, 17);
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let grad = casc.pair_gradient_per_particle(0);
        for (i, &g) in grad.iter().enumerate() {
            assert_eq!(g, (n - 1) as u64,
                "point {} grad at root = {}, expected {}", i, g, n - 1);
        }
    }

    #[test]
    fn pair_gradient_3d_brute_force() {
        // Direct: pick a moderate N, compute brute-force pair gradient at
        // multiple levels, compare to cascade.
        let n = 40;
        let bits = 8;
        let pts: Vec<[u64; 3]> = (0..n as u64).map(|i| {
            let x = (i.wrapping_mul(2654435761)) % (1 << bits);
            let y = (i.wrapping_mul(40503)) % (1 << bits);
            let z = (i.wrapping_mul(2246822519)) % (1 << bits);
            [x, y, z]
        }).collect();
        let trimmed = TrimmedPoints::from_points(pts.clone());
        let casc = BitVecCascade::<3>::build(trimmed, None);
        let stats = casc.analyze();
        let eff = casc.points.range.effective_bits;

        for l in 1..stats.len().min(4) {
            let grad = casc.pair_gradient_per_particle(l);
            // Brute force: for each point i, count points j != i sharing cell at level l
            let mut bf_grad = vec![0u64; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j { continue; }
                    let mut same = true;
                    for d in 0..3 {
                        let e = eff[d] as usize;
                        let l_use = l.min(e);
                        let shift = if l_use >= e { 0 } else { e - l_use };
                        if (casc.points.points[i][d] >> shift) != (casc.points.points[j][d] >> shift) {
                            same = false;
                            break;
                        }
                    }
                    if same { bf_grad[i] += 1; }
                }
            }
            for i in 0..n {
                assert_eq!(grad[i], bf_grad[i],
                    "level {} point {}: cascade grad = {}, brute force = {}",
                    l, i, grad[i], bf_grad[i]);
            }
        }
    }

    // ---- Default crossover threshold ----

    #[test]
    fn default_crossover_threshold_floors_at_64() {
        // Small N: floor of 64 prevents the crossover from triggering on
        // every cell visit (which would slow down small problems).
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(0), 64);
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(1), 64);
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(100), 64);
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(4096), 64);
        // Above 64 * 64 = 4096, the N/64 term takes over.
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(8192), 128);
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(64000), 1000);
        assert_eq!(BitVecCascade::<2>::default_crossover_threshold(1_000_000), 15625);
    }

    #[test]
    fn build_uses_adaptive_default() {
        // Ensure build() picks up the adaptive value, not a hard-coded constant.
        let pts: Vec<[u64; 2]> = (0..200_000u64).map(|i| {
            // splitmix64 mixing — full-range coords
            let mut s = i.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); s = z ^ (z >> 31);
            let x = s & 0xFFFF;
            let mut s2 = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s2; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); s2 = z ^ (z >> 31);
            let y = s2 & 0xFFFF;
            [x, y]
        }).collect();
        let n = pts.len();
        let trimmed = TrimmedPoints::from_points(pts);
        let casc = BitVecCascade::<2>::build(trimmed, None);
        let expected = BitVecCascade::<2>::default_crossover_threshold(n);
        assert_eq!(casc.crossover_threshold, expected,
            "build() should use the adaptive default for N={}", n);
        assert!(expected > 64,
            "for N=200k the adaptive default should be > 64, got {}", expected);
    }
}
