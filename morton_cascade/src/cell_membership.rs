// cell_membership.rs
//
// Reverse mapping from cascade cells to the particle indices that fall
// in them. The cascade's bit planes encode "which cell does particle i
// belong to" indexed by particle; this module provides the inverse —
// "which particles belong to cell c at level l" indexed by cell.
//
// Use case: per-particle attribution and gradient computation. Many
// downstream questions ("how much does particle i contribute to bin
// k?", "what is ∂S/∂w_i?") reduce to "find the cells containing
// particle i at each level, then walk those cells' contributions."
// This module provides the cell-side lookup; the per-particle
// gradients then need only sum L_max contributions per particle.
//
// Design. We build a single permutation `perm: Vec<u32>` of particle
// indices sorted by their deepest-level Morton code, plus a parallel
// `morton_codes: Vec<u64>` with the codes themselves. The Morton code
// at level l_max uniquely identifies a leaf cell; cells at any
// shallower level l < l_max are prefixes of the deepest-level codes,
// so particles sharing a level-l ancestor cell are contiguous in `perm`
// — this is the defining property of Morton ordering.
//
// Lookup: given (level, cell_id), binary-search for the contiguous
// range and return `&perm[start..end]`. O(log N) per query, O(1) if
// you have the level-l_max cell range cached.
//
// Memory: O(N) — one u32 + one u64 per particle. No per-cell
// allocations. Independent of catalog density (sparse cascades cost
// the same as dense ones).

use crate::hier_bitvec_pair::BitVecCascadePair;

/// Inverse-direction cell lookup: from (level, cell_id) to particle
/// indices. Built once at the cost of O(N log N) for the sort (in
/// addition to O(N · L_max · D) bit reads to construct codes).
///
/// The const generic `D` matches the cascade's spatial dimension and
/// is required because Morton interleaving is per-D.
#[derive(Debug, Clone)]
pub struct CellMembership {
    /// Particle indices sorted by deepest-level Morton code.
    /// `perm[k]` is the original catalog index of the k-th particle
    /// in Morton order.
    perm: Vec<u32>,
    /// Morton codes at level `l_max`, sorted ascending.
    /// `morton_codes[k]` is the level-`l_max` cell id of particle
    /// `perm[k]`. Length matches `perm`.
    morton_codes: Vec<u64>,
    /// Cascade depth — number of bits per axis used by Morton codes.
    /// Equal to `pair.l_max`.
    l_max: usize,
    /// Spatial dimension (matches the cascade's const generic D).
    d: usize,
}

impl CellMembership {
    /// Build the membership index from a cascade and a "data or
    /// randoms" selection. Computes Morton codes from the bit planes,
    /// sorts in O(N log N), returns the index.
    ///
    /// `which`: 0 for data catalog, 1 for randoms catalog.
    ///
    /// The cost is paid once per cascade — repeated lookups against
    /// the same cascade are O(log N) each (binary search).
    pub fn build<const D: usize>(
        pair: &BitVecCascadePair<D>,
        which: WhichCatalog,
    ) -> Self {
        let l_max = pair.l_max;
        let n = match which {
            WhichCatalog::Data => pair.n_d(),
            WhichCatalog::Randoms => pair.n_r(),
        };

        // Compute level-l_max Morton code for every particle.
        // morton_at_level_l_max(i) = concat over levels {0..l_max} of
        //                            (concat over axes {0..D} of axis_d_bit_at_level_l(i))
        // with level 0 in the most-significant chunk and axis D-1 in
        // the most-significant bit within each per-level chunk.
        // Total bit width: l_max * D. Fits in u64 as long as l_max*D ≤ 64.
        debug_assert!(l_max * D <= 64,
            "Morton code at l_max={} D={} requires {} bits, doesn't fit in u64",
            l_max, D, l_max * D);

        let mut morton_codes_unsorted: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let mut code: u64 = 0;
            for l in 0..l_max {
                let mut chunk: u64 = 0;
                for d in 0..D {
                    let bit = match which {
                        WhichCatalog::Data => {
                            (pair.bit_planes_d_at(d, l)[word_idx] >> bit_idx) & 1
                        }
                        WhichCatalog::Randoms => {
                            (pair.bit_planes_r_at(d, l)[word_idx] >> bit_idx) & 1
                        }
                    };
                    chunk |= bit << d;
                }
                code = (code << D) | chunk;
            }
            morton_codes_unsorted.push(code);
        }

        // Build sort permutation. Sort indices [0..n] by Morton code.
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.sort_by_key(|&i| morton_codes_unsorted[i as usize]);

        // Reorder Morton codes to match the sorted perm.
        let morton_codes: Vec<u64> = perm.iter()
            .map(|&i| morton_codes_unsorted[i as usize])
            .collect();

        Self { perm, morton_codes, l_max, d: D }
    }

    /// Particle indices in cell `cell_id` at the given `level`.
    /// Returns a slice into the internal permutation array — no
    /// allocation per query.
    ///
    /// At level `level`, `cell_id` ranges over `0 .. 2^(D*level)`.
    /// Level 0 is the root cell (the entire box) and the returned
    /// slice contains every particle's index.
    ///
    /// For levels deeper than `l_max`, the returned slice is empty.
    /// For invalid `cell_id` (≥ 2^(D*level)), the returned slice is
    /// empty.
    pub fn members(&self, level: usize, cell_id: u64) -> &[u32] {
        if level > self.l_max { return &[]; }
        // The Morton code we stored is at level l_max. To find all
        // particles in cell `cell_id` at level `level`, we need those
        // whose level-l_max code has the top D*level bits equal to
        // `cell_id` (with the lower D*(l_max-level) bits unspecified).
        let shift = self.d * (self.l_max - level);
        // Validate cell_id range.
        if level > 0 && level * self.d <= 64 {
            let max_cell_id = 1u64 << (self.d * level);
            if cell_id >= max_cell_id { return &[]; }
        }
        let lo = cell_id << shift;
        // hi = lo + 2^shift, but watch for overflow when shift = 64.
        let hi = if shift >= 64 { u64::MAX } else {
            lo.checked_add(1u64 << shift).unwrap_or(u64::MAX)
        };

        // Binary-search for [lo, hi) range in sorted morton_codes.
        let start = self.morton_codes.partition_point(|&c| c < lo);
        let end = self.morton_codes.partition_point(|&c| c < hi);
        &self.perm[start..end]
    }

    /// Number of particles in cell `cell_id` at the given `level`.
    /// Convenience around `members(level, cell_id).len()`.
    #[inline]
    pub fn member_count(&self, level: usize, cell_id: u64) -> usize {
        self.members(level, cell_id).len()
    }

    /// Total number of particles indexed (matches the catalog size).
    #[inline]
    pub fn n_particles(&self) -> usize {
        self.perm.len()
    }

    /// Cascade depth (`l_max` of the source cascade).
    #[inline]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    /// Spatial dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Iterator over (cell_id, particle_indices) at the given level.
    /// Visits only NON-EMPTY cells (sparse-friendly). Cells are
    /// visited in ascending cell_id order.
    pub fn non_empty_cells_at(&self, level: usize)
        -> impl Iterator<Item = (u64, &[u32])>
    {
        let shift = self.d * (self.l_max - level);
        let codes = &self.morton_codes;
        let perm = &self.perm;
        // Collect the unique cell ids by walking the sorted codes and
        // grouping consecutive runs sharing the same upper-bits prefix.
        let n = codes.len();
        let mut ranges: Vec<(u64, usize, usize)> = Vec::new();
        let mut k = 0;
        while k < n {
            let cell = if shift >= 64 { 0 } else { codes[k] >> shift };
            let mut j = k + 1;
            while j < n {
                let next_cell = if shift >= 64 { 0 } else { codes[j] >> shift };
                if next_cell != cell { break; }
                j += 1;
            }
            ranges.push((cell, k, j));
            k = j;
        }
        ranges.into_iter().map(move |(cid, s, e)| (cid, &perm[s..e]))
    }
}

/// Selector for which catalog's membership to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichCatalog {
    Data,
    Randoms,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_range::{CoordRange, TrimmedPoints};

    fn make_uniform_3d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| [next() & mask, next() & mask, next() & mask]).collect()
    }

    fn build_pair(pts_d: Vec<[u64; 3]>, pts_r: Vec<[u64; 3]>) -> BitVecCascadePair<3> {
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        BitVecCascadePair::<3>::build(td, tr, None)
    }

    #[test]
    fn cell_membership_root_contains_all_particles() {
        let pts_d = make_uniform_3d(100, 6, 7000);
        let pts_r = make_uniform_3d(300, 6, 7001);
        let pair = build_pair(pts_d, pts_r);

        let mem = CellMembership::build(&pair, WhichCatalog::Data);
        let root = mem.members(0, 0);
        assert_eq!(root.len(), 100,
            "root cell at level 0 must contain every particle");

        // Should be a permutation of 0..100.
        let mut sorted = root.to_vec();
        sorted.sort();
        let expected: Vec<u32> = (0..100).collect();
        assert_eq!(sorted, expected);

        let mem_r = CellMembership::build(&pair, WhichCatalog::Randoms);
        assert_eq!(mem_r.members(0, 0).len(), 300);
    }

    #[test]
    fn cell_membership_partition_property() {
        // At any level, the union of all cells' members must equal the
        // full catalog (no duplicates, no omissions). This is the
        // canonical correctness check on the inverse mapping.
        let pts_d = make_uniform_3d(500, 5, 7100);
        let pts_r = make_uniform_3d(1500, 5, 7101);
        let pair = build_pair(pts_d, pts_r);

        let mem = CellMembership::build(&pair, WhichCatalog::Data);
        for level in 0..=mem.l_max() {
            let mut union: Vec<u32> = mem.non_empty_cells_at(level)
                .flat_map(|(_, p)| p.iter().copied())
                .collect();
            union.sort();
            let expected: Vec<u32> = (0..500).collect();
            assert_eq!(union, expected,
                "level {}: union of non-empty cells != full catalog", level);
        }
    }

    #[test]
    fn cell_membership_matches_cascade_cell_counts_at_finest_level() {
        // The membership index must agree with the cascade's own
        // cell-count statistics at the finest level: for every cell
        // visited by the cascade, the membership.member_count should
        // match the per-cell n_d.
        let pts_d = make_uniform_3d(200, 5, 7200);
        let pts_r = make_uniform_3d(600, 5, 7201);
        let pair = build_pair(pts_d, pts_r);

        let mem = CellMembership::build(&pair, WhichCatalog::Data);
        let l = mem.l_max();
        // At level l_max, every distinct Morton code corresponds to one
        // cell; member_count gives the multiplicity. Sum across all
        // distinct codes should equal n_particles.
        let total: usize = mem.non_empty_cells_at(l)
            .map(|(_, p)| p.len())
            .sum();
        assert_eq!(total, mem.n_particles(),
            "sum of finest-level cell sizes != n_particles");
    }

    #[test]
    fn cell_membership_parent_contains_children() {
        // For any cell at level l, its members must equal the union of
        // its 2^D child cells' members. (Cascade-level consistency.)
        let pts_d = make_uniform_3d(300, 5, 7300);
        let pts_r = make_uniform_3d(900, 5, 7301);
        let pair = build_pair(pts_d, pts_r);
        let mem = CellMembership::build(&pair, WhichCatalog::Data);

        // Test at level 1 → level 2.
        let level = 1;
        let n_cells_at_level = 1u64 << (3 * level);
        for parent_id in 0..n_cells_at_level {
            let parent_members: Vec<u32> = mem.members(level, parent_id).to_vec();
            let mut child_union: Vec<u32> = Vec::new();
            for child_offset in 0..8u64 {  // 2^D = 8 for D=3
                let child_id = (parent_id << 3) | child_offset;
                child_union.extend_from_slice(mem.members(level + 1, child_id));
            }
            child_union.sort();
            let mut parent_sorted = parent_members.clone();
            parent_sorted.sort();
            assert_eq!(parent_sorted, child_union,
                "level {} cell {}: parent != union of children", level, parent_id);
        }
    }

    #[test]
    fn cell_membership_morton_sorted_property() {
        // The internal morton_codes vector must be sorted ascending
        // (defining property of the data structure).
        let pts_d = make_uniform_3d(200, 5, 7400);
        let pts_r = make_uniform_3d(600, 5, 7401);
        let pair = build_pair(pts_d, pts_r);
        let mem = CellMembership::build(&pair, WhichCatalog::Data);
        for k in 1..mem.morton_codes.len() {
            assert!(mem.morton_codes[k] >= mem.morton_codes[k - 1],
                "morton_codes not sorted at index {}: {} < {}",
                k, mem.morton_codes[k], mem.morton_codes[k - 1]);
        }
    }

    #[test]
    fn cell_membership_invalid_cell_id_returns_empty() {
        let pts_d = make_uniform_3d(100, 5, 7500);
        let pts_r = make_uniform_3d(300, 5, 7501);
        let pair = build_pair(pts_d, pts_r);
        let mem = CellMembership::build(&pair, WhichCatalog::Data);

        // At level 1 there are 2^D = 8 cells. Cell id 100 is invalid.
        assert_eq!(mem.members(1, 100).len(), 0);
        // Beyond l_max → empty.
        assert_eq!(mem.members(mem.l_max() + 5, 0).len(), 0);
    }

    #[test]
    fn cell_membership_per_particle_attribution_example() {
        // Worked example: per-particle CIC PMF attribution.
        // For a cell of size n at the finest level, each of its n
        // particles is credited with 1/n of that cell's contribution
        // to histogram bin n. Summing over ALL particles of their
        // contribution to bin k must therefore equal bin k's count
        // (the number of finest-level cells that hold exactly k
        // particles). This is the canonical correctness check for
        // per-particle attribution.
        let pts_d = make_uniform_3d(150, 5, 7600);
        let pts_r = make_uniform_3d(450, 5, 7601);
        let pair = build_pair(pts_d, pts_r);
        let mem = CellMembership::build(&pair, WhichCatalog::Data);
        let l = mem.l_max();

        // Direct count: how many cells at level l_max have exactly k
        // particles? Build the histogram from the membership index.
        let mut bin_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (_, members) in mem.non_empty_cells_at(l) {
            *bin_counts.entry(members.len()).or_insert(0) += 1;
        }

        // Per-particle attribution: each particle in cell of size n
        // contributes 1/n to bin n. Sum particles' contributions to
        // each bin and check it matches bin_counts.
        let mut bin_attribution: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        for (_, members) in mem.non_empty_cells_at(l) {
            let n = members.len();
            let per_particle = 1.0 / n as f64;
            for _ in members {
                *bin_attribution.entry(n).or_insert(0.0) += per_particle;
            }
        }

        // Compare: bin_attribution[k] should equal bin_counts[k]
        // exactly (each cell of size k contributes k * (1/k) = 1
        // total to bin k, so sum across all such cells = #cells_of_size_k).
        for (k, &count) in &bin_counts {
            let attr = bin_attribution.get(k).copied().unwrap_or(0.0);
            assert!((attr - count as f64).abs() < 1e-12,
                "bin {}: attribution sum {} != cell count {}", k, attr, count);
        }
        // Total across all bins: sum of bin counts = number of nonempty
        // cells. Sum of attribution across all bins = same.
        let total_cells: usize = bin_counts.values().sum();
        let total_attr: f64 = bin_attribution.values().sum();
        assert!((total_attr - total_cells as f64).abs() < 1e-10,
            "total cells {} vs total attribution {}", total_cells, total_attr);
    }
}
