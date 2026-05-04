//! # Cascade visitor types
//!
//! Foundation types for traversing the cascade. The library traverses
//! a Morton-ordered dyadic cell hierarchy once; many statistics can
//! be computed during that traversal. This module provides the types
//! that make a *single* policy for "how do we use the random catalog"
//! visible to every statistic, instead of having each statistic
//! reinvent the policy locally.
//!
//! ## Two-catalog cell summary
//!
//! For each cell visited during a cascade walk, the walker computes
//! a [`CellVisit`]. This captures:
//!
//! - which level we're at, and the cell's spatial extent
//! - the data-catalog cell summary ([`CatalogCell`]: count and weight)
//! - the random-catalog cell summary
//! - the global mean-density ratio `alpha = sum_w_d / sum_w_r`
//!
//! From these, every statistic that needs a footprint-aware density
//! contrast can use the canonical formula via [`CellVisit::delta`]:
//!
//! ```text
//!     delta = sw_d / (alpha * sw_r) - 1     if sw_r > w_r_min
//!     None  (i.e. cell is outside footprint) otherwise
//! ```
//!
//! ## Footprint policy
//!
//! [`FootprintCutoff`] holds the parameters that decide whether a cell
//! is "inside the survey" — currently just `w_r_min` (the minimum
//! random-catalog weight required to count a cell as in-footprint).
//! Future extensions (per-cell footprint masks, position-dependent
//! cutoffs) can extend this struct without changing the per-statistic
//! visitor code.

/// Catalog summary at a single cell. Captured for both the data catalog
/// and the random catalog at every cell the cascade visits.
///
/// `count` is the unweighted number of points in the cell. `sum_w` is
/// the total weight (= `count` if no per-point weights). `sum_w_sq` is
/// the sum of squared weights, used by some statistics for shot-noise
/// estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatalogCell {
    pub count: u64,
    pub sum_w: f64,
    pub sum_w_sq: f64,
}

impl CatalogCell {
    pub const EMPTY: Self = CatalogCell { count: 0, sum_w: 0.0, sum_w_sq: 0.0 };

    /// True if this catalog has zero points in the cell.
    #[inline]
    pub fn is_empty(&self) -> bool { self.count == 0 }
}

/// Footprint-cutoff policy: how do we decide whether a cell is in
/// the survey?
///
/// Currently a single threshold on the random-catalog weight per cell.
/// A cell with `sum_w_r > w_r_min` is considered in-footprint.
#[derive(Clone, Copy, Debug)]
pub struct FootprintCutoff {
    /// Minimum random-catalog weight to count a cell as in-footprint.
    /// Cells below this threshold are excluded from in-footprint
    /// estimators and may be flagged by outside-footprint diagnostics
    /// if they contain data points.
    pub w_r_min: f64,
}

impl FootprintCutoff {
    /// Default policy: any cell with at least one random point is in
    /// the footprint. Use [`FootprintCutoff::strict`] for stricter
    /// thresholds.
    pub const ANY_RANDOM: Self = FootprintCutoff { w_r_min: 0.0 };

    /// Stricter policy: require `w_r_min` random-catalog weight to
    /// count a cell as in-footprint. Useful for noisy randoms.
    pub fn strict(w_r_min: f64) -> Self { FootprintCutoff { w_r_min } }
}

impl Default for FootprintCutoff {
    fn default() -> Self { Self::ANY_RANDOM }
}

/// Per-cell snapshot passed to every statistic visitor.
///
/// `D` is the spatial dimension (2, 3, ...).
#[derive(Clone, Copy, Debug)]
pub struct CellVisit<const D: usize> {
    /// Level in the cascade. 0 is the root (whole box); larger is finer.
    pub level: usize,
    /// Cell side in trimmed (tree) coordinates: `2^(L_max - level)`.
    /// Multiply by `box_size / 2^L_max` to get physical units.
    pub cell_side_trimmed: u64,
    /// Linear index of this cell within its level (0 .. 2^(D*level)).
    /// Useful for statistics that need a per-cell map.
    pub cell_id: u64,
    /// Data-catalog cell summary.
    pub data: CatalogCell,
    /// Random-catalog cell summary.
    pub randoms: CatalogCell,
    /// Global mean-density ratio: `total_w_d / total_w_r`. Constant for
    /// the duration of a single cascade walk.
    pub alpha: f64,
}

impl<const D: usize> CellVisit<D> {
    /// Footprint-aware density contrast: `δ = W_d / (α · W_r) - 1`,
    /// or `None` if the cell is outside the survey footprint per the
    /// given cutoff.
    ///
    /// This is the canonical estimator for the density contrast in a
    /// cell of a survey with non-trivial geometry. It is correctly
    /// normalized: `delta = 0` when the cell has the mean weighted
    /// number density of the survey.
    #[inline]
    pub fn delta(&self, footprint: &FootprintCutoff) -> Option<f64> {
        if self.randoms.sum_w > footprint.w_r_min && self.alpha > 0.0 {
            Some(self.data.sum_w / (self.alpha * self.randoms.sum_w) - 1.0)
        } else {
            None
        }
    }

    /// Whether this cell is in-footprint per the given cutoff.
    #[inline]
    pub fn in_footprint(&self, footprint: &FootprintCutoff) -> bool {
        self.randoms.sum_w > footprint.w_r_min
    }

    /// Whether data is present at a cell that is outside the footprint.
    /// This is the canonical "catalog-quality" diagnostic: data exists
    /// where the random catalog says no survey volume exists.
    #[inline]
    pub fn data_outside_footprint(&self, footprint: &FootprintCutoff) -> bool {
        !self.in_footprint(footprint) && !self.data.is_empty()
    }

    /// True if the cell has neither data nor randoms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.randoms.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(sw_d: f64, sw_r: f64, alpha: f64) -> CellVisit<3> {
        CellVisit::<3> {
            level: 0,
            cell_side_trimmed: 1024,
            cell_id: 0,
            data: CatalogCell { count: sw_d as u64, sum_w: sw_d, sum_w_sq: sw_d },
            randoms: CatalogCell { count: sw_r as u64, sum_w: sw_r, sum_w_sq: sw_r },
            alpha,
        }
    }

    #[test]
    fn delta_basic() {
        // alpha = 1, sw_d = sw_r → delta = 0
        let c = mk(10.0, 10.0, 1.0);
        assert_eq!(c.delta(&FootprintCutoff::ANY_RANDOM), Some(0.0));
    }

    #[test]
    fn delta_overdensity() {
        // alpha = 1, sw_d = 2 * sw_r → delta = 1
        let c = mk(20.0, 10.0, 1.0);
        assert_eq!(c.delta(&FootprintCutoff::ANY_RANDOM), Some(1.0));
    }

    #[test]
    fn delta_underdensity() {
        // alpha = 1, sw_d = 0 → delta = -1
        let c = mk(0.0, 10.0, 1.0);
        assert_eq!(c.delta(&FootprintCutoff::ANY_RANDOM), Some(-1.0));
    }

    #[test]
    fn delta_alpha_normalization() {
        // alpha = 0.5 (50x more randoms), matched mean density:
        // expect delta = sw_d / (alpha * sw_r) - 1 = 100 / (0.5 * 200) - 1 = 0
        let c = mk(100.0, 200.0, 0.5);
        assert!((c.delta(&FootprintCutoff::ANY_RANDOM).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn delta_outside_footprint() {
        // No randoms in cell → outside footprint → None
        let c = mk(5.0, 0.0, 1.0);
        assert_eq!(c.delta(&FootprintCutoff::ANY_RANDOM), None);
    }

    #[test]
    fn footprint_strict_excludes_low_random_cells() {
        // sw_r = 1.0 doesn't pass strict cutoff at 5.0
        let c = mk(10.0, 1.0, 1.0);
        let strict = FootprintCutoff::strict(5.0);
        assert!(!c.in_footprint(&strict));
        assert_eq!(c.delta(&strict), None);
        // But it does pass the lenient cutoff
        assert!(c.in_footprint(&FootprintCutoff::ANY_RANDOM));
    }

    #[test]
    fn data_outside_footprint_diagnostic() {
        // Data present, no randoms → outside footprint
        let c = mk(5.0, 0.0, 1.0);
        assert!(c.data_outside_footprint(&FootprintCutoff::ANY_RANDOM));
        // No data, no randoms → empty cell, not "outside"
        let c = mk(0.0, 0.0, 1.0);
        assert!(!c.data_outside_footprint(&FootprintCutoff::ANY_RANDOM));
        // Data and randoms → in footprint, not "outside"
        let c = mk(5.0, 10.0, 1.0);
        assert!(!c.data_outside_footprint(&FootprintCutoff::ANY_RANDOM));
    }

    #[test]
    fn is_empty_test() {
        assert!(mk(0.0, 0.0, 1.0).is_empty());
        assert!(!mk(0.0, 1.0, 1.0).is_empty());
        assert!(!mk(1.0, 0.0, 1.0).is_empty());
        assert!(!mk(1.0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn alpha_zero_returns_none() {
        // Pathological case: alpha = 0 (no data, but somehow we got
        // here). Should be safely handled.
        let c = mk(10.0, 10.0, 0.0);
        assert_eq!(c.delta(&FootprintCutoff::ANY_RANDOM), None);
    }
}

// ============================================================================
// CascadeVisitor trait
// ============================================================================

/// Trait implemented by statistics that consume cascade walks.
///
/// A single cascade walk visits every non-empty cell at every level
/// (modulo the crossover-to-pointlist optimization, which is invisible
/// to visitors). The walker calls `enter_cell` on each cell, descends
/// into children, then calls `after_children` on the parent so
/// statistics that need to combine information from a parent's
/// children (Haar wavelets, scattering coefficients) can do so.
///
/// All methods have default no-op implementations, so a visitor
/// implements only what it needs.
///
/// ## Composition
///
/// Multiple visitors can be combined to compute several statistics
/// in a single walk via tuple impls (see [`CompositeVisitor`]).
///
/// ## Example
///
/// ```ignore
/// struct CountActiveCells(Vec<u64>);
///
/// impl CascadeVisitor<3> for CountActiveCells {
///     fn enter_cell(&mut self, cell: &CellVisit<3>) {
///         if cell.in_footprint(&FootprintCutoff::ANY_RANDOM) {
///             self.0[cell.level] += 1;
///         }
///     }
/// }
/// ```
pub trait CascadeVisitor<const D: usize> {
    /// Called when the walker enters each cell, before descending
    /// into its children. Most one-point statistics (variance,
    /// moments, CIC histograms) only need this method.
    #[allow(unused_variables)]
    fn enter_cell(&mut self, cell: &CellVisit<D>) {}

    /// Called for cells at the finest level (`level == l_max`).
    /// Defaults to delegating to `enter_cell`. Override only if
    /// finest-level cells need special treatment (e.g. they should
    /// not contribute to a wavelet decomposition because there are
    /// no further children).
    #[allow(unused_variables)]
    fn leaf_cell(&mut self, cell: &CellVisit<D>) {}

    /// Called on a parent cell *after* the walker has descended into
    /// all 2^D children. The `children` slice is in canonical order:
    /// `children[c]` is the cell at axis-pattern `c` ∈ {0, ..., 2^D-1}
    /// where bit `d` of `c` selects upper-half (1) or lower-half (0)
    /// along axis `d`.
    ///
    /// This is the entry point for parent-from-children statistics
    /// like Haar wavelet coefficients, scattering coefficients,
    /// gradient estimators, and any kernel that needs a 2^D stencil.
    ///
    /// `children` is passed as a slice of length `1 << D`. Empty
    /// children (no data and no randoms) still appear in the slice
    /// with `data` and `randoms` zero so that the indexing is
    /// regular.
    #[allow(unused_variables)]
    fn after_children(&mut self, parent: &CellVisit<D>, children: &[CellVisit<D>]) {}
}

/// Tuple visitor that runs two visitors in parallel during one walk.
/// Composition is associative: `((A, B), C)` and `(A, (B, C))` give
/// the same dispatching behavior, modulo invocation order.
///
/// For more than two visitors, nest tuples or use [`CompositeVisitor`].
impl<const D: usize, A: CascadeVisitor<D>, B: CascadeVisitor<D>>
    CascadeVisitor<D> for (&mut A, &mut B)
{
    fn enter_cell(&mut self, cell: &CellVisit<D>) {
        self.0.enter_cell(cell);
        self.1.enter_cell(cell);
    }
    fn leaf_cell(&mut self, cell: &CellVisit<D>) {
        self.0.leaf_cell(cell);
        self.1.leaf_cell(cell);
    }
    fn after_children(&mut self, parent: &CellVisit<D>, children: &[CellVisit<D>]) {
        self.0.after_children(parent, children);
        self.1.after_children(parent, children);
    }
}

/// Run an arbitrary number of visitors in parallel during one walk.
/// Heap-allocated; for static-typed two-visitor composition prefer the
/// tuple impl.
pub struct CompositeVisitor<'a, const D: usize> {
    visitors: Vec<&'a mut dyn CascadeVisitor<D>>,
}

impl<'a, const D: usize> CompositeVisitor<'a, D> {
    pub fn new() -> Self { Self { visitors: Vec::new() } }
    pub fn with_capacity(n: usize) -> Self {
        Self { visitors: Vec::with_capacity(n) }
    }
    pub fn push(&mut self, v: &'a mut dyn CascadeVisitor<D>) {
        self.visitors.push(v);
    }
    pub fn len(&self) -> usize { self.visitors.len() }
    pub fn is_empty(&self) -> bool { self.visitors.is_empty() }
}

impl<'a, const D: usize> Default for CompositeVisitor<'a, D> {
    fn default() -> Self { Self::new() }
}

impl<'a, const D: usize> CascadeVisitor<D> for CompositeVisitor<'a, D> {
    fn enter_cell(&mut self, cell: &CellVisit<D>) {
        for v in self.visitors.iter_mut() { v.enter_cell(cell); }
    }
    fn leaf_cell(&mut self, cell: &CellVisit<D>) {
        for v in self.visitors.iter_mut() { v.leaf_cell(cell); }
    }
    fn after_children(&mut self, parent: &CellVisit<D>, children: &[CellVisit<D>]) {
        for v in self.visitors.iter_mut() { v.after_children(parent, children); }
    }
}

#[cfg(test)]
mod visitor_tests {
    use super::*;

    /// Trivial visitor that counts cells per level.
    struct LevelCounter(Vec<u64>);

    impl CascadeVisitor<3> for LevelCounter {
        fn enter_cell(&mut self, cell: &CellVisit<3>) {
            if cell.level >= self.0.len() {
                self.0.resize(cell.level + 1, 0);
            }
            self.0[cell.level] += 1;
        }
    }

    /// Trivial visitor that sums data weights per level.
    struct WeightSum(Vec<f64>);

    impl CascadeVisitor<3> for WeightSum {
        fn enter_cell(&mut self, cell: &CellVisit<3>) {
            if cell.level >= self.0.len() {
                self.0.resize(cell.level + 1, 0.0);
            }
            self.0[cell.level] += cell.data.sum_w;
        }
    }

    fn fake_cell(level: usize, sw_d: f64, sw_r: f64) -> CellVisit<3> {
        CellVisit::<3> {
            level,
            cell_side_trimmed: 1024 >> level,
            cell_id: 0,
            data: CatalogCell { count: sw_d as u64, sum_w: sw_d, sum_w_sq: sw_d },
            randoms: CatalogCell { count: sw_r as u64, sum_w: sw_r, sum_w_sq: sw_r },
            alpha: 1.0,
        }
    }

    #[test]
    fn tuple_composition_dispatches_to_both() {
        let mut a = LevelCounter(vec![]);
        let mut b = WeightSum(vec![]);
        let mut both = (&mut a, &mut b);
        both.enter_cell(&fake_cell(0, 100.0, 100.0));
        both.enter_cell(&fake_cell(1, 50.0, 50.0));
        assert_eq!(a.0, vec![1, 1]);
        assert_eq!(b.0, vec![100.0, 50.0]);
    }

    #[test]
    fn composite_visitor_dispatches_to_all() {
        let mut a = LevelCounter(vec![]);
        let mut b = WeightSum(vec![]);
        let mut c = LevelCounter(vec![]);
        {
            let mut comp = CompositeVisitor::<3>::new();
            comp.push(&mut a);
            comp.push(&mut b);
            comp.push(&mut c);
            assert_eq!(comp.len(), 3);
            comp.enter_cell(&fake_cell(0, 10.0, 10.0));
            comp.enter_cell(&fake_cell(0, 20.0, 20.0));
            comp.enter_cell(&fake_cell(2, 5.0, 5.0));
        }
        assert_eq!(a.0, vec![2, 0, 1]);
        assert_eq!(b.0, vec![30.0, 0.0, 5.0]);
        assert_eq!(c.0, vec![2, 0, 1]);
    }

    #[test]
    fn default_methods_are_no_op() {
        // A visitor that defines nothing should compile and run
        // happily, doing nothing.
        struct DoNothing;
        impl CascadeVisitor<3> for DoNothing {}
        let mut v = DoNothing;
        v.enter_cell(&fake_cell(0, 1.0, 1.0));
        v.leaf_cell(&fake_cell(0, 1.0, 1.0));
        v.after_children(&fake_cell(0, 1.0, 1.0), &[]);
        // No assertions: just confirming it compiles and doesn't panic.
    }
}
