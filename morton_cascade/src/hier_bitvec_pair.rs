// hier_bitvec_pair.rs
//
// Two-catalog bit-vector cascade. Same depth-first traversal as
// `hier_bitvec`, but tracks two point sets (data D, randoms R) sharing a
// single cell hierarchy. Per non-empty cell at level l we accumulate three
// pair counts:
//
//   DD(l) += n_d(c) * (n_d(c) - 1) / 2
//   RR(l) += n_r(c) * (n_r(c) - 1) / 2
//   DR(l) += n_d(c) * n_r(c)
//
// where n_d(c), n_r(c) are the number of data and random points in cell c.
// Cumulative-pair vectors are differenced between adjacent levels to give
// per-shell DD/RR/DR, then combined into the Landy-Szalay estimator.
//
// Coordinate alignment: both catalogs are trimmed against a SHARED
// `CoordRange` (use `CoordRange::analyze_pair`) so cells at every level line
// up between D and R.
//
// Pruning: a cell is descended into iff (n_d > 0 OR n_r > 0). A cell with
// n_d=0, n_r>=1 still contributes to RR; a cell with n_d>=1, n_r=0 still
// contributes to DD. Only DR contributions vanish when either side is empty,
// but those cells are still visited by their non-empty side.
//
// Crossover: when BOTH n_d and n_r drop at or below the crossover threshold,
// the descendants are processed via point-list partition (same trick as the
// single-catalog cascade).
//
// Weights: optional per-point weights (Vec<f64> for each catalog) are
// provided. When weights are present the accumulated pair counts become
// weighted sums:
//   DD_w(l) += sum_c [(sum_{i in c} w_d(i))^2 - sum_{i in c} w_d(i)^2] / 2
//   DR_w(l) += sum_c (sum_{i in c} w_d(i)) * (sum_{j in c} w_r(j))
// (and analogously for RR). For unit weights this reduces to the count
// formulas above.

use crate::coord_range::TrimmedPoints;

/// How the cascade interprets the boundary of the analysis region.
///
/// Most cosmological survey data with a randoms catalog uses [`Isolated`],
/// because the survey footprint already encodes the geometry (cells
/// outside the footprint have W_r = 0 and are excluded). Periodic-box
/// simulations without a randoms catalog use [`Periodic`], in which case
/// the box volume itself defines the mean density and α is computed from
/// box geometry rather than from the random catalog.
///
/// [`Isolated`]: BoundaryMode::Isolated
/// [`Periodic`]: BoundaryMode::Periodic
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Default. Cells are independent of one another; cells outside the
    /// random-catalog footprint are excluded; α = ΣW_d / ΣW_r is read from
    /// the catalogs. Behavior matches the original (pre-`BoundaryMode`)
    /// library.
    Isolated,
    /// Periodic cubic box. The box itself defines the volume and mean
    /// density. When no randoms catalog is provided, α is set so that
    /// the expected count per cell equals (ΣW_d / V_box) * V_cell, and
    /// every cell is in-footprint. Cells at the box boundary are stitched
    /// across via wraparound during traversal.
    Periodic,
}

impl Default for BoundaryMode {
    fn default() -> Self { BoundaryMode::Isolated }
}

/// Per-level cumulative pair counts for two catalogs sharing the same cell
/// hierarchy.
#[derive(Clone, Debug)]
pub struct PairLevelStats {
    /// Tree level (0 = whole box, larger = finer).
    pub level: usize,
    /// Total number of cells at this level (2^(D*level)).
    pub n_total_cells: u64,
    /// Number of cells with n_d > 0 visited at this level.
    pub n_nonempty_d: u64,
    /// Number of cells with n_r > 0 visited at this level.
    pub n_nonempty_r: u64,
    /// Number of cells with both n_d > 0 AND n_r > 0 visited at this level.
    pub n_nonempty_both: u64,
    /// Cumulative DD pair count at this level (pairs of D-points sharing
    /// a cell at this level or any deeper).
    pub cumulative_dd: f64,
    /// Cumulative RR pair count at this level.
    pub cumulative_rr: f64,
    /// Cumulative DR cross pair count at this level.
    pub cumulative_dr: f64,
}

/// Per-shell pair counts and Landy-Szalay correlation in one cube-shell bin.
#[derive(Clone, Debug)]
pub struct XiShell {
    /// Tree level (cube-side at level l = box_side / 2^l).
    pub level: usize,
    /// Cube side at this level, in trimmed-coordinate units.
    pub cell_side_trimmed: f64,
    /// Outer scale of the shell (= cube side at the parent level).
    pub r_outer_trimmed: f64,
    /// Inner scale of the shell (= cube side at this level).
    pub r_inner_trimmed: f64,
    /// DD pairs falling in this shell.
    pub dd: f64,
    /// RR pairs falling in this shell.
    pub rr: f64,
    /// DR cross pairs falling in this shell.
    pub dr: f64,
    /// Landy-Szalay estimator: (DD_norm - 2 DR_norm + RR_norm) / RR_norm.
    /// Returns NaN if RR_norm == 0.
    pub xi_ls: f64,
}

/// Per-level density-field statistics — moments and PDF of the cell-by-cell
/// density contrast δ(c) = W_d(c) / (α · W_r(c)) − 1, where α = ΣW_d / ΣW_r
/// is the global mean-density ratio.
///
/// Cells with W_r(c) ≤ `w_r_min` are excluded from all moments and PDFs;
/// they correspond to "outside the survey footprint" regions where the
/// random catalog has no presence. (This is the cosmological-survey way of
/// handling masks: the random catalog *encodes* the selection function via
/// its spatial density, including window, fiber-completeness, imaging
/// systematics, dust correction, etc.)
///
/// Moments are weighted by W_r (effective volume): a cell that the survey
/// integrates over twice as much contributes twice as much weight to
/// `<δ>_W_r`, `<δ²>_W_r`, etc. For matched-density data and randoms,
/// `<δ>_W_r ≈ 0` is exact by construction (it's the global α normalization
/// ensuring this).
#[derive(Clone, Debug)]
pub struct DensityFieldStats {
    /// Tree level.
    pub level: usize,
    /// Cube side at this level, in trimmed-coordinate units.
    pub cell_side_trimmed: f64,
    /// Number of cells at this level with W_r(c) > w_r_min (active cells,
    /// inside footprint).
    pub n_cells_active: u64,
    /// Sum of W_r over active cells. With the global α normalization this
    /// equals ΣW_r exactly (modulo the w_r_min cut).
    pub sum_w_r_active: f64,
    /// W_r-weighted first moment of δ. Should be ≈ 0 for the global
    /// normalization (sanity check).
    pub mean_delta: f64,
    /// W_r-weighted central second moment of δ: variance of the density
    /// contrast field smoothed at this cell scale.
    pub var_delta: f64,
    /// W_r-weighted central third moment of δ: <δ³>_W_r.
    /// (NOT the standardized skewness — divide by var^{3/2} for that.)
    pub m3_delta: f64,
    /// W_r-weighted central fourth moment of δ: <δ⁴>_W_r.
    /// (NOT the standardized kurtosis — divide by var² for that.)
    pub m4_delta: f64,
    /// Min and max δ observed at active cells of this level.
    pub min_delta: f64,
    pub max_delta: f64,
    /// W_r-weighted reduced skewness S_3 = <δ³>_W_r / <δ²>_W_r²
    /// (the cosmological reduced cumulant; equals 34/7 + γ at LO for ZA).
    pub s3_delta: f64,
    // ---- Diagnostic: data outside footprint ----
    /// Number of cells visited at this level where data is present
    /// (n_d > 0) but the random weight is at or below `w_r_min`. These
    /// cells are EXCLUDED from the moments above (no W_r ⇒ δ undefined).
    /// A non-zero count is a catalog-quality signal: data points falling
    /// in regions the random catalog says are outside the survey.
    pub n_cells_data_outside: u64,
    /// Sum of W_d over those outside-footprint cells. Compare to
    /// `sum_w_d_active = sum_w_r_active * α` to gauge what fraction of
    /// the total data weight lies outside footprint at this scale.
    pub sum_w_d_outside: f64,
    /// Histogram of δ values, in bins evenly spaced in `log(1+δ)` (i.e.
    /// uniform in the log of the density-relative-to-mean). Empty if
    /// `hist_bins == 0` was requested. Each bin holds the W_r-weighted
    /// fraction of cells in that bin.
    /// `hist_bin_edges` has `hist_bins + 1` entries when populated.
    pub hist_bin_edges: Vec<f64>,
    /// W_r-weighted PDF: `hist_density[k]` is the fraction of total W_r in
    /// bin k. Sums to (1 − fraction of cells with 1+δ outside the binned
    /// log-range). The over/under-flow are reported in `hist_underflow_w_r`
    /// and `hist_overflow_w_r`.
    pub hist_density: Vec<f64>,
    pub hist_underflow_w_r: f64,
    pub hist_overflow_w_r: f64,
    /// W_r-weighted raw sums of δ^k for k = 1..=4, in the order
    /// `[Σw_r·δ, Σw_r·δ², Σw_r·δ³, Σw_r·δ⁴]`. Together with
    /// `sum_w_r_active` these are the moment generators — they pool
    /// additively across multiple cascade runs (used by
    /// [`crate::multi_run::CascadeRunner`]'s aggregation methods).
    pub raw_sum_w_r_delta_pow: [f64; 4],
}

/// Per-level cell-count PMF computed by the cascade.
///
/// `histogram_density[k]` is the probability that a random cell at this
/// level contains exactly `k` data points.
///
/// **Normalization conventions:**
/// - In [`BoundaryMode::Periodic`] (no randoms): the PMF is taken over
///   the full box. `histogram_density` sums to 1, and `histogram_density[0]`
///   includes the contribution of unvisited (zero-data) cells via an
///   analytic correction.
/// - In [`BoundaryMode::Isolated`]: the PMF is taken over the cells the
///   cascade actually descended into (i.e. cells with at least one data or
///   random point). `histogram_density` sums to 1 over those cells, but
///   the unsampled "outside-footprint" volume is not represented.
///
/// `histogram_counts[k]` always reports the raw integer count of *visited*
/// cells with k data points. In Periodic mode the unvisited count
/// (`n_cells_total − n_cells_visited`) all lands in the k=0 bin of
/// `histogram_density`, but is **not** reflected in `histogram_counts[0]`.
#[derive(Clone, Debug)]
pub struct CicPmfStats {
    pub level: usize,
    /// Cube side at this level, in trimmed-coordinate units.
    pub cell_side_trimmed: f64,
    /// Cells the cascade actually descended into (had data and/or
    /// randoms). Equal to the sum of `histogram_counts`.
    pub n_cells_visited: u64,
    /// Total cells at this level: `2^(D*level)`. In Periodic mode this is
    /// the box-tiling count; in Isolated mode it's the same number but the
    /// PMF only spans `n_cells_visited` of them.
    pub n_cells_total: u64,
    /// Raw integer count of visited cells holding exactly k data points.
    /// Length = `max_k_observed + 1`.
    pub histogram_counts: Vec<u64>,
    /// Probability that a random cell at this level contains k data points.
    /// Length matches `histogram_counts`. In Periodic mode `[0]` includes
    /// the analytic correction for unvisited cells; in Isolated mode the
    /// density is over visited cells only.
    pub histogram_density: Vec<f64>,
    /// Mean cell count: ⟨N⟩. Equal to `total_n_d / n_cells_total`
    /// in Periodic mode (an exact identity); shot-noise-fluctuating
    /// in Isolated mode.
    pub mean: f64,
    /// Variance of cell count: ⟨N²⟩ − ⟨N⟩².
    pub var: f64,
    /// Standardized 3rd moment (μ₃ / σ³).
    pub skew: f64,
    /// Standardized 4th moment (μ₄ / σ⁴). Excess kurtosis = kurt − 3.
    pub kurt: f64,
}

/// Configuration for [`BitVecCascadePair::analyze_cic_pmf`].
#[derive(Clone, Debug)]
pub struct CicPmfConfig {
    /// Maximum number of histogram bins to retain. Cells with N points
    /// above this cap are clipped to the top bin (and a warning could be
    /// surfaced; we just fold them in). Defaults to 2,000,000 to match
    /// the dense-grid PMF's protection against pathological inputs.
    pub max_bins: usize,
}

impl Default for CicPmfConfig {
    fn default() -> Self {
        Self { max_bins: 2_000_000 }
    }
}

/// Configuration for the histogram of δ in `analyze_field_stats`.
#[derive(Clone, Debug)]
pub struct FieldStatsConfig {
    /// Cells with W_r(c) ≤ w_r_min are excluded (outside footprint).
    /// Pass 0.0 to include every cell visited (still excludes empty cells
    /// not visited by the cascade).
    pub w_r_min: f64,
    /// Number of histogram bins. Pass 0 to disable PDF estimation.
    pub hist_bins: usize,
    /// Histogram covers `log10(1 + δ)` ∈ [log_min, log_max]. Defaults of
    /// (-3.0, +3.0) cover overdensities from 10^-3 to 10^3 times the mean.
    pub hist_log_min: f64,
    pub hist_log_max: f64,
    /// Use Neumaier compensated summation for the outer cell-aggregator
    /// accumulators. Costs ~4× per add but recovers ~full f64 precision
    /// regardless of summation order or magnitude spread. Useful when
    /// per-particle weights span wide dynamic range (e.g., FKP weights
    /// spanning multiple decades) or when δ moments would otherwise
    /// suffer catastrophic cancellation across many cells. Default: false
    /// (naive summation).
    pub compensated_sums: bool,
}

impl Default for FieldStatsConfig {
    fn default() -> Self {
        Self { w_r_min: 0.0, hist_bins: 50, hist_log_min: -3.0, hist_log_max: 3.0,
               compensated_sums: false }
    }
}

/// Per-level cell-wavelet anisotropy statistics. **D-generic.**
///
/// At each level the cascade visits parent cells and their `2^D` children.
/// The child cell-counts decompose into one isotropic average (which is
/// just the parent's count, recovered) plus `2^D - 1` axis-pattern detail
/// coefficients — the Haar wavelets of the cell-count field. This works
/// in every dimension; only the labels of the patterns and their geometric
/// interpretation differ.
///
/// We accumulate the W_r-weighted variance of each detail coefficient.
/// The patterns factor by Hamming weight:
///
/// - **Weight 1** (D patterns): axis-aligned. These are the natural
///   probes of anisotropy along a privileged axis (e.g. line-of-sight in
///   redshift surveys).
/// - **Weight 2** (D(D−1)/2 patterns): face-diagonal in 3D and above.
/// - ...
/// - **Weight D** (1 pattern): body-diagonal.
///
/// **Quadrupole-like LoS-vs-transverse moment** (D ≥ 2):
///
///   Q_2 = ⟨w_LoS²⟩ − mean over D−1 transverse axes of ⟨w_perp²⟩
///
/// where the LoS is the **last axis** by convention (axis index D−1).
/// In 3D with axis ordering (x, y, z) this matches the conventional
/// definition: Q_2 = ⟨w_z²⟩ − ½(⟨w_x²⟩ + ⟨w_y²⟩). In 2D with axes
/// (x, y) it becomes Q_2 = ⟨w_y²⟩ − ⟨w_x²⟩. In 1D Q_2 ≡ 0 (no
/// transverse direction exists).
///
/// This is a CASCADE-NATIVE observable: it is NOT the same as HIPSTER's
/// P_2(k) (Legendre multipole of the pair-count power spectrum), but it
/// captures the same physics — anisotropy along a chosen axis — at every
/// dyadic scale.
#[derive(Clone, Debug)]
pub struct AnisotropyStats {
    pub level: usize,
    /// Cube side at this level, trimmed-coord units.
    pub cell_side_trimmed: f64,
    /// Number of parent cells visited at this level (cells where every
    /// child passed the footprint cutoff; see strict-mode policy).
    pub n_parents: u64,
    /// Sum of W_r over all visited parent cells.
    pub sum_w_r_parents: f64,
    /// W_r-weighted mean of squared wavelet coefficient for **each
    /// non-trivial pattern** e ∈ {1, ..., 2^D − 1}. Indexed by raw
    /// pattern bits: position `e` (for e in 1..2^D) holds ⟨w_e²⟩.
    /// Position 0 is unused (constant-pattern, identically zero).
    /// Length is exactly `2^D`.
    pub mean_w_squared_by_pattern: Vec<f64>,
    /// W_r-weighted mean of squared axis-aligned wavelet coefficients
    /// in canonical axis order: position `d` holds ⟨w_axis_d²⟩.
    /// Length is exactly `D`. The LoS-by-convention axis is the last
    /// element (index D−1).
    pub mean_w_squared_axis: Vec<f64>,
    /// Quadrupole-like LoS-vs-transverse moment, with LoS = axis D−1.
    /// 0.0 in 1D.
    pub quadrupole_los: f64,
    /// `quadrupole_los` divided by the mean of axis variances.
    /// Dimensionless, comparable across scales.
    pub reduced_quadrupole_los: f64,
}

impl AnisotropyStats {
    /// Convenience: get ⟨w_e²⟩ by raw pattern index `e ∈ 1..2^D`.
    /// Returns 0.0 for out-of-range indices.
    #[inline]
    pub fn mean_w_squared_pattern(&self, e: usize) -> f64 {
        self.mean_w_squared_by_pattern.get(e).copied().unwrap_or(0.0)
    }

    /// Convenience: get ⟨w²⟩ for a specific axis-aligned direction.
    /// Returns 0.0 for axis indices ≥ D.
    #[inline]
    pub fn mean_w_squared_for_axis(&self, axis: usize) -> f64 {
        self.mean_w_squared_axis.get(axis).copied().unwrap_or(0.0)
    }

    /// Patterns of a given Hamming weight, in increasing pattern-bit order.
    /// Weight 1 = axis-aligned (returns D entries), weight 2 = face-diagonal
    /// (D(D−1)/2 entries), ..., weight D = body-diagonal (1 entry).
    pub fn pattern_indices_with_hamming_weight(&self, weight: u32) -> Vec<usize> {
        let n = self.mean_w_squared_by_pattern.len();
        (1..n).filter(|&e| (e as u32).count_ones() == weight).collect()
    }
}

/// Second-order Haar scattering coefficients at a single (l_1, l_2) pair of
/// cascade levels with l_2 < l_1 (l_2 coarser than l_1).
///
/// First-order Haar scattering produces the field |w_{e_1}^(l_1)|(c) defined
/// on level-l_1 parent cells: the magnitude of the wavelet response at scale
/// R_{l_1} along direction e_1. Second-order asks how that field varies at a
/// coarser scale R_{l_2}: we compute Haar wavelet coefficients of the |w|
/// field at level l_2, again per direction e_2.
///
/// Following Mallat (2012, "Group invariant scattering"), this captures
/// non-Gaussian information complementary to the bispectrum: how localized
/// fluctuations at one scale themselves cluster at coarser scales.
///
/// We restrict to axis-aligned directions e_1, e_2 ∈ {x, y, z}: this covers
/// 9 = 3 × 3 direction pairs per (l_1, l_2). Face-diagonal and body-diagonal
/// extensions are straightforward but produce less interpretable output.
#[derive(Clone, Debug)]
pub struct ScatteringStats {
    /// Finer scale level (where the first-order wavelet was computed).
    pub level_fine: usize,
    /// Coarser scale level (where the second-order wavelet is taken).
    pub level_coarse: usize,
    /// Cube side at the finer level (trimmed coords).
    pub cell_side_fine_trimmed: f64,
    /// Cube side at the coarser level (trimmed coords).
    pub cell_side_coarse_trimmed: f64,
    /// Number of coarse-level parent cells contributing.
    pub n_parents_coarse: u64,
    /// Sum of W_r over those parent cells.
    pub sum_w_r_parents: f64,
    /// First-order coefficient value at the fine level, indexed by [e_1] for
    /// e_1 ∈ {x=0, y=1, z=2}. Reproduced here for self-contained output.
    /// Equals < |w_{e_1}^(l_1)|² > over level-l_1 cells (W_r-weighted).
    /// (At l_2 = l_1, the second-order trivially equals the first-order.)
    pub first_order: [f64; 3],
    /// Second-order coefficient: < |w_{e_2}^(l_2)[ |w_{e_1}^(l_1)| ]|² >
    /// indexed by [e_1][e_2] with e_1, e_2 ∈ {x=0, y=1, z=2}.
    pub second_order: [[f64; 3]; 3],
}

/// The two-catalog bit-vector cascade.
pub struct BitVecCascadePair<const D: usize> {
    /// Trimmed data points (D-catalog).
    pub data: TrimmedPoints<D>,
    /// Trimmed random points (R-catalog).
    pub randoms: TrimmedPoints<D>,
    /// Optional per-data-point weights. Same length as `data.points` if Some.
    pub weights_d: Option<Vec<f64>>,
    /// Optional per-random-point weights. Same length as `randoms.points` if Some.
    pub weights_r: Option<Vec<f64>>,
    /// Number of cascade levels (max effective bits across axes; SHARED with
    /// `data.range` and `randoms.range`).
    pub l_max: usize,
    /// Number of u64 words per data bit-vector (= ceil(N_d / 64)).
    pub n_words_d: usize,
    /// Number of u64 words per random bit-vector (= ceil(N_r / 64)).
    pub n_words_r: usize,
    /// Crossover threshold: when BOTH cell counts (D and R) drop to this many
    /// or fewer, switch to point-list representation.
    pub crossover_threshold: usize,
    /// `bit_planes_d[d][l]`: N_d-bit vector, bit i set iff D-point i has
    /// bit (eff_d - 1 - l) of its trimmed axis-d coord set.
    pub bit_planes_d: Vec<Vec<Vec<u64>>>,
    /// Same for the R catalog.
    pub bit_planes_r: Vec<Vec<Vec<u64>>>,
    /// How the cascade should treat the boundary of the analysis region.
    /// Defaults to [`BoundaryMode::Isolated`] (the historical behavior).
    pub boundary_mode: BoundaryMode,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Default crossover threshold for `build`. See
    /// `BitVecCascade::default_crossover_threshold` for rationale; the pair
    /// version scales with `max(N_d, N_r)` because the bit-vector path's
    /// cost-per-cell is set by the larger catalog.
    pub fn default_crossover_threshold(n_d: usize, n_r: usize) -> usize {
        (n_d.max(n_r) / 64).max(64)
    }

    /// Build the pair cascade from two TrimmedPoints sharing a single
    /// CoordRange.
    ///
    /// PRECONDITION: `data.range` and `randoms.range` must be equal. Use
    /// `CoordRange::analyze_pair` and `TrimmedPoints::from_points_with_range`
    /// to guarantee this. Panics if ranges differ.
    ///
    /// Defaults to [`BoundaryMode::Isolated`]. Use
    /// [`Self::build_full_with_boundary`] to choose explicitly.
    pub fn build(
        data: TrimmedPoints<D>,
        randoms: TrimmedPoints<D>,
        l_max: Option<usize>,
    ) -> Self {
        let thresh = Self::default_crossover_threshold(
            data.points.len(), randoms.points.len());
        Self::build_full_with_boundary(
            data, randoms, None, None, l_max, thresh, BoundaryMode::Isolated)
    }

    /// Build with per-point weights and explicit crossover threshold.
    /// Defaults to [`BoundaryMode::Isolated`]; use
    /// [`Self::build_full_with_boundary`] to choose.
    ///
    /// PRECONDITION as in `build`. Weight slices, if Some, must match the
    /// corresponding catalog length.
    pub fn build_full(
        data: TrimmedPoints<D>,
        randoms: TrimmedPoints<D>,
        weights_d: Option<Vec<f64>>,
        weights_r: Option<Vec<f64>>,
        l_max: Option<usize>,
        crossover_threshold: usize,
    ) -> Self {
        Self::build_full_with_boundary(
            data, randoms, weights_d, weights_r, l_max, crossover_threshold,
            BoundaryMode::Isolated)
    }

    /// Build with per-point weights, explicit crossover threshold, and an
    /// explicit [`BoundaryMode`]. The other `build*` methods delegate here
    /// with `BoundaryMode::Isolated`.
    pub fn build_full_with_boundary(
        data: TrimmedPoints<D>,
        randoms: TrimmedPoints<D>,
        weights_d: Option<Vec<f64>>,
        weights_r: Option<Vec<f64>>,
        l_max: Option<usize>,
        crossover_threshold: usize,
        boundary_mode: BoundaryMode,
    ) -> Self {
        // Range alignment is what makes cells line up. Enforce it.
        assert_eq!(data.range.bit_min, randoms.range.bit_min,
            "data.range.bit_min != randoms.range.bit_min — \
             call CoordRange::analyze_pair and from_points_with_range");
        assert_eq!(data.range.effective_bits, randoms.range.effective_bits,
            "data.range.effective_bits != randoms.range.effective_bits");

        if let Some(ref w) = weights_d {
            assert_eq!(w.len(), data.points.len(),
                "weights_d length {} != data.points length {}", w.len(), data.points.len());
        }
        if let Some(ref w) = weights_r {
            assert_eq!(w.len(), randoms.points.len(),
                "weights_r length {} != randoms.points length {}", w.len(), randoms.points.len());
        }

        let n_d = data.points.len();
        let n_r = randoms.points.len();
        let n_words_d = (n_d + 63) / 64;
        let n_words_r = (n_r + 63) / 64;

        let supported = data.range.max_supported_l_max() as usize;
        let l_max = l_max.map(|l| l.min(supported)).unwrap_or(supported);

        // Build bit planes for both catalogs.
        let mut bit_planes_d: Vec<Vec<Vec<u64>>> =
            vec![vec![vec![0u64; n_words_d]; l_max]; D];
        for (i, p) in data.points.iter().enumerate() {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let mask = 1u64 << bit_idx;
            for d in 0..D {
                let eff = data.range.effective_bits[d] as usize;
                for l in 0..l_max.min(eff) {
                    let coord_bit = eff - 1 - l;
                    if (p[d] >> coord_bit) & 1 == 1 {
                        bit_planes_d[d][l][word_idx] |= mask;
                    }
                }
            }
        }
        let mut bit_planes_r: Vec<Vec<Vec<u64>>> =
            vec![vec![vec![0u64; n_words_r]; l_max]; D];
        for (i, p) in randoms.points.iter().enumerate() {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let mask = 1u64 << bit_idx;
            for d in 0..D {
                let eff = randoms.range.effective_bits[d] as usize;
                for l in 0..l_max.min(eff) {
                    let coord_bit = eff - 1 - l;
                    if (p[d] >> coord_bit) & 1 == 1 {
                        bit_planes_r[d][l][word_idx] |= mask;
                    }
                }
            }
        }

        Self {
            data, randoms,
            weights_d, weights_r,
            l_max, n_words_d, n_words_r,
            crossover_threshold,
            bit_planes_d, bit_planes_r,
            boundary_mode,
        }
    }

    /// Override the boundary mode after construction. Returns the previous
    /// mode. Useful for tests and for sharing one cascade across analyses
    /// that differ only in boundary treatment.
    pub fn set_boundary_mode(&mut self, mode: BoundaryMode) -> BoundaryMode {
        let prev = self.boundary_mode;
        self.boundary_mode = mode;
        prev
    }

    /// Build a periodic-box cascade from a single data catalog and box
    /// dimensions, with no randoms catalog.
    ///
    /// The box geometry — given as `box_bits[d]`, defining a side of
    /// `2^box_bits[d]` along each axis — defines the mean density and the
    /// per-cell expected weight. Internally the randoms catalog is left
    /// empty and the field-stats / anisotropy visitors compute α from the
    /// box volume rather than from a random catalog.
    ///
    /// All `data.points[i][d]` must satisfy `0 ≤ p[i][d] < 2^box_bits[d]`.
    /// The `data.range` is replaced by [`CoordRange::for_box_bits`] so that
    /// the cascade resolution matches the box, not the data extent.
    ///
    /// Use [`Self::build`] for survey-style analyses with a real randoms
    /// catalog (the [`BoundaryMode::Isolated`] case).
    pub fn build_periodic(
        data: TrimmedPoints<D>,
        box_bits: [u32; D],
        l_max: Option<usize>,
    ) -> Self {
        Self::build_periodic_full(data, box_bits, None, l_max,
            Self::default_crossover_threshold(0, 0))
    }

    /// As [`Self::build_periodic`] but accepts per-point weights and an
    /// explicit crossover threshold. The crossover threshold defaults to
    /// `(N_d / 64).max(64)` if you don't have a strong reason to override.
    pub fn build_periodic_full(
        data: TrimmedPoints<D>,
        box_bits: [u32; D],
        weights_d: Option<Vec<f64>>,
        l_max: Option<usize>,
        crossover_threshold: usize,
    ) -> Self {
        // Replace the data range with the box-derived range so the cascade
        // covers the full box, not just the data extent.
        let box_range = crate::coord_range::CoordRange::<D>::for_box_bits(box_bits);
        let data_with_box = crate::coord_range::TrimmedPoints::<D> {
            points: data.points,
            range: box_range.clone(),
        };
        // Empty randoms catalog with the same range so cells align.
        let empty_randoms = crate::coord_range::TrimmedPoints::<D> {
            points: Vec::new(),
            range: box_range,
        };
        // Adjust crossover threshold to use N_d when caller passed the
        // "no-randoms" sentinel of 64.
        let thresh = if crossover_threshold == Self::default_crossover_threshold(0, 0) {
            Self::default_crossover_threshold(data_with_box.points.len(), 0)
        } else {
            crossover_threshold
        };
        Self::build_full_with_boundary(
            data_with_box, empty_randoms, weights_d, None, l_max, thresh,
            BoundaryMode::Periodic)
    }

    /// Number of D-points.
    pub fn n_d(&self) -> usize { self.data.points.len() }
    /// Number of R-points.
    pub fn n_r(&self) -> usize { self.randoms.points.len() }
    /// Cascade depth (l_max). Number of bits per axis used to address
    /// the deepest cells.
    pub fn l_max(&self) -> usize { self.l_max }

    /// Read-only access to a data-catalog bit plane: axis `d`, level
    /// `l`. Returns the per-particle bit array (one bit per point,
    /// packed into u64 words). Useful for building inverse-direction
    /// indices (see `cell_membership`).
    pub fn bit_planes_d_at(&self, d: usize, l: usize) -> &[u64] {
        &self.bit_planes_d[d][l]
    }

    /// Read-only access to a randoms-catalog bit plane: axis `d`,
    /// level `l`. See [`Self::bit_planes_d_at`].
    pub fn bit_planes_r_at(&self, d: usize, l: usize) -> &[u64] {
        &self.bit_planes_r[d][l]
    }

    /// Read-only access to data-catalog per-particle weights, if any.
    /// Returns None for unit-weighted catalogs.
    pub fn weights_d(&self) -> Option<&[f64]> {
        self.weights_d.as_deref()
    }

    /// Read-only access to randoms-catalog per-particle weights, if any.
    /// Returns None for unit-weighted catalogs.
    pub fn weights_r(&self) -> Option<&[f64]> {
        self.weights_r.as_deref()
    }

    /// Build the per-particle cell-membership index for the data
    /// catalog. Inverse of the cascade's bit-plane representation:
    /// given any (level, cell_id), returns the indices of the data
    /// particles in that cell. See [`crate::cell_membership`] for
    /// design rationale and complexity.
    ///
    /// Cost: O(N log N) for the Morton sort. Pay once per cascade,
    /// then lookups are O(log N) each.
    pub fn cell_membership_data(&self) -> crate::cell_membership::CellMembership {
        crate::cell_membership::CellMembership::build(
            self, crate::cell_membership::WhichCatalog::Data)
    }

    /// Build the per-particle cell-membership index for the randoms
    /// catalog. See [`Self::cell_membership_data`].
    pub fn cell_membership_randoms(&self) -> crate::cell_membership::CellMembership {
        crate::cell_membership::CellMembership::build(
            self, crate::cell_membership::WhichCatalog::Randoms)
    }

    /// Sum of D-weights (= n_d if unit weights).
    pub fn sum_w_d(&self) -> f64 {
        match &self.weights_d {
            Some(w) => w.iter().sum(),
            None => self.n_d() as f64,
        }
    }
    /// Sum of R-weights (= n_r if unit weights).
    pub fn sum_w_r(&self) -> f64 {
        match &self.weights_r {
            Some(w) => w.iter().sum(),
            None => self.n_r() as f64,
        }
    }

    /// Run the joint cascade traversal and accumulate per-level pair counts.
    pub fn analyze(&self) -> Vec<PairLevelStats> {
        let n_d = self.n_d();
        let n_r = self.n_r();
        let n_words_d = self.n_words_d;
        let n_words_r = self.n_words_r;
        let n_levels = self.l_max + 1;

        let mut cum_dd = vec![0.0f64; n_levels];
        let mut cum_rr = vec![0.0f64; n_levels];
        let mut cum_dr = vec![0.0f64; n_levels];
        let mut nne_d = vec![0u64; n_levels];
        let mut nne_r = vec![0u64; n_levels];
        let mut nne_both = vec![0u64; n_levels];

        // Initial root memberships
        let mut root_d = vec![u64::MAX; n_words_d.max(1)];
        if n_d == 0 {
            root_d.iter_mut().for_each(|w| *w = 0);
        } else if n_d % 64 != 0 {
            let last = n_d % 64;
            root_d[n_words_d - 1] = (1u64 << last) - 1;
        }
        let mut root_r = vec![u64::MAX; n_words_r.max(1)];
        if n_r == 0 {
            root_r.iter_mut().for_each(|w| *w = 0);
        } else if n_r % 64 != 0 {
            let last = n_r % 64;
            root_r[n_words_r - 1] = (1u64 << last) - 1;
        }

        // Truncate root vectors to actual word counts (n_words_d/r may be 0)
        let root_d_view: &[u64] = if n_d == 0 { &[] } else { &root_d[..n_words_d] };
        let root_r_view: &[u64] = if n_r == 0 { &[] } else { &root_r[..n_words_r] };

        if n_d > 0 || n_r > 0 {
            self.recurse(
                root_d_view, root_r_view, 0,
                &mut cum_dd, &mut cum_rr, &mut cum_dr,
                &mut nne_d, &mut nne_r, &mut nne_both,
            );
        }

        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            out.push(PairLevelStats {
                level: l,
                n_total_cells: 1u64 << (D * l),
                n_nonempty_d: nne_d[l],
                n_nonempty_r: nne_r[l],
                n_nonempty_both: nne_both[l],
                cumulative_dd: cum_dd[l],
                cumulative_rr: cum_rr[l],
                cumulative_dr: cum_dr[l],
            });
        }
        out
    }

    // Helper: weighted cell sums given a membership vector. Returns
    // (count, sum_w, sum_w2) where count is unweighted cardinality.
    #[inline]
    fn cell_sums_d(&self, mem: &[u64]) -> (u64, f64, f64) {
        let count = popcount_vec(mem);
        match &self.weights_d {
            None => (count, count as f64, count as f64),
            Some(w) => {
                let mut s = 0.0;
                let mut s2 = 0.0;
                for (w_idx, &word) in mem.iter().enumerate() {
                    let mut wb = word;
                    let base = w_idx * 64;
                    while wb != 0 {
                        let i = base + wb.trailing_zeros() as usize;
                        let wi = w[i];
                        s += wi;
                        s2 += wi * wi;
                        wb &= wb - 1;
                    }
                }
                (count, s, s2)
            }
        }
    }
    #[inline]
    fn cell_sums_r(&self, mem: &[u64]) -> (u64, f64, f64) {
        let count = popcount_vec(mem);
        match &self.weights_r {
            None => (count, count as f64, count as f64),
            Some(w) => {
                let mut s = 0.0;
                let mut s2 = 0.0;
                for (w_idx, &word) in mem.iter().enumerate() {
                    let mut wb = word;
                    let base = w_idx * 64;
                    while wb != 0 {
                        let i = base + wb.trailing_zeros() as usize;
                        let wi = w[i];
                        s += wi;
                        s2 += wi * wi;
                        wb &= wb - 1;
                    }
                }
                (count, s, s2)
            }
        }
    }

    // Same but for an explicit point-list representation
    #[inline]
    fn cell_sums_d_list(&self, indices: &[u32]) -> (u64, f64, f64) {
        let count = indices.len() as u64;
        match &self.weights_d {
            None => (count, count as f64, count as f64),
            Some(w) => {
                let mut s = 0.0;
                let mut s2 = 0.0;
                for &i in indices {
                    let wi = w[i as usize];
                    s += wi;
                    s2 += wi * wi;
                }
                (count, s, s2)
            }
        }
    }
    #[inline]
    fn cell_sums_r_list(&self, indices: &[u32]) -> (u64, f64, f64) {
        let count = indices.len() as u64;
        match &self.weights_r {
            None => (count, count as f64, count as f64),
            Some(w) => {
                let mut s = 0.0;
                let mut s2 = 0.0;
                for &i in indices {
                    let wi = w[i as usize];
                    s += wi;
                    s2 += wi * wi;
                }
                (count, s, s2)
            }
        }
    }

    fn accumulate_cell(
        &self,
        level: usize,
        n_d: u64, sw_d: f64, sw2_d: f64,
        n_r: u64, sw_r: f64, sw2_r: f64,
        cum_dd: &mut [f64], cum_rr: &mut [f64], cum_dr: &mut [f64],
        nne_d: &mut [u64], nne_r: &mut [u64], nne_both: &mut [u64],
    ) {
        // Pair-count contributions for this cell at this level.
        // Unit weights: (sw_d, sw2_d) = (n_d, n_d), so (sw^2 - sw2)/2 = n(n-1)/2.
        cum_dd[level] += 0.5 * (sw_d * sw_d - sw2_d);
        cum_rr[level] += 0.5 * (sw_r * sw_r - sw2_r);
        cum_dr[level] += sw_d * sw_r;
        if n_d > 0 { nne_d[level] += 1; }
        if n_r > 0 { nne_r[level] += 1; }
        if n_d > 0 && n_r > 0 { nne_both[level] += 1; }
    }

    /// Bit-vector recursion. `mem_d` and `mem_r` are membership slices for the
    /// current cell (one per catalog).
    fn recurse(
        &self,
        mem_d: &[u64],
        mem_r: &[u64],
        level: usize,
        cum_dd: &mut [f64], cum_rr: &mut [f64], cum_dr: &mut [f64],
        nne_d: &mut [u64], nne_r: &mut [u64], nne_both: &mut [u64],
    ) {
        let (n_d, sw_d, sw2_d) = self.cell_sums_d(mem_d);
        let (n_r, sw_r, sw2_r) = self.cell_sums_r(mem_r);
        if n_d == 0 && n_r == 0 {
            return;
        }
        self.accumulate_cell(level, n_d, sw_d, sw2_d, n_r, sw_r, sw2_r,
                              cum_dd, cum_rr, cum_dr,
                              nne_d, nne_r, nne_both);

        if level >= self.l_max {
            return;
        }

        // Crossover: if BOTH counts are at/below threshold, switch to lists.
        // (If only one side is below threshold but the other has many points,
        // bit-vec is still cheaper for the heavy side.)
        let thresh = self.crossover_threshold as u64;
        if n_d <= thresh && n_r <= thresh {
            // Extract index lists once for both catalogs.
            let idx_d = mem_to_indices(mem_d, n_d as usize);
            let idx_r = mem_to_indices(mem_r, n_r as usize);
            // Partition into 2^D children at descent step `level`, then recurse
            // on each non-empty child at level+1. This mirrors the bit-vec path
            // which calls recurse once per child at level+1.
            let n_children = 1usize << D;
            let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            for &i in &idx_d {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_d[child_id].push(i);
            }
            for &i in &idx_r {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_r[child_id].push(i);
            }
            for c in 0..n_children {
                if !buckets_d[c].is_empty() || !buckets_r[c].is_empty() {
                    self.recurse_pointlist(&buckets_d[c], &buckets_r[c], level + 1,
                                            cum_dd, cum_rr, cum_dr,
                                            nne_d, nne_r, nne_both);
                }
            }
            return;
        }

        // Bit-vec descent: compute upper-half vectors for each axis & catalog,
        // then enumerate 2^D children using the same membership AND structure.
        let nwd = self.n_words_d;
        let nwr = self.n_words_r;

        let mut up_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
        let mut lo_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
        for d in 0..D {
            for w in 0..nwd {
                let bp = self.bit_planes_d[d][level][w];
                up_d[d][w] = mem_d[w] & bp;
                lo_d[d][w] = mem_d[w] & !bp;
            }
        }
        let mut up_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
        let mut lo_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
        for d in 0..D {
            for w in 0..nwr {
                let bp = self.bit_planes_r[d][level][w];
                up_r[d][w] = mem_r[w] & bp;
                lo_r[d][w] = mem_r[w] & !bp;
            }
        }

        let mut child_d = vec![0u64; nwd];
        let mut child_r = vec![0u64; nwr];
        for child in 0..(1u32 << D) {
            // Axis-0 initialization
            let b0 = (child & 1) != 0;
            child_d.copy_from_slice(if b0 { &up_d[0] } else { &lo_d[0] });
            child_r.copy_from_slice(if b0 { &up_r[0] } else { &lo_r[0] });
            // AND axes 1..D
            for d in 1..D {
                let bd = (child >> d) & 1 != 0;
                let half_d = if bd { &up_d[d] } else { &lo_d[d] };
                let half_r = if bd { &up_r[d] } else { &lo_r[d] };
                for w in 0..nwd { child_d[w] &= half_d[w]; }
                for w in 0..nwr { child_r[w] &= half_r[w]; }
            }
            self.recurse(&child_d, &child_r, level + 1,
                         cum_dd, cum_rr, cum_dr,
                         nne_d, nne_r, nne_both);
        }
    }

    /// Point-list recursion. Both catalogs given as explicit index lists.
    fn recurse_pointlist(
        &self,
        idx_d: &[u32],
        idx_r: &[u32],
        level: usize,
        cum_dd: &mut [f64], cum_rr: &mut [f64], cum_dr: &mut [f64],
        nne_d: &mut [u64], nne_r: &mut [u64], nne_both: &mut [u64],
    ) {
        let (n_d, sw_d, sw2_d) = self.cell_sums_d_list(idx_d);
        let (n_r, sw_r, sw2_r) = self.cell_sums_r_list(idx_r);
        if n_d == 0 && n_r == 0 {
            return;
        }
        self.accumulate_cell(level, n_d, sw_d, sw2_d, n_r, sw_r, sw2_r,
                              cum_dd, cum_rr, cum_dr,
                              nne_d, nne_r, nne_both);

        if level >= self.l_max {
            return;
        }

        // Partition both lists into 2^D children using bit_planes at this descent
        // step. The bit-plane index for descending FROM `level` to `level+1`
        // is `level` (matches the convention used in the single-catalog cascade).
        let descent_level = level;
        let n_children = 1usize << D;
        let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        for &i in idx_d {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_d[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_d[child_id].push(i);
        }
        for &i in idx_r {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_r[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_r[child_id].push(i);
        }

        for c in 0..n_children {
            if !buckets_d[c].is_empty() || !buckets_r[c].is_empty() {
                self.recurse_pointlist(&buckets_d[c], &buckets_r[c], level + 1,
                                        cum_dd, cum_rr, cum_dr,
                                        nne_d, nne_r, nne_both);
            }
        }
    }

    // ========================================================================
    // Walker — single source of truth for cascade traversal
    // ========================================================================

    /// Walk every non-empty cell of the cascade once, calling the
    /// provided visitor at each cell. This is the recommended way to
    /// implement statistics that need to consume the cascade.
    ///
    /// The visitor receives:
    /// - [`CascadeVisitor::enter_cell`] on every cell as it is entered
    /// - [`CascadeVisitor::leaf_cell`] on cells at the finest level
    ///   (`level == l_max`)
    /// - [`CascadeVisitor::after_children`] on each parent cell after
    ///   all 2^D children have been visited, with a slice giving each
    ///   child's [`CellVisit`] in canonical axis-pattern order
    ///
    /// Multiple statistics can be computed in a single walk by composing
    /// visitors via tuples or [`CompositeVisitor`].
    ///
    /// ## Example
    ///
    /// ```ignore
    /// use morton_cascade::cascade_visitor::{CascadeVisitor, CellVisit, FootprintCutoff};
    ///
    /// struct CountInFootprint(u64);
    /// impl CascadeVisitor<3> for CountInFootprint {
    ///     fn enter_cell(&mut self, cell: &CellVisit<3>) {
    ///         if cell.in_footprint(&FootprintCutoff::ANY_RANDOM) {
    ///             self.0 += 1;
    ///         }
    ///     }
    /// }
    /// let mut v = CountInFootprint(0);
    /// pair.walk(&mut v);
    /// ```
    pub fn walk<V: crate::cascade_visitor::CascadeVisitor<D>>(&self, visitor: &mut V) {
        let n_d = self.n_d();
        let n_r = self.n_r();
        if n_d == 0 && n_r == 0 { return; }

        // α policy depends on boundary mode + presence of randoms.
        //   Isolated (always) and Periodic with non-empty randoms:
        //     α = ΣW_d / ΣW_r (catalog ratio).
        //   Periodic with empty randoms:
        //     α = ΣW_d (taking V_box as the unit; each cell at level l is
        //     synthesized to have W_r = 2^(-D*l) so δ = sw_d/(α·sw_r) − 1
        //     reduces to (2^(D*l)·sw_d/ΣW_d) − 1, the box-mean density
        //     contrast).
        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        let alpha = match (self.boundary_mode, n_r > 0) {
            (BoundaryMode::Periodic, false) => total_w_d,
            _ => if total_w_r > 0.0 { total_w_d / total_w_r } else { 0.0 },
        };

        // Build root membership: all bits set up to the actual count
        let mut root_d = vec![u64::MAX; self.n_words_d.max(1)];
        if n_d == 0 {
            for w in root_d.iter_mut() { *w = 0; }
        } else if n_d % 64 != 0 {
            let last = n_d % 64;
            root_d[self.n_words_d - 1] = (1u64 << last) - 1;
        }
        let mut root_r = vec![u64::MAX; self.n_words_r.max(1)];
        if n_r == 0 {
            for w in root_r.iter_mut() { *w = 0; }
        } else if n_r % 64 != 0 {
            let last = n_r % 64;
            root_r[self.n_words_r - 1] = (1u64 << last) - 1;
        }
        let root_d_view: &[u64] = if n_d == 0 { &[] } else { &root_d[..self.n_words_d] };
        let root_r_view: &[u64] = if n_r == 0 { &[] } else { &root_r[..self.n_words_r] };

        // Recurse from level 0, cell_id 0
        let _ = self.walk_recurse(root_d_view, root_r_view, 0, 0, alpha, visitor);
    }

    /// True iff this cascade should synthesize per-cell W_r from the box
    /// geometry instead of reading it from the (empty) randoms catalog.
    /// Defined as `boundary_mode == Periodic && N_r == 0`.
    #[inline]
    pub(crate) fn synthesizes_randoms_from_box(&self) -> bool {
        self.boundary_mode == BoundaryMode::Periodic && self.n_r() == 0
    }

    /// Make a CellVisit from cell sums and metadata. Used by the walker.
    ///
    /// In Periodic-mode-with-no-randoms, the actually-popcount'd
    /// `(n_r, sw_r, sw2_r)` are zero and we replace them with the synthetic
    /// per-cell volume W_r = 2^(-D*level) so that `cell.delta(footprint)`
    /// returns the correct box-mean density contrast for the data field.
    #[inline]
    fn make_cell_visit(
        &self, n_d: u64, sw_d: f64, sw2_d: f64,
        n_r: u64, sw_r: f64, sw2_r: f64,
        level: usize, cell_id: u64, alpha: f64,
    ) -> crate::cascade_visitor::CellVisit<D> {
        use crate::cascade_visitor::{CellVisit, CatalogCell};
        let max_eff = self.data.range.max_supported_l_max() as usize;
        let cell_side_trimmed = if level <= max_eff {
            1u64 << (max_eff - level)
        } else { 1 };

        // Periodic + no randoms: synthesize per-cell W_r from box geometry.
        //
        // The synthesis only fills sw_r and sw_r_sq with the cell-volume
        // expectation (in V_box=1 units). The synthesized count stays 0 so
        // the walker's `is_empty` short-circuit still prunes empty cells —
        // periodic-mode visitors then add an analytic correction at
        // finalization time for the unvisited (zero-data) cell volume.
        let (n_r_eff, sw_r_eff, sw2_r_eff) = if self.synthesizes_randoms_from_box() {
            // Volume of one level-l cell, in units where V_box = 1: 2^(-D*l).
            // Level 0 → 1.0 (whole box). At depths so deep this underflows
            // f64, fall back to MIN_POSITIVE so footprint tests with tiny
            // w_r_min still classify the cell as in-footprint.
            let depth_bits = D.saturating_mul(level);
            let cell_vol: f64 = if depth_bits >= 1024 {
                f64::MIN_POSITIVE
            } else {
                2.0_f64.powi(-(depth_bits as i32))
            };
            (0u64, cell_vol, cell_vol * cell_vol)
        } else {
            (n_r, sw_r, sw2_r)
        };

        CellVisit::<D> {
            level,
            cell_side_trimmed,
            cell_id,
            data: CatalogCell { count: n_d, sum_w: sw_d, sum_w_sq: sw2_d },
            randoms: CatalogCell { count: n_r_eff, sum_w: sw_r_eff, sum_w_sq: sw2_r_eff },
            alpha,
        }
    }

    /// Bit-vec walk recursion. Returns the cell's CellVisit so the
    /// parent can include it in `children` for after_children dispatch.
    fn walk_recurse<V: crate::cascade_visitor::CascadeVisitor<D>>(
        &self, mem_d: &[u64], mem_r: &[u64],
        level: usize, cell_id: u64, alpha: f64,
        visitor: &mut V,
    ) -> crate::cascade_visitor::CellVisit<D> {
        use crate::cascade_visitor::CellVisit;

        let (n_d, sw_d, sw2_d) = self.cell_sums_d(mem_d);
        let (n_r, sw_r, sw2_r) = self.cell_sums_r(mem_r);
        let cell = self.make_cell_visit(n_d, sw_d, sw2_d, n_r, sw_r, sw2_r,
                                        level, cell_id, alpha);
        if cell.is_empty() {
            return cell;
        }
        visitor.enter_cell(&cell);
        if level >= self.l_max {
            visitor.leaf_cell(&cell);
            return cell;
        }

        // Decide pointlist vs bitvec descent
        let thresh = self.crossover_threshold as u64;
        let n_children = 1usize << D;
        let mut children: Vec<CellVisit<D>> = Vec::with_capacity(n_children);

        if n_d <= thresh && n_r <= thresh {
            let idx_d = mem_to_indices(mem_d, n_d as usize);
            let idx_r = mem_to_indices(mem_r, n_r as usize);
            let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            for &i in &idx_d {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_d[child_id].push(i);
            }
            for &i in &idx_r {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_r[child_id].push(i);
            }
            for c in 0..n_children {
                let child_cell_id = (cell_id << D) | (c as u64);
                let child = self.walk_recurse_pointlist(
                    &buckets_d[c], &buckets_r[c], level + 1, child_cell_id, alpha, visitor);
                children.push(child);
            }
        } else {
            // Bit-vec descent
            let nwd = self.n_words_d;
            let nwr = self.n_words_r;
            let mut up_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
            let mut lo_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
            for d in 0..D {
                for w in 0..nwd {
                    let bp = self.bit_planes_d[d][level][w];
                    up_d[d][w] = mem_d[w] & bp;
                    lo_d[d][w] = mem_d[w] & !bp;
                }
            }
            let mut up_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
            let mut lo_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
            for d in 0..D {
                for w in 0..nwr {
                    let bp = self.bit_planes_r[d][level][w];
                    up_r[d][w] = mem_r[w] & bp;
                    lo_r[d][w] = mem_r[w] & !bp;
                }
            }
            let mut child_d = vec![0u64; nwd];
            let mut child_r = vec![0u64; nwr];
            for c in 0..(1u32 << D) {
                let b0 = (c & 1) != 0;
                child_d.copy_from_slice(if b0 { &up_d[0] } else { &lo_d[0] });
                child_r.copy_from_slice(if b0 { &up_r[0] } else { &lo_r[0] });
                for d in 1..D {
                    let bd = (c >> d) & 1 != 0;
                    let half_d = if bd { &up_d[d] } else { &lo_d[d] };
                    let half_r = if bd { &up_r[d] } else { &lo_r[d] };
                    for w in 0..nwd { child_d[w] &= half_d[w]; }
                    for w in 0..nwr { child_r[w] &= half_r[w]; }
                }
                let child_cell_id = (cell_id << D) | (c as u64);
                let child = self.walk_recurse(
                    &child_d, &child_r, level + 1, child_cell_id, alpha, visitor);
                children.push(child);
            }
        }

        // Defensive: if any child slot is missing (shouldn't happen given the
        // loop above, but to avoid panics in dispatch we ensure 2^D entries)
        while children.len() < n_children {
            children.push(self.make_cell_visit(0, 0.0, 0.0, 0, 0.0, 0.0,
                                                level + 1, 0, alpha));
        }

        visitor.after_children(&cell, &children);
        cell
    }

    /// Point-list walk recursion.
    fn walk_recurse_pointlist<V: crate::cascade_visitor::CascadeVisitor<D>>(
        &self, idx_d: &[u32], idx_r: &[u32],
        level: usize, cell_id: u64, alpha: f64,
        visitor: &mut V,
    ) -> crate::cascade_visitor::CellVisit<D> {
        use crate::cascade_visitor::CellVisit;

        let (n_d, sw_d, sw2_d) = self.cell_sums_d_list(idx_d);
        let (n_r, sw_r, sw2_r) = self.cell_sums_r_list(idx_r);
        let cell = self.make_cell_visit(n_d, sw_d, sw2_d, n_r, sw_r, sw2_r,
                                        level, cell_id, alpha);
        if cell.is_empty() {
            return cell;
        }
        visitor.enter_cell(&cell);
        if level >= self.l_max {
            visitor.leaf_cell(&cell);
            return cell;
        }

        // Bucket points into 2^D children
        let descent_level = level;
        let n_children = 1usize << D;
        let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        for &i in idx_d {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_d[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_d[child_id].push(i);
        }
        for &i in idx_r {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_r[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_r[child_id].push(i);
        }

        let mut children: Vec<CellVisit<D>> = Vec::with_capacity(n_children);
        for c in 0..n_children {
            let child_cell_id = (cell_id << D) | (c as u64);
            let child = self.walk_recurse_pointlist(
                &buckets_d[c], &buckets_r[c], level + 1, child_cell_id, alpha, visitor);
            children.push(child);
        }
        while children.len() < n_children {
            children.push(self.make_cell_visit(0, 0.0, 0.0, 0, 0.0, 0.0,
                                                level + 1, 0, alpha));
        }

        visitor.after_children(&cell, &children);
        cell
    }

    /// Convert per-level cumulative DD/DR/RR into per-shell bins and apply the
    /// Landy-Szalay estimator
    ///
    /// ```text
    ///     ξ(r) = (DD/N_DD - 2 DR/N_DR + RR/N_RR) / (RR/N_RR)
    /// ```
    ///
    /// where the normalizations are
    ///   N_DD = W_D * (W_D - 1) / 2  (using sum_w_d for weighted catalogs;
    ///                               for unit weights this is N_d (N_d - 1)/2)
    ///   N_RR = W_R * (W_R - 1) / 2
    ///   N_DR = W_D * W_R
    ///
    /// For the deepest level (no level beyond it) the shell holds whatever
    /// pairs share the finest cell.
    pub fn xi_landy_szalay(&self, stats: &[PairLevelStats]) -> Vec<XiShell> {
        let max_eff = self.data.range.max_supported_l_max() as usize;
        let n_levels = stats.len();

        let wd = self.sum_w_d();
        let wr = self.sum_w_r();
        let norm_dd = if wd > 1.0 { wd * (wd - 1.0) / 2.0 } else { 0.0 };
        let norm_rr = if wr > 1.0 { wr * (wr - 1.0) / 2.0 } else { 0.0 };
        let norm_dr = wd * wr;

        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else {
                1.0
            };
            let r_outer = if l == 0 {
                (1u64 << max_eff) as f64
            } else if (l - 1) <= max_eff {
                (1u64 << (max_eff - (l - 1))) as f64
            } else {
                1.0
            };
            let (dd, rr, dr) = if l + 1 < n_levels {
                (
                    stats[l].cumulative_dd - stats[l + 1].cumulative_dd,
                    stats[l].cumulative_rr - stats[l + 1].cumulative_rr,
                    stats[l].cumulative_dr - stats[l + 1].cumulative_dr,
                )
            } else {
                (stats[l].cumulative_dd, stats[l].cumulative_rr, stats[l].cumulative_dr)
            };
            let xi_ls = if norm_rr > 0.0 && rr > 0.0 {
                let dd_n = if norm_dd > 0.0 { dd / norm_dd } else { 0.0 };
                let dr_n = if norm_dr > 0.0 { dr / norm_dr } else { 0.0 };
                let rr_n = rr / norm_rr;
                (dd_n - 2.0 * dr_n + rr_n) / rr_n
            } else {
                f64::NAN
            };
            out.push(XiShell {
                level: l,
                cell_side_trimmed: side_l,
                r_outer_trimmed: r_outer,
                r_inner_trimmed: side_l,
                dd, rr, dr,
                xi_ls,
            });
        }
        out
    }

    /// Walk the cascade and accumulate per-cell density-contrast moments and
    /// (optionally) histogram of δ = W_d / (α · W_r) − 1. See
    /// [`DensityFieldStats`] for the full definition.
    ///
    /// `cfg.w_r_min` excludes cells whose summed random weight is at or
    /// below the threshold (these are "outside the survey footprint" cells).
    /// Pass 0.0 to keep every visited cell.
    ///
    /// Implementation: delegates to [`Self::analyze_field_stats_v2`], which
    /// uses the cascade walker. Numerically identical to the legacy
    /// `analyze_field_stats_legacy` (kept as a parity-test reference).
    pub fn analyze_field_stats(&self, cfg: &FieldStatsConfig) -> Vec<DensityFieldStats> {
        self.analyze_field_stats_v2(cfg)
    }

    /// Legacy (pre-walker) field-stats implementation. Kept temporarily as
    /// a parity-test reference; will be removed once the visitor migration
    /// is fully validated.
    ///
    /// New code should use [`Self::analyze_field_stats`] directly.
    #[doc(hidden)]
    pub fn analyze_field_stats_legacy(&self, cfg: &FieldStatsConfig) -> Vec<DensityFieldStats> {
        let n_d = self.n_d();
        let n_r = self.n_r();
        let n_levels = self.l_max + 1;

        // Global α = ΣW_d / ΣW_r. Cells outside footprint (W_r ≤ w_r_min)
        // are excluded from moments but α uses the GLOBAL sums so that
        // <δ>_W_r ≈ 0 holds on the full surveyed region.
        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 {
            // Pathological: no random-catalog weight at all. Return
            // zero-filled stats per level rather than NaN.
            return (0..n_levels).map(|l| empty_density_stats::<D>(self, l, cfg)).collect();
        }
        let alpha = total_w_d / total_w_r;

        // Accumulators per level. We use raw moments of (δ * W_r) and then
        // form central moments at the end. Keeping running raw moments is
        // numerically fine as long as we centre at the (small) sample mean.
        // For very large W-totals one would want Welford's algorithm; here
        // the variance is small (cosmological δ ~ O(0.1-1)) and W_r is
        // bounded so straightforward sums are stable.
        let mut sum_w_r = vec![0.0f64; n_levels];
        let mut sum_w_r_delta1 = vec![0.0f64; n_levels];
        let mut sum_w_r_delta2 = vec![0.0f64; n_levels];
        let mut sum_w_r_delta3 = vec![0.0f64; n_levels];
        let mut sum_w_r_delta4 = vec![0.0f64; n_levels];
        let mut n_active = vec![0u64; n_levels];
        let mut min_delta = vec![f64::INFINITY; n_levels];
        let mut max_delta = vec![f64::NEG_INFINITY; n_levels];
        // Diagnostic: data-bearing cells with random weight at-or-below w_r_min.
        let mut n_outside = vec![0u64; n_levels];
        let mut sw_d_outside = vec![0.0f64; n_levels];

        // Histogram bins (uniform in log10(1+δ))
        let mut hist: Vec<Vec<f64>> = vec![vec![0.0f64; cfg.hist_bins]; n_levels];
        let mut hist_under = vec![0.0f64; n_levels];
        let mut hist_over = vec![0.0f64; n_levels];

        // Bit-vector or empty: walk the cascade.
        if n_d == 0 && n_r == 0 {
            return (0..n_levels).map(|l| empty_density_stats::<D>(self, l, cfg)).collect();
        }

        let mut root_d = vec![u64::MAX; self.n_words_d.max(1)];
        if n_d == 0 {
            for w in root_d.iter_mut() { *w = 0; }
        } else if n_d % 64 != 0 {
            let last = n_d % 64;
            root_d[self.n_words_d - 1] = (1u64 << last) - 1;
        }
        let mut root_r = vec![u64::MAX; self.n_words_r.max(1)];
        if n_r == 0 {
            for w in root_r.iter_mut() { *w = 0; }
        } else if n_r % 64 != 0 {
            let last = n_r % 64;
            root_r[self.n_words_r - 1] = (1u64 << last) - 1;
        }
        let root_d_view: &[u64] = if n_d == 0 { &[] } else { &root_d[..self.n_words_d] };
        let root_r_view: &[u64] = if n_r == 0 { &[] } else { &root_r[..self.n_words_r] };

        self.field_recurse(
            root_d_view, root_r_view, 0, alpha, cfg,
            &mut sum_w_r, &mut sum_w_r_delta1, &mut sum_w_r_delta2,
            &mut sum_w_r_delta3, &mut sum_w_r_delta4,
            &mut n_active, &mut min_delta, &mut max_delta,
            &mut n_outside, &mut sw_d_outside,
            &mut hist, &mut hist_under, &mut hist_over,
        );

        // Build per-level output. For each level, central moments are computed
        // from the raw W_r-weighted moments using the standard transform:
        //   <(δ-μ)^k>_W = sum_W_r δ^k / sum_W_r computed about μ = sum_W_r δ / sum_W_r.
        // We compute all moments in one pass by accumulating raw E[δ], E[δ²],
        // E[δ³], E[δ⁴] (W_r-weighted) and then transforming.
        let mut hist_edges = Vec::new();
        if cfg.hist_bins > 0 {
            hist_edges = (0..=cfg.hist_bins).map(|i| {
                cfg.hist_log_min
                    + (cfg.hist_log_max - cfg.hist_log_min) * (i as f64) / (cfg.hist_bins as f64)
            }).collect();
        }

        let mut out: Vec<DensityFieldStats> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let total_w = sum_w_r[l];
            let (mean, var, m3, m4) = if total_w > 0.0 {
                let m1 = sum_w_r_delta1[l] / total_w;
                let m2_raw = sum_w_r_delta2[l] / total_w;
                let m3_raw = sum_w_r_delta3[l] / total_w;
                let m4_raw = sum_w_r_delta4[l] / total_w;
                // Standard raw -> central moment transform
                let var = m2_raw - m1 * m1;
                let m3c = m3_raw - 3.0 * m1 * m2_raw + 2.0 * m1 * m1 * m1;
                let m4c = m4_raw - 4.0 * m1 * m3_raw + 6.0 * m1 * m1 * m2_raw - 3.0 * m1 * m1 * m1 * m1;
                (m1, var.max(0.0), m3c, m4c)
            } else { (0.0, 0.0, 0.0, 0.0) };
            let s3 = if var > 0.0 { m3 / (var * var) } else { 0.0 };

            // Convert histogram counts to W_r-normalized density
            let hist_density: Vec<f64> = if total_w > 0.0 && cfg.hist_bins > 0 {
                hist[l].iter().map(|&c| c / total_w).collect()
            } else {
                vec![0.0; cfg.hist_bins]
            };
            let hist_under_n = if total_w > 0.0 { hist_under[l] / total_w } else { 0.0 };
            let hist_over_n = if total_w > 0.0 { hist_over[l] / total_w } else { 0.0 };

            // Cube side at this level.
            let max_eff = self.data.range.max_supported_l_max() as usize;
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else { 1.0 };

            out.push(DensityFieldStats {
                level: l,
                cell_side_trimmed: side_l,
                n_cells_active: n_active[l],
                sum_w_r_active: total_w,
                mean_delta: mean,
                var_delta: var,
                m3_delta: m3,
                m4_delta: m4,
                min_delta: if n_active[l] > 0 { min_delta[l] } else { 0.0 },
                max_delta: if n_active[l] > 0 { max_delta[l] } else { 0.0 },
                s3_delta: s3,
                n_cells_data_outside: n_outside[l],
                sum_w_d_outside: sw_d_outside[l],
                hist_bin_edges: hist_edges.clone(),
                hist_density,
                hist_underflow_w_r: hist_under_n,
                hist_overflow_w_r: hist_over_n,
                raw_sum_w_r_delta_pow: [
                    sum_w_r_delta1[l],
                    sum_w_r_delta2[l],
                    sum_w_r_delta3[l],
                    sum_w_r_delta4[l],
                ],
            });
        }
        out
    }

    /// Cascade walker for `analyze_field_stats`. Single-pass: at every
    /// non-empty visited cell, accumulate W_r-weighted moments of δ and
    /// optionally the histogram bin.
    fn field_recurse(
        &self,
        mem_d: &[u64],
        mem_r: &[u64],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        sum_w_r: &mut [f64],
        sum_w_r_delta1: &mut [f64],
        sum_w_r_delta2: &mut [f64],
        sum_w_r_delta3: &mut [f64],
        sum_w_r_delta4: &mut [f64],
        n_active: &mut [u64],
        min_delta: &mut [f64],
        max_delta: &mut [f64],
        n_outside: &mut [u64],
        sw_d_outside: &mut [f64],
        hist: &mut [Vec<f64>],
        hist_under: &mut [f64],
        hist_over: &mut [f64],
    ) {
        let (n_d, sw_d, _) = self.cell_sums_d(mem_d);
        let (n_r, sw_r, _) = self.cell_sums_r(mem_r);
        if n_d == 0 && n_r == 0 { return; }

        // Active cell test: include this cell in stats iff its random weight
        // exceeds the footprint threshold. Cells that fail are still
        // descended into (their children might pass).
        if sw_r > cfg.w_r_min {
            let delta = if alpha > 0.0 && sw_r > 0.0 {
                sw_d / (alpha * sw_r) - 1.0
            } else { -1.0 };  // empty random cell shouldn't reach here, but guard
            sum_w_r[level] += sw_r;
            sum_w_r_delta1[level] += sw_r * delta;
            let d2 = delta * delta;
            sum_w_r_delta2[level] += sw_r * d2;
            sum_w_r_delta3[level] += sw_r * d2 * delta;
            sum_w_r_delta4[level] += sw_r * d2 * d2;
            n_active[level] += 1;
            if delta < min_delta[level] { min_delta[level] = delta; }
            if delta > max_delta[level] { max_delta[level] = delta; }

            // Histogram bin in log10(1+δ)
            if cfg.hist_bins > 0 {
                let one_plus = 1.0 + delta;
                if one_plus > 0.0 {
                    let log_v = one_plus.log10();
                    if log_v < cfg.hist_log_min {
                        hist_under[level] += sw_r;
                    } else if log_v >= cfg.hist_log_max {
                        hist_over[level] += sw_r;
                    } else {
                        let frac = (log_v - cfg.hist_log_min)
                            / (cfg.hist_log_max - cfg.hist_log_min);
                        let bin = (frac * cfg.hist_bins as f64) as usize;
                        let bin = bin.min(cfg.hist_bins - 1);
                        hist[level][bin] += sw_r;
                    }
                } else {
                    // δ = -1 exactly (W_d = 0 but W_r > w_r_min): underflow
                    hist_under[level] += sw_r;
                }
            }
        } else if n_d > 0 {
            // Diagnostic: data is here, but the random catalog says we're
            // outside the surveyed footprint. Track count and W_d so users
            // can see this as a catalog-quality signal.
            n_outside[level] += 1;
            sw_d_outside[level] += sw_d;
        }

        if level >= self.l_max { return; }

        // Crossover and descent: same logic as `recurse`, structurally.
        let thresh = self.crossover_threshold as u64;
        if n_d <= thresh && n_r <= thresh {
            let idx_d = mem_to_indices(mem_d, n_d as usize);
            let idx_r = mem_to_indices(mem_r, n_r as usize);
            let n_children = 1usize << D;
            let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
            for &i in &idx_d {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_d[child_id].push(i);
            }
            for &i in &idx_r {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..D {
                    let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_r[child_id].push(i);
            }
            for c in 0..n_children {
                if !buckets_d[c].is_empty() || !buckets_r[c].is_empty() {
                    self.field_recurse_pointlist(
                        &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg,
                        sum_w_r, sum_w_r_delta1, sum_w_r_delta2,
                        sum_w_r_delta3, sum_w_r_delta4,
                        n_active, min_delta, max_delta,
                        n_outside, sw_d_outside,
                        hist, hist_under, hist_over,
                    );
                }
            }
            return;
        }

        // Bit-vec descent
        let nwd = self.n_words_d;
        let nwr = self.n_words_r;
        let mut up_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
        let mut lo_d: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwd]);
        for d in 0..D {
            for w in 0..nwd {
                let bp = self.bit_planes_d[d][level][w];
                up_d[d][w] = mem_d[w] & bp;
                lo_d[d][w] = mem_d[w] & !bp;
            }
        }
        let mut up_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
        let mut lo_r: [Vec<u64>; D] = std::array::from_fn(|_| vec![0u64; nwr]);
        for d in 0..D {
            for w in 0..nwr {
                let bp = self.bit_planes_r[d][level][w];
                up_r[d][w] = mem_r[w] & bp;
                lo_r[d][w] = mem_r[w] & !bp;
            }
        }
        let mut child_d = vec![0u64; nwd];
        let mut child_r = vec![0u64; nwr];
        for child in 0..(1u32 << D) {
            let b0 = (child & 1) != 0;
            child_d.copy_from_slice(if b0 { &up_d[0] } else { &lo_d[0] });
            child_r.copy_from_slice(if b0 { &up_r[0] } else { &lo_r[0] });
            for d in 1..D {
                let bd = (child >> d) & 1 != 0;
                let half_d = if bd { &up_d[d] } else { &lo_d[d] };
                let half_r = if bd { &up_r[d] } else { &lo_r[d] };
                for w in 0..nwd { child_d[w] &= half_d[w]; }
                for w in 0..nwr { child_r[w] &= half_r[w]; }
            }
            self.field_recurse(
                &child_d, &child_r, level + 1, alpha, cfg,
                sum_w_r, sum_w_r_delta1, sum_w_r_delta2,
                sum_w_r_delta3, sum_w_r_delta4,
                n_active, min_delta, max_delta,
                n_outside, sw_d_outside,
                hist, hist_under, hist_over,
            );
        }
    }

    fn field_recurse_pointlist(
        &self,
        idx_d: &[u32],
        idx_r: &[u32],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        sum_w_r: &mut [f64],
        sum_w_r_delta1: &mut [f64],
        sum_w_r_delta2: &mut [f64],
        sum_w_r_delta3: &mut [f64],
        sum_w_r_delta4: &mut [f64],
        n_active: &mut [u64],
        min_delta: &mut [f64],
        max_delta: &mut [f64],
        n_outside: &mut [u64],
        sw_d_outside: &mut [f64],
        hist: &mut [Vec<f64>],
        hist_under: &mut [f64],
        hist_over: &mut [f64],
    ) {
        let (n_d, sw_d, _) = self.cell_sums_d_list(idx_d);
        let (n_r, sw_r, _) = self.cell_sums_r_list(idx_r);
        if n_d == 0 && n_r == 0 { return; }

        if sw_r > cfg.w_r_min {
            let delta = if alpha > 0.0 && sw_r > 0.0 {
                sw_d / (alpha * sw_r) - 1.0
            } else { -1.0 };
            sum_w_r[level] += sw_r;
            sum_w_r_delta1[level] += sw_r * delta;
            let d2 = delta * delta;
            sum_w_r_delta2[level] += sw_r * d2;
            sum_w_r_delta3[level] += sw_r * d2 * delta;
            sum_w_r_delta4[level] += sw_r * d2 * d2;
            n_active[level] += 1;
            if delta < min_delta[level] { min_delta[level] = delta; }
            if delta > max_delta[level] { max_delta[level] = delta; }

            if cfg.hist_bins > 0 {
                let one_plus = 1.0 + delta;
                if one_plus > 0.0 {
                    let log_v = one_plus.log10();
                    if log_v < cfg.hist_log_min { hist_under[level] += sw_r; }
                    else if log_v >= cfg.hist_log_max { hist_over[level] += sw_r; }
                    else {
                        let frac = (log_v - cfg.hist_log_min)
                            / (cfg.hist_log_max - cfg.hist_log_min);
                        let bin = (frac * cfg.hist_bins as f64) as usize;
                        let bin = bin.min(cfg.hist_bins - 1);
                        hist[level][bin] += sw_r;
                    }
                } else { hist_under[level] += sw_r; }
            }
        } else if n_d > 0 {
            n_outside[level] += 1;
            sw_d_outside[level] += sw_d;
        }

        if level >= self.l_max { return; }

        let descent_level = level;
        let n_children = 1usize << D;
        let mut buckets_d: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        let mut buckets_r: Vec<Vec<u32>> = (0..n_children).map(|_| Vec::new()).collect();
        for &i in idx_d {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_d[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_d[child_id].push(i);
        }
        for &i in idx_r {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..D {
                let bit = (self.bit_planes_r[d][descent_level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_r[child_id].push(i);
        }
        for c in 0..n_children {
            if !buckets_d[c].is_empty() || !buckets_r[c].is_empty() {
                self.field_recurse_pointlist(
                    &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg,
                    sum_w_r, sum_w_r_delta1, sum_w_r_delta2,
                    sum_w_r_delta3, sum_w_r_delta4,
                    n_active, min_delta, max_delta,
                    n_outside, sw_d_outside,
                    hist, hist_under, hist_over,
                );
            }
        }
    }

    /// Compute axis-aligned cell-wavelet anisotropy statistics at every
    /// dyadic scale. This walks the cascade once, computes the 8 child
    /// cell-counts of each parent cell, and decomposes them via the Haar
    /// wavelet transform into 1 isotropic average plus 7 directional
    /// detail coefficients.
    ///
    /// The 7 detail coefficients factor by axis pattern:
    ///   - 3 axis-aligned: x-only, y-only, z-only differences
    ///   - 3 face-diagonal: yz, xz, xy plane patterns
    ///   - 1 body-diagonal: full 3D xor
    ///
    /// We accumulate W_r-weighted variance of each coefficient per level.
    /// The quadrupole-like moment Q_2 = <w_z²> − ½(<w_x²> + <w_y²>)
    /// directly probes line-of-sight (z) versus transverse anisotropy.
    ///
    /// Only 3D is supported. For other D this returns an empty vector.
    ///
    /// **Relation to wavelet scattering transforms:** This is structurally
    /// **first-order Haar scattering** (Mallat 2012; Bruna & Mallat 2013) on
    /// the cell-count field, applied along the cascade's dyadic hierarchy
    /// with axis-aligned Haar wavelets and a `|·|²` summary nonlinearity in
    /// place of the canonical `|·|`. Second-order scattering coefficients
    /// would arise from recursing the cascade on the derived `|w_e|` field
    /// — same machinery, different input.
    ///
    /// **Works in any dimension D ≥ 1.** Delegates to the visitor-based
    /// implementation [`Self::analyze_anisotropy_v2`].
    pub fn analyze_anisotropy(&self, cfg: &FieldStatsConfig) -> Vec<AnisotropyStats> {
        self.analyze_anisotropy_v2(cfg)
    }

    /// Legacy 3D-only anisotropy implementation, kept for parity testing
    /// during the visitor refactor. Returns an empty vector for D != 3.
    /// New code should use [`Self::analyze_anisotropy`] which works in
    /// any dimension and uses the shared cascade walker.
    #[doc(hidden)]
    pub fn analyze_anisotropy_legacy(&self, cfg: &FieldStatsConfig) -> Vec<AnisotropyStats> {
        if D != 3 {
            return Vec::new();
        }
        let n_d = self.n_d();
        let n_r = self.n_r();
        let n_levels = self.l_max + 1;

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 {
            return (0..n_levels).map(|l| empty_anisotropy(self, l)).collect();
        }
        let alpha = total_w_d / total_w_r;

        // Per-level accumulators. For each parent cell we accumulate W_r-weighted
        // squared wavelet coefficients in 7 directions. The 7 directions are
        // indexed by their bit pattern e in 1..8 (0 is the trivial all-positive
        // average that recovers the parent count).
        // Layout: dir_index[pattern] in 1..8.
        let mut sum_w_r_parents = vec![0.0f64; n_levels];
        let mut n_parents = vec![0u64; n_levels];
        // For each level and each of the 7 wavelet directions, accumulate the
        // W_r-weighted mean of w_pattern²:
        let mut sum_wr_w2 = vec![[0.0f64; 8]; n_levels];

        if n_d == 0 && n_r == 0 {
            return (0..n_levels).map(|l| empty_anisotropy(self, l)).collect();
        }

        // Build root membership vectors
        let mut root_d = vec![u64::MAX; self.n_words_d.max(1)];
        if n_d == 0 {
            for w in root_d.iter_mut() { *w = 0; }
        } else if n_d % 64 != 0 {
            let last = n_d % 64;
            root_d[self.n_words_d - 1] = (1u64 << last) - 1;
        }
        let mut root_r = vec![u64::MAX; self.n_words_r.max(1)];
        if n_r == 0 {
            for w in root_r.iter_mut() { *w = 0; }
        } else if n_r % 64 != 0 {
            let last = n_r % 64;
            root_r[self.n_words_r - 1] = (1u64 << last) - 1;
        }
        let root_d_view: &[u64] = if n_d == 0 { &[] } else { &root_d[..self.n_words_d] };
        let root_r_view: &[u64] = if n_r == 0 { &[] } else { &root_r[..self.n_words_r] };

        // Walk and accumulate
        self.aniso_recurse(
            root_d_view, root_r_view, 0, alpha, cfg,
            &mut sum_w_r_parents, &mut n_parents, &mut sum_wr_w2,
        );

        // Build per-level output
        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let total_w = sum_w_r_parents[l];
            // Divide accumulators to get W_r-weighted means
            let mut means = [0.0f64; 8];
            if total_w > 0.0 {
                for k in 1..8 {
                    means[k] = sum_wr_w2[l][k] / total_w;
                }
            }
            // Axis-aligned directions: pattern 1=x-only (001), 2=y-only (010), 4=z-only (100)
            let w2_x = means[1];
            let w2_y = means[2];
            let w2_z = means[4];
            // Face-diagonals: pattern 3=xy (011), 5=xz (101), 6=yz (110)
            // Body-diagonal: pattern 7=xyz (111)
            // (Now flowed directly through means_by_pattern.)

            // Quadrupole moment along z axis (LoS by convention)
            let q2 = w2_z - 0.5 * (w2_x + w2_y);
            let total_axis = w2_x + w2_y + w2_z;
            let reduced = if total_axis > 0.0 { q2 / (total_axis / 3.0) } else { 0.0 };

            let max_eff = self.data.range.max_supported_l_max() as usize;
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else { 1.0 };

            // Build the D-generic output. For D=3 (this legacy path
            // only runs for D=3) means_by_pattern is means[0..8]
            // and means_axis is [w_x², w_y², w_z²].
            let mut means_by_pattern = vec![0.0f64; 8];
            for k in 1..8 { means_by_pattern[k] = means[k]; }
            let means_axis = vec![w2_x, w2_y, w2_z];

            out.push(AnisotropyStats {
                level: l,
                cell_side_trimmed: side_l,
                n_parents: n_parents[l],
                sum_w_r_parents: total_w,
                mean_w_squared_by_pattern: means_by_pattern,
                mean_w_squared_axis: means_axis,
                quadrupole_los: q2,
                reduced_quadrupole_los: reduced,
            });
        }
        out
    }

    /// Recursive walker for `analyze_anisotropy`. Returns this cell's
    /// (W_d, W_r) so the caller can use them to compute child-deltas.
    /// 3D only — assumes D == 3 (caller has guarded).
    fn aniso_recurse(
        &self,
        mem_d: &[u64],
        mem_r: &[u64],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        sum_w_r_parents: &mut [f64],
        n_parents: &mut [u64],
        sum_wr_w2: &mut [[f64; 8]],
    ) -> (f64, f64) {
        let (n_d, sw_d, _) = self.cell_sums_d(mem_d);
        let (n_r, sw_r, _) = self.cell_sums_r(mem_r);
        if n_d == 0 && n_r == 0 { return (0.0, 0.0); }

        // Leaf cell: cannot decompose further. Return its W_d, W_r.
        if level >= self.l_max {
            return (sw_d, sw_r);
        }

        // Get child cell sums by descending one level. We need (W_d, W_r) for
        // each of the 8 children to compute wavelet coefficients of δ at THIS
        // (parent) level.
        let mut child_w_d = [0.0f64; 8];
        let mut child_w_r = [0.0f64; 8];

        let thresh = self.crossover_threshold as u64;
        let use_pointlist = n_d <= thresh && n_r <= thresh;

        if use_pointlist {
            // Convert to point-list for the descent and use point-list recursion
            let idx_d = mem_to_indices(mem_d, n_d as usize);
            let idx_r = mem_to_indices(mem_r, n_r as usize);
            let mut buckets_d: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            let mut buckets_r: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            for &i in &idx_d {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..3 {
                    let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_d[child_id].push(i);
            }
            for &i in &idx_r {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..3 {
                    let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_r[child_id].push(i);
            }
            for c in 0..8 {
                let (cwd, cwr) = self.aniso_recurse_pointlist(
                    &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg,
                    sum_w_r_parents, n_parents, sum_wr_w2,
                );
                child_w_d[c] = cwd;
                child_w_r[c] = cwr;
            }
        } else {
            // Bit-vec descent
            let nwd = self.n_words_d;
            let nwr = self.n_words_r;
            let mut up_d: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwd]);
            let mut lo_d: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwd]);
            for d in 0..3 {
                for w in 0..nwd {
                    let bp = self.bit_planes_d[d][level][w];
                    up_d[d][w] = mem_d[w] & bp;
                    lo_d[d][w] = mem_d[w] & !bp;
                }
            }
            let mut up_r: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwr]);
            let mut lo_r: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwr]);
            for d in 0..3 {
                for w in 0..nwr {
                    let bp = self.bit_planes_r[d][level][w];
                    up_r[d][w] = mem_r[w] & bp;
                    lo_r[d][w] = mem_r[w] & !bp;
                }
            }
            let mut child_d = vec![0u64; nwd];
            let mut child_r = vec![0u64; nwr];
            for child in 0..8u32 {
                let b0 = (child & 1) != 0;
                child_d.copy_from_slice(if b0 { &up_d[0] } else { &lo_d[0] });
                child_r.copy_from_slice(if b0 { &up_r[0] } else { &lo_r[0] });
                for d in 1..3 {
                    let bd = (child >> d) & 1 != 0;
                    let half_d = if bd { &up_d[d] } else { &lo_d[d] };
                    let half_r = if bd { &up_r[d] } else { &lo_r[d] };
                    for w in 0..nwd { child_d[w] &= half_d[w]; }
                    for w in 0..nwr { child_r[w] &= half_r[w]; }
                }
                let (cwd, cwr) = self.aniso_recurse(
                    &child_d, &child_r, level + 1, alpha, cfg,
                    sum_w_r_parents, n_parents, sum_wr_w2,
                );
                child_w_d[child as usize] = cwd;
                child_w_r[child as usize] = cwr;
            }
        }

        // Now we have all 8 child (W_d, W_r) pairs at this (parent's level+1)
        // = (level+1) cells. The wavelet decomposition is performed on the δ
        // field one level FINER than `level`, so the wavelet stats are
        // accumulated at level (level+1)? No — the WAVELET COEFFICIENT at scale
        // R_l describes the difference structure at scale R_l, derived from the
        // R_(l+1) children of the R_l parent. So we accumulate at `level+1`
        // because the wavelet coefficient lives at the finer scale but is
        // computed FROM the parent level partition.
        //
        // However the conventional dyadic-band labeling is to associate the
        // wavelet coefficient with the parent's scale R_l (cells of side R_l
        // containing 8 children of side R_l/2). Either convention is fine as
        // long as we're consistent. We use the parent's scale (level).

        // Compute per-child δ from (W_d, W_r) values, with footprint cutoff
        // applied at the child level (W_r > w_r_min for that child).
        let mut delta = [0.0f64; 8];
        let mut active = [false; 8];
        for c in 0..8 {
            if child_w_r[c] > cfg.w_r_min {
                delta[c] = if alpha > 0.0 && child_w_r[c] > 0.0 {
                    child_w_d[c] / (alpha * child_w_r[c]) - 1.0
                } else { -1.0 };
                active[c] = true;
            }
        }

        // We need a parent cell with all 8 children "well-defined" for the
        // wavelet coefficients to be unambiguous. If a child fails the
        // footprint cut, its δ is undefined. For the cell-wavelet observable
        // to be physically meaningful, we use only parent cells where ALL 8
        // children are active (have non-trivial random presence). This is the
        // strict mode; one could also impute δ=0 for missing children, but
        // that biases the wavelet statistics. Strict mode is cleaner.
        let all_active = active.iter().all(|&a| a);
        if !all_active {
            return (sw_d, sw_r);
        }

        // Compute the 7 non-trivial wavelet coefficients via the Haar transform
        // matrix. For pattern e ∈ {1,...,7} and child σ ∈ {0,...,7}:
        //   w_e = (1/8) Σ_σ (-1)^(e·σ) δ_σ
        // where e·σ is the binary dot product.
        let mut w_pattern = [0.0f64; 8];
        for e in 1..8 {
            let mut acc = 0.0;
            for sigma in 0..8 {
                let sign = if ((e & sigma) as u32).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                acc += sign * delta[sigma];
            }
            w_pattern[e] = acc / 8.0;
        }

        // Use the parent's W_r as the weight for these wavelet stats (it's the
        // total W_r in the 8 children, since they tile the parent).
        let parent_wr = sw_r;
        // Check cutoff at parent level too
        if parent_wr > cfg.w_r_min {
            n_parents[level] += 1;
            sum_w_r_parents[level] += parent_wr;
            for e in 1..8 {
                sum_wr_w2[level][e] += parent_wr * w_pattern[e] * w_pattern[e];
            }
        }

        (sw_d, sw_r)
    }

    /// Point-list variant of aniso_recurse. Same return convention.
    fn aniso_recurse_pointlist(
        &self,
        idx_d: &[u32],
        idx_r: &[u32],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        sum_w_r_parents: &mut [f64],
        n_parents: &mut [u64],
        sum_wr_w2: &mut [[f64; 8]],
    ) -> (f64, f64) {
        let (n_d, sw_d, _) = self.cell_sums_d_list(idx_d);
        let (n_r, sw_r, _) = self.cell_sums_r_list(idx_r);
        if n_d == 0 && n_r == 0 { return (0.0, 0.0); }

        if level >= self.l_max {
            return (sw_d, sw_r);
        }

        // Bucket points into 8 children
        let mut buckets_d: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        let mut buckets_r: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        for &i in idx_d {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..3 {
                let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_d[child_id].push(i);
        }
        for &i in idx_r {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..3 {
                let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_r[child_id].push(i);
        }

        let mut child_w_d = [0.0f64; 8];
        let mut child_w_r = [0.0f64; 8];
        for c in 0..8 {
            let (cwd, cwr) = self.aniso_recurse_pointlist(
                &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg,
                sum_w_r_parents, n_parents, sum_wr_w2,
            );
            child_w_d[c] = cwd;
            child_w_r[c] = cwr;
        }

        // Same wavelet computation as in aniso_recurse
        let mut delta = [0.0f64; 8];
        let mut active = [false; 8];
        for c in 0..8 {
            if child_w_r[c] > cfg.w_r_min {
                delta[c] = if alpha > 0.0 && child_w_r[c] > 0.0 {
                    child_w_d[c] / (alpha * child_w_r[c]) - 1.0
                } else { -1.0 };
                active[c] = true;
            }
        }
        let all_active = active.iter().all(|&a| a);
        if !all_active {
            return (sw_d, sw_r);
        }

        let mut w_pattern = [0.0f64; 8];
        for e in 1..8 {
            let mut acc = 0.0;
            for sigma in 0..8 {
                let sign = if ((e & sigma) as u32).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                acc += sign * delta[sigma];
            }
            w_pattern[e] = acc / 8.0;
        }

        let parent_wr = sw_r;
        if parent_wr > cfg.w_r_min {
            n_parents[level] += 1;
            sum_w_r_parents[level] += parent_wr;
            for e in 1..8 {
                sum_wr_w2[level][e] += parent_wr * w_pattern[e] * w_pattern[e];
            }
        }

        (sw_d, sw_r)
    }

    /// Compute second-order Haar scattering coefficients at every (l_1, l_2)
    /// pair of cascade levels with l_2 < l_1. Restricted to axis-aligned
    /// wavelets in 3D (e_1, e_2 ∈ {x, y, z}).
    ///
    /// First-order scattering already produces |w_{e_1}^(l_1)| on every
    /// level-l_1 parent cell. Second-order scattering treats this magnitude
    /// field as a new scalar field and computes its Haar wavelet coefficients
    /// at every coarser level l_2 < l_1, again per direction e_2.
    ///
    /// Following Mallat (2012) and Bruna & Mallat (2013), second-order
    /// scattering captures non-Gaussian structural information complementary
    /// to first-order (which is power-spectrum-like) and to the bispectrum.
    ///
    /// Output is a vector of [`ScatteringStats`] entries, one per (l_1, l_2)
    /// pair with l_2 < l_1. Each entry contains:
    /// - the first-order coefficient < |w_{e_1}^(l_1)|² > at l_1, indexed by e_1
    /// - the second-order coefficient < |w_{e_2}^(l_2)[ |w_{e_1}^(l_1)| ]|² >
    ///   at l_2, indexed by (e_1, e_2)
    ///
    /// Only 3D is supported. For other D this returns an empty vector.
    pub fn analyze_scattering_2nd_order(
        &self, cfg: &FieldStatsConfig,
    ) -> Vec<ScatteringStats> {
        if D != 3 { return Vec::new(); }
        let n_d = self.n_d();
        let n_r = self.n_r();
        let n_levels = self.l_max + 1;
        if n_levels == 0 { return Vec::new(); }

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || (n_d == 0 && n_r == 0) {
            return Vec::new();
        }
        let alpha = total_w_d / total_w_r;

        // First-order accumulators: per level l_1, per axis direction e_1 ∈
        // {x=0, y=1, z=2}, sum of W_r-weighted |w_{e_1}^(l_1)|² over all
        // level-l_1 parent cells.
        let mut sum_wr_first = vec![[0.0f64; 3]; n_levels];
        // Second-order accumulators: indexed by (l_1, l_2, e_1, e_2), with
        // l_2 < l_1. Layout: sum_wr_second[l_1][l_2][e_1][e_2].
        let mut sum_wr_second = vec![vec![[[0.0f64; 3]; 3]; n_levels]; n_levels];
        // Per (l_1, l_2) parent counts at level-l_2:
        let mut n_parents_coarse = vec![vec![0u64; n_levels]; n_levels];
        let mut sum_wr_parents_coarse = vec![vec![0.0f64; n_levels]; n_levels];
        // Per level-l_1 parent counts (for first-order normalization):
        let mut n_parents_fine = vec![0u64; n_levels];
        let mut sum_wr_parents_fine = vec![0.0f64; n_levels];

        // Build root membership vectors
        let mut root_d = vec![u64::MAX; self.n_words_d.max(1)];
        if n_d == 0 {
            for w in root_d.iter_mut() { *w = 0; }
        } else if n_d % 64 != 0 {
            let last = n_d % 64;
            root_d[self.n_words_d - 1] = (1u64 << last) - 1;
        }
        let mut root_r = vec![u64::MAX; self.n_words_r.max(1)];
        if n_r == 0 {
            for w in root_r.iter_mut() { *w = 0; }
        } else if n_r % 64 != 0 {
            let last = n_r % 64;
            root_r[self.n_words_r - 1] = (1u64 << last) - 1;
        }
        let root_d_view: &[u64] = if n_d == 0 { &[] } else { &root_d[..self.n_words_d] };
        let root_r_view: &[u64] = if n_r == 0 { &[] } else { &root_r[..self.n_words_r] };

        // Each cell's recursion returns its (W_d, W_r) plus a vector of
        // per-(l_1, e_1) mean-|w_{e_1}^(l_1)| values for all l_1 ≥ this cell's
        // level. We use a flat layout: w_means[l_1 * 3 + e_1] = mean of
        // |w_{e_1}^(l_1)| over this cell's level-l_1 descendants. Entries for
        // l_1 < cell's level are unused (zero).
        let mut _root_w_means = vec![0.0f64; n_levels * 3];
        self.scatter_recurse(
            root_d_view, root_r_view, 0, alpha, cfg, n_levels,
            &mut sum_wr_first, &mut sum_wr_second,
            &mut n_parents_coarse, &mut sum_wr_parents_coarse,
            &mut n_parents_fine, &mut sum_wr_parents_fine,
            &mut _root_w_means,
        );

        // Build output: one ScatteringStats entry per (l_1, l_2) pair with
        // l_2 < l_1.
        let max_eff = self.data.range.max_supported_l_max() as usize;
        let mut out = Vec::with_capacity(n_levels * (n_levels - 1) / 2);
        for l1 in 1..n_levels {
            // Normalize first-order at this l_1
            let total_fine = sum_wr_parents_fine[l1];
            let mut first_order = [0.0f64; 3];
            if total_fine > 0.0 {
                for e in 0..3 {
                    first_order[e] = sum_wr_first[l1][e] / total_fine;
                }
            }
            for l2 in 0..l1 {
                let total_coarse = sum_wr_parents_coarse[l1][l2];
                let mut second_order = [[0.0f64; 3]; 3];
                if total_coarse > 0.0 {
                    for e1 in 0..3 {
                        for e2 in 0..3 {
                            second_order[e1][e2] =
                                sum_wr_second[l1][l2][e1][e2] / total_coarse;
                        }
                    }
                }
                let side_fine = if l1 <= max_eff {
                    (1u64 << (max_eff - l1)) as f64 } else { 1.0 };
                let side_coarse = if l2 <= max_eff {
                    (1u64 << (max_eff - l2)) as f64 } else { 1.0 };
                out.push(ScatteringStats {
                    level_fine: l1,
                    level_coarse: l2,
                    cell_side_fine_trimmed: side_fine,
                    cell_side_coarse_trimmed: side_coarse,
                    n_parents_coarse: n_parents_coarse[l1][l2],
                    sum_w_r_parents: total_coarse,
                    first_order,
                    second_order,
                });
            }
        }
        out
    }

    /// Recursive walk for `analyze_scattering_2nd_order`. 3D only.
    /// Returns (W_d, W_r) and writes into `w_means_out` the per-(l_1, e_1)
    /// mean-|w_{e_1}^(l_1)| values for all l_1 ≥ this cell's level.
    fn scatter_recurse(
        &self,
        mem_d: &[u64],
        mem_r: &[u64],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        n_levels: usize,
        sum_wr_first: &mut [[f64; 3]],
        sum_wr_second: &mut [Vec<[[f64; 3]; 3]>],
        n_parents_coarse: &mut [Vec<u64>],
        sum_wr_parents_coarse: &mut [Vec<f64>],
        n_parents_fine: &mut [u64],
        sum_wr_parents_fine: &mut [f64],
        w_means_out: &mut [f64],
    ) -> (f64, f64) {
        // Zero out our slot
        for v in w_means_out.iter_mut() { *v = 0.0; }

        let (n_d, sw_d, _) = self.cell_sums_d(mem_d);
        let (n_r, sw_r, _) = self.cell_sums_r(mem_r);
        if n_d == 0 && n_r == 0 { return (0.0, 0.0); }
        if level >= self.l_max {
            return (sw_d, sw_r);
        }

        // Allocate per-child w_means and child W's
        // Layout: child_w_means[c * (n_levels * 3) + l1 * 3 + e1]
        let stride = n_levels * 3;
        let mut child_w_means = vec![0.0f64; 8 * stride];
        let mut child_w_d = [0.0f64; 8];
        let mut child_w_r = [0.0f64; 8];

        let thresh = self.crossover_threshold as u64;
        let use_pointlist = n_d <= thresh && n_r <= thresh;

        if use_pointlist {
            let idx_d = mem_to_indices(mem_d, n_d as usize);
            let idx_r = mem_to_indices(mem_r, n_r as usize);
            let mut buckets_d: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            let mut buckets_r: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
            for &i in &idx_d {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..3 {
                    let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_d[child_id].push(i);
            }
            for &i in &idx_r {
                let word_idx = (i / 64) as usize;
                let bit_idx = (i % 64) as u64;
                let mut child_id: usize = 0;
                for d in 0..3 {
                    let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                    if bit != 0 { child_id |= 1 << d; }
                }
                buckets_r[child_id].push(i);
            }
            for c in 0..8 {
                let slot_start = c * stride;
                let slot_end = slot_start + stride;
                let (cwd, cwr) = self.scatter_recurse_pointlist(
                    &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg, n_levels,
                    sum_wr_first, sum_wr_second,
                    n_parents_coarse, sum_wr_parents_coarse,
                    n_parents_fine, sum_wr_parents_fine,
                    &mut child_w_means[slot_start..slot_end],
                );
                child_w_d[c] = cwd;
                child_w_r[c] = cwr;
            }
        } else {
            let nwd = self.n_words_d;
            let nwr = self.n_words_r;
            let mut up_d: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwd]);
            let mut lo_d: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwd]);
            for d in 0..3 {
                for w in 0..nwd {
                    let bp = self.bit_planes_d[d][level][w];
                    up_d[d][w] = mem_d[w] & bp;
                    lo_d[d][w] = mem_d[w] & !bp;
                }
            }
            let mut up_r: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwr]);
            let mut lo_r: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; nwr]);
            for d in 0..3 {
                for w in 0..nwr {
                    let bp = self.bit_planes_r[d][level][w];
                    up_r[d][w] = mem_r[w] & bp;
                    lo_r[d][w] = mem_r[w] & !bp;
                }
            }
            let mut child_d = vec![0u64; nwd];
            let mut child_r = vec![0u64; nwr];
            for c in 0..8u32 {
                let b0 = (c & 1) != 0;
                child_d.copy_from_slice(if b0 { &up_d[0] } else { &lo_d[0] });
                child_r.copy_from_slice(if b0 { &up_r[0] } else { &lo_r[0] });
                for d in 1..3 {
                    let bd = (c >> d) & 1 != 0;
                    let half_d = if bd { &up_d[d] } else { &lo_d[d] };
                    let half_r = if bd { &up_r[d] } else { &lo_r[d] };
                    for w in 0..nwd { child_d[w] &= half_d[w]; }
                    for w in 0..nwr { child_r[w] &= half_r[w]; }
                }
                let slot_start = (c as usize) * stride;
                let slot_end = slot_start + stride;
                let (cwd, cwr) = self.scatter_recurse(
                    &child_d, &child_r, level + 1, alpha, cfg, n_levels,
                    sum_wr_first, sum_wr_second,
                    n_parents_coarse, sum_wr_parents_coarse,
                    n_parents_fine, sum_wr_parents_fine,
                    &mut child_w_means[slot_start..slot_end],
                );
                child_w_d[c as usize] = cwd;
                child_w_r[c as usize] = cwr;
            }
        }

        // Now we have the 8 children's (W_d, W_r) and per-child w_means arrays.
        // Compute first-order δ at THIS level (the children's δ values).
        let mut delta = [0.0f64; 8];
        let mut all_active = true;
        for c in 0..8 {
            if child_w_r[c] > cfg.w_r_min {
                delta[c] = if alpha > 0.0 && child_w_r[c] > 0.0 {
                    child_w_d[c] / (alpha * child_w_r[c]) - 1.0
                } else { -1.0 };
            } else {
                all_active = false;
            }
        }

        // First-order axis-aligned wavelets: e ∈ {1=x, 2=y, 4=z}.
        // For consistency with AnisotropyStats indexing, use [e1] = [0=x, 1=y, 2=z]
        // and pattern bits 1, 2, 4 respectively.
        let parent_wr = sw_r;
        let mut w_axis = [0.0f64; 3];
        if all_active && parent_wr > cfg.w_r_min {
            let patterns: [u32; 3] = [1, 2, 4];
            for (e1, &pat) in patterns.iter().enumerate() {
                let mut acc = 0.0;
                for sigma in 0..8u32 {
                    let sign = if ((pat & sigma) as u32).count_ones() % 2 == 0 {
                        1.0 } else { -1.0 };
                    acc += sign * delta[sigma as usize];
                }
                w_axis[e1] = acc / 8.0;
            }
            // Accumulate first-order at this level (= l_1 = `level`)
            n_parents_fine[level] += 1;
            sum_wr_parents_fine[level] += parent_wr;
            for e1 in 0..3 {
                sum_wr_first[level][e1] += parent_wr * w_axis[e1] * w_axis[e1];
            }
        }

        // Now: write THIS cell's w_means slots.
        // For l_1 == level: store |w_{e_1}^(level)| if all_active, else 0.
        // For l_1 > level: average the 8 children's w_means at (l_1, e_1).
        if level < n_levels {
            for e1 in 0..3 {
                w_means_out[level * 3 + e1] =
                    if all_active && parent_wr > cfg.w_r_min { w_axis[e1].abs() } else { 0.0 };
            }
        }
        for l1 in (level + 1)..n_levels {
            for e1 in 0..3 {
                let mut sum = 0.0;
                for c in 0..8 {
                    sum += child_w_means[c * stride + l1 * 3 + e1];
                }
                w_means_out[l1 * 3 + e1] = sum / 8.0;
            }
        }

        // Second-order: for each l_1 > level, compute Haar wavelet
        // decomposition of |w_{e_1}^(l_1)| field at level=`level`, using the
        // 8 children's mean-|w_{e_1}^(l_1)| values.
        if all_active && parent_wr > cfg.w_r_min {
            for l1 in (level + 1)..n_levels {
                for e1 in 0..3 {
                    // Gather 8 children's mean-|w_{e_1}^(l_1)|
                    let mut child_w_field = [0.0f64; 8];
                    for c in 0..8 {
                        child_w_field[c] =
                            child_w_means[c * stride + l1 * 3 + e1];
                    }
                    // Haar decomposition for axis-aligned e_2
                    let patterns: [u32; 3] = [1, 2, 4];
                    for (e2, &pat) in patterns.iter().enumerate() {
                        let mut acc = 0.0;
                        for sigma in 0..8u32 {
                            let sign = if ((pat & sigma) as u32).count_ones() % 2 == 0 {
                                1.0 } else { -1.0 };
                            acc += sign * child_w_field[sigma as usize];
                        }
                        let w_2 = acc / 8.0;
                        sum_wr_second[l1][level][e1][e2] += parent_wr * w_2 * w_2;
                    }
                }
                // Bookkeeping at (l_1, l_2 = level)
                n_parents_coarse[l1][level] += 1;
                sum_wr_parents_coarse[l1][level] += parent_wr;
            }
        }

        (sw_d, sw_r)
    }

    /// Point-list variant of scatter_recurse. Same signature.
    fn scatter_recurse_pointlist(
        &self,
        idx_d: &[u32],
        idx_r: &[u32],
        level: usize,
        alpha: f64,
        cfg: &FieldStatsConfig,
        n_levels: usize,
        sum_wr_first: &mut [[f64; 3]],
        sum_wr_second: &mut [Vec<[[f64; 3]; 3]>],
        n_parents_coarse: &mut [Vec<u64>],
        sum_wr_parents_coarse: &mut [Vec<f64>],
        n_parents_fine: &mut [u64],
        sum_wr_parents_fine: &mut [f64],
        w_means_out: &mut [f64],
    ) -> (f64, f64) {
        for v in w_means_out.iter_mut() { *v = 0.0; }

        let (n_d, sw_d, _) = self.cell_sums_d_list(idx_d);
        let (n_r, sw_r, _) = self.cell_sums_r_list(idx_r);
        if n_d == 0 && n_r == 0 { return (0.0, 0.0); }
        if level >= self.l_max {
            return (sw_d, sw_r);
        }

        let stride = n_levels * 3;
        let mut child_w_means = vec![0.0f64; 8 * stride];
        let mut child_w_d = [0.0f64; 8];
        let mut child_w_r = [0.0f64; 8];

        let mut buckets_d: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        let mut buckets_r: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        for &i in idx_d {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..3 {
                let bit = (self.bit_planes_d[d][level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_d[child_id].push(i);
        }
        for &i in idx_r {
            let word_idx = (i / 64) as usize;
            let bit_idx = (i % 64) as u64;
            let mut child_id: usize = 0;
            for d in 0..3 {
                let bit = (self.bit_planes_r[d][level][word_idx] >> bit_idx) & 1;
                if bit != 0 { child_id |= 1 << d; }
            }
            buckets_r[child_id].push(i);
        }

        for c in 0..8 {
            let slot_start = c * stride;
            let slot_end = slot_start + stride;
            let (cwd, cwr) = self.scatter_recurse_pointlist(
                &buckets_d[c], &buckets_r[c], level + 1, alpha, cfg, n_levels,
                sum_wr_first, sum_wr_second,
                n_parents_coarse, sum_wr_parents_coarse,
                n_parents_fine, sum_wr_parents_fine,
                &mut child_w_means[slot_start..slot_end],
            );
            child_w_d[c] = cwd;
            child_w_r[c] = cwr;
        }

        let mut delta = [0.0f64; 8];
        let mut all_active = true;
        for c in 0..8 {
            if child_w_r[c] > cfg.w_r_min {
                delta[c] = if alpha > 0.0 && child_w_r[c] > 0.0 {
                    child_w_d[c] / (alpha * child_w_r[c]) - 1.0
                } else { -1.0 };
            } else {
                all_active = false;
            }
        }

        let parent_wr = sw_r;
        let mut w_axis = [0.0f64; 3];
        if all_active && parent_wr > cfg.w_r_min {
            let patterns: [u32; 3] = [1, 2, 4];
            for (e1, &pat) in patterns.iter().enumerate() {
                let mut acc = 0.0;
                for sigma in 0..8u32 {
                    let sign = if ((pat & sigma) as u32).count_ones() % 2 == 0 {
                        1.0 } else { -1.0 };
                    acc += sign * delta[sigma as usize];
                }
                w_axis[e1] = acc / 8.0;
            }
            n_parents_fine[level] += 1;
            sum_wr_parents_fine[level] += parent_wr;
            for e1 in 0..3 {
                sum_wr_first[level][e1] += parent_wr * w_axis[e1] * w_axis[e1];
            }
        }

        if level < n_levels {
            for e1 in 0..3 {
                w_means_out[level * 3 + e1] =
                    if all_active && parent_wr > cfg.w_r_min { w_axis[e1].abs() } else { 0.0 };
            }
        }
        for l1 in (level + 1)..n_levels {
            for e1 in 0..3 {
                let mut sum = 0.0;
                for c in 0..8 {
                    sum += child_w_means[c * stride + l1 * 3 + e1];
                }
                w_means_out[l1 * 3 + e1] = sum / 8.0;
            }
        }

        if all_active && parent_wr > cfg.w_r_min {
            for l1 in (level + 1)..n_levels {
                for e1 in 0..3 {
                    let mut child_w_field = [0.0f64; 8];
                    for c in 0..8 {
                        child_w_field[c] =
                            child_w_means[c * stride + l1 * 3 + e1];
                    }
                    let patterns: [u32; 3] = [1, 2, 4];
                    for (e2, &pat) in patterns.iter().enumerate() {
                        let mut acc = 0.0;
                        for sigma in 0..8u32 {
                            let sign = if ((pat & sigma) as u32).count_ones() % 2 == 0 {
                                1.0 } else { -1.0 };
                            acc += sign * child_w_field[sigma as usize];
                        }
                        let w_2 = acc / 8.0;
                        sum_wr_second[l1][level][e1][e2] += parent_wr * w_2 * w_2;
                    }
                }
                n_parents_coarse[l1][level] += 1;
                sum_wr_parents_coarse[l1][level] += parent_wr;
            }
        }

        (sw_d, sw_r)
    }

    // ========================================================================
    // analyze_field_stats_v2: visitor-based reimplementation
    // ========================================================================

    /// Visitor-based implementation of `analyze_field_stats`. Walks the
    /// cascade exactly once via [`Self::walk`] and produces results
    /// numerically identical to the legacy `analyze_field_stats`.
    ///
    /// This is the migration target for [`Self::analyze_field_stats`];
    /// the legacy method will become a thin wrapper around this once
    /// all statistics have visitor-based equivalents.
    pub fn analyze_field_stats_v2(&self, cfg: &FieldStatsConfig) -> Vec<DensityFieldStats> {
        let n_levels = self.l_max + 1;
        // Pathological-case shortcuts. In Periodic mode without randoms,
        // sum_w_r is 0 but α is computed from box volume; that's a valid
        // configuration, not an empty-result case.
        let synth = self.synthesizes_randoms_from_box();
        if !synth && self.sum_w_r() <= 0.0 {
            return (0..n_levels).map(|l| empty_density_stats::<D>(self, l, cfg)).collect();
        }
        if self.n_d() == 0 && self.n_r() == 0 {
            return (0..n_levels).map(|l| empty_density_stats::<D>(self, l, cfg)).collect();
        }
        let mut visitor = FieldStatsVisitor::<D>::new(self, n_levels, cfg);
        self.walk(&mut visitor);
        visitor.into_results(self, cfg)
    }

    /// Cascade-based cell-count PMF (CIC: cell-in-cube).
    ///
    /// For each dyadic level l, returns the histogram of integer data-point
    /// counts over cells of side 2^(L_max − l). The cascade visits each
    /// non-empty cell exactly once; in [`BoundaryMode::Periodic`] the
    /// unvisited (zero-data) cells are folded into the k=0 bin of the
    /// density histogram via an analytic correction, so the density sums
    /// to 1 over the full box.
    ///
    /// Cost: O(N log N) traversal, O(N) memory — much cheaper than the
    /// dense-grid `cascade_with_pmf_windows` which is O(M^D). The trade
    /// is that this version produces dyadic-spaced sides only; for
    /// log-spaced non-dyadic sides use the dense-grid version.
    ///
    /// **Works in any dimension D ≥ 1.** Ignores point weights — the
    /// CIC PMF is a histogram of object counts; weighted-moments are
    /// the job of [`Self::analyze_field_stats`].
    pub fn analyze_cic_pmf(&self, cfg: &CicPmfConfig) -> Vec<CicPmfStats> {
        let n_levels = self.l_max + 1;
        // Note: we deliberately do NOT short-circuit on empty inputs here.
        // In Periodic mode the visitor's finalization adds the missing-cell
        // analytic correction even when no `enter_cell` was ever called,
        // producing the correct degenerate PMF (density[0] = 1).
        let mut visitor = CicPmfVisitor::<D>::new(self, n_levels, cfg);
        self.walk(&mut visitor);
        visitor.into_results(self)
    }

    /// Walker-based anisotropy implementation. Produces results
    /// numerically identical to [`Self::analyze_anisotropy`] but via the
    /// shared cascade Walker / Visitor machinery, so the cascade is
    /// traversed only once when this is composed with other visitors.
    ///
    /// **Works in any dimension D ≥ 1.**
    pub fn analyze_anisotropy_v2(&self, cfg: &FieldStatsConfig) -> Vec<AnisotropyStats> {
        let n_levels = self.l_max + 1;
        let synth = self.synthesizes_randoms_from_box();
        if !synth && self.sum_w_r() <= 0.0 {
            return (0..n_levels).map(|l| empty_anisotropy::<D>(self, l)).collect();
        }
        if self.n_d() == 0 && self.n_r() == 0 {
            return (0..n_levels).map(|l| empty_anisotropy::<D>(self, l)).collect();
        }
        let mut visitor = AnisotropyVisitor::<D>::new(self, n_levels, cfg);
        self.walk(&mut visitor);
        visitor.into_results(self)
    }
}

// ----------------------------------------------------------------------------
// FieldStatsVisitor: density-moments visitor for field-stats
// ----------------------------------------------------------------------------

/// Visitor that accumulates W_r-weighted moments of the density contrast δ
/// at every cascade level, plus an outside-footprint diagnostic and
/// optional histogram of log10(1+δ).
///
/// Numerically identical to the legacy `field_recurse`-based pipeline for
/// the [`BoundaryMode::Isolated`] case. In [`BoundaryMode::Periodic`] mode
/// (with a synthesized random catalog from box volume), an analytic
/// correction for unvisited zero-data cells is applied at finalization
/// so that the W_r-weighted moments are taken over the **full box volume**
/// rather than the data-touched volume only.
pub struct FieldStatsVisitor<'a, const D: usize> {
    cfg: &'a FieldStatsConfig,
    footprint: crate::cascade_visitor::FootprintCutoff,
    /// Whether the host cascade is in Periodic + no-randoms mode. When true,
    /// `into_results` adds an analytic δ=−1 contribution from unvisited
    /// (zero-data) cell volume per level.
    periodic_box_correction: bool,
    sum_w_r: Vec<f64>,
    sum_w_r_delta1: Vec<f64>,
    sum_w_r_delta2: Vec<f64>,
    sum_w_r_delta3: Vec<f64>,
    sum_w_r_delta4: Vec<f64>,
    n_active: Vec<u64>,
    min_delta: Vec<f64>,
    max_delta: Vec<f64>,
    n_outside: Vec<u64>,
    sw_d_outside: Vec<f64>,
    hist: Vec<Vec<f64>>,
    hist_under: Vec<f64>,
    hist_over: Vec<f64>,
    /// Parallel Neumaier-compensated accumulators. Populated only when
    /// `cfg.compensated_sums == true`. `into_results` reads these in
    /// place of the naive `Vec<f64>` accumulators when active. Empty
    /// vectors when compensated mode is off (zero allocation overhead).
    sum_w_r_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sum_w_r_delta1_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sum_w_r_delta2_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sum_w_r_delta3_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sum_w_r_delta4_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sw_d_outside_comp: Vec<crate::compensated_sum::CompensatedSum>,
}

impl<'a, const D: usize> FieldStatsVisitor<'a, D> {
    pub fn new(pair: &BitVecCascadePair<D>, n_levels: usize, cfg: &'a FieldStatsConfig) -> Self {
        let footprint = crate::cascade_visitor::FootprintCutoff::strict(cfg.w_r_min);
        let comp_len = if cfg.compensated_sums { n_levels } else { 0 };
        let comp_zero = || vec![crate::compensated_sum::CompensatedSum::new(); comp_len];
        Self {
            cfg, footprint,
            periodic_box_correction: pair.synthesizes_randoms_from_box(),
            sum_w_r: vec![0.0; n_levels],
            sum_w_r_delta1: vec![0.0; n_levels],
            sum_w_r_delta2: vec![0.0; n_levels],
            sum_w_r_delta3: vec![0.0; n_levels],
            sum_w_r_delta4: vec![0.0; n_levels],
            n_active: vec![0; n_levels],
            min_delta: vec![f64::INFINITY; n_levels],
            max_delta: vec![f64::NEG_INFINITY; n_levels],
            n_outside: vec![0; n_levels],
            sw_d_outside: vec![0.0; n_levels],
            hist: vec![vec![0.0; cfg.hist_bins]; n_levels],
            hist_under: vec![0.0; n_levels],
            hist_over: vec![0.0; n_levels],
            sum_w_r_comp: comp_zero(),
            sum_w_r_delta1_comp: comp_zero(),
            sum_w_r_delta2_comp: comp_zero(),
            sum_w_r_delta3_comp: comp_zero(),
            sum_w_r_delta4_comp: comp_zero(),
            sw_d_outside_comp: comp_zero(),
        }
    }

    /// Finalize accumulators into per-level DensityFieldStats results.
    pub fn into_results(self, pair: &BitVecCascadePair<D>, cfg: &FieldStatsConfig) -> Vec<DensityFieldStats> {
        let n_levels = self.sum_w_r.len();
        let mut hist_edges = Vec::new();
        if cfg.hist_bins > 0 {
            hist_edges = (0..=cfg.hist_bins).map(|i| {
                cfg.hist_log_min
                    + (cfg.hist_log_max - cfg.hist_log_min) * (i as f64) / (cfg.hist_bins as f64)
            }).collect();
        }

        // When compensated_sums is active, snapshot the per-level finalized
        // values from the parallel CompensatedSum accumulators. Otherwise
        // use the naive vectors directly.
        let comp = cfg.compensated_sums;
        let read_swr        = |l: usize| if comp { self.sum_w_r_comp[l].value() } else { self.sum_w_r[l] };
        let read_sd1        = |l: usize| if comp { self.sum_w_r_delta1_comp[l].value() } else { self.sum_w_r_delta1[l] };
        let read_sd2        = |l: usize| if comp { self.sum_w_r_delta2_comp[l].value() } else { self.sum_w_r_delta2[l] };
        let read_sd3        = |l: usize| if comp { self.sum_w_r_delta3_comp[l].value() } else { self.sum_w_r_delta3[l] };
        let read_sd4        = |l: usize| if comp { self.sum_w_r_delta4_comp[l].value() } else { self.sum_w_r_delta4[l] };
        let read_swd_out    = |l: usize| if comp { self.sw_d_outside_comp[l].value() } else { self.sw_d_outside[l] };

        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            // Periodic + no randoms: add analytic correction for unvisited
            // (zero-data) cells. Each contributes δ=−1 weighted by its
            // volume. Total cell volume per level = 1.0 (V_box units).
            let (sw_r, sd1, sd2, sd3, sd4) = if self.periodic_box_correction {
                let visited = read_swr(l);
                let missed = (1.0 - visited).max(0.0);
                (
                    visited + missed,
                    read_sd1(l) + missed * (-1.0),
                    read_sd2(l) + missed,
                    read_sd3(l) + missed * (-1.0),
                    read_sd4(l) + missed,
                )
            } else {
                (read_swr(l), read_sd1(l), read_sd2(l), read_sd3(l), read_sd4(l))
            };

            let total_w = sw_r;
            let (mean, var, m3, m4) = if total_w > 0.0 {
                let m1 = sd1 / total_w;
                let m2_raw = sd2 / total_w;
                let m3_raw = sd3 / total_w;
                let m4_raw = sd4 / total_w;
                let var = m2_raw - m1 * m1;
                let m3c = m3_raw - 3.0 * m1 * m2_raw + 2.0 * m1 * m1 * m1;
                let m4c = m4_raw - 4.0 * m1 * m3_raw + 6.0 * m1 * m1 * m2_raw - 3.0 * m1 * m1 * m1 * m1;
                (m1, var.max(0.0), m3c, m4c)
            } else { (0.0, 0.0, 0.0, 0.0) };
            let s3 = if var > 0.0 { m3 / (var * var) } else { 0.0 };

            let hist_density: Vec<f64> = if total_w > 0.0 && cfg.hist_bins > 0 {
                self.hist[l].iter().map(|&c| c / total_w).collect()
            } else {
                vec![0.0; cfg.hist_bins]
            };
            let hist_under_n = if total_w > 0.0 { self.hist_under[l] / total_w } else { 0.0 };
            let hist_over_n = if total_w > 0.0 { self.hist_over[l] / total_w } else { 0.0 };

            let max_eff = pair.data.range.max_supported_l_max() as usize;
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else { 1.0 };

            // For periodic mode, min_delta has to incorporate the empty
            // cells' contribution of −1.
            let min_d = if self.periodic_box_correction {
                let v = if self.n_active[l] > 0 { self.min_delta[l] } else { 0.0 };
                let unvisited_at_level =
                    (1.0 - read_swr(l)).max(0.0) > 0.0;
                if unvisited_at_level { v.min(-1.0) } else { v }
            } else if self.n_active[l] > 0 {
                self.min_delta[l]
            } else { 0.0 };
            let max_d = if self.n_active[l] > 0 { self.max_delta[l] } else { 0.0 };

            out.push(DensityFieldStats {
                level: l,
                cell_side_trimmed: side_l,
                n_cells_active: self.n_active[l],
                sum_w_r_active: total_w,
                mean_delta: mean,
                var_delta: var,
                m3_delta: m3,
                m4_delta: m4,
                min_delta: min_d,
                max_delta: max_d,
                s3_delta: s3,
                n_cells_data_outside: self.n_outside[l],
                sum_w_d_outside: read_swd_out(l),
                hist_bin_edges: hist_edges.clone(),
                hist_density,
                hist_underflow_w_r: hist_under_n,
                hist_overflow_w_r: hist_over_n,
                raw_sum_w_r_delta_pow: [sd1, sd2, sd3, sd4],
            });
        }
        out
    }
}

impl<'a, const D: usize> crate::cascade_visitor::CascadeVisitor<D> for FieldStatsVisitor<'a, D> {
    fn enter_cell(&mut self, cell: &crate::cascade_visitor::CellVisit<D>) {
        let level = cell.level;
        if level >= self.sum_w_r.len() { return; }

        if let Some(delta) = cell.delta(&self.footprint) {
            // In-footprint cell: accumulate moments and histogram
            let sw_r = cell.randoms.sum_w;
            self.sum_w_r[level] += sw_r;
            self.sum_w_r_delta1[level] += sw_r * delta;
            let d2 = delta * delta;
            self.sum_w_r_delta2[level] += sw_r * d2;
            self.sum_w_r_delta3[level] += sw_r * d2 * delta;
            self.sum_w_r_delta4[level] += sw_r * d2 * d2;
            // Mirror into compensated accumulators when active. The branch
            // is once per cell visit (not per particle) — modern branch
            // prediction makes this near-zero overhead since the same
            // direction is taken for every cell of an analysis.
            if self.cfg.compensated_sums {
                self.sum_w_r_comp[level].add(sw_r);
                self.sum_w_r_delta1_comp[level].add(sw_r * delta);
                self.sum_w_r_delta2_comp[level].add(sw_r * d2);
                self.sum_w_r_delta3_comp[level].add(sw_r * d2 * delta);
                self.sum_w_r_delta4_comp[level].add(sw_r * d2 * d2);
            }
            self.n_active[level] += 1;
            if delta < self.min_delta[level] { self.min_delta[level] = delta; }
            if delta > self.max_delta[level] { self.max_delta[level] = delta; }

            if self.cfg.hist_bins > 0 {
                let one_plus = 1.0 + delta;
                if one_plus > 0.0 {
                    let log_v = one_plus.log10();
                    if log_v < self.cfg.hist_log_min {
                        self.hist_under[level] += sw_r;
                    } else if log_v >= self.cfg.hist_log_max {
                        self.hist_over[level] += sw_r;
                    } else {
                        let frac = (log_v - self.cfg.hist_log_min)
                            / (self.cfg.hist_log_max - self.cfg.hist_log_min);
                        let bin = (frac * self.cfg.hist_bins as f64) as usize;
                        let bin = bin.min(self.cfg.hist_bins - 1);
                        self.hist[level][bin] += sw_r;
                    }
                } else {
                    self.hist_under[level] += sw_r;
                }
            }
        } else if cell.data_outside_footprint(&self.footprint) {
            self.n_outside[level] += 1;
            self.sw_d_outside[level] += cell.data.sum_w;
            if self.cfg.compensated_sums {
                self.sw_d_outside_comp[level].add(cell.data.sum_w);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// AnisotropyVisitor: cell-wavelet anisotropy moments via after_children
// ----------------------------------------------------------------------------

/// Visitor that accumulates W_r-weighted squared Haar wavelet
/// coefficients of the density-contrast field at each dyadic scale.
///
/// **Works in any dimension D ≥ 1.** At each parent cell the wavelet
/// coefficients are computed from the `2^D` children's δ values via
/// the standard `2^D`-point Haar transform:
///
/// ```text
///     w_e = (1/2^D) Σ_σ (-1)^(e·σ) δ_σ      e ∈ {1, ..., 2^D − 1},
///                                            σ ∈ {0, ..., 2^D − 1}
/// ```
///
/// The non-trivial wavelet patterns factor by Hamming weight: D
/// axis-aligned (weight 1), D(D-1)/2 face-diagonal (weight 2), ...,
/// 1 body-diagonal (weight D). All are produced; convenience
/// accessors on [`AnisotropyStats`] expose the axis-aligned subset.
///
/// **Anisotropic-survey context:** the line-of-sight is taken to be
/// the **last axis** (axis index D − 1). The quadrupole-like moment
///   Q_2 = ⟨w_LoS²⟩ − mean over D−1 transverse axes of ⟨w_perp²⟩
/// directly probes Kaiser-vs-FoG-vs-isotropic anisotropy at every
/// dyadic scale. In 1D Q_2 ≡ 0 trivially.
///
/// Numerically identical to the legacy `aniso_recurse`-based pipeline
/// for D == 3 (see `analyze_anisotropy_legacy`).
pub struct AnisotropyVisitor<'a, const D: usize> {
    cfg: &'a FieldStatsConfig,
    footprint: crate::cascade_visitor::FootprintCutoff,
    /// Per-level: total parent cells that contributed (had all 2^D
    /// children in-footprint and parent in-footprint).
    n_parents: Vec<u64>,
    /// Per-level: sum of parent W_r over contributing parents.
    sum_w_r_parents: Vec<f64>,
    /// Per-level × pattern (length 2^D): W_r-weighted Σ w_pattern² over
    /// contributing parents. Pattern 0 (the constant) is unused.
    sum_wr_w2: Vec<Vec<f64>>,
    /// Parallel Neumaier-compensated accumulators. Populated only when
    /// `cfg.compensated_sums == true`. Empty (zero-allocation) when off.
    sum_w_r_parents_comp: Vec<crate::compensated_sum::CompensatedSum>,
    sum_wr_w2_comp: Vec<Vec<crate::compensated_sum::CompensatedSum>>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, const D: usize> AnisotropyVisitor<'a, D> {
    pub fn new(_pair: &BitVecCascadePair<D>, n_levels: usize, cfg: &'a FieldStatsConfig) -> Self {
        let footprint = crate::cascade_visitor::FootprintCutoff::strict(cfg.w_r_min);
        let n_patterns = 1usize << D;
        let comp_levels = if cfg.compensated_sums { n_levels } else { 0 };
        let comp_patterns = if cfg.compensated_sums { n_patterns } else { 0 };
        Self {
            cfg, footprint,
            n_parents: vec![0u64; n_levels],
            sum_w_r_parents: vec![0.0f64; n_levels],
            sum_wr_w2: vec![vec![0.0f64; n_patterns]; n_levels],
            sum_w_r_parents_comp: vec![
                crate::compensated_sum::CompensatedSum::new(); comp_levels],
            sum_wr_w2_comp: vec![
                vec![crate::compensated_sum::CompensatedSum::new(); comp_patterns];
                comp_levels],
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn into_results(self, pair: &BitVecCascadePair<D>) -> Vec<AnisotropyStats> {
        let n_levels = self.n_parents.len();
        let max_eff = pair.data.range.max_supported_l_max() as usize;
        let n_patterns = 1usize << D;
        let comp = self.cfg.compensated_sums;
        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let total_w = if comp { self.sum_w_r_parents_comp[l].value() }
                          else    { self.sum_w_r_parents[l] };

            // Per-pattern means (positions 1..n_patterns); position 0 is unused
            let mut means_by_pattern = vec![0.0f64; n_patterns];
            if total_w > 0.0 {
                for e in 1..n_patterns {
                    let num = if comp { self.sum_wr_w2_comp[l][e].value() }
                              else    { self.sum_wr_w2[l][e] };
                    means_by_pattern[e] = num / total_w;
                }
            }

            // Axis-aligned subset: pattern with bit-d set is the axis-d
            // wavelet, i.e. pattern (1 << d).
            let mut means_axis = vec![0.0f64; D];
            for d in 0..D {
                means_axis[d] = means_by_pattern[1usize << d];
            }

            // LoS quadrupole: LoS = last axis. Q_2 = <w_LoS²> − mean of
            // the (D-1) transverse axis variances. In 1D Q_2 ≡ 0.
            let (q2, reduced) = if D >= 2 {
                let los = means_axis[D - 1];
                let transverse_sum: f64 = means_axis[..D - 1].iter().sum();
                let transverse_mean = transverse_sum / (D - 1) as f64;
                let q = los - transverse_mean;
                let total_axis: f64 = means_axis.iter().sum();
                let reduced = if total_axis > 0.0 { q / (total_axis / D as f64) } else { 0.0 };
                (q, reduced)
            } else {
                (0.0, 0.0)
            };

            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else { 1.0 };

            out.push(AnisotropyStats {
                level: l,
                cell_side_trimmed: side_l,
                n_parents: self.n_parents[l],
                sum_w_r_parents: total_w,
                mean_w_squared_by_pattern: means_by_pattern,
                mean_w_squared_axis: means_axis,
                quadrupole_los: q2,
                reduced_quadrupole_los: reduced,
            });
        }
        out
    }
}

impl<'a, const D: usize> crate::cascade_visitor::CascadeVisitor<D> for AnisotropyVisitor<'a, D> {
    /// Anisotropy is a parent-from-children statistic: it needs all 2^D
    /// children's deltas to form the wavelet decomposition.
    fn after_children(
        &mut self,
        parent: &crate::cascade_visitor::CellVisit<D>,
        children: &[crate::cascade_visitor::CellVisit<D>],
    ) {
        let n_children = 1usize << D;
        if children.len() != n_children { return; }
        let level = parent.level;
        if level >= self.n_parents.len() { return; }

        // Strict mode: every child must be in-footprint and have a
        // well-defined δ. Otherwise the wavelet coefficients would be
        // ill-defined and imputing δ=0 would bias the estimator.
        let mut delta = vec![0.0f64; n_children];
        for c in 0..n_children {
            match children[c].delta(&self.footprint) {
                Some(d) => delta[c] = d,
                None => return,
            }
        }

        if !parent.in_footprint(&self.footprint) { return; }

        // 2^D − 1 non-trivial Haar wavelet coefficients via the
        // 2^D x 2^D transform matrix. Pattern 0 (constant) is excluded.
        // For pattern e ∈ {1, ..., 2^D − 1}:
        //     w_e = (1 / 2^D) Σ_σ (-1)^popcount(e & σ) δ_σ
        let scale = 1.0 / (n_children as f64);
        let parent_wr = parent.randoms.sum_w;
        let comp = self.cfg.compensated_sums;
        for e in 1..n_children {
            let mut acc = 0.0;
            for sigma in 0..n_children {
                let parity = ((e & sigma) as u32).count_ones() % 2;
                let sign = if parity == 0 { 1.0 } else { -1.0 };
                acc += sign * delta[sigma];
            }
            let w_e = acc * scale;
            let contrib = parent_wr * w_e * w_e;
            self.sum_wr_w2[level][e] += contrib;
            if comp {
                self.sum_wr_w2_comp[level][e].add(contrib);
            }
        }
        self.n_parents[level] += 1;
        self.sum_w_r_parents[level] += parent_wr;
        if comp {
            self.sum_w_r_parents_comp[level].add(parent_wr);
        }
    }
}

fn empty_density_stats<const D: usize>(
    pair: &BitVecCascadePair<D>, level: usize, cfg: &FieldStatsConfig,
) -> DensityFieldStats {
    let max_eff = pair.data.range.max_supported_l_max() as usize;
    let side_l = if level <= max_eff {
        (1u64 << (max_eff - level)) as f64
    } else { 1.0 };
    let edges: Vec<f64> = if cfg.hist_bins > 0 {
        (0..=cfg.hist_bins).map(|i| {
            cfg.hist_log_min
                + (cfg.hist_log_max - cfg.hist_log_min) * (i as f64) / (cfg.hist_bins as f64)
        }).collect()
    } else { Vec::new() };
    DensityFieldStats {
        level, cell_side_trimmed: side_l,
        n_cells_active: 0, sum_w_r_active: 0.0,
        mean_delta: 0.0, var_delta: 0.0, m3_delta: 0.0, m4_delta: 0.0,
        min_delta: 0.0, max_delta: 0.0, s3_delta: 0.0,
        n_cells_data_outside: 0, sum_w_d_outside: 0.0,
        hist_bin_edges: edges,
        hist_density: vec![0.0; cfg.hist_bins],
        hist_underflow_w_r: 0.0, hist_overflow_w_r: 0.0,
        raw_sum_w_r_delta_pow: [0.0; 4],
    }
}

fn empty_anisotropy<const D: usize>(
    pair: &BitVecCascadePair<D>, level: usize,
) -> AnisotropyStats {
    let max_eff = pair.data.range.max_supported_l_max() as usize;
    let side_l = if level <= max_eff {
        (1u64 << (max_eff - level)) as f64
    } else { 1.0 };
    AnisotropyStats {
        level,
        cell_side_trimmed: side_l,
        n_parents: 0,
        sum_w_r_parents: 0.0,
        mean_w_squared_by_pattern: vec![0.0; 1usize << D],
        mean_w_squared_axis: vec![0.0; D],
        quadrupole_los: 0.0,
        reduced_quadrupole_los: 0.0,
    }
}

// ============================================================================
// CicPmfVisitor: per-level histogram of integer cell counts
// ============================================================================

/// Visitor that builds a per-level histogram of `data.count` over visited
/// cells. In Periodic mode the unvisited (zero-data) cells are folded
/// into the k=0 density bin at finalization.
///
/// Ignores point weights — a histogram of integer object counts.
pub struct CicPmfVisitor<'a, const D: usize> {
    cfg: &'a CicPmfConfig,
    /// Whether to apply the box-volume correction at finalization.
    periodic_box_correction: bool,
    /// `histograms[l]` is the per-level cell-count histogram. We grow these
    /// as we encounter larger N values, capped at `cfg.max_bins`.
    histograms: Vec<Vec<u64>>,
    /// `n_visited[l]` mirrors `sum(histograms[l])`; tracked separately to
    /// avoid recomputation at finalization.
    n_visited: Vec<u64>,
}

impl<'a, const D: usize> CicPmfVisitor<'a, D> {
    pub fn new(pair: &BitVecCascadePair<D>, n_levels: usize, cfg: &'a CicPmfConfig) -> Self {
        Self {
            cfg,
            periodic_box_correction: pair.synthesizes_randoms_from_box(),
            histograms: vec![Vec::new(); n_levels],
            n_visited: vec![0u64; n_levels],
        }
    }

    pub fn into_results(self, pair: &BitVecCascadePair<D>) -> Vec<CicPmfStats> {
        let n_levels = self.histograms.len();
        let max_eff = pair.data.range.max_supported_l_max() as usize;
        let mut out = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let side_l = if l <= max_eff {
                (1u64 << (max_eff - l)) as f64
            } else { 1.0 };
            let depth_bits = D.saturating_mul(l);
            let n_total: u64 = if depth_bits >= 63 {
                // Practically unreachable, but keep arithmetic safe.
                u64::MAX
            } else {
                1u64 << depth_bits
            };

            let counts = self.histograms[l].clone();
            let visited = self.n_visited[l];

            // Density normalization: in Periodic mode use n_total (full box
            // tiling); in Isolated mode use visited (only the cascade-touched
            // cells form the sample).
            let denom = if self.periodic_box_correction { n_total } else { visited };

            let mut density: Vec<f64> = if denom == 0 {
                vec![0.0; counts.len()]
            } else {
                let denom_f = denom as f64;
                counts.iter().map(|&c| c as f64 / denom_f).collect()
            };

            // Periodic correction: unvisited cells all contribute to bin 0.
            if self.periodic_box_correction && n_total > visited {
                if density.is_empty() { density.push(0.0); }
                let missed = (n_total - visited) as f64;
                density[0] += missed / (n_total as f64);
            }

            // Cell-count moments. In Periodic mode we use the corrected
            // density (full-box mean/variance); in Isolated we use the
            // raw counts over visited cells.
            let (mean, var, skew, kurt) = compute_count_moments(&density);

            out.push(CicPmfStats {
                level: l,
                cell_side_trimmed: side_l,
                n_cells_visited: visited,
                n_cells_total: n_total,
                histogram_counts: counts,
                histogram_density: density,
                mean, var, skew, kurt,
            });
        }
        out
    }
}

impl<'a, const D: usize> crate::cascade_visitor::CascadeVisitor<D> for CicPmfVisitor<'a, D> {
    fn enter_cell(&mut self, cell: &crate::cascade_visitor::CellVisit<D>) {
        let level = cell.level;
        if level >= self.histograms.len() { return; }
        // Use raw integer count (ignores weights, by design).
        let n = cell.data.count as usize;
        // Cap bin index at max_bins - 1 to bound memory.
        let cap = self.cfg.max_bins.saturating_sub(1);
        let idx = n.min(cap);

        let h = &mut self.histograms[level];
        if idx >= h.len() {
            h.resize(idx + 1, 0);
        }
        h[idx] += 1;
        self.n_visited[level] += 1;
    }
}

/// Compute (mean, var, standardized skew, standardized kurtosis) of an
/// integer-count distribution from its density vector.
fn compute_count_moments(density: &[f64]) -> (f64, f64, f64, f64) {
    if density.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut total = 0.0;
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for (k, &p) in density.iter().enumerate() {
        if p == 0.0 { continue; }
        let x = k as f64;
        total += p;
        m1 += p * x;
        m2 += p * x * x;
        m3 += p * x * x * x;
        m4 += p * x * x * x * x;
    }
    if total <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = m1 / total;
    let var = (m2 / total - mean * mean).max(0.0);
    let mu3 = m3 / total - 3.0 * mean * (m2 / total) + 2.0 * mean.powi(3);
    let mu4 = m4 / total - 4.0 * mean * (m3 / total)
        + 6.0 * mean * mean * (m2 / total) - 3.0 * mean.powi(4);
    let sigma = var.sqrt();
    let skew = if sigma > 0.0 { mu3 / sigma.powi(3) } else { 0.0 };
    let kurt = if var > 0.0 { mu4 / (var * var) } else { 0.0 };
    (mean, var, skew, kurt)
}

// -- helpers ----------------------------------------------------------------

#[inline]
fn popcount_vec(v: &[u64]) -> u64 {
    let mut c: u64 = 0;
    for &w in v { c += w.count_ones() as u64; }
    c
}

#[inline]
fn mem_to_indices(mem: &[u64], expected_count: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(expected_count);
    for (w_idx, &word) in mem.iter().enumerate() {
        let mut w = word;
        let base = (w_idx * 64) as u32;
        while w != 0 {
            out.push(base + w.trailing_zeros());
            w &= w - 1;
        }
    }
    out
}

// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_range::CoordRange;

    fn make_uniform_2d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 2]> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| [next() & mask, next() & mask]).collect()
    }

    fn make_uniform_3d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| [next() & mask, next() & mask, next() & mask]).collect()
    }

    fn build_aligned<const D: usize>(
        d: Vec<[u64; D]>, r: Vec<[u64; D]>,
    ) -> (TrimmedPoints<D>, TrimmedPoints<D>) {
        let range = CoordRange::analyze_pair(&d, &r);
        (
            TrimmedPoints::from_points_with_range(d, range.clone()),
            TrimmedPoints::from_points_with_range(r, range),
        )
    }

    // ===========================================================================
    // BoundaryMode (Step 1: presence + defaults + setter only;
    // semantic differences come in Step 2 and beyond)
    // ===========================================================================

    #[test]
    fn boundary_mode_default_is_isolated() {
        // BoundaryMode::default() and Default::default() must agree on Isolated
        // so that adding the new field doesn't change behavior of any existing
        // code path or test fixture.
        assert_eq!(BoundaryMode::default(), BoundaryMode::Isolated);
    }

    #[test]
    fn build_defaults_to_isolated_boundary() {
        // The convenience `build` constructor must continue to give the
        // historical Isolated behavior so every existing test still holds.
        let pts_d = make_uniform_2d(100, 8, 1);
        let pts_r = make_uniform_2d(100, 8, 2);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        assert_eq!(pair.boundary_mode, BoundaryMode::Isolated);
    }

    #[test]
    fn build_full_defaults_to_isolated_boundary() {
        let pts_d = make_uniform_2d(100, 8, 3);
        let pts_r = make_uniform_2d(100, 8, 4);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<2>::build_full(td, tr, None, None, None, 64);
        assert_eq!(pair.boundary_mode, BoundaryMode::Isolated);
    }

    #[test]
    fn build_full_with_boundary_threads_choice() {
        let pts_d = make_uniform_2d(100, 8, 5);
        let pts_r = make_uniform_2d(100, 8, 6);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<2>::build_full_with_boundary(
            td, tr, None, None, None, 64, BoundaryMode::Periodic);
        assert_eq!(pair.boundary_mode, BoundaryMode::Periodic);
    }

    #[test]
    fn set_boundary_mode_returns_previous_and_swaps() {
        let pts_d = make_uniform_2d(50, 8, 7);
        let pts_r = make_uniform_2d(50, 8, 8);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let mut pair = BitVecCascadePair::<2>::build(td, tr, None);
        let prev = pair.set_boundary_mode(BoundaryMode::Periodic);
        assert_eq!(prev, BoundaryMode::Isolated);
        assert_eq!(pair.boundary_mode, BoundaryMode::Periodic);
        let prev2 = pair.set_boundary_mode(BoundaryMode::Isolated);
        assert_eq!(prev2, BoundaryMode::Periodic);
        assert_eq!(pair.boundary_mode, BoundaryMode::Isolated);
    }

    #[test]
    fn boundary_mode_does_not_yet_change_field_stats() {
        // Step 1 is non-semantic: setting BoundaryMode::Periodic on a pair
        // built with a real randoms catalog should not (yet) change any of
        // the analyze_* outputs. The semantic effect kicks in in Step 2,
        // when build_periodic + no-randoms-catalog mode arrives.
        let pts_d = make_sm_uniform_3d(800, 8, 100);
        let pts_r = make_sm_uniform_3d(2400, 8, 101);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };

        let mut pair = BitVecCascadePair::<3>::build(td, tr, None);
        let before = pair.analyze_field_stats(&cfg);
        pair.set_boundary_mode(BoundaryMode::Periodic);
        let after = pair.analyze_field_stats(&cfg);

        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a.mean_delta, b.mean_delta, "Step 1 must be non-semantic");
            assert_eq!(a.var_delta, b.var_delta);
            assert_eq!(a.m3_delta, b.m3_delta);
            assert_eq!(a.n_cells_active, b.n_cells_active);
        }
    }

    #[test]
    fn build_periodic_field_stats_mean_delta_is_zero() {
        // In Periodic mode without randoms, α comes from the box volume.
        // For uniform-Poisson data the mean δ at every level should be
        // statistically consistent with zero (it is exactly zero at the
        // root because the root cell IS the box and contains all data).
        let pts_d = make_sm_uniform_3d(2_000, 8, 42);
        let range = CoordRange::analyze_pair(&pts_d, &pts_d);  // self-pair → box-derived
        let td = TrimmedPoints::<3>::from_points_with_range(pts_d, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [8, 8, 8], None);
        assert_eq!(pair.boundary_mode, BoundaryMode::Periodic);
        assert_eq!(pair.n_r(), 0, "Periodic build should leave randoms empty");

        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);

        // Root: cell IS the box, contains all N points. α normalization
        // gives δ = sw_d / (α · sw_r) − 1 with sw_r = 1, α = ΣW_d
        // ⇒ δ_root = ΣW_d / ΣW_d − 1 = 0 exactly.
        assert!((stats[0].mean_delta).abs() < 1e-12,
            "level 0 (full-box) mean δ should be 0, got {}", stats[0].mean_delta);
        assert!((stats[0].var_delta).abs() < 1e-12,
            "level 0 var δ should be 0 (one-cell), got {}", stats[0].var_delta);
    }

    #[test]
    fn build_periodic_field_stats_clustered_gives_positive_variance() {
        // A clustered field in Periodic mode should give positive variance
        // at intermediate levels — the basic sanity check that δ is being
        // computed from the box-volume α and not silently zero.
        let mut s = 12345u64;
        let bits = 9;
        let max = (1u64 << bits) as i64;
        let n = 4_000;
        let n_clusters = 40;
        let center_max = (max as u64) - 64;
        let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| [
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
        ]).collect();
        let mut pts: Vec<[u64; 3]> = Vec::with_capacity(n);
        while pts.len() < n {
            let c = &centers[(sm64(&mut s) as usize) % n_clusters];
            let dx = ((sm64(&mut s) % 64) as i64) - 32;
            let dy = ((sm64(&mut s) % 64) as i64) - 32;
            let dz = ((sm64(&mut s) % 64) as i64) - 32;
            pts.push([
                ((c[0] as i64 + dx).rem_euclid(max)) as u64,
                ((c[1] as i64 + dy).rem_euclid(max)) as u64,
                ((c[2] as i64 + dz).rem_euclid(max)) as u64,
            ]);
        }
        let range = CoordRange::analyze_pair(&pts, &pts);
        let td = TrimmedPoints::<3>::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits, bits, bits], None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);

        // At intermediate scales (cluster scale) we should see substantial
        // variance — the clusters concentrate points into some cells.
        let mut found_clustering_signal = false;
        for st in &stats {
            if st.n_cells_active < 50 { continue; }
            // Cluster radius ~32 trimmed units. Levels with cell side
            // 16-128 should see the strongest clustering signal.
            // Threshold: var(δ) > 0.1 — well above uniform-Poisson shot noise.
            if st.cell_side_trimmed >= 16.0 && st.cell_side_trimmed <= 128.0
                && st.var_delta > 0.1 {
                found_clustering_signal = true;
                break;
            }
        }
        assert!(found_clustering_signal,
            "Expected substantial var(δ) at cluster scales in Periodic mode");
    }

    // ===========================================================================
    // Step 2: build_periodic + box-derived α
    // ===========================================================================

    #[test]
    fn build_periodic_sets_periodic_mode_and_empty_randoms() {
        let pts_d = make_sm_uniform_3d(500, 8, 200);
        let range = CoordRange::for_box_bits([8u32; 3]);
        let td = TrimmedPoints::from_points_with_range(pts_d, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [8u32; 3], None);
        assert_eq!(pair.boundary_mode, BoundaryMode::Periodic);
        assert_eq!(pair.n_d(), 500);
        assert_eq!(pair.n_r(), 0);
        // l_max should match the box bits, not the data extent.
        assert_eq!(pair.l_max, 8);
    }

    #[test]
    fn build_periodic_box_bits_define_resolution_independent_of_data() {
        // Even if all data lives in a small corner of the box, the cascade
        // should reach down to the full box resolution.
        let bits = 8u32;
        let pts: Vec<[u64; 3]> = (0..50).map(|i| {
            // All in the [0, 8)^3 sub-corner of a 256^3 box
            [(i % 8) as u64, ((i / 8) % 8) as u64, ((i / 64) % 8) as u64]
        }).collect();
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        assert_eq!(pair.l_max, 8, "box bits, not data extent, must drive l_max");
    }

    #[test]
    fn periodic_uniform_field_gives_mean_delta_zero() {
        // Uniform-random data filling the whole box, periodic mode, no
        // randoms catalog. The synthesized α=ΣW_d, sw_r=cell_volume should
        // give <δ> = 0 at every level (shot noise cancels in the mean).
        let bits = 8u32;
        let pts = make_sm_uniform_3d(20_000, bits, 300);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let cfg = FieldStatsConfig { w_r_min: 0.0, hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);

        for st in &stats {
            // The W_r-weighted mean is a sum over all cells weighted by
            // their volume — that's a fully sampled box-integral, so it
            // must equal 0 to floating-point precision (sum_d W_d / sum_d
            // (alpha * cell_vol) - 1 = 1 - 1 = 0).
            assert!(st.mean_delta.abs() < 1e-12,
                "level {}: mean_delta = {:.3e} (must be 0 for periodic uniform)",
                st.level, st.mean_delta);
        }
    }

    #[test]
    fn periodic_no_data_gives_zero_variance() {
        // Empty data + empty randoms with Periodic mode: walk should
        // short-circuit and produce a zero result, not panic.
        let pts: Vec<[u64; 3]> = vec![];
        let range = CoordRange::for_box_bits([6u32; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [6u32; 3], None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        for st in &stats {
            assert_eq!(st.n_cells_active, 0);
            assert_eq!(st.var_delta, 0.0);
        }
    }

    #[test]
    fn periodic_variance_increases_with_clustering() {
        // A clustered field has more variance than a uniform field,
        // monotonically with scale (at fixed N_d). Smoke test that the
        // periodic-mode visitor reports a sensible monotonic trend.
        let bits = 8u32;
        let n = 4_000;

        // Uniform field
        let pts_unif = make_sm_uniform_3d(n, bits, 400);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td_u = TrimmedPoints::from_points_with_range(pts_unif, range.clone());
        let pair_u = BitVecCascadePair::<3>::build_periodic(td_u, [bits; 3], None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats_u = pair_u.analyze_field_stats(&cfg);

        // Clustered field: 50 clusters, each ~80 points within a 16-wide cube
        let mut s = 401u64;
        let n_clusters = 50;
        let max = 1u64 << bits;
        let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| [
            sm64(&mut s) % max,
            sm64(&mut s) % max,
            sm64(&mut s) % max,
        ]).collect();
        let pts_clust: Vec<[u64; 3]> = (0..n).map(|_| {
            let c = &centers[(sm64(&mut s) as usize) % n_clusters];
            let dx = ((sm64(&mut s) % 32) as i64) - 16;
            let dy = ((sm64(&mut s) % 32) as i64) - 16;
            let dz = ((sm64(&mut s) % 32) as i64) - 16;
            [
                ((c[0] as i64 + dx).rem_euclid(max as i64)) as u64,
                ((c[1] as i64 + dy).rem_euclid(max as i64)) as u64,
                ((c[2] as i64 + dz).rem_euclid(max as i64)) as u64,
            ]
        }).collect();
        let td_c = TrimmedPoints::from_points_with_range(pts_clust, range);
        let pair_c = BitVecCascadePair::<3>::build_periodic(td_c, [bits; 3], None);
        let stats_c = pair_c.analyze_field_stats(&cfg);

        // At cluster scales (cell side ~16-64 in a 256^3 box with cluster
        // radius ~16) the clustered field's variance should clearly exceed
        // the uniform's. At very fine scales both are shot-noise dominated;
        // at very coarse scales clustering averages away. Restrict to the
        // band where the contrast is expected to be visible.
        let mut had_clear_contrast = false;
        let mut any_level_compared = false;
        for (u, c) in stats_u.iter().zip(stats_c.iter()) {
            let side = u.cell_side_trimmed;
            if !(16.0..=64.0).contains(&side) { continue; }
            if u.n_cells_active < 50 || c.n_cells_active < 50 { continue; }
            if u.var_delta < 1e-3 { continue; }
            any_level_compared = true;
            if c.var_delta > u.var_delta * 1.5 {
                had_clear_contrast = true;
            }
        }
        assert!(any_level_compared,
            "no level in side range [16,64] had usable signal");
        assert!(had_clear_contrast,
            "expected at least one level in [16,64] with clustered var > 1.5x uniform var");
    }

    #[test]
    fn periodic_matches_isolated_with_uniform_randoms() {
        // The whole point of Periodic mode: it should be equivalent to
        // running Isolated mode with a fully populated uniform random
        // catalog of the same box, in the limit of infinitely many randoms.
        // We use a finite but heavy random catalog and check that the
        // means agree to within shot noise.
        let bits = 7u32;  // 128^3 box; modest size for test speed
        let n_d = 5_000;
        let pts_d = make_sm_uniform_3d(n_d, bits, 500);

        // Periodic-mode (no randoms) result
        let range = CoordRange::for_box_bits([bits; 3]);
        let td_p = TrimmedPoints::from_points_with_range(pts_d.clone(), range.clone());
        let pair_p = BitVecCascadePair::<3>::build_periodic(td_p, [bits; 3], None);
        let cfg = FieldStatsConfig { w_r_min: 0.0, hist_bins: 0, ..Default::default() };
        let stats_p = pair_p.analyze_field_stats(&cfg);

        // Isolated-mode with heavy uniform random catalog
        let pts_r = make_sm_uniform_3d(n_d * 100, bits, 501);
        let td_i = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr_i = TrimmedPoints::from_points_with_range(pts_r, range);
        let pair_i = BitVecCascadePair::<3>::build(td_i, tr_i, None);
        let stats_i = pair_i.analyze_field_stats(&cfg);

        // Both should have the same number of levels.
        assert_eq!(stats_p.len(), stats_i.len());

        // At coarse-to-mid levels (where cells contain many points),
        // the two estimators should agree closely on mean δ (≈ 0) and
        // on the variance, modulo shot noise from the finite randoms.
        for (p, i) in stats_p.iter().zip(stats_i.iter()) {
            // Skip the finest levels where shot noise dominates and the
            // few-random-points-per-cell case kicks in.
            if p.level >= bits as usize - 1 { continue; }
            if i.n_cells_active < 50 { continue; }
            // Mean δ: periodic must be 0 exactly; isolated must be O(shot noise)
            assert!(p.mean_delta.abs() < 1e-12, "periodic mean δ at l={}", p.level);
            assert!(i.mean_delta.abs() < 0.05, "isolated mean δ at l={}", p.level);
            // Variance: should agree to within ~30% (shot noise). Test only
            // levels with non-trivial signal.
            if p.var_delta < 1e-4 { continue; }
            let rel_diff = (p.var_delta - i.var_delta).abs() / p.var_delta.max(i.var_delta);
            assert!(rel_diff < 0.30,
                "level {}: var(periodic)={:.3e} var(isolated)={:.3e} \
                 rel diff {:.2}", p.level, p.var_delta, i.var_delta, rel_diff);
        }
    }

    // ===========================================================================
    // Step 3: cascade-based CIC PMF
    // ===========================================================================

    #[test]
    fn cic_pmf_periodic_density_sums_to_one() {
        let bits = 7u32;
        let pts = make_sm_uniform_3d(2_000, bits, 600);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let cfg = CicPmfConfig::default();
        let stats = pair.analyze_cic_pmf(&cfg);
        assert!(!stats.is_empty(), "expected per-level results");
        for st in &stats {
            let s: f64 = st.histogram_density.iter().sum();
            assert!((s - 1.0).abs() < 1e-12,
                "level {}: density sum = {} (must be exactly 1 in periodic)",
                st.level, s);
        }
    }

    #[test]
    fn cic_pmf_periodic_mean_is_exact_n_per_cell() {
        // In Periodic mode the cascade tiles the full box with 2^(D*l) cells
        // per level. The mean cell count must equal n_d / n_total exactly,
        // because every data point falls in exactly one cell.
        let bits = 7u32;
        let n_d = 3_000usize;
        let pts = make_sm_uniform_3d(n_d, bits, 700);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &stats {
            let expected = (n_d as f64) / (st.n_cells_total as f64);
            assert!((st.mean - expected).abs() < 1e-9,
                "level {}: mean={} expected={} (n_d={}, n_total={})",
                st.level, st.mean, expected, n_d, st.n_cells_total);
        }
    }

    #[test]
    fn cic_pmf_periodic_no_data_concentrates_at_zero() {
        // Empty data + periodic mode: every cell is unvisited → density[0] = 1.
        let pts: Vec<[u64; 3]> = vec![];
        let range = CoordRange::for_box_bits([6u32; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [6u32; 3], None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &stats {
            assert_eq!(st.n_cells_visited, 0);
            assert!(!st.histogram_density.is_empty(),
                "periodic empty case must produce a [1.0, ...] density not an empty vec");
            assert!((st.histogram_density[0] - 1.0).abs() < 1e-15,
                "level {}: density[0]={} (expected 1.0)", st.level, st.histogram_density[0]);
            assert_eq!(st.mean, 0.0);
            assert_eq!(st.var, 0.0);
        }
    }

    #[test]
    fn cic_pmf_periodic_root_level_concentrates_at_n_d() {
        // At level 0 the only cell is the whole box → histogram density
        // must concentrate at k = n_d with probability 1.
        let bits = 8u32;
        let n_d = 1_500usize;
        let pts = make_sm_uniform_3d(n_d, bits, 800);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());
        let root = &stats[0];
        assert_eq!(root.n_cells_total, 1);
        assert_eq!(root.n_cells_visited, 1);
        // density[k] = 1 for k = n_d, 0 elsewhere
        assert!(root.histogram_density.len() > n_d,
            "root histogram needs to span up to n_d={}", n_d);
        for (k, &p) in root.histogram_density.iter().enumerate() {
            if k == n_d {
                assert!((p - 1.0).abs() < 1e-15, "p[{}] should be 1, got {}", k, p);
            } else {
                assert_eq!(p, 0.0, "p[{}] should be 0, got {}", k, p);
            }
        }
        assert_eq!(root.mean, n_d as f64);
        assert!(root.var.abs() < 1e-9, "var at root = {}, must be 0", root.var);
    }

    #[test]
    fn cic_pmf_isolated_density_sums_to_one_over_visited() {
        // In Isolated mode the density is normalized over visited cells only,
        // because the cascade only descends into cells with at least one data
        // or random point. So density sums to 1 over visited cells (regardless
        // of how many cells were never touched).
        let pts_d = make_sm_uniform_3d(800, 8, 900);
        let pts_r = make_sm_uniform_3d(2_400, 8, 901);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &stats {
            if st.n_cells_visited == 0 {
                assert!(st.histogram_density.is_empty(),
                    "no-visited isolated should yield empty density (no inference)");
                continue;
            }
            let s: f64 = st.histogram_density.iter().sum();
            assert!((s - 1.0).abs() < 1e-12,
                "level {}: isolated density sum = {} (expected 1.0)", st.level, s);
        }
    }

    #[test]
    fn cic_pmf_visited_count_equals_sum_of_histogram_counts() {
        let pts_d = make_sm_uniform_3d(500, 7, 1000);
        let pts_r = make_sm_uniform_3d(1500, 7, 1001);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &stats {
            let sum: u64 = st.histogram_counts.iter().sum();
            assert_eq!(st.n_cells_visited, sum,
                "level {}: n_cells_visited={} but histogram_counts sum={}",
                st.level, st.n_cells_visited, sum);
        }
    }

    #[test]
    fn cic_pmf_cascade_mean_matches_dense_grid_at_dyadic_sides() {
        // The cascade-based PMF (cells tiling the box) and the dense
        // sliding-window PMF (all M^D origins) measure related but
        // distinct distributions. Their MEANS must agree exactly:
        // each is total_n_d * (cell_volume / box_volume), regardless of
        // how the windows are placed.
        use crate::hier_nd::cascade_with_pmf_windows;
        let bits = 6u32;
        let n_d = 800usize;
        let pts_u16: Vec<[u16; 3]> = (0..n_d).map(|i| {
            let mut s = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let m = 1u64 << bits;
            // two short rounds of mixing
            s ^= s >> 33; s = s.wrapping_mul(0xff51afd7ed558ccd); s ^= s >> 33;
            let x = (s % m) as u16;
            s ^= s >> 33; s = s.wrapping_mul(0xc4ceb9fe1a85ec53); s ^= s >> 33;
            let y = (s % m) as u16;
            s ^= s >> 33; s = s.wrapping_mul(0xff51afd7ed558ccd); s ^= s >> 33;
            let z = (s % m) as u16;
            [x, y, z]
        }).collect();

        // Dense sliding-window PMF: all sides 1, 2, 4, 8, 16, 32, 64
        let m = 1usize << bits;
        let sides: Vec<usize> = (0..=bits).map(|l| 1usize << l).collect();
        let dense = cascade_with_pmf_windows::<3>(
            &pts_u16, bits as usize, 0, true, &sides);

        // Cascade PMF
        let pts_u64: Vec<[u64; 3]> = pts_u16.iter().map(|p|
            [p[0] as u64, p[1] as u64, p[2] as u64]).collect();
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts_u64, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let cascade = pair.analyze_cic_pmf(&CicPmfConfig::default());

        // Cascade level l ↔ side 2^(L_max-l). Cross-reference by side.
        for st in &cascade {
            let cascade_side = st.cell_side_trimmed as usize;
            if cascade_side > m { continue; }
            // Find the dense entry with the same side
            let dense_entry = dense.iter().find(|d| d.window_side == cascade_side);
            let dense_entry = match dense_entry { Some(d) => d, None => continue };
            // Means must agree exactly: both are n_d * (side^D / m^D)
            let expected = (n_d as f64) * (cascade_side as f64).powi(3) / (m as f64).powi(3);
            assert!((st.mean - expected).abs() < 1e-9,
                "side {}: cascade mean={} expected={}", cascade_side, st.mean, expected);
            assert!((dense_entry.mean - expected).abs() < 1e-9,
                "side {}: dense mean={} expected={}", cascade_side, dense_entry.mean, expected);
        }
    }

    #[test]
    fn cic_pmf_works_in_2d_and_4d() {
        // 2D
        let pts2: Vec<[u64; 2]> = (0..200).map(|i|
            [(i as u64 * 7) % 256, (i as u64 * 13) % 256]).collect();
        let range2 = CoordRange::for_box_bits([8u32; 2]);
        let td2 = TrimmedPoints::from_points_with_range(pts2, range2);
        let pair2 = BitVecCascadePair::<2>::build_periodic(td2, [8u32; 2], None);
        let s2 = pair2.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &s2 {
            assert_eq!(st.n_cells_total, 1u64 << (2 * st.level));
            let s: f64 = st.histogram_density.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }

        // 4D
        let pts4: Vec<[u64; 4]> = (0..200).map(|i| {
            let i = i as u64;
            [(i * 7) % 32, (i * 13) % 32, (i * 17) % 32, (i * 19) % 32]
        }).collect();
        let range4 = CoordRange::for_box_bits([5u32; 4]);
        let td4 = TrimmedPoints::from_points_with_range(pts4, range4);
        let pair4 = BitVecCascadePair::<4>::build_periodic(td4, [5u32; 4], None);
        let s4 = pair4.analyze_cic_pmf(&CicPmfConfig::default());
        for st in &s4 {
            assert_eq!(st.n_cells_total, 1u64 << (4 * st.level));
            let s: f64 = st.histogram_density.iter().sum();
            assert!((s - 1.0).abs() < 1e-12,
                "4D level {}: density sum = {} (expected 1.0)", st.level, s);
        }
    }

    #[test]
    fn cic_pmf_periodic_uniform_approaches_multinomial_poisson_limit() {
        // For a uniform-random catalog the cascade PMF at level l with K =
        // 2^(D*l) cells is exactly multinomial: N points distributed among
        // K cells uniformly. This gives var/mean = 1 - 1/K. For K large
        // the limit is Poisson (var/mean → 1); for small K the variance
        // is suppressed by 1/K.
        //
        // Test: at levels with K >= 1024 AND mean count between 1 and 100,
        // var/mean must be within 5% of (1 - 1/K). Excludes coarse levels
        // (small K, large fluctuations) and very fine levels (μ << 1,
        // where shot noise dominates the *ratio* of two small numbers).
        let bits = 8u32;
        let n_d = 20_000usize;
        let pts = make_sm_uniform_3d(n_d, bits, 1100);
        let range = CoordRange::for_box_bits([bits; 3]);
        let td = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build_periodic(td, [bits; 3], None);
        let stats = pair.analyze_cic_pmf(&CicPmfConfig::default());

        let mut tested = 0;
        for st in &stats {
            if st.n_cells_total < 1024 { continue; }
            if !(1.0..=100.0).contains(&st.mean) { continue; }
            let k = st.n_cells_total as f64;
            let expected_ratio = 1.0 - 1.0 / k;
            let observed = st.var / st.mean;
            assert!((observed - expected_ratio).abs() < 0.05,
                "level {} (K={}, μ={:.2}): var/mean = {:.4}, \
                 multinomial expectation = {:.4}",
                st.level, st.n_cells_total, st.mean, observed, expected_ratio);
            tested += 1;
        }
        assert!(tested >= 1, "expected at least one level in the multinomial-Poisson regime");
    }


    #[test]
    fn dd_matches_single_catalog_cascade() {
        // The pair cascade's cumulative DD must equal what we get from the
        // single-catalog cascade run on data alone.
        use crate::hier_bitvec::BitVecCascade;
        let pts_d = make_uniform_2d(300, 10, 7);
        let pts_r = make_uniform_2d(500, 10, 11);

        // Single-catalog DD on data alone. To compare we need the SAME range
        // so cells line up; otherwise effective_bits could differ between the
        // two trim approaches.
        let (td, tr) = build_aligned(pts_d.clone(), pts_r.clone());
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let pstats = pair.analyze();

        let single_t = TrimmedPoints::from_points_with_range(
            pts_d, pair.data.range.clone());
        let single = BitVecCascade::<2>::build(single_t, Some(pair.l_max));
        let sstats = single.analyze();

        for l in 0..pstats.len() {
            assert_eq!(pstats[l].cumulative_dd as u64,
                       sstats[l].cumulative_pairs,
                       "level {}: pair DD = {}, single = {}",
                       l, pstats[l].cumulative_dd, sstats[l].cumulative_pairs);
        }
    }

    #[test]
    fn rr_matches_single_catalog_cascade() {
        use crate::hier_bitvec::BitVecCascade;
        let pts_d = make_uniform_2d(200, 10, 41);
        let pts_r = make_uniform_2d(400, 10, 43);

        let (td, tr) = build_aligned(pts_d.clone(), pts_r.clone());
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let pstats = pair.analyze();

        let single_t = TrimmedPoints::from_points_with_range(
            pts_r, pair.data.range.clone());
        let single = BitVecCascade::<2>::build(single_t, Some(pair.l_max));
        let sstats = single.analyze();

        for l in 0..pstats.len() {
            assert_eq!(pstats[l].cumulative_rr as u64,
                       sstats[l].cumulative_pairs,
                       "level {} RR mismatch", l);
        }
    }

    #[test]
    fn dr_matches_brute_force_2d() {
        // For small enough catalogs, brute-force enumerate all D×R cross pairs
        // sharing a cell at every level.
        let n_d = 30;
        let n_r = 50;
        let pts_d = make_uniform_2d(n_d, 8, 99);
        let pts_r = make_uniform_2d(n_r, 8, 100);

        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let stats = pair.analyze();
        let eff = pair.data.range.effective_bits;

        for l in 0..stats.len() {
            let mut bf = 0u64;
            for i in 0..pair.n_d() {
                for j in 0..pair.n_r() {
                    let mut same = true;
                    for d in 0..2 {
                        let e = eff[d] as usize;
                        let l_use = l.min(e);
                        let shift = if l_use >= e { 0 } else { e - l_use };
                        let ci = pair.data.points[i][d] >> shift;
                        let cj = pair.randoms.points[j][d] >> shift;
                        if ci != cj { same = false; break; }
                    }
                    if same { bf += 1; }
                }
            }
            assert_eq!(stats[l].cumulative_dr as u64, bf,
                "level {}: cascade DR = {}, brute = {}",
                l, stats[l].cumulative_dr, bf);
        }
    }

    #[test]
    fn dr_matches_brute_force_3d() {
        let n_d = 25;
        let n_r = 40;
        let pts_d = make_uniform_3d(n_d, 7, 13);
        let pts_r = make_uniform_3d(n_r, 7, 17);

        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let stats = pair.analyze();
        let eff = pair.data.range.effective_bits;

        for l in 0..stats.len() {
            let mut bf = 0u64;
            for i in 0..pair.n_d() {
                for j in 0..pair.n_r() {
                    let mut same = true;
                    for d in 0..3 {
                        let e = eff[d] as usize;
                        let l_use = l.min(e);
                        let shift = if l_use >= e { 0 } else { e - l_use };
                        if (pair.data.points[i][d] >> shift)
                            != (pair.randoms.points[j][d] >> shift) {
                            same = false;
                            break;
                        }
                    }
                    if same { bf += 1; }
                }
            }
            assert_eq!(stats[l].cumulative_dr as u64, bf,
                "level {}: 3D DR cascade={}, brute={}", l,
                stats[l].cumulative_dr, bf);
        }
    }

    #[test]
    fn shells_sum_to_total_pairs() {
        let pts_d = make_uniform_2d(150, 10, 1);
        let pts_r = make_uniform_2d(200, 10, 2);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let dd_total: f64 = shells.iter().map(|s| s.dd).sum();
        let rr_total: f64 = shells.iter().map(|s| s.rr).sum();
        let dr_total: f64 = shells.iter().map(|s| s.dr).sum();
        assert!((dd_total - 150.0 * 149.0 / 2.0).abs() < 1e-6);
        assert!((rr_total - 200.0 * 199.0 / 2.0).abs() < 1e-6);
        assert!((dr_total - 150.0 * 200.0).abs() < 1e-6);
    }

    #[test]
    fn ls_xi_near_zero_for_matched_uniform() {
        // Same-distribution data and randoms (independent draws from the same
        // uniform distribution) should give ξ ≈ 0 within sample noise across
        // most shells. We check the sum of |ξ| weighted by RR is small relative
        // to the per-shell noise floor.
        let n = 4000;
        let pts_d = make_uniform_3d(n, 10, 1234);
        let pts_r = make_uniform_3d(n, 10, 5678);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);

        // For shells with reasonable pair counts, |ξ| should be O(1/sqrt(RR)).
        // We just sanity-check the mid-range shells have small ξ.
        let n_levels = shells.len();
        let mid_start = n_levels / 4;
        let mid_end = n_levels.saturating_sub(2);
        let mut max_abs = 0.0f64;
        let mut min_rr_in_mid = f64::INFINITY;
        for s in &shells[mid_start..mid_end.max(mid_start + 1)] {
            if s.rr > 100.0 {
                if s.xi_ls.abs() > max_abs { max_abs = s.xi_ls.abs(); }
                if s.rr < min_rr_in_mid { min_rr_in_mid = s.rr; }
            }
        }
        // For matched Poisson the standard noise scale on ξ is roughly
        // 1/sqrt(min_rr). We allow a generous 5x margin.
        if min_rr_in_mid.is_finite() {
            let bound = 5.0 / min_rr_in_mid.sqrt();
            assert!(max_abs < bound.max(0.1),
                "matched-uniform ξ_max = {}, bound {}", max_abs, bound);
        }
    }

    #[test]
    fn unit_weights_match_unweighted() {
        // A pair cascade with unit weights must give the same DD/RR/DR as one
        // built without weights.
        let pts_d = make_uniform_2d(120, 9, 21);
        let pts_r = make_uniform_2d(180, 9, 22);

        let (td_a, tr_a) = build_aligned(pts_d.clone(), pts_r.clone());
        let pair_a = BitVecCascadePair::<2>::build(td_a, tr_a, None);
        let stats_a = pair_a.analyze();

        let (td_b, tr_b) = build_aligned(pts_d, pts_r);
        let wd = vec![1.0; pair_a.n_d()];
        let wr = vec![1.0; pair_a.n_r()];
        let pair_b = BitVecCascadePair::<2>::build_full(
            td_b, tr_b, Some(wd), Some(wr), None, 64);
        let stats_b = pair_b.analyze();

        for l in 0..stats_a.len() {
            assert!((stats_a[l].cumulative_dd - stats_b[l].cumulative_dd).abs() < 1e-6,
                "level {} DD differs", l);
            assert!((stats_a[l].cumulative_rr - stats_b[l].cumulative_rr).abs() < 1e-6,
                "level {} RR differs", l);
            assert!((stats_a[l].cumulative_dr - stats_b[l].cumulative_dr).abs() < 1e-6,
                "level {} DR differs", l);
        }
    }

    #[test]
    fn weighted_dd_matches_brute_force() {
        // For small N, brute-force the weighted DD sum and compare.
        let n_d = 25;
        let pts_d = make_uniform_2d(n_d, 8, 7);
        let pts_r = make_uniform_2d(15, 8, 8);
        let mut s = 12345u64;
        let weights: Vec<f64> = (0..n_d).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64) / (1u64 << 31) as f64
        }).collect();

        let (td, tr) = build_aligned(pts_d.clone(), pts_r);
        let nd_for_w = td.points.len();
        let pair = BitVecCascadePair::<2>::build_full(
            td, tr, Some(weights.clone()), None, None, 64);
        assert_eq!(nd_for_w, weights.len());
        let stats = pair.analyze();
        let eff = pair.data.range.effective_bits;

        for l in 0..stats.len() {
            let mut bf = 0.0f64;
            for i in 0..pair.n_d() {
                for j in (i+1)..pair.n_d() {
                    let mut same = true;
                    for d in 0..2 {
                        let e = eff[d] as usize;
                        let l_use = l.min(e);
                        let shift = if l_use >= e { 0 } else { e - l_use };
                        if (pair.data.points[i][d] >> shift)
                            != (pair.data.points[j][d] >> shift) {
                            same = false; break;
                        }
                    }
                    if same { bf += weights[i] * weights[j]; }
                }
            }
            assert!((stats[l].cumulative_dd - bf).abs() < 1e-9,
                "level {}: weighted cascade DD = {}, brute = {}",
                l, stats[l].cumulative_dd, bf);
        }
    }

    #[test]
    fn ls_xi_recovers_clustering_for_clustered_data() {
        // Place data points clustered into a few small blobs, with uniform
        // randoms. ξ at small scales should be substantially positive.
        let n_clusters = 40;
        let pts_per_cluster = 25;
        let bits = 10;
        let max = 1u64 << bits;

        // PRNG note: LCGs have very short cycles in their LOW bits (the low k
        // bits cycle with period ≤ 2^k). Naively masking with `s & mask` for
        // k=10 gives only 1024 distinct values across thousands of draws,
        // collapsing 3D "random" points onto a tiny lattice. We extract the
        // top `bits` of each LCG output instead, where the period is full.
        let mut s = 9999u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let draw = |next: &mut dyn FnMut() -> u64, modulus: u64| -> u64 {
            // Take high bits, then reduce modulo to handle non-power-of-two moduli.
            (next() >> 32) % modulus
        };
        let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| {
            [draw(&mut next, max), draw(&mut next, max), draw(&mut next, max)]
        }).collect();

        // Each cluster: gaussian-ish ball of radius ~ max/64
        let mut data: Vec<[u64; 3]> = Vec::new();
        for c in &centers {
            for _ in 0..pts_per_cluster {
                let dx = draw(&mut next, max / 32) as i64 - (max / 64) as i64;
                let dy = draw(&mut next, max / 32) as i64 - (max / 64) as i64;
                let dz = draw(&mut next, max / 32) as i64 - (max / 64) as i64;
                let x = ((c[0] as i64 + dx).rem_euclid(max as i64)) as u64;
                let y = ((c[1] as i64 + dy).rem_euclid(max as i64)) as u64;
                let z = ((c[2] as i64 + dz).rem_euclid(max as i64)) as u64;
                data.push([x, y, z]);
            }
        }
        let n_d = data.len();
        let randoms: Vec<[u64; 3]> = (0..n_d * 3).map(|_| {
            [draw(&mut next, max), draw(&mut next, max), draw(&mut next, max)]
        }).collect();

        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);

        // At small scales (deep levels) ξ should be strongly positive.
        // The cluster size is ~max/32 so the shell at side ~max/32 (level
        // log2(32) = 5) should show clear excess.
        let mut max_xi_small_scale = f64::NEG_INFINITY;
        for s in &shells {
            // small scale = side <= max/16
            if s.cell_side_trimmed <= (max / 16) as f64 && s.cell_side_trimmed >= 1.0 {
                if s.rr > 10.0 && s.xi_ls > max_xi_small_scale {
                    max_xi_small_scale = s.xi_ls;
                }
            }
        }
        assert!(max_xi_small_scale > 1.0,
            "expected strong positive ξ at small scales; got max = {}",
            max_xi_small_scale);
    }

    #[test]
    fn crossover_thresholds_agree() {
        // The pair cascade with threshold = 0 (immediate point-list) and
        // threshold = huge (pure bit-vec) must produce identical cumulative
        // DD/DR/RR.
        let pts_d = make_uniform_3d(150, 9, 100);
        let pts_r = make_uniform_3d(200, 9, 200);

        let (td_a, tr_a) = build_aligned(pts_d.clone(), pts_r.clone());
        let pair_a = BitVecCascadePair::<3>::build_full(
            td_a, tr_a, None, None, None, 0);
        let stats_a = pair_a.analyze();

        let (td_b, tr_b) = build_aligned(pts_d, pts_r);
        let pair_b = BitVecCascadePair::<3>::build_full(
            td_b, tr_b, None, None, None, 100_000);
        let stats_b = pair_b.analyze();

        for l in 0..stats_a.len() {
            assert!((stats_a[l].cumulative_dd - stats_b[l].cumulative_dd).abs() < 1e-9,
                "level {} DD differs across thresholds", l);
            assert!((stats_a[l].cumulative_rr - stats_b[l].cumulative_rr).abs() < 1e-9,
                "level {} RR differs across thresholds", l);
            assert!((stats_a[l].cumulative_dr - stats_b[l].cumulative_dr).abs() < 1e-9,
                "level {} DR differs across thresholds", l);
        }
    }

    #[test]
    fn empty_data_or_random_is_safe() {
        // Construct catalogs where one is empty. Should not panic.
        let pts_d: Vec<[u64; 2]> = vec![[10, 20], [30, 40]];
        let pts_r: Vec<[u64; 2]> = Vec::new();
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let stats = pair.analyze();
        // RR should be all zero
        for s in &stats {
            assert_eq!(s.cumulative_rr, 0.0);
            assert_eq!(s.cumulative_dr, 0.0);
        }
        // DD at root = N(N-1)/2 = 1
        assert_eq!(stats[0].cumulative_dd, 1.0);
    }

    #[test]
    fn pair_default_crossover_uses_max_of_n_d_n_r() {
        // The pair cascade cost is set by the larger catalog. Confirm the
        // default threshold scales with max(n_d, n_r).
        assert_eq!(BitVecCascadePair::<3>::default_crossover_threshold(100, 100), 64);
        assert_eq!(BitVecCascadePair::<3>::default_crossover_threshold(100, 100_000), 100_000 / 64);
        assert_eq!(BitVecCascadePair::<3>::default_crossover_threshold(100_000, 100), 100_000 / 64);
        assert_eq!(BitVecCascadePair::<3>::default_crossover_threshold(0, 0), 64);
    }

    // ============================================================================
    // Density-field statistics — δ = W_d/(α W_r) − 1, W_r-weighted moments
    // ============================================================================

    /// SplitMix64 — high-quality PRNG, no LCG low-bit pathology.
    fn sm64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn make_sm_uniform_3d(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| {
            [sm64(&mut s) & mask, sm64(&mut s) & mask, sm64(&mut s) & mask]
        }).collect()
    }

    #[test]
    fn field_stats_data_equals_randoms_gives_zero_delta() {
        // If data is literally the same point set as randoms, then for every
        // cell W_d = W_r, α = 1, and δ = 0 everywhere. All central moments
        // must be exactly 0.
        let pts = make_sm_uniform_3d(500, 10, 1234);
        let (td, tr) = build_aligned(pts.clone(), pts);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        for (l, st) in stats.iter().enumerate() {
            // All cells with W_r > 0 will also have W_d > 0 (same set), so
            // δ = (W_r/W_r) − 1 = 0. Mean, var, m3, m4 all zero exactly.
            assert!(st.mean_delta.abs() < 1e-12,
                "level {}: mean_delta = {}", l, st.mean_delta);
            assert!(st.var_delta < 1e-20,
                "level {}: var_delta = {}", l, st.var_delta);
            assert!(st.m3_delta.abs() < 1e-20,
                "level {}: m3_delta = {}", l, st.m3_delta);
            assert!(st.m4_delta < 1e-20,
                "level {}: m4_delta = {}", l, st.m4_delta);
        }
    }

    #[test]
    fn field_stats_mean_delta_zero_at_root() {
        // With one cell containing everything (level 0), W_d_root = ΣW_d
        // and W_r_root = ΣW_r so δ = α/α − 1 = 0 exactly. This is the
        // algebraic identity that the global α normalization buys us.
        let data = make_sm_uniform_3d(800, 10, 4242);
        let randoms = make_sm_uniform_3d(2000, 10, 9999);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        // At level 0 there is exactly one cell, so δ_root = 0 and mean=var=0.
        assert_eq!(stats[0].n_cells_active, 1);
        assert!(stats[0].mean_delta.abs() < 1e-12,
            "level 0 mean_delta = {} (should be ~0)", stats[0].mean_delta);
        assert!(stats[0].var_delta < 1e-20,
            "level 0 var_delta = {} (single-cell trivially 0)", stats[0].var_delta);
    }

    #[test]
    fn field_stats_mean_delta_zero_at_level_zero_only() {
        // The W_r-weighted mean of δ is exactly 0 at the root by algebraic
        // identity (ΣW_d / α = ΣW_r). At deeper levels, cells where data
        // is present but randoms are absent (n_d > 0, n_r = 0) get excluded
        // from the moment sums (no W_r ⇒ δ undefined), so the global α
        // normalization no longer balances and <δ> drifts negative
        // (intuition: the "absent data" cells near randoms contribute δ=-1
        // and aren't compensated by the missing positive contributions).
        // For sparse data this drift can be order unity at fine scales —
        // it's a real diagnostic of the catalog, not a bug.
        let data = make_sm_uniform_3d(1000, 10, 100);
        let randoms = make_sm_uniform_3d(3000, 10, 200);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        // Hard assertion: level 0 mean is exactly 0 (single cell).
        assert!(stats[0].mean_delta.abs() < 1e-12,
            "level 0: mean_delta = {} (must be 0 exactly at the root)",
            stats[0].mean_delta);
        // Soft assertion: level 1 mean stays small (8 cells, most have both
        // data and randoms at this resolution).
        assert!(stats[1].mean_delta.abs() < 0.05,
            "level 1: mean_delta = {} (small N, mostly populated cells)",
            stats[1].mean_delta);
    }

    #[test]
    fn field_stats_var_grows_then_shot_noise_dominates() {
        // For matched-density uniform data + randoms, the only signal in δ
        // is shot noise. var(δ) should grow from ~0 at the root toward
        // larger values at intermediate scales. At very fine scales most
        // cells contain just 0 or 1 random point and the metric becomes
        // degenerate (cells are either δ = -1 or unobserved), so we restrict
        // the trend test to the well-behaved scale range.
        let data = make_sm_uniform_3d(8000, 10, 7);
        let randoms = make_sm_uniform_3d(24000, 10, 8);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        // Level 0: var = 0 exactly.
        assert!(stats[0].var_delta < 1e-20,
            "level 0 var should be 0; got {}", stats[0].var_delta);
        // Find a "well-behaved" range: levels where mean random count per
        // active cell is at least ~3 (so δ has multiple discrete values
        // available rather than just {-1, very_large}). For 24000 randoms
        // in 3D box, mean per cell at level l = 24000 / 8^l, so >= 3 needs
        // 8^l <= 8000, i.e. l <= 4.
        let var_l1 = stats[1].var_delta;
        let var_l4 = stats[4].var_delta;
        assert!(var_l1 > 0.0 && var_l4 > var_l1,
            "var should grow from level 1 to level 4 (shot noise rising); \
             var(1) = {} var(4) = {}", var_l1, var_l4);
    }

    #[test]
    fn field_stats_w_r_min_excludes_low_random_cells() {
        // With w_r_min set high enough to exclude every cell, n_cells_active
        // should be 0 and all moments 0.
        let data = make_sm_uniform_3d(200, 8, 11);
        let randoms = make_sm_uniform_3d(400, 8, 12);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig {
            w_r_min: 1e18,    // larger than any plausible cell W_r sum
            hist_bins: 0,
            ..Default::default()
        };
        let stats = pair.analyze_field_stats(&cfg);
        for st in &stats {
            assert_eq!(st.n_cells_active, 0,
                "with w_r_min huge, no cells should pass; got {} active at level {}",
                st.n_cells_active, st.level);
            assert_eq!(st.mean_delta, 0.0);
            assert_eq!(st.var_delta, 0.0);
        }
    }

    #[test]
    fn field_stats_histogram_normalizes_to_one() {
        // hist_density values plus underflow + overflow should sum to 1 at
        // every level with active cells.
        let data = make_sm_uniform_3d(500, 10, 33);
        let randoms = make_sm_uniform_3d(1500, 10, 34);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0,
            hist_bins: 50,
            hist_log_min: -3.0,
            hist_log_max: 3.0,
            ..Default::default()
        };
        let stats = pair.analyze_field_stats(&cfg);
        for st in &stats {
            if st.n_cells_active == 0 { continue; }
            let total: f64 = st.hist_density.iter().sum::<f64>()
                + st.hist_underflow_w_r + st.hist_overflow_w_r;
            assert!((total - 1.0).abs() < 1e-9,
                "level {}: histogram sum = {} (should be 1.0)", st.level, total);
            // Edges should have hist_bins + 1 entries
            assert_eq!(st.hist_bin_edges.len(), 51);
            assert_eq!(st.hist_density.len(), 50);
        }
    }

    #[test]
    fn field_stats_with_per_point_weights() {
        // With non-trivial weights, δ should still satisfy the global mean
        // identity at level 0 (single cell).
        let data = make_sm_uniform_3d(300, 9, 55);
        let randoms = make_sm_uniform_3d(900, 9, 56);
        let n_d = data.len();
        let n_r = randoms.len();
        // Random-ish weights in [0.5, 1.5]
        let mut s = 77u64;
        let w_d: Vec<f64> = (0..n_d).map(|_|
            0.5 + (sm64(&mut s) >> 53) as f64 / (1u64 << 53) as f64
        ).collect();
        let w_r: Vec<f64> = (0..n_r).map(|_|
            0.5 + (sm64(&mut s) >> 53) as f64 / (1u64 << 53) as f64
        ).collect();

        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build_full(
            td, tr, Some(w_d), Some(w_r), None,
            BitVecCascadePair::<3>::default_crossover_threshold(n_d, n_r),
        );
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        // Level 0: one cell, δ = 0 by α normalization.
        assert!(stats[0].mean_delta.abs() < 1e-12,
            "weighted: level 0 mean_delta = {}", stats[0].mean_delta);
        assert!(stats[0].var_delta < 1e-20);
    }

    #[test]
    fn field_stats_empty_random_catalog_safe() {
        // Pathological case: no randoms at all. Should not panic; should
        // return zero stats with a meaningful structure.
        let data = make_sm_uniform_3d(100, 8, 99);
        let randoms: Vec<[u64; 3]> = vec![];
        let range = CoordRange::analyze_pair(&data, &randoms);
        let td = TrimmedPoints::from_points_with_range(data, range.clone());
        let tr = TrimmedPoints::from_points_with_range(randoms, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 10, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);
        for st in &stats {
            assert_eq!(st.n_cells_active, 0);
            assert_eq!(st.sum_w_r_active, 0.0);
            assert_eq!(st.mean_delta, 0.0);
            assert_eq!(st.var_delta, 0.0);
            assert_eq!(st.hist_density.len(), 10);
            assert!(st.hist_density.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn field_stats_data_outside_footprint_counted() {
        // Construct a case where some data points lie in cells with no
        // corresponding randoms. Verify they appear in n_cells_data_outside
        // and sum_w_d_outside, while NOT contaminating moments.
        //
        // At fine scales (cells smaller than mean random spacing) the
        // "outside" counter loses physical meaning because even within the
        // footprint many cells have 0 randoms by chance. The diagnostic is
        // most informative at scales where the random density is high.
        let bits = 6;
        let max = (1u64 << bits) as i64;
        // Randoms restricted to lower half-cube (x < max/2).
        // Data: clustered, with one cluster INSIDE footprint and one OUTSIDE.
        let mut s = 12345u64;
        // Heavy random sampling in lower half so no random-empty cells.
        let randoms: Vec<[u64; 3]> = (0..50_000).map(|_| [
            (sm64(&mut s) % (max as u64 / 2)),
            (sm64(&mut s) % (max as u64)),
            (sm64(&mut s) % (max as u64)),
        ]).collect();
        // Inside-footprint data: a tight cluster in lower half
        let data_in: Vec<[u64; 3]> = (0..100).map(|_| [
            (max as u64 / 4) + (sm64(&mut s) % 4),  // tight cluster around max/4
            (max as u64 / 2) + (sm64(&mut s) % 4),
            (max as u64 / 2) + (sm64(&mut s) % 4),
        ]).collect();
        // Outside-footprint data: a tight cluster in upper half
        let data_out: Vec<[u64; 3]> = (0..30).map(|_| [
            (3 * max as u64 / 4) + (sm64(&mut s) % 4),  // tight cluster in upper half
            (max as u64 / 2) + (sm64(&mut s) % 4),
            (max as u64 / 2) + (sm64(&mut s) % 4),
        ]).collect();
        let mut data: Vec<[u64; 3]> = Vec::new();
        data.extend(data_in.iter().copied());
        data.extend(data_out.iter().copied());

        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_field_stats(&cfg);

        // At a moderate level (cell size ≈ box/8 = 8 units), the outside
        // cluster sits in cells with no random because the random catalog
        // doesn't extend there.
        let mid_level = 3;
        assert!(stats[mid_level].n_cells_data_outside >= 1,
            "level {}: expected outside cells, got {}",
            mid_level, stats[mid_level].n_cells_data_outside);
        // The W_d in outside cells at this level should account for most of
        // the outside cluster (some may be in cells touching the boundary).
        assert!(stats[mid_level].sum_w_d_outside >= 25.0,
            "level {}: outside W_d = {} (expected ≥ 25 of 30 outside points)",
            mid_level, stats[mid_level].sum_w_d_outside);

        // At level 0 (root): there's only one cell, it's everywhere, randoms
        // are present, so n_cells_data_outside should be 0.
        assert_eq!(stats[0].n_cells_data_outside, 0);
        assert_eq!(stats[0].sum_w_d_outside, 0.0);

        // The moments at the mid level should NOT be contaminated by the
        // outside cluster — they're computed only from the in-footprint
        // (i.e., random-bearing) cells.
        // sum_w_r_active equals total randoms (since all randoms are in
        // footprint by construction).
        let total_w_r = 50_000.0;
        assert!((stats[mid_level].sum_w_r_active - total_w_r).abs() < 1e-6,
            "level {}: sum_w_r_active = {} (expected {})",
            mid_level, stats[mid_level].sum_w_r_active, total_w_r);
    }

    // ===========================================================================
    // Anisotropy: cell-wavelet axis-aligned moments
    // ===========================================================================

    #[test]
    fn anisotropy_isotropic_field_gives_small_quadrupole() {
        // Truly isotropic data + matched isotropic randoms. The quadrupole
        // moment Q_2 should be statistically consistent with zero (it's not
        // zero exactly because of finite-sample shot noise).
        let pts_d = make_sm_uniform_3d(20_000, 10, 100);
        let pts_r = make_sm_uniform_3d(60_000, 10, 200);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);

        // For each level with enough parents to give clean stats, check that
        // the reduced quadrupole is small. A stricter test is that the three
        // axis-aligned moments are approximately equal to one another.
        for st in &stats {
            if st.n_parents < 50 { continue; }
            // The three axis-aligned w² should be equal to within ~20% for
            // a truly isotropic random field (Poisson shot noise has same
            // variance along every axis).
            let wx2 = st.mean_w_squared_for_axis(0);
            let wy2 = st.mean_w_squared_for_axis(1);
            let wz2 = st.mean_w_squared_for_axis(2);
            let mean = (wx2 + wy2 + wz2) / 3.0;
            if mean < 1e-12 { continue; }  // skip levels with no signal
            for &v in &[wx2, wy2, wz2] {
                let frac_dev = (v - mean).abs() / mean;
                assert!(frac_dev < 0.20,
                    "level {}: axis variances differ by > 20%: \
                    [wx², wy², wz²] = [{:.4e}, {:.4e}, {:.4e}], \
                    mean = {:.4e}, frac_dev = {:.3}",
                    st.level, wx2, wy2, wz2, mean, frac_dev);
            }
            // Reduced quadrupole should be small (|Q_2| < ~30% of mean axis variance)
            assert!(st.reduced_quadrupole_los.abs() < 0.30,
                "level {}: reduced Q_2 = {:.3} (should be ~0 for isotropic)",
                st.level, st.reduced_quadrupole_los);
        }
    }

    #[test]
    fn anisotropy_z_squashed_field_gives_negative_quadrupole() {
        // Construct anisotropic data: clustered with cluster shape squashed
        // along z. This means the density field has more variation along z
        // (sharp z structure) than along x or y. The wavelet coefficient
        // along z therefore has higher variance than perpendicular: Q_2 > 0.
        //
        // Conversely, extending clusters along z (FoG-like) gives Q_2 < 0.
        // Here we build a Kaiser-like enhancement: clusters squashed along z.
        let mut s = 4242u64;
        let bits = 10;
        let max = (1u64 << bits) as i64;
        let n = 8000;
        let n_clusters = 100;
        let center_max = max as u64 - 64;
        let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| [
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
        ]).collect();
        let mut data: Vec<[u64; 3]> = Vec::with_capacity(n);
        while data.len() < n {
            let c = &centers[(sm64(&mut s) as usize) % n_clusters];
            // Clusters extended in x, y but tight in z (Kaiser-like):
            // x, y spread over ±32 units, z spread over ±4 units
            let dx = ((sm64(&mut s) % 64) as i64) - 32;
            let dy = ((sm64(&mut s) % 64) as i64) - 32;
            let dz = ((sm64(&mut s) % 8) as i64) - 4;
            let p = [
                ((c[0] as i64 + dx).max(0).min(max - 1)) as u64,
                ((c[1] as i64 + dy).max(0).min(max - 1)) as u64,
                ((c[2] as i64 + dz).max(0).min(max - 1)) as u64,
            ];
            data.push(p);
        }
        let randoms = make_sm_uniform_3d(40_000, bits, 8888);
        let (td, tr) = build_aligned(data, randoms);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);

        // At scales comparable to the cluster size (cell ~ 16-64), the
        // tight-z structure means small z-extent variance vs larger x,y
        // extent. The axis-aligned wavelet picks up sharp boundaries:
        // sharp z transitions → large wz², smooth x,y → smaller wx², wy².
        // So at the relevant scales we expect wz² > wx², wz² > wy² → Q_2 > 0.
        let mut found_positive_q2 = false;
        for st in &stats {
            if st.n_parents < 20 { continue; }
            // Skip very fine scales where shot noise dominates
            let cell = st.cell_side_trimmed;
            if cell < 8.0 || cell > 256.0 { continue; }
            let wx2 = st.mean_w_squared_for_axis(0);
            let wy2 = st.mean_w_squared_for_axis(1);
            let wz2 = st.mean_w_squared_for_axis(2);
            // Tight-z clusters → wz² should be larger than wx², wy²
            if wz2 > wx2 * 1.1 && wz2 > wy2 * 1.1 {
                found_positive_q2 = true;
                break;
            }
        }
        assert!(found_positive_q2,
            "Expected a positive Q_2 signature at cluster scale (8-256), \
            but no level showed wz² > wx² and wz² > wy² by a clear margin.");
    }

    #[test]
    fn anisotropy_data_equals_randoms_zero_everywhere() {
        // If data == randoms exactly, δ = 0 in every cell, all wavelet
        // coefficients are zero, and Q_2 = 0 to machine precision.
        let pts = make_sm_uniform_3d(2000, 10, 7);
        let (td, tr) = build_aligned(pts.clone(), pts);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);
        for st in &stats {
            if st.n_parents == 0 { continue; }
            assert!(st.quadrupole_los.abs() < 1e-20,
                "level {}: Q_2 = {} (should be 0)", st.level, st.quadrupole_los);
            for &v in &st.mean_w_squared_axis {
                assert!(v < 1e-20,
                    "level {}: w² should be 0 when data = randoms; got {}",
                    st.level, v);
            }
        }
    }

    #[test]
    fn anisotropy_empty_random_catalog_safe() {
        let data = make_sm_uniform_3d(100, 8, 99);
        let randoms: Vec<[u64; 3]> = vec![];
        let range = CoordRange::analyze_pair(&data, &randoms);
        let td = TrimmedPoints::from_points_with_range(data, range.clone());
        let tr = TrimmedPoints::from_points_with_range(randoms, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);
        for st in &stats {
            assert_eq!(st.n_parents, 0);
            assert_eq!(st.quadrupole_los, 0.0);
            assert_eq!(st.mean_w_squared_axis, vec![0.0; 3]);
        }
    }

    #[test]
    fn anisotropy_works_in_2d() {
        // The cell-Haar wavelet observable is dimension-generic. In 2D
        // each parent has 4 children, giving 3 nontrivial wavelet
        // patterns: weight-1 (axis x and axis y) and weight-2 (xy).
        // Q_2 in 2D = <w_y²> − <w_x²>.
        let pts_d = make_uniform_2d(2000, 10, 100);
        let pts_r = make_uniform_2d(8000, 10, 200);
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::<2>::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::<2>::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);

        // Should produce 1 entry per level
        assert!(!stats.is_empty(), "expected non-empty anisotropy in 2D");
        for st in &stats {
            // Pattern vector of length 4 (= 2^2)
            assert_eq!(st.mean_w_squared_by_pattern.len(), 4,
                "expected 2^D = 4 patterns in 2D");
            // Axis-aligned vector of length 2
            assert_eq!(st.mean_w_squared_axis.len(), 2,
                "expected D = 2 axis-aligned values in 2D");
        }

        // For matched isotropic uniform fields, the two axis variances
        // should agree to within shot noise at scales where statistics
        // are decent.
        for st in &stats {
            if st.n_parents < 200 { continue; }
            let wx2 = st.mean_w_squared_for_axis(0);
            let wy2 = st.mean_w_squared_for_axis(1);
            let mean = (wx2 + wy2) / 2.0;
            if mean < 1e-12 { continue; }
            let frac_dev = (wx2 - wy2).abs() / mean;
            assert!(frac_dev < 0.30,
                "level {}: 2D axis variances differ by > 30%: \
                wx² = {:.4e}, wy² = {:.4e}, frac_dev = {:.3}",
                st.level, wx2, wy2, frac_dev);
        }
    }

    #[test]
    fn anisotropy_works_in_4d() {
        // In 4D each parent has 16 children, giving 15 nontrivial
        // wavelet patterns. We just verify the shapes and that an
        // isotropic field gives roughly equal axis variances.
        let pts: Vec<[u64; 4]> = (0..500).map(|i| {
            let x = (i as u64 * 31 + 7) % 256;
            let y = (i as u64 * 37 + 11) % 256;
            let z = (i as u64 * 41 + 13) % 256;
            let w = (i as u64 * 43 + 17) % 256;
            [x, y, z, w]
        }).collect();
        let range = CoordRange::analyze_pair(&pts, &pts);
        let td = TrimmedPoints::<4>::from_points_with_range(pts.clone(), range.clone());
        let tr = TrimmedPoints::<4>::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<4>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);
        for st in &stats {
            assert_eq!(st.mean_w_squared_by_pattern.len(), 16,
                "expected 2^D = 16 patterns in 4D");
            assert_eq!(st.mean_w_squared_axis.len(), 4,
                "expected D = 4 axis-aligned values in 4D");
            // Hamming-weight grouping: weight-1 is axis-aligned
            let weight1 = st.pattern_indices_with_hamming_weight(1);
            assert_eq!(weight1.len(), 4,
                "expected 4 weight-1 patterns in 4D, got {}", weight1.len());
            let weight4 = st.pattern_indices_with_hamming_weight(4);
            assert_eq!(weight4.len(), 1,
                "expected 1 weight-4 (body-diagonal) pattern in 4D");
            // data == randoms ⇒ δ = 0 ⇒ all coefficients exactly zero
            for &v in &st.mean_w_squared_by_pattern {
                assert_eq!(v, 0.0,
                    "level {}: data=randoms ⇒ wavelet coefficient should be 0",
                    st.level);
            }
        }
    }

    #[test]
    fn anisotropy_works_in_1d() {
        // In 1D each parent has 2 children, so there's exactly 1 wavelet
        // coefficient (w_e=1 = the difference). Q_2 ≡ 0 in 1D since there
        // is no transverse direction.
        let pts: Vec<[u64; 1]> = (0..200).map(|i| [(i as u64 * 7) % 1024]).collect();
        let range = CoordRange::analyze_pair(&pts, &pts);
        let td = TrimmedPoints::<1>::from_points_with_range(pts.clone(), range.clone());
        let tr = TrimmedPoints::<1>::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<1>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_anisotropy(&cfg);
        for st in &stats {
            assert_eq!(st.mean_w_squared_by_pattern.len(), 2,
                "expected 2^D = 2 patterns in 1D");
            assert_eq!(st.mean_w_squared_axis.len(), 1);
            // Q_2 ≡ 0 in 1D (no transverse direction)
            assert_eq!(st.quadrupole_los, 0.0,
                "1D Q_2 should be identically zero");
            assert_eq!(st.reduced_quadrupole_los, 0.0);
        }
    }

    /// Helper: verify that v1 (legacy aniso_recurse) and v2 (visitor)
    /// produce identical AnisotropyStats for the given input. Used by
    /// the parity tests below to cover several configurations.
    fn assert_aniso_parity(
        pts_d: Vec<[u64; 3]>, pts_r: Vec<[u64; 3]>,
        cfg: FieldStatsConfig, label: &str,
    ) {
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::<3>::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::<3>::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        // The public method now delegates to the visitor; the legacy
        // (3D-only, hand-written recursion) is preserved as
        // `analyze_anisotropy_legacy` precisely for this parity check.
        let v1 = pair.analyze_anisotropy_legacy(&cfg);
        let v2 = pair.analyze_anisotropy_v2(&cfg);

        assert_eq!(v1.len(), v2.len(),
            "{}: result lengths differ: v1={} v2={}",
            label, v1.len(), v2.len());

        for (l, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(a.level, b.level,
                "{}: level field differs at index {}", label, l);
            // Counters must be exactly equal.
            assert_eq!(a.n_parents, b.n_parents,
                "{}: level {} n_parents differ: v1={} v2={}",
                label, l, a.n_parents, b.n_parents);
            // Compare every pattern (3D has 8 entries in mean_w_squared_by_pattern).
            // The walker visits cells in the same canonical axis-pattern
            // order as the legacy recursion, so values should be bit-identical
            // (or differ only by a tiny order-of-summation amount).
            assert_eq!(a.mean_w_squared_by_pattern.len(), b.mean_w_squared_by_pattern.len(),
                "{}: level {} pattern-vector length differs", label, l);
            assert_eq!(a.mean_w_squared_axis.len(), b.mean_w_squared_axis.len(),
                "{}: level {} axis-vector length differs", label, l);

            let mut comparisons: Vec<(String, f64, f64)> = vec![
                ("sum_w_r_parents".to_string(), a.sum_w_r_parents, b.sum_w_r_parents),
                ("quadrupole_los".to_string(), a.quadrupole_los, b.quadrupole_los),
                ("reduced_quadrupole_los".to_string(), a.reduced_quadrupole_los,
                    b.reduced_quadrupole_los),
            ];
            for d in 0..a.mean_w_squared_axis.len() {
                comparisons.push((format!("axis_{}", d),
                    a.mean_w_squared_axis[d], b.mean_w_squared_axis[d]));
            }
            for e in 1..a.mean_w_squared_by_pattern.len() {
                comparisons.push((format!("pattern_{:03b}", e),
                    a.mean_w_squared_by_pattern[e],
                    b.mean_w_squared_by_pattern[e]));
            }

            for (name, av, bv) in comparisons {
                if av != bv {
                    // Allow tiny floating-point drift only if the absolute
                    // value is also tiny — a sign of an order-of-summation
                    // difference rather than an algorithmic error.
                    let diff = (av - bv).abs();
                    let mag = av.abs().max(bv.abs()).max(1e-30);
                    let rel = diff / mag;
                    assert!(rel < 1e-13,
                        "{}: level {} {}: v1={:.16e} v2={:.16e} (rel diff {:.3e})",
                        label, l, name, av, bv, rel);
                }
            }
        }
    }

    #[test]
    fn anisotropy_v2_matches_v1_uniform() {
        let pts_d = make_sm_uniform_3d(8_000, 9, 50);
        let pts_r = make_sm_uniform_3d(20_000, 9, 51);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        assert_aniso_parity(pts_d, pts_r, cfg, "uniform 8k+20k");
    }

    #[test]
    fn anisotropy_v2_matches_v1_clustered() {
        // Anisotropic clustered field — clusters squashed in z,
        // matching the same setup used by anisotropy_z_squashed test.
        let mut s = 1234u64;
        let bits = 9;
        let max = (1u64 << bits) as i64;
        let n = 5_000;
        let n_clusters = 80;
        let center_max = max as u64 - 64;
        let centers: Vec<[u64; 3]> = (0..n_clusters).map(|_| [
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
            (sm64(&mut s) % center_max) + 32,
        ]).collect();
        let mut pts_d: Vec<[u64; 3]> = Vec::with_capacity(n);
        while pts_d.len() < n {
            let c = &centers[(sm64(&mut s) as usize) % n_clusters];
            let dx = ((sm64(&mut s) % 64) as i64) - 32;
            let dy = ((sm64(&mut s) % 64) as i64) - 32;
            let dz = ((sm64(&mut s) % 8) as i64) - 4;
            pts_d.push([
                ((c[0] as i64 + dx).max(0).min(max - 1)) as u64,
                ((c[1] as i64 + dy).max(0).min(max - 1)) as u64,
                ((c[2] as i64 + dz).max(0).min(max - 1)) as u64,
            ]);
        }
        let pts_r = make_sm_uniform_3d(20_000, bits, 999);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        assert_aniso_parity(pts_d, pts_r, cfg, "clustered z-squashed");
    }

    #[test]
    fn anisotropy_v2_matches_v1_with_footprint_cutoff() {
        // Force some parent cells to fail the footprint cutoff by using
        // a sparse random catalog and a non-trivial w_r_min.
        let pts_d = make_sm_uniform_3d(4_000, 8, 200);
        let pts_r = make_sm_uniform_3d(5_000, 8, 201);
        let cfg = FieldStatsConfig {
            w_r_min: 0.5,
            hist_bins: 0,
            ..Default::default()
        };
        assert_aniso_parity(pts_d, pts_r, cfg, "footprint cutoff");
    }

    #[test]
    fn anisotropy_v2_matches_v1_empty_inputs() {
        // Empty random catalog → both implementations should return
        // identical zero-stats vectors.
        let pts_d = make_sm_uniform_3d(100, 7, 0);
        let pts_r: Vec<[u64; 3]> = vec![];
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::<3>::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::<3>::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig::default();
        let v1 = pair.analyze_anisotropy_legacy(&cfg);
        let v2 = pair.analyze_anisotropy_v2(&cfg);
        assert_eq!(v1.len(), v2.len());
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a.n_parents, b.n_parents);
            assert_eq!(a.sum_w_r_parents, b.sum_w_r_parents);
            assert_eq!(a.quadrupole_los, b.quadrupole_los);
        }
    }

    // ===========================================================================
    // Second-order Haar scattering
    // ===========================================================================

    #[test]
    fn scattering_2nd_order_data_equals_randoms_zero_everywhere() {
        // δ = 0 everywhere → all wavelet coefficients zero, so all scattering
        // coefficients (first AND second order) are zero.
        let pts = make_sm_uniform_3d(2000, 10, 7);
        let (td, tr) = build_aligned(pts.clone(), pts);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_scattering_2nd_order(&cfg);
        for st in &stats {
            for &v in &st.first_order {
                assert!(v < 1e-20,
                    "level pair (l1={}, l2={}): first-order should be 0 \
                    when data=randoms; got {}",
                    st.level_fine, st.level_coarse, v);
            }
            for row in &st.second_order {
                for &v in row {
                    assert!(v < 1e-20,
                        "(l1={}, l2={}): second-order should be 0; got {}",
                        st.level_fine, st.level_coarse, v);
                }
            }
        }
    }

    #[test]
    fn scattering_2nd_order_isotropic_field_axes_balanced() {
        // For an isotropic random field, second-order coefficients along
        // x, y, z should be statistically equivalent.
        let pts_d = make_sm_uniform_3d(20_000, 9, 100);
        let pts_r = make_sm_uniform_3d(60_000, 9, 200);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let stats = pair.analyze_scattering_2nd_order(&cfg);

        // Pick (l1, l2) pairs with enough statistics
        for st in &stats {
            if st.n_parents_coarse < 50 { continue; }
            // For each e_1 direction, the three e_2 second-order coefficients
            // should be approximately equal (since underlying field is
            // isotropic).
            for e1 in 0..3 {
                let row = &st.second_order[e1];
                let mean = (row[0] + row[1] + row[2]) / 3.0;
                if mean < 1e-12 { continue; }
                for &v in row {
                    let frac = (v - mean).abs() / mean;
                    assert!(frac < 0.40,
                        "(l1={}, l2={}, e1={}): second-order axes differ by \
                        > 40%: row = {:?}, mean = {:.3e}, frac = {:.3}",
                        st.level_fine, st.level_coarse, e1, row, mean, frac);
                }
            }
        }
    }

    #[test]
    fn scattering_2nd_order_first_order_matches_anisotropy() {
        // The first-order coefficients reported by the second-order machinery
        // should agree with what analyze_anisotropy reports for the same
        // axis-aligned directions, modulo statistical equivalence.
        // (analyze_anisotropy weights every level-l parent cell that has all
        // 8 children active; analyze_scattering_2nd_order applies the same
        // gating.)
        let pts_d = make_sm_uniform_3d(8000, 9, 13);
        let pts_r = make_sm_uniform_3d(40_000, 9, 17);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let aniso = pair.analyze_anisotropy(&cfg);
        let scat = pair.analyze_scattering_2nd_order(&cfg);

        // Pick (l1, l2 = l1 - 1) pairs and compare first-order to anisotropy.
        // Map: scat entries with level_fine == l_1 should have
        // first_order[e] = aniso[l_1].mean_w_squared_axis[e] when n_parents
        // is the same.
        for st in &scat {
            if st.level_coarse != st.level_fine - 1 { continue; }
            let l1 = st.level_fine;
            if l1 >= aniso.len() { continue; }
            let a = &aniso[l1];
            if a.n_parents == 0 || st.n_parents_coarse == 0 { continue; }
            // Both functions should be using the same parent-cell set, so the
            // first-order means should match up to floating-point.
            for e1 in 0..3 {
                let s_val = st.first_order[e1];
                let a_val = a.mean_w_squared_axis[e1];
                if a_val.abs() < 1e-15 { continue; }
                let rel = (s_val - a_val).abs() / a_val.abs();
                assert!(rel < 1e-9,
                    "l_1={}, e_1={}: first-order from scattering ({:.6e}) \
                    differs from analyze_anisotropy ({:.6e}), rel = {:.3e}",
                    l1, e1, s_val, a_val, rel);
            }
        }
    }

    #[test]
    fn scattering_2nd_order_only_3d() {
        let pts_2d: Vec<[u64; 2]> = (0..100).map(|i| [i as u64 * 3, i as u64 * 5]).collect();
        let range = CoordRange::analyze_pair(&pts_2d, &pts_2d);
        let td = TrimmedPoints::from_points_with_range(pts_2d.clone(), range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_2d, range);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let cfg = FieldStatsConfig::default();
        let stats = pair.analyze_scattering_2nd_order(&cfg);
        assert!(stats.is_empty(), "scattering should return empty Vec for D!=3");
    }

    // ========================================================================
    // Walker verification: walk() visits the right cells
    // ========================================================================

    #[test]
    fn walk_visits_correct_per_level_active_counts() {
        // The walker should visit each non-empty cell exactly once. We
        // verify by counting active cells at each level via a trivial
        // visitor and comparing against analyze_field_stats's
        // n_cells_active.
        use crate::cascade_visitor::{CascadeVisitor, CellVisit, FootprintCutoff};

        struct ActiveCount {
            n_active: Vec<u64>,
        }
        impl CascadeVisitor<3> for ActiveCount {
            fn enter_cell(&mut self, cell: &CellVisit<3>) {
                if cell.in_footprint(&FootprintCutoff::ANY_RANDOM) {
                    if cell.level >= self.n_active.len() {
                        self.n_active.resize(cell.level + 1, 0);
                    }
                    self.n_active[cell.level] += 1;
                }
            }
        }

        let pts_d = make_sm_uniform_3d(2_000, 9, 100);
        let pts_r = make_sm_uniform_3d(6_000, 9, 200);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        // Walker counts
        let mut visitor = ActiveCount { n_active: Vec::new() };
        pair.walk(&mut visitor);

        // Reference: analyze_field_stats.n_cells_active
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let ref_stats = pair.analyze_field_stats(&cfg);

        for st in &ref_stats {
            let walker_count = visitor.n_active.get(st.level).copied().unwrap_or(0);
            assert_eq!(walker_count, st.n_cells_active,
                "level {}: walker n_active={} vs analyze_field_stats={}",
                st.level, walker_count, st.n_cells_active);
        }
    }

    #[test]
    fn walk_sum_w_r_matches_analyze() {
        // The walker should see the same total W_r per level as
        // analyze_field_stats reports.
        use crate::cascade_visitor::{CascadeVisitor, CellVisit, FootprintCutoff};

        struct SumWR {
            sum: Vec<f64>,
        }
        impl CascadeVisitor<3> for SumWR {
            fn enter_cell(&mut self, cell: &CellVisit<3>) {
                if cell.in_footprint(&FootprintCutoff::ANY_RANDOM) {
                    if cell.level >= self.sum.len() {
                        self.sum.resize(cell.level + 1, 0.0);
                    }
                    self.sum[cell.level] += cell.randoms.sum_w;
                }
            }
        }

        let pts_d = make_sm_uniform_3d(1_500, 8, 11);
        let pts_r = make_sm_uniform_3d(4_500, 8, 22);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        let mut visitor = SumWR { sum: Vec::new() };
        pair.walk(&mut visitor);

        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let ref_stats = pair.analyze_field_stats(&cfg);

        for st in &ref_stats {
            let w = visitor.sum.get(st.level).copied().unwrap_or(0.0);
            // sum_w_r_active in the reference is sum over in-footprint
            // cells at this level. Should match.
            let rel_err = if st.sum_w_r_active.abs() > 1e-12 {
                (w - st.sum_w_r_active).abs() / st.sum_w_r_active.abs()
            } else { (w - st.sum_w_r_active).abs() };
            assert!(rel_err < 1e-12,
                "level {}: walker sum_w_r={} vs analyze sum_w_r_active={}, rel_err={}",
                st.level, w, st.sum_w_r_active, rel_err);
        }
    }

    #[test]
    fn walk_after_children_provides_correct_decomposition() {
        // after_children should give us 8 children whose sum of (W_d, W_r)
        // equals the parent's (W_d, W_r). Verify this invariant.
        use crate::cascade_visitor::{CascadeVisitor, CellVisit};

        struct DecompCheck {
            ok: bool,
            failures: Vec<(usize, u64, f64, f64, f64, f64)>,  // (level, cell_id, parent_d, sum_d, parent_r, sum_r)
        }
        impl CascadeVisitor<3> for DecompCheck {
            fn after_children(&mut self, parent: &CellVisit<3>, children: &[CellVisit<3>]) {
                let sum_d: f64 = children.iter().map(|c| c.data.sum_w).sum();
                let sum_r: f64 = children.iter().map(|c| c.randoms.sum_w).sum();
                let d_ok = (sum_d - parent.data.sum_w).abs() < 1e-9;
                let r_ok = (sum_r - parent.randoms.sum_w).abs() < 1e-9;
                if !d_ok || !r_ok {
                    self.ok = false;
                    self.failures.push((parent.level, parent.cell_id,
                                         parent.data.sum_w, sum_d,
                                         parent.randoms.sum_w, sum_r));
                }
            }
        }

        let pts_d = make_sm_uniform_3d(800, 8, 33);
        let pts_r = make_sm_uniform_3d(2_400, 8, 44);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        let mut check = DecompCheck { ok: true, failures: Vec::new() };
        pair.walk(&mut check);
        assert!(check.ok,
            "8-child decomposition violated at: {:?}", &check.failures[..check.failures.len().min(3)]);
    }

    #[test]
    fn walk_handles_empty_safely() {
        // Empty data, empty randoms.
        let pts: Vec<[u64; 3]> = vec![];
        let range = CoordRange::analyze_pair(&pts, &pts);
        let td = TrimmedPoints::from_points_with_range(pts.clone(), range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        struct CountVisits(usize);
        impl crate::cascade_visitor::CascadeVisitor<3> for CountVisits {
            fn enter_cell(&mut self, _: &crate::cascade_visitor::CellVisit<3>) {
                self.0 += 1;
            }
        }
        let mut v = CountVisits(0);
        pair.walk(&mut v);
        assert_eq!(v.0, 0, "walker should visit no cells when both catalogs are empty");
    }

    #[test]
    fn walk_with_pointlist_crossover_visits_same_cells() {
        // A small catalog where the crossover threshold kicks in early:
        // verify the pointlist path of walk_recurse also gives the right
        // active counts.
        use crate::cascade_visitor::{CascadeVisitor, CellVisit, FootprintCutoff};

        struct ActiveCount(Vec<u64>);
        impl CascadeVisitor<3> for ActiveCount {
            fn enter_cell(&mut self, cell: &CellVisit<3>) {
                if cell.in_footprint(&FootprintCutoff::ANY_RANDOM) {
                    if cell.level >= self.0.len() {
                        self.0.resize(cell.level + 1, 0);
                    }
                    self.0[cell.level] += 1;
                }
            }
        }

        let pts_d = make_sm_uniform_3d(100, 6, 5);  // small N to force crossover
        let pts_r = make_sm_uniform_3d(300, 6, 6);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, Some(8));  // low crossover

        let mut walker = ActiveCount(Vec::new());
        pair.walk(&mut walker);

        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let ref_stats = pair.analyze_field_stats(&cfg);

        for st in &ref_stats {
            let w = walker.0.get(st.level).copied().unwrap_or(0);
            assert_eq!(w, st.n_cells_active,
                "level {}: walker n_active={} vs analyze={}",
                st.level, w, st.n_cells_active);
        }
    }

    // ========================================================================
    // Visitor-based v2 vs legacy: bit-exact parity tests
    // ========================================================================

    fn assert_field_stats_match(v1: &[DensityFieldStats], v2: &[DensityFieldStats], ctx: &str) {
        assert_eq!(v1.len(), v2.len(), "{}: level counts differ", ctx);
        for (a, b) in v1.iter().zip(v2.iter()) {
            let l = a.level;
            assert_eq!(a.level, b.level, "{} level mismatch", ctx);
            assert_eq!(a.n_cells_active, b.n_cells_active,
                "{} l={}: n_cells_active {} vs {}", ctx, l, a.n_cells_active, b.n_cells_active);
            assert_eq!(a.n_cells_data_outside, b.n_cells_data_outside,
                "{} l={}: n_cells_data_outside {} vs {}", ctx, l,
                a.n_cells_data_outside, b.n_cells_data_outside);
            // Floats: relative or absolute tolerance — should be bit-exact since
            // the visitor accumulates in the same order, but allow tiny rounding.
            for (name, av, bv) in &[
                ("sum_w_r_active", a.sum_w_r_active, b.sum_w_r_active),
                ("mean_delta", a.mean_delta, b.mean_delta),
                ("var_delta", a.var_delta, b.var_delta),
                ("m3_delta", a.m3_delta, b.m3_delta),
                ("m4_delta", a.m4_delta, b.m4_delta),
                ("s3_delta", a.s3_delta, b.s3_delta),
                ("sum_w_d_outside", a.sum_w_d_outside, b.sum_w_d_outside),
                ("hist_underflow_w_r", a.hist_underflow_w_r, b.hist_underflow_w_r),
                ("hist_overflow_w_r", a.hist_overflow_w_r, b.hist_overflow_w_r),
            ] {
                let abs_err = (av - bv).abs();
                let rel_err = if av.abs() > 1e-12 { abs_err / av.abs() } else { abs_err };
                assert!(rel_err < 1e-12,
                    "{} l={}: {} mismatch: v1={}, v2={}, rel_err={:.2e}",
                    ctx, l, name, av, bv, rel_err);
            }
            // Histogram bins
            assert_eq!(a.hist_density.len(), b.hist_density.len(),
                "{} l={}: hist_density length", ctx, l);
            for (bin, (h1, h2)) in a.hist_density.iter().zip(b.hist_density.iter()).enumerate() {
                let abs = (h1 - h2).abs();
                let rel = if h1.abs() > 1e-12 { abs / h1.abs() } else { abs };
                assert!(rel < 1e-12,
                    "{} l={} bin={}: hist {} vs {}, rel_err={:.2e}",
                    ctx, l, bin, h1, h2, rel);
            }
        }
    }

    #[test]
    fn field_stats_v2_matches_v1_uniform() {
        let pts_d = make_sm_uniform_3d(2_000, 9, 100);
        let pts_r = make_sm_uniform_3d(6_000, 9, 200);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 30, ..Default::default() };
        let v1 = pair.analyze_field_stats(&cfg);
        let v2 = pair.analyze_field_stats_v2(&cfg);
        assert_field_stats_match(&v1, &v2, "uniform");
    }

    #[test]
    fn field_stats_v2_matches_v1_empty_random() {
        // Empty random catalog → both should return empty stats
        let data = make_sm_uniform_3d(100, 8, 99);
        let randoms: Vec<[u64; 3]> = vec![];
        let range = CoordRange::analyze_pair(&data, &randoms);
        let td = TrimmedPoints::from_points_with_range(data, range.clone());
        let tr = TrimmedPoints::from_points_with_range(randoms, range);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 10, ..Default::default() };
        let v1 = pair.analyze_field_stats(&cfg);
        let v2 = pair.analyze_field_stats_v2(&cfg);
        assert_field_stats_match(&v1, &v2, "empty_randoms");
    }

    #[test]
    fn field_stats_v2_matches_v1_with_outside_footprint() {
        // Realistic survey-like geometry: contamination outside the footprint.
        let mut pts_d = Vec::new();
        // In-footprint clustered data
        for i in 0..1500 {
            let x = (i as u64 * 37) % 256 + 64;
            let y = (i as u64 * 41) % 256 + 64;
            let z = (i as u64 * 43) % 256 + 256;
            pts_d.push([x, y, z]);
        }
        // Out-of-footprint contamination
        for i in 0..100 {
            let x = (i as u64 * 31) % 64;
            let y = (i as u64 * 29) % 64;
            let z = (i as u64 * 23) % 32;  // randoms have z >= 256, so these are out
            pts_d.push([x, y, z]);
        }
        // Randoms only in upper-z half
        let mut pts_r = Vec::new();
        for i in 0..6000 {
            let x = (i as u64 * 79) % 512;
            let y = (i as u64 * 83) % 512;
            let z = (i as u64 * 89) % 256 + 256;
            pts_r.push([x, y, z]);
        }
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let cfg = FieldStatsConfig { w_r_min: 0.5, hist_bins: 20, ..Default::default() };
        let v1 = pair.analyze_field_stats(&cfg);
        let v2 = pair.analyze_field_stats_v2(&cfg);
        assert_field_stats_match(&v1, &v2, "outside_footprint");
    }

    #[test]
    fn field_stats_v2_matches_v1_2d() {
        let pts_d: Vec<[u64; 2]> = (0..300).map(|i| [(i * 7) as u64, (i * 11) as u64]).collect();
        let pts_r: Vec<[u64; 2]> = (0..900).map(|i| [(i * 5) as u64, (i * 13) as u64]).collect();
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        let pair = BitVecCascadePair::<2>::build(td, tr, None);
        let cfg = FieldStatsConfig { hist_bins: 15, ..Default::default() };
        let v1 = pair.analyze_field_stats(&cfg);
        let v2 = pair.analyze_field_stats_v2(&cfg);
        assert_field_stats_match(&v1, &v2, "2d");
    }

    /// Commit 6: `compensated_sums: true` must produce numerically
    /// identical results to `compensated_sums: false` on benign data
    /// (cosmology-realistic cell counts and weights). The compensated
    /// path is a precision *insurance* — it should never CHANGE the
    /// answer for problems where naive summation is already accurate.
    /// This test is the regression guarantee that flipping the flag
    /// is safe for routine use.
    ///
    /// The compensated mode's actual benefit (recovering precision
    /// against pathological dynamic-range cancellation) is covered
    /// by the `CompensatedSum` unit tests in `compensated_sum.rs`,
    /// which exercise the underlying primitive directly.
    #[test]
    fn compensated_sums_matches_naive_on_benign_data() {
        let pts_d = make_uniform_3d(1500, 7, 12345);
        let pts_r = make_uniform_3d(4500, 7, 12346);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        let cfg_naive = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 20, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let cfg_comp = FieldStatsConfig {
            compensated_sums: true,
            ..cfg_naive.clone()
        };

        let stats_naive = pair.analyze_field_stats(&cfg_naive);
        let stats_comp  = pair.analyze_field_stats(&cfg_comp);

        assert_eq!(stats_naive.len(), stats_comp.len());
        for (l, (a, b)) in stats_naive.iter().zip(stats_comp.iter()).enumerate() {
            assert_eq!(a.n_cells_active, b.n_cells_active,
                "level {}: cell count mismatch", l);
            // For benign data both modes should agree to ~1e-12 relative
            // (small Neumaier reordering vs. naive can shift the very
            // last few bits, but no more). Use mixed abs+rel tolerance:
            // some quantities are O(1) (var, kurtosis), others are
            // identically ~0 (mean_delta, by α-normalization).
            let close = |x: f64, y: f64, name: &str| {
                if !x.is_finite() && !y.is_finite() { return; }
                let scale = x.abs().max(y.abs());
                let abs_diff = (x - y).abs();
                let abs_tol = 1e-12;
                let rel_tol = 1e-10;
                let ok = abs_diff < abs_tol || abs_diff < rel_tol * scale;
                assert!(ok,
                    "level {} {}: naive={} compensated={} (abs diff {}, rel {})",
                    l, name, x, y, abs_diff, abs_diff / scale.max(1e-300));
            };
            close(a.sum_w_r_active,    b.sum_w_r_active,    "sum_w_r_active");
            close(a.mean_delta,         b.mean_delta,        "mean_delta");
            close(a.var_delta,          b.var_delta,         "var_delta");
            close(a.m3_delta,           b.m3_delta,          "m3_delta");
            close(a.m4_delta,           b.m4_delta,          "m4_delta");
            close(a.s3_delta,           b.s3_delta,          "s3_delta");
        }
    }

    /// Commit 6: confirm the `_comp` accumulator vectors are
    /// allocated (and zero-length when off, n_levels when on).
    /// This is a structural guarantee — ensures the opt-in flag
    /// actually toggles the parallel-accumulator code path. Without
    /// this we'd have no way to know if a future refactor accidentally
    /// dropped the wiring while keeping the parity test passing.
    #[test]
    fn compensated_sums_flag_controls_visitor_allocation() {
        let pts_d = make_uniform_3d(100, 6, 12347);
        let pts_r = make_uniform_3d(300, 6, 12348);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let n_levels = pair.l_max + 1;

        // Off: comp vectors should be empty (zero allocation overhead).
        let cfg_off = FieldStatsConfig {
            compensated_sums: false, ..Default::default()
        };
        let v_off = FieldStatsVisitor::<3>::new(&pair, n_levels, &cfg_off);
        assert_eq!(v_off.sum_w_r_comp.len(), 0,
            "compensated_sums: false → comp vector must be empty");
        assert_eq!(v_off.sum_w_r_delta3_comp.len(), 0,
            "compensated_sums: false → δ³ comp vector must be empty");
        assert_eq!(v_off.sw_d_outside_comp.len(), 0,
            "compensated_sums: false → sw_d_outside comp vector must be empty");

        // On: comp vectors should match n_levels.
        let cfg_on = FieldStatsConfig {
            compensated_sums: true, ..Default::default()
        };
        let v_on = FieldStatsVisitor::<3>::new(&pair, n_levels, &cfg_on);
        assert_eq!(v_on.sum_w_r_comp.len(), n_levels);
        assert_eq!(v_on.sum_w_r_delta1_comp.len(), n_levels);
        assert_eq!(v_on.sum_w_r_delta2_comp.len(), n_levels);
        assert_eq!(v_on.sum_w_r_delta3_comp.len(), n_levels);
        assert_eq!(v_on.sum_w_r_delta4_comp.len(), n_levels);
        assert_eq!(v_on.sw_d_outside_comp.len(), n_levels);
    }

    /// Commit 6: same parity check for AnisotropyVisitor.
    /// `compensated_sums: true` must produce numerically identical
    /// results to `false` on benign data.
    #[test]
    fn compensated_sums_matches_naive_anisotropy_on_benign_data() {
        let pts_d = make_uniform_3d(1500, 7, 22345);
        let pts_r = make_uniform_3d(4500, 7, 22346);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);

        let cfg_naive = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let cfg_comp = FieldStatsConfig {
            compensated_sums: true,
            ..cfg_naive.clone()
        };

        let stats_naive = pair.analyze_anisotropy(&cfg_naive);
        let stats_comp  = pair.analyze_anisotropy(&cfg_comp);

        assert_eq!(stats_naive.len(), stats_comp.len());
        for (l, (a, b)) in stats_naive.iter().zip(stats_comp.iter()).enumerate() {
            assert_eq!(a.n_parents, b.n_parents,
                "level {}: parent count mismatch", l);
            let close = |x: f64, y: f64, name: &str| {
                if !x.is_finite() && !y.is_finite() { return; }
                let scale = x.abs().max(y.abs());
                let abs_diff = (x - y).abs();
                let ok = abs_diff < 1e-12 || abs_diff < 1e-10 * scale;
                assert!(ok,
                    "level {} {}: naive={} compensated={} (abs diff {})",
                    l, name, x, y, abs_diff);
            };
            close(a.sum_w_r_parents,    b.sum_w_r_parents,    "sum_w_r_parents");
            close(a.quadrupole_los,     b.quadrupole_los,     "quadrupole_los");
            for (k, (m_n, m_c)) in a.mean_w_squared_by_pattern.iter()
                    .zip(b.mean_w_squared_by_pattern.iter()).enumerate() {
                close(*m_n, *m_c, &format!("mean_w_squared_by_pattern[{}]", k));
            }
        }
    }

    /// Commit 6: AnisotropyVisitor allocation toggling — same
    /// structural guarantee as for FieldStatsVisitor.
    #[test]
    fn compensated_sums_flag_controls_anisotropy_visitor_allocation() {
        let pts_d = make_uniform_3d(100, 6, 22347);
        let pts_r = make_uniform_3d(300, 6, 22348);
        let (td, tr) = build_aligned(pts_d, pts_r);
        let pair = BitVecCascadePair::<3>::build(td, tr, None);
        let n_levels = pair.l_max + 1;

        let cfg_off = FieldStatsConfig {
            compensated_sums: false, ..Default::default()
        };
        let v_off = AnisotropyVisitor::<3>::new(&pair, n_levels, &cfg_off);
        assert_eq!(v_off.sum_w_r_parents_comp.len(), 0,
            "compensated_sums: false → parent-W comp vector must be empty");
        assert_eq!(v_off.sum_wr_w2_comp.len(), 0,
            "compensated_sums: false → wavelet comp vector must be empty");

        let cfg_on = FieldStatsConfig {
            compensated_sums: true, ..Default::default()
        };
        let v_on = AnisotropyVisitor::<3>::new(&pair, n_levels, &cfg_on);
        assert_eq!(v_on.sum_w_r_parents_comp.len(), n_levels);
        assert_eq!(v_on.sum_wr_w2_comp.len(), n_levels);
        // Inner: 2^D = 8 patterns per level for D=3
        assert_eq!(v_on.sum_wr_w2_comp[0].len(), 8);
    }
}
