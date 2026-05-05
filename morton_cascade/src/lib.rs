//! # morton_cascade
//!
//! Hierarchical multi-statistic spatial analysis for point catalogs, with native
//! two-catalog (data + randoms) survey-geometry handling.
//!
//! ## What this library does
//!
//! Given a catalog of N points in a D-dimensional box (and, optionally, a second
//! "randoms" catalog encoding a survey footprint and selection function), the
//! library builds a **single sparse Morton-ordered dyadic cell hierarchy** and
//! produces a wide family of spatial statistics from that one construction. The
//! cascade visits cells of side `R_l = 2^(L_max - l)` for `l = 0, 1, ..., L_max`,
//! using bit-vector encoding so empty regions cost nothing.
//!
//! ## Statistics families produced
//!
//! From one cascade construction the library can report, at every dyadic scale:
//!
//! - **Counts-in-cells (CIC) family**
//!   - Cell-count probability mass function `P_N(V)` at log-spaced cube volumes
//!   - Volume distribution function `P(δ; R)` of the density contrast
//!   - Void probability function as the empty-cell tail of `P_N`
//!
//! - **Moments family**
//!   - Variance `σ²(R)` (the σ²-vs-R curve in one pass)
//!   - Schur residual `ΔV_l` decomposing `Var` by scale: `Var = Σ_l ΔV_l`
//!   - Skewness, kurtosis: `<δ³>(R)`, `<δ⁴>(R)`
//!   - Reduced cumulants `S_3(R) = <δ³>/<δ²>²`, etc.
//!
//! - **Two-point family**
//!   - Pair counts `DD(R)`, `RR(R)`, `DR(R)` at every dyadic scale
//!   - Landy-Szalay correlation function `ξ(R)`
//!   - Power-spectrum-equivalent observables via Fourier integration of `ξ(R)`
//!
//! - **Three-point and beyond**
//!   - Cube-window-averaged bispectrum equivalent: `<δ³>` at every scale
//!   - Higher cumulants and reduced moments to arbitrary order
//!
//! - **Wavelet / scattering family**
//!   - First-order Haar scattering: cell-wavelet variance per axis-aligned direction
//!   - Anisotropy moments: quadrupole `Q_2 = <w_z²> − ½(<w_x²> + <w_y²>)` at every scale
//!     ([`hier_bitvec_pair::BitVecCascadePair::analyze_anisotropy`])
//!   - Second-order scattering planned (recursive cascade on |w| derived fields)
//!
//! - **Footprint and catalog-quality diagnostics**
//!   - Outside-footprint mass `Σ W_d outside survey` per scale
//!   - Active-cell counts and effective-volume diagnostics
//!
//! - **Cross-statistics**
//!   - All of the above, between two tracer catalogs
//!
//! ## Why the cascade rather than mesh-based estimators?
//!
//! Conventional cosmological clustering codes (FFT-based, fixed-grid) commit to a
//! grid resolution at the start, requiring O(N_cells^D) memory and O(N_cells^D
//! log N_cells) FFT cost, and producing one set of statistics at one resolution.
//! Multiscale analysis means re-pixelizing at multiple resolutions. Survey
//! footprints require separate window-function or Φ-correction machinery.
//!
//! The cascade has three structural advantages over mesh-based approaches:
//!
//! 1. **No mesh.** The bit-vector encoding adapts to wherever the points are;
//!    empty regions cost nothing.  No grid-resolution decisions, no discretization
//!    artifacts at scales near the cell size.
//!
//! 2. **All scales in one pass.** The dyadic structure produces every scale from
//!    the box-size down to the finest cell in a single traversal, with internal
//!    consistency between scales guaranteed.
//!
//! 3. **Footprint-native.** The two-catalog construction is built in, not bolted
//!    on. The random catalog encodes the survey selection function (footprint,
//!    fiber completeness, imaging systematics, dust correction); cell counts of
//!    randoms naturally weight every estimator by its effective volume.
//!
//! ## Performance characteristics
//!
//! Most cascade statistics scale as O(N log N) on N points; many scale as O(N) in
//! the relevant high-density regime. In head-to-head benchmarks against the
//! `HIPSTER` C++ pair-count power spectrum estimator (Philcox & Eisenstein 2019)
//! at survey-relevant N: ~30× wall-time speedup on `P(k)` monopole, ~150× on the
//! bispectrum equivalent, with the cascade additionally producing many other
//! statistics simultaneously at no extra cost. See `examples/` for benchmark
//! scripts.
//!
//! ## Two principal interfaces
//!
//! - [`hier_bitvec`] — single-catalog bit-vector cascade. Produces moments,
//!   pair counts, and density statistics on one point set.
//!
//! - [`hier_bitvec_pair`] — two-catalog bit-vector cascade. The main interface
//!   for survey-realistic analysis: data + randoms, with optional per-point
//!   weights, native footprint handling, and outside-footprint diagnostics.
//!
//! Plus several legacy and specialized variants:
//!
//! - [`hier::cascade_hierarchical_bc`] — 2D, dense u64 buffers, single-thread
//! - [`hier_3d::cascade_3d_with_tpcf`] — 3D version with optional TPCF
//! - [`hier_nd::cascade_nd::<const D: usize>`] — generic D, fastest at every D
//!   for D ≥ 2 thanks to LLVM monomorphization
//! - [`hier_packed::cascade_adaptive`] — 2D with per-level adaptive bit-width
//!   (u8/u16/u32/u64), giving ~2× speedup in MCMC inner loops where the same
//!   field shape is sampled many times
//! - [`hier_par::cascade_hierarchical_par`] — 2D rayon-threaded
//! - [`hier::cascade_hierarchical_with_tpcf`] — 2D with two-point correlation
//! - [`hier::cascade_with_pmf`] — 2D with full per-level cell-count PMFs
//!
//! ## CLI
//!
//! All major statistics families are exposed via the `morton-cascade` binary
//! with subcommands: `cascade`, `pmf`, `pmf-windows`, `tpcf`, `pairs`, `xi`,
//! `field-stats`, `anisotropy`, `gradient`, `fingerprint`. Run
//! `morton-cascade --help` for usage.
//!
//! ## Algorithmic detail: shift averaging
//!
//! The cascade computes statistics shift-averaged over all 2^D parities of the
//! cell-grid origin (the "all-shifts on" trick), so the estimators are isotropic
//! with respect to the cell-grid orientation. For a Poisson process at the
//! finest level, the per-scale variance contribution `ΔV_l / <N>_l = 1 - (2^D −
//! 1) / 4^D` (e.g. 13/16 in 2D, 57/64 in 3D); deviations probe non-Poissonian
//! clustering structure.
//!
//! ## Quick start
//!
//! ```no_run
//! use morton_cascade::hier;
//!
//! // pts: Vec<(u16, u16)> in [0, 2^16)^2 — the unit periodic box, scaled to u16
//! # let pts: Vec<(u16, u16)> = vec![(0, 0)];
//! let (level_stats, n_distinct) = hier::cascade_hierarchical_bc(&pts, 1, true);
//!
//! for (l, st) in level_stats.iter().enumerate() {
//!     let R = 1usize << (morton_cascade::L_MAX - l);
//!     println!("R={:>3}  <N>={:>10.2}  sigma^2={:.4}",
//!              R, st.mean, st.var / st.mean.powi(2));
//! }
//! ```
//!
//! ## Coordinate convention
//!
//! For the legacy 2D modules, input points are unsigned 16-bit per axis (`u16`),
//! interpreted as the unit periodic box \[0, 1)^D scaled to \[0, 2^16). For the
//! bit-vector cascades ([`hier_bitvec`], [`hier_bitvec_pair`]) input is `u64`
//! per axis with `2^L_max` resolution; pass `f64` physical coordinates and a
//! box size to the CLI which packs them automatically.
//!
//! In the cascade, the finest reported level `L_max` corresponds to a cell side
//! of 1 in "tree coordinates" (i.e. the box is `2^L_max` per side, with
//! `L_max=8` in 2D by default giving 256×256 finest cells, or `L_max=7` in 3D
//! giving 128×128×128).

pub mod hier;
pub mod hier_3d;
pub mod hier_nd;
pub mod hier_packed;
pub mod hier_par;
pub mod all_shifts;
pub mod coord_range;
pub mod hier_bitvec;
pub mod hier_bitvec_pair;
pub mod cascade_visitor;
pub mod cell_membership;
pub mod compensated_sum;
pub mod anisotropy_gradient;
pub mod field_stats_gradient;
pub mod xi_gradient;
pub mod multi_run;
pub mod jvp;
pub mod cell_kernels;

// ============================================================================
// Public constants and types shared by the 2D modules
// ============================================================================

/// Default finest level for the 2D cascade. Tree-coord box is 2^L_MAX per side
/// (so L_MAX=8 means 256x256 finest cells).
///
/// To use a different resolution, the [`hier_nd::cascade_nd::<2>`] variant accepts
/// `l_max` as a runtime parameter.
pub const L_MAX: usize = 8;

/// Number of levels reported by the 2D cascade (= L_MAX + 1).
pub const N_LEVELS: usize = L_MAX + 1;

/// Per-level statistics returned by the 2D cascade variants.
#[derive(Clone, Debug)]
pub struct LevelStats {
    /// Total number of cells at this level: 4^l.
    pub n_cells_total: usize,
    /// Mean cell count, averaged over all (cell, shift) pairs at this level.
    pub mean: f64,
    /// Variance of cell counts at this level. For a clustered field,
    /// `sigma^2_field(R) = (var - mean) / mean^2`.
    pub var: f64,
    /// Schur residual (scale-localized variance contribution). For a Poisson
    /// process at the finest level, `dvar / mean = 1 - 3/16 = 13/16`.
    pub dvar: f64,
}

// Re-exports to keep the user-facing namespace tidy
pub use hier_3d::{LevelStats3D, L_MAX_3D, N_LEVELS_3D};
pub use hier_nd::LevelStatsND;
