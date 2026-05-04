//! Cell pair-separation kernels for relating $\xi(r)$ to $\sigma^2(V)$.
//!
//! Given a cell shape $V$ and the probability density $f_V(r)$ that two
//! uniform random points in $V$ are at separation $r$, the
//! cell-averaged correlation function is
//!
//! ```text
//!   ξ̄_V = ∫₀^{r_max} f_V(r) ξ(r) dr
//! ```
//!
//! The clustering variance of counts in $V$ equals $\bar\xi_V$ (Poisson
//! shot noise must be added separately for total observed variance).
//!
//! This module provides:
//!
//! - [`CellKernel`] trait — abstract pair-separation density evaluator
//! - [`AnalyticBoxKernel1D`] — closed-form triangular density for 1D
//!   line segments. Cheap, exact.
//! - [`MonteCarloKernel`] — generic MC-tabulated density that works
//!   for any cell shape in any dimension. Pre-computed by sampling
//!   pairs in a unit cell and histogramming distances.
//! - [`variance_from_xi`] — kernel-weighted integral that turns
//!   per-shell $\xi$ into a single $\bar\xi_V$ value.
//!
//! # Why no closed-form 2D/3D analytic kernels?
//!
//! The pair-separation density inside a 2D square or 3D cube has known
//! closed forms (Philip 2007; Mathai 1999), but they are piecewise —
//! 2 pieces in 2D, 3 pieces in 3D — with `arcsin`/`arccos` in the
//! middle pieces. Transcribing them correctly is error-prone, and
//! once you have them you still verify via Monte Carlo. We skip the
//! closed forms entirely: a Monte Carlo kernel with $10^5$ pair
//! samples gives sub-percent accuracy in milliseconds, computed
//! once and reused. For users who specifically want the analytic
//! forms, this module's MC kernel is a verification tool.
//!
//! # Differentiability
//!
//! The kernel-weighted variance $\hat\sigma^2(V) = \sum_s K_V(\bar r_s) \xi_s \Delta r_s$
//! is a **linear combination of pooled-ξ shell values**. Its gradient
//! w.r.t. data or random weights is therefore one call to the
//! existing `gradient_xi_{data,random}_pooled_aggregate` API with
//! `betas[g][s] = K_V(\bar r_s) · Δr_s`. No new gradient code is
//! required. See the module-level test
//! `variance_from_xi_gradient_via_aggregate_matches_finite_difference`
//! for an end-to-end demonstration.
//!
//! # When to use this
//!
//! - Cross-checking the cascade's two variance estimators
//!   (`analyze_field_stats.var_delta` vs kernel-from-ξ).
//!   Disagreement signals systematics (mask, weighting, shot-noise
//!   estimation).
//! - Predicting $\sigma^2(V)$ from a theoretical $\xi(r)$ without
//!   running a separate variance walk.
//! - Differentiable cell-averaged correlation functions for
//!   parameter-fitting workflows (the gradient comes free; see
//!   above).

use crate::multi_run::{AggregatedXi, XiResizeGroup};

/// A pair-separation density on a cell of side `s`.
///
/// Given a cell of side `s` (linear scale; for D-cubes this is the
/// edge length), the density `f(r)` satisfies
/// $\int_0^{s\sqrt{D}} f(r) dr = 1$ and represents the probability
/// density that two uniform random points in the cell are separated
/// by distance $r$.
pub trait CellKernel {
    /// Evaluate the density `f(r)` at separation `r` for a cell of
    /// linear scale `s`. Returns 0 outside the support.
    fn density(&self, r: f64, s: f64) -> f64;

    /// Maximum separation (support upper bound) for a cell of scale `s`.
    /// For a D-dimensional cube, this is `s * sqrt(D)`.
    fn support_max(&self, s: f64) -> f64;

    /// Dimension D of the cell. Used for default support (cube
    /// diagonal), and as a sanity check by callers.
    fn dim(&self) -> usize;
}

/// Closed-form pair-separation density for a 1D line segment.
///
/// On a segment of length `s`, the distance $r = |x_1 - x_2|$
/// between two uniform random points has triangular density
/// $f(r) = 2(s - r) / s^2$ for $r \in [0, s]$, integrating to 1.
///
/// Trivially exact; useful as a verification reference for the
/// Monte Carlo kernel and for any actually-1D applications.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnalyticBoxKernel1D;

impl CellKernel for AnalyticBoxKernel1D {
    fn density(&self, r: f64, s: f64) -> f64 {
        if r < 0.0 || r > s || s <= 0.0 { return 0.0; }
        2.0 * (s - r) / (s * s)
    }
    fn support_max(&self, s: f64) -> f64 { s }
    fn dim(&self) -> usize { 1 }
}

/// Generic Monte Carlo pair-separation kernel.
///
/// Constructed by sampling `n_pairs` pairs of uniform random points
/// in a unit-side cell of dimension `D`, computing their pairwise
/// distances, and histogramming into `n_bins` bins of constant
/// linear width on $[0, \sqrt{D}]$. Stored as a normalized density
/// table; queried via linear interpolation at evaluation time.
///
/// Once constructed, evaluation is O(1) per call. Construction with
/// $n_\text{pairs} = 10^5, n_\text{bins} = 200$ takes a few
/// milliseconds and gives sub-percent kernel accuracy at the bins
/// where the density is non-negligible.
///
/// **Reproducibility**: the constructor takes a `seed`. Same seed
/// produces the same kernel.
///
/// **Cell shape**: the constructor is generic over a sampler closure,
/// so this works for any cell shape (cube, sphere, mask-shaped
/// region) — pass a sampler that returns uniform points in your
/// cell and the appropriate `support_max` value.
#[derive(Clone, Debug)]
pub struct MonteCarloKernel {
    dim: usize,
    /// Density `f(r)` tabulated at bin midpoints, on a cell of side 1.
    /// Length `n_bins`. Already normalized to integrate to 1 on
    /// `[0, support_max_unit]`.
    density_table: Vec<f64>,
    /// Bin width on the unit cell.
    bin_width_unit: f64,
    /// Support upper bound on the unit cell.
    support_max_unit: f64,
    n_pairs_used: usize,
}

impl MonteCarloKernel {
    /// Construct an MC kernel for a D-dimensional unit cube.
    ///
    /// Convenience constructor: samples uniform pairs in $[0,1]^D$
    /// and tabulates the resulting pair-separation density on
    /// $[0, \sqrt{D}]$.
    pub fn unit_cube<const D: usize>(n_pairs: usize, n_bins: usize, seed: u64) -> Self {
        let mut s = seed;
        let support_max = (D as f64).sqrt();
        let bin_width = support_max / n_bins as f64;
        let mut counts = vec![0u64; n_bins];

        for _ in 0..n_pairs {
            // Two random points in [0,1]^D.
            let mut p = [0.0_f64; 32];   // up to D=32 inline
            assert!(D <= 32, "MonteCarloKernel::unit_cube requires D ≤ 32");
            for d in 0..D {
                p[d] = rand_u01(&mut s);
            }
            let mut q = [0.0_f64; 32];
            for d in 0..D {
                q[d] = rand_u01(&mut s);
            }
            let mut r2 = 0.0_f64;
            for d in 0..D {
                let diff = p[d] - q[d];
                r2 += diff * diff;
            }
            let r = r2.sqrt();
            let bin = ((r / bin_width) as usize).min(n_bins - 1);
            counts[bin] += 1;
        }

        let total = counts.iter().sum::<u64>() as f64;
        let inv = 1.0 / (total * bin_width);
        let density_table: Vec<f64> = counts.iter()
            .map(|&c| c as f64 * inv).collect();

        Self {
            dim: D,
            density_table,
            bin_width_unit: bin_width,
            support_max_unit: support_max,
            n_pairs_used: n_pairs,
        }
    }

    /// Number of MC pair samples used to build this kernel.
    pub fn n_pairs(&self) -> usize { self.n_pairs_used }
}

impl CellKernel for MonteCarloKernel {
    fn density(&self, r: f64, s: f64) -> f64 {
        if s <= 0.0 || r < 0.0 { return 0.0; }
        // Rescale to unit cell: density on a side-s cell at r equals
        // (1/s) · density_unit(r/s).
        let r_unit = r / s;
        if r_unit > self.support_max_unit { return 0.0; }
        // Linear interpolation between bin midpoints.
        let idx_f = r_unit / self.bin_width_unit - 0.5;
        let n = self.density_table.len();
        if idx_f < 0.0 {
            return self.density_table[0] / s;
        }
        let i = idx_f.floor() as usize;
        if i + 1 >= n {
            return self.density_table[n - 1] / s;
        }
        let frac = idx_f - i as f64;
        let v = (1.0 - frac) * self.density_table[i]
            + frac * self.density_table[i + 1];
        v / s
    }

    fn support_max(&self, s: f64) -> f64 { s * self.support_max_unit }
    fn dim(&self) -> usize { self.dim }
}

/// Compute the kernel-weighted integral $\bar\xi_V = \int K_V(r) \xi(r) dr$
/// on a single resize-group of pooled $\xi$ shells, for a cell of
/// linear scale `cell_side`.
///
/// **Discretization**: midpoint rule, summing
/// `density(r̄_s, cell_side) · ξ_s · Δr_s` over shells where
/// `r̄_s = (r_inner + r_outer) / 2` and `Δr_s = r_outer - r_inner`.
///
/// **Returned value**: the cell-averaged correlation function
/// $\bar\xi_V$ (≡ clustering-variance contribution). Add the
/// Poisson shot-noise term $1/N_\text{eff}$ separately if comparing
/// to a directly-measured variance that includes shot noise.
///
/// **Accuracy**: limited by the cascade's shell granularity. For
/// cells whose volume corresponds to a few cascade levels of
/// support, accuracy is typically sub-percent. For cells at the
/// finest level (kernel support ≲ one shell width), discretization
/// error becomes significant.
pub fn variance_from_xi<K: CellKernel>(
    group: &XiResizeGroup,
    kernel: &K,
    cell_side: f64,
) -> f64 {
    let support_max = kernel.support_max(cell_side);
    let mut total = 0.0_f64;
    for shell in &group.shells {
        // Use the cascade's per-shell physical r_inner/r_outer scaled
        // by the resize group's spec_scale (already in physical units
        // here — see XiShellPooled in multi_run/mod.rs which exposes
        // r_center and r_half_width).
        let r_inner = (shell.r_center - shell.r_half_width).max(0.0);
        let r_outer = shell.r_center + shell.r_half_width;
        if r_inner >= support_max { continue; }
        let r_mid = shell.r_center;
        let dr = r_outer - r_inner;
        if !shell.xi_naive.is_finite() { continue; }
        total += kernel.density(r_mid, cell_side) * shell.xi_naive * dr;
    }
    total
}

/// Vector form: compute $\bar\xi_V$ for every resize group in an
/// `AggregatedXi`, given a per-group cell side. Useful when each
/// resize-group corresponds to a different cascade scale.
///
/// `cell_sides[g]` is the cell side to use for resize group `g`.
/// Length must match `agg.by_resize.len()`.
pub fn variance_from_xi_per_group<K: CellKernel>(
    agg: &AggregatedXi<3>,
    kernel: &K,
    cell_sides: &[f64],
) -> Vec<f64> {
    assert_eq!(cell_sides.len(), agg.by_resize.len(),
        "cell_sides length {} != number of resize groups {}",
        cell_sides.len(), agg.by_resize.len());
    agg.by_resize.iter().zip(cell_sides.iter())
        .map(|(g, &s)| variance_from_xi(g, kernel, s))
        .collect()
}

/// Compute the per-shell `betas` weights for use with the cascade's
/// existing `gradient_xi_{data,random}_pooled_aggregate` API.
///
/// Returns `betas[s] = K_V(\bar r_s) · Δr_s` per shell of the given
/// resize group, suitable for passing as the inner `Vec<f64>` in the
/// `betas: &[Vec<f64>]` argument of the aggregate gradient API.
///
/// Combined with the aggregate API, this gives you
/// $\partial \bar\xi_V / \partial w$ via
///
/// ```ignore
/// let mut betas: Vec<Vec<f64>> = vec![vec![]; agg.by_resize.len()];
/// betas[g] = compute_kernel_betas_for_group(&agg.by_resize[g], &kernel, cell_side);
/// // (other resize groups left as empty Vec<f64> — they don't contribute)
/// // ...but the API requires correct shape, so use:
/// let betas: Vec<Vec<f64>> = agg.by_resize.iter().enumerate().map(|(gi, group)| {
///     if gi == g {
///         compute_kernel_betas_for_group(group, &kernel, cell_side)
///     } else {
///         vec![0.0; group.shells.len()]
///     }
/// }).collect();
/// let grad = runner.gradient_xi_data_pooled_aggregate(scale_tol, &betas);
/// ```
pub fn compute_kernel_betas_for_group<K: CellKernel>(
    group: &XiResizeGroup,
    kernel: &K,
    cell_side: f64,
) -> Vec<f64> {
    let support_max = kernel.support_max(cell_side);
    group.shells.iter().map(|shell| {
        let r_inner = (shell.r_center - shell.r_half_width).max(0.0);
        let r_outer = shell.r_center + shell.r_half_width;
        if r_inner >= support_max { return 0.0; }
        let r_mid = shell.r_center;
        let dr = r_outer - r_inner;
        kernel.density(r_mid, cell_side) * dr
    }).collect()
}

// ---------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------

fn rand_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    let bits = z ^ (z >> 31);
    ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D analytic kernel integrates to 1 on [0, s].
    #[test]
    fn analytic_1d_kernel_normalizes_to_one() {
        let kernel = AnalyticBoxKernel1D;
        let s = 2.5_f64;
        let n_grid = 10_000;
        let dx = s / n_grid as f64;
        let mut total = 0.0_f64;
        for i in 0..n_grid {
            let r = (i as f64 + 0.5) * dx;
            total += kernel.density(r, s) * dx;
        }
        // Trapezoidal/midpoint rule on a smooth function: high accuracy.
        assert!((total - 1.0).abs() < 1e-6,
            "1D kernel integral = {} (expected 1.0)", total);
    }

    /// 2D MC kernel integrates to 1 within MC tolerance.
    #[test]
    fn mc_2d_kernel_normalizes_to_one() {
        let kernel = MonteCarloKernel::unit_cube::<2>(50_000, 200, 42);
        let s = 1.0_f64;  // unit cell
        let n_grid = 5000;
        let support_max = kernel.support_max(s);
        let dx = support_max / n_grid as f64;
        let mut total = 0.0_f64;
        for i in 0..n_grid {
            let r = (i as f64 + 0.5) * dx;
            total += kernel.density(r, s) * dx;
        }
        // MC histogram + linear interp: ~1% normalization error.
        assert!((total - 1.0).abs() < 0.02,
            "2D MC kernel integral = {} (expected ~1.0)", total);
    }

    /// 3D MC kernel integrates to 1.
    #[test]
    fn mc_3d_kernel_normalizes_to_one() {
        let kernel = MonteCarloKernel::unit_cube::<3>(100_000, 300, 43);
        let s = 1.0_f64;
        let n_grid = 5000;
        let support_max = kernel.support_max(s);
        let dx = support_max / n_grid as f64;
        let mut total = 0.0_f64;
        for i in 0..n_grid {
            let r = (i as f64 + 0.5) * dx;
            total += kernel.density(r, s) * dx;
        }
        assert!((total - 1.0).abs() < 0.02,
            "3D MC kernel integral = {} (expected ~1.0)", total);
    }

    /// 1D MC kernel agrees with analytic 1D at the bin midpoints.
    /// MC noise is ~ 1/sqrt(N_pairs / N_bins).
    #[test]
    fn mc_1d_kernel_matches_analytic_within_mc_noise() {
        let analytic = AnalyticBoxKernel1D;
        let kernel = MonteCarloKernel::unit_cube::<1>(200_000, 100, 44);
        let s = 1.0_f64;
        // Compare at several r values (avoid endpoints where MC binning
        // has more noise).
        let r_values = [0.1, 0.2, 0.4, 0.6, 0.8];
        let mut max_err = 0.0_f64;
        for &r in &r_values {
            let an = analytic.density(r, s);
            let mc = kernel.density(r, s);
            let rel = (mc - an).abs() / an.max(1e-10);
            max_err = max_err.max(rel);
        }
        // 200k pairs / 100 bins → ~2k samples/bin → 1/sqrt(2k) ≈ 2.2% std.
        assert!(max_err < 0.05,
            "1D MC vs analytic max relative error = {:.2e}", max_err);
    }

    /// MC density at moderate `r` agrees with the analytic 1D
    /// kernel where they share dimension (1D). For 2D and 3D the
    /// closed forms aren't checked here; the normalization and
    /// scaling tests are the substantive correctness checks.
    #[test]
    fn mc_1d_kernel_matches_analytic_at_multiple_radii() {
        let analytic = AnalyticBoxKernel1D;
        let kernel = MonteCarloKernel::unit_cube::<1>(500_000, 200, 45);
        let s = 1.0_f64;
        // Test at several radii avoiding bin-edge artifacts.
        let r_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85];
        let mut max_err = 0.0_f64;
        for &r in &r_values {
            let an = analytic.density(r, s);
            let mc = kernel.density(r, s);
            let rel = (mc - an).abs() / an.max(1e-10);
            max_err = max_err.max(rel);
        }
        // 500k pairs / 200 bins = 2500 / bin → 1/sqrt(2500) ≈ 2% std.
        // But MC values fluctuate around analytic at each bin; max error
        // at most ~5%.
        assert!(max_err < 0.05,
            "1D MC vs analytic at multiple radii max relative error = {:.2e}", max_err);
    }

    /// Kernel evaluated outside support returns 0.
    #[test]
    fn kernels_zero_outside_support() {
        let analytic = AnalyticBoxKernel1D;
        assert_eq!(analytic.density(2.0, 1.0), 0.0);
        assert_eq!(analytic.density(-0.1, 1.0), 0.0);

        let mc = MonteCarloKernel::unit_cube::<3>(1000, 50, 46);
        let s = 1.0_f64;
        let beyond = mc.support_max(s) + 0.01;
        assert_eq!(mc.density(beyond, s), 0.0);
        assert_eq!(mc.density(-0.1, s), 0.0);
    }

    /// Scaling: density on side-s cell at r equals (1/s) × density on
    /// side-1 cell at r/s.
    #[test]
    fn kernel_scaling_invariance() {
        let analytic = AnalyticBoxKernel1D;
        let r_unit = 0.4;
        let f_unit = analytic.density(r_unit, 1.0);
        let s = 3.7_f64;
        let f_scaled = analytic.density(r_unit * s, s);
        assert!((f_scaled - f_unit / s).abs() < 1e-12,
            "1D kernel scaling: f({}, s={}) = {} vs (1/s) f({}, 1) = {}",
            r_unit * s, s, f_scaled, r_unit, f_unit / s);

        let mc = MonteCarloKernel::unit_cube::<2>(20_000, 100, 47);
        let r_unit_mc = 0.5_f64;
        let f_mc_unit = mc.density(r_unit_mc, 1.0);
        let s_mc = 2.5_f64;
        let f_mc_scaled = mc.density(r_unit_mc * s_mc, s_mc);
        assert!((f_mc_scaled - f_mc_unit / s_mc).abs() < 1e-9,
            "2D MC kernel scaling: violated");
    }

    /// Unit test of the kernel-weighted integration on a synthetic
    /// `XiResizeGroup` with constant ξ. The kernel integrates to 1, so
    /// the result must equal that constant ξ value.
    ///
    /// This decouples the integration routine's correctness from the
    /// cascade's specific shell structure. The numerical agreement of
    /// `variance_from_xi` with `analyze_field_stats.var_delta` is a
    /// separate matter that depends on cascade discretization, shot-
    /// noise estimation conventions, and shell boundary details — too
    /// many sources of systematic to make a clean unit test out of.
    /// Users wanting that cross-check should run it on their own
    /// catalogs and inspect the discrepancies.
    #[test]
    fn variance_from_xi_constant_xi_recovers_xi() {
        use crate::multi_run::{XiShellPooled, XiResizeGroup};

        // Synthetic XiResizeGroup: shells covering [0, 1] uniformly,
        // each with the same ξ value. After kernel-weighted integration
        // (kernel integrates to 1), result should equal ξ_const.
        let n_shells = 50;
        let s = 1.0_f64;
        let r_max = s;  // 1D: support is [0, s]
        let dr = r_max / n_shells as f64;
        let xi_const = 0.7_f64;
        let shells: Vec<XiShellPooled> = (0..n_shells).map(|i| {
            let r_mid = (i as f64 + 0.5) * dr;
            XiShellPooled {
                level: i,
                r_center: r_mid,
                r_half_width: dr / 2.0,
                dd_sum: 0.0,
                rr_sum: 1.0,
                dr_sum: 0.0,
                n_d_sum: 100,
                n_r_sum: 100,
                xi_naive: xi_const,
                xi_shift_bootstrap_var: 0.0,
            }
        }).collect();
        let group = XiResizeGroup { scale: 1.0, n_shifts: 1, shells };
        let kernel = AnalyticBoxKernel1D;
        let result = variance_from_xi(&group, &kernel, s);
        // Kernel integrates to 1 over [0, s]; ξ is constant; so result = ξ_const.
        // Midpoint discretization on the triangular kernel converges as
        // O(h²); with h = 0.02 we get ~1e-4 accuracy.
        assert!((result - xi_const).abs() < 1e-3,
            "constant ξ recovery: result {} vs expected {}",
            result, xi_const);
    }

    /// Unit test of `compute_kernel_betas_for_group`: consistency with
    /// `variance_from_xi`. The betas dotted with the shells' xi values
    /// should equal `variance_from_xi(group, kernel, cell_side)`.
    #[test]
    fn compute_kernel_betas_consistent_with_variance_from_xi() {
        use crate::multi_run::{XiShellPooled, XiResizeGroup};

        let n_shells = 30;
        let s = 1.5_f64;
        let r_max = s;
        let dr = r_max / n_shells as f64;
        let shells: Vec<XiShellPooled> = (0..n_shells).map(|i| {
            let r_mid = (i as f64 + 0.5) * dr;
            // Vary ξ across shells.
            let xi_val = 1.0 - r_mid / s;
            XiShellPooled {
                level: i,
                r_center: r_mid,
                r_half_width: dr / 2.0,
                dd_sum: 0.0,
                rr_sum: 1.0,
                dr_sum: 0.0,
                n_d_sum: 100,
                n_r_sum: 100,
                xi_naive: xi_val,
                xi_shift_bootstrap_var: 0.0,
            }
        }).collect();
        let group = XiResizeGroup { scale: 1.0, n_shifts: 1, shells };
        let kernel = AnalyticBoxKernel1D;
        let result = variance_from_xi(&group, &kernel, s);
        let betas = compute_kernel_betas_for_group(&group, &kernel, s);
        let expected: f64 = group.shells.iter().zip(betas.iter())
            .map(|(shell, beta)| shell.xi_naive * beta).sum();
        // Should be exactly equal (same arithmetic).
        assert!((result - expected).abs() < 1e-12,
            "betas-vs-variance_from_xi: {} vs {}", result, expected);
    }

    /// End-to-end differentiability demonstration: the gradient of
    /// kernel-weighted variance w.r.t. data weights, computed by
    /// passing kernel-weighted `betas` to the existing
    /// `gradient_xi_data_pooled_aggregate` API, matches the FD
    /// gradient of `variance_from_xi` re-evaluated at perturbed
    /// weights.
    ///
    /// This proves the assertion in the module docstring: kernel-
    /// weighted variance is differentiable for free, no new gradient
    /// code needed beyond the existing pooled-ξ aggregate API.
    #[test]
    fn variance_from_xi_gradient_via_aggregate_matches_finite_difference() {
        use crate::multi_run::{CascadeRunner, CascadeRunPlan};

        let mut s = 4421_u64;
        let bits = 5u32;
        let mask = (1u64 << bits) - 1;
        let n_d = 30;
        let n_r = 90;
        let pts_d: Vec<[u64; 3]> = (0..n_d).map(|_| [
            (((rand_u01(&mut s) * (1u64 << bits) as f64) as u64) & mask),
            (((rand_u01(&mut s) * (1u64 << bits) as f64) as u64) & mask),
            (((rand_u01(&mut s) * (1u64 << bits) as f64) as u64) & mask),
        ]).collect();
        let mut s2 = 5532_u64;
        let pts_r: Vec<[u64; 3]> = (0..n_r).map(|_| [
            (((rand_u01(&mut s2) * (1u64 << bits) as f64) as u64) & mask),
            (((rand_u01(&mut s2) * (1u64 << bits) as f64) as u64) & mask),
            (((rand_u01(&mut s2) * (1u64 << bits) as f64) as u64) & mask),
        ]).collect();

        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 6643);
        let scale_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        // MC kernel.
        let kernel = MonteCarloKernel::unit_cube::<3>(50_000, 100, 7754);

        // Pick a cell side roughly matching the catalog scale.
        let cell_side = (1u64 << bits) as f64 * 0.25;

        // Compute variance_from_xi at base weights.
        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let agg_base = runner_base.analyze_xi(scale_tol);
        if agg_base.by_resize.is_empty() {
            return;
        }
        let group_idx = 0;
        let v_base = variance_from_xi(
            &agg_base.by_resize[group_idx], &kernel, cell_side);

        // Build kernel betas for the chosen group only.
        let betas: Vec<Vec<f64>> = agg_base.by_resize.iter().enumerate()
            .map(|(g, group)| {
                if g == group_idx {
                    compute_kernel_betas_for_group(group, &kernel, cell_side)
                } else {
                    vec![0.0; group.shells.len()]
                }
            })
            .collect();
        // Compute the gradient via the existing aggregate API.
        let grad = runner_base.gradient_xi_data_pooled_aggregate(scale_tol, &betas);

        // FD: perturb each base weight, recompute variance_from_xi.
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let runner_pert = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd_pert),
                pts_r.clone(), None, bits_arr, plan.clone());
            let agg_pert = runner_pert.analyze_xi(scale_tol);
            // Re-find the same group_idx in the perturbed result.
            // Resize groups should be in the same order (sorted by scale).
            let v_pert = variance_from_xi(
                &agg_pert.by_resize[group_idx], &kernel, cell_side);
            let fd = (v_pert - v_base) / eps;
            let an = grad[i];
            let abs_diff = (fd - an).abs();
            // Tolerance: ξ FD has ~5e-3 absolute error per shell, kernel-
            // weighted sum amplifies this by a small factor.
            assert!(abs_diff < 1e-1,
                "particle {}: FD {} vs analytic {} (diff {})",
                i, fd, an, abs_diff);
        }
    }
}
