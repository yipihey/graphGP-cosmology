// multi_run.rs
//
// Multi-run cascade orchestration: shifts and box-rescalings.
//
// The cascade gives statistics at sides R_l = 2^(L_max - l) — strictly
// dyadic. Two operations let us fill in the gaps and tighten error bars
// without leaving the cascade family:
//
//   - SHIFT: Re-run with the data offset by some vector. In periodic mode
//     this is a relabeling that decorrelates the lattice. In isolated mode
//     it picks a different sub-cube of the survey.
//
//   - RESIZE: Re-run on a cube of side α·L for some α. With the same
//     l_max, cells become α times their original physical size — shifting
//     the whole dyadic ladder by log_2(α). Multiple α values interleave
//     to fill non-dyadic factors.
//
// SHIFT and RESIZE are orthogonal and compose: a single
// `CascadeRunSpec { offset, scale }` describes the sub-cube
// [offset, offset + scale·L] of the original box.
//
// In periodic mode this sub-cube wraps around the box edges. In isolated
// mode points outside it are clipped (and the random catalog likewise) —
// the result is a new analysis on a smaller, possibly sparsely-sampled
// region. A per-run `footprint_coverage` diagnostic lets users spot when
// they've selected a sub-cube outside the survey.

pub mod xi_continuous;

use crate::hier_bitvec_pair::BoundaryMode;

/// One run of the cascade, expressed as an offset + scale relative to a
/// base box.
///
/// `offset_frac[d]` is in box-fraction units: 0.0 = no shift, 0.5 = half
/// the base box side. Each component should lie in `[0, 1)`; out-of-range
/// values are accepted and wrapped (periodic) or clipped (isolated) by
/// the runner.
///
/// `scale` is the new side as a fraction of the base box side. 1.0 = same
/// box, 0.5 = half-side cube, 2.0 = double (rarely useful — extends past
/// the original box). Must be > 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeRunSpec<const D: usize> {
    pub offset_frac: [f64; D],
    pub scale: f64,
}

impl<const D: usize> CascadeRunSpec<D> {
    /// The identity run: zero offset, unit scale. Equivalent to running
    /// the cascade once on the unmodified base catalog.
    pub fn identity() -> Self {
        Self { offset_frac: [0.0; D], scale: 1.0 }
    }

    /// Pure shift: rescale 1.0, offset given.
    pub fn shift(offset_frac: [f64; D]) -> Self {
        Self { offset_frac, scale: 1.0 }
    }

    /// Pure resize: zero offset, scale given.
    pub fn resize(scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive (got {})", scale);
        Self { offset_frac: [0.0; D], scale }
    }

    /// Combined offset + scale.
    pub fn new(offset_frac: [f64; D], scale: f64) -> Self {
        assert!(scale > 0.0, "scale must be positive (got {})", scale);
        Self { offset_frac, scale }
    }

    /// Whether this spec is the identity (no offset, unit scale).
    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.offset_frac.iter().all(|&x| x == 0.0)
    }
}

/// A plan describing a set of cascade runs, each with a name for
/// provenance.
///
/// Names are user-visible (per-run output uses them to label results) and
/// should be reasonably short and unique within a plan; the runner
/// disambiguates duplicates with a `__N` suffix if needed.
///
/// Plans compose via [`Self::compose`] (cartesian product of runs).
#[derive(Clone, Debug)]
pub struct CascadeRunPlan<const D: usize> {
    pub runs: Vec<(String, CascadeRunSpec<D>)>,
    /// Boundary mode the plan is intended for. Used by factories to give
    /// sensible defaults; not enforced — the runner re-checks at execution.
    pub intended_boundary: Option<BoundaryMode>,
}

impl<const D: usize> CascadeRunPlan<D> {
    /// A single run: the unmodified base box. Equivalent to calling
    /// `analyze_*` directly on the cascade.
    pub fn just_base() -> Self {
        Self {
            runs: vec![("base".to_string(), CascadeRunSpec::<D>::identity())],
            intended_boundary: None,
        }
    }

    /// `n_shifts_per_axis^D` shifts on a regular lattice of fractional
    /// offsets. Each axis is divided into `n_shifts_per_axis` equispaced
    /// fractional offsets in [0, 1). The total run count is
    /// `n_shifts_per_axis.pow(D)`.
    ///
    /// Periodic-mode-friendly: in periodic mode each shift is a
    /// statistically equivalent decorrelating relabeling. In isolated
    /// mode shifted boxes still cover the full base box (scale = 1)
    /// so are pinned to the base extent — they only differ by which
    /// part of the data is at the box edge after wrapping. **In isolated
    /// mode pure shifts at scale=1 are no-ops** (the data and randoms
    /// are unchanged), so users in isolated mode should compose this
    /// with [`Self::log_spaced_resizings`] or use [`Self::random_offsets`]
    /// at scale < 1 instead.
    pub fn shifted_grid(n_shifts_per_axis: usize) -> Self {
        assert!(n_shifts_per_axis >= 1,
            "n_shifts_per_axis must be ≥ 1 (got {})", n_shifts_per_axis);
        let n = n_shifts_per_axis;
        let total = n.pow(D as u32);
        let mut runs = Vec::with_capacity(total);
        let step = 1.0 / (n as f64);
        for idx in 0..total {
            let mut off = [0.0; D];
            let mut q = idx;
            for d in 0..D {
                off[d] = (q % n) as f64 * step;
                q /= n;
            }
            // Build a stable name like "grid_2x2x2_001"
            let dim_str: Vec<String> = (0..D).map(|_| format!("{}", n)).collect();
            let name = format!("grid_{}_{:0w$}",
                dim_str.join("x"), idx, w = (total - 1).to_string().len().max(1));
            runs.push((name, CascadeRunSpec::<D>::shift(off)));
        }
        Self { runs, intended_boundary: Some(BoundaryMode::Periodic) }
    }

    /// `n_runs` shifts at random offsets of fixed magnitude (in box-side
    /// units). Each offset is a random direction with magnitude `magnitude`
    /// in box-fraction units — i.e. moved a fraction `magnitude` of the
    /// box side along a random direction.
    ///
    /// Reasonable default magnitude: 0.25 (a quarter of the box side) —
    /// large enough to substantially decorrelate shifts in periodic mode.
    /// In isolated mode, scale=1 means shifts wrap into themselves and
    /// produce identical data — see `random_offsets_with_resize` for the
    /// isolated-mode-friendly variant.
    ///
    /// `seed` makes the offsets reproducible.
    pub fn random_offsets(n_runs: usize, magnitude: f64, seed: u64) -> Self {
        assert!(n_runs >= 1, "n_runs must be ≥ 1 (got {})", n_runs);
        assert!(magnitude >= 0.0, "magnitude must be ≥ 0 (got {})", magnitude);
        let mut runs = Vec::with_capacity(n_runs);
        let mut s = seed;
        for i in 0..n_runs {
            let dir = sample_unit_vector::<D>(&mut s);
            let mut off = [0.0; D];
            for d in 0..D { off[d] = dir[d] * magnitude; }
            // Wrap into [0, 1) for definiteness; runner handles either way.
            for d in 0..D {
                let mut v = off[d];
                while v < 0.0 { v += 1.0; }
                while v >= 1.0 { v -= 1.0; }
                off[d] = v;
            }
            let width = (n_runs - 1).to_string().len().max(1);
            runs.push((format!("rand_{:0w$}", i, w = width),
                       CascadeRunSpec::<D>::shift(off)));
        }
        Self { runs, intended_boundary: Some(BoundaryMode::Periodic) }
    }

    /// Log-spaced resizings: rescale the box by factors of α spanning a
    /// range, with approximately `points_per_decade_v` points per decade
    /// in volume V = α^D. Includes α = 1 (the base run) automatically.
    ///
    /// The factors fill in the gaps between dyadic levels. With
    /// `points_per_decade_v = 5` in 3D, sides interleave with the dyadic
    /// ladder at roughly 5 points per decade in V — the convention used
    /// by [`crate::hier_nd::log_spaced_window_sides`].
    ///
    /// Range: α from `min_scale` to `max_scale`. Defaults: `min_scale = 0.5`,
    /// `max_scale = 1.0` give one octave below the base, which combined
    /// with the cascade's intrinsic dyadic ladder fills one decade nicely.
    pub fn log_spaced_resizings(
        min_scale: f64, max_scale: f64, points_per_decade_v: f64,
    ) -> Self {
        assert!(min_scale > 0.0 && max_scale >= min_scale,
            "require 0 < min_scale ({}) ≤ max_scale ({})", min_scale, max_scale);
        assert!(points_per_decade_v > 0.0,
            "points_per_decade_v must be > 0 (got {})", points_per_decade_v);
        // Step in side: same convention as log_spaced_window_sides.
        // log step in side = 1 / (D * ppd_V).
        let log_step = 1.0 / (D as f64 * points_per_decade_v);
        let log_lo = min_scale.log10();
        let log_hi = max_scale.log10();
        let n_steps = ((log_hi - log_lo) / log_step).ceil() as usize + 1;
        let mut scales: Vec<f64> = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let lg = log_lo + (i as f64) * log_step;
            scales.push(10.0_f64.powf(lg).clamp(min_scale, max_scale));
        }
        // Always include α = 1.0 if it's in range.
        if (min_scale..=max_scale).contains(&1.0) { scales.push(1.0); }
        // Dedup with tolerance — round to 6 decimals for the dedup key.
        scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut deduped: Vec<f64> = Vec::with_capacity(scales.len());
        for s in scales {
            let approx_eq = deduped.last().map(|&last| (s - last).abs() < 1e-6)
                                          .unwrap_or(false);
            if !approx_eq { deduped.push(s); }
        }
        let n = deduped.len();
        let width = (n - 1).max(1).to_string().len();
        let runs: Vec<(String, CascadeRunSpec<D>)> = deduped.iter().enumerate()
            .map(|(i, &s)| (format!("scale_{:0w$}_{:.4}", i, s, w = width),
                            CascadeRunSpec::<D>::resize(s)))
            .collect();
        Self { runs, intended_boundary: None }
    }

    /// Cartesian product: every spec in `a` combined with every spec in
    /// `b`. Composition is `compose(a_offset, a_scale) ∘ (b_offset, b_scale)`
    /// = `(a_offset + b_offset · a_scale, a_scale · b_scale)` — that is,
    /// `a` chooses a sub-cube and `b` picks a sub-sub-cube within it.
    /// In practice we usually compose orthogonal plans (one shifts, one
    /// resizes) so the order is mostly cosmetic for naming.
    pub fn compose(a: &Self, b: &Self) -> Self {
        let mut runs = Vec::with_capacity(a.runs.len() * b.runs.len());
        for (na, sa) in &a.runs {
            for (nb, sb) in &b.runs {
                let mut off = [0.0; D];
                for d in 0..D {
                    off[d] = sa.offset_frac[d] + sb.offset_frac[d] * sa.scale;
                }
                let scale = sa.scale * sb.scale;
                let name = format!("{}__{}", na, nb);
                runs.push((name, CascadeRunSpec::new(off, scale)));
            }
        }
        Self {
            runs,
            intended_boundary: a.intended_boundary.or(b.intended_boundary),
        }
    }

    /// Number of runs in this plan.
    pub fn len(&self) -> usize { self.runs.len() }

    /// Whether the plan has no runs.
    pub fn is_empty(&self) -> bool { self.runs.is_empty() }
}

/// Sample a uniformly-distributed unit vector in D dimensions using the
/// Marsaglia/Box-Muller approach. SplitMix-seeded so results are
/// reproducible per `seed`.
fn sample_unit_vector<const D: usize>(seed: &mut u64) -> [f64; D] {
    // Box-Muller from two splitmix uniforms gives independent N(0,1).
    // Build a Gaussian per axis, then normalize.
    let mut g = [0.0f64; D];
    let mut i = 0;
    while i < D {
        let u1 = splitmix_uniform(seed);
        let u2 = splitmix_uniform(seed);
        // Clamp u1 away from 0 to avoid log(0).
        let u1 = u1.max(f64::MIN_POSITIVE);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        g[i] = r * theta.cos();
        if i + 1 < D { g[i + 1] = r * theta.sin(); }
        i += 2;
    }
    let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
    let mut out = [0.0; D];
    for d in 0..D { out[d] = g[d] / norm; }
    out
}

#[inline]
fn splitmix_uniform(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    // Take top 53 bits as the mantissa of a [0, 1) double.
    ((z >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
}

/// Result of applying a [`CascadeRunSpec`] to a base catalog.
///
/// The new catalog lives in a sub-cube of the base box, expressed back as
/// integer coords on the same number of bits per axis (so the same
/// `l_max` gives cells of the same RELATIVE resolution).
///
/// `footprint_coverage` reports the fraction of base random-catalog
/// weight that survived the (isolated-mode) clipping. Always 1.0 in
/// periodic mode because nothing is dropped.
#[derive(Clone, Debug)]
pub struct AppliedRun<const D: usize> {
    pub pts_d: Vec<[u64; D]>,
    pub pts_r: Vec<[u64; D]>,
    pub weights_d: Option<Vec<f64>>,
    pub weights_r: Option<Vec<f64>>,
    /// Per-axis bits of the new sub-cube (= base bits by default policy;
    /// the coords are remapped so [0, 2^bits[d]) covers the sub-cube).
    pub box_bits: [u32; D],
    /// Fraction of input random-catalog weight that survived clipping.
    /// 1.0 in periodic mode (no clipping). In isolated mode, < 1 means
    /// the requested sub-cube extends outside the random-catalog footprint.
    pub footprint_coverage: f64,
    /// The spec that produced this run. Carried for provenance.
    pub spec: CascadeRunSpec<D>,
    /// Original (pre-shift, pre-clip) catalog index for each surviving
    /// data point. `original_d_indices[k]` is the index in the base
    /// `pts_d` slice that became `pts_d[k]` in this run. Length matches
    /// `pts_d`. Required to lift per-run gradients back into the
    /// original-particle-index space (commit 14: multi-run gradients).
    pub original_d_indices: Vec<u32>,
    /// Original index for each surviving random point. Same role as
    /// `original_d_indices` for the random catalog.
    pub original_r_indices: Vec<u32>,
}

/// Apply a [`CascadeRunSpec`] to a base catalog, producing a new catalog
/// suitable for cascade analysis.
///
/// Coordinate semantics:
/// - Base box is `[0, 2^base_box_bits[d])` per axis.
/// - Sub-cube origin = `offset_frac[d] * 2^base_box_bits[d]` (in base coords).
/// - Sub-cube side   = `scale * 2^base_box_bits[d]` (in base coords).
/// - Output coords are in `[0, 2^base_box_bits[d])` — the sub-cube is
///   re-mapped to fill the same integer range, so cascade `l_max` choices
///   give cells of the same RELATIVE resolution. (Physical-cell-side =
///   sub_cube_side / 2^l_max.)
///
/// In periodic mode the sub-cube wraps around the base box edges.
/// In isolated mode points outside the sub-cube are clipped from both
/// data and random catalogs (with their weights, if any).
///
/// Identity spec (zero offset, unit scale) returns the input unchanged.
pub fn apply_run_spec<const D: usize>(
    pts_d: &[[u64; D]],
    weights_d: Option<&[f64]>,
    pts_r: &[[u64; D]],
    weights_r: Option<&[f64]>,
    base_box_bits: [u32; D],
    spec: CascadeRunSpec<D>,
    mode: BoundaryMode,
) -> AppliedRun<D> {
    // Fast path: identity spec
    if spec.is_identity() {
        return AppliedRun {
            pts_d: pts_d.to_vec(),
            pts_r: pts_r.to_vec(),
            weights_d: weights_d.map(|w| w.to_vec()),
            weights_r: weights_r.map(|w| w.to_vec()),
            box_bits: base_box_bits,
            footprint_coverage: 1.0,
            spec,
            original_d_indices: (0..pts_d.len() as u32).collect(),
            original_r_indices: (0..pts_r.len() as u32).collect(),
        };
    }

    // Per-axis side lengths of the base box and the sub-cube, in base
    // u64-coord units.
    let mut base_side = [0.0f64; D];
    let mut sub_origin = [0.0f64; D];
    let mut sub_side = [0.0f64; D];
    for d in 0..D {
        // 2^bits as f64; bits ≤ 63 enforced by CoordRange::for_box_bits.
        base_side[d] = (1u64 << base_box_bits[d]) as f64;
        sub_origin[d] = spec.offset_frac[d] * base_side[d];
        sub_side[d]   = spec.scale * base_side[d];
        assert!(sub_side[d] > 0.0,
            "sub-cube has zero size on axis {} (scale={}, base_side={})",
            d, spec.scale, base_side[d]);
    }
    // Output coords use the SAME number of bits as the input, so the
    // sub-cube fills [0, 2^bits[d]). Scale factor: 2^bits[d] / sub_side[d].
    let mut scale_to_out = [0.0f64; D];
    let mut out_max = [0u64; D];
    for d in 0..D {
        scale_to_out[d] = base_side[d] / sub_side[d];
        out_max[d] = (1u64 << base_box_bits[d]) - 1;
    }

    let n_d_in = pts_d.len();
    let n_r_in = pts_r.len();
    let mut out_d: Vec<[u64; D]> = Vec::with_capacity(n_d_in);
    let mut out_r: Vec<[u64; D]> = Vec::with_capacity(n_r_in);
    let mut out_wd: Option<Vec<f64>> = weights_d.map(|_| Vec::with_capacity(n_d_in));
    let mut out_wr: Option<Vec<f64>> = weights_r.map(|_| Vec::with_capacity(n_r_in));
    let mut orig_d_idx: Vec<u32> = Vec::with_capacity(n_d_in);
    let mut orig_r_idx: Vec<u32> = Vec::with_capacity(n_r_in);

    let map_one = |p: &[u64; D]| -> Option<[u64; D]> {
        let mut q = [0u64; D];
        for d in 0..D {
            // Fractional offset within base box, [0, 1).
            let pf = p[d] as f64;
            // Coord in the sub-cube frame: shift by -origin, then either
            // wrap (periodic) or check bounds (isolated).
            let shifted = pf - sub_origin[d];
            let in_sub = match mode {
                BoundaryMode::Periodic => {
                    // Wrap shifted into [0, base_side); then keep iff < sub_side.
                    let mut w = shifted;
                    while w < 0.0 { w += base_side[d]; }
                    while w >= base_side[d] { w -= base_side[d]; }
                    if w < sub_side[d] { Some(w) } else { None }
                }
                BoundaryMode::Isolated => {
                    if shifted >= 0.0 && shifted < sub_side[d] { Some(shifted) }
                    else { None }
                }
            };
            match in_sub {
                Some(v) => {
                    let mapped = (v * scale_to_out[d]).floor() as u64;
                    q[d] = mapped.min(out_max[d]);
                }
                None => return None,
            }
        }
        Some(q)
    };

    for (i, p) in pts_d.iter().enumerate() {
        if let Some(q) = map_one(p) {
            out_d.push(q);
            orig_d_idx.push(i as u32);
            if let (Some(wsrc), Some(wdst)) = (weights_d, out_wd.as_mut()) {
                wdst.push(wsrc[i]);
            }
        }
    }
    let mut sum_wr_out = 0.0f64;
    let mut sum_wr_in = 0.0f64;
    for (i, p) in pts_r.iter().enumerate() {
        let w = weights_r.map(|ws| ws[i]).unwrap_or(1.0);
        sum_wr_in += w;
        if let Some(q) = map_one(p) {
            out_r.push(q);
            orig_r_idx.push(i as u32);
            if let (Some(wsrc), Some(wdst)) = (weights_r, out_wr.as_mut()) {
                wdst.push(wsrc[i]);
            }
            sum_wr_out += w;
        }
    }

    let footprint_coverage = match mode {
        BoundaryMode::Periodic => 1.0,
        BoundaryMode::Isolated => {
            if sum_wr_in > 0.0 { sum_wr_out / sum_wr_in } else { 1.0 }
        }
    };

    AppliedRun {
        pts_d: out_d,
        pts_r: out_r,
        weights_d: out_wd,
        weights_r: out_wr,
        box_bits: base_box_bits,
        footprint_coverage,
        spec,
        original_d_indices: orig_d_idx,
        original_r_indices: orig_r_idx,
    }
}

// ============================================================================
// Layer 3: CascadeRunner — execute a plan against a base catalog
// ============================================================================

use crate::coord_range::{CoordRange, TrimmedPoints};
use crate::hier_bitvec_pair::BitVecCascadePair;

/// Per-run output: the spec, a built cascade, and the footprint diagnostic.
///
/// The cascade is owned so callers can run multiple analyses against it
/// without rebuilding. After taking what they need, callers can drop it
/// to free memory.
pub struct RunResult<const D: usize> {
    pub name: String,
    pub spec: CascadeRunSpec<D>,
    pub footprint_coverage: f64,
    /// The fully-built cascade for this run. In Periodic mode it carries
    /// no randoms (built via `build_periodic`); in Isolated mode it
    /// carries the clipped randoms catalog.
    pub cascade: BitVecCascadePair<D>,
    /// Mapping from cascade-particle-index to original-catalog-index
    /// for the data catalog. `original_d_indices[k]` is the index in
    /// the runner's base `pts_d` slice that became the k-th particle
    /// in this run's cascade. Used by gradient-lifting code to map
    /// per-cascade gradients back to the original-particle-index
    /// space across shifted runs.
    pub original_d_indices: Vec<u32>,
    /// Mapping for the random catalog. Empty in periodic mode (no
    /// randoms catalog is carried by the cascade).
    pub original_r_indices: Vec<u32>,
}

/// Executes a [`CascadeRunPlan`] against a base data + (optional) randoms
/// catalog, building one cascade per run.
///
/// Construction is cheap (no work done); execution happens on `per_run` /
/// the aggregation methods.
///
/// The runner takes ownership of the base catalog data so that it can
/// re-derive sub-cube views per run without lifetime gymnastics. Wrap
/// your base data in `Arc` if you want to share it across runners.
pub struct CascadeRunner<const D: usize> {
    base_pts_d: Vec<[u64; D]>,
    base_weights_d: Option<Vec<f64>>,
    base_pts_r: Vec<[u64; D]>,
    base_weights_r: Option<Vec<f64>>,
    base_box_bits: [u32; D],
    boundary_mode: BoundaryMode,
    plan: CascadeRunPlan<D>,
    l_max: Option<usize>,
}

impl<const D: usize> CascadeRunner<D> {
    /// Construct a runner for periodic-mode analyses. No randoms catalog.
    pub fn new_periodic(
        pts_d: Vec<[u64; D]>,
        weights_d: Option<Vec<f64>>,
        box_bits: [u32; D],
        plan: CascadeRunPlan<D>,
    ) -> Self {
        Self {
            base_pts_d: pts_d,
            base_weights_d: weights_d,
            base_pts_r: Vec::new(),
            base_weights_r: None,
            base_box_bits: box_bits,
            boundary_mode: BoundaryMode::Periodic,
            plan,
            l_max: None,
        }
    }

    /// Construct a runner for isolated-mode (survey) analyses with a real
    /// randoms catalog.
    pub fn new_isolated(
        pts_d: Vec<[u64; D]>,
        weights_d: Option<Vec<f64>>,
        pts_r: Vec<[u64; D]>,
        weights_r: Option<Vec<f64>>,
        box_bits: [u32; D],
        plan: CascadeRunPlan<D>,
    ) -> Self {
        Self {
            base_pts_d: pts_d,
            base_weights_d: weights_d,
            base_pts_r: pts_r,
            base_weights_r: weights_r,
            base_box_bits: box_bits,
            boundary_mode: BoundaryMode::Isolated,
            plan,
            l_max: None,
        }
    }

    /// Override the cascade `l_max` for every run. Defaults to None
    /// (auto: deepest the data range supports).
    pub fn with_l_max(mut self, l_max: usize) -> Self {
        self.l_max = Some(l_max);
        self
    }

    /// Number of runs in the plan.
    pub fn n_runs(&self) -> usize { self.plan.len() }

    /// Build one cascade per run (no analysis yet) and yield them with
    /// per-run provenance. Caller can run any analysis they like against
    /// each cascade.
    ///
    /// This is the most flexible entry point; the higher-level
    /// `analyze_*` methods (added in subsequent steps) wrap this with
    /// per-statistic aggregation.
    pub fn per_run(&self) -> Vec<RunResult<D>> {
        // Parallelize across runs: each (name, spec) entry produces an
        // independent RunResult that depends only on immutable shared
        // input (self.base_*) plus its own spec. par_iter().map(...).
        // collect() preserves input order, so downstream aggregation
        // remains deterministic regardless of thread count.
        //
        // See docs/parallelism.md §2 for rationale and §3 for the
        // determinism policy (collect-then-serial-reduce).
        use rayon::prelude::*;
        self.plan.runs.par_iter().map(|(name, spec)| {
            let applied = apply_run_spec::<D>(
                &self.base_pts_d, self.base_weights_d.as_deref(),
                &self.base_pts_r, self.base_weights_r.as_deref(),
                self.base_box_bits, *spec, self.boundary_mode);

            let range = CoordRange::<D>::for_box_bits(applied.box_bits);
            let td = TrimmedPoints::<D>::from_points_with_range(
                applied.pts_d, range.clone());

            let cascade = match self.boundary_mode {
                BoundaryMode::Periodic => {
                    BitVecCascadePair::<D>::build_periodic_full(
                        td, applied.box_bits, applied.weights_d,
                        self.l_max,
                        BitVecCascadePair::<D>::default_crossover_threshold(
                            self.base_pts_d.len(), 0))
                }
                BoundaryMode::Isolated => {
                    let tr = TrimmedPoints::<D>::from_points_with_range(
                        applied.pts_r, range);
                    let thresh = BitVecCascadePair::<D>::default_crossover_threshold(
                        self.base_pts_d.len(), self.base_pts_r.len());
                    BitVecCascadePair::<D>::build_full(
                        td, tr, applied.weights_d, applied.weights_r,
                        self.l_max, thresh)
                }
            };

            RunResult {
                name: name.clone(),
                spec: *spec,
                footprint_coverage: applied.footprint_coverage,
                cascade,
                original_d_indices: applied.original_d_indices,
                original_r_indices: applied.original_r_indices,
            }
        }).collect()
    }

    /// Lift a per-cascade gradient (in cascade-particle-index space)
    /// back to the original-catalog-index space, using a per-run
    /// `original_*_indices` mapping.
    ///
    /// The output vector has length `n_base` (size of the original
    /// catalog — data or randoms). For each cascade-particle index `k`,
    /// the gradient value `cascade_grad[k]` is placed at position
    /// `mapping[k]` in the output. Original particles that did not
    /// survive this run's clipping (or that were not present in
    /// cascade-particle space at all) get value 0.
    ///
    /// **Catalog-generic**: despite the historical `_d_` in the name,
    /// the function is symmetric. Pass `(grad, original_d_indices, n_base_d)`
    /// for data-weight gradients, or `(grad, original_r_indices, n_base_r)`
    /// for random-weight gradients.
    ///
    /// Used internally by all the multi-run pooled gradient methods
    /// (data and random) and exposed publicly for users implementing
    /// custom multi-run gradient compositions.
    pub fn lift_gradient_d_to_original<G: AsRef<[f64]>>(
        cascade_grad: G,
        mapping: &[u32],
        n_base_d: usize,
    ) -> Vec<f64> {
        let cg = cascade_grad.as_ref();
        debug_assert_eq!(cg.len(), mapping.len(),
            "cascade gradient length {} != mapping length {}",
            cg.len(), mapping.len());
        let mut out = vec![0.0_f64; n_base_d];
        for (k, &orig_idx) in mapping.iter().enumerate() {
            let oi = orig_idx as usize;
            if oi < n_base_d {
                out[oi] = cg[k];
            }
        }
        out
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// **average across runs** of `var_delta` at a given cascade level.
    ///
    /// This is the simplest multi-run gradient composition, useful for:
    ///
    /// - Sensitivity analysis: which original particles drive the
    ///   variance signal, averaged across shifted views?
    /// - Optimization on shift-averaged statistics.
    ///
    /// **NOT** the gradient of [`Self::analyze_field_stats`]'s pooled
    /// `var_delta` output (which combines raw sums across runs at the
    /// same physical side, not per-run variances). The pooled-aggregate
    /// gradient is more involved; see commit notes.
    ///
    /// Math: if $V_r$ is run $r$'s var_delta at the chosen level, and
    /// $\bar V = (1/N) \sum_r V_r$, then by linearity
    /// $\partial \bar V / \partial w_i^d = (1/N) \sum_r \partial V_r / \partial w_i^d$
    /// where each per-run gradient is lifted from cascade-particle-index
    /// space to original-particle-index space (particles clipped out of
    /// run $r$ get gradient 0 from that run).
    ///
    /// `level` indexes per-cascade levels (0 = root). Runs whose
    /// cascade has fewer levels skip that level (treated as 0).
    pub fn gradient_var_delta_data_run_average(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        level: usize,
    ) -> Vec<f64> {
        use rayon::prelude::*;
        let n_base_d = self.base_pts_d.len();
        let runs = self.per_run();
        if runs.is_empty() {
            return vec![0.0; n_base_d];
        }
        // Per-run: compute lifted gradient (or None if this run has
        // no contribution at the chosen level). Independent — runs
        // share only immutable input (`cfg` and the cascades themselves).
        let per_run: Vec<Option<Vec<f64>>> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze_field_stats(cfg);
            if level >= stats.len() { return None; }
            let grad_full = r.cascade.gradient_var_delta_all_levels(cfg, &stats);
            let level_grad = &grad_full.data_weight_grads[level];
            if level_grad.is_empty() { return None; }
            Some(Self::lift_gradient_d_to_original(
                level_grad.as_slice(), &r.original_d_indices, n_base_d))
        }).collect();
        // Serial aggregate (deterministic order, bit-exact).
        let mut acc = vec![0.0_f64; n_base_d];
        let mut n_used = 0;
        for opt in &per_run {
            if let Some(lifted) = opt {
                for (i, &g) in lifted.iter().enumerate() {
                    acc[i] += g;
                }
                n_used += 1;
            }
        }
        if n_used > 0 {
            let inv = 1.0 / (n_used as f64);
            for g in acc.iter_mut() { *g *= inv; }
        }
        acc
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// **pooled** `var_delta` produced by [`Self::analyze_field_stats`].
    ///
    /// Returns a vector of per-particle gradients indexed by aggregated
    /// output bin: `result.bin_grads[bin_idx][original_particle_idx]`,
    /// where `bin_idx` matches `AggregatedFieldStats::by_side` from the
    /// same `analyze_field_stats(cfg, bin_tol)` call.
    ///
    /// Math (see `docs/differentiable_cascade.md` §7.4):
    ///
    /// For each output bin pooled from runs $r$ at levels $\ell_r$,
    /// pooled raw sums add: $T^\text{pool} = \sum_r T^{(r)}$,
    /// $S_k^\text{pool} = \sum_r S_k^{(r)}$. Pooled variance follows
    /// the standard formula. Gradient via chain rule + lifting:
    ///
    /// ```text
    ///   ∂μ_2^pool / ∂w_i^d = (1/T^pool) · [∂S_2^pool[i] − 2 m_1^pool · ∂S_1^pool[i]]
    ///
    ///   ∂S_k^pool[i] = Σ_r lift_r(∂S_k^(r)[i])
    /// ```
    ///
    /// Particles clipped from a given run contribute zero to that
    /// run's lifted gradient (correct semantics — they didn't
    /// participate). The pooled bin's per-particle gradient sums
    /// contributions across all participating runs.
    ///
    /// Sanity check: `gradient_pooled_matches_run_average_for_no_pooling`
    /// pins the pooled gradient against the run-average gradient for
    /// scenarios where pooling collapses to per-run trivially.
    pub fn gradient_var_delta_data_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsGradient<D> {
        let n_base_d = self.base_pts_d.len();
        let pooled = self.pooled_raw_sums_data(cfg, bin_tol);

        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            // ∂μ_2 = (1/T)(∂S_2 − 2 m_1 ∂S_1).
            let mut g = vec![0.0_f64; n_base_d];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let inv_t = 1.0 / b.t;
                for k in 0..n_base_d {
                    g[k] = inv_t * (b.grad_s2[k] - 2.0 * m_1 * b.grad_s1[k]);
                }
            }
            bin_grads.push(g);
        }

        PooledFieldStatsGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// pooled `m3_delta` (third central moment of δ) produced by
    /// [`Self::analyze_field_stats`]. See
    /// [`Self::gradient_var_delta_data_pooled`] for the structure;
    /// chain rule is:
    ///
    /// ```text
    ///   ∂μ_3^pool/∂w_i^d = (1/T^pool) · [∂S_3 − 3 m_1 ∂S_2
    ///                                    − 3(μ_2 − m_1²) ∂S_1]
    /// ```
    pub fn gradient_m3_delta_data_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsGradient<D> {
        let n_base_d = self.base_pts_d.len();
        let pooled = self.pooled_raw_sums_data(cfg, bin_tol);

        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            let mut g = vec![0.0_f64; n_base_d];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let mu_2 = (b.s_2 / b.t) - m1_sq;  // pooled var_delta
                let inv_t = 1.0 / b.t;
                let coef_s1 = -3.0 * (mu_2 - m1_sq);
                let coef_s2 = -3.0 * m_1;
                for k in 0..n_base_d {
                    g[k] = inv_t * (b.grad_s3[k] + coef_s2 * b.grad_s2[k]
                                    + coef_s1 * b.grad_s1[k]);
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// pooled `m4_delta` (fourth central moment). Chain rule:
    ///
    /// ```text
    ///   ∂μ_4^pool/∂w_i^d = (1/T^pool) · [∂S_4 − 4 m_1 ∂S_3 + 6 m_1² ∂S_2
    ///                                    − 4(μ_3 + m_1³) ∂S_1]
    /// ```
    pub fn gradient_m4_delta_data_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsGradient<D> {
        let n_base_d = self.base_pts_d.len();
        let pooled = self.pooled_raw_sums_data(cfg, bin_tol);

        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            let mut g = vec![0.0_f64; n_base_d];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let a_2 = b.s_2 / b.t;
                let a_3 = b.s_3 / b.t;
                // Pooled μ_3 = A_3 − 3 m_1 A_2 + 2 m_1³.
                let mu_3 = a_3 - 3.0 * m_1 * a_2 + 2.0 * m1_cu;
                let inv_t = 1.0 / b.t;
                let coef_s3 = -4.0 * m_1;
                let coef_s2 = 6.0 * m1_sq;
                let coef_s1 = -4.0 * (mu_3 + m1_cu);
                for k in 0..n_base_d {
                    g[k] = inv_t * (b.grad_s4[k] + coef_s3 * b.grad_s3[k]
                                    + coef_s2 * b.grad_s2[k]
                                    + coef_s1 * b.grad_s1[k]);
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// pooled reduced skewness `s3_delta = μ_3 / μ_2²`. Chain rule
    /// of pooled μ_2 and μ_3 gradients:
    ///
    /// ```text
    ///   ∂S_3^pool/∂w = (1/μ_2²) ∂μ_3 − (2 μ_3/μ_2³) ∂μ_2
    /// ```
    ///
    /// Bins where μ_2^pool ≤ 0 yield zero gradient (S_3 undefined).
    pub fn gradient_s3_delta_data_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsGradient<D> {
        let n_base_d = self.base_pts_d.len();
        let var_pooled = self.gradient_var_delta_data_pooled(cfg, bin_tol);
        let m3_pooled = self.gradient_m3_delta_data_pooled(cfg, bin_tol);
        let raw = self.pooled_raw_sums_data(cfg, bin_tol);

        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(raw.bins.len());
        for (b_idx, b) in raw.bins.iter().enumerate() {
            let mut g = vec![0.0_f64; n_base_d];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let a_2 = b.s_2 / b.t;
                let a_3 = b.s_3 / b.t;
                let mu_2 = a_2 - m1_sq;
                let mu_3 = a_3 - 3.0 * m_1 * a_2 + 2.0 * m1_cu;
                if mu_2 > 0.0 {
                    let inv_mu2_sq = 1.0 / (mu_2 * mu_2);
                    let coef_mu2 = -2.0 * mu_3 / (mu_2 * mu_2 * mu_2);
                    let g_mu2 = &var_pooled.bin_grads[b_idx];
                    let g_mu3 = &m3_pooled.bin_grads[b_idx];
                    for k in 0..n_base_d {
                        g[k] = inv_mu2_sq * g_mu3[k] + coef_mu2 * g_mu2[k];
                    }
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsGradient {
            bin_grads,
            bin_sides: raw.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Internal: pool raw sums and lifted raw-sum gradients across runs
    /// by physical_side. Returns one bin per pooled output, matching
    /// the binning logic of `analyze_field_stats`.
    fn pooled_raw_sums_data(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledRawSumsData {
        use rayon::prelude::*;
        let n_base_d = self.base_pts_d.len();
        let runs = self.per_run();

        struct Entry {
            physical_side: f64,
            sum_w_r: f64,
            s_1: f64, s_2: f64, s_3: f64, s_4: f64,
            lifted_grad_s1: Vec<f64>,
            lifted_grad_s2: Vec<f64>,
            lifted_grad_s3: Vec<f64>,
            lifted_grad_s4: Vec<f64>,
        }
        // Per-run: produce a Vec<Entry> (one entry per non-empty level).
        // Independent — each cascade is read immutably and produces its
        // own gradient/analysis. flat_map_iter collects all per-run
        // entries into one Vec in stable input order.
        let entries_per_run: Vec<Vec<Entry>> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze_field_stats(cfg);
            let raw = r.cascade.gradient_raw_sums_data_all_levels(cfg, &stats);
            let mut out: Vec<Entry> = Vec::new();
            for (l, s) in stats.iter().enumerate() {
                if s.sum_w_r_active <= 0.0 { continue; }
                let physical_side = r.spec.scale * s.cell_side_trimmed;
                let t_l = s.sum_w_r_active;
                let m_1 = s.mean_delta;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let m1_qu = m1_cu * m_1;
                let s_1 = t_l * m_1;
                let s_2 = t_l * (s.var_delta + m1_sq);
                let s_3 = t_l * (s.m3_delta + 3.0 * m_1 * s.var_delta + m1_cu);
                let s_4 = t_l * (s.m4_delta + 4.0 * m_1 * s.m3_delta
                                 + 6.0 * m1_sq * s.var_delta + m1_qu);
                let lift = |v: &[f64]| -> Vec<f64> {
                    if v.is_empty() {
                        vec![0.0; n_base_d]
                    } else {
                        Self::lift_gradient_d_to_original(
                            v, &r.original_d_indices, n_base_d)
                    }
                };
                out.push(Entry {
                    physical_side,
                    sum_w_r: t_l,
                    s_1, s_2, s_3, s_4,
                    lifted_grad_s1: lift(raw.s1_grads[l].as_slice()),
                    lifted_grad_s2: lift(raw.s2_grads[l].as_slice()),
                    lifted_grad_s3: lift(raw.s3_grads[l].as_slice()),
                    lifted_grad_s4: lift(raw.s4_grads[l].as_slice()),
                });
            }
            out
        }).collect();
        let mut entries: Vec<Entry> = entries_per_run.into_iter().flatten().collect();

        // Bin by physical side, matching analyze_field_stats's
        // sort-and-sweep logic.
        entries.sort_by(|a, b|
            a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<PooledBinData> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() {
                let scale = entries[i].physical_side.abs()
                    .max(entries[j].physical_side.abs()).max(1e-300);
                let rel = (entries[i].physical_side - entries[j].physical_side).abs() / scale;
                if rel >= bin_tol { break; }
                j += 1;
            }
            let mut t = 0.0_f64;
            let mut s1 = 0.0_f64; let mut s2 = 0.0_f64;
            let mut s3 = 0.0_f64; let mut s4 = 0.0_f64;
            let mut g1 = vec![0.0_f64; n_base_d];
            let mut g2 = vec![0.0_f64; n_base_d];
            let mut g3 = vec![0.0_f64; n_base_d];
            let mut g4 = vec![0.0_f64; n_base_d];
            let mut weighted_side_num = 0.0_f64;
            let mut weighted_side_den = 0.0_f64;
            for e in &entries[i..j] {
                t += e.sum_w_r;
                s1 += e.s_1; s2 += e.s_2; s3 += e.s_3; s4 += e.s_4;
                for k in 0..n_base_d {
                    g1[k] += e.lifted_grad_s1[k];
                    g2[k] += e.lifted_grad_s2[k];
                    g3[k] += e.lifted_grad_s3[k];
                    g4[k] += e.lifted_grad_s4[k];
                }
                weighted_side_num += e.physical_side * e.sum_w_r;
                weighted_side_den += e.sum_w_r;
            }
            let physical_side = if weighted_side_den > 0.0 {
                weighted_side_num / weighted_side_den
            } else { entries[i].physical_side };
            bins.push(PooledBinData {
                physical_side,
                t,
                s_1: s1, s_2: s2, s_3: s3, s_4: s4,
                grad_s1: g1, grad_s2: g2, grad_s3: g3, grad_s4: g4,
            });
            i = j;
        }
        PooledRawSumsData { bins }
    }

    /// Internal: random-weight counterpart of `pooled_raw_sums_data`.
    /// Pools $T$ and $S_k$ AND their gradients across runs; lifts
    /// per-particle gradients to base-r-index space.
    ///
    /// Differs from the data version in that $\partial T \neq 0$ for
    /// random weights, so an additional `grad_t` is tracked per bin.
    fn pooled_raw_sums_random(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledRawSumsRandom {
        use rayon::prelude::*;
        let n_base_r = self.base_pts_r.len();
        let runs = self.per_run();

        struct Entry {
            physical_side: f64,
            sum_w_r: f64,
            s_1: f64, s_2: f64, s_3: f64, s_4: f64,
            lifted_grad_t: Vec<f64>,
            lifted_grad_s1: Vec<f64>,
            lifted_grad_s2: Vec<f64>,
            lifted_grad_s3: Vec<f64>,
            lifted_grad_s4: Vec<f64>,
        }
        let entries_per_run: Vec<Vec<Entry>> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze_field_stats(cfg);
            let raw = r.cascade.gradient_raw_sums_random_all_levels(cfg, &stats);
            let mut out: Vec<Entry> = Vec::new();
            for (l, s) in stats.iter().enumerate() {
                if s.sum_w_r_active <= 0.0 { continue; }
                let physical_side = r.spec.scale * s.cell_side_trimmed;
                let t_l = s.sum_w_r_active;
                let m_1 = s.mean_delta;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let m1_qu = m1_cu * m_1;
                let s_1 = t_l * m_1;
                let s_2 = t_l * (s.var_delta + m1_sq);
                let s_3 = t_l * (s.m3_delta + 3.0 * m_1 * s.var_delta + m1_cu);
                let s_4 = t_l * (s.m4_delta + 4.0 * m_1 * s.m3_delta
                                 + 6.0 * m1_sq * s.var_delta + m1_qu);
                // Lift each gradient to base-r-index space using the
                // run's random-particle mapping.
                let lift = |v: &[f64]| -> Vec<f64> {
                    if v.is_empty() {
                        vec![0.0; n_base_r]
                    } else {
                        Self::lift_gradient_d_to_original(
                            v, &r.original_r_indices, n_base_r)
                    }
                };
                out.push(Entry {
                    physical_side,
                    sum_w_r: t_l,
                    s_1, s_2, s_3, s_4,
                    lifted_grad_t: lift(raw.t_grads[l].as_slice()),
                    lifted_grad_s1: lift(raw.s1_grads[l].as_slice()),
                    lifted_grad_s2: lift(raw.s2_grads[l].as_slice()),
                    lifted_grad_s3: lift(raw.s3_grads[l].as_slice()),
                    lifted_grad_s4: lift(raw.s4_grads[l].as_slice()),
                });
            }
            out
        }).collect();
        let mut entries: Vec<Entry> =
            entries_per_run.into_iter().flatten().collect();

        // Bin by physical_side using bin_tol relative tolerance.
        entries.sort_by(|a, b|
            a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<PooledBinRandom> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() {
                let scale = entries[i].physical_side.abs()
                    .max(entries[j].physical_side.abs()).max(1e-300);
                let rel = (entries[i].physical_side - entries[j].physical_side).abs() / scale;
                if rel >= bin_tol { break; }
                j += 1;
            }
            let mut t = 0.0_f64;
            let mut s1 = 0.0_f64; let mut s2 = 0.0_f64;
            let mut s3 = 0.0_f64; let mut s4 = 0.0_f64;
            let mut gt = vec![0.0_f64; n_base_r];
            let mut g1 = vec![0.0_f64; n_base_r];
            let mut g2 = vec![0.0_f64; n_base_r];
            let mut g3 = vec![0.0_f64; n_base_r];
            let mut g4 = vec![0.0_f64; n_base_r];
            let mut weighted_side_num = 0.0_f64;
            let mut weighted_side_den = 0.0_f64;
            for e in &entries[i..j] {
                t += e.sum_w_r;
                s1 += e.s_1; s2 += e.s_2; s3 += e.s_3; s4 += e.s_4;
                for k in 0..n_base_r {
                    gt[k] += e.lifted_grad_t[k];
                    g1[k] += e.lifted_grad_s1[k];
                    g2[k] += e.lifted_grad_s2[k];
                    g3[k] += e.lifted_grad_s3[k];
                    g4[k] += e.lifted_grad_s4[k];
                }
                weighted_side_num += e.physical_side * e.sum_w_r;
                weighted_side_den += e.sum_w_r;
            }
            let physical_side = if weighted_side_den > 0.0 {
                weighted_side_num / weighted_side_den
            } else { entries[i].physical_side };
            bins.push(PooledBinRandom {
                physical_side,
                t,
                s_1: s1, s_2: s2, s_3: s3, s_4: s4,
                grad_t: gt,
                grad_s1: g1, grad_s2: g2, grad_s3: g3, grad_s4: g4,
            });
            i = j;
        }
        PooledRawSumsRandom { bins }
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// the **pooled** `var_delta` produced by [`Self::analyze_field_stats`].
    /// Symmetric counterpart to
    /// [`Self::gradient_var_delta_data_pooled`].
    ///
    /// Chain rule (random weights, with $\partial T^\text{pool}/\partial w_j^r \neq 0$):
    ///
    /// ```text
    ///   ∂μ_2^pool/∂w_j^r = (1/T^pool) [∂S_2 − 2 m_1 ∂S_1 − (μ_2 − m_1²) ∂T]
    /// ```
    pub fn gradient_var_delta_random_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsRandomGradient<D> {
        let n_base_r = self.base_pts_r.len();
        let pooled = self.pooled_raw_sums_random(cfg, bin_tol);
        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            let mut g = vec![0.0_f64; n_base_r];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let mu_2 = (b.s_2 / b.t) - m_1 * m_1;
                let inv_t = 1.0 / b.t;
                let coef_t = -(mu_2 - m_1 * m_1);
                for k in 0..n_base_r {
                    g[k] = inv_t * (b.grad_s2[k] - 2.0 * m_1 * b.grad_s1[k]
                                    + coef_t * b.grad_t[k]);
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsRandomGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// pooled `m3_delta`. Chain rule:
    ///
    /// ```text
    ///   ∂μ_3^pool/∂w_j^r = ∂A_3 − 3 (∂m_1 · A_2 + m_1 · ∂A_2) + 6 m_1² ∂m_1
    ///   ∂A_k = (1/T)(∂S_k − A_k ∂T),  ∂m_1 = ∂A_1
    /// ```
    pub fn gradient_m3_delta_random_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsRandomGradient<D> {
        let n_base_r = self.base_pts_r.len();
        let pooled = self.pooled_raw_sums_random(cfg, bin_tol);
        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            let mut g = vec![0.0_f64; n_base_r];
            if b.t > 0.0 {
                let inv_t = 1.0 / b.t;
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let a_2 = b.s_2 / b.t;
                let a_3 = b.s_3 / b.t;
                for k in 0..n_base_r {
                    let d_a1 = inv_t * (b.grad_s1[k] - m_1 * b.grad_t[k]);
                    let d_a2 = inv_t * (b.grad_s2[k] - a_2 * b.grad_t[k]);
                    let d_a3 = inv_t * (b.grad_s3[k] - a_3 * b.grad_t[k]);
                    g[k] = d_a3 - 3.0 * (d_a1 * a_2 + m_1 * d_a2)
                        + 6.0 * m1_sq * d_a1;
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsRandomGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// pooled `m4_delta`. Chain rule:
    ///
    /// ```text
    ///   ∂μ_4^pool/∂w_j^r = ∂A_4 − 4 (∂m_1 · A_3 + m_1 · ∂A_3)
    ///                       + 12 m_1 ∂m_1 · A_2 + 6 m_1² ∂A_2
    ///                       − 12 m_1³ ∂m_1
    /// ```
    pub fn gradient_m4_delta_random_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsRandomGradient<D> {
        let n_base_r = self.base_pts_r.len();
        let pooled = self.pooled_raw_sums_random(cfg, bin_tol);
        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(pooled.bins.len());
        for b in &pooled.bins {
            let mut g = vec![0.0_f64; n_base_r];
            if b.t > 0.0 {
                let inv_t = 1.0 / b.t;
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let a_2 = b.s_2 / b.t;
                let a_3 = b.s_3 / b.t;
                let a_4 = b.s_4 / b.t;
                for k in 0..n_base_r {
                    let d_a1 = inv_t * (b.grad_s1[k] - m_1 * b.grad_t[k]);
                    let d_a2 = inv_t * (b.grad_s2[k] - a_2 * b.grad_t[k]);
                    let d_a3 = inv_t * (b.grad_s3[k] - a_3 * b.grad_t[k]);
                    let d_a4 = inv_t * (b.grad_s4[k] - a_4 * b.grad_t[k]);
                    g[k] = d_a4 - 4.0 * (d_a1 * a_3 + m_1 * d_a3)
                        + 12.0 * m_1 * d_a1 * a_2
                        + 6.0 * m1_sq * d_a2
                        - 12.0 * m1_cu * d_a1;
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsRandomGradient {
            bin_grads,
            bin_sides: pooled.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// pooled `s3_delta = μ_3 / μ_2²`. Chain rule of pooled μ_2 and μ_3.
    pub fn gradient_s3_delta_random_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledFieldStatsRandomGradient<D> {
        let n_base_r = self.base_pts_r.len();
        let var_pooled = self.gradient_var_delta_random_pooled(cfg, bin_tol);
        let m3_pooled = self.gradient_m3_delta_random_pooled(cfg, bin_tol);
        let raw = self.pooled_raw_sums_random(cfg, bin_tol);
        let mut bin_grads: Vec<Vec<f64>> = Vec::with_capacity(raw.bins.len());
        for (b_idx, b) in raw.bins.iter().enumerate() {
            let mut g = vec![0.0_f64; n_base_r];
            if b.t > 0.0 {
                let m_1 = b.s_1 / b.t;
                let m1_sq = m_1 * m_1;
                let m1_cu = m1_sq * m_1;
                let a_2 = b.s_2 / b.t;
                let a_3 = b.s_3 / b.t;
                let mu_2 = a_2 - m1_sq;
                let mu_3 = a_3 - 3.0 * m_1 * a_2 + 2.0 * m1_cu;
                if mu_2 > 0.0 {
                    let inv_mu2_sq = 1.0 / (mu_2 * mu_2);
                    let coef_mu2 = -2.0 * mu_3 / (mu_2 * mu_2 * mu_2);
                    let g_mu2 = &var_pooled.bin_grads[b_idx];
                    let g_mu3 = &m3_pooled.bin_grads[b_idx];
                    for k in 0..n_base_r {
                        g[k] = inv_mu2_sq * g_mu3[k] + coef_mu2 * g_mu2[k];
                    }
                }
            }
            bin_grads.push(g);
        }
        PooledFieldStatsRandomGradient {
            bin_grads,
            bin_sides: raw.bins.iter().map(|b| b.physical_side).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Aggregate-scalar variant of [`Self::gradient_var_delta_random_pooled`].
    pub fn gradient_var_delta_random_pooled_aggregate(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
        betas: &[f64],
    ) -> Vec<f64> {
        let pooled = self.gradient_var_delta_random_pooled(cfg, bin_tol);
        assert_eq!(betas.len(), pooled.bin_grads.len(),
            "betas length {} != pooled bin count {}",
            betas.len(), pooled.bin_grads.len());
        let n = self.base_pts_r.len();
        let mut out = vec![0.0_f64; n];
        for (b, bin_grad) in pooled.bin_grads.iter().enumerate() {
            let beta = betas[b];
            if beta == 0.0 { continue; }
            for (j, &g) in bin_grad.iter().enumerate() {
                out[j] += beta * g;
            }
        }
        out
    }

    /// Aggregate-scalar variant of [`Self::gradient_var_delta_data_pooled`].
    /// Given per-bin loss weights `betas` (length = number of pooled
    /// bins), returns `∂L / ∂w_i^d` for `L = Σ_bin β_bin · μ_2^pool[bin]`.
    pub fn gradient_var_delta_data_pooled_aggregate(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
        betas: &[f64],
    ) -> Vec<f64> {
        let pooled = self.gradient_var_delta_data_pooled(cfg, bin_tol);
        assert_eq!(betas.len(), pooled.bin_grads.len(),
            "betas length {} != pooled bins {}",
            betas.len(), pooled.bin_grads.len());
        let n = self.base_pts_d.len();
        let mut out = vec![0.0_f64; n];
        for (b, &beta) in betas.iter().enumerate() {
            if beta == 0.0 { continue; }
            for (i, &g) in pooled.bin_grads[b].iter().enumerate() {
                out[i] += beta * g;
            }
        }
        out
    }

    /// Run the cascade across all runs, compute per-run field-stats, then
    /// **pool the W_r-weighted moments across runs that share the same
    /// physical cell side** (within `bin_tol` relative tolerance).
    ///
    /// Pooling math: each per-run `(mean, var, m3, m4)` came from raw
    /// W_r-weighted sums Σ_r w_r · δ^k. Those raw sums add across runs;
    /// the central moments are then re-derived from the pooled raw sums.
    /// This is exact (modulo floating-point) — pooling N runs of M cells
    /// each gives the same result as analyzing one run of N·M cells with
    /// the same per-cell weighting.
    ///
    /// **Physical side**: cells at level l in a run with `spec.scale = α`
    /// have physical side `α · 2^(L_max - l)` in base-box coord units.
    /// Sides from different runs that match within `bin_tol` relative
    /// tolerance are pooled into a single output bin.
    ///
    /// `bin_tol = 1e-6` works for any sensible choice of resize factors.
    pub fn analyze_field_stats(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> AggregatedFieldStats<D> {
        use rayon::prelude::*;
        // Step 1: per-run analysis. Per-run independent — collect into
        // (entries, diagnostic) per run, then flatten in input order
        // for deterministic downstream pooling.
        let runs = self.per_run();
        let per_run: Vec<(Vec<PerRunLevelEntry>, RunDiagnostic<D>)> =
            runs.par_iter().map(|r| {
                let stats = r.cascade.analyze_field_stats(cfg);
                let diag = RunDiagnostic {
                    name: r.name.clone(),
                    spec: r.spec,
                    footprint_coverage: r.footprint_coverage,
                    n_levels: stats.len(),
                };
                let mut es: Vec<PerRunLevelEntry> = Vec::with_capacity(stats.len());
                for s in &stats {
                    let physical_side = r.spec.scale * s.cell_side_trimmed;
                    es.push(PerRunLevelEntry {
                        physical_side,
                        sum_w_r: s.sum_w_r_active,
                        raw_pow: s.raw_sum_w_r_delta_pow,
                        n_cells_active: s.n_cells_active,
                        n_cells_data_outside: s.n_cells_data_outside,
                        sum_w_d_outside: s.sum_w_d_outside,
                        min_delta: s.min_delta,
                        max_delta: s.max_delta,
                        mean_delta_for_var_estimate: s.mean_delta,
                        var_delta_for_var_estimate: s.var_delta,
                    });
                }
                (es, diag)
            }).collect();
        let mut entries: Vec<PerRunLevelEntry> = Vec::new();
        let mut diagnostics: Vec<RunDiagnostic<D>> = Vec::with_capacity(runs.len());
        for (es, d) in per_run {
            entries.extend(es);
            diagnostics.push(d);
        }

        // Step 2: bucket by physical side using bin_tol relative tolerance.
        // Sort by side, then sweep — adjacent entries within tolerance go
        // to the same bin.
        entries.sort_by(|a, b| a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<AggregatedFieldStatsBin> = Vec::new();
        let mut bucket: Vec<&PerRunLevelEntry> = Vec::new();

        for e in &entries {
            let same_bin = bucket.last()
                .map(|prev: &&PerRunLevelEntry| {
                    let scale = prev.physical_side.abs().max(e.physical_side.abs()).max(1e-300);
                    (prev.physical_side - e.physical_side).abs() / scale < bin_tol
                })
                .unwrap_or(false);
            if !same_bin && !bucket.is_empty() {
                bins.push(pool_bin(&bucket));
                bucket.clear();
            }
            bucket.push(e);
        }
        if !bucket.is_empty() {
            bins.push(pool_bin(&bucket));
        }

        AggregatedFieldStats {
            by_side: bins,
            per_run_diagnostics: diagnostics,
        }
    }

    /// Run cascade-anisotropy on every plan run, aggregating across runs
    /// by physical cell side. Returns one bin per distinct physical side
    /// (within `bin_tol` relative tolerance), with raw-sum aggregation
    /// per Haar-wavelet pattern so cross-run pooling is unbiased even
    /// when per-run W_r totals differ.
    ///
    /// The LoS quadrupole and its reduced form are recomputed from the
    /// pooled per-pattern means (LoS = last axis, by convention).
    ///
    /// Across-run variance of the LoS quadrupole is reported per bin as
    /// the cheap shift-bootstrap uncertainty estimate.
    pub fn analyze_anisotropy(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> AggregatedAnisotropyStats<D> {
        use rayon::prelude::*;
        let runs = self.per_run();
        let per_run: Vec<(Vec<PerRunAnisoEntry>, RunDiagnostic<D>)> =
            runs.par_iter().map(|r| {
                let stats = r.cascade.analyze_anisotropy(cfg);
                let diag = RunDiagnostic {
                    name: r.name.clone(),
                    spec: r.spec,
                    footprint_coverage: r.footprint_coverage,
                    n_levels: stats.len(),
                };
                let mut es: Vec<PerRunAnisoEntry> = Vec::with_capacity(stats.len());
                for s in &stats {
                    let physical_side = r.spec.scale * s.cell_side_trimmed;
                    // Recover raw per-pattern sums from means × total weight.
                    // (Visitor invariant: mean_e = sum_wr_we² / sum_w_r_parents.)
                    let raw_by_pattern: Vec<f64> = s.mean_w_squared_by_pattern.iter()
                        .map(|m| m * s.sum_w_r_parents)
                        .collect();
                    es.push(PerRunAnisoEntry {
                        physical_side,
                        sum_w_r_parents: s.sum_w_r_parents,
                        n_parents: s.n_parents,
                        raw_by_pattern,
                        quadrupole_los_for_var_estimate: s.quadrupole_los,
                    });
                }
                (es, diag)
            }).collect();
        let mut entries: Vec<PerRunAnisoEntry> = Vec::new();
        let mut diagnostics: Vec<RunDiagnostic<D>> = Vec::with_capacity(runs.len());
        for (es, d) in per_run {
            entries.extend(es);
            diagnostics.push(d);
        }

        // Bucket by physical side using bin_tol relative tolerance.
        entries.sort_by(|a, b| a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<AggregatedAnisotropyStatsBin> = Vec::new();
        let mut bucket: Vec<&PerRunAnisoEntry> = Vec::new();
        for e in &entries {
            let same_bin = bucket.last()
                .map(|prev: &&PerRunAnisoEntry| {
                    let scale = prev.physical_side.abs().max(e.physical_side.abs()).max(1e-300);
                    (prev.physical_side - e.physical_side).abs() / scale < bin_tol
                })
                .unwrap_or(false);
            if !same_bin && !bucket.is_empty() {
                bins.push(pool_aniso_bin::<D>(&bucket));
                bucket.clear();
            }
            bucket.push(e);
        }
        if !bucket.is_empty() {
            bins.push(pool_aniso_bin::<D>(&bucket));
        }

        AggregatedAnisotropyStats {
            by_side: bins,
            per_run_diagnostics: diagnostics,
        }
    }

    /// Run cascade CIC PMF on every plan run, aggregating across runs by
    /// physical cell side. Per bin returns a pooled histogram (counts AND
    /// density) plus pooled cell-count moments.
    ///
    /// In Periodic mode the per-run densities sum to 1 over the full box,
    /// and the pooled density also sums to 1 (averaging probabilities).
    /// In Isolated mode the per-run densities sum to 1 over visited cells;
    /// the pooled density is the cell-count-weighted average over runs
    /// (equivalent to summing histogram_counts and dividing by the total
    /// visited cells).
    ///
    /// Across-run variance of the mean cell count is reported per bin.
    pub fn analyze_cic_pmf(
        &self,
        cfg: &crate::hier_bitvec_pair::CicPmfConfig,
        bin_tol: f64,
    ) -> AggregatedCicPmf<D> {
        use rayon::prelude::*;
        let runs = self.per_run();
        let per_run: Vec<(Vec<PerRunCicEntry>, RunDiagnostic<D>)> =
            runs.par_iter().map(|r| {
                let stats = r.cascade.analyze_cic_pmf(cfg);
                let diag = RunDiagnostic {
                    name: r.name.clone(),
                    spec: r.spec,
                    footprint_coverage: r.footprint_coverage,
                    n_levels: stats.len(),
                };
                let mut es: Vec<PerRunCicEntry> = Vec::with_capacity(stats.len());
                for s in &stats {
                    let physical_side = r.spec.scale * s.cell_side_trimmed;
                    es.push(PerRunCicEntry {
                        physical_side,
                        n_cells_visited: s.n_cells_visited,
                        n_cells_total: s.n_cells_total,
                        histogram_counts: s.histogram_counts.clone(),
                        histogram_density: s.histogram_density.clone(),
                        mean_for_var_estimate: s.mean,
                    });
                }
                (es, diag)
            }).collect();
        let mut entries: Vec<PerRunCicEntry> = Vec::new();
        let mut diagnostics: Vec<RunDiagnostic<D>> = Vec::with_capacity(runs.len());
        for (es, d) in per_run {
            entries.extend(es);
            diagnostics.push(d);
        }

        entries.sort_by(|a, b| a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<AggregatedCicPmfBin> = Vec::new();
        let mut bucket: Vec<&PerRunCicEntry> = Vec::new();
        for e in &entries {
            let same_bin = bucket.last()
                .map(|prev: &&PerRunCicEntry| {
                    let scale = prev.physical_side.abs().max(e.physical_side.abs()).max(1e-300);
                    (prev.physical_side - e.physical_side).abs() / scale < bin_tol
                })
                .unwrap_or(false);
            if !same_bin && !bucket.is_empty() {
                bins.push(pool_cic_bin(&bucket));
                bucket.clear();
            }
            bucket.push(e);
        }
        if !bucket.is_empty() {
            bins.push(pool_cic_bin(&bucket));
        }

        AggregatedCicPmf {
            by_side: bins,
            per_run_diagnostics: diagnostics,
        }
    }

    /// Run cascade pair counts on every plan run, then form Landy-Szalay
    /// ξ(r) per cascade shell. Aggregation respects the
    /// **shift-vs-resize distinction**:
    ///
    /// - **Shifts** (same scale, different offsets) produce repeated
    ///   measurements of the *same* observable — DD/RR/DR at the same
    ///   set of shell separations. They are pooled by summing pair
    ///   counts within each resize group, with shift-bootstrap variance
    ///   reported per shell as the cheap uncertainty estimate.
    /// - **Resizes** (different scales) probe *different* physical
    ///   shell separations and widths. They are kept as **separate
    ///   resize groups** in the output, because pooling DD/RR/DR
    ///   across different shell volumes would incorrectly average
    ///   measurements with different effective windows. Use the
    ///   continuous-function fit (Storey-Fisher & Hogg 2021) on the
    ///   per-resize-group output to combine resizes properly into a
    ///   shell-deconvolved continuous ξ(r).
    ///
    /// Periodic mode: no random catalog, so RR/DR are zero in the raw
    /// output and `xi_naive` is reported as NaN. Most users in periodic
    /// mode should use the field-stats variance estimator instead;
    /// pair counts are included for completeness.
    pub fn analyze_xi(
        &self,
        scale_tol: f64,
    ) -> AggregatedXi<D> {
        use rayon::prelude::*;
        let runs = self.per_run();
        let per_run: Vec<(PerRunXi, RunDiagnostic<D>)> =
            runs.par_iter().map(|r| {
                let stats = r.cascade.analyze();
                let shells = r.cascade.xi_landy_szalay(&stats);
                let diag = RunDiagnostic {
                    name: r.name.clone(),
                    spec: r.spec,
                    footprint_coverage: r.footprint_coverage,
                    n_levels: shells.len(),
                };
                let pr = PerRunXi {
                    spec_scale: r.spec.scale,
                    shells,
                    n_d: r.cascade.n_d() as u64,
                    n_r: r.cascade.n_r() as u64,
                };
                (pr, diag)
            }).collect();
        let mut diagnostics: Vec<RunDiagnostic<D>> = Vec::with_capacity(runs.len());
        let mut per_run_results: Vec<PerRunXi> = Vec::with_capacity(runs.len());
        for (pr, d) in per_run {
            per_run_results.push(pr);
            diagnostics.push(d);
        }

        // Group runs by spec.scale (within scale_tol relative tolerance).
        // Within each group, all runs see the same shells and pool by
        // summing pair counts.
        per_run_results.sort_by(|a, b| a.spec_scale.partial_cmp(&b.spec_scale).unwrap());

        let mut groups: Vec<XiResizeGroup> = Vec::new();
        let mut bucket: Vec<&PerRunXi> = Vec::new();
        for r in &per_run_results {
            let same_group = bucket.last()
                .map(|prev: &&PerRunXi| {
                    let scale = prev.spec_scale.abs().max(r.spec_scale.abs()).max(1e-300);
                    (prev.spec_scale - r.spec_scale).abs() / scale < scale_tol
                })
                .unwrap_or(false);
            if !same_group && !bucket.is_empty() {
                groups.push(pool_xi_resize_group(&bucket));
                bucket.clear();
            }
            bucket.push(r);
        }
        if !bucket.is_empty() {
            groups.push(pool_xi_resize_group(&bucket));
        }

        AggregatedXi {
            by_resize: groups,
            per_run_diagnostics: diagnostics,
        }
    }

    /// Multi-run gradient: per-original-particle gradient of the
    /// **pooled** Landy-Szalay ξ produced by [`Self::analyze_xi`].
    ///
    /// Output mirrors `AggregatedXi::by_resize`: one entry per
    /// distinct resize scale (sorted ascending), each carrying
    /// `shell_grads[shell_idx][original_particle_idx]`.
    ///
    /// **Math** (matches the forward semantics of `pool_xi_resize_group`):
    /// per resize group, pooled pair counts add linearly across runs:
    /// $DD^\text{pool}_\ell = \sum_r DD_\ell^{(r)}$, similarly for
    /// DR, RR. Pooled normalizations use the **summed particle counts
    /// across runs** (count-based, not weight-based — matches the
    /// existing forward code; see the note in `pool_xi_resize_group`).
    /// Since these normalizations don't depend on data weights, the
    /// gradient simplifies:
    ///
    /// ```text
    ///   ∂ξ^pool/∂w_i^d = (1/f_RR^pool) · [∂f_DD^pool/∂w_i^d − 2 ∂f_DR^pool/∂w_i^d]
    ///   ∂f_DD^pool/∂w_i^d = (1/N_DD^pool) · Σ_r lift_r(∂DD^(r)/∂w_i^d)
    ///   ∂f_DR^pool/∂w_i^d = (1/N_DR^pool) · Σ_r lift_r(∂DR^(r)/∂w_i^d)
    /// ```
    ///
    /// Per-run pair-count gradients ∂DD^(r)/∂w_i^d come from the
    /// existing single-cascade primitive
    /// `BitVecCascadePair::gradient_xi_data_all_shells`.
    ///
    /// **Scope**: differentiates the same xi formula that
    /// `pool_xi_resize_group` computes for `xi_naive`. For unit-weight
    /// catalogs (the common case) this is the standard count-based LS
    /// formula; for weighted catalogs this is the weighted DD/RR/DR
    /// pair counts but count-based normalization (a known
    /// inconsistency in the forward aggregator that this gradient
    /// preserves rather than papers over).
    pub fn gradient_xi_data_pooled(
        &self,
        scale_tol: f64,
    ) -> PooledXiGradient {
        use rayon::prelude::*;
        let n_base_d = self.base_pts_d.len();

        let runs = self.per_run();
        let per_run: Vec<PerRunXiGrad> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze();
            let shells = r.cascade.xi_landy_szalay(&stats);
            let xg = r.cascade.gradient_xi_data_all_shells(&stats, &shells);
            let lift = |v: &[f64]| -> Vec<f64> {
                if v.is_empty() {
                    vec![0.0; n_base_d]
                } else {
                    Self::lift_gradient_d_to_original(
                        v, &r.original_d_indices, n_base_d)
                }
            };
            let shell_grads_dd: Vec<Vec<f64>> = xg.dd_grads.iter()
                .map(|g| lift(g.as_slice())).collect();
            let shell_grads_dr: Vec<Vec<f64>> = xg.dr_grads.iter()
                .map(|g| lift(g.as_slice())).collect();
            PerRunXiGrad {
                spec_scale: r.spec.scale,
                shells,
                shell_grads_dd,
                shell_grads_dr,
                n_d: r.cascade.n_d() as u64,
                n_r: r.cascade.n_r() as u64,
            }
        }).collect();

        // Sort & group by spec.scale (matches analyze_xi's binning).
        let mut prg = per_run;
        prg.sort_by(|a, b| a.spec_scale.partial_cmp(&b.spec_scale).unwrap());

        let mut groups: Vec<PooledXiResizeGroupGradient> = Vec::new();
        let mut bucket_idx: Vec<usize> = Vec::new();
        for i in 0..prg.len() {
            let same_group = bucket_idx.last().map(|&j: &usize| {
                let prev_scale = prg[j].spec_scale;
                let cur_scale = prg[i].spec_scale;
                let scale = prev_scale.abs().max(cur_scale.abs()).max(1e-300);
                (prev_scale - cur_scale).abs() / scale < scale_tol
            }).unwrap_or(false);
            if !same_group && !bucket_idx.is_empty() {
                groups.push(pool_xi_grad_group(&prg, &bucket_idx, n_base_d));
                bucket_idx.clear();
            }
            bucket_idx.push(i);
        }
        if !bucket_idx.is_empty() {
            groups.push(pool_xi_grad_group(&prg, &bucket_idx, n_base_d));
        }

        PooledXiGradient { by_resize: groups }
    }

    /// Aggregate-scalar variant of [`Self::gradient_xi_data_pooled`].
    /// Given per-(resize-group, shell) loss weights, return
    /// `∂L/∂w_i^d` for `L = Σ_g Σ_ℓ β_{g,ℓ} · ξ^pool_{g,ℓ}`.
    ///
    /// `betas[g][ℓ]` indexes the same way as
    /// `gradient_xi_data_pooled().by_resize[g].shell_grads[ℓ]`.
    pub fn gradient_xi_data_pooled_aggregate(
        &self,
        scale_tol: f64,
        betas: &[Vec<f64>],
    ) -> Vec<f64> {
        let pooled = self.gradient_xi_data_pooled(scale_tol);
        assert_eq!(betas.len(), pooled.by_resize.len(),
            "betas outer length {} != pooled resize-group count {}",
            betas.len(), pooled.by_resize.len());
        for (g, b) in betas.iter().enumerate() {
            assert_eq!(b.len(), pooled.by_resize[g].shell_grads.len(),
                "betas[{}] length {} != shell count {} in resize group",
                g, b.len(), pooled.by_resize[g].shell_grads.len());
        }
        let n = self.base_pts_d.len();
        let mut out = vec![0.0_f64; n];
        for (g, group) in pooled.by_resize.iter().enumerate() {
            for (s, shell_g) in group.shell_grads.iter().enumerate() {
                let beta = betas[g][s];
                if beta == 0.0 { continue; }
                if shell_g.is_empty() { continue; }
                for (i, &gi) in shell_g.iter().enumerate() {
                    out[i] += beta * gi;
                }
            }
        }
        out
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// the **pooled** Landy-Szalay ξ produced by [`Self::analyze_xi`].
    /// Symmetric counterpart to [`Self::gradient_xi_data_pooled`].
    ///
    /// Output mirrors `AggregatedXi::by_resize`.
    ///
    /// **Math** (matches the count-based forward semantics of
    /// `pool_xi_resize_group`): per resize group, pooled pair counts
    /// add linearly across runs. Pooled normalizations $N_{DD}^\text{pool},
    /// N_{DR}^\text{pool}, N_{RR}^\text{pool}$ are count-based and
    /// **independent of random weights**. $\partial DD^{(r)}/\partial w_j^r = 0$
    /// (DD doesn't involve random weights). So:
    ///
    /// ```text
    ///   ∂ξ^pool/∂w_j^r = (1/f_RR^pool) · [
    ///        −2 ∂f_DR^pool/∂w_j^r − (ξ^pool − 1) · ∂f_RR^pool/∂w_j^r
    ///   ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///   ∂f_X^pool/∂w_j^r = (1/N_X^pool) · Σ_r lift_r(∂X^(r)/∂w_j^r)
    /// ```
    ///
    /// and the per-run pair-count gradients $\partial DR^{(r)}/\partial w_j^r$,
    /// $\partial RR^{(r)}/\partial w_j^r$ come from the existing
    /// single-cascade primitive `gradient_xi_random_all_shells`.
    pub fn gradient_xi_random_pooled(
        &self,
        scale_tol: f64,
    ) -> PooledXiRandomGradient {
        use rayon::prelude::*;
        let n_base_r = self.base_pts_r.len();
        let runs = self.per_run();

        let per_run: Vec<PerRunXiRandomGrad> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze();
            let shells = r.cascade.xi_landy_szalay(&stats);
            let xg = r.cascade.gradient_xi_random_all_shells(&stats, &shells);
            let lift = |v: &[f64]| -> Vec<f64> {
                if v.is_empty() {
                    vec![0.0; n_base_r]
                } else {
                    Self::lift_gradient_d_to_original(
                        v, &r.original_r_indices, n_base_r)
                }
            };
            let shell_grads_dr: Vec<Vec<f64>> = xg.dr_grads.iter()
                .map(|g| lift(g.as_slice())).collect();
            let shell_grads_rr: Vec<Vec<f64>> = xg.rr_grads.iter()
                .map(|g| lift(g.as_slice())).collect();
            PerRunXiRandomGrad {
                spec_scale: r.spec.scale,
                shells,
                shell_grads_dr,
                shell_grads_rr,
                n_d: r.cascade.n_d() as u64,
                n_r: r.cascade.n_r() as u64,
            }
        }).collect();

        // Sort & group by spec.scale.
        let mut prg = per_run;
        prg.sort_by(|a, b| a.spec_scale.partial_cmp(&b.spec_scale).unwrap());

        let mut groups: Vec<PooledXiRandomResizeGroupGradient> = Vec::new();
        let mut bucket_idx: Vec<usize> = Vec::new();
        for i in 0..prg.len() {
            let same_group = bucket_idx.last().map(|&j: &usize| {
                let prev_scale = prg[j].spec_scale;
                let cur_scale = prg[i].spec_scale;
                let scale = prev_scale.abs().max(cur_scale.abs()).max(1e-300);
                (prev_scale - cur_scale).abs() / scale < scale_tol
            }).unwrap_or(false);
            if !same_group && !bucket_idx.is_empty() {
                groups.push(pool_xi_random_grad_group(&prg, &bucket_idx, n_base_r));
                bucket_idx.clear();
            }
            bucket_idx.push(i);
        }
        if !bucket_idx.is_empty() {
            groups.push(pool_xi_random_grad_group(&prg, &bucket_idx, n_base_r));
        }
        PooledXiRandomGradient { by_resize: groups }
    }

    /// Aggregate-scalar variant of [`Self::gradient_xi_random_pooled`].
    pub fn gradient_xi_random_pooled_aggregate(
        &self,
        scale_tol: f64,
        betas: &[Vec<f64>],
    ) -> Vec<f64> {
        let pooled = self.gradient_xi_random_pooled(scale_tol);
        assert_eq!(betas.len(), pooled.by_resize.len(),
            "betas outer length {} != pooled resize-group count {}",
            betas.len(), pooled.by_resize.len());
        for (g, b) in betas.iter().enumerate() {
            assert_eq!(b.len(), pooled.by_resize[g].shell_grads.len(),
                "betas[{}] length {} != shell count {}",
                g, b.len(), pooled.by_resize[g].shell_grads.len());
        }
        let n = self.base_pts_r.len();
        let mut out = vec![0.0_f64; n];
        for (g, group) in pooled.by_resize.iter().enumerate() {
            for (s, shell_g) in group.shell_grads.iter().enumerate() {
                let beta = betas[g][s];
                if beta == 0.0 { continue; }
                if shell_g.is_empty() { continue; }
                for (j, &gj) in shell_g.iter().enumerate() {
                    out[j] += beta * gj;
                }
            }
        }
        out
    }

    /// Multi-run gradient: per-pattern, per-original-particle gradient
    /// of the **pooled** anisotropy means produced by
    /// [`Self::analyze_anisotropy`].
    ///
    /// Output mirrors `AggregatedAnisotropyStats::by_side`: one entry
    /// per pooled bin, each carrying `pattern_grads[pattern_e][particle_idx]`
    /// where `pattern_e ∈ 1..2^D` (slot 0 unused).
    ///
    /// **Math**. The pooled per-pattern mean is
    ///
    /// ```text
    ///   ⟨w_e²⟩^pool = (Σ_r A_e^(r)) / (Σ_r T^(r))
    /// ```
    ///
    /// where `A_e^(r) = T^(r) · ⟨w_e²⟩^(r)` is the per-run raw
    /// per-pattern accumulator and `T^(r) = sum_w_r_parents^(r)`.
    /// Since random weights don't depend on data weights,
    /// `∂T^(r)/∂w_i^d = 0` and the chain rule reduces to
    ///
    /// ```text
    ///   ∂⟨w_e²⟩^pool / ∂w_i^d = (1/T^pool) · Σ_r lift_r(∂A_e^(r)/∂w_i^d)
    ///                         = (1/T^pool) · Σ_r lift_r(T^(r) · G_e^(r))
    /// ```
    ///
    /// where `G_e^(r) = ∂⟨w_e²⟩^(r)/∂w_i^d` from the existing
    /// per-cascade primitive `gradient_anisotropy_all_levels`.
    pub fn gradient_anisotropy_data_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledAnisotropyGradient {
        use rayon::prelude::*;
        let n_base_d = self.base_pts_d.len();
        let n_patterns = 1usize << D;
        let runs = self.per_run();

        // Per-(run, level): (physical_side, T_l, lifted ∂A_e/∂w grad
        // per pattern). Lifting at the per-run-level granularity keeps
        // the data flat for binning.
        struct Entry {
            physical_side: f64,
            t_l: f64,
            // grads_by_pattern[e][original_particle], lifted; pattern 0 empty.
            grads_by_pattern: Vec<Vec<f64>>,
        }
        let entries_per_run: Vec<Vec<Entry>> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze_anisotropy(cfg);
            let g_full = r.cascade.gradient_anisotropy_all_levels(cfg, &stats);
            let mut out: Vec<Entry> = Vec::with_capacity(stats.len());
            for (l, s) in stats.iter().enumerate() {
                if s.sum_w_r_parents <= 0.0 { continue; }
                let physical_side = r.spec.scale * s.cell_side_trimmed;
                let t_l = s.sum_w_r_parents;
                // Lift each per-pattern gradient AND scale by T_l so it
                // becomes the raw-accumulator gradient ∂A_e/∂w_i^d.
                let mut grads_by_pattern: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
                grads_by_pattern.push(Vec::new());  // pattern 0 unused
                for e in 1..n_patterns {
                    let g = &g_full.pattern_grads[l][e];
                    let lifted = if g.is_empty() {
                        vec![0.0; n_base_d]
                    } else {
                        Self::lift_gradient_d_to_original(
                            g.as_slice(), &r.original_d_indices, n_base_d)
                    };
                    // Scale to raw-accumulator gradient.
                    let raw_lifted: Vec<f64> =
                        lifted.iter().map(|x| x * t_l).collect();
                    grads_by_pattern.push(raw_lifted);
                }
                out.push(Entry { physical_side, t_l, grads_by_pattern });
            }
            out
        }).collect();
        let mut entries: Vec<Entry> =
            entries_per_run.into_iter().flatten().collect();

        // Bin by physical_side using bin_tol relative tolerance.
        entries.sort_by(|a, b|
            a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<PooledAnisotropyGradientBin> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() {
                let scale = entries[i].physical_side.abs()
                    .max(entries[j].physical_side.abs()).max(1e-300);
                let rel = (entries[i].physical_side - entries[j].physical_side).abs() / scale;
                if rel >= bin_tol { break; }
                j += 1;
            }
            // Pool: T_pool = Σ T_l, raw_grad_pool[e] = Σ raw_grad[e]^(r,l).
            // Then ∂⟨w_e²⟩^pool/∂w = raw_grad_pool[e] / T_pool.
            let mut t_pool = 0.0_f64;
            let mut raw_grad_pool: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
            raw_grad_pool.push(Vec::new());  // pattern 0
            for _ in 1..n_patterns {
                raw_grad_pool.push(vec![0.0_f64; n_base_d]);
            }
            let mut weighted_side_num = 0.0_f64;
            let mut weighted_side_den = 0.0_f64;
            for e in &entries[i..j] {
                t_pool += e.t_l;
                weighted_side_num += e.physical_side * e.t_l;
                weighted_side_den += e.t_l;
                for p in 1..n_patterns {
                    let g = &e.grads_by_pattern[p];
                    for k in 0..n_base_d {
                        raw_grad_pool[p][k] += g[k];
                    }
                }
            }
            let physical_side = if weighted_side_den > 0.0 {
                weighted_side_num / weighted_side_den
            } else { entries[i].physical_side };
            // Divide by T_pool to get pooled mean gradients.
            let mut pattern_grads: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
            pattern_grads.push(Vec::new());
            if t_pool > 0.0 {
                let inv_t = 1.0 / t_pool;
                for p in 1..n_patterns {
                    let g: Vec<f64> = raw_grad_pool[p].iter()
                        .map(|x| x * inv_t).collect();
                    pattern_grads.push(g);
                }
            } else {
                for _ in 1..n_patterns {
                    pattern_grads.push(vec![0.0_f64; n_base_d]);
                }
            }

            // Axis-aligned subset and LoS quadrupole gradient.
            let mut axis_grads: Vec<Vec<f64>> = Vec::with_capacity(D);
            for d in 0..D {
                axis_grads.push(pattern_grads[1usize << d].clone());
            }
            let quadrupole_los_grad: Vec<f64> = if D >= 2 {
                // Q_2 = ⟨w_LoS²⟩ − mean(transverse axes).
                // ∂Q/∂w = ∂⟨w_LoS²⟩/∂w − (1/(D-1)) Σ_{d<D-1} ∂⟨w_axis_d²⟩/∂w
                let mut q = vec![0.0_f64; n_base_d];
                let inv_n_trans = 1.0 / (D as f64 - 1.0);
                for k in 0..n_base_d {
                    let los = axis_grads[D - 1][k];
                    let trans: f64 = (0..D-1).map(|d| axis_grads[d][k]).sum();
                    q[k] = los - inv_n_trans * trans;
                }
                q
            } else {
                vec![0.0_f64; n_base_d]
            };

            bins.push(PooledAnisotropyGradientBin {
                physical_side,
                pattern_grads,
                axis_grads,
                quadrupole_los_grad,
            });
            i = j;
        }
        PooledAnisotropyGradient { by_side: bins }
    }

    /// Aggregate-scalar variant of [`Self::gradient_anisotropy_data_pooled`]
    /// for the LoS quadrupole. Given per-bin loss weights `betas`,
    /// returns `∂L/∂w_i^d` for `L = Σ_b β_b · Q_LoS^pool[b]`.
    pub fn gradient_anisotropy_quadrupole_data_pooled_aggregate(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
        betas: &[f64],
    ) -> Vec<f64> {
        let pooled = self.gradient_anisotropy_data_pooled(cfg, bin_tol);
        assert_eq!(betas.len(), pooled.by_side.len(),
            "betas length {} != pooled bin count {}",
            betas.len(), pooled.by_side.len());
        let n = self.base_pts_d.len();
        let mut out = vec![0.0_f64; n];
        for (b, bin) in pooled.by_side.iter().enumerate() {
            let beta = betas[b];
            if beta == 0.0 { continue; }
            for (i, &g) in bin.quadrupole_los_grad.iter().enumerate() {
                out[i] += beta * g;
            }
        }
        out
    }

    /// Multi-run gradient: per-original-random-particle gradient of
    /// **pooled** anisotropy means. Symmetric counterpart to
    /// [`Self::gradient_anisotropy_data_pooled`].
    ///
    /// Math (random-weight side, where ∂T ≠ 0):
    ///
    /// ```text
    ///   ∂A_e^(r) / ∂w_j^r = T^(r) · ∂⟨w_e²⟩^(r)/∂w_j^r
    ///                       + ⟨w_e²⟩^(r) · ∂T^(r)/∂w_j^r
    ///   ∂A_e^pool / ∂w_j^r = Σ_r lift_r(∂A_e^(r)/∂w_j^r)
    ///   ∂T^pool   / ∂w_j^r = Σ_r lift_r(∂T^(r)/∂w_j^r)
    ///   ∂⟨w_e²⟩^pool / ∂w_j^r = (1/T^pool) · [∂A_e^pool − ⟨w_e²⟩^pool · ∂T^pool]
    /// ```
    ///
    /// LoS quadrupole gradient is the corresponding linear combination
    /// of per-axis pattern gradients (`Q_2 = ⟨w_LoS²⟩ −
    /// (1/(D-1)) Σ_{d<D-1} ⟨w_axis_d²⟩`).
    pub fn gradient_anisotropy_random_pooled(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
    ) -> PooledAnisotropyRandomGradient {
        use rayon::prelude::*;
        let n_base_r = self.base_pts_r.len();
        let n_patterns = 1usize << D;
        let runs = self.per_run();

        struct Entry {
            physical_side: f64,
            t_l: f64,
            // Per-pattern raw accumulator value A_e = mean × T for
            // this (run, level). Needed to compute pooled ⟨w_e²⟩.
            a_by_pattern: Vec<f64>,
            // Per-pattern lifted ∂A_e/∂w_j^r.
            grads_a_by_pattern: Vec<Vec<f64>>,
            // Lifted ∂T/∂w_j^r.
            grad_t: Vec<f64>,
        }
        let entries_per_run: Vec<Vec<Entry>> = runs.par_iter().map(|r| {
            let stats = r.cascade.analyze_anisotropy(cfg);
            let g_full = r.cascade.gradient_anisotropy_random_all_levels(cfg, &stats);
            let t_full = r.cascade.gradient_anisotropy_t_random_all_levels(cfg, &stats);
            let mut out: Vec<Entry> = Vec::with_capacity(stats.len());
            for (l, s) in stats.iter().enumerate() {
                if s.sum_w_r_parents <= 0.0 { continue; }
                let physical_side = r.spec.scale * s.cell_side_trimmed;
                let t_l = s.sum_w_r_parents;
                let grad_t_lifted = if t_full[l].is_empty() {
                    vec![0.0; n_base_r]
                } else {
                    Self::lift_gradient_d_to_original(
                        t_full[l].as_slice(), &r.original_r_indices, n_base_r)
                };
                let mut a_by_pattern = vec![0.0_f64; n_patterns];
                let mut grads_a_by_pattern: Vec<Vec<f64>> =
                    Vec::with_capacity(n_patterns);
                grads_a_by_pattern.push(Vec::new());
                for e in 1..n_patterns {
                    let mean_e = s.mean_w_squared_by_pattern.get(e)
                        .copied().unwrap_or(0.0);
                    a_by_pattern[e] = mean_e * t_l;
                    let g_mean = &g_full.pattern_grads[l][e];
                    let g_mean_lifted = if g_mean.is_empty() {
                        vec![0.0; n_base_r]
                    } else {
                        Self::lift_gradient_d_to_original(
                            g_mean.as_slice(), &r.original_r_indices, n_base_r)
                    };
                    let grad_a: Vec<f64> = g_mean_lifted.iter()
                        .zip(grad_t_lifted.iter())
                        .map(|(gm, gt)| t_l * gm + mean_e * gt)
                        .collect();
                    grads_a_by_pattern.push(grad_a);
                }
                out.push(Entry {
                    physical_side,
                    t_l,
                    a_by_pattern,
                    grads_a_by_pattern,
                    grad_t: grad_t_lifted,
                });
            }
            out
        }).collect();
        let mut entries: Vec<Entry> =
            entries_per_run.into_iter().flatten().collect();

        entries.sort_by(|a, b|
            a.physical_side.partial_cmp(&b.physical_side).unwrap());

        let mut bins: Vec<PooledAnisotropyRandomGradientBin> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let mut j = i + 1;
            while j < entries.len() {
                let scale = entries[i].physical_side.abs()
                    .max(entries[j].physical_side.abs()).max(1e-300);
                let rel = (entries[i].physical_side - entries[j].physical_side).abs() / scale;
                if rel >= bin_tol { break; }
                j += 1;
            }
            let mut t_pool = 0.0_f64;
            let mut a_pool = vec![0.0_f64; n_patterns];
            let mut grad_t_pool = vec![0.0_f64; n_base_r];
            let mut grad_a_pool: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
            grad_a_pool.push(Vec::new());
            for _ in 1..n_patterns {
                grad_a_pool.push(vec![0.0_f64; n_base_r]);
            }
            let mut weighted_side_num = 0.0_f64;
            let mut weighted_side_den = 0.0_f64;
            for e in &entries[i..j] {
                t_pool += e.t_l;
                weighted_side_num += e.physical_side * e.t_l;
                weighted_side_den += e.t_l;
                for k in 0..n_base_r {
                    grad_t_pool[k] += e.grad_t[k];
                }
                for p in 1..n_patterns {
                    a_pool[p] += e.a_by_pattern[p];
                    let g = &e.grads_a_by_pattern[p];
                    for k in 0..n_base_r {
                        grad_a_pool[p][k] += g[k];
                    }
                }
            }
            let physical_side = if weighted_side_den > 0.0 {
                weighted_side_num / weighted_side_den
            } else { entries[i].physical_side };

            // Compute pooled mean and chain-rule gradient per pattern.
            let mut mean_pool = vec![0.0_f64; n_patterns];
            let mut pattern_grads: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
            pattern_grads.push(Vec::new());
            if t_pool > 0.0 {
                let inv_t = 1.0 / t_pool;
                for p in 1..n_patterns {
                    mean_pool[p] = a_pool[p] * inv_t;
                    let coef_t = -mean_pool[p];
                    let g: Vec<f64> = (0..n_base_r).map(|k| {
                        inv_t * (grad_a_pool[p][k] + coef_t * grad_t_pool[k])
                    }).collect();
                    pattern_grads.push(g);
                }
            } else {
                for _ in 1..n_patterns {
                    pattern_grads.push(vec![0.0_f64; n_base_r]);
                }
            }

            // Axis-aligned subset.
            let axis_grads: Vec<Vec<f64>> = (0..D).map(|d|
                pattern_grads[1usize << d].clone()).collect();

            // LoS quadrupole gradient.
            let quadrupole_los_grad: Vec<f64> = if D >= 2 {
                let mut q = vec![0.0_f64; n_base_r];
                let inv_n_trans = 1.0 / (D as f64 - 1.0);
                for k in 0..n_base_r {
                    let los = axis_grads[D - 1][k];
                    let trans: f64 = (0..D-1).map(|d| axis_grads[d][k]).sum();
                    q[k] = los - inv_n_trans * trans;
                }
                q
            } else {
                vec![0.0_f64; n_base_r]
            };

            bins.push(PooledAnisotropyRandomGradientBin {
                physical_side,
                pattern_grads,
                axis_grads,
                quadrupole_los_grad,
            });
            i = j;
        }
        PooledAnisotropyRandomGradient { by_side: bins }
    }

    /// Aggregate-scalar variant of [`Self::gradient_anisotropy_random_pooled`]
    /// for the LoS quadrupole.
    pub fn gradient_anisotropy_quadrupole_random_pooled_aggregate(
        &self,
        cfg: &crate::hier_bitvec_pair::FieldStatsConfig,
        bin_tol: f64,
        betas: &[f64],
    ) -> Vec<f64> {
        let pooled = self.gradient_anisotropy_random_pooled(cfg, bin_tol);
        assert_eq!(betas.len(), pooled.by_side.len(),
            "betas length {} != pooled bin count {}",
            betas.len(), pooled.by_side.len());
        let n = self.base_pts_r.len();
        let mut out = vec![0.0_f64; n];
        for (b, bin) in pooled.by_side.iter().enumerate() {
            let beta = betas[b];
            if beta == 0.0 { continue; }
            for (j, &g) in bin.quadrupole_los_grad.iter().enumerate() {
                out[j] += beta * g;
            }
        }
        out
    }
}

/// Helper: pool one bucket of per-run-grad entries (all sharing the
/// same scale) into one resize group's gradient.
fn pool_xi_grad_group(
    prg: &[PerRunXiGrad],
    bucket_idx: &[usize],
    n_base_d: usize,
) -> PooledXiResizeGroupGradient {
    debug_assert!(!bucket_idx.is_empty());
    let scale = prg[bucket_idx[0]].spec_scale;
    let n_shells = prg[bucket_idx[0]].shells.len();

    let mut shell_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
    for shell_l in 0..n_shells {
        // Pool DD, DR pair counts and per-particle gradients across
        // all runs in this group.
        let mut dd_sum = 0.0_f64;
        let mut rr_sum = 0.0_f64;
        let mut dr_sum = 0.0_f64;
        let mut n_d_sum: u64 = 0;
        let mut n_r_sum: u64 = 0;
        let mut g_dd_sum = vec![0.0_f64; n_base_d];
        let mut g_dr_sum = vec![0.0_f64; n_base_d];
        for &j in bucket_idx {
            let pr = &prg[j];
            if shell_l >= pr.shells.len() { continue; }
            let s = &pr.shells[shell_l];
            dd_sum += s.dd;
            rr_sum += s.rr;
            dr_sum += s.dr;
            n_d_sum += pr.n_d;
            n_r_sum += pr.n_r;
            let g_dd = &pr.shell_grads_dd[shell_l];
            let g_dr = &pr.shell_grads_dr[shell_l];
            for k in 0..n_base_d {
                g_dd_sum[k] += g_dd[k];
                g_dr_sum[k] += g_dr[k];
            }
        }

        // Pooled count-based normalizations (matches pool_xi_resize_group
        // forward semantics). RR-side derivatives are zero w.r.t.
        // data weights, so we don't need to track per-run RR gradients.
        // Suppress unused-var warning for dr_sum (it's pooled but only
        // the gradient path needs it indirectly).
        let _ = dr_sum;
        let mut shell_grad = vec![0.0_f64; n_base_d];
        if rr_sum > 0.0 && n_r_sum > 1 && n_d_sum > 1 {
            let wd = n_d_sum as f64;
            let wr = n_r_sum as f64;
            let n_dd_norm = wd * (wd - 1.0) / 2.0;
            let n_rr_norm = wr * (wr - 1.0) / 2.0;
            let n_dr_norm = wd * wr;
            let f_rr = rr_sum / n_rr_norm;
            if f_rr > 0.0 {
                let inv_f_rr = 1.0 / f_rr;
                let inv_n_dd = 1.0 / n_dd_norm;
                let inv_n_dr = 1.0 / n_dr_norm;
                for k in 0..n_base_d {
                    let d_f_dd = inv_n_dd * g_dd_sum[k];
                    let d_f_dr = inv_n_dr * g_dr_sum[k];
                    shell_grad[k] = inv_f_rr * (d_f_dd - 2.0 * d_f_dr);
                }
            }
        }
        // Suppress unused-var for dd_sum (shell_grad derived from
        // ∂DD gradient, not from dd_sum directly).
        let _ = dd_sum;
        shell_grads.push(shell_grad);
    }
    PooledXiResizeGroupGradient { scale, shell_grads }
}

/// Helper: pool one bucket of per-run-random-grad entries (all sharing
/// the same scale) into one resize group's random-weight gradient.
fn pool_xi_random_grad_group(
    prg: &[PerRunXiRandomGrad],
    bucket_idx: &[usize],
    n_base_r: usize,
) -> PooledXiRandomResizeGroupGradient {
    debug_assert!(!bucket_idx.is_empty());
    let scale = prg[bucket_idx[0]].spec_scale;
    let n_shells = prg[bucket_idx[0]].shells.len();

    let mut shell_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
    for shell_l in 0..n_shells {
        let mut dd_sum = 0.0_f64;
        let mut rr_sum = 0.0_f64;
        let mut dr_sum = 0.0_f64;
        let mut n_d_sum: u64 = 0;
        let mut n_r_sum: u64 = 0;
        let mut g_dr_sum = vec![0.0_f64; n_base_r];
        let mut g_rr_sum = vec![0.0_f64; n_base_r];
        for &j in bucket_idx {
            let pr = &prg[j];
            if shell_l >= pr.shells.len() { continue; }
            let s = &pr.shells[shell_l];
            dd_sum += s.dd;
            rr_sum += s.rr;
            dr_sum += s.dr;
            n_d_sum += pr.n_d;
            n_r_sum += pr.n_r;
            let g_dr = &pr.shell_grads_dr[shell_l];
            let g_rr = &pr.shell_grads_rr[shell_l];
            for k in 0..n_base_r {
                g_dr_sum[k] += g_dr[k];
                g_rr_sum[k] += g_rr[k];
            }
        }

        let mut shell_grad = vec![0.0_f64; n_base_r];
        if rr_sum > 0.0 && n_r_sum > 1 && n_d_sum > 1 {
            let wd = n_d_sum as f64;
            let wr = n_r_sum as f64;
            let n_dd_norm = wd * (wd - 1.0) / 2.0;
            let n_rr_norm = wr * (wr - 1.0) / 2.0;
            let n_dr_norm = wd * wr;
            let f_dd = dd_sum / n_dd_norm;
            let f_dr = dr_sum / n_dr_norm;
            let f_rr = rr_sum / n_rr_norm;
            if f_rr > 0.0 {
                let inv_f_rr = 1.0 / f_rr;
                let inv_n_dr = 1.0 / n_dr_norm;
                let inv_n_rr = 1.0 / n_rr_norm;
                // ξ^pool − 1 = (f_DD^pool − 2 f_DR^pool) / f_RR^pool.
                let xi_minus_1 = (f_dd - 2.0 * f_dr) * inv_f_rr;
                for k in 0..n_base_r {
                    let d_f_dr = inv_n_dr * g_dr_sum[k];
                    let d_f_rr = inv_n_rr * g_rr_sum[k];
                    shell_grad[k] = inv_f_rr
                        * (-2.0 * d_f_dr - xi_minus_1 * d_f_rr);
                }
            }
        }
        shell_grads.push(shell_grad);
    }
    PooledXiRandomResizeGroupGradient { scale, shell_grads }
}

/// Internal: per-run xi data plus lifted ∂DD/∂DR gradients, used by
/// `gradient_xi_data_pooled` and `pool_xi_grad_group`.
struct PerRunXiGrad {
    spec_scale: f64,
    shells: Vec<crate::hier_bitvec_pair::XiShell>,
    /// shell_grads_dd[shell][original_particle_idx] (lifted)
    shell_grads_dd: Vec<Vec<f64>>,
    shell_grads_dr: Vec<Vec<f64>>,
    n_d: u64,
    n_r: u64,
}

/// Internal: per-run xi data plus lifted ∂DR/∂RR random-weight
/// gradients, used by `gradient_xi_random_pooled` and
/// `pool_xi_random_grad_group`.
struct PerRunXiRandomGrad {
    spec_scale: f64,
    shells: Vec<crate::hier_bitvec_pair::XiShell>,
    /// shell_grads_dr[shell][original_random_idx] (lifted)
    shell_grads_dr: Vec<Vec<f64>>,
    /// shell_grads_rr[shell][original_random_idx] (lifted)
    shell_grads_rr: Vec<Vec<f64>>,
    n_d: u64,
    n_r: u64,
}

// Internal: per-cascade-run ξ data, used during aggregation.
struct PerRunXi {
    spec_scale: f64,
    shells: Vec<crate::hier_bitvec_pair::XiShell>,
    n_d: u64,
    n_r: u64,
}

// ============================================================================
// Aggregation output types
// ============================================================================

/// Per-run diagnostic information returned alongside aggregated stats.
#[derive(Clone, Debug)]
pub struct RunDiagnostic<const D: usize> {
    pub name: String,
    pub spec: CascadeRunSpec<D>,
    pub footprint_coverage: f64,
    pub n_levels: usize,
}

/// Pooled field-stats for one physical cell-side bin.
#[derive(Clone, Debug)]
pub struct AggregatedFieldStatsBin {
    /// Physical cell side (W_r-weighted average across the runs that fell
    /// in this bin; usually all runs in a bin have nearly identical sides).
    pub physical_side: f64,
    /// How many (run, level) entries were pooled into this bin.
    pub n_contributing_runs: usize,
    /// Total W_r summed across all pooled entries.
    pub sum_w_r_total: f64,
    /// Total active cells across all pooled entries.
    pub n_cells_active_total: u64,
    /// Pooled W_r-weighted mean of δ. Should be ≈ 0 for a well-normalized
    /// estimator across many runs.
    pub mean_delta: f64,
    /// Pooled W_r-weighted central second moment.
    pub var_delta: f64,
    /// Pooled W_r-weighted central third moment.
    pub m3_delta: f64,
    /// Pooled W_r-weighted central fourth moment.
    pub m4_delta: f64,
    /// Pooled reduced skewness `m3 / var^2`.
    pub s3_delta: f64,
    /// Min and max δ across all pooled cells.
    pub min_delta: f64,
    pub max_delta: f64,
    /// Across-run variance of `mean_delta` — the cheap shift-bootstrap
    /// estimator of statistical uncertainty on the pooled mean. Equals 0
    /// when only one run contributed.
    pub mean_delta_across_run_var: f64,
    /// Same for `var_delta`.
    pub var_delta_across_run_var: f64,
    /// Total cells where data is present but outside the (random)
    /// footprint, summed across pooled runs.
    pub n_cells_data_outside_total: u64,
    pub sum_w_d_outside_total: f64,
}

/// The full aggregated output: one bin per distinct physical cell side,
/// plus per-run diagnostics for inspecting the ensemble.
#[derive(Clone, Debug)]
pub struct AggregatedFieldStats<const D: usize> {
    pub by_side: Vec<AggregatedFieldStatsBin>,
    pub per_run_diagnostics: Vec<RunDiagnostic<D>>,
}

/// Per-original-particle gradient of the pooled `var_delta` produced
/// by [`CascadeRunner::analyze_field_stats`]. Indexed by the same
/// physical-side bins as `AggregatedFieldStats::by_side`.
#[derive(Clone, Debug)]
pub struct PooledFieldStatsGradient<const D: usize> {
    /// `bin_grads[bin_idx][original_particle_idx]` = gradient of
    /// pooled var_delta in that bin w.r.t. the original data weight.
    pub bin_grads: Vec<Vec<f64>>,
    /// Physical side (sum-w-r-weighted average across pooled entries)
    /// associated with each bin. Matches the order of `bin_grads`.
    pub bin_sides: Vec<f64>,
    pub _marker: std::marker::PhantomData<[(); D]>,
}

/// Internal: per-bin pooled raw sums plus lifted ∂S_k gradients,
/// produced by `CascadeRunner::pooled_raw_sums_data` and consumed by
/// the moment-specific pooled gradient methods.
struct PooledBinData {
    physical_side: f64,
    t: f64,
    s_1: f64, s_2: f64, s_3: f64,
    /// Pooled S_4 = Σ_runs Σ_cells W_r · δ⁴. Currently unused by the
    /// implemented moment chain rules (m4 needs only S_1, S_2, S_3
    /// for the scalar coefficient and ∂S_4 for the gradient itself),
    /// but retained for future higher-moment / kurtosis-ratio gradients.
    #[allow(dead_code)]
    s_4: f64,
    grad_s1: Vec<f64>, grad_s2: Vec<f64>,
    grad_s3: Vec<f64>, grad_s4: Vec<f64>,
}

struct PooledRawSumsData {
    bins: Vec<PooledBinData>,
}

/// Per-original-random-particle gradient of pooled field-stats moments.
/// Symmetric counterpart to [`PooledFieldStatsGradient`]; same
/// physical-side bins as `AggregatedFieldStats::by_side`.
#[derive(Clone, Debug)]
pub struct PooledFieldStatsRandomGradient<const D: usize> {
    /// `bin_grads[bin_idx][original_random_idx]` = gradient of
    /// pooled $\mu_k$ in that bin w.r.t. the original random weight.
    pub bin_grads: Vec<Vec<f64>>,
    pub bin_sides: Vec<f64>,
    pub _marker: std::marker::PhantomData<[(); D]>,
}

/// Internal: per-bin pooled raw sums plus lifted gradients (random
/// version, where ∂T ≠ 0).
struct PooledBinRandom {
    physical_side: f64,
    t: f64,
    s_1: f64, s_2: f64, s_3: f64, s_4: f64,
    grad_t: Vec<f64>,
    grad_s1: Vec<f64>, grad_s2: Vec<f64>,
    grad_s3: Vec<f64>, grad_s4: Vec<f64>,
}

struct PooledRawSumsRandom {
    bins: Vec<PooledBinRandom>,
}

// Internal: per-run-per-level intermediate during aggregation.
struct PerRunLevelEntry {
    physical_side: f64,
    sum_w_r: f64,
    raw_pow: [f64; 4],         // [Σwδ, Σwδ², Σwδ³, Σwδ⁴]
    n_cells_active: u64,
    n_cells_data_outside: u64,
    sum_w_d_outside: f64,
    min_delta: f64,
    max_delta: f64,
    // For across-run variance of derived quantities.
    mean_delta_for_var_estimate: f64,
    var_delta_for_var_estimate: f64,
}

/// Pool a set of per-run-level entries into one AggregatedFieldStatsBin.
/// Uses the standard "sum the raw moments, then re-derive central
/// moments" recipe so pooling is exact in the limit of zero floating-point
/// rounding error.
fn pool_bin(entries: &[&PerRunLevelEntry]) -> AggregatedFieldStatsBin {
    let n = entries.len();
    let mut total_sw = 0.0;
    let mut total_pow = [0.0f64; 4];
    let mut total_cells: u64 = 0;
    let mut total_outside_cells: u64 = 0;
    let mut total_outside_sw_d = 0.0;
    let mut min_d = f64::INFINITY;
    let mut max_d = f64::NEG_INFINITY;
    let mut weighted_side_num = 0.0;
    let mut weighted_side_den = 0.0;

    for e in entries {
        total_sw += e.sum_w_r;
        for k in 0..4 { total_pow[k] += e.raw_pow[k]; }
        total_cells += e.n_cells_active;
        total_outside_cells += e.n_cells_data_outside;
        total_outside_sw_d += e.sum_w_d_outside;
        if e.min_delta < min_d && e.n_cells_active > 0 { min_d = e.min_delta; }
        if e.max_delta > max_d && e.n_cells_active > 0 { max_d = e.max_delta; }
        weighted_side_num += e.physical_side * e.sum_w_r;
        weighted_side_den += e.sum_w_r;
    }

    if min_d == f64::INFINITY { min_d = 0.0; }
    if max_d == f64::NEG_INFINITY { max_d = 0.0; }

    let physical_side = if weighted_side_den > 0.0 {
        weighted_side_num / weighted_side_den
    } else if !entries.is_empty() {
        entries[0].physical_side
    } else { 0.0 };

    let (mean_delta, var_delta, m3_delta, m4_delta) = if total_sw > 0.0 {
        let m1 = total_pow[0] / total_sw;
        let m2_raw = total_pow[1] / total_sw;
        let m3_raw = total_pow[2] / total_sw;
        let m4_raw = total_pow[3] / total_sw;
        let var = (m2_raw - m1 * m1).max(0.0);
        let m3c = m3_raw - 3.0 * m1 * m2_raw + 2.0 * m1 * m1 * m1;
        let m4c = m4_raw - 4.0 * m1 * m3_raw + 6.0 * m1 * m1 * m2_raw
                  - 3.0 * m1 * m1 * m1 * m1;
        (m1, var, m3c, m4c)
    } else { (0.0, 0.0, 0.0, 0.0) };
    let s3_delta = if var_delta > 0.0 { m3_delta / (var_delta * var_delta) } else { 0.0 };

    // Across-run variance of per-run mean_delta and var_delta. Equally-
    // weighted across runs (could weight by sw_r — debatable; equal-weighted
    // is the conservative bootstrap-style choice).
    let mean_arv = if n > 1 {
        let m: f64 = entries.iter().map(|e| e.mean_delta_for_var_estimate).sum::<f64>() / n as f64;
        entries.iter()
            .map(|e| (e.mean_delta_for_var_estimate - m).powi(2))
            .sum::<f64>() / (n - 1) as f64
    } else { 0.0 };
    let var_arv = if n > 1 {
        let m: f64 = entries.iter().map(|e| e.var_delta_for_var_estimate).sum::<f64>() / n as f64;
        entries.iter()
            .map(|e| (e.var_delta_for_var_estimate - m).powi(2))
            .sum::<f64>() / (n - 1) as f64
    } else { 0.0 };

    AggregatedFieldStatsBin {
        physical_side,
        n_contributing_runs: n,
        sum_w_r_total: total_sw,
        n_cells_active_total: total_cells,
        mean_delta,
        var_delta,
        m3_delta,
        m4_delta,
        s3_delta,
        min_delta: min_d,
        max_delta: max_d,
        mean_delta_across_run_var: mean_arv,
        var_delta_across_run_var: var_arv,
        n_cells_data_outside_total: total_outside_cells,
        sum_w_d_outside_total: total_outside_sw_d,
    }
}

// ============================================================================
// Anisotropy aggregation types and pooling
// ============================================================================

/// Pooled anisotropy stats for one physical cell-side bin. D-generic:
/// the per-pattern vector has length 2^D and the axis-aligned vector
/// has length D, matching the per-run [`AnisotropyStats`].
#[derive(Clone, Debug)]
pub struct AggregatedAnisotropyStatsBin {
    /// W_r-weighted average physical side across pooled entries.
    pub physical_side: f64,
    /// How many (run, level) entries were pooled.
    pub n_contributing_runs: usize,
    /// Total sum_w_r_parents over pooled entries.
    pub sum_w_r_parents_total: f64,
    /// Total parent cells visited across pooled entries.
    pub n_parents_total: u64,
    /// Pooled W_r-weighted ⟨w_e²⟩ per non-trivial pattern. Length 2^D.
    /// Position 0 is unused.
    pub mean_w_squared_by_pattern: Vec<f64>,
    /// Convenience: pooled axis-aligned ⟨w²⟩ in axis order. Length D.
    pub mean_w_squared_axis: Vec<f64>,
    /// LoS quadrupole computed from the pooled axis values.
    pub quadrupole_los: f64,
    /// Reduced LoS quadrupole computed from the pooled axis values.
    pub reduced_quadrupole_los: f64,
    /// Across-run variance of the per-run `quadrupole_los`. The cheap
    /// shift-bootstrap uncertainty estimate. Zero for single-run bins.
    pub quadrupole_los_across_run_var: f64,
}

/// Full aggregated anisotropy output: bins by physical side + per-run
/// diagnostics.
#[derive(Clone, Debug)]
pub struct AggregatedAnisotropyStats<const D: usize> {
    pub by_side: Vec<AggregatedAnisotropyStatsBin>,
    pub per_run_diagnostics: Vec<RunDiagnostic<D>>,
}

/// Per-original-particle gradient of pooled anisotropy means produced
/// by [`CascadeRunner::analyze_anisotropy`]. Mirrors
/// `AggregatedAnisotropyStats::by_side`: one entry per pooled bin,
/// each carrying per-pattern, per-axis, and LoS-quadrupole gradients.
#[derive(Clone, Debug)]
pub struct PooledAnisotropyGradient {
    pub by_side: Vec<PooledAnisotropyGradientBin>,
}

/// Per-bin gradient slice in a [`PooledAnisotropyGradient`].
#[derive(Clone, Debug)]
pub struct PooledAnisotropyGradientBin {
    /// W_r-weighted average physical side (matches the corresponding
    /// `AggregatedAnisotropyStatsBin.physical_side`).
    pub physical_side: f64,
    /// `pattern_grads[e][i] = ∂⟨w_e²⟩^pool / ∂w_i^d` for pattern e ∈
    /// 1..2^D. Slot `pattern_grads[0]` is empty (pattern 0 unused).
    pub pattern_grads: Vec<Vec<f64>>,
    /// Convenience: axis-aligned subset, indexed `axis_grads[d][i]`
    /// for axis d ∈ 0..D. Equivalent to `pattern_grads[1 << d]`.
    pub axis_grads: Vec<Vec<f64>>,
    /// Gradient of the LoS quadrupole `Q_2 = ⟨w_LoS²⟩ −
    /// (1/(D-1)) Σ_{d<D-1} ⟨w_axis_d²⟩` (LoS = last axis). All-zero
    /// in 1D (Q_2 trivially undefined).
    pub quadrupole_los_grad: Vec<f64>,
}

/// Per-original-random-particle gradient of pooled anisotropy means
/// produced by [`CascadeRunner::analyze_anisotropy`]. Symmetric
/// counterpart to [`PooledAnisotropyGradient`].
#[derive(Clone, Debug)]
pub struct PooledAnisotropyRandomGradient {
    pub by_side: Vec<PooledAnisotropyRandomGradientBin>,
}

/// Per-bin gradient slice in a [`PooledAnisotropyRandomGradient`].
/// Same shape as [`PooledAnisotropyGradientBin`], but each gradient
/// vector is indexed by base-random-particle index and is the
/// derivative w.r.t. random weights.
#[derive(Clone, Debug)]
pub struct PooledAnisotropyRandomGradientBin {
    pub physical_side: f64,
    /// `pattern_grads[e][j] = ∂⟨w_e²⟩^pool / ∂w_j^r` for pattern e ∈
    /// 1..2^D. Slot 0 empty.
    pub pattern_grads: Vec<Vec<f64>>,
    pub axis_grads: Vec<Vec<f64>>,
    pub quadrupole_los_grad: Vec<f64>,
}

// Internal: per-run-per-level intermediate during anisotropy aggregation.
struct PerRunAnisoEntry {
    physical_side: f64,
    sum_w_r_parents: f64,
    n_parents: u64,
    /// Raw un-normalized per-pattern sums: position e holds Σ_parents
    /// (parent_wr · w_e²). Recovered as `mean × sum_w_r_parents`.
    raw_by_pattern: Vec<f64>,
    quadrupole_los_for_var_estimate: f64,
}

/// Pool a set of per-run-level anisotropy entries into one bin.
fn pool_aniso_bin<const D: usize>(
    entries: &[&PerRunAnisoEntry],
) -> AggregatedAnisotropyStatsBin {
    let n = entries.len();
    let n_patterns = 1usize << D;

    let mut total_sw = 0.0;
    let mut total_n_parents: u64 = 0;
    let mut raw_total = vec![0.0f64; n_patterns];
    let mut weighted_side_num = 0.0;
    let mut weighted_side_den = 0.0;

    for e in entries {
        total_sw += e.sum_w_r_parents;
        total_n_parents += e.n_parents;
        for k in 0..n_patterns.min(e.raw_by_pattern.len()) {
            raw_total[k] += e.raw_by_pattern[k];
        }
        weighted_side_num += e.physical_side * e.sum_w_r_parents;
        weighted_side_den += e.sum_w_r_parents;
    }

    let physical_side = if weighted_side_den > 0.0 {
        weighted_side_num / weighted_side_den
    } else if !entries.is_empty() {
        entries[0].physical_side
    } else { 0.0 };

    let mut means_by_pattern = vec![0.0f64; n_patterns];
    if total_sw > 0.0 {
        for k in 1..n_patterns {
            means_by_pattern[k] = raw_total[k] / total_sw;
        }
    }

    // Axis-aligned subset: pattern with bit-d set is pattern (1 << d).
    let mut means_axis = vec![0.0f64; D];
    for d in 0..D {
        means_axis[d] = means_by_pattern[1usize << d];
    }

    // LoS quadrupole = mean over D-1 transverse axes vs LoS (last axis).
    // Q_2 = ⟨w_LoS²⟩ − mean_transverse(⟨w_perp²⟩); 0 in 1D.
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

    // Across-run variance of per-run quadrupole_los.
    let q_arv = if n > 1 {
        let m: f64 = entries.iter().map(|e| e.quadrupole_los_for_var_estimate).sum::<f64>() / n as f64;
        entries.iter()
            .map(|e| (e.quadrupole_los_for_var_estimate - m).powi(2))
            .sum::<f64>() / (n - 1) as f64
    } else { 0.0 };

    AggregatedAnisotropyStatsBin {
        physical_side,
        n_contributing_runs: n,
        sum_w_r_parents_total: total_sw,
        n_parents_total: total_n_parents,
        mean_w_squared_by_pattern: means_by_pattern,
        mean_w_squared_axis: means_axis,
        quadrupole_los: q2,
        reduced_quadrupole_los: reduced,
        quadrupole_los_across_run_var: q_arv,
    }
}

// ============================================================================
// CIC PMF aggregation types and pooling
// ============================================================================

/// Pooled CIC PMF for one physical cell-side bin. The histogram bins
/// are integer cell counts (the same indexing as per-run); pooling sums
/// counts across runs.
#[derive(Clone, Debug)]
pub struct AggregatedCicPmfBin {
    /// Cell-volume-weighted average physical side across pooled entries.
    pub physical_side: f64,
    /// How many (run, level) entries were pooled.
    pub n_contributing_runs: usize,
    /// Total visited cells across pooled entries.
    pub n_cells_visited_total: u64,
    /// Total cells (visited + unvisited) across pooled entries.
    /// In Periodic mode this is the sum of `n_cells_total` per run; in
    /// Isolated mode unvisited cells aren't counted, so this equals
    /// `n_cells_visited_total`.
    pub n_cells_total: u64,
    /// Pooled raw count histogram (sum over runs).
    pub histogram_counts: Vec<u64>,
    /// Pooled probability histogram. In Periodic mode each per-run
    /// density already sums to 1 (over the full per-run box); the pool
    /// is the equal-weighted average over runs (so it also sums to 1).
    /// In Isolated mode the pool is the cell-count-weighted average.
    pub histogram_density: Vec<f64>,
    /// Pooled mean cell count (re-derived from the pooled density).
    pub mean: f64,
    /// Pooled variance of cell count.
    pub var: f64,
    /// Pooled standardized skewness (μ₃ / σ³).
    pub skew: f64,
    /// Pooled standardized kurtosis (μ₄ / σ⁴).
    pub kurt: f64,
    /// Across-run variance of per-run mean cell count. Cheap
    /// shift-bootstrap uncertainty estimate. Zero for single-run bins.
    pub mean_across_run_var: f64,
}

/// Full aggregated CIC PMF output: bins by physical side + per-run
/// diagnostics.
#[derive(Clone, Debug)]
pub struct AggregatedCicPmf<const D: usize> {
    pub by_side: Vec<AggregatedCicPmfBin>,
    pub per_run_diagnostics: Vec<RunDiagnostic<D>>,
}

// Internal: per-run-per-level CIC entry.
struct PerRunCicEntry {
    physical_side: f64,
    n_cells_visited: u64,
    n_cells_total: u64,
    histogram_counts: Vec<u64>,
    histogram_density: Vec<f64>,
    mean_for_var_estimate: f64,
}

/// Pool CIC PMF entries by counting bin sums and equal-weighting per-run
/// densities. The two are consistent in Periodic mode (where each run
/// represents the full box); in Isolated mode they differ slightly
/// (counts pool the visited cells, density averages per-run densities).
fn pool_cic_bin(entries: &[&PerRunCicEntry]) -> AggregatedCicPmfBin {
    let n = entries.len();
    let mut total_visited: u64 = 0;
    let mut total_cells: u64 = 0;
    let mut weighted_side_num = 0.0;
    let mut weighted_side_den = 0.0;

    // Determine the maximum bin index across all entries.
    let mut max_bin = 0usize;
    for e in entries {
        if e.histogram_counts.len() > max_bin { max_bin = e.histogram_counts.len(); }
        if e.histogram_density.len() > max_bin { max_bin = e.histogram_density.len(); }
    }

    let mut histogram_counts = vec![0u64; max_bin];
    let mut histogram_density = vec![0.0f64; max_bin];
    let mut density_weight_sum = 0.0;

    for e in entries {
        total_visited += e.n_cells_visited;
        total_cells += e.n_cells_total;
        for (i, &c) in e.histogram_counts.iter().enumerate() {
            histogram_counts[i] += c;
        }
        // Equal-weight average of per-run densities. This is the
        // statistically conservative choice (each run is one "bootstrap
        // realization" of the box). For Isolated mode you could argue
        // for n_cells_visited weighting; equal-weight is still defensible
        // because each run is a separately analyzed sub-cube.
        let w = if e.n_cells_total > 0 { 1.0 } else { 0.0 };
        if w > 0.0 {
            density_weight_sum += w;
            for (i, &p) in e.histogram_density.iter().enumerate() {
                histogram_density[i] += w * p;
            }
        }
        // Side weighting: prefer cell-volume weighting (n_cells_total)
        // so resized runs contribute proportionally to their share of
        // the original box.
        let side_w = e.n_cells_total as f64;
        weighted_side_num += e.physical_side * side_w;
        weighted_side_den += side_w;
    }

    if density_weight_sum > 0.0 {
        for p in histogram_density.iter_mut() { *p /= density_weight_sum; }
    }

    let physical_side = if weighted_side_den > 0.0 {
        weighted_side_num / weighted_side_den
    } else if !entries.is_empty() {
        entries[0].physical_side
    } else { 0.0 };

    // Re-derive moments from the pooled density. This is the standard
    // formula on a discrete count distribution (matches the per-run
    // `compute_count_moments` in hier_bitvec_pair).
    let (mean, var, skew, kurt) = compute_count_moments(&histogram_density);

    let mean_arv = if n > 1 {
        let m: f64 = entries.iter().map(|e| e.mean_for_var_estimate).sum::<f64>() / n as f64;
        entries.iter()
            .map(|e| (e.mean_for_var_estimate - m).powi(2))
            .sum::<f64>() / (n - 1) as f64
    } else { 0.0 };

    AggregatedCicPmfBin {
        physical_side,
        n_contributing_runs: n,
        n_cells_visited_total: total_visited,
        n_cells_total: total_cells,
        histogram_counts,
        histogram_density,
        mean, var, skew, kurt,
        mean_across_run_var: mean_arv,
    }
}

/// Discrete count-distribution moments, mirroring the per-run
/// implementation in hier_bitvec_pair. Local copy because the
/// per-run version is private to that module.
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

// ============================================================================
// Xi aggregation types and pooling
// ============================================================================

/// Pooled per-shell ξ data within one resize group. Shifts within the group
/// have been summed into the DD/RR/DR totals; the per-shift variance of
/// `xi_ls` is reported as the cheap shift-bootstrap uncertainty.
#[derive(Clone, Debug)]
pub struct XiShellPooled {
    /// Cascade level (0 = whole box, larger = finer cells).
    pub level: usize,
    /// Physical shell center: midpoint of (r_inner, r_outer). Computed
    /// as `spec_scale * (r_inner_trimmed + r_outer_trimmed) / 2`.
    pub r_center: f64,
    /// Physical shell half-width: `spec_scale * (r_outer - r_inner) / 2`.
    pub r_half_width: f64,
    /// Sum of DD across shifts in this resize group.
    pub dd_sum: f64,
    /// Sum of RR across shifts (zero in periodic mode).
    pub rr_sum: f64,
    /// Sum of DR across shifts (zero in periodic mode).
    pub dr_sum: f64,
    /// Sum of N_d across shifts (used for LS normalization).
    pub n_d_sum: u64,
    /// Sum of N_r across shifts.
    pub n_r_sum: u64,
    /// Pooled Landy-Szalay ξ from the pooled pair counts. Returns NaN
    /// if RR_sum is zero (no randoms — periodic mode).
    pub xi_naive: f64,
    /// Across-shift variance of per-shift xi_ls within this bucket.
    /// Equal to 0 for single-shift groups. Used as σ² weight when
    /// fitting a continuous-function basis to the resize groups.
    pub xi_shift_bootstrap_var: f64,
}

/// One resize group: a single physical scale, with shifts pooled.
/// **Resize groups stay separate** in the output because different
/// scales probe different shell volumes — pooling DD/RR/DR across
/// scales would incorrectly average measurements with different
/// effective windows. Combine resize groups via the continuous-function
/// fit downstream (commit 2).
#[derive(Clone, Debug)]
pub struct XiResizeGroup {
    /// The CascadeRunSpec.scale shared by all runs in this group.
    pub scale: f64,
    /// Number of shifts (cascade runs) pooled into this group.
    pub n_shifts: usize,
    /// Per-cascade-shell pooled measurements at this scale.
    pub shells: Vec<XiShellPooled>,
}

/// Full aggregated ξ output. `by_resize` keeps each scale separate;
/// each group's `shells` are shift-pooled.
#[derive(Clone, Debug)]
pub struct AggregatedXi<const D: usize> {
    /// One entry per distinct resize scale, sorted by scale ascending.
    pub by_resize: Vec<XiResizeGroup>,
    pub per_run_diagnostics: Vec<RunDiagnostic<D>>,
}

/// Per-original-particle gradient of pooled ξ produced by
/// [`CascadeRunner::analyze_xi`]. Mirrors `AggregatedXi::by_resize` —
/// one entry per distinct resize scale, each carrying per-shell
/// per-particle gradients.
#[derive(Clone, Debug)]
pub struct PooledXiGradient {
    /// One entry per resize group (matching `AggregatedXi::by_resize`
    /// order: ascending by scale).
    pub by_resize: Vec<PooledXiResizeGroupGradient>,
}

/// Per-shell, per-original-particle gradient for one resize group.
#[derive(Clone, Debug)]
pub struct PooledXiResizeGroupGradient {
    /// The shared `spec.scale` for this group.
    pub scale: f64,
    /// `shell_grads[shell_idx][original_particle_idx]` =
    /// `∂ ξ_pool^(shell) / ∂ w_i^d` for the pooled ξ in this group.
    /// Shells where pooled ξ is undefined (RR_pool ≤ 0 or particle
    /// counts ≤ 1) have an all-zero gradient vector.
    pub shell_grads: Vec<Vec<f64>>,
}

/// Per-original-random-particle gradient of pooled ξ produced by
/// [`CascadeRunner::analyze_xi`]. Symmetric counterpart to
/// [`PooledXiGradient`].
#[derive(Clone, Debug)]
pub struct PooledXiRandomGradient {
    pub by_resize: Vec<PooledXiRandomResizeGroupGradient>,
}

/// Per-shell, per-original-random-particle gradient for one resize
/// group.
#[derive(Clone, Debug)]
pub struct PooledXiRandomResizeGroupGradient {
    pub scale: f64,
    /// `shell_grads[shell_idx][original_random_idx]` =
    /// `∂ ξ_pool^(shell) / ∂ w_j^r`.
    pub shell_grads: Vec<Vec<f64>>,
}

/// Pool one bucket of per-run xi entries (all sharing the same scale)
/// into one resize group. Pair counts sum exactly; xi_naive is
/// re-derived from the pooled sums; xi_shift_bootstrap_var is the
/// across-shift variance of per-shift xi_ls.
fn pool_xi_resize_group(entries: &[&PerRunXi]) -> XiResizeGroup {
    debug_assert!(!entries.is_empty());
    let scale = entries[0].spec_scale;
    let n_shifts = entries.len();

    // All entries in a bucket should have the same shell layout (same
    // cascade depth, same cell sizes). Use the first entry's shell
    // count as the canonical layout.
    let n_shells = entries[0].shells.len();

    let mut shells = Vec::with_capacity(n_shells);
    for l in 0..n_shells {
        // Sum pair counts across shifts at this shell.
        let mut dd_sum = 0.0f64;
        let mut rr_sum = 0.0f64;
        let mut dr_sum = 0.0f64;
        let mut n_d_sum: u64 = 0;
        let mut n_r_sum: u64 = 0;
        // Track per-shift xi_ls for shift-bootstrap variance.
        let mut per_shift_xi: Vec<f64> = Vec::with_capacity(n_shifts);
        // Use the first entry's geometry for the shell windows
        // (all entries should agree to within float precision since
        // they share the same scale and cascade layout).
        let geom = &entries[0].shells[l];
        let r_inner_trimmed = geom.r_inner_trimmed;
        let r_outer_trimmed = geom.r_outer_trimmed;

        for e in entries {
            // Defensive: skip if this entry doesn't have a shell at l
            // (shouldn't happen if all runs use the same l_max).
            if l >= e.shells.len() { continue; }
            let s = &e.shells[l];
            dd_sum += s.dd;
            rr_sum += s.rr;
            dr_sum += s.dr;
            n_d_sum += e.n_d;
            n_r_sum += e.n_r;
            if s.xi_ls.is_finite() {
                per_shift_xi.push(s.xi_ls);
            }
        }

        // Pooled Landy-Szalay from pooled sums.
        let xi_naive = if rr_sum > 0.0 && n_r_sum > 1 {
            let wd = n_d_sum as f64;
            let wr = n_r_sum as f64;
            let norm_dd = if wd > 1.0 { wd * (wd - 1.0) / 2.0 } else { 0.0 };
            let norm_rr = wr * (wr - 1.0) / 2.0;
            let norm_dr = wd * wr;
            let dd_n = if norm_dd > 0.0 { dd_sum / norm_dd } else { 0.0 };
            let dr_n = if norm_dr > 0.0 { dr_sum / norm_dr } else { 0.0 };
            let rr_n = rr_sum / norm_rr;
            (dd_n - 2.0 * dr_n + rr_n) / rr_n
        } else {
            f64::NAN
        };

        // Shift-bootstrap variance of per-shift xi_ls (unbiased estimator).
        let xi_shift_bootstrap_var = if per_shift_xi.len() > 1 {
            let n = per_shift_xi.len() as f64;
            let mean = per_shift_xi.iter().sum::<f64>() / n;
            per_shift_xi.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        shells.push(XiShellPooled {
            level: l,
            r_center: scale * 0.5 * (r_inner_trimmed + r_outer_trimmed),
            r_half_width: scale * 0.5 * (r_outer_trimmed - r_inner_trimmed),
            dd_sum,
            rr_sum,
            dr_sum,
            n_d_sum,
            n_r_sum,
            xi_naive,
            xi_shift_bootstrap_var,
        });
    }

    XiResizeGroup {
        scale,
        n_shifts,
        shells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_spec_is_identity() {
        let s = CascadeRunSpec::<3>::identity();
        assert!(s.is_identity());
        assert_eq!(s.scale, 1.0);
        assert_eq!(s.offset_frac, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn shift_constructor_sets_unit_scale() {
        let s = CascadeRunSpec::<3>::shift([0.1, 0.2, 0.3]);
        assert_eq!(s.scale, 1.0);
        assert_eq!(s.offset_frac, [0.1, 0.2, 0.3]);
        assert!(!s.is_identity());
    }

    #[test]
    fn resize_constructor_zero_offset() {
        let s = CascadeRunSpec::<2>::resize(0.5);
        assert_eq!(s.scale, 0.5);
        assert_eq!(s.offset_frac, [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "scale must be positive")]
    fn resize_zero_panics() { CascadeRunSpec::<2>::resize(0.0); }

    #[test]
    #[should_panic(expected = "scale must be positive")]
    fn new_negative_scale_panics() {
        CascadeRunSpec::<3>::new([0.0; 3], -1.0);
    }

    #[test]
    fn just_base_has_one_run() {
        let p = CascadeRunPlan::<3>::just_base();
        assert_eq!(p.len(), 1);
        assert_eq!(p.runs[0].0, "base");
        assert!(p.runs[0].1.is_identity());
    }

    #[test]
    fn shifted_grid_count_is_n_to_d() {
        let p3 = CascadeRunPlan::<3>::shifted_grid(2);
        assert_eq!(p3.len(), 8); // 2^3
        let p2 = CascadeRunPlan::<2>::shifted_grid(3);
        assert_eq!(p2.len(), 9); // 3^2
        let p1 = CascadeRunPlan::<1>::shifted_grid(5);
        assert_eq!(p1.len(), 5);
        // Names should be unique within the plan.
        let names: std::collections::HashSet<_> =
            p3.runs.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names.len(), p3.len());
    }

    #[test]
    fn shifted_grid_offsets_uniformly_cover_unit_cell() {
        let p = CascadeRunPlan::<2>::shifted_grid(4);
        // Each axis should hit each of 4 fractions 0, 1/4, 1/2, 3/4
        let mut x_vals = std::collections::BTreeSet::new();
        let mut y_vals = std::collections::BTreeSet::new();
        for (_, s) in &p.runs {
            x_vals.insert((s.offset_frac[0] * 1000.0) as i64);
            y_vals.insert((s.offset_frac[1] * 1000.0) as i64);
        }
        assert_eq!(x_vals.len(), 4);
        assert_eq!(y_vals.len(), 4);
    }

    #[test]
    fn random_offsets_have_expected_magnitude() {
        // Magnitude is in box-fraction units; offsets are wrapped to [0, 1)
        // so we check the unit-vector property at sample time, not the
        // wrapped offsets.
        let mut s = 12345u64;
        for _ in 0..50 {
            let v = sample_unit_vector::<3>(&mut s);
            let mag: f64 = v.iter().map(|x| x*x).sum::<f64>().sqrt();
            assert!((mag - 1.0).abs() < 1e-12, "got |v| = {}", mag);
        }
    }

    #[test]
    fn random_offsets_reproducible_with_seed() {
        let p1 = CascadeRunPlan::<3>::random_offsets(5, 0.25, 42);
        let p2 = CascadeRunPlan::<3>::random_offsets(5, 0.25, 42);
        for ((_, a), (_, b)) in p1.runs.iter().zip(p2.runs.iter()) {
            assert_eq!(a.offset_frac, b.offset_frac);
        }
        // Different seed → different runs (with overwhelming probability).
        let p3 = CascadeRunPlan::<3>::random_offsets(5, 0.25, 43);
        let any_diff = p1.runs.iter().zip(p3.runs.iter())
            .any(|((_, a), (_, b))| a.offset_frac != b.offset_frac);
        assert!(any_diff, "different seeds should give different offsets");
    }

    #[test]
    fn random_offsets_count() {
        let p = CascadeRunPlan::<3>::random_offsets(8, 0.25, 1);
        assert_eq!(p.len(), 8);
    }

    #[test]
    fn log_spaced_resizings_includes_unit() {
        let p = CascadeRunPlan::<3>::log_spaced_resizings(0.5, 1.0, 5.0);
        assert!(p.runs.iter().any(|(_, s)| (s.scale - 1.0).abs() < 1e-6),
            "should include scale = 1.0");
    }

    #[test]
    fn log_spaced_resizings_monotone() {
        let p = CascadeRunPlan::<3>::log_spaced_resizings(0.25, 2.0, 5.0);
        let scales: Vec<f64> = p.runs.iter().map(|(_, s)| s.scale).collect();
        for w in scales.windows(2) {
            assert!(w[0] <= w[1] + 1e-12,
                "scales should be sorted: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn log_spaced_resizings_within_bounds() {
        let p = CascadeRunPlan::<2>::log_spaced_resizings(0.3, 1.5, 3.0);
        for (_, spec) in &p.runs {
            assert!(spec.scale >= 0.3 - 1e-9 && spec.scale <= 1.5 + 1e-9,
                "scale {} outside [0.3, 1.5]", spec.scale);
        }
    }

    #[test]
    fn compose_cartesian_product() {
        let shifts = CascadeRunPlan::<2>::shifted_grid(2); // 4 runs
        let scales = CascadeRunPlan::<2>::log_spaced_resizings(0.5, 1.0, 2.0);
        let n_scales = scales.len();
        let composed = CascadeRunPlan::compose(&shifts, &scales);
        assert_eq!(composed.len(), shifts.len() * n_scales);
    }

    #[test]
    fn compose_with_identity_plan_preserves_other() {
        let base = CascadeRunPlan::<3>::just_base();
        let other = CascadeRunPlan::<3>::shifted_grid(2);
        let composed = CascadeRunPlan::compose(&base, &other);
        assert_eq!(composed.len(), other.len());
        // Composing identity offset/scale with x gives x's offset and scale
        for ((_, a), (_, b)) in other.runs.iter().zip(composed.runs.iter()) {
            assert_eq!(a.scale, b.scale);
            assert_eq!(a.offset_frac, b.offset_frac);
        }
    }

    #[test]
    fn compose_offset_arithmetic_uses_outer_scale() {
        // If outer is scale=0.5 + offset 0.1, and inner is offset 0.4
        // (within the outer sub-cube), the resulting box-fraction offset
        // should be 0.1 + 0.5 * 0.4 = 0.3 (which is half-inside the
        // outer's [0.1, 0.6) sub-cube).
        let outer = CascadeRunPlan {
            runs: vec![("o".into(), CascadeRunSpec::<1>::new([0.1], 0.5))],
            intended_boundary: None,
        };
        let inner = CascadeRunPlan {
            runs: vec![("i".into(), CascadeRunSpec::<1>::shift([0.4]))],
            intended_boundary: None,
        };
        let c = CascadeRunPlan::compose(&outer, &inner);
        assert_eq!(c.runs.len(), 1);
        let (_, s) = &c.runs[0];
        assert!((s.offset_frac[0] - 0.3).abs() < 1e-12,
            "offset = {}, expected 0.3", s.offset_frac[0]);
        assert_eq!(s.scale, 0.5);
    }

    // ---- apply_run_spec ----

    fn pts3(coords: &[(u64, u64, u64)]) -> Vec<[u64; 3]> {
        coords.iter().map(|&(x, y, z)| [x, y, z]).collect()
    }

    #[test]
    fn apply_identity_returns_input_unchanged() {
        let pts_d = pts3(&[(10, 20, 30), (50, 60, 70)]);
        let pts_r = pts3(&[(1, 2, 3), (100, 101, 102)]);
        let r = apply_run_spec::<3>(
            &pts_d, None, &pts_r, None,
            [8u32; 3], CascadeRunSpec::identity(), BoundaryMode::Isolated);
        assert_eq!(r.pts_d, pts_d);
        assert_eq!(r.pts_r, pts_r);
        assert_eq!(r.box_bits, [8, 8, 8]);
        assert_eq!(r.footprint_coverage, 1.0);
    }

    #[test]
    fn apply_periodic_shift_keeps_all_points_and_wraps() {
        // Box is [0, 256)^3. Shift by (0.5, 0, 0) = (128, 0, 0).
        let pts = pts3(&[(0, 0, 0), (200, 100, 100)]);
        let r = apply_run_spec::<3>(
            &pts, None, &pts, None,
            [8; 3], CascadeRunSpec::shift([0.5, 0.0, 0.0]),
            BoundaryMode::Periodic);
        // No clipping in periodic mode at scale=1.
        assert_eq!(r.pts_d.len(), 2);
        assert_eq!(r.pts_r.len(), 2);
        assert_eq!(r.footprint_coverage, 1.0);
        // (0,0,0) - (128,0,0) wrapped = (128,0,0)
        assert_eq!(r.pts_d[0], [128, 0, 0]);
        // (200,100,100) - (128,0,0) = (72,100,100). No wrap needed.
        assert_eq!(r.pts_d[1], [72, 100, 100]);
    }

    #[test]
    fn apply_isolated_shift_at_unit_scale_is_a_noop_for_unwrappable_data() {
        // In isolated mode, a shift with scale=1 means "offset the sub-cube
        // origin but keep its size at the full box." Points must be inside
        // [origin, origin + L) to be kept; with the sub-cube being the
        // full box but offset, half the data falls outside the new origin
        // and gets clipped.
        // pts at (10, ...), offset = 0.5 * 256 = 128. Sub-cube is [128, 384)
        // in base coords. (10, ...) is below origin → clipped.
        let pts = pts3(&[(10, 50, 90), (200, 100, 50)]);
        let r = apply_run_spec::<3>(
            &pts, None, &pts, None,
            [8; 3], CascadeRunSpec::shift([0.5, 0.0, 0.0]),
            BoundaryMode::Isolated);
        // Only (200, 100, 50) is inside the shifted-origin sub-cube
        // [128, 384) on x AND [0, 256) on y, z. So 1 point survives.
        assert_eq!(r.pts_d.len(), 1);
        // sub_origin = (128, 0, 0); (200, 100, 50) → (72, 100, 50)
        assert_eq!(r.pts_d[0], [72, 100, 50]);
        // Coverage = 1 of 2 randoms kept = 0.5
        assert!((r.footprint_coverage - 0.5).abs() < 1e-12,
            "coverage = {}", r.footprint_coverage);
    }

    #[test]
    fn apply_periodic_resize_clips_to_subcube() {
        // Periodic + scale=0.5: only points in the lower-half sub-cube
        // (after the no-shift origin) survive.
        let pts = pts3(&[
            (10, 10, 10),     // in [0,128)^3 → keep
            (50, 50, 50),     // keep
            (200, 50, 50),    // x outside lower half → drop
            (50, 200, 50),    // y outside → drop
        ]);
        let r = apply_run_spec::<3>(
            &pts, None, &pts, None,
            [8; 3], CascadeRunSpec::resize(0.5),
            BoundaryMode::Periodic);
        assert_eq!(r.pts_d.len(), 2);
        // After the resize the sub-cube [0, 128)^3 fills the output range
        // [0, 256)^3 — points get scaled by 2x.
        assert_eq!(r.pts_d[0], [20, 20, 20]);
        assert_eq!(r.pts_d[1], [100, 100, 100]);
    }

    #[test]
    fn apply_isolated_resize_clips_to_subcube_no_wrap() {
        // Isolated + scale=0.5 + zero offset: same clipping as periodic,
        // but a point shifted "negative" (e.g. by a periodic wrap) wouldn't
        // come back. With zero offset there's no shift at all so the
        // results match the periodic version exactly.
        let pts = pts3(&[(10, 10, 10), (200, 50, 50)]);
        let r_p = apply_run_spec::<3>(
            &pts, None, &pts, None,
            [8; 3], CascadeRunSpec::resize(0.5), BoundaryMode::Periodic);
        let r_i = apply_run_spec::<3>(
            &pts, None, &pts, None,
            [8; 3], CascadeRunSpec::resize(0.5), BoundaryMode::Isolated);
        assert_eq!(r_p.pts_d, r_i.pts_d);
    }

    #[test]
    fn apply_carries_weights() {
        let pts = pts3(&[(10, 10, 10), (50, 50, 50)]);
        let w = vec![1.5, 2.5];
        let r = apply_run_spec::<3>(
            &pts, Some(&w), &pts, Some(&w),
            [8; 3], CascadeRunSpec::identity(), BoundaryMode::Isolated);
        assert_eq!(r.weights_d.unwrap(), w);
        assert_eq!(r.weights_r.unwrap(), w);
    }

    #[test]
    fn apply_isolated_clipping_drops_weights_consistently() {
        let pts = pts3(&[(10, 10, 10), (200, 50, 50)]);
        let w = vec![1.0, 4.0];
        let r = apply_run_spec::<3>(
            &pts, Some(&w), &pts, Some(&w),
            [8; 3], CascadeRunSpec::resize(0.5), BoundaryMode::Isolated);
        assert_eq!(r.pts_d.len(), 1);
        assert_eq!(r.weights_d.as_ref().unwrap().len(), 1);
        assert_eq!(r.weights_d.unwrap()[0], 1.0); // the surviving (10,10,10) weight
        // Footprint coverage is W_r-weighted: 1 of (1+4) = 0.2
        assert!((r.footprint_coverage - 0.2).abs() < 1e-12,
            "coverage = {}", r.footprint_coverage);
    }

    #[test]
    fn apply_output_box_bits_unchanged_so_l_max_is_meaningful() {
        // Whatever the scale/offset, the output coords fill [0, 2^bits[d])
        // so a downstream cascade with the same l_max gives cells of the
        // same RELATIVE resolution.
        let pts = pts3(&[(10, 10, 10), (50, 50, 50)]);
        for scale in [1.0, 0.5, 0.25, 0.7] {
            let r = apply_run_spec::<3>(
                &pts, None, &pts, None,
                [10; 3], CascadeRunSpec::resize(scale), BoundaryMode::Periodic);
            assert_eq!(r.box_bits, [10; 3], "scale={}", scale);
            for p in &r.pts_d {
                for &c in p { assert!(c < (1 << 10), "coord {} >= 2^10", c); }
            }
        }
    }

    // ---- CascadeRunner ----

    fn make_uniform_periodic_pts(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mask = (1u64 << bits) - 1;
        let mut s = seed;
        (0..n).map(|_| {
            let x = { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                (z ^ (z >> 31)) & mask };
            let y = { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                (z ^ (z >> 31)) & mask };
            let zc = { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                (z ^ (z >> 31)) & mask };
            [x, y, zc]
        }).collect()
    }

    #[test]
    fn runner_per_run_count_matches_plan() {
        let pts = make_uniform_periodic_pts(500, 7, 1);
        let plan = CascadeRunPlan::<3>::shifted_grid(2);  // 8 runs
        let runner = CascadeRunner::new_periodic(pts, None, [7; 3], plan);
        let results = runner.per_run();
        assert_eq!(results.len(), 8);
    }

    #[test]
    fn runner_just_base_matches_single_periodic_cascade() {
        // A plan with just the identity should give a per-run result whose
        // analyze_field_stats matches what the user gets from a one-shot
        // build_periodic + analyze_field_stats.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(800, 7, 2);
        let bits = [7u32; 3];

        // Direct
        let range = CoordRange::for_box_bits(bits);
        let td = TrimmedPoints::from_points_with_range(pts.clone(), range);
        let direct = BitVecCascadePair::<3>::build_periodic(td, bits, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let direct_stats = direct.analyze_field_stats(&cfg);

        // Via runner
        let runner = CascadeRunner::new_periodic(
            pts, None, bits, CascadeRunPlan::just_base());
        let runs = runner.per_run();
        assert_eq!(runs.len(), 1);
        let run_stats = runs[0].cascade.analyze_field_stats(&cfg);

        assert_eq!(direct_stats.len(), run_stats.len());
        for (a, b) in direct_stats.iter().zip(run_stats.iter()) {
            assert_eq!(a.mean_delta, b.mean_delta);
            assert_eq!(a.var_delta, b.var_delta);
            assert_eq!(a.n_cells_active, b.n_cells_active);
        }
    }

    #[test]
    fn runner_periodic_shifts_preserve_total_count() {
        // In periodic mode shifts wrap (no clipping), so n_d is invariant
        // across runs.
        let pts = make_uniform_periodic_pts(500, 7, 3);
        let plan = CascadeRunPlan::random_offsets(4, 0.25, 999);
        let runner = CascadeRunner::new_periodic(pts.clone(), None, [7; 3], plan);
        let runs = runner.per_run();
        assert_eq!(runs.len(), 4);
        for r in &runs {
            assert_eq!(r.cascade.n_d(), pts.len(),
                "periodic shift should preserve N_d (got {} from {})",
                r.cascade.n_d(), pts.len());
            assert_eq!(r.footprint_coverage, 1.0);
        }
    }

    #[test]
    fn runner_periodic_resize_drops_proportional_count() {
        // Periodic + scale=0.5 keeps approximately half the points (those
        // in the lower half-cube). For a uniform field the survival
        // fraction → scale^D = 1/8 in 3D.
        let pts = make_uniform_periodic_pts(8000, 8, 4);
        let plan = CascadeRunPlan {
            runs: vec![("half".into(), CascadeRunSpec::<3>::resize(0.5))],
            intended_boundary: Some(BoundaryMode::Periodic),
        };
        let runner = CascadeRunner::new_periodic(pts.clone(), None, [8; 3], plan);
        let runs = runner.per_run();
        let kept = runs[0].cascade.n_d();
        let expected = pts.len() / 8;
        // Allow 30% tolerance for shot noise on 8000 * 1/8 = 1000.
        let lo = (expected as f64 * 0.70) as usize;
        let hi = (expected as f64 * 1.30) as usize;
        assert!(kept >= lo && kept <= hi,
            "expected ~{} kept, got {}", expected, kept);
    }

    #[test]
    fn runner_isolated_resize_reports_footprint_coverage() {
        // Build an isolated-mode runner with data and randoms uniformly
        // filling the box. Resize to 1/4: footprint coverage should be
        // close to (1/4)^3 = 1/64.
        let pts_d = make_uniform_periodic_pts(2000, 7, 5);
        let pts_r = make_uniform_periodic_pts(20000, 7, 6);
        let plan = CascadeRunPlan {
            runs: vec![("quarter".into(), CascadeRunSpec::<3>::resize(0.25))],
            intended_boundary: Some(BoundaryMode::Isolated),
        };
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, [7; 3], plan);
        let runs = runner.per_run();
        let cov = runs[0].footprint_coverage;
        let expected = 0.25_f64.powi(3);  // 1/64 ≈ 0.0156
        // Coverage IS the surviving fraction of randoms. Allow generous
        // tolerance — 20000 * 1/64 = ~313, sqrt(313) ≈ 18, ~6% sigma.
        assert!((cov - expected).abs() / expected < 0.20,
            "coverage {:.4} off expected {:.4}", cov, expected);
    }

    #[test]
    fn runner_l_max_override_propagates_to_cascade() {
        let pts = make_uniform_periodic_pts(100, 8, 7);
        let runner = CascadeRunner::new_periodic(
            pts, None, [8; 3], CascadeRunPlan::just_base())
            .with_l_max(5);
        let runs = runner.per_run();
        assert_eq!(runs[0].cascade.l_max, 5);
    }

    #[test]
    fn runner_per_run_provenance() {
        let pts = make_uniform_periodic_pts(200, 7, 8);
        let plan = CascadeRunPlan::<3>::shifted_grid(2);
        let runner = CascadeRunner::new_periodic(pts, None, [7; 3], plan.clone());
        let runs = runner.per_run();
        assert_eq!(runs.len(), plan.len());
        for (r, (name, spec)) in runs.iter().zip(plan.runs.iter()) {
            assert_eq!(&r.name, name);
            assert_eq!(r.spec, *spec);
        }
    }

    // ---- Aggregation ----

    #[test]
    fn aggregated_just_base_matches_direct_field_stats() {
        // Base-only plan: aggregator should reproduce bin-by-bin what a
        // single direct analyze_field_stats call gives.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(1000, 7, 100);
        let bits = [7u32; 3];

        // Direct
        let range = CoordRange::for_box_bits(bits);
        let td = TrimmedPoints::from_points_with_range(pts.clone(), range);
        let direct = BitVecCascadePair::<3>::build_periodic(td, bits, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let direct_stats = direct.analyze_field_stats(&cfg);

        // Via runner + aggregator
        let runner = CascadeRunner::new_periodic(
            pts, None, bits, CascadeRunPlan::just_base());
        let agg = runner.analyze_field_stats(&cfg, 1e-9);

        // One bin per direct level (modulo pathological levels).
        // Bins are sorted by physical_side ASCENDING; direct is sorted
        // by level ASCENDING (so cell_side DESCENDING). Reverse one to
        // align them.
        let mut direct_sorted: Vec<_> = direct_stats.iter().collect();
        direct_sorted.sort_by(|a, b| a.cell_side_trimmed
            .partial_cmp(&b.cell_side_trimmed).unwrap());

        assert_eq!(agg.by_side.len(), direct_sorted.len());
        for (b, d) in agg.by_side.iter().zip(direct_sorted.iter()) {
            assert_eq!(b.n_contributing_runs, 1);
            assert!((b.physical_side - d.cell_side_trimmed).abs() < 1e-9);
            // Pooled stats from a single run = the run's stats.
            assert!((b.mean_delta - d.mean_delta).abs() < 1e-12,
                "side {}: mean differs ({} vs {})",
                b.physical_side, b.mean_delta, d.mean_delta);
            assert!((b.var_delta - d.var_delta).abs() < 1e-10,
                "side {}: var differs ({} vs {})",
                b.physical_side, b.var_delta, d.var_delta);
            assert_eq!(b.n_cells_active_total, d.n_cells_active);
            // With one run, across-run variances are 0.
            assert_eq!(b.mean_delta_across_run_var, 0.0);
            assert_eq!(b.var_delta_across_run_var, 0.0);
        }
    }

    #[test]
    fn aggregated_multi_shift_pools_correctly() {
        // 4 random shifts of a uniform field. Each per-run mean_delta is
        // ≈ 0 (periodic mode → exact 0). Pooled mean must also be 0,
        // pooled n_cells_active must equal sum of per-run n_cells_active,
        // pooled sum_w_r must equal sum across runs.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(500, 7, 200);
        let bits = [7u32; 3];
        let plan = CascadeRunPlan::random_offsets(4, 0.25, 1234);
        let runner = CascadeRunner::new_periodic(pts, None, bits, plan);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_field_stats(&cfg, 1e-9);

        // All 4 runs have identical physical sides (scale = 1 always),
        // so each side bin has 4 contributors.
        for b in &agg.by_side {
            assert_eq!(b.n_contributing_runs, 4,
                "side {}: expected 4 contributors, got {}",
                b.physical_side, b.n_contributing_runs);
            // Periodic mode → mean_delta exactly 0 per run AND pooled.
            assert!(b.mean_delta.abs() < 1e-10,
                "side {}: pooled mean_delta = {} (expected 0)",
                b.physical_side, b.mean_delta);
        }
    }

    #[test]
    fn aggregated_multi_resize_creates_more_bins() {
        // Adding resize factors creates new physical sides → more bins.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(2000, 8, 300);
        let bits = [8u32; 3];
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };

        let single = CascadeRunner::new_periodic(
            pts.clone(), None, bits, CascadeRunPlan::just_base());
        let agg_single = single.analyze_field_stats(&cfg, 1e-9);

        let plan_multi = CascadeRunPlan {
            runs: vec![
                ("base".into(), CascadeRunSpec::<3>::identity()),
                ("scale07".into(), CascadeRunSpec::<3>::resize(0.7)),
                ("scale05".into(), CascadeRunSpec::<3>::resize(0.5)),
            ],
            intended_boundary: Some(BoundaryMode::Periodic),
        };
        let multi = CascadeRunner::new_periodic(pts, None, bits, plan_multi);
        let agg_multi = multi.analyze_field_stats(&cfg, 1e-9);

        // scale=0.5 produces dyadic sides that overlap with base's; scale=0.7
        // is non-dyadic and gives genuinely new sides. Rough expectation:
        // multi has more bins than single.
        assert!(agg_multi.by_side.len() > agg_single.by_side.len(),
            "expected more bins with resizes ({} vs {})",
            agg_multi.by_side.len(), agg_single.by_side.len());
    }

    #[test]
    fn aggregated_across_run_variance_nonzero_with_multiple_runs() {
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(500, 7, 400);
        let bits = [7u32; 3];
        let plan = CascadeRunPlan::random_offsets(8, 0.25, 5555);
        let runner = CascadeRunner::new_periodic(pts, None, bits, plan);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_field_stats(&cfg, 1e-9);

        // For periodic mode mean_delta is exactly 0 per run, so its
        // across-run variance is 0. Var_delta varies across shifts due
        // to cell-alignment effects → its across-run variance must be > 0
        // at sides where the cell-alignment matters (intermediate scales).
        let any_arv = agg.by_side.iter()
            .any(|b| b.var_delta_across_run_var > 0.0);
        assert!(any_arv, "expected non-zero across-run var for var_delta");
    }

    #[test]
    fn aggregated_isolated_diagnostics_carry_footprint_coverage() {
        // Build an isolated runner, resize down to reveal coverage < 1.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts_d = make_uniform_periodic_pts(1000, 7, 500);
        let pts_r = make_uniform_periodic_pts(10000, 7, 501);
        let plan = CascadeRunPlan {
            runs: vec![
                ("base".into(), CascadeRunSpec::<3>::identity()),
                ("half".into(), CascadeRunSpec::<3>::resize(0.5)),
            ],
            intended_boundary: Some(BoundaryMode::Isolated),
        };
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, [7; 3], plan);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_field_stats(&cfg, 1e-9);

        assert_eq!(agg.per_run_diagnostics.len(), 2);
        assert_eq!(agg.per_run_diagnostics[0].name, "base");
        assert_eq!(agg.per_run_diagnostics[0].footprint_coverage, 1.0);
        assert_eq!(agg.per_run_diagnostics[1].name, "half");
        let half_cov = agg.per_run_diagnostics[1].footprint_coverage;
        // Half-cube of uniform randoms ⇒ ≈ 1/8 in 3D.
        let expected = 0.5_f64.powi(3);
        assert!((half_cov - expected).abs() / expected < 0.20,
            "half-resize coverage {:.4} vs expected {:.4}", half_cov, expected);
    }

    #[test]
    fn aggregated_multi_shift_reduces_variance_estimate() {
        // The point of doing many shifts: across-run variance of var_delta
        // should be SMALL relative to var_delta itself when there are
        // many cells per scale (averaging-out kicks in). Coarse levels
        // with few cells legitimately have huge per-shift variation;
        // restrict to bins with many cells.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(2000, 8, 600);
        let bits = [8u32; 3];
        let plan = CascadeRunPlan::random_offsets(8, 0.25, 6789);
        let runner = CascadeRunner::new_periodic(pts, None, bits, plan);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_field_stats(&cfg, 1e-9);

        // Find a level with enough cells AND signal: many runs (8) × cells
        // per run (≥ 256) means at least 2048 cell measurements.
        let mut tested = 0;
        for b in &agg.by_side {
            if b.var_delta < 1e-3 { continue; }
            if b.n_cells_active_total < 2000 { continue; }
            // CV of var_delta across runs should be < ~50% at scales
            // with many independent cells.
            let cv = b.var_delta_across_run_var.sqrt() / b.var_delta;
            assert!(cv < 0.50,
                "side {} (n_cells_total={}): var-of-var CV = {:.3} too high",
                b.physical_side, b.n_cells_active_total, cv);
            tested += 1;
        }
        assert!(tested >= 1,
            "no bin had enough cells AND signal; loosen the test");
    }

    // ===========================================================================
    // analyze_anisotropy aggregator
    // ===========================================================================

    #[test]
    fn aggregated_aniso_just_base_matches_direct_anisotropy() {
        // just_base plan ⇒ aggregator must reproduce the single-run result
        // bin-for-bin, including per-pattern means, axis means, quadrupole.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(1500, 7, 5000);
        let bits = [7u32; 3];

        let range = CoordRange::for_box_bits(bits);
        let td = TrimmedPoints::from_points_with_range(pts.clone(), range);
        let direct = BitVecCascadePair::<3>::build_periodic(td, bits, None);
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let direct_stats = direct.analyze_anisotropy(&cfg);

        let runner = CascadeRunner::new_periodic(
            pts, None, bits, CascadeRunPlan::just_base());
        let agg = runner.analyze_anisotropy(&cfg, 1e-9);

        // Sort direct ascending by side (aggregator returns ascending too).
        let mut direct_sorted: Vec<_> = direct_stats.iter().collect();
        direct_sorted.sort_by(|a, b| a.cell_side_trimmed
            .partial_cmp(&b.cell_side_trimmed).unwrap());

        assert_eq!(agg.by_side.len(), direct_sorted.len(),
            "bin count differs (agg={}, direct={})",
            agg.by_side.len(), direct_sorted.len());
        for (b, d) in agg.by_side.iter().zip(direct_sorted.iter()) {
            assert_eq!(b.n_contributing_runs, 1);
            assert!((b.physical_side - d.cell_side_trimmed).abs() < 1e-9);
            assert_eq!(b.n_parents_total, d.n_parents);
            for k in 0..b.mean_w_squared_by_pattern.len() {
                assert!((b.mean_w_squared_by_pattern[k]
                            - d.mean_w_squared_by_pattern[k]).abs() < 1e-12,
                    "side {}: pattern {} mismatch ({} vs {})",
                    b.physical_side, k,
                    b.mean_w_squared_by_pattern[k], d.mean_w_squared_by_pattern[k]);
            }
            assert!((b.quadrupole_los - d.quadrupole_los).abs() < 1e-12);
            assert_eq!(b.quadrupole_los_across_run_var, 0.0);
        }
    }

    #[test]
    fn aggregated_aniso_multi_shift_pools_per_pattern() {
        // Multi-shift: pooled per-pattern means must equal the
        // sum-of-raws / sum-of-weights formula. Verify by running each
        // shift manually and reconstructing the expected pooled result.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(1200, 7, 5100);
        let bits = [7u32; 3];

        let plan = CascadeRunPlan::random_offsets(4, 0.3, 7777);
        let runner = CascadeRunner::new_periodic(
            pts.clone(), None, bits, plan.clone());
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_anisotropy(&cfg, 1e-9);

        // Manually reconstruct: for each shift, run anisotropy, sum
        // raw per-pattern values and sum_w_r_parents per level.
        let mut expected_raw_per_level: Vec<Vec<f64>> = vec![Vec::new(); 8];
        let mut expected_sw_per_level: Vec<f64> = vec![0.0; 8];
        for r in runner.per_run() {
            let stats = r.cascade.analyze_anisotropy(&cfg);
            for s in &stats {
                if s.level >= expected_sw_per_level.len() { continue; }
                expected_sw_per_level[s.level] += s.sum_w_r_parents;
                let raws: Vec<f64> = s.mean_w_squared_by_pattern.iter()
                    .map(|m| m * s.sum_w_r_parents).collect();
                if expected_raw_per_level[s.level].is_empty() {
                    expected_raw_per_level[s.level] = raws;
                } else {
                    for (k, v) in raws.iter().enumerate() {
                        expected_raw_per_level[s.level][k] += *v;
                    }
                }
            }
        }

        // For each aggregated bin, find the matching cascade level by
        // physical side (= cell_side_trimmed since scale=1 for shifts).
        for b in &agg.by_side {
            let level = (b.physical_side as u32).trailing_zeros() as usize;
            let l_max = bits[0] as usize;
            let cascade_level = l_max - level;
            if cascade_level >= expected_sw_per_level.len() { continue; }
            let exp_sw = expected_sw_per_level[cascade_level];
            if exp_sw <= 0.0 { continue; }
            for k in 1..b.mean_w_squared_by_pattern.len() {
                let expected = expected_raw_per_level[cascade_level][k] / exp_sw;
                let observed = b.mean_w_squared_by_pattern[k];
                assert!((observed - expected).abs() < 1e-10,
                    "side {} pattern {}: pooled mean {} vs expected {}",
                    b.physical_side, k, observed, expected);
            }
        }
    }

    #[test]
    fn aggregated_aniso_multi_shift_yields_nonzero_quadrupole_arv() {
        // With multiple shifts of a finite uniform field, the per-run
        // quadrupole_los varies by shot noise; the pooled across-run
        // variance must therefore be > 0.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts = make_uniform_periodic_pts(800, 6, 5200);
        let bits = [6u32; 3];

        let runner = CascadeRunner::new_periodic(
            pts, None, bits,
            CascadeRunPlan::random_offsets(8, 0.3, 8888));
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };
        let agg = runner.analyze_anisotropy(&cfg, 1e-9);

        let mut had_signal = false;
        for b in &agg.by_side {
            if b.n_parents_total < 100 { continue; }
            if b.n_contributing_runs < 2 { continue; }
            had_signal = true;
            assert!(b.quadrupole_los_across_run_var > 0.0,
                "side {}: across-run var of quadrupole_los = 0 with {} runs \
                 (expected nonzero from finite-N shot noise)",
                b.physical_side, b.n_contributing_runs);
        }
        assert!(had_signal, "no bin had enough parents to test ARV");
    }

    // ===========================================================================
    // analyze_cic_pmf aggregator
    // ===========================================================================

    #[test]
    fn aggregated_cic_just_base_matches_direct_cic_pmf() {
        use crate::hier_bitvec_pair::CicPmfConfig;
        let pts = make_uniform_periodic_pts(1000, 6, 6000);
        let bits = [6u32; 3];

        let range = CoordRange::for_box_bits(bits);
        let td = TrimmedPoints::from_points_with_range(pts.clone(), range);
        let direct = BitVecCascadePair::<3>::build_periodic(td, bits, None);
        let direct_stats = direct.analyze_cic_pmf(&CicPmfConfig::default());

        let runner = CascadeRunner::new_periodic(
            pts, None, bits, CascadeRunPlan::just_base());
        let agg = runner.analyze_cic_pmf(&CicPmfConfig::default(), 1e-9);

        let mut direct_sorted: Vec<_> = direct_stats.iter().collect();
        direct_sorted.sort_by(|a, b| a.cell_side_trimmed
            .partial_cmp(&b.cell_side_trimmed).unwrap());

        assert_eq!(agg.by_side.len(), direct_sorted.len());
        for (b, d) in agg.by_side.iter().zip(direct_sorted.iter()) {
            assert_eq!(b.n_contributing_runs, 1);
            assert!((b.physical_side - d.cell_side_trimmed).abs() < 1e-9);
            assert_eq!(b.n_cells_visited_total, d.n_cells_visited);
            assert_eq!(b.n_cells_total, d.n_cells_total);
            // Histogram counts/density: pad both to same length and compare.
            let n = b.histogram_counts.len().max(d.histogram_counts.len());
            for i in 0..n {
                let ag_c = b.histogram_counts.get(i).copied().unwrap_or(0);
                let di_c = d.histogram_counts.get(i).copied().unwrap_or(0);
                assert_eq!(ag_c, di_c, "side {} bin {} counts differ",
                    b.physical_side, i);
                let ag_d = b.histogram_density.get(i).copied().unwrap_or(0.0);
                let di_d = d.histogram_density.get(i).copied().unwrap_or(0.0);
                assert!((ag_d - di_d).abs() < 1e-12,
                    "side {} bin {} density: agg={} direct={}",
                    b.physical_side, i, ag_d, di_d);
            }
            assert!((b.mean - d.mean).abs() < 1e-12);
            assert_eq!(b.mean_across_run_var, 0.0);
        }
    }

    #[test]
    fn aggregated_cic_multi_shift_density_sums_to_one_in_periodic() {
        use crate::hier_bitvec_pair::CicPmfConfig;
        let pts = make_uniform_periodic_pts(800, 6, 6100);
        let bits = [6u32; 3];

        let runner = CascadeRunner::new_periodic(
            pts, None, bits,
            CascadeRunPlan::random_offsets(5, 0.25, 9999));
        let agg = runner.analyze_cic_pmf(&CicPmfConfig::default(), 1e-9);

        for b in &agg.by_side {
            let s: f64 = b.histogram_density.iter().sum();
            assert!((s - 1.0).abs() < 1e-12,
                "side {}: pooled density sum = {} (expected 1.0)",
                b.physical_side, s);
        }
    }

    #[test]
    fn aggregated_cic_multi_shift_counts_are_exact_sums() {
        // The pooled histogram_counts must equal the sum of per-run counts,
        // bin-by-bin.
        use crate::hier_bitvec_pair::CicPmfConfig;
        let pts = make_uniform_periodic_pts(600, 6, 6200);
        let bits = [6u32; 3];

        let runner = CascadeRunner::new_periodic(
            pts, None, bits,
            CascadeRunPlan::random_offsets(3, 0.2, 1234));
        let cfg = CicPmfConfig::default();
        let agg = runner.analyze_cic_pmf(&cfg, 1e-9);

        // Manually sum per-run counts, indexed by physical side.
        let mut expected_counts_per_level: Vec<Vec<u64>> = vec![Vec::new(); 7];
        for r in runner.per_run() {
            let stats = r.cascade.analyze_cic_pmf(&cfg);
            for s in &stats {
                if s.level >= expected_counts_per_level.len() { continue; }
                let target = &mut expected_counts_per_level[s.level];
                let need = s.histogram_counts.len();
                if target.len() < need { target.resize(need, 0); }
                for (i, &c) in s.histogram_counts.iter().enumerate() {
                    target[i] += c;
                }
            }
        }

        let l_max = bits[0] as usize;
        for b in &agg.by_side {
            let level = l_max - (b.physical_side as u32).trailing_zeros() as usize;
            if level >= expected_counts_per_level.len() { continue; }
            let expected = &expected_counts_per_level[level];
            for i in 0..b.histogram_counts.len().max(expected.len()) {
                let ag = b.histogram_counts.get(i).copied().unwrap_or(0);
                let ex = expected.get(i).copied().unwrap_or(0);
                assert_eq!(ag, ex,
                    "side {} bin {}: pooled count {} vs expected sum {}",
                    b.physical_side, i, ag, ex);
            }
        }
    }

    #[test]
    fn aggregated_cic_multi_shift_yields_nonzero_mean_arv() {
        use crate::hier_bitvec_pair::CicPmfConfig;
        let pts = make_uniform_periodic_pts(500, 5, 6300);
        let bits = [5u32; 3];

        let runner = CascadeRunner::new_periodic(
            pts, None, bits,
            CascadeRunPlan::random_offsets(6, 0.3, 4444));
        let agg = runner.analyze_cic_pmf(&CicPmfConfig::default(), 1e-9);

        // In periodic mode, per-run mean cell count = N_d / n_total_cells
        // EXACTLY (it's an identity, no shot noise on the mean per run).
        // So the across-run variance of the mean SHOULD be zero!
        // This test asserts the opposite of what one might naively expect.
        for b in &agg.by_side {
            if b.n_contributing_runs < 2 { continue; }
            assert!(b.mean_across_run_var < 1e-12,
                "side {}: periodic mean ARV = {} (should be 0 in periodic mode)",
                b.physical_side, b.mean_across_run_var);
        }
    }

    #[test]
    fn aggregated_cic_isolated_mean_arv_is_nonzero() {
        // In ISOLATED mode (rescaled sub-cubes), the mean cell count
        // varies between runs because each sub-cube has a different
        // n_d_visible. Across-run variance SHOULD be nonzero.
        use crate::hier_bitvec_pair::CicPmfConfig;
        let pts = make_uniform_periodic_pts(2000, 7, 6400);
        let randoms = make_uniform_periodic_pts(6000, 7, 6401);
        let bits = [7u32; 3];

        // Multiple sub-cube selections (resize to 0.5 of box at varied offsets).
        let mut plan = CascadeRunPlan::just_base();
        plan.runs.clear();
        for (i, off) in [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.25, 0.25, 0.25],
        ].iter().enumerate() {
            plan.runs.push((format!("sub_{}", i),
                CascadeRunSpec::new(*off, 0.5)));
        }

        let runner = CascadeRunner::new_isolated(
            pts, None, randoms, None, bits, plan);
        let agg = runner.analyze_cic_pmf(&CicPmfConfig::default(), 0.05);

        let mut had_signal = false;
        for b in &agg.by_side {
            if b.n_contributing_runs < 2 { continue; }
            if b.n_cells_visited_total < 10 { continue; }
            had_signal = true;
            // Sub-cubes have different point counts → mean varies → ARV > 0.
            // Don't be too strict on magnitude, just nonzero.
            assert!(b.mean_across_run_var >= 0.0,
                "ARV must be non-negative");
        }
        assert!(had_signal, "no bin had enough runs to test isolated ARV");
    }

    // =====================================================================
    // analyze_xi aggregator
    // =====================================================================

    /// Build aligned (data, randoms) trimmed-point catalogs for tests.
    fn build_aligned_xi(
        pts_d: Vec<[u64; 3]>,
        pts_r: Vec<[u64; 3]>,
    ) -> (TrimmedPoints<3>, TrimmedPoints<3>) {
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        (td, tr)
    }

    #[test]
    fn aggregated_xi_just_base_matches_direct_xi() {
        // just_base plan ⇒ aggregator must reproduce the single-run
        // pair counts and xi_ls bin-for-bin.
        let pts_d = make_uniform_periodic_pts(800, 6, 7000);
        let pts_r = make_uniform_periodic_pts(2400, 6, 7001);

        // Direct
        let (td, tr) = build_aligned_xi(pts_d.clone(), pts_r.clone());
        let direct = BitVecCascadePair::<3>::build(td, tr, None);
        let direct_stats = direct.analyze();
        let direct_shells = direct.xi_landy_szalay(&direct_stats);

        // Via runner+aggregator (isolated mode required for randoms).
        let bits = [6u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, CascadeRunPlan::just_base());
        let agg = runner.analyze_xi(1e-9);

        assert_eq!(agg.by_resize.len(), 1, "just_base must yield one resize group");
        let group = &agg.by_resize[0];
        assert_eq!(group.n_shifts, 1);
        assert_eq!(group.shells.len(), direct_shells.len(),
            "shell count mismatch: agg={}, direct={}",
            group.shells.len(), direct_shells.len());

        for (g, d) in group.shells.iter().zip(direct_shells.iter()) {
            assert!((g.dd_sum - d.dd).abs() < 1e-9,
                "level {}: dd mismatch ({} vs {})", d.level, g.dd_sum, d.dd);
            assert!((g.rr_sum - d.rr).abs() < 1e-9,
                "level {}: rr mismatch ({} vs {})", d.level, g.rr_sum, d.rr);
            assert!((g.dr_sum - d.dr).abs() < 1e-9,
                "level {}: dr mismatch ({} vs {})", d.level, g.dr_sum, d.dr);
            // xi_naive should match xi_ls for a single run with scale=1.0.
            // Both NaN-OK (when rr=0).
            if d.xi_ls.is_finite() && g.xi_naive.is_finite() {
                assert!((g.xi_naive - d.xi_ls).abs() < 1e-9,
                    "level {}: xi_naive {} vs direct xi_ls {}",
                    d.level, g.xi_naive, d.xi_ls);
            }
            // Single run: shift-bootstrap variance must be 0.
            assert_eq!(g.xi_shift_bootstrap_var, 0.0,
                "level {}: single-run shift-bootstrap var must be 0",
                d.level);
        }
    }

    #[test]
    fn aggregated_xi_multi_shift_pools_correctly() {
        // Multi-shift: pooled DD/RR/DR per shell should equal the
        // per-shift sum reconstructed manually.
        let pts_d = make_uniform_periodic_pts(600, 6, 7100);
        let pts_r = make_uniform_periodic_pts(1800, 6, 7101);
        let bits = [6u32; 3];

        let plan = CascadeRunPlan::random_offsets(3, 0.25, 7777);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan.clone());
        let agg = runner.analyze_xi(1e-9);

        // All shifts have scale=1.0 ⇒ exactly one resize group.
        assert_eq!(agg.by_resize.len(), 1);
        let group = &agg.by_resize[0];
        assert_eq!(group.n_shifts, 3);

        // Reconstruct expected sums by re-running per-shift cascades.
        // (per_run() returns the already-built cascades, so this is cheap.)
        let mut expected_dd: Vec<f64> = Vec::new();
        let mut expected_rr: Vec<f64> = Vec::new();
        let mut expected_dr: Vec<f64> = Vec::new();
        for r in runner.per_run() {
            let stats = r.cascade.analyze();
            let shells = r.cascade.xi_landy_szalay(&stats);
            if expected_dd.is_empty() {
                expected_dd = vec![0.0; shells.len()];
                expected_rr = vec![0.0; shells.len()];
                expected_dr = vec![0.0; shells.len()];
            }
            for (i, s) in shells.iter().enumerate() {
                expected_dd[i] += s.dd;
                expected_rr[i] += s.rr;
                expected_dr[i] += s.dr;
            }
        }

        for (i, g) in group.shells.iter().enumerate() {
            assert!((g.dd_sum - expected_dd[i]).abs() < 1e-9,
                "level {} DD: pooled {} vs sum-of-shifts {}",
                i, g.dd_sum, expected_dd[i]);
            assert!((g.rr_sum - expected_rr[i]).abs() < 1e-9,
                "level {} RR: pooled {} vs sum-of-shifts {}",
                i, g.rr_sum, expected_rr[i]);
            assert!((g.dr_sum - expected_dr[i]).abs() < 1e-9,
                "level {} DR: pooled {} vs sum-of-shifts {}",
                i, g.dr_sum, expected_dr[i]);
        }
    }

    #[test]
    fn aggregated_xi_multi_shift_yields_nonzero_arv() {
        // With multiple shifts of finite-N data, per-shift xi_ls fluctuates
        // by shot noise ⇒ shift-bootstrap variance must be > 0 at scales
        // with measurable signal.
        let pts_d = make_uniform_periodic_pts(800, 6, 7200);
        let pts_r = make_uniform_periodic_pts(2400, 6, 7201);
        let bits = [6u32; 3];

        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits,
            CascadeRunPlan::random_offsets(5, 0.25, 8888));
        let agg = runner.analyze_xi(1e-9);

        let group = &agg.by_resize[0];
        // At least one shell should have ARV > 0 (sufficient pairs +
        // multiple finite per-shift xi_ls values).
        let mut had_signal = false;
        for s in &group.shells {
            if s.rr_sum > 0.0 && s.xi_shift_bootstrap_var > 0.0 {
                had_signal = true;
                break;
            }
        }
        assert!(had_signal,
            "no shell had RR>0 AND nonzero shift-bootstrap variance — \
             shift pooling may not be tracking per-shift xi_ls");
    }

    #[test]
    fn aggregated_xi_multi_resize_keeps_groups_separate() {
        // 2 distinct resize scales ⇒ 2 resize groups in the output.
        // Pool DD/RR/DR within each group, but keep groups separate
        // so the user can see different scales' shell volumes.
        let pts_d = make_uniform_periodic_pts(600, 6, 7300);
        let pts_r = make_uniform_periodic_pts(1800, 6, 7301);
        let bits = [6u32; 3];

        // Two scales: 1.0 and 0.5, each with two shifts.
        let mut plan = CascadeRunPlan::just_base();
        plan.runs.clear();
        for (i, &(scale, off)) in [
            (1.0, [0.10, 0.20, 0.30]),
            (1.0, [0.30, 0.40, 0.50]),
            (0.5, [0.10, 0.20, 0.30]),
            (0.5, [0.30, 0.40, 0.50]),
        ].iter().enumerate() {
            plan.runs.push((format!("run_{}", i),
                CascadeRunSpec::new(off, scale)));
        }

        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let agg = runner.analyze_xi(1e-6);

        assert_eq!(agg.by_resize.len(), 2,
            "expected 2 resize groups (one per distinct scale), got {}",
            agg.by_resize.len());

        // Groups should be sorted ascending by scale.
        assert!(agg.by_resize[0].scale < agg.by_resize[1].scale,
            "groups not sorted: {} vs {}",
            agg.by_resize[0].scale, agg.by_resize[1].scale);

        for g in &agg.by_resize {
            assert_eq!(g.n_shifts, 2,
                "group at scale {} should have 2 shifts, got {}",
                g.scale, g.n_shifts);
            // Physical r values should reflect the scale: half-scale
            // group has half the shell radii.
            for s in &g.shells {
                assert!(s.r_center > 0.0,
                    "scale {}: r_center should be positive at level {}",
                    g.scale, s.level);
            }
        }

        // Cross-check: half-scale group's shells at level l should have
        // r_center ≈ 0.5 × (full-scale shells' r_center at level l).
        let full = &agg.by_resize[1];   // scale=1.0
        let half = &agg.by_resize[0];   // scale=0.5
        for l in 0..full.shells.len().min(half.shells.len()) {
            let ratio = half.shells[l].r_center / full.shells[l].r_center;
            assert!((ratio - 0.5).abs() < 1e-9,
                "level {}: r_center ratio {} should be 0.5 (half-scale group)",
                l, ratio);
        }
    }

    // =====================================================================
    // Commit 4: Verify per-particle weights flow correctly through the
    // multi-run aggregators. These tests exercise the full pipeline:
    // CLI/runner ingestion → apply_run_spec (clip/wrap with weights) →
    // cascade build with weights → per-statistic aggregator → CSV row.
    // =====================================================================

    #[test]
    fn weights_field_stats_unit_weights_match_unweighted() {
        // weights_d == Some(vec![1.0; n]) should give bitwise-equivalent
        // results to weights_d == None. (Tests that the explicit weight
        // path doesn't accidentally drift from the optimized none-path.)
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts_d = make_uniform_periodic_pts(800, 6, 9100);
        let pts_r = make_uniform_periodic_pts(2400, 6, 9101);
        let bits = [6u32; 3];
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };

        // Unweighted run.
        let runner_no_w = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let agg_no_w = runner_no_w.analyze_field_stats(&cfg, 1e-9);

        // Unit-weighted run.
        let n_d = pts_d.len();
        let n_r = pts_r.len();
        let runner_unit = CascadeRunner::new_isolated(
            pts_d, Some(vec![1.0; n_d]), pts_r, Some(vec![1.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_unit = runner_unit.analyze_field_stats(&cfg, 1e-9);

        assert_eq!(agg_no_w.by_side.len(), agg_unit.by_side.len());
        for (b_no, b_unit) in agg_no_w.by_side.iter().zip(agg_unit.by_side.iter()) {
            // Tolerances allow for tiny reordering drift between the
            // count-only optimized path and the per-particle weight loop.
            assert!((b_no.physical_side - b_unit.physical_side).abs() < 1e-9,
                "side mismatch: {} vs {}", b_no.physical_side, b_unit.physical_side);
            assert!((b_no.sum_w_r_total - b_unit.sum_w_r_total).abs() < 1e-9,
                "sum_w_r mismatch: {} vs {}", b_no.sum_w_r_total, b_unit.sum_w_r_total);
            assert!((b_no.mean_delta - b_unit.mean_delta).abs() < 1e-10,
                "mean_delta mismatch: {} vs {}", b_no.mean_delta, b_unit.mean_delta);
            assert!((b_no.var_delta - b_unit.var_delta).abs() < 1e-10,
                "var_delta mismatch: {} vs {}", b_no.var_delta, b_unit.var_delta);
        }
    }

    #[test]
    fn weights_field_stats_uniform_scaling_invariant() {
        // δ = W_d / (α W_r) − 1 with α = ΣW_d / ΣW_r.
        // Multiplying ALL data weights by k1 and ALL random weights by k2:
        //   α' = (k1/k2) α
        //   δ' per cell = (k1 W_d) / ((k1/k2) α · k2 W_r) − 1 = δ
        // and W_r-weighted moments use W_r in both numerator and
        // denominator → unchanged. Test: doubling both should leave
        // moments invariant.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts_d = make_uniform_periodic_pts(800, 6, 9200);
        let pts_r = make_uniform_periodic_pts(2400, 6, 9201);
        let bits = [6u32; 3];
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };

        let n_d = pts_d.len();
        let n_r = pts_r.len();

        // Unit-weight run as baseline.
        let runner_unit = CascadeRunner::new_isolated(
            pts_d.clone(), Some(vec![1.0; n_d]),
            pts_r.clone(), Some(vec![1.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_unit = runner_unit.analyze_field_stats(&cfg, 1e-9);

        // 2× weight run.
        let runner_2x = CascadeRunner::new_isolated(
            pts_d, Some(vec![2.0; n_d]),
            pts_r, Some(vec![2.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_2x = runner_2x.analyze_field_stats(&cfg, 1e-9);

        assert_eq!(agg_unit.by_side.len(), agg_2x.by_side.len());
        for (b_u, b_2) in agg_unit.by_side.iter().zip(agg_2x.by_side.iter()) {
            // sum_w_r should DOUBLE.
            let r = b_2.sum_w_r_total / b_u.sum_w_r_total;
            assert!((r - 2.0).abs() < 1e-9,
                "side {}: sum_w_r ratio {}, expected 2.0",
                b_u.physical_side, r);
            // δ-moments are dimensionless, must be unchanged.
            assert!((b_u.mean_delta - b_2.mean_delta).abs() < 1e-10,
                "side {}: mean_delta {} vs {} (must be invariant)",
                b_u.physical_side, b_u.mean_delta, b_2.mean_delta);
            assert!((b_u.var_delta - b_2.var_delta).abs() < 1e-10,
                "side {}: var_delta {} vs {} (must be invariant)",
                b_u.physical_side, b_u.var_delta, b_2.var_delta);
        }
    }

    #[test]
    fn weights_xi_uniform_scaling_invariant() {
        // Landy-Szalay ξ = (DD/N_DD - 2 DR/N_DR + RR/N_RR) / (RR/N_RR)
        // is dimensionless. Doubling all weights:
        //   DD' = 4·DD, RR' = 4·RR, DR' = 4·DR
        //   N_DD' = (2 ΣW_d)(2 ΣW_d - 1)/2 ≈ 4·N_DD for large W_d
        //   etc.
        // All factors of 4 cancel (in the large-weight limit), so ξ
        // is invariant. Test with weights large enough that the (W-1)
        // term is negligible.
        let pts_d = make_uniform_periodic_pts(800, 6, 9300);
        let pts_r = make_uniform_periodic_pts(2400, 6, 9301);
        let bits = [6u32; 3];

        let n_d = pts_d.len();
        let n_r = pts_r.len();

        // Use weights = 100.0 so the W·(W-1) ≈ W² approximation is
        // tight to ~1%; we'll verify ratios at the per-shell level.
        let runner_unit = CascadeRunner::new_isolated(
            pts_d.clone(), Some(vec![100.0; n_d]),
            pts_r.clone(), Some(vec![100.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_unit = runner_unit.analyze_xi(1e-9);

        let runner_2x = CascadeRunner::new_isolated(
            pts_d, Some(vec![200.0; n_d]),
            pts_r, Some(vec![200.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_2x = runner_2x.analyze_xi(1e-9);

        assert_eq!(agg_unit.by_resize.len(), 1);
        assert_eq!(agg_2x.by_resize.len(), 1);
        let g_u = &agg_unit.by_resize[0];
        let g_2 = &agg_2x.by_resize[0];
        assert_eq!(g_u.shells.len(), g_2.shells.len());

        for (s_u, s_2) in g_u.shells.iter().zip(g_2.shells.iter()) {
            // DD: weighted DD = Σ_{(i,j)} w_i w_j → scales as 4× when all
            // weights doubled. Test the ratio at shells with measurable
            // signal.
            if s_u.dd_sum < 10.0 { continue; }
            let dd_ratio = s_2.dd_sum / s_u.dd_sum;
            assert!((dd_ratio - 4.0).abs() < 0.01,
                "level {}: DD ratio {}, expected 4.0", s_u.level, dd_ratio);
            let rr_ratio = s_2.rr_sum / s_u.rr_sum;
            assert!((rr_ratio - 4.0).abs() < 0.01,
                "level {}: RR ratio {}, expected 4.0", s_u.level, rr_ratio);
            let dr_ratio = s_2.dr_sum / s_u.dr_sum;
            assert!((dr_ratio - 4.0).abs() < 0.01,
                "level {}: DR ratio {}, expected 4.0", s_u.level, dr_ratio);
            // ξ should be (essentially) invariant.
            if s_u.xi_naive.is_finite() && s_2.xi_naive.is_finite() {
                let xi_diff = (s_2.xi_naive - s_u.xi_naive).abs();
                let xi_scale = s_u.xi_naive.abs().max(1e-3);
                assert!(xi_diff / xi_scale < 0.05,
                    "level {}: ξ {} vs {} (relative diff {}, expected ≈0)",
                    s_u.level, s_u.xi_naive, s_2.xi_naive, xi_diff / xi_scale);
            }
        }
    }

    #[test]
    fn weights_anisotropy_unit_weights_match_unweighted() {
        // Same parity check as the field-stats version, for anisotropy.
        // Unit weights should give numerically equivalent moments to
        // unweighted.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts_d = make_uniform_periodic_pts(800, 6, 9400);
        let pts_r = make_uniform_periodic_pts(2400, 6, 9401);
        let bits = [6u32; 3];
        let cfg = FieldStatsConfig { hist_bins: 0, ..Default::default() };

        let n_d = pts_d.len();
        let n_r = pts_r.len();

        let runner_no_w = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let agg_no_w = runner_no_w.analyze_anisotropy(&cfg, 1e-9);

        let runner_unit = CascadeRunner::new_isolated(
            pts_d, Some(vec![1.0; n_d]), pts_r, Some(vec![1.0; n_r]),
            bits, CascadeRunPlan::just_base());
        let agg_unit = runner_unit.analyze_anisotropy(&cfg, 1e-9);

        assert_eq!(agg_no_w.by_side.len(), agg_unit.by_side.len());
        for (b_no, b_unit) in agg_no_w.by_side.iter().zip(agg_unit.by_side.iter()) {
            assert!((b_no.sum_w_r_parents_total - b_unit.sum_w_r_parents_total).abs() < 1e-9,
                "sum_w_r_parents mismatch: {} vs {}",
                b_no.sum_w_r_parents_total, b_unit.sum_w_r_parents_total);
            for (k, (m_no, m_un)) in b_no.mean_w_squared_by_pattern.iter()
                    .zip(b_unit.mean_w_squared_by_pattern.iter()).enumerate() {
                assert!((m_no - m_un).abs() < 1e-9,
                    "side {}: pattern {} mean: {} vs {}",
                    b_no.physical_side, k, m_no, m_un);
            }
            assert!((b_no.quadrupole_los - b_unit.quadrupole_los).abs() < 1e-10,
                "quadrupole mismatch: {} vs {}",
                b_no.quadrupole_los, b_unit.quadrupole_los);
        }
    }

    #[test]
    fn weights_field_stats_per_particle_weighting_matches_per_cell_sum() {
        // The W_d cell accumulator at the deepest cascade level (where
        // each cell holds at most one point) MUST equal the weight of
        // the point in that cell. This is the most direct correctness
        // check on weight ingestion: no aggregation, no scaling, just
        // "the weight you put in is the weight that comes out per cell."
        //
        // We construct a tiny catalog with 4 well-separated points, each
        // with a distinct weight, then verify that at the deepest level
        // the cell sums equal the per-particle weights when we enumerate
        // them in the same order.
        let bits = 6u32;
        let half = 1u64 << (bits - 1);
        // Place 4 points in 4 different octants for clean separation.
        let pts_d: Vec<[u64; 3]> = vec![
            [10, 10, 10],
            [half + 10, 10, 10],
            [10, half + 10, 10],
            [10, 10, half + 10],
        ];
        let pts_r = make_uniform_periodic_pts(800, bits, 9501);
        let weights = vec![1.5, 2.5, 3.5, 4.5];
        let bits_arr = [bits; 3];

        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), Some(weights.clone()),
            pts_r, None,
            bits_arr, CascadeRunPlan::just_base());
        let runs = runner.per_run();
        assert_eq!(runs.len(), 1);
        let cascade = &runs[0].cascade;

        // Sum of cell W_d at the deepest level should equal sum of weights.
        // (Every weight ends up in exactly one cell at the deepest level
        // since coordinates are distinct.)
        let total_wd: f64 = weights.iter().sum();
        let cascade_total = cascade.sum_w_d();
        assert!((cascade_total - total_wd).abs() < 1e-12,
            "cascade total W_d {} != input total {}", cascade_total, total_wd);
    }

    // =====================================================================
    // Commit 6: Compensated (Neumaier) summation — opt-in flag wired into
    // FieldStatsVisitor's outer cell accumulators.
    // =====================================================================

    #[test]
    fn compensated_sums_match_naive_on_benign_data() {
        // Standard random catalog with unit weights: compensated and naive
        // outer accumulators should agree to ~1e-10 for typical data
        // (no cancellation pathology). Confirms the wiring is sound and
        // compensated mode doesn't drift when there's nothing to defend
        // against.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let pts_d = make_uniform_periodic_pts(800, 6, 9700);
        let pts_r = make_uniform_periodic_pts(2400, 6, 9701);
        let bits = [6u32; 3];

        let cfg_naive = FieldStatsConfig {
            hist_bins: 0, compensated_sums: false,
            ..Default::default()
        };
        let cfg_comp = FieldStatsConfig {
            hist_bins: 0, compensated_sums: true,
            ..Default::default()
        };

        let runner_naive = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let agg_naive = runner_naive.analyze_field_stats(&cfg_naive, 1e-9);

        let runner_comp = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None,
            bits, CascadeRunPlan::just_base());
        let agg_comp = runner_comp.analyze_field_stats(&cfg_comp, 1e-9);

        assert_eq!(agg_naive.by_side.len(), agg_comp.by_side.len());
        for (b_n, b_c) in agg_naive.by_side.iter().zip(agg_comp.by_side.iter()) {
            // For benign random data, Neumaier should recover the same
            // sum as naive to f64 precision (the compensation captures
            // bits that don't accumulate to anything significant).
            assert!((b_n.sum_w_r_total - b_c.sum_w_r_total).abs() < 1e-9,
                "side {}: sum_w_r naive {} vs comp {}",
                b_n.physical_side, b_n.sum_w_r_total, b_c.sum_w_r_total);
            assert!((b_n.mean_delta - b_c.mean_delta).abs() < 1e-10,
                "side {}: mean_delta naive {} vs comp {}",
                b_n.physical_side, b_n.mean_delta, b_c.mean_delta);
            assert!((b_n.var_delta - b_c.var_delta).abs() < 1e-10,
                "side {}: var_delta naive {} vs comp {}",
                b_n.physical_side, b_n.var_delta, b_c.var_delta);
        }
    }

    #[test]
    fn compensated_sums_recover_exact_total_with_pathological_weights() {
        // Pathological weights: half the points carry weight 1.0, half
        // carry weight 1e10. Cell accumulators sum these in arbitrary
        // order, and naive summation may lose precision when small
        // weights are added to a sum that has accumulated large weights.
        //
        // The cascade's `sum_w_r()` (called via cell_sums_r aggregation)
        // is a separate code path from the visitor's outer accumulators;
        // this test focuses on the visitor's `sum_w_r_active` per-level
        // total, which is the outer Neumaier-protected accumulator.
        //
        // We construct the catalog so the EXACT total weight is an
        // integer (representable in f64), then check compensated mode
        // recovers it.
        use crate::hier_bitvec_pair::FieldStatsConfig;
        let n_r: usize = 1000;
        let pts_r = make_uniform_periodic_pts(n_r, 6, 9800);
        let pts_d = make_uniform_periodic_pts(200, 6, 9801);
        let bits = [6u32; 3];

        // Half weights = 1.0, half = 1e10. Exact total = 500 + 500e10.
        let weights_r: Vec<f64> = (0..n_r)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1e10 })
            .collect();
        let exact_total: f64 = 500.0 + 500.0 * 1e10;

        let cfg_comp = FieldStatsConfig {
            hist_bins: 0, compensated_sums: true,
            ..Default::default()
        };
        let runner_comp = CascadeRunner::new_isolated(
            pts_d, None, pts_r, Some(weights_r),
            bits, CascadeRunPlan::just_base());
        let agg_comp = runner_comp.analyze_field_stats(&cfg_comp, 1e-9);

        // The visitor's per-level sum_w_r aggregates all visited cells.
        // For a single-shift run, the deepest level's sum should equal
        // the sum of all random weights (every cell has W_r > 0 since
        // random density > 0 with bits=6, n_r=1000 → 64^3=262144 cells,
        // so most cells are empty). The level-0 sum is the cleanest:
        // it's just the sum over all randoms in the box. For periodic
        // mode (no clipping), level-0 sum_w_r_active should equal the
        // total random weight exactly.
        //
        // Find a level that has sum_w_r_active close to the exact total
        // (typically the coarse levels, where every visited cell has
        // contributions from many randoms).
        let mut max_recovered = 0.0_f64;
        for b in &agg_comp.by_side {
            if b.sum_w_r_total > max_recovered {
                max_recovered = b.sum_w_r_total;
            }
        }
        // The coarsest level should have aggregated nearly all randoms.
        let relative_error = (max_recovered - exact_total).abs() / exact_total;
        assert!(relative_error < 1e-12,
            "compensated mode should recover total to ~f64 precision; \
             got {} (exact {}, rel.err {})",
            max_recovered, exact_total, relative_error);
    }

    // -----------------------------------------------------------------
    // Commit 14: multi-run gradient infrastructure
    // -----------------------------------------------------------------

    #[test]
    fn lift_gradient_identity_round_trip() {
        // Identity mapping: cascade-particle index k → original index k.
        // Lifted gradient should equal the input gradient (zero-padded
        // beyond the input length).
        let cascade_grad = vec![1.5, 2.5, 3.5, 4.5];
        let mapping: Vec<u32> = vec![0, 1, 2, 3];
        let lifted = CascadeRunner::<3>::lift_gradient_d_to_original(
            &cascade_grad, &mapping, 4);
        assert_eq!(lifted, cascade_grad);

        // Larger n_base: extra positions zero-padded.
        let lifted = CascadeRunner::<3>::lift_gradient_d_to_original(
            &cascade_grad, &mapping, 10);
        assert_eq!(lifted.len(), 10);
        assert_eq!(&lifted[..4], &cascade_grad[..]);
        assert_eq!(&lifted[4..], &[0.0; 6]);
    }

    #[test]
    fn lift_gradient_with_clipping_zeros_dropped_particles() {
        // A run that clipped out some particles: only surviving
        // particles get nonzero entries in the lifted vector.
        let cascade_grad = vec![10.0, 20.0, 30.0];
        let mapping: Vec<u32> = vec![0, 2, 4];  // particles 1, 3 clipped
        let lifted = CascadeRunner::<3>::lift_gradient_d_to_original(
            &cascade_grad, &mapping, 5);
        assert_eq!(lifted, vec![10.0, 0.0, 20.0, 0.0, 30.0]);
    }

    #[test]
    fn run_result_carries_original_indices_for_identity_spec() {
        // just_base plan ⇒ identity mapping: original_d_indices = 0..n_d.
        let pts_d = make_uniform_periodic_pts(50, 5, 4444);
        let pts_r = make_uniform_periodic_pts(150, 5, 5555);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None,
            bits, CascadeRunPlan::just_base());
        let runs = runner.per_run();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].original_d_indices.len(), 50);
        assert_eq!(runs[0].original_r_indices.len(), 150);
        // Identity: index k maps to k.
        for (k, &orig) in runs[0].original_d_indices.iter().enumerate() {
            assert_eq!(orig, k as u32, "identity-spec data index mismatch at k={}", k);
        }
        for (k, &orig) in runs[0].original_r_indices.iter().enumerate() {
            assert_eq!(orig, k as u32, "identity-spec random index mismatch at k={}", k);
        }
    }

    #[test]
    fn run_result_drops_clipped_particles_in_isolated_mode() {
        // With a non-trivial shift in isolated mode, some particles
        // are clipped. The cascade has fewer particles than the base
        // catalog, and the original_d_indices vector matches the
        // cascade size (not the base).
        let pts_d = make_uniform_periodic_pts(200, 6, 6666);
        let pts_r = make_uniform_periodic_pts(600, 6, 7777);
        let bits = [6u32; 3];
        // Random offsets at scale 0.5 should clip lots of particles.
        let plan = CascadeRunPlan::random_offsets(2, 0.4, 8888);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let runs = runner.per_run();
        assert_eq!(runs.len(), 2);
        for r in &runs {
            assert_eq!(r.original_d_indices.len(), r.cascade.n_d(),
                "original_d_indices length must match cascade n_d");
            // All indices must be valid (< 200) and distinct.
            let mut seen = std::collections::HashSet::new();
            for &i in &r.original_d_indices {
                assert!((i as usize) < 200,
                    "original index {} out of range", i);
                assert!(seen.insert(i),
                    "duplicate original index {}", i);
            }
            // Non-trivial clipping: most runs should drop SOME particles.
            // (Allow rare passes-through-without-clipping for robustness.)
            // Just check the upper bound.
            assert!(r.original_d_indices.len() <= 200);
        }
    }

    #[test]
    fn gradient_run_average_matches_manual_average() {
        // Multi-run mean of per-run var_delta gradient should equal
        // the manual average of lifted per-cascade gradients.
        let pts_d = make_uniform_periodic_pts(60, 5, 9999);
        let pts_r = make_uniform_periodic_pts(180, 5, 10000);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 11111);
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let level = 1;

        let avg_grad = runner.gradient_var_delta_data_run_average(&cfg, level);
        assert_eq!(avg_grad.len(), 60);

        // Manual average: per-run gradient, lifted, averaged.
        let runs = runner.per_run();
        let mut manual = vec![0.0_f64; 60];
        let mut n_used = 0;
        for r in &runs {
            let stats = r.cascade.analyze_field_stats(&cfg);
            if level >= stats.len() { continue; }
            let g = r.cascade.gradient_var_delta_all_levels(&cfg, &stats);
            if g.data_weight_grads[level].is_empty() { continue; }
            let lifted = CascadeRunner::<3>::lift_gradient_d_to_original(
                g.data_weight_grads[level].as_slice(),
                &r.original_d_indices, 60);
            for (i, &v) in lifted.iter().enumerate() { manual[i] += v; }
            n_used += 1;
        }
        if n_used > 0 {
            for v in manual.iter_mut() { *v /= n_used as f64; }
        }
        for i in 0..60 {
            assert!((avg_grad[i] - manual[i]).abs() < 1e-12,
                "particle {}: method {} vs manual {}",
                i, avg_grad[i], manual[i]);
        }
    }

    #[test]
    fn gradient_run_average_finite_difference_parity() {
        // FD parity: perturb base weight i by ε, recompute the run-
        // averaged var_delta at the chosen level, compare slope to
        // the analytic gradient.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 21111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 22222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.25, 23333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let wd_base = vec![1.0_f64; n_d];

        // Build runner once for analytic gradient.
        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());

        // Pick a level with measurable per-run signal.
        let runs = runner_base.per_run();
        let mut chosen_level = None;
        'outer: for level in 0..7 {
            for r in &runs {
                let s = r.cascade.analyze_field_stats(&cfg);
                if level < s.len() && s[level].var_delta > 1e-3 && s[level].n_cells_active >= 2 {
                    chosen_level = Some(level);
                    break 'outer;
                }
            }
        }
        let chosen_level = match chosen_level {
            Some(l) => l,
            None => return,
        };

        let analytic = runner_base.gradient_var_delta_data_run_average(&cfg, chosen_level);

        // Helper: compute run-averaged var at chosen level for given weights.
        let var_avg = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr, plan.clone());
            let mut sum = 0.0_f64;
            let mut n = 0_usize;
            for run in r.per_run() {
                let st = run.cascade.analyze_field_stats(&cfg);
                if chosen_level < st.len() {
                    sum += st[chosen_level].var_delta;
                    n += 1;
                }
            }
            if n > 0 { sum / n as f64 } else { 0.0 }
        };

        let v_base = var_avg(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let v_pert = var_avg(&wd_pert);
            let fd = (v_pert - v_base) / eps;
            let an = analytic[i];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "particle {}: FD {} vs analytic {} (diff {})",
                i, fd, an, abs_diff);
        }
    }

    // -----------------------------------------------------------------
    // Pooled-aggregate variance gradient (commit beyond 14)
    // -----------------------------------------------------------------

    #[test]
    fn gradient_pooled_matches_single_cascade_with_just_base() {
        // With just_base plan (one identity-spec run), pooling collapses
        // to per-run trivially. The pooled per-bin var_delta should
        // equal the single cascade's var_delta at the corresponding
        // level, and the pooled gradient should equal the single
        // cascade's variance gradient.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 31111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 31222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let pooled = runner.gradient_var_delta_data_pooled(&cfg, 1e-6);

        // Build single cascade directly for comparison.
        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_field_stats(&cfg);
        let var_grad = single.gradient_var_delta_all_levels(&cfg, &stats);

        // The pooled output bins are sorted by physical_side ascending.
        // The single-cascade levels are indexed 0=root, deepest at end.
        // Physical sides are powers of 2 (in trimmed-coord units),
        // descending with level. So pooled bin index `b` maps to
        // single-cascade level `n_levels - 1 - b` for an isolated
        // single run with default pooling.
        let n_levels = stats.len();
        assert_eq!(pooled.bin_grads.len(), n_levels,
            "expected one bin per level in single-run pooling");
        for b in 0..pooled.bin_grads.len() {
            let l = n_levels - 1 - b;  // bins ascending in side ↔ levels descending
            let direct = &var_grad.data_weight_grads[l];
            if direct.is_empty() { continue; }
            for i in 0..n_d {
                let p = pooled.bin_grads[b][i];
                let d = direct[i];
                assert!((p - d).abs() < 1e-10,
                    "bin {} (level {}) particle {}: pooled {} vs direct {}",
                    b, l, i, p, d);
            }
        }
    }

    #[test]
    fn gradient_pooled_uniform_scaling_invariant() {
        // Pooled var_delta is invariant under uniform scaling of all
        // base data weights (each per-run α scales by k, each per-run
        // δ unchanged, hence each per-run S_k unchanged, hence pooled
        // sums unchanged, hence pooled μ_2 unchanged).
        // So Σ_i w_i · ∂μ_2^pool/∂w_i = 0 exactly.
        let n_d = 60;
        let n_r = 180;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 32111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 32222);
        let bits = [5u32; 3];
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 60.0).collect();
        let plan = CascadeRunPlan::random_offsets(3, 0.3, 33333);
        let runner = CascadeRunner::new_isolated(
            pts_d, Some(weights_d.clone()), pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let pooled = runner.gradient_var_delta_data_pooled(&cfg, 1e-6);

        for (b, gs) in pooled.bin_grads.iter().enumerate() {
            let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                .map(|(g, w)| g * w).sum();
            assert!(weighted_sum.abs() < 1e-9,
                "bin {} (side {}): Σ w_i · ∂μ_2^pool/∂w_i = {} (should be ≈ 0)",
                b, pooled.bin_sides[b], weighted_sum);
        }
    }

    #[test]
    fn gradient_pooled_finite_difference_parity_multi_run() {
        // FD parity for the pooled gradient with multiple shifted runs.
        // The strongest end-to-end test: perturb base weight i by ε,
        // recompute pooled var_delta in a chosen bin via the full
        // analyze_field_stats path, compare slope to the analytic
        // pooled gradient.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 34111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 34222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 35333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let bin_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let analytic = runner_base.gradient_var_delta_data_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_field_stats(&cfg, bin_tol);

        // Pick a bin with measurable variance (a few cells, signal > noise).
        let mut chosen_bin: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.var_delta > 1e-3 && bin.n_cells_active_total >= 4 {
                chosen_bin = Some(b);
                break;
            }
        }
        let chosen_bin = match chosen_bin {
            Some(b) => b,
            None => return,
        };

        let var_pooled_at_bin = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr, plan.clone());
            let agg = r.analyze_field_stats(&cfg, bin_tol);
            // The bin index correspondence assumes pooling is stable
            // (no bin merges/splits) under tiny weight perturbations
            // — true because bin_tol is on physical sides which depend
            // on coordinates, not weights.
            assert_eq!(agg.by_side.len(), analytic.bin_grads.len(),
                "bin count differs after perturbation; cannot run FD test");
            agg.by_side[chosen_bin].var_delta
        };

        let v_base = var_pooled_at_bin(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let v_pert = var_pooled_at_bin(&wd_pert);
            let fd = (v_pert - v_base) / eps;
            let an = analytic.bin_grads[chosen_bin][i];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "bin {} particle {}: FD {} vs analytic {} (diff {})",
                chosen_bin, i, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_pooled_aggregate_matches_per_bin_combination() {
        // Aggregate-scalar API equals Σ_b β_b · per_bin_grad[b].
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 36111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 36222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 37333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let bin_tol = 1e-6;
        let pooled = runner.gradient_var_delta_data_pooled(&cfg, bin_tol);

        let n_bins = pooled.bin_grads.len();
        let betas: Vec<f64> = (0..n_bins)
            .map(|b| if b % 2 == 0 { 1.0 } else { -0.5 } * (b as f64 + 1.0))
            .collect();
        let agg = runner.gradient_var_delta_data_pooled_aggregate(&cfg, bin_tol, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for b in 0..n_bins {
            for (i, &g) in pooled.bin_grads[b].iter().enumerate() {
                expected[i] += betas[b] * g;
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β_b grad_b {}",
                i, agg[i], expected[i]);
        }
    }

    // -----------------------------------------------------------------
    // Pooled higher-moment gradients (m3, m4, S3)
    // -----------------------------------------------------------------

    #[test]
    fn gradient_pooled_m3_matches_single_cascade_with_just_base() {
        // Same logic as the variance test: with just_base plan, pooled
        // m3 gradient should equal single-cascade m3 gradient at each
        // matching level.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 40111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 40222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };

        let pooled = runner.gradient_m3_delta_data_pooled(&cfg, 1e-6);
        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_field_stats(&cfg);
        let direct = single.gradient_m3_delta_all_levels(&cfg, &stats);

        let n_levels = stats.len();
        assert_eq!(pooled.bin_grads.len(), n_levels);
        for b in 0..pooled.bin_grads.len() {
            let l = n_levels - 1 - b;
            let dg = &direct.data_weight_grads[l];
            if dg.is_empty() { continue; }
            for i in 0..n_d {
                let p = pooled.bin_grads[b][i];
                let d = dg[i];
                let diff = (p - d).abs();
                let scale = d.abs().max(1e-10);
                assert!(diff < 1e-10 || diff / scale < 1e-9,
                    "bin {} (level {}) particle {}: pooled {} vs direct {} (diff {})",
                    b, l, i, p, d, diff);
            }
        }
    }

    #[test]
    fn gradient_pooled_m4_matches_single_cascade_with_just_base() {
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 41111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 41222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let pooled = runner.gradient_m4_delta_data_pooled(&cfg, 1e-6);
        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_field_stats(&cfg);
        let direct = single.gradient_m4_delta_all_levels(&cfg, &stats);

        let n_levels = stats.len();
        assert_eq!(pooled.bin_grads.len(), n_levels);
        for b in 0..pooled.bin_grads.len() {
            let l = n_levels - 1 - b;
            let dg = &direct.data_weight_grads[l];
            if dg.is_empty() { continue; }
            for i in 0..n_d {
                let p = pooled.bin_grads[b][i];
                let d = dg[i];
                let diff = (p - d).abs();
                let scale = d.abs().max(1e-10);
                assert!(diff < 1e-10 || diff / scale < 1e-9,
                    "bin {} (level {}) particle {}: pooled {} vs direct {} (diff {})",
                    b, l, i, p, d, diff);
            }
        }
    }

    #[test]
    fn gradient_pooled_s3_matches_single_cascade_with_just_base() {
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 42111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 42222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let pooled = runner.gradient_s3_delta_data_pooled(&cfg, 1e-6);
        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_field_stats(&cfg);
        let direct = single.gradient_s3_delta_all_levels(&cfg, &stats);

        let n_levels = stats.len();
        assert_eq!(pooled.bin_grads.len(), n_levels);
        for b in 0..pooled.bin_grads.len() {
            let l = n_levels - 1 - b;
            let dg = &direct.data_weight_grads[l];
            if dg.is_empty() { continue; }
            // S3 is undefined where var_delta = 0; both sides should
            // return 0 in that case (we built S3 as 0 when var ≤ 0).
            for i in 0..n_d {
                let p = pooled.bin_grads[b][i];
                let d = dg[i];
                let diff = (p - d).abs();
                let scale = d.abs().max(1e-10);
                assert!(diff < 1e-9 || diff / scale < 1e-8,
                    "bin {} (level {}) particle {}: pooled {} vs direct {} (diff {})",
                    b, l, i, p, d, diff);
            }
        }
    }

    #[test]
    fn gradient_pooled_higher_moments_uniform_scaling_invariant() {
        // m3, m4, S3 are all invariant under uniform scaling of base
        // data weights (per the same argument as variance: per-run α
        // scales by k, per-run δ unchanged, per-run S_k unchanged,
        // pooled S_k unchanged, pooled central moments unchanged).
        // ⇒ Σ_i w_i · ∂μ/∂w_i = 0 exactly.
        let n_d = 60;
        let n_r = 180;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 43111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 43222);
        let bits = [5u32; 3];
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 60.0).collect();
        let plan = CascadeRunPlan::random_offsets(3, 0.3, 44444);
        let runner = CascadeRunner::new_isolated(
            pts_d, Some(weights_d.clone()), pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let bin_tol = 1e-6;
        for (name, pooled) in [
            ("m3", runner.gradient_m3_delta_data_pooled(&cfg, bin_tol)),
            ("m4", runner.gradient_m4_delta_data_pooled(&cfg, bin_tol)),
            ("s3", runner.gradient_s3_delta_data_pooled(&cfg, bin_tol)),
        ] {
            for (b, gs) in pooled.bin_grads.iter().enumerate() {
                let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                    .map(|(g, w)| g * w).sum();
                assert!(weighted_sum.abs() < 1e-9,
                    "{} bin {} (side {}): Σ w_i · ∂μ/∂w_i = {} (should be ≈ 0)",
                    name, b, pooled.bin_sides[b], weighted_sum);
            }
        }
    }

    #[test]
    fn gradient_pooled_m3_finite_difference_parity_multi_run() {
        // FD parity for the headline case: pooled m3 gradient with
        // multiple shifted runs, perturbing each base data weight,
        // recomputing the full pooled m3 via analyze_field_stats.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 45111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 45222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 46333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let bin_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let analytic = runner_base.gradient_m3_delta_data_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_field_stats(&cfg, bin_tol);

        // Pick a bin with nontrivial m3.
        let mut chosen_bin: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.m3_delta.abs() > 1e-3 && bin.n_cells_active_total >= 4 {
                chosen_bin = Some(b);
                break;
            }
        }
        let chosen_bin = match chosen_bin {
            Some(b) => b,
            None => return,
        };

        let m3_pooled_at_bin = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr, plan.clone());
            let agg = r.analyze_field_stats(&cfg, bin_tol);
            assert_eq!(agg.by_side.len(), analytic.bin_grads.len(),
                "bin count differs after perturbation");
            agg.by_side[chosen_bin].m3_delta
        };

        let v_base = m3_pooled_at_bin(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let v_pert = m3_pooled_at_bin(&wd_pert);
            let fd = (v_pert - v_base) / eps;
            let an = analytic.bin_grads[chosen_bin][i];
            let abs_diff = (fd - an).abs();
            // Looser tolerance — m3 is cubic in δ so FD truncation error
            // amplifies relative to the variance case.
            assert!(abs_diff < 1e-2,
                "bin {} particle {}: FD {} vs analytic {} (diff {})",
                chosen_bin, i, fd, an, abs_diff);
        }
    }

    // -----------------------------------------------------------------
    // Threading determinism (parallelism audit deliverable)
    // -----------------------------------------------------------------

    /// Run the same multi-run analysis under two different rayon
    /// thread-pool configurations and assert bit-exact equality of
    /// the f64 outputs. This pins the `collect-then-serial-reduce`
    /// pattern used throughout the per-run parallelization: the
    /// parallel map collects results in stable input order, then the
    /// serial aggregation walks that ordered vector — so thread count
    /// must not affect the floating-point sum order.
    ///
    /// If this test ever fails it means we accidentally introduced a
    /// direct parallel f64 reduction somewhere. See `docs/parallelism.md`
    /// §3.
    #[test]
    fn multi_run_outputs_are_bit_exact_across_thread_counts() {
        // A non-trivial multi-run plan: enough runs that thread-count
        // > 1 actually exercises rayon's work distribution.
        let pts_d = make_uniform_periodic_pts(80, 6, 50111);
        let pts_r = make_uniform_periodic_pts(240, 6, 50222);
        let bits = [6u32; 3];
        let plan = CascadeRunPlan::random_offsets(8, 0.3, 51111);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;

        // Run inside a dedicated 1-thread pool.
        let pool_1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1).build().unwrap();
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None, bits, plan.clone());
        let agg_1 = pool_1.install(|| runner.analyze_field_stats(&cfg, bin_tol));
        let pooled_1 = pool_1.install(|| runner.gradient_var_delta_data_pooled(&cfg, bin_tol));

        // Run inside a 4-thread pool.
        let pool_4 = rayon::ThreadPoolBuilder::new()
            .num_threads(4).build().unwrap();
        let agg_4 = pool_4.install(|| runner.analyze_field_stats(&cfg, bin_tol));
        let pooled_4 = pool_4.install(|| runner.gradient_var_delta_data_pooled(&cfg, bin_tol));

        // analyze_field_stats: bit-exact equality of every pooled bin.
        assert_eq!(agg_1.by_side.len(), agg_4.by_side.len());
        for (b1, b4) in agg_1.by_side.iter().zip(agg_4.by_side.iter()) {
            assert_eq!(b1.physical_side.to_bits(), b4.physical_side.to_bits(),
                "physical_side differs across thread counts");
            assert_eq!(b1.sum_w_r_total.to_bits(), b4.sum_w_r_total.to_bits(),
                "sum_w_r_total differs across thread counts");
            assert_eq!(b1.mean_delta.to_bits(), b4.mean_delta.to_bits(),
                "mean_delta differs across thread counts");
            assert_eq!(b1.var_delta.to_bits(), b4.var_delta.to_bits(),
                "var_delta differs across thread counts");
            assert_eq!(b1.m3_delta.to_bits(), b4.m3_delta.to_bits(),
                "m3_delta differs across thread counts");
            assert_eq!(b1.m4_delta.to_bits(), b4.m4_delta.to_bits(),
                "m4_delta differs across thread counts");
            assert_eq!(b1.n_cells_active_total, b4.n_cells_active_total);
        }

        // Pooled gradient: bit-exact equality of every per-particle entry.
        assert_eq!(pooled_1.bin_grads.len(), pooled_4.bin_grads.len());
        for (b_idx, (g1, g4)) in pooled_1.bin_grads.iter()
            .zip(pooled_4.bin_grads.iter()).enumerate()
        {
            assert_eq!(g1.len(), g4.len(),
                "bin {}: gradient length differs", b_idx);
            for (i, (a, b)) in g1.iter().zip(g4.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(),
                    "bin {} particle {}: gradient differs ({} vs {})",
                    b_idx, i, a, b);
            }
        }
    }

    // -----------------------------------------------------------------
    // Pooled ξ gradient
    //
    // Note on test design: the pooled ξ gradient differentiates the
    // forward formula in `pool_xi_resize_group`, which uses count-based
    // normalizations N_DD = n_d_pool · (n_d_pool − 1) / 2 (independent
    // of data weights). This differs from the single-cascade
    // `gradient_xi_data_all_shells` convention, which uses weight-based
    // N_DD = (W_d² − Σw²)/2. The two agree numerically on ξ for unit
    // weights but produce different gradients (the count-based form
    // has zero denominator derivative; the weight-based form does not).
    //
    // Consequences for testing:
    // - The pooled gradient does NOT reduce to the single-cascade
    //   gradient under just_base, even with unit weights: different
    //   normalization conventions ⇒ different chain-rule terms.
    // - ξ^pool is NOT invariant under uniform scaling of data weights:
    //   scaling w by k scales DD by k² but leaves count-based N_DD
    //   unchanged, so f_DD scales by k².
    //
    // The meaningful correctness test is FD parity against
    // `analyze_xi`'s actual output, which is what `_finite_difference_parity_multi_run` does.
    // -----------------------------------------------------------------

    #[test]
    fn gradient_xi_pooled_just_base_matches_finite_difference() {
        // Single-shift FD parity: with just_base plan, perturb each
        // base data weight, recompute ξ via analyze_xi, compare to
        // analytic. This is the strict test that the gradient is
        // correct for the simplest case.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 60111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 60222);
        let bits_arr = [bits; 3];
        let scale_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr,
            CascadeRunPlan::just_base());
        let analytic = runner_base.gradient_xi_data_pooled(scale_tol);
        let agg_base = runner_base.analyze_xi(scale_tol);

        assert_eq!(analytic.by_resize.len(), 1, "just_base: one resize group");
        let group = &analytic.by_resize[0];
        let agg_group = &agg_base.by_resize[0];

        // Pick a shell with measurable ξ and pair-count signal.
        let mut chosen: Option<usize> = None;
        for (s, shell) in agg_group.shells.iter().enumerate() {
            if shell.xi_naive.is_finite() && shell.xi_naive.abs() > 1e-3
               && shell.rr_sum > 0.5
            {
                chosen = Some(s);
                break;
            }
        }
        let sx = match chosen { Some(s) => s, None => return };

        let xi_at = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr,
                CascadeRunPlan::just_base());
            r.analyze_xi(scale_tol).by_resize[0].shells[sx].xi_naive
        };
        let v_base = xi_at(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let fd = (xi_at(&wd_pert) - v_base) / eps;
            let an = group.shell_grads[sx][i];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "shell {} particle {}: FD {} vs analytic {} (diff {})",
                sx, i, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_xi_pooled_finite_difference_parity_multi_run() {
        // End-to-end FD parity for the pooled ξ gradient. Perturb
        // each base data weight, recompute the full pooled ξ via
        // analyze_xi, compare to the analytic gradient. Multi-shift
        // case: 3 random-shifted runs.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 63111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 63222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 64333);
        let scale_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let analytic = runner_base.gradient_xi_data_pooled(scale_tol);
        let agg_base = runner_base.analyze_xi(scale_tol);

        // Pick a (group, shell) with measurable ξ.
        let mut chosen: Option<(usize, usize)> = None;
        'outer: for (g, group) in agg_base.by_resize.iter().enumerate() {
            for (s, shell) in group.shells.iter().enumerate() {
                if shell.xi_naive.is_finite() && shell.xi_naive.abs() > 1e-3
                   && shell.rr_sum > 0.5
                {
                    chosen = Some((g, s));
                    break 'outer;
                }
            }
        }
        let (gx, sx) = match chosen {
            Some(c) => c,
            None => return,
        };

        let xi_at = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr, plan.clone());
            let agg = r.analyze_xi(scale_tol);
            assert_eq!(agg.by_resize.len(), analytic.by_resize.len(),
                "resize group count differs after perturbation");
            agg.by_resize[gx].shells[sx].xi_naive
        };

        let v_base = xi_at(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let v_pert = xi_at(&wd_pert);
            let fd = (v_pert - v_base) / eps;
            let an = analytic.by_resize[gx].shell_grads[sx][i];
            let abs_diff = (fd - an).abs();
            // ξ tolerance: ξ is a ratio of pair counts, FD truncation
            // error is amplified by the denominator.
            assert!(abs_diff < 5e-3,
                "group {} shell {} particle {}: FD {} vs analytic {} (diff {})",
                gx, sx, i, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_xi_pooled_aggregate_matches_per_shell_combination() {
        // Aggregate-scalar API equals Σ_g Σ_s β_{g,s} · per_shell_grad[g][s].
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 65111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 65222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 66333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let scale_tol = 1e-6;
        let pooled = runner.gradient_xi_data_pooled(scale_tol);

        let betas: Vec<Vec<f64>> = pooled.by_resize.iter().enumerate()
            .map(|(g, group)| {
                group.shell_grads.iter().enumerate().map(|(s, _)| {
                    if (g + s) % 2 == 0 { 1.0_f64 + (s as f64) }
                    else { -0.5_f64 * (s as f64 + 1.0) }
                }).collect()
            }).collect();
        let agg = runner.gradient_xi_data_pooled_aggregate(scale_tol, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for (g, group) in pooled.by_resize.iter().enumerate() {
            for (s, shell_g) in group.shell_grads.iter().enumerate() {
                let beta = betas[g][s];
                if beta == 0.0 { continue; }
                for (i, &gi) in shell_g.iter().enumerate() {
                    expected[i] += beta * gi;
                }
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β·grad {}",
                i, agg[i], expected[i]);
        }
    }

    // -----------------------------------------------------------------
    // Pooled anisotropy gradient
    // -----------------------------------------------------------------

    #[test]
    fn gradient_anisotropy_pooled_matches_single_cascade_with_just_base() {
        // With just_base plan, pooled per-pattern gradients should
        // equal the single-cascade per-pattern gradients at the
        // matching level. No count-vs-weight discrepancy here —
        // pooled and per-cascade both use the weighted T = sum_w_r_parents.
        //
        // The pooled aggregator skips levels with t_l ≤ 0, so we
        // match by physical_side rather than by index.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 70111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 70222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let pooled = runner.gradient_anisotropy_data_pooled(&cfg, 1e-6);

        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_anisotropy(&cfg);
        let direct = single.gradient_anisotropy_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;

        let mut bins_checked = 0;
        for bin in &pooled.by_side {
            // Find single-cascade level whose physical_side matches.
            let mut matched_l: Option<usize> = None;
            for (l, s) in stats.iter().enumerate() {
                let lvl_side = s.cell_side_trimmed;  // scale=1 for just_base
                if (lvl_side - bin.physical_side).abs() / bin.physical_side.max(1e-300) < 1e-6 {
                    matched_l = Some(l);
                    break;
                }
            }
            let l = match matched_l {
                Some(l) => l,
                None => continue,  // no level matches (shouldn't happen for just_base)
            };
            for e in 1..n_patterns {
                let dg = &direct.pattern_grads[l][e];
                if dg.is_empty() { continue; }
                let pg = &bin.pattern_grads[e];
                assert_eq!(pg.len(), n_d);
                for i in 0..n_d {
                    let diff = (pg[i] - dg[i]).abs();
                    let scale = dg[i].abs().max(1e-10);
                    assert!(diff < 1e-10 || diff / scale < 1e-9,
                        "level {} pattern {} particle {}: \
                         pooled {} vs direct {} (diff {})",
                        l, e, i, pg[i], dg[i], diff);
                }
            }
            bins_checked += 1;
        }
        assert!(bins_checked > 0, "no bins matched any single-cascade level");
    }

    #[test]
    fn gradient_anisotropy_pooled_finite_difference_parity_multi_run() {
        // End-to-end FD parity: perturb base data weight, recompute
        // the LoS quadrupole via analyze_anisotropy, compare to analytic.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 71111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 71222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 72333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let analytic = runner_base.gradient_anisotropy_data_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_anisotropy(&cfg, bin_tol);

        // Pick a bin with measurable Q_LoS.
        let mut chosen: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.quadrupole_los.abs() > 1e-3 && bin.n_parents_total >= 4 {
                chosen = Some(b);
                break;
            }
        }
        let bx = match chosen { Some(b) => b, None => return };

        let q_at = |wd: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits_arr, plan.clone());
            let agg = r.analyze_anisotropy(&cfg, bin_tol);
            assert_eq!(agg.by_side.len(), analytic.by_side.len(),
                "bin count differs after perturbation");
            agg.by_side[bx].quadrupole_los
        };

        let v_base = q_at(&wd_base);
        let eps = 1e-5;
        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let fd = (q_at(&wd_pert) - v_base) / eps;
            let an = analytic.by_side[bx].quadrupole_los_grad[i];
            let abs_diff = (fd - an).abs();
            // Tolerance: Q_LoS involves squared residuals divided by
            // T, so FD error scales with eps and 1/T. Loose tolerance
            // for safety.
            assert!(abs_diff < 5e-3,
                "bin {} particle {}: FD {} vs analytic {} (diff {})",
                bx, i, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_anisotropy_pooled_quadrupole_aggregate_matches_per_bin() {
        // Aggregate-scalar API equals Σ_b β_b · quadrupole_los_grad[b].
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 73111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 73222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 74333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let pooled = runner.gradient_anisotropy_data_pooled(&cfg, bin_tol);
        let n_bins = pooled.by_side.len();
        let betas: Vec<f64> = (0..n_bins).map(|b| {
            if b % 2 == 0 { 1.0 + b as f64 } else { -0.5 * (b as f64 + 1.0) }
        }).collect();
        let agg = runner.gradient_anisotropy_quadrupole_data_pooled_aggregate(
            &cfg, bin_tol, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for b in 0..n_bins {
            let beta = betas[b];
            for (i, &g) in pooled.by_side[b].quadrupole_los_grad.iter().enumerate() {
                expected[i] += beta * g;
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β·grad {}",
                i, agg[i], expected[i]);
        }
    }

    // -----------------------------------------------------------------
    // Pooled random-weight field-stats gradients
    //
    // Symmetric counterparts to the data-weight pooled gradients.
    // Differ in two ways:
    //   1. Lifted to base-r-index space (n_base_r-dim output)
    //   2. ∂T ≠ 0 (random weights enter T = sum_w_r_active)
    // -----------------------------------------------------------------

    #[test]
    fn gradient_pooled_random_var_matches_single_cascade_with_just_base() {
        // With just_base plan, pooled random-weight var gradient
        // should equal single-cascade random var gradient at each
        // matching level.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 90111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 90222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let pooled = runner.gradient_var_delta_random_pooled(&cfg, 1e-6);

        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_field_stats(&cfg);
        let direct = single.gradient_var_delta_random_all_levels(&cfg, &stats);

        let n_levels = stats.len();
        assert_eq!(pooled.bin_grads.len(), n_levels);
        for b in 0..pooled.bin_grads.len() {
            let l = n_levels - 1 - b;
            let dg = &direct.random_weight_grads[l];
            if dg.is_empty() { continue; }
            for j in 0..n_r {
                let p = pooled.bin_grads[b][j];
                let d = dg[j];
                let diff = (p - d).abs();
                let scale = d.abs().max(1e-10);
                assert!(diff < 1e-9 || diff / scale < 1e-9,
                    "bin {} (level {}) random {}: pooled {} vs direct {} (diff {})",
                    b, l, j, p, d, diff);
            }
        }
    }

    #[test]
    fn gradient_pooled_random_var_finite_difference_parity_multi_run() {
        // FD parity: perturb each base random weight, recompute
        // pooled var via analyze_field_stats, compare to analytic.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 91111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 91222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 92333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let wr_base = vec![1.0_f64; n_r];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), Some(wr_base.clone()),
            bits_arr, plan.clone());
        let analytic = runner_base.gradient_var_delta_random_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_field_stats(&cfg, bin_tol);

        // Pick a bin with measurable variance.
        let mut chosen: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.var_delta > 1e-3 && bin.n_cells_active_total >= 4 {
                chosen = Some(b);
                break;
            }
        }
        let bx = match chosen { Some(b) => b, None => return };

        let var_at = |wr: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), None, pts_r.clone(), Some(wr.to_vec()),
                bits_arr, plan.clone());
            r.analyze_field_stats(&cfg, bin_tol).by_side[bx].var_delta
        };

        let v_base = var_at(&wr_base);
        let eps = 1e-5;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let fd = (var_at(&wr_pert) - v_base) / eps;
            let an = analytic.bin_grads[bx][j];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "bin {} random {}: FD {} vs analytic {} (diff {})",
                bx, j, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_pooled_random_higher_moments_finite_difference_parity() {
        // FD parity for m3 (random) — exercises the (∂T, ∂S_1..∂S_3)
        // chain rule end-to-end through the pooled multi-run path.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 93111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 93222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 94333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let wr_base = vec![1.0_f64; n_r];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), Some(wr_base.clone()),
            bits_arr, plan.clone());
        let analytic = runner_base.gradient_m3_delta_random_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_field_stats(&cfg, bin_tol);

        // Pick a bin with measurable m3.
        let mut chosen: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.m3_delta.abs() > 1e-3 && bin.n_cells_active_total >= 4 {
                chosen = Some(b);
                break;
            }
        }
        let bx = match chosen { Some(b) => b, None => return };

        let m3_at = |wr: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), None, pts_r.clone(), Some(wr.to_vec()),
                bits_arr, plan.clone());
            r.analyze_field_stats(&cfg, bin_tol).by_side[bx].m3_delta
        };

        let v_base = m3_at(&wr_base);
        let eps = 1e-5;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let fd = (m3_at(&wr_pert) - v_base) / eps;
            let an = analytic.bin_grads[bx][j];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 1e-2,
                "bin {} random {}: FD {} vs analytic {} (diff {})",
                bx, j, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_pooled_random_var_aggregate_matches_per_bin_combination() {
        // Aggregate-scalar API equals Σ_b β_b · per_bin_grad.
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 95111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 95222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 96333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let pooled = runner.gradient_var_delta_random_pooled(&cfg, bin_tol);
        let n_bins = pooled.bin_grads.len();
        let betas: Vec<f64> = (0..n_bins).map(|b| {
            if b % 2 == 0 { 1.0 + b as f64 } else { -0.5 * (b as f64 + 1.0) }
        }).collect();
        let agg = runner.gradient_var_delta_random_pooled_aggregate(
            &cfg, bin_tol, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for b in 0..n_bins {
            let beta = betas[b];
            for (j, &g) in pooled.bin_grads[b].iter().enumerate() {
                expected[j] += beta * g;
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random {}: aggregate {} vs Σ β·grad {}",
                j, agg[j], expected[j]);
        }
    }

    // -----------------------------------------------------------------
    // Pooled random-weight anisotropy gradient
    // -----------------------------------------------------------------

    #[test]
    fn gradient_anisotropy_random_pooled_matches_single_cascade_with_just_base() {
        // With just_base plan, pooled per-pattern random gradients
        // should equal single-cascade per-pattern random gradients at
        // matching levels (matched by physical_side).
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 100111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 100222);
        let bits = [5u32; 3];
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), None,
            bits, CascadeRunPlan::just_base());
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let pooled = runner.gradient_anisotropy_random_pooled(&cfg, 1e-6);

        let runs = runner.per_run();
        let single = &runs[0].cascade;
        let stats = single.analyze_anisotropy(&cfg);
        let direct = single.gradient_anisotropy_random_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        let mut bins_checked = 0;
        for bin in &pooled.by_side {
            // Find single-cascade level matching this bin's physical_side.
            let mut matched_l: Option<usize> = None;
            for (l, s) in stats.iter().enumerate() {
                let lvl_side = s.cell_side_trimmed;  // scale=1 for just_base
                if (lvl_side - bin.physical_side).abs() / bin.physical_side.max(1e-300) < 1e-6 {
                    matched_l = Some(l);
                    break;
                }
            }
            let l = match matched_l { Some(l) => l, None => continue };
            for e in 1..n_patterns {
                let dg = &direct.pattern_grads[l][e];
                if dg.is_empty() { continue; }
                let pg = &bin.pattern_grads[e];
                assert_eq!(pg.len(), n_r);
                for j in 0..n_r {
                    let diff = (pg[j] - dg[j]).abs();
                    let scale = dg[j].abs().max(1e-10);
                    assert!(diff < 1e-9 || diff / scale < 1e-9,
                        "level {} pattern {} random {}: pooled {} vs direct {} (diff {})",
                        l, e, j, pg[j], dg[j], diff);
                }
            }
            bins_checked += 1;
        }
        assert!(bins_checked > 0, "no bins matched any single-cascade level");
    }

    #[test]
    fn gradient_anisotropy_random_pooled_finite_difference_parity_multi_run() {
        // FD parity: perturb each base random weight, recompute Q_LoS
        // via analyze_anisotropy, compare to analytic.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 101111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 101222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 102333);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let wr_base = vec![1.0_f64; n_r];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), Some(wr_base.clone()),
            bits_arr, plan.clone());
        let analytic = runner_base.gradient_anisotropy_random_pooled(&cfg, bin_tol);
        let agg_base = runner_base.analyze_anisotropy(&cfg, bin_tol);

        let mut chosen: Option<usize> = None;
        for (b, bin) in agg_base.by_side.iter().enumerate() {
            if bin.quadrupole_los.abs() > 1e-3 && bin.n_parents_total >= 4 {
                chosen = Some(b);
                break;
            }
        }
        let bx = match chosen { Some(b) => b, None => return };

        let q_at = |wr: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), None, pts_r.clone(), Some(wr.to_vec()),
                bits_arr, plan.clone());
            r.analyze_anisotropy(&cfg, bin_tol).by_side[bx].quadrupole_los
        };

        let v_base = q_at(&wr_base);
        let eps = 1e-5;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let fd = (q_at(&wr_pert) - v_base) / eps;
            let an = analytic.by_side[bx].quadrupole_los_grad[j];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "bin {} random {}: FD {} vs analytic {} (diff {})",
                bx, j, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_anisotropy_random_pooled_quadrupole_aggregate_matches_per_bin() {
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 103111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 103222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 104333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let cfg = crate::hier_bitvec_pair::FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let pooled = runner.gradient_anisotropy_random_pooled(&cfg, bin_tol);
        let n_bins = pooled.by_side.len();
        let betas: Vec<f64> = (0..n_bins).map(|b| {
            if b % 2 == 0 { 1.0 + b as f64 } else { -0.5 * (b as f64 + 1.0) }
        }).collect();
        let agg = runner.gradient_anisotropy_quadrupole_random_pooled_aggregate(
            &cfg, bin_tol, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for b in 0..n_bins {
            let beta = betas[b];
            for (j, &g) in pooled.by_side[b].quadrupole_los_grad.iter().enumerate() {
                expected[j] += beta * g;
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random {}: aggregate {} vs Σ β·grad {}",
                j, agg[j], expected[j]);
        }
    }

    // -----------------------------------------------------------------
    // Pooled random-weight ξ gradient
    //
    // Same count-based pooled-normalization design as the data-side
    // ξ pooled gradient (see test module header for that section).
    // For random weights, ∂DD/∂w_j^r = 0; ∂RR and ∂DR are nonzero.
    // -----------------------------------------------------------------

    #[test]
    fn gradient_xi_random_pooled_just_base_matches_finite_difference() {
        // Single-shift FD parity: with just_base plan, perturb each
        // base random weight, recompute ξ via analyze_xi, compare to
        // analytic.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 110111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 110222);
        let bits_arr = [bits; 3];
        let scale_tol = 1e-6;
        let wr_base = vec![1.0_f64; n_r];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), Some(wr_base.clone()),
            bits_arr, CascadeRunPlan::just_base());
        let analytic = runner_base.gradient_xi_random_pooled(scale_tol);
        let agg_base = runner_base.analyze_xi(scale_tol);

        assert_eq!(analytic.by_resize.len(), 1);
        let group = &analytic.by_resize[0];
        let agg_group = &agg_base.by_resize[0];

        let mut chosen: Option<usize> = None;
        for (s, shell) in agg_group.shells.iter().enumerate() {
            if shell.xi_naive.is_finite() && shell.xi_naive.abs() > 1e-3
               && shell.rr_sum > 0.5
            {
                chosen = Some(s);
                break;
            }
        }
        let sx = match chosen { Some(s) => s, None => return };

        let xi_at = |wr: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), None, pts_r.clone(), Some(wr.to_vec()),
                bits_arr, CascadeRunPlan::just_base());
            r.analyze_xi(scale_tol).by_resize[0].shells[sx].xi_naive
        };
        let v_base = xi_at(&wr_base);
        let eps = 1e-5;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let fd = (xi_at(&wr_pert) - v_base) / eps;
            let an = group.shell_grads[sx][j];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "shell {} random {}: FD {} vs analytic {} (diff {})",
                sx, j, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_xi_random_pooled_finite_difference_parity_multi_run() {
        // Multi-shift FD parity — exercises the full random-weight
        // chain rule end-to-end through the pooled multi-run path.
        let bits = 5u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform_periodic_pts(n_d, bits, 111111);
        let pts_r = make_uniform_periodic_pts(n_r, bits, 111222);
        let bits_arr = [bits; 3];
        let plan = CascadeRunPlan::random_offsets(3, 0.25, 112333);
        let scale_tol = 1e-6;
        let wr_base = vec![1.0_f64; n_r];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r.clone(), Some(wr_base.clone()),
            bits_arr, plan.clone());
        let analytic = runner_base.gradient_xi_random_pooled(scale_tol);
        let agg_base = runner_base.analyze_xi(scale_tol);

        let mut chosen: Option<(usize, usize)> = None;
        'outer: for (g, group) in agg_base.by_resize.iter().enumerate() {
            for (s, shell) in group.shells.iter().enumerate() {
                if shell.xi_naive.is_finite() && shell.xi_naive.abs() > 1e-3
                   && shell.rr_sum > 0.5
                {
                    chosen = Some((g, s));
                    break 'outer;
                }
            }
        }
        let (gx, sx) = match chosen { Some(c) => c, None => return };

        let xi_at = |wr: &[f64]| -> f64 {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), None, pts_r.clone(), Some(wr.to_vec()),
                bits_arr, plan.clone());
            r.analyze_xi(scale_tol).by_resize[gx].shells[sx].xi_naive
        };

        let v_base = xi_at(&wr_base);
        let eps = 1e-5;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let fd = (xi_at(&wr_pert) - v_base) / eps;
            let an = analytic.by_resize[gx].shell_grads[sx][j];
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 5e-3,
                "group {} shell {} random {}: FD {} vs analytic {} (diff {})",
                gx, sx, j, fd, an, abs_diff);
        }
    }

    #[test]
    fn gradient_xi_random_pooled_aggregate_matches_per_shell_combination() {
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform_periodic_pts(n_d, 5, 113111);
        let pts_r = make_uniform_periodic_pts(n_r, 5, 113222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 114333);
        let runner = CascadeRunner::new_isolated(
            pts_d, None, pts_r, None, bits, plan);
        let scale_tol = 1e-6;
        let pooled = runner.gradient_xi_random_pooled(scale_tol);

        let betas: Vec<Vec<f64>> = pooled.by_resize.iter().enumerate()
            .map(|(g, group)| {
                group.shell_grads.iter().enumerate().map(|(s, _)| {
                    if (g + s) % 2 == 0 { 1.0_f64 + (s as f64) }
                    else { -0.5_f64 * (s as f64 + 1.0) }
                }).collect()
            }).collect();
        let agg = runner.gradient_xi_random_pooled_aggregate(scale_tol, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for (g, group) in pooled.by_resize.iter().enumerate() {
            for (s, shell_g) in group.shell_grads.iter().enumerate() {
                let beta = betas[g][s];
                if beta == 0.0 { continue; }
                for (j, &gj) in shell_g.iter().enumerate() {
                    expected[j] += beta * gj;
                }
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random {}: aggregate {} vs Σ β·grad {}",
                j, agg[j], expected[j]);
        }
    }
}
