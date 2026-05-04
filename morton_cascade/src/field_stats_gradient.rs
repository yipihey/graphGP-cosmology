// field_stats_gradient.rs
//
// Backward-mode gradient of field-stats variance with respect to
// per-particle data weights. For each level l and each data particle
// i, this computes ∂ var_delta[l] / ∂ w_i^d.
//
// Math derivation (see commit 8 design notes):
//
// Define per-level accumulators
//   T_l   = Σ_c W_r(c)
//   S1_l  = Σ_c W_r(c) δ(c)         (canonically ≈ 0 by α normalization)
//   S2_l  = Σ_c W_r(c) δ(c)²
//   m_l   = S1_l / T_l               (W_r-weighted mean of δ)
//   var_l = S2_l / T_l − m_l²
// where α = Σ_i w_i^d / Σ_j w_j^r and δ(c) = W_d(c) / (α W_r(c)) − 1.
//
// Cell-sensitivity to data weights:
//   ∂ var_l / ∂ W_d(c) = 2(δ(c) − m_l) / (α T_l)        [for cell c at level l]
//
// Alpha-sensitivity:
//   ∂ var_l / ∂ α      = −2 var_l / α
//
// Per-particle gradient (chain rule):
//   ∂ var_l / ∂ w_i^d  = ∂ var_l / ∂ W_d(c_i^(l)) · 1
//                        + (∂ var_l / ∂ α) · (∂ α / ∂ w_i^d)
//                      = 2(δ(c_i^(l)) − m_l) / (α T_l)
//                        + (−2 var_l / α) · (1 / Σ w_r)
// where c_i^(l) is the level-l cell containing particle i.
//
// For cells outside footprint (W_r(c) ≤ w_r_min), the cell does not
// contribute to var_l, so the cell-sensitivity term is 0 — but the
// alpha-sensitivity term still applies (alpha uses GLOBAL sums).
//
// Implementation strategy:
//   1. Compute global α and the constant α-sensitivity term once.
//   2. For each level l: walk the cell-membership index, compute δ(c)
//      from the per-cell W_d(c)/W_r(c), and credit each cell's
//      particles with the per-cell sensitivity term.
//   3. Add the constant α-sensitivity to every entry.
//
// Cost: O(L_max · N_d) — same as one forward pass over the cascade.
// Memory: depends on output mode (see [`gradient_var_delta_all_levels`]
// vs [`gradient_var_delta_aggregate`]).

use crate::cell_membership::{CellMembership, WhichCatalog};
use crate::hier_bitvec_pair::{BitVecCascadePair, DensityFieldStats, FieldStatsConfig};

/// Per-level data-weight gradient of the variance statistic.
///
/// Indexed as `data_weight_grads[level][particle_idx]`.
#[derive(Clone, Debug)]
pub struct FieldStatsGradient {
    /// `data_weight_grads[l]` is a length-N_d vector with entry `i`
    /// holding `∂ var_delta[l] / ∂ w_i^d`. Empty for levels where the
    /// variance is undefined (no active cells).
    pub data_weight_grads: Vec<Vec<f64>>,
}

/// Per-level random-weight gradient of the variance statistic.
///
/// Indexed as `random_weight_grads[level][random_particle_idx]`.
#[derive(Clone, Debug)]
pub struct RandomWeightFieldStatsGradient {
    /// `random_weight_grads[l]` is a length-N_r vector with entry `j`
    /// holding `∂ var_delta[l] / ∂ w_j^r`. Empty for levels where the
    /// variance is undefined (no active cells).
    pub random_weight_grads: Vec<Vec<f64>>,
}

/// Per-particle data-weight gradients of the **raw** field-stats
/// accumulators $S_1$, $S_2$, $S_3$, $S_4$ at every cascade level.
///
/// These are the primitive quantities needed for downstream gradient
/// compositions that pool sums across multiple cascades (see
/// `CascadeRunner::gradient_var_delta_data_pooled` and the higher-moment
/// pooled gradient methods).
///
/// Indexed as `s{k}_grads[level][particle_idx]`. Each vector at a given
/// level has length `n_d` or is empty (for levels with undefined α /
/// footprint).
#[derive(Clone, Debug)]
pub struct RawSumDataGradient {
    /// `s1_grads[l][i] = ∂ S_1^(l) / ∂ w_i^d`
    /// where $S_1 = \sum_c W_r(c) \delta(c)$ at level l.
    pub s1_grads: Vec<Vec<f64>>,
    /// `s2_grads[l][i] = ∂ S_2^(l) / ∂ w_i^d`
    /// where $S_2 = \sum_c W_r(c) \delta(c)^2$ at level l.
    pub s2_grads: Vec<Vec<f64>>,
    /// `s3_grads[l][i] = ∂ S_3^(l) / ∂ w_i^d`
    /// where $S_3 = \sum_c W_r(c) \delta(c)^3$ at level l.
    pub s3_grads: Vec<Vec<f64>>,
    /// `s4_grads[l][i] = ∂ S_4^(l) / ∂ w_i^d`
    /// where $S_4 = \sum_c W_r(c) \delta(c)^4$ at level l.
    pub s4_grads: Vec<Vec<f64>>,
}

/// Per-particle random-weight gradients of the **raw** field-stats
/// accumulators $T$, $S_1$, $S_2$, $S_3$, $S_4$ at every cascade level.
///
/// Symmetric counterpart to [`RawSumDataGradient`]. Used by the
/// random-weight pooled gradient methods (multi-run case): pooled
/// $\mu_k$ depends on both pooled $S_k$ and pooled $T$ when those are
/// random-weight-dependent, so both gradients are needed.
///
/// Indexed as `t_grads[level][particle_idx]` and
/// `s{k}_grads[level][particle_idx]`. Each vector at a given level has
/// length `n_r` or is empty (for levels with undefined α / footprint).
///
/// **Math** (single cascade, level $\ell$):
///
/// ```text
///   ∂T^(l) / ∂w_j^r = 𝟙[j's cell at level l in footprint]
///   ∂S_k^(l) / ∂w_j^r = local_term(j) + α_term
///       local_term(j) = (1−k) δ(c_j)^k − k δ(c_j)^(k−1)   [if c_j in footprint]
///       α_term = +k (S_k + S_{k−1}) / Σ w_r              [global, all j]
/// ```
///
/// (with $S_0 \equiv T$). Out-of-footprint particles get only the
/// global α-term in $\partial S_k$ and zero in $\partial T$.
/// (Sign on the α-term is positive: scaling $w^r → k w^r$ leaves $\delta$
/// invariant but multiplies $S_k$ by $k$, so by Euler
/// $\Sigma_j w_j^r \partial S_k/\partial w_j^r = S_k$.)
#[derive(Clone, Debug)]
pub struct RawSumRandomGradient {
    /// `t_grads[l][j] = ∂ T^(l) / ∂ w_j^r` where
    /// $T = \sum_{c \in \text{footprint}} W_r(c)$ is `sum_w_r_active`.
    pub t_grads: Vec<Vec<f64>>,
    /// `s1_grads[l][j] = ∂ S_1^(l) / ∂ w_j^r`.
    pub s1_grads: Vec<Vec<f64>>,
    pub s2_grads: Vec<Vec<f64>>,
    pub s3_grads: Vec<Vec<f64>>,
    pub s4_grads: Vec<Vec<f64>>,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Compute the per-particle gradient of `var_delta` at every level
    /// with respect to the data-catalog weights.
    ///
    /// Requires a previously-computed `analyze_field_stats` result
    /// (`results`) and the matching config (`cfg`). The returned
    /// gradient is `Vec<Vec<f64>>` indexed `[level][particle_idx]`.
    ///
    /// **Cost**: O(L_max · N_d). Memory: O(L_max · N_d).
    /// For large N_d use [`Self::gradient_var_delta_aggregate`] instead.
    pub fn gradient_var_delta_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> FieldStatsGradient {
        self.gradient_central_moment_all_levels(cfg, results, 2)
    }

    /// Aggregate-scalar gradient: given per-level loss weights `betas`
    /// (length `results.len()`), compute `∂L / ∂ w_i^d` for the scalar
    /// loss `L = Σ_l β_l · var_delta[l]`. Returns one length-N_d
    /// vector.
    ///
    /// This is the optimization-friendly path: O(L_max · N_d) compute,
    /// O(N_d) memory regardless of how many levels are active.
    pub fn gradient_var_delta_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_aggregate(cfg, results, betas, 2)
    }

    /// Per-level gradient of the third central moment `m3_delta` with
    /// respect to data weights. Same shape and complexity as
    /// [`Self::gradient_var_delta_all_levels`].
    pub fn gradient_m3_delta_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> FieldStatsGradient {
        self.gradient_central_moment_all_levels(cfg, results, 3)
    }

    /// Aggregate-scalar gradient for `m3_delta`. See
    /// [`Self::gradient_var_delta_aggregate`].
    pub fn gradient_m3_delta_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_aggregate(cfg, results, betas, 3)
    }

    /// Per-level gradient of the fourth central moment `m4_delta` with
    /// respect to data weights.
    pub fn gradient_m4_delta_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> FieldStatsGradient {
        self.gradient_central_moment_all_levels(cfg, results, 4)
    }

    /// Aggregate-scalar gradient for `m4_delta`.
    pub fn gradient_m4_delta_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_aggregate(cfg, results, betas, 4)
    }

    /// Per-level gradient of the reduced skewness `s3_delta = m3 / m2²`
    /// with respect to data weights. Computed from m2 and m3 gradients
    /// via the chain rule:
    ///
    ///   ∂(m3/m2²)/∂w = (1/m2²)·∂m3/∂w − (2 m3/m2³)·∂m2/∂w
    ///
    /// At levels where `var_delta` is zero or non-finite the gradient
    /// is reported as zero (S3 is undefined there).
    pub fn gradient_s3_delta_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> FieldStatsGradient {
        let m2_grad = self.gradient_central_moment_all_levels(cfg, results, 2);
        let m3_grad = self.gradient_central_moment_all_levels(cfg, results, 3);
        let n_d = self.n_d();
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(results.len());
        for l in 0..results.len() {
            let m2 = results[l].var_delta;
            let m3 = results[l].m3_delta;
            if m2 <= 0.0 || !m2.is_finite() {
                out.push(vec![0.0; n_d]);
                continue;
            }
            let inv_m2_sq = 1.0 / (m2 * m2);
            let coeff_m2 = -2.0 * m3 / (m2 * m2 * m2);
            let g2 = &m2_grad.data_weight_grads[l];
            let g3 = &m3_grad.data_weight_grads[l];
            let mut row = vec![0.0_f64; n_d];
            for i in 0..n_d {
                row[i] = inv_m2_sq * g3[i] + coeff_m2 * g2[i];
            }
            out.push(row);
        }
        FieldStatsGradient { data_weight_grads: out }
    }

    /// Aggregate-scalar gradient for `s3_delta`.
    pub fn gradient_s3_delta_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} does not match results length {}",
            betas.len(), results.len());
        let full = self.gradient_s3_delta_all_levels(cfg, results);
        let n_d = self.n_d();
        let mut out = vec![0.0_f64; n_d];
        for l in 0..results.len() {
            if betas[l] == 0.0 { continue; }
            for (i, &g) in full.data_weight_grads[l].iter().enumerate() {
                out[i] += betas[l] * g;
            }
        }
        out
    }

    /// Per-level gradient of `var_delta` with respect to the
    /// **random-catalog** weights. For each level $\ell$ and each
    /// random particle $j$, returns $\partial \mu_2^{(\ell)} /
    /// \partial w_j^r$.
    ///
    /// Math (see `docs/differentiable_cascade.md` §2.6):
    ///
    /// ```text
    ///   ∂μ_2 / ∂w_j^r = 2 μ_2 / Σ w_r
    ///                  − (1/T) [d_j² + 2(1 + δ̄) d_j + μ_2]
    /// ```
    ///
    /// where d_j = δ(c_j^(ℓ)) − δ̄ and c_j^(ℓ) is the cell at level
    /// ℓ containing random particle j. The first term is constant
    /// across all random particles for given (ℓ); the second is
    /// per-particle.
    ///
    /// **Eligibility**: if random particle j is in a cell that's
    /// outside the footprint cut (W_r(c) ≤ w_r_min), the local cell
    /// term is omitted (the cell isn't part of T_ℓ or S_k_ℓ in the
    /// forward pass), and only the global α term remains.
    ///
    /// Returned shape: `[level][random_particle_idx]`. Length-N_r per
    /// level (or empty for levels with var undefined).
    pub fn gradient_var_delta_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RandomWeightFieldStatsGradient {
        self.gradient_central_moment_random_all_levels(cfg, results, 2)
    }

    /// Aggregate-scalar gradient for `var_delta` with respect to
    /// random weights. Given per-level betas, returns
    /// `∂L / ∂w_j^r` for `L = Σ_ℓ β_ℓ · var_delta[ℓ]`.
    pub fn gradient_var_delta_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_random_aggregate(cfg, results, betas, 2)
    }

    /// Per-level gradient of `m3_delta` (third central moment) with
    /// respect to **random-catalog** weights. Same structure and
    /// complexity as [`Self::gradient_var_delta_random_all_levels`].
    /// Math: see `docs/differentiable_cascade.md` §6.10 (unified
    /// formula for k = 2, 3, 4).
    pub fn gradient_m3_delta_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RandomWeightFieldStatsGradient {
        self.gradient_central_moment_random_all_levels(cfg, results, 3)
    }

    /// Aggregate-scalar gradient for `m3_delta` w.r.t. random weights.
    pub fn gradient_m3_delta_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_random_aggregate(cfg, results, betas, 3)
    }

    /// Per-level gradient of `m4_delta` (fourth central moment) with
    /// respect to **random-catalog** weights.
    pub fn gradient_m4_delta_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RandomWeightFieldStatsGradient {
        self.gradient_central_moment_random_all_levels(cfg, results, 4)
    }

    /// Aggregate-scalar gradient for `m4_delta` w.r.t. random weights.
    pub fn gradient_m4_delta_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        self.gradient_central_moment_random_aggregate(cfg, results, betas, 4)
    }

    /// Per-level gradient of `s3_delta = m3 / m2²` with respect to
    /// **random-catalog** weights. Computed via the chain rule:
    ///
    ///   ∂(m3/m2²)/∂w_j^r = (1/m2²)·∂m3/∂w_j^r − (2 m3/m2³)·∂m2/∂w_j^r
    ///
    /// At levels where `var_delta` is zero or non-finite the gradient
    /// is reported as zero (S3 is undefined there).
    pub fn gradient_s3_delta_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RandomWeightFieldStatsGradient {
        let m2_grad = self.gradient_central_moment_random_all_levels(cfg, results, 2);
        let m3_grad = self.gradient_central_moment_random_all_levels(cfg, results, 3);
        let n_r = self.n_r();
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(results.len());
        for l in 0..results.len() {
            let m2 = results[l].var_delta;
            let m3 = results[l].m3_delta;
            if m2 <= 0.0 || !m2.is_finite() {
                out.push(vec![0.0; n_r]);
                continue;
            }
            let inv_m2_sq = 1.0 / (m2 * m2);
            let coeff_m2 = -2.0 * m3 / (m2 * m2 * m2);
            let g2 = &m2_grad.random_weight_grads[l];
            let g3 = &m3_grad.random_weight_grads[l];
            let mut row = vec![0.0_f64; n_r];
            for j in 0..n_r {
                row[j] = inv_m2_sq * g3[j] + coeff_m2 * g2[j];
            }
            out.push(row);
        }
        RandomWeightFieldStatsGradient { random_weight_grads: out }
    }

    /// Aggregate-scalar gradient for `s3_delta` w.r.t. random weights.
    pub fn gradient_s3_delta_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} does not match results length {}",
            betas.len(), results.len());
        let full = self.gradient_s3_delta_random_all_levels(cfg, results);
        let n_r = self.n_r();
        let mut out = vec![0.0_f64; n_r];
        for l in 0..results.len() {
            if betas[l] == 0.0 { continue; }
            for (j, &g) in full.random_weight_grads[l].iter().enumerate() {
                out[j] += betas[l] * g;
            }
        }
        out
    }

    /// Per-level data-weight gradients of the raw accumulators
    /// $S_1$, $S_2$, $S_3$, $S_4$ at every cascade level.
    ///
    /// Math (in-footprint cells; out-of-footprint cells get only the
    /// global α-term). Unified formula for $k = 1, 2, 3, 4$:
    ///
    /// ```text
    ///   ∂S_k^(l) / ∂w_i^d = 𝟙[c_i in footprint] · (k · δ(c_i)^(k−1) / α)
    ///                       − k · (S_k^(l) + S_{k−1}^(l)) / (α · Σ w_r)
    /// ```
    ///
    /// (with $S_0 \equiv T$). $\partial T^{(l)}/\partial w_i^d = 0$
    /// identically (T sums random weights only).
    ///
    /// These primitives are the building blocks for downstream gradient
    /// compositions that pool raw sums across multiple cascades. See
    /// `CascadeRunner::gradient_var_delta_data_pooled` (and the
    /// higher-moment pooled gradient methods).
    ///
    /// Sanity check: combining ∂S_1, ∂S_2 via the per-cascade variance
    /// chain rule recovers `gradient_var_delta_all_levels` exactly
    /// (verified by `raw_sum_gradient_combines_to_var_gradient`).
    /// Higher moments combine analogously; see
    /// `raw_sum_gradient_combines_to_higher_moment_gradients`.
    pub fn gradient_raw_sums_data_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RawSumDataGradient {
        let n_d = self.n_d();
        let n_levels = results.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return RawSumDataGradient {
                s1_grads: vec![Vec::new(); n_levels],
                s2_grads: vec![Vec::new(); n_levels],
                s3_grads: vec![Vec::new(); n_levels],
                s4_grads: vec![Vec::new(); n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut s1_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s2_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s3_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s4_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let (g1, g2, g3, g4) = per_level_raw_sum_data_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_d,
            );
            s1_grads.push(g1);
            s2_grads.push(g2);
            s3_grads.push(g3);
            s4_grads.push(g4);
        }
        RawSumDataGradient { s1_grads, s2_grads, s3_grads, s4_grads }
    }

    /// Per-level random-weight gradients of the raw accumulators
    /// $T$, $S_1$, $S_2$, $S_3$, $S_4$ at every cascade level.
    /// Symmetric counterpart to
    /// [`Self::gradient_raw_sums_data_all_levels`]; building block for
    /// the random-weight multi-run pooled gradient methods.
    ///
    /// Math (in-footprint cells; out-of-footprint cells get only the
    /// global α-term for $S_k$ and zero for $T$):
    ///
    /// ```text
    ///   ∂T^(l) / ∂w_j^r = 𝟙[c_j^(l) in footprint]
    ///   ∂S_k^(l) / ∂w_j^r = 𝟙[c_j^(l) in footprint] · ((1−k) δ^k − k δ^(k−1))
    ///                       + k · (S_k + S_{k−1}) / Σ w_r
    /// ```
    ///
    /// (with $S_0 \equiv T$). Sign on the α-term is positive (Euler:
    /// scaling $w^r → k w^r$ leaves $\delta$ invariant and multiplies
    /// $S_k$ by $k$).
    ///
    /// Sanity check: combining ∂T and ∂S_k via the per-cascade
    /// random-weight chain rule (with $\partial T^{(r)}/\partial w_j^r$
    /// nonzero in the random case) recovers
    /// `gradient_central_moment_random_all_levels` exactly. Verified
    /// by `raw_sum_random_gradient_combines_to_central_moment_gradients`.
    pub fn gradient_raw_sums_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
    ) -> RawSumRandomGradient {
        let n_r = self.n_r();
        let n_levels = results.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return RawSumRandomGradient {
                t_grads: vec![Vec::new(); n_levels],
                s1_grads: vec![Vec::new(); n_levels],
                s2_grads: vec![Vec::new(); n_levels],
                s3_grads: vec![Vec::new(); n_levels],
                s4_grads: vec![Vec::new(); n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut t_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s1_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s2_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s3_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        let mut s4_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let (gt, g1, g2, g3, g4) = per_level_raw_sum_random_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_r,
            );
            t_grads.push(gt);
            s1_grads.push(g1);
            s2_grads.push(g2);
            s3_grads.push(g3);
            s4_grads.push(g4);
        }
        RawSumRandomGradient { t_grads, s1_grads, s2_grads, s3_grads, s4_grads }
    }

    /// Internal: per-level random-weight gradient shared by var/m3/m4.
    fn gradient_central_moment_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        k: u32,
    ) -> RandomWeightFieldStatsGradient {
        let n_r = self.n_r();
        let n_levels = results.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return RandomWeightFieldStatsGradient {
                random_weight_grads: vec![Vec::new(); n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            grads.push(per_level_central_moment_random_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_r, k,
            ));
        }
        RandomWeightFieldStatsGradient { random_weight_grads: grads }
    }

    /// Internal: aggregate-scalar random-weight gradient shared by var/m3/m4.
    fn gradient_central_moment_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
        k: u32,
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} != results length {}",
            betas.len(), results.len());
        let n_r = self.n_r();
        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return vec![0.0; n_r];
        }
        let alpha = total_w_d / total_w_r;

        let mut out = vec![0.0_f64; n_r];
        for l in 0..results.len() {
            if betas[l] == 0.0 { continue; }
            let per_level = per_level_central_moment_random_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_r, k,
            );
            for (j, &g) in per_level.iter().enumerate() {
                out[j] += betas[l] * g;
            }
        }
        out
    }

    /// Internal: per-level gradient computation shared by var, m3, m4.
    fn gradient_central_moment_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        k: u32,
    ) -> FieldStatsGradient {
        let n_d = self.n_d();
        let n_levels = results.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return FieldStatsGradient {
                data_weight_grads: vec![Vec::new(); n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            grads.push(per_level_central_moment_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_d, k,
            ));
        }
        FieldStatsGradient { data_weight_grads: grads }
    }

    /// Internal: aggregate-scalar gradient computation shared by var, m3, m4.
    fn gradient_central_moment_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[DensityFieldStats],
        betas: &[f64],
        k: u32,
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} does not match results length {}",
            betas.len(), results.len());

        let n_d = self.n_d();
        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return vec![0.0; n_d];
        }
        let alpha = total_w_d / total_w_r;

        let mut out = vec![0.0_f64; n_d];
        for l in 0..results.len() {
            if betas[l] == 0.0 { continue; }
            let per_level = per_level_central_moment_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_d, k,
            );
            for (i, &g) in per_level.iter().enumerate() {
                out[i] += betas[l] * g;
            }
        }
        out
    }
}

/// Compute the per-particle gradient of the k-th central moment of δ
/// at a single level. Implements the unified formula
///
///   ∂ μ_k / ∂ w_i^d = (k / (α T_l)) · [(δ(c_i) − m_l)^(k−1) − μ_{k−1}]
///                     + (−k μ_k / α) · (1 / Σ w_r)
///
/// where μ_1 ≡ 0 by convention. Valid for k ∈ {2, 3, 4}.
fn per_level_central_moment_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_d: usize,
    k: u32,
) -> Vec<f64> {
    debug_assert!(k >= 2 && k <= 4, "moment order k must be 2, 3, or 4");
    let mut grad = vec![0.0_f64; n_d];

    let stats = &results[level];
    let m1 = stats.mean_delta;
    let t_l = stats.sum_w_r_active;

    // Look up μ_k and μ_{k-1} from the analysis result.
    let mu_k = match k {
        2 => stats.var_delta,
        3 => stats.m3_delta,
        4 => stats.m4_delta,
        _ => unreachable!(),
    };
    let mu_km1 = match k {
        2 => 0.0,                 // μ_1 ≡ 0 by definition (centered)
        3 => stats.var_delta,     // μ_2
        4 => stats.m3_delta,      // μ_3
        _ => unreachable!(),
    };

    // Constant α-sensitivity: ∂ μ_k / ∂ α = −k μ_k / α; chain through
    // ∂ α / ∂ w_i^d = 1 / Σ w_r gives a uniform per-particle term.
    let alpha_term = if alpha > 0.0 && total_w_r > 0.0 {
        -(k as f64) * mu_k / (alpha * total_w_r)
    } else {
        0.0
    };
    for g in grad.iter_mut() { *g = alpha_term; }

    if alpha <= 0.0 || t_l <= 0.0 {
        return grad;
    }
    let cell_prefactor = k as f64 / (alpha * t_l);

    for (cell_id, members_d) in mem_d.non_empty_cells_at(level) {
        let w_d_c: f64 = match pair.weights_d() {
            Some(w) => members_d.iter().map(|&i| w[i as usize]).sum(),
            None => members_d.len() as f64,
        };
        let members_r = mem_r.members(level, cell_id);
        let w_r_c: f64 = match pair.weights_r() {
            Some(w) => members_r.iter().map(|&i| w[i as usize]).sum(),
            None => members_r.len() as f64,
        };
        if w_r_c <= cfg.w_r_min { continue; }
        let delta = w_d_c / (alpha * w_r_c) - 1.0;
        let dev = delta - m1;
        // (δ - m_1)^(k-1)
        let dev_pow = match k {
            2 => dev,
            3 => dev * dev,
            4 => dev * dev * dev,
            _ => unreachable!(),
        };
        let cell_term = cell_prefactor * (dev_pow - mu_km1);
        for &i in members_d {
            grad[i as usize] += cell_term;
        }
    }

    grad
}

/// Compute the per-particle gradient at a single level (variance only).
/// Thin wrapper around the generic central-moment helper for k=2.
#[allow(dead_code)]
fn per_level_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_d: usize,
) -> Vec<f64> {
    per_level_central_moment_gradient(
        pair, mem_d, mem_r, level, results, alpha, total_w_r, cfg, n_d, 2)
}

/// Compute the per-particle gradient of `var_delta` at a single
/// level with respect to **random** weights.
/// Compute the per-particle gradient of the k-th central moment of δ
/// at a single level with respect to **random** weights, for k ∈ {2,3,4}.
///
/// Implements the unified formula derived in `docs/differentiable_cascade.md`
/// §6.10:
///
///   ∂μ_k / ∂w_j^r = k μ_k / Σ w_r
///                  − (1/T_ℓ) [(k−1) d_j^k + k(1+δ̄) d_j^(k−1)
///                              − k(1+δ̄) μ_{k−1} + μ_k]
///
/// where d_j = δ(c_j) − δ̄ for the level-ℓ cell containing random
/// particle j, and μ_1 ≡ 0 by convention. The local cell term is
/// evaluated only when c_j passes the footprint cut; for cells outside
/// footprint, only the global α term applies.
fn per_level_central_moment_random_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_r: usize,
    k: u32,
) -> Vec<f64> {
    debug_assert!(k >= 2 && k <= 4, "moment order k must be 2, 3, or 4");
    let mut grad = vec![0.0_f64; n_r];

    let stats = &results[level];
    let m_bar = stats.mean_delta;
    let t_l = stats.sum_w_r_active;

    // Look up μ_k and μ_{k-1} from the analysis result.
    let mu_k = match k {
        2 => stats.var_delta,
        3 => stats.m3_delta,
        4 => stats.m4_delta,
        _ => unreachable!(),
    };
    let mu_km1 = match k {
        2 => 0.0,                 // μ_1 ≡ 0 by definition
        3 => stats.var_delta,     // μ_2
        4 => stats.m3_delta,      // μ_3
        _ => unreachable!(),
    };

    // Constant α-term: ∂μ_k / ∂α = -k μ_k / α; chain via
    // ∂α / ∂w_j^r = -α / Σ w_r gives +k μ_k / Σ w_r — same value
    // for every random particle, applies regardless of footprint.
    let alpha_term = if total_w_r > 0.0 {
        (k as f64) * mu_k / total_w_r
    } else {
        0.0
    };
    for g in grad.iter_mut() { *g = alpha_term; }

    if alpha <= 0.0 || t_l <= 0.0 {
        return grad;
    }
    let inv_t = 1.0 / t_l;
    let kf = k as f64;

    for (cell_id, members_r) in mem_r.non_empty_cells_at(level) {
        let w_r_c: f64 = match pair.weights_r() {
            Some(w) => members_r.iter().map(|&j| w[j as usize]).sum(),
            None => members_r.len() as f64,
        };
        if w_r_c <= cfg.w_r_min { continue; }
        let members_d = mem_d.members(level, cell_id);
        let w_d_c: f64 = match pair.weights_d() {
            Some(w) => members_d.iter().map(|&i| w[i as usize]).sum(),
            None => members_d.len() as f64,
        };
        let delta = w_d_c / (alpha * w_r_c) - 1.0;
        let d_j = delta - m_bar;
        // d_j^(k-1) and d_j^k
        let dj_km1 = match k {
            2 => d_j,
            3 => d_j * d_j,
            4 => d_j * d_j * d_j,
            _ => unreachable!(),
        };
        let dj_k = dj_km1 * d_j;
        // Local term (negated form of the bracket):
        //   −(1/T)[(k−1) d_j^k + k(1+δ̄) d_j^(k−1) − k(1+δ̄) μ_{k−1} + μ_k]
        let bracket =
            (kf - 1.0) * dj_k
            + kf * (1.0 + m_bar) * dj_km1
            - kf * (1.0 + m_bar) * mu_km1
            + mu_k;
        let local_term = -inv_t * bracket;
        for &j in members_r {
            grad[j as usize] += local_term;
        }
    }

    grad
}

/// Backward-compatible wrapper for the k=2 case.
#[allow(dead_code)]
fn per_level_variance_random_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_r: usize,
) -> Vec<f64> {
    per_level_central_moment_random_gradient(
        pair, mem_d, mem_r, level, results, alpha, total_w_r, cfg, n_r, 2)
}

/// Compute the per-particle data-weight gradients of the raw
/// accumulators $S_1$ and $S_2$ at a single level. Returns
/// `(grad_s1, grad_s2)`, each length $n_d$.
///
/// In-footprint cells contribute the local term; particles in
/// out-of-footprint cells receive only the global α-term. Particles
/// outside any cell at this level (impossible for nonempty levels)
/// also receive only the α-term — both end up in the same code path.
fn per_level_raw_sum_data_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_d: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let stats = &results[level];
    // Recover raw moments from central moments:
    //   S_1 = T · m_1
    //   S_2 = T · (μ_2 + m_1²)
    //   S_3 = T · (μ_3 + 3 m_1 μ_2 + m_1³)
    //   S_4 = T · (μ_4 + 4 m_1 μ_3 + 6 m_1² μ_2 + m_1⁴)
    let t_l = stats.sum_w_r_active;
    let m_1 = stats.mean_delta;
    let mu_2 = stats.var_delta;
    let mu_3 = stats.m3_delta;
    let mu_4 = stats.m4_delta;
    let m1_sq = m_1 * m_1;
    let m1_cu = m1_sq * m_1;
    let m1_qu = m1_cu * m_1;
    let s_1 = t_l * m_1;
    let s_2 = t_l * (mu_2 + m1_sq);
    let s_3 = t_l * (mu_3 + 3.0 * m_1 * mu_2 + m1_cu);
    let s_4 = t_l * (mu_4 + 4.0 * m_1 * mu_3 + 6.0 * m1_sq * mu_2 + m1_qu);

    // Global α-term (uniform per particle):
    //   α-term for ∂S_k = -k·(S_k + S_{k-1}) / (α · Σw_r), with S_0 = T.
    let (a1, a2, a3, a4) = if alpha > 0.0 && total_w_r > 0.0 {
        let inv = 1.0 / (alpha * total_w_r);
        (-(s_1 + t_l) * inv,
         -2.0 * (s_2 + s_1) * inv,
         -3.0 * (s_3 + s_2) * inv,
         -4.0 * (s_4 + s_3) * inv)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let mut g1 = vec![a1; n_d];
    let mut g2 = vec![a2; n_d];
    let mut g3 = vec![a3; n_d];
    let mut g4 = vec![a4; n_d];

    // Local cell terms (in-footprint only):
    //   local term for ∂S_k = k · δ(c)^(k-1) / α
    if alpha <= 0.0 || t_l <= 0.0 {
        return (g1, g2, g3, g4);
    }
    let inv_alpha = 1.0 / alpha;
    for (cell_id, members_d) in mem_d.non_empty_cells_at(level) {
        let members_r = mem_r.members(level, cell_id);
        let w_r_c: f64 = match pair.weights_r() {
            Some(w) => members_r.iter().map(|&j| w[j as usize]).sum(),
            None => members_r.len() as f64,
        };
        if w_r_c <= cfg.w_r_min { continue; }
        let w_d_c: f64 = match pair.weights_d() {
            Some(w) => members_d.iter().map(|&i| w[i as usize]).sum(),
            None => members_d.len() as f64,
        };
        let delta = w_d_c / (alpha * w_r_c) - 1.0;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let local_s1 = inv_alpha;
        let local_s2 = 2.0 * delta * inv_alpha;
        let local_s3 = 3.0 * d2 * inv_alpha;
        let local_s4 = 4.0 * d3 * inv_alpha;
        for &i in members_d {
            g1[i as usize] += local_s1;
            g2[i as usize] += local_s2;
            g3[i as usize] += local_s3;
            g4[i as usize] += local_s4;
        }
    }
    (g1, g2, g3, g4)
}

/// Compute the per-particle random-weight gradients of $T$ and the
/// raw accumulators $S_1$, $S_2$, $S_3$, $S_4$ at a single level.
/// Returns `(grad_t, grad_s1, grad_s2, grad_s3, grad_s4)`, each
/// length $n_r$.
///
/// In-footprint cells contribute the local term; particles in
/// out-of-footprint cells receive only the global α-term in $\partial S_k$
/// and zero in $\partial T$.
///
/// Math (in-footprint cell $c$ containing particle $j$):
///
/// ```text
///   ∂T / ∂w_j^r = 1
///   ∂S_k / ∂w_j^r |_{local} = (1−k) δ(c)^k − k δ(c)^(k−1)
/// ```
///
/// Global α-term applied to all particles regardless of footprint:
///
/// ```text
///   ∂S_k / ∂w_j^r |_{α-term} = −k (S_k + S_{k−1}) / Σ w_r
/// ```
///
/// (with $S_0 \equiv T$).
fn per_level_raw_sum_random_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    _mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[DensityFieldStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_r: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let stats = &results[level];
    // Recover raw moments from central moments (same as data path).
    let t_l = stats.sum_w_r_active;
    let m_1 = stats.mean_delta;
    let mu_2 = stats.var_delta;
    let mu_3 = stats.m3_delta;
    let mu_4 = stats.m4_delta;
    let m1_sq = m_1 * m_1;
    let m1_cu = m1_sq * m_1;
    let m1_qu = m1_cu * m_1;
    let s_1 = t_l * m_1;
    let s_2 = t_l * (mu_2 + m1_sq);
    let s_3 = t_l * (mu_3 + 3.0 * m_1 * mu_2 + m1_cu);
    let s_4 = t_l * (mu_4 + 4.0 * m_1 * mu_3 + 6.0 * m1_sq * mu_2 + m1_qu);

    // Global α-terms (uniform across all random particles):
    //   α-term for ∂S_k = +k (S_k + S_{k−1}) / W_r_total, with S_0 ≡ T.
    // Sign is positive: scaling w_r by k leaves δ invariant but
    // multiplies S_k by k, so by Euler's identity Σ_j w_j ∂S_k/∂w_j = S_k.
    let (a1, a2, a3, a4) = if total_w_r > 0.0 {
        let inv = 1.0 / total_w_r;
        ((s_1 + t_l) * inv,
         2.0 * (s_2 + s_1) * inv,
         3.0 * (s_3 + s_2) * inv,
         4.0 * (s_4 + s_3) * inv)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let mut gt = vec![0.0_f64; n_r];
    let mut g1 = vec![a1; n_r];
    let mut g2 = vec![a2; n_r];
    let mut g3 = vec![a3; n_r];
    let mut g4 = vec![a4; n_r];

    if alpha <= 0.0 || t_l <= 0.0 {
        return (gt, g1, g2, g3, g4);
    }

    // Local cell terms (in-footprint cells only).
    //
    // Per-particle local term per the math:
    //   ∂T  /∂w_j^r |_{local} = 1                              (j in any in-footprint cell)
    //   ∂S_k/∂w_j^r |_{local} = (1−k) δ^k − k δ^(k−1)
    //
    //   k=1: 0·δ − 1·δ⁰ = −1
    //   k=2: −1·δ² − 2·δ
    //   k=3: −2·δ³ − 3·δ²
    //   k=4: −3·δ⁴ − 4·δ³
    for (cell_id, members_r) in mem_r.non_empty_cells_at(level) {
        let w_r_c: f64 = match pair.weights_r() {
            Some(w) => members_r.iter().map(|&j| w[j as usize]).sum(),
            None => members_r.len() as f64,
        };
        if w_r_c <= cfg.w_r_min { continue; }
        let w_d_c: f64 = match pair.weights_d() {
            Some(w) => _mem_d.members(level, cell_id).iter()
                .map(|&i| w[i as usize]).sum(),
            None => _mem_d.members(level, cell_id).len() as f64,
        };
        let delta = w_d_c / (alpha * w_r_c) - 1.0;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d3 * delta;
        let local_t = 1.0_f64;
        let local_s1 = -1.0_f64;
        let local_s2 = -d2 - 2.0 * delta;
        let local_s3 = -2.0 * d3 - 3.0 * d2;
        let local_s4 = -3.0 * d4 - 4.0 * d3;
        for &j in members_r {
            let jj = j as usize;
            gt[jj] += local_t;
            g1[jj] += local_s1;
            g2[jj] += local_s2;
            g3[jj] += local_s3;
            g4[jj] += local_s4;
        }
    }
    (gt, g1, g2, g3, g4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_range::{CoordRange, TrimmedPoints};

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn make_uniform(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mask = (1u64 << bits) - 1;
        (0..n)
            .map(|_| [splitmix64(&mut s) & mask, splitmix64(&mut s) & mask, splitmix64(&mut s) & mask])
            .collect()
    }

    fn build_pair(
        pts_d: Vec<[u64; 3]>, pts_r: Vec<[u64; 3]>,
        wd: Option<Vec<f64>>, wr: Option<Vec<f64>>,
    ) -> BitVecCascadePair<3> {
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        BitVecCascadePair::<3>::build_full(td, tr, wd, wr, None, 64)
    }

    #[test]
    fn gradient_finite_difference_parity_unweighted() {
        // The headline correctness test: for each particle i, perturb
        // its weight by ε and compute (var(w + ε e_i) − var(w)) / ε.
        // This finite-difference derivative must match the analytic
        // gradient to ~1e-6.
        //
        // We use small N (~50 particles) to keep the test fast since
        // we evaluate the cascade ~N times. With unit weights becoming
        // (1+ε), the cell weighted sums shift in a controlled way.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 4444);
        let pts_r = make_uniform(n_r, bits, 5555);

        // Start with unit weights so we have a well-defined baseline.
        let wd_base = vec![1.0_f64; n_d];
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_var_delta_all_levels(&cfg, &stats_base);

        // Pick a level with a meaningful variance to test against.
        let test_level = stats_base.iter()
            .enumerate()
            .find(|(_, s)| s.var_delta > 1e-3 && s.n_cells_active >= 2)
            .map(|(l, _)| l);
        let test_level = match test_level {
            Some(l) => l,
            None => return,  // test data didn't yield sufficient signal; skip
        };

        let eps = 1e-5;
        let var_base = stats_base[test_level].var_delta;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let var_pert = stats_pert[test_level].var_delta;
            let fd_grad = (var_pert - var_base) / eps;
            let an_grad = grad.data_weight_grads[test_level][i];
            // Use absolute tolerance because the gradient itself can
            // be small near the optimum; relative tolerance would be
            // too strict for tiny gradients.
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-4,
                "particle {} at level {}: FD grad {} vs analytic {} \
                 (abs diff {}, var_base {})",
                i, test_level, fd_grad, an_grad, abs_diff, var_base);
        }
    }

    #[test]
    fn gradient_uniform_scaling_invariant() {
        // Variance is invariant under uniform data-weight scaling
        // (because α scales by the same factor, so δ(c) is unchanged).
        // By chain rule: Σ_i w_i · ∂ var_l / ∂ w_i = 0.
        // Test this analytic invariant.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 7777);
        let pts_r = make_uniform(n_r, 5, 8888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let grad = pair.gradient_var_delta_all_levels(&cfg, &stats);

        for (l, gs) in grad.data_weight_grads.iter().enumerate() {
            if gs.is_empty() { continue; }
            let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                .map(|(g, w)| g * w).sum();
            // Sum should be ~0 to f64 precision (it's exactly zero
            // analytically, but floats accumulate ~1e-13 noise per op).
            assert!(weighted_sum.abs() < 1e-9,
                "level {}: Σ w_i · ∂var/∂w_i = {} (should be ≈ 0 \
                 by uniform-scaling invariance)",
                l, weighted_sum);
        }
    }

    #[test]
    fn gradient_aggregate_matches_per_level_combination() {
        // The aggregate-scalar API (with arbitrary betas) must equal
        // Σ_l β_l · per_level_grad[l]. This is just a refactoring
        // identity but worth pinning so future optimizations don't
        // accidentally diverge.
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 1111);
        let pts_r = make_uniform(n_r, 5, 2222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let grad = pair.gradient_var_delta_all_levels(&cfg, &stats);

        // Use distinctive betas (not all 1) to exercise the weighting.
        let betas: Vec<f64> = (0..stats.len())
            .map(|l| if l % 2 == 0 { 1.0 } else { -0.5 } * (l as f64 + 1.0))
            .collect();
        let agg = pair.gradient_var_delta_aggregate(&cfg, &stats, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for l in 0..stats.len() {
            for (i, &g) in grad.data_weight_grads[l].iter().enumerate() {
                expected[i] += betas[l] * g;
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β_l grad_l {}",
                i, agg[i], expected[i]);
        }
    }

    #[test]
    fn gradient_shape_matches_n_data_per_level() {
        // Sanity: every level's gradient vector must be n_d long
        // (or empty for degenerate levels).
        let n_d = 100;
        let n_r = 300;
        let pts_d = make_uniform(n_d, 5, 3333);
        let pts_r = make_uniform(n_r, 5, 4444);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let grad = pair.gradient_var_delta_all_levels(&cfg, &stats);
        assert_eq!(grad.data_weight_grads.len(), stats.len());
        for (l, gs) in grad.data_weight_grads.iter().enumerate() {
            assert!(gs.is_empty() || gs.len() == n_d,
                "level {}: gradient length {} (expected 0 or {})",
                l, gs.len(), n_d);
        }
    }

    // -----------------------------------------------------------------
    // Commit 9: higher-moment gradients (m3, m4, S3)
    // -----------------------------------------------------------------

    /// Helper: pick a level with meaningful signal in moment k.
    fn pick_signal_level(stats: &[DensityFieldStats], k: u32, threshold: f64) -> Option<usize> {
        for (l, s) in stats.iter().enumerate() {
            if s.n_cells_active < 2 { continue; }
            let value = match k {
                2 => s.var_delta,
                3 => s.m3_delta,
                4 => s.m4_delta,
                _ => unreachable!(),
            };
            if value.abs() > threshold {
                return Some(l);
            }
        }
        None
    }

    #[test]
    fn gradient_m3_finite_difference_parity() {
        // FD parity for m3, the third central moment of δ.
        // Same template as the variance test, with m3 in place of var.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 6444);
        let pts_r = make_uniform(n_r, bits, 6555);
        let wd_base = vec![1.0_f64; n_d];
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_m3_delta_all_levels(&cfg, &stats_base);

        let test_level = match pick_signal_level(&stats_base, 3, 1e-3) {
            Some(l) => l,
            None => return,
        };
        let eps = 1e-5;
        let m3_base = stats_base[test_level].m3_delta;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let m3_pert = stats_pert[test_level].m3_delta;
            let fd_grad = (m3_pert - m3_base) / eps;
            let an_grad = grad.data_weight_grads[test_level][i];
            // Looser absolute tolerance because m3 has cubic dependence
            // on δ — second-order FD truncation error grows accordingly.
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "m3: particle {} at level {}: FD {} vs analytic {} \
                 (abs diff {})",
                i, test_level, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn gradient_m4_finite_difference_parity() {
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 6644);
        let pts_r = make_uniform(n_r, bits, 6655);
        let wd_base = vec![1.0_f64; n_d];
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_m4_delta_all_levels(&cfg, &stats_base);

        let test_level = match pick_signal_level(&stats_base, 4, 1e-3) {
            Some(l) => l,
            None => return,
        };
        let eps = 1e-5;
        let m4_base = stats_base[test_level].m4_delta;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let m4_pert = stats_pert[test_level].m4_delta;
            let fd_grad = (m4_pert - m4_base) / eps;
            let an_grad = grad.data_weight_grads[test_level][i];
            // Even looser for m4 — quartic in δ.
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-2,
                "m4: particle {} at level {}: FD {} vs analytic {} \
                 (abs diff {})",
                i, test_level, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn gradient_s3_finite_difference_parity() {
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 6744);
        let pts_r = make_uniform(n_r, bits, 6755);
        let wd_base = vec![1.0_f64; n_d];
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_s3_delta_all_levels(&cfg, &stats_base);

        // S3 = m3/m2² — pick a level where both are non-trivial.
        let test_level = stats_base.iter().enumerate()
            .find(|(_, s)| s.var_delta > 1e-3 && s.m3_delta.abs() > 1e-3
                  && s.n_cells_active >= 2)
            .map(|(l, _)| l);
        let test_level = match test_level {
            Some(l) => l,
            None => return,
        };
        let eps = 1e-5;
        let s3_base = stats_base[test_level].s3_delta;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let s3_pert = stats_pert[test_level].s3_delta;
            let fd_grad = (s3_pert - s3_base) / eps;
            let an_grad = grad.data_weight_grads[test_level][i];
            // S3 = m3/m2² is a ratio; small denominators amplify FD
            // noise. Use a tolerance scaled to |s3_base| for robustness.
            let scale = s3_base.abs().max(1.0);
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-2 * scale,
                "S3: particle {} at level {}: FD {} vs analytic {} \
                 (abs diff {}, |s3| {})",
                i, test_level, fd_grad, an_grad, abs_diff, s3_base.abs());
        }
    }

    #[test]
    fn gradient_higher_moments_uniform_scaling_invariant() {
        // All central moments (and S3) are invariant under uniform
        // data-weight scaling. Σ_i w_i · ∂ μ_k / ∂ w_i = 0 by chain
        // rule for each k.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 7777);
        let pts_r = make_uniform(n_r, 5, 8888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);

        for (name, grad) in [
            ("m3", pair.gradient_m3_delta_all_levels(&cfg, &stats)),
            ("m4", pair.gradient_m4_delta_all_levels(&cfg, &stats)),
            ("s3", pair.gradient_s3_delta_all_levels(&cfg, &stats)),
        ] {
            for (l, gs) in grad.data_weight_grads.iter().enumerate() {
                if gs.is_empty() { continue; }
                let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                    .map(|(g, w)| g * w).sum();
                assert!(weighted_sum.abs() < 1e-9,
                    "{} level {}: Σ w_i · ∂μ/∂w_i = {} (should be ≈ 0)",
                    name, l, weighted_sum);
            }
        }
    }

    #[test]
    fn gradient_s3_matches_chain_rule_combination() {
        // Direct check of the S3 chain rule:
        //   ∂(m3/m2²) = (1/m2²)·∂m3 − (2 m3/m2³)·∂m2
        // Compute m2 and m3 gradients separately and combine; compare
        // to the dedicated S3 method.
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 9111);
        let pts_r = make_uniform(n_r, 5, 9222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let g_m2 = pair.gradient_var_delta_all_levels(&cfg, &stats);
        let g_m3 = pair.gradient_m3_delta_all_levels(&cfg, &stats);
        let g_s3 = pair.gradient_s3_delta_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let m2 = stats[l].var_delta;
            let m3 = stats[l].m3_delta;
            if m2 <= 0.0 { continue; }
            let inv_m2_sq = 1.0 / (m2 * m2);
            let coeff_m2 = -2.0 * m3 / (m2 * m2 * m2);
            for i in 0..n_d {
                let expected = inv_m2_sq * g_m3.data_weight_grads[l][i]
                             + coeff_m2 * g_m2.data_weight_grads[l][i];
                let got = g_s3.data_weight_grads[l][i];
                assert!((got - expected).abs() < 1e-10,
                    "S3 chain rule mismatch at level {} particle {}: \
                     manual {} vs method {}",
                    l, i, expected, got);
            }
        }
    }

    #[test]
    fn gradient_higher_moments_aggregate_matches_per_level() {
        // Aggregate ↔ per-level identity for m3, m4, s3.
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 1411);
        let pts_r = make_uniform(n_r, 5, 2422);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let betas: Vec<f64> = (0..stats.len())
            .map(|l| if l % 2 == 0 { 1.0 } else { -0.5 } * (l as f64 + 1.0))
            .collect();

        for (name, full, agg) in [
            ("m3",
             pair.gradient_m3_delta_all_levels(&cfg, &stats),
             pair.gradient_m3_delta_aggregate(&cfg, &stats, &betas)),
            ("m4",
             pair.gradient_m4_delta_all_levels(&cfg, &stats),
             pair.gradient_m4_delta_aggregate(&cfg, &stats, &betas)),
            ("s3",
             pair.gradient_s3_delta_all_levels(&cfg, &stats),
             pair.gradient_s3_delta_aggregate(&cfg, &stats, &betas)),
        ] {
            let mut expected = vec![0.0_f64; n_d];
            for l in 0..stats.len() {
                for (i, &g) in full.data_weight_grads[l].iter().enumerate() {
                    expected[i] += betas[l] * g;
                }
            }
            for i in 0..n_d {
                assert!((agg[i] - expected[i]).abs() < 1e-10,
                    "{} particle {}: aggregate {} vs Σ β_l grad_l {}",
                    name, i, agg[i], expected[i]);
            }
        }
    }

    // -----------------------------------------------------------------
    // Commit 12: random-weight variance gradient
    // -----------------------------------------------------------------

    #[test]
    fn gradient_var_random_finite_difference_parity() {
        // FD parity: perturb each random weight by ε, recompute var,
        // compare (var(w+ε e_j) - var(w))/ε to the analytic gradient.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 80;
        let pts_d = make_uniform(n_d, bits, 18444);
        let pts_r = make_uniform(n_r, bits, 18555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_var_delta_random_all_levels(&cfg, &stats_base);

        // Pick a level with measurable variance (need both signal AND
        // multiple cells for the per-particle terms to differ).
        let test_level = stats_base.iter().enumerate()
            .find(|(_, s)| s.var_delta > 1e-3 && s.n_cells_active >= 2)
            .map(|(l, _)| l);
        let test_level = match test_level {
            Some(l) => l,
            None => return,
        };

        let eps = 1e-5;
        let var_base = stats_base[test_level].var_delta;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let var_pert = stats_pert[test_level].var_delta;
            let fd_grad = (var_pert - var_base) / eps;
            let an_grad = grad.random_weight_grads[test_level][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "random j={} at level {}: FD {} vs analytic {} (diff {})",
                j, test_level, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn gradient_var_random_uniform_scaling_invariant() {
        // Variance is invariant under uniform scaling of all random
        // weights: w_j^r → k · w_j^r implies α → α/k and W_r(c) → k W_r(c),
        // so δ(c) is unchanged. By chain rule,
        //   Σ_j w_j^r · ∂ var_delta / ∂w_j^r = 0  (exactly).
        let n_d = 50;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 19777);
        let pts_r = make_uniform(n_r, 5, 19888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 60.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let grad = pair.gradient_var_delta_random_all_levels(&cfg, &stats);

        for (l, gs) in grad.random_weight_grads.iter().enumerate() {
            if gs.is_empty() { continue; }
            let weighted_sum: f64 = gs.iter().zip(weights_r.iter())
                .map(|(g, w)| g * w).sum();
            assert!(weighted_sum.abs() < 1e-9,
                "level {}: Σ w_j^r · ∂var/∂w_j^r = {} (should be ≈ 0)",
                l, weighted_sum);
        }
    }

    #[test]
    fn gradient_var_random_aggregate_matches_per_level() {
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 20111);
        let pts_r = make_uniform(n_r, 5, 20222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let grad = pair.gradient_var_delta_random_all_levels(&cfg, &stats);

        let betas: Vec<f64> = (0..stats.len())
            .map(|l| if l % 2 == 0 { 1.0 } else { -0.5 } * (l as f64 + 1.0))
            .collect();
        let agg = pair.gradient_var_delta_random_aggregate(&cfg, &stats, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for l in 0..stats.len() {
            for (j, &g) in grad.random_weight_grads[l].iter().enumerate() {
                expected[j] += betas[l] * g;
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random j={}: aggregate {} vs Σ β_l grad_l {}",
                j, agg[j], expected[j]);
        }
    }

    // -----------------------------------------------------------------
    // Commit 12 extended: m3, m4, S3 random-weight gradients
    // -----------------------------------------------------------------

    /// Helper: pick a level with measurable signal in moment k.
    fn pick_signal_level_random(stats: &[DensityFieldStats], k: u32, threshold: f64) -> Option<usize> {
        for (l, s) in stats.iter().enumerate() {
            if s.n_cells_active < 2 { continue; }
            let value = match k {
                2 => s.var_delta,
                3 => s.m3_delta,
                4 => s.m4_delta,
                _ => unreachable!(),
            };
            if value.abs() > threshold {
                return Some(l);
            }
        }
        None
    }

    #[test]
    fn gradient_m3_random_finite_difference_parity() {
        // FD parity for ∂m3/∂w_j^r. Same template as variance, with
        // m3 in place of var. Tolerance scales with the cubic
        // dependence of m3 on δ.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 80;
        let pts_d = make_uniform(n_d, bits, 24444);
        let pts_r = make_uniform(n_r, bits, 24555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_m3_delta_random_all_levels(&cfg, &stats_base);

        let test_level = match pick_signal_level_random(&stats_base, 3, 1e-3) {
            Some(l) => l,
            None => return,
        };

        let eps = 1e-5;
        let m3_base = stats_base[test_level].m3_delta;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let m3_pert = stats_pert[test_level].m3_delta;
            let fd_grad = (m3_pert - m3_base) / eps;
            let an_grad = grad.random_weight_grads[test_level][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "m3 random j={} at level {}: FD {} vs analytic {} (diff {})",
                j, test_level, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn gradient_m4_random_finite_difference_parity() {
        // FD parity for ∂m4/∂w_j^r. Quartic dependence — looser tolerance.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 80;
        let pts_d = make_uniform(n_d, bits, 24644);
        let pts_r = make_uniform(n_r, bits, 24655);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_m4_delta_random_all_levels(&cfg, &stats_base);

        let test_level = match pick_signal_level_random(&stats_base, 4, 1e-3) {
            Some(l) => l,
            None => return,
        };

        let eps = 1e-5;
        let m4_base = stats_base[test_level].m4_delta;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let m4_pert = stats_pert[test_level].m4_delta;
            let fd_grad = (m4_pert - m4_base) / eps;
            let an_grad = grad.random_weight_grads[test_level][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-2,
                "m4 random j={} at level {}: FD {} vs analytic {} (diff {})",
                j, test_level, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn gradient_s3_random_finite_difference_parity() {
        let bits = 4u32;
        let n_d = 30;
        let n_r = 80;
        let pts_d = make_uniform(n_d, bits, 24744);
        let pts_r = make_uniform(n_r, bits, 24755);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_field_stats(&cfg);
        let grad = pair_base.gradient_s3_delta_random_all_levels(&cfg, &stats_base);

        // S3 needs both var and m3 nontrivial.
        let test_level = stats_base.iter().enumerate()
            .find(|(_, s)| s.var_delta > 1e-3 && s.m3_delta.abs() > 1e-3
                  && s.n_cells_active >= 2)
            .map(|(l, _)| l);
        let test_level = match test_level {
            Some(l) => l,
            None => return,
        };

        let eps = 1e-5;
        let s3_base = stats_base[test_level].s3_delta;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_field_stats(&cfg);
            let s3_pert = stats_pert[test_level].s3_delta;
            let fd_grad = (s3_pert - s3_base) / eps;
            let an_grad = grad.random_weight_grads[test_level][j];
            let scale = s3_base.abs().max(1.0);
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-2 * scale,
                "S3 random j={} at level {}: FD {} vs analytic {} (diff {}, |s3| {})",
                j, test_level, fd_grad, an_grad, abs_diff, s3_base.abs());
        }
    }

    #[test]
    fn gradient_higher_moments_random_uniform_scaling_invariant() {
        // All central moments + S3 invariant under uniform scaling of
        // random weights → Σ_j w_j^r · ∂μ_k/∂w_j^r = 0 exactly.
        // Strong invariant; catches any missing terms.
        let n_d = 50;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 25777);
        let pts_r = make_uniform(n_r, 5, 25888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 60.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);

        for (name, grad) in [
            ("m3", pair.gradient_m3_delta_random_all_levels(&cfg, &stats)),
            ("m4", pair.gradient_m4_delta_random_all_levels(&cfg, &stats)),
            ("s3", pair.gradient_s3_delta_random_all_levels(&cfg, &stats)),
        ] {
            for (l, gs) in grad.random_weight_grads.iter().enumerate() {
                if gs.is_empty() { continue; }
                let weighted_sum: f64 = gs.iter().zip(weights_r.iter())
                    .map(|(g, w)| g * w).sum();
                assert!(weighted_sum.abs() < 1e-9,
                    "{} level {}: Σ w_j^r · ∂μ/∂w_j^r = {} (should be ≈ 0)",
                    name, l, weighted_sum);
            }
        }
    }

    #[test]
    fn gradient_s3_random_chain_rule_consistency() {
        // Direct check of S3 chain rule: dedicated S3 method should
        // equal (1/m_2²)·∂m_3 − (2 m_3/m_2³)·∂m_2 computed from m_2,
        // m_3 random-weight gradients separately.
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 26111);
        let pts_r = make_uniform(n_r, 5, 26222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let g_m2 = pair.gradient_var_delta_random_all_levels(&cfg, &stats);
        let g_m3 = pair.gradient_m3_delta_random_all_levels(&cfg, &stats);
        let g_s3 = pair.gradient_s3_delta_random_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let m2 = stats[l].var_delta;
            let m3 = stats[l].m3_delta;
            if m2 <= 0.0 { continue; }
            let inv_m2_sq = 1.0 / (m2 * m2);
            let coeff_m2 = -2.0 * m3 / (m2 * m2 * m2);
            for j in 0..n_r {
                let expected = inv_m2_sq * g_m3.random_weight_grads[l][j]
                             + coeff_m2 * g_m2.random_weight_grads[l][j];
                let got = g_s3.random_weight_grads[l][j];
                assert!((got - expected).abs() < 1e-10,
                    "S3 random chain rule mismatch level {} j={}: \
                     manual {} vs method {}",
                    l, j, expected, got);
            }
        }
    }

    #[test]
    fn gradient_higher_moments_random_aggregate_matches_per_level() {
        // Aggregate ↔ per-level identity for m3, m4, s3 random-weight.
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 27111);
        let pts_r = make_uniform(n_r, 5, 27222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let betas: Vec<f64> = (0..stats.len())
            .map(|l| if l % 2 == 0 { 1.0 } else { -0.5 } * (l as f64 + 1.0))
            .collect();

        for (name, full, agg) in [
            ("m3",
             pair.gradient_m3_delta_random_all_levels(&cfg, &stats),
             pair.gradient_m3_delta_random_aggregate(&cfg, &stats, &betas)),
            ("m4",
             pair.gradient_m4_delta_random_all_levels(&cfg, &stats),
             pair.gradient_m4_delta_random_aggregate(&cfg, &stats, &betas)),
            ("s3",
             pair.gradient_s3_delta_random_all_levels(&cfg, &stats),
             pair.gradient_s3_delta_random_aggregate(&cfg, &stats, &betas)),
        ] {
            let mut expected = vec![0.0_f64; n_r];
            for l in 0..stats.len() {
                for (j, &g) in full.random_weight_grads[l].iter().enumerate() {
                    expected[j] += betas[l] * g;
                }
            }
            for j in 0..n_r {
                assert!((agg[j] - expected[j]).abs() < 1e-10,
                    "{} random j={}: aggregate {} vs Σ β_l grad_l {}",
                    name, j, agg[j], expected[j]);
            }
        }
    }

    // -----------------------------------------------------------------
    // Raw-sum gradient: building block for multi-run pooled gradient
    // -----------------------------------------------------------------

    #[test]
    fn raw_sum_gradient_combines_to_var_gradient() {
        // Strict consistency: for each particle i and each level l,
        //   ∂ μ_2 / ∂w_i^d = (1/T) [∂S_2 - 2 m_1 ∂S_1]
        // where ∂S_k come from gradient_raw_sums_data_all_levels and
        // ∂μ_2 from gradient_var_delta_all_levels. Both should agree
        // to f64 precision.
        //
        // This test pins the new raw-sum primitive against the
        // existing variance-gradient primitive and is the foundational
        // check that the multi-run pooled gradient (which uses raw-sum
        // gradients) is numerically equivalent to per-cascade work.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 30111);
        let pts_r = make_uniform(n_r, 5, 30222);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.7 + (j as f64) / 200.0).collect();
        let pair = build_pair(pts_d, pts_r,
            Some(weights_d), Some(weights_r));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let raw = pair.gradient_raw_sums_data_all_levels(&cfg, &stats);
        let var_grad = pair.gradient_var_delta_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let t_l = stats[l].sum_w_r_active;
            let m_1 = stats[l].mean_delta;
            if t_l <= 0.0 { continue; }
            let inv_t = 1.0 / t_l;
            let s1 = &raw.s1_grads[l];
            let s2 = &raw.s2_grads[l];
            let vg = &var_grad.data_weight_grads[l];
            assert_eq!(s1.len(), n_d);
            assert_eq!(s2.len(), n_d);
            assert_eq!(vg.len(), n_d);
            for i in 0..n_d {
                let combined = inv_t * (s2[i] - 2.0 * m_1 * s1[i]);
                let direct = vg[i];
                let diff = (combined - direct).abs();
                let scale = direct.abs().max(1e-10);
                assert!(diff < 1e-10 || diff / scale < 1e-10,
                    "level {} particle {}: combined {} vs direct {} (diff {})",
                    l, i, combined, direct, diff);
            }
        }
    }

    #[test]
    fn raw_sum_gradient_combines_to_higher_moment_gradients() {
        // Strict consistency: combining (∂S_1, ∂S_2, ∂S_3, ∂S_4) via
        // the chain rule recovers the m3 and m4 single-cascade gradients
        // to f64 precision.
        //
        // Chain rule (data weights, where ∂T/∂w_i^d = 0):
        //   ∂μ_3/∂w = (1/T)[∂S_3 − 3 m_1 ∂S_2 − 3(μ_2 − m_1²) ∂S_1]
        //   ∂μ_4/∂w = (1/T)[∂S_4 − 4 m_1 ∂S_3 + 6 m_1² ∂S_2
        //                   − 4(μ_3 + m_1³) ∂S_1]
        //
        // Pins the extended raw-sum primitive against the existing m3/m4
        // single-cascade gradients. Foundation for pooled m3/m4 gradients.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 38111);
        let pts_r = make_uniform(n_r, 5, 38222);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.7 + (j as f64) / 200.0).collect();
        let pair = build_pair(pts_d, pts_r,
            Some(weights_d), Some(weights_r));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let raw = pair.gradient_raw_sums_data_all_levels(&cfg, &stats);
        let m3_grad = pair.gradient_m3_delta_all_levels(&cfg, &stats);
        let m4_grad = pair.gradient_m4_delta_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let t_l = stats[l].sum_w_r_active;
            let m_1 = stats[l].mean_delta;
            let mu_2 = stats[l].var_delta;
            let mu_3 = stats[l].m3_delta;
            if t_l <= 0.0 { continue; }
            let inv_t = 1.0 / t_l;
            let m1_sq = m_1 * m_1;
            let m1_cu = m1_sq * m_1;
            let s1 = &raw.s1_grads[l];
            let s2 = &raw.s2_grads[l];
            let s3 = &raw.s3_grads[l];
            let s4 = &raw.s4_grads[l];
            let g3 = &m3_grad.data_weight_grads[l];
            let g4 = &m4_grad.data_weight_grads[l];
            assert_eq!(s3.len(), n_d);
            assert_eq!(s4.len(), n_d);
            for i in 0..n_d {
                let combined_m3 = inv_t * (s3[i]
                    - 3.0 * m_1 * s2[i]
                    - 3.0 * (mu_2 - m1_sq) * s1[i]);
                let combined_m4 = inv_t * (s4[i]
                    - 4.0 * m_1 * s3[i]
                    + 6.0 * m1_sq * s2[i]
                    - 4.0 * (mu_3 + m1_cu) * s1[i]);
                let direct_m3 = g3[i];
                let direct_m4 = g4[i];
                let diff3 = (combined_m3 - direct_m3).abs();
                let scale3 = direct_m3.abs().max(1e-10);
                assert!(diff3 < 1e-10 || diff3 / scale3 < 1e-9,
                    "m3 level {} particle {}: combined {} vs direct {} (diff {})",
                    l, i, combined_m3, direct_m3, diff3);
                let diff4 = (combined_m4 - direct_m4).abs();
                let scale4 = direct_m4.abs().max(1e-10);
                assert!(diff4 < 1e-10 || diff4 / scale4 < 1e-9,
                    "m4 level {} particle {}: combined {} vs direct {} (diff {})",
                    l, i, combined_m4, direct_m4, diff4);
            }
        }
    }

    #[test]
    fn raw_sum_random_gradient_combines_to_var_gradient() {
        // Strict consistency: combining (∂T, ∂S_1, ∂S_2) via the
        // random-weight chain rule recovers the existing single-cascade
        // random-weight variance gradient.
        //
        // Chain rule (random weights, with ∂T/∂w_j^r ≠ 0):
        //   ∂μ_2/∂w = (1/T)[∂S_2 − 2 m_1 ∂S_1 − (μ_2 − m_1²) ∂T]
        //
        // This pins the new random-weight raw-sum primitive against
        // the existing variance gradient and is the foundational check
        // for the random-weight pooled gradients.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 80111);
        let pts_r = make_uniform(n_r, 5, 80222);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.7 + (j as f64) / 200.0).collect();
        let pair = build_pair(pts_d, pts_r,
            Some(weights_d), Some(weights_r));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let raw = pair.gradient_raw_sums_random_all_levels(&cfg, &stats);
        let var_grad = pair.gradient_var_delta_random_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let t_l = stats[l].sum_w_r_active;
            let m_1 = stats[l].mean_delta;
            let mu_2 = stats[l].var_delta;
            let m1_sq = m_1 * m_1;
            if t_l <= 0.0 { continue; }
            let inv_t = 1.0 / t_l;
            let gt = &raw.t_grads[l];
            let s1 = &raw.s1_grads[l];
            let s2 = &raw.s2_grads[l];
            let vg = &var_grad.random_weight_grads[l];
            assert_eq!(gt.len(), n_r);
            assert_eq!(s1.len(), n_r);
            assert_eq!(s2.len(), n_r);
            assert_eq!(vg.len(), n_r);
            for j in 0..n_r {
                let combined = inv_t * (s2[j] - 2.0 * m_1 * s1[j]
                    - (mu_2 - m1_sq) * gt[j]);
                let direct = vg[j];
                let diff = (combined - direct).abs();
                let scale = direct.abs().max(1e-10);
                assert!(diff < 1e-9 || diff / scale < 1e-9,
                    "level {} particle {}: combined {} vs direct {} (diff {})",
                    l, j, combined, direct, diff);
            }
        }
    }

    #[test]
    fn raw_sum_random_gradient_combines_to_higher_moment_gradients() {
        // Strict consistency: combining (∂T, ∂S_1, ..., ∂S_4) via the
        // random-weight chain rule recovers m3 and m4 single-cascade
        // random gradients.
        //
        // Chain rule (random weights, ∂T ≠ 0):
        //   ∂μ_3/∂w = (1/T)[∂S_3 − 3 m_1 ∂S_2
        //                   − 3(μ_2 − m_1²) ∂S_1
        //                   − (μ_3 − 3 m_1 μ_2 + 3 m_1³ − m_1·something) ∂T]
        //
        // Easier: derive ∂A_k = (1/T)(∂S_k − A_k ∂T), then ∂μ_k via
        // the same algebraic combinations of ∂A_k as the data-weight case.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 81111);
        let pts_r = make_uniform(n_r, 5, 81222);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.7 + (j as f64) / 200.0).collect();
        let pair = build_pair(pts_d, pts_r,
            Some(weights_d), Some(weights_r));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_field_stats(&cfg);
        let raw = pair.gradient_raw_sums_random_all_levels(&cfg, &stats);
        let m3_grad = pair.gradient_m3_delta_random_all_levels(&cfg, &stats);
        let m4_grad = pair.gradient_m4_delta_random_all_levels(&cfg, &stats);

        for l in 0..stats.len() {
            let t_l = stats[l].sum_w_r_active;
            let m_1 = stats[l].mean_delta;
            let mu_2 = stats[l].var_delta;
            let mu_3 = stats[l].m3_delta;
            if t_l <= 0.0 { continue; }
            let inv_t = 1.0 / t_l;
            let m1_sq = m_1 * m_1;
            let m1_cu = m1_sq * m_1;
            let a_2 = (mu_2 + m1_sq);
            let a_3 = (mu_3 + 3.0 * m_1 * mu_2 + m1_cu);
            let gt = &raw.t_grads[l];
            let s1 = &raw.s1_grads[l];
            let s2 = &raw.s2_grads[l];
            let s3 = &raw.s3_grads[l];
            let s4 = &raw.s4_grads[l];
            let g3 = &m3_grad.random_weight_grads[l];
            let g4 = &m4_grad.random_weight_grads[l];
            // ∂A_k = (1/T)(∂S_k − A_k ∂T)
            // m_1 = A_1, ∂m_1 = ∂A_1
            // ∂μ_3 = ∂A_3 − 3 (∂m_1 A_2 + m_1 ∂A_2) + 6 m_1² ∂m_1
            // ∂μ_4 = ∂A_4 − 4 (∂m_1 A_3 + m_1 ∂A_3) + 12 m_1 ∂m_1 A_2
            //         + 6 m_1² ∂A_2 − 12 m_1³ ∂m_1
            for j in 0..n_r {
                let inv = inv_t;
                let d_a1 = inv * (s1[j] - m_1 * gt[j]);
                let d_a2 = inv * (s2[j] - a_2 * gt[j]);
                let d_a3 = inv * (s3[j] - a_3 * gt[j]);
                let a_4 = (stats[l].m4_delta + 4.0 * m_1 * mu_3 + 6.0 * m1_sq * mu_2 + m1_sq * m1_sq);
                let d_a4 = inv * (s4[j] - a_4 * gt[j]);
                let d_m1 = d_a1;
                let combined_m3 = d_a3 - 3.0 * (d_m1 * a_2 + m_1 * d_a2)
                    + 6.0 * m1_sq * d_m1;
                let combined_m4 = d_a4 - 4.0 * (d_m1 * a_3 + m_1 * d_a3)
                    + 12.0 * m_1 * d_m1 * a_2 + 6.0 * m1_sq * d_a2
                    - 12.0 * m1_cu * d_m1;
                let direct_m3 = g3[j];
                let direct_m4 = g4[j];
                let diff3 = (combined_m3 - direct_m3).abs();
                let scale3 = direct_m3.abs().max(1e-10);
                assert!(diff3 < 1e-9 || diff3 / scale3 < 1e-9,
                    "m3 level {} particle {}: combined {} vs direct {} (diff {})",
                    l, j, combined_m3, direct_m3, diff3);
                let diff4 = (combined_m4 - direct_m4).abs();
                let scale4 = direct_m4.abs().max(1e-10);
                assert!(diff4 < 1e-9 || diff4 / scale4 < 1e-8,
                    "m4 level {} particle {}: combined {} vs direct {} (diff {})",
                    l, j, combined_m4, direct_m4, diff4);
            }
        }
    }
}
