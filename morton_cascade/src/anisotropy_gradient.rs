// anisotropy_gradient.rs
//
// Backward-mode gradient of anisotropy moments ⟨w_e²⟩ with respect
// to per-particle data weights. For each level l, each non-trivial
// Haar pattern e ∈ {1, ..., 2^D − 1}, and each data particle i, this
// computes ∂⟨w_e²⟩^(l) / ∂w_i^d.
//
// Math derivation (commit 10 design notes):
//
// Forward statistic. At each level l we form the W_r-weighted mean of
// squared Haar coefficients over all eligible parents:
//
//   ⟨w_e²⟩^(l) = A_e^(l) / T^(l)
//   A_e^(l)    = Σ_p W_r(p) · w_e(p)²
//   T^(l)      = Σ_p W_r(p)
//   w_e(p)     = (1/2^D) Σ_σ (-1)^|e ∧ σ| · δ_σ(p)
//
// where σ ranges over the 2^D children of parent p, |·| is popcount,
// and δ_σ(p) = W_d(c_σ) / (α W_r(c_σ)) − 1 is the density contrast of
// child cell c_σ (a level-(l+1) cell). Eligibility: every child must
// have W_r > w_r_min AND parent must too.
//
// Cell-sensitivity. For a particle i in child cell c_σ_i of parent p_i:
//
//   ∂δ_σ(p) / ∂w_i^d = δ_{σ=σ_i, p=p_i} · 1/(α W_r(c_σ_i))
//
// So:
//   ∂w_e(p_i) / ∂w_i^d = (1/2^D) (-1)^|e ∧ σ_i| / (α W_r(c_σ_i))
//   ∂A_e / ∂w_i^d = W_r(p_i) · 2 w_e(p_i) · (1/2^D) (-1)^|e ∧ σ_i| / (α W_r(c_σ_i))
//   ∂⟨w_e²⟩^(l) / ∂(W_d-cell-direct) =
//     2 w_e(p_i) · W_r(p_i) · (-1)^|e ∧ σ_i| / (2^D α W_r(c_σ_i) T^(l))
//
// Alpha-sensitivity. Using ∂δ_σ/∂α = -(δ_σ + 1)/α and the cancellation
// identity Σ_σ (-1)^|e ∧ σ| = 0 for e ≠ 0:
//
//   ∂w_e(p) / ∂α = -w_e(p) / α
//   ∂⟨w_e²⟩^(l) / ∂α = -2 ⟨w_e²⟩^(l) / α
//
// Per-particle gradient (chain rule, ∂α/∂w_i^d = 1/Σw_r):
//
//   ∂⟨w_e²⟩^(l) / ∂w_i^d
//     = 2 w_e(p_i) · W_r(p_i) · (-1)^|e ∧ σ_i| / (2^D α W_r(c_σ_i) T^(l))
//       − 2 ⟨w_e²⟩^(l) / (α · Σw_r)
//
// The first term is per-particle (depends on parent + child position).
// The second is constant across all particles for a given (l, e).
//
// Implementation:
//   1. Walk parents at level l via the data-cell membership index.
//   2. For each parent, look up its 2^D child cells' W_d, W_r;
//      compute child δ_σ, then w_e(p) for each pattern e.
//   3. Skip ineligible parents (any child or parent below w_r_min).
//   4. For each eligible parent, distribute the per-particle term
//      to particles in each child cell (one term per (e, child)).
//   5. Add the global α-term to every particle.
//
// Cost: O(L_max · n_parents · (2^D)²). For typical D=3 the inner
// factor is 64, modest. Total: same order as one anisotropy forward
// pass.

use crate::cell_membership::{CellMembership, WhichCatalog};
use crate::hier_bitvec_pair::{
    AnisotropyStats, BitVecCascadePair, FieldStatsConfig,
};

/// Per-(level, pattern) data-weight gradient of anisotropy moments.
///
/// Indexed as `pattern_grads[level][pattern_e][particle_idx]`.
/// `pattern_e` ranges over `0..2^D`; index 0 is empty (constant
/// pattern is identically zero by construction, so its gradient is
/// trivially zero).
#[derive(Clone, Debug)]
pub struct AnisotropyGradient {
    pub pattern_grads: Vec<Vec<Vec<f64>>>,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Per-particle gradient of ⟨w_e²⟩ at every (level, non-trivial
    /// pattern) with respect to the data-catalog weights. The result
    /// is `pattern_grads[level][pattern_e][particle_idx]`. Slot
    /// `pattern_grads[l][0]` is empty since pattern 0 is the constant.
    ///
    /// Memory: O(L_max · 2^D · N_d). For large catalogs use the
    /// [`Self::gradient_anisotropy_aggregate`] variant.
    pub fn gradient_anisotropy_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[AnisotropyStats],
    ) -> AnisotropyGradient {
        let n_d = self.n_d();
        let n_levels = results.len();
        let n_patterns = 1usize << D;

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return AnisotropyGradient {
                pattern_grads: vec![vec![Vec::new(); n_patterns]; n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut grads: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            grads.push(per_level_anisotropy_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_d, n_patterns,
            ));
        }
        AnisotropyGradient { pattern_grads: grads }
    }

    /// Aggregate-scalar gradient: given a `(level, pattern)` weight
    /// matrix `betas` of shape `[n_levels][2^D]`, compute
    /// `∂L / ∂w_i^d` for the scalar loss
    ///
    ///   L = Σ_l Σ_e β_{l,e} · ⟨w_e²⟩^(l)
    ///
    /// Returns one length-N_d vector. Pattern 0's coefficients are
    /// silently ignored (its contribution is zero by construction).
    ///
    /// Common cosmology use cases:
    ///   - LoS-only: set β_{l, 1<<(D-1)} = 1; everything else 0.
    ///   - Quadrupole: set β_{l, 1<<(D-1)} = 1, β_{l, 1<<d} = -1/(D-1)
    ///     for d ≠ D-1.
    pub fn gradient_anisotropy_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[AnisotropyStats],
        betas: &[Vec<f64>],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} does not match results length {}",
            betas.len(), results.len());
        let n_patterns = 1usize << D;
        for (l, b) in betas.iter().enumerate() {
            assert_eq!(b.len(), n_patterns,
                "betas[{}] length {} != 2^D = {}", l, b.len(), n_patterns);
        }

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
            // Skip levels with no nonzero betas — saves a parent walk.
            let any_nonzero = betas[l].iter().any(|&b| b != 0.0);
            if !any_nonzero { continue; }
            let per_level = per_level_anisotropy_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_d, n_patterns,
            );
            for e in 1..n_patterns {
                if betas[l][e] == 0.0 { continue; }
                let coef = betas[l][e];
                for (i, &g) in per_level[e].iter().enumerate() {
                    out[i] += coef * g;
                }
            }
        }
        out
    }
}

/// Per-(level, pattern) random-weight gradient of anisotropy moments.
///
/// Indexed as `pattern_grads[level][pattern_e][random_particle_idx]`.
/// `pattern_e` ranges over `0..2^D`; index 0 is empty (constant pattern
/// is identically zero by construction).
#[derive(Clone, Debug)]
pub struct AnisotropyRandomGradient {
    pub pattern_grads: Vec<Vec<Vec<f64>>>,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Per-particle gradient of `⟨w_e²⟩` at every (level, non-trivial
    /// pattern) with respect to the **random-catalog** weights.
    ///
    /// Math (see `docs/differentiable_cascade.md` §6.12):
    ///
    /// ```text
    ///   ∂⟨w_e²⟩^(ℓ) / ∂w_j^r = 2 ⟨w_e²⟩^(ℓ) / Σ w_r
    ///                        + (1/T^(ℓ)) [ w_e(p_j)² − ⟨w_e²⟩^(ℓ)
    ///                                     − 2 W_r(p_j) w_e(p_j) (-1)^|e ∧ σ_j|
    ///                                       (δ_σ_j(p_j) + 1) / (2^D W_r(c_σ_j)) ]
    /// ```
    ///
    /// where p_j is the level-ℓ parent of random particle j and σ_j is
    /// j's child position (level ℓ+1) inside that parent. Path-2
    /// (the bracket) is included only when p_j is eligible (all 2^D
    /// children in footprint).
    ///
    /// The result is `pattern_grads[level][pattern_e][random_particle_idx]`.
    /// Slot `pattern_grads[l][0]` is empty (constant pattern).
    pub fn gradient_anisotropy_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[AnisotropyStats],
    ) -> AnisotropyRandomGradient {
        let n_r = self.n_r();
        let n_levels = results.len();
        let n_patterns = 1usize << D;

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        if total_w_r <= 0.0 || total_w_d <= 0.0 {
            return AnisotropyRandomGradient {
                pattern_grads: vec![vec![Vec::new(); n_patterns]; n_levels],
            };
        }
        let alpha = total_w_d / total_w_r;

        let mut grads: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            grads.push(per_level_anisotropy_random_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_r, n_patterns,
            ));
        }
        AnisotropyRandomGradient { pattern_grads: grads }
    }

    /// Per-level random-weight gradient of $T^{(\ell)} = $
    /// `sum_w_r_parents` (the eligible-parent random-weight sum used
    /// as the denominator in $\overline{w_e^2}$).
    ///
    /// `t_grads[level][random_particle_idx] = ∂T^{(ℓ)} / ∂w_j^r`.
    /// Equals 1 if `j`'s level-`ℓ` parent cell is fully eligible
    /// (parent in footprint and all 2^D children in footprint), else 0.
    ///
    /// Used by the multi-run pooled anisotropy random-weight gradient.
    /// Single-cascade users typically don't need this directly — the
    /// existing `gradient_anisotropy_random_all_levels` already
    /// includes the full chain-rule effect on $\overline{w_e^2}$.
    pub fn gradient_anisotropy_t_random_all_levels(
        &self,
        cfg: &FieldStatsConfig,
        results: &[AnisotropyStats],
    ) -> Vec<Vec<f64>> {
        let n_r = self.n_r();
        let n_levels = results.len();
        let n_children = 1usize << D;

        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        let mut t_grads: Vec<Vec<f64>> = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let stats = &results[l];
            if stats.sum_w_r_parents <= 0.0 {
                t_grads.push(vec![0.0_f64; n_r]);
                continue;
            }
            let mut g = vec![0.0_f64; n_r];
            for (parent_id, parent_members_r) in mem_r.non_empty_cells_at(l) {
                let parent_w_r: f64 = match self.weights_r() {
                    Some(w) => parent_members_r.iter().map(|&i| w[i as usize]).sum(),
                    None => parent_members_r.len() as f64,
                };
                if parent_w_r <= cfg.w_r_min { continue; }

                // Check eligibility: all 2^D children in footprint.
                let level_child = l + 1;
                let mut all_eligible = true;
                for child_offset in 0..n_children {
                    let child_id = (parent_id << D) | (child_offset as u64);
                    let r_members = mem_r.members(level_child, child_id);
                    let w_r: f64 = match self.weights_r() {
                        Some(w) => r_members.iter().map(|&i| w[i as usize]).sum(),
                        None => r_members.len() as f64,
                    };
                    if w_r <= cfg.w_r_min { all_eligible = false; break; }
                }
                if !all_eligible { continue; }

                // Credit +1 to every random particle in this parent.
                // The contribution of j to T^(ℓ) at this parent is W_r(parent)
                // when j is in the parent. ∂(parent_W_r)/∂w_j^r = 1 iff
                // j ∈ parent. So ∂T/∂w_j^r += 1 for each j in this eligible parent.
                for &j in parent_members_r {
                    g[j as usize] += 1.0;
                }
            }
            t_grads.push(g);
        }
        t_grads
    }

    /// Aggregate-scalar gradient for anisotropy with respect to random
    /// weights. `betas` shape `[n_levels][2^D]`; pattern-0 slot ignored.
    pub fn gradient_anisotropy_random_aggregate(
        &self,
        cfg: &FieldStatsConfig,
        results: &[AnisotropyStats],
        betas: &[Vec<f64>],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), results.len(),
            "betas length {} does not match results length {}",
            betas.len(), results.len());
        let n_patterns = 1usize << D;
        for (l, b) in betas.iter().enumerate() {
            assert_eq!(b.len(), n_patterns,
                "betas[{}] length {} != 2^D = {}", l, b.len(), n_patterns);
        }

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
            let any_nonzero = betas[l].iter().any(|&b| b != 0.0);
            if !any_nonzero { continue; }
            let per_level = per_level_anisotropy_random_gradient(
                self, &mem_d, &mem_r, l, results, alpha, total_w_r, cfg, n_r, n_patterns,
            );
            for e in 1..n_patterns {
                if betas[l][e] == 0.0 { continue; }
                let coef = betas[l][e];
                for (j, &g) in per_level[e].iter().enumerate() {
                    out[j] += coef * g;
                }
            }
        }
        out
    }
}

/// Per-pattern random-weight gradient at a single level. Returns
/// `Vec<Vec<f64>>` indexed `[pattern_e][random_particle_idx]`.
/// Slot `[0]` is empty (constant pattern unused).
fn per_level_anisotropy_random_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[AnisotropyStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_r: usize,
    n_patterns: usize,
) -> Vec<Vec<f64>> {
    let stats = &results[level];
    let t_l = stats.sum_w_r_parents;

    // Allocate per-pattern gradient vectors. Slot [0] stays empty.
    let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
    grads.push(Vec::new());  // pattern 0 unused
    for _ in 1..n_patterns {
        grads.push(vec![0.0_f64; n_r]);
    }

    // Constant α-term: same for every random particle, applies
    // regardless of whether j's parent is eligible.
    if alpha > 0.0 && total_w_r > 0.0 {
        for e in 1..n_patterns {
            let mean_we_sq = stats.mean_w_squared_by_pattern.get(e)
                .copied().unwrap_or(0.0);
            let alpha_term = 2.0 * mean_we_sq / total_w_r;
            for g in grads[e].iter_mut() { *g = alpha_term; }
        }
    }

    if alpha <= 0.0 || t_l <= 0.0 {
        return grads;
    }

    // Path-2: walk eligible parents, distribute per-particle terms to
    // random particles in each child. Same walk as the data-weight
    // anisotropy gradient.
    let n_children = 1usize << D;
    let inv_two_d = 1.0 / (n_children as f64);
    let inv_t = 1.0 / t_l;

    for (parent_id, parent_members_r) in mem_r.non_empty_cells_at(level) {
        let parent_w_r: f64 = match pair.weights_r() {
            Some(w) => parent_members_r.iter().map(|&j| w[j as usize]).sum(),
            None => parent_members_r.len() as f64,
        };
        if parent_w_r <= cfg.w_r_min { continue; }

        // Compute each child's W_d, W_r, δ. Track child member lists.
        let mut child_w_r = [0.0_f64; 256];
        let mut child_delta = [0.0_f64; 256];
        let mut child_r_members: [&[u32]; 256] = [&[]; 256];
        debug_assert!(n_children <= 256, "anisotropy random gradient assumes D ≤ 8");

        let mut all_eligible = true;
        for child_offset in 0..n_children {
            let child_id = (parent_id << D) | (child_offset as u64);
            let level_child = level + 1;
            let r_members = mem_r.members(level_child, child_id);
            let d_members = mem_d.members(level_child, child_id);
            let w_r: f64 = match pair.weights_r() {
                Some(w) => r_members.iter().map(|&j| w[j as usize]).sum(),
                None => r_members.len() as f64,
            };
            let w_d: f64 = match pair.weights_d() {
                Some(w) => d_members.iter().map(|&i| w[i as usize]).sum(),
                None => d_members.len() as f64,
            };
            if w_r <= cfg.w_r_min { all_eligible = false; break; }
            child_w_r[child_offset] = w_r;
            child_delta[child_offset] = w_d / (alpha * w_r) - 1.0;
            child_r_members[child_offset] = r_members;
        }
        if !all_eligible { continue; }

        // Compute w_e(p) for each pattern.
        let mut w_e = vec![0.0_f64; n_patterns];
        for e in 1..n_patterns {
            let mut acc = 0.0_f64;
            for sigma in 0..n_children {
                let parity = ((e & sigma) as u32).count_ones() % 2;
                let sign = if parity == 0 { 1.0 } else { -1.0 };
                acc += sign * child_delta[sigma];
            }
            w_e[e] = acc * inv_two_d;
        }

        // Path-2 contribution to each random particle in each child.
        // For pattern e and child σ:
        //   pattern-independent part:  (w_e(p)² − ⟨w_e²⟩) / T
        //   pattern-and-child-specific part:
        //       − 2 W_r(p) w_e(p) sign(e,σ) (δ_σ + 1) / (2^D W_r(c_σ) T)
        for sigma in 0..n_children {
            let members = child_r_members[sigma];
            if members.is_empty() { continue; }
            let inv_w_r_child = 1.0 / child_w_r[sigma];
            let delta_sigma = child_delta[sigma];
            for e in 1..n_patterns {
                let mean_we_sq = stats.mean_w_squared_by_pattern.get(e)
                    .copied().unwrap_or(0.0);
                let parity = ((e & sigma) as u32).count_ones() % 2;
                let sign = if parity == 0 { 1.0 } else { -1.0 };
                let pattern_indep = inv_t * (w_e[e] * w_e[e] - mean_we_sq);
                let pattern_dep = -2.0 * parent_w_r * w_e[e] * sign
                                  * (delta_sigma + 1.0)
                                  * inv_two_d * inv_w_r_child * inv_t;
                let term = pattern_indep + pattern_dep;
                for &j in members {
                    grads[e][j as usize] += term;
                }
            }
        }
    }

    grads
}

/// Per-pattern gradient at a single level. Returns `Vec<Vec<f64>>`
/// indexed `[pattern_e][particle_idx]`. Slot `[0]` is empty.
fn per_level_anisotropy_gradient<const D: usize>(
    pair: &BitVecCascadePair<D>,
    mem_d: &CellMembership,
    mem_r: &CellMembership,
    level: usize,
    results: &[AnisotropyStats],
    alpha: f64,
    total_w_r: f64,
    cfg: &FieldStatsConfig,
    n_d: usize,
    n_patterns: usize,
) -> Vec<Vec<f64>> {
    let stats = &results[level];
    let t_l = stats.sum_w_r_parents;

    // Allocate per-pattern gradient vectors. Slot [0] stays empty.
    let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n_patterns);
    grads.push(Vec::new());  // pattern 0 unused
    for _ in 1..n_patterns {
        grads.push(vec![0.0_f64; n_d]);
    }

    // Constant α-sensitivity term: ∂⟨w_e²⟩/∂α = -2⟨w_e²⟩/α; chain via
    // ∂α/∂w_i^d = 1/Σw_r gives a uniform per-particle term per pattern.
    if alpha > 0.0 && total_w_r > 0.0 {
        for e in 1..n_patterns {
            let mean_we_sq = stats.mean_w_squared_by_pattern.get(e)
                .copied().unwrap_or(0.0);
            let alpha_term = -2.0 * mean_we_sq / (alpha * total_w_r);
            for g in grads[e].iter_mut() { *g = alpha_term; }
        }
    }

    // Cell-sensitivity term: walk parents at level `level`, check
    // eligibility, distribute per-particle terms.
    if alpha <= 0.0 || t_l <= 0.0 {
        return grads;
    }

    // Need parents at level `level`. The membership index iterates
    // non-empty data cells. To enumerate parents that have ALL 2^D
    // children eligible (regardless of whether each child has data),
    // we should iterate parents via the random catalog (since
    // eligibility is keyed on W_r > w_r_min for parent and every
    // child). But we only need to credit gradients to data particles.
    //
    // Strategy: iterate parents via the random membership (covers all
    // parents that could possibly be eligible — empty parents have
    // W_r = 0 and are ineligible by definition). For each parent,
    // look up its data-membership too.
    let n_children = 1usize << D;
    let inv_two_d = 1.0 / (n_children as f64);
    let inv_alpha = 1.0 / alpha;

    for (parent_id, parent_members_r) in mem_r.non_empty_cells_at(level) {
        // Parent W_r and footprint check.
        let parent_w_r: f64 = match pair.weights_r() {
            Some(w) => parent_members_r.iter().map(|&i| w[i as usize]).sum(),
            None => parent_members_r.len() as f64,
        };
        if parent_w_r <= cfg.w_r_min { continue; }

        // Compute each child's W_d, W_r, δ. Track child member lists.
        let mut child_w_r = [0.0_f64; 256];   // arbitrary cap for D ≤ 8
        let mut child_w_d = [0.0_f64; 256];
        let mut child_delta = [0.0_f64; 256];
        let mut child_d_members: [&[u32]; 256] = [&[]; 256];
        debug_assert!(n_children <= 256, "anisotropy gradient assumes D ≤ 8");

        let mut all_eligible = true;
        for child_offset in 0..n_children {
            let child_id = (parent_id << D) | (child_offset as u64);
            let level_child = level + 1;
            let r_members = mem_r.members(level_child, child_id);
            let d_members = mem_d.members(level_child, child_id);
            let w_r: f64 = match pair.weights_r() {
                Some(w) => r_members.iter().map(|&i| w[i as usize]).sum(),
                None => r_members.len() as f64,
            };
            let w_d: f64 = match pair.weights_d() {
                Some(w) => d_members.iter().map(|&i| w[i as usize]).sum(),
                None => d_members.len() as f64,
            };
            // Strict-mode footprint: every child must pass.
            if w_r <= cfg.w_r_min { all_eligible = false; break; }
            child_w_r[child_offset] = w_r;
            child_w_d[child_offset] = w_d;
            child_delta[child_offset] = w_d / (alpha * w_r) - 1.0;
            child_d_members[child_offset] = d_members;
        }
        if !all_eligible { continue; }

        // Compute w_e(p) for each pattern e ∈ 1..2^D.
        let mut w_e = vec![0.0_f64; n_patterns];
        for e in 1..n_patterns {
            let mut acc = 0.0_f64;
            for sigma in 0..n_children {
                let parity = ((e & sigma) as u32).count_ones() % 2;
                let sign = if parity == 0 { 1.0 } else { -1.0 };
                acc += sign * child_delta[sigma];
            }
            w_e[e] = acc * inv_two_d;
        }

        // Distribute per-particle gradient term:
        //   term(e, σ) = 2 w_e(p) · W_r(p) · sign(e, σ) / (2^D α W_r(c_σ) T^(l))
        // for every particle i in child cell σ. The W_r(p) factor enters
        // because A_e = Σ_p W_r(p) · w_e(p)², so ∂A_e/∂W_d(c) carries
        // the parent's random weight as a multiplier.
        for sigma in 0..n_children {
            let members = child_d_members[sigma];
            if members.is_empty() { continue; }
            let inv_w_r_child = 1.0 / child_w_r[sigma];
            for e in 1..n_patterns {
                let parity = ((e & sigma) as u32).count_ones() % 2;
                let sign = if parity == 0 { 1.0 } else { -1.0 };
                let term = 2.0 * w_e[e] * parent_w_r * sign
                           * inv_two_d * inv_alpha * inv_w_r_child / t_l;
                for &i in members {
                    grads[e][i as usize] += term;
                }
            }
        }
    }

    grads
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
    fn anisotropy_gradient_finite_difference_parity() {
        // FD parity for the per-pattern anisotropy moments. Same
        // template as field-stats: perturb each weight, recompute
        // ⟨w_e²⟩ per pattern, compare to analytic gradient.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 9444);
        let pts_r = make_uniform(n_r, bits, 9555);
        let wd_base = vec![1.0_f64; n_d];
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_anisotropy(&cfg);
        let grad = pair_base.gradient_anisotropy_all_levels(&cfg, &stats_base);

        // Find a (level, pattern) with measurable signal.
        let mut test_lvl_pat = None;
        for (l, s) in stats_base.iter().enumerate() {
            if s.n_parents < 2 { continue; }
            for e in 1..(1usize << 3) {
                let v = s.mean_w_squared_by_pattern[e];
                if v > 1e-3 {
                    test_lvl_pat = Some((l, e));
                    break;
                }
            }
            if test_lvl_pat.is_some() { break; }
        }
        let (test_level, test_pattern) = match test_lvl_pat {
            Some(lp) => lp,
            None => return,
        };

        let eps = 1e-5;
        let v_base = stats_base[test_level].mean_w_squared_by_pattern[test_pattern];

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze_anisotropy(&cfg);
            let v_pert = stats_pert[test_level].mean_w_squared_by_pattern[test_pattern];
            let fd_grad = (v_pert - v_base) / eps;
            let an_grad = grad.pattern_grads[test_level][test_pattern][i];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "pattern {} at level {}: particle {}: FD {} vs analytic {} \
                 (abs diff {})",
                test_pattern, test_level, i, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn anisotropy_gradient_uniform_scaling_invariant() {
        // ⟨w_e²⟩ depends on δ which is invariant under uniform scaling
        // of data weights → Σ_i w_i · ∂⟨w_e²⟩/∂w_i = 0 for all (l, e).
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 9777);
        let pts_r = make_uniform(n_r, 5, 9888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let grad = pair.gradient_anisotropy_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        for (l, level_grads) in grad.pattern_grads.iter().enumerate() {
            for e in 1..n_patterns {
                let gs = &level_grads[e];
                if gs.is_empty() { continue; }
                let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                    .map(|(g, w)| g * w).sum();
                assert!(weighted_sum.abs() < 1e-9,
                    "level {} pattern {}: Σ w_i · ∂⟨w_e²⟩/∂w_i = {} \
                     (should be ≈ 0 by uniform-scaling invariance)",
                    l, e, weighted_sum);
            }
        }
    }

    #[test]
    fn anisotropy_gradient_aggregate_matches_per_level_combination() {
        // The aggregate API with arbitrary betas must equal
        // Σ_l Σ_e β_{l,e} · per_level_grad[l][e].
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 1411);
        let pts_r = make_uniform(n_r, 5, 2422);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let grad = pair.gradient_anisotropy_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        // Distinctive betas per (level, pattern) — alternating sign.
        let betas: Vec<Vec<f64>> = (0..stats.len())
            .map(|l| (0..n_patterns)
                .map(|e| {
                    if e == 0 { 0.0 }
                    else if (l + e) % 2 == 0 { l as f64 + 1.0 }
                    else { -0.5 * (e as f64) }
                })
                .collect())
            .collect();
        let agg = pair.gradient_anisotropy_aggregate(&cfg, &stats, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for l in 0..stats.len() {
            for e in 1..n_patterns {
                let coef = betas[l][e];
                if coef == 0.0 { continue; }
                let gs = &grad.pattern_grads[l][e];
                if gs.is_empty() { continue; }
                for (i, &g) in gs.iter().enumerate() {
                    expected[i] += coef * g;
                }
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β_l,e grad_l,e {}",
                i, agg[i], expected[i]);
        }
    }

    #[test]
    fn anisotropy_gradient_shape() {
        let n_d = 80;
        let n_r = 240;
        let pts_d = make_uniform(n_d, 5, 3333);
        let pts_r = make_uniform(n_r, 5, 4444);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let grad = pair.gradient_anisotropy_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        assert_eq!(grad.pattern_grads.len(), stats.len());
        for level_grads in &grad.pattern_grads {
            assert_eq!(level_grads.len(), n_patterns);
            // Slot 0 is empty (constant pattern unused).
            assert!(level_grads[0].is_empty());
            for e in 1..n_patterns {
                assert!(level_grads[e].is_empty() || level_grads[e].len() == n_d,
                    "pattern {}: gradient length {} (expected 0 or {})",
                    e, level_grads[e].len(), n_d);
            }
        }
    }

    // -----------------------------------------------------------------
    // Commit 12c: anisotropy random-weight gradients
    // -----------------------------------------------------------------

    #[test]
    fn anisotropy_random_gradient_finite_difference_parity() {
        // FD parity for ∂⟨w_e²⟩/∂w_j^r at a (level, pattern) with
        // measurable signal. The full chain rule combines local
        // child-W_r effect, parent-W_r effect (via T), and global α.
        let bits = 4u32;
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, bits, 35444);
        let pts_r = make_uniform(n_r, bits, 35555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats_base = pair_base.analyze_anisotropy(&cfg);
        let grad = pair_base.gradient_anisotropy_random_all_levels(&cfg, &stats_base);

        // Find a (level, pattern) with measurable signal.
        let mut test_lvl_pat = None;
        for (l, s) in stats_base.iter().enumerate() {
            if s.n_parents < 2 { continue; }
            for e in 1..(1usize << 3) {
                let v = s.mean_w_squared_by_pattern[e];
                if v > 1e-3 {
                    test_lvl_pat = Some((l, e));
                    break;
                }
            }
            if test_lvl_pat.is_some() { break; }
        }
        let (test_level, test_pattern) = match test_lvl_pat {
            Some(lp) => lp,
            None => return,
        };

        let eps = 1e-5;
        let v_base = stats_base[test_level].mean_w_squared_by_pattern[test_pattern];

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_anisotropy(&cfg);
            let v_pert = stats_pert[test_level].mean_w_squared_by_pattern[test_pattern];
            let fd_grad = (v_pert - v_base) / eps;
            let an_grad = grad.pattern_grads[test_level][test_pattern][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "pattern {} at level {}: random j={}: FD {} vs analytic {} \
                 (abs diff {})",
                test_pattern, test_level, j, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn anisotropy_random_gradient_uniform_scaling_invariant() {
        // ⟨w_e²⟩ invariant under uniform scaling of random weights:
        //   w_j^r → k w_j^r ⇒ α → α/k, all W_r(c) → k W_r(c) and
        //   W_r(p) → k W_r(p), δ(c) unchanged ⇒ w_e(p) unchanged ⇒
        //   numerator and denominator of ⟨w_e²⟩ both scale by k ⇒
        //   ratio unchanged.
        // Therefore Σ_j w_j^r · ∂⟨w_e²⟩/∂w_j^r = 0 exactly.
        let n_d = 50;
        let n_r = 150;
        let pts_d = make_uniform(n_d, 5, 36777);
        let pts_r = make_uniform(n_r, 5, 36888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 75.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let grad = pair.gradient_anisotropy_random_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        for (l, level_grads) in grad.pattern_grads.iter().enumerate() {
            for e in 1..n_patterns {
                let gs = &level_grads[e];
                if gs.is_empty() { continue; }
                let weighted_sum: f64 = gs.iter().zip(weights_r.iter())
                    .map(|(g, w)| g * w).sum();
                assert!(weighted_sum.abs() < 1e-9,
                    "level {} pattern {}: Σ w_j^r · ∂⟨w_e²⟩/∂w_j^r = {} \
                     (should be ≈ 0)",
                    l, e, weighted_sum);
            }
        }
    }

    #[test]
    fn anisotropy_random_aggregate_matches_per_level() {
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 37111);
        let pts_r = make_uniform(n_r, 5, 37222);
        let pair = build_pair(pts_d, pts_r, None, None);
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let grad = pair.gradient_anisotropy_random_all_levels(&cfg, &stats);

        let n_patterns = 1usize << 3;
        let betas: Vec<Vec<f64>> = (0..stats.len())
            .map(|l| (0..n_patterns)
                .map(|e| {
                    if e == 0 { 0.0 }
                    else if (l + e) % 2 == 0 { l as f64 + 1.0 }
                    else { -0.5 * (e as f64) }
                })
                .collect())
            .collect();
        let agg = pair.gradient_anisotropy_random_aggregate(&cfg, &stats, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for l in 0..stats.len() {
            for e in 1..n_patterns {
                let coef = betas[l][e];
                if coef == 0.0 { continue; }
                let gs = &grad.pattern_grads[l][e];
                if gs.is_empty() { continue; }
                for (j, &g) in gs.iter().enumerate() {
                    expected[j] += coef * g;
                }
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random j={}: aggregate {} vs Σ β_l,e grad_l,e {}",
                j, agg[j], expected[j]);
        }
    }

    #[test]
    fn anisotropy_t_random_gradient_finite_difference_parity() {
        // ∂T^(ℓ) / ∂w_j^r should equal 1 if j's parent at level ℓ is
        // fully eligible, else 0. FD parity against analyze_anisotropy
        // confirms the eligibility logic matches the forward exactly.
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, 5, 99111);
        let pts_r = make_uniform(n_r, 5, 99222);
        let wd = vec![1.0_f64; n_d];
        let wr_base = vec![1.0_f64; n_r];
        let pair = build_pair(pts_d.clone(), pts_r.clone(),
            Some(wd.clone()), Some(wr_base.clone()));
        let cfg = FieldStatsConfig {
            w_r_min: 0.0, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0,
            compensated_sums: false,
        };
        let stats = pair.analyze_anisotropy(&cfg);
        let analytic = pair.gradient_anisotropy_t_random_all_levels(&cfg, &stats);

        // FD: perturb each wr[j], rebuild the pair, recompute
        // sum_w_r_parents.
        let eps = 1e-4_f64;
        // Pick a level with measurable T.
        let mut chosen_l: Option<usize> = None;
        for (l, s) in stats.iter().enumerate() {
            if s.sum_w_r_parents > 5.0 && s.n_parents >= 2 {
                chosen_l = Some(l);
                break;
            }
        }
        let lx = match chosen_l { Some(l) => l, None => return };
        let t_base = stats[lx].sum_w_r_parents;
        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze_anisotropy(&cfg);
            let t_pert = stats_pert[lx].sum_w_r_parents;
            let fd = (t_pert - t_base) / eps;
            let an = analytic[lx][j];
            // The eligibility-flip discontinuity is rare and we use unit
            // weights well above w_r_min, so FD should match analytic
            // to f64 precision on the "smooth" particles. Allow a small
            // tolerance for the few particles near a threshold.
            let abs_diff = (fd - an).abs();
            assert!(abs_diff < 1e-6,
                "level {} random {}: FD {} vs analytic {} (diff {})",
                lx, j, fd, an, abs_diff);
        }
    }
}
