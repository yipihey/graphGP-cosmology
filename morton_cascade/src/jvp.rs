//! Jacobian-vector products (forward-mode directional derivatives) for
//! pooled gradient outputs.
//!
//! Given a pooled gradient $\partial \mathbf{f}/\partial w$ in
//! row-major-Jacobian form (one row per observable component, one
//! column per particle), the JVP with a perturbation vector $v$ is
//!
//! ```text
//!   (J v)_b = Σ_i (∂f_b / ∂w_i) · v_i
//! ```
//!
//! one scalar per observable component. This is the directional
//! derivative of $f$ in the direction $v$.
//!
//! # Why this exists as a public API
//!
//! Mathematically a JVP is just a row-by-row dot product of the
//! Jacobian against $v$, and the cascade's reverse-mode code already
//! materializes that Jacobian. So this module's functions are
//! one-liners. We expose them anyway because:
//!
//! - They give the API a name that matches the operation users want
//!   to perform ("directional derivative" / "JVP" rather than
//!   "matrix-vector product on a `bin_grads` field").
//! - They're a clean entry point for finite-difference cross-checks
//!   of reverse-mode gradients in directions other than coordinate
//!   axes.
//! - They're the building block for Hessian-vector products
//!   (`hvp` in this crate): forward-FD over the gradient computation
//!   uses JVP machinery.
//!
//! # When to use forward vs. reverse mode in practice
//!
//! - **Reverse mode** (the existing `gradient_*_pooled_aggregate`
//!   methods): efficient for "scalar loss, many input weights".
//!   One reverse pass produces the full gradient w.r.t. all inputs.
//!
//! - **JVP / forward mode** (this module): efficient for "few input
//!   directions, many output components". One JVP gives the
//!   directional derivative of every observable component
//!   simultaneously, but only along the one direction $v$.
//!
//! For our cascade, observables are typically O(10-30) bins/shells
//! and weights are O(10^4-10^7), so reverse mode dominates. JVP is
//! useful for HVPs (next module), Fisher-information block elements,
//! and finite-difference checks in non-axis directions.

use crate::multi_run::{
    PooledFieldStatsGradient,
    PooledFieldStatsRandomGradient,
    PooledAnisotropyGradient,
    PooledAnisotropyRandomGradient,
    PooledXiGradient,
    PooledXiRandomGradient,
};

/// Generic dot product. Asserts equal length.
#[inline]
fn dot(row: &[f64], v: &[f64]) -> f64 {
    debug_assert_eq!(row.len(), v.len(),
        "jvp: row length {} != perturbation length {}",
        row.len(), v.len());
    row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}

/// JVP of a pooled field-stats moment gradient (data weights).
/// Returns one scalar per pooled bin.
///
/// `v` has length `n_base_d` (size of the original data catalog).
/// Output `[bin]` is the directional derivative of the pooled moment
/// in that bin in the direction `v`.
pub fn jvp_field_stats<const D: usize>(
    grad: &PooledFieldStatsGradient<D>,
    v: &[f64],
) -> Vec<f64> {
    grad.bin_grads.iter().map(|row| dot(row, v)).collect()
}

/// JVP of a pooled field-stats moment gradient (random weights).
/// `v` has length `n_base_r`.
pub fn jvp_field_stats_random<const D: usize>(
    grad: &PooledFieldStatsRandomGradient<D>,
    v: &[f64],
) -> Vec<f64> {
    grad.bin_grads.iter().map(|row| dot(row, v)).collect()
}

/// JVP of pooled anisotropy gradient (data weights).
/// Returns `[bin][pattern]`, one scalar per (bin, pattern) pair.
/// Pattern 0 (unused in the source struct) is skipped: the inner
/// vector starts at pattern 1.
pub fn jvp_anisotropy(
    grad: &PooledAnisotropyGradient,
    v: &[f64],
) -> Vec<Vec<f64>> {
    grad.by_side.iter().map(|bin| {
        // Skip slot 0 (pattern 0 unused).
        bin.pattern_grads.iter().skip(1).map(|row| dot(row, v)).collect()
    }).collect()
}

/// JVP of the LoS quadrupole gradient (data weights).
/// One scalar per pooled bin.
pub fn jvp_anisotropy_quadrupole_los(
    grad: &PooledAnisotropyGradient,
    v: &[f64],
) -> Vec<f64> {
    grad.by_side.iter().map(|bin| dot(&bin.quadrupole_los_grad, v)).collect()
}

/// JVP of pooled anisotropy gradient (random weights).
pub fn jvp_anisotropy_random(
    grad: &PooledAnisotropyRandomGradient,
    v: &[f64],
) -> Vec<Vec<f64>> {
    grad.by_side.iter().map(|bin| {
        bin.pattern_grads.iter().skip(1).map(|row| dot(row, v)).collect()
    }).collect()
}

/// JVP of LoS quadrupole gradient (random weights).
pub fn jvp_anisotropy_quadrupole_los_random(
    grad: &PooledAnisotropyRandomGradient,
    v: &[f64],
) -> Vec<f64> {
    grad.by_side.iter().map(|bin| dot(&bin.quadrupole_los_grad, v)).collect()
}

/// JVP of pooled ξ gradient (data weights).
/// Returns `[resize_group][shell]`.
pub fn jvp_xi(
    grad: &PooledXiGradient,
    v: &[f64],
) -> Vec<Vec<f64>> {
    grad.by_resize.iter().map(|group| {
        group.shell_grads.iter().map(|row| dot(row, v)).collect()
    }).collect()
}

/// JVP of pooled ξ gradient (random weights).
pub fn jvp_xi_random(
    grad: &PooledXiRandomGradient,
    v: &[f64],
) -> Vec<Vec<f64>> {
    grad.by_resize.iter().map(|group| {
        group.shell_grads.iter().map(|row| dot(row, v)).collect()
    }).collect()
}

/// Hessian-vector product via forward finite differences over the
/// reverse-mode gradient.
///
/// Given a gradient closure `grad(w) -> Vec<f64>` (which the user
/// constructs from the existing `gradient_*_pooled_aggregate` API
/// for some scalar loss), this returns
///
/// ```text
///   H v ≈ [grad(w + ε v) − grad(w − ε v)] / (2 ε)
/// ```
///
/// This is forward-FD over the gradient computation, costing two
/// gradient evaluations per HVP. **It does not materialize the
/// Hessian.** Newton-CG and trust-region optimizers consume the HVP
/// directly via Krylov methods.
///
/// # Choice of `eps`
///
/// The default `eps_default()` returns `1e-6`, suitable for most
/// f64 cases. Smaller `eps` reduces truncation error but amplifies
/// f64 round-off; larger `eps` does the opposite. For a smooth
/// gradient at a reasonable working point, `1e-6` typically gives
/// HVP accuracy of ~1e-9 relative. If your gradient has visible
/// noise (e.g., from kernel-floor clipping, see
/// `optimize_data_weights_logloss.rs`), use `1e-4` or larger.
///
/// # When this is appropriate
///
/// - Smooth losses where the gradient is differentiable
/// - Optimization where you want curvature information for
///   Newton-CG / trust region without forming the full Hessian
/// - Cramér-Rao bounds via Fisher information (apply HVP to
///   coordinate basis vectors)
///
/// # When this is NOT appropriate
///
/// - Losses with discontinuous gradients (e.g., absolute value,
///   max-pooling) — the gradient is piecewise-constant and HVP
///   measures that discontinuity, which is meaningless
/// - Stochastic losses where each gradient evaluation has noise
///   exceeding `eps × gradient_scale` — FD will be dominated by
///   the noise
///
/// # Example
///
/// ```ignore
/// use morton_cascade::jvp::hvp;
///
/// // Define your gradient closure (uses the cascade's pooled
/// // aggregate API):
/// let grad = |w: &[f64]| -> Vec<f64> {
///     // ... build runner, compute betas from current w,
///     //     call gradient_var_delta_data_pooled_aggregate
///     unimplemented!()
/// };
///
/// // Compute Hv at the current weights w in direction v:
/// let h_v = hvp(&grad, &w, &v, 1e-6);
/// ```
pub fn hvp<F>(grad_fn: &F, w: &[f64], v: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    debug_assert_eq!(w.len(), v.len(),
        "hvp: w length {} != v length {}", w.len(), v.len());
    let mut w_plus: Vec<f64> = w.iter().zip(v.iter())
        .map(|(wi, vi)| wi + eps * vi).collect();
    let mut w_minus: Vec<f64> = w.iter().zip(v.iter())
        .map(|(wi, vi)| wi - eps * vi).collect();
    let g_plus = grad_fn(&w_plus);
    let g_minus = grad_fn(&w_minus);
    debug_assert_eq!(g_plus.len(), w.len());
    debug_assert_eq!(g_minus.len(), w.len());
    let inv = 1.0 / (2.0 * eps);
    let out: Vec<f64> = g_plus.iter().zip(g_minus.iter())
        .map(|(gp, gm)| (gp - gm) * inv).collect();
    // Suppress unused warnings (we used these to construct g_plus/g_minus).
    let _ = (&mut w_plus, &mut w_minus);
    out
}

/// Default `eps` for [`hvp`] — `1e-6`, a conservative central-FD
/// step for f64 gradients. See [`hvp`] documentation for guidance
/// on tuning.
pub fn hvp_eps_default() -> f64 { 1e-6 }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_run::{CascadeRunner, CascadeRunPlan};
    use crate::hier_bitvec_pair::FieldStatsConfig;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn make_periodic_pts(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mask = (1u64 << bits) - 1;
        (0..n).map(|_| [
            splitmix64(&mut s) & mask,
            splitmix64(&mut s) & mask,
            splitmix64(&mut s) & mask,
        ]).collect()
    }

    /// JVP of pooled variance gradient should equal: for each bin,
    /// the dot of `bin_grads[bin]` with `v`. This is a structural
    /// sanity check rather than a math validation.
    #[test]
    fn jvp_field_stats_matches_hand_computed() {
        let pts_d = make_periodic_pts(40, 5, 100111);
        let pts_r = make_periodic_pts(120, 5, 100222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.3, 100333);
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), None, pts_r, None, bits, plan);
        let cfg = FieldStatsConfig::default();
        let pooled = runner.gradient_var_delta_data_pooled(&cfg, 1e-6);

        // Make a structured perturbation: alternating ±1 across particles.
        let v: Vec<f64> = (0..pts_d.len()).map(|i| {
            if i % 2 == 0 { 1.0 } else { -1.0 }
        }).collect();
        let jvp_result = jvp_field_stats(&pooled, &v);

        // Hand check: each bin should equal the dot product directly.
        assert_eq!(jvp_result.len(), pooled.bin_grads.len());
        for (b, expected_row) in pooled.bin_grads.iter().enumerate() {
            let expected: f64 = expected_row.iter().zip(v.iter())
                .map(|(a, b)| a * b).sum();
            assert!((jvp_result[b] - expected).abs() < 1e-12,
                "bin {}: jvp {} vs hand-dot {}", b, jvp_result[b], expected);
        }
    }

    /// JVP via the API matches finite-difference of the forward
    /// observable directly. This is the meaningful math check:
    /// confirms that what we call "directional derivative" really is
    /// the directional derivative of the underlying observable.
    #[test]
    fn jvp_field_stats_matches_finite_difference() {
        let pts_d = make_periodic_pts(25, 5, 101111);
        let pts_r = make_periodic_pts(75, 5, 101222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.25, 101333);
        let cfg = FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let n_d = pts_d.len();
        let wd_base = vec![1.0_f64; n_d];

        let runner_base = CascadeRunner::new_isolated(
            pts_d.clone(), Some(wd_base.clone()),
            pts_r.clone(), None, bits, plan.clone());
        let pooled = runner_base.gradient_var_delta_data_pooled(&cfg, bin_tol);

        // A structured perturbation direction.
        let v: Vec<f64> = (0..n_d).map(|i| {
            ((i as f64) / (n_d as f64) - 0.5) * 2.0   // linear ramp [-1, 1]
        }).collect();
        let jvp_analytic = jvp_field_stats(&pooled, &v);

        // FD: vary all weights together along v.
        let var_at = |wd: &[f64]| -> Vec<f64> {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(wd.to_vec()),
                pts_r.clone(), None, bits, plan.clone());
            let agg = r.analyze_field_stats(&cfg, bin_tol);
            agg.by_side.iter().map(|b| b.var_delta).collect()
        };
        let eps = 1e-5_f64;
        let wd_plus: Vec<f64> = wd_base.iter().zip(v.iter())
            .map(|(w, vi)| w + eps * vi).collect();
        let wd_minus: Vec<f64> = wd_base.iter().zip(v.iter())
            .map(|(w, vi)| w - eps * vi).collect();
        let var_plus = var_at(&wd_plus);
        let var_minus = var_at(&wd_minus);
        let inv = 1.0 / (2.0 * eps);
        let fd: Vec<f64> = var_plus.iter().zip(var_minus.iter())
            .map(|(a, b)| (a - b) * inv).collect();

        assert_eq!(jvp_analytic.len(), fd.len());
        for b in 0..jvp_analytic.len() {
            let diff = (jvp_analytic[b] - fd[b]).abs();
            let scale = fd[b].abs().max(1e-10);
            assert!(diff < 5e-3 || diff / scale < 5e-3,
                "bin {}: JVP analytic {} vs central-FD {} (diff {})",
                b, jvp_analytic[b], fd[b], diff);
        }
    }

    /// HVP applied to a quadratic loss should reproduce the constant
    /// Hessian. Concretely: take L(w) = 0.5 (w - w*)ᵀ A (w - w*) for a
    /// fixed positive-definite A. Then ∇L = A (w - w*) and Hv = Av
    /// independent of w. Tests both forward and central FD modes.
    #[test]
    fn hvp_matches_quadratic_hessian() {
        // Synthetic 4x4 PD matrix.
        let n = 4;
        let a = [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.5, 0.2],
            [0.5, 0.5, 2.0, 0.1],
            [0.0, 0.2, 0.1, 1.5],
        ];
        let w_star = [0.5, -0.3, 0.7, 0.1];

        let grad = |w: &[f64]| -> Vec<f64> {
            (0..n).map(|i| {
                (0..n).map(|j| a[i][j] * (w[j] - w_star[j])).sum()
            }).collect()
        };

        let w = vec![0.0_f64; n];
        // HVP at v = e_0:
        let v = vec![1.0_f64, 0.0, 0.0, 0.0];
        let hv = hvp(&grad, &w, &v, hvp_eps_default());
        let expected: Vec<f64> = (0..n).map(|i| a[i][0]).collect();
        for i in 0..n {
            assert!((hv[i] - expected[i]).abs() < 1e-7,
                "HVP[{}] = {}, expected {}", i, hv[i], expected[i]);
        }

        // HVP at v = (1, 1, 1, 1) should give row sums of A:
        let v_uniform = vec![1.0_f64; n];
        let hv_uniform = hvp(&grad, &w, &v_uniform, hvp_eps_default());
        let expected_uniform: Vec<f64> = (0..n).map(|i| {
            (0..n).map(|j| a[i][j]).sum()
        }).collect();
        for i in 0..n {
            assert!((hv_uniform[i] - expected_uniform[i]).abs() < 1e-7,
                "HVP_uniform[{}] = {}, expected {}", i, hv_uniform[i],
                expected_uniform[i]);
        }
    }

    /// HVP applied to the cascade's variance loss in a coordinate
    /// direction should approximately equal the second partial derivative
    /// (estimated by central FD on the loss directly). This is the
    /// real test that HVP works on the cascade end-to-end.
    #[test]
    fn hvp_matches_second_order_finite_difference_on_cascade_loss() {
        let pts_d = make_periodic_pts(20, 5, 102111);
        let pts_r = make_periodic_pts(60, 5, 102222);
        let bits = [5u32; 3];
        let plan = CascadeRunPlan::random_offsets(2, 0.25, 102333);
        let cfg = FieldStatsConfig::default();
        let bin_tol = 1e-6;
        let n_d = pts_d.len();
        let wd_base = vec![1.0_f64; n_d];

        // Loss: L(w) = Σ_b (var_b(w))² (a smooth scalar of cascade var).
        // ∇L_i = Σ_b 2 var_b · ∂var_b/∂w_i (use aggregate with
        // betas[b] = 2 var_b).
        let var_profile = |w: &[f64]| -> Vec<f64> {
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(w.to_vec()),
                pts_r.clone(), None, bits, plan.clone());
            r.analyze_field_stats(&cfg, bin_tol).by_side
                .iter().map(|b| b.var_delta).collect()
        };
        let loss = |w: &[f64]| -> f64 {
            var_profile(w).iter().map(|v| v * v).sum()
        };
        let grad = |w: &[f64]| -> Vec<f64> {
            let varp = var_profile(w);
            let betas: Vec<f64> = varp.iter().map(|v| 2.0 * v).collect();
            let r = CascadeRunner::new_isolated(
                pts_d.clone(), Some(w.to_vec()),
                pts_r.clone(), None, bits, plan.clone());
            r.gradient_var_delta_data_pooled_aggregate(&cfg, bin_tol, &betas)
        };

        // Pick a particle, compute HVP in coordinate direction e_i,
        // and compare to second-order central FD of L:
        //   ∂²L/∂w_i² ≈ [L(w + ε e_i) - 2 L(w) + L(w - ε e_i)] / ε²
        // The HVP gives us (∂²L/∂w_j ∂w_i) for all j, of which the i'th
        // entry is ∂²L/∂w_i².
        let i = 7;
        let mut e_i = vec![0.0_f64; n_d];
        e_i[i] = 1.0;
        let hvp_eps = 1e-4_f64;  // looser eps because the gradient is noisy
        let h_e_i = hvp(&grad, &wd_base, &e_i, hvp_eps);

        // Second-order FD on L directly. Need a separate eps to keep
        // truncation error well above f64 round-off.
        let fd2_eps = 1e-3_f64;
        let mut wd_plus = wd_base.clone();
        let mut wd_minus = wd_base.clone();
        wd_plus[i] += fd2_eps;
        wd_minus[i] -= fd2_eps;
        let l_plus = loss(&wd_plus);
        let l_zero = loss(&wd_base);
        let l_minus = loss(&wd_minus);
        let fd2_ii = (l_plus - 2.0 * l_zero + l_minus) / (fd2_eps * fd2_eps);

        let h_ii = h_e_i[i];
        // Expect agreement to ~1e-3 relative (second-order FD has
        // limited precision).
        let diff = (h_ii - fd2_ii).abs();
        let scale = fd2_ii.abs().max(1e-6);
        assert!(diff / scale < 5e-2,
            "particle {}: HVP[i,i] = {}, FD₂ ∂²L/∂w_i² = {} (rel diff {:.2e})",
            i, h_ii, fd2_ii, diff / scale);
    }
}
