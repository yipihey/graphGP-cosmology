// xi_continuous.rs
//
// Continuous-function fit of ξ(r) from a multi-run AggregatedXi.
//
// Adapted from Storey-Fisher & Hogg (2021, ApJ 909, 220; arxiv 2011.01836)
// to the cascade case where each measurement is a top-hat windowed
// estimate over a known shell — rather than reaching into the cascade
// for per-pair separations, we treat each cascade shell as the basis
// projection that the cascade's binning naturally provides.
//
// Math. Each measurement i = (resize-group, level) is a window-averaged
// estimate of ξ:
//
//   y_i = (1/N_i) ∫_{r_i-Δ_i}^{r_i+Δ_i} ξ(r) w_i(r) dr
//   N_i = ∫_{r_i-Δ_i}^{r_i+Δ_i} w_i(r) dr
//
// where w_i(r) is the pair density inside the shell. Substituting
// ξ = Σ_n c_n ψ_n with linear B-splines in log r:
//
//   y_i = Σ_n c_n A_in,
//   A_in = (1/N_i) ∫_{shell_i} ψ_n(r) w_i(r) dr
//
// Solve the weighted normal equations
//
//   ĉ = (Aᵀ W A)⁻¹ Aᵀ W y,  W = diag(1/σ_i²)
//
// with coefficient covariance (Aᵀ W A)⁻¹. Integrals computed by
// Gauss-Legendre quadrature inside each shell — basis-function-agnostic
// and trivially correct for any future basis choice.
//
// Weighting choice. In isolated mode w_i can be set from the cascade's
// own measured RR per shell (constant inside the shell since RR is one
// number per shell — equivalent to "uniform pair density inside shell").
// In periodic mode RR is unavailable so fall back to the Euclidean
// w(r) = r^(D-1) which is the analytic pair density for uniform points.

use crate::multi_run::AggregatedXi;

/// Choice of basis for the continuous fit.
#[derive(Clone, Debug)]
pub enum XiBasis {
    /// Linear (tent) B-splines on a uniform grid in log r. K knots
    /// produce K basis functions, each piecewise-linear in log r.
    /// Knot range is [log(r_min), log(r_max)].
    LinearBSplineLogR {
        n_knots: usize,
        r_min: f64,
        r_max: f64,
    },
}

/// How to weight the integrand `ξ(r) w(r) / ∫w(r)` over each shell.
#[derive(Clone, Copy, Debug)]
pub enum XiWindowWeighting {
    /// Use Euclidean pair-density weighting w(r) = r^(D-1). Correct
    /// for periodic mode and for isolated mode in the limit of
    /// uniform random catalogs without survey edges.
    EuclideanRPow { d: usize },
    /// Use the cascade's empirical RR_sum as a constant pair-density
    /// inside each shell (i.e., assume RR is uniform within the shell
    /// width). Strictly more correct than Euclidean for real surveys
    /// with footprints. Falls back to Euclidean if RR_sum = 0
    /// (periodic mode).
    EmpiricalRR { fallback_d: usize },
}

/// How to weight measurements in the least-squares fit.
#[derive(Clone, Copy, Debug)]
pub enum XiMeasurementWeighting {
    /// 1/σ² with σ² = `xi_shift_bootstrap_var` per shell. Pure
    /// shift-bootstrap weighting; meaningful only when n_shifts ≥ 2.
    /// Shells with σ² = 0 are excluded from the fit.
    ShiftBootstrap,
    /// 1/σ² with σ² = max(xi_shift_bootstrap_var, poisson_floor),
    /// where the Poisson floor is 1/(DD_sum + 1) — a crude
    /// shot-noise envelope to keep zero-count shells from being
    /// excluded entirely.
    ShiftBootstrapPlusPoisson,
    /// Equal weights — simplest, worst-conditioned. Useful for
    /// sanity-checking against the weighted fits.
    Uniform,
}

/// Result of a continuous-function fit.
#[derive(Clone, Debug)]
pub struct XiContinuousFit {
    pub basis: XiBasis,
    /// Fitted coefficients c_n.
    pub coefs: Vec<f64>,
    /// Coefficient covariance: (Aᵀ W A)⁻¹.
    pub coef_cov: Vec<Vec<f64>>,
    /// Number of measurements (shells across all resize groups) that
    /// went into the fit. Excludes shells dropped due to non-finite
    /// xi_naive or zero σ when ShiftBootstrap weighting was selected.
    pub n_measurements_used: usize,
    /// Number of basis functions (= n_knots).
    pub n_basis: usize,
    /// Reduced χ² of the fit: Σ_i (y_i - Σ_n c_n A_in)² / σ_i² / (n_meas - n_basis).
    /// Should be ≈ 1 for a well-specified model.
    pub reduced_chi_squared: f64,
}

impl XiContinuousFit {
    /// Evaluate the fitted ξ(r) at a single point.
    pub fn evaluate(&self, r: f64) -> f64 {
        let bvals = basis_values_at(&self.basis, r);
        bvals.iter().enumerate()
            .map(|(n, b)| b * self.coefs[n])
            .sum()
    }

    /// Evaluate fitted ξ(r) at many points.
    pub fn evaluate_many(&self, rs: &[f64]) -> Vec<f64> {
        rs.iter().map(|&r| self.evaluate(r)).collect()
    }

    /// 1-σ uncertainty on ξ(r) at a single r, from coefficient covariance.
    /// σ_ξ²(r) = bᵀ Σ b where b are basis values and Σ = coef_cov.
    pub fn sigma_at(&self, r: f64) -> f64 {
        let bvals = basis_values_at(&self.basis, r);
        let mut v = 0.0;
        for n in 0..bvals.len() {
            for m in 0..bvals.len() {
                v += bvals[n] * self.coef_cov[n][m] * bvals[m];
            }
        }
        v.max(0.0).sqrt()
    }
}

/// Fit a continuous ξ(r) from a multi-run AggregatedXi by weighted
/// least squares onto the chosen basis.
///
/// Returns `Err` if the design matrix is rank-deficient (e.g., too
/// few non-degenerate measurements for the requested basis size).
pub fn fit_xi_continuous<const D: usize>(
    agg: &AggregatedXi<D>,
    basis: &XiBasis,
    window: XiWindowWeighting,
    measurement: XiMeasurementWeighting,
) -> Result<XiContinuousFit, String> {
    // Step 1: collect (shell-window, y_i, σ_i) measurements, dropping
    // shells with non-finite y_i or zero width.
    struct Measurement {
        r_lo: f64,
        r_hi: f64,
        rr_sum: f64,
        y: f64,
        sigma2: f64,
    }
    let mut meas: Vec<Measurement> = Vec::new();
    for g in &agg.by_resize {
        for s in &g.shells {
            if !s.xi_naive.is_finite() { continue; }
            if s.r_half_width <= 0.0 { continue; }
            let r_lo = s.r_center - s.r_half_width;
            let r_hi = s.r_center + s.r_half_width;
            if r_lo <= 0.0 { continue; }
            let sigma2 = match measurement {
                XiMeasurementWeighting::ShiftBootstrap => {
                    if s.xi_shift_bootstrap_var > 0.0 {
                        s.xi_shift_bootstrap_var
                    } else { continue; }
                }
                XiMeasurementWeighting::ShiftBootstrapPlusPoisson => {
                    let poisson_floor = 1.0 / (s.dd_sum + 1.0);
                    s.xi_shift_bootstrap_var.max(poisson_floor)
                }
                XiMeasurementWeighting::Uniform => 1.0,
            };
            if !sigma2.is_finite() || sigma2 <= 0.0 { continue; }
            meas.push(Measurement {
                r_lo, r_hi,
                rr_sum: s.rr_sum,
                y: s.xi_naive,
                sigma2,
            });
        }
    }

    let n_basis = match basis {
        XiBasis::LinearBSplineLogR { n_knots, .. } => *n_knots,
    };
    if meas.len() < n_basis {
        return Err(format!(
            "fit_xi_continuous: only {} usable measurements for {} basis functions; \
             reduce n_knots or supply more cascade runs",
            meas.len(), n_basis));
    }

    // Step 2: build design matrix A and weights W.
    // A_in = (1/N_i) ∫_shell_i ψ_n(r) w_i(r) dr,  N_i = ∫ w_i(r) dr.
    let mut a_mat: Vec<Vec<f64>> = vec![vec![0.0; n_basis]; meas.len()];
    for (i, m) in meas.iter().enumerate() {
        let row = build_design_row(basis, &window, m.r_lo, m.r_hi, m.rr_sum);
        a_mat[i] = row;
    }

    // Step 3: solve weighted normal equations via dense Cholesky.
    // M = AᵀWA  (n_basis × n_basis)
    // b = AᵀWy
    // Solve M c = b; Σ = M⁻¹.
    let mut m_mat: Vec<Vec<f64>> = vec![vec![0.0; n_basis]; n_basis];
    let mut b_vec: Vec<f64> = vec![0.0; n_basis];
    for (i, mi) in meas.iter().enumerate() {
        let w = 1.0 / mi.sigma2;
        for n in 0..n_basis {
            b_vec[n] += w * a_mat[i][n] * mi.y;
            for k in 0..n_basis {
                m_mat[n][k] += w * a_mat[i][n] * a_mat[i][k];
            }
        }
    }

    // Add a tiny ridge to the diagonal for numerical stability when
    // some basis functions have no support overlap with any measurement.
    let trace_avg = (0..n_basis).map(|n| m_mat[n][n]).sum::<f64>() / n_basis as f64;
    let ridge = trace_avg * 1e-12;
    for n in 0..n_basis { m_mat[n][n] += ridge; }

    let coefs = cholesky_solve(&m_mat, &b_vec)
        .map_err(|e| format!("fit_xi_continuous: Cholesky failure: {}", e))?;
    let coef_cov = cholesky_inverse(&m_mat)
        .map_err(|e| format!("fit_xi_continuous: covariance inverse failure: {}", e))?;

    // Step 4: reduced χ².
    let mut chi2 = 0.0;
    for (i, mi) in meas.iter().enumerate() {
        let pred: f64 = (0..n_basis).map(|n| coefs[n] * a_mat[i][n]).sum();
        let resid = mi.y - pred;
        chi2 += resid * resid / mi.sigma2;
    }
    let dof = (meas.len() as i64 - n_basis as i64).max(1) as f64;
    let reduced_chi_squared = chi2 / dof;

    Ok(XiContinuousFit {
        basis: basis.clone(),
        coefs,
        coef_cov,
        n_measurements_used: meas.len(),
        n_basis,
        reduced_chi_squared,
    })
}

// ============================================================================
// Linear B-spline basis evaluation
// ============================================================================

/// Evaluate all basis functions at point r. Returns a vector of length
/// n_basis. Most entries are zero (B-splines have compact support).
fn basis_values_at(basis: &XiBasis, r: f64) -> Vec<f64> {
    match basis {
        XiBasis::LinearBSplineLogR { n_knots, r_min, r_max } => {
            let mut out = vec![0.0; *n_knots];
            if r <= 0.0 { return out; }
            let log_r = r.ln();
            let log_min = r_min.ln();
            let log_max = r_max.ln();
            // Knot positions in log r, uniformly spaced.
            // Knot n is at u_n = log_min + n * (log_max - log_min) / (n_knots - 1).
            // ψ_n is the tent function peaked at u_n with support [u_{n-1}, u_{n+1}],
            // clamped at the endpoints.
            if *n_knots < 2 { return out; }
            let du = (log_max - log_min) / (*n_knots as f64 - 1.0);
            // Find which interval log_r is in.
            // x = (log_r - log_min) / du gives the fractional index.
            let x = (log_r - log_min) / du;
            // x in [n, n+1] means r is between knots n and n+1, contributing
            // to basis ψ_n (with weight 1 - frac) and ψ_{n+1} (with weight frac).
            if x < 0.0 || x > (*n_knots as f64 - 1.0) {
                // Outside the basis range — clamp to nearest endpoint with
                // zero weight (extrapolation is the user's problem).
                return out;
            }
            let n_lo = x.floor() as usize;
            if n_lo + 1 >= *n_knots {
                // At the right endpoint exactly.
                out[*n_knots - 1] = 1.0;
                return out;
            }
            let frac = x - n_lo as f64;
            out[n_lo] = 1.0 - frac;
            out[n_lo + 1] = frac;
            out
        }
    }
}

/// Build one row of the design matrix: A_in = (1/N_i) ∫ ψ_n(r) w_i(r) dr
/// over the shell [r_lo, r_hi]. Uses Gauss-Legendre quadrature.
fn build_design_row(
    basis: &XiBasis,
    window: &XiWindowWeighting,
    r_lo: f64,
    r_hi: f64,
    rr_sum: f64,
) -> Vec<f64> {
    // 8-point Gauss-Legendre on [-1, 1]; we rescale to [r_lo, r_hi].
    // Nodes and weights from standard tables.
    const N: usize = 8;
    const NODES: [f64; N] = [
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498,  0.1834346424956498,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363,
    ];
    const WEIGHTS: [f64; N] = [
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
        0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763,
    ];

    let half = 0.5 * (r_hi - r_lo);
    let mid = 0.5 * (r_hi + r_lo);

    let n_basis = match basis {
        XiBasis::LinearBSplineLogR { n_knots, .. } => *n_knots,
    };
    let mut numerator = vec![0.0; n_basis];
    let mut denominator = 0.0;

    for k in 0..N {
        let r = mid + half * NODES[k];
        let dr = half * WEIGHTS[k];
        let w_r = pair_density(window, r, rr_sum);
        denominator += w_r * dr;
        let bvals = basis_values_at(basis, r);
        for n in 0..n_basis {
            numerator[n] += bvals[n] * w_r * dr;
        }
    }

    if denominator > 0.0 {
        for n in 0..n_basis {
            numerator[n] /= denominator;
        }
    }
    numerator
}

/// Evaluate the pair density w_i(r) inside a shell.
fn pair_density(window: &XiWindowWeighting, r: f64, rr_sum: f64) -> f64 {
    match window {
        XiWindowWeighting::EuclideanRPow { d } => {
            // r^(D-1)
            r.powi(*d as i32 - 1)
        }
        XiWindowWeighting::EmpiricalRR { fallback_d } => {
            if rr_sum > 0.0 {
                // Constant inside the shell — RR is one number per shell.
                // The actual constant cancels in A_in normalization, so we
                // can use any positive value. Use rr_sum itself for clarity.
                rr_sum
            } else {
                r.powi(*fallback_d as i32 - 1)
            }
        }
    }
}

// ============================================================================
// Small dense linear algebra (no external dependencies)
// ============================================================================

/// In-place Cholesky factorization: M = L Lᵀ. Stores L in the lower
/// triangle (and diagonal) of the input. Returns Err if M is not
/// positive-definite within numerical tolerance.
fn cholesky_factor(m: &mut Vec<Vec<f64>>) -> Result<(), String> {
    let n = m.len();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = m[i][j];
            for k in 0..j {
                sum -= m[i][k] * m[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "non-positive diagonal at row {} during Cholesky: {}", i, sum));
                }
                m[i][j] = sum.sqrt();
            } else {
                m[i][j] = sum / m[j][j];
            }
        }
        // Zero the upper triangle for cleanliness.
        for j in (i + 1)..n {
            m[i][j] = 0.0;
        }
    }
    Ok(())
}

/// Solve Mc = b given L (Cholesky factor in lower triangle).
fn cholesky_solve_with_factor(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    // Forward solve L y = b.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    // Back solve Lᵀ x = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Convenience: factor M then solve Mc = b.
fn cholesky_solve(m: &Vec<Vec<f64>>, b: &[f64]) -> Result<Vec<f64>, String> {
    let mut m_copy = m.clone();
    cholesky_factor(&mut m_copy)?;
    Ok(cholesky_solve_with_factor(&m_copy, b))
}

/// Compute M⁻¹ via Cholesky factorization. Returns the dense inverse
/// matrix. Intended for small (n ≤ ~100) systems.
fn cholesky_inverse(m: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
    let n = m.len();
    let mut m_copy = m.clone();
    cholesky_factor(&mut m_copy)?;
    // Solve M x_j = e_j for each unit vector e_j.
    let mut inv: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut e = vec![0.0; n];
    for j in 0..n {
        for k in 0..n { e[k] = 0.0; }
        e[j] = 1.0;
        let x = cholesky_solve_with_factor(&m_copy, &e);
        for i in 0..n { inv[i][j] = x[i]; }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Linear algebra primitives ----

    #[test]
    fn cholesky_solves_diagonal_system() {
        // Diagonal positive matrix: M_ii = 1/(i+1)^2, b_i = 1
        // ⇒ x_i = (i+1)^2.
        let n = 5;
        let mut m: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        let mut b: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            m[i][i] = 1.0 / ((i + 1) as f64).powi(2);
            b[i] = 1.0;
        }
        let x = cholesky_solve(&m, &b).unwrap();
        for i in 0..n {
            let expected = ((i + 1) as f64).powi(2);
            assert!((x[i] - expected).abs() < 1e-10,
                "x[{}] = {}, expected {}", i, x[i], expected);
        }
    }

    #[test]
    fn cholesky_inverse_recovers_identity() {
        // 3×3 SPD matrix, M·M⁻¹ = I.
        let m: Vec<Vec<f64>> = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.7],
            vec![0.5, 0.7, 2.0],
        ];
        let inv = cholesky_inverse(&m).unwrap();
        // Verify M * inv = I.
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let v: f64 = (0..n).map(|k| m[i][k] * inv[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-10,
                    "(M·M⁻¹)[{},{}] = {}, expected {}", i, j, v, expected);
            }
        }
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        // Singular matrix: row 1 = 2 × row 0.
        let m: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        let b = vec![1.0, 2.0];
        assert!(cholesky_solve(&m, &b).is_err(),
            "should detect singular matrix");
    }

    // ---- Linear B-spline basis ----

    #[test]
    fn linear_bspline_partition_of_unity() {
        // At any point inside the basis range, the basis values must
        // sum to 1 (partition of unity for tent functions).
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 10, r_min: 1.0, r_max: 1000.0,
        };
        for r in [1.5, 5.0, 17.0, 50.0, 200.0, 800.0] {
            let bv = basis_values_at(&basis, r);
            let s: f64 = bv.iter().sum();
            assert!((s - 1.0).abs() < 1e-12,
                "partition-of-unity violated at r={}: sum = {}", r, s);
        }
    }

    #[test]
    fn linear_bspline_at_knot_is_one_hot() {
        // At knot u_n = log_min + n * du, basis value should be 1 at
        // index n, 0 elsewhere.
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 5, r_min: 1.0, r_max: 10000.0,
        };
        let log_min = 1.0_f64.ln();
        let log_max = 10000.0_f64.ln();
        let du = (log_max - log_min) / 4.0;
        for n in 0..5 {
            let r = (log_min + n as f64 * du).exp();
            let bv = basis_values_at(&basis, r);
            for k in 0..5 {
                let expected = if k == n { 1.0 } else { 0.0 };
                assert!((bv[k] - expected).abs() < 1e-10,
                    "knot {}: basis[{}] = {}, expected {}",
                    n, k, bv[k], expected);
            }
        }
    }

    #[test]
    fn linear_bspline_outside_range_is_zero() {
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 5, r_min: 1.0, r_max: 100.0,
        };
        // Below r_min
        let bv = basis_values_at(&basis, 0.5);
        let s: f64 = bv.iter().sum();
        assert!(s.abs() < 1e-12, "below range should give zero sum, got {}", s);
        // Above r_max
        let bv = basis_values_at(&basis, 200.0);
        let s: f64 = bv.iter().sum();
        assert!(s.abs() < 1e-12, "above range should give zero sum, got {}", s);
    }

    // ---- Continuous fit, end-to-end ----

    #[test]
    fn fit_recovers_constant_xi_exactly() {
        // ξ(r) = constant ⇒ all basis coefficients should equal that
        // constant (partition-of-unity ⇒ constant function = sum of all
        // basis funcs with equal weight).
        use crate::multi_run::{AggregatedXi, XiResizeGroup, XiShellPooled, RunDiagnostic};
        let xi_const = 0.42;
        let mut shells = Vec::new();
        for (l, r) in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0].iter().enumerate() {
            shells.push(XiShellPooled {
                level: l,
                r_center: *r,
                r_half_width: r * 0.05,  // narrow shells
                dd_sum: 1000.0,
                rr_sum: 3000.0,
                dr_sum: 0.0,
                n_d_sum: 100,
                n_r_sum: 300,
                xi_naive: xi_const,
                xi_shift_bootstrap_var: 0.001,
            });
        }
        let agg: AggregatedXi<3> = AggregatedXi {
            by_resize: vec![XiResizeGroup { scale: 1.0, n_shifts: 1, shells }],
            per_run_diagnostics: Vec::<RunDiagnostic<3>>::new(),
        };
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 4, r_min: 1.0, r_max: 100.0,
        };
        let fit = fit_xi_continuous(&agg, &basis,
            XiWindowWeighting::EuclideanRPow { d: 3 },
            XiMeasurementWeighting::Uniform).unwrap();
        // All coefficients should equal xi_const within numerics.
        for (n, c) in fit.coefs.iter().enumerate() {
            assert!((c - xi_const).abs() < 1e-6,
                "coef[{}] = {}, expected {}", n, c, xi_const);
        }
        // Evaluate at a few points → should give xi_const.
        for r in [3.0, 10.0, 30.0] {
            let v = fit.evaluate(r);
            assert!((v - xi_const).abs() < 1e-6,
                "evaluate({}) = {}, expected {}", r, v, xi_const);
        }
    }

    #[test]
    fn fit_recovers_log_linear_xi() {
        // ξ(r) = a + b * log(r) ⇒ linear B-splines in log r should
        // recover this exactly because each tent's value at any point
        // is a linear interpolation of the knot values, and a linear
        // function evaluates exactly under linear interpolation.
        use crate::multi_run::{AggregatedXi, XiResizeGroup, XiShellPooled, RunDiagnostic};
        let a = 1.5;
        let b = -0.3;
        let mut shells = Vec::new();
        for (l, r) in [2.0_f64, 4.0, 8.0, 16.0, 32.0, 64.0].iter().enumerate() {
            // Use narrow shells so the windowed average ≈ pointwise value
            let r_half = r * 0.01;
            // Window-averaged value of (a + b log r) over [r-Δ, r+Δ]
            // weighted by r²: not exactly a + b log(r_center), but
            // very close for narrow shells. Set y_i directly to the
            // theoretical value at r_center to test the basis fit.
            let xi = a + b * r.ln();
            shells.push(XiShellPooled {
                level: l,
                r_center: *r,
                r_half_width: r_half,
                dd_sum: 1000.0,
                rr_sum: 3000.0,
                dr_sum: 0.0,
                n_d_sum: 100,
                n_r_sum: 300,
                xi_naive: xi,
                xi_shift_bootstrap_var: 0.001,
            });
        }
        let agg: AggregatedXi<3> = AggregatedXi {
            by_resize: vec![XiResizeGroup { scale: 1.0, n_shifts: 1, shells }],
            per_run_diagnostics: Vec::<RunDiagnostic<3>>::new(),
        };
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 5, r_min: 1.0, r_max: 100.0,
        };
        let fit = fit_xi_continuous(&agg, &basis,
            XiWindowWeighting::EuclideanRPow { d: 3 },
            XiMeasurementWeighting::Uniform).unwrap();
        // Evaluate at several r and check ξ_fit ≈ a + b·log(r).
        // Tolerance is loose because narrow-shell window averaging in
        // r²-weighted integrand and the basis tents in log r aren't
        // exactly linear together.
        for r in [3.0, 7.0, 12.0, 25.0, 50.0] {
            let observed = fit.evaluate(r);
            let expected = a + b * r.ln();
            assert!((observed - expected).abs() < 0.01,
                "evaluate({}) = {}, expected {} (loose tol)",
                r, observed, expected);
        }
    }

    #[test]
    fn fit_combines_multiple_resize_groups() {
        // Two resize groups (different shell volumes), constant ξ.
        // Fit must combine them and recover the constant.
        use crate::multi_run::{AggregatedXi, XiResizeGroup, XiShellPooled, RunDiagnostic};
        let xi_const = 0.7;
        let mk_shell = |r: f64, hw_frac: f64, l: usize| XiShellPooled {
            level: l,
            r_center: r,
            r_half_width: r * hw_frac,
            dd_sum: 500.0,
            rr_sum: 1500.0,
            dr_sum: 0.0,
            n_d_sum: 100,
            n_r_sum: 300,
            xi_naive: xi_const,
            xi_shift_bootstrap_var: 0.005,
        };
        // Group 1 (scale 1.0): wider shells at r = 4, 8, 16, 32
        let g1 = XiResizeGroup {
            scale: 1.0, n_shifts: 1,
            shells: vec![
                mk_shell(4.0, 0.1, 0), mk_shell(8.0, 0.1, 1),
                mk_shell(16.0, 0.1, 2), mk_shell(32.0, 0.1, 3),
            ],
        };
        // Group 2 (scale 0.7): different shell layout
        let g2 = XiResizeGroup {
            scale: 0.7, n_shifts: 1,
            shells: vec![
                mk_shell(2.8, 0.1, 0), mk_shell(5.6, 0.1, 1),
                mk_shell(11.2, 0.1, 2), mk_shell(22.4, 0.1, 3),
            ],
        };
        let agg: AggregatedXi<3> = AggregatedXi {
            by_resize: vec![g1, g2],
            per_run_diagnostics: Vec::<RunDiagnostic<3>>::new(),
        };
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 4, r_min: 1.0, r_max: 100.0,
        };
        let fit = fit_xi_continuous(&agg, &basis,
            XiWindowWeighting::EuclideanRPow { d: 3 },
            XiMeasurementWeighting::Uniform).unwrap();
        assert_eq!(fit.n_measurements_used, 8,
            "expected all 8 shells (4 per group × 2 groups) to contribute");
        for c in &fit.coefs {
            assert!((c - xi_const).abs() < 1e-3,
                "coef = {}, expected {}", c, xi_const);
        }
    }

    #[test]
    fn fit_rejects_too_few_measurements() {
        use crate::multi_run::{AggregatedXi, XiResizeGroup, XiShellPooled, RunDiagnostic};
        let agg: AggregatedXi<3> = AggregatedXi {
            by_resize: vec![XiResizeGroup {
                scale: 1.0, n_shifts: 1,
                shells: vec![XiShellPooled {
                    level: 0,
                    r_center: 5.0,
                    r_half_width: 0.5,
                    dd_sum: 100.0,
                    rr_sum: 300.0,
                    dr_sum: 0.0,
                    n_d_sum: 50,
                    n_r_sum: 150,
                    xi_naive: 0.1,
                    xi_shift_bootstrap_var: 0.001,
                }],
            }],
            per_run_diagnostics: Vec::<RunDiagnostic<3>>::new(),
        };
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 5, r_min: 1.0, r_max: 100.0,
        };
        let result = fit_xi_continuous(&agg, &basis,
            XiWindowWeighting::EuclideanRPow { d: 3 },
            XiMeasurementWeighting::Uniform);
        assert!(result.is_err(),
            "expected error: 1 measurement < 5 basis functions");
    }

    #[test]
    fn fit_evaluate_outside_basis_range_returns_zero() {
        // Outside [r_min, r_max] all basis functions evaluate to 0,
        // so fit.evaluate(r) returns 0. (User's responsibility to ask
        // only inside the basis range.)
        use crate::multi_run::{AggregatedXi, XiResizeGroup, XiShellPooled, RunDiagnostic};
        let mut shells = Vec::new();
        for (l, r) in [4.0, 8.0, 16.0, 32.0].iter().enumerate() {
            shells.push(XiShellPooled {
                level: l, r_center: *r, r_half_width: 0.1,
                dd_sum: 500.0, rr_sum: 1500.0, dr_sum: 0.0,
                n_d_sum: 100, n_r_sum: 300,
                xi_naive: 0.5, xi_shift_bootstrap_var: 0.001,
            });
        }
        let agg: AggregatedXi<3> = AggregatedXi {
            by_resize: vec![XiResizeGroup { scale: 1.0, n_shifts: 1, shells }],
            per_run_diagnostics: Vec::<RunDiagnostic<3>>::new(),
        };
        let basis = XiBasis::LinearBSplineLogR {
            n_knots: 4, r_min: 2.0, r_max: 50.0,
        };
        let fit = fit_xi_continuous(&agg, &basis,
            XiWindowWeighting::EuclideanRPow { d: 3 },
            XiMeasurementWeighting::Uniform).unwrap();
        // Inside range: ≈ xi_naive
        let inside = fit.evaluate(10.0);
        assert!((inside - 0.5).abs() < 0.01, "inside = {}", inside);
        // Outside range: 0
        let outside_low = fit.evaluate(0.5);
        assert_eq!(outside_low, 0.0);
        let outside_high = fit.evaluate(200.0);
        assert_eq!(outside_high, 0.0);
    }
}
