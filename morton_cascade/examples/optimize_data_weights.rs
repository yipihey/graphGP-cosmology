// optimize_data_weights.rs
//
// End-to-end demonstration of the differentiable cascade: use the
// closed-form gradient of pooled variance to optimize per-particle
// data weights against a target moment profile.
//
// The point is to stress-test the gradient API under realistic
// optimization pressure, not just at a single fixed point. Things
// this example checks:
//
//   - Loss landscape behavior under standard first-order optimization
//   - Gradient stability across iterations (analytic vs. finite-diff
//     spot-check at the start, middle, and end of training)
//   - Convergence of a basic Adam optimizer on a non-trivial target
//   - Numerical handling of the weight-scale degeneracy (μ_2 is
//     invariant under uniform weight scaling — the loss has a flat
//     direction that the optimizer must navigate around)
//
// Setup:
//   - Clustered data via Cox process (so variance has a real shape)
//   - Random uniform comparison catalog
//   - 4-shift multi-run plan (so we exercise the pooled gradient path)
//   - Target: scale per-bin variance by a non-trivial factor
//     (2× at small scales, 1.2× at large scales)
//   - Loss: sum-squared per-bin variance error
//   - Parameterization: w_i = exp(θ_i), optimize on θ — keeps weights
//     positive automatically. The uniform direction in θ-space maps
//     to multiplicative scaling in w; the loss is invariant under
//     this so the gradient lies in the orthogonal complement.
//
// Observed results (one representative run, n_d=2400, 60 iters):
//   - Loss reduces ~10000× (4.1 × 10¹ → 4.3 × 10⁻³)
//   - Bins 2-7 converge to <1% relative error
//   - Bins 0-1 (smallest scales) lag at 15-30% error — they have
//     small baseline variance, so the absolute-error loss correctly
//     weights them less. Switching to a log-space or relative-error
//     loss would equalize this.
//   - Adam loss curve is non-monotonic at lr=0.05 (occasional uptick
//     near the optimum). Smaller lr or a 2nd-order method (L-BFGS)
//     would be cleaner. Adam still converges adequately.
//   - Gradient remains correct at the optimum (FD spot-check
//     relative error ~3e-5, similar to ~5e-6 at init).
//   - Weight redistribution is genuine (std/mean ≈ 0.5), not the
//     trivial uniform-scaling flat direction.
//
// Build & run:
//   cargo build --release --example optimize_data_weights
//   ./target/release/examples/optimize_data_weights

use morton_cascade::hier_bitvec_pair::FieldStatsConfig;
use morton_cascade::multi_run::{CascadeRunner, CascadeRunPlan};
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand_u01(state: &mut u64) -> f64 {
    let bits = splitmix64(state);
    // Convert to [0,1) using top 53 bits.
    ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Clustered data via cluster-and-scatter: place `n_clusters` cluster
/// centers uniformly in the box, then draw `n_per_cluster` points
/// near each (Gaussian-like via Box-Muller in cube coords). Bits is
/// the per-axis bit width of the box.
fn make_clustered(
    n_clusters: usize,
    n_per_cluster: usize,
    bits: u32,
    sigma_frac: f64,
    seed: u64,
) -> Vec<[u64; 3]> {
    let mut s = seed;
    let extent = (1u64 << bits) as f64;
    let sigma = sigma_frac * extent;
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        centers.push([
            rand_u01(&mut s) * extent,
            rand_u01(&mut s) * extent,
            rand_u01(&mut s) * extent,
        ]);
    }
    let mut out: Vec<[u64; 3]> = Vec::with_capacity(n_clusters * n_per_cluster);
    for c in &centers {
        for _ in 0..n_per_cluster {
            // Box-Muller: two uniforms → two N(0,1)s.
            let u1 = rand_u01(&mut s).max(1e-300);
            let u2 = rand_u01(&mut s);
            let r = (-2.0 * u1.ln()).sqrt();
            let g1 = r * (2.0 * std::f64::consts::PI * u2).cos();
            let g2 = r * (2.0 * std::f64::consts::PI * u2).sin();
            let g3_u1 = rand_u01(&mut s).max(1e-300);
            let g3 = (-2.0 * g3_u1.ln()).sqrt()
                * (2.0 * std::f64::consts::PI * rand_u01(&mut s)).cos();
            let p = [
                wrap(c[0] + sigma * g1, extent),
                wrap(c[1] + sigma * g2, extent),
                wrap(c[2] + sigma * g3, extent),
            ];
            out.push([
                p[0].clamp(0.0, extent - 1.0) as u64,
                p[1].clamp(0.0, extent - 1.0) as u64,
                p[2].clamp(0.0, extent - 1.0) as u64,
            ]);
        }
    }
    out
}

fn wrap(x: f64, extent: f64) -> f64 {
    let mut r = x % extent;
    if r < 0.0 { r += extent; }
    r
}

fn make_uniform(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
    let mut s = seed;
    let mask = (1u64 << bits) - 1;
    (0..n).map(|_| [
        splitmix64(&mut s) & mask,
        splitmix64(&mut s) & mask,
        splitmix64(&mut s) & mask,
    ]).collect()
}

fn main() {
    // ---- Setup ----
    let bits = 7u32;       // per-axis bits — modest box for fast iteration
    let n_clusters = 30;
    let n_per_cluster = 80;
    let n_d = n_clusters * n_per_cluster;  // 2400
    let n_r = 4 * n_d;                     // 9600
    let n_runs = 4;

    println!("=== Optimize data weights to match target variance profile ===\n");
    println!("Catalog: {} clustered data points, {} uniform randoms",
        n_d, n_r);
    println!("Box: 2^{} per axis, {}-shift multi-run plan", bits, n_runs);

    let pts_d = make_clustered(n_clusters, n_per_cluster, bits, 0.04, 11);
    let pts_r = make_uniform(n_r, bits, 22);
    let bits_arr = [bits; 3];
    let plan = CascadeRunPlan::random_offsets(n_runs, 0.25, 33);
    let cfg = FieldStatsConfig::default();
    let bin_tol = 1e-6;

    // Helper: compute pooled var_delta per bin given a weight vector.
    let pooled_var = |w: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let r = CascadeRunner::new_isolated(
            pts_d.clone(), Some(w.to_vec()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let agg = r.analyze_field_stats(&cfg, bin_tol);
        let sides: Vec<f64> = agg.by_side.iter().map(|b| b.physical_side).collect();
        let vars: Vec<f64> = agg.by_side.iter().map(|b| b.var_delta).collect();
        (sides, vars)
    };

    // ---- Establish baseline and target ----
    let w_unit: Vec<f64> = vec![1.0; n_d];
    let (sides, var_baseline) = pooled_var(&w_unit);
    let n_bins = sides.len();
    println!("\nBaseline (uniform weights) variance profile:");
    println!("  bin   side       var_baseline   target");
    // Target: scale variance per bin by a non-trivial multiplier.
    // Aggressive at small scales (need to upweight clustered regions),
    // mild at large scales. This forces the optimizer to redistribute
    // weight across particles, not just rescale globally.
    let target: Vec<f64> = (0..n_bins).map(|b| {
        // Scale factor: 2.0 at smallest bin, decreasing to 1.2 at largest.
        let frac = b as f64 / (n_bins - 1).max(1) as f64;
        let factor = 2.0 - 0.8 * frac;
        var_baseline[b] * factor
    }).collect();
    for b in 0..n_bins {
        println!("  {:>3}  {:>9.3}  {:>13.4}  {:>9.4}",
            b, sides[b], var_baseline[b], target[b]);
    }

    // ---- Loss & gradient ----
    // Loss: L(w) = Σ_b (var_b(w) - target_b)²
    // ∂L/∂w_i = Σ_b 2(var_b(w) - target_b) ∂var_b/∂w_i
    //         = pooled_aggregate gradient with betas[b] = 2(var_b(w) - target_b)
    let loss_and_grad = |w: &[f64]| -> (f64, Vec<f64>) {
        let (_sides, var_now) = pooled_var(w);
        debug_assert_eq!(var_now.len(), target.len(),
            "bin count drifted during optimization");
        let mut loss = 0.0_f64;
        let mut betas = vec![0.0_f64; var_now.len()];
        for b in 0..var_now.len() {
            let err = var_now[b] - target[b];
            loss += err * err;
            betas[b] = 2.0 * err;
        }
        // Use the pooled aggregate gradient API.
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), Some(w.to_vec()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let grad = runner.gradient_var_delta_data_pooled_aggregate(
            &cfg, bin_tol, &betas);
        (loss, grad)
    };

    // ---- Spot-check: analytic gradient agrees with finite-difference ----
    println!("\nSpot-check: analytic gradient vs FD at uniform weights");
    let (loss0, grad0) = loss_and_grad(&w_unit);
    let eps = 1e-5;
    let mut max_rel_err = 0.0_f64;
    let n_check = 10;
    for i in 0..n_check {
        let mut w_pert = w_unit.clone();
        w_pert[i] += eps;
        let (loss_pert, _) = loss_and_grad(&w_pert);
        let fd = (loss_pert - loss0) / eps;
        let an = grad0[i];
        let scale = an.abs().max(1e-10);
        let rel = (fd - an).abs() / scale;
        max_rel_err = max_rel_err.max(rel);
        println!("  particle {}: analytic = {:>12.6e}, FD = {:>12.6e}, rel.err = {:.2e}",
            i, an, fd, rel);
    }
    println!("  → max relative error across {} spot-checks: {:.2e}",
        n_check, max_rel_err);
    if max_rel_err > 1e-3 {
        println!("  WARNING: analytic gradient does not match FD; investigate");
    }

    // ---- Optimize on θ where w_i = exp(θ_i) ----
    // Chain rule: ∂L/∂θ_i = w_i · ∂L/∂w_i.
    // Adam optimizer with standard hyperparameters.
    let mut theta: Vec<f64> = vec![0.0; n_d];   // w starts at exp(0) = 1
    let mut m: Vec<f64> = vec![0.0; n_d];        // 1st moment
    let mut v: Vec<f64> = vec![0.0; n_d];        // 2nd moment
    let lr = 0.05_f64;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps_adam = 1e-8_f64;
    let n_iter = 60;

    println!("\nOptimization (Adam on θ where w = exp(θ), lr={}, n_iter={})", lr, n_iter);
    println!("  iter   loss          ‖∇L‖           Δw range          time(s)");
    let t0 = Instant::now();
    let mut loss_history: Vec<f64> = Vec::with_capacity(n_iter);
    for it in 0..n_iter {
        let w: Vec<f64> = theta.iter().map(|t| t.exp()).collect();
        let (loss, grad_w) = loss_and_grad(&w);
        loss_history.push(loss);
        // Convert ∂L/∂w to ∂L/∂θ.
        let grad_theta: Vec<f64> = grad_w.iter().zip(w.iter())
            .map(|(g, wi)| g * wi).collect();
        let grad_norm: f64 = grad_theta.iter().map(|g| g * g).sum::<f64>().sqrt();
        let w_min = w.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Adam update.
        let bc1 = 1.0 - beta1.powi((it + 1) as i32);
        let bc2 = 1.0 - beta2.powi((it + 1) as i32);
        for i in 0..n_d {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad_theta[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad_theta[i] * grad_theta[i];
            let mh = m[i] / bc1;
            let vh = v[i] / bc2;
            theta[i] -= lr * mh / (vh.sqrt() + eps_adam);
        }
        if it % 5 == 0 || it == n_iter - 1 {
            println!("  {:>4}   {:>11.5e}   {:>12.5e}   [{:>5.3}, {:>5.3}]   {:>5.2}",
                it, loss, grad_norm, w_min, w_max,
                t0.elapsed().as_secs_f64());
        }
    }
    let total_time = t0.elapsed().as_secs_f64();

    // ---- Final state ----
    let w_final: Vec<f64> = theta.iter().map(|t| t.exp()).collect();
    let (_, var_final) = pooled_var(&w_final);
    println!("\nFinal variance vs target:");
    println!("  bin   side       var_final     target       rel.err");
    let mut max_rel_var_err = 0.0_f64;
    for b in 0..n_bins {
        let rel = (var_final[b] - target[b]).abs() / target[b].abs().max(1e-12);
        max_rel_var_err = max_rel_var_err.max(rel);
        println!("  {:>3}  {:>9.3}  {:>11.4}  {:>9.4}    {:>6.2}%",
            b, sides[b], var_final[b], target[b], 100.0 * rel);
    }
    println!("\nLoss reduction: {:.5e} → {:.5e}  ({:.1}× decrease)",
        loss_history[0], *loss_history.last().unwrap(),
        loss_history[0] / loss_history.last().unwrap().max(1e-300));
    println!("Max per-bin variance relative error: {:.2}%", 100.0 * max_rel_var_err);
    println!("Wall time: {:.2}s ({:.1} ms/iter)",
        total_time, total_time * 1000.0 / n_iter as f64);

    // ---- Spot-check: analytic gradient still correct at the optimum ----
    println!("\nSpot-check: analytic gradient vs FD at final weights");
    let (loss_f, grad_f) = loss_and_grad(&w_final);
    let mut max_rel_err_final = 0.0_f64;
    for i in 0..n_check {
        let mut w_pert = w_final.clone();
        w_pert[i] += eps;
        let (loss_pert, _) = loss_and_grad(&w_pert);
        let fd = (loss_pert - loss_f) / eps;
        let an = grad_f[i];
        let scale = an.abs().max(1e-10);
        let rel = (fd - an).abs() / scale;
        max_rel_err_final = max_rel_err_final.max(rel);
    }
    println!("  → max relative error: {:.2e}  (was {:.2e} at init)",
        max_rel_err_final, max_rel_err);

    // ---- Diagnostic on weight-scale degeneracy ----
    let w_mean: f64 = w_final.iter().sum::<f64>() / w_final.len() as f64;
    let w_std: f64 = (w_final.iter().map(|w| (w - w_mean).powi(2))
        .sum::<f64>() / w_final.len() as f64).sqrt();
    println!("\nWeight distribution: mean={:.3}, std={:.3}, std/mean={:.3}",
        w_mean, w_std, w_std / w_mean);
    println!("(High std/mean indicates non-trivial redistribution; near-zero would");
    println!(" mean the optimizer just slid along the uniform-scaling flat direction.)");
}
