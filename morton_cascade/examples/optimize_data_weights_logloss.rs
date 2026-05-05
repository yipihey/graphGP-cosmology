// optimize_data_weights_logloss.rs
//
// Companion to `optimize_data_weights.rs` showing how to fit per-bin
// fractional errors equally instead of letting small-target bins
// dominate the residual budget.
//
// The original example minimized `L = Σ_b (μ₂_actual_b - μ₂_target_b)²`
// — sum-of-squares in absolute units. With variance targets that
// span 3-4 orders of magnitude across scales, this loss is dominated
// by the largest bins and the smallest bins converge poorly (often
// 15-30% relative error while large bins reach <1%).
//
// This example uses a **log-space loss**:
//
//   L = Σ_b (log μ₂_actual_b − log μ₂_target_b)²
//
// which weights every bin equally on a fractional-error scale. The
// chain rule gives:
//
//   ∂L/∂w_i = Σ_b 2 · (log μ₂_actual_b − log μ₂_target_b) / μ₂_actual_b
//                 · ∂μ₂_b/∂w_i
//
// so the per-bin β passed to `gradient_var_delta_data_pooled_aggregate`
// is simply `2 · (log_err) / μ₂_actual_b`.
//
// This is purely a loss-function change; the cascade gradient
// machinery is unchanged. The same trick applies to ξ, anisotropy,
// or any other observable where you want equal fractional error.
//
// Practical notes (learned the hard way):
//
//   - Floor `var_now[b]` to a small ε before forming `1/var_now`.
//     During optimization a bin can momentarily approach zero and
//     produce an exploding β; floor protects the gradient norm.
//   - Use a smaller learning rate than the absolute-error case (this
//     example: 0.02 vs 0.05). Log-space loss has steeper effective
//     curvature near the optimum because the residual scale shrinks
//     as the bins converge.
//   - For FD spot-checks of the log-loss gradient, use a larger
//     perturbation (this example: 1e-3 vs 1e-5 in the absolute case).
//     Smaller eps gets swamped by f64 truncation in `log()`.
//
// Observed results (one representative run, n_d=2400, 100 iters):
//   - Loss reduces ~40000× (1.85 → 4.5e-5)
//   - All bins converge to ≤ 0.4% relative error (vs 15-30% for the
//     small bins under absolute-error loss)
//   - Weight redistribution std/mean ≈ 0.34 (similar magnitude to the
//     absolute-error example)
//   - FD spot-check matches analytic gradient to ~1e-4 relative error
//   - ~23 ms/iter, same per-iteration cost as the absolute-error case
//
// Build & run:
//   cargo build --release --example optimize_data_weights_logloss
//   ./target/release/examples/optimize_data_weights_logloss

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
    ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

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
    let bits = 7u32;
    let n_clusters = 30;
    let n_per_cluster = 80;
    let n_d = n_clusters * n_per_cluster;
    let n_r = 4 * n_d;
    let n_runs = 4;

    println!("=== Optimize data weights with log-space loss ===");
    println!("(equal fractional error per bin, regardless of bin magnitude)\n");
    println!("Catalog: {} clustered data points, {} uniform randoms",
        n_d, n_r);
    println!("Box: 2^{} per axis, {}-shift multi-run plan", bits, n_runs);

    let pts_d = make_clustered(n_clusters, n_per_cluster, bits, 0.04, 11);
    let pts_r = make_uniform(n_r, bits, 22);
    let bits_arr = [bits; 3];
    let plan = CascadeRunPlan::random_offsets(n_runs, 0.25, 33);
    let cfg = FieldStatsConfig::default();
    let bin_tol = 1e-6;

    let pooled_var = |w: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let r = CascadeRunner::new_isolated(
            pts_d.clone(), Some(w.to_vec()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let agg = r.analyze_field_stats(&cfg, bin_tol);
        let sides: Vec<f64> = agg.by_side.iter().map(|b| b.physical_side).collect();
        let vars: Vec<f64> = agg.by_side.iter().map(|b| b.var_delta).collect();
        (sides, vars)
    };

    let w_unit: Vec<f64> = vec![1.0; n_d];
    let (sides, var_baseline) = pooled_var(&w_unit);
    let n_bins = sides.len();
    println!("\nBaseline (uniform weights) variance profile:");
    println!("  bin   side       var_baseline   target");
    let target: Vec<f64> = (0..n_bins).map(|b| {
        let frac = b as f64 / (n_bins - 1).max(1) as f64;
        let factor = 2.0 - 0.8 * frac;
        var_baseline[b] * factor
    }).collect();
    for b in 0..n_bins {
        println!("  {:>3}  {:>9.3}  {:>13.4}  {:>9.4}",
            b, sides[b], var_baseline[b], target[b]);
    }

    // Log-space loss & gradient.
    // Skip bins where target ≤ 0 or actual ≤ var_floor (log undefined or
    // numerically explosive). The var_floor protects against gradient
    // blowup when var_now → 0 during optimization.
    let var_floor = 1e-6_f64;
    let loss_and_grad = |w: &[f64]| -> (f64, Vec<f64>) {
        let (_sides, var_now) = pooled_var(w);
        let mut loss = 0.0_f64;
        let mut betas = vec![0.0_f64; var_now.len()];
        let mut n_active = 0;
        for b in 0..var_now.len() {
            if var_now[b] <= var_floor || target[b] <= var_floor { continue; }
            let log_err = var_now[b].ln() - target[b].ln();
            loss += log_err * log_err;
            // ∂(log v)/∂w = (1/v) ∂v/∂w; chain rule gives β = 2·log_err/v.
            betas[b] = 2.0 * log_err / var_now[b];
            n_active += 1;
        }
        debug_assert!(n_active > 0, "no active bins for log-loss");
        let runner = CascadeRunner::new_isolated(
            pts_d.clone(), Some(w.to_vec()),
            pts_r.clone(), None, bits_arr, plan.clone());
        let grad = runner.gradient_var_delta_data_pooled_aggregate(
            &cfg, bin_tol, &betas);
        (loss, grad)
    };

    // FD spot-check.
    // Note: log-space loss is non-linear and bin-floor-clipped, so use
    // a slightly larger eps than the linear-loss case to get clean FD
    // signal above truncation noise.
    println!("\nSpot-check: analytic gradient vs FD at uniform weights");
    let (loss0, grad0) = loss_and_grad(&w_unit);
    let eps = 1e-3;
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

    // Adam on θ where w = exp(θ).
    let mut theta: Vec<f64> = vec![0.0; n_d];
    let mut m: Vec<f64> = vec![0.0; n_d];
    let mut v: Vec<f64> = vec![0.0; n_d];
    let lr = 0.02_f64;
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps_adam = 1e-8_f64;
    let n_iter = 100;

    println!("\nOptimization (Adam on θ where w = exp(θ), lr={}, n_iter={})",
        lr, n_iter);
    println!("  iter   loss          ‖∇L‖           Δw range          time(s)");
    let t0 = Instant::now();
    let mut loss_history: Vec<f64> = Vec::with_capacity(n_iter);
    for it in 0..n_iter {
        let w: Vec<f64> = theta.iter().map(|t| t.exp()).collect();
        let (loss, grad_w) = loss_and_grad(&w);
        loss_history.push(loss);
        let grad_theta: Vec<f64> = grad_w.iter().zip(w.iter())
            .map(|(g, wi)| g * wi).collect();
        let grad_norm: f64 = grad_theta.iter().map(|g| g * g).sum::<f64>().sqrt();
        let w_min = w.iter().cloned().fold(f64::INFINITY, f64::min);
        let w_max = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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

    let w_mean: f64 = w_final.iter().sum::<f64>() / w_final.len() as f64;
    let w_std: f64 = (w_final.iter().map(|w| (w - w_mean).powi(2))
        .sum::<f64>() / w_final.len() as f64).sqrt();
    println!("\nWeight distribution: mean={:.3}, std={:.3}, std/mean={:.3}",
        w_mean, w_std, w_std / w_mean);

    println!("\nCompare to optimize_data_weights.rs (absolute-error loss):");
    println!("- Absolute loss: small bins lag because they contribute less to L");
    println!("- Log-space loss: every bin weighted equally on fractional scale");
}
