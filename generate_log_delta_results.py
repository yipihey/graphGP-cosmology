#!/usr/bin/env python3
"""
Generate log-delta GP reconstruction results from existing density results.

Since the original Quijote simulation data may not be available, this script:
1. Loads positions and labels from existing density results
2. Runs the full log-delta GP reconstruction pipeline
3. Computes clustering statistics (two-point, CIC, three-point)
4. Also updates the density results with clustering statistics
5. Saves both result files
"""

import os
import sys
import numpy as np

import jax
import jax.numpy as jnp

import graphgp as gp
import optax

# Import pipeline functions
from graphGP_cosmo import (
    make_kernel, build_combined_graph,
    optimize_field_log_delta, optimize_kernel,
    compute_gp_derivatives, compute_hessian_quadratic_fit,
    compute_kernel_fisher, partial_corr,
    compute_two_point_function, compute_counts_in_cells,
    compute_three_point_function,
    N0, K_NEIGHBORS, N_VOL_POINTS, OUTPUT_DIR, L_BOX,
    INIT_VARIANCE, INIT_SCALE,
    N_OPTIM_STEPS, LEARNING_RATE, N_KERNEL_STEPS, N_ALTERNATING_ROUNDS,
)


def main():
    # Load existing density results
    density_path = os.path.join(OUTPUT_DIR, "gp_reconstruction_results.npz")
    print(f"Loading existing density results from {density_path}")
    d = np.load(density_path, allow_pickle=True)

    positions = d["positions"]  # (5000, 3) in Mpc/h
    delta_density = d["delta"]
    label_a = d["label_a"]
    label_b = d["label_b"]
    N = len(positions)

    print(f"  N = {N}")
    print(f"  Position range: [{positions.min():.1f}, {positions.max():.1f}] Mpc/h")
    print(f"  Density delta range: [{delta_density.min():.3f}, {delta_density.max():.3f}]")

    # ── Normalize positions to [0,1] ──────────────────────────────
    points_norm = jnp.array(positions / L_BOX)

    # ── Build graphs ──────────────────────────────────────────────
    print("\nBuilding graphs...")
    graph_halo = gp.build_graph(points_norm, n0=N0, k=K_NEIGHBORS)

    graph_combined, n_halo, n_vol, vol_points = \
        build_combined_graph(points_norm, n_vol=N_VOL_POINTS)

    n_bar = float(N)

    # ── Initial kernel ────────────────────────────────────────────
    log_var = jnp.log(jnp.array(INIT_VARIANCE, dtype=jnp.float32))
    log_scale = jnp.log(jnp.array(INIT_SCALE / L_BOX, dtype=jnp.float32))

    # ── Alternating optimization (log-delta) ──────────────────────
    print("\nRunning log-delta GP reconstruction...")
    all_losses = []
    xi_current = None

    for rnd in range(N_ALTERNATING_ROUNDS):
        print(f"\n{'*'*60}")
        print(f"  LOG-DELTA ROUND {rnd+1}/{N_ALTERNATING_ROUNDS}")
        print(f"{'*'*60}")

        xi_current, f_map, delta_map, losses = optimize_field_log_delta(
            graph_combined, n_bar, log_var, log_scale,
            n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
            xi_init=xi_current, n_halo=n_halo, n_vol=n_vol)
        all_losses.extend(losses)

        # Kernel optimization uses f (the GP field) not delta
        log_var, log_scale = optimize_kernel(
            f_map, graph_combined, log_var, log_scale,
            n_steps=N_KERNEL_STEPS, lr=1e-3)

    # Final field optimization
    xi_current, f_map, delta_map, losses = optimize_field_log_delta(
        graph_combined, n_bar, log_var, log_scale,
        n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
        xi_init=xi_current, n_halo=n_halo, n_vol=n_vol)
    all_losses.extend(losses)

    # Extract results
    f_halo = np.array(f_map[:n_halo])
    delta_halo = np.array(delta_map[:n_halo])
    f_vol = np.array(f_map[n_halo:])
    delta_vol = np.array(delta_map[n_halo:])
    vol_points_np = np.array(vol_points)

    var_val = float(jnp.exp(log_var))
    scale_val = float(jnp.exp(log_scale)) * L_BOX

    print(f"\n  Final kernel: variance = {var_val:.4f}, "
          f"scale = {scale_val:.1f} Mpc/h")
    print(f"  f_halo range: [{f_halo.min():.3f}, {f_halo.max():.3f}]")
    print(f"  delta_halo range: [{delta_halo.min():.3f}, {delta_halo.max():.3f}]")
    print(f"  All densities positive: {np.all(delta_halo > -1)}")

    # ── Fisher matrix ─────────────────────────────────────────────
    print("\nComputing Fisher matrix...")
    fisher, fisher_unc = compute_kernel_fisher(
        f_map, graph_combined, log_var, log_scale)

    # ── Hessian and cosmic web (log-delta) ────────────────────────
    print("\nComputing Hessian and cosmic web classification...")
    cov = make_kernel(log_var, log_scale)

    gradient_f, hessian_f, eigenvalues_f, labels_geo_f, laplacian_f, s_squared_f = \
        compute_gp_derivatives(graph_halo, cov, jnp.array(f_halo),
                               log_variance=log_var, log_scale=log_scale)

    # Convert to delta-space Hessian via chain rule
    exp_f = np.exp(f_halo)
    hessian_delta = np.zeros_like(hessian_f)
    for i in range(len(f_halo)):
        outer_grad = np.outer(gradient_f[i], gradient_f[i])
        hessian_delta[i] = exp_f[i] * (hessian_f[i] + outer_grad)

    eigenvalues_delta = np.linalg.eigvalsh(hessian_delta)
    eigenvalues_delta = eigenvalues_delta[:, ::-1]

    n_positive_neg_H = np.sum(-eigenvalues_delta > 0, axis=1)
    labels_geo = np.array(["void"] * len(f_halo), dtype=object)
    labels_geo[n_positive_neg_H == 1] = "sheet"
    labels_geo[n_positive_neg_H == 2] = "filament"
    labels_geo[n_positive_neg_H == 3] = "peak"

    laplacian_delta = np.trace(hessian_delta, axis1=1, axis2=2)
    trace_part = (laplacian_delta / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    s_ij = hessian_delta - trace_part
    s_squared_delta = np.sum(s_ij ** 2, axis=(1, 2))

    # ── Clustering statistics ─────────────────────────────────────
    print("\nComputing clustering statistics...")

    # Two-point function
    print("  Two-point correlation function...")
    r_2pt, xi_2pt, xi_2pt_err = compute_two_point_function(
        positions, n_bins=20, r_max=150.0, box_size=L_BOX)

    # Counts in cells
    print("  Counts-in-cells...")
    for nc_label, nc in [("8", 8), ("16", 16)]:
        counts, density_mean, cic_bins, cic_pdf, cic_var, cic_skew, cic_S3 = \
            compute_counts_in_cells(positions, delta_halo, n_cells=nc,
                                     box_size=L_BOX)
        print(f"    n_cells={nc}: variance={cic_var:.2f}, "
              f"skewness={cic_skew:.4f}, S3={cic_S3:.3f}")

    # Use n_cells=10 for saved results
    counts_10, density_mean_10, cic_bins_10, cic_pdf_10, cic_var_10, cic_skew_10, cic_S3_10 = \
        compute_counts_in_cells(positions, delta_halo, n_cells=10, box_size=L_BOX)

    # Three-point function
    print("  Three-point function (equilateral)...")
    r_3pt, Q_3pt, Q_3pt_err, zeta_3pt, xi_3pt = compute_three_point_function(
        positions, n_bins=12, r_max=80.0, box_size=L_BOX)

    # ── Save log-delta results ────────────────────────────────────
    outfile = os.path.join(OUTPUT_DIR, "gp_log_delta_results.npz")
    np.savez(
        outfile,
        positions=positions,
        f_halo=f_halo,
        delta=delta_halo,
        f_vol=f_vol,
        delta_vol=delta_vol,
        vol_points=vol_points_np,
        gradient_f=gradient_f,
        hessian_f=hessian_f,
        eigenvalues_f=eigenvalues_f,
        gradient_delta=gradient_f * exp_f[:, None],
        hessian_delta=hessian_delta,
        eigenvalues=eigenvalues_delta,
        laplacian=laplacian_delta,
        s_squared=s_squared_delta,
        labels_geo=labels_geo,
        label_a=label_a,
        label_b=label_b,
        kernel_variance=var_val,
        kernel_scale_mpc_h=scale_val,
        fisher_matrix=np.array(fisher),
        fisher_uncertainties=np.array(fisher_unc),
        losses=np.array(all_losses),
        r_2pt=r_2pt,
        xi_2pt=xi_2pt,
        xi_2pt_err=xi_2pt_err,
        cic_bins=cic_bins_10,
        cic_pdf=cic_pdf_10,
        cic_variance=cic_var_10,
        cic_skewness=cic_skew_10,
        cic_S3=cic_S3_10,
        cic_counts=counts_10,
        r_3pt=r_3pt,
        Q_3pt=Q_3pt,
        Q_3pt_err=Q_3pt_err,
        zeta_3pt=zeta_3pt,
        xi_3pt=xi_3pt,
    )
    print(f"\n  Log-delta results saved to: {outfile}")

    # ── Also update density results with clustering statistics ────
    print("\nUpdating density results with clustering statistics...")

    # Compute clustering for density approach too
    r_2pt_d, xi_2pt_d, xi_2pt_err_d = compute_two_point_function(
        positions, n_bins=20, r_max=150.0, box_size=L_BOX)
    counts_10d, dm_10d, cic_bins_10d, cic_pdf_10d, cic_var_10d, cic_skew_10d, cic_S3_10d = \
        compute_counts_in_cells(positions, np.array(d["delta"]), n_cells=10, box_size=L_BOX)
    r_3pt_d, Q_3pt_d, Q_3pt_err_d, zeta_3pt_d, xi_3pt_d = compute_three_point_function(
        positions, n_bins=12, r_max=80.0, box_size=L_BOX)

    # Merge with existing density results
    density_out = os.path.join(OUTPUT_DIR, "gp_reconstruction_results.npz")
    existing = dict(d)
    existing.update({
        "r_2pt": r_2pt_d,
        "xi_2pt": xi_2pt_d,
        "xi_2pt_err": xi_2pt_err_d,
        "cic_bins": cic_bins_10d,
        "cic_pdf": cic_pdf_10d,
        "cic_variance": cic_var_10d,
        "cic_skewness": cic_skew_10d,
        "cic_S3": cic_S3_10d,
        "cic_counts": counts_10d,
        "r_3pt": r_3pt_d,
        "Q_3pt": Q_3pt_d,
        "Q_3pt_err": Q_3pt_err_d,
        "zeta_3pt": zeta_3pt_d,
        "xi_3pt": xi_3pt_d,
    })
    np.savez(density_out, **existing)
    print(f"  Updated density results saved to: {density_out}")

    print("\n" + "=" * 62)
    print("  All results generated successfully!")
    print("=" * 62)


if __name__ == "__main__":
    main()
