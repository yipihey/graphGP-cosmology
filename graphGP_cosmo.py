#!/opt/homebrew/opt/python@3.11/bin/python3.11
"""
GP Density Field Reconstruction from Quijote Halo Catalogs
==========================================================

Given halo positions with labels (mass, velocity), reconstruct the smooth
underlying density field and its Hessian using GraphGP, then ask whether
labels depend on the local tidal environment.

Data: Quijote simulation halos (Cuesta-Lazaro & Mishra-Sharma 2023).
  - 5000 heaviest halos per simulation
  - Box size: L = 1000 Mpc/h (periodic)
  - Features: positions (x,y,z), velocities (vx,vy,vz), mass

Pipeline:
  0. Load data
  1. Build GraphGP neighbor graph
  2. Set up kernel (correlation function)
  3. MAP optimization (alternating field & kernel)
  4. Compute Hessian → cosmic web classification
  5. Label-environment correlations (Q1-Q4)

Author: Abel, Frank, Dodge, Clark, Wechsler & collaborators
Date: March 2026
"""

import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit

import graphgp as gp
import optax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

DATA_DIR = "/Users/tabel/Research/data/quijote_halos_set_diffuser_data"
L_BOX = 1000.0

# GraphGP parameters
N0 = 100
K_NEIGHBORS = 15

# Volume-filling points for likelihood integral
N_VOL_POINTS = 3000
VOL_SEED = 99

# Kernel initial guesses
INIT_VARIANCE = 1.0
INIT_SCALE = 30.0  # Mpc/h

# Optimization
N_OPTIM_STEPS = 200
LEARNING_RATE = 1e-2
N_KERNEL_STEPS = 50
N_ALTERNATING_ROUNDS = 3

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# STEP 0: LOAD DATA
# =====================================================================

def load_data(data_dir, sim_idx=0):
    """
    Load one simulation from train_halos.npy.

    Returns:
        positions: (N, 3) in Mpc/h
        velocities: (N, 3) in km/s
        masses: (N,) in Msun/h
    """
    print("=" * 60)
    print(f"STEP 0: Loading simulation {sim_idx}")
    print("=" * 60)

    halos = np.load(os.path.join(data_dir, "train_halos.npy"))
    print(f"  Full dataset shape: {halos.shape}")  # (1800, 5000, 7)

    sim = halos[sim_idx]  # (5000, 7): x,y,z,vx,vy,vz,mass
    positions = sim[:, :3]
    velocities = sim[:, 3:6]
    masses = sim[:, 6]

    N = len(positions)
    print(f"  Loaded {N} halos from sim {sim_idx}")
    print(f"  Position range: [{positions.min():.1f}, {positions.max():.1f}] Mpc/h")
    v_mag = np.linalg.norm(velocities, axis=1)
    print(f"  Velocity range: [{v_mag.min():.1f}, {v_mag.max():.1f}] km/s")
    print(f"  Mass range: [{masses.min():.2e}, {masses.max():.2e}] Msun/h")

    return positions, velocities, masses


# =====================================================================
# STEP 1: BUILD THE GRAPHGP NEIGHBOR GRAPH
# =====================================================================

def build_graph(points):
    """
    Build the Vecchia-approximation neighbor graph on [0,1]^3 positions.

    Args:
        points: (N, 3) positions in Mpc/h

    Returns:
        graph: GraphGP Graph object
        points_norm: (N, 3) JAX array in [0,1] box units
    """
    print("\n" + "=" * 60)
    print("STEP 1: Building neighbor graph")
    print("=" * 60)

    points_norm = jnp.array(points / L_BOX)

    print(f"  N = {len(points)}, n0 = {N0}, k = {K_NEIGHBORS}")
    graph = gp.build_graph(points_norm, n0=N0, k=K_NEIGHBORS)
    print(f"  Graph built successfully.")

    return graph, points_norm


def build_combined_graph(halo_points_norm, n_vol=N_VOL_POINTS, seed=VOL_SEED):
    """
    Build graph on halo positions + uniform volume-filling points.

    The volume points provide unbiased Monte Carlo integration for the
    Poisson likelihood normalization (the integral of lambda over the box).

    Args:
        halo_points_norm: (N_halo, 3) halo positions in [0,1] box units
        n_vol: number of uniform volume-filling points
        seed: random seed for volume point generation

    Returns:
        graph: GraphGP Graph on combined points (halos first, then volume)
        n_halo: number of halo points
        n_vol: number of volume points
        vol_points: (n_vol, 3) volume point positions in [0,1]
    """
    n_halo = len(halo_points_norm)
    print(f"\n  Building combined graph: {n_halo} halos + {n_vol} volume points")

    rng = np.random.RandomState(seed)
    vol_points = jnp.array(rng.uniform(0, 1, size=(n_vol, 3)).astype(np.float32))

    combined = jnp.concatenate([halo_points_norm, vol_points], axis=0)
    N_total = len(combined)

    n0 = min(N0, N_total // 2)
    k = min(K_NEIGHBORS, n0 - 1)

    print(f"  N_total = {N_total}, n0 = {n0}, k = {k}")
    graph = gp.build_graph(combined, n0=n0, k=k)
    print(f"  Combined graph built.")

    return graph, n_halo, n_vol, vol_points


# =====================================================================
# STEP 2: KERNEL
# =====================================================================

def make_kernel(log_variance, log_scale):
    """
    Differentiable RBF kernel using make_cov_bins + manual formula.

    Works in normalized [0,1] box units.
    Returns (r_bins, c_vals) tuple for GraphGP.
    """
    variance = jnp.exp(log_variance)
    scale = jnp.exp(log_scale)

    r_bins = gp.extras.make_cov_bins(r_min=1e-5, r_max=0.5, n_bins=1000)
    c_vals = variance * jnp.exp(-0.5 * (r_bins / scale) ** 2)
    # Add jitter at r=0 for numerical stability
    c_vals = c_vals.at[0].add(1e-4 * variance)

    return (r_bins, c_vals)


# =====================================================================
# STEP 3: MAP OPTIMIZATION (xi-space for field, delta-space for kernel)
# =====================================================================

def poisson_log_likelihood(delta, n_bar, n_halo, n_vol=0):
    """
    Poisson point process log-likelihood with volume integral.

    ln L = sum_{i in halos} ln[n_bar * (1 + delta_i)]
         - n_bar * (V / N_vol) * sum_{j in vol} max(0, 1 + delta_j)

    The volume integral is estimated via Monte Carlo over uniform points.
    Density is clipped to >= 0 in the integral (physical constraint).
    """
    delta_halo = delta[:n_halo]
    density_halo = jnp.clip(1.0 + delta_halo, 1e-10, None)
    ll = jnp.sum(jnp.log(n_bar * density_halo))

    if n_vol > 0:
        delta_vol = delta[n_halo:]
        density_vol = jnp.clip(1.0 + delta_vol, 1e-10, None)
        integral = n_bar * (1.0 / n_vol) * jnp.sum(density_vol)
        ll -= integral
    else:
        ll -= n_halo

    return ll


def optimize_field(graph, n_bar, log_variance, log_scale,
                   n_steps=200, lr=1e-2, xi_init=None,
                   n_halo=None, n_vol=0):
    """
    MAP field estimation in xi-space (white-noise parameterization).

    delta = gp.generate(graph, cov, xi)
    Prior: p(xi) = N(0, I), so log-prior = -0.5 * ||xi||^2

    When n_vol > 0, the Poisson likelihood includes a Monte Carlo volume
    integral over uniform volume-filling points (indices n_halo..N-1).

    Returns:
        xi_map: (N,) MAP white-noise parameters
        delta_map: (N,) MAP density field (in original point order)
        losses: list of -log_posterior values
    """
    print("\n" + "=" * 60)
    print("STEP 3a: Optimizing density field (fixed kernel, xi-space)")
    print("=" * 60)

    N = len(graph.points)
    cov = make_kernel(log_variance, log_scale)

    # Default: all points are halos (backward compatible)
    _n_halo = n_halo if n_halo is not None else N
    _n_vol = n_vol

    if xi_init is None:
        key = jax.random.PRNGKey(42)
        xi = 0.01 * jax.random.normal(key, shape=(N,))
    else:
        xi = xi_init

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(xi)

    @jit
    def loss_fn(xi):
        delta = gp.generate(graph, cov, xi)
        ll = poisson_log_likelihood(delta, n_bar, _n_halo, _n_vol)
        log_prior = -0.5 * jnp.dot(xi, xi)
        return -(ll + log_prior)

    grad_fn = jit(grad(loss_fn))

    losses = []
    for step in range(n_steps):
        loss_val = loss_fn(xi)
        grads = grad_fn(xi)
        updates, opt_state = optimizer.update(grads, opt_state, xi)
        xi = optax.apply_updates(xi, updates)
        losses.append(float(loss_val))

        if step % 50 == 0 or step == n_steps - 1:
            delta = gp.generate(graph, cov, xi)
            d_halo = delta[:_n_halo]
            msg = (f"  Step {step:4d}: loss = {loss_val:.2f}, "
                   f"delta_halo = [{float(d_halo.min()):.3f}, "
                   f"{float(d_halo.max()):.3f}]")
            if _n_vol > 0:
                d_vol = delta[_n_halo:]
                msg += f", <delta_vol> = {float(d_vol.mean()):.3f}"
            print(msg)

    delta_map = gp.generate(graph, cov, xi)
    return xi, delta_map, losses


def optimize_kernel(delta_fixed, graph,
                    init_log_var, init_log_scale,
                    n_steps=50, lr=1e-3):
    """
    Optimize kernel hyperparameters at fixed field (delta-space).

    At fixed delta, the Poisson term is constant w.r.t. kernel params.
    Only the GP log-prior matters:
      loss = 0.5 * ||xi||^2 + generate_logdet(graph, cov)
    where xi = generate_inv(graph, cov, delta_fixed).

    Returns:
        (opt_log_var, opt_log_scale)
    """
    print("\n" + "=" * 60)
    print("STEP 3b: Optimizing kernel hyperparameters (fixed field)")
    print("=" * 60)

    params = jnp.array([init_log_var, init_log_scale])
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jit
    def loss_fn(params):
        cov = make_kernel(params[0], params[1])
        xi = gp.generate_inv(graph, cov, delta_fixed)
        logdet_L = gp.generate_logdet(graph, cov)
        # -log p(delta|theta) = 0.5*||xi||^2 + logdet_L  (up to constant)
        return 0.5 * jnp.dot(xi, xi) + logdet_L

    grad_fn = jit(grad(loss_fn))

    for step in range(n_steps):
        loss_val = loss_fn(params)
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 10 == 0 or step == n_steps - 1:
            var = float(jnp.exp(params[0]))
            scale = float(jnp.exp(params[1]))
            print(f"  Step {step:4d}: loss = {loss_val:.2f}, "
                  f"variance = {var:.4f}, "
                  f"scale = {scale * L_BOX:.1f} Mpc/h")

    return params[0], params[1]


# =====================================================================
# STEP 3c: FISHER MATRIX FOR KERNEL UNCERTAINTIES
# =====================================================================

def compute_kernel_fisher(delta, graph, opt_log_var, opt_log_scale):
    """
    Fisher matrix (Hessian of neg-log-marginal-likelihood) at MAP kernel.

    Returns:
        fisher: (2, 2) matrix
        uncertainties: (2,) = sqrt(diag(inv(fisher)))
    """
    print("\n  Computing Fisher matrix for kernel uncertainties...")

    def neg_log_marginal(params):
        cov = make_kernel(params[0], params[1])
        xi = gp.generate_inv(graph, cov, delta)
        logdet_L = gp.generate_logdet(graph, cov)
        return 0.5 * jnp.dot(xi, xi) + logdet_L

    params_map = jnp.array([opt_log_var, opt_log_scale])
    fisher = jax.hessian(neg_log_marginal)(params_map)

    try:
        fisher_inv = jnp.linalg.inv(fisher)
        uncertainties = jnp.sqrt(jnp.abs(jnp.diag(fisher_inv)))
    except Exception:
        uncertainties = jnp.array([jnp.nan, jnp.nan])

    print(f"  Fisher matrix:\n    {fisher}")
    print(f"  log-variance uncertainty: {float(uncertainties[0]):.4f}")
    print(f"  log-scale uncertainty:    {float(uncertainties[1]):.4f}")

    return fisher, uncertainties


# =====================================================================
# STEP 4: COMPUTE HESSIAN AND CLASSIFY LOCAL GEOMETRY
# =====================================================================

def compute_hessian_quadratic_fit(delta, points):
    """
    Estimate gradient and Hessian at each point via local quadratic fits
    using k-nearest-neighbor weighted least squares.

    Args:
        delta: (N,) field values
        points: (N, 3) normalized positions in [0,1]

    Returns:
        gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared
    """
    print("\n" + "=" * 60)
    print("STEP 4: Computing Hessian (local quadratic fits)")
    print("=" * 60)

    from scipy.spatial import cKDTree

    N = len(points)
    pos_np = np.array(points)
    delta_np = np.array(delta)

    tree = cKDTree(pos_np, boxsize=1.0)
    k_fit = min(20, N - 1)

    print(f"  Fitting local quadratics using {k_fit} neighbors...")

    gradient = np.zeros((N, 3))
    hessian = np.zeros((N, 3, 3))

    for i in range(N):
        dists, idxs = tree.query(pos_np[i], k=k_fit + 1)
        idxs = idxs[1:]

        dx = pos_np[idxs] - pos_np[i]
        dx = dx - np.round(dx)  # periodic wrap

        df = delta_np[idxs] - delta_np[i]

        A = np.column_stack([
            dx[:, 0], dx[:, 1], dx[:, 2],
            0.5 * dx[:, 0]**2,
            0.5 * dx[:, 1]**2,
            0.5 * dx[:, 2]**2,
            dx[:, 0] * dx[:, 1],
            dx[:, 0] * dx[:, 2],
            dx[:, 1] * dx[:, 2],
        ])

        weights = 1.0 / (dists[1:] + 1e-10)
        Aw = A * weights[:, None]
        dfw = df * weights

        try:
            coeffs, _, _, _ = np.linalg.lstsq(Aw, dfw, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(9)

        gradient[i] = coeffs[:3]
        hessian[i, 0, 0] = coeffs[3]
        hessian[i, 1, 1] = coeffs[4]
        hessian[i, 2, 2] = coeffs[5]
        hessian[i, 0, 1] = hessian[i, 1, 0] = coeffs[6]
        hessian[i, 0, 2] = hessian[i, 2, 0] = coeffs[7]
        hessian[i, 1, 2] = hessian[i, 2, 1] = coeffs[8]

    eigenvalues = np.linalg.eigvalsh(hessian)
    eigenvalues = eigenvalues[:, ::-1]  # descending: l1 >= l2 >= l3

    # Classify by number of positive eigenvalues of -H
    n_positive_neg_H = np.sum(-eigenvalues > 0, axis=1)
    labels_geo = np.array(["void"] * N, dtype=object)
    labels_geo[n_positive_neg_H == 1] = "sheet"
    labels_geo[n_positive_neg_H == 2] = "filament"
    labels_geo[n_positive_neg_H == 3] = "peak"

    unique, counts = np.unique(labels_geo, return_counts=True)
    print(f"\n  Cosmic web classification:")
    for u, c in zip(unique, counts):
        print(f"    {u:10s}: {c:5d} ({100*c/N:.1f}%)")

    laplacian = np.trace(hessian, axis1=1, axis2=2)

    trace_part = (laplacian / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    s_ij = hessian - trace_part
    s_squared = np.sum(s_ij ** 2, axis=(1, 2))

    print(f"\n  Laplacian range: [{laplacian.min():.4f}, {laplacian.max():.4f}]")
    print(f"  Tidal shear s^2 range: [{s_squared.min():.4f}, {s_squared.max():.4f}]")

    return gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared


def compute_gp_derivatives(graph, cov, delta, log_variance=None, log_scale=None):
    """
    Compute gradient and Hessian of the GP posterior mean at each point
    using Vecchia-local conditionals and JAX autodiff.

    For refined points (i >= n0): differentiate the Vecchia conditional mean
      mu(x) = k(x, X_nbrs) @ K_nbrs^{-1} @ delta_nbrs
    For dense points (i < n0): differentiate
      mu(x) = k(x, X_dense) @ K_dense^{-1} @ delta_dense

    Uses the analytical RBF kernel C(r) = var * exp(-0.5*(r/l)^2) for
    differentiation (not jnp.interp, which is piecewise-linear and has
    zero second derivatives).  The cov tuple is still used to build K_nn
    and K_dd via jnp.interp (where only values, not derivatives, matter).

    Args:
        graph: GraphGP Graph object
        cov: (r_bins, c_vals) kernel tuple (used for covariance matrices)
        delta: (N,) field values in original point order
        log_variance: log of kernel variance (if None, estimated from cov)
        log_scale: log of kernel scale (if None, estimated from cov)

    Returns:
        gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared
    """
    print("\n" + "=" * 60)
    print("STEP 4: Computing Hessian (GP derivatives via autodiff)")
    print("=" * 60)

    r_bins, c_vals = cov
    n0 = graph.offsets[0]
    N = len(graph.points)

    # Extract kernel parameters
    if log_variance is None or log_scale is None:
        # Estimate from the tabulated kernel
        variance = c_vals[0]  # C(0) ≈ variance (+ small jitter)
        # Find scale from C(r) = variance/e ⟹ r = scale
        half_idx = jnp.searchsorted(-c_vals, -variance / jnp.e)
        scale = r_bins[half_idx]
    else:
        variance = jnp.exp(log_variance)
        scale = jnp.exp(log_scale)

    print(f"  Kernel: variance = {float(variance):.4f}, "
          f"scale = {float(scale):.4f} box units")

    # Reorder delta into graph order
    delta_graph = delta[graph.indices]

    def rbf_kernel(r_sq):
        """Smooth RBF kernel as a function of squared distance."""
        return variance * jnp.exp(-0.5 * r_sq / (scale ** 2))

    def safe_dist_sq(x, axis=-1):
        """Squared distance (no norm singularity)."""
        return jnp.sum(x ** 2, axis=axis)

    # --- Refined points (i >= n0): Vecchia-local conditional ---
    n_refined = N - n0
    k = graph.neighbors.shape[1]
    print(f"  Refined points: {n_refined}, neighbors k = {k}")
    print(f"  Dense points: {n0}")

    # Gather neighbor data for all refined points
    neighbor_points = graph.points[graph.neighbors]        # (n_refined, k, 3)
    neighbor_values = delta_graph[graph.neighbors]         # (n_refined, k)
    refined_points = graph.points[n0:]                     # (n_refined, 3)

    def refined_conditional_mean(x_query, nbr_pts, nbr_vals):
        """Conditional mean at x_query given neighbor points and values."""
        # Cross-covariance: use smooth RBF for differentiability
        diffs = x_query[None, :] - nbr_pts                # (k, 3)
        k_star = rbf_kernel(safe_dist_sq(diffs, axis=1))   # (k,)

        # Neighbor-neighbor covariance: use interp (only values needed, not derivatives)
        diffs_nn = nbr_pts[:, None, :] - nbr_pts[None, :, :]
        dists_nn = jnp.sqrt(safe_dist_sq(diffs_nn, axis=2) + 1e-30)
        K_nn = jnp.interp(dists_nn, r_bins, c_vals)       # (k, k)

        alpha = jnp.linalg.solve(K_nn, nbr_vals)
        return jnp.dot(k_star, alpha)

    def refined_grad(x_query, nbr_pts, nbr_vals):
        return jax.grad(refined_conditional_mean)(x_query, nbr_pts, nbr_vals)

    def refined_hess(x_query, nbr_pts, nbr_vals):
        return jax.hessian(refined_conditional_mean)(x_query, nbr_pts, nbr_vals)

    print("  Computing refined-point gradients and Hessians...")
    grad_refined = jax.vmap(refined_grad)(refined_points, neighbor_points, neighbor_values)
    hess_refined = jax.vmap(refined_hess)(refined_points, neighbor_points, neighbor_values)

    # --- Dense points (i < n0): condition on all n0 dense points ---
    # Using rbf_kernel(r_sq) avoids the norm singularity entirely — the
    # squared-distance formulation is smooth everywhere including at r=0.
    dense_points = graph.points[:n0]                       # (n0, 3)
    dense_values = delta_graph[:n0]                        # (n0,)

    # Precompute alpha_dense = K_dd^{-1} @ delta_dense
    diffs_dd = dense_points[:, None, :] - dense_points[None, :, :]
    dists_dd = jnp.sqrt(safe_dist_sq(diffs_dd, axis=2) + 1e-30)
    K_dd = jnp.interp(dists_dd, r_bins, c_vals)
    alpha_dense = jnp.linalg.solve(K_dd, dense_values)

    def dense_mean_at(x_query):
        """Conditional mean at x_query using precomputed alpha_dense."""
        diffs = x_query[None, :] - dense_points            # (n0, 3)
        k_star = rbf_kernel(safe_dist_sq(diffs, axis=1))   # (n0,)
        return jnp.dot(k_star, alpha_dense)

    print("  Computing dense-point gradients and Hessians...")
    grad_dense = jax.vmap(jax.grad(dense_mean_at))(dense_points)
    hess_dense = jax.vmap(jax.hessian(dense_mean_at))(dense_points)

    # --- Combine in graph order, then reorder to original order ---
    grad_graph = jnp.concatenate([grad_dense, grad_refined], axis=0)   # (N, 3)
    hess_graph = jnp.concatenate([hess_dense, hess_refined], axis=0)   # (N, 3, 3)

    # Reorder from graph order to original order
    inv_perm = jnp.argsort(graph.indices)
    gradient = np.array(grad_graph[inv_perm])
    hessian = np.array(hess_graph[inv_perm])

    # Symmetrize (remove floating-point asymmetry ~1e-7 relative)
    hessian = 0.5 * (hessian + np.transpose(hessian, (0, 2, 1)))

    # --- Classify ---
    eigenvalues = np.linalg.eigvalsh(hessian)
    eigenvalues = eigenvalues[:, ::-1]  # descending: l1 >= l2 >= l3

    n_positive_neg_H = np.sum(-eigenvalues > 0, axis=1)
    labels_geo = np.array(["void"] * N, dtype=object)
    labels_geo[n_positive_neg_H == 1] = "sheet"
    labels_geo[n_positive_neg_H == 2] = "filament"
    labels_geo[n_positive_neg_H == 3] = "peak"

    unique, counts = np.unique(labels_geo, return_counts=True)
    print(f"\n  Cosmic web classification:")
    for u, c in zip(unique, counts):
        print(f"    {u:10s}: {c:5d} ({100*c/N:.1f}%)")

    laplacian = np.trace(hessian, axis1=1, axis2=2)

    trace_part = (laplacian / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    s_ij = hessian - trace_part
    s_squared = np.sum(s_ij ** 2, axis=(1, 2))

    print(f"\n  Laplacian range: [{laplacian.min():.4f}, {laplacian.max():.4f}]")
    print(f"  Tidal shear s^2 range: [{s_squared.min():.4f}, {s_squared.max():.4f}]")

    return gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared


def predict_at_points(delta_known, points_known, new_points, cov, k=15):
    """
    Predict GP field at new locations via k-NN conditional mean.

    For each new point, finds k nearest neighbors among known points
    and computes the Vecchia conditional mean:
      mu(x) = k(x, X_nbrs) @ K_nbrs^{-1} @ delta_nbrs

    Args:
        delta_known: (N,) field values at known positions
        points_known: (N, 3) known positions (JAX or numpy)
        new_points: (M, 3) positions to predict at
        cov: (r_bins, c_vals) kernel tuple
        k: number of neighbors

    Returns:
        predictions: (M,) predicted field values
    """
    from scipy.spatial import cKDTree

    pts = np.array(points_known)
    vals = np.array(delta_known)
    new_pts = np.array(new_points)
    r_bins = np.array(cov[0])
    c_vals = np.array(cov[1])

    tree = cKDTree(pts, boxsize=1.0)
    M = len(new_pts)
    predictions = np.zeros(M)

    for i in range(M):
        dists, idxs = tree.query(new_pts[i], k=k)

        # Cross-covariance: k(x_new, neighbors)
        k_star = np.interp(dists, r_bins, c_vals)

        # Neighbor-neighbor covariance
        nbr_pts = pts[idxs]
        diffs = nbr_pts[:, None, :] - nbr_pts[None, :, :]
        diffs = diffs - np.round(diffs)  # periodic wrap
        dists_nn = np.sqrt(np.sum(diffs ** 2, axis=2))
        K_nn = np.interp(dists_nn, r_bins, c_vals)

        alpha = np.linalg.solve(K_nn, vals[idxs])
        predictions[i] = np.dot(k_star, alpha)

    return predictions


# =====================================================================
# STEP 5: LABEL-ENVIRONMENT CORRELATIONS
# =====================================================================

def partial_corr(x, y, z):
    """Pearson correlation of x and y after regressing out z."""
    from scipy.stats import pearsonr
    from numpy.polynomial import polynomial as P
    cx = P.polyfit(z, x, 1)
    cy = P.polyfit(z, y, 1)
    rx = x - P.polyval(z, cx)
    ry = y - P.polyval(z, cy)
    return pearsonr(rx, ry)


def environment_label_analysis(delta, laplacian, s_squared, eigenvalues,
                                labels_geo, label_a, label_b,
                                label_a_name="log10(Mass)",
                                label_b_name="|Velocity|"):
    """
    Test whether halo properties correlate with local environment
    beyond density alone (Q1-Q4).
    """
    print("\n" + "=" * 60)
    print("STEP 5: Label-environment correlations")
    print("=" * 60)

    from scipy.stats import pearsonr

    # Q1: correlation with density
    r_a_delta, p_a_delta = pearsonr(delta, label_a)
    r_b_delta, p_b_delta = pearsonr(delta, label_b)
    print(f"\n  Q1: Correlation with density")
    print(f"    Corr({label_a_name}, d) = {r_a_delta:+.4f}  (p = {p_a_delta:.2e})")
    print(f"    Corr({label_b_name}, d) = {r_b_delta:+.4f}  (p = {p_b_delta:.2e})")

    # Q2: partial correlation with tidal shear at fixed delta
    r_a_s2, p_a_s2 = partial_corr(label_a, s_squared, delta)
    r_b_s2, p_b_s2 = partial_corr(label_b, s_squared, delta)
    print(f"\n  Q2: Partial correlation with tidal shear s^2 (at fixed d)")
    print(f"    Corr({label_a_name}, s^2 | d) = {r_a_s2:+.4f}  (p = {p_a_s2:.2e})")
    print(f"    Corr({label_b_name}, s^2 | d) = {r_b_s2:+.4f}  (p = {p_b_s2:.2e})")

    # Q3: partial correlation with prolateness
    denom = np.where(np.abs(eigenvalues[:, 2]) > 1e-10,
                     eigenvalues[:, 2], 1e-10)
    prolateness = eigenvalues[:, 0] / denom

    r_a_prol, p_a_prol = partial_corr(label_a, prolateness, delta)
    r_b_prol, p_b_prol = partial_corr(label_b, prolateness, delta)
    print(f"\n  Q3: Partial correlation with prolateness l1/l3 (at fixed d)")
    print(f"    Corr({label_a_name}, l1/l3 | d) = {r_a_prol:+.4f}  (p = {p_a_prol:.2e})")
    print(f"    Corr({label_b_name}, l1/l3 | d) = {r_b_prol:+.4f}  (p = {p_b_prol:.2e})")

    # Q4: mean labels by cosmic web type
    print(f"\n  Q4: Mean labels by cosmic web type")
    print(f"    {'Type':10s} {'N':>6s} {'<'+label_a_name+'>':>14s} "
          f"{'<'+label_b_name+'>':>14s}")
    print(f"    {'-'*50}")
    for geo_type in ["peak", "filament", "sheet", "void"]:
        mask = labels_geo == geo_type
        if mask.sum() > 0:
            print(f"    {geo_type:10s} {mask.sum():6d} "
                  f"{label_a[mask].mean():14.4f} "
                  f"{label_b[mask].mean():14.4f}")

    results = {
        "r_a_delta": r_a_delta, "p_a_delta": p_a_delta,
        "r_b_delta": r_b_delta, "p_b_delta": p_b_delta,
        "r_a_s2": r_a_s2, "p_a_s2": p_a_s2,
        "r_b_s2": r_b_s2, "p_b_s2": p_b_s2,
        "r_a_prol": r_a_prol, "p_a_prol": p_a_prol,
        "r_b_prol": r_b_prol, "p_b_prol": p_b_prol,
        "prolateness": prolateness,
    }
    return results


# =====================================================================
# STEP 5b: Q4 LABEL-DEPENDENT CLUSTERING
# =====================================================================

def label_dependent_clustering(positions, label_a, label_a_name, graph_full):
    """
    Q4 extension: split halos by median label, learn separate kernels,
    compare correlation lengths.
    """
    print("\n" + "=" * 60)
    print("STEP 5b: Label-dependent clustering (Q4)")
    print("=" * 60)

    median_a = np.median(label_a)
    mask_hi = label_a >= median_a
    mask_lo = label_a < median_a

    results = {}
    for name, mask in [("high", mask_hi), ("low", mask_lo)]:
        pos_sub = positions[mask]
        N_sub = len(pos_sub)
        print(f"\n  {name}-{label_a_name} subsample: N = {N_sub}")

        points_sub = jnp.array(pos_sub / L_BOX)
        n0_sub = min(N0, N_sub // 2)
        k_sub = min(K_NEIGHBORS, n0_sub - 1)

        graph_sub = gp.build_graph(points_sub, n0=n0_sub, k=k_sub)
        n_bar_sub = float(N_sub)

        # Quick field + kernel optimization
        init_lv = jnp.log(jnp.array(INIT_VARIANCE, dtype=jnp.float32))
        init_ls = jnp.log(jnp.array(INIT_SCALE / L_BOX, dtype=jnp.float32))

        _, delta_sub, _ = optimize_field(
            graph_sub, n_bar_sub, init_lv, init_ls,
            n_steps=100, lr=LEARNING_RATE)

        opt_lv, opt_ls = optimize_kernel(
            delta_sub, graph_sub, init_lv, init_ls,
            n_steps=30, lr=1e-3)

        var_val = float(jnp.exp(opt_lv))
        scale_val = float(jnp.exp(opt_ls)) * L_BOX
        print(f"  {name}: variance = {var_val:.4f}, scale = {scale_val:.1f} Mpc/h")
        results[name] = {"variance": var_val, "scale_mpc_h": scale_val}

    print(f"\n  Scale ratio (high/low): "
          f"{results['high']['scale_mpc_h'] / results['low']['scale_mpc_h']:.3f}")
    return results


# =====================================================================
# PLOTTING
# =====================================================================

def make_plots(positions, delta, eigenvalues, labels_geo,
               label_a, label_b, s_squared, laplacian, losses,
               opt_log_var, opt_log_scale, fisher_unc=None,
               q4_results=None,
               label_a_name="log10(Mass)", label_b_name="|Velocity|"):
    """Generate summary plots."""

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16))

    # (a) Convergence
    ax = axes[0, 0]
    ax.plot(losses)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("-log posterior")
    ax.set_title("(a) MAP convergence")

    # (b) Density field slice
    ax = axes[0, 1]
    z_slice = positions[:, 2]
    mask = (z_slice > 0.45) & (z_slice < 0.55)
    sc = ax.scatter(positions[mask, 0], positions[mask, 1],
                    c=delta[mask], s=8, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("x [box units]")
    ax.set_ylabel("y [box units]")
    ax.set_title("(b) Reconstructed d (thin z-slice)")
    plt.colorbar(sc, ax=ax, label="d")

    # (c) Cosmic web classification
    ax = axes[0, 2]
    color_map = {"peak": "red", "filament": "orange",
                 "sheet": "lightblue", "void": "navy"}
    colors = [color_map.get(g, "gray") for g in labels_geo[mask]]
    ax.scatter(positions[mask, 0], positions[mask, 1], c=colors, s=8)
    ax.set_xlabel("x [box units]")
    ax.set_ylabel("y [box units]")
    ax.set_title("(c) Cosmic web (thin z-slice)")
    for label, color in color_map.items():
        ax.scatter([], [], c=color, label=label, s=30)
    ax.legend(loc="upper right", fontsize=8)

    # (d) Label a vs density
    ax = axes[1, 0]
    ax.scatter(delta, label_a, s=1, alpha=0.3)
    ax.set_xlabel("d (reconstructed)")
    ax.set_ylabel(label_a_name)
    ax.set_title(f"(d) {label_a_name} vs density")

    # (e) Label b vs tidal shear
    ax = axes[1, 1]
    ax.scatter(s_squared, label_b, s=1, alpha=0.3)
    ax.set_xlabel("s^2 (tidal shear)")
    ax.set_ylabel(label_b_name)
    ax.set_title(f"(e) {label_b_name} vs tidal shear")

    # (f) Eigenvalue distribution
    ax = axes[1, 2]
    ax.hist(eigenvalues[:, 0], bins=50, alpha=0.5, label="l1", density=True)
    ax.hist(eigenvalues[:, 1], bins=50, alpha=0.5, label="l2", density=True)
    ax.hist(eigenvalues[:, 2], bins=50, alpha=0.5, label="l3", density=True)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title("(f) Hessian eigenvalue distributions")
    ax.legend()

    # (g) Learned kernel C(r) with Fisher uncertainty
    ax = axes[2, 0]
    variance = float(jnp.exp(opt_log_var))
    scale = float(jnp.exp(opt_log_scale))
    r_plot = np.linspace(0, 0.3, 300)
    cr_plot = variance * np.exp(-0.5 * (r_plot / scale) ** 2)
    ax.plot(r_plot * L_BOX, cr_plot, 'b-', lw=2, label="MAP kernel")
    if fisher_unc is not None:
        sig_lv = float(fisher_unc[0])
        sig_ls = float(fisher_unc[1])
        # Upper/lower bounds via +/- 1sigma in log-space
        for sign, ls in [(1, '--'), (-1, '--')]:
            v_b = np.exp(np.log(variance) + sign * sig_lv)
            s_b = np.exp(np.log(scale) + sign * sig_ls)
            cr_b = v_b * np.exp(-0.5 * (r_plot / s_b) ** 2)
            ax.plot(r_plot * L_BOX, cr_b, 'b' + ls, alpha=0.4)
        ax.fill_between(
            r_plot * L_BOX,
            np.exp(np.log(variance) - sig_lv) * np.exp(-0.5 * (r_plot / np.exp(np.log(scale) - sig_ls)) ** 2),
            np.exp(np.log(variance) + sig_lv) * np.exp(-0.5 * (r_plot / np.exp(np.log(scale) + sig_ls)) ** 2),
            alpha=0.15, color='b', label="1-sigma band")
    ax.set_xlabel("r [Mpc/h]")
    ax.set_ylabel("C(r)")
    ax.set_title("(g) Learned kernel")
    ax.legend()

    # (h) Q4: kernel comparison if available
    ax = axes[2, 1]
    if q4_results is not None:
        for name, res in q4_results.items():
            v = res["variance"]
            s = res["scale_mpc_h"] / L_BOX
            r_p = np.linspace(0, 0.3, 300)
            cr = v * np.exp(-0.5 * (r_p / s) ** 2)
            ax.plot(r_p * L_BOX, cr, lw=2, label=f"{name} {label_a_name}")
        ax.set_xlabel("r [Mpc/h]")
        ax.set_ylabel("C(r)")
        ax.set_title(f"(h) Q4: Kernel by {label_a_name} split")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Q4 not computed", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title("(h) Q4: Kernel comparison")

    # (i) empty/extra
    ax = axes[2, 2]
    ax.axis("off")
    ax.set_title("(i) Reserved")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "gp_density_reconstruction.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(sim_idx=0):
    """Run the full GP density reconstruction pipeline."""

    print("=" * 62)
    print("  GP Density Reconstruction from Quijote Halo Catalogs")
    print("  Using GraphGP (Vecchia approximation)")
    print("=" * 62)

    # ── Step 0: Load data ────────────────────────────────────────
    positions, velocities, masses = load_data(DATA_DIR, sim_idx=sim_idx)
    N = len(positions)

    label_a = np.log10(masses + 1e-10)
    label_b = np.linalg.norm(velocities, axis=1)
    label_a_name = "log10(M)"
    label_b_name = "|v| [km/s]"

    print(f"\n  Label a ({label_a_name}): "
          f"range [{label_a.min():.2f}, {label_a.max():.2f}]")
    print(f"  Label b ({label_b_name}): "
          f"range [{label_b.min():.1f}, {label_b.max():.1f}]")

    # ── Step 1: Build graphs ─────────────────────────────────────
    # Halo-only graph (for Hessian computation later)
    graph_halo, points_norm = build_graph(positions)
    n_bar = float(N)

    # Combined graph with volume-filling points (for corrected likelihood)
    graph_combined, n_halo, n_vol, vol_points = \
        build_combined_graph(points_norm, n_vol=N_VOL_POINTS)

    # ── Step 2: Initial kernel ───────────────────────────────────
    log_var = jnp.log(jnp.array(INIT_VARIANCE, dtype=jnp.float32))
    log_scale = jnp.log(jnp.array(INIT_SCALE / L_BOX, dtype=jnp.float32))

    print(f"\n  Initial kernel: variance = {INIT_VARIANCE:.2f}, "
          f"scale = {INIT_SCALE:.1f} Mpc/h "
          f"({INIT_SCALE/L_BOX:.4f} box units)")

    # ── Step 3: Alternating optimization (combined graph) ────────
    all_losses = []
    xi_current = None

    for rnd in range(N_ALTERNATING_ROUNDS):
        print(f"\n{'*'*60}")
        print(f"  ALTERNATING ROUND {rnd+1}/{N_ALTERNATING_ROUNDS}")
        print(f"{'*'*60}")

        # Field optimization (xi-space) with volume integral
        xi_current, delta_map, losses = optimize_field(
            graph_combined, n_bar, log_var, log_scale,
            n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
            xi_init=xi_current, n_halo=n_halo, n_vol=n_vol)
        all_losses.extend(losses)

        # Kernel optimization on combined graph
        log_var, log_scale = optimize_kernel(
            delta_map, graph_combined, log_var, log_scale,
            n_steps=N_KERNEL_STEPS, lr=1e-3)

    # Final field optimization with learned kernel
    xi_current, delta_map, losses = optimize_field(
        graph_combined, n_bar, log_var, log_scale,
        n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
        xi_init=xi_current, n_halo=n_halo, n_vol=n_vol)
    all_losses.extend(losses)

    # Extract halo delta from the optimization
    delta_halo = np.array(delta_map[:n_halo])
    vol_points_np = np.array(vol_points)

    print(f"\n  Final kernel: variance = {float(jnp.exp(log_var)):.4f}, "
          f"scale = {float(jnp.exp(log_scale)) * L_BOX:.1f} Mpc/h")
    print(f"  Halo delta range: [{delta_halo.min():.3f}, {delta_halo.max():.3f}]")

    # Predict delta at volume points via GP conditional mean from halos.
    # This gives physically smooth interpolation (not optimization artifacts).
    print("  Predicting field at volume points via GP conditional mean...")
    cov_learned = make_kernel(log_var, log_scale)
    delta_vol = predict_at_points(
        delta_halo, points_norm, vol_points, cov_learned, k=K_NEIGHBORS)
    delta_vol = np.array(delta_vol)
    print(f"  Volume delta: mean = {delta_vol.mean():.3f}, "
          f"range = [{delta_vol.min():.3f}, {delta_vol.max():.3f}]")

    # ── Fisher matrix for kernel uncertainties ───────────────────
    fisher, fisher_unc = compute_kernel_fisher(
        delta_map, graph_combined, log_var, log_scale)

    var_val = float(jnp.exp(log_var))
    scale_val = float(jnp.exp(log_scale)) * L_BOX
    sig_lv = float(fisher_unc[0])
    sig_ls = float(fisher_unc[1])
    print(f"  Kernel: variance = {var_val:.4f} +/- {var_val*(np.exp(sig_lv)-1):.4f}")
    print(f"  Kernel: scale = {scale_val:.1f} +/- {scale_val*(np.exp(sig_ls)-1):.1f} Mpc/h")

    # Convert to numpy (halo-only)
    delta_np = delta_halo
    points_np = np.array(points_norm)

    # ── Step 4: Hessian and cosmic web (halo-only graph) ─────────
    gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared = \
        compute_gp_derivatives(graph_halo, make_kernel(log_var, log_scale),
                               jnp.array(delta_halo),
                               log_variance=log_var, log_scale=log_scale)

    # Quadratic-fit Hessian for comparison
    gradient_qf, hessian_qf, eigenvalues_qf, labels_geo_qf, laplacian_qf, s_squared_qf = \
        compute_hessian_quadratic_fit(delta_np, points_np)

    # ── Step 5: Label-environment correlations ───────────────────
    corr_results = environment_label_analysis(
        delta_np, laplacian, s_squared, eigenvalues, labels_geo,
        label_a, label_b,
        label_a_name=label_a_name, label_b_name=label_b_name)

    # ── Step 5b: Q4 label-dependent clustering ───────────────────
    q4_results = label_dependent_clustering(
        positions, label_a, label_a_name, graph_halo)

    # ── Clustering statistics ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Computing clustering statistics...")
    print("=" * 60)

    print("  Two-point correlation function...")
    r_2pt, xi_2pt, xi_2pt_err = compute_two_point_function(
        positions, n_bins=20, r_max=150.0, box_size=L_BOX)

    print("  Counts-in-cells...")
    counts_10, density_mean_10, cic_bins_10, cic_pdf_10, cic_var_10, cic_skew_10, cic_S3_10 = \
        compute_counts_in_cells(positions, delta_halo, n_cells=10, box_size=L_BOX)
    print(f"    n_cells=10: variance={cic_var_10:.2f}, "
          f"skewness={cic_skew_10:.4f}, S3={cic_S3_10:.3f}")

    print("  Three-point function (equilateral)...")
    r_3pt, Q_3pt, Q_3pt_err, zeta_3pt, xi_3pt = compute_three_point_function(
        positions, n_bins=12, r_max=80.0, box_size=L_BOX)

    # ── Plots ────────────────────────────────────────────────────
    make_plots(
        points_np, delta_np, eigenvalues, labels_geo,
        label_a, label_b, s_squared, laplacian, all_losses,
        log_var, log_scale, fisher_unc=fisher_unc,
        q4_results=q4_results,
        label_a_name=label_a_name, label_b_name=label_b_name)

    # ── Save results ─────────────────────────────────────────────
    outfile = os.path.join(OUTPUT_DIR, "gp_reconstruction_results.npz")
    np.savez(
        outfile,
        positions=positions,
        delta=delta_np,
        gradient=gradient,
        hessian=hessian,
        eigenvalues=eigenvalues,
        laplacian=laplacian,
        s_squared=s_squared,
        label_a=label_a,
        label_b=label_b,
        labels_geo=labels_geo,
        # Quadratic-fit Hessian for comparison
        gradient_qf=gradient_qf,
        hessian_qf=hessian_qf,
        eigenvalues_qf=eigenvalues_qf,
        laplacian_qf=laplacian_qf,
        s_squared_qf=s_squared_qf,
        labels_geo_qf=labels_geo_qf,
        # Volume-filling evaluation
        delta_vol=delta_vol,
        vol_points=vol_points_np,
        n_vol_points=n_vol,
        # Kernel info
        kernel_variance=var_val,
        kernel_scale_mpc_h=scale_val,
        fisher_matrix=np.array(fisher),
        fisher_uncertainties=np.array(fisher_unc),
        losses=np.array(all_losses),
        # Clustering statistics
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
    print(f"\n  Results saved to: {outfile}")

    print("\n" + "=" * 62)
    print("  Pipeline complete!")
    print("=" * 62)


# =====================================================================
# LOG-DELTA APPROACH
# =====================================================================
# Instead of reconstructing delta directly (where rho = n_bar*(1+delta)),
# we reconstruct f(x) = log(1 + delta(x)), the log-density contrast.
# This guarantees positive densities (delta = exp(f) - 1 >= -1) and
# is better suited for high-contrast regions of the cosmic web.
#
# Key differences from the linear (delta) approach:
#   - GP models f(x) = log(1 + delta) instead of delta
#   - Poisson log-likelihood: sum_i f_i - n_bar * (1/N_vol) * sum_j exp(f_j)
#   - Density recovered as delta = exp(f) - 1
# =====================================================================

def log_delta_poisson_log_likelihood(f, n_bar, n_halo, n_vol=0):
    """
    Poisson log-likelihood in log-density space.

    f = log(1 + delta), so (1 + delta) = exp(f).

    ln L = sum_{i in halos} [ln(n_bar) + f_i]
         - n_bar * (V / N_vol) * sum_{j in vol} exp(f_j)

    Args:
        f: (N,) log-density field values at all points
        n_bar: mean number density
        n_halo: number of halo points (first n_halo entries)
        n_vol: number of volume-filling points (remaining entries)
    """
    f_halo = f[:n_halo]
    ll = jnp.sum(jnp.log(n_bar) + f_halo)

    if n_vol > 0:
        f_vol = f[n_halo:]
        # Monte Carlo integral of n_bar * exp(f) over unit volume
        integral = n_bar * (1.0 / n_vol) * jnp.sum(jnp.exp(f_vol))
        ll -= integral
    else:
        ll -= n_halo

    return ll


def optimize_field_log_delta(graph, n_bar, log_variance, log_scale,
                              n_steps=200, lr=1e-2, xi_init=None,
                              n_halo=None, n_vol=0):
    """
    MAP field estimation in log-density space.

    f = gp.generate(graph, cov, xi) models log(1+delta).

    Returns:
        xi_map: (N,) MAP white-noise parameters
        f_map: (N,) MAP log-density field
        delta_map: (N,) MAP density contrast (exp(f) - 1)
        losses: list of -log_posterior values
    """
    print("\n" + "=" * 60)
    print("LOG-DELTA: Optimizing log-density field (fixed kernel, xi-space)")
    print("=" * 60)

    N = len(graph.points)
    cov = make_kernel(log_variance, log_scale)

    _n_halo = n_halo if n_halo is not None else N
    _n_vol = n_vol

    if xi_init is None:
        key = jax.random.PRNGKey(42)
        xi = 0.01 * jax.random.normal(key, shape=(N,))
    else:
        xi = xi_init

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(xi)

    @jit
    def loss_fn(xi):
        f = gp.generate(graph, cov, xi)
        ll = log_delta_poisson_log_likelihood(f, n_bar, _n_halo, _n_vol)
        log_prior = -0.5 * jnp.dot(xi, xi)
        return -(ll + log_prior)

    grad_fn = jit(grad(loss_fn))

    losses = []
    for step in range(n_steps):
        loss_val = loss_fn(xi)
        grads = grad_fn(xi)
        updates, opt_state = optimizer.update(grads, opt_state, xi)
        xi = optax.apply_updates(xi, updates)
        losses.append(float(loss_val))

        if step % 50 == 0 or step == n_steps - 1:
            f = gp.generate(graph, cov, xi)
            f_halo = f[:_n_halo]
            delta_halo = jnp.exp(f_halo) - 1.0
            msg = (f"  Step {step:4d}: loss = {loss_val:.2f}, "
                   f"f_halo = [{float(f_halo.min()):.3f}, "
                   f"{float(f_halo.max()):.3f}], "
                   f"delta = [{float(delta_halo.min()):.3f}, "
                   f"{float(delta_halo.max()):.3f}]")
            if _n_vol > 0:
                f_vol = f[_n_halo:]
                msg += f", <f_vol> = {float(f_vol.mean()):.3f}"
            print(msg)

    f_map = gp.generate(graph, cov, xi)
    delta_map = jnp.exp(f_map) - 1.0
    return xi, f_map, delta_map, losses


# =====================================================================
# CLUSTERING STATISTICS
# =====================================================================

def compute_two_point_function(positions, n_bins=25, r_max=150.0,
                                box_size=None):
    """
    Estimate the two-point correlation function xi(r) using the
    Landy-Szalay estimator: xi(r) = (DD - 2DR + RR) / RR.

    Args:
        positions: (N, 3) positions in Mpc/h
        n_bins: number of radial bins
        r_max: maximum separation in Mpc/h
        box_size: periodic box size (if None, no periodic wrapping)

    Returns:
        r_centers: (n_bins,) bin centers
        xi: (n_bins,) two-point correlation function
        xi_err: (n_bins,) Poisson error estimate
    """
    from scipy.spatial import cKDTree

    N = len(positions)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Build tree with periodic boundary conditions
    if box_size is not None:
        tree = cKDTree(positions, boxsize=box_size)
    else:
        tree = cKDTree(positions)

    # DD counts
    DD = np.zeros(n_bins)
    for i in range(N):
        dists = tree.query_ball_point(positions[i], r_max)
        dists_vals = np.array([np.linalg.norm(
            _periodic_diff(positions[j] - positions[i], box_size)
            if box_size else positions[j] - positions[i])
            for j in dists if j != i])
        if len(dists_vals) > 0:
            hist, _ = np.histogram(dists_vals, bins=r_edges)
            DD += hist
    DD /= 2.0  # each pair counted twice

    # RR counts (analytic for uniform random in a periodic box)
    if box_size is not None:
        V = box_size ** 3
    else:
        V = (positions.max(axis=0) - positions.min(axis=0)).prod()

    n_bar_vol = N / V
    shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    RR = 0.5 * N * (N - 1) * shell_volumes / V

    # Landy-Szalay (simplified: DR ~ RR for periodic box)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(RR > 0, DD / RR - 1.0, 0.0)

    # Poisson error: sigma_xi ~ (1 + xi) / sqrt(DD)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi_err = np.where(DD > 0, (1.0 + np.abs(xi)) / np.sqrt(DD), 0.0)

    return r_centers, xi, xi_err


def _periodic_diff(dx, box_size):
    """Apply periodic wrapping to displacement vector."""
    if box_size is None:
        return dx
    return dx - box_size * np.round(dx / box_size)


def compute_counts_in_cells(positions, delta, n_cells=10, box_size=None):
    """
    Counts-in-cells statistics: bin halos into a 3D grid and compute
    the PDF of cell counts, plus moments.

    Args:
        positions: (N, 3) positions in Mpc/h
        delta: (N,) density field values (used for density-weighted counts)
        n_cells: number of cells per dimension
        box_size: box size in Mpc/h

    Returns:
        counts: (n_cells^3,) raw counts per cell
        density_mean: (n_cells^3,) mean density per cell
        count_pdf_bins: bin edges for the count PDF
        count_pdf: count PDF values
        variance: variance of counts
        skewness: skewness of counts
        S3: reduced skewness S_3 = <delta^3> / <delta^2>^2
    """
    if box_size is None:
        box_size = positions.max() - positions.min()

    pos_min = positions.min(axis=0)

    # Assign to cells
    cell_idx = np.floor((positions - pos_min) / box_size * n_cells).astype(int)
    cell_idx = np.clip(cell_idx, 0, n_cells - 1)

    # Count halos per cell
    flat_idx = cell_idx[:, 0] * n_cells ** 2 + cell_idx[:, 1] * n_cells + cell_idx[:, 2]
    n_total_cells = n_cells ** 3
    counts = np.bincount(flat_idx, minlength=n_total_cells).astype(float)

    # Mean density per cell
    density_sum = np.bincount(flat_idx, weights=delta, minlength=n_total_cells)
    cell_counts_nonzero = np.where(counts > 0, counts, 1)
    density_mean = density_sum / cell_counts_nonzero

    # Count PDF
    max_count = int(counts.max()) + 1
    count_pdf_bins = np.arange(-0.5, max_count + 1.5, 1.0)
    count_pdf, _ = np.histogram(counts, bins=count_pdf_bins, density=True)
    count_pdf_centers = 0.5 * (count_pdf_bins[:-1] + count_pdf_bins[1:])

    # Moments
    mean_count = counts.mean()
    variance = counts.var()
    delta_c = (counts - mean_count) / mean_count if mean_count > 0 else counts * 0
    skewness = np.mean(delta_c ** 3)
    var_delta = np.mean(delta_c ** 2)
    S3 = skewness / var_delta ** 2 if var_delta > 0 else 0.0

    return counts, density_mean, count_pdf_centers, count_pdf, variance, skewness, S3


def compute_three_point_function(positions, n_bins=10, r_max=80.0,
                                  box_size=None, n_triplets_max=500000,
                                  seed=42):
    """
    Estimate the reduced three-point function Q(r1, r2, theta) for
    equilateral configurations (r1 = r2 = r).

    Q = zeta(r, r, r) / [xi(r)^2 * 3]

    where zeta is the connected three-point function.

    Uses Monte Carlo sampling of triplets for efficiency.

    Args:
        positions: (N, 3) positions in Mpc/h
        n_bins: number of radial bins
        r_max: maximum triangle side length
        box_size: periodic box size
        n_triplets_max: max number of triplet samples
        seed: random seed

    Returns:
        r_centers: (n_bins,) bin centers
        Q_r: (n_bins,) reduced three-point function
        Q_err: (n_bins,) bootstrap error
        zeta: (n_bins,) connected three-point function
        xi_r: (n_bins,) two-point function at same scales
    """
    from scipy.spatial import cKDTree

    N = len(positions)
    rng = np.random.RandomState(seed)

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    if box_size is not None:
        tree = cKDTree(positions, boxsize=box_size)
    else:
        tree = cKDTree(positions)

    # First compute xi(r) at these scales
    _, xi_r, _ = compute_two_point_function(positions, n_bins=n_bins,
                                             r_max=r_max, box_size=box_size)

    # Count equilateral triplets: DDD(r)
    DDD = np.zeros(n_bins)
    RRR = np.zeros(n_bins)

    if box_size is not None:
        V = box_size ** 3
    else:
        V = (positions.max(axis=0) - positions.min(axis=0)).prod()

    n_bar_vol = N / V

    # Sample anchor points
    n_anchors = min(N, 2000)
    anchor_idx = rng.choice(N, n_anchors, replace=False)

    for a_idx in anchor_idx:
        # Find neighbors within r_max
        nbrs = tree.query_ball_point(positions[a_idx], r_max + dr)
        if len(nbrs) < 2:
            continue

        nbr_pos = positions[nbrs]
        diffs_a = nbr_pos - positions[a_idx]
        if box_size is not None:
            diffs_a = diffs_a - box_size * np.round(diffs_a / box_size)
        dists_a = np.linalg.norm(diffs_a, axis=1)

        for b in range(n_bins):
            r_lo, r_hi = r_edges[b], r_edges[b + 1]
            mask_b = (dists_a >= r_lo) & (dists_a < r_hi)
            b_indices = np.where(mask_b)[0]

            if len(b_indices) < 2:
                continue

            # For each pair in this shell, check if they are also separated by r
            for ii in range(min(len(b_indices), 50)):
                idx_i = b_indices[ii]
                for jj in range(ii + 1, min(len(b_indices), 50)):
                    idx_j = b_indices[jj]
                    diff_ij = nbr_pos[idx_j] - nbr_pos[idx_i]
                    if box_size is not None:
                        diff_ij = diff_ij - box_size * np.round(diff_ij / box_size)
                    dist_ij = np.linalg.norm(diff_ij)
                    if r_lo <= dist_ij < r_hi:
                        DDD[b] += 1

    # Normalize
    scale = N / n_anchors
    DDD *= scale

    # RRR for uniform random (analytic for equilateral in periodic box)
    shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    for b in range(n_bins):
        RRR[b] = (N * (N - 1) * (N - 2) / 6.0) * (shell_vol[b] / V) ** 2

    # Connected three-point function
    with np.errstate(divide='ignore', invalid='ignore'):
        zeta = np.where(RRR > 0, DDD / RRR - 1.0, 0.0)

    # Reduced Q = zeta / (3 * xi^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 3.0 * xi_r ** 2
        Q_r = np.where(denom > 1e-10, zeta / denom, 0.0)

    # Error estimate (Poisson-based)
    with np.errstate(divide='ignore', invalid='ignore'):
        Q_err = np.where(DDD > 0, np.abs(Q_r) / np.sqrt(DDD + 1), 0.0)

    return r_centers, Q_r, Q_err, zeta, xi_r


# =====================================================================
# LOG-DELTA MAIN PIPELINE
# =====================================================================

def main_log_delta(sim_idx=0):
    """Run the full GP log-density reconstruction pipeline."""

    print("=" * 62)
    print("  GP Log-Density Reconstruction from Quijote Halo Catalogs")
    print("  Using GraphGP (Vecchia approximation)")
    print("  Field: f(x) = log(1 + delta(x))")
    print("=" * 62)

    # ── Step 0: Load data ────────────────────────────────────────
    positions, velocities, masses = load_data(DATA_DIR, sim_idx=sim_idx)
    N = len(positions)

    label_a = np.log10(masses + 1e-10)
    label_b = np.linalg.norm(velocities, axis=1)
    label_a_name = "log10(M)"
    label_b_name = "|v| [km/s]"

    # ── Step 1: Build graphs ─────────────────────────────────────
    graph_halo, points_norm = build_graph(positions)
    n_bar = float(N)

    graph_combined, n_halo, n_vol, vol_points = \
        build_combined_graph(points_norm, n_vol=N_VOL_POINTS)

    # ── Step 2: Initial kernel ───────────────────────────────────
    log_var = jnp.log(jnp.array(INIT_VARIANCE, dtype=jnp.float32))
    log_scale = jnp.log(jnp.array(INIT_SCALE / L_BOX, dtype=jnp.float32))

    # ── Step 3: Alternating optimization (log-delta) ─────────────
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

    # Extract halo and volume values
    f_halo = np.array(f_map[:n_halo])
    delta_halo = np.array(delta_map[:n_halo])
    f_vol = np.array(f_map[n_halo:])
    delta_vol = np.array(delta_map[n_halo:])
    vol_points_np = np.array(vol_points)

    print(f"\n  Final kernel: variance = {float(jnp.exp(log_var)):.4f}, "
          f"scale = {float(jnp.exp(log_scale)) * L_BOX:.1f} Mpc/h")
    print(f"  f_halo range: [{f_halo.min():.3f}, {f_halo.max():.3f}]")
    print(f"  delta_halo range: [{delta_halo.min():.3f}, {delta_halo.max():.3f}]")
    print(f"  All densities positive: {np.all(delta_halo > -1)}")

    # ── Fisher matrix ────────────────────────────────────────────
    fisher, fisher_unc = compute_kernel_fisher(
        f_map, graph_combined, log_var, log_scale)

    var_val = float(jnp.exp(log_var))
    scale_val = float(jnp.exp(log_scale)) * L_BOX
    sig_lv = float(fisher_unc[0])
    sig_ls = float(fisher_unc[1])

    # ── Hessian and cosmic web ───────────────────────────────────
    # Compute Hessian of the log-density field f, then convert to
    # Hessian of delta = exp(f) - 1 via chain rule:
    #   d^2(delta)/dx_i dx_j = exp(f) * [d^2f/dx_i dx_j + df/dx_i * df/dx_j]
    gradient_f, hessian_f, eigenvalues_f, labels_geo_f, laplacian_f, s_squared_f = \
        compute_gp_derivatives(graph_halo, make_kernel(log_var, log_scale),
                               jnp.array(f_halo),
                               log_variance=log_var, log_scale=log_scale)

    # Convert to delta-space Hessian
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

    # ── Clustering statistics ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Computing clustering statistics...")
    print("=" * 60)

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

    # ── Label-environment correlations ───────────────────────────
    corr_results = environment_label_analysis(
        delta_halo, laplacian_delta, s_squared_delta, eigenvalues_delta, labels_geo,
        label_a, label_b,
        label_a_name=label_a_name, label_b_name=label_b_name)

    # ── Save results ─────────────────────────────────────────────
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
        gradient_delta=gradient_f * exp_f[:, None],  # chain rule
        hessian_delta=hessian_delta,
        eigenvalues=eigenvalues_delta,
        laplacian=laplacian_delta,
        s_squared=s_squared_delta,
        labels_geo=labels_geo,
        label_a=label_a,
        label_b=label_b,
        # Kernel
        kernel_variance=var_val,
        kernel_scale_mpc_h=scale_val,
        fisher_matrix=np.array(fisher),
        fisher_uncertainties=np.array(fisher_unc),
        losses=np.array(all_losses),
        # Clustering statistics
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

    print("\n" + "=" * 62)
    print("  Log-delta pipeline complete!")
    print("=" * 62)


if __name__ == "__main__":
    import sys
    sim_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "density"
    if mode == "log_delta":
        main_log_delta(sim_idx=sim_idx)
    else:
        main(sim_idx=sim_idx)
