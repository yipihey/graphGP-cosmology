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

def poisson_log_likelihood(delta, n_bar):
    """
    Poisson point process log-likelihood.

    ln L = sum_i ln[n_bar * (1 + delta_i)] - N
    """
    density = jnp.clip(1.0 + delta, 1e-10, None)
    ll = jnp.sum(jnp.log(n_bar * density))
    N = len(delta)
    ll -= N
    return ll


def optimize_field(graph, n_bar, log_variance, log_scale,
                   n_steps=200, lr=1e-2, xi_init=None):
    """
    MAP field estimation in xi-space (white-noise parameterization).

    delta = gp.generate(graph, cov, xi)
    Prior: p(xi) = N(0, I), so log-prior = -0.5 * ||xi||^2
    No logdet or generate_inv needed during field optimization.

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
        ll = poisson_log_likelihood(delta, n_bar)
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
            print(f"  Step {step:4d}: loss = {loss_val:.2f}, "
                  f"delta range = [{float(delta.min()):.3f}, "
                  f"{float(delta.max()):.3f}]")

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

def compute_hessian_local_quadratic(delta, points):
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

    # ── Step 1: Build graph ──────────────────────────────────────
    graph, points_norm = build_graph(positions)
    n_bar = float(N)

    # ── Step 2: Initial kernel ───────────────────────────────────
    log_var = jnp.log(jnp.array(INIT_VARIANCE, dtype=jnp.float32))
    log_scale = jnp.log(jnp.array(INIT_SCALE / L_BOX, dtype=jnp.float32))

    print(f"\n  Initial kernel: variance = {INIT_VARIANCE:.2f}, "
          f"scale = {INIT_SCALE:.1f} Mpc/h "
          f"({INIT_SCALE/L_BOX:.4f} box units)")

    # ── Step 3: Alternating optimization ─────────────────────────
    all_losses = []
    xi_current = None

    for rnd in range(N_ALTERNATING_ROUNDS):
        print(f"\n{'*'*60}")
        print(f"  ALTERNATING ROUND {rnd+1}/{N_ALTERNATING_ROUNDS}")
        print(f"{'*'*60}")

        # Field optimization (xi-space)
        xi_current, delta_map, losses = optimize_field(
            graph, n_bar, log_var, log_scale,
            n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
            xi_init=xi_current)
        all_losses.extend(losses)

        # Kernel optimization (delta-space)
        log_var, log_scale = optimize_kernel(
            delta_map, graph, log_var, log_scale,
            n_steps=N_KERNEL_STEPS, lr=1e-3)

    # Final field optimization with learned kernel
    xi_current, delta_map, losses = optimize_field(
        graph, n_bar, log_var, log_scale,
        n_steps=N_OPTIM_STEPS, lr=LEARNING_RATE,
        xi_init=xi_current)
    all_losses.extend(losses)

    print(f"\n  Final kernel: variance = {float(jnp.exp(log_var)):.4f}, "
          f"scale = {float(jnp.exp(log_scale)) * L_BOX:.1f} Mpc/h")
    print(f"  Field range: [{float(delta_map.min()):.3f}, "
          f"{float(delta_map.max()):.3f}]")

    # ── Fisher matrix for kernel uncertainties ───────────────────
    fisher, fisher_unc = compute_kernel_fisher(
        delta_map, graph, log_var, log_scale)

    var_val = float(jnp.exp(log_var))
    scale_val = float(jnp.exp(log_scale)) * L_BOX
    sig_lv = float(fisher_unc[0])
    sig_ls = float(fisher_unc[1])
    print(f"  Kernel: variance = {var_val:.4f} +/- {var_val*(np.exp(sig_lv)-1):.4f}")
    print(f"  Kernel: scale = {scale_val:.1f} +/- {scale_val*(np.exp(sig_ls)-1):.1f} Mpc/h")

    # Convert to numpy
    delta_np = np.array(delta_map)
    points_np = np.array(points_norm)

    # ── Step 4: Hessian and cosmic web ───────────────────────────
    gradient, hessian, eigenvalues, labels_geo, laplacian, s_squared = \
        compute_hessian_local_quadratic(delta_np, points_np)

    # ── Step 5: Label-environment correlations ───────────────────
    corr_results = environment_label_analysis(
        delta_np, laplacian, s_squared, eigenvalues, labels_geo,
        label_a, label_b,
        label_a_name=label_a_name, label_b_name=label_b_name)

    # ── Step 5b: Q4 label-dependent clustering ───────────────────
    q4_results = label_dependent_clustering(
        positions, label_a, label_a_name, graph)

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
        kernel_variance=var_val,
        kernel_scale_mpc_h=scale_val,
        fisher_matrix=np.array(fisher),
        fisher_uncertainties=np.array(fisher_unc),
        losses=np.array(all_losses),
    )
    print(f"\n  Results saved to: {outfile}")

    print("\n" + "=" * 62)
    print("  Pipeline complete!")
    print("=" * 62)


if __name__ == "__main__":
    import sys
    sim_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(sim_idx=sim_idx)
