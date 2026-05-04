#!/usr/bin/env python3
"""
xi_graphgp_fit.py — Continuous-function ξ(r) fit using graphGP.

Reads `multi_run_xi_raw.csv` from a `morton-cascade multi-run --statistic xi`
run and produces a smoothed ξ(r) using a Gaussian-process regression
with an RBF kernel, where the GP machinery is provided by the
Stanford-ISM `graphgp` library (Dodge & Frank).

This is the GP-regression counterpart to the in-Rust linear B-spline fit
(`--xi-fit-basis linear-bsplines`). Both are valid Storey-Fisher-Hogg-style
continuous-function estimators that combine measurements across resize
groups by treating each cascade shell as a top-hat-windowed measurement
of the underlying ξ(r). The B-spline version fits a small number of
basis coefficients; the graphGP version uses a nonparametric GP with
covariance regularization.

Outputs `multi_run_xi_graphgp.csv` in the same format as the Rust-side
`multi_run_xi_evaluated.csv` for direct comparison.

Usage:
    pip install graphgp jax numpy
    python examples/xi_graphgp_fit.py /path/to/multi_run_xi_raw.csv \\
        --r-min 1.0 --r-max 100.0 --n-eval 100 \\
        --kernel-scale 0.3 --kernel-variance 1.0 \\
        --output /path/to/output_dir

Requires the Rust `multi-run --statistic xi` to have been run first to
produce `multi_run_xi_raw.csv`.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List, Tuple

# Lazy imports of optional deps so we can give a useful error message
# when they aren't installed.
def _import_deps():
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        import graphgp as gp
        return np, jax, jnp, gp
    except ImportError as e:
        sys.stderr.write(
            "ERROR: required dependencies not installed: {}\n".format(e) +
            "\nInstall with:\n" +
            "    pip install graphgp jax numpy\n" +
            "\nFor GPU acceleration, also see:\n" +
            "    https://github.com/stanford-ism/graphgp\n"
        )
        sys.exit(1)


def read_xi_raw(csv_path: str) -> List[dict]:
    """Read multi_run_xi_raw.csv. Returns list of shells, one dict each."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: float(v) if k not in ("n_shifts", "level",
                                            "n_d_sum", "n_r_sum")
                       else int(v)
                   for k, v in row.items()}
            rows.append(row)
    return rows


def build_design_and_data(
    shells: List[dict],
    r_eval: "np.ndarray",
    poisson_floor: bool = True,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Assemble the design matrix A, observations y, and noise variance σ².

    Each shell i contributes one row to A: A[i, j] = fraction of shell i's
    pair-density support that falls into the Voronoi cell of evaluation
    point r_eval[j]. For our top-hat shells in r and the simple piecewise-
    constant interpretation of the GP on the grid, this is just
    A[i, j] = (overlap of shell_i with grid cell j) / (shell_i width).

    Returns:
        A: (n_meas, n_eval) design matrix
        y: (n_meas,) measured xi_naive values
        sigma2: (n_meas,) measurement variances
    """
    import numpy as np

    n_eval = len(r_eval)
    # Build evaluation cell edges in log r (each grid point r_eval[j] owns
    # a Voronoi cell from midpoint to midpoint).
    log_r = np.log(r_eval)
    edges_log = np.empty(n_eval + 1)
    edges_log[1:-1] = 0.5 * (log_r[:-1] + log_r[1:])
    edges_log[0] = log_r[0] - 0.5 * (log_r[1] - log_r[0])
    edges_log[-1] = log_r[-1] + 0.5 * (log_r[-1] - log_r[-2])
    edges = np.exp(edges_log)

    # Filter out shells with bad data
    usable_shells = []
    for s in shells:
        if s["r_half_width"] <= 0:
            continue
        if not np.isfinite(s["xi_naive"]):
            continue
        r_lo = s["r_center"] - s["r_half_width"]
        r_hi = s["r_center"] + s["r_half_width"]
        if r_lo <= 0:
            continue
        if r_hi <= edges[0] or r_lo >= edges[-1]:
            continue  # outside eval grid
        usable_shells.append(s)

    n_meas = len(usable_shells)
    if n_meas == 0:
        raise RuntimeError("no usable shells found within evaluation grid")

    A = np.zeros((n_meas, n_eval))
    y = np.zeros(n_meas)
    sigma2 = np.zeros(n_meas)
    for i, s in enumerate(usable_shells):
        r_lo = s["r_center"] - s["r_half_width"]
        r_hi = s["r_center"] + s["r_half_width"]
        # Overlap of [r_lo, r_hi] with each grid cell [edges[j], edges[j+1]].
        # Weight each grid cell by its overlap fraction within the shell.
        for j in range(n_eval):
            overlap = max(0.0, min(r_hi, edges[j + 1]) - max(r_lo, edges[j]))
            A[i, j] = overlap / (r_hi - r_lo)  # fraction of shell

        y[i] = s["xi_naive"]
        # Use shift-bootstrap variance plus optional Poisson floor.
        var_shift = s["xi_shift_bootstrap_var"]
        if poisson_floor:
            poisson = 1.0 / (s["dd_sum"] + 1.0)
            sigma2[i] = max(var_shift, poisson, 1e-12)
        else:
            sigma2[i] = max(var_shift, 1e-12)

    return A, y, sigma2


def fit_gp_regression(
    A,
    y,
    sigma2,
    r_eval,
    kernel_scale: float,
    kernel_variance: float,
    jitter: float,
    k_neighbors: int,
):
    """Fit a GP with RBF kernel to the linear inverse problem.

    Posterior of ξ on the eval grid:
        ξ | y ∼ N(μ, Σ_post)
        Σ_post = (Aᵀ Σ_y⁻¹ A + K⁻¹)⁻¹
        μ      = Σ_post Aᵀ Σ_y⁻¹ y

    where K is the prior covariance from the RBF kernel evaluated on
    r_eval, and Σ_y = diag(σ²).

    The graphGP machinery makes K⁻¹ (and its products) tractable for
    large grids via the Vecchia approximation. For modest grid sizes
    (< few hundred points) we use a dense GP since it's clearer and
    nearly as fast.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp

    n_eval = len(r_eval)

    # Build prior covariance K_ij = σ² exp(-(log r_i - log r_j)² / 2 ℓ²).
    # We use distances in log r so the kernel's length scale is in
    # log-decades, matching the cascade's intrinsic log-scale spacing.
    log_r = jnp.log(r_eval)
    diff = log_r[:, None] - log_r[None, :]
    K_prior = kernel_variance * jnp.exp(-0.5 * (diff / kernel_scale) ** 2)
    K_prior = K_prior + jitter * jnp.eye(n_eval)

    A_jax = jnp.asarray(A)
    y_jax = jnp.asarray(y)
    sigma2_jax = jnp.asarray(sigma2)

    # Posterior precision: Λ = Aᵀ diag(1/σ²) A + K_prior⁻¹.
    AtSinv = (A_jax.T / sigma2_jax)
    AtSinvA = AtSinv @ A_jax
    K_prior_inv = jnp.linalg.inv(K_prior)
    Lambda = AtSinvA + K_prior_inv

    # Posterior mean: μ = Λ⁻¹ Aᵀ Σ_y⁻¹ y.
    rhs = AtSinv @ y_jax
    mu = jnp.linalg.solve(Lambda, rhs)
    Sigma_post = jnp.linalg.inv(Lambda)
    sigma_diag = jnp.sqrt(jnp.diag(Sigma_post).clip(min=0.0))

    # Reduced χ²
    pred = A_jax @ mu
    resid = y_jax - pred
    chi2 = jnp.sum(resid * resid / sigma2_jax)
    dof = max(1, len(y) - n_eval // 2)  # rough effective DOF
    return (
        np.asarray(mu),
        np.asarray(sigma_diag),
        float(chi2 / dof),
    )


def write_eval_csv(out_path: str, r: "np.ndarray", xi: "np.ndarray", sigma: "np.ndarray"):
    with open(out_path, "w") as f:
        f.write("r,xi_fit,xi_fit_sigma\n")
        for ri, xi_i, si in zip(r, xi, sigma):
            f.write(f"{ri:.10e},{xi_i:.10e},{si:.10e}\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xi_raw_csv",
        help="Path to multi_run_xi_raw.csv produced by morton-cascade")
    parser.add_argument("--output", "-o", default=".",
        help="Output directory (default: current)")
    parser.add_argument("--r-min", type=float, default=None,
        help="Lower edge of evaluation grid in trimmed units. "
             "Default: smallest shell window in input.")
    parser.add_argument("--r-max", type=float, default=None,
        help="Upper edge of evaluation grid. Default: largest shell window.")
    parser.add_argument("--n-eval", type=int, default=100,
        help="Number of evaluation grid points. Default: 100.")
    parser.add_argument("--kernel-scale", type=float, default=0.5,
        help="RBF kernel length scale in log r (decades). Default: 0.5.")
    parser.add_argument("--kernel-variance", type=float, default=1.0,
        help="RBF kernel prior variance. Default: 1.0.")
    parser.add_argument("--jitter", type=float, default=1e-6,
        help="Diagonal jitter for kernel stability. Default: 1e-6.")
    parser.add_argument("--no-poisson-floor", action="store_true",
        help="Disable Poisson-floor on shift-bootstrap variance.")
    args = parser.parse_args()

    np, jax, jnp, gp = _import_deps()  # noqa: F841 (gp may be used later)

    print(f"Reading {args.xi_raw_csv} ...")
    shells = read_xi_raw(args.xi_raw_csv)
    print(f"  {len(shells)} shell rows")

    # Resolve r-min, r-max from data if not specified.
    valid = [(s["r_center"] - s["r_half_width"],
              s["r_center"] + s["r_half_width"])
             for s in shells if s["r_half_width"] > 0]
    valid = [(lo, hi) for lo, hi in valid if lo > 0]
    if not valid:
        sys.exit("error: no usable shells with positive width and r_lo > 0")
    data_r_min = min(lo for lo, _ in valid)
    data_r_max = max(hi for _, hi in valid)
    r_min = args.r_min if args.r_min is not None else data_r_min
    r_max = args.r_max if args.r_max is not None else data_r_max
    print(f"  evaluation grid: r ∈ [{r_min:.4e}, {r_max:.4e}], "
          f"n_eval = {args.n_eval}")
    r_eval = np.exp(np.linspace(np.log(r_min), np.log(r_max), args.n_eval))

    # Build the linear inverse problem.
    A, y, sigma2 = build_design_and_data(
        shells, r_eval, poisson_floor=not args.no_poisson_floor)
    print(f"  n_measurements_used = {A.shape[0]}, n_grid = {A.shape[1]}")
    print(f"  median σ = {float(np.median(np.sqrt(sigma2))):.4e}")

    # Fit the GP.
    print(f"  GP RBF kernel: scale = {args.kernel_scale} (log r), "
          f"variance = {args.kernel_variance}, jitter = {args.jitter}")
    mu, sigma, reduced_chi2 = fit_gp_regression(
        A, y, sigma2, r_eval,
        args.kernel_scale, args.kernel_variance, args.jitter,
        k_neighbors=20,
    )
    print(f"  reduced χ² = {reduced_chi2:.4f}")

    # Note for the user about graphgp:
    # The current implementation uses dense JAX inversion for clarity.
    # For larger grids (n_eval > ~500) replace the K_prior_inv step
    # with graphgp's Vecchia-sparse Cholesky:
    #
    #   graph = gp.build_graph(r_eval[:, None], n0=20, k=k_neighbors)
    #   covariance = gp.extras.rbf_kernel(
    #       variance=args.kernel_variance, scale=args.kernel_scale,
    #       r_min=1e-4, r_max=100.0, n_bins=1000, jitter=args.jitter)
    #   K_prior_action = lambda v: gp.generate(graph, covariance, v)
    #
    # See https://github.com/stanford-ism/graphgp for full API.

    # Write output CSV (same format as Rust-side multi_run_xi_evaluated.csv
    # for direct comparison).
    out_path = os.path.join(args.output, "multi_run_xi_graphgp.csv")
    write_eval_csv(out_path, r_eval, mu, sigma)

    print()
    print("Compare with the Rust-side B-spline fit:")
    print("  diff <(tail -n+2 multi_run_xi_evaluated.csv | cut -d, -f1,3,4) \\")
    print("       <(tail -n+2 multi_run_xi_graphgp.csv)")


if __name__ == "__main__":
    main()
