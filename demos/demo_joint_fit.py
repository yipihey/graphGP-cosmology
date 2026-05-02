"""Joint MAP fit of (Omega_m, sigma_8, b^2) from a noisy xi_data via
scipy.minimize + jax.grad.

Closes the cosmology forward-model loop end-to-end:

  1. Generate xi_data at a known (Omega_m_true, sigma_8_true, b^2_true)
     using syren-halofit + FFTLog. Add diagonal Gaussian noise.
  2. Define a chi^2 over a fixed log-spaced s-grid:
        L(theta) = (1/2) sum_s ((xi_data(s) - xi_pred(s, theta))/sigma)^2
  3. Optimise via scipy L-BFGS-B; gradient is jax-jitted on theta.
  4. Compare best-fit to truth + plot the recovered xi.

Two PNGs::

  joint_fit_xi.png    - xi_data + best-fit xi_pred + truth, residuals.
  joint_fit_pdf.png   - 1D marginal slices of the loss around the best
                        fit for Om, sigma_8, b^2.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density import cosmology as cj
from twopt_density.fit import map_fit
from twopt_density.spectra import (
    FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog,
)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # --- truth --------------------------------------------------------
    Om_true, sigma8_true, b2_true = 0.31, 0.80, 2.50
    Ob_fixed, h_fixed, ns_fixed = 0.049, 0.68, 0.965

    k = make_log_k_grid(1e-4, 1e2, 2048)
    fft = FFTLogP2xi(k, l=0)
    s = jnp.asarray(np.logspace(np.log10(3.0), np.log10(40.0), 18))

    def xi_pred(theta):
        Om, sigma8, b2 = theta
        P = cj.run_halofit(k, sigma8=sigma8, Om=Om, Ob=Ob_fixed,
                           h=h_fixed, ns=ns_fixed, a=1.0)
        return b2 * xi_from_Pk_fftlog(s, fft, P)

    print("generating mock xi_data at truth ...")
    xi_truth = np.asarray(xi_pred(jnp.array([Om_true, sigma8_true, b2_true])))
    rng = np.random.default_rng(42)
    sigma = 0.05 * np.maximum(np.abs(xi_truth), 0.1)  # 5% relative noise
    xi_data = xi_truth + rng.normal(scale=sigma)
    sigma_j = jnp.asarray(sigma)
    xi_data_j = jnp.asarray(xi_data)

    # --- joint fit ----------------------------------------------------
    def loss(theta):
        return 0.5 * jnp.sum(((xi_data_j - xi_pred(theta)) / sigma_j) ** 2)

    theta0 = jnp.array([0.27, 0.75, 1.5])  # away from truth
    bounds = [(0.20, 0.45), (0.6, 1.0), (0.1, 10.0)]

    print("MAP fit (L-BFGS-B with jax-jitted gradient) ...")
    t0 = time.perf_counter()
    res = map_fit(loss, theta0, bounds=bounds)
    dt = time.perf_counter() - t0
    Om_fit, sigma8_fit, b2_fit = res.theta
    print(f"  ({dt:.2f} s, {res.nfev} forward evals, success={res.success})")
    print(f"  fit  : Om={Om_fit:.4f}  sigma8={sigma8_fit:.4f}  b2={b2_fit:.4f}")
    print(f"  truth: Om={Om_true:.4f}  sigma8={sigma8_true:.4f}  b2={b2_true:.4f}")

    # --- panel 1: xi_data, fit, truth ---------------------------------
    s_fine = jnp.asarray(np.logspace(np.log10(2.5), np.log10(50.0), 80))

    def xi_at(theta, s_eval):
        Om, sigma8, b2 = theta
        P = cj.run_halofit(k, sigma8=sigma8, Om=Om, Ob=Ob_fixed,
                           h=h_fixed, ns=ns_fixed, a=1.0)
        return b2 * xi_from_Pk_fftlog(s_eval, fft, P)

    xi_truth_fine = np.asarray(xi_at(
        jnp.array([Om_true, sigma8_true, b2_true]), s_fine,
    ))
    xi_fit_fine = np.asarray(xi_at(jnp.asarray(res.theta), s_fine))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1]))
    ax_xi, ax_res = axes
    ax_xi.errorbar(np.asarray(s), xi_data, yerr=sigma, fmt="ok",
                   markersize=4, capsize=3, label="mock $\\xi_{data}$ (5% noise)")
    ax_xi.plot(np.asarray(s_fine), xi_truth_fine, "-", color="C2", lw=1.5,
               alpha=0.7, label="truth")
    ax_xi.plot(np.asarray(s_fine), xi_fit_fine, "-", color="C1", lw=2,
               label="MAP fit")
    ax_xi.set_xscale("log")
    ax_xi.set_ylabel(r"$\xi(s)$")
    ax_xi.set_title(
        rf"Joint MAP fit (jax.grad + scipy LBFGS-B): "
        rf"$\Omega_m={Om_fit:.3f}\ ({Om_true})$, "
        rf"$\sigma_8={sigma8_fit:.3f}\ ({sigma8_true})$, "
        rf"$b^2={b2_fit:.2f}\ ({b2_true})$"
    )
    ax_xi.legend()
    ax_xi.axhline(0, color="k", lw=0.5, alpha=0.3)
    # residuals
    xi_fit_at_data = np.asarray(xi_at(jnp.asarray(res.theta), s))
    chi = (xi_data - xi_fit_at_data) / sigma
    ax_res.errorbar(np.asarray(s), chi, yerr=1.0, fmt="ok",
                    markersize=4, capsize=3)
    ax_res.axhline(0, color="C1", lw=1)
    ax_res.set_xlabel("s [Mpc/h]")
    ax_res.set_ylabel(r"$(\xi_{data}-\xi_{fit})/\sigma$")
    ax_res.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "joint_fit_xi.png"), dpi=140)
    plt.close(fig)
    print("  wrote joint_fit_xi.png")

    # --- panel 2: 1D loss slices around best fit ----------------------
    def loss_slice(idx, vals):
        out = np.empty_like(vals)
        for i, v in enumerate(vals):
            theta = res.theta.copy()
            theta[idx] = v
            out[i] = float(loss(jnp.asarray(theta)))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    grid = [
        ("$\\Omega_m$", 0, np.linspace(0.25, 0.37, 60), Om_true),
        ("$\\sigma_8$", 1, np.linspace(0.7, 0.9, 60), sigma8_true),
        ("$b^2$",       2, np.linspace(1.5, 3.5, 60), b2_true),
    ]
    for ax, (label, idx, vals, truth) in zip(axes, grid):
        L = loss_slice(idx, vals)
        ax.plot(vals, L - res.loss, "-", lw=2)
        ax.axvline(res.theta[idx], color="C1", ls="--", lw=1, label="MAP")
        ax.axvline(truth, color="C2", ls=":", lw=2, label="truth")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta(-\log L)$")
        ax.set_yscale("symlog", linthresh=1)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Loss profile around MAP best fit (1D slices)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "joint_fit_pdf.png"), dpi=140)
    plt.close(fig)
    print("  wrote joint_fit_pdf.png")


if __name__ == "__main__":
    main()
