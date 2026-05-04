"""Quantify and correct the wide-z bias in Quaia clustering analysis.

The standard "fit a single bias at z_eff" estimator on a wide-redshift
sample like Quaia (z = 0.8 - 2.5) recovers the geometric mean

    b_recovered = sqrt(<b^2(z) D^2(z)>_pair) / D(z_eff)

over the pair-z distribution -- not b at z_eff. For an evolving b(z)
this is a few-percent bias that biases all downstream cosmology.

This demo builds the evolution-aware forward model (twopt_density.
evolution.wp_pair_evolved) and the Modi & White-style optimal weights
(optimal_clustering_weights) on Quaia G < 20:

  1. Compute the empirical pair-z PDF for pairs surviving pi_max.
  2. For several b(z) shapes b0 (1+z)^alpha, compute A_eff = the
     fractional bias of the single-z approximation.
  3. Compare wp(rp) predictions: single-z vs pair-evolved.
  4. Per-galaxy optimal weights w_i ~ b(z_i) D(z_i); show how they
     re-weight the sample toward epochs with strongest signal.

Output: ``quaia_evolution_diagnostics.png`` -- 4-panel plot showing
the pair-z PDF, A_eff vs b(z) shape, single-vs-evolved wp predictions,
and the optimal-weight distribution.
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

from twopt_density.distance import DistanceCosmo
from twopt_density.evolution import (
    effective_amplitude_under_evolution, optimal_clustering_weights,
    pair_z_distribution, wp_pair_evolved,
)
from twopt_density.limber import (
    linear_growth, sigma_chi_from_sigma_z, wp_observed,
)
from twopt_density.quaia import load_quaia


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_float(n, d):
    v = os.environ.get(n)
    return float(v) if v else d


def _b_of_z(z, b0, alpha, z_pivot=1.5):
    return b0 * ((1.0 + z) / (1.0 + z_pivot)) ** alpha


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68)
    pi_max = _env_float("QUAIA_PI_MAX", 200.0)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)

    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    z_d = cat.z_data[md]
    sig_z = cat.z_data_err[md]
    print(f"  N_data after z-cut: {len(z_d):,}, "
          f"z range [{z_d.min():.2f}, {z_d.max():.2f}], "
          f"median = {np.median(z_d):.3f}")

    # 1) empirical pair-z distribution
    print(f"\n1) pair-z distribution (pi_max = {pi_max:.0f} Mpc/h) ...")
    z_g, pdf_pair, n_kept = pair_z_distribution(
        z_d, fid, pi_max=pi_max, n_pairs=100_000, n_bins=80,
    )
    n_z_hist, _ = np.histogram(z_d, bins=z_g.size, density=True)
    z_centres_data = np.linspace(z_g.min(), z_g.max(), n_z_hist.size)
    print(f"  pairs surviving pi_max cut: {n_kept:,}/100,000 "
          f"({n_kept / 1000:.1f}%)")
    print(f"  pair-z mean: {np.trapezoid(pdf_pair * z_g, z_g):.3f} "
          f"(data n(z) median: {np.median(z_d):.3f})")

    # 2) A_eff for several b(z) shapes
    print("\n2) effective bias amplification for various b(z) shapes")
    print(f"  (b(z) = b0 ((1+z)/(1+1.5))^alpha; A_eff = single-z bias / b(z_eff))")
    z_eff = float(np.median(z_d))
    alphas = np.linspace(0.0, 2.0, 5)
    b0_test = 2.5
    A_effs = []
    for alpha in alphas:
        b_z = _b_of_z(z_g, b0_test, alpha)
        A = effective_amplitude_under_evolution(z_g, pdf_pair, b_z, fid,
                                                  z_eff=z_eff)
        A_effs.append(A)
        print(f"    alpha = {alpha:.2f}: A_eff = {A:.4f} "
              f"(single-z bias is {(A - 1) * 100:+.1f}% from b(z_eff))")
    A_effs = np.array(A_effs)

    # 3) compare single-z vs pair-evolved wp predictions
    print("\n3) wp(rp) predictions: single-z vs pair-evolved")
    sigma_chi_per_obj = np.asarray(sigma_chi_from_sigma_z(z_d, sig_z, fid))
    sigma_chi_eff_single = float(np.sqrt(2.0) * np.median(sigma_chi_per_obj))
    # per-z sigma_chi: medianed over galaxies near each z_g
    sigma_chi_z = np.array([
        np.sqrt(2.0) * np.median(sigma_chi_per_obj[
            np.abs(z_d - zg) < 0.05
        ]) if (np.abs(z_d - zg) < 0.05).sum() > 50 else sigma_chi_eff_single
        for zg in z_g
    ])
    rp_grid = np.logspace(np.log10(8.0), np.log10(80.0), 12)

    alpha_use = 1.5    # representative
    b_z_use = _b_of_z(z_g, b0_test, alpha_use)
    wp_single = np.asarray(wp_observed(
        jnp.asarray(rp_grid), z_eff=z_eff,
        sigma_chi_eff=sigma_chi_eff_single, cosmo=fid, bias=b0_test,
        pi_max=pi_max,
    ))
    wp_evolved = np.asarray(wp_pair_evolved(
        jnp.asarray(rp_grid), z_g, pdf_pair, b_z_use, sigma_chi_z, fid,
        pi_max=pi_max,
    ))
    print(f"  rp_c    wp_single    wp_evolved   ratio")
    for r, ws, we in zip(rp_grid, wp_single, wp_evolved):
        print(f"  {r:6.1f}  {ws:8.3f}     {we:8.3f}    {we / ws:.3f}")

    # 4) optimal weights
    print("\n4) optimal clustering weights w_i ~ b(z_i) D(z_i)")
    w_opt = optimal_clustering_weights(z_d, b_z_use, z_g, fid,
                                         P_target_mpc3h3=0.0)
    print(f"  mean = {w_opt.mean():.4f}, std = {w_opt.std():.4f}, "
          f"range [{w_opt.min():.3f}, {w_opt.max():.3f}]")
    # SNR (variance of weight distribution -> theoretical FoM gain)
    # FoM gain ~ <w>^2 / <w^2> for a uniform-noise estimator; for
    # b(z) = const this is 1; for w(z)=b(z)D(z) it improves the
    # variance of the bias-amplitude measurement.
    snr_gain = w_opt.mean() ** 2 / (w_opt ** 2).mean()
    print(f"  variance ratio (1 = no gain, < 1 = optimal weighting "
          f"reduces noise): {snr_gain:.4f}")

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    ax_pdf, ax_aeff, ax_wp, ax_w = axs.flatten()

    n_z_hist_full, edges_d = np.histogram(z_d, bins=z_g.size, density=True)
    z_centres_data = 0.5 * (edges_d[:-1] + edges_d[1:])
    ax_pdf.plot(z_centres_data, n_z_hist_full, "C0-", lw=2, label="data n(z)")
    ax_pdf.plot(z_g, pdf_pair, "C1-", lw=2, label=fr"pair-z PDF ($\pi_{{\max}}={pi_max:.0f}$ Mpc/h)")
    ax_pdf.set_xlabel("z"); ax_pdf.set_ylabel("PDF")
    ax_pdf.set_title("Quaia G$<$20: data n(z) vs pair-z distribution")
    ax_pdf.legend(fontsize=9); ax_pdf.grid(alpha=0.3)

    ax_aeff.plot(alphas, A_effs, "C0-o", ms=6,
                  label=fr"$b(z)=b_0((1+z)/2.5)^\alpha$, $z_{{\rm eff}}={z_eff:.2f}$")
    ax_aeff.axhline(1.0, color="k", lw=0.5, ls=":")
    ax_aeff.set_xlabel(r"bias evolution slope $\alpha$")
    ax_aeff.set_ylabel(r"$A_{\rm eff}$")
    ax_aeff.set_title(r"single-z fit bias / true $b(z_{\rm eff})$"
                       r" vs $b(z)$ shape")
    ax_aeff.legend(fontsize=9); ax_aeff.grid(alpha=0.3)

    ax_wp.plot(rp_grid, wp_single, "C0-o", ms=4,
                label=fr"single-z $w_p$ at $b={b0_test}$, $z_{{\rm eff}}={z_eff:.2f}$")
    ax_wp.plot(rp_grid, wp_evolved, "C3-s", ms=4,
                label=fr"pair-evolved $w_p$ ($b_0={b0_test}$, $\alpha={alpha_use}$)")
    ax_wp.set_xscale("log"); ax_wp.set_yscale("symlog", linthresh=0.5)
    ax_wp.set_xlabel(r"$r_p$ [Mpc/h]")
    ax_wp.set_ylabel(r"$w_p(r_p)$ [Mpc/h]")
    ax_wp.set_title(r"forward-model $w_p(r_p)$: single-z vs pair-evolved")
    ax_wp.legend(fontsize=9); ax_wp.grid(alpha=0.3, which="both")

    ax_w.hist(w_opt, bins=60, color="C2", alpha=0.7,
                label=fr"$w_i \propto b(z_i) D(z_i)$, $\alpha={alpha_use}$")
    ax_w.axvline(1.0, color="k", lw=0.5, ls=":")
    ax_w.set_xlabel("per-galaxy weight"); ax_w.set_ylabel("# galaxies")
    ax_w.set_title(r"optimal clustering weights -- variance ratio "
                    rf"$\langle w \rangle^2 / \langle w^2 \rangle = {snr_gain:.3f}$")
    ax_w.legend(fontsize=9); ax_w.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_evolution_diagnostics.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
