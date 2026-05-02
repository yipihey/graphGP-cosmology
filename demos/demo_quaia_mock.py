"""End-to-end Quaia-mock survey workflow.

Runs the full survey pipeline against a Quaia-shape synthetic
catalog (see ``twopt_density.quaia.make_mock_quaia``):

  1. Galactic-plane-masked all-sky (RA, Dec, z) data + matched random.
  2. JAX (RA, Dec, z) -> comoving Mpc/h under fiducial cosmology.
  3. SF&H basis-projected xi(s) on the survey.
  4. syren-halofit ``b^2 xi_NL(s)`` overlay at fiducial cosmology.

The mock uses a simple "uniform + light Gaussian-blob" model for the
data clustering -- it has the right xi amplitude (a few, in the quasar
ballpark) but not the LambdaCDM shape, so a full cosmology fit on this
mock would be biased. ``demos/demo_joint_fit.py`` already demonstrates
the MAP-recovery machinery on a clean cosmology mock; what this demo
adds is the survey-shape end-to-end: angular masking, n(z), randoms,
RA/Dec/z -> comoving, SF&H xi(s).

When the real ``quaia_G20.5.fits`` and ``quaia_G20.5_random.fits`` are
on hand, swap ``make_mock_quaia(...)`` for ``load_quaia(catalog_path,
randoms_path)`` -- everything else runs unchanged, and the cosmology
fit becomes meaningful.

Three PNGs::

  quaia_skymap.png   - Mollweide projection of data + random subsample.
  quaia_nz.png       - n(z) histograms for data and random.
  quaia_xi.png       - SF&H xi(s) on the mock + syren-halofit overlay.
"""

from __future__ import annotations

import os
import time

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density import cosmology as cj
from twopt_density.basis_xi import JAXBasis, xi_LS_basis_AP
from twopt_density.differentiable_lisa import build_state
from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import make_mock_quaia
from twopt_density.spectra import (
    FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog,
)


FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def panel_skymap(cat, out_path):
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111, projection="mollweide")
    ra_d = np.deg2rad(cat.ra_data) - np.pi
    dec_d = np.deg2rad(cat.dec_data)
    rng = np.random.default_rng(0)
    sub = rng.choice(cat.N_random, size=min(cat.N_random, 6000), replace=False)
    ra_r = np.deg2rad(cat.ra_random[sub]) - np.pi
    dec_r = np.deg2rad(cat.dec_random[sub])
    ax.scatter(ra_r, dec_r, s=0.3, alpha=0.15, color="C7", label="random (subset)")
    ax.scatter(ra_d, dec_d, s=0.4, alpha=0.4, color="C0",
               label=f"data (N={cat.N_data})")
    ax.set_title("Quaia-shape mock: Mollweide sky map "
                 "(Galactic-plane masked)")
    ax.legend(loc="lower right", fontsize=8, markerscale=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_nz(cat, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.5, 4.5, 60)
    ax.hist(cat.z_data, bins=bins, density=True, alpha=0.6, color="C0",
            label="data")
    ax.hist(cat.z_random, bins=bins, density=True, alpha=0.4, color="C7",
            label="random")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("normalized density")
    ax.set_title("Quaia-shape n(z): bimodal target")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_xi(s_data, xi_data, sigma_xi, out_path):
    """Measured xi(s) + syren-halofit ``b^2 * xi_NL`` overlays at three b^2."""
    k = make_log_k_grid(1e-4, 1e2, 2048)
    fft = FFTLogP2xi(k, l=0)
    s_fine = jnp.asarray(np.logspace(np.log10(5.0), np.log10(100.0), 100))
    P = cj.run_halofit(k, sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)
    xi_NL = np.asarray(xi_from_Pk_fftlog(s_fine, fft, P))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(np.asarray(s_data), xi_data, yerr=np.asarray(sigma_xi),
                fmt="ok", markersize=4, capsize=3,
                label="$\\xi_{data}$ (Quaia mock)")
    for b2, ls in [(3.0, "-"), (9.0, "--"), (12.0, ":")]:
        ax.plot(np.asarray(s_fine), b2 * xi_NL, ls, lw=2,
                label=rf"$b^2 \xi_{{NL}}(s),\ b^2={b2}$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title("SF&H $\\xi(s)$ on Quaia-shape mock + syren-halofit overlays")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)
    print("synthesise Quaia-shape mock ...")
    t0 = time.perf_counter()
    cat = make_mock_quaia(
        n_data=15000, n_random=60000, fid_cosmo=fid, seed=11,
    )
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"N_d={cat.N_data}, N_r={cat.N_random}, "
          f"z_med={np.median(cat.z_data):.3f}")

    panel_skymap(cat, os.path.join(FIG_DIR, "quaia_skymap.png"))
    print("  wrote quaia_skymap.png")
    panel_nz(cat, os.path.join(FIG_DIR, "quaia_nz.png"))
    print("  wrote quaia_nz.png")

    positions, randoms, box = cat.shift_to_positive()
    r_edges = np.logspace(np.log10(5.0), np.log10(100.0), 14)
    print("build_state on the survey ...")
    t0 = time.perf_counter()
    state = build_state(
        positions, r_edges, box, randoms=randoms,
        los=np.array([0.0, 0.0, 1.0]), cache_rr=True,
    )
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"DD={state.DD_pi.size}, DR={state.DR_pi.size}, "
          f"RR={state.RR_d.size}")

    jb = JAXBasis.from_cubic_spline(n_basis=14, r_min=5.0, r_max=100.0,
                                    n_grid=2000)
    w_d = jnp.ones(state.N_D)
    w_r = jnp.ones(state.N_R)

    s_data = jnp.asarray(np.logspace(np.log10(6.0), np.log10(80.0), 18))
    print("basis-projected xi_data ...")
    t0 = time.perf_counter()
    xi_data = np.asarray(xi_LS_basis_AP(state, jb, w_d, w_r, 1.0, 1.0, s_data))
    print(f"  {time.perf_counter()-t0:.1f}s, "
          f"xi range: [{xi_data.min():.3f}, {xi_data.max():.3f}]")

    sigma_xi = 0.05 * np.maximum(np.abs(xi_data), 0.005)
    panel_xi(s_data, xi_data, sigma_xi,
             os.path.join(FIG_DIR, "quaia_xi.png"))
    print("  wrote quaia_xi.png")

    print()
    print("Pipeline ran end-to-end on a Quaia-shape mock.")
    print("Swap make_mock_quaia(...) for load_quaia(path, path) to run")
    print("on the real Storey-Fisher+24 catalogue once you have the FITS")
    print("files on a machine that can reach Zenodo.")


if __name__ == "__main__":
    main()
