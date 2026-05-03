"""End-to-end real-Quaia survey workflow.

Runs the full survey pipeline against the published
Storey-Fisher+24 G < 20 quasar catalogue (Storey-Fisher et al. 2024,
arXiv:2306.17749). The data file ``quaia_G20.0.fits`` (755,850 sources,
0.084 < z < 4.537) lives at::

    data/quaia/quaia_G20.0.fits

and the random catalogue is generated locally from the published
healpix selection-function map at::

    data/quaia/selection_function_NSIDE64_G20.0.fits

(no need to download the 150 MB ``random_G20.0_10x.fits`` -- the
selection map encodes the same completeness used by the published
random, and ``make_random_from_selection_function`` reproduces the
recipe at the per-pixel level.)

Pipeline steps:

  1. ``load_quaia(catalog_path, selection_path=...)`` -- read FITS,
     synthesise selection-aware random with z drawn from the data n(z),
     project to comoving Mpc/h under the fiducial cosmology.
  2. ``build_state`` on the survey -- DD / DR / RR pair counts via
     the cKDTree machinery in ``differentiable_lisa``.
  3. SF&H basis-projected xi(s) on the survey.
  4. syren-halofit ``b^2 xi_NL(s)`` overlay at the fiducial cosmology.

Three PNGs::

  quaia_real_skymap.png   - Mollweide projection of data + random subsample.
  quaia_real_nz.png       - n(z) for data and selection-aware random.
  quaia_real_xi.png       - SF&H xi(s) on real Quaia + syren-halofit overlay.

Defaults are sized to finish in a few minutes on a single laptop core.
Override via environment variables:

  QUAIA_N_DATA     -- max number of data points to keep (default 80000)
  QUAIA_N_RANDOM   -- random multiplier (default 5x N_data)
  QUAIA_Z_MIN      -- minimum redshift cut (default 0.8)
  QUAIA_Z_MAX      -- maximum redshift cut (default 2.5)
  QUAIA_DATA       -- override path to the quaia FITS
  QUAIA_SELECTION  -- override path to the selection-function FITS
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
from twopt_density.quaia import (
    QuaiaCatalog, load_quaia, load_selection_function,
    make_random_from_selection_function,
)
from twopt_density.spectra import (
    FFTLogP2xi, make_log_k_grid, xi_from_Pk_fftlog,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def panel_skymap(cat, out_path):
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111, projection="mollweide")
    rng = np.random.default_rng(0)
    sub_d = rng.choice(cat.N_data, size=min(cat.N_data, 8000), replace=False)
    sub_r = rng.choice(cat.N_random, size=min(cat.N_random, 8000), replace=False)
    ra_d = np.deg2rad(cat.ra_data[sub_d]) - np.pi
    dec_d = np.deg2rad(cat.dec_data[sub_d])
    ra_r = np.deg2rad(cat.ra_random[sub_r]) - np.pi
    dec_r = np.deg2rad(cat.dec_random[sub_r])
    ax.scatter(ra_r, dec_r, s=0.3, alpha=0.15, color="C7",
               label=f"random (N={cat.N_random:,}, subset)")
    ax.scatter(ra_d, dec_d, s=0.4, alpha=0.4, color="C0",
               label=f"data (N={cat.N_data:,}, subset)")
    ax.set_title("Quaia G<20: Mollweide sky map "
                 "(selection-function-masked random)")
    ax.legend(loc="lower right", fontsize=8, markerscale=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_nz(cat, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    z_min = float(min(cat.z_data.min(), cat.z_random.min()))
    z_max = float(max(cat.z_data.max(), cat.z_random.max()))
    bins = np.linspace(z_min, z_max, 80)
    ax.hist(cat.z_data, bins=bins, density=True, alpha=0.6, color="C0",
            label=f"data (N={cat.N_data:,})")
    ax.hist(cat.z_random, bins=bins, density=True, alpha=0.4, color="C7",
            label=f"random (N={cat.N_random:,}, n(z) drawn from data)")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("normalized density")
    ax.set_title("Quaia G<20: n(z) for data vs selection-aware random")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def panel_xi(s_data, xi_data, sigma_xi, out_path):
    """Measured xi(s) on real Quaia + syren-halofit ``b^2 xi_NL`` overlays."""
    k = make_log_k_grid(1e-4, 1e2, 2048)
    fft = FFTLogP2xi(k, l=0)
    s_fine = jnp.asarray(np.logspace(np.log10(5.0), np.log10(120.0), 100))
    P = cj.run_halofit(k, sigma8=0.8, Om=0.31, Ob=0.049, h=0.68, ns=0.965, a=1.0)
    xi_NL = np.asarray(xi_from_Pk_fftlog(s_fine, fft, P))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(np.asarray(s_data), xi_data, yerr=np.asarray(sigma_xi),
                fmt="ok", markersize=4, capsize=3,
                label=r"$\xi_{data}$ (Quaia G$<$20)")
    for b2, ls in [(3.0, "-"), (6.0, "--"), (9.0, ":")]:
        ax.plot(np.asarray(s_fine), b2 * xi_NL, ls, lw=2,
                label=rf"$b^2 \xi_{{NL}}(s),\ b^2={b2}$")
    ax.set_xscale("log")
    ax.set_xlabel("s [Mpc/h]")
    ax.set_ylabel(r"$\xi(s)$")
    ax.set_title(r"SF&H $\xi(s)$ on real Quaia G<20 + syren-halofit overlays")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _downsample_catalog(cat: QuaiaCatalog, n_data: int, n_random: int,
                        seed: int = 0) -> QuaiaCatalog:
    rng = np.random.default_rng(seed)
    di = rng.choice(cat.N_data, size=min(cat.N_data, n_data), replace=False)
    ri = rng.choice(cat.N_random, size=min(cat.N_random, n_random),
                    replace=False)
    return QuaiaCatalog(
        ra_data=cat.ra_data[di], dec_data=cat.dec_data[di],
        z_data=cat.z_data[di], xyz_data=cat.xyz_data[di],
        ra_random=cat.ra_random[ri], dec_random=cat.dec_random[ri],
        z_random=cat.z_random[ri], xyz_random=cat.xyz_random[ri],
        fid_cosmo=cat.fid_cosmo,
    )


def main():
    fid = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)
    cat_path = os.environ.get("QUAIA_DATA",
                              os.path.join(DATA_DIR, "quaia_G20.0.fits"))
    sel_path = os.environ.get("QUAIA_SELECTION", os.path.join(
        DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    n_data_max = _env_int("QUAIA_N_DATA", 80000)
    n_random_factor_for_load = 5     # generate 5x random against full N_data
    n_random_max_factor = _env_int("QUAIA_N_RANDOM", 5)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)

    print(f"load real Quaia from {cat_path}")
    print(f"  selection map: {sel_path}")
    t0 = time.perf_counter()
    cat = load_quaia(
        catalog_path=cat_path, selection_path=sel_path, fid_cosmo=fid,
        n_random_factor=n_random_factor_for_load, rng_seed=0,
    )
    print(f"  {time.perf_counter()-t0:.1f}s -> N_d={cat.N_data:,}, "
          f"N_r={cat.N_random:,}, z range "
          f"[{cat.z_data.min():.2f}, {cat.z_data.max():.2f}]")

    # redshift cut: focus on the well-populated regime to keep pair counts tight
    print(f"redshift cut {z_min} < z < {z_max}")
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    mr = (cat.z_random >= z_min) & (cat.z_random <= z_max)
    cat = QuaiaCatalog(
        ra_data=cat.ra_data[md], dec_data=cat.dec_data[md],
        z_data=cat.z_data[md], xyz_data=cat.xyz_data[md],
        ra_random=cat.ra_random[mr], dec_random=cat.dec_random[mr],
        z_random=cat.z_random[mr], xyz_random=cat.xyz_random[mr],
        fid_cosmo=cat.fid_cosmo,
    )
    print(f"  -> N_d={cat.N_data:,}, N_r={cat.N_random:,}")

    # downsample for tractable runtime; full catalogue is supported via env vars
    if cat.N_data > n_data_max:
        n_random_target = n_random_max_factor * n_data_max
        print(f"downsample to N_d={n_data_max:,}, N_r={n_random_target:,} "
              "(set QUAIA_N_DATA / QUAIA_N_RANDOM to scale up)")
        cat = _downsample_catalog(cat, n_data=n_data_max,
                                  n_random=n_random_target, seed=1)
        print(f"  -> N_d={cat.N_data:,}, N_r={cat.N_random:,}")

    panel_skymap(cat, os.path.join(FIG_DIR, "quaia_real_skymap.png"))
    print("  wrote quaia_real_skymap.png")
    panel_nz(cat, os.path.join(FIG_DIR, "quaia_real_nz.png"))
    print("  wrote quaia_real_nz.png")

    positions, randoms, box = cat.shift_to_positive()
    print(f"comoving cube side: {float(box):.0f} Mpc/h")
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
             os.path.join(FIG_DIR, "quaia_real_xi.png"))
    print("  wrote quaia_real_xi.png")

    print()
    print("Pipeline ran end-to-end on the real Storey-Fisher+24 Quaia G<20 "
          "catalogue.")
    print(f"FITS rows used: data={cat.N_data:,}, random={cat.N_random:,} "
          f"(selection-aware, generated locally from the NSIDE=64 healpix "
          "completeness map).")


if __name__ == "__main__":
    main()
