"""Cross-correlation pipeline: DESI Y1 QSO x Quaia G<20 via joint kNN-CDF.

Companion to ``demos/quaia_full_pipeline.py`` and
``demos/desi_full_pipeline.py``. Runs the four passes needed for the
asymmetric Landy-Szalay cross estimator::

    xi_LS_xy(theta; z) = (mu_DD - mu_DR - mu_RD + mu_RR) / mu_RR

where each ``mu_X = nbar_X / N_neighbor_per_zn`` puts the per-cap
counts on a common per-cap-volume scale. ``x = DESI``, ``y = Quaia``.

Passes (all flavor="RD" since the catalogs are distinct objects, no
self-exclusion needed):

- ``DD_xy``: query=DESI data, neighbor=Quaia data
- ``DR_xy``: query=DESI data, neighbor=Quaia random
- ``RD_xy``: query=DESI random, neighbor=Quaia data
- ``RR_xy``: query=DESI random, neighbor=Quaia random

Random catalogs are re-drawn with the same seeds used by the
auto-correlation pipelines (DESI: rng_seed=2025 in
``desi_full_pipeline.py``; Quaia: rng_seed=2025 in
``quaia_full_pipeline.py``), so the cross artifacts are paired with
the auto artifacts on identical random sets.

Saves four artifacts under ``output/`` with key prefixes mirroring the
auto pipelines (``rd_*``-style):

- ``quaia_x_desi_dd.npz``  (DD_xy)
- ``quaia_x_desi_dr.npz``  (DR_xy)
- ``quaia_x_desi_rd.npz``  (RD_xy)
- ``quaia_x_desi_rr.npz``  (RR_xy)

We use ``k_max=0`` for all four passes — only the moment cubes
``sum_n``, ``sum_n2`` are needed for the xi_LS estimator.
``H_geq_k`` (kNN ladder for CIC PMF) is not needed here.

Tunables via env vars (mirrors the auto pipelines):
    PAPER_THETA_MIN_DEG     default 0.1
    PAPER_THETA_MAX_DEG     default 12.0
    PAPER_N_THETA_BINS      default 90
    PAPER_Z_MIN             default 0.8  (overlap range)
    PAPER_Z_MAX             default 2.1
    PAPER_N_R_FACTOR_NUM    default 1
    PAPER_N_R_FACTOR_DEN    default 5    (-> N_R = 0.2 N_D, both surveys)
    PAPER_NSIDE_LOOKUP      default 512
    PAPER_N_THREADS         default 8
    PAPER_CHUNK_SIZE        default 5000
"""

from __future__ import annotations

import os
import time

import numpy as np

from twopt_density.distance import DistanceCosmo
from twopt_density.desi import (
    load_desi_qso,
    random_queries_desi_per_region,
    split_n_random_by_data_fraction,
)
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.knn_cdf import joint_knn_cdf
from twopt_density.knn_analytic_rr import random_queries_from_selection_function


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESI_DATA_DIR = os.path.join(REPO_ROOT, "data", "desi")
QUAIA_DATA_DIR = "/Users/tabel/Research/data/quaia"
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DESI_COMPLETENESS_N = os.path.join(
    DESI_DATA_DIR, "desi_qso_y1_completeness_N_NSIDE64.fits")
DESI_COMPLETENESS_S = os.path.join(
    DESI_DATA_DIR, "desi_qso_y1_completeness_S_NSIDE64.fits")


def _save_pass(path, label, query_kind, neigh_kind, res, n_q, n_n, t_elapsed,
               desi_random_photsys=None):
    """Save a single cross-correlation pass artifact with consistent keys."""
    extras = {}
    if desi_random_photsys is not None:
        extras["random_photsys_desi"] = desi_random_photsys
    np.savez_compressed(
        path,
        label=label,
        query_kind=query_kind,        # "data_x" or "random_x"
        neigh_kind=neigh_kind,        # "data_y" or "random_y"
        n_query=n_q,
        n_neigh=n_n,
        theta_radii_rad=res.theta_radii_rad,
        z_q_edges=res.z_q_edges,
        z_n_edges=res.z_n_edges,
        sum_n=res.sum_n,
        sum_n2=res.sum_n2,
        N_q=res.N_q,
        elapsed_s=t_elapsed,
        **extras,
    )


def main():
    theta_min = float(os.environ.get("PAPER_THETA_MIN_DEG", 0.05))
    theta_max = float(os.environ.get("PAPER_THETA_MAX_DEG", 12.0))
    n_theta = int(os.environ.get("PAPER_N_THETA_BINS", 90))
    z_min = float(os.environ.get("PAPER_Z_MIN", 0.8))
    z_max = float(os.environ.get("PAPER_Z_MAX", 2.1))
    nr_num = int(os.environ.get("PAPER_N_R_FACTOR_NUM", 1))
    nr_den = int(os.environ.get("PAPER_N_R_FACTOR_DEN", 5))
    nside_lookup = int(os.environ.get("PAPER_NSIDE_LOOKUP", 512))
    n_threads = int(os.environ.get("PAPER_N_THREADS", 8))
    chunk_size = int(os.environ.get("PAPER_CHUNK_SIZE", 5000))

    t_start = time.time()
    fid = DistanceCosmo(Om=0.31, h=0.68)

    # ---- Load DESI ---------------------------------------------------
    print("loading DESI Y1 QSO (NGC + SGC) ...")
    desi = load_desi_qso(
        catalog_paths=[
            os.path.join(DESI_DATA_DIR, "QSO_NGC_clustering.dat.fits"),
            os.path.join(DESI_DATA_DIR, "QSO_SGC_clustering.dat.fits"),
        ],
        randoms_paths=None,
        fid_cosmo=fid, z_min=z_min, z_max=z_max, with_weight_fkp=True,
        with_photsys=True,
    )
    n_dx = desi.ra_data.size
    print(f"  N_data_x (DESI) = {n_dx}")
    print(f"  z range = [{desi.z_data.min():.2f}, {desi.z_data.max():.2f}]")
    print(f"  per-region: "
          f"N={int((desi.photsys_data=='N').sum())}, "
          f"S={int((desi.photsys_data=='S').sum())}")

    import healpy as hp
    desi_sel_N = hp.read_map(DESI_COMPLETENESS_N, verbose=False)
    desi_sel_S = hp.read_map(DESI_COMPLETENESS_S, verbose=False)
    desi_nside_mask = hp.npix2nside(desi_sel_N.size)
    if hp.npix2nside(desi_sel_S.size) != desi_nside_mask:
        raise SystemExit("DESI N and S completeness maps must share NSIDE.")

    # ---- Load Quaia --------------------------------------------------
    print("\nloading Quaia G<20 ...")
    quaia = load_quaia(
        catalog_path=os.path.join(QUAIA_DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            QUAIA_DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    quaia_sel_map, quaia_nside_mask = load_selection_function(
        os.path.join(QUAIA_DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md_y = (quaia.z_data >= z_min) & (quaia.z_data <= z_max)
    where_y = np.where(md_y)[0]
    ra_dy = quaia.ra_data[where_y]
    dec_dy = quaia.dec_data[where_y]
    z_dy = quaia.z_data[where_y]
    n_dy = ra_dy.size
    print(f"  N_data_y (Quaia, z in [{z_min},{z_max}]) = {n_dy}")

    # ---- Random catalogs (same seeds as auto pipelines) -------------
    n_rx = (n_dx * nr_num) // nr_den
    n_ry = (n_dy * nr_num) // nr_den
    print(f"\ndrawing random catalogs (seed=2025) ...")
    print(f"  N_random_x (DESI) = {n_rx}")
    print(f"  N_random_y (Quaia) = {n_ry}")

    n_rx_per_region = split_n_random_by_data_fraction(
        n_rx, desi.photsys_data)
    print(f"  DESI random per-region split: {n_rx_per_region}")
    ra_rx, dec_rx, z_rx, photsys_rx = random_queries_desi_per_region(
        region_sel_maps={"N": desi_sel_N, "S": desi_sel_S},
        region_z_pools={
            "N": desi.z_data[desi.photsys_data == "N"],
            "S": desi.z_data[desi.photsys_data == "S"],
        },
        n_random_per_region=n_rx_per_region,
        nside=desi_nside_mask, rng=np.random.default_rng(2025),
    )
    ra_ry, dec_ry, z_ry = random_queries_from_selection_function(
        sel_map=quaia_sel_map, z_data=z_dy, n_random=n_ry,
        nside=quaia_nside_mask, rng=np.random.default_rng(2025),
    )

    # ---- Common grids ------------------------------------------------
    theta_radii = np.deg2rad(np.geomspace(theta_min, theta_max, n_theta))
    z_edges = np.expm1(np.linspace(np.log1p(z_min), np.log1p(z_max), 5))
    z_q_edges = z_n_edges = z_edges
    print(f"\n  theta bins (deg): {np.round(np.degrees(theta_radii), 3)}")
    print(f"  z shells: {np.round(z_edges, 3)}")

    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        nside_lookup=nside_lookup, n_threads=n_threads,
        query_chunk_size=chunk_size, progress=True,
        k_max=0, flavor="RD",   # cross-catalog: no self-exclusion
    )

    # ---- DD_xy: query=D_x (DESI data), neighbor=D_y (Quaia data) ----
    print(f"\n=== DD_xy: query=DESI data ({n_dx}), "
          f"neighbor=Quaia data ({n_dy}) ===")
    t0 = time.time()
    res_dd = joint_knn_cdf(
        desi.ra_data, desi.dec_data, desi.z_data,
        ra_dy, dec_dy, z_dy,
        weights_neigh=None,
        **common,
    )
    t_dd = time.time() - t0
    print(f"  DD_xy done in {t_dd/60:.1f} min")
    _save_pass(
        os.path.join(OUTPUT_DIR, "quaia_x_desi_dd.npz"),
        "DD_xy", "data_x", "data_y",
        res_dd, n_dx, n_dy, t_dd,
    )
    print(f"  wrote quaia_x_desi_dd.npz")

    # ---- DR_xy: query=D_x (DESI data), neighbor=R_y (Quaia random) --
    print(f"\n=== DR_xy: query=DESI data ({n_dx}), "
          f"neighbor=Quaia random ({n_ry}) ===")
    t0 = time.time()
    res_dr = joint_knn_cdf(
        desi.ra_data, desi.dec_data, desi.z_data,
        ra_ry, dec_ry, z_ry,
        weights_neigh=None,
        **common,
    )
    t_dr = time.time() - t0
    print(f"  DR_xy done in {t_dr/60:.1f} min")
    _save_pass(
        os.path.join(OUTPUT_DIR, "quaia_x_desi_dr.npz"),
        "DR_xy", "data_x", "random_y",
        res_dr, n_dx, n_ry, t_dr,
    )
    print(f"  wrote quaia_x_desi_dr.npz")

    # ---- RD_xy: query=R_x (DESI random), neighbor=D_y (Quaia data) --
    print(f"\n=== RD_xy: query=DESI random ({n_rx}), "
          f"neighbor=Quaia data ({n_dy}) ===")
    t0 = time.time()
    res_rd = joint_knn_cdf(
        ra_rx, dec_rx, z_rx,
        ra_dy, dec_dy, z_dy,
        weights_neigh=None,
        **common,
    )
    t_rd = time.time() - t0
    print(f"  RD_xy done in {t_rd/60:.1f} min")
    _save_pass(
        os.path.join(OUTPUT_DIR, "quaia_x_desi_rd.npz"),
        "RD_xy", "random_x", "data_y",
        res_rd, n_rx, n_dy, t_rd,
        desi_random_photsys=photsys_rx,
    )
    print(f"  wrote quaia_x_desi_rd.npz")

    # ---- RR_xy: query=R_x (DESI random), neighbor=R_y (Quaia random) -
    print(f"\n=== RR_xy: query=DESI random ({n_rx}), "
          f"neighbor=Quaia random ({n_ry}) ===")
    t0 = time.time()
    res_rr = joint_knn_cdf(
        ra_rx, dec_rx, z_rx,
        ra_ry, dec_ry, z_ry,
        weights_neigh=None,
        **common,
    )
    t_rr = time.time() - t0
    print(f"  RR_xy done in {t_rr/60:.1f} min")
    _save_pass(
        os.path.join(OUTPUT_DIR, "quaia_x_desi_rr.npz"),
        "RR_xy", "random_x", "random_y",
        res_rr, n_rx, n_ry, t_rr,
        desi_random_photsys=photsys_rx,
    )
    print(f"  wrote quaia_x_desi_rr.npz")

    t_total = time.time() - t_start
    print(f"\n=== Cross pipeline complete: {t_total/60:.1f} min wall ===")
    print(f"  DD_xy: {t_dd/60:.1f} min  ({n_dx} x {n_dy})")
    print(f"  DR_xy: {t_dr/60:.1f} min  ({n_dx} x {n_ry})")
    print(f"  RD_xy: {t_rd/60:.1f} min  ({n_rx} x {n_dy})")
    print(f"  RR_xy: {t_rr/60:.1f} min  ({n_rx} x {n_ry})")


if __name__ == "__main__":
    main()
