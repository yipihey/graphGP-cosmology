"""Phase C: N_R / N_D convergence sweep on full Quaia.

For each ``N_R / N_D`` value, runs an RD-flavor kNN-CDF pass with
single-pass jackknife and saves the moment + per-region cubes to
``output/quaia_n_r_sweep.npz``. Phase D loads this artifact, computes
``sigma2_clust`` and its jackknife covariance per N_R, and plots the
SNR convergence curves.

Tunables via env vars:
    PAPER_N_R_FACTORS   default "1,2,5,10"  (comma-separated)
    PAPER_NSIDE_LOOKUP  default 512
    PAPER_N_REGIONS     default 25
    PAPER_THETA_MAX_DEG default 8.0
    PAPER_N_THREADS     default 8
    PAPER_CHUNK_SIZE    default 5000
"""

from __future__ import annotations

import os
import time

import numpy as np

from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.knn_cdf import joint_knn_cdf
from twopt_density.knn_analytic_rr import random_queries_from_selection_function
from twopt_density.jackknife import jackknife_region_labels


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/Users/tabel/Research/data/quaia"
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
ARTIFACT = os.path.join(OUTPUT_DIR, "quaia_n_r_sweep.npz")


def main():
    n_r_factors = [int(s) for s in os.environ.get(
        "PAPER_N_R_FACTORS", "1,2,5,10").split(",")]
    nside_lookup = int(os.environ.get("PAPER_NSIDE_LOOKUP", 512))
    n_regions = int(os.environ.get("PAPER_N_REGIONS", 25))
    theta_max_deg = float(os.environ.get("PAPER_THETA_MAX_DEG", 8.0))
    n_threads = int(os.environ.get("PAPER_N_THREADS", 8))
    chunk_size = int(os.environ.get("PAPER_CHUNK_SIZE", 5000))

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading full Quaia G < 20 ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    sel_map, nside_mask = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))

    md = (cat.z_data >= 0.5) & (cat.z_data <= 3.0)
    where = np.where(md)[0]
    ra_d = cat.ra_data[where]
    dec_d = cat.dec_data[where]
    z_d = cat.z_data[where]
    n_d = ra_d.size
    print(f"  using {n_d} galaxies in z=[0.5, 3.0]")

    theta_radii = np.deg2rad(np.geomspace(0.3, theta_max_deg, 9))
    z_edges = np.expm1(np.linspace(np.log1p(0.5), np.log1p(3.0), 5))
    z_q_edges = z_n_edges = z_edges

    print(f"  theta bins (deg): {np.round(np.degrees(theta_radii), 2)}")
    print(f"  z shells: {np.round(z_edges, 3)}")
    print(f"  N_R/N_D sweep: {n_r_factors}")

    artifact = dict(
        n_d=n_d,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions,
        n_r_factors=np.array(n_r_factors, dtype=np.int64),
    )

    for nr_factor in n_r_factors:
        n_rand = nr_factor * n_d
        print(f"\n=== N_R/N_D = {nr_factor}  (N_R = {n_rand}) ===")
        t0 = time.time()
        ra_r, dec_r, z_r = random_queries_from_selection_function(
            sel_map=sel_map, z_data=z_d, n_random=n_rand,
            nside=nside_mask, rng=np.random.default_rng(1000 + nr_factor),
        )
        print(f"  drew {n_rand} randoms in {time.time() - t0:.1f}s")

        # Per-query jackknife region labels.
        labels_r, _ = jackknife_region_labels(
            ra_r, dec_r, n_regions=n_regions, nside_jack=4,
        )

        t0 = time.time()
        res = joint_knn_cdf(
            ra_r, dec_r, z_r, ra_d, dec_d, z_d,
            theta_radii_rad=theta_radii,
            z_q_edges=z_q_edges, z_n_edges=z_n_edges,
            k_max=0, flavor="RD",
            region_labels_query=labels_r, n_regions=n_regions,
            nside_lookup=nside_lookup, n_threads=n_threads,
            query_chunk_size=chunk_size, progress=True,
        )
        elapsed = time.time() - t0
        print(f"  RD pass done in {elapsed/60:.1f} min")

        prefix = f"nr_{nr_factor}"
        artifact[f"{prefix}_sum_n"] = res.sum_n
        artifact[f"{prefix}_sum_n2"] = res.sum_n2
        artifact[f"{prefix}_N_q"] = res.N_q
        artifact[f"{prefix}_sum_n_per_region"] = res.sum_n_per_region
        artifact[f"{prefix}_sum_n2_per_region"] = res.sum_n2_per_region
        artifact[f"{prefix}_N_q_per_region"] = res.N_q_per_region
        artifact[f"{prefix}_elapsed_s"] = elapsed

        # Save after each N_R so partial progress survives interruption.
        np.savez_compressed(ARTIFACT, **artifact)
        print(f"  saved {ARTIFACT} ({os.path.getsize(ARTIFACT)/1024/1024:.1f} MB)")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
