"""Unified full-Quaia kNN-CDF pipeline: DD + RD + RR in one process.

Replaces the separate ``quaia_full_dd_rd.py`` + ``quaia_rd1x_kmax.py``
+ ``quaia_full_rr.py`` scripts with a single pass that:

- Loads Quaia + selection function ONCE.
- Runs ONE DD pass on a merged theta grid covering both small-scale
  and BAO-scale views (geomspace(theta_min, theta_max, n_theta)).
- Runs ONE RD pass at N_R/N_D = 0.2 (the precision sweet spot
  established by the 0.2x/1x/5x convergence study).
- Runs ONE RR pass at the same N_R for Landy-Szalay xi_LS.

Each pass uses one-pass jackknife (region_labels) so the per-region
cubes come for free. Saves four artifacts:

    output/quaia_full_dd.npz             # DD merged grid (legacy keys
                                           dd_bao_*, dd_small_* derived
                                           by slicing for back-compat)
    output/quaia_full_rd_1x_kmax.npz     # RD cube; key prefix ``rd_*``
    output/quaia_full_rd_0p2x.npz        # alias of the above for the
                                           convergence panel ``nr_0p2_*``
                                           prefix (back-compat)
    output/quaia_full_rr.npz             # RR cube; key prefix ``rr_*``

Total wall time on 8 threads: ~12 min (vs ~33 min for the previous
multi-script pipeline). The convergence sweep
(``output/quaia_n_r_sweep.npz``) is not regenerated here — those cubes
remain from the one-shot calibration that established N_R = 0.2 N_D
is sufficient.

Tunables via env vars:
    PAPER_THETA_MIN_DEG          default 0.1
    PAPER_THETA_MAX_DEG          default 12.0
    PAPER_N_THETA_BINS           default 12
    PAPER_THETA_SMALL_SPLIT_DEG  default 1.5  (legacy slice cutoff for
                                                dd_small_* compatibility)
    PAPER_N_R_FACTOR_NUM         default 1   (numerator)
    PAPER_N_R_FACTOR_DEN         default 5   (denominator -> N_R = 1/5 * N_D)
    PAPER_K_MAX                  default 400
    PAPER_NSIDE_LOOKUP           default 512
    PAPER_N_REGIONS              default 25
    PAPER_N_THREADS              default 8
    PAPER_CHUNK_SIZE             default 5000
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
from twopt_density.zgrid import construct_z_grid


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/Users/tabel/Research/data/quaia"
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _higher_moment_keys(prefix, res):
    """Return a dict of sum_n3, sum_n4 (and per-region) keys, prefixed
    by ``prefix``. Empty if the result has no higher moments (e.g.
    older artifacts from before note v4_1 §6 was implemented). Used to
    splat into the np.savez_compressed kwargs."""
    out = {}
    for name in ("sum_n3", "sum_n4",
                 "sum_n3_per_region", "sum_n4_per_region"):
        v = getattr(res, name, None)
        if v is not None:
            out[f"{prefix}{name}"] = v
    return out


def main():
    theta_min = float(os.environ.get("PAPER_THETA_MIN_DEG", 0.05))
    theta_max = float(os.environ.get("PAPER_THETA_MAX_DEG", 12.0))
    n_theta = int(os.environ.get("PAPER_N_THETA_BINS", 12))
    theta_split = float(os.environ.get("PAPER_THETA_SMALL_SPLIT_DEG", 1.5))
    z_min = float(os.environ.get("PAPER_Z_MIN", 0.5))
    z_max = float(os.environ.get("PAPER_Z_MAX", 3.0))
    nr_num = int(os.environ.get("PAPER_N_R_FACTOR_NUM", 1))
    nr_den = int(os.environ.get("PAPER_N_R_FACTOR_DEN", 5))
    k_max = int(os.environ.get("PAPER_K_MAX", 302))
    nside_lookup = int(os.environ.get("PAPER_NSIDE_LOOKUP", 512))
    n_regions = int(os.environ.get("PAPER_N_REGIONS", 25))
    n_threads = int(os.environ.get("PAPER_N_THREADS", 8))
    chunk_size = int(os.environ.get("PAPER_CHUNK_SIZE", 5000))
    do_dr_pass = os.environ.get("PAPER_DR_PASS", "1") == "1"
    offdiag_n_z = int(os.environ.get("PAPER_OFFDIAG_N_Z", 0))

    # ---- Load Quaia ONCE -------------------------------------------------
    t_start = time.time()
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

    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    where = np.where(md)[0]
    ra_d = cat.ra_data[where]
    dec_d = cat.dec_data[where]
    z_d = cat.z_data[where]
    n_d = ra_d.size
    print(f"  z range used: [{z_min}, {z_max}]")
    n_rand = (n_d * nr_num) // nr_den
    print(f"  N_data={n_d}, N_random={n_rand}  "
          f"(N_R/N_D = {nr_num}/{nr_den} = {nr_num/nr_den:.2f})")

    # ---- Merged theta grid + z shell edges -------------------------------
    n_z_shells = int(os.environ.get("PAPER_N_Z_SHELLS", 4))
    diagonal_only = os.environ.get("PAPER_DIAGONAL_ONLY", "1") == "1"
    z_grid_spec = os.environ.get("PAPER_Z_GRID", "log1pz")
    theta_radii = np.deg2rad(np.geomspace(theta_min, theta_max, n_theta))
    # Note v4_1 A.2 R-centered defaults n_deciles=9; D-centered
    # defaults n_bins=90. PAPER_N_Z_SHELLS overrides those defaults.
    z_edges = construct_z_grid(
        z_grid_spec, z_d, z_min=z_min, z_max=z_max, n_shells=n_z_shells)
    z_q_edges = z_n_edges = z_edges
    # Artifact suffix so non-log1pz runs don't overwrite legacy
    # files. log1pz keeps the legacy filenames untouched.
    artifact_suffix = "" if z_grid_spec == "log1pz" else f"_{z_grid_spec}"
    print(f"  theta bins (deg): {np.round(np.degrees(theta_radii), 3)}")
    print(f"  z grid spec:      {z_grid_spec}{artifact_suffix or ' (legacy)'}")
    print(f"  z shells:         {np.round(z_edges, 3)}")
    print(f"  n_z_shells:       {n_z_shells}  diagonal_only={diagonal_only}")
    print(f"  k_max:            {k_max}")
    print(f"  jackknife regions: {n_regions}")

    # Find the index that splits "small" and "bao" theta for legacy
    # back-compat slicing.
    small_idx = int(np.searchsorted(np.degrees(theta_radii), theta_split))
    print(f"  legacy small/bao split at theta={theta_split} deg "
          f"(index {small_idx} of {n_theta})")

    backend = os.environ.get("PAPER_BACKEND", "numba")
    print(f"  backend:          {backend}  "
          f"({'morton_cascade Rust' if backend == 'cascade' else 'numba+HEALPix'})")
    common = dict(
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        nside_lookup=nside_lookup, n_threads=n_threads,
        query_chunk_size=chunk_size, progress=True,
        backend=backend,
        diagonal_only=diagonal_only,
    )

    # ---- DD pass (data queries on data, self-excluded) -----------------
    print(f"\n=== DD pass (full Quaia, theta {theta_min}–{theta_max} deg) ===")
    labels_d, _ = jackknife_region_labels(
        ra_d, dec_d, n_regions=n_regions, nside_jack=4,
    )
    t0 = time.time()
    res_dd = joint_knn_cdf(
        ra_d, dec_d, z_d, ra_d, dec_d, z_d,
        k_max=k_max, flavor="DD",
        region_labels_query=labels_d, n_regions=n_regions,
        **common,
    )
    t_dd = time.time() - t0
    print(f"  DD done in {t_dd/60:.1f} min")

    # Save DD with both merged ("dd_bao_*") and sliced ("dd_small_*")
    # views for back-compat with the existing HTML demo.
    dd_artifact = dict(
        n_d=n_d,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions, k_max=k_max,
        theta_radii_bao_rad=theta_radii,
        theta_radii_small_rad=theta_radii[:max(small_idx, 1)],
        # full merged cube under dd_bao_* keys
        dd_bao_H_geq_k=res_dd.H_geq_k,
        dd_bao_sum_n=res_dd.sum_n,
        dd_bao_sum_n2=res_dd.sum_n2,
        dd_bao_N_q=res_dd.N_q,
        dd_bao_sum_n_per_region=res_dd.sum_n_per_region,
        dd_bao_sum_n2_per_region=res_dd.sum_n2_per_region,
        dd_bao_N_q_per_region=res_dd.N_q_per_region,
        dd_bao_H_geq_k_per_region=res_dd.H_geq_k_per_region,
        dd_bao_elapsed_s=t_dd,
        # legacy small-scale slice (subset of merged cube)
        dd_small_H_geq_k=res_dd.H_geq_k[:max(small_idx, 1)],
        dd_small_sum_n=res_dd.sum_n[:max(small_idx, 1)],
        dd_small_sum_n2=res_dd.sum_n2[:max(small_idx, 1)],
        dd_small_N_q=res_dd.N_q,
        dd_small_sum_n_per_region=res_dd.sum_n_per_region[:max(small_idx, 1)],
        dd_small_sum_n2_per_region=res_dd.sum_n2_per_region[:max(small_idx, 1)],
        dd_small_N_q_per_region=res_dd.N_q_per_region,
        dd_small_H_geq_k_per_region=res_dd.H_geq_k_per_region[:max(small_idx, 1)],
        dd_small_elapsed_s=0.0,  # implied: free with dd_bao
    )
    dd_artifact.update(_higher_moment_keys("dd_bao_", res_dd))
    # Sliced higher-moment cubes (note: small-scale artifact uses the
    # same theta-prefix slice so sum_n3/sum_n4 keep their (theta, z)
    # leading axis aligned with the H/sum_n cubes).
    if getattr(res_dd, "sum_n3", None) is not None:
        dd_artifact["dd_small_sum_n3"] = res_dd.sum_n3[:max(small_idx, 1)]
        dd_artifact["dd_small_sum_n4"] = res_dd.sum_n4[:max(small_idx, 1)]
        if res_dd.sum_n3_per_region is not None:
            dd_artifact["dd_small_sum_n3_per_region"] = (
                res_dd.sum_n3_per_region[:max(small_idx, 1)])
            dd_artifact["dd_small_sum_n4_per_region"] = (
                res_dd.sum_n4_per_region[:max(small_idx, 1)])
    dd_path = os.path.join(
        OUTPUT_DIR, f"quaia_full_dd{artifact_suffix}.npz")
    np.savez_compressed(dd_path, **dd_artifact)
    print(f"  wrote {os.path.basename(dd_path)}")

    # ---- Random catalog (drawn ONCE; shared by RD and RR) ---------------
    print("\ndrawing random catalog from selection function ...")
    rng = np.random.default_rng(2025)
    ra_r, dec_r, z_r = random_queries_from_selection_function(
        sel_map=sel_map, z_data=z_d, n_random=n_rand,
        nside=nside_mask, rng=rng,
    )
    labels_r, _ = jackknife_region_labels(
        ra_r, dec_r, n_regions=n_regions, nside_jack=4,
    )

    # ---- RD pass (random queries on data, k_max>0 for CIC PMF) ---------
    print(f"\n=== RD pass (N_R/N_D = {nr_num}/{nr_den}, k_max={k_max}) ===")
    t0 = time.time()
    res_rd = joint_knn_cdf(
        ra_r, dec_r, z_r, ra_d, dec_d, z_d,
        k_max=k_max, flavor="RD",
        region_labels_query=labels_r, n_regions=n_regions,
        **common,
    )
    t_rd = time.time() - t0
    print(f"  RD done in {t_rd/60:.1f} min")

    # Two artifact files for back-compat with HTML demo's loaders:
    #   quaia_full_rd_1x_kmax.npz with rd_* keys (CIC PMF source)
    #   quaia_full_rd_0p2x.npz with nr_0p2_* keys (convergence panel)
    rd_artifact = dict(
        n_d=n_d, n_r=n_rand, k_max=k_max,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions,
        rd_H_geq_k=res_rd.H_geq_k,
        rd_sum_n=res_rd.sum_n,
        rd_sum_n2=res_rd.sum_n2,
        rd_N_q=res_rd.N_q,
        rd_sum_n_per_region=res_rd.sum_n_per_region,
        rd_sum_n2_per_region=res_rd.sum_n2_per_region,
        rd_N_q_per_region=res_rd.N_q_per_region,
        rd_H_geq_k_per_region=res_rd.H_geq_k_per_region,
        rd_elapsed_s=t_rd,
    )
    rd_artifact.update(_higher_moment_keys("rd_", res_rd))
    rd_path = os.path.join(
        OUTPUT_DIR, f"quaia_full_rd_1x_kmax{artifact_suffix}.npz")
    np.savez_compressed(rd_path, **rd_artifact)
    print(f"  wrote {os.path.basename(rd_path)}")

    rd_conv_artifact = dict(
        n_d=n_d, n_r=n_rand,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions,
        nr_0p2_sum_n=res_rd.sum_n,
        nr_0p2_sum_n2=res_rd.sum_n2,
        nr_0p2_N_q=res_rd.N_q,
        nr_0p2_sum_n_per_region=res_rd.sum_n_per_region,
        nr_0p2_sum_n2_per_region=res_rd.sum_n2_per_region,
        nr_0p2_N_q_per_region=res_rd.N_q_per_region,
        nr_0p2_elapsed_s=t_rd,
    )
    rd_conv_artifact.update(_higher_moment_keys("nr_0p2_", res_rd))
    rd_conv_path = os.path.join(
        OUTPUT_DIR, f"quaia_full_rd_0p2x{artifact_suffix}.npz")
    np.savez_compressed(rd_conv_path, **rd_conv_artifact)
    print(f"  wrote {os.path.basename(rd_conv_path)}")

    # ---- RR pass (random queries on randoms, self-excluded) -------------
    print(f"\n=== RR pass (queries=randoms, neighbors=randoms, "
          f"self-excluded) ===")
    t0 = time.time()
    res_rr = joint_knn_cdf(
        ra_r, dec_r, z_r, ra_r, dec_r, z_r,    # SAME object → DD self-exclusion
        k_max=0, flavor="DD",                   # DD-flavor enables self-exclusion
        region_labels_query=labels_r, n_regions=n_regions,
        **common,
    )
    t_rr = time.time() - t0
    print(f"  RR done in {t_rr/60:.1f} min")

    rr_artifact = dict(
        n_d=n_d, n_r=n_rand, n_r_factor=nr_num/nr_den,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions,
        rr_sum_n=res_rr.sum_n,
        rr_sum_n2=res_rr.sum_n2,
        rr_N_q=res_rr.N_q,
        rr_sum_n_per_region=res_rr.sum_n_per_region,
        rr_sum_n2_per_region=res_rr.sum_n2_per_region,
        rr_N_q_per_region=res_rr.N_q_per_region,
        rr_elapsed_s=t_rr,
    )
    rr_artifact.update(_higher_moment_keys("rr_", res_rr))
    rr_path = os.path.join(
        OUTPUT_DIR, f"quaia_full_rr{artifact_suffix}.npz")
    np.savez_compressed(rr_path, **rr_artifact)
    print(f"  wrote {os.path.basename(rr_path)}")

    # ---- DR pass (data queries on randoms) — completes the four-flavor
    # set {DD, DR, RD, RR} from note v4_1 §2. Required for the Hamilton
    # estimator (Eq. 15) and the Landy-Szalay estimator of higher
    # moments (Eq. 13). Skip with PAPER_DR_PASS=0 to keep legacy
    # 3-flavor behaviour. -----------------------------------------------
    t_dr = 0.0
    if do_dr_pass:
        print(f"\n=== DR pass (data queries, random neighbours; "
              f"k_max={k_max}) ===")
        t0 = time.time()
        res_dr = joint_knn_cdf(
            ra_d, dec_d, z_d, ra_r, dec_r, z_r,
            k_max=k_max, flavor="DR",
            region_labels_query=labels_d, n_regions=n_regions,
            **common,
        )
        t_dr = time.time() - t0
        print(f"  DR done in {t_dr/60:.1f} min")

        dr_artifact = dict(
            n_d=n_d, n_r=n_rand, k_max=k_max,
            theta_radii_rad=theta_radii,
            z_q_edges=z_q_edges, z_n_edges=z_n_edges,
            n_regions=n_regions,
            dr_H_geq_k=res_dr.H_geq_k,
            dr_sum_n=res_dr.sum_n,
            dr_sum_n2=res_dr.sum_n2,
            dr_N_q=res_dr.N_q,
            dr_sum_n_per_region=res_dr.sum_n_per_region,
            dr_sum_n2_per_region=res_dr.sum_n2_per_region,
            dr_N_q_per_region=res_dr.N_q_per_region,
            dr_H_geq_k_per_region=res_dr.H_geq_k_per_region,
            dr_elapsed_s=t_dr,
        )
        dr_artifact.update(_higher_moment_keys("dr_", res_dr))
        dr_path = os.path.join(
            OUTPUT_DIR, f"quaia_full_dr{artifact_suffix}.npz")
        np.savez_compressed(dr_path, **dr_artifact)
        print(f"  wrote {os.path.basename(dr_path)}")
    else:
        print("\n=== DR pass skipped (PAPER_DR_PASS=0) ===")

    # ---- Off-diagonal pass set (note v4_1 §4-5): coarse n_z full
    # (z₁, z₂) plane to support ∂lnσ²/∂ln(1+z) and ξ_ℓ(s) projection.
    # Skip with PAPER_OFFDIAG_N_Z=0 (default). -------------------------
    t_off = 0.0
    if offdiag_n_z > 0:
        from twopt_density.zgrid import log1pz_grid
        z_off = log1pz_grid(z_min, z_max, offdiag_n_z)
        common_off = dict(common)
        common_off["z_q_edges"] = z_off
        common_off["z_n_edges"] = z_off
        common_off["diagonal_only"] = False
        print(f"\n=== Off-diagonal passes (n_z={offdiag_n_z} full plane) ===")
        t0 = time.time()
        res_off_dd = joint_knn_cdf(
            ra_d, dec_d, z_d, ra_d, dec_d, z_d,
            k_max=0, flavor="DD",
            region_labels_query=labels_d, n_regions=n_regions,
            **common_off,
        )
        res_off_rd = joint_knn_cdf(
            ra_r, dec_r, z_r, ra_d, dec_d, z_d,
            k_max=0, flavor="RD",
            region_labels_query=labels_r, n_regions=n_regions,
            **common_off,
        )
        res_off_rr = joint_knn_cdf(
            ra_r, dec_r, z_r, ra_r, dec_r, z_r,
            k_max=0, flavor="DD",
            region_labels_query=labels_r, n_regions=n_regions,
            **common_off,
        )
        res_off_dr = None
        if do_dr_pass:
            res_off_dr = joint_knn_cdf(
                ra_d, dec_d, z_d, ra_r, dec_r, z_r,
                k_max=0, flavor="DR",
                region_labels_query=labels_d, n_regions=n_regions,
                **common_off,
            )
        t_off = time.time() - t0
        off_artifact = dict(
            n_d=n_d, n_r=n_rand,
            z_q_edges=z_off, z_n_edges=z_off,
            theta_radii_rad=theta_radii,
            n_regions=n_regions,
            offdiag_dd_sum_n=res_off_dd.sum_n,
            offdiag_dd_sum_n2=res_off_dd.sum_n2,
            offdiag_dd_sum_n_per_region=res_off_dd.sum_n_per_region,
            offdiag_dd_N_q=res_off_dd.N_q,
            offdiag_rd_sum_n=res_off_rd.sum_n,
            offdiag_rd_sum_n2=res_off_rd.sum_n2,
            offdiag_rd_sum_n_per_region=res_off_rd.sum_n_per_region,
            offdiag_rd_N_q=res_off_rd.N_q,
            offdiag_rr_sum_n=res_off_rr.sum_n,
            offdiag_rr_sum_n2=res_off_rr.sum_n2,
            offdiag_rr_sum_n_per_region=res_off_rr.sum_n_per_region,
            offdiag_rr_N_q=res_off_rr.N_q,
            offdiag_elapsed_s=t_off,
        )
        if res_off_dr is not None:
            off_artifact.update(
                offdiag_dr_sum_n=res_off_dr.sum_n,
                offdiag_dr_sum_n2=res_off_dr.sum_n2,
                offdiag_dr_sum_n_per_region=res_off_dr.sum_n_per_region,
                offdiag_dr_N_q=res_off_dr.N_q,
            )
        off_pairs = [
            ("offdiag_dd_", res_off_dd),
            ("offdiag_rd_", res_off_rd),
            ("offdiag_rr_", res_off_rr),
        ]
        if res_off_dr is not None:
            off_pairs.append(("offdiag_dr_", res_off_dr))
        for prefix, res_off in off_pairs:
            off_artifact.update(_higher_moment_keys(prefix, res_off))
        off_path = os.path.join(
            OUTPUT_DIR, f"quaia_full_offdiag{artifact_suffix}.npz")
        np.savez_compressed(off_path, **off_artifact)
        print(f"  off-diagonal done in {t_off/60:.1f} min")
        print(f"  wrote {os.path.basename(off_path)}")

    t_total = time.time() - t_start
    print(f"\n=== Pipeline complete: {t_total/60:.1f} min wall ===")
    print(f"  DD:        {t_dd/60:.1f} min")
    print(f"  RD:        {t_rd/60:.1f} min")
    print(f"  RR:        {t_rr/60:.1f} min")
    if do_dr_pass:
        print(f"  DR:        {t_dr/60:.1f} min")
    if offdiag_n_z > 0:
        print(f"  off-diag:  {t_off/60:.1f} min")
    print(f"  overhead:  "
          f"{(t_total - t_dd - t_rd - t_rr - t_dr - t_off)/60:.1f} min")


if __name__ == "__main__":
    main()
