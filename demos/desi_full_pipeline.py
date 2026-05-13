"""Unified full-DESI-Y1-QSO kNN-CDF pipeline: DD + RD + RR in one process.

Mirrors ``demos/quaia_full_pipeline.py`` for the DESI Y1 QSO clustering
catalogue. Loads NGC + SGC, applies z-cut, draws randoms from the
locally-built completeness map (data/desi/desi_qso_y1_completeness_*),
and runs DD (with WEIGHT * WEIGHT_FKP per object), RD (random queries
on data), and RR (random queries on randoms, self-excluded) in one
script with one Quaia-mask load.

Saves four artifacts under ``output/``:
- ``desi_full_dd.npz``           DD merged-grid cubes (legacy
                                   ``dd_bao_*`` and ``dd_small_*`` keys
                                   for HTML demo back-compat)
- ``desi_full_rd_1x_kmax.npz``  RD cube (CIC PMF + LS source)
- ``desi_full_rd_0p2x.npz``     alias of above for convergence panel
- ``desi_full_rr.npz``           RR cube (k_max=0, self-excluded)

Tunables via env vars (mirroring quaia_full_pipeline.py):
    PAPER_THETA_MIN_DEG          default 0.1
    PAPER_THETA_MAX_DEG          default 12.0
    PAPER_N_THETA_BINS           default 90
    PAPER_THETA_SMALL_SPLIT_DEG  default 1.5  (legacy slice cutoff)
    PAPER_Z_MIN                  default 0.8  (DESI Y1 QSO range)
    PAPER_Z_MAX                  default 2.1
    PAPER_N_R_FACTOR_NUM         default 1   (-> N_R = num/den * N_D)
    PAPER_N_R_FACTOR_DEN         default 5
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
from twopt_density.desi import (
    load_desi_qso,
    random_queries_desi_per_region,
    split_n_random_by_data_fraction,
)
from twopt_density.knn_cdf import joint_knn_cdf
from twopt_density.jackknife import jackknife_region_labels
from twopt_density.zgrid import construct_z_grid


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "desi")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
COMPLETENESS_N = os.path.join(DATA_DIR, "desi_qso_y1_completeness_N_NSIDE64.fits")
COMPLETENESS_S = os.path.join(DATA_DIR, "desi_qso_y1_completeness_S_NSIDE64.fits")


def _higher_moment_keys(prefix, res):
    """Return a dict of sum_n3, sum_n4 (and per-region) keys, prefixed
    by ``prefix``. Empty if the result has no higher moments. Used to
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
    n_theta = int(os.environ.get("PAPER_N_THETA_BINS", 90))
    theta_split = float(os.environ.get("PAPER_THETA_SMALL_SPLIT_DEG", 1.5))
    z_min = float(os.environ.get("PAPER_Z_MIN", 0.8))
    z_max = float(os.environ.get("PAPER_Z_MAX", 2.1))
    nr_num = int(os.environ.get("PAPER_N_R_FACTOR_NUM", 1))
    nr_den = int(os.environ.get("PAPER_N_R_FACTOR_DEN", 5))
    k_max = int(os.environ.get("PAPER_K_MAX", 302))
    nside_lookup = int(os.environ.get("PAPER_NSIDE_LOOKUP", 512))
    n_regions = int(os.environ.get("PAPER_N_REGIONS", 25))
    n_threads = int(os.environ.get("PAPER_N_THREADS", 8))
    chunk_size = int(os.environ.get("PAPER_CHUNK_SIZE", 5000))
    do_dr_pass = os.environ.get("PAPER_DR_PASS", "1") == "1"
    offdiag_n_z = int(os.environ.get("PAPER_OFFDIAG_N_Z", 0))

    # ---- Load DESI Y1 QSO + completeness map -------------------------
    t_start = time.time()
    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading DESI Y1 QSO (NGC + SGC) ...")
    cat = load_desi_qso(
        catalog_paths=[
            os.path.join(DATA_DIR, "QSO_NGC_clustering.dat.fits"),
            os.path.join(DATA_DIR, "QSO_SGC_clustering.dat.fits"),
        ],
        randoms_paths=None,
        fid_cosmo=fid, z_min=z_min, z_max=z_max, with_weight_fkp=True,
        with_photsys=True,
    )
    n_d = cat.ra_data.size
    print(f"  N_data={n_d}  z=[{cat.z_data.min():.2f}, {cat.z_data.max():.2f}]")
    print(f"  weights: mean={cat.w_data.mean():.3f}  "
          f"min={cat.w_data.min():.2f}  max={cat.w_data.max():.2f}")
    if cat.photsys_data.size != n_d:
        raise SystemExit(
            "PHOTSYS not loaded; per-region random recipe requires it. "
            "Confirm twopt_density/desi.py loader and re-run."
        )
    region_counts = {r: int((cat.photsys_data == r).sum())
                     for r in ("N", "S")}
    print(f"  per-region data counts: {region_counts}")

    for path in (COMPLETENESS_N, COMPLETENESS_S):
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing {path}. Run demos/build_desi_completeness_map.py "
                "first to build the per-region completeness maps."
            )
    import healpy as hp
    sel_N = hp.read_map(COMPLETENESS_N, verbose=False)
    sel_S = hp.read_map(COMPLETENESS_S, verbose=False)
    nside_mask = hp.npix2nside(sel_N.size)
    if hp.npix2nside(sel_S.size) != nside_mask:
        raise SystemExit("N and S completeness maps must share NSIDE.")
    print(f"  per-region completeness maps: NSIDE={nside_mask}, "
          f"sky frac N={(sel_N>0).mean():.3f}, S={(sel_S>0).mean():.3f}")

    n_rand = (n_d * nr_num) // nr_den
    n_rand_per_region = split_n_random_by_data_fraction(
        n_rand, cat.photsys_data)
    print(f"  N_random = {n_rand}  (N_R/N_D = {nr_num}/{nr_den} "
          f"= {nr_num/nr_den:.2f})  per-region: {n_rand_per_region}")

    # ---- Merged theta grid + z shell edges --------------------------
    n_z_shells = int(os.environ.get("PAPER_N_Z_SHELLS", 4))
    diagonal_only = os.environ.get("PAPER_DIAGONAL_ONLY", "1") == "1"
    z_grid_spec = os.environ.get("PAPER_Z_GRID", "log1pz")
    theta_radii = np.deg2rad(np.geomspace(theta_min, theta_max, n_theta))
    # Note v4_1 A.2: rdeciles for R-centered, dquantiles for D-centered.
    # PAPER_N_Z_SHELLS overrides the default n_deciles=9 / n_bins=90.
    z_edges = construct_z_grid(
        z_grid_spec, cat.z_data, z_min=z_min, z_max=z_max,
        n_shells=n_z_shells)
    z_q_edges = z_n_edges = z_edges
    artifact_suffix = "" if z_grid_spec == "log1pz" else f"_{z_grid_spec}"
    print(f"  theta bins (deg): {np.round(np.degrees(theta_radii), 3)}")
    print(f"  z grid spec:      {z_grid_spec}{artifact_suffix or ' (legacy)'}")
    print(f"  z shells:         {np.round(z_edges, 3)}")
    print(f"  n_z_shells:       {n_z_shells}  diagonal_only={diagonal_only}")
    print(f"  k_max:            {k_max}")
    print(f"  jackknife regions: {n_regions}")

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

    # ---- DD pass (data queries on data, self-excluded, weighted) ----
    print(f"\n=== DD pass (full DESI, theta {theta_min}-{theta_max} deg) ===")
    labels_d, _ = jackknife_region_labels(
        cat.ra_data, cat.dec_data, n_regions=n_regions, nside_jack=4,
    )
    t0 = time.time()
    res_dd = joint_knn_cdf(
        cat.ra_data, cat.dec_data, cat.z_data,
        cat.ra_data, cat.dec_data, cat.z_data,
        weights_neigh=cat.w_data,
        k_max=k_max, flavor="DD",
        region_labels_query=labels_d, n_regions=n_regions,
        **common,
    )
    t_dd = time.time() - t0
    print(f"  DD done in {t_dd/60:.1f} min")

    dd_artifact = dict(
        n_d=n_d,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions, k_max=k_max,
        theta_radii_bao_rad=theta_radii,
        theta_radii_small_rad=theta_radii[:max(small_idx, 1)],
        dd_bao_H_geq_k=res_dd.H_geq_k,
        dd_bao_sum_n=res_dd.sum_n,
        dd_bao_sum_n2=res_dd.sum_n2,
        dd_bao_N_q=res_dd.N_q,
        dd_bao_sum_n_per_region=res_dd.sum_n_per_region,
        dd_bao_sum_n2_per_region=res_dd.sum_n2_per_region,
        dd_bao_N_q_per_region=res_dd.N_q_per_region,
        dd_bao_H_geq_k_per_region=res_dd.H_geq_k_per_region,
        dd_bao_elapsed_s=t_dd,
        dd_small_H_geq_k=res_dd.H_geq_k[:max(small_idx, 1)],
        dd_small_sum_n=res_dd.sum_n[:max(small_idx, 1)],
        dd_small_sum_n2=res_dd.sum_n2[:max(small_idx, 1)],
        dd_small_N_q=res_dd.N_q,
        dd_small_sum_n_per_region=res_dd.sum_n_per_region[:max(small_idx, 1)],
        dd_small_sum_n2_per_region=res_dd.sum_n2_per_region[:max(small_idx, 1)],
        dd_small_N_q_per_region=res_dd.N_q_per_region,
        dd_small_H_geq_k_per_region=res_dd.H_geq_k_per_region[:max(small_idx, 1)],
        dd_small_elapsed_s=0.0,
    )
    dd_artifact.update(_higher_moment_keys("dd_bao_", res_dd))
    if getattr(res_dd, "sum_n3", None) is not None:
        dd_artifact["dd_small_sum_n3"] = res_dd.sum_n3[:max(small_idx, 1)]
        dd_artifact["dd_small_sum_n4"] = res_dd.sum_n4[:max(small_idx, 1)]
        if res_dd.sum_n3_per_region is not None:
            dd_artifact["dd_small_sum_n3_per_region"] = (
                res_dd.sum_n3_per_region[:max(small_idx, 1)])
            dd_artifact["dd_small_sum_n4_per_region"] = (
                res_dd.sum_n4_per_region[:max(small_idx, 1)])
    dd_path = os.path.join(
        OUTPUT_DIR, f"desi_full_dd{artifact_suffix}.npz")
    np.savez_compressed(dd_path, **dd_artifact)
    print(f"  wrote {os.path.basename(dd_path)}")

    # ---- Random catalog (drawn ONCE per region; shared by RD and RR) -
    print("\ndrawing per-region random catalogs (N from BASS/MzLS, "
          "S from DECaLS) ...")
    region_z_pools = {
        "N": cat.z_data[cat.photsys_data == "N"],
        "S": cat.z_data[cat.photsys_data == "S"],
    }
    ra_r, dec_r, z_r, photsys_r = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools=region_z_pools,
        n_random_per_region=n_rand_per_region,
        nside=nside_mask, rng=np.random.default_rng(2025),
    )
    print(f"  drawn N={ (photsys_r=='N').sum()}, "
          f"S={(photsys_r=='S').sum()}, total={ra_r.size}")
    labels_r, _ = jackknife_region_labels(
        ra_r, dec_r, n_regions=n_regions, nside_jack=4,
    )

    # ---- RD pass (random queries on data, weighted) ------------------
    print(f"\n=== RD pass (N_R/N_D = {nr_num}/{nr_den}, k_max={k_max}) ===")
    t0 = time.time()
    res_rd = joint_knn_cdf(
        ra_r, dec_r, z_r,
        cat.ra_data, cat.dec_data, cat.z_data,
        weights_neigh=cat.w_data,
        k_max=k_max, flavor="RD",
        region_labels_query=labels_r, n_regions=n_regions,
        **common,
    )
    t_rd = time.time() - t0
    print(f"  RD done in {t_rd/60:.1f} min")

    rd_artifact = dict(
        n_d=n_d, n_r=n_rand, k_max=k_max,
        theta_radii_rad=theta_radii,
        z_q_edges=z_q_edges, z_n_edges=z_n_edges,
        n_regions=n_regions,
        random_photsys=photsys_r,
        n_rand_per_region=np.array(
            [n_rand_per_region.get("N", 0), n_rand_per_region.get("S", 0)],
            dtype=np.int64,
        ),
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
        OUTPUT_DIR, f"desi_full_rd_1x_kmax{artifact_suffix}.npz")
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
        OUTPUT_DIR, f"desi_full_rd_0p2x{artifact_suffix}.npz")
    np.savez_compressed(rd_conv_path, **rd_conv_artifact)
    print(f"  wrote {os.path.basename(rd_conv_path)}")

    # ---- RR pass (random queries on randoms, self-excluded) ----------
    # For LS-grade RR we'd want N_R^RR = N_D, but mirror the unified
    # Quaia pipeline's lower-N_R RR for now -- the LS panel will use
    # the same statistics as the Quaia equivalent.
    print(f"\n=== RR pass (queries=randoms, neighbors=randoms, "
          "self-excluded) ===")
    t0 = time.time()
    res_rr = joint_knn_cdf(
        ra_r, dec_r, z_r, ra_r, dec_r, z_r,
        k_max=0, flavor="DD",
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
        random_photsys=photsys_r,
        n_rand_per_region=np.array(
            [n_rand_per_region.get("N", 0), n_rand_per_region.get("S", 0)],
            dtype=np.int64,
        ),
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
        OUTPUT_DIR, f"desi_full_rr{artifact_suffix}.npz")
    np.savez_compressed(rr_path, **rr_artifact)
    print(f"  wrote {os.path.basename(rr_path)}")

    # ---- DR pass (data queries on randoms) — completes the four-flavor
    # set {DD, DR, RD, RR} from note v4_1 §2. Required for the Hamilton
    # estimator (Eq. 15) and Landy-Szalay higher moments (Eq. 13).
    # Skip with PAPER_DR_PASS=0 to keep legacy 3-flavor behaviour.
    # Note on weights: queries are unweighted data positions; neighbour
    # randoms are unweighted (RR-style randoms have no FKP weight).
    # ------------------------------------------------------------------
    t_dr = 0.0
    if do_dr_pass:
        print(f"\n=== DR pass (data queries, random neighbours; "
              f"k_max={k_max}) ===")
        t0 = time.time()
        res_dr = joint_knn_cdf(
            cat.ra_data, cat.dec_data, cat.z_data,
            ra_r, dec_r, z_r,
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
            random_photsys=photsys_r,
            n_rand_per_region=np.array(
                [n_rand_per_region.get("N", 0),
                 n_rand_per_region.get("S", 0)],
                dtype=np.int64,
            ),
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
            OUTPUT_DIR, f"desi_full_dr{artifact_suffix}.npz")
        np.savez_compressed(dr_path, **dr_artifact)
        print(f"  wrote {os.path.basename(dr_path)}")
    else:
        print("\n=== DR pass skipped (PAPER_DR_PASS=0) ===")

    # ---- Off-diagonal pass set (note v4_1 §4-5): coarse n_z full
    # (z₁, z₂) plane to support ∂lnσ²/∂ln(1+z) and ξ_ℓ(s). Skip
    # with PAPER_OFFDIAG_N_Z=0 (default). -----------------------------
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
            cat.ra_data, cat.dec_data, cat.z_data,
            cat.ra_data, cat.dec_data, cat.z_data,
            weights_neigh=cat.w_data,
            k_max=0, flavor="DD",
            region_labels_query=labels_d, n_regions=n_regions,
            **common_off,
        )
        res_off_rd = joint_knn_cdf(
            ra_r, dec_r, z_r,
            cat.ra_data, cat.dec_data, cat.z_data,
            weights_neigh=cat.w_data,
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
                cat.ra_data, cat.dec_data, cat.z_data,
                ra_r, dec_r, z_r,
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
            OUTPUT_DIR, f"desi_full_offdiag{artifact_suffix}.npz")
        np.savez_compressed(off_path, **off_artifact)
        print(f"  off-diagonal done in {t_off/60:.1f} min")
        print(f"  wrote {os.path.basename(off_path)}")

    t_total = time.time() - t_start
    print(f"\n=== Pipeline complete: {t_total/60:.1f} min wall ===")
    print(f"  DD: {t_dd/60:.1f} min")
    print(f"  RD: {t_rd/60:.1f} min")
    print(f"  RR: {t_rr/60:.1f} min")
    if do_dr_pass:
        print(f"  DR: {t_dr/60:.1f} min")
    if offdiag_n_z > 0:
        print(f"  off-diag: {t_off/60:.1f} min")


if __name__ == "__main__":
    main()
