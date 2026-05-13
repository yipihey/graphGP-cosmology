"""Cascade-tree backend for the angular kNN-CDF.

Drop-in replacement for ``twopt_density.knn_cdf.joint_knn_cdf`` that
runs the per-query angular pair-counting in Rust via the
``morton-cascade angular-knn-cdf`` CLI subcommand. Output is
**bit-identical** in observable semantics to the numba backend:
``H_geq_k`` integers match exactly, ``sum_n``/``sum_n2`` match to
machine precision (modulo parallel-summation order).

Mapping
-------
- ``(RA, Dec)`` → S² unit vectors via
  ``twopt_density.morton_backend.radec_to_chord_box`` (already shared
  with the existing 3D-cascade demos).
- Angular separation θ → chord ``d = 2 sin(θ/2)``; the per-query
  range query is run against that chord radius. Small-angle error
  <0.07% at θ=12°.

Weighting
---------
Per-neighbour weights propagate through the kernel exactly as in
``joint_knn_cdf``: ``sum_n``/``sum_n2`` accumulate weighted, and
the H-ladder uses ``int(weighted_sum)`` as the integer threshold.

Self-exclusion
--------------
For DD same-catalogue (object identity), the kernel mirrors the
numba behaviour: count the self-pair with its full weight, then
subtract 1.0 from ``n_cap[t, jn_self]`` and clip to 0 (matches
``_aggregate_query_global`` at ``knn_cdf.py:117–130``). To trigger
this, pass ``flavor='DD'`` AND identical ``query_*`` and ``neigh_*``
arrays (object identity is detected via ``is`` checks).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from .knn_cdf import KnnCdfResult
from .morton_backend import radec_to_chord_box, _resolve_binary


def joint_knn_cdf_cascade(
    query_ra_deg: np.ndarray, query_dec_deg: np.ndarray, query_z: np.ndarray,
    neigh_ra_deg: np.ndarray, neigh_dec_deg: np.ndarray, neigh_z: np.ndarray,
    theta_radii_rad: np.ndarray,
    z_q_edges: np.ndarray, z_n_edges: np.ndarray,
    k_max: int = 10,
    weights_neigh: Optional[np.ndarray] = None,
    region_labels_query: Optional[np.ndarray] = None,
    n_regions: int = 0,
    flavor: str = "DD",
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    quiet: bool = True,
    diagonal_only: bool = False,
) -> KnnCdfResult:
    """Cascade-tree drop-in for :func:`joint_knn_cdf`.

    Same input contract; same ``KnnCdfResult`` shape and observable
    semantics. Heavy lifting is in the Rust binary, driven by binary
    file I/O through a tempdir.
    """
    bin_path = _resolve_binary()

    flavor = flavor.upper()
    if flavor not in ("DD", "DR", "RD", "RR"):
        raise ValueError(
            f"unknown flavor {flavor!r}; expected DD, DR, RD or RR")

    same_catalog = (
        flavor == "DD"
        and (query_ra_deg is neigh_ra_deg)
        and (query_dec_deg is neigh_dec_deg)
        and (query_z is neigh_z)
    )

    n_q = int(query_ra_deg.size)
    n_n = int(neigh_ra_deg.size)
    n_theta = int(theta_radii_rad.size)
    n_z_q = int(z_q_edges.size - 1)
    n_z_n = int(z_n_edges.size - 1)

    # Map RA/Dec → S² unit vectors. radec_to_chord_box returns
    # (pts, L); we don't need L here because the CLI walks coordinates
    # directly without a box assumption.
    query_pts, _ = radec_to_chord_box(query_ra_deg, query_dec_deg)
    neigh_pts, _ = radec_to_chord_box(neigh_ra_deg, neigh_dec_deg)
    # The cascade uses chord = 2 sin(θ/2). Note: radec_to_chord_box
    # shifts to [0, L)^3 with a uniform offset, which is irrelevant
    # for chord distances (cancels in dx/dy/dz).
    chord_radii = 2.0 * np.sin(np.asarray(theta_radii_rad) / 2.0)

    # Synthetic TARGETIDs for self-exclusion: when same_catalog, the
    # query and neighbour catalogues are object-identical so row
    # index = TARGETID identifies pairs.
    if same_catalog:
        query_targetid = np.arange(n_q, dtype=np.int64)
        neigh_targetid = query_targetid  # alias
    else:
        query_targetid = None
        neigh_targetid = None

    cleanup = workdir is None
    if cleanup:
        workdir = tempfile.mkdtemp(prefix="aknn_")
    os.makedirs(workdir, exist_ok=True)
    out_dir = os.path.join(workdir, "out")
    os.makedirs(out_dir, exist_ok=True)

    def _wbin_f64(name, arr):
        p = os.path.join(workdir, name)
        np.ascontiguousarray(arr, dtype="<f8").tofile(p)
        return p

    def _wbin_i64(name, arr):
        p = os.path.join(workdir, name)
        np.ascontiguousarray(arr, dtype="<i8").tofile(p)
        return p

    try:
        cmd = [
            bin_path, "angular-knn-cdf",
            "--query-data", _wbin_f64("query_pts.bin", query_pts),
            "--query-z",    _wbin_f64("query_z.bin", query_z),
            "--neigh-data", _wbin_f64("neigh_pts.bin", neigh_pts),
            "--neigh-z",    _wbin_f64("neigh_z.bin", neigh_z),
            "--chord-radii", _wbin_f64("chord.bin", chord_radii),
            "--z-q-edges",  _wbin_f64("zq.bin", z_q_edges),
            "--z-n-edges",  _wbin_f64("zn.bin", z_n_edges),
            "--k-max", str(int(k_max)),
            "--output", out_dir,
        ]
        if quiet: cmd.append("--quiet")
        if weights_neigh is not None:
            cmd.extend(["--weights-neigh", _wbin_f64("w.bin", weights_neigh)])
        if region_labels_query is not None and n_regions > 0:
            cmd.extend([
                "--region-labels",
                _wbin_i64("regions.bin", region_labels_query),
                "--n-regions", str(int(n_regions)),
            ])
        if same_catalog:
            cmd.extend([
                "--self-exclude",
                "--query-targetid", _wbin_i64("qtid.bin", query_targetid),
                "--neigh-targetid", _wbin_i64("ntid.bin", neigh_targetid),
            ])
        if diagonal_only:
            cmd.append("--diagonal-only")

        try:
            res = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"angular-knn-cdf failed (exit {e.returncode}):\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from None

        meta_path = os.path.join(out_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["n_theta"] == n_theta
        assert meta["n_z_q"] == n_z_q
        assert meta["n_z_n"] == n_z_n
        h_max = max(int(k_max), 1)
        is_diagonal = bool(meta.get("is_diagonal", False))

        has_higher_moments = bool(meta.get("has_higher_moments", False))

        def _maybe_read(name, dtype, shape):
            """Read an optional binary if present; return None otherwise.
            Used for higher-moment cubes that older meta.json files
            won't advertise."""
            p = os.path.join(out_dir, name)
            if not os.path.exists(p):
                return None
            return np.fromfile(p, dtype=dtype).reshape(shape)

        if is_diagonal:
            # Cube layouts collapse the (z_q, z_n) plane to single n_z.
            n_z = n_z_q
            cube_shape = (n_theta, n_z)
            H_geq_k = np.fromfile(os.path.join(out_dir, "H_geq_k.bin"),
                                    dtype="<i8").reshape(n_theta, n_z, h_max)
            sum_n = np.fromfile(os.path.join(out_dir, "sum_n.bin"),
                                  dtype="<f8").reshape(cube_shape)
            sum_n2 = np.fromfile(os.path.join(out_dir, "sum_n2.bin"),
                                   dtype="<f8").reshape(cube_shape)
        else:
            cube_shape = (n_theta, n_z_q, n_z_n)
            H_geq_k = np.fromfile(os.path.join(out_dir, "H_geq_k.bin"),
                                    dtype="<i8").reshape(
                n_theta, n_z_q, n_z_n, h_max
            )
            sum_n = np.fromfile(os.path.join(out_dir, "sum_n.bin"),
                                  dtype="<f8").reshape(cube_shape)
            sum_n2 = np.fromfile(os.path.join(out_dir, "sum_n2.bin"),
                                   dtype="<f8").reshape(cube_shape)
        sum_n3 = _maybe_read("sum_n3.bin", "<f8", cube_shape) \
            if has_higher_moments else None
        sum_n4 = _maybe_read("sum_n4.bin", "<f8", cube_shape) \
            if has_higher_moments else None
        N_q = np.fromfile(os.path.join(out_dir, "N_q.bin"), dtype="<i8")

        H_geq_k_per_region = sum_n_per_region = None
        sum_n2_per_region = N_q_per_region = None
        sum_n3_per_region = sum_n4_per_region = None
        if meta.get("has_per_region", False):
            if is_diagonal:
                pr_cube_shape = (n_theta, n_z_q, n_regions)
                H_geq_k_per_region = np.fromfile(
                    os.path.join(out_dir, "H_geq_k_per_region.bin"),
                    dtype="<i8",
                ).reshape(n_theta, n_z_q, h_max, n_regions)
            else:
                pr_cube_shape = (n_theta, n_z_q, n_z_n, n_regions)
                H_geq_k_per_region = np.fromfile(
                    os.path.join(out_dir, "H_geq_k_per_region.bin"),
                    dtype="<i8",
                ).reshape(n_theta, n_z_q, n_z_n, h_max, n_regions)
            sum_n_per_region = np.fromfile(
                os.path.join(out_dir, "sum_n_per_region.bin"),
                dtype="<f8",
            ).reshape(pr_cube_shape)
            sum_n2_per_region = np.fromfile(
                os.path.join(out_dir, "sum_n2_per_region.bin"),
                dtype="<f8",
            ).reshape(pr_cube_shape)
            if has_higher_moments:
                sum_n3_per_region = _maybe_read(
                    "sum_n3_per_region.bin", "<f8", pr_cube_shape)
                sum_n4_per_region = _maybe_read(
                    "sum_n4_per_region.bin", "<f8", pr_cube_shape)
            N_q_per_region = np.fromfile(
                os.path.join(out_dir, "N_q_per_region.bin"), dtype="<i8",
            ).reshape(n_z_q, n_regions)

        return KnnCdfResult(
            H_geq_k=H_geq_k,
            sum_n=sum_n,
            sum_n2=sum_n2,
            N_q=N_q,
            theta_radii_rad=np.asarray(theta_radii_rad, dtype=np.float64),
            z_q_edges=np.asarray(z_q_edges, dtype=np.float64),
            z_n_edges=np.asarray(z_n_edges, dtype=np.float64),
            flavor=flavor,
            backend_used="cascade",
            area_per_cap=2.0 * np.pi * (
                1.0 - np.cos(np.asarray(theta_radii_rad))),
            H_geq_k_per_region=H_geq_k_per_region,
            sum_n_per_region=sum_n_per_region,
            sum_n2_per_region=sum_n2_per_region,
            N_q_per_region=N_q_per_region,
            is_diagonal=is_diagonal,
            sum_n3=sum_n3,
            sum_n4=sum_n4,
            sum_n3_per_region=sum_n3_per_region,
            sum_n4_per_region=sum_n4_per_region,
        )
    finally:
        if cleanup and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
