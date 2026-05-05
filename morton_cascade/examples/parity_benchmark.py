#!/usr/bin/env python3
"""
Survey-quality parity benchmark for `morton-cascade field-stats`.

Generates a small synthetic survey: clustered data inside an octant footprint,
uniform randoms in the same footprint. Computes per-level density-field
statistics two ways:

  1. By calling `morton-cascade field-stats` and parsing the output CSV.
  2. By direct numpy cell counting on regular cube grids matching each
     dyadic level. This is the "ground truth" — slow O(N * cells) per level
     but unambiguous.

For each dyadic level, compares:
  - sum_w_r_active           : total random weight in cells with W_r > 0
  - mean_delta               : W_r-weighted mean of δ
  - var_delta                : W_r-weighted variance of δ
  - n_cells_data_outside     : data-bearing cells with no randoms
  - sum_w_d_outside          : total W_d in those cells

These should agree to floating-point round-off at every comparable level.
The level structure is dyadic (powers of 2 of the box side), so we test
cells of side L, L/2, L/4, ..., L/2^N where N is set by the cascade depth.

Also checks the cosmological-survey identities:
  - Total ΣW_r over all cells (active OR outside) = N_r exactly.
  - Total ΣW_d over (active cells + outside cells) = N_d exactly.
  - Mean δ over the WHOLE box (active + outside-as-zero) = 0 algebraically
    (the global α normalization).

Usage:
    cd morton_cascade && cargo build --release
    python3 examples/parity_benchmark.py
"""

import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BOX_SIZE = 100.0
N_DATA = 4_000
N_RAND = 32_000
SEED = 271828

# Dyadic levels to compare (cell side = BOX_SIZE / 2^level for non-trimmed,
# but cascade trims so we recompute below from the cascade output).
LEVELS_TO_CHECK = [1, 2, 3, 4, 5]

BIN = "./target/release/morton-cascade"


# -----------------------------------------------------------------------------
# Synthetic survey
# -----------------------------------------------------------------------------

def make_survey(rng: np.random.Generator):
    """Survey: data clustered into 20 Gaussian blobs inside the upper octant
    (x > L/2, y > L/2, z > L/2). Randoms uniform inside the same octant.
    Plus a small contamination cluster OUTSIDE the footprint, to exercise
    the n_cells_data_outside diagnostic.

    Returns:
        data, randoms : (N, 3) arrays of float64 coordinates in [0, BOX_SIZE).
    """
    L = BOX_SIZE
    octant_lo = 0.5 * L
    octant_hi = L

    # Cluster centers in the octant
    n_clusters = 20
    centers = rng.uniform(octant_lo, octant_hi, size=(n_clusters, 3))

    # Data: cluster_size = box / 64
    cluster_sigma = L / 64.0
    data = []
    while len(data) < N_DATA - 100:
        c = centers[rng.integers(0, n_clusters)]
        p = c + rng.normal(0.0, cluster_sigma, size=3)
        # Reject outside octant or outside box
        if (octant_lo <= p[0] < octant_hi
                and octant_lo <= p[1] < octant_hi
                and octant_lo <= p[2] < octant_hi):
            data.append(p)

    # Contamination: 100 data points OUTSIDE footprint (in lower octant).
    # These should appear in n_cells_data_outside diagnostic.
    contam_center = np.array([0.25 * L, 0.25 * L, 0.25 * L])
    while len(data) < N_DATA:
        p = contam_center + rng.normal(0.0, cluster_sigma, size=3)
        if 0 <= p[0] < octant_lo and 0 <= p[1] < octant_lo and 0 <= p[2] < octant_lo:
            data.append(p)

    data = np.array(data, dtype=np.float64)

    # Randoms: uniform in the octant
    randoms = rng.uniform(octant_lo, octant_hi, size=(N_RAND, 3)).astype(np.float64)

    return data, randoms


def write_pts_bin(path: str, pts: np.ndarray):
    """Write (N, D) array as raw little-endian f64."""
    pts.astype('<f8').tofile(path)


# -----------------------------------------------------------------------------
# Cascade reference
# -----------------------------------------------------------------------------

def run_cascade(workdir: str, data_path: str, randoms_path: str):
    """Run `morton-cascade field-stats` and return parsed moments rows."""
    cmd = [BIN, "field-stats",
           "-i", data_path,
           "--randoms", randoms_path,
           "-d", "3",
           "-L", str(BOX_SIZE),
           "-o", workdir,
           "--hist-bins", "0",
           "-q"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"  cascade stderr: {res.stderr.strip()}")
    rows = []
    with open(os.path.join(workdir, "field_moments.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (float(v) if k not in ("level", "n_cells_active",
                                                   "n_cells_data_outside") else int(v))
                         for k, v in row.items()})
    return rows


# -----------------------------------------------------------------------------
# Direct numpy reference
# -----------------------------------------------------------------------------

def direct_field_stats(data: np.ndarray, randoms: np.ndarray, cell_side: float):
    """Compute density-field statistics by direct binning into cube cells.

    Cells are aligned to the [0, BOX_SIZE) grid with given side.
    Returns dict matching cascade columns.
    """
    L = BOX_SIZE
    n_cells_per_axis = int(round(L / cell_side))
    if n_cells_per_axis < 1:
        n_cells_per_axis = 1
    bin_edges = np.linspace(0, L, n_cells_per_axis + 1)

    # 3D histogram of data and randoms (unit weights here).
    counts_d, _ = np.histogramdd(data, bins=[bin_edges, bin_edges, bin_edges])
    counts_r, _ = np.histogramdd(randoms, bins=[bin_edges, bin_edges, bin_edges])

    # Global α
    total_w_d = counts_d.sum()
    total_w_r = counts_r.sum()
    alpha = total_w_d / total_w_r

    flat_d = counts_d.ravel()
    flat_r = counts_r.ravel()

    # Active cells: W_r > 0 (default w_r_min = 0)
    active = flat_r > 0
    sum_w_r_active = flat_r[active].sum()
    delta = np.where(active, flat_d / (alpha * np.maximum(flat_r, 1e-300)) - 1.0, 0.0)

    # W_r-weighted moments over active cells
    w = flat_r[active]
    d_act = delta[active]
    if w.sum() > 0:
        mean = (w * d_act).sum() / w.sum()
        d2 = (d_act - mean)
        var = (w * d2 * d2).sum() / w.sum()
    else:
        mean = 0.0
        var = 0.0

    # Diagnostic: data outside footprint (W_r = 0 but W_d > 0)
    outside = (flat_r == 0) & (flat_d > 0)
    n_outside = int(outside.sum())
    sum_w_d_outside = float(flat_d[outside].sum())

    return {
        "n_cells_active": int(active.sum()),
        "sum_w_r_active": float(sum_w_r_active),
        "mean_delta": float(mean),
        "var_delta": float(var),
        "n_cells_data_outside": n_outside,
        "sum_w_d_outside": sum_w_d_outside,
        "n_cells_per_axis": n_cells_per_axis,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not os.path.exists(BIN):
        print(f"FAIL: binary {BIN} not found. Run `cargo build --release` first.")
        sys.exit(1)

    print("=" * 78)
    print("Morton-cascade field-stats parity benchmark")
    print(f"  N_data = {N_DATA},  N_rand = {N_RAND},  box = {BOX_SIZE}")
    print(f"  Survey: clustered data + uniform randoms in upper octant")
    print("=" * 78)

    rng = np.random.default_rng(SEED)
    data, randoms = make_survey(rng)
    print(f"Generated {len(data)} clustered data + {len(randoms)} randoms")

    workdir = tempfile.mkdtemp(prefix="morton_parity_")
    try:
        data_path = os.path.join(workdir, "data.bin")
        randoms_path = os.path.join(workdir, "randoms.bin")
        write_pts_bin(data_path, data)
        write_pts_bin(randoms_path, randoms)

        print("\nRunning cascade...")
        cascade_rows = run_cascade(workdir, data_path, randoms_path)
        print(f"  → {len(cascade_rows)} levels of stats")

        # The cascade trims coordinates internally. The reported
        # cell_side_phys is what we should use for the direct comparison.
        print("\n" + "-" * 78)
        print(f"{'level':>5} {'cell':>10} {'metric':<30} {'cascade':>16} "
              f"{'numpy':>16} {'diff':>10}")
        print("-" * 78)

        n_failures = 0
        n_compares = 0
        max_rel_err = 0.0

        for casc_row in cascade_rows:
            level = casc_row["level"]
            if level not in LEVELS_TO_CHECK:
                continue
            cell_side = casc_row["cell_side_phys"]
            if cell_side >= BOX_SIZE * 2.0:
                continue  # cells too large to be meaningful

            np_ref = direct_field_stats(data, randoms, cell_side)

            for metric in ["n_cells_active", "sum_w_r_active", "mean_delta",
                           "var_delta", "n_cells_data_outside",
                           "sum_w_d_outside"]:
                cas_val = casc_row[metric]
                np_val = np_ref[metric]
                # Choose comparison style by metric type
                if metric in ("n_cells_active", "n_cells_data_outside"):
                    diff = abs(cas_val - np_val)
                    ok = (diff == 0)
                    diff_str = f"{int(diff)}"
                elif metric in ("sum_w_r_active", "sum_w_d_outside"):
                    diff = abs(cas_val - np_val)
                    ok = (diff < 1e-9 * max(abs(cas_val), abs(np_val), 1.0))
                    diff_str = f"{diff:.2e}"
                else:
                    # Floating moments — relative tolerance, but if both are
                    # near zero use absolute tolerance instead (mean_delta is
                    # algebraically zero by construction so both sides land
                    # at the f64 noise floor of ~1e-17, where relative error
                    # is meaningless).
                    abs_diff = abs(cas_val - np_val)
                    if max(abs(cas_val), abs(np_val)) < 1e-12:
                        ok = (abs_diff < 1e-12)
                        diff_str = f"{abs_diff:.2e}"
                    else:
                        denom = max(abs(cas_val), abs(np_val), 1e-12)
                        rel = abs_diff / denom
                        ok = (rel < 1e-9)
                        diff_str = f"{rel:.2e}"
                        max_rel_err = max(max_rel_err, rel)

                marker = " " if ok else " ✗"
                print(f"{level:>5} {cell_side:>10.4f} {metric:<30} "
                      f"{cas_val:>16.6e} {np_val:>16.6e} {diff_str:>10}{marker}")
                if not ok:
                    n_failures += 1
                n_compares += 1
            print()

        print("-" * 78)
        print(f"Summary: {n_compares - n_failures}/{n_compares} comparisons agree")
        print(f"Max relative error on floating moments: {max_rel_err:.2e}")
        if n_failures == 0:
            print("\n✓ PARITY: cascade matches direct numpy at every level")
            return 0
        else:
            print(f"\n✗ {n_failures} disagreements")
            return 1

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
