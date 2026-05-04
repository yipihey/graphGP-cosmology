"""Python wrapper for the morton_cascade Rust crate.

morton_cascade builds a sparse Morton-ordered dyadic cell hierarchy
in O(N log N) and produces a wide family of spatial statistics
(counts-in-cells, sigma^2(R), Schur residuals, Landy-Szalay xi(r),
field moments and PDF, axis-aligned anisotropy, per-particle
gradients, multi-run shift-bootstrap aggregation) at every dyadic
scale in one pass. The Rust binary is built via ``cargo build
--release`` from ``morton_cascade/`` at the repository root.

This wrapper writes numpy point arrays as binary little-endian f64
``[N, D]`` files (the format the CLI expects), invokes the
appropriate subcommand, and parses the CSV output back into numpy
records. The first call builds the binary if it is not already
compiled.

Top-level entry points:

  cascade(positions, box_size, dim=None, periodic=True)
      -> per-level mean / variance / dvar / sigma^2(R) (no randoms)

  xi_landy_szalay(positions, randoms, box_size, dim=None,
                   periodic=False, weights_data=None,
                   weights_randoms=None)
      -> per-shell DD / RR / DR / xi_LS (Landy-Szalay)

  field_stats(positions, randoms, box_size, dim=None,
                w_r_min=0.0, hist_bins=50, hist_log_min=-3.0,
                hist_log_max=3.0)
      -> per-level moments and PDF of delta(c) = W_d / (alpha W_r) - 1

  pmf_windows(positions, box_size, dim=None,
                points_per_decade=5.0, side_min=1, side_max=None)
      -> P_N(V) at log-spaced cube-window volumes

  anisotropy(positions, box_size, periodic=True)
      -> per-level Haar-wavelet anisotropy moments (3D only)

Each entry point returns a numpy structured array (record array) with
named columns matching the CSV output.

Why the wrapper:
  - the cascade is much faster than NaMaster / Corrfunc on the same
    estimators (~30x on P(k) monopole, ~150x on bispectrum-equivalent
    according to the upstream README);
  - one cascade pass produces every dyadic scale of every observable
    listed above -- well-suited to MCMC inner loops;
  - native two-catalog (data + randoms) survey-geometry handling
    bypasses the sandbox dependency on healpy / pymaster for window
    functions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATE_DIR = os.path.join(REPO_ROOT, "morton_cascade")
BIN_PATH = os.path.join(CRATE_DIR, "target", "release", "morton-cascade")


def _ensure_binary():
    """Compile the morton-cascade CLI if it isn't built yet."""
    if os.path.exists(BIN_PATH):
        return BIN_PATH
    if not os.path.isdir(CRATE_DIR):
        raise FileNotFoundError(
            f"morton_cascade crate not found at {CRATE_DIR}. The crate "
            "should live alongside twopt_density at the repo root."
        )
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            "cargo not on PATH; cannot build morton_cascade. Install "
            "rustup, then `cargo build --release` in morton_cascade/."
        )
    res = subprocess.run(
        [cargo, "build", "--release", "--bin", "morton-cascade"],
        cwd=CRATE_DIR, capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            "cargo build failed:\n" + res.stdout + "\n" + res.stderr,
        )
    return BIN_PATH


def _write_points_bin(positions: np.ndarray, path: str):
    """Write ``(N, D)`` positions to ``path`` as binary f64 little-endian."""
    arr = np.ascontiguousarray(positions, dtype="<f8")
    if arr.ndim != 2:
        raise ValueError("positions must be (N, D)")
    arr.tofile(path)


def _read_csv(path: str) -> np.ndarray:
    """Read a cascade CSV output as a structured numpy array."""
    return np.genfromtxt(path, delimiter=",", names=True,
                            dtype=None, encoding="utf-8")


def _common_args(input_path, output_dir, dim, box_size,
                  periodic=True, packed=False, quiet=True,
                  s_subshift=1):
    args = ["-i", input_path, "-o", output_dir,
              "-d", str(dim), "-L", f"{box_size:g}",
              "-s", str(s_subshift)]
    if not periodic:
        args.append("--non-periodic")
    if packed:
        args.append("--packed")
    if quiet:
        args.append("-q")
    return args


def _run(subcommand, args, capture_stdout: bool = False):
    bin_path = _ensure_binary()
    cmd = [bin_path, subcommand, *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"morton-cascade {subcommand} failed:\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return res.stdout if capture_stdout else None


def cascade(
    positions: np.ndarray,
    box_size: float,
    dim: Optional[int] = None,
    periodic: bool = True,
) -> np.ndarray:
    """Per-level mean / var / dvar / sigma^2(R) cascade.

    Parameters
    ----------
    positions : (N, D) array of point coordinates inside ``[0, box_size]^D``.
    box_size : edge length of the bounding cube.
    dim : spatial dimension; inferred from positions if not given.
    periodic : True for a periodic box, False for an isolated survey.

    Returns
    -------
    structured numpy array with columns
    ``level, R_tree, n_cells, mean, var, dvar, sigma2_field``.
    """
    pos = np.asarray(positions)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "pts.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_path)
        _run("cascade", _common_args(in_path, out_dir, D, box_size,
                                          periodic=periodic))
        return _read_csv(os.path.join(out_dir, "level_stats.csv"))


def xi_landy_szalay(
    positions: np.ndarray, randoms: np.ndarray,
    box_size: float,
    dim: Optional[int] = None,
    periodic: bool = False,
    weights_data: Optional[np.ndarray] = None,
    weights_randoms: Optional[np.ndarray] = None,
    crossover_threshold: Optional[int] = None,
) -> np.ndarray:
    """Two-catalog Landy-Szalay xi(r) via the bit-vector pair cascade.

    Returns a structured array with columns:
        level, cell_side_trimmed, cell_side_phys, r_inner_phys, r_outer_phys,
        dd, rr, dr, xi_ls, cumulative_dd, cumulative_rr, cumulative_dr.
    """
    pos = np.asarray(positions); ran = np.asarray(randoms)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_d = os.path.join(tmp, "data.bin")
        in_r = os.path.join(tmp, "rand.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_d)
        _write_points_bin(ran, in_r)
        args = _common_args(in_d, out_dir, D, box_size, periodic=periodic)
        args += ["--randoms", in_r]
        if weights_data is not None:
            wd_path = os.path.join(tmp, "wd.bin")
            np.ascontiguousarray(weights_data, dtype="<f8").tofile(wd_path)
            args += ["--weights-data", wd_path]
        if weights_randoms is not None:
            wr_path = os.path.join(tmp, "wr.bin")
            np.ascontiguousarray(weights_randoms, dtype="<f8").tofile(wr_path)
            args += ["--weights-randoms", wr_path]
        if crossover_threshold is not None:
            args += ["--crossover-threshold", str(int(crossover_threshold))]
        _run("xi", args)
        return _read_csv(os.path.join(out_dir, "xi_landy_szalay.csv"))


def field_stats(
    positions: np.ndarray, randoms: np.ndarray,
    box_size: float,
    dim: Optional[int] = None,
    periodic: bool = False,
    w_r_min: float = 0.0,
    hist_bins: int = 50,
    hist_log_min: float = -3.0,
    hist_log_max: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Density-field cell-count statistics: per-level moments and PDF
    of ``delta(c) = W_d / (alpha W_r) - 1`` weighted by ``W_r`` (effective
    volume). Returns ``(moments, pdf)``."""
    pos = np.asarray(positions); ran = np.asarray(randoms)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_d = os.path.join(tmp, "data.bin")
        in_r = os.path.join(tmp, "rand.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_d)
        _write_points_bin(ran, in_r)
        args = _common_args(in_d, out_dir, D, box_size, periodic=periodic)
        args += ["--randoms", in_r,
                  "--w-r-min", f"{w_r_min:g}",
                  "--hist-bins", str(hist_bins),
                  "--hist-log-min", f"{hist_log_min:g}",
                  "--hist-log-max", f"{hist_log_max:g}"]
        _run("field-stats", args)
        moments = _read_csv(os.path.join(out_dir, "field_moments.csv"))
        pdf = _read_csv(os.path.join(out_dir, "field_pdf.csv"))
        return moments, pdf


def pmf_windows(
    positions: np.ndarray, box_size: float,
    dim: Optional[int] = None, periodic: bool = True,
    points_per_decade: float = 5.0,
    side_min: int = 1, side_max: Optional[int] = None,
) -> np.ndarray:
    """``P_N(V)`` at log-spaced cube-window volumes."""
    pos = np.asarray(positions)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "pts.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_path)
        args = _common_args(in_path, out_dir, D, box_size,
                              periodic=periodic)
        args += ["--points-per-decade", f"{points_per_decade:g}",
                  "--side-min", str(side_min)]
        if side_max is not None:
            args += ["--side-max", str(side_max)]
        _run("pmf-windows", args)
        return _read_csv(os.path.join(out_dir, "pmf_windows.csv"))


def anisotropy(
    positions: np.ndarray, randoms: np.ndarray,
    box_size: float, periodic: bool = True,
    w_r_min: float = 0.0,
) -> np.ndarray:
    """Per-level axis-aligned cell-wavelet anisotropy moments
    (3D only). Like ``field_stats``, ``anisotropy`` is a
    *footprint-aware* observable: random points encode the survey
    selection function and their counts weight the wavelet moments
    by effective volume.

    Returns ``field_anisotropy.csv`` as a structured array."""
    pos = np.asarray(positions); ran = np.asarray(randoms)
    if pos.shape[1] != 3:
        raise ValueError("anisotropy is 3D only; got dim={}".format(
            pos.shape[1]))
    with tempfile.TemporaryDirectory() as tmp:
        in_d = os.path.join(tmp, "data.bin")
        in_r = os.path.join(tmp, "rand.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_d)
        _write_points_bin(ran, in_r)
        args = _common_args(in_d, out_dir, 3, box_size,
                              periodic=periodic)
        args += ["--randoms", in_r,
                  "--w-r-min", f"{w_r_min:g}"]
        _run("anisotropy", args)
        return _read_csv(os.path.join(out_dir, "field_anisotropy.csv"))


def gradient(
    positions: np.ndarray, box_size: float,
    target_level: int, dim: Optional[int] = None, periodic: bool = True,
) -> np.ndarray:
    """Per-particle pair-count gradient at ``target_level``: returns
    ``N`` integers ``grad[i] = (number of points sharing a cell with
    point i at that dyadic level) - 1``. Useful as a per-galaxy
    density-environment estimator."""
    pos = np.asarray(positions)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "pts.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_path)
        args = _common_args(in_path, out_dir, D, box_size,
                              periodic=periodic)
        args += ["--target-level", str(int(target_level))]
        _run("gradient", args)
        return _read_csv(os.path.join(out_dir, "gradient.csv"))
