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


def pairs(
    positions: np.ndarray, box_size: float,
    dim: Optional[int] = None, periodic: bool = True,
    max_depth: Optional[int] = None,
    crossover_threshold: Optional[int] = None,
) -> np.ndarray:
    """Per-shell DD pair counts at every dyadic scale, no randoms
    needed. Returns a structured array with columns
    ``level, cell_side_trimmed, cell_side_phys, r_inner_phys,
    r_outer_phys, n_pairs, cumulative_pairs``.

    The headline observable for combining with our
    ``twopt_density.analytic_rr`` machinery: the cascade gives DD in
    one ``O(N log N)`` pass and the analytic-window RR/DR fill in
    the random side without instantiating an MC random catalogue.
    """
    pos = np.asarray(positions)
    D = int(dim) if dim is not None else pos.shape[1]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "pts.bin")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_bin(pos, in_path)
        args = _common_args(in_path, out_dir, D, box_size,
                              periodic=periodic)
        if max_depth is not None:
            args += ["--max-depth", str(int(max_depth))]
        if crossover_threshold is not None:
            args += ["--crossover-threshold",
                       str(int(crossover_threshold))]
        _run("pairs", args)
        return _read_csv(os.path.join(out_dir, "pairs.csv"))


def xi_landy_szalay_from_window(
    positions: np.ndarray,
    ra_deg: np.ndarray, dec_deg: np.ndarray, z_data: np.ndarray,
    sel_map: np.ndarray, nside: int, cosmo,
    box_size: Optional[float] = None,
    n_random_factor: int = 10,
    rng_seed: int = 0,
    weights_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Landy-Szalay xi(r) per dyadic shell, with the random catalogue
    generated **internally** from the survey window (mask + n(z)). The
    user never instantiates randoms.

    Pipeline:
      1. Sample ``n_random_factor * N_d`` random points in (RA, Dec, z)
         from the separable window ``W(Omega, z) = sel_map(Omega)
         n(z)`` via ``twopt_density.quaia.make_random_from_selection_function``.
      2. Convert to comoving xyz under ``cosmo`` and shift both data
         and random to non-negative coords (cKDTree requirement).
      3. Run the morton_cascade ``xi`` subcommand. Returns full LS
         output at every dyadic shell.

    For surveys whose selection function is angular-only at NSIDE
    (Quaia, DESI footprint), this is the natural way to combine the
    cascade's O(N log N) pair-count machinery with our analytic-window
    ``2D randoms``: the synthesised random encodes the same ``M(Omega)
    n(z)`` factorisation that ``analytic_rr`` uses, so LS is consistent
    with the existing wp(rp) / sigma^2(R) pipelines.

    Parameters
    ----------
    positions : (N_d, 3) data comoving positions [Mpc/h].
    ra_deg, dec_deg, z_data : data sky coordinates and redshifts;
        used to build the random-side n(z) and to anchor the survey
        footprint sampling.
    sel_map : (NPIX,) HEALPix angular completeness map (Storey-Fisher
        style), in [0, 1].
    nside : HEALPix NSIDE of ``sel_map``.
    cosmo : ``DistanceCosmo`` for comoving conversions.
    box_size : optional bounding-box side (Mpc/h); inferred from the
        union of (data, random) positions if not given.

    Returns
    -------
    structured array as ``xi_landy_szalay`` (per-shell DD/RR/DR/xi_LS).
    """
    from .distance import radec_z_to_cartesian
    from .quaia import make_random_from_selection_function

    pos_d = np.asarray(positions, dtype=np.float64)
    if pos_d.shape[1] != 3:
        raise ValueError("positions must be (N_d, 3) comoving xyz")
    n_d = len(pos_d)
    n_r = max(int(n_random_factor) * n_d, 1)

    rng = np.random.default_rng(rng_seed)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=sel_map, n_random=n_r, z_data=np.asarray(z_data),
        nside=nside, rng=rng,
    )
    pos_r = radec_z_to_cartesian(ra_r, dec_r, z_r, cosmo)

    all_xyz = np.vstack([pos_d, pos_r])
    shift = -all_xyz.min(axis=0) + 100.0
    pos_d_s = pos_d + shift
    pos_r_s = pos_r + shift
    if box_size is None:
        box_size = float(np.max(np.vstack([pos_d_s, pos_r_s])) + 100.0)

    return xi_landy_szalay(
        pos_d_s, pos_r_s, box_size=box_size, dim=3, periodic=False,
        weights_data=weights_data,
    )


def analytic_window_grid(
    sel_map: np.ndarray, nside: int,
    z_data: np.ndarray, cosmo,
    n_grid: int = 64,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    chi_pad: float = 100.0,
    z_kde_bandwidth: float = 0.05,
):
    """Deterministic 'analytic random' on a Cartesian grid: tile the
    bounding cube of the survey at resolution ``n_grid^3`` and assign
    each cell centre a weight proportional to the integrated survey
    window ``W(Omega, z) = sel_map(Omega) * n(z)`` over that cell.

    Returns ``(positions, weights, box_size)`` where ``positions`` is
    ``(n_grid^3, 3)`` Cartesian (Mpc/h, shifted into the non-negative
    octant) and ``weights[i] = M(Omega_i) * n(z_i) / (chi_i^2
    dchi/dz_i) * V_cell`` (proportional to the expected count of
    randoms in cell ``i`` for any uniformly-window-sampled random
    catalogue). Cells outside the survey radial range have weight 0.

    Used by ``xi_landy_szalay_analytic_RR`` as the cascade's
    "random catalogue": running cascade ``xi`` with these
    weights returns LS xi(r) per dyadic shell with **MC-noise-free
    analytic RR**.

    Parameters
    ----------
    sel_map : (NPIX,) HEALPix angular completeness in [0, 1].
    nside : HEALPix NSIDE of ``sel_map``.
    z_data : observed redshifts (used to build the n(z) PDF).
    cosmo : ``DistanceCosmo``.
    n_grid : grid resolution per axis. Cell side = box_side / n_grid.
        Defaults to 64; for the cascade to faithfully sample dyadic
        levels down to L, ``n_grid`` should be ``>= 2^L``.
    z_min, z_max : radial bounds; default to the data redshift range.
    chi_pad : padding in Mpc/h around chi(z_max).
    z_kde_bandwidth : Gaussian KDE bandwidth in z, used to smooth the
        n(z) histogram.
    """
    import healpy as hp
    import jax.numpy as jnp

    from .distance import (
        cartesian_to_radec_z, comoving_distance,
    )

    if z_min is None:
        z_min = float(np.min(z_data))
    if z_max is None:
        z_max = float(np.max(z_data))

    chi_at_z = np.asarray(comoving_distance(
        jnp.asarray([z_min, z_max], dtype=jnp.float64), cosmo))
    chi_min, chi_max = float(chi_at_z[0]), float(chi_at_z[1])

    half = chi_max + chi_pad
    edges = np.linspace(-half, half, n_grid + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    cell_side = float(edges[1] - edges[0])
    dV = cell_side ** 3

    XX, YY, ZZ = np.meshgrid(centres, centres, centres, indexing="ij")
    pos = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    chi = np.linalg.norm(pos, axis=1)
    in_radial = (chi >= chi_min) & (chi <= chi_max)

    # smooth n(z) PDF
    z_arr = np.asarray(z_data)
    z_pdf_grid = np.linspace(z_min - 0.05, z_max + 0.05, 200)
    delta_z = z_pdf_grid[:, None] - z_arr[None, :]
    nz_pdf = np.exp(-0.5 * (delta_z / z_kde_bandwidth) ** 2).sum(axis=1)
    nz_pdf = nz_pdf / max(np.trapezoid(nz_pdf, z_pdf_grid), 1e-30)

    # convert all grid points to (ra, dec, z); weight = M * n(z) / chi^2 dchi/dz
    ra_g, dec_g, z_g = cartesian_to_radec_z(
        jnp.asarray(pos, dtype=jnp.float64), cosmo)
    ra_g = np.asarray(ra_g); dec_g = np.asarray(dec_g)
    z_g = np.asarray(z_g)
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_g), np.deg2rad(ra_g))
    M_at = np.asarray(sel_map)[pix]
    n_at = np.interp(z_g, z_pdf_grid, nz_pdf, left=0.0, right=0.0)

    # dchi/dz at z_g via numerical derivative on a dense grid
    z_dense = np.linspace(0.0, z_max + 0.5, 4000)
    chi_dense = np.asarray(comoving_distance(
        jnp.asarray(z_dense, dtype=jnp.float64), cosmo))
    dchi_dz_dense = np.gradient(chi_dense, z_dense)
    dchi_dz_g = np.interp(z_g, z_dense, dchi_dz_dense)

    chi_safe = np.where(chi > 0, chi, 1.0)
    weights = np.where(in_radial,
                          M_at * n_at / (chi_safe ** 2 * np.maximum(
                              dchi_dz_g, 1e-12)) * dV,
                          0.0)
    weights = np.where(weights > 0, weights, 0.0)

    # shift positions into the non-negative octant for cKDTree
    shift = -pos.min(axis=0) + 100.0
    pos_s = pos + shift
    box_size = float(np.max(pos_s) + 100.0)

    return pos_s, weights, box_size


def xi_landy_szalay_analytic_RR(
    positions: np.ndarray,
    sel_map: np.ndarray, nside: int,
    z_data: np.ndarray, cosmo,
    n_grid: int = 64,
    weights_data: Optional[np.ndarray] = None,
    weight_threshold: float = 1e-12,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
):
    """Cascade Landy-Szalay xi(r) with **deterministic analytic RR**:
    no MC random catalogue.

    The "random catalogue" is replaced by a regular Cartesian grid
    of ``n_grid^3`` cell centres, each carrying a weight proportional
    to the survey window ``M(Omega) n(z) / chi^2 dchi/dz`` integrated
    over the cell. Running cascade ``xi`` with these weighted-grid
    randoms produces the analytic RR with zero Monte-Carlo noise --
    only Riemann-sum discretisation set by ``n_grid``.

    For the cascade to resolve dyadic level L, ``n_grid >= 2^L``. The
    default ``n_grid = 64`` resolves ``L <= 6`` cleanly which on Quaia
    covers the BAO range (cell side > 125 Mpc/h on an 8 Gpc/h box).

    Parameters
    ----------
    positions : (N_d, 3) data comoving xyz.
    sel_map, nside : HEALPix angular completeness.
    z_data : observed redshifts (for n(z)).
    cosmo : ``DistanceCosmo``.
    n_grid : grid resolution per axis (default 64).
    weights_data : optional per-data weights.
    weight_threshold : drop grid cells whose weight is below this
        fraction of the maximum (typically negligible at the survey
        edge / outside the footprint).

    Returns
    -------
    structured array as ``xi_landy_szalay``.
    """
    pos_d = np.asarray(positions, dtype=np.float64)
    if pos_d.shape[1] != 3:
        raise ValueError("positions must be (N_d, 3)")
    grid_pos, grid_w, grid_box = analytic_window_grid(
        sel_map, nside, z_data, cosmo, n_grid=n_grid,
        z_min=z_min, z_max=z_max,
    )

    # shift the data into the same coordinate frame as the grid
    # (positions are typically already in the survey frame; we shift to
    # match the grid box)
    shift = -np.minimum(pos_d.min(axis=0), 0.0) + 100.0
    pos_d_s = pos_d + shift
    box_size = max(grid_box, float(np.max(pos_d_s) + 100.0))

    # filter grid cells with negligible weight to keep the random
    # catalogue compact (the cascade still scales O(N log N) but
    # smaller N is faster)
    keep = grid_w > weight_threshold * grid_w.max()
    grid_pos_k = grid_pos[keep]
    grid_w_k = grid_w[keep]
    # Rescale weights so ``sum(w_k) = N_kept``. The cascade normalises
    # weighted-RR by ``N_r_points (N_r_points - 1) / 2`` (count-of-points,
    # not weighted-pair-norm), so unit-mean weights make the LS xi
    # consistent with an unweighted N_kept-point uniform random in the
    # survey window. The relative shape of grid_w (which encodes the
    # angular mask + n(z)) is preserved.
    s_w = float(grid_w_k.sum())
    if s_w > 0:
        grid_w_k = grid_w_k * (len(grid_w_k) / s_w)

    return xi_landy_szalay(
        pos_d_s, grid_pos_k, box_size=box_size, dim=3, periodic=False,
        weights_data=weights_data, weights_randoms=grid_w_k,
    )


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
