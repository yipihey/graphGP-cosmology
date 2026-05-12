"""Python wrapper around the ``morton_cascade`` Rust crate's CLI.

Drives the Rust binary at
``morton_cascade/target/release/morton-cascade`` via subprocess, with
binary point files written to a tempdir and CSV results read back as
numpy arrays. The wrapper handles:

- Comoving Cartesian conversion from ``(RA, Dec, z)`` (using
  ``twopt_density.distance.DistanceCosmo``) into a [0, L)^3 cube
  centered on the data centroid (the cascade requires periodic-box
  geometry; we use ``--non-periodic`` and pad the box so all points
  fit).
- Binding ``xi`` (Landy-Szalay 3D correlation function) with
  per-point weights.
- ``pairs`` (raw per-shell pair counts).

The crate's ``xi`` produces ξ(r) at axis-aligned dyadic scales in one
pass (paper Eq. 12 in 3D Cartesian frame). For typical survey N
~10^6 this is many×× faster than per-r Corrfunc calls because the
cascade visits each cell only once per scale.

Build the binary first:

    cd morton_cascade && cargo build --release

The wrapper auto-discovers the binary at the standard path; override
via ``MORTON_CASCADE_BIN`` env var if needed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(
    REPO_ROOT, "morton_cascade/target/release/morton-cascade",
)


def _resolve_binary() -> str:
    p = os.environ.get("MORTON_CASCADE_BIN", DEFAULT_BIN)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"morton-cascade binary not found at {p!r}. Build with "
            f"`cargo build --release` in {os.path.dirname(p)}."
        )
    return p


def _write_points_binary(pts: np.ndarray, path: str) -> None:
    """Write an (N, D) f64 array as little-endian binary, C-order."""
    pts = np.ascontiguousarray(pts, dtype="<f8")
    if pts.ndim != 2:
        raise ValueError(f"expected 2D (N, D) array; got shape {pts.shape}")
    pts.tofile(path)


def radec_to_chord_box(
    ra_deg: np.ndarray, dec_deg: np.ndarray,
    box_pad_frac: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Map ``(RA, Dec)`` to unit vectors on S² and shift to a
    non-periodic ``[0, L)^3`` box of side L = 2 (1 + pad).

    The cascade then sees dyadic shells in **chord length**
    ``d = 2 sin(θ/2)``. For θ ≪ 1 rad the small-angle relation
    ``d ≈ θ`` is exact to (θ/2)²/3 — at θ = 12° ≈ 0.21 rad the
    chord-vs-arc error is ~0.07%; at θ = 30° it grows to ~0.4%. So
    angular pair counts at chord ≤ d_θ ≡ 2 sin(θ/2) are the same as
    angular pair counts at separation ≤ θ to that accuracy.

    Returns ``(pts, L)`` where ``pts`` is ``(N, 3)`` in [0, L)^3 and
    L is the box side. The cascade should be run with
    ``--non-periodic`` over this box.
    """
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    theta_polar = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg % 360.0)
    x = np.sin(theta_polar) * np.cos(phi)
    y = np.sin(theta_polar) * np.sin(phi)
    z = np.cos(theta_polar)
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    # Shift to [0, L)^3. Sphere always fits in [-1, 1]^3, so
    # L = 2 (1 + pad) and shift is uniform.
    L = 2.0 * (1.0 + box_pad_frac)
    pts = pts + (L / 2.0)
    return pts, L


def chord_to_theta_deg(chord: np.ndarray) -> np.ndarray:
    """Inverse of ``d = 2 sin(θ/2)``: given chord lengths return θ
    in degrees, clipping chord into the valid [0, 2] range first."""
    c = np.clip(np.asarray(chord, dtype=np.float64), 0.0, 2.0)
    return np.degrees(2.0 * np.arcsin(c / 2.0))


def radec_z_to_cartesian_for_cascade(
    ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray,
    fid_cosmo,
    box_pad_frac: float = 0.05,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Convert (RA, Dec, z) → centred Cartesian (x, y, z) in Mpc/h
    coordinates suitable for the cascade.

    Returns
    -------
    pts : (N, 3) float64 array
        Coordinates shifted to ``[0, L)^3`` so the cascade can be run
        with ``--non-periodic`` over a box of side L.
    L : float
        The box side length used.
    centroid : (3,) array
        The centroid before shifting (so callers can recover sky
        positions if needed).
    """
    from .distance import DistanceCosmo, comoving_distance
    import jax.numpy as jnp
    chi = np.asarray(comoving_distance(jnp.asarray(z), fid_cosmo))
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg)
    x = chi * np.sin(theta) * np.cos(phi)
    y = chi * np.sin(theta) * np.sin(phi)
    z_c = chi * np.cos(theta)
    pts = np.stack([x, y, z_c], axis=1).astype(np.float64)
    centroid = pts.mean(axis=0)
    extent = (pts.max(axis=0) - pts.min(axis=0)).max()
    L = float(extent) * (1.0 + box_pad_frac)
    pts -= pts.min(axis=0) - 0.5 * (L - (pts.max(axis=0) - pts.min(axis=0)))
    return pts, L, centroid


@dataclass
class XiResult:
    """Output of ``run_xi``.

    ``r``, ``xi``, ``DD``, ``RR``, ``DR`` are length-``n_levels``
    numpy arrays from the cascade's per-level CSV. ``r`` is the
    physical (Mpc/h) shell radius for that dyadic level.
    """

    r: np.ndarray
    xi: np.ndarray
    DD: np.ndarray
    RR: np.ndarray
    DR: np.ndarray
    n_levels: int
    box_size: float
    n_data: int
    n_random: int
    elapsed_s: float
    raw_csv_path: Optional[str] = None


def run_xi(
    pts_data: np.ndarray, pts_random: np.ndarray,
    box_size: float,
    weights_data: Optional[np.ndarray] = None,
    weights_random: Optional[np.ndarray] = None,
    crossover_threshold: Optional[int] = None,
    s_subshift: int = 1,
    periodic: bool = False,
    workdir: Optional[str] = None,
    keep_workdir: bool = False,
    quiet: bool = True,
) -> XiResult:
    """Invoke ``morton-cascade xi`` and parse the resulting ``xi.csv``.

    Parameters
    ----------
    pts_data, pts_random
        ``(N, 3)`` float64 arrays in ``[0, L)^3``. See
        ``radec_z_to_cartesian_for_cascade`` for a converter from
        survey (RA, Dec, z).
    box_size
        Side length L (Mpc/h). Must contain all points.
    weights_data, weights_random
        Optional ``(N,)`` per-point weights (e.g.,
        WEIGHT*WEIGHT_FKP).
    crossover_threshold
        Adaptive default (``max(64, max(N_d, N_r) / 64)``) when
        ``None``. Override only for benchmarking.
    s_subshift
        Cell-grid origin sub-shift averaging — increases isotropy of
        the dyadic cells. Default 1 (matches the CLI default).
    periodic
        Most surveys are non-periodic. Set ``True`` only for box-mock
        catalogs where the volume genuinely wraps.

    Returns
    -------
    XiResult
    """
    import time as _t

    bin_path = _resolve_binary()
    if pts_data.ndim != 2 or pts_data.shape[1] != 3:
        raise ValueError(f"pts_data must be (N, 3); got {pts_data.shape}")
    if pts_random.ndim != 2 or pts_random.shape[1] != 3:
        raise ValueError(f"pts_random must be (N, 3); got {pts_random.shape}")

    cleanup = workdir is None
    if cleanup:
        workdir = tempfile.mkdtemp(prefix="morton_xi_")
    os.makedirs(workdir, exist_ok=True)
    try:
        data_path = os.path.join(workdir, "data.bin")
        rand_path = os.path.join(workdir, "rand.bin")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        _write_points_binary(pts_data, data_path)
        _write_points_binary(pts_random, rand_path)

        cmd = [
            bin_path, "xi",
            "--input", data_path,
            "--randoms", rand_path,
            "--output", out_dir,
            "--dim", "3",
            "--box-size", repr(float(box_size)),
            "--subshift", str(int(s_subshift)),
        ]
        if not periodic:
            cmd.append("--non-periodic")
        if crossover_threshold is not None:
            cmd.extend(["--crossover-threshold", str(int(crossover_threshold))])
        if weights_data is not None:
            wd_path = os.path.join(workdir, "wd.bin")
            np.ascontiguousarray(weights_data, dtype="<f8").tofile(wd_path)
            cmd.extend(["--weights-data", wd_path])
        if weights_random is not None:
            wr_path = os.path.join(workdir, "wr.bin")
            np.ascontiguousarray(weights_random, dtype="<f8").tofile(wr_path)
            cmd.extend(["--weights-randoms", wr_path])
        if quiet:
            cmd.append("--quiet")

        t0 = _t.time()
        try:
            res = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"morton-cascade xi failed (exit {e.returncode}):\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            ) from None
        elapsed = _t.time() - t0

        # CLI version 0.1 writes xi_landy_szalay.csv; older
        # builds wrote xi.csv. Try both.
        csv_path = None
        for name in ("xi_landy_szalay.csv", "xi.csv"):
            cand = os.path.join(out_dir, name)
            if os.path.exists(cand):
                csv_path = cand
                break
        if csv_path is None:
            files = sorted(os.listdir(out_dir))
            raise RuntimeError(
                f"no xi CSV in {out_dir}; got files {files}\n"
                f"stdout: {res.stdout}"
            )

        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        # Best-effort column resolution — column names vary slightly
        # across CLI versions.
        names = data.dtype.names

        def _col(*candidates):
            for n in candidates:
                if n in names:
                    return data[n]
            raise KeyError(
                f"none of {candidates!r} in xi.csv columns {names!r}"
            )

        # Each level represents a shell from r_inner_phys to
        # r_outer_phys; we report the geometric-mean shell radius as
        # the per-level summary.
        r_inner = np.asarray(_col("r_inner_phys"))
        r_outer = np.asarray(_col("r_outer_phys"))
        # Geometric mean handles the level-0 case where inner==outer.
        with np.errstate(invalid="ignore"):
            r = np.where(r_inner == r_outer, r_outer,
                         np.sqrt(r_inner * r_outer))
        xi = np.asarray(_col("xi_ls", "xi"))
        DD = np.asarray(_col("dd", "DD"))
        RR = np.asarray(_col("rr", "RR"))
        DR = np.asarray(_col("dr", "DR"))

        return XiResult(
            r=r, xi=xi, DD=DD, RR=RR, DR=DR,
            n_levels=r.size, box_size=float(box_size),
            n_data=int(pts_data.shape[0]),
            n_random=int(pts_random.shape[0]),
            elapsed_s=elapsed,
            raw_csv_path=csv_path if keep_workdir else None,
        )
    finally:
        if cleanup and not keep_workdir:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
