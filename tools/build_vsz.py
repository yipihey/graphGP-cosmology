"""Veusz-based publication-quality figure pipeline.

Each panel group writes:

- ``vsz/<group>.vsz``   — Veusz document. Seeded once by the build script
  and then HAND-EDITED. Subsequent builds NEVER overwrite it; they only
  refresh the data CSVs the .vsz imports. Style/layout edits survive.
- ``vsz/<group>_<cat>.csv`` — per-catalog data, regenerated every build.
- ``vsz/_snapshots/{ISO}/`` — captured copy of every hand-editable file
  before each build, so ``tools/propagate_vsz_edits.py`` can diff.

Public API:
    sigma2_group(quaia, desi, z_indices, theta_deg, out_dir, svg_path)
    snapshot_vsz_dir(vsz_dir, keep=20)
    export_svg(vsz_path, svg_path)

This module emits .vsz textually; the Veusz Python embed API is not
required. Veusz's CLI (``/Applications/Veusz.app/Contents/MacOS/veusz.exe``)
handles export.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


VEUSZ_CLI = "/Applications/Veusz.app/Contents/MacOS/veusz.exe"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _hand_editable_files(vsz_dir: Path) -> list[Path]:
    """Files that the user hand-edits and the build script must never
    overwrite. Auto-generated CSVs and SVGs are excluded."""
    out: list[Path] = []
    for p in sorted(vsz_dir.glob("*.vsz")):
        out.append(p)
    style_log = vsz_dir / "STYLE_LOG.md"
    if style_log.exists():
        out.append(style_log)
    return out


def snapshot_vsz_dir(vsz_dir: str | os.PathLike, keep: int = 20) -> Path:
    """Copy all hand-editable files in ``vsz_dir`` into
    ``vsz_dir/_snapshots/{ISO timestamp}/``. Prunes oldest snapshots so
    at most ``keep`` directories remain.

    Returns the path of the snapshot just created (or the most recent
    existing snapshot if there's nothing to capture).
    """
    vsz_dir = Path(vsz_dir)
    snap_root = vsz_dir / "_snapshots"
    snap_root.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = snap_root / ts
    dest.mkdir(parents=True, exist_ok=False)

    files = _hand_editable_files(vsz_dir)
    for f in files:
        shutil.copy2(f, dest / f.name)

    # Prune old snapshots
    snaps = sorted(snap_root.iterdir())
    while len(snaps) > keep:
        old = snaps.pop(0)
        shutil.rmtree(old, ignore_errors=True)

    return dest


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_svg(vsz_path: str | os.PathLike,
               svg_path: str | os.PathLike) -> None:
    """Render ``vsz_path`` to SVG via the Veusz CLI."""
    vsz_path = Path(vsz_path)
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(VEUSZ_CLI).exists():
        raise RuntimeError(
            f"Veusz CLI not found at {VEUSZ_CLI}. Install Veusz.app in "
            "/Applications or set tools.build_vsz.VEUSZ_CLI.")
    subprocess.run(
        [VEUSZ_CLI, "--export", str(svg_path), str(vsz_path)],
        check=True,
    )


# ---------------------------------------------------------------------------
# σ² panel group
# ---------------------------------------------------------------------------


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Write a simple CSV with the given named columns. All columns
    must have the same length. Used for Veusz's ImportFileCSV."""
    names = list(columns)
    arrays = [np.asarray(columns[n], dtype=np.float64) for n in names]
    n_rows = arrays[0].size
    for a, n in zip(arrays, names):
        if a.size != n_rows:
            raise ValueError(
                f"column {n!r} has size {a.size}, expected {n_rows}")
    with open(path, "w") as f:
        f.write(",".join(names) + "\n")
        for i in range(n_rows):
            f.write(",".join(_fmt(a[i]) for a in arrays) + "\n")


def _fmt(x: float) -> str:
    """Format a float compactly for CSV. Non-finite values are written
    as the empty string so Veusz skips them in plotting."""
    if not np.isfinite(x):
        return ""
    return f"{x:.6g}"


def _sigma2_csvs(
    s2_diag: np.ndarray,        # (n_theta, n_z) on diagonal
    se_diag: np.ndarray,        # same shape
    theta_deg: np.ndarray,
    z_indices: Sequence[int],
    csv_path: Path,
    cat_prefix: str,            # "q_" or "d_"
) -> dict[str, str]:
    """Write a CSV with theta + per-z-bin (sigma2, se) pairs.

    Column names are pre-prefixed with ``cat_prefix`` so that
    ``ImportFileCSV`` (without its own prefix kwarg, which behaves
    unexpectedly) creates datasets named ``q_theta_deg`` etc.

    Returns a mapping ``{z_index_str: (s2_col_name, se_col_name)}``.
    """
    cols: dict[str, np.ndarray] = {
        f"{cat_prefix}theta_deg": np.asarray(theta_deg),
    }
    name_map: dict[str, tuple[str, str]] = {}
    for slot, iq in enumerate(z_indices):
        s2_name = f"{cat_prefix}s2_z{slot}"
        se_name = f"{cat_prefix}se_z{slot}"
        cols[s2_name] = s2_diag[:, iq]
        cols[se_name] = se_diag[:, iq]
        name_map[str(iq)] = (s2_name, se_name)
    _write_csv(csv_path, cols)
    return name_map


# Seed template for the σ² Veusz document. This text is written to
# ``vsz/sigma2.vsz`` ONCE if the file doesn't exist; subsequent builds
# leave it alone so the user's GUI edits survive.
#
# The data CSVs (``vsz/data_sigma2_quaia.csv`` and
# ``vsz/data_sigma2_desi.csv``) are re-written each build, and Veusz
# re-reads them via the ``ImportFileCSV(..., linked=True)`` directive.
#
# The grid is 2 rows × 4 cols (Quaia top, DESI bottom; 4 z-quartile
# midpoints across).
SIGMA2_VSZ_SEED = """# Veusz saved document (auto-seeded)
# Hand-edit freely; subsequent rebuilds leave this file alone.
# Data CSVs are regenerated every build.

ImportFileCSV(u'data_sigma2_quaia.csv', linked=True)
ImportFileCSV(u'data_sigma2_desi.csv',  linked=True)

Set('width', '28cm')
Set('height', '14cm')
Set('colorTheme', u'default1')

# ---- Global stylesheet ---------------------------------------------------
Set('StyleSheet/Font/font', u'Helvetica')
Set('StyleSheet/Font/size', u'10pt')
Set('StyleSheet/axis/Label/size', u'11pt')
Set('StyleSheet/axis/TickLabels/size', u'9pt')
Set('StyleSheet/axis/MajorTicks/length', u'4pt')
Set('StyleSheet/axis/MinorTicks/length', u'2pt')
Set('StyleSheet/axis/Line/width', u'0.6pt')
Set('StyleSheet/xy/markerSize', u'3pt')
Set('StyleSheet/xy/PlotLine/width', u'1pt')

Add('page', name='page1', autoadd=False)
To('page1')
Add('grid', name='grid1', autoadd=False)
To('grid1')
Set('rows', 2)
Set('columns', 4)
Set('leftMargin', '1.2cm')
Set('rightMargin', '0.3cm')
Set('topMargin', '0.5cm')
Set('bottomMargin', '1.0cm')
Set('internalMargin', '0.4cm')

{panels}

To('..')  # leave grid1
To('..')  # leave page1
"""


# Per-panel template: one graph with x (log θ in deg), y (log σ²), and an
# xy trace bound to the right CSV column. ``{...}`` placeholders are
# filled by the build script.
PANEL_TEMPLATE = """
# ---- Panel {slot}: {cat_label} z={z_mid:.2f} -----------------------
Add('graph', name='panel_{cat}_{slot}', autoadd=False)
To('panel_{cat}_{slot}')
Add('axis', name='x', autoadd=False)
To('x')
Set('label', u'{xlabel}')
Set('log', True)
Set('min', 0.05)
Set('max', 15.0)
To('..')
Add('axis', name='y', autoadd=False)
To('y')
Set('label', u'{ylabel}')
Set('direction', 'vertical')
Set('log', True)
Set('min', 1e-3)
Set('max', 1.0)
To('..')
Add('label', name='title', autoadd=False)
To('title')
Set('label', u'{title}')
Set('xPos', [0.5])
Set('yPos', [0.92])
Set('alignHorz', u'centre')
Set('Text/size', u'10pt')
To('..')
Add('xy', name='dd', autoadd=False)
To('dd')
Set('xData', u'{x_col}')
Set('yData', u'{y_col}')
Set('errorStyle', u'barends')
Set('marker', u'{marker}')
Set('MarkerFill/color', u'{color}')
Set('MarkerLine/color', u'{color}')
Set('PlotLine/color', u'{color}')
Set('ErrorBarLine/color', u'{color}')
Set('key', u'{cat_label}')
To('..')
To('..')  # leave panel
"""


def _build_sigma2_seed(z_indices: Sequence[int],
                       z_mids: Sequence[float]) -> str:
    """Render the seed sigma2.vsz body for the given z_indices."""
    panels = []
    # Row 1: Quaia (4 z slots). Row 2: DESI (4 z slots). Veusz's grid
    # layout fills row-major, so emit Quaia panels then DESI panels.
    for cat, prefix, color, marker, label in (
        ("quaia", "q", "#1f77b4", "circle", "Quaia G<20"),
        ("desi",  "d", "#ff7f0e", "square", "DESI Y1 QSO"),
    ):
        for slot, iq in enumerate(z_indices):
            panels.append(PANEL_TEMPLATE.format(
                slot=slot, cat=cat, cat_label=label,
                z_mid=z_mids[slot],
                xlabel=r"\\theta [deg]" if cat == "desi" else "",
                ylabel=r"\\sigma^2_{clust}" if slot == 0 else "",
                title=f"z = {z_mids[slot]:.2f}",
                x_col=f"{prefix}_theta_deg",
                y_col=f"{prefix}_s2_z{slot}",
                e_col=f"{prefix}_se_z{slot}",
                marker=marker, color=color,
            ))
    body = SIGMA2_VSZ_SEED.format(panels="".join(panels))
    return body


def sigma2_group(
    s2_q_diag: np.ndarray,   # (n_theta, n_z) Quaia diagonal σ²
    se_q_diag: np.ndarray,
    s2_d_diag: np.ndarray,   # (n_theta, n_z) DESI diagonal σ²
    se_d_diag: np.ndarray,
    theta_deg: np.ndarray,
    z_indices: Sequence[int],
    z_mids: Sequence[float],
    vsz_dir: str | os.PathLike,
) -> Path:
    """Refresh the σ² panel group.

    Always writes the CSVs (auto-regenerated each build). Seeds
    ``sigma2.vsz`` if missing; otherwise leaves it for hand-editing.

    Returns the path to ``sigma2.vsz``.
    """
    vsz_dir = Path(vsz_dir)
    vsz_dir.mkdir(parents=True, exist_ok=True)

    # 1) write CSVs (column names baked with q_/d_ prefix)
    _sigma2_csvs(
        s2_q_diag, se_q_diag, theta_deg, z_indices,
        vsz_dir / "data_sigma2_quaia.csv", cat_prefix="q_",
    )
    _sigma2_csvs(
        s2_d_diag, se_d_diag, theta_deg, z_indices,
        vsz_dir / "data_sigma2_desi.csv", cat_prefix="d_",
    )

    # 2) seed sigma2.vsz if it doesn't exist
    vsz_path = vsz_dir / "sigma2.vsz"
    if not vsz_path.exists():
        seed = _build_sigma2_seed(z_indices, z_mids)
        with open(vsz_path, "w") as f:
            f.write(seed)
        print(f"  seeded {vsz_path} ({len(seed)/1024:.1f} KB)",
              file=sys.stderr)
    return vsz_path
