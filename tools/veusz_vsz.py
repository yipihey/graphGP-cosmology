"""Generate self-contained, browser-editable Veusz ``.vsz`` figures from Python.

A ``.vsz`` is a plain-text Veusz command script. The browser embed
(``<veusz-figure src="x.vsz">`` + the hosted ``veusz-embed.js``, the WASM build
from the Veusz.jl / veusz-tauri fork) fetches the ``.vsz`` and renders it
*editable* in any modern browser — the reader can pan/zoom, change colours,
fonts, ranges, markers, and re-export, with no server, no Python, no Julia.

This module writes such ``.vsz`` files directly, with data **embedded**
(``ImportString`` / ``ImportString2D``, exactly the form Veusz itself uses when
saving with embedded datasets — see veusz/datasets/{oned,twod}.py), so each
figure is a single portable file. No Veusz install is needed to *write* them.

High-level builders:
  * ``scatter`` — one graph, N xy series (points and/or lines, per-series colour,
    marker, transparency/alpha, error bars, key), optional log axes.
  * ``grid`` — a multi-panel grid of graphs (used for the many inpaint
    before/after panels), each panel a list of xy series.
  * ``image`` — a 2-D array as a colour image with a colorbar (e.g. ξ(Δθ,Δz)).

Conventions (match the rest of the report): right-handed astronomical RA is
*wrapped* with :func:`wrap_ra` so a survey cap straddling 0/360 is contiguous and
centred near 0; pass ``invert_x=True`` to draw RA increasing leftwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


def wrap_ra(ra):
    """Wrap RA (deg) to (-180, 180], so a cap across 0/360 is contiguous and
    centred near 0 (the CMASS-South footprint then runs ≈ -43°…+45°)."""
    return ((np.asarray(ra, float) + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# low-level dataset serialisation (mirrors veusz' own save format)
# ---------------------------------------------------------------------------
def _imp1d(name, data, serr=None, perr=None, nerr=None):
    """``ImportString('name(numeric)[,+-/+/-]', '''…''')`` for a 1-D dataset.

    Columns are written ``value [serr] [perr] [nerr]`` per row in ``%e`` form,
    exactly as :meth:`Dataset.saveDataDumpToText`."""
    data = np.asarray(data, float)
    desc = name + "(numeric)"
    cols = [data]
    if serr is not None:
        desc += ",+-"; cols.append(np.asarray(serr, float))
    if perr is not None:
        desc += ",+"; cols.append(np.asarray(perr, float))
    if nerr is not None:
        desc += ",-"; cols.append(np.asarray(nerr, float))
    rows = np.column_stack(cols)
    body = "\n".join(" ".join("%e" % v for v in row) for row in rows)
    return "ImportString(%r,'''\n%s\n''')\n" % (desc, body)


def _imp2d(name, grid, xrange, yrange):
    """``ImportString2D('name', '''xrange…/yrange…/matrix''')`` for a 2-D array.

    ``grid`` is indexed ``[iy, ix]`` (row = y); written low-y row first, matching
    Veusz' ``ImportString2D`` reader and :meth:`Dataset2D.saveDataDumpToText`."""
    grid = np.asarray(grid, float)
    head = "xrange %e %e\nyrange %e %e\n" % (xrange[0], xrange[1], yrange[0], yrange[1])
    body = "\n".join(" ".join("%e" % v for v in row) for row in grid)
    return "ImportString2D(%r, '''\n%s%s\n''')\n" % (name, head, body)


# ---------------------------------------------------------------------------
# figure description objects
# ---------------------------------------------------------------------------
@dataclass
class Series:
    """One xy plotter: x, y arrays + style."""
    x: Sequence[float]
    y: Sequence[float]
    label: str = ""                  # legend key ("" => not in key)
    color: str = "#1f77b4"
    marker: str = "circle"           # 'circle','square','none',...
    line: bool = False               # draw a connecting PlotLine
    line_only: bool = False          # line, no markers
    size: str = "3pt"
    alpha: float = 1.0               # 1.0 opaque .. 0.0 transparent
    yerr: Optional[Sequence[float]] = None
    line_style: str = "solid"        # 'solid','dashed','dotted'
    cdata: Optional[Sequence[float]] = None   # per-point values -> colour-mapped markers
    colormap: str = "viridis"


@dataclass
class Panel:
    """One graph: a set of series sharing axes."""
    series: List[Series]
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    xrange: Optional[tuple] = None   # (min,max) or None=auto
    yrange: Optional[tuple] = None
    xlog: bool = False
    ylog: bool = False
    invert_x: bool = False           # RA increasing leftwards
    equal_aspect: bool = False


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
class _Vsz:
    def __init__(self, width="18cm", height="12cm"):
        self.imports: List[str] = []
        self.body: List[str] = []
        self._n = 0
        self.width, self.height = width, height

    def data(self, prefix, x, y, yerr=None):
        """Register x/y(/yerr) datasets under unique names; return their names."""
        self._n += 1
        xn, yn = f"{prefix}{self._n}x", f"{prefix}{self._n}y"
        self.imports.append(_imp1d(xn, x))
        self.imports.append(_imp1d(yn, y, serr=yerr))
        return xn, yn

    def data2d(self, name, grid, xrange, yrange):
        self.imports.append(_imp2d(name, grid, xrange, yrange))
        return name

    def _axis(self, name, label, rng, log, vertical, invert=False):
        b = self.body
        b.append(f"Add('axis', name={name!r}, autoadd=False)"); b.append(f"To({name!r})")
        b.append(f"Set('label', {label!r})")
        if rng is not None:
            b.append(f"Set('min', {float(rng[0])!r})"); b.append(f"Set('max', {float(rng[1])!r})")
        else:
            b.append("Set('min', 'Auto')"); b.append("Set('max', 'Auto')")
        b.append(f"Set('log', {bool(log)})")
        if invert:                                   # RA increasing leftwards
            b.append("Set('reflect', True)")
        if vertical:
            b.append("Set('direction', 'vertical')")
        b.append("To('..')")

    def _series(self, s: Series):
        b = self.body
        xn, yn = self.data("d", s.x, s.y, yerr=s.yerr)
        nm = f"xy{self._n}"
        b.append(f"Add('xy', name={nm!r}, autoadd=False)"); b.append(f"To({nm!r})")
        b.append(f"Set('xData', {xn!r})"); b.append(f"Set('yData', {yn!r})")
        marker = "none" if s.line_only else s.marker
        b.append(f"Set('marker', {marker!r})")
        b.append(f"Set('markerSize', {s.size!r})")
        if s.label:
            b.append(f"Set('key', {s.label!r})")
        # line
        if s.line or s.line_only:
            b.append("Set('PlotLine/hide', False)")
            b.append(f"Set('PlotLine/color', {s.color!r})")
            b.append(f"Set('PlotLine/style', {s.line_style!r})")
            b.append("Set('PlotLine/width', '1.5pt')")
        else:
            b.append("Set('PlotLine/hide', True)")
        # markers + alpha (Veusz transparency is 0..100 %, 0 = opaque)
        transp = int(round((1.0 - float(s.alpha)) * 100))
        if s.cdata is not None:                       # per-point colour-mapped markers
            self._n += 1; cn = f"c{self._n}"
            self.imports.append(_imp1d(cn, s.cdata))
            b.append(f"Set('Color/points', {cn!r})")
            b.append(f"Set('MarkerFill/colorMap', {s.colormap!r})")
            b.append("Set('MarkerLine/hide', True)")
        else:
            b.append(f"Set('MarkerFill/color', {s.color!r})")
            b.append(f"Set('MarkerLine/color', {s.color!r})")
            b.append(f"Set('MarkerLine/transparency', {transp})")
        b.append(f"Set('MarkerFill/transparency', {transp})")
        if s.yerr is not None:
            b.append("Set('errorStyle', 'barends')")
            b.append(f"Set('ErrorBarLine/color', {s.color!r})")
        b.append("To('..')")

    def graph(self, p: Panel, name="graph1", top=False):
        b = self.body
        b.append(f"Add('graph', name={name!r}, autoadd=False)"); b.append(f"To({name!r})")
        if p.equal_aspect:
            b.append("Set('aspect', True)")
        self._axis("x", p.xlabel, p.xrange, p.xlog, False, invert=p.invert_x)
        self._axis("y", p.ylabel, p.yrange, p.ylog, True)
        if p.title:
            b.append("Add('label', name='title', autoadd=False)"); b.append("To('title')")
            b.append(f"Set('label', {p.title!r})")
            b.append("Set('xPos', [0.5])"); b.append("Set('yPos', [1.04])")
            b.append("Set('alignHorz', 'centre')"); b.append("Set('Text/size', '11pt')")
            b.append("To('..')")
        for s in p.series:
            self._series(s)
        b.append("To('..')")

    def render(self):
        head = [
            "# Veusz saved document (version 4.2)",
            "SetCompatLevel(0)",
            f"Set('width', {self.width!r})",
            f"Set('height', {self.height!r})",
            "Set('StyleSheet/Font/font', 'Helvetica')",
            "Set('StyleSheet/Font/size', '11pt')",
            "Set('StyleSheet/axis/Line/width', '0.8pt')",
        ]
        return "\n".join(head) + "\n" + "".join(self.imports) + "\n".join(self.body) + "\n"


def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------
def scatter(path, panel: Panel, *, width="18cm", height="12cm"):
    """Write a single-graph figure (N series) to ``path`` (.vsz)."""
    v = _Vsz(width, height)
    v.body.append("Add('page', name='page1', autoadd=False)"); v.body.append("To('page1')")
    v.graph(panel)
    v.body.append("To('..')")
    return _write(path, v.render())


def grid(path, panels: List[Panel], *, rows, cols, width="26cm", height="26cm"):
    """Write a ``rows×cols`` grid of graphs (each a :class:`Panel`) to ``path``."""
    v = _Vsz(width, height)
    v.body.append("Add('page', name='page1', autoadd=False)"); v.body.append("To('page1')")
    v.body.append("Add('grid', name='grid1', autoadd=False)"); v.body.append("To('grid1')")
    v.body.append(f"Set('rows', {int(rows)})"); v.body.append(f"Set('columns', {int(cols)})")
    v.body.append("Set('leftMargin', '0.4cm')"); v.body.append("Set('rightMargin', '0.2cm')")
    v.body.append("Set('topMargin', '0.2cm')"); v.body.append("Set('bottomMargin', '0.4cm')")
    v.body.append("Set('internalMargin', '0.5cm')")
    for i, p in enumerate(panels):
        v.graph(p, name=f"g{i}")
    v.body.append("To('..')")
    v.body.append("To('..')")
    return _write(path, v.render())


def image(path, grid2d, *, xrange, yrange, xlabel="", ylabel="", title="",
          colormap="viridis", xlog=False, ylog=False, width="16cm", height="13cm"):
    """Write a 2-D array as a colour image with a colorbar to ``path``."""
    v = _Vsz(width, height)
    v.body.append("Add('page', name='page1', autoadd=False)"); v.body.append("To('page1')")
    v.body.append("Add('graph', name='graph1', autoadd=False)"); v.body.append("To('graph1')")
    v._axis("x", xlabel, xrange, xlog, False)
    v._axis("y", ylabel, yrange, ylog, True)
    if title:
        v.body.append("Add('label', name='title', autoadd=False)"); v.body.append("To('title')")
        v.body.append(f"Set('label', {title!r})")
        v.body.append("Set('xPos', [0.5])"); v.body.append("Set('yPos', [1.04])")
        v.body.append("Set('alignHorz', 'centre')"); v.body.append("To('..')")
    v.data2d("img", grid2d, xrange, yrange)
    v.body.append("Add('image', name='image1', autoadd=False)"); v.body.append("To('image1')")
    v.body.append("Set('data', 'img')"); v.body.append(f"Set('colorMap', {colormap!r})")
    v.body.append("Add('colorbar', name='colorbar1', autoadd=False)"); v.body.append("To('colorbar1')")
    v.body.append("Set('image', 'image1')"); v.body.append("To('..')")
    v.body.append("To('..')")
    v.body.append("To('..')")
    v.body.append("To('..')")
    return _write(path, v.render())


EMBED_SCRIPT_VERSION = "v4.5.0"
_EMBED_BASE = f"https://yipihey.github.io/veusz/embed/{EMBED_SCRIPT_VERSION}"
# The browser embed runs veusz headless in a (shared, singleton) Pyodide worker;
# it only installs veusz when given the wheel URL via the `veusz-wheel` attribute.
WHEEL_URL = f"{_EMBED_BASE}/veusz-{EMBED_SCRIPT_VERSION[1:]}-py3-none-any.whl"
# the Rust/WASM paint module lives under a wasm/ subdir; without wasm-base the
# embed does import("undefined/veusz_paint_wasm.js") -> "module script failed".
WASM_BASE = f"{_EMBED_BASE}/wasm"
EMBED_SCRIPT = f'<script type="module" src="{_EMBED_BASE}/veusz-embed.js"></script>'


def embed_tag(vsz_relpath, *, width=720, height=460, poster=True):
    """HTML ``<veusz-figure>`` element referencing a ``.vsz`` (browser-editable).

    Carries ``veusz-wheel`` (so the Pyodide worker installs veusz) and
    ``wasm-base`` (so the WASM paint module resolves) — both required when the page
    is served from a different origin than the embed assets.

    With ``poster`` (default: the sibling ``.png``) the figure shows a **static
    image** immediately and only boots the heavy interactive engine on demand —
    so the report displays no matter what (no WebGPU, slow Pyodide, etc.) and the
    editable Veusz is an opt-in enhancement."""
    poster_attr = ""
    if poster:
        png = poster if isinstance(poster, str) else (
            vsz_relpath[:-4] + ".png" if vsz_relpath.endswith(".vsz") else vsz_relpath + ".png")
        poster_attr = f'poster="{png}" eager="false" '
    return (f'<veusz-figure src="{vsz_relpath}" veusz-wheel="{WHEEL_URL}" '
            f'wasm-base="{WASM_BASE}" {poster_attr}width="{width}" height="{height}" '
            f'style="display:block;max-width:100%"></veusz-figure>')
