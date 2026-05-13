# Style decision log

Append-only log of plot-style decisions made through the Veusz
edit-and-propagate workflow. Each entry: ISO timestamp + 1-line
summary of what changed and why. Future Claude reads this file before
seeding any new panel group so style stays consistent across the
report without re-asking.

## 2026-05-13  (initial seed from matplotlib defaults)

- **Fonts**: Helvetica 10pt body, panel titles 10pt, axis labels 11pt
- **Quaia**: `#1f77b4`, circle markers, 1pt line
- **DESI**: `#ff7f0e`, square markers, 1pt line
- **Per-shell panel grid**: 2 rows (Quaia top / DESI bottom) × 4 cols
  using `Z_PLOT_IDX` quartile-midpoint indices (z = 0.93, 1.22, 1.54,
  1.91 at the canonical 64-shell DESI z-range)
- **σ² axes**: log θ ∈ [0.05°, 15°]; log σ² ∈ [1e-3, 1.0]
- **Panel size**: 28cm × 14cm overall (2×4 grid → ~6×6cm per panel)
- **Margins**: 1.2cm left, 0.3cm right, 0.5cm top, 1.0cm bottom,
  0.4cm internal
- **Errors**: not yet wired (jackknife SE in CSV but not plotted in
  MVP; planned via Veusz CSV `+-` column convention)

## How propagation works

`tools/propagate_vsz_edits.py` runs at the start of each Veusz-mode
build. It diffs each `vsz/*.vsz` against the most recent
`vsz/_snapshots/{ISO}/` copy. For each `Set('property/path', value)`
line that changed, it consults a scope table:

- `Set('StyleSheet/...')` → **global** (would propagate to other panel
  groups when they exist)
- `Set('StyleSheet/xy/...')` → global trace-style defaults
- `Set('panel_*/x|y/min|max', ...)` → **local** (per-panel)
- Trace-level color/marker/PlotLine on a single panel → **group**
  (currently the σ² group only)

Unknown properties default to **local + log warning** so unintentional
global propagation is opt-in.

## 2026-05-13T18:41Z  (propagate_vsz_edits)

- `sigma2.vsz` :: `Set('StyleSheet/xy/PlotLine/width')` u'1pt' → u'1.6pt' *(global)*
