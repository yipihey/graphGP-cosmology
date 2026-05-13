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

## 2026-05-13T19:15Z  (propagate_vsz_edits)

- `sigma2.vsz` :: `Set('colorTheme')` u'default1' → 'default1' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/Font/font')` u'Helvetica' → 'Helvetica' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/Font/size')` u'10pt' → '10pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis/Line/width')` u'0.6pt' → '0.6pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis/Label/size')` u'11pt' → '11pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis/TickLabels/size')` u'9pt' → '9pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis/MajorTicks/length')` u'4pt' → '4pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis/MinorTicks/length')` u'2pt' → '2pt' *(global)*
- `sigma2.vsz` :: `Set('StyleSheet/axis-function/autoRange')` <new> → 'next-tick' *(local-warning)*
- `sigma2.vsz` :: `Set('StyleSheet/xy/PlotLine/width')` u'1pt' → '1pt' *(global)*
- `sigma2.vsz` :: `Set('leftMargin')` '1.2cm' → '0cm' *(local-warning)*
- `sigma2.vsz` :: `Set('rightMargin')` '0.3cm' → '0cm' *(local-warning)*
- `sigma2.vsz` :: `Set('topMargin')` '0.5cm' → '0cm' *(local-warning)*
- `sigma2.vsz` :: `Set('bottomMargin')` '1.0cm' → '0cm' *(local-warning)*
- `sigma2.vsz` :: `Set('label')` u'z = 1.91' → 'z = 1.91' *(local-warning)*
- `sigma2.vsz` :: `Set('min')` 1e-3 → 0.03 *(local-warning)*
- `sigma2.vsz` :: `Set('max')` 1.0 → 0.15 *(local-warning)*
- `sigma2.vsz` :: `Set('log')` True → False *(local-warning)*
- `sigma2.vsz` :: `Set('alignHorz')` u'centre' → 'centre' *(local-warning)*
- `sigma2.vsz` :: `Set('Text/size')` u'10pt' → '10pt' *(local-warning)*
- `sigma2.vsz` :: `Set('marker')` u'square' → 'square' *(local-warning)*
- `sigma2.vsz` :: `Set('xData')` u'd_theta_deg' → 'd_theta_deg' *(local-warning)*
- `sigma2.vsz` :: `Set('yData')` u'd_s2_z3' → 'd_s2_z3' *(local-warning)*
- `sigma2.vsz` :: `Set('key')` u'DESI Y1 QSO' → 'DESI Y1 QSO' *(local-warning)*
- `sigma2.vsz` :: `Set('errorStyle')` u'barends' → 'barends' *(local-warning)*
- `sigma2.vsz` :: `Set('PlotLine/color')` u'#ff7f0e' → '#ff7f0e' *(local-warning)*
- `sigma2.vsz` :: `Set('MarkerLine/color')` u'#ff7f0e' → '#ff7f0e' *(local-warning)*
- `sigma2.vsz` :: `Set('MarkerFill/color')` u'#ff7f0e' → '#ff7f0e' *(local-warning)*
- `sigma2.vsz` :: `Set('ErrorBarLine/color')` u'#ff7f0e' → '#ff7f0e' *(local-warning)*
- `sigma2.vsz` :: `Set('StyleSheet/xy/markerSize')` u'3pt' → <removed> *(global)*

## 2026-05-13T20:01Z  (propagate_vsz_edits)

- `sigma2.vsz` :: `Set('min')` 0.03 → 0.1 *(local-warning)*
- `sigma2.vsz` :: `Set('max')` 0.15 → 0.2 *(local-warning)*
