# Veusz publication-quality plot workflow

Iterate to publication-ready figures by editing them visually in
[Veusz](https://veusz.github.io/) instead of describing changes in
natural language.

## Quickstart

1. Make sure you have Veusz installed at
   `/Applications/Veusz.app` (override `tools.build_vsz.VEUSZ_CLI`
   if installed elsewhere).
2. Generate the Veusz panel files:
   ```bash
   PAPER_USE_VEUSZ=1 /Users/tabel/pyqt/bin/python \
       demos/build_knn_cdf_desi_quaia_presentation.py
   ```
   This writes `vsz/sigma2.vsz`, refreshes the data CSVs, and exports
   `docs/sigma2.svg`.
3. Start the local helper:
   ```bash
   python tools/veusz_helper.py        # http://localhost:8765/
   ```
4. Open `http://localhost:8765/` in a browser. Click the σ² panel
   under the "σ²(θ;z)" tab → Veusz opens with `vsz/sigma2.vsz`.
5. Tweak axes, line weights, fonts, … in Veusz. Save (`⌘S`).
6. Trigger a rebuild:
   ```bash
   curl -X POST http://localhost:8765/rebuild
   ```
   Or rerun step 2 manually. The HTML page refreshes; the σ² panel
   reflects your edits. `vsz/STYLE_LOG.md` gains one entry per
   property changed.

## File layout

```
vsz/
  sigma2.vsz                ← seeded once; HAND-EDITED freely
  data_sigma2_quaia.csv     ← auto-regenerated each build
  data_sigma2_desi.csv      ← auto-regenerated each build
  STYLE_LOG.md              ← decision log (append-only)
  _snapshots/{ISO}/         ← copies of hand-edited files captured at
                              the start of each build (used by
                              propagate_vsz_edits.py)
```

**Invariants the build script obeys:**

- It NEVER overwrites `*.vsz` (other than freshly seeding one if
  missing).
- It refreshes CSVs every run; Veusz re-reads them via
  `ImportFileCSV(linked=True)`.
- Before regenerating, it snapshots `*.vsz` + `STYLE_LOG.md` to
  `_snapshots/{ISO}/` so the propagator can diff.

## Style propagation

`tools/propagate_vsz_edits.py` runs at the start of each
PAPER_USE_VEUSZ build:

1. Reads the most recent snapshot.
2. Diffs each current `vsz/*.vsz` against its snapshot copy.
3. For each `Set('path', value)` line that changed, classifies the
   scope from `SCOPE_RULES`:
   - `Set('StyleSheet/...')` → **global** (would replicate to other
     panel groups when they exist)
   - Per-panel axis range → **local**
   - Trace styling on a panel → **group** (within the σ² group)
4. Appends one line per change to `STYLE_LOG.md` with the scope tag.

For the MVP (σ² is the only group), propagation is mostly a logging
pass — but the scaffolding is in place so when more groups land
(σ²_DP, ξ_LS, S₃, …) the same diff pass replicates global changes
across them.

## Adding more panel groups

For each new group `<g>`:

1. Add a function in `tools/build_vsz.py`:
   ```python
   def <g>_group(quaia_data, desi_data, ..., vsz_dir):
       # write data_<g>_*.csv and seed vsz/<g>.vsz once
   ```
   Use the existing `sigma2_group` as a template. Keep style rules
   identical to whatever `STYLE_LOG.md` records (Helvetica, line
   weights, color hex codes, …).
2. Wire it into `demos/build_knn_cdf_desi_quaia_presentation.py`
   alongside the σ² Veusz block.
3. Embed the SVG in the corresponding HTML tab with a
   `<a href="http://localhost:8765/open?vsz=vsz/<g>.vsz">` wrapper.

## Caveats

- Veusz's CLI prints lots of noisy `Error interpreting item ...`
  warnings to stderr that come from macOS preferences parsing — they
  are harmless and can be filtered with
  `grep -vE "Error interpreting|SAMP"`.
- `ImportFileCSV(prefix='q_')` does NOT prefix dataset names as you
  might expect; we bake the prefix into the CSV column headers
  instead.
- Symmetric error bars use Veusz's CSV `+-` column convention (a
  column literally named `+-` after the data column). Not yet
  wired in the MVP.
- The auto-snapshot retention is 20 most-recent dirs; older
  snapshots are pruned by `tools.build_vsz.snapshot_vsz_dir`.
