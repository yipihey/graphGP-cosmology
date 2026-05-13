"""Diff-and-propagate Veusz style edits across panel groups.

Reads the most recent ``vsz/_snapshots/{ISO}/`` and the current
``vsz/*.vsz``. For each ``Set('path', value)`` line that changed,
classifies the scope from ``SCOPE_RULES`` and applies global/group
changes to the appropriate hand-edited files. Appends one line per
property to ``vsz/STYLE_LOG.md``.

For the MVP the σ² group is the only group, so "propagation" mostly
means logging — this scaffolding is in place so when more groups land
(σ²_DP, ξ_LS, S₃, …) the same diff pass applies global changes
everywhere.

Run via:

    python tools/propagate_vsz_edits.py [--dry-run]

Or call ``propagate(vsz_dir, dry_run=False)`` from another script.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VSZ_DIR = REPO_ROOT / "vsz"
STYLE_LOG = VSZ_DIR / "STYLE_LOG.md"

# scope rules. Earliest matching glob wins. Add entries as more
# panel groups land.
SCOPE_RULES: list[tuple[str, str]] = [
    ("StyleSheet/Font/*",          "global"),
    ("StyleSheet/axis/*",          "global"),
    ("StyleSheet/xy/*",            "global"),
    ("colorTheme",                 "global"),
    ("width",                      "global"),
    ("height",                     "global"),
    ("*/x/min", "panel"),
    ("*/x/max", "panel"),
    ("*/y/min", "panel"),
    ("*/y/max", "panel"),
    ("*/leftMargin",   "group"),
    ("*/rightMargin",  "group"),
    ("*/topMargin",    "group"),
    ("*/bottomMargin", "group"),
    ("*/internalMargin", "group"),
    ("panel_*/title/*", "panel"),
    ("panel_*/dd/MarkerFill/color", "group"),
    ("panel_*/dd/MarkerLine/color", "group"),
    ("panel_*/dd/PlotLine/color",   "group"),
    ("panel_*/dd/marker",           "group"),
]


# ``Set('path', value)`` — capture path and value as separate groups
SET_RE = re.compile(r"^Set\(\s*[u]?['\"](?P<path>[^'\"]+)['\"]\s*,\s*"
                    r"(?P<value>.+?)\s*\)\s*$")

# u'foo' vs 'foo' are semantically identical in the Veusz format; PyQt
# drops the unicode prefix on re-save. Normalise both sides before
# comparing so this drift doesn't pollute STYLE_LOG.md.
_UNICODE_PREFIX = re.compile(r"u(['\"])")


def _normalise(v: str) -> str:
    """Canonicalise a Veusz Set() value for diff comparison."""
    return _UNICODE_PREFIX.sub(r"\1", v).strip()


def _scope_for(path: str) -> str:
    for pat, scope in SCOPE_RULES:
        if fnmatch.fnmatchcase(path, pat):
            return scope
    return "local-warning"


def _parse_sets(text: str) -> dict[str, str]:
    """Return {path: value} for every `Set(...)` line. If a path appears
    multiple times, the LAST occurrence wins (Veusz semantics)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = SET_RE.match(line.strip())
        if m:
            out[m.group("path")] = m.group("value")
    return out


def _latest_snapshot(vsz_dir: Path) -> Path | None:
    snap_root = vsz_dir / "_snapshots"
    if not snap_root.exists():
        return None
    snaps = sorted(snap_root.iterdir())
    return snaps[-1] if snaps else None


def _diff_file(current: Path, snapshot: Path) -> list[tuple[str, str, str]]:
    """Return list of (path, old, new) for each property change.

    Values are normalised before comparison so that the ``u'foo'`` →
    ``'foo'`` re-serialisation drift that PyQt introduces on save
    doesn't appear in the log.
    """
    if not snapshot.exists():
        return []  # New file — no propagation, just log on first run.
    cur = _parse_sets(current.read_text())
    snp = _parse_sets(snapshot.read_text())
    changes: list[tuple[str, str, str]] = []
    for path, new in cur.items():
        old = snp.get(path)
        if old is None or _normalise(old) != _normalise(new):
            changes.append((path, old or "<new>", new))
    # Properties removed from current
    for path, old in snp.items():
        if path not in cur:
            changes.append((path, old, "<removed>"))
    return changes


def _append_log(entries: list[str]) -> None:
    if not entries:
        return
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    blob = f"\n## {ts}  (propagate_vsz_edits)\n\n"
    blob += "\n".join(f"- {e}" for e in entries) + "\n"
    with open(STYLE_LOG, "a") as f:
        f.write(blob)


def propagate(vsz_dir: Path | str, dry_run: bool = False) -> int:
    """Run one propagation pass. Returns number of recorded changes."""
    vsz_dir = Path(vsz_dir)
    snap = _latest_snapshot(vsz_dir)
    if snap is None:
        print("propagate: no snapshot yet; nothing to diff.",
              file=sys.stderr)
        return 0

    log_entries: list[str] = []
    for vsz_file in sorted(vsz_dir.glob("*.vsz")):
        snap_file = snap / vsz_file.name
        changes = _diff_file(vsz_file, snap_file)
        if not changes:
            continue
        for path, old, new in changes:
            scope = _scope_for(path)
            log_entries.append(
                f"`{vsz_file.name}` :: `Set('{path}')` "
                f"{old!s:.40} → {new!s:.40} *({scope})*"
            )
            # In the MVP we don't yet replicate the change into other
            # files — there's only one panel group. The scope tag
            # records the intended propagation for when more groups
            # land. Print to stderr so the user sees what would happen.
            if scope in ("global", "group") and not dry_run:
                print(f"propagate: {vsz_file.name} {path} → {scope} "
                      f"(would propagate when more groups exist)",
                      file=sys.stderr)
    if log_entries and not dry_run:
        _append_log(log_entries)
    print(f"propagate: {len(log_entries)} change(s) recorded "
          f"({'dry-run' if dry_run else 'logged'})", file=sys.stderr)
    return len(log_entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write to STYLE_LOG.md")
    args = ap.parse_args()
    propagate(VSZ_DIR, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
