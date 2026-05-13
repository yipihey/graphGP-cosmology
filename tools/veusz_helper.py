"""Local HTTP helper for the Veusz click-through workflow.

Serves docs/index.html (and assets) at http://localhost:8765/ AND
exposes:

- ``GET  /open?vsz=<repo-relative-path>`` → opens the .vsz file in
  the Veusz GUI via a direct binary invocation. Returns 200.
- ``POST /rebuild`` → re-runs
  ``demos/build_knn_cdf_desi_quaia_presentation.py``. Returns JSON
  ``{"status":"ok","tail":"..."}`` when done. Stdout/stderr are
  captured.
- ``GET /<anything>`` → static-serves files under ``docs/`` (and
  ``vsz/`` for the SVG/PNG previews referenced from the HTML).

CORS is permissive (``Access-Control-Allow-Origin: *``) so that the
🔄 button in the HTML works whether the page is loaded via
``http://localhost:8765/`` or ``file://``.

Optional ``--watch`` mode polls ``vsz/*.vsz`` mtimes in a background
thread and triggers a rebuild ~1s after the latest save (debounced).
Useful if you want zero-click rebuilds. Use ``--watch-debounce N`` to
change the wait interval.

Usage:
    python tools/veusz_helper.py [--port 8765] [--watch]

Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
VSZ_DIR = REPO_ROOT / "vsz"
BUILD_SCRIPT = REPO_ROOT / "demos" / "build_knn_cdf_desi_quaia_presentation.py"
PYTHON = os.environ.get("PYTHON_BIN", "/Users/tabel/pyqt/bin/python")
# Direct binary invocation — `open -a Veusz <file>` doesn't pass the
# file argument cleanly to the PyQt Veusz build on macOS (it surfaces
# an empty editor window). Calling the binary directly with the path
# as argv[1] reliably opens the document.
VEUSZ_BIN = "/Applications/Veusz.app/Contents/MacOS/veusz.exe"


def _safe_repo_relative(rel: str) -> Path | None:
    """Resolve ``rel`` against REPO_ROOT, refusing escapes via ``..``."""
    if not rel:
        return None
    p = (REPO_ROOT / rel).resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return p


# ---------------------------------------------------------------------------
# Rebuild — shared by POST /rebuild and the watcher
# ---------------------------------------------------------------------------


_REBUILD_LOCK = threading.Lock()


def _do_rebuild() -> tuple[bool, str]:
    """Run the build script. Serialised so concurrent rebuilds (button
    + watcher firing simultaneously) collapse to one execution."""
    env = os.environ.copy()
    env.setdefault("PAPER_USE_VEUSZ", "1")
    with _REBUILD_LOCK:
        try:
            res = subprocess.run(
                [PYTHON, str(BUILD_SCRIPT)], cwd=REPO_ROOT, env=env,
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            return False, f"exit {e.returncode}: {e.stderr[-2000:]}"
        tail = "\n".join(res.stdout.splitlines()[-10:])
        return True, tail


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[veusz_helper] {fmt % args}\n")

    # CORS — permissive so a button on file:// can POST here.
    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    # /open?vsz=<path>
    def _handle_open(self, query: dict) -> None:
        rel = query.get("vsz", [""])[0]
        target = _safe_repo_relative(rel)
        if target is None or not target.exists():
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            self.wfile.write(f"file not found: {rel}".encode())
            return
        try:
            subprocess.Popen(
                [VEUSZ_BIN, str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            self.send_response(500)
            self._send_cors()
            self.end_headers()
            self.wfile.write(
                f"Veusz binary not found at {VEUSZ_BIN}".encode())
            return
        except Exception as e:
            self.send_response(500)
            self._send_cors()
            self.end_headers()
            self.wfile.write(f"open failed: {e}".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self._send_cors()
        self.end_headers()
        self.wfile.write(f"opened {rel} in Veusz".encode())

    # POST /rebuild
    def _handle_rebuild(self) -> None:
        ok, msg = _do_rebuild()
        if not ok:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self._send_cors()
            self.end_headers()
            self.wfile.write(f"rebuild failed: {msg}".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors()
        self.end_headers()
        self.wfile.write(
            json.dumps({"status": "ok", "tail": msg}).encode())

    # GET static
    def _serve_static(self, path: str) -> None:
        if path in ("/", ""):
            target = DOCS_DIR / "index.html"
        elif path.startswith("/vsz/"):
            target = VSZ_DIR / path[len("/vsz/"):]
        else:
            target = DOCS_DIR / path.lstrip("/")
        target = target.resolve()
        try:
            target.relative_to(REPO_ROOT)
        except ValueError:
            self.send_response(403)
            self._send_cors()
            self.end_headers()
            return
        if not target.exists() or not target.is_file():
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            self.wfile.write(b"not found")
            return
        ctype = _content_type(target)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(target.stat().st_size))
        # Cache-Control: no-cache so /rebuild + browser reload always
        # picks up the freshly-written SVG. Without this, the browser
        # may serve a stale 304-cached copy.
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        self._send_cors()
        self.end_headers()
        with open(target, "rb") as f:
            self.wfile.write(f.read())

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/open":
            self._handle_open(urllib.parse.parse_qs(u.query))
            return
        self._serve_static(u.path)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/rebuild":
            self._handle_rebuild()
            return
        self.send_response(404)
        self._send_cors()
        self.end_headers()


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".svg":  "image/svg+xml",
        ".png":  "image/png",
        ".pdf":  "application/pdf",
        ".css":  "text/css",
        ".js":   "application/javascript",
        ".json": "application/json",
        ".vsz":  "text/plain",
        ".csv":  "text/csv",
    }.get(suffix, "application/octet-stream")


# ---------------------------------------------------------------------------
# Optional file watcher
# ---------------------------------------------------------------------------


def _watch_loop(debounce_s: float) -> None:
    """Poll vsz/*.vsz mtimes; rebuild ~debounce_s after the most recent
    change. Coalesces rapid-fire saves (Veusz writes the file in a few
    operations on ⌘S) into a single rebuild.

    Skips files written by the build script itself (data_*.csv, .svg,
    sigma2.svg) — only hand-edited *.vsz files trigger a rebuild.
    """
    def _scan() -> dict[str, float]:
        return {
            p.name: p.stat().st_mtime
            for p in VSZ_DIR.glob("*.vsz")
        }

    print(f"[veusz_helper] watch mode: {debounce_s:.1f}s debounce on "
          f"{VSZ_DIR}/*.vsz", file=sys.stderr)
    last = _scan()
    pending_since: float | None = None

    while True:
        time.sleep(0.5)
        try:
            cur = _scan()
        except OSError:
            continue
        if cur != last:
            last = cur
            pending_since = time.time()
            print("[veusz_helper] watch: change detected, "
                  f"debouncing {debounce_s:.1f}s", file=sys.stderr)
        elif pending_since is not None and \
                time.time() - pending_since > debounce_s:
            pending_since = None
            print("[veusz_helper] watch: triggering rebuild",
                  file=sys.stderr)
            ok, msg = _do_rebuild()
            print(f"[veusz_helper] watch: rebuild {'OK' if ok else 'FAILED'}",
                  file=sys.stderr)
            # Refresh mtime snapshot since the build wrote new files.
            try:
                last = _scan()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--watch", action="store_true",
                    help="auto-rebuild when vsz/*.vsz changes")
    ap.add_argument("--watch-debounce", type=float, default=1.5,
                    help="seconds to wait after last change before "
                         "rebuilding (default 1.5)")
    args = ap.parse_args()

    print(f"veusz_helper: serving {DOCS_DIR} on http://localhost:{args.port}/",
          file=sys.stderr)
    print(f"  GET  /open?vsz=<repo-relative-path>   open in Veusz GUI",
          file=sys.stderr)
    print(f"  POST /rebuild                         re-run build script",
          file=sys.stderr)

    if args.watch:
        t = threading.Thread(
            target=_watch_loop, args=(args.watch_debounce,),
            daemon=True,
        )
        t.start()

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
