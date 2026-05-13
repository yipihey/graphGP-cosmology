"""Local HTTP helper for the Veusz click-through workflow.

Serves docs/index.html (and assets) at http://localhost:8765/ AND
exposes:

- ``GET  /open?vsz=<repo-relative-path>`` → opens the .vsz file in
  the Veusz GUI via ``open -a Veusz <abs path>``. Returns 200.
- ``POST /rebuild`` → re-runs
  ``demos/build_knn_cdf_desi_quaia_presentation.py``. Returns 200 when
  done. Stdout/stderr streamed to the helper's terminal.
- ``GET /<anything>`` → static-serves files under ``docs/`` (and
  ``vsz/`` for the SVG/PNG previews referenced from the HTML).

Usage:
    python tools/veusz_helper.py [--port 8765]

Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


# ---------------------------------------------------------------------------


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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[veusz_helper] {fmt % args}\n")

    # /open?vsz=<path>
    def _handle_open(self, query: dict) -> None:
        rel = query.get("vsz", [""])[0]
        target = _safe_repo_relative(rel)
        if target is None or not target.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"file not found: {rel}".encode())
            return
        try:
            # Detached Popen — Veusz lives independently of the helper.
            # Send stdout/stderr to /dev/null so the noisy macOS-prefs
            # parse warnings don't fill the helper's terminal.
            subprocess.Popen(
                [VEUSZ_BIN, str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(
                f"Veusz binary not found at {VEUSZ_BIN}".encode())
            return
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"open failed: {e}".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"opened {rel} in Veusz".encode())

    # POST /rebuild
    def _handle_rebuild(self) -> None:
        env = os.environ.copy()
        env.setdefault("PAPER_USE_VEUSZ", "1")
        try:
            res = subprocess.run(
                [PYTHON, str(BUILD_SCRIPT)], cwd=REPO_ROOT, env=env,
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(
                f"rebuild failed (exit {e.returncode}):\n"
                f"{e.stderr[-2000:]}".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        # Last few lines of stdout for context
        tail = "\n".join(res.stdout.splitlines()[-10:])
        self.wfile.write(f"rebuild OK\n\n{tail}".encode())

    # GET static
    def _serve_static(self, path: str) -> None:
        # Map "/" → docs/index.html. Anything else under /docs/ or /vsz/.
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
            self.end_headers()
            return
        if not target.exists() or not target.is_file():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        ctype = _content_type(target)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with open(target, "rb") as f:
            self.wfile.write(f.read())

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"veusz_helper: serving {DOCS_DIR} on http://localhost:{args.port}/",
          file=sys.stderr)
    print(f"  GET  /open?vsz=<repo-relative-path>   open in Veusz GUI",
          file=sys.stderr)
    print(f"  POST /rebuild                         re-run build script",
          file=sys.stderr)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
