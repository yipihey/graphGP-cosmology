#!/usr/bin/env bash
# drive_upload.sh — upload one or more files to the project's
# Google Drive results folder via rclone.
#
# One-time setup: configure rclone with Google Drive access. Run
# `rclone config` interactively and create a remote named `gdrive`
# (or set PAPER_DRIVE_REMOTE to your remote name). See
# tools/UPLOAD_README.md.
#
# Usage:
#   tools/drive_upload.sh <file>...
#   PAPER_DRIVE_REMOTE=gdrive PAPER_DRIVE_PATH="tmp/Quaia" tools/drive_upload.sh output/foo.html
#
# Environment variables:
#   PAPER_DRIVE_REMOTE   rclone remote name (default: gdrive)
#   PAPER_DRIVE_PATH     destination path under the remote
#                        (default: tmp/Quaia — the "Quaia" folder
#                         inside My Drive/tmp/, the project's
#                         results bucket)
#
# Exits 0 on success; non-zero if rclone is missing or upload fails.

set -euo pipefail

REMOTE="${PAPER_DRIVE_REMOTE:-gdrive}"
DEST_PATH="${PAPER_DRIVE_PATH:-tmp/Quaia}"

if ! command -v rclone >/dev/null 2>&1; then
    echo "drive_upload.sh: rclone not found. Install with:" >&2
    echo "  brew install rclone" >&2
    exit 1
fi

# Verify the remote exists before attempting an upload.
if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE}:$"; then
    echo "drive_upload.sh: rclone remote '${REMOTE}:' not configured." >&2
    echo "  Run \`rclone config\` and create a Google Drive remote." >&2
    echo "  See tools/UPLOAD_README.md." >&2
    exit 2
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <file>..." >&2
    exit 1
fi

DEST="${REMOTE}:${DEST_PATH}"

echo "drive_upload: uploading $# file(s) to ${DEST} ..."
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "  skip (not a file): $f" >&2
        continue
    fi
    size=$(wc -c < "$f" | tr -d ' ')
    rclone copy --progress "$f" "${DEST}"
    echo "  ✓ $(basename "$f") (${size} bytes)"
done
echo "drive_upload: done. View at https://drive.google.com/drive/folders/16FquBrDQdwxFArqwr35uoKHuAr5e_aRA"
