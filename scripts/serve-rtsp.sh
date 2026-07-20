#!/usr/bin/env bash
# Serve a multicam-sim scene as N live RTSP camera streams (fake IP cameras).
#
# Renders a domain-neutral scene with the pure-numpy rasterizer and publishes
# each camera as its own rtsp:// stream through a local mediamtx, then prints the
# URLs. multicam-rt's `ingest-rtsp` consumes them like real cameras.
#
# Dependencies (macOS / Homebrew):
#   brew install mediamtx        # zero-dependency RTSP server
#   brew install ffmpeg@7        # H.264 encode; @7 matches the Rust consumer
#
# RTSP carries PIXELS ONLY, no ground truth — this is the realism / network-path
# test, not the eval path (see docs/rtsp.md).
#
#   scripts/serve-rtsp.sh [--cameras 3] [--fps 15] [...]   # forwarded to python
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

command -v mediamtx >/dev/null 2>&1 || {
  echo "mediamtx not found. Install it: brew install mediamtx" >&2; exit 1;
}
if [ ! -x /opt/homebrew/opt/ffmpeg@7/bin/ffmpeg ] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it: brew install ffmpeg@7" >&2; exit 1
fi

exec env PYTHONPATH="$repo_root/src" python3 "$repo_root/scripts/serve_rtsp.py" "$@"
