#!/usr/bin/env bash
# == Demo screenshots ==
# Renders fixed views of the running demo (python visual/server.py) with headless Chrome,
# so the README images can be regenerated after any UI change.
set -euo pipefail
# First browser found: $CHROME, Chrome on macOS, Edge or Chrome on Windows, chromium on Linux.
for c in "${CHROME:-}" "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"          "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"          "/c/Program Files/Google/Chrome/Application/chrome.exe" "$(command -v chromium || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CHROME="$c"; break; fi
done
BASE="${BASE:-http://127.0.0.1:8765}"
OUT="$(cd "$(dirname "$0")" && pwd)/screenshots"

# Windows browsers need a Windows path for --screenshot
file_path() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }

shoot() {
  "$CHROME" --headless=new --hide-scrollbars --window-size=1600,1000 \
    --virtual-time-budget=15000 --screenshot="$(file_path "$OUT/$1.png")" "$BASE/#$2" 2>/dev/null
}

shoot pokemon-park replay/ovid/58688771
shoot fake-road replay/ovid/61154591
shoot name-spam replay/ovid/63588505
shoot false-alarm replay/ovid/56356326
shoot live-osm live/osm/170865351
shoot overview overview/ovid
shoot deleted-landmark replay/ovid_modern/69064265
