#!/usr/bin/env bash
# == Demo screenshots ==
# Renders fixed views of the running demo (python visual/server.py) with headless Chrome,
# so the README images can be regenerated after any UI change.
set -euo pipefail
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
BASE="${BASE:-http://127.0.0.1:8765}"
OUT="$(dirname "$0")/screenshots"

shoot() {
  "$CHROME" --headless=new --hide-scrollbars --window-size=1440,1500 \
    --virtual-time-budget=15000 --screenshot="$OUT/$1.png" "$BASE/#$2" 2>/dev/null
}

shoot pokemon-park replay/ovid/58688771
shoot fake-road replay/ovid/61154591
shoot name-spam replay/ovid/63588505
shoot false-alarm replay/ovid/56356326
shoot live-osm live/osm/170865351
