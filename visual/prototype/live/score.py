"""Score one OpenStreetMap changeset from the command line - the same code path as the dashboard.

    .venv/Scripts/python live/score.py 190804354
    .venv/Scripts/python live/score.py 190804354 --json      # the exact payload the page receives

Needs the network (OSM API). For the saved research scores of Ovid's test set, use the
dashboard's replay mode instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Engine, ScoreError  # noqa: E402

REPORT = Path(__file__).resolve().parent / "model" / "scorer_report.json"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cid", type=int, help="changeset id")
    ap.add_argument("--json", action="store_true", help="print the full payload as JSON")
    ap.add_argument("--rate", type=float, default=1.5, help="OSM API requests per second")
    args = ap.parse_args()

    try:
        item = Engine(rate=args.rate).score_changeset(args.cid)
    except ScoreError as exc:
        print(f"cannot score: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(item, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        return 0

    thr = json.loads(REPORT.read_text())["live_threshold"]
    s = item["score"]["live"]
    print(f"changeset {item['cid']}  score {s:.4f}  -> {'FLAG for review' if s >= thr else 'looks clean'} "
          f"(live threshold {thr:.2f})")
    print(f"  {item['editor']}  |  {item['n_edits']} edits {item['ops']}  |  author: {item['user_n_prior']}"
          f"{'+' if item['user_history_capped'] else ''} earlier changesets")
    print(f"  comment: {item['comment'] or '(none)'}")
    print("  why (score drop when the signal is replaced by a typical value):")
    for b in item["signals"]["bars"]:
        print(f"    {b['label']:<22} {b['delta']:+.3f}")
    ranked = sorted(range(len(item["edits"])), key=lambda i: -item["edits"][i]["push"])[:3]
    print("  top edits by token push:")
    for i in ranked:
        e = item["edits"][i]
        print(f"    #{i + 1:<3} {e['a']:<6} {e['t']:<8} push {e['push']:+.2f}  {' '.join(e['tok'][1:6])}  {e['geo']}")
    print(f"  fetch {item['timing']['fetch_s']} s, scoring {item['timing']['score_ms']} ms  |  "
          f"model: {item['model']['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
