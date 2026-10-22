"""Record the committed fixtures that make the demo work offline, and the golden expectations.

    .venv/Scripts/python live/build_fixtures.py            # replay + live (needs the network)
    .venv/Scripts/python live/build_fixtures.py --no-live  # replay only, keep the recorded live.json

Writes to fixtures/:
  replay.json  every test-set point (scores, truth, location) and full detail for the demo
               changesets, exactly as the server builds them from release v1.1.0
  live.json    real changesets scored through core.Engine.score_changeset, with their feature rows
  golden.json  what the tests must reproduce: per changeset the score, the top-ranked edit, the
               number of map points and the signal bars
No author ids or usernames are written (core.PRIVATE strips them; replay payloads never had them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay  # noqa: E402
from core import Engine, ScoreError, parse_changeset_metas, OSM_API_URL  # noqa: E402

FIX = HERE.parent / "fixtures"
SCENARIOS = [73656923, 61155395, 25818789, 22578095, 56564752]
N_LIVE = 8


def top_index(d: dict) -> int:
    key = (lambda e: e["w"] or 0) if d["rank_by"] == "attention" else (lambda e: e["push"])
    return max(range(len(d["edits"])), key=lambda i: key(d["edits"][i])) if d["edits"] else -1


def golden_entry(d: dict, score_key: str) -> dict:
    t = top_index(d)
    return {"cid": d["cid"], "score": d["score"][score_key], "top_edit": t,
            "top_tokens": d["edits"][t]["tok"][:6] if t >= 0 else [],
            "n_edits_listed": len(d["edits"]), "n_nodes": len(d["nodes"]), "n_ways": len(d["ways"]),
            "signals": {b["key"]: b["delta"] for b in (d["signals"] or {}).get("bars", [])}}


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    FIX.mkdir(exist_ok=True)

    engine = Engine(store_dir=replay.STORE / "feature_store")
    pts, meta, detail = replay.load_from_release(engine.explainer)
    examples = json.loads((replay.BUILD / "examples.json").read_text(encoding="utf-8"))
    cids = list(dict.fromkeys(SCENARIOS + [e["cid"] for e in examples]))
    details = {c: detail(c) for c in cids if detail(c)}
    (FIX / "replay.json").write_text(dump({"meta": meta, "points": pts, "details": details,
                                           "scenarios": SCENARIOS,
                                           "source": "release v1.1.0 + round-4 scores (build/scores.json)"}),
                                     encoding="utf-8")
    golden = {"tolerance": 1e-6, "replay": [golden_entry(details[c], "stack_emb") for c in SCENARIOS]}
    print(f"replay.json: {len(pts):,} points, {len(details)} detailed changesets")

    if args.no_live:
        golden["live"] = json.loads((FIX / "golden.json").read_text())["live"]
    else:
        r = engine.sess.get(f"{OSM_API_URL}/changesets", params={"closed": "true", "limit": 100})
        metas = [m for m in parse_changeset_metas(r.text) if m.has_bbox and m.num_changes and m.num_changes <= 300]
        items = []
        for m in metas:
            if len(items) >= N_LIVE:
                break
            try:
                items.append(engine.score_meta(m))
                print(f"  live {m.cid}: {items[-1]['score']['live']:.4f}  {m.comment[:50]!r}")
            except ScoreError as exc:
                print(f"  skip {m.cid}: {exc}")
        (FIX / "live.json").write_text(dump(items), encoding="utf-8")
        golden["live"] = [golden_entry(it, "live") for it in items]
    (FIX / "golden.json").write_text(json.dumps(golden, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in ("replay.json", "live.json", "golden.json"):
        print(f"{f}: {(FIX / f).stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
