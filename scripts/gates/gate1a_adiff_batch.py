"""Gate 1a - does Overpass adiff actually deliver before/after geometry at batch scale?

Probe P2/P3 of the plan review showed adiff works on single changesets. That is not the
same as it working on a random sample of the actual corpus, where changesets are empty,
tag-only, global, or older than the attic. This gate measures the real yield.

Pass criterion (PROJECT_PLAN section 8.4): at least 90 of 100 sampled changesets return
usable before/after geometry, counting the versioned-multi-fetch fallback for the ones
adiff should not be asked about.

    python scripts/gate1a_adiff_batch.py --sample 100
    python scripts/gate1a_adiff_batch.py --sample 20 --seed 7   # quick look
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mapguard.config import LABELS, OVERPASS_RATE, RESULTS, DEFAULT_RATE
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import fetch_changeset_meta, fetch_versioned
from mapguard.data.overpass import fetch_geometry, should_use_adiff


def load_labels() -> list[tuple[int, int]]:
    rows = []
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            rows.append((int(cid), 1 if label.strip().lower() == "true" else 0))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = load_labels()
    rng = random.Random(args.seed)
    # Stratify by class so a yield difference between vandalism and clean edits shows up.
    pos = [c for c, y in labels if y == 1]
    neg = [c for c, y in labels if y == 0]
    half = args.sample // 2
    sample = [(c, 1) for c in rng.sample(pos, half)] + [(c, 0) for c in rng.sample(neg, args.sample - half)]
    rng.shuffle(sample)

    api = PoliteSession(DEFAULT_RATE)
    ovp = PoliteSession(OVERPASS_RATE, max_retries=3)

    records = []
    started = time.monotonic()

    for n, (cid, label) in enumerate(sample, 1):
        rec = {"cid": cid, "label": label}
        meta, r = fetch_changeset_meta(api, cid)
        rec["meta_status"] = r.status
        if meta is None:
            rec["outcome"] = "no_meta"
            records.append(rec)
            print(f"[{n}/{len(sample)}] {cid} meta {r.status} -> no_meta", flush=True)
            continue

        rec["num_changes"] = meta.num_changes
        rec["has_bbox"] = meta.has_bbox
        usable, why = should_use_adiff(meta)
        rec["adiff_eligible"] = usable
        rec["ineligible_reason"] = None if usable else why

        if usable:
            t0 = time.monotonic()
            actions, resp, note = fetch_geometry(ovp, meta)
            rec["overpass_status"] = resp.status if resp else None
            rec["overpass_seconds"] = round(time.monotonic() - t0, 2)
            rec["note"] = note
            rec["n_actions"] = len(actions)
            rec["n_with_old"] = sum(1 for a in actions if a.old is not None)
            rec["n_way"] = sum(1 for a in actions if a.otype == "way")
            rec["n_way_with_coords"] = sum(
                1 for a in actions if a.otype == "way" and ((a.new and a.new.coords) or (a.old and a.old.coords))
            )
            rec["route"] = "adiff"
            rec["outcome"] = "geometry" if actions else "adiff_empty"
        else:
            # Fallback route: previous states in one batched call. Nodes come back with
            # coordinates inline, which is what the displacement features need.
            rec["route"] = "multifetch"
            rec["outcome"] = "fallback_only"
            rec["n_actions"] = 0

        records.append(rec)
        print(
            f"[{n}/{len(sample)}] {cid} label={label} route={rec['route']} "
            f"outcome={rec['outcome']} actions={rec.get('n_actions')}",
            flush=True,
        )

    ok = [r for r in records if r.get("outcome") == "geometry"]
    empty = [r for r in records if r.get("outcome") == "adiff_empty"]
    fallback = [r for r in records if r.get("outcome") == "fallback_only"]
    nometa = [r for r in records if r.get("outcome") == "no_meta"]
    times = [r["overpass_seconds"] for r in records if r.get("overpass_seconds")]

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "1a",
        "question": "does Overpass adiff deliver before/after geometry at batch scale?",
        "pass_criterion": "at least 90% of sampled changesets yield usable geometry",
        "sample": len(records),
        "seed": args.seed,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
        "outcomes": {
            "geometry": len(ok),
            "adiff_empty": len(empty),
            "fallback_only": len(fallback),
            "no_meta": len(nometa),
        },
        "geometry_yield": round(len(ok) / max(1, len(records)), 3),
        "overpass_seconds_median": round(statistics.median(times), 2) if times else None,
        "overpass_seconds_max": round(max(times), 2) if times else None,
        "ways_seen": sum(r.get("n_way", 0) or 0 for r in records),
        "ways_with_coords": sum(r.get("n_way_with_coords", 0) or 0 for r in records),
        "yield_by_class": {
            "vandalism": round(
                sum(1 for r in ok if r["label"] == 1) / max(1, sum(1 for r in records if r["label"] == 1)), 3
            ),
            "clean": round(
                sum(1 for r in ok if r["label"] == 0) / max(1, sum(1 for r in records if r["label"] == 0)), 3
            ),
        },
        "ineligible_reasons": {},
        "records": records,
    }
    for r in records:
        why = r.get("ineligible_reason")
        if why:
            key = why.split("(")[0].strip()
            summary["ineligible_reasons"][key] = summary["ineligible_reasons"].get(key, 0) + 1

    summary["verdict"] = "PASS" if summary["geometry_yield"] >= 0.90 else "FAIL"

    out = Path(args.out) if args.out else RESULTS / "gates" / "gate1a_adiff_batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- Gate 1a ---")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
