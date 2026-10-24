"""Gate 0 - does the public `real-changesets` bucket cover Ovid's 2014-2019 labels?

The whole data plan assumes we can pull changeset *contents* (before/after tags and
node coordinates) straight from AWS Open Data instead of reconstructing them from a
150 GB history file. The bucket is populated by OSMCha's pipeline, and a listing of
it starts at 2024-era ids - so coverage of the older labelled range is unverified.

This script measures the hit rate with cheap HEAD requests and prints a verdict
against the thresholds in PROJECT_PLAN.md section 8:

    > 80%   proceed as designed
    30-80%  proceed on the covered subset, document reduced N
    < 30%   switch to Fallback B (OSM API changeset download)

Usage:
    python scripts/gate0_coverage.py [--sample 150] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels" / "ovid_labels.tsv"
OUT = ROOT / "results" / "gates" / "gate0_coverage.json"

BUCKET = "https://real-changesets.s3.us-west-2.amazonaws.com"
TIMEOUT = 20
N_BUCKETS = 10  # id-range strata, so we see *where* coverage starts


def load_labels(path: Path) -> list[tuple[int, bool]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = next(fh)
        if not header.startswith("changeset"):
            raise ValueError(f"unexpected header: {header!r}")
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            rows.append((int(cid), label.strip().lower() == "true"))
    return rows


def stratify(rows: list[tuple[int, bool]], n: int) -> list[tuple[int, bool]]:
    """Sample evenly across the id range and across both classes.

    Uniform random sampling would tell us the average hit rate but not whether
    coverage has a cutoff date, which is the thing we actually need to know.
    """
    rows = sorted(rows)
    per_bucket = max(1, n // N_BUCKETS)
    size = len(rows) / N_BUCKETS
    picked: list[tuple[int, bool]] = []

    for b in range(N_BUCKETS):
        chunk = rows[int(b * size):int((b + 1) * size)]
        if not chunk:
            continue
        pos = [r for r in chunk if r[1]]
        neg = [r for r in chunk if not r[1]]
        half = max(1, per_bucket // 2)
        for pool in (pos, neg):
            if not pool:
                continue
            step = max(1, len(pool) // half)
            picked.extend(pool[::step][:half])
    return picked


def probe(item: tuple[int, bool]) -> dict:
    cid, is_vandalism = item
    url = f"{BUCKET}/{cid}.json"
    rec = {"changeset": cid, "vandalism": is_vandalism, "status": None, "bytes": None}
    try:
        r = requests.head(url, timeout=TIMEOUT)
        rec["status"] = r.status_code
        if r.status_code == 200:
            rec["bytes"] = int(r.headers.get("Content-Length", 0))
    except requests.RequestException as exc:
        rec["status"] = f"ERR:{type(exc).__name__}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8, help="keep modest; public bucket")
    args = ap.parse_args()

    rows = load_labels(LABELS)
    lo, hi = min(r[0] for r in rows), max(r[0] for r in rows)
    print(f"labels          : {len(rows):,}  ({sum(r[1] for r in rows):,} vandalism)")
    print(f"changeset range : {lo:,} .. {hi:,}")

    sample = stratify(rows, args.sample)
    print(f"probing         : {len(sample)} ids, {args.workers} workers\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(probe, sample))
    elapsed = time.time() - t0

    hits = [r for r in results if r["status"] == 200]
    rate = len(hits) / len(results) if results else 0.0

    # where does coverage begin? report per id-range stratum
    print(f"{'id range':>26}  {'hit':>5}  {'n':>4}  rate")
    print("-" * 52)
    size = len(sample) / N_BUCKETS
    strata = []
    ordered = sorted(results, key=lambda r: r["changeset"])
    for b in range(N_BUCKETS):
        chunk = ordered[int(b * size):int((b + 1) * size)]
        if not chunk:
            continue
        h = sum(1 for r in chunk if r["status"] == 200)
        r0, r1 = chunk[0]["changeset"], chunk[-1]["changeset"]
        pct = h / len(chunk)
        strata.append({"from": r0, "to": r1, "hits": h, "n": len(chunk), "rate": pct})
        bar = "#" * int(pct * 20)
        print(f"{r0:>11,}..{r1:>11,}  {h:>5}  {len(chunk):>4}  {pct:6.1%} {bar}")

    pos = [r for r in results if r["vandalism"]]
    neg = [r for r in results if not r["vandalism"]]
    pos_rate = sum(1 for r in pos if r["status"] == 200) / len(pos) if pos else 0
    neg_rate = sum(1 for r in neg if r["status"] == 200) / len(neg) if neg else 0

    codes: dict[str, int] = {}
    for r in results:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1

    if rate > 0.80:
        verdict, action = "PASS", "Proceed as designed. Full label set usable."
    elif rate >= 0.30:
        verdict, action = "PARTIAL", "Proceed on the covered subset; document reduced N."
    else:
        verdict, action = "FAIL", "Switch to Fallback B: OSM API changeset download."

    est = int(rate * len(rows))
    print("\n" + "=" * 52)
    print(f"overall hit rate : {rate:.1%}  ({len(hits)}/{len(results)})")
    print(f"  vandalism      : {pos_rate:.1%}")
    print(f"  clean          : {neg_rate:.1%}")
    print(f"status codes     : {codes}")
    print(f"elapsed          : {elapsed:.1f}s")
    print(f"projected usable : ~{est:,} of {len(rows):,} labelled changesets")
    print("=" * 52)
    print(f"VERDICT: {verdict}  -  {action}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bucket": BUCKET,
        "labels_total": len(rows),
        "changeset_range": [lo, hi],
        "sampled": len(results),
        "hit_rate": rate,
        "hit_rate_vandalism": pos_rate,
        "hit_rate_clean": neg_rate,
        "status_codes": codes,
        "strata": strata,
        "projected_usable_labels": est,
        "verdict": verdict,
        "action": action,
        "probes": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
