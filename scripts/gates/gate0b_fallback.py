"""Gate 0b - Gate 0 failed, so establish what the alternatives actually give us.

Two questions, since the 2014-2019 labels have no contents in the S3 bucket:

  A. Does Fallback B work? `GET /api/0.6/changeset/<id>/download` on real labelled
     ids - does it respond, what does it contain, how long does it take, and does
     it carry the *previous* state we need for the geometry branch?

  B. Where does real-changesets coverage actually begin? If it starts in year X,
     mining our own labels from >= X becomes viable and is arguably better science
     than replaying a 2019 benchmark.

Usage:
    python scripts/gate0b_fallback.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels" / "ovid_labels.tsv"
OUT = ROOT / "results" / "gates" / "gate0b_fallback.json"

OSM_API = "https://api.openstreetmap.org/api/0.6"
BUCKET = "https://real-changesets.s3.us-west-2.amazonaws.com"
UA = {"User-Agent": "MapGuard-research/0.1 (university project; contact via GitHub ayushmgarg)"}
TIMEOUT = 45


def test_osm_api(ids: list[int]) -> list[dict]:
    """Can we pull changeset contents from the OSM API for old changesets?"""
    out = []
    for cid in ids:
        rec: dict = {"changeset": cid}
        t0 = time.time()
        try:
            r = requests.get(f"{OSM_API}/changeset/{cid}/download", headers=UA, timeout=TIMEOUT)
            rec["status"] = r.status_code
            rec["seconds"] = round(time.time() - t0, 2)
            rec["bytes"] = len(r.content)
            if r.status_code == 200:
                x = r.text
                rec["ops"] = {
                    op: len(re.findall(rf"<{op}>", x)) for op in ("create", "modify", "delete")
                }
                rec["n_node"] = x.count("<node ")
                rec["n_way"] = x.count("<way ")
                rec["n_relation"] = x.count("<relation ")
                rec["has_lat_lon"] = ' lat="' in x
                # the crucial question: is the PREVIOUS version present?
                vers = [int(v) for v in re.findall(r' version="(\d+)"', x)]
                rec["max_version"] = max(vers) if vers else None
                rec["has_v_gt_1"] = any(v > 1 for v in vers)
        except requests.RequestException as exc:
            rec["status"] = f"ERR:{type(exc).__name__}"
        out.append(rec)
        print(f"  cs {cid:>11,}  {rec.get('status')}  "
              f"{rec.get('bytes', 0):>8,}B  {rec.get('seconds', 0):>5}s  "
              f"nodes={rec.get('n_node')} ways={rec.get('n_way')} "
              f"latlon={rec.get('has_lat_lon')} v>1={rec.get('has_v_gt_1')}")
        time.sleep(1.0)  # be polite
    return out


def find_coverage_start() -> list[dict]:
    """Ladder-probe the bucket to find roughly where coverage begins."""
    rungs = [80, 90, 100, 110, 120, 130, 140, 145, 150, 160, 170]
    out = []
    for m in rungs:
        base = m * 1_000_000
        probes = [base + off for off in (1_111, 22_222, 333_333, 444_444, 555_555)]
        hits = 0
        for cid in probes:
            try:
                r = requests.head(f"{BUCKET}/{cid}.json", timeout=20)
                if r.status_code == 200:
                    hits += 1
            except requests.RequestException:
                pass
        rate = hits / len(probes)
        out.append({"id_millions": m, "hits": hits, "n": len(probes), "rate": rate})
        print(f"  ~{m:>3}M  {hits}/{len(probes)}  {rate:5.0%}  {'#' * int(rate * 20)}")
    return out


def main() -> None:
    rows = []
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            rows.append((int(cid), label.strip().lower() == "true"))

    vandal = [c for c, v in rows if v]
    picks = [vandal[0], vandal[len(vandal) // 2], vandal[-1]]

    print("A. OSM API changeset download on labelled vandalism changesets")
    print("   (Fallback B viability)\n")
    api = test_osm_api(picks)

    print("\nB. real-changesets coverage ladder")
    print("   (can we mine our own modern corpus instead?)\n")
    ladder = find_coverage_start()

    ok = sum(1 for r in api if r.get("status") == 200)
    print("\n" + "=" * 58)
    print(f"Fallback B viable : {ok}/{len(api)} changesets fetched")
    covered = [r for r in ladder if r["rate"] > 0.5]
    if covered:
        print(f"Bucket coverage   : solid from ~{covered[0]['id_millions']}M onward")
    else:
        print("Bucket coverage   : sparse across all probed rungs")
    print("=" * 58)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "osm_api_probes": api,
        "bucket_ladder": ladder,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
