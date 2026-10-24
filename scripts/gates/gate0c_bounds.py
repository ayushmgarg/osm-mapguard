"""Gate 0c - pin the exact usable window, now that 0a failed and 0b opened a door.

Three remaining unknowns before the data plan can be rewritten:

  A. What is the current maximum changeset id? (tells us whether the 170M miss in
     Gate 0b means "coverage stopped" or "those changesets don't exist yet")
  B. Where does bucket coverage end? Probe the top end finely.
  C. Does the previous-version endpoint work? The osmChange from Fallback B carries
     only the NEW version of each object; the geometry branch needs the old one via
     `GET /api/0.6/<type>/<id>/<version-1>`.

Usage:
    python scripts/gate0c_bounds.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "gates" / "gate0c_bounds.json"

OSM_API = "https://api.openstreetmap.org/api/0.6"
BUCKET = "https://real-changesets.s3.us-west-2.amazonaws.com"
UA = {"User-Agent": "MapGuard-research/0.1 (university project; contact via GitHub ayushmgarg)"}


def current_max_changeset() -> dict:
    """Latest changeset id, from the public changesets feed."""
    r = requests.get(f"{OSM_API}/changesets", headers=UA, timeout=45)
    rec = {"status": r.status_code}
    if r.status_code == 200:
        ids = [int(i) for i in re.findall(r'<changeset id="(\d+)"', r.text)]
        dates = re.findall(r'created_at="([^"]+)"', r.text)
        rec["max_id"] = max(ids) if ids else None
        rec["newest_created_at"] = dates[0] if dates else None
    return rec


def bucket_upper_bound(max_id: int | None) -> list[dict]:
    top = (max_id // 1_000_000) if max_id else 170
    rungs = [top - 20, top - 15, top - 10, top - 6, top - 3, top - 1, top]
    out = []
    for m in rungs:
        if m < 140:
            continue
        base = m * 1_000_000
        probes = [base + off for off in (1_111, 22_222, 333_333, 444_444, 555_555, 777_777)]
        hits = 0
        for cid in probes:
            try:
                if requests.head(f"{BUCKET}/{cid}.json", timeout=20).status_code == 200:
                    hits += 1
            except requests.RequestException:
                pass
        rate = hits / len(probes)
        out.append({"id_millions": m, "hits": hits, "n": len(probes), "rate": rate})
        print(f"  ~{m:>3}M  {hits}/{len(probes)}  {rate:5.0%}  {'#' * int(rate * 20)}")
    return out


def test_previous_version() -> list[dict]:
    """Fetch a changeset, find a modified object, then pull its previous version."""
    out = []
    # a labelled vandalism changeset known to contain a v>1 object
    cid = 19739784
    r = requests.get(f"{OSM_API}/changeset/{cid}/download", headers=UA, timeout=45)
    if r.status_code != 200:
        return [{"error": f"changeset fetch {r.status_code}"}]

    found = re.findall(r'<(node|way|relation) id="(\d+)"[^>]*version="(\d+)"', r.text)
    targets = [(t, i, int(v)) for t, i, v in found if int(v) > 1][:3]
    print(f"  changeset {cid:,}: {len(found)} objects, {len(targets)} with version > 1")

    for otype, oid, ver in targets:
        prev = ver - 1
        t0 = time.time()
        pr = requests.get(f"{OSM_API}/{otype}/{oid}/{prev}", headers=UA, timeout=45)
        rec = {
            "type": otype, "id": oid, "version_new": ver, "version_prev": prev,
            "status": pr.status_code, "seconds": round(time.time() - t0, 2),
            "bytes": len(pr.content),
        }
        if pr.status_code == 200:
            rec["has_lat_lon"] = ' lat="' in pr.text
            rec["has_nd_refs"] = "<nd " in pr.text
            rec["n_tags"] = pr.text.count("<tag ")
        out.append(rec)
        print(f"  {otype} {oid} v{prev}: {pr.status_code}  {rec['bytes']:,}B  "
              f"{rec['seconds']}s  tags={rec.get('n_tags')} "
              f"latlon={rec.get('has_lat_lon')} nd={rec.get('has_nd_refs')}")
        time.sleep(1.0)
    return out


def main() -> None:
    print("A. Current maximum changeset id\n")
    cur = current_max_changeset()
    print(f"  max_id = {cur.get('max_id'):,}" if cur.get("max_id") else f"  {cur}")
    print(f"  newest = {cur.get('newest_created_at')}\n")

    print("B. Bucket coverage, upper end\n")
    ladder = bucket_upper_bound(cur.get("max_id"))

    print("\nC. Previous-version endpoint (needed for before/after geometry)\n")
    prev = test_previous_version()

    covered = [r for r in ladder if r["rate"] >= 0.5]
    print("\n" + "=" * 58)
    if covered:
        print(f"Bucket usable to  : ~{covered[-1]['id_millions']}M")
    prev_ok = sum(1 for p in prev if p.get("status") == 200)
    print(f"Prev-version API  : {prev_ok}/{len(prev)} fetched")
    print("=" * 58)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "current_changeset": cur,
        "bucket_upper_ladder": ladder,
        "previous_version_probes": prev,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
