"""Record map geometry ahead of time, so `server.py --offline` can still draw maps.

    python visual/precache_maps.py                # scenarios + 25 per outcome group, both tasks
    python visual/precache_maps.py --per-group 50

For each test task it takes the most telling changesets of every outcome at the headline
model's threshold (highest-scored catches and false alarms, lowest-scored misses and clears)
and stores their before/after geometry in visual/data/geometry/ - the same cache the online
server fills as you browse. Requests go through the shared rate limiter (4 per second).
"""

from __future__ import annotations

import argparse
import json

import osm_live
import server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-group", type=int, default=25)
    args = ap.parse_args()
    wanted: list[int] = [s["cid"] for s in json.loads((server.DATA / "scenarios.json").read_text())]
    for task in server.TASKS:
        r = server.Replay(task)
        t = r.models[r.headline]["threshold"]
        s = r.scores[[r.headline, "label"]].rename(columns={r.headline: "p"})
        groups = (
            s[(s.p >= t) & (s.label == 1)].sort_values("p", ascending=False),   # caught
            s[(s.p >= t) & (s.label == 0)].sort_values("p", ascending=False),   # false alarm
            s[(s.p < t) & (s.label == 1)].sort_values("p"),                     # missed
            s[(s.p < t) & (s.label == 0)].sort_values("p"),                     # cleared
        )
        for g in groups:
            wanted += [int(c) for c in g.index[: args.per_group]]
    unique = list(dict.fromkeys(wanted))
    done = failed = 0
    for i, cid in enumerate(unique, 1):
        if (osm_live.GEOMETRY_DIR / f"osm_{cid}.json").exists():
            done += 1
            continue
        try:
            osm_live.cached_display(server.session, cid)
            done += 1
        except Exception as exc:  # noqa: BLE001 - one missing map must not stop the batch
            failed += 1
            print(f"{cid}: {exc}", flush=True)
        if i % 20 == 0:
            print(f"{i}/{len(unique)} changesets", flush=True)
    print(f"recorded {done}, failed {failed}; cache: {osm_live.GEOMETRY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
