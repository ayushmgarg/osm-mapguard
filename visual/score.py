"""Score one changeset from the command line, through the exact functions the web app calls.

    python visual/score.py 58688771                    # replayed: round 4's stored outputs (ovid test set)
    python visual/score.py 58688771 --task ovid_modern
    python visual/score.py 123456 --live sandbox        # fetch from the OSM sandbox now and score on CPU
    python visual/score.py 190804354 --live osm --json

The web app's "Score a changeset" box and this script both go through `server.live_detail`,
and the replay views both go through `Replay.detail`, so they print the same numbers
(tests/test_demo.py checks the replay path against round 4's stored predictions).
"""

from __future__ import annotations

import argparse
import json
import sys

import server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cid", type=int)
    ap.add_argument("--task", default="ovid", choices=server.TASKS)
    ap.add_argument("--live", choices=("osm", "sandbox"), help="fetch now instead of replaying round 4")
    ap.add_argument("--json", action="store_true", help="print the full detail the web app receives")
    args = ap.parse_args()

    if args.live:
        server.REPLAYS[server.LIVE_TASK] = server.Replay(server.LIVE_TASK)
        try:
            detail = server.live_detail(args.live, args.cid)
        except (server.osm_live.ChangesetNotFound, server.osm_live.UpstreamError) as exc:
            print(f"could not score changeset {args.cid} from {args.live}: {exc}", file=sys.stderr)
            return 2
    else:
        server.REPLAYS[args.task] = server.Replay(args.task)
        try:
            detail = server.replay(args.task).detail(args.cid)
        except server.NotInReplay as exc:
            print(f"{exc}. Use --live osm to fetch and score it now.", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(detail, default=server.jsonable, indent=1))
        return 0
    m = detail["models"][detail["headline_model"]]
    flagged = m["score"] >= m["threshold"]
    cs = detail["changeset"]
    where = f", {detail.get('api')}" if detail["source"] == "live" else ""
    print(f"changeset {args.cid} ({detail['source']}{where})")
    print(f"  score      {100 * m['score']:.0f}/100 by {m['name']}; flags at {100 * m['threshold']:.0f} -> "
          f"{'FLAG FOR REVIEW' if flagged else 'no action'}")
    if detail["label"] is not None:
        print(f"  truth      {'reverted by another mapper (damaging)' if detail['label'] else 'not reverted (clean)'}")
    print(f"  edits      {cs.get('n_edits')} ({cs.get('n_create')} created, {cs.get('n_modify')} modified, "
          f"{cs.get('n_delete')} deleted); comment: {cs.get('comment') or '(none)'}")
    s = detail["signals"]
    print(f"  signals    who {s['who']:+.1f}  what {s['what']:+.1f}  text {s['text']:+.1f}  (log-odds)")
    rank = detail["edit_ranking"]
    top = max(detail["edits"], key=lambda e: e[rank] or 0, default=None)
    if top:
        print(f"  top edit   {top['action']} {top['otype']} {top['oid']}: {(top['tokens'] or '')[:90]}")
    others = [(k, v) for k, v in detail["models"].items() if k != detail["headline_model"]]
    if others:
        print("  models     " + "  ".join(f"{k} {100 * v['score']:.0f}" for k, v in others))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
