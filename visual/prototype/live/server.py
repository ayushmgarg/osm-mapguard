"""MapGuard live map - a local web server.

    .venv/Scripts/python live/server.py              # then open http://localhost:8765
    .venv/Scripts/python live/server.py --offline    # no network at all: committed fixtures only
    .venv/Scripts/python live/server.py --per-minute 4 --port 8765

Two modes on one OpenStreetMap map:

  Test set   Ovid's 3,607 held-out changesets with the repo's real round-4 scores (stack +
  (replay)   embeddings, mean of 3 seeds), read from release v1.1.0 - or, with --offline, from
             fixtures/replay.json (every test point, full detail for the demo changesets).
  Live       every minute the server asks the OSM API for the newest closed changesets and scores
             a few with core.Engine.score_changeset - the same function live/score.py calls. Any
             changeset id can also be scored on request. With --offline, recorded results from
             fixtures/live.json are shown instead and nothing is fetched.

OSM is volunteer-run: all requests go through the repo's PoliteSession (one global token bucket,
backs off on 429/509), at most --per-minute changesets are scored, and nothing is cached on disk.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(HERE))

import replay  # noqa: E402

STATIC = HERE / "static"
FIXTURES = DEMO / "fixtures"
REPORT = HERE / "model" / "scorer_report.json"
MAX_LIVE = 300


# ------------------------------------------------------------------ live feed
class Live:
    def __init__(self, engine, per_minute: int):
        self.engine, self.per_minute = engine, per_minute
        self.items: "OrderedDict[int, dict]" = OrderedDict()
        self.seen: set[int] = set()
        self.seq = 0
        self.lock = threading.Lock()
        self.status = {"state": "starting", "polls": 0, "scored": 0, "skipped": 0, "errors": 0,
                       "last_poll": None, "last_poll_epoch": None, "poll_ms": None, "fetch_s": None,
                       "score_ms": None, "last_error": None, "offline": False}

    def add(self, item: dict) -> dict:
        with self.lock:
            self.seq += 1
            item["seq"] = self.seq
            self.items.pop(item["cid"], None)
            self.items[item["cid"]] = item
            while len(self.items) > MAX_LIVE:
                self.items.popitem(last=False)
        return item

    def run(self):
        while True:
            t0 = time.monotonic()
            try:
                self.poll()
            except Exception as exc:                 # keep the feed alive
                self.status.update(errors=self.status["errors"] + 1, last_error=repr(exc)[:200])
                traceback.print_exc()
            time.sleep(max(5.0, 60.0 - (time.monotonic() - t0)))

    def poll(self):
        from core import MAX_CHANGES, OSM_API_URL, ScoreError, parse_changeset_metas
        self.status.update(state="polling OSM", last_poll=time.strftime("%H:%M:%S"), last_poll_epoch=time.time())
        t0 = time.monotonic()
        r = self.engine.sess.get(f"{OSM_API_URL}/changesets", params={"closed": "true", "limit": 100})
        self.status["poll_ms"] = round((time.monotonic() - t0) * 1000)
        if not r.ok:
            raise RuntimeError(f"changesets HTTP {r.status}")
        metas = [m for m in parse_changeset_metas(r.text) if m.cid not in self.seen]
        self.status["polls"] += 1
        fresh = [m for m in metas if m.has_bbox and m.num_changes and 0 < m.num_changes <= MAX_CHANGES]
        for m in metas:
            self.seen.add(m.cid)
        random.shuffle(fresh)
        for meta in fresh[: self.per_minute]:
            self.status["state"] = f"scoring changeset {meta.cid}"
            try:
                item = self.engine.score_meta(meta)
            except ScoreError as exc:
                self.status["skipped"] += 1
                self.status["last_error"] = str(exc)[:200]
                continue
            self.add(item)
            self.status.update(scored=self.status["scored"] + 1, fetch_s=item["timing"]["fetch_s"],
                               score_ms=item["timing"]["score_ms"])
            print(f"live {meta.cid}  score {item['score']['live']:.3f}  {meta.editor[:20]!r}  {meta.comment[:50]!r}",
                  flush=True)
        self.status["state"] = "waiting for the next minute"

    def score_on_request(self, cid: int) -> dict:
        from core import ScoreError
        if cid in self.items:
            return self.items[cid]
        self.status["state"] = f"scoring changeset {cid} (requested)"
        try:
            item = self.engine.score_changeset(cid)
        finally:
            self.status["state"] = "waiting for the next minute"
        item["requested"] = True
        self.status.update(fetch_s=item["timing"]["fetch_s"], score_ms=item["timing"]["score_ms"])
        return self.add(item)

    def since(self, seq: int) -> list[dict]:
        with self.lock:
            return [{k: it.get(k) for k in ("cid", "seq", "lat", "lon", "score", "threshold", "comment", "editor",
                                            "closed_at", "n_edits", "requested", "recorded")}
                    for it in self.items.values() if it["seq"] > seq]


# ------------------------------------------------------------------ HTTP
def make_handler(test_points, test_meta, test_detail, live: Live, report: dict, offline: bool):
    config = {"offline": offline,
              "replay_source": "fixtures/replay.json (offline copy)" if offline else "release v1.1.0 + saved round-4 scores",
              "detail_cids": sorted(test_detail.__self__) if offline else None}

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC), **kw)

        def log_message(self, fmt, *args):
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            if parts[:1] != ["api"]:
                return super().do_GET()
            try:
                if parts == ["api", "config"]:
                    return self.send_json(config)
                if parts == ["api", "test"]:
                    return self.send_json({"meta": test_meta, "points": test_points})
                if len(parts) == 3 and parts[1] == "test":
                    d = test_detail(int(parts[2]))
                    if d:
                        return self.send_json(d)
                    msg = ("offline mode keeps full detail only for the demo changesets"
                           if offline else "not in the test set")
                    return self.send_json({"error": msg}, 404)
                if parts == ["api", "live"]:
                    seq = int((dict(q.split("=") for q in u.query.split("&") if "=" in q)).get("since", 0))
                    return self.send_json({"items": live.since(seq), "status": live.status, "report": report})
                if len(parts) == 3 and parts[1] == "live":
                    it = live.items.get(int(parts[2]))
                    return self.send_json(it) if it else self.send_json({"error": "no longer in the feed"}, 404)
                if len(parts) == 3 and parts[1] == "score":
                    if offline:
                        return self.send_json({"error": "offline mode: scoring a new changeset needs the OSM API"}, 503)
                    from core import ScoreError
                    try:
                        return self.send_json(live.score_on_request(int(parts[2])))
                    except ScoreError as exc:
                        return self.send_json({"error": str(exc)}, exc.status)
            except ValueError:
                return self.send_json({"error": "bad request: a changeset id is a whole number"}, 400)
            return self.send_json({"error": "not found"}, 404)
    return H


def build(args):
    """Load everything the server needs; returns (handler class, live)."""
    report = json.loads(REPORT.read_text())
    report.pop("tabular_columns", None)
    if args.offline:
        pts, meta, detail = replay.load_from_fixtures(FIXTURES)
        live = Live(engine=None, per_minute=0)
        for item in json.loads((FIXTURES / "live.json").read_text(encoding="utf-8")):
            item["recorded"] = True
            live.add(item)
        live.status.update(state="offline: recorded live results, nothing is fetched", offline=True)
    else:
        from core import Engine
        engine = Engine(rate=args.rate, store_dir=replay.STORE / "feature_store")
        print("loading test set ...", flush=True)
        pts, meta, detail = replay.load_from_release(engine.explainer)
        live = Live(engine, args.per_minute)
        if args.no_live:
            live.status["state"] = "live feed off (--no-live); changesets can still be scored on request"
        else:
            threading.Thread(target=live.run, daemon=True).start()
    print(f"  {meta['n_located']:,} of {meta['n']:,} test changesets located", flush=True)
    return make_handler(pts, meta, detail, live, report, args.offline), live


def main() -> int:
    # changeset comments come in every script; a Windows console codepage must not crash the feed
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--per-minute", type=int, default=5, help="live changesets scored per minute")
    ap.add_argument("--rate", type=float, default=1.5, help="OSM API requests per second, all threads")
    ap.add_argument("--no-live", action="store_true", help="no background polling (scoring on request still works)")
    ap.add_argument("--offline", action="store_true", help="no network at all: committed fixtures only")
    args = ap.parse_args()
    if not args.offline and not replay.release_available():
        print("dataset release v1.1.0 not found under data/ - starting in --offline mode", flush=True)
        args.offline = True
    handler, _ = build(args)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"MapGuard live map{' (offline)' if args.offline else ''}: http://localhost:{args.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
