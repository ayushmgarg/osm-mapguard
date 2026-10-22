"""MapGuard demo server: replay of round 4's outputs, live scoring, and a live feed.

A local process rather than a static site because live scoring has to call the OSM API and
run the tree model; nothing here needs TensorFlow. Standard-library HTTP server: the API is
six read-only routes, and a framework would be the largest dependency in the demo.

    python visual/server.py            # http://127.0.0.1:8765
    python visual/server.py --offline  # no network: replay and recorded maps only

Routes (all GET):
    /api/config                      tasks, models, thresholds, scenarios
    /api/replay/<task>               every test changeset of a task, one line each
    /api/replay/<task>/<cid>         one stored changeset: every model's score, why, edits, map
    /api/live/<osm|sandbox>/<cid>    fetch a changeset now and score it with the tree model
    /api/feed                        the newest real changesets, scored as they close
    /api/metrics/<task>              overview: outcome counts and round 4's multi-seed summary
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import osm_live
import pandas as pd
import tree_model

from mapguard.config import OSM_API
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import parse_changeset_metas

WEB = tree_model.VISUAL / "web"
DATA = tree_model.VISUAL / "data"
TASKS = ("ovid", "ovid_modern")
# Live changesets are modern, so they are scored by the model trained on both eras.
LIVE_TASK = "ovid_modern"
# Well under OSM's usage policy: one shared bucket for every request this server makes.
OSM_RATE = 4.0
FEED_POLL_SECONDS = 60
FEED_NEW_PER_POLL = 5
FEED_KEEP = 40
# The feed polls only while someone is looking at it, so an idle laptop is not a scraper.
FEED_IDLE_STOP_SECONDS = 300

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
ROUND4 = tree_model.REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds"
# Ovid (2021), as published - the reference line on the overview.
OVID_PUBLISHED = {"precision": 0.8035, "recall": 0.8302, "f1": 0.8166, "accuracy": 0.8137}
# --offline: no network at all. Replay answers from exported data and the geometry cache;
# live scoring and the feed say clearly that they are off. The fallback if the room's
# network fails.
OFFLINE = False


def google_maps_key() -> str | None:
    """GOOGLE_MAPS_API_KEY from the environment or visual/.env (git-ignored). Never committed;
    served only to this laptop's own page (the Host check below)."""
    import os
    if os.environ.get("GOOGLE_MAPS_API_KEY"):
        return os.environ["GOOGLE_MAPS_API_KEY"].strip()
    env = tree_model.VISUAL / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GOOGLE_MAPS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"') or None
    return None

session = PoliteSession(OSM_RATE)


class NotInReplay(Exception):
    pass


class Replay:
    """Round 4's exported outputs for one task, indexed by changeset id."""

    def __init__(self, task: str):
        root = DATA / task
        self.task = task
        self.models = json.loads((root / "models.json").read_text())
        self.headline = next(k for k, v in self.models.items() if v["headline"])
        self.scores = pd.read_parquet(root / "scores.parquet").set_index("cid")
        self.changesets = pd.read_parquet(root / "changesets.parquet").set_index("cid")
        edits = pd.read_parquet(root / "edits.parquet")
        self.edits = {cid: frame for cid, frame in edits.groupby("cid")}
        # one map point per changeset for the overview: the median of its edits' positions
        pts = edits.dropna(subset=["lat", "lon"]).groupby("cid")[["lat", "lon"]].median()
        self.points = {int(c): (round(float(r.lat), 4), round(float(r.lon), 4)) for c, r in pts.iterrows()}
        self.bundle = tree_model.load(task)

    def listing(self) -> list[dict]:
        cs = self.changesets
        return [{"cid": int(cid), "label": int(row["label"]), "score": round(float(row[self.headline]), 4),
                 "n_edits": int(cs.at[cid, "n_edits"]), "created_at": cs.at[cid, "created_at"],
                 "comment": (cs.at[cid, "comment"] or "")[:80],
                 "point": self.points.get(int(cid))}
                for cid, row in self.scores.iterrows()]

    def metrics(self) -> dict:
        """Everything on the overview, from round 4's stored outputs only.

        Confusion counts come from the exported scores at the headline model's validation
        threshold; the multi-seed summary is round 4's analysis/seeds.json, unchanged."""
        t = self.models[self.headline]["threshold"]
        s, y = self.scores[self.headline].to_numpy(), self.scores["label"].to_numpy()
        flag = s >= t
        counts = {"caught": int((flag & (y == 1)).sum()), "false_alarm": int((flag & (y == 0)).sum()),
                  "missed": int((~flag & (y == 1)).sum()), "cleared": int((~flag & (y == 0)).sum())}
        seeds = json.loads((ROUND4 / "analysis" / "seeds.json").read_text())["tasks"].get(self.task, {})
        # what the edit-token model learned, as round 4 stored it (seed 20260904 hybrid run)
        runs = sorted((ROUND4 / "raw/experiments/runs/baselines").glob(f"*_{self.task}_R1b_s20260904"))
        hyb = json.loads((runs[0] / "results.json").read_text())["models"]["hybrid"] if runs else {}
        return {"task": self.task, "headline": self.headline, "threshold": t, "n": int(len(y)),
                "counts": counts, "round4": {k: {m: v.get(m) for m in ("pr_auc", "one_per_author", "f1", "accuracy")}
                                             for k, v in seeds.get("models", {}).items()},
                "vs_hybrid": seeds.get("vs_hybrid_seed_averaged", {}),
                "ovid_published": OVID_PUBLISHED if self.task == "ovid" else None,
                "tokens": {"damaging": hyb.get("most_damaging_tokens", [])[:12],
                           "benign": hyb.get("most_benign_tokens", [])[:12]},
                "source": (ROUND4.relative_to(tree_model.REPO) / "analysis" / "seeds.json").as_posix()}

    def detail(self, cid: int) -> dict:
        if cid not in self.scores.index:
            raise NotInReplay(f"changeset {cid} is not in the {self.task} test set")
        row, cs = self.scores.loc[cid], self.changesets.loc[cid]
        models = {key: {**info, "score": float(row[key]),
                        "seeds": [float(row[c]) for c in row.index if c.startswith(f"{key}_s")]}
                  for key, info in self.models.items()}
        edits = self.edits.get(cid, pd.DataFrame())
        token_scores = tree_model.edit_token_scores(self.bundle, edits["diff_tokens"].tolist())
        stored = [{"position": int(e.position), "otype": e.otype, "oid": int(e.oid), "action": e.action,
                   "version": int(e.version) if pd.notna(e.version) else None,
                   "tokens": e.diff_tokens, "text": e.text, "old_text": e.old_text,
                   "attention": float(e.attention), "token_score": float(s)}
                  for e, s in zip(edits.itertuples(), token_scores)]
        return {"source": "replay", "task": self.task, "cid": cid, "label": int(row["label"]),
                "headline_model": self.headline, "models": models,
                "changeset": changeset_summary(cs), "signals": json.loads(cs["signals_json"]),
                "local_hybrid": float(cs["local_hybrid"]),
                "reasons": tree_model.token_reasons(self.bundle, cs["diff_tokens"]),
                "edits": stored, "edit_ranking": "attention", "map": replay_map(cid)}


def replay_map(cid: int) -> dict | None:
    if OFFLINE:
        cached = osm_live.GEOMETRY_DIR / f"osm_{cid}.json"
        return json.loads(cached.read_text()) if cached.exists() else {
            "error": "offline mode: this changeset's map was not recorded (python visual/precache_maps.py)"}
    try:
        return osm_live.cached_display(session, cid)
    except (osm_live.ChangesetNotFound, osm_live.UpstreamError) as exc:
        return {"error": str(exc)}


SUMMARY_KEYS = ("created_at", "comment", "editor", "n_edits", "n_create", "n_modify", "n_delete",
                "user_n_prior", "user_days_active_span", "user_history_known", "corpus")


def changeset_summary(cs: pd.Series) -> dict:
    return {k: (None if pd.isna(cs[k]) else cs[k]) for k in SUMMARY_KEYS if k in cs}


def jsonable(value):
    """numpy scalars -> Python; json.dumps calls this only for types it cannot encode."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


@lru_cache(maxsize=256)
def live_detail(api_name: str, cid: int) -> dict:
    """Fetch and score one changeset. Closed changesets never change, so results are cached."""
    if OFFLINE:
        raise osm_live.UpstreamError("offline mode: live scoring needs the network (restart without --offline)")
    bundle = REPLAYS[LIVE_TASK].bundle
    t0 = time.monotonic()
    fetched = osm_live.fetch_changeset(session, cid, api_name)
    fetch_ms = round(1000 * (time.monotonic() - t0))
    frame = pd.DataFrame([fetched.changeset])
    p, signals = tree_model.score(bundle, frame)
    token_scores = tree_model.edit_token_scores(bundle, [e["diff_tokens"] for e in fetched.edits])
    edits = [{"position": e["position"], "otype": e["otype"], "oid": e["oid"], "action": e["action"],
              "tokens": e["diff_tokens"], "text": e["text"], "old_text": e["old_text"],
              "attention": None, "token_score": float(s)}
             for e, s in zip(fetched.edits, token_scores)]
    return {"source": "live", "api": api_name, "task": LIVE_TASK, "cid": cid, "label": None,
            "headline_model": "live_hybrid",
            "models": {"live_hybrid": {"name": "Hybrid trees (live, CPU)", "score": float(p[0]),
                                       "threshold": bundle["threshold"], "headline": True,
                                       "test_pr_auc": round(bundle["test_pr_auc"], 4), "seeds": []}},
            "changeset": changeset_summary(pd.Series(fetched.changeset)), "signals": signals[0],
            "local_hybrid": float(p[0]), "reasons": tree_model.token_reasons(bundle, fetched.changeset["diff_tokens"]),
            "edits": edits, "edit_ranking": "token_score", "map": osm_live.display_payload(fetched),
            "timing": {"fetch_ms": fetch_ms, "total_ms": round(1000 * (time.monotonic() - t0))}}


class Feed:
    """Newest closed changesets on the real OSM, scored in the background while watched."""

    def __init__(self):
        self.items: list[dict] = []
        self.seen: set[int] = set()
        self.last_request = 0.0
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def snapshot(self) -> list[dict]:
        self.last_request = time.monotonic()
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
            return list(self.items)

    def run(self) -> None:
        while time.monotonic() - self.last_request < FEED_IDLE_STOP_SECONDS:
            self.poll()
            time.sleep(FEED_POLL_SECONDS)

    def poll(self) -> None:
        r = session.get(f"{OSM_API}/changesets", params={"closed": "true", "limit": 20})
        if not r.ok:
            return
        try:
            metas = parse_changeset_metas(r.text)
        except ET.ParseError:
            return
        fresh = [m for m in metas if m.cid not in self.seen][:FEED_NEW_PER_POLL]
        for meta in fresh:
            self.seen.add(meta.cid)
            try:
                d = live_detail("osm", meta.cid)
            except (osm_live.ChangesetNotFound, osm_live.UpstreamError):
                continue
            item = {"cid": meta.cid, "score": d["models"]["live_hybrid"]["score"],
                    "threshold": d["models"]["live_hybrid"]["threshold"],
                    "created_at": meta.created_at, "comment": meta.comment[:80],
                    "n_edits": d["changeset"].get("n_edits"), "editor": meta.editor[:40]}
            with self.lock:
                self.items = [item, *self.items][:FEED_KEEP]


REPLAYS: dict[str, Replay] = {}


def live_answer(api_name: str, cid: int) -> dict:
    """live_detail, marked as cached when the answer came from its cache (no new fetch)."""
    misses = live_detail.cache_info().misses
    detail = live_detail(api_name, cid)
    if live_detail.cache_info().misses == misses:
        return {**detail, "timing": {"cached": True}}
    return detail


def replay(task: str) -> Replay:
    if task not in REPLAYS:
        raise NotInReplay(f"no replay data for task {task!r}")
    return REPLAYS[task]
FEED = Feed()
CID = r"(\d{1,12})"
ROUTES = [
    (re.compile(r"^/api/config$"), lambda: config()),
    (re.compile(r"^/api/replay/(\w+)$"), lambda task: replay(task).listing()),
    (re.compile(rf"^/api/replay/(\w+)/{CID}$"), lambda task, cid: replay(task).detail(int(cid))),
    (re.compile(rf"^/api/live/(osm|sandbox)/{CID}$"), lambda api, cid: live_answer(api, int(cid))),
    (re.compile(r"^/api/feed$"), lambda: [] if OFFLINE else FEED.snapshot()),
    (re.compile(r"^/api/metrics/(\w+)$"), lambda task: replay(task).metrics()),
]


def config() -> dict:
    live = REPLAYS[LIVE_TASK].bundle
    return {"tasks": {t: {"models": r.models, "headline": r.headline} for t, r in REPLAYS.items()},
            "scenarios": json.loads((DATA / "scenarios.json").read_text()),
            "live": {"task": LIVE_TASK, "threshold": live["threshold"],
                     "test_pr_auc": round(live["test_pr_auc"], 4)},
            "apis": {"osm": "https://www.openstreetmap.org",
                     "sandbox": "https://master.apis.dev.openstreetmap.org"},
            "offline": OFFLINE, "google_maps_key": None if OFFLINE else google_maps_key(),
            "round": ROUND4.name, "ovid_published": OVID_PUBLISHED}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self) -> None:
        # Loopback binding is not enough on its own: a web page can point a rebound hostname
        # at 127.0.0.1 and use this server to drive OSM API calls. Only local names pass.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in LOCAL_HOSTS:
            return self.send_json(HTTPStatus.FORBIDDEN, {"error": "local use only"})
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return super().do_GET()
        for pattern, handler in ROUTES:
            m = pattern.match(path)
            if m:
                return self.answer(handler, m.groups())
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "no such route"})

    def answer(self, handler, args) -> None:
        try:
            self.send_json(HTTPStatus.OK, handler(*args))
        except NotInReplay as exc:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except osm_live.ChangesetNotFound as exc:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except osm_live.UpstreamError as exc:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - last resort, logged with the error
            # A demo page must always get an answer it can show, never a dropped socket.
            self.log_error("unhandled %r", exc)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {exc!r}"})

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: HTTPStatus, body) -> None:
        blob = json.dumps(body, default=jsonable, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, fmt, *args) -> None:
        if "/api/feed" not in self.path:
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--offline", action="store_true",
                    help="no network: replay and recorded maps only (fallback for a demo room)")
    args = ap.parse_args()
    global OFFLINE
    OFFLINE = args.offline
    for task in TASKS:
        REPLAYS[task] = Replay(task)
    # Loopback only: the demo is for this laptop, and the API has no authentication.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MapGuard demo on http://127.0.0.1:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
