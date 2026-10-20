"""score_changeset(cid): the one scoring path shared by the CLI (live/score.py) and the dashboard
(live/server.py). Fetch -> features (the repo's own builder) -> CPU live scorer -> explanation.

Everything the page shows about a live changeset comes out of `Engine.score_meta`, so a number
on screen and a number printed by the CLI are produced by the same code.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # demo/live -> repository root
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "train")]
os.environ.setdefault("MAPGUARD_CONTACT", "course demo; https://github.com/ayushmgarg/osm-mapguard")

from mapguard.config import OSM_API as OSM_API_URL  # noqa: E402,F401  (server polls through it)
from mapguard.data.http import PoliteSession  # noqa: E402
from mapguard.data.osm_api import ChangesetMeta, fetch_changeset_meta, parse_changeset_metas  # noqa: E402,F401
from mapguard.features.geometry import haversine_m  # noqa: E402

import train_scorer  # noqa: E402  (pickle needs the Scorer class importable)
from explain import Explainer  # noqa: E402
from features import featurise  # noqa: E402

sys.modules["__main__"].Scorer = train_scorer.Scorer
MODEL_DIR = HERE / "model"
STATS = MODEL_DIR / "explainer_stats.json"
MAX_CHANGES = 1500
PRIVATE = ("uid", "user_uid", "user")


class ScoreError(Exception):
    """A changeset that cannot be scored, with a reason a person can act on."""

    def __init__(self, msg: str, status: int = 422):
        super().__init__(msg)
        self.status = status


def load_scorer():
    with open(MODEL_DIR / "scorer.pkl", "rb") as fh:
        return pickle.load(fh)


def load_explainer(scorer, store_dir: Path | None = None) -> Explainer:
    """From saved stats when present (no dataset needed), else from the store, then saved."""
    if STATS.exists():
        return Explainer(scorer, stats=json.loads(STATS.read_text()))
    if store_dir is None:
        raise FileNotFoundError(f"{STATS} missing and no feature store given to rebuild it")
    ex = Explainer(scorer, pd.read_parquet(store_dir / "changesets.parquet"))
    STATS.write_text(json.dumps(ex.stats()))
    return ex


def model_info() -> dict:
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime((MODEL_DIR / "scorer.pkl").stat().st_mtime))
    return {"name": "MapGuard CPU live scorer",
            "detail": "hybrid trees re-trained on release v1.1.0 (the round-4 network weights are not "
                      "published); held-out PR-AUC 0.884",
            "version": f"scorer.pkl, trained {mtime}"}


class Engine:
    def __init__(self, rate: float = 1.5, sess: PoliteSession | None = None, store_dir: Path | None = None):
        self.scorer = load_scorer()
        self.explainer = load_explainer(self.scorer, store_dir)
        self.model = model_info()
        self.sess = sess or PoliteSession(rate)

    def score_changeset(self, cid: int) -> dict:
        meta, r = fetch_changeset_meta(self.sess, int(cid))
        if meta is None:
            if r.status in (404, 410):
                raise ScoreError(f"changeset {cid} does not exist on OpenStreetMap", 404)
            raise ScoreError(f"OSM API answered HTTP {r.status} for changeset {cid}", 502)
        return self.score_meta(meta)

    def score_meta(self, meta: ChangesetMeta) -> dict:
        if not meta.num_changes:
            raise ScoreError(f"changeset {meta.cid} is empty")
        if meta.num_changes > MAX_CHANGES:
            raise ScoreError(f"changeset {meta.cid} has {meta.num_changes:,} edits; the demo scores up to "
                             f"{MAX_CHANGES:,} to stay polite to the OSM API")
        t0 = time.monotonic()
        try:
            f = featurise(self.sess, meta)
        except RuntimeError as exc:
            raise ScoreError(f"could not fetch changeset {meta.cid}: {exc}", 502) from exc
        fetch_s = round(time.monotonic() - t0, 2)
        row = f["row"]
        t1 = time.monotonic()
        p = float(self.scorer.score_frame(pd.DataFrame([row]))[0])
        score_ms = round((time.monotonic() - t1) * 1000, 1)
        for e in f["edits"]:
            e["push"] = self.explainer.edit_push(e["tok"])
            e["geo"] = ""
        for k, la, lo, pla, plo in f["nodes"]:
            if pla is not None and (pla, plo) != (la, lo):
                f["edits"][k]["geo"] = f"moved {haversine_m((pla, plo), (la, lo)):,.0f} m"
        for k, line in f["ways"]:
            if not f["edits"][k]["geo"]:
                f["edits"][k]["geo"] = f"{len(line)} nodes"
        if meta.has_bbox:
            lat, lon = (meta.min_lat + meta.max_lat) / 2, (meta.min_lon + meta.max_lon) / 2
            bbox = [meta.min_lat, meta.min_lon, meta.max_lat, meta.max_lon]
        elif f["nodes"]:
            lat = sum(n[1] for n in f["nodes"]) / len(f["nodes"])
            lon = sum(n[2] for n in f["nodes"]) / len(f["nodes"])
            bbox = None
        else:
            raise ScoreError(f"changeset {meta.cid} has no location (no bounding box and no node positions)")
        return {
            "cid": meta.cid, "mode": "live", "score": {"live": round(p, 6)}, "model": self.model,
            "signals": self.explainer.signals(row), "rank_by": "push",
            "ops": {k: int(row.get(f"n_{k}", 0)) for k in ("create", "modify", "delete")},
            "timing": {"fetch_s": fetch_s, "score_ms": score_ms, "scored_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            "threshold": round(float(self.scorer.threshold), 4),
            "lat": round(lat, 5), "lon": round(lon, 5), "bbox": bbox,
            "comment": meta.comment[:300], "editor": meta.editor[:80], "created_at": meta.created_at,
            "closed_at": meta.closed_at, "open": meta.closed_at is None,
            "n_edits": row["n_edits"], "user_n_prior": f["user_n_prior"],
            "user_history_capped": f["user_n_prior"] >= 100,
            "edits": f["edits"], "nodes": f["nodes"], "ways": f["ways"],
            "why": self.scorer.explain(row),
            # the exact feature row, so a recorded result can be re-scored offline (golden test);
            # author ids are dropped: they are not model inputs and must never leave this process
            "row": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.items()
                    if k not in PRIVATE},
        }
