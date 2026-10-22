"""Replay data: Ovid's 3,607 held-out test changesets with the repo's saved round-4 scores.

Two sources, same payloads:
  load_from_release(explainer)  dataset release v1.1.0 + build/ (scores, attention) - the source of truth
  load_from_fixtures(dir)       the small committed fixtures (fixtures/replay.json) for offline use,
                                written from the release by live/build_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
STORE = DEMO / "data/release_v1.1.0/extracted"
BUILD = DEMO / "build"
MODELS = ("stack_emb", "hybrid", "deep")
NOTE = "Ways are drawn only in live mode: the release keeps node positions, not way shapes."


def geo_text(r) -> str:
    """One short geometry delta for an edit row of the feature store."""
    shift = getattr(r, "centroid_shift_m", None)
    if shift is not None and pd.notna(shift) and shift >= 1:
        return f"moved {shift:,.0f} m"
    d = getattr(r, "d_refs", None)
    if d is not None and pd.notna(d) and d:
        return f"{int(d):+d} nodes"
    return ""


def release_available() -> bool:
    return (STORE / "feature_store/changesets.parquet").exists() and (BUILD / "scores.json").exists()


def load_from_release(explainer):
    """Returns (points, meta, detail) where detail(cid) -> payload or None."""
    S = json.loads((BUILD / "scores.json").read_text())
    cids = np.array(S["cids"])
    a = pd.read_parquet(STORE / "corpus_a_changesets.parquet",
                        columns=["changeset_id", "min_lat", "max_lat", "min_lon", "max_lon"]).set_index("changeset_id")
    full = pd.read_parquet(STORE / "feature_store/changesets.parquet")
    idx = {int(c): i for i, c in enumerate(cids)}
    rows = full[full["cid"].isin(set(idx))].set_index("cid")
    ed = pd.read_parquet(STORE / "feature_store/edits.parquet",
                         columns=["cid", "position", "action", "otype", "lat", "lon", "diff_tokens", "text", "old_text",
                                  "centroid_shift_m", "d_refs"])
    ed = ed[ed["cid"].isin(set(idx))]
    att = np.load(BUILD / "deep_attention.npz")
    att_row = {int(c): i for i, c in enumerate(att["cids"])}
    edits_by = {int(c): g.sort_values("position") for c, g in ed.groupby("cid")}

    points = []
    for i, c in enumerate(cids):
        c = int(c)
        lat = lon = None
        if c in a.index and pd.notna(a.at[c, "min_lat"]):
            lat = (a.at[c, "min_lat"] + a.at[c, "max_lat"]) / 2
            lon = (a.at[c, "min_lon"] + a.at[c, "max_lon"]) / 2
        elif c in edits_by and edits_by[c]["lat"].notna().any():
            g = edits_by[c]
            lat, lon = float(g["lat"].mean()), float(g["lon"].mean())
        if lat is None:
            continue
        points.append([c, round(float(lat), 5), round(float(lon), 5), int(S["y"][i]),
                       *[S["models"][m]["p"][i] for m in MODELS]])
    summ = json.loads((BUILD / "summary.json").read_text())
    meta = {"thresholds": {m: v["threshold"] for m, v in S["models"].items()}, "n": len(cids),
            "n_located": len(points), "tokens": summ["tokens"]}
    runs = S["models"]["stack_emb"].get("runs", [])
    model = {"name": "MapGuard round 4: stack + embeddings",
             "detail": "the research result: mean of 3 seeds, scores saved by the training runs",
             "version": (runs[0].split("_stack_")[0] + " (+2 more seeds)") if runs else "round 4"}
    cache: dict = {}

    def detail(c: int) -> dict | None:
        if c in cache:
            return cache[c]
        if c not in idx:
            return None
        i = idx[c]
        g = edits_by.get(c)
        w = att["att"][att_row[c]] if c in att_row else None
        edits, nodes = [], []
        if g is not None:
            for k, r in enumerate(g.itertuples(index=False)):
                toks = str(r.diff_tokens or "").split()
                edits.append({"a": r.action, "t": r.otype, "tok": toks[:10],
                              "old": r.old_text or "", "new": r.text or "",
                              "w": round(float(w[k]), 4) if w is not None and k < len(w) else None,
                              "push": explainer.edit_push(toks), "geo": geo_text(r)})
                if pd.notna(r.lat):
                    nodes.append([k, round(float(r.lat), 6), round(float(r.lon), 6), None, None])
        row = rows.loc[c] if c in rows.index else None
        bbox = None
        if c in a.index and pd.notna(a.at[c, "min_lat"]):
            bbox = [float(a.at[c, k]) for k in ("min_lat", "min_lon", "max_lat", "max_lon")]
        cache[c] = out = {
            "cid": c, "mode": "test", "label": int(S["y"][i]), "model": model,
            "signals": explainer.signals({"cid": c, **row.to_dict()}) if row is not None else None,
            "rank_by": "attention", "bbox": bbox,
            "ops": {k: int(row[f"n_{k}"]) for k in ("create", "modify", "delete")} if row is not None else None,
            "score": {m: S["models"][m]["p"][i] for m in MODELS},
            "threshold": S["models"]["stack_emb"]["threshold"],
            "comment": "" if row is None else str(row["comment"] or ""),
            "editor": "" if row is None else str(row["editor"] or ""),
            "created_at": "" if row is None else str(row["created_at"]),
            "n_edits": None if row is None else int(row["n_edits"]),
            "user_n_prior": None if row is None or pd.isna(row["user_n_prior"]) else int(row["user_n_prior"]),
            "edits": edits, "nodes": nodes, "ways": [], "note": NOTE}
        return out
    return points, meta, detail


def load_from_fixtures(fixdir: Path):
    doc = json.loads((fixdir / "replay.json").read_text(encoding="utf-8"))
    details = {int(k): v for k, v in doc["details"].items()}
    return doc["points"], doc["meta"], details.get
