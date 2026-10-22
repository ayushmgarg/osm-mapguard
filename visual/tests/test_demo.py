"""The demo must show round 4's numbers, not numbers that merely look like them.

The core check reads the round-4 prediction files independently of the export code and
compares them with what the running server sends to the page, over HTTP, for every model.

Needs the exported data and refit models (visual/README.md, steps 1-2). Run from the repo root:

    visual/.venv/bin/python -m pytest visual/tests -m "not network"
"""

from __future__ import annotations

import json
import math
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

VISUAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUAL))

import export_demo
import server

pytestmark = pytest.mark.skipif(not (VISUAL / "data" / "ovid" / "scores.parquet").exists(),
                                reason="run visual/tree_model.py and visual/export_demo.py first")
SAMPLE_PER_TASK = 40
EXACT = 1e-12


def stored_test_scores(task: str, key: str) -> pd.Series:
    """Seed-mean test score per changeset, read straight from round 4's run folders."""
    stage, pattern, fname, _ = export_demo.MODELS[key]
    per_seed = []
    for seed in export_demo.SEEDS:
        (run,) = (export_demo.ROUND4_RUNS / stage).glob(pattern.format(task=task, seed=seed))
        z = np.load(run / fname)
        per_seed.append(pd.Series(z["test_p"].astype("float64"), index=z["test_cids"]))
    return pd.concat(per_seed, axis=1).mean(axis=1)


@pytest.fixture(scope="module")
def base_url():
    server.replay_map = lambda cid: None          # map geometry needs the network; scores do not
    for task in server.TASKS:
        server.REPLAYS[task] = server.Replay(task)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url: str) -> dict:
    with urlopen(url) as r:
        return json.loads(r.read())


def sample_cids(task: str) -> list[int]:
    cids = pd.read_parquet(VISUAL / "data" / task / "scores.parquet")["cid"]
    scenario = [s["cid"] for s in json.loads((VISUAL / "data" / "scenarios.json").read_text())
                if s["task"] == task]
    return scenario + cids.sample(SAMPLE_PER_TASK, random_state=0).tolist()


@pytest.mark.parametrize("task", export_demo.TASKS)
def test_served_score_equals_stored_output(base_url, task):
    stored = {key: stored_test_scores(task, key) for key in export_demo.MODELS}
    # The ensemble has no stored rows; its rebuild is pinned to the recorded PR-AUC below.
    stored["ensemble"] = pd.concat([export_demo.ensemble_scores(task, s)["test"]
                                    for s in export_demo.SEEDS], axis=1).mean(axis=1)
    for cid in sample_cids(task):
        detail = get(f"{base_url}/api/replay/{task}/{cid}")
        for key, scores in stored.items():
            assert abs(detail["models"][key]["score"] - scores[cid]) < EXACT, (task, cid, key)


@pytest.mark.parametrize("task", export_demo.TASKS)
def test_listing_shows_every_test_changeset_at_its_stored_score(base_url, task):
    """The picker list covers the whole test set, every row at the headline model's score."""
    headline = export_demo.HEADLINE_MODEL[task]
    if headline == "ensemble":
        stored = pd.concat([export_demo.ensemble_scores(task, s)["test"] for s in export_demo.SEEDS],
                           axis=1).mean(axis=1)
    else:
        stored = stored_test_scores(task, headline)
    listing = get(f"{base_url}/api/replay/{task}")
    assert sorted(r["cid"] for r in listing) == sorted(stored.index.tolist())
    for row in listing:
        assert abs(row["score"] - stored[row["cid"]]) <= 5e-5     # listing rounds to 4 places


@pytest.mark.parametrize("task", export_demo.TASKS)
def test_served_label_equals_stored_label(base_url, task):
    (run,) = (export_demo.ROUND4_RUNS / "baselines").glob(f"*_{task}_R1b_s{export_demo.SEEDS[0]}")
    z = np.load(run / "predictions_hybrid.npz")
    labels = dict(zip(z["test_cids"].tolist(), z["test_y"].tolist()))
    for cid in sample_cids(task):
        assert get(f"{base_url}/api/replay/{task}/{cid}")["label"] == labels[cid]


@pytest.mark.parametrize("task", export_demo.TASKS)
@pytest.mark.parametrize("seed", export_demo.SEEDS)
def test_rebuilt_ensemble_reproduces_recorded_pr_auc(task, seed):
    export_demo.ensemble_scores(task, seed)       # raises SystemExit on a mismatch


@pytest.mark.parametrize("task", export_demo.TASKS)
def test_attention_is_a_distribution_over_each_changesets_edits(task):
    edits = pd.read_parquet(VISUAL / "data" / task / "edits.parquet", columns=["cid", "attention"])
    sums = edits.groupby("cid")["attention"].sum()
    assert np.allclose(sums, 1.0, atol=0.01)      # stored as float16 by the deep model


@pytest.mark.parametrize("task", export_demo.TASKS)
def test_signal_bars_add_up_to_the_tree_score(task):
    cs = pd.read_parquet(VISUAL / "data" / task / "changesets.parquet",
                         columns=["local_hybrid", "signals_json"]).head(200)
    for p, blob in zip(cs["local_hybrid"], cs["signals_json"]):
        s = json.loads(blob)
        logit = math.log(p / (1 - p))
        assert abs(s["who"] + s["what"] + s["text"] + s["bias"] - logit) < 1e-2


def test_unknown_routes_and_ids_are_404(base_url):
    for path in ("/api/replay/ovid/1", "/api/replay/nope", "/api/nothing"):
        with pytest.raises(Exception) as err:
            get(base_url + path)
        assert "404" in str(err.value)


@pytest.mark.network
def test_live_features_match_the_feature_store():
    """A stored modern changeset, fetched again today, gets the same features and score."""
    import osm_live
    import tree_model

    from mapguard.data.http import PoliteSession
    from mapguard.training.dataset import TABULAR_FEATURES

    bundle = tree_model.load("ovid_modern")
    cs = pd.read_parquet(VISUAL / "data" / "ovid_modern" / "changesets.parquet")
    rows = cs[cs["corpus"] == "R"].sample(3, random_state=1)
    session = PoliteSession(4.0)
    for _, row in rows.iterrows():
        live = pd.DataFrame([osm_live.fetch_changeset(session, int(row["cid"])).changeset])
        assert live["diff_tokens"].iloc[0] == row["diff_tokens"]
        for col in TABULAR_FEATURES:
            stored = 0.0 if pd.isna(row[col]) else float(row[col])
            assert math.isclose(stored, float(live[col].iloc[0] or 0.0), rel_tol=1e-3, abs_tol=1e-6), col
        p_live, _ = tree_model.score(bundle, live)
        p_stored, _ = tree_model.score(bundle, cs[cs["cid"] == row["cid"]])
        assert abs(p_live[0] - p_stored[0]) < 1e-6
