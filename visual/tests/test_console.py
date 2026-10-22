"""Checks for the review-console additions: overview metrics, offline mode, the CLI, the
listing's map points and the Google key handling. Like test_demo.py, every number is compared
with round 4's own files, never with a value the console computed itself.

    visual/.venv/Scripts/python -m pytest visual/tests -m "not network"
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

VISUAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUAL))

import server  # noqa: E402

pytestmark = pytest.mark.skipif(not (VISUAL / "data" / "ovid" / "scores.parquet").exists(),
                                reason="run visual/tree_model.py and visual/export_demo.py first")
SEEDS_JSON = server.ROUND4 / "analysis" / "seeds.json"


@pytest.fixture(scope="module")
def replays():
    for task in server.TASKS:
        server.REPLAYS.setdefault(task, server.Replay(task))
    return server.REPLAYS


@pytest.mark.parametrize("task", server.TASKS)
def test_overview_metrics_are_round4_analysis_unchanged(replays, task):
    m = replays[task].metrics()
    stored = json.loads(SEEDS_JSON.read_text())["tasks"][task]
    for model, values in stored["models"].items():
        for k in ("pr_auc", "one_per_author", "f1", "accuracy"):
            assert m["round4"][model][k] == values.get(k)
    assert m["vs_hybrid"] == stored["vs_hybrid_seed_averaged"]


@pytest.mark.parametrize("task", server.TASKS)
def test_outcome_counts_partition_the_test_set_at_the_threshold(replays, task):
    r = replays[task]
    m = r.metrics()
    s = r.scores[r.headline].to_numpy()
    y = r.scores["label"].to_numpy()
    t = r.models[r.headline]["threshold"]
    assert sum(m["counts"].values()) == len(y) == m["n"]
    assert m["counts"]["caught"] == int(((s >= t) & (y == 1)).sum())
    assert m["counts"]["missed"] == int(((s < t) & (y == 1)).sum())


def test_ovid_counts_match_the_error_analysis(replays):
    # experiments/rounds/r4_*/analysis/error_analysis.md: 382 false positives, 258 false negatives
    c = replays["ovid"].metrics()["counts"]
    assert (c["false_alarm"], c["missed"]) == (382, 258)


@pytest.mark.parametrize("task", server.TASKS)
def test_map_points_come_from_each_changesets_own_edits(replays, task):
    """A point is the median of the changeset's own edit coordinates; changesets with none
    (tag-only way/relation edits: 457 of 3,607 on ovid) get no point and are not drawn."""
    r = replays[task]
    rows = r.listing()
    edits = pd.read_parquet(VISUAL / "data" / task / "edits.parquet").dropna(subset=["lat", "lon"])
    with_coords = set(edits["cid"].astype(int))
    assert all((row["point"] is not None) == (row["cid"] in with_coords) for row in rows)
    placed = [row for row in rows if row["point"] is not None]
    sample = placed[:: max(1, len(placed) // 25)]
    for row in sample:
        e = edits[edits["cid"] == row["cid"]]
        assert row["point"] == (round(float(np.median(e["lat"])), 4), round(float(np.median(e["lon"])), 4))


def test_offline_mode_uses_only_the_cache_and_refuses_live(replays, monkeypatch):
    monkeypatch.setattr(server, "OFFLINE", True)
    cached = next(server.osm_live.GEOMETRY_DIR.glob("osm_*.json"), None)
    if cached is None:
        pytest.skip("no recorded maps (run visual/precache_maps.py)")
    cid = int(cached.stem.split("_")[1])
    assert server.replay_map(cid) == json.loads(cached.read_text())
    assert "offline" in server.replay_map(1)["error"]
    with pytest.raises(server.osm_live.UpstreamError, match="offline"):
        server.live_detail.__wrapped__("osm", 1)
    assert server.config()["offline"] is True and server.config()["google_maps_key"] is None
    feed = next(h for pat, h in server.ROUTES if pat.pattern == r"^/api/feed$")
    assert feed() == []


def test_google_key_comes_from_env_or_an_ignored_file(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key-123")
    assert server.google_maps_key() == "test-key-123"
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY")
    (tmp_path / ".env").write_text('OTHER=1\nGOOGLE_MAPS_API_KEY="from-file-456"\n', encoding="utf-8")
    monkeypatch.setattr(server.tree_model, "VISUAL", tmp_path)
    assert server.google_maps_key() == "from-file-456"
    ignored = subprocess.run(["git", "check-ignore", "-q", "visual/.env"], cwd=VISUAL.parent)
    assert ignored.returncode == 0, "visual/.env must be git-ignored"


def test_every_route_refuses_a_foreign_host_and_nothing_is_cached(replays):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        for path in ("/api/config", "/api/metrics/ovid", "/api/replay/ovid"):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Host": "evil.example"})
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req)
            assert err.value.code == 403
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/metrics/ovid") as r:
            assert r.headers["Cache-Control"] == "no-store"
            assert json.loads(r.read())["n"] == 3607
    finally:
        httpd.shutdown()


def test_cli_prints_the_server_score(replays):
    detail = replays["ovid"].detail(58688771)
    m = detail["models"][detail["headline_model"]]
    out = subprocess.run([sys.executable, str(VISUAL / "score.py"), "58688771", "--json"],
                         capture_output=True, text=True, cwd=VISUAL, check=True).stdout
    cli = json.loads(out)
    assert cli["models"][cli["headline_model"]]["score"] == pytest.approx(m["score"], abs=1e-12)
