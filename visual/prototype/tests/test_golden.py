"""Golden-fixture tests: what the dashboard shows must match the recorded expectations.

    .venv/Scripts/python -m pytest -q tests
    MAPGUARD_NETWORK_TESTS=1 .venv/Scripts/python -m pytest -q tests     # also the live CLI/dashboard check

fixtures/golden.json holds, per demo changeset: the score, the top-ranked edit, the number of map
points and the signal bars. Everything except the network test runs from a clean checkout with
no network and no dataset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

DEMO = Path(__file__).resolve().parents[1]
LIVE = DEMO / "live"
FIX = DEMO / "fixtures"
sys.path.insert(0, str(LIVE))

import core  # noqa: E402
import replay  # noqa: E402
import server  # noqa: E402

GOLDEN = json.loads((FIX / "golden.json").read_text(encoding="utf-8"))
TOL = GOLDEN["tolerance"]


def top_index(d: dict) -> int:
    key = (lambda e: e["w"] or 0) if d["rank_by"] == "attention" else (lambda e: e["push"])
    return max(range(len(d["edits"])), key=lambda i: key(d["edits"][i]))


def check(d: dict, g: dict, score_key: str):
    assert abs(d["score"][score_key] - g["score"]) <= TOL, (d["cid"], d["score"][score_key], g["score"])
    assert top_index(d) == g["top_edit"], d["cid"]
    assert d["edits"][g["top_edit"]]["tok"][:6] == g["top_tokens"], d["cid"]
    assert len(d["nodes"]) == g["n_nodes"] and len(d["ways"]) == g["n_ways"], d["cid"]
    got = {b["key"]: b["delta"] for b in d["signals"]["bars"]}
    for k, v in g["signals"].items():
        assert abs(got[k] - v) <= 1e-4, (d["cid"], k, got[k], v)


@pytest.fixture(scope="module")
def offline_server():
    handler, live = server.build(SimpleNamespace(offline=True, rate=1.0, per_minute=0, no_live=True))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# --------------------------------------------------------------- offline (clean checkout)
def test_dashboard_replays_every_scenario_exactly(offline_server):
    status, cfg = get(f"{offline_server}/api/config")
    assert status == 200 and cfg["offline"] is True
    for g in GOLDEN["replay"]:
        status, d = get(f"{offline_server}/api/test/{g['cid']}")
        assert status == 200
        check(d, g, "stack_emb")


def test_offline_server_never_fetches(offline_server):
    status, d = get(f"{offline_server}/api/score/190804354")
    assert status == 503 and "offline" in d["error"]
    status, live = get(f"{offline_server}/api/live?since=0")
    assert len(live["items"]) == len(GOLDEN["live"]) and all(i["recorded"] for i in live["items"])
    for g in GOLDEN["live"]:
        status, d = get(f"{offline_server}/api/live/{g['cid']}")
        check(d, g, "live")


def test_recorded_live_results_rescore_identically():
    """The live scorer and explainer reproduce every recorded score from its feature row."""
    scorer = core.load_scorer()
    ex = core.load_explainer(scorer)
    items = {it["cid"]: it for it in json.loads((FIX / "live.json").read_text(encoding="utf-8"))}
    for g in GOLDEN["live"]:
        row = items[g["cid"]]["row"]
        p = float(scorer.score_frame(pd.DataFrame([row]))[0])
        assert abs(p - g["score"]) <= TOL, (g["cid"], p, g["score"])
        bars = {b["key"]: b["delta"] for b in ex.signals(row)["bars"]}
        for k, v in g["signals"].items():
            assert abs(bars[k] - v) <= 1e-4


def test_fixtures_carry_no_author_identity():
    for f in ("replay.json", "live.json", "golden.json"):
        text = (FIX / f).read_text(encoding="utf-8")
        for key in core.PRIVATE + ("display_name",):
            assert f'"{key}"' not in text, (f, key)


# --------------------------------------------------------------- against the source data
@pytest.mark.skipif(not replay.release_available(), reason="dataset release v1.1.0 not unpacked under data/")
def test_fixtures_still_match_the_release():
    ex = core.load_explainer(core.load_scorer())
    _, _, detail = replay.load_from_release(ex)
    for g in GOLDEN["replay"]:
        check(detail(g["cid"]), g, "stack_emb")


# --------------------------------------------------------------- CLI == dashboard, live
@pytest.mark.skipif(os.environ.get("MAPGUARD_NETWORK_TESTS") != "1", reason="set MAPGUARD_NETWORK_TESTS=1 (hits the OSM API)")
def test_cli_and_dashboard_give_the_same_score():
    g = GOLDEN["live"][0]
    out = subprocess.run([sys.executable, str(LIVE / "score.py"), str(g["cid"]), "--json"],
                         capture_output=True, text=True, encoding="utf-8", timeout=300, check=True)
    cli = json.loads(out.stdout)
    engine = core.Engine(rate=1.0)
    live = server.Live(engine, per_minute=0)
    dash = live.score_on_request(g["cid"])          # the function behind /api/score/<cid>
    assert abs(cli["score"]["live"] - dash["score"]["live"]) <= TOL
    assert abs(cli["score"]["live"] - g["score"]) <= TOL, "the changeset or its history changed on OSM since recording"
