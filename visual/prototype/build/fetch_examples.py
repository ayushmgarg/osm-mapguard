"""Fetch example changesets from the public OSM API for the demo page.

Picks the most confidently damaging and most confidently clean test changesets (ovid task,
round-4 stack + embeddings, seed-averaged) plus a few confident mistakes, then downloads their
edits and previous object versions so the page can show what each edit did, where it was, and
which edit the deep model attended to. No user ids or usernames are kept.

    python fetch_examples.py <repo> <build dir>
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO, BUILD = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(REPO / "src")]
import os  # noqa: E402

os.environ["MAPGUARD_CONTACT"] = "course demo; https://github.com/ayushmgarg/osm-mapguard"
from mapguard.config import OSM_API  # noqa: E402
from mapguard.data.http import PoliteSession  # noqa: E402
from mapguard.data.osm_api import fetch_versioned, parse_changeset_metas  # noqa: E402
from mapguard.features.build_store import is_text_key  # noqa: E402
from mapguard.features.edit_tokens import edit_tokens  # noqa: E402

PER_GROUP = {"tp": 9, "tn": 9, "fp": 4, "fn": 4}
MIN_EDITS, MAX_EDITS = 3, 80

S = json.loads((BUILD / "scores.json").read_text())
att = np.load(BUILD / "deep_attention.npz")
cids = np.array(S["cids"])
y = np.array(S["y"])
p = {m: np.array(v["p"]) for m, v in S["models"].items()}
n_edits = dict(zip(att["cids"].tolist(), att["n"].tolist()))
att_row = {int(c): i for i, c in enumerate(att["cids"])}
ok_size = np.array([MIN_EDITS <= n_edits[int(c)] <= MAX_EDITS for c in cids])

order = np.argsort(-p["stack_emb"])
cand = {
    "tp": [int(cids[i]) for i in order if y[i] == 1 and ok_size[i]][:40],
    "tn": [int(cids[i]) for i in order[::-1] if y[i] == 0 and ok_size[i]][:40],
}
err_csv = REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds/analysis/error_analysis_ovid_stack_emb.csv"
rows = list(csv.DictReader(err_csv.open(encoding="utf-8")))
cand["fp"] = [int(r["cid"]) for r in rows if r["error"] == "false_positive"
              and MIN_EDITS <= int(r["n_edits"]) <= MAX_EDITS][:12]
cand["fn"] = [int(r["cid"]) for r in rows if r["error"] == "false_negative"
              and MIN_EDITS <= int(r["n_edits"]) <= MAX_EDITS][:12]

sess = PoliteSession(1.0)
idx = {int(c): i for i, c in enumerate(cids)}


def parse_change(xml):
    """Edits in document order, with way node refs (osmchange.parse drops them)."""
    root = ET.fromstring(xml)
    edits = []
    for block in root:
        if block.tag not in ("create", "modify", "delete"):
            continue
        for el in block:
            if el.tag not in ("node", "way", "relation"):
                continue
            a = el.attrib
            edits.append({
                "action": block.tag, "otype": el.tag, "oid": int(a["id"]),
                "version": int(a.get("version", 0)),
                "lat": float(a["lat"]) if "lat" in a else None,
                "lon": float(a["lon"]) if "lon" in a else None,
                "tags": {t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")},
                "refs": [int(n.attrib["ref"]) for n in el.findall("nd")],
            })
    return edits


def build(cid, group, meta):
    r = sess.get(f"{OSM_API}/changeset/{cid}/download")
    if not r.ok:
        return None, f"download {r.status}"
    edits = parse_change(r.text)
    want = {"node": [], "way": [], "relation": []}
    for e in edits:
        if e["action"] in ("modify", "delete") and e["version"] > 1:
            want[e["otype"]].append((e["oid"], e["version"] - 1))
    prev = {}
    for ot, refs in want.items():
        if refs:
            for (oid, ver), el in fetch_versioned(sess, ot, refs).items():
                prev[(ot, oid)] = el
    # the store drops edits whose previous version is hidden; keep only changesets with none,
    # so attention slot k is exactly the k-th edit in document order
    missing = sum(1 for ot, refs in want.items() for oid, _ in refs if (ot, oid) not in prev)
    if missing:
        return None, f"{missing} hidden previous versions"
    if len(edits) != n_edits[cid]:
        return None, f"edit count {len(edits)} != store {n_edits[cid]}"

    node_pos = {e["oid"]: (e["lat"], e["lon"]) for e in edits if e["otype"] == "node" and e["lat"] is not None}
    a = att["att"][att_row[cid]][: len(edits)].astype(float)
    out_edits, nodes, ways = [], [], []
    for k, e in enumerate(edits):
        pel = prev.get((e["otype"], e["oid"]))
        old_tags = ({t.attrib["k"]: t.attrib["v"] for t in pel.findall("tag")} if pel is not None
                    else ({} if e["action"] == "create" else None))
        toks = edit_tokens(e["action"], e["otype"], old_tags, e["tags"])
        new_text = " ".join(v for kk, v in e["tags"].items() if is_text_key(kk) and v)[:80]
        old_text = " ".join(v for kk, v in (old_tags or {}).items() if is_text_key(kk) and v)[:80]
        out_edits.append({"a": e["action"], "t": e["otype"], "tok": toks[:10],
                          "old": old_text, "new": new_text if e["action"] != "delete" else "",
                          "w": round(float(a[k]), 4)})
        if e["otype"] == "node":
            plat = float(pel.attrib["lat"]) if pel is not None and "lat" in pel.attrib else None
            plon = float(pel.attrib["lon"]) if pel is not None and "lon" in pel.attrib else None
            lat, lon = (e["lat"], e["lon"]) if e["lat"] is not None else (plat, plon)
            if lat is not None:
                nodes.append([k, round(lat, 6), round(lon, 6),
                              None if plat is None else round(plat, 6),
                              None if plon is None else round(plon, 6)])
        elif e["otype"] == "way" and e["refs"]:
            pts = [node_pos[r_] for r_ in e["refs"] if r_ in node_pos]
            if len(pts) >= 2:
                ways.append([k, [[round(la, 6), round(lo, 6)] for la, lo in pts]])
    i = idx[cid]
    return {
        "cid": cid, "group": group, "label": int(y[i]),
        "score": {m: round(float(p[m][i]), 4) for m in p},
        "created_at": meta.get("created_at"), "comment": (meta.get("comment") or "")[:200],
        "editor": (meta.get("editor") or "")[:60], "n_edits": len(edits),
        "edits": out_edits, "nodes": nodes, "ways": ways,
    }, "ok"


all_cids = [c for g in cand.values() for c in g]
metas = {}
for i in range(0, len(all_cids), 100):
    r = sess.get(f"{OSM_API}/changesets", params={"changesets": ",".join(map(str, all_cids[i:i + 100]))})
    for m in parse_changeset_metas(r.text) if r.ok else []:
        metas[m.cid] = {"created_at": m.created_at, "comment": m.comment, "editor": m.editor}

examples, log = [], []
for group, lst in cand.items():
    got = 0
    for cid in lst:
        if got >= PER_GROUP[group]:
            break
        if cid not in metas:
            log.append((group, cid, "no metadata"))
            continue
        try:
            ex, why = build(cid, group, metas[cid])
        except ET.ParseError as exc:
            ex, why = None, f"parse error {exc}"
        log.append((group, cid, why))
        print(group, cid, why, flush=True)
        if ex and (ex["nodes"] or ex["ways"]):
            examples.append(ex)
            got += 1
(BUILD / "examples.json").write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
print(json.dumps({g: sum(1 for e in examples if e["group"] == g) for g in PER_GROUP}))
