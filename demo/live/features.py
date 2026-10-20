"""Turn one live changeset into the same feature row the training store has.

Uses the repo's own feature code (mapguard.features.build_store.changeset_row, diff_bag,
user_history.build), so a live changeset is featurised exactly like a training row. The only
difference: the previous object versions come from an in-memory API fetch instead of the
prev_versions.sqlite store.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # demo/live/features.py -> repository root
sys.path.insert(0, str(REPO / "src"))

from mapguard.config import OSM_API  # noqa: E402
from mapguard.data.osm_api import ChangesetMeta, fetch_versioned  # noqa: E402
from mapguard.data.osmchange import parse_safe  # noqa: E402
from mapguard.data.overpass import GeometryAction, GeometryState  # noqa: E402
from mapguard.features.build_store import (  # noqa: E402
    ROUTE_VERSION, changeset_row, diff_bag, is_text_key, prev_tags,
)
from mapguard.features.edit_tokens import edit_tokens  # noqa: E402
from mapguard.features.user_history import build as build_user, fetch_user_history  # noqa: E402

MAX_VERSIONS = 2000      # same cap as scripts/fetch_osm.py VERSIONS_PER_CHANGESET


def way_refs(xml: str) -> list[list[int]]:
    """<nd ref> lists per edit, in document order (osmchange.parse keeps only the count)."""
    out = []
    for block in ET.fromstring(xml):
        if block.tag in ("create", "modify", "delete"):
            for el in block:
                if el.tag in ("node", "way", "relation"):
                    out.append([int(n.attrib["ref"]) for n in el.findall("nd")])
    return out


def featurise(sess, meta: ChangesetMeta) -> dict:
    """Fetch contents, previous versions and author history; return row + map geometry."""
    r = sess.get(f"{OSM_API}/changeset/{meta.cid}/download")
    if not r.ok:
        raise RuntimeError(f"download HTTP {r.status}")
    edits, err = parse_safe(r.text)
    if err or not edits:
        raise RuntimeError(err or "empty changeset")
    refs = way_refs(r.text)

    want: dict[str, list] = {"node": [], "way": [], "relation": []}
    for e in edits:
        if e.action in ("modify", "delete") and e.version and e.version > 1:
            want[e.otype].append((e.oid, e.version - 1))
    prev = {}
    budget = MAX_VERSIONS
    for ot, lst in want.items():
        lst = lst[:budget]
        budget -= len(lst)
        if lst:
            for (oid, _), el in fetch_versioned(sess, ot, lst).items():
                prev[(ot, oid)] = el

    actions, routes, prev_meta, old_tags = [], {}, {}, {}
    for e in edits:
        el = prev.get((e.otype, e.oid))
        if el is None or e.action not in ("modify", "delete"):
            continue
        a = el.attrib
        key = (e.otype, e.oid)
        prev_meta[key] = (a.get("timestamp"), int(a["uid"]) if a.get("uid", "").isdigit() else None)
        tags = {t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")}
        old_tags[key] = tags
        is_node = e.otype == "node"
        old = GeometryState(e.otype, e.oid, e.version - 1,
                            [(float(a["lat"]), float(a["lon"]))] if is_node and "lat" in a else [], tags)
        new = None if e.action == "delete" else GeometryState(
            e.otype, e.oid, e.version, [(e.lat, e.lon)] if is_node and e.lat is not None else [], dict(e.tags))
        actions.append(GeometryAction(e.action, e.otype, e.oid, None, old, new))
        routes[key] = ROUTE_VERSION

    hist, status = fetch_user_history(sess, meta.uid, meta.created_at) if meta.uid else ([], 0)
    uf = build_user(meta.uid, hist, meta.created_at or "")
    m = {"created_at": meta.created_at, "uid": meta.uid, "num_changes": meta.num_changes,
         "comment": meta.comment, "editor": meta.editor, "min_lat": meta.min_lat, "min_lon": meta.min_lon,
         "max_lat": meta.max_lat, "max_lon": meta.max_lon, "tags_json": json.dumps(meta.tags)}
    row = changeset_row(meta.cid, 0, "live", m, edits, actions, routes, None, hist, status == 200, uf,
                        prev_meta, {"diff_tokens": diff_bag(edits, old_tags)})

    # what the page draws: every edit's tokens, text, and position
    node_pos = {e.oid: (e.lat, e.lon) for e in edits if e.otype == "node" and e.lat is not None}
    out_edits, nodes, ways = [], [], []
    for k, e in enumerate(edits[:400]):
        ot = prev_tags(e, old_tags)
        toks = edit_tokens(e.action, e.otype, ot, e.tags)
        out_edits.append({"a": e.action, "t": e.otype, "tok": toks[:10],
                          "old": " ".join(v for kk, v in (ot or {}).items() if is_text_key(kk) and v)[:80],
                          "new": "" if e.action == "delete" else
                          " ".join(v for kk, v in e.tags.items() if is_text_key(kk) and v)[:80]})
        if e.otype == "node":
            el = prev.get(("node", e.oid))
            p = (float(el.attrib["lat"]), float(el.attrib["lon"])) if el is not None and "lat" in el.attrib else None
            here = (e.lat, e.lon) if e.lat is not None else p
            if here:
                nodes.append([k, round(here[0], 6), round(here[1], 6),
                              round(p[0], 6) if p else None, round(p[1], 6) if p else None])
        elif e.otype == "way":
            pts = [node_pos[r_] for r_ in refs[k] if r_ in node_pos]
            if len(pts) >= 2:
                ways.append([k, [[round(a, 6), round(b, 6)] for a, b in pts]])
    return {"row": row, "edits": out_edits, "nodes": nodes, "ways": ways,
            "history_status": status, "user_n_prior": uf.n_prior}
