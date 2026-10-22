"""Fetch one changeset from an OSM API and turn it into model features plus map geometry.

Separate from the server so the export (replay geometry) and the live path share one fetcher.

Features come from the training feature builder itself (`mapguard.features.build_store`):
previous object versions go into an in-memory copy of the `versions` table that
`drop_hidden` and `version_actions` read, so a live changeset is featurised by the same
lines of code as the 46,578 training rows. Two known differences, both recorded on the row:

  - no Overpass adiff (3 queries/minute is too slow for a live demo), so geometry comes
    from the previous-version route only; `geom_frac_adiff` is 0, as it is for ~67% of
    training rows;
  - account age is not fetched; it was never a model input.

Map geometry is display-only and separate: ways are drawn from their node references, using
this changeset's own node positions where it moved them and today's positions otherwise.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from fetch_osm import MAX_PARAM_CHARS, VERSIONS_SCHEMA

from mapguard.config import MULTIFETCH_MAX, OSM_API
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import fetch_changeset_download, fetch_changeset_meta
from mapguard.data.osmchange import parse_safe
from mapguard.features.build_store import (
    ROUTE_VERSION,
    changeset_row,
    diff_bag,
    drop_hidden,
    edit_rows,
    version_actions,
)
from mapguard.features.geometry import features_for
from mapguard.features.user_history import build as build_user
from mapguard.features.user_history import fetch_user_history

# OSM's official test instance: edits there touch no real map data.
SANDBOX_API = "https://master.apis.dev.openstreetmap.org/api/0.6"
APIS = {"osm": OSM_API, "sandbox": SANDBOX_API}
PLURAL = {"node": "nodes", "way": "ways", "relation": "relations"}
# Display only: a mass edit can reference tens of thousands of nodes, and the demo must not
# spend minutes of a volunteer API's capacity drawing one changeset.
MAX_NODE_LOOKUPS = 3000
GEOMETRY_DIR = Path(__file__).resolve().parent / "data" / "geometry"
# The API is telling us to slow down or is failing: never mistake these for a hidden element.
RETRYABLE = (0, 429, 509)


class ChangesetNotFound(Exception):
    pass


class UpstreamError(Exception):
    pass


@dataclass
class Fetched:
    cid: int
    api: str
    changeset: dict                      # one changesets.parquet row
    edits: list[dict]                    # edits.parquet rows (the sampled <= 256)
    display: list[dict] = field(default_factory=list)   # per-edit tags and geometry for the map
    bbox: list[float] | None = None
    geometry_partial: bool = False


# --------------------------------------------------------------------------- fetching


def fetch_elements(session: PoliteSession, api: str, otype: str, ids: list[str]) -> tuple[list, dict]:
    """Multi-fetch `ids` ("123" or "123v4"). Returns (elements, {id: failing status}).

    One hidden or deleted element fails a whole batch with 4xx, so a failed batch is split
    until the culprits stand alone; their status is what `drop_hidden` reads.
    """
    if not ids:
        return [], {}
    r = session.get(f"{api}/{PLURAL[otype]}", params={PLURAL[otype]: ",".join(ids)})
    if r.ok:
        return ET.fromstring(r.text).findall(otype), {}
    if r.status in RETRYABLE or r.status >= 500:
        raise UpstreamError(f"{otype} fetch failed with HTTP {r.status}")
    if len(ids) == 1:
        return [], {ids[0]: r.status}
    mid = len(ids) // 2
    a_els, a_bad = fetch_elements(session, api, otype, ids[:mid])
    b_els, b_bad = fetch_elements(session, api, otype, ids[mid:])
    return a_els + b_els, {**a_bad, **b_bad}


def fetch_in_batches(session: PoliteSession, api: str, otype: str, ids: list[str]) -> tuple[list, dict]:
    elements, failed, batch, length = [], {}, [], 0
    for i in [*ids, None]:
        full = len(batch) >= MULTIFETCH_MAX or length + len(i or "") + 1 > MAX_PARAM_CHARS
        if i is None or (batch and full):
            got, bad = fetch_elements(session, api, otype, batch)
            elements += got
            failed.update(bad)
            batch, length = [], 0
        if i is not None:
            batch.append(i)
            length += len(i) + 1
    return elements, failed


def version_row(otype: str, el: ET.Element) -> tuple:
    """One `versions` row, in the column order scripts/fetch_osm.py writes."""
    a = el.attrib
    n_refs = len(el.findall("nd")) if otype == "way" else len(el.findall("member"))
    return (otype, int(a["id"]), int(a["version"]), 200, 0 if a.get("visible") == "false" else 1,
            float(a["lat"]) if "lat" in a else None, float(a["lon"]) if "lon" in a else None,
            n_refs, json.dumps({t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")}),
            a.get("timestamp"), int(a["uid"]) if a.get("uid", "").isdigit() else None,
            int(a["changeset"]) if a.get("changeset", "").isdigit() else None)


def previous_versions(session: PoliteSession, api: str, edits) -> tuple[sqlite3.Connection, dict]:
    """In-memory `versions` table for every modify/delete, plus old way node refs for the map."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(VERSIONS_SCHEMA)
    old_way_refs: dict[int, list[int]] = {}
    for otype in PLURAL:
        wanted = sorted({(e.oid, e.version - 1) for e in edits if e.otype == otype
                         and e.action in ("modify", "delete") and e.version and e.version > 1})
        elements, failed = fetch_in_batches(session, api, otype, [f"{i}v{v}" for i, v in wanted])
        rows = [version_row(otype, el) for el in elements]
        for ref, status in failed.items():
            oid, version = ref.split("v")
            rows.append((otype, int(oid), int(version), status) + (None,) * 8)
        conn.executemany("INSERT OR REPLACE INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        if otype == "way":
            old_way_refs = {int(el.attrib["id"]): [int(nd.attrib["ref"]) for nd in el.findall("nd")]
                            for el in elements}
    return conn, old_way_refs


# --------------------------------------------------------------------------- geometry


def new_way_refs(osmchange_xml: str) -> dict[int, list[int]]:
    root = ET.fromstring(osmchange_xml)
    return {int(w.attrib["id"]): [int(nd.attrib["ref"]) for nd in w.findall("nd")]
            for block in root for w in block.findall("way")}


def node_positions(session: PoliteSession, api: str, edits, vconn, refs: set[int]) -> tuple[dict, dict, bool]:
    """(before, after) position per node id. Nodes this changeset touched use its own
    coordinates; every other referenced node is unchanged by it, so today's position serves
    for both sides."""
    after = {e.oid: (e.lat, e.lon) for e in edits
             if e.otype == "node" and e.lat is not None and e.action != "delete"}
    before = {oid: (lat, lon) for oid, lat, lon in vconn.execute(
        "SELECT oid, lat, lon FROM versions WHERE otype='node' AND status=200 AND lat IS NOT NULL")}
    touched = {e.oid for e in edits if e.otype == "node"}
    untouched = sorted(refs - touched)
    lookups = [str(i) for i in untouched[:MAX_NODE_LOOKUPS]]
    try:
        elements, failed = fetch_in_batches(session, api, "node", lookups)
    except UpstreamError:
        # Display only: a map with fewer nodes beats no score at all.
        elements, failed = [], dict.fromkeys(lookups, 0)
    partial = len(untouched) > MAX_NODE_LOOKUPS or bool(failed)
    for el in elements:
        if "lat" in el.attrib:
            pos = (float(el.attrib["lat"]), float(el.attrib["lon"]))
            before.setdefault(int(el.attrib["id"]), pos)
            after.setdefault(int(el.attrib["id"]), pos)
    return before, after, partial


def line(refs: list[int] | None, positions: dict) -> list[list[float]] | None:
    if not refs:
        return None
    pts = [list(positions[r]) for r in refs if r in positions]
    return pts or None


def display_edits(edit_rows_: list[dict], edits, old_tags: dict, geo: dict) -> list[dict]:
    out = []
    for row in edit_rows_:
        e = edits[row["position"]]      # position, not (otype, oid): an object can be edited twice
        key = (e.otype, e.oid)
        if e.otype == "node":
            before = [list(geo["before"][e.oid])] if e.action != "create" and e.oid in geo["before"] else None
            after = [[e.lat, e.lon]] if e.lat is not None and e.action != "delete" else None
        elif e.otype == "way":
            before = line(geo["old_refs"].get(e.oid), geo["before"]) if e.action != "create" else None
            after = line(geo["new_refs"].get(e.oid), geo["after"]) if e.action != "delete" else None
        else:
            before = after = None
        out.append({"position": row["position"], "action": e.action, "otype": e.otype, "oid": e.oid,
                    "version": e.version, "tokens": row["diff_tokens"],
                    "tags_before": old_tags.get(key, {} if e.action == "create" else None),
                    "tags_after": {} if e.action == "delete" else dict(e.tags),
                    "before": before, "after": after})
    return out


# ----------------------------------------------------------------------------- main


def fetch_changeset(session: PoliteSession, cid: int, api_name: str = "osm") -> Fetched:
    api = APIS[api_name]
    meta, r = fetch_changeset_meta(session, cid, api)
    if r.status == 404:
        raise ChangesetNotFound(f"changeset {cid} not found on {api_name}")
    if meta is None:
        raise UpstreamError(f"changeset {cid} metadata: HTTP {r.status}")
    if meta.closed_at is None:
        raise UpstreamError(f"changeset {cid} is still open; score it once it closes")
    dl = fetch_changeset_download(session, cid, api)
    if not dl.ok:
        raise UpstreamError(f"changeset {cid} download: HTTP {dl.status}")
    edits, err = parse_safe(dl.text)
    if err:
        raise UpstreamError(f"changeset {cid}: {err}")

    vconn, old_way_refs = previous_versions(session, api, edits)
    edits, _ = drop_hidden(edits, vconn)
    if not edits:
        # Training drops these too (build_store: "dropped_empty"): there is nothing to judge.
        raise UpstreamError(f"changeset {cid} has no edits whose history OSM still serves")
    prev_meta: dict = {}
    actions, prev_refs, _ = version_actions(edits, vconn, set(), prev_meta)
    routes = {(a.otype, a.oid): ROUTE_VERSION for a in actions}
    geom_by_key = {(a.otype, a.oid): features_for(a) for a in actions}
    old_tags = {(a.otype, a.oid): (a.old.tags if a.old else {}) for a in actions}

    history, status = (fetch_user_history(session, meta.uid, meta.created_at, api)
                       if meta.uid is not None else ([], 0))
    meta_row = {"created_at": meta.created_at, "uid": meta.uid, "num_changes": meta.num_changes,
                "comment": meta.comment, "editor": meta.editor, "min_lat": meta.min_lat,
                "min_lon": meta.min_lon, "max_lat": meta.max_lat, "max_lon": meta.max_lon,
                "tags_json": json.dumps(meta.tags)}
    row = changeset_row(cid, -1, "live", meta_row, edits, actions, routes, None, history,
                        status == 200, build_user(meta.uid, history, meta.created_at or ""),
                        prev_meta, {"r_stratum": None, "revert_mentions_vandalism": None,
                                    "diff_tokens": diff_bag(edits, old_tags)})
    rows = edit_rows(cid, edits, geom_by_key, routes, prev_refs, prev_meta, old_tags) if edits else []

    new_refs = new_way_refs(dl.text)
    shown = {(r_["otype"], r_["oid"]) for r_ in rows}
    refs = {n for w, ns in {**old_way_refs, **new_refs}.items() if ("way", w) in shown for n in ns}
    before, after, partial = node_positions(session, api, edits, vconn, refs)
    geo = {"before": before, "after": after, "old_refs": old_way_refs, "new_refs": new_refs}
    return Fetched(cid=cid, api=api_name, changeset=row, edits=rows,
                   display=display_edits(rows, edits, old_tags, geo),
                   bbox=list(meta.bbox) if meta.bbox else None, geometry_partial=partial)


def display_payload(fetched: Fetched) -> dict:
    return {"display": fetched.display, "bbox": fetched.bbox,
            "geometry_partial": fetched.geometry_partial}


def cached_display(session: PoliteSession, cid: int) -> dict:
    """Map geometry for a replayed (real OSM) changeset, fetched once and kept on disk.

    A closed changeset never changes, so the cache needs no expiry.
    """
    path = GEOMETRY_DIR / f"osm_{cid}.json"
    if path.exists():
        return json.loads(path.read_text())
    payload = display_payload(fetch_changeset(session, cid, "osm"))
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload
