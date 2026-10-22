"""Typed wrappers over the three OSM API 0.6 endpoints this project uses.

    /changeset/<id>                      metadata: who, when, bbox, comment, editor
    /changeset/<id>/download             osmChange: the actual edits
    /{nodes,ways,relations}?ids=<id>v<n> previous object states, up to 725 per request

The third one is the correction from the plan review. PROJECT_PLAN v1 assumed a
previous object version costs one request each, and concluded that geometry for the
2014-2019 labels was unaffordable. It is not: the multi-fetch endpoints accept a
version suffix, so 725 previous states cost one call. Verified 2026-09-04, see
results/gate0d_review_probes.json probe P1.

Everything here returns plain dataclasses or dicts. Nothing writes to disk - storage
policy belongs to the caller, not to the transport layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from mapguard.config import MULTIFETCH_MAX, OSM_API
from mapguard.data.http import PoliteSession, Response

_PLURAL = {"node": "nodes", "way": "ways", "relation": "relations"}


@dataclass
class ChangesetMeta:
    """Changeset metadata. Everything the B5 tabular branch and baseline B0 need."""

    cid: int
    created_at: str | None
    closed_at: str | None
    uid: int | None
    user: str | None
    num_changes: int | None
    min_lat: float | None
    min_lon: float | None
    max_lat: float | None
    max_lon: float | None
    comments_count: int | None
    tags: dict[str, str]

    @property
    def has_bbox(self) -> bool:
        return None not in (self.min_lat, self.min_lon, self.max_lat, self.max_lon)

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """(south, west, north, east) - the order Overpass expects."""
        if not self.has_bbox:
            return None
        return (self.min_lat, self.min_lon, self.max_lat, self.max_lon)  # type: ignore[return-value]

    @property
    def comment(self) -> str:
        return self.tags.get("comment", "")

    @property
    def editor(self) -> str:
        return self.tags.get("created_by", "")


def _f(attrs: dict, key: str) -> float | None:
    v = attrs.get(key)
    return float(v) if v not in (None, "") else None


def _i(attrs: dict, key: str) -> int | None:
    v = attrs.get(key)
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_changeset_meta(xml: str) -> ChangesetMeta | None:
    """Parse a /changeset/<id> response.

    The changeset count attribute is spelled `changes_count` by the API and
    `num_changes` by the replication stream. Accept both, or the feature silently
    becomes null for one of the two sources.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    el = root.find("changeset")
    if el is None:
        return None
    return _meta_from_element(el)


def parse_changeset_metas(xml: str) -> list[ChangesetMeta]:
    """Parse a /changesets?changesets=<id,id,...> response: up to 100 changesets per call."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    return [_meta_from_element(el) for el in root.findall("changeset")]


def _meta_from_element(el) -> ChangesetMeta:
    a = el.attrib
    return ChangesetMeta(
        cid=int(a["id"]),
        created_at=a.get("created_at"),
        closed_at=a.get("closed_at"),
        uid=_i(a, "uid"),
        user=a.get("user"),
        num_changes=_i(a, "changes_count") if "changes_count" in a else _i(a, "num_changes"),
        min_lat=_f(a, "min_lat"),
        min_lon=_f(a, "min_lon"),
        max_lat=_f(a, "max_lat"),
        max_lon=_f(a, "max_lon"),
        comments_count=_i(a, "comments_count"),
        tags={t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")},
    )


def fetch_changeset_meta(
    session: PoliteSession, cid: int, api: str = OSM_API
) -> tuple[ChangesetMeta | None, Response]:
    r = session.get(f"{api}/changeset/{cid}")
    return (parse_changeset_meta(r.text) if r.ok else None), r


def fetch_changeset_download(session: PoliteSession, cid: int, api: str = OSM_API) -> Response:
    """The osmChange document. Parsed by mapguard.data.osmchange."""
    return session.get(f"{api}/changeset/{cid}/download")


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_versioned(
    session: PoliteSession,
    otype: str,
    refs: list[tuple[int, int]],
    api: str = OSM_API,
) -> dict[tuple[int, int], ET.Element]:
    """Fetch specific (id, version) states in batches of up to 725.

    This is the fallback geometry route (plan section 7, Decision 7). It gives previous
    tags and, for nodes, previous coordinates inline. Ways come back as `<nd>` refs with
    no coordinates - resolving those needs the node versions current at that moment,
    which is exactly the work Overpass adiff does for us, so prefer adiff when it works.
    """
    plural = _PLURAL[otype]
    out: dict[tuple[int, int], ET.Element] = {}
    wanted = [(int(i), int(v)) for i, v in refs if v >= 1]

    for batch in _chunks(wanted, MULTIFETCH_MAX):
        param = ",".join(f"{i}v{v}" for i, v in batch)
        r = session.get(f"{api}/{plural}", params={plural: param})
        if not r.ok:
            continue
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            continue
        for el in root.findall(otype):
            key = (int(el.attrib["id"]), int(el.attrib["version"]))
            out[key] = el
    return out


def current_changeset_id(session: PoliteSession) -> int | None:
    """Highest changeset id right now. Used to bound sampling of the modern corpora."""
    r = session.get(f"{OSM_API}/changesets", params={"limit": 1})
    if not r.ok:
        return None
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return None
    ids = [int(c.attrib["id"]) for c in root.findall("changeset")]
    return max(ids) if ids else None
