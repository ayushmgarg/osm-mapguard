"""User history features, computed strictly from the past.

This module exists because of one mistake that would invalidate every result in the
paper. B4 consumes "the user's past changesets". If those are read from a full dump or a
current API snapshot, then for a 2016 target the model sees that the account was blocked
in 2017 - a perfect predictor of a label derived from that same later history, and
completely useless in production, where the future has not happened yet.

**The rule (PROJECT_PLAN section 5.4, Rule 2): every feature here uses only changesets
with `created_at < target.created_at`.** It is enforced in `point_in_time` below and
asserted by `tests/test_user_history.py`, not left to whoever is editing the file.

The one exception is `account_created`, which is fixed for the life of the account and so
carries no information about what happened after the target. It lives in
`mapguard.data.users`.

Source of history: `GET /api/0.6/changesets?user=<uid>&time=<from>,<to>`. One call per
*user*, not per changeset, and the corpus has roughly four thousand distinct users. The
API returns newest first, capped at 100, so a prolific mapper is truncated - that is
recorded as `history_truncated` rather than hidden, because a truncated history is itself
a signal about the account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from mapguard.config import OSM_API
from mapguard.data.http import PoliteSession

HISTORY_LIMIT = 100      # the API caps here
SEQUENCE_LENGTH = 200    # B4 input cap; see PROJECT_PLAN section 5.4


@dataclass
class PriorChangeset:
    """One earlier changeset by the same user. Deliberately thin."""

    cid: int
    created_at: str
    num_changes: int
    editor: str
    comment_length: int
    bbox_diag_deg: float

    @property
    def ts(self) -> float:
        return _epoch(self.created_at) or 0.0


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso_week(iso: str) -> str:
    ts = _epoch(iso) or 0.0
    y, w, _ = datetime.fromtimestamp(ts, tz=timezone.utc).isocalendar()
    return f"{y}-W{w:02d}"


def parse_changeset_list(xml: str) -> list[PriorChangeset]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out = []
    for el in root.findall("changeset"):
        a = el.attrib
        tags = {t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")}
        try:
            lat_span = abs(float(a["max_lat"]) - float(a["min_lat"]))
            lon_span = abs(float(a["max_lon"]) - float(a["min_lon"]))
            diag = (lat_span**2 + lon_span**2) ** 0.5
        except (KeyError, ValueError):
            diag = 0.0
        out.append(
            PriorChangeset(
                cid=int(a["id"]),
                created_at=a.get("created_at", ""),
                num_changes=int(a.get("changes_count", a.get("num_changes", 0)) or 0),
                editor=tags.get("created_by", ""),
                comment_length=len(tags.get("comment", "")),
                bbox_diag_deg=round(diag, 6),
            )
        )
    return out


def fetch_user_history(
    session: PoliteSession, uid: int, before: str, api: str = OSM_API
) -> tuple[list[PriorChangeset], int]:
    """Changesets by `uid` closed before `before`, newest first, capped at 100.

    The `time=<a>,<b>` form asks for changesets closed in that interval. Passing an open
    lower bound and the target timestamp as the upper bound means the API itself does the
    point-in-time filtering, and `point_in_time` below re-checks it - the API is not the
    place to put a correctness guarantee this important.
    """
    r = session.get(
        f"{api}/changesets",
        params={"user": uid, "time": f"2005-01-01T00:00:00Z,{before}", "limit": HISTORY_LIMIT},
    )
    if not r.ok:
        return [], r.status
    return parse_changeset_list(r.text), r.status


def point_in_time(history: list[PriorChangeset], target_created_at: str) -> list[PriorChangeset]:
    """Filter to strictly-earlier changesets, oldest first.

    Belt and braces over the API filter. If this function is ever bypassed, the model
    starts reading the future and every number in the paper becomes meaningless.
    """
    cutoff = _epoch(target_created_at)
    if cutoff is None:
        return []
    prior = [h for h in history if (h.ts or 0) < cutoff]
    prior.sort(key=lambda h: h.ts)
    return prior


@dataclass
class UserFeatures:
    """Aggregates for B5, plus the ordered sequence B4 consumes."""

    uid: int | None
    n_prior: int
    history_truncated: bool
    days_since_prev: float | None
    days_active_span: float
    changesets_per_active_day: float
    mean_prior_changes: float
    max_prior_changes: int
    total_prior_changes: int
    distinct_editors: int
    frac_prior_with_comment: float
    mean_bbox_diag_deg: float
    max_bbox_diag_deg: float
    active_weeks: int = 0          # Ovid: distinct ISO weeks with an earlier changeset
    sequence: list[tuple[float, float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("sequence")
        return d


def build(
    uid: int | None,
    history: list[PriorChangeset],
    target_created_at: str,
) -> UserFeatures:
    """Point-in-time user features for one target changeset."""
    prior = point_in_time(history, target_created_at)
    cutoff = _epoch(target_created_at) or 0.0

    if not prior:
        # A first-ever edit is not missing data - it is the strongest single fact about
        # this account, and "New mapper" is the top harmful reason in the OSMCha labels.
        return UserFeatures(
            uid=uid,
            n_prior=0,
            history_truncated=len(history) >= HISTORY_LIMIT,
            days_since_prev=None,
            days_active_span=0.0,
            changesets_per_active_day=0.0,
            mean_prior_changes=0.0,
            max_prior_changes=0,
            total_prior_changes=0,
            distinct_editors=0,
            frac_prior_with_comment=0.0,
            mean_bbox_diag_deg=0.0,
            max_bbox_diag_deg=0.0,
            sequence=[],
        )

    changes = [p.num_changes for p in prior]
    span_days = max(0.0, (cutoff - prior[0].ts) / 86400.0)

    seq: list[tuple[float, float, float, float]] = []
    for i, p in enumerate(prior[-SEQUENCE_LENGTH:]):
        gap = (p.ts - prior[max(0, i - 1)].ts) / 86400.0 if i else 0.0
        seq.append((float(p.num_changes), gap, float(p.comment_length), p.bbox_diag_deg))

    return UserFeatures(
        uid=uid,
        n_prior=len(prior),
        history_truncated=len(history) >= HISTORY_LIMIT,
        days_since_prev=round((cutoff - prior[-1].ts) / 86400.0, 4),
        days_active_span=round(span_days, 4),
        changesets_per_active_day=round(len(prior) / max(1.0, span_days), 4),
        mean_prior_changes=round(sum(changes) / len(changes), 3),
        max_prior_changes=max(changes),
        total_prior_changes=sum(changes),
        distinct_editors=len({p.editor for p in prior if p.editor}),
        frac_prior_with_comment=round(
            sum(1 for p in prior if p.comment_length > 0) / len(prior), 4
        ),
        mean_bbox_diag_deg=round(sum(p.bbox_diag_deg for p in prior) / len(prior), 6),
        max_bbox_diag_deg=round(max(p.bbox_diag_deg for p in prior), 6),
        active_weeks=len({_iso_week(p.created_at) for p in prior if p.created_at}),
        sequence=seq,
    )
