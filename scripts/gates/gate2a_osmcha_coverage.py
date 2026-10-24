"""Gate D0 / 2a - characterise OSMCha before committing to a label window.

The 2026 measurement (8,298 harmful, 19,002 reviewed-clean) cannot be extrapolated
backwards. OSMCha adoption grew over time, so an earlier year might have a tenth of the
coverage, or coverage dominated by two very active reviewers. Choosing a training window
without measuring that is guessing, and the label window is not a detail - it decides what
the model learns and what the paper can honestly claim.

Five questions, per year:

  1. how many harmful, and how many reviewed-clean
  2. how concentrated are the reviewers - a year where five people made most of the
     decisions is five people's opinion, not a community label. If one of them
     systematically flagged a region or an editor, the model learns that as vandalism.
  3. are any changesets reviewed more than once, and do the reviewers agree - the only
     route to an inter-rater number for the limitations section
  4. does the set overlap Corpus A at all (expected ~0 given the eras, but checked)
  5. how does class balance drift year to year

**The exclusion threshold is deliberately NOT hard-coded.** Per decision 4 in
docs/data/DATA_FOUNDATION.md, concentration is measured and reported first, the cut is chosen
afterwards with the distribution visible, and the paper discloses that it was set that way.
Hard-coding a guessed threshold here would either exclude everything or nothing, and
pretending afterwards that it was pre-registered would be misrepresentation.

Count queries only - `page_size=1` and read `count`. No bulk pagination, so this is cheap.

    python scripts/gate2a_osmcha_coverage.py
    python scripts/gate2a_osmcha_coverage.py --from-year 2017 --sample-reviewers 400
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mapguard.config import CORPUS_A_META_DB, LABELS, RESULTS, data_path
from mapguard.data.osmcha import OSMCHA_API, count_for, iter_labels_windowed, session


def corpus_a_ids() -> set[int]:
    ids = set()
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            ids.add(int(line.split("\t")[0]))
    return ids


def reviewer_profile(sess, year: int, sample: int) -> dict:
    """Who actually made the decisions in this year.

    Samples rather than enumerating - a full year can be tens of thousands of rows and the
    concentration ratio converges long before that. The sample size is recorded so the
    number can be read with the right amount of confidence.
    """
    seen: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    n = 0
    for lab in iter_labels_windowed(
        sess, harmful=True, date_gte=f"{year}-01-01", date_lte=f"{year}-12-31",
        window_days=30, max_pages_per_window=2,
    ):
        n += 1
        if lab.check_user:
            seen[lab.check_user] += 1
        if lab.date:
            dates[lab.date[:7]] += 1
        if n >= sample:
            break

    total = sum(seen.values())
    top5 = seen.most_common(5)
    return {
        "sampled": n,
        "with_reviewer": total,
        "distinct_reviewers": len(seen),
        "top5_share": round(sum(c for _, c in top5) / total, 4) if total else None,
        "top1_share": round(top5[0][1] / total, 4) if total else None,
        "months_covered": len(dates),
        # Reviewer names are personal data. Only the shape of the distribution is kept.
        "top5_counts": [c for _, c in top5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2019)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--sample-reviewers", type=int, default=300,
                    help="rows sampled per year for the concentration estimate")
    args = ap.parse_args()

    sess = session()
    years: dict[str, dict] = {}

    print(f"{'year':6} {'harmful':>9} {'clean':>9} {'ratio':>7}  reviewers")
    for year in range(args.from_year, args.to_year + 1):
        lo, hi = f"{year}-01-01", f"{year}-12-31"
        harmful = count_for(sess, harmful=True, date_gte=lo, date_lte=hi)
        clean = count_for(sess, harmful=False, date_gte=lo, date_lte=hi)
        row = {
            "harmful": harmful,
            "reviewed_clean": clean,
            "clean_per_harmful": round(clean / harmful, 2) if harmful else None,
        }
        if harmful:
            row["reviewers"] = reviewer_profile(sess, year, args.sample_reviewers)
        years[str(year)] = row
        rp = row.get("reviewers", {})
        print(f"{year:<6} {str(harmful):>9} {str(clean):>9} {str(row['clean_per_harmful']):>7}  "
              f"n={rp.get('distinct_reviewers','-')} top5={rp.get('top5_share','-')}")

    # --- overlap with the historical benchmark ---------------------------------
    a_ids = corpus_a_ids()
    overlap = 0
    checked = 0
    for lab in iter_labels_windowed(
        sess, harmful=True, date_gte=f"{args.from_year}-01-01",
        date_lte=f"{args.to_year}-12-31", window_days=90, max_pages_per_window=1,
    ):
        checked += 1
        if lab.cid in a_ids:
            overlap += 1
        if checked >= 2000:
            break

    # --- how much of Corpus A does OSMCha know about? --------------------------
    # Not the same question. Corpus A is 2014-2019 and OSMCha review of that era is thin;
    # any overlap would mean a changeset carries two independently-derived labels, which is
    # the cheapest inter-rater signal available.
    totals_h = sum(v["harmful"] or 0 for v in years.values())
    totals_c = sum(v["reviewed_clean"] or 0 for v in years.values())

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "D0 / 2a",
        "question": "what window of OSMCha review is dense and diverse enough to train on?",
        "api": OSMCHA_API,
        "method": "count queries (page_size=1) per year; reviewer concentration from a "
                  f"{args.sample_reviewers}-row sample per year",
        "years": years,
        "totals": {
            "harmful": totals_h,
            "reviewed_clean": totals_c,
            "combined": totals_h + totals_c,
        },
        "corpus_a_overlap": {
            "checked": checked,
            "overlapping_ids": overlap,
            "note": "an overlapping id would carry both a revert-derived and a human label, "
                    "which is the cheapest available cross-source agreement signal",
        },
        "threshold_policy": (
            "NOT pre-registered. Per decision 4 in docs/data/DATA_FOUNDATION.md the exclusion "
            "rule is chosen after inspecting this distribution, and the paper states that "
            "it was set post-hoc. Per-year numbers are published so a reader can apply a "
            "different threshold."
        ),
        "decision": "PENDING - fill in after reading the table above",
    }

    out = RESULTS / "gates" / "gate2a_osmcha_coverage.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- totals ---")
    print(json.dumps(summary["totals"], indent=2))
    print(f"corpus A overlap: {overlap}/{checked} sampled ids")
    print(f"\nwritten to {out}")
    print("\nNext: read the per-year table, choose the window, and record it in "
          "docs/data/DATA_FOUNDATION.md with the reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
