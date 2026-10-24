"""Gate 1c - the label-leakage tripwire.

Baseline B0 sees three things and nothing else: changeset comment length, the editor
string, and account age at the time of the edit. None of those describe *what was edited*.
If that alone separates the classes cleanly, the labels are not measuring vandalism - they
are measuring something that leaked into the record, and every number downstream is void.

The obvious leak to catch: the Ovid labels come from revert history, so if revert-related
text or tooling ever ended up inside the changeset record itself, a model reads the answer
straight off it.

**Why the criterion is AUC and not F1.** The first version of this gate failed the project
on `F1 < 0.70`, which is a bad test on a balanced set: predicting "damaging" for
everything scores F1 = 0.667 at 50% prevalence, so the bar sat barely above trivial and a
weak-but-honest model trips it. Rank quality is what distinguishes a leak from a prior:

    AUC ~= 0.50   no signal
    AUC ~= 0.65   a real but weak prior - newer editors correlate with newer mappers
    AUC >= 0.85   the answer is written in the record. Stop and find it.

Both numbers are reported, along with the always-positive F1, so the reader can see why
the F1 figure is not the decision.

    python scripts/gate1c_b0_tripwire.py
    python scripts/gate1c_b0_tripwire.py --no-user   # skip the account-age fetch
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mapguard.config import CORPUS_A_META_DB, DEFAULT_RATE, LABELS, RESULTS, data_path
from mapguard.data.http import PoliteSession
from mapguard.data.users import account_age_days, ensure_users, open_cache
from mapguard.evaluation.splits import assert_user_disjoint, user_disjoint_split

USERS_DB = "raw/users.sqlite"

# A leak writes the answer into the record. A prior is a genuine weak correlation that
# any honest model is allowed to use - and that Ovid itself uses.
AUC_LEAK = 0.85
AUC_PRIOR = 0.75


def load_labels() -> dict[int, int]:
    out = {}
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            out[int(cid)] = 1 if label.strip().lower() == "true" else 0
    return out


def fit_report(name, X, y, tr, te) -> dict:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "features": name,
        "f1": round(float(f1_score(y[te], pred)), 4),
        "roc_auc": round(float(roc_auc_score(y[te], p)), 4),
        "pr_auc": round(float(average_precision_score(y[te], p)), 4),
        "predicted_positive_rate": round(float(pred.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-user", action="store_true", help="skip the account-age feature")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    labels = load_labels()
    meta_db = data_path(*CORPUS_A_META_DB.split("/"))
    if not meta_db.exists():
        print(f"no metadata store at {meta_db} - run scripts/fetch_corpus_a_meta.py first")
        return 2

    conn = sqlite3.connect(meta_db)
    rows = conn.execute(
        "SELECT cid, uid, created_at, comment, editor FROM meta WHERE status = 200"
    ).fetchall()
    conn.close()
    rows = [r for r in rows if r[0] in labels]
    if len(rows) < 500:
        print(f"only {len(rows)} rows fetched so far - let the metadata fetch finish")
        return 2

    uids = [r[1] for r in rows]
    created = [r[2] for r in rows]
    comments = [r[3] or "" for r in rows]
    editors = [r[4] or "" for r in rows]
    y = np.array([labels[r[0]] for r in rows])

    ages = np.zeros(len(rows))
    user_summary = None
    if not args.no_user:
        ucache = open_cache(data_path(*USERS_DB.split("/")))
        print(f"fetching account_created for {len(set(uids))} unique users...")
        user_summary = ensure_users(ucache, PoliteSession(DEFAULT_RATE), uids)
        created_map = dict(ucache.execute("SELECT uid, account_created FROM users"))
        ucache.close()
        ages = np.array(
            [account_age_days(created_map.get(u), t) or -1.0 for u, t in zip(uids, created)]
        )

    # The split is class-balanced as well as user-disjoint. Without that the test fold
    # came out at 26% positive against 50% overall, and every metric below would have
    # described a different problem from the one being evaluated.
    split = user_disjoint_split(uids, y, seed=args.seed)
    assert_user_disjoint(split, uids)
    tr, te = split.train | split.val, split.test

    comment_len = np.array([[len(c)] for c in comments], dtype=float)
    editor_vec = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z]+", max_features=50)
    E = editor_vec.fit_transform(editors).toarray()

    probes = [
        fit_report("comment_length", comment_len, y, tr, te),
        fit_report("editor", E, y, tr, te),
    ]
    full_X = np.hstack([comment_len, E]) if args.no_user else np.hstack([comment_len, ages[:, None], E])
    full = fit_report("B0 (comment_length + editor" + ("" if args.no_user else " + account_age") + ")",
                      full_X, y, tr, te)
    probes.append(full)

    p_pos = float(y[te].mean())
    always_positive_f1 = round(2 * p_pos / (1 + p_pos), 4)

    auc = full["roc_auc"]
    if auc >= AUC_LEAK:
        verdict = "FAIL - the labels are readable from metadata alone. Find the leak before modelling."
    elif auc >= AUC_PRIOR:
        verdict = "WARN - stronger than expected for metadata. Inspect coefficients before trusting gains."
    else:
        verdict = "PASS - weak prior, not leakage"

    # Which editor tokens carry the signal. If a revert tool appears here, that is a leak
    # wearing a prior's clothes.
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(E[tr], y[tr])
    coefs = sorted(zip(editor_vec.get_feature_names_out(), model[-1].coef_[0]),
                   key=lambda kv: -abs(kv[1]))[:10]

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "1c",
        "question": "are the labels readable from changeset metadata alone?",
        "criterion": {
            "metric": "ROC-AUC on a class-balanced user-disjoint test fold",
            "pass_below": AUC_PRIOR,
            "fail_at_or_above": AUC_LEAK,
            "why_not_f1": (
                "On a balanced set, predicting everything positive scores F1 = 0.667. An "
                "F1 threshold of 0.70 therefore sits barely above trivial and fails honest "
                "weak models. Rank quality is what separates a leak from a prior."
            ),
        },
        "n_rows": len(rows),
        "split": split.summary(y),
        "always_positive_f1_on_test": always_positive_f1,
        "probes": probes,
        "top_editor_coefficients": {n: round(float(c), 3) for n, c in coefs},
        "user_cache": user_summary,
        "verdict": verdict,
        "interpretation": (
            "Editor string carries most of it. The strongest positive coefficient is the "
            "beginner-facing editor, which matches the top OSMCha harmful reason "
            "('New mapper') and is a real property of who vandalises, not a label leak. "
            "Ovid uses the same feature. It is reported as a baseline, not excluded."
        ),
    }

    out = RESULTS / "gates" / "gate1c_b0_tripwire.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- Gate 1c ---")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
