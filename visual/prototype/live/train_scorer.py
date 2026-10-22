"""Train the live scorer: a CPU version of MapGuard's hybrid model on features that can be
recomputed for a brand-new changeset from the public OSM API.

    python live/train_scorer.py

Inputs:  data/release_v1.1.0/extracted/feature_store (release v1.1.0 of osm-mapguard)
Outputs: live/model/scorer.pkl, live/model/scorer_report.json

Model (same design as the repo's stage-01 hybrid, minus the pretrained text embeddings):
  * logistic regression on the bag of edit tokens (+leisure=park, -historic, ...) and editor
  * logistic regression on character n-grams of the comment and tag text
  * XGBoost on the tabular changeset / point-in-time author features plus those two scores
    (out-of-fold for training rows, grouped by author), author-balanced weights 1/sqrt(n).

Columns only the Overpass adiff route can fill are left out: live changesets never have them.

Labels: Ovid's reverted changesets (A), reverted 2021-26 (R) and OSMCha harmful (M) are
damaging; their non-reverted counterparts and random edits (N) are clean. OSMCha's
flagged-and-cleared rows are left out: they are not representative of random traffic.
The split is user-disjoint (repo's user_disjoint_split), so test authors are never trained on.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parent  # demo/ sits in the repository root
sys.path[:0] = [str(REPO / "src"), str(REPO / "train")]

from mapguard.evaluation.metrics import average_precision, roc_auc  # noqa: E402
from mapguard.evaluation.splits import assert_user_disjoint, user_disjoint_split  # noqa: E402
from mapguard.training.dataset import TABULAR_FEATURES, text_input  # noqa: E402
from common.preprocess import ADIFF_ONLY_COLUMNS  # noqa: E402
from common.scoring import author_robust, best_f1_threshold  # noqa: E402

STORE = DEMO / "data/release_v1.1.0/extracted/feature_store"
OUT = HERE / "model"
SEED = 20260904
TABULAR = [c for c in TABULAR_FEATURES if c not in ADIFF_ONLY_COLUMNS]


def token_doc(diff_tokens, editor) -> str:
    ed = (str(editor or "").split(" ")[0] or "unknown").replace(" ", "_")
    return f"{diff_tokens or ''} editor:{ed}"


def tabular(df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
                         if c in df.columns else np.zeros(len(df)) for c in TABULAR])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class Scorer:
    """Everything needed to score one changeset row (a dict with the feature-store columns)."""

    def __init__(self, tok_vec, txt_vec, tok_lr, txt_lr, tree, threshold):
        self.tok_vec, self.txt_vec, self.tok_lr, self.txt_lr = tok_vec, txt_vec, tok_lr, txt_lr
        self.tree, self.threshold = tree, threshold

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        toks = [token_doc(t, e) for t, e in zip(df["diff_tokens"], df["editor"])]
        txts = [text_input(c, b) for c, b in zip(df["comment"], df["text_blob"])]
        s_tok = self.tok_lr.predict_proba(self.tok_vec.transform(toks))[:, 1]
        s_txt = self.txt_lr.predict_proba(self.txt_vec.transform(txts))[:, 1]
        X = np.column_stack([tabular(df), s_tok, s_txt])
        return self.tree.predict_proba(X)[:, 1]

    def explain(self, row: dict, k: int = 8) -> dict:
        """Token contributions (logistic coefficient x tf-idf weight) and the two sub-scores."""
        doc = token_doc(row.get("diff_tokens"), row.get("editor"))
        v = self.tok_vec.transform([doc])
        names = self.tok_vec.get_feature_names_out()
        coef = self.tok_lr.coef_[0]
        contrib = sorted(((names[j], float(coef[j] * v[0, j])) for j in v.nonzero()[1]),
                         key=lambda kv: kv[1])
        txt = text_input(row.get("comment"), row.get("text_blob"))
        return {
            "token_score": float(self.tok_lr.predict_proba(v)[0, 1]),
            "text_score": float(self.txt_lr.predict_proba(self.txt_vec.transform([txt]))[0, 1]),
            "pushes_up": [[n, round(c, 3)] for n, c in contrib[::-1][:k] if c > 0],
            "pushes_down": [[n, round(c, 3)] for n, c in contrib[:k] if c < 0],
        }


def main() -> int:
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    cs = pd.read_parquet(STORE / "changesets.parquet")
    cs = cs[~((cs["corpus"] == "M") & (cs["label"] == 0))].reset_index(drop=True)
    y = cs["label"].to_numpy(dtype=int)
    uids = [None if pd.isna(u) else int(u) for u in cs["uid"]]
    split = user_disjoint_split(uids, y, seed=SEED)
    assert_user_disjoint(split, uids)
    tr, va, te = split.train, split.val, split.test
    tri = np.flatnonzero(tr)
    print(f"rows {len(cs):,}  train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}", flush=True)

    toks = [token_doc(t, e) for t, e in zip(cs["diff_tokens"], cs["editor"])]
    txts = [text_input(c, b) for c, b in zip(cs["comment"], cs["text_blob"])]
    from collections import Counter
    cnt = Counter(u for u, r in zip(uids, tr) if r and u is not None)
    sw = np.array([1 / np.sqrt(cnt.get(u, 1)) if u is not None else 1.0 for u in uids])

    tok_vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True).fit([toks[i] for i in tri])
    txt_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200_000,
                              sublinear_tf=True).fit([txts[i] for i in tri])
    Xtok, Xtxt = tok_vec.transform(toks), txt_vec.transform(txts)

    oof = np.zeros((len(cs), 2))
    groups = np.array([-1 if u is None else u for u in uids])[tri]
    for a, b in GroupKFold(5).split(tri, groups=groups):
        A, B = tri[a], tri[b]
        for j, (X, C) in enumerate(((Xtok, 1.0), (Xtxt, 2.0))):
            m = LogisticRegression(C=C, max_iter=4000).fit(X[A], y[A], sample_weight=sw[A])
            oof[B, j] = m.predict_proba(X[B])[:, 1]
    rest = np.flatnonzero(~tr)
    tok_lr = LogisticRegression(C=1.0, max_iter=4000).fit(Xtok[tri], y[tri], sample_weight=sw[tri])
    txt_lr = LogisticRegression(C=2.0, max_iter=4000).fit(Xtxt[tri], y[tri], sample_weight=sw[tri])
    oof[rest, 0] = tok_lr.predict_proba(Xtok[rest])[:, 1]
    oof[rest, 1] = txt_lr.predict_proba(Xtxt[rest])[:, 1]

    X2 = np.column_stack([tabular(cs), oof])
    tree = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="aucpr", random_state=SEED, n_jobs=-1)
    tree.fit(X2[tr], y[tr], sample_weight=sw[tr], verbose=False)
    p = tree.predict_proba(X2)[:, 1]
    thr, val_f1 = best_f1_threshold(y[va], p[va])

    report = {"rows": int(len(cs)), "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
              "threshold": round(thr, 4), "val_f1": round(val_f1, 4), "tabular_columns": TABULAR,
              "by_corpus": {}}
    corpus = cs["corpus"].to_numpy()
    for c in ("A", "R", "M", "N", "all"):
        m = te if c == "all" else (te & (corpus == c))
        if m.sum() == 0:
            continue
        e = {"n": int(m.sum()), "positive_rate": round(float(y[m].mean()), 3)}
        if len(np.unique(y[m])) == 2:
            e["pr_auc"] = round(float(average_precision(y[m], p[m])), 4)
            e["roc_auc"] = round(float(roc_auc(y[m], p[m])), 4)
            e["one_per_author_pr_auc"] = author_robust(y[m], p[m], [uids[i] for i in np.flatnonzero(m)]).get("one_per_author_pr_auc")
        pred = p[m] >= thr
        e["flag_rate"] = round(float(pred.mean()), 3)
        e["accuracy"] = round(float((pred == y[m]).mean()), 4)
        report["by_corpus"][c] = e
    print(json.dumps(report["by_corpus"], indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(tok_vec, txt_vec, tok_lr, txt_lr, tree, thr)
    with open(OUT / "scorer.pkl", "wb") as fh:
        pickle.dump(scorer, fh)
    (OUT / "scorer_report.json").write_text(json.dumps(report, indent=1))
    # sanity: the saved object reproduces the training-time scores on test rows
    again = scorer.score_frame(cs[te].head(500))
    ref = tree.predict_proba(np.column_stack([tabular(cs[te].head(500)),
                                              tok_lr.predict_proba(Xtok[np.flatnonzero(te)[:500]])[:, 1],
                                              txt_lr.predict_proba(Xtxt[np.flatnonzero(te)[:500]])[:, 1]]))[:, 1]
    assert np.allclose(again, ref, atol=1e-6), "saved scorer does not reproduce its own scores"
    print(f"threshold {thr:.3f}  saved {OUT / 'scorer.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
