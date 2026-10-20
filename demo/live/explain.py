"""Explanations for the demo, computed from the live scorer's real outputs.

Signal bars (the checklist's "why flagged: geometry / text / user / scale"): occlusion. Each signal
group is replaced by the value a typical changeset has (median over the training store), the
changeset is scored again, and the bar is the score drop. Positive: this signal pushed the score
up. Groups interact, so the bars do not add up to the score.

Edit push: for live changesets, each edit is ranked by the sum of the token model's logistic
coefficients over that edit's tokens - how much what this edit did pushes towards "damaging".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import train_scorer as ts
from mapguard.training.dataset import text_input

TEXT_COLS = ("comment_length", "text_chars", "n_text_edits")
GROUPS = [
    ("tokens", "What the edits did", "edit tokens such as +leisure=park"),
    ("text", "Comment & tag text", "the comment and the names typed into tags"),
    ("author", "Author history", "the author's earlier changesets"),
    ("scale", "Size & mix of edits", "how many edits, creates vs deletes, versions"),
    ("geometry", "Geometry", "node moves, shape changes, bounding box"),
]


def _group_of(col: str) -> str:
    if col.startswith("user_"):
        return "author"
    if col in TEXT_COLS:
        return "text"
    if col.startswith("geom_") or col in ("frac_edits_prev_version", "bbox_lat_span", "bbox_lon_span"):
        return "geometry"
    return "scale"


class Explainer:
    """`reference` (the training store) computes the typical-changeset values; `stats` (from
    `stats()`, saved as model/explainer_stats.json) restores them without the 80 MB store."""

    def __init__(self, scorer, reference: pd.DataFrame | None = None, stats: dict | None = None):
        self.s = scorer
        self.cols = list(ts.TABULAR)
        self.col_group = np.array([_group_of(c) for c in self.cols])
        if stats is not None:
            if stats["columns"] != self.cols:
                raise ValueError("explainer_stats.json was built for different scorer columns")
            self.tab_median = np.array(stats["tab_median"], dtype=float)
            self.tok_median, self.txt_median = stats["tok_median"], stats["txt_median"]
        else:
            ref = reference.sample(min(4000, len(reference)), random_state=0)
            self.tab_median = np.median(ts.tabular(ref), axis=0)
            tok, txt = self._subscores(ref)
            self.tok_median, self.txt_median = float(np.median(tok)), float(np.median(txt))
        vocab = scorer.tok_vec.vocabulary_
        coef = scorer.tok_lr.coef_[0]
        self.token_coef = {t: float(coef[j]) for t, j in vocab.items()}

    def stats(self) -> dict:
        return {"columns": self.cols, "tab_median": [float(v) for v in self.tab_median],
                "tok_median": self.tok_median, "txt_median": self.txt_median,
                "rule": "medians over a 4,000-changeset sample (random_state 0) of release v1.1.0"}

    def _subscores(self, df: pd.DataFrame):
        toks = [ts.token_doc(t, e) for t, e in zip(df["diff_tokens"], df["editor"])]
        txts = [text_input(c, b) for c, b in zip(df["comment"], df["text_blob"])]
        return (self.s.tok_lr.predict_proba(self.s.tok_vec.transform(toks))[:, 1],
                self.s.txt_lr.predict_proba(self.s.txt_vec.transform(txts))[:, 1])

    def signals(self, row: dict) -> dict:
        df = pd.DataFrame([row])
        tok, txt = self._subscores(df)
        base = np.concatenate([ts.tabular(df)[0], [tok[0], txt[0]]])
        X = [base]
        for g, _, _ in GROUPS:
            x = base.copy()
            if g == "tokens":
                x[-2] = self.tok_median
            else:
                m = self.col_group == g
                x[:-2][m] = self.tab_median[m]
                if g == "text":
                    x[-1] = self.txt_median
            X.append(x)
        p = self.s.tree.predict_proba(np.vstack(X))[:, 1]
        bars = [{"key": g, "label": lab, "what": what, "delta": round(float(p[0] - p[i + 1]), 4)}
                for i, (g, lab, what) in enumerate(GROUPS)]
        return {"score": round(float(p[0]), 4), "bars": bars,
                "method": "score drop when this signal is replaced by a typical changeset's value "
                          "(CPU scorer, occlusion; signals interact, so bars do not add up)"}

    def edit_push(self, tokens: list[str]) -> float:
        return round(sum(self.token_coef.get(t.lower(), 0.0) for t in tokens), 3)
