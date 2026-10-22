"""[BASELINES · R1/R2] B0 leakage tripwire, logistic regression and XGBoost.

Run order 1 of 6. Plan stage: baselines. CPU only - no TensorFlow needed.

    python train/01_baselines/baselines_train.py
    python train/01_baselines/baselines_train.py --set task=T1
    python train/01_baselines/baselines_train.py --set split=R1a --set tag=ovid_protocol

Three models, three jobs:

  B0        comment length + editor string + time since the author's first earlier
            changeset (point-in-time), and nothing about what was
            edited. The tripwire: if this separates the classes well, the labels are
            readable from metadata and every later number is suspect. Gate 1c measured
            AUC 0.656 on Corpus A; that is a weak prior, not a leak.
  LogReg    every tabular feature - the floor a neural model must clear to justify itself
  XGBoost   the strongest non-neural reference; if Stage 2 cannot beat it, say so
  Hybrid    XGBoost on the tabular features plus two out-of-fold text models: a bag of edit
            tokens (what each edit did - see mapguard/features/edit_tokens.py) and character
            n-grams of the comment and tag text. Trained with author-balanced weights. This is
            the "gradient-boosted trees over engineered + learned signals" design that large
            wiki and map moderation systems use; round 1 showed a bare deep model only ties
            plain XGBoost, and this is what it has to beat instead.

Running this first also creates the frozen split, so Stage 2 trains on exactly these rows.
"""

import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TRAIN), str(TRAIN.parent / "src")]

import numpy as np  # noqa: E402

from common import scoring, stage  # noqa: E402
from common.preprocess import Normaliser  # noqa: E402


def b0_features(store, rows_all):
    """Metadata only. Columns that are absent are recorded, never silently zero-filled."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cs = store.changesets
    used, missing = [], []
    blocks = []
    # account_age_days was replaced by the point-in-time user_days_active_span: account age is
    # read from today's account, and deleted accounts (mostly vandals) made its missingness a
    # leak. See the note above TABULAR_FEATURES in mapguard/training/dataset.py.
    for col in ("comment_length", "user_days_active_span"):
        if col in cs.columns:
            v = cs[col].to_numpy(dtype="float64", na_value=np.nan)
            blocks.append(np.nan_to_num(np.sign(v) * np.log1p(np.abs(v)), nan=-1.0)[:, None])
            used.append(col)
        else:
            missing.append(col)
    vec = None
    if "editor" in cs.columns:
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z]+", max_features=50)
        editors = cs["editor"].fillna("").astype(str).to_numpy()
        vec.fit(editors[rows_all])
        blocks.append(vec.transform(editors).toarray())
        used.append("editor (tf-idf, 50 tokens)")
    else:
        missing.append("editor")
    if not blocks:
        return None, used, missing, None
    return np.hstack(blocks), used, missing, vec


def fit_predict_logreg(X, y, tr, va, te, seed):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)
    clf.fit(X[tr], y[tr])
    return clf, clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]


def author_weights(uids, rows) -> np.ndarray:
    """1/sqrt(changesets by the same author among `rows`): a bulk-reverted account with 200
    positives counts ~14x one changeset, not 200x. None-author rows get weight 1."""
    from collections import Counter

    counts = Counter(u for u, r in zip(uids, rows) if r and u is not None)
    return np.array([1.0 / np.sqrt(counts.get(u, 1)) if u is not None else 1.0 for u in uids])


def fit_hybrid(store, prep, tr, va, te, y, seed, params, weighting=True, *,
               extra_tokens=None, extra_tab=None, dense_views=None, extra_stack=None, tune=False,
               keep_models=None):
    """Out-of-fold stacking: text models are fitted on 4/5 of the training authors and score
    the held-out fifth, so the tree never sees a text score that was fitted on its own row.

    v3 options (all None = the round-2 hybrid, unchanged):
      extra_tokens  list[str] per row appended to the edit-token bag (object context tokens)
      extra_tab     (N, k) extra columns for the tree (object-context numerics)
      dense_views   {name: (N, d)} dense matrices, each gets its own out-of-fold logistic score
                    (pretrained text embeddings)
      extra_stack   {name: (N,)} scores already out-of-fold for training rows (deep OOF score)
      tune          pick the tree settings from TUNE_GRID by 3-fold author-grouped CV on the
                    training rows only (validation and test are never looked at)
      keep_models   a dict to receive the fitted vectorisers, view models and tree, so a caller
                    (the visual/ demo) can score new changesets with exactly this model
    """
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from mapguard.training.dataset import text_input

    cs = store.changesets
    if "diff_tokens" not in cs.columns:
        return None
    tokens = cs["diff_tokens"].fillna("").astype(str).tolist()
    if extra_tokens is not None:
        tokens = [f"{a} {b}" for a, b in zip(tokens, extra_tokens)]
    texts = [text_input(c, b) for c, b in zip(cs.get("comment", [""] * len(cs)), cs["text_blob"])]
    tri, vai, tei = (np.flatnonzero(m) for m in (tr, va, te))
    sw = author_weights(prep.uids, tr) if weighting else np.ones(len(y))

    tv = TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                         max_features=200_000, sublinear_tf=True)
    tv.fit([tokens[i] for i in tri])
    cv.fit([texts[i] for i in tri])
    views = [("tokens_score", tv.transform(tokens), 1.0), ("text_score", cv.transform(texts), 2.0)]
    for name, mat in (dense_views or {}).items():
        views.append((f"{name}_score", np.asarray(mat, dtype="float32"), 1.0))

    oof = np.zeros((len(y), len(views)))
    groups = np.array([-1 if u is None else u for u in prep.uids])[tri]
    for a, b in GroupKFold(5).split(tri, groups=groups):
        A, B = tri[a], tri[b]
        for j, (_, X, C) in enumerate(views):
            m = LogisticRegression(C=C, max_iter=4000).fit(X[A], y[A], sample_weight=sw[A])
            oof[B, j] = m.predict_proba(X[B])[:, 1]
    rest = np.concatenate([vai, tei])
    fitted = []
    for j, (_, X, C) in enumerate(views):
        m = LogisticRegression(C=C, max_iter=4000).fit(X[tri], y[tri], sample_weight=sw[tri])
        oof[rest, j] = m.predict_proba(X[rest])[:, 1]
        fitted.append(m)

    raw = np.nan_to_num(prep.arrays["tabular"], nan=0.0, posinf=0.0, neginf=0.0)
    blocks, names = [raw], [f"tab:{c}" for c in prep.tabular_columns]
    if extra_tab is not None:
        blocks.append(np.asarray(extra_tab, dtype="float64"))
        names += [f"extra:{j}" for j in range(np.asarray(extra_tab).shape[1])]
    blocks.append(oof)
    names += [v[0] for v in views]
    for name, col in (extra_stack or {}).items():
        blocks.append(np.asarray(col, dtype="float64").reshape(-1, 1))
        names.append(name)
    X2 = np.column_stack(blocks)
    tuning = None
    if tune:
        params, tuning = tune_tree(X2, y, tri, groups, sw, seed)
    clf = xgb.XGBClassifier(**params, eval_metric="aucpr", random_state=seed, n_jobs=-1)
    clf.fit(X2[tri], y[tri], sample_weight=sw[tri], eval_set=[(X2[vai], y[vai])], verbose=False)
    if keep_models is not None:
        keep_models.update(token_vectoriser=tv, text_vectoriser=cv, view_models=fitted,
                           view_names=[v[0] for v in views], tree=clf, feature_names=names)
    coef = sorted(zip(tv.get_feature_names_out(), fitted[0].coef_[0]), key=lambda kv: kv[1])
    stacked = [n for n in names if not n.startswith(("tab:", "extra:"))]
    imp = dict(zip(names, clf.feature_importances_))
    info = {"author_weighting": "1/sqrt(n)" if weighting else None,
            "most_damaging_tokens": [k for k, _ in coef[-20:][::-1]],
            "most_benign_tokens": [k for k, _ in coef[:20]],
            "stack_importance": {n: round(float(imp[n]), 4) for n in stacked},
            "tree_params": params, "tuning": tuning}
    return clf.predict_proba(X2[vai])[:, 1], clf.predict_proba(X2[tei])[:, 1], info


TUNE_GRID = [  # (max_depth, learning_rate, n_estimators); the round-2 setting is (6, 0.08, 400)
    (4, 0.03, 800), (4, 0.08, 400), (6, 0.03, 800), (6, 0.08, 400), (8, 0.03, 800), (8, 0.08, 400),
]


def tune_tree(X, y, tri, groups, sw, seed):
    """Choose tree settings by 3-fold GroupKFold (authors) on training rows only.

    The stacked text scores in X are out-of-fold over 5 author folds of the same training rows,
    so a tuning fold's scores were fitted partly on its sibling folds: selection is slightly
    optimistic, evaluation is not (validation/test never enter)."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    from mapguard.evaluation.metrics import average_precision

    scores = []
    for depth, lr, n in TUNE_GRID:
        cand = {"max_depth": depth, "learning_rate": lr, "n_estimators": n,
                "subsample": 0.8, "colsample_bytree": 0.8}
        aps = []
        for a, b in GroupKFold(3).split(tri, groups=groups):
            A, B = tri[a], tri[b]
            m = xgb.XGBClassifier(**cand, eval_metric="aucpr", random_state=seed, n_jobs=-1)
            m.fit(X[A], y[A], sample_weight=sw[A], verbose=False)
            aps.append(float(average_precision(y[B], m.predict_proba(X[B])[:, 1])))
        scores.append((float(np.mean(aps)), cand))
        print(f"  tune depth={depth} lr={lr} n={n}: cv PR-AUC {np.mean(aps):.4f}", flush=True)
    best = max(scores, key=lambda s: s[0])
    return best[1], {"grid": [{"cv_pr_auc": round(s, 4), **c} for s, c in scores],
                     "chosen": best[1], "rule": "3-fold GroupKFold on training rows"}


def hybrid_inputs(store, prep, variant: str):
    """Keyword arguments for fit_hybrid for one named variant, or None if its data is missing.

    base      round-2 hybrid
    ctx       + object context (tokens into the token bag, numerics into the tree)
    emb       + pretrained multilingual sentence embeddings of comment and tag text
    ctx_emb   both
    where     + WHERE distances (kept to report the negative result reproducibly)
    """
    from common import extras

    kw: dict = {"tune": "tuned" in variant}
    if "ctx" in variant:
        got = extras.object_context(store, prep.cids)
        if got is None:
            return None
        kw["extra_tokens"], kw["extra_tab"], _ = got
    if "emb" in variant:
        got = extras.text_embeddings(store, prep.cids)
        if got is None:
            return None
        kw["dense_views"] = {"emb_comment": got["comment"], "emb_tags": got["tags"]}
    if variant == "where":
        got = extras.where_features(store, prep.cids)
        if got is None:
            return None
        kw["extra_tab"] = got[0]
    return kw


def main() -> int:
    args = stage.parse_args(__doc__.splitlines()[0], Path(__file__).with_name("config.json"))
    cfg, run, store, prep, masks = stage.setup(args)
    b = cfg.get("baselines", {})
    seed = int(cfg.get("seed", 0))
    tr, va, te = masks["train"], masks["val"], masks["test"]
    y = prep.y.astype(int)
    results = {"models": {}, "split_leaky": cfg["split"] == "R1a",
               "data_missing_optional_columns": store.missing_optional}

    # --- B0 tripwire -------------------------------------------------------------
    if b.get("b0_tripwire", True):
        X0, used, missing, vec = b0_features(store, tr)
        if X0 is None:
            results["models"]["b0_tripwire"] = {"skipped": "no metadata columns available",
                                                "missing": missing}
        else:
            clf, pv, pt = fit_predict_logreg(X0, y, tr, va, te, seed)
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            auc = rep["test_intervals_95"]["roc_auc"][0]
            leak_at = float(b.get("b0_leak_auc", 0.85))
            results["models"]["b0_tripwire"] = {
                "features_used": used, "features_missing": missing,
                "headline": scoring.headline(rep), "evaluation": rep,
                "verdict": ("FAIL - labels readable from metadata alone; investigate before "
                            "trusting any other result" if auc >= leak_at else
                            "PASS - weak prior, not leakage"),
            }
            print(f"B0 tripwire  ROC-AUC {auc}  -> {results['models']['b0_tripwire']['verdict']}")

    # --- tabular features, normalised on training rows ---------------------------
    norm = Normaliser().fit(prep, tr, groups=("tabular",))
    Xt = norm._z(np.sign(prep.arrays["tabular"]) * np.log1p(np.abs(prep.arrays["tabular"])),
                 "tabular", prep.tabular_columns)
    norm.save(run.path("normaliser_tabular.json"))

    if b.get("logistic_regression", True):
        clf, pv, pt = fit_predict_logreg(Xt, y, tr, va, te, seed)
        rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
        coefs = sorted(zip(prep.tabular_columns, clf.coef_[0]), key=lambda kv: -abs(kv[1]))[:15]
        results["models"]["logistic_regression"] = {
            "n_features": int(Xt.shape[1]), "headline": scoring.headline(rep), "evaluation": rep,
            "top_coefficients": {k: round(float(v), 4) for k, v in coefs},
        }
        np.savez_compressed(run.path("predictions_logreg.npz"), test_cids=prep.cids[te],
                            test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
        print(f"LogReg       PR-AUC {rep['test_intervals_95']['pr_auc']}")

    if b.get("xgboost", True):
        try:
            import xgboost as xgb
        except ImportError:
            results["models"]["xgboost"] = {"skipped": "xgboost not installed"}
        else:
            raw = np.nan_to_num(prep.arrays["tabular"], nan=0.0, posinf=0.0, neginf=0.0)
            clf = xgb.XGBClassifier(**b.get("xgboost_params", {}), eval_metric="aucpr",
                                    random_state=seed, n_jobs=-1)
            clf.fit(raw[tr], y[tr], eval_set=[(raw[va], y[va])], verbose=False)
            pv, pt = clf.predict_proba(raw[va])[:, 1], clf.predict_proba(raw[te])[:, 1]
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            imp = sorted(zip(prep.tabular_columns, clf.feature_importances_), key=lambda kv: -kv[1])[:15]
            results["models"]["xgboost"] = {
                "params": b.get("xgboost_params", {}), "headline": scoring.headline(rep),
                "evaluation": rep, "top_importances": {k: round(float(v), 4) for k, v in imp},
            }
            np.savez_compressed(run.path("predictions_xgboost.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"XGBoost      PR-AUC {rep['test_intervals_95']['pr_auc']}")

    if b.get("hybrid", True):
        for variant in b.get("hybrid_variants", ["base"]):
            key = "hybrid" if variant == "base" else f"hybrid_{variant}"
            kw = hybrid_inputs(store, prep, variant)
            if kw is None:
                results["models"][key] = {"skipped": f"side table for '{variant}' not in the store"}
                print(f"{key:24s} skipped (no side table)")
                continue
            try:
                got = fit_hybrid(store, prep, tr, va, te, y, seed, b.get("xgboost_params", {}),
                                 weighting=b.get("author_weighting", "sqrt") == "sqrt", **kw)
            except ImportError as exc:
                got, results["models"][key] = None, {"skipped": repr(exc)}
            if got is None:
                results["models"].setdefault(key, {"skipped": "store has no diff_tokens (schema < 4)"})
                continue
            pv, pt, info = got
            rep = scoring.report(y[va], pv, y[te], pt, seed=seed, test_groups=prep.uids[te])
            results["models"][key] = {**info, "variant": variant, "headline": scoring.headline(rep),
                                      "evaluation": rep}
            np.savez_compressed(run.path(f"predictions_{key}.npz"), test_cids=prep.cids[te],
                                test_y=y[te], test_p=pt, val_cids=prep.cids[va], val_y=y[va], val_p=pv)
            print(f"{key:24s} PR-AUC {rep['test_intervals_95']['pr_auc']}  "
                  f"one-per-author {rep['author_robust'].get('one_per_author_pr_auc')}")

    best = max(((k, v["headline"]["test_pr_auc"][0]) for k, v in results["models"].items()
                if "headline" in v and k != "b0_tripwire"), key=lambda kv: kv[1], default=None)
    results["headline"] = {"best_baseline": best[0], "best_test_pr_auc": best[1]} if best else None
    run.finish(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
