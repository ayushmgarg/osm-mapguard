"""The demo's tree model: round 4's "hybrid", refit on CPU so live changesets can be scored.

Round 4 saved each model's predictions, not the fitted trees, so the live path needs a refit.
It runs the training code itself (`fit_hybrid` in train/01_baselines), on the released feature
store, with the frozen R1b split and seed 20260904, so the refit is round 4's model and not
a look-alike. `fit` checks that by comparing test PR-AUC with round 4's stored predictions.

The hybrid, not the stack, is what runs live: the stack needs the TensorFlow deep model,
and the demo has to run on a laptop CPU.

    python visual/tree_model.py --task ovid_modern
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

VISUAL = Path(__file__).resolve().parent
REPO = VISUAL.parent
sys.path[:0] = [str(REPO / "train"), str(REPO / "train" / "01_baselines"), str(REPO / "src")]

from baselines_train import fit_hybrid
from common import data_contract, preprocess, scoring, splits_io

from mapguard.evaluation.metrics import average_precision
from mapguard.training.dataset import build_batch, text_input

SEED = 20260904
# Round 4's tree settings (train/01_baselines/config.json); the refit must not drift from them.
TREE_PARAMS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
               "subsample": 0.8, "colsample_bytree": 0.8}
MODELS_DIR = VISUAL / "models"
ROUND4_RUNS = REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds/raw/experiments/runs"
# Library versions differ from the cluster (xgboost 1.7 there), so the refit is close, not equal.
REFIT_TOLERANCE = 0.005

# Signal bars: each tree input belongs to one of the three kinds of evidence in the README.
WHO_PREFIX = "tab:user_"
TEXT_FEATURES = {"text_score", "tab:comment_length", "tab:text_chars", "tab:n_text_edits"}


def default_store() -> Path:
    root = Path(os.environ.get("MAPGUARD_DATA", Path.home() / "osm-data-mac"))
    return root / "release" / "v1.1.0" / "feature_store"


def signal_of(feature: str) -> str:
    if feature.startswith(WHO_PREFIX):
        return "who"
    if feature in TEXT_FEATURES:
        return "text"
    return "what"


def round4_hybrid_pr_auc(task: str) -> float:
    path = next((ROUND4_RUNS / "baselines").glob(f"*_{task}_R1b_s{SEED}/predictions_hybrid.npz"))
    z = np.load(path)
    return float(average_precision(z["test_y"], z["test_p"]))


def fit(task: str, store_dir: Path) -> dict:
    store = data_contract.load(store_dir, task=task)
    prep = preprocess.prepare(store, {})
    masks, _, _ = splits_io.get_split(prep, task=task, split="R1b", seed=SEED)
    y = prep.y.astype(int)
    kept: dict = {}
    got = fit_hybrid(store, prep, masks["train"], masks["val"], masks["test"], y,
                     SEED, TREE_PARAMS, weighting=True, keep_models=kept)
    if got is None:
        raise SystemExit("feature store has no diff_tokens (schema < 4); use release v1.1.0 or later")
    p_val, p_test, _ = got
    threshold, _ = scoring.best_f1_threshold(y[masks["val"]], p_val)
    refit = float(average_precision(y[masks["test"]], p_test))
    round4 = round4_hybrid_pr_auc(task)
    if abs(refit - round4) > REFIT_TOLERANCE:
        raise SystemExit(f"refit test PR-AUC {refit:.4f} is not round 4's {round4:.4f}")
    return {**kept, "task": task, "seed": SEED, "threshold": threshold,
            "test_pr_auc": refit, "round4_test_pr_auc": round4,
            "test_cids": prep.cids[masks["test"]], "test_p": p_test}


def features(bundle: dict, changesets: pd.DataFrame) -> np.ndarray:
    """Tree inputs for feature-store-shaped changeset rows, built the way training built them."""
    tabular = build_batch(changesets, None, None).tabular.astype("float64")
    tokens = changesets["diff_tokens"].fillna("").astype(str).tolist()
    texts = [text_input(c, b) for c, b in zip(changesets["comment"], changesets["text_blob"])]
    views = [bundle["token_vectoriser"].transform(tokens), bundle["text_vectoriser"].transform(texts)]
    scores = [m.predict_proba(X)[:, 1] for m, X in zip(bundle["view_models"], views)]
    return np.column_stack([np.nan_to_num(tabular, nan=0.0, posinf=0.0, neginf=0.0), *scores])


def score(bundle: dict, changesets: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Damage probability per row, and each row's who/what/text pull in log-odds.

    The pulls are the tree's own SHAP contributions summed by evidence kind, so the bars
    add up (with the bias) to exactly the score shown, rather than being a separate story.
    """
    import xgboost as xgb

    X = features(bundle, changesets)
    p = bundle["tree"].predict_proba(X)[:, 1]
    contribs = bundle["tree"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    names = bundle["feature_names"]
    signals = []
    for row in contribs:
        pulls = {"who": 0.0, "what": 0.0, "text": 0.0}
        for name, value in zip(names, row[:-1]):
            pulls[signal_of(name)] += float(value)
        top = sorted(zip(names, row[:-1]), key=lambda kv: -abs(kv[1]))[:6]
        signals.append({**{k: round(v, 4) for k, v in pulls.items()}, "bias": round(float(row[-1]), 4),
                        "top": [{"feature": n, "pull": round(float(v), 4)} for n, v in top]})
    return p, signals


def token_reasons(bundle: dict, diff_tokens: str, k: int = 5) -> dict:
    """The edit tokens in this changeset that pushed the token model hardest, each way.

    Weight is coefficient x the token's tf-idf value in this changeset, so a token repeated
    across many edits counts for more, the way the model saw it.
    """
    vec = bundle["token_vectoriser"].transform([diff_tokens or ""])
    coef = bundle["view_models"][0].coef_[0]
    vocab = bundle["token_vectoriser"].get_feature_names_out()
    weights = sorted(((vocab[j], float(coef[j] * v)) for j, v in zip(vec.indices, vec.data)),
                     key=lambda kv: kv[1])
    damaging = [p for p in weights[::-1] if p[1] > 0][:k]
    benign = [p for p in weights if p[1] < 0][:k]
    return {name: [{"token": t, "weight": round(w, 3)} for t, w in pairs]
            for name, pairs in (("damaging", damaging), ("benign", benign))}


def edit_token_scores(bundle: dict, edit_tokens: list[str]) -> np.ndarray:
    """The token model applied to one edit at a time: which edits look damaging on their own.

    The deep model's attention answers this for replayed changesets; it needs TensorFlow, so
    the live path ranks edits with this instead, and the page labels which one it shows.
    """
    if not edit_tokens:
        return np.zeros(0)
    X = bundle["token_vectoriser"].transform([t or "" for t in edit_tokens])
    return bundle["view_models"][0].predict_proba(X)[:, 1]


def model_path(task: str) -> Path:
    return MODELS_DIR / f"hybrid_{task}.joblib"


def load(task: str) -> dict:
    return joblib.load(model_path(task))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--store", type=Path, default=default_store())
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    for task in args.task:
        bundle = fit(task, args.store)
        joblib.dump(bundle, model_path(task))
        print(json.dumps({"task": task, "test_pr_auc": round(bundle["test_pr_auc"], 4),
                          "round4_test_pr_auc": round(bundle["round4_test_pr_auc"], 4),
                          "threshold": round(bundle["threshold"], 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
