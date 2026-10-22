"""Round 4's stored outputs -> the demo's replay dataset (visual/data/<task>/).

The demo shows what round 4 actually produced, not a re-run: every score is read from the
run folders in experiments/rounds/r4_..., and the page shows the mean over the three model
seeds. The ensemble was never saved per row, so it is rebuilt from its recorded weight and
its two inputs, and each seed's rebuild must reproduce the PR-AUC that run recorded.

    scores.parquet       cid, label, <model>, <model>_s<seed> for every model
    models.json          per model: name, val-chosen threshold, test PR-AUC
    changesets.parquet   feature-store rows of the test changesets, plus why-flagged signals
    edits.parquet        their sampled edits, with the deep model's attention per edit
    scenarios.json       curated examples (visual/data/scenarios.json, all tasks)
    geometry/            before/after map geometry for the curated examples, from the OSM API

    python visual/export_demo.py                 # everything
    python visual/export_demo.py --no-geometry   # skip the OSM API calls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import osm_live
import pandas as pd
import tree_model
from common import scoring
from tree_model import ROUND4_RUNS

from mapguard.data.http import PoliteSession
from mapguard.evaluation.metrics import average_precision

DATA = tree_model.VISUAL / "data"
TASKS = ("ovid", "ovid_modern")
SEEDS = (20260904, 20260905, 20260906)
# Recorded PR-AUC is rounded to 4 places in results.json.
RECORDED_TOLERANCE = 6e-5

# model key -> (stage folder, run folder pattern, predictions file, display name)
MODELS = {
    "logreg": ("baselines", "*_{task}_R1b_s{seed}", "predictions_logreg.npz", "Logistic regression"),
    "xgboost": ("baselines", "*_{task}_R1b_s{seed}", "predictions_xgboost.npz", "XGBoost (tabular)"),
    "hybrid": ("baselines", "*_{task}_R1b_s{seed}", "predictions_hybrid.npz", "Hybrid trees"),
    "hybrid_emb": ("baselines", "*_{task}_R1b_s{seed}", "predictions_hybrid_emb.npz", "Hybrid + embeddings"),
    "deep": ("stage2_supervised", "*_{task}_R1b_s{seed}", "predictions.npz", "Deep MIL model"),
    "stack": ("stack", "*_{task}_R1b_stack_s{seed}", "predictions_stack_base.npz", "Stack"),
    "stack_emb": ("stack", "*_{task}_R1b_stack_s{seed}", "predictions_stack_emb.npz", "Stack + embeddings"),
}
ENSEMBLE_NAME = "Ensemble (deep + hybrid)"
# The best system per task in round 4 (experiments/rounds/r4_.../README.md, "What this round settles").
HEADLINE_MODEL = {"ovid": "stack_emb", "ovid_modern": "ensemble"}


def run_dir(stage: str, pattern: str, task: str, seed: int) -> Path:
    found = list((ROUND4_RUNS / stage).glob(pattern.format(task=task, seed=seed)))
    if len(found) != 1:
        raise SystemExit(f"expected one {stage} run for {task} seed {seed}, found {found}")
    return found[0]


def split_scores(npz: Path) -> dict[str, pd.Series]:
    z = np.load(npz)
    return {f: pd.Series(z[f"{f}_p"].astype("float64"), index=z[f"{f}_cids"].astype("int64"))
            for f in ("val", "test")}


def ensemble_scores(task: str, seed: int) -> dict[str, pd.Series]:
    """deep_weight * deep + (1 - deep_weight) * hybrid, as train/07_ensemble/ensemble.py did."""
    for d in (ROUND4_RUNS / "ensemble").glob(f"*_{task}_R1b"):
        cfg = json.loads((d / "config.json").read_text())
        if not Path(cfg["hybrid_run"]).name.endswith(f"_{task}_R1b_s{seed}"):
            continue
        w = json.loads((d / "results.json").read_text())["deep_weight"]
        hybrid_npz = run_dir("baselines", MODELS["hybrid"][1], task, seed) / "predictions_hybrid.npz"
        deep = split_scores(run_dir("stage2_supervised", MODELS["deep"][1], task, seed) / "predictions.npz")
        hyb = split_scores(hybrid_npz)
        blended = {f: w * deep[f] + (1 - w) * hyb[f].reindex(deep[f].index) for f in deep}
        check_recorded(d, blended["test"], labels_for(hybrid_npz))
        return blended
    raise SystemExit(f"no ensemble run for {task} seed {seed}")


def check_recorded(run: Path, test: pd.Series, y: pd.Series) -> None:
    """The rebuilt ensemble must score what the round recorded, or it is not that ensemble."""
    recorded = json.loads((run / "results.json").read_text())["headline"]["test_pr_auc"][0]
    got = average_precision(y.reindex(test.index).to_numpy(), test.to_numpy())
    if abs(got - recorded) > RECORDED_TOLERANCE:
        raise SystemExit(f"{run.name}: rebuilt ensemble PR-AUC {got:.4f} != recorded {recorded}")


def labels_for(npz: Path) -> pd.Series:
    z = np.load(npz)
    return pd.Series(z["test_y"].astype("int64"), index=z["test_cids"].astype("int64"))


def model_scores(task: str) -> tuple[pd.DataFrame, dict]:
    """Test scores per model and seed, plus each model's threshold and test PR-AUC."""
    hybrid_npz = run_dir("baselines", MODELS["hybrid"][1], task, SEEDS[0]) / "predictions_hybrid.npz"
    y = labels_for(hybrid_npz)
    frame = pd.DataFrame({"cid": y.index, "label": y.to_numpy()}).set_index("cid")
    info = {}
    keys = [*MODELS, "ensemble"]
    for key in keys:
        val, test = [], []
        for seed in SEEDS:
            if key == "ensemble":
                got = ensemble_scores(task, seed)
            else:
                stage, pattern, fname, _ = MODELS[key]
                got = split_scores(run_dir(stage, pattern, task, seed) / fname)
            val.append(got["val"])
            frame[f"{key}_s{seed}"] = got["test"].reindex(frame.index)
            test.append(got["test"])
        frame[key] = frame[[f"{key}_s{s}" for s in SEEDS]].mean(axis=1)
        val_mean = pd.concat(val, axis=1).mean(axis=1)
        val_y = labels_for_val(task).reindex(val_mean.index)
        if val_y.isna().any() or val_mean.isna().any():
            raise SystemExit(f"{task} {key}: validation ids disagree between models")
        threshold, _ = scoring.best_f1_threshold(val_y.to_numpy(), val_mean.to_numpy())
        info[key] = {"name": MODELS[key][3] if key in MODELS else ENSEMBLE_NAME,
                     "threshold": round(threshold, 6),
                     "test_pr_auc": round(float(average_precision(frame["label"].to_numpy(),
                                                                  frame[key].to_numpy())), 4),
                     "headline": key == HEADLINE_MODEL[task]}
    if frame.isna().any().any():
        raise SystemExit(f"{task}: models disagree on the test changesets")
    return frame.reset_index(), info


def labels_for_val(task: str) -> pd.Series:
    z = np.load(run_dir("baselines", MODELS["hybrid"][1], task, SEEDS[0]) / "predictions_hybrid.npz")
    return pd.Series(z["val_y"].astype("int64"), index=z["val_cids"].astype("int64"))


def attention(task: str, edits: pd.DataFrame) -> pd.Series:
    """Deep-model attention per stored edit row, averaged over seeds.

    Slot k of a changeset's attention row is its k-th row in edits.parquet: that is how
    `build_batch` filled the model's input, so the mapping is exact, and checked by count.
    """
    slots = edits.groupby("cid").cumcount()
    att = []
    for seed in SEEDS:
        z = np.load(run_dir("stage2_supervised", MODELS["deep"][1], task, seed) / "predictions.npz")
        row_of = {int(c): i for i, c in enumerate(z["test_cids"])}
        n_edits = edits.groupby("cid").size()
        for cid, n in n_edits.items():
            if int(z["test_n_edits"][row_of[cid]]) != n:
                raise SystemExit(f"{task} cid {cid}: attention covers {z['test_n_edits'][row_of[cid]]} edits, store has {n}")
        rows = edits["cid"].map(row_of).to_numpy()
        att.append(z["test_attention"][rows, slots.to_numpy()].astype("float64"))
    return pd.Series(np.mean(att, axis=0), index=edits.index)


def export_task(task: str, store: Path) -> pd.DataFrame:
    out = DATA / task
    out.mkdir(parents=True, exist_ok=True)
    scores, info = model_scores(task)
    cids = set(scores["cid"])
    cs = pd.read_parquet(store / "changesets.parquet")
    cs = cs[cs["cid"].isin(cids)].reset_index(drop=True)
    # Full read, then filter: attention slots follow file order, which a filtered read does
    # not promise to keep.
    edits = pd.read_parquet(store / "edits.parquet")
    edits = edits[edits["cid"].isin(cids)].reset_index(drop=True)
    if not (edits.groupby("cid")["position"].diff().dropna() > 0).all():
        raise SystemExit("edits.parquet rows are not in position order within a changeset")
    edits["attention"] = attention(task, edits)

    bundle = tree_model.load(task)
    local_p, signals = tree_model.score(bundle, cs)
    cs["local_hybrid"] = local_p
    cs["signals_json"] = [json.dumps(s) for s in signals]

    scores.to_parquet(out / "scores.parquet", index=False)
    cs.to_parquet(out / "changesets.parquet", index=False)
    edits.to_parquet(out / "edits.parquet", index=False)
    (out / "models.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"task": task, "changesets": len(cs), "edits": len(edits),
                      **{k: v["test_pr_auc"] for k, v in info.items()}}))
    return scores.merge(cs, on=["cid", "label"])


# ------------------------------------------------------------------------ scenarios

# Each scenario: (key, task, title, blurb, rule). A rule picks rows from the merged
# scores+changesets frame of its task; the most clear-cut row (by the headline score) wins.
def _has(token: str):
    return lambda df: df["diff_tokens"].fillna("").str.contains(token, regex=False)


SCENARIOS = [
    ("pokemon_park", "ovid", "A park that isn't there",
     "Pokemon Go players drew fake parks to spawn game items. Look for +leisure=park on a new account.",
     lambda df: (df.label == 1) & _has("+leisure=park")(df) & (df.n_edits <= 12), "high"),
    ("fake_road", "ovid", "An invented road",
     "A motorway or primary road drawn where none exists.",
     lambda df: (df.label == 1) & (_has("+highway=motorway")(df) | _has("+highway=primary")(df))
     & (df.n_edits <= 30), "high"),
    ("deleted_landmark", "ovid_modern", "A landmark deleted",
     "Historic features removed outright.",
     lambda df: (df.label == 1) & _has("-historic")(df) & (df.n_edits <= 20), "high"),
    # Chosen by hand: most name vandalism in the test set is hate speech, and this one makes
    # the same point (free text the model has to read) without putting slurs on screen.
    ("name_spam", "ovid", "An advert typed into a street name",
     "Detroit Avenue renamed to a paragraph of car-donation spam. The text signal carries this one.",
     lambda df: df.cid == 63588505, "high"),
    ("modern_vandalism", "ovid_modern", "Reverted as vandalism (2021+)",
     "A recent changeset another mapper reverted, citing vandalism.",
     lambda df: (df.label == 1) & (df.corpus == "R") & (df.n_edits <= 20), "high"),
    ("careful_mapper", "ovid_modern", "A careful, experienced mapper",
     "Many edits, sources cited, long history. The model stays calm.",
     lambda df: (df.label == 0) & (df.n_edits >= 50) & (df.user_n_prior >= 50), "low"),
    ("false_alarm", "ovid", "A false alarm",
     "Scored high, but nobody reverted it. Where the model is wrong, and why a human still decides.",
     lambda df: (df.label == 0), "high"),
    ("missed", "ovid", "A miss",
     "Reverted, yet scored low. The edit looked ordinary and the account looked established.",
     lambda df: (df.label == 1), "low"),
]


def pick_scenarios(frames: dict[str, pd.DataFrame]) -> list[dict]:
    picked = []
    for key, task, title, blurb, rule, end in SCENARIOS:
        df = frames[task]
        hits = df[rule(df)].sort_values(HEADLINE_MODEL[task], ascending=(end == "low"))
        if hits.empty:
            print(f"scenario {key}: no candidate, skipped")
            continue
        picked.append({"key": key, "task": task, "cid": int(hits.iloc[0]["cid"]),
                       "title": title, "blurb": blurb})
    return picked


def fetch_geometry(scenarios: list[dict]) -> None:
    session = PoliteSession(4.0)
    for s in scenarios:
        try:
            got = osm_live.cached_display(session, s["cid"])
        except (osm_live.ChangesetNotFound, osm_live.UpstreamError) as exc:
            print(f"geometry {s['cid']}: {exc}")
            continue
        print(f"geometry {s['cid']}: {len(got['display'])} edits")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, default=tree_model.default_store())
    ap.add_argument("--no-geometry", action="store_true")
    args = ap.parse_args()
    frames = {task: export_task(task, args.store) for task in TASKS}
    scenarios = pick_scenarios(frames)
    (DATA / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    if not args.no_geometry:
        fetch_geometry(scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
