"""Package round 4's trained models for a GitHub release.

    python scripts/package_checkpoints.py            # -> dist/mapguard-checkpoints-r4/ + three zips

Three archives, each with a README and per-file SHA-256:

  mapguard-checkpoints-r4-deep.zip    the deep model (stage 2, v3: two transformer blocks +
                                      text embeddings), 6 tasks x 3 seeds, plus the round-1-style
                                      Ovid reproduction (ovid, 3 seeds)
  mapguard-checkpoints-r4-stack.zip   the deep fold models behind stacking: 6 tasks x 3 seeds x 5
                                      author folds, plus each stack run's out-of-fold scores
  mapguard-hybrid-refit-cpu.zip       the hybrid tree models the review console scores live with
                                      (visual/tree_model.py refit on CPU, frozen R1b split)

Per deep run: model.weights.h5 (the evaluated weights), model_config.json (build kwargs: rebuild
with train/common/fit.build), normaliser.json (feature scaling fitted on the training rows),
config.json, results.json, provenance.json. best.weights.h5 is the same checkpoint before the
reload and is left out. Nothing here contains user ids: inputs are pseudonymised.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds/raw/experiments/runs"
OUT = REPO / "dist" / "mapguard-checkpoints-r4"
RUN_FILES = ("model.weights.h5", "model_config.json", "normaliser.json", "config.json",
             "results.json", "provenance.json")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack(zip_path: Path, files: list[tuple[Path, str]], readme: str) -> dict:
    sums = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", readme)
        for src, arc in files:
            z.write(src, arc)
            sums[arc] = sha(src)
        z.writestr("SHA256SUMS.json", json.dumps(sums, indent=1))
    return {"zip": zip_path.name, "files": len(files), "mb": round(zip_path.stat().st_size / 1e6, 1),
            "sha256": sha(zip_path)}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    summary = {}

    deep_dirs = sorted(d for d in (RUNS / "stage2_supervised").iterdir()
                       if d.is_dir() and not d.name.endswith("_quick"))
    files = [(d / f, f"deep/{d.name}/{f}") for d in deep_dirs for f in RUN_FILES if (d / f).exists()]
    index = []
    for d in deep_dirs:
        r = json.loads((d / "results.json").read_text())
        index.append({"run": d.name, "task": r.get("task"), "test_pr_auc": r["headline"]["test_pr_auc"][0],
                      "text_emb_dim": r["training"].get("text_emb_dim"), "edit_layers": r["training"].get("edit_layers")})
    readme = (
        "# MapGuard round-4 deep models\n\n"
        f"{len(deep_dirs)} runs: the v3 deep model for 6 tasks x 3 seeds, and the Ovid reproduction "
        "(B5 only, mean pooling) for ovid x 3 seeds. Trained on an NVIDIA H100 with TensorFlow 2.12 "
        "(Keras 2), mixed precision; split R1b frozen (split_seed 20260904).\n\n"
        "Rebuild and load (inputs must be prepared exactly as in training - train/common/preprocess.py, "
        "with the saved normaliser):\n\n```python\n"
        "import json, sys; sys.path[:0] = ['train', 'src']\n"
        "from common.fit import build\n"
        "kw = json.load(open('deep/<run>/model_config.json'))['build_kwargs']\n"
        "model = build(kw); model.load_weights('deep/<run>/model.weights.h5')\n```\n\n"
        "Models with text_emb_dim = 768 expect the stage-08 sentence embeddings "
        "(paraphrase-multilingual-MiniLM-L12-v2) as input `text_emb`.\n\n"
        "| Run | Task | Test PR-AUC | text_emb_dim | edit_layers |\n|---|---|---|---|---|\n"
        + "".join(f"| {i['run']} | {i['task']} | {i['test_pr_auc']} | {i['text_emb_dim']} | {i['edit_layers']} |\n" for i in index)
        + "\nLicence: model weights derived from ODbL data (c) OpenStreetMap contributors; code MIT.\n")
    summary["deep"] = pack(OUT / "mapguard-checkpoints-r4-deep.zip", files, readme)

    stack_dirs = sorted(d for d in (RUNS / "stack").iterdir() if d.is_dir() and not d.name.endswith("_quick"))
    files = []
    for s in stack_dirs:
        for f in ("results.json", "config.json", "deep_oof.npz"):
            if (s / f).exists():
                files.append((s / f, f"stack/{s.name}/{f}"))
        for fold in sorted((s / "folds" / "stack_fold").glob("*")):
            for f in ("model.weights.h5", "model_config.json", "normaliser.json"):
                if (fold / f).exists():
                    files.append((fold / f, f"stack/{s.name}/folds/{fold.name}/{f}"))
    readme = (
        "# MapGuard round-4 stacking fold models\n\n"
        f"{len(stack_dirs)} stack runs (6 tasks x 3 seeds). Each trained the deep model on 5 author folds "
        "of the training rows; every training changeset's deep score comes from a fold model that never "
        "saw it, and validation/test get the mean of the 5. `deep_oof.npz` holds those scores (cids, deep, "
        "role: 0 train, 1 val, 2 test). The trees on top are refit from them with "
        "train/09_stack/stack_train.py.\n\nLoad a fold model like the deep models (see the deep archive).\n")
    summary["stack"] = pack(OUT / "mapguard-checkpoints-r4-stack.zip", files, readme)

    models = REPO / "visual" / "models"
    files = [(p, f"hybrid/{p.name}") for p in sorted(models.glob("*.joblib"))]
    readme = (
        "# MapGuard hybrid tree models, refit on CPU\n\n"
        "The models the review console (visual/) scores live changesets with. visual/tree_model.py refits "
        "round 4's hybrid (XGBoost on tabular features + out-of-fold edit-token and text models) from the "
        "released feature store v1.1.0 with the same code, the frozen R1b split and seed 20260904, and "
        "checks test PR-AUC against round 4's stored predictions (tolerance 0.005).\n\n"
        "Files are joblib pickles: **load them only from this release**, with the library versions in "
        "visual/requirements.txt (xgboost 3.4.1, scikit-learn 1.9.1, numpy 2.5.4, Python 3.12). Or "
        "regenerate them in ~20 s: `python visual/tree_model.py --task ovid --task ovid_modern`.\n")
    if files:
        summary["hybrid"] = pack(OUT / "mapguard-hybrid-refit-cpu.zip", files, readme)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
