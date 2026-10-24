"""Generate the cluster notebooks - one per training stage, plus one that runs everything.

Notebooks are build artefacts, not source. Each one is four thin cells around a single
call to `train/run_pipeline.py`, so the notebook, the M4 terminal and any other machine all
run *exactly* the same code path. There is no training logic in here to drift.

    python cluster/make_notebooks.py

Folders mirror train/:

    cluster/00_wiring_check/            run first on every new image
    cluster/01_baselines/               CPU is enough
    cluster/02_stage2_supervised/
    cluster/03_stage2_ablations/        run 04 first so row 07 has an encoder
    cluster/04_stage1_pretrain/
    cluster/05_stage1_label_efficiency/ needs a finished 04
    cluster/06_temporal_evaluation/
    cluster/99_full_pipeline/           everything, in dependency order, in one job

Hand-editing a generated notebook while debugging on the cluster is fine; regenerating
overwrites it.
"""

from __future__ import annotations

import json
from pathlib import Path

CLUSTER = Path(__file__).resolve().parent

STAGES = [
    ("00_wiring_check", "00", "Wiring check",
     "Pushes random tensors through every real code path - every ablation branch set, weight "
     "save/rebuild, Stage 1 pretraining and encoder transfer - for one tiny epoch each. "
     "Produces **no results**, only PASS/FAIL. Run it first on every new image: it is the only "
     "proof that the model builds under this image's Keras version.",
     "under 5 min on a GPU", "nothing"),
    ("01_baselines", "01", "Baselines (B0 tripwire, logistic regression, XGBoost)",
     "Also **creates the frozen split** in experiments/splits/ that every later stage reuses. "
     "If B0 (metadata only) reaches ROC-AUC >= 0.85 it stops: the labels are readable from "
     "metadata and nothing downstream can be trusted.",
     "a few minutes, CPU", "nothing"),
    ("02_stage2_supervised", "02", "Stage 2 - the supervised five-branch classifier",
     "The headline R1 result on the user-disjoint split (R1b), which is also Ovid's own "
     "protocol. For the ovid task it also trains the Ovid-equivalent model (ovid_repro.json) "
     "on the same split, the reproduction row to set beside Ovid's published F1 of 81.66.",
     "30-90 min on one MIG slice", "01 (for the frozen split; created automatically if absent)"),
    ("03_stage2_ablations", "03", "Stage 2 ablation grid",
     "Seven rows x two seeds through the same training core as stage 02. Row 07 (pretrained "
     "encoder) is skipped unless a stage 04 run exists for this task.",
     "several hours - submit as its own job", "04 for row 07"),
    ("04_stage1_pretrain", "04", "Stage 1 - corruption-contrast pretraining + zero-shot check",
     "Pretrains the edit encoder on training-fold edits only (evaluation authors excluded, "
     "proof in disjointness.json), then scores real labelled changesets with the frozen "
     "pretext model. Report the zero-shot number whatever it is.",
     "20-60 min", "nothing"),
    ("05_stage1_label_efficiency", "05", "Label efficiency (R4): pretrained vs scratch",
     "Four label budgets x two seeds x {scratch, pretrained}, nested and class-stratified.",
     "several hours - submit as its own job", "04"),
    ("06_temporal_evaluation", "06", "Temporal evaluation (R3)",
     "Train strictly before the cutoff, test strictly after, with authors active on both sides "
     "removed from training. ovid: 2014-2018 -> 2019. T1/T2: -> 2025-07 onwards.",
     "30-90 min", "02 for the in-distribution reference (optional)"),
    ("99_full_pipeline", "00,01,02,04,03,05,06", "Full pipeline",
     "Every stage in dependency order in one job. Stops at the first failure.",
     "most of a day - run --quick first", "nothing"),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip()}


def notebook(cells: list[dict]) -> dict:
    out = []
    for c in cells:
        lines = c["source"].split("\n")
        out.append({**c, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r'''
# 1. Find the bundle, unpack it once, install only what is missing.
import glob, json, os, pathlib, subprocess, sys, zipfile

HOME = pathlib.Path.cwd()
if (HOME / "train" / "run_pipeline.py").exists():   # cell re-run after chdir below
    HOME = HOME.parent
candidates = sorted(glob.glob("mapguard_bundle*.zip") + glob.glob("../mapguard_bundle*.zip")
                    + glob.glob("/data/**/mapguard_bundle*.zip", recursive=True),
                    key=os.path.getmtime)
ROOT = HOME / "mapguard"
# Unpack when there is no copy yet, or when a newer bundle was uploaded. experiments/runs is
# never inside a bundle, so earlier results survive an update.
if candidates:
    with zipfile.ZipFile(candidates[-1]) as z:
        fresh = json.loads(z.read("BUNDLE_MANIFEST.json"))
        current = ROOT / "BUNDLE_MANIFEST.json"
        if not current.exists() or json.loads(current.read_text()).get("generated") != fresh.get("generated"):
            z.extractall(ROOT)
            print("unpacked", candidates[-1], "->", ROOT)
elif not (ROOT / "train" / "run_pipeline.py").exists():
    raise SystemExit("upload cluster/_bundles/mapguard_bundle.zip next to this notebook first")
os.chdir(ROOT)
manifest = json.loads((ROOT / "BUNDLE_MANIFEST.json").read_text())
print(json.dumps({k: manifest.get(k) for k in ("git_sha", "git_dirty", "generated", "feature_schema")}, indent=2))

# Install missing packages without letting pip upgrade numpy: the image's TensorFlow is
# built against its own numpy, and a silent upgrade breaks it.
import importlib.util, numpy
missing = [pkg for mod, pkg in (("pyarrow", "pyarrow"), ("pandas", "pandas"),
                                ("sklearn", "scikit-learn"), ("xgboost", "xgboost"))
           if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing,
                           f"numpy=={numpy.__version__}"])
import tensorflow as tf
print("python", sys.version.split()[0], "| tensorflow", tf.__version__,
      "| GPUs", [g.name for g in tf.config.list_physical_devices("GPU")])
'''.strip()

RUN = r'''
# 2. Run. QUICK=True is a smoke test with tiny epochs - never report its numbers.
TASK = "ovid"          # ovid | T1 | T2  (T1/T2 need corpora M and N in the feature store)
QUICK = False
STAGES = "{stages}"

cmd = [sys.executable, "-u", "train/run_pipeline.py", "--feature-store", "data/interim",
       "--task", TASK, "--stages", STAGES] + (["--quick"] if QUICK else [])
print(" ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
print("exit code", proc.wait())
'''.strip()

SHOW = r'''
# 3. Headlines of the runs this job produced.
summaries = sorted(glob.glob("experiments/pipeline/*.json"), key=os.path.getmtime)
summary = json.loads(pathlib.Path(summaries[-1]).read_text())
print(json.dumps(summary["stages"], indent=1))
for run_dir in summary["runs_created"]:
    res = pathlib.Path(run_dir) / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        print(f"\n{run_dir}\n  headline: {json.dumps(r.get('headline'))[:600]}")
'''.strip()

PACK = r'''
# 4. Pack results for download (Files -> right-click -> Download). Weights included.
import shutil, time
out = HOME / f"mapguard_results_{STAGES.replace(',', '-')}_{TASK}_{time.strftime('%Y%m%d-%H%M%S')}"
shutil.make_archive(str(out), "zip", root_dir=ROOT, base_dir="experiments")
print("download:", str(out) + ".zip")
'''.strip()


def stage_notebook(folder, stages, title, what, time_needed, needs) -> dict:
    return notebook([
        md(f"""
# {folder[:2]} - {title}

{what}

| | |
|---|---|
| Expected time | {time_needed} |
| Needs first | {needs} |
| Runs | `python train/run_pipeline.py --feature-store data/interim --task <TASK> --stages {stages}` |

**Before running:** upload `mapguard_bundle.zip` (built with `python cluster/bundle.py`) next to
this notebook. Submit with the `tensorflow_pbs` image, 1 GPU, 8 CPUs, 32 GB. See RUN_ALTAIR.md.

Outputs land in `mapguard/experiments/runs/<stage>/...`; cell 4 zips them for download.
Every run directory carries its config, git SHA, data fingerprint and split file.
"""),
        code(SETUP),
        code(RUN.replace("{stages}", stages)),
        code(SHOW),
        code(PACK),
    ])


README = """# {folder}

{title}

{what}

- Expected time: {time_needed}
- Needs first: {needs}
- Equivalent command on the M4 or any machine:

```bash
python train/run_pipeline.py --feature-store data/interim --task ovid --stages {stages}
```

Generated by `cluster/make_notebooks.py` - edit that, not this file.
"""


def main() -> None:
    for folder, stages, title, what, time_needed, needs in STAGES:
        d = CLUSTER / folder
        d.mkdir(parents=True, exist_ok=True)
        nb = stage_notebook(folder, stages, title, what, time_needed, needs)
        (d / f"{folder}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        (d / "README.md").write_text(README.format(folder=folder, title=title, what=what,
                                                   time_needed=time_needed, needs=needs,
                                                   stages=stages), encoding="utf-8")
        print("wrote", d)


if __name__ == "__main__":
    main()
