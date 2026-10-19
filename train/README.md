# train/ — every training and evaluation entry point

Everything that fits a model, evaluates one, or produces a number for the paper starts
here. Library code (feature extraction, the model definition, metrics, splits) lives in
`src/mapguard/`; these folders are the *runs* built on top of it.

Every stage reads the feature store through one contract,
[`common/data_contract.py`](common/data_contract.py), which
`python -m mapguard.features.build_store` writes to. Default task is `ovid` (Corpus A, the
only corpus complete today); `T1`/`T2` need corpora M and N - see [RUNBOOK.md](../docs/running/RUNBOOK.md).

## One command

```bash
python train/run_pipeline.py --feature-store data/interim --task ovid --quick
```

runs 00 -> 08 -> 01 -> 02 -> 07 -> 09 -> 04 -> 03 -> 05 -> 06 (`ORDER` in
[`run_pipeline.py`](run_pipeline.py)), wiring each stage's output into the next: 01 and 02 into
the 07 ensemble, the Stage 1 encoder into 03 and 05, and the R1b run into 06. `--stages 02,06` runs a subset;
later stages find earlier runs from any session through `experiments/runs/index.jsonl`.
`--quick` is a smoke test and tags every run directory `_quick`: never report those numbers.
The cluster notebooks call exactly this.

## Folder names: a number, then the plan stage

The plan numbers stages by *idea* — Stage 1 is self-supervised pretraining, Stage 2 is the
supervised classifier. They do not run in that order: the supervised model is built and
validated first, and pretraining is week-4 conditional (PROJECT_PLAN §13). So folders are
numbered, and tagged with the plan stage they belong to.

| Folder | Plan stage | What it produces | Machine |
|---|---|---|---|
| [`00_wiring_check/`](00_wiring_check/) | — | proof the code runs on this machine; **no results** | any, first |
| [`08_text_embeddings/`](08_text_embeddings/) | v3 · optional input | multilingual sentence embeddings written next to the feature store (needs `sentence-transformers`) | GPU preferred |
| [`01_baselines/`](01_baselines/) | baselines · R1, R2 | B0 tripwire, logistic regression, XGBoost, hybrid trees | CPU, any |
| [`02_stage2_supervised/`](02_stage2_supervised/) | **Stage 2** · R1, R2 | the six-branch classifier + its evaluation | M4 or cluster |
| [`07_ensemble/`](07_ensemble/) | v2 | blend of 02 and the 01 hybrid, weight chosen on validation | CPU, any |
| [`09_stack/`](09_stack/) | v3 | 5 author-fold deep models whose out-of-fold score becomes a tree feature | M4 or cluster |
| [`04_stage1_pretrain/`](04_stage1_pretrain/) | **Stage 1** · conditional | pretrained edit encoder + zero-shot transfer check | cluster |
| [`03_stage2_ablations/`](03_stage2_ablations/) | **Stage 2** · ablation grid | 15 rows in `rows/` × seeds | cluster |
| [`05_stage1_label_efficiency/`](05_stage1_label_efficiency/) | **Stage 1 → 2** · R4 | pretrained vs scratch across label budgets | cluster |
| [`06_temporal_evaluation/`](06_temporal_evaluation/) | **Stage 2** · R3 | train on the past, test on the future | M4 or cluster |

The table is in run order. Folder numbers are the order the stages were *added*, so 07–09
run earlier than their numbers suggest.

Stages 4 and 5 are the first thing cut if the timeline slips. Stage 2 was designed to stand
alone; nothing in 01–03 or 06 depends on them.

## How a run works

```bash
python train/02_stage2_supervised/stage2_train.py --config train/02_stage2_supervised/config.json
```

Each script:

1. loads its `config.json` (override any key with `--set train.epochs=5`)
2. validates the feature store against the data contract — **fails loudly, never fills
   missing columns with zeros**
3. loads or creates a **frozen split file**, so every model on the same task sees exactly
   the same rows
4. writes everything to `experiments/runs/<stage>/<timestamp>_<task>_<split>_<tag>/`:
   `config.json`, `provenance.json`, `results.json`, weights, normaliser, training log

A run directory is self-describing. `stage2_evaluate.py --run <dir>` rebuilds the model
from what is inside it, with nothing else needed.

## Rules these scripts enforce — do not route around them

- **Splits are frozen files.** A split recomputed at training time silently changes when the
  split code changes. `common/splits_io.py` refuses to reuse a split whose ids no longer
  match the data.
- **Normalisation statistics come from training rows only**, and are saved with the run.
- **Every row has at least one real edit.** A fully masked row makes attention produce NaN
  and poisons the whole batch; `ensure_nonempty` inserts a neutral null edit.
- **Masking is checked on the real model**, before and after training. A failure aborts the
  run rather than producing a number.
- **Pretraining never sees evaluation rows.** Stage 1 excludes every changeset id *and* uid
  in the validation and test splits, and writes the proof to the run directory.
- **Tasks are explicit.** `T1` (triage: harmful vs reviewed-clean), `T2` (detection: harmful
  vs random unreviewed), or `ovid` (the historical benchmark). A result without its task is
  not reportable, so it is part of the run directory name.

## Environment

TensorFlow 2.12 (the cluster image) up to 2.19. Built to run under both Keras 2 (the cluster's
`tensorflow_pbs:23.07-tf2-py3`) and Keras 3 (a current laptop or M4) —
see the header of `src/mapguard/models/stage2.py` for why that needs care.

On Apple Silicon, time one epoch with `"device": "cpu"` and with `"auto"` before choosing.
Masked LSTMs and GRUs often miss the fast Metal kernels and can run slower on the GPU.

**Run `00_wiring_check` first on every new machine or image.**
