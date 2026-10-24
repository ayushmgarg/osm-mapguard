# experiments/rounds/r1_2026-09-21_m4 — first full training round (2026-09-21)

Run on a MacBook M4 (16 GB), bundle release v1.0.1, code `3c97b88`. **CPU only**: the
Apple GPU was not used (TensorFlow 2.19.1 was installed and `tensorflow-metal` did not attach).
Each full training took ~3–8 min on 10 cores.

## Layout

```
experiments/rounds/r1_2026-09-21_m4/
  raw/experiments/        the folder as it came back - untouched, nothing edited or removed
    pipeline/             one log + summary per run_pipeline invocation
    runs/<stage>/<run>/   results.json, provenance.json, training_log.csv, weights, predictions
    runs/index.jsonl      every finished run
    splits/               identical to experiments/splits in the repo (all 12 checked)
  analysis/
    summary.md            every table below, generated
    summary.json          the same numbers, machine-readable
```

Regenerate the analysis with `python scripts/summarise_results.py experiments/rounds/r1_2026-09-21_m4`.
Directories ending in `_quick` (19) are the smoke test and are excluded from every table.
Weights (`*.h5`, 64 MB) are git-ignored and stay local.

## What was run

| Task | Stages | Note |
|---|---|---|
| `ovid` | 00–06, all | 00–03 in one invocation (its log has no summary - it stopped after the ablation grid), 05–06 in a second |
| `ovid_modern` | 01, 02, 06 | |
| `T1` | 01, 02, 06 | |
| `T2` | 01, 02, 06 | |

## Findings

1. **The deep model ties XGBoost; it does not beat it.** ovid: 0.844 vs 0.857 PR-AUC, F1 78.1
   both. ovid_modern: 0.908 vs 0.909. T1: 0.786 vs 0.809. T2: 0.869 vs 0.877. Every gap sits
   inside the confidence intervals.
2. **Below Ovid's published F1.** Our best on Ovid's labels and protocol is F1 78.1 (deep and
   XGBoost) against 81.66. Our Ovid-equivalent network gets 75.0, so *we do not reproduce
   Ovid's number*: the gap comes from the setup (split, features, tuning), not from our additions.
3. **Text is what helps.** Ablations (2 seeds): the text branch (comment + tag text) adds
   +0.035 PR-AUC on its own and is the whole gain of the full model (full 0.844 ≈ text-only
   0.845). Geometry adds +0.014 and sequences/user history +0.016, but neither adds anything
   on top of text.
4. **Self-supervision does not help.** Pretext AUC 0.997 (easy), zero-shot 0.611 (weak real
   signal), and pretrained minus scratch is within ±0.003 at every label budget. With 10% of
   labels, scratch already reaches 0.814 of 0.844: labels are not the bottleneck.
5. **Overfitting starts early.** Train PR-AUC 0.97 vs validation 0.86 by epoch 9; the best
   epoch is 5 (Ovid-equivalent: epoch 1). The model is under-regularised for 12.6k training rows.
6. **Time:** ovid 2014–18 → 2019 drops 0.076 (0.844 → 0.768). ovid_modern 2014–19 → 2021–26
   drops only 0.018 (0.891), but its test set is the vandalism-flag corpus where metadata alone
   scores 0.839 (B0), so part of that is an easy test set. It needs the same temporal run for
   XGBoost and B0 before it can be claimed. T1 holds (+0.007), T2 drops 0.042.
7. **Label noise in Ovid's data.** April 2019 inverts (ROC 0.38): one account's university
   mapathon (`#6mapathonutm_nazmin`, 206 changesets, 62% of that month's positives) was bulk-
   reverted, and the model reads it as organised clean mapping. In Corpus A, 56 authors supply
   37% of positives; negatives come from 6,444 authors. Corpus R caps authors at 5 by design.
8. **Operational reality.** At 1% prevalence, precision is 2.6% (PR-AUC 0.07); precision in the
   top 100 is 0.92. Calibration: ECE 0.04–0.15, worst on T2.
9. **Explanations are weak.** Edit-attention weights are nearly uniform (~1/n), so "which edit"
   is not informative yet. Branch occlusion does work: B5 tabular > B2 text > B4 user > B1 > B3.

## Next plan - proposed, not started

See the bottom of `docs/archive/CHECKLIST_v2.md`.
