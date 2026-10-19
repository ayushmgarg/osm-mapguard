# MapGuard runbook — from raw OSM data to trained models

Three places, three jobs. Nothing crosses the lines below.

| Where | Does | Never does |
|---|---|---|
| **Laptop, on a good connection** | fetch, build the feature store, bundle | train a neural model |
| **SVKM cluster** (`cluster/*/*.ipynb`) | train and evaluate | fetch from OSM or OSMCha |
| **M4** (`train/setup_env.sh`) | train and evaluate | fetch from OSM or OSMCha |

Every command is resumable: stop with Ctrl+C, run it again, it continues.

---

## Shortcut — train straight from the release

Every fetch below is already done (2026-09-19). The GitHub releases carry the results:

- **v1.1.0** (feature store schema v4, the input of every training round):
  - `mapguard-dataset-v1.1.0.zip`: the full dataset (pseudonymised feature store, OSMCha
    labels, corpus R revert provenance, splits, dataset card). Unzip it and pass
    `--feature-store <dir>/feature_store` to `train/run_pipeline.py`.
  - `mapguard_bundle.zip`: code, notebooks, the feature store and frozen splits, as of the
    release commit. Upload it to the cluster, or unzip it on the M4, and go to Step 4.
    The code inside is a snapshot; for current code, use a clone plus the dataset zip.
- **v1.2.0**: optional side tables (`object_context`, `where_features`), keyed by changeset id.

Pseudonymisation does not change results: baselines on the release bundle match the raw store
exactly (XGBoost `ovid` PR-AUC 0.8595 on both).

```bash
gh release download v1.1.0 -p mapguard-dataset-v1.1.0.zip
```

Steps 1–3 are only for rebuilding the data from scratch.

## Step 1 — fetch (laptop)

Which source gives what, and why Geofabrik/OSMnx/QuickOSM are not used:
[docs/data/DATA_SOURCES.md](../data/DATA_SOURCES.md).

### Order at a glance

| Step | Command | Gives | Time (measured or estimated) |
|---|---|---|---|
| 1-0 | `mine_changesets.py` | corpus R candidates: 5 years of reverts + a 1/400 sample | measured 2 h 10 min at 0.45 MB/s for 3.9 GB |
| 1a | `fetch_osmcha_labels.py` | corpus M labels (2023+) | osmcha.org-bound; it was overloaded on 2026-09-18 |
| 1b | `fetch_osm.py select --corpora R` / `--corpora MN` | chosen rows | seconds |
| 1c | `fetch_osm.py changesets` | metadata (100 per call) + osmChange | ~3–4 h for ~24k downloads |
| 1d | `fetch_osm.py versions` | previous versions | Corpus A measured ~1 h (458k objects) |
| 1e | `fetch_osm.py users`, `histories` | account age, point-in-time history | Corpus A measured: users 2 min; histories 14,924 calls ~2 h |

**The `ovid` task only needs Corpus A**, which is fully fetched. So Step 2 can run as soon
as 1d and 1e have finished for Corpus A, and training can start while R and M are still
being fetched. Rebuild and re-bundle when they finish.

### 1-0. Mine five years of reverts from the changeset dump (~3 h)

```bash
python scripts/mine_changesets.py --since 2021-09-01 --until 2026-09-14
```

It reads only the byte range from 2021-09 onwards of `changesets-latest.osm.bz2`, in
parallel range requests, and stores reverts, a hash sample and monthly counts in
`raw/changeset_dump.sqlite`. It is resumable per 16 MB chunk. `--stats` shows progress.

Needs `.env` with `OSMCHA_TOKEN` and `MAPGUARD_CONTACT` (see `.env.example`). Data goes to
`MAPGUARD_DATA` (default `C:\osm-data`). Keep the laptop plugged in and awake.

Progress of everything below, at any time:

```bash
python scripts/fetch_osm.py status
```

### 1a. Finish the OSMCha labels (~1–3 h, depends on osmcha.org)

The store has gaps: 2023Q4 has no harmful labels, and 2025Q4–2026Q1 have no cleared ones.

```bash
python scripts/fetch_osmcha_labels.py --since 2023-01-01 --until 2026-09-16 --no-count
```

```bash
python scripts/fetch_osmcha_labels.py --since 2023-01-01 --until 2026-09-16 --no-count --retry-gaps
```

Check `python scripts/fetch_osmcha_labels.py --stats`. If osmcha.org is returning HTTP 0/5xx
for every window, it is down — stop and try later, do not hammer it.

### 1b. Choose the rows (seconds)

Corpus R, as soon as 1-0 has finished:

```bash
python scripts/fetch_osm.py select --corpora R
```

Corpora M and N, only after 1a is complete:

```bash
python scripts/fetch_osm.py select --corpora MN
```

4,000 harmful + 4,000 reviewed-clean (M), water-filled across quarters, plus ~6,400 random
candidate ids for N (oversampled; 4,000 are kept). Re-running does nothing; `--force`
rebuilds, and must only be used before 1c.

### 1c. Modern metadata + osmChange contents (~3–4 h at 2 req/s)

```bash
python scripts/fetch_osm.py changesets --rate 2
```

### 1d. Previous object versions (duration unmeasured — one call per 725 objects) — can run alongside 1c

Parses every stored osmChange (Corpus A now, M/N when 1c has finished), then multi-fetches
the previous version of each modified or deleted object, 725 per call.

```bash
python scripts/fetch_osm.py versions --rate 1
```

Run it **again after 1c finishes** so M and N are planned too.

### 1e. Accounts and point-in-time histories (~2–5 h) — after 1c

```bash
python scripts/fetch_osm.py users --rate 1
```

```bash
python scripts/fetch_osm.py histories --rate 1.5
```

Keep the total OSM API rate around 4 req/s across all terminals.

### 1f. Optional — more adiff geometry, overnight

```bash
python scripts/fetch_corpus_a_geometry.py
```

~122 s per changeset, so ~300 per night. It now runs in hash order, so wherever it stops the
enriched set is an unbiased random sample. (The first 1,055 were fetched in id order and lean
towards 2014 — the build report says how many rows have adiff; B1 claims on full way
geometry must be made on that subset and must name its size.)

---

## Step 2 — build the feature store (laptop, ~20–40 min, no network)

```bash
python -m mapguard.features.build_store
```

Writes `C:\osm-data\interim\{changesets,edits,user_seq}.parquet` and `build_report.json`.

**Read `build_report.json` before going further.** For each corpus:

| Check | Acceptable | If not |
|---|---|---|
| `dropped_no_contents` | a handful | re-run 1c |
| `history_missing_label_0` vs `_label_1` | similar *rates* | re-run 1e; if still unequal, B4 results measure the fetch — report that |
| `account_age_known_label_*` | similar rates | re-run `users` |
| `edits_prev_version / edits_eligible_prev_version` | > 90% | re-run 1d |

Then copy the report into the repo so the paper's numbers have a source:

```bash
cp C:/osm-data/interim/build_report.json results/feature_store_build_report.json
```

Optional, CPU-only sanity check before anything goes to a GPU (a few minutes):

```bash
python train/01_baselines/baselines_train.py --set task=ovid
```

B0 must stay below ROC-AUC 0.85. If it does not, stop: the labels leak through metadata.

---

## Step 3 — bundle (laptop, ~2 min)

Commit first, so the bundle's git SHA describes it.

```bash
python cluster/make_notebooks.py
```

```bash
python cluster/bundle.py
```

Produces `cluster/_bundles/mapguard_bundle.zip` — code, notebooks, feature store, frozen
splits and a manifest. The same zip goes to the cluster and to the M4.

---

## Training plan — round 2 (model v2, see docs/model/MODEL_V2.md)

Every command is `python train/run_pipeline.py --feature-store data/interim` plus:

| # | Arguments | Answers | Priority |
|---|---|---|---|
| 0 | `--task ovid --quick` | everything runs (smoke test, ~20 min) | first |
| 1 | `--task ovid --stages 00,01,02,07` | v2 headline vs XGBoost, hybrid, ensemble, Ovid's 81.66 | **must** |
| 2 | `--task ovid --stages 03` | 9 ablation rows × 2 seeds: which branch, MIL vs mean, author weighting | **must** |
| 3 | `--task ovid_modern --stages 01,02,07,06` | 2014–19 → 2021–26 under the same rule | **must** |
| 4 | `--task T1 --stages 01,02,07,06` | human-reviewed triage | should |
| 5 | `--task T2 --stages 01,02,07,06` | detection vs random traffic | should |
| 6 | `--task ovid --stages 04,05` | Stage 1 again (round 1: no gain) | only if time |

- Lead with `test_author_robust.one_per_author_pr_auc`, not only PR-AUC: Ovid's labels are
  concentrated in a few bulk-reverted accounts (docs/model/MODEL_V2.md §3).
- Stage 07 needs the same task's 01 and 02 runs; the pipeline orders it after them.
- Never report a directory ending in `_quick`.
- Hardware is used automatically (see train/setup_env.sh). If the Apple GPU does not attach,
  CPU training takes ~5–8 min per model.

## Step 4a — train on the SVKM cluster

See RUN_ALTAIR.md for the portal. For every job: image `tensorflow_pbs:23.07-tf2-py3`,
1 GPU, 8 CPUs, 32 GB; upload `mapguard_bundle.zip` and the notebook for the stage.

| Order | Notebook | Time | Needs |
|---|---|---|---|
| 1 | `cluster/00_wiring_check` | < 5 min | — **run first; stop if FAIL** |
| 2 | `cluster/99_full_pipeline` with `QUICK = True` | ~15 min | 1 |
| 3 | `cluster/01_baselines` | minutes | — |
| 4 | `cluster/02_stage2_supervised` | 30–90 min | 3 |
| 5 | `cluster/04_stage1_pretrain` | 20–60 min | 3 |
| 6 | `cluster/03_stage2_ablations` | hours | 5 |
| 7 | `cluster/05_stage1_label_efficiency` | hours | 5 |
| 8 | `cluster/06_temporal_evaluation` | 30–90 min | 4 |

Jobs 6, 7 and 8 are independent of each other — submit them in parallel if slices are free.
Each notebook's last cell zips `experiments/` for download.

## Step 4b — train on the M4

```bash
unzip mapguard_bundle.zip -d mapguard && cd mapguard
```

```bash
bash train/setup_env.sh
```

```bash
source .venv-train/bin/activate && python train/run_pipeline.py --feature-store data/interim --task ovid --quick
```

```bash
python train/run_pipeline.py --feature-store data/interim --task ovid
```

Same stages, same order, same outputs as the cluster. For the modern tasks add `--task T1`
or `--task T2` (needs step 1b–1e done).

---

## What to report, and what not to

- Report runs **without** `_quick` in the directory name only.
- R1b is the result, and it is Ovid's own protocol (user-disjoint, their section 4.1): compare
  `ovid_table` F1/accuracy with Ovid's 81.66 / 81.37 on R1b. R1a is user-leaky; if you run it,
  label it so.
- `03` ablation deltas for B1 on the full corpus mix edits with and without full geometry;
  the build report says how many changesets have adiff.
- `04` zero-shot AUC: report it whatever it is.
- Every `results.json` has a sibling `provenance.json` with the git SHA, data fingerprints
  and split file — quote those, not memory.
