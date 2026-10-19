# MapGuard

**Detecting damaging edits to OpenStreetMap.**

OpenStreetMap (OSM) is edited by millions of volunteers. A small share of those edits does
damage: fake parks drawn to spawn game items, invented roads, deleted landmarks, abuse typed
into names. Volunteer reviewers catch these by hand, using rule-based flags.

MapGuard learns from edits that were later reverted or rejected. It gives every *changeset*
(one upload of edits) a score for how likely it is to be damaging, and points to the edit
that drove the score.

---

## The project at a glance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture/mapguard-pipeline-dark.png">
  <img alt="MapGuard pipeline. Four sources (the OSM API, the Ovid label file, the planet changeset dump and OSMCha reviews) feed two fetch steps: the edit fetchers and the label miners. The label miners pass the changeset ids to fetch to the edit fetchers, which write a point-in-time feature store. The feature store is published as a pseudonymised release zip. It also feeds two models, hybrid XGBoost trees and a deep multiple-instance model. Their scores are stacked and blended into the final score, which is scored for PR-AUC on a user-disjoint test set." src="docs/architecture/mapguard-pipeline-light.png">
</picture>

<sub>Interactive version, with search, focus, tracing and export:
[`docs/architecture/mapguard-pipeline.html`](docs/architecture/mapguard-pipeline.html)
(download it and open it in a browser). It is generated from
[`mapguard-pipeline.dataflow.json`](docs/architecture/mapguard-pipeline.dataflow.json);
see [docs/architecture/](docs/architecture/README.md).</sub>

Read the diagram left to right. Each column is a stage, with its code:

| Stage | What happens | Where the code lives |
|---|---|---|
| **1. Sources** | Four sources:<br>• the **OSM API** (changesets, edits, earlier versions of edited objects, user history);<br>• the **planet changeset dump** (every changeset ever made, scanned for reverts);<br>• **OSMCha** (human review verdicts);<br>• the **Ovid label file** (the 2014–19 benchmark, kept in the repo). | [`src/mapguard/data/`](src/mapguard/data/) |
| **2. Fetch + label** | • The **label miners** decide which changesets are damaging or clean. [`mine_changesets.py`](scripts/mine_changesets.py) finds reverts in the dump; [`fetch_osmcha_labels.py`](scripts/fetch_osmcha_labels.py) collects review verdicts.<br>• The **edit fetchers** ([`fetch_osm.py`](scripts/fetch_osm.py)) then download each chosen changeset and everything around it.<br>• Every request goes through one shared rate limit ([`PoliteSession`](src/mapguard/data/http.py)).<br>• Results go into resumable SQLite files, so a rerun costs nothing. | [`scripts/`](scripts/), [`src/mapguard/data/`](src/mapguard/data/) |
| **3. Feature store** | [`build_store.py`](src/mapguard/features/build_store.py) turns the raw files into three Parquet tables:<br>• `changesets`: one row per changeset, with labels, counts, text and account features;<br>• `edits`: up to 256 edits per changeset;<br>• `user_seq`: the author's history.<br>User history is **point-in-time**: only changesets made *before* the one being scored. The store is also published as a pseudonymised [release](../../releases). | [`src/mapguard/features/`](src/mapguard/features/) |
| **4. Train** | Two model families:<br>• **hybrid trees**: XGBoost on account and count features, plus out-of-fold text and token scores;<br>• a **deep multiple-instance model** that reads every edit.<br>[`run_pipeline.py`](train/run_pipeline.py) runs every stage in order. | [`train/`](train/), [`src/mapguard/models/stage2.py`](src/mapguard/models/stage2.py) |
| **5. Evaluate** | **Stacking** feeds the deep model's score into the trees; **blending** averages the two families.<br>Every model is scored on a test set whose authors never appear in training. | [`src/mapguard/evaluation/`](src/mapguard/evaluation/), [`train/common/scoring.py`](train/common/scoring.py) |

### Words used throughout

| Term | Meaning |
|---|---|
| **changeset** | One upload of edits by one account, with a comment. This is the unit MapGuard scores. |
| **revert** | A later changeset that undoes an earlier one. A revert by a *different* account within 90 days is the main "damaging" label. |
| **OSMCha** | The community review tool. Reviewers mark changesets *harmful* or *good*. |
| **Ovid** | The published model for this task (2021). Its labels are corpus A and its numbers are the baseline to beat. |
| **user-disjoint split (R1b)** | Train, validation and test share no author. Without this, a model can score well by recognising a few prolific vandals. R1a is the leaky random split, reported only for comparison. |
| **PR-AUC** | Area under the precision–recall curve. 1.0 is perfect; on balanced data a random guess scores about 0.5. |
| **one-per-author score** | PR-AUC counting only one changeset per author. It is a stricter check that the model has learnt behaviour, not identities. |
| **MIL (multiple-instance learning)** | Treat a changeset as a bag of edits that is bad if *any* edit is bad, and learn which edit that is. |
| **OOF (out-of-fold)** | A score for each training row from a model that never saw that row. It lets one model feed another without leaking labels. |

---

## Results

Every number is measured on a **user-disjoint test set**: no author in the test set appears in
training. Each is the mean of 3 seeds. Each interval is a 95% bootstrap interval that resamples
authors, not rows. Every number traces back to a file in
[`experiments/rounds/`](experiments/rounds/).

### Against Ovid (the published model for this task, 2021)

Same benchmark labels (changesets later reverted, 2014–2019) and the same protocol
(user-disjoint split, threshold chosen on validation):

| | Precision | Recall | **F1** | **Accuracy** |
|---|---|---|---|---|
| Ovid (published) | 0.804 | 0.830 | 0.817 | 0.814 |
| **MapGuard (stack + embeddings)** | 0.801 | 0.856 | **0.828** [0.814, 0.840] | **0.823** [0.810, 0.835] |

MapGuard is **+1.1 F1 and +0.9 accuracy points above Ovid**. Ovid's figure falls just inside our
interval (P(ours higher) = 0.95). So this is a gain under the same protocol, not proof on an
identical test set: the changesets were re-fetched, and edits that OSM has since hidden were
dropped.

### Every task (PR-AUC, the area under the precision–recall curve; 1.0 is perfect)

| Task | Labels | Best model | **MapGuard** | Tree baseline | Gain [95% CI] |
|---|---|---|---|---|---|
| **ovid** | reverted, 2014–19 (Ovid's benchmark) | stack + emb | **0.906** | 0.898 | **+0.010** [0.006, 0.014] |
| ovid_modern | Ovid's rule (revert mentions vandalism) on 2014–19 + 2021–26 together | ensemble | **0.944** | 0.936 | **+0.009** [0.005, 0.012] |
| revert_vandalism | reverted with "vandalism" in the revert, 2021–26 | ensemble | **0.970** | 0.963 | **+0.007** [0.003, 0.012] |
| revert | reverted by another account, 2021–26 | ensemble | **0.925** | 0.913 | **+0.012** [0.004, 0.020] |
| T1 | OSMCha human review, harmful vs flagged-and-cleared | ensemble | 0.859 | 0.852 | +0.008 [−0.002, 0.019] |
| T2 | OSMCha harmful vs random edits | hybrid + emb | 0.898 | 0.895 | +0.003 [−0.002, 0.008] |

The tree baseline is XGBoost on account, count and text features, plus out-of-fold token and
text scores. It is already stronger than Ovid. A gain in **bold** has a 95% interval that
excludes zero.

**The stricter score.** In Ovid's labels, 56 accounts supply 37% of the damaging changesets.
A model can therefore score well just by recognising a few accounts. Counting **one changeset
per author**, a plain XGBoost drops from 0.86 to 0.77; MapGuard's best ovid model keeps
**0.844**. Every result in this repository reports this score alongside the standard one.

---

## How it works

A changeset is judged on three kinds of evidence. They complement each other: under the
one-per-author measure, each scores only 0.60–0.63 on its own.

| | Question | Signals |
|---|---|---|
| **Who** | Does this account know the map? | account history, edit rate, editor |
| **What** | What did each edit actually do? | edit tokens: `+leisure=park`, `-historic`, `~highway:track>motorway` |
| **Text** | What did they write? | comment and tag text, plus pretrained multilingual sentence embeddings |

The edit tokens ([`edit_tokens.py`](src/mapguard/features/edit_tokens.py)) are the most useful
single signal. A linear model ranks these as the most damaging:
- `+leisure=park`, `+highway=footway`, `+tourism`: the documented Pokémon Go vandalism;
- `+highway=motorway`: fake roads;
- `-historic`: deleted landmarks.

The most benign are `+source`, `+access=private` and `+highway=track`, the habits of careful
mappers.

**Models**

```
                 ┌─ hybrid: gradient-boosted trees on account/count features
changeset ───────┤          + out-of-fold logistic scores of edit tokens, text, embeddings
                 │
                 └─ deep: six-branch neural network (src/mapguard/models/stage2.py)
                      edits ──► per-edit encoder (numbers + hashed tokens + old→new text chars)
                            ──► BiLSTM over edit order
                            ──► 2 transformer blocks over edits
                            ──► gated-attention multiple-instance pooling + max pooling
                      comment/tags ──► character CNN + sentence embeddings
                      account history ──► GRU        account/counts ──► MLP
                                                │
              stack: the deep model's out-of-fold score becomes a feature of the trees
              ensemble: blend of deep + hybrid, weight chosen on validation
```

- **Multiple-instance pooling.** One bad edit among 200 good ones makes a changeset bad, so
  the model learns attention over edits rather than averaging them. With the residual
  connection, attention concentrates on one edit: 2.6× more than uniform, and sharper on
  damaging changesets. That edit is the answer to "which edit?".
- **Author-balanced training.** Each changeset is weighted 1/√(changesets by its author). This
  added +0.05 to the deep model's one-per-author score.
- **Stacking without leakage.** The deep model is trained on 5 author folds. Each training
  changeset gets its score from a fold model that never saw it. Validation and test rows get
  the mean of the fold models.

### What did *not* help (measured, kept as negative results)

| Idea | Result |
|---|---|
| Edit geometry (shape and area change) | ≈ +0.001 in every ablation (within seed noise) |
| Where the author usually maps (distance to past edits, from the full planet changeset history) | −0.0005 [−0.006, +0.004] |
| What the edited object was (Wikidata link, name, historic, …) | within ±0.007 on every task: real signal, but redundant with the edit tokens |
| Tuning the trees | +0.0007 |
| Self-supervised pretraining of the edit encoder | ±0.003 |

### Limits, measured

- **Revert labels are noisy.** An [error analysis](experiments/rounds/r4_2026-10-13_altair_h100_seeds/analysis/error_analysis.md)
  looked at the 60 most confident mistakes:
  - most missed "damaging" changesets are good-faith work reverted for other reasons:
    mapathon imports, bulk buildings, edit wars, technical tag disputes;
  - most false alarms are newcomers adding real parks and footpaths. At least two look like
    vandalism that was never reverted.
- **Over time.** Trained on 2014–19 and tested on 2021–26, the tree model holds (0.905). The
  deepest neural variant drops more (0.874, one seed), so the combined models are the robust
  choice.
- **Prevalence.** Real vandalism is well under 1% of edits. Every score above is on balanced
  data; the reports also give precision at realistic prevalence (1% and 0.1%).

---

## Data

**46,578 changesets** from four corpora:

| Corpus | Period | Label | Rows |
|---|---|---|---|
| A | 2014–2019 | Ovid's benchmark: later reverted | 18,037 |
| R | 2021-09 → 2026-06 | reverted by another account within 90 days, mined from the planet changeset dump | 16,718 |
| M | 2023 → 2026 | OSMCha human review (harmful vs flagged-and-cleared) | 7,892 |
| N | 2023 → 2026 | random changesets, presumed clean | 3,931 |

Each changeset carries:
- its edits, with the previous version recovered for 99%+ of modified objects;
- the author's history from *before* that changeset only.

User ids are pseudonymised (salted hash) in every released file, and usernames are removed.

- **Download:** [GitHub releases](../../releases): the feature store, frozen splits and
  dataset card.
- **Sources and coverage:** [docs/data/DATA_SOURCES.md](docs/data/DATA_SOURCES.md) and the
  [dataset card](docs/data/DATASET_CARD.md).
- **Licence:** the data is a derivative of OpenStreetMap: **ODbL-1.0, © OpenStreetMap
  contributors**.

---

## Quick start

There are three entry points, depending on what you want to do.

### 1. Run the tests (data and feature code)

Python 3.11 or 3.12. TensorFlow has no wheels for Python 3.13+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                    # add -m "not network" to stay offline
```

### 2. Train on the released data (no OSM fetching needed)

From a clone of this repository:

```bash
gh release download v1.1.0 -p mapguard-dataset-v1.1.0.zip   # pseudonymised feature store, 80 MB
unzip mapguard-dataset-v1.1.0.zip -d release
bash train/setup_env.sh                                     # TensorFlow for this machine (add --cpu to skip the Apple GPU)
source .venv-train/bin/activate
python train/run_pipeline.py --feature-store release/feature_store --task ovid --quick   # smoke test, minutes
python train/run_pipeline.py --feature-store release/feature_store --task ovid           # the real run
```

**Text embeddings.** Stage 08 needs `pip install sentence-transformers`, which is not in the
requirements files. Without it the stage fails softly and the models train without
embeddings. The [`v1.2.0`](../../releases/tag/v1.2.0) release adds optional side tables
that reproduce two negative results.

**Options:**
- `--task`: one of `ovid`, `ovid_modern`, `revert`, `revert_vandalism`, `T1`, `T2`.
- `--seed N`: another model seed. The test split stays frozen.
- `--stages 00,08,01,02,07,09`: run a subset. Stages always run in pipeline order.

`--quick` numbers are a smoke test: they are tagged `quick` and must never be reported.

**Stages, in the order they run:**

| Stage | What it does | Needs |
|---|---|---|
| 00 | wiring check: the model builds, masks padding and trains on this machine (stops the run if it fails) | — |
| 08 | pretrained multilingual sentence embeddings, written next to the feature store | — |
| 01 | baselines: metadata tripwire, logistic regression, XGBoost, hybrid trees | — |
| 02 | the deep model | — |
| 07 | ensemble: blend of 02 and the 01 hybrid | 01, 02 |
| 09 | stacking: trains its own 5 author-fold deep models and feeds their OOF score into the trees | — |
| 04 | self-supervised pretraining of the edit encoder | — |
| 03 | ablation grid (15 rows in [`train/03_stage2_ablations/rows/`](train/03_stage2_ablations/rows/)) | 04 for the pretrained row |
| 05 | label efficiency: pretrained vs scratch across label budgets | 04 |
| 06 | temporal evaluation: train on the past, test on the future | 02 |

Without `--stages`, all ten stages run, including the slow research stages 04, 03 and 05. The
final round ran 08 once, then `01,02,07,09` for every task and seed, then 06 and a four-row
slice of 03. The exact plan is
[`altair/plans/r4_seeds.txt`](altair/plans/r4_seeds.txt).

Each stage writes a self-describing run directory under `experiments/runs/<stage>/`
containing `config.json`, `provenance.json`, `results.json` and predictions. Later stages find
earlier runs through `experiments/runs/index.jsonl`.

### 3. Rebuild the data from OpenStreetMap

This takes hours of polite API traffic. Follow [docs/running/RUNBOOK.md](docs/running/RUNBOOK.md):
mine → fetch → `python -m mapguard.features.build_store` → bundle. Copy `.env.example` to
`.env` first; OSMCha needs a free token.

**Compute.** Training rounds 2–4 ran on an NVIDIA H100 GPU (a 40 GB slice) on the Altair HPC
cluster ([altair/README.md](altair/README.md)); round 1 ran on an Apple M4 CPU. A full round
(all six tasks × 3 seeds, stacking and temporal evaluation) takes about 3.7 hours there. Every
stage also runs on a CPU, more slowly.

---

## Repository map

```
src/mapguard/         the library
  data/               one client per source (OSM API, Overpass, OSMCha, planet dump) + PoliteSession
  features/           edit tokens, geometry, point-in-time user history, build_store (the feature store)
  training/           dataset.py: parquet tables → padded model arrays
  models/stage2.py    the deep model
  evaluation/         splits (random, user-disjoint, temporal) and metrics (PR-AUC, prevalence-weighted)
train/                one folder per stage + run_pipeline.py; train/common/ is the shared training core
scripts/              data harvesting, release packaging, result summaries, data-gate probes
data/                 the Ovid label file (the one tracked dataset) and small test fixtures
altair/               job script, bundle builder and step plans for the GPU training rounds
cluster/              bundle.py + generated notebook wrappers for running stages on a notebook cluster
experiments/rounds/   every training round: raw outputs, analysis, a README with findings
experiments/splits/   the frozen train/validation/test splits
results/              data-gate measurements, early baselines, probes
docs/                 architecture, model, data, running and planning documents
tests/                test suite
```

**Where to start reading:**
- **The model:** [`src/mapguard/models/stage2.py`](src/mapguard/models/stage2.py), then
  [`docs/model/MODEL_V3.md`](docs/model/MODEL_V3.md).
- **The features:** [`src/mapguard/features/build_store.py`](src/mapguard/features/build_store.py)
  and [`edit_tokens.py`](src/mapguard/features/edit_tokens.py).
- **Training:** [`train/README.md`](train/README.md), then
  [`train/common/fit.py`](train/common/fit.py), the one training loop every deep stage shares.
- **The results:** the round READMEs in [`experiments/rounds/`](experiments/rounds/). Read them
  in order (r1 → r4) to see how the model got here.
- **The rules:** [CONTRIBUTING.md](CONTRIBUTING.md) lists the rules that keep results valid:
  point-in-time features, user-disjoint splits, and polite OSM traffic.

The full documentation index is [docs/README.md](docs/README.md).

---

## Licence

Code: [MIT](LICENSE). Data derived from OpenStreetMap: ODbL-1.0, © OpenStreetMap contributors.
