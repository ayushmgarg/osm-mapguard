# MapGuard — four-week build checklist

**2026-09-04 → 2026-10-02.** Rewritten from the 15-week version when the timeline changed.
Scope cuts and their justification are in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §13.

**Conventions**

- `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped, with a reason
- **A/C** = acceptance criterion. Not done until the A/C is demonstrably met.
- Every gate writes JSON to `results/`. No file means the gate did not run.
- Owner is the workstream: WS1 data · WS2 evaluation · WS3 Stage 2 · WS4 Stage 1 and figures.

---

## Already done (2026-09-04)

- [x] Plan reviewed, two false premises corrected, `docs/planning/PLAN_REVIEW.md`
- [x] Repo restored onto a clean `main`; `.claude/` gitignored; conventions in `CONTRIBUTING.md`
- [x] `MAPGUARD_DATA` = `C:\osm-data`, outside OneDrive, wired through every module
- [x] pytest harness, **38 tests**, no network
- [x] OSM account, OSMCha token verified against the live API, stored in a gitignored `.env`
- [x] **Gate 1a PASS** — 93/100 geometry yield, 2,109/2,109 ways with coordinates
- [x] Data modules: `http` (global token bucket, 509/429 backoff), `osm_api`, `overpass`
      (adiff), `replication`, `osmcha`, `users`, `store`
- [x] `features/geometry.py` — displacement, length, area, Hausdorff, tag deltas, summary
- [x] `osmchange.parse()` streaming leak fixed; parser tested against six document shapes
- [x] `data/mock/` — 200 synthetic changesets in the real schema, so WS2–4 are not blocked
- [x] Corpus D live harvest working — 1,426 changesets from 60 minutes of replication
- [x] OSMCha labels harvesting — 3,200+ harmful for 2026, plus reviewed negatives

---

## Week 1 · Sep 4 – Sep 10 · data pipeline and the remaining gates

**Goal: Gates 1b/1c/1d green, Corpus A collected and featurised, schema frozen.**
WS2–4 work against `data/mock/` all week. Do not wait for WS1.

### W1.1 Finish the gates

- [ ] **W1.1.1** Gate 1b — Overpass capacity. Run alone, 30 minutes. `python scripts/gate1b_overpass_capacity.py --minutes 30`
      **A/C:** a written decision — full 18,276, a mirror, or a subset **sampled by changeset id, never by size or user**. → `results/gate1b_overpass_capacity.json`
- [ ] **W1.1.2** Gate 1c — B0 leakage tripwire, once the metadata fetch finishes. `python scripts/gate1c_b0_tripwire.py`
      **A/C: F1 < 0.70** on the user-disjoint split. **If it reaches ~0.8, stop the project and find the leak.** → `results/gate1c_b0_tripwire.json`
- [ ] **W1.1.3** Gate 1d — framework spike. Toy model: 5 heterogeneous inputs, ragged edit set, masked multi-head attention, one epoch. Verify padded rows do not change the output.
      **A/C:** trains, masking verified. **Decision by Sep 8.** Pass → Keras 3. Fail → PyTorch. → `results/gate1d_framework_spike.json`
- [~] **W1.1.4** Gate 1e — Stage-1 throughput on Kaggle. **Deferred to week 4**, and only if Stage 1 survives.

### W1.2 Corpus A collection

- [~] **W1.2.1** Metadata for all 18,276 — running, ~3/s. **A/C:** ≥98% HTTP 200, failures recorded with status.
- [ ] **W1.2.2** Contents (osmChange) for all 18,276. `python -m mapguard.data.fetch_corpus_a` — update its defaults to the measured 1.2 s latency first.
- [ ] **W1.2.3** Verify resume: kill mid-run, restart, confirm no refetch and no gap.
- [ ] **W1.2.4** Geometry enrichment via adiff, at whatever scope Gate 1b permits. `python scripts/fetch_corpus_a_geometry.py`
      **A/C:** geometry coverage reported per object type. → `results/corpus_a_geometry_coverage.json`
- [ ] **W1.2.5** Multi-fetch fallback wired for changesets adiff cannot serve (no bbox, global bbox).
- [ ] **W1.2.6** Presumed-negative pool for R2 — 50,000 unlabelled Ovid-era ids, ~4.2 h. **Stretch goal**; R2 works without it via prevalence correction.

### W1.3 User histories — point-in-time

- [ ] **W1.3.1** Per-uid ordered changeset table. Source: OSM API user endpoint plus the changesets already fetched.
- [ ] **W1.3.2** **Enforce the point-in-time rule** — only changesets with `created_at < target.created_at`.
- [ ] **W1.3.3** Write `test_user_features_are_point_in_time`. Construct a user with a known later event; assert it never appears in an earlier target's features. **Not optional.**
- [ ] **W1.3.4** Cap the sequence at 200 changesets; record the truncation rate.

### W1.4 Feature store

- [ ] **W1.4.1** Extend `build_corpus_a.py` to emit `changesets.parquet`, `edits.parquet`, `geometry.parquet`.
- [ ] **W1.4.2** **Exact** whole-changeset summaries over all N edits — op mix, type mix, tag-key histogram, geometry moments. These never truncate. (Plan §5.4 Rule 1, gap G4.)
- [ ] **W1.4.3** **Stratified edit sample**, K = 256: top displacement, top area change, all text-bearing edits, then random fill. Store `K/N` as a feature.
- [ ] **W1.4.4** Text features — old and new values of `name`, `name:*`, `description`, `note`, `operator` as character sequences for B2.
- [ ] **W1.4.5** Tabular features — a documented superset of Ovid's, so the reproduction is honest.
- [ ] **W1.4.6** **Freeze the schema Friday Sep 10.** After that it is a versioned change, not an edit.
- [ ] **W1.4.7** Upload featurised Parquet to Drive once.

### W1.5 Accounts and admin

- [ ] **W1.5.1** Kaggle accounts for all four (30 GPU-h/week each — that is the compute budget).
- [ ] **W1.5.2** Shared Drive folder `mapguard/`.
- [ ] **W1.5.3** W&B project, shared.
- [ ] **W1.5.4** Workstream owners named in `README.md`.
- [ ] **W1.5.5** Delete the stale branches — `master`, `setup`, both `claude/*` — and remove the worktree.

---

## Week 2 · Sep 11 – Sep 17 · baselines, evaluation harness, first two branches

**Goal: a defensible number on the board, and the harness every later result flows through.**

### W2.1 Splits and metrics · WS2

- [ ] **W2.1.1** R1a split — random 70/10/20, seeded. Ovid's protocol. **Label it user-leaky in every output.**
- [ ] **W2.1.2** R1b split — user-disjoint, no uid crosses splits. Assert it in a test.
- [ ] **W2.1.3** Persist split assignments to disk so every model sees identical splits.
- [ ] **W2.1.4** Metrics: F1, precision, recall, ROC-AUC, **PR-AUC**, precision@k.
- [ ] **W2.1.5** **Prevalence-corrected PR curves** — reweight negatives by `w = ((1-p)/p) * (n_pos/n_neg)`. Unit-test against a synthetic set with known prevalence.
- [ ] **W2.1.6** Bootstrap confidence intervals on every headline number. A point estimate without a CI does not go in the paper.
- [ ] **W2.1.7** Calibration: reliability diagram, Brier score, isotonic recalibration.

### W2.2 Baselines · WS2

- [ ] **W2.2.1** B0 tripwire re-run on the final feature store.
- [ ] **W2.2.2** Logistic regression on the full tabular set.
- [ ] **W2.2.3** XGBoost, lightly tuned. (Random forest only if time — GBDT is the stronger reference.)
- [ ] **W2.2.4** **Ovid reproduction** — Ovid's feature set only, under R1a.
      **A/C:** within a few F1 points of 81.66. If far off, say so and diagnose **before** claiming any improvement over it.
- [ ] **W2.2.5** All baselines write to `results/baselines.json` with split, seed, metric, CI, git SHA.

### W2.3 First branches · WS3

- [ ] **W2.3.1** **B5 tabular MLP** end to end — the parity baseline and the simplest full path.
- [ ] **W2.3.2** **B2 char-CNN** over tag text, multi-width parallel kernels (the Inception idea).
- [ ] **W2.3.3** Training loop: early stopping on **validation PR-AUC**, never accuracy.
- [ ] **W2.3.4** **Focal loss** implemented, compared against BCE.
- [ ] **W2.3.5** Checkpoint to Drive every epoch; verify a killed run resumes.
- [ ] **W2.3.6** W&B logging: config, metrics, split, seed, git SHA on every run.

### W2.4 Corpus D · WS1

- [ ] **W2.4.1** Harvest a defined 2026 window. **State the exact dates — they go in the paper.**
- [ ] **W2.4.2** Join OSMCha labels to the harvested changesets; report prevalence.
- [ ] **W2.4.3** Fetch contents and adiff geometry for the labelled subset.
- [ ] **W2.4.4** Featurise into the frozen schema.
- [ ] **W2.4.5** Verify the temporal boundary: **no Corpus D changeset may predate the Corpus A training cut-off.**

---

## Week 3 · Sep 18 – Sep 24 · the full model, R1 and R2

**Goal: the complete architecture trains, and the two headline regimes have numbers.**
**Paper drafting starts this week, in parallel. Not in week 4.**

### W3.1 Remaining branches · WS3

- [ ] **W3.1.1** **B1 geometry CNN** — 2-channel before/after raster per edit, plus scalar geometry features.
- [ ] **W3.1.2** Document the rasterisation: extent, resolution, normalisation, how a single node renders.
- [ ] **W3.1.3** **B3 edit-sequence BiLSTM** over the sampled edits in commit order.
- [ ] **W3.1.4** **B4 user-history GRU** over the point-in-time sequence.
- [ ] **W3.1.5** **Multi-head attention pooling** over the K sampled edits, masking verified.
- [ ] **W3.1.6** Fusion head: concatenate → dense → dropout → LayerNorm → sigmoid.
- [ ] **W3.1.7** Three outputs wired: `p(damaging)`, per-edit attention weights, branch attribution.

### W3.2 Course-syllabus deliverables · WS3 + WS4

These are marked work, not optional colour. Each is a report section.

- [ ] **W3.2.1** **U5** — train a vanilla RNN on 200-step user histories deliberately, plot gradient-norm collapse, then justify LSTM/GRU.
- [ ] **W3.2.2** **U3** — optimiser comparison (SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam) on the real task, reusing the lab expt 6 protocol.
- [ ] **W3.2.3** **U2** — activation ablation (ReLU / LeakyReLU / GELU).
- [ ] **W3.2.4** **U4** — parameter count per branch, tabulated.

### W3.3 Results · WS2

- [ ] **W3.3.1** **R1a** — balanced, Ovid protocol, reported against 81.66 and flagged as leaky.
- [ ] **W3.3.2** **R1b** — user-disjoint. **Report the R1a–R1b gap explicitly; it is a finding.**
- [ ] **W3.3.3** **R2** — prevalence-corrected PR curves at 1% and 0.1%. PR-AUC and precision@k, no F1 headline.
- [ ] **W3.3.4** Error analysis: 20 false positives and 20 false negatives read by hand, grouped into causes.

### W3.4 Paper drafting starts · all four

- [ ] **W3.4.1** Overleaf project, IEEE template, all four as authors.
- [ ] **W3.4.2** Related work and the gap table — from plan §2 and §3.
- [ ] **W3.4.3** Method section — two stages, five branches, hierarchical encoding, point-in-time rule.
- [ ] **W3.4.4** Data section — three corpora, and the adiff and replication routes described so others can reproduce them.

---

## Week 4 · Sep 25 – Oct 2 · temporal test, ablations, demo, submission

### W4.1 R3b — the temporal contribution · WS2

- [ ] **W4.1.1** Train on Corpus A, test on Corpus D (live 2026, OSMCha-labelled).
- [ ] **W4.1.2** Report the drop, **and separate two effects**: five years of drift, and a different label source (human review, not revert history). Do not conflate them.
- [ ] **W4.1.3** Calibration on Corpus D — a model calibrated on 2019 will not be calibrated on 2026.

### W4.2 Stage 1 — conditional · WS4

**Run only if the week-3 model is stable. If it is not, drop this section and say so in the paper.**

- [ ] **W4.2.1** Gate 1e — measured edits/sec on one Kaggle P100. Derive the budget; do not guess.
- [ ] **W4.2.2** Edit encoder: tag embedding + char-CNN + geometry conv + dense → `z ∈ R^128`.
- [ ] **W4.2.3** **Pretext C only** — corruption contrast. Geometry displacement, name scrambling, ×100 area, chunk deletion, tag-value swap. Document why each mimics a real vandalism mode.
- [ ] **W4.2.4** Corpus C from the bucket window, **excluding every Corpus D changeset id and uid**. Write `test_pretrain_eval_disjoint`.
- [ ] **W4.2.5** **Zero-shot transfer check** — score the frozen corruption head on the real labelled set before fine-tuning. Report the AUC. Near 0.5 means the corruptions are the wrong distribution; report that too.
- [ ] **W4.2.6** **R4** — label efficiency at 10/25/50/100%, pretrained vs scratch, ≥2 seeds.

### W4.3 Ablations — six rows, two seeds · WS3

- [ ] **W4.3.1** B0 tripwire · **W4.3.2** Ovid-repro · **W4.3.3** +geometry · **W4.3.4** +text · **W4.3.5** +sequences · **W4.3.6** Full
- [ ] **W4.3.7** Full-pretrained, **only if W4.2 ran**.
- [ ] **W4.3.8** 2 seeds per config, mean ± std. A single-seed ablation row is not evidence.
- [ ] **W4.3.9** Export to `results/ablations.json`; generate the paper table from it programmatically.
- [ ] **W4.3.10** Per-vandalism-type breakdown — geometric / semantic / destructive. **First thing to cut if the week slips.**

### W4.4 Figures and walkthrough · WS4

- [ ] **W4.4.1** Architecture diagram, exported clean from the mermaid source.
- [ ] **W4.4.2** **R4 label-efficiency curve** if Stage 1 ran; otherwise the ablation bar chart is the headline figure.
- [ ] **W4.4.3** PR curves at balanced and realistic prevalence on one axis.
- [ ] **W4.4.4** Gradient-norm collapse plot (U5).
- [ ] **W4.4.5** Attention weights on a worked vandalism example — which edit, which signal.
- [ ] **W4.4.6** Scripted walkthrough: pull a live changeset from the minute stream, score it, show the flagged edit. **A recording, not a hosted service.**
- [ ] **W4.4.7** Every figure regenerable by one script from `results/`. No hand-edited images.

### W4.5 Paper and submission · all four

- [ ] **W4.5.1** Results section — tables generated from `results/`, never retyped.
- [ ] **W4.5.2** Limitations — label noise, presumed-negative contamination, relations out of scope, geometry subsetting if Gate 1b forced it, Stage 1 if it was dropped.
- [ ] **W4.5.3** Ethics and licensing (plan §16) — ODbL attribution, salted-hash uids, no account-level claims.
- [ ] **W4.5.4** Reproducibility appendix: seeds, split files, git SHA, `requirements.lock`.
- [ ] **W4.5.5** Plagiarism check.
- [ ] **W4.5.6** Internal review — each member reviews a section they did not write.
- [ ] **W4.5.7** Syllabus coverage audit against plan §6 — every unit U2–U5 used, artefact named.
- [ ] **W4.5.8** Final repo tidy: README current, no data committed, `results/` complete, no stale docstrings.
- [ ] **W4.5.9** Submit.

---

## Dropped, on purpose

Recorded so nobody re-adds them by accident, and so the paper can say what was not done.

- [-] **Corpus B as a labelled corpus** — Corpus D supersedes it with better labels on newer data. Costs regime R3a. The bucket survives as the Corpus C pretraining source.
- [-] **Pretext A (masked tag modelling) and Pretext B (geometry reconstruction)** — one pretext task fits the timeline; these become future work. Costs the λ ablation.
- [-] **The ~40-run ablation grid** — six rows, two seeds.
- [-] **Interactive Folium demo** — a scripted walkthrough instead.
- [-] **Random forest baseline** — XGBoost is the stronger reference and one GBDT is enough.

### If it compresses to three weeks

Drop in this order and stop when it fits:

1. **Stage 1 entirely** (W4.2) — R4 goes; the paper becomes a supervised-fusion contribution
2. **B3 edit-sequence** (W3.1.3) — B4 already carries the sequence argument for U5
3. **R2 sampled 1:100** (W1.2.6) — keep the weight-corrected version only
4. **Per-type breakdown** (W4.3.10)

**G1, G2, G3, G5 and G6 survive even that.** Only G4 and the self-supervision claim are at
risk, and both were designed to be ablatable.

---

## Standing rules — check at every merge

- [ ] No data in the repository. Bulk data lives in `C:\osm-data\`.
- [ ] No secrets. The OSMCha token stays in the gitignored `.env`.
- [ ] Metrics go to `results/*.json`, not stdout, not a screenshot.
- [ ] Every run records config, split, seed and git SHA.
- [ ] Notebooks stay thin: `git pull`, install, import, run.
- [ ] OSM requests: one shared token bucket, honest User-Agent, back off on 509. Never send Overpass a planet-sized bbox.
- [ ] No `Co-Authored-By` trailers, no generated-by footers, no tool-named branches.
- [ ] **A red gate stops the week.** Re-plan; do not build past it.
