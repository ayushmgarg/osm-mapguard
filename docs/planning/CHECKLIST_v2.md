# MapGuard — v2 build checklist

**Purpose:** execution checklist after reviewing `PROJECT_PLAN.md`, `CHECKLIST.md`,
`docs/planning/PLAN_REVIEW.md`, the current source tree, and the tracked gate artifacts.

**Scope:** preserve the research contribution, make the evidence trail truthful, and
ship a visual, inspectable demo alongside the paper. This file is additive; the v1
checklist remains unchanged for historical reference.

## Current baseline

- [x] Plan review recorded the original data and evaluation flaws and their fixes.
- [x] Gate 1a passed: 93/100 sampled changesets yielded usable geometry.
- [x] Gate 1b measured the real constraint: public Overpass projects to 96.2 hours;
  choose a mirror or an id-sampled subset.
- [x] Gate 1d passed: Keras 3 masked multi-input spike trained and padding perturbation
  was zero.
- [x] Corpus D replication harvest, OSMCha labels, mock data, parser, geometry helpers,
  splits, and padded dataset assembly exist.
- [x] Gate 1c has a tracked result and an explicit leakage decision.
- [x] Corpus A has been fully fetched, geometry-enriched, featurised, and schema-frozen.
  *Fetched: yes, 18,276/18,276 metadata and contents. Geometry: 1,055 adiff + 65 skipped of the 5,972-changeset subset. Featurised: no. Schema frozen: no.*
  *Done 2026-09-19: fetched, featurised (schema v3), schema manifest written. Geometry: previous-version route for every modified object, plus adiff on 1,006.*
- [~] A real trainable model, checkpoint, inference path, and experiment results exist.
  *Training code exists in `train/00`–`06`; none of the TensorFlow code has run yet.*
  *Code is verified end to end (Keras 2 + 3, real data, quick mode). Real training runs next, on the cluster or M4.*
  *Trained 2026-09-21 (experiments/rounds/r1_2026-09-21_m4/). Checkpoints and results exist; a single-changeset inference path for the demo is still to build.*
- [ ] A visual demo exists and scores the same input through the same inference path as
  the research results.

## Audit 2026-09-17 — what blocks the first real training run

Legend used from this audit on: `[x]` done and verified on real data · `[~]` code exists
but has not been run on real data or is partial · `[ ]` nothing yet.

How it was checked: row counts read directly from every store in `C:\osm-data\raw`. The test suite
passed (103). Every training script byte-compiled. The feature builder was run on 400 real
changesets. `01_baselines` ran end to end on that probe, and `--dry-run` passed for stages
02, 03 and 04. All of this ran in an isolated copy, so no split or run was written to the repo.

### A. Code that makes the data trainable — done 2026-09-17, needs data runs at home
- [x] `build_store.py` rewritten: reads the real table names, writes `corpus`/`quarter`,
  builds A, M and N, and gives a per-corpus, per-label coverage report. It was verified on 400
  real Corpus A changesets, and `tests/test_build_store.py` runs it on stores with the real schemas.
- [x] `comments_count` removed from the features. It is read years after the edit, and
  reviewers comment on the changesets they revert, so it leaked the label.
- [x] Missing data is now explicit (`account_age_known`, `user_history_known`,
  `frac_edits_prev_version`, `geom_frac_adiff`), never a silent zero that reads as
  "new account" or "nothing moved".
- [x] Previous-version geometry route (`scripts/fetch_osm.py versions`): tag diffs for every
  modified or deleted object, node displacement, way node-count change. It needs one API call
  per 725 objects, against ~122 s per changeset for adiff; the real duration is measured on the
  first run. `geom_route` records which route each edit got.
- [x] Point-in-time histories are fetched per target, and a truncated later window is never reused
  (`mapguard.data.corpora.history_for`, tested).
- [x] adiff geometry is fetched in hash order, so any stopping point is an unbiased sample.
  The first 1,055 were fetched in id order and lean towards 2014; disclose that.
- [x] Default task in every `train/*/config.json` is now `ovid`. The temporal cutoff for ovid is
  2019-01-01, so 2014–2018 is train and 2019 is test.
- [x] Corpus A fetched and built (2026-09-18). 18,050 changesets, 100% point-in-time history,
  and 456,422 of 458,126 previous versions; 1,704 are moderator-hidden or missing and are stored
  as unresolved. The report is `results/feature_store_build_report.json`.
- [x] Two more future leaks were found in the built data and removed:
  - Account age: 97 deleted accounts own 599 changesets, 83% of them damaging, so it is no
    longer a model input. B0 fell from 0.711 to 0.657 once it went.
  - Empty changesets: 226 rows, 224 of them clean, a free "clean" signal. They are now dropped.
- [x] Baselines on the real store (R1b): XGBoost PR-AUC 0.852, LogReg 0.804, B0 0.657. The
  frozen split `experiments/splits/ovid_R1b_seed20260904.json` is committed.
- [x] Schema manifest: `scripts/schema_manifest.py` writes `results/feature_schema.json`
  (dtype, unit, meaning and null rate for every column), generated from the built store.

### B. Remote execution — done 2026-09-17, needs a first run on a TensorFlow machine
- [x] `train/run_pipeline.py`: one command for every stage on any machine. It wires the Stage 1
  encoder into 03/05 and the R1b run into 06, and finds runs from earlier jobs. `--quick` runs
  are tagged and never reused by real runs.
- [x] `cluster/` regenerated as one notebook per stage plus `99_full_pipeline`, each calling
  run_pipeline. The stale two-output notebooks were deleted. `cluster/bundle.py` builds one
  zip (code, feature store, splits, manifest). This was verified by unzipping in a clean
  directory and running stage 01 from it.
- [x] `train/setup_env.sh` sets up TensorFlow on the M4 or Linux and runs the wiring check.
  `requirements-train.txt` added.
- [x] Static review of the TensorFlow paths turned up three fixes:
  - Keras 2 only restores the best weights when early stopping fires, so the checkpoint is
    now reloaded explicitly.
  - Fully masked user histories knocked whole batches off cuDNN, so a null step is inserted.
  - Stage 1 corruptions could write geometry onto edits that had none, a shortcut the
    encoder would learn. They now respect `geom_route`.
- [x] `00_wiring_check` PASS on **Keras 3.15** (TF 2.18, Python 3.12) and on **Keras 2.13.1**
  (TF 2.13, Python 3.10, the cluster image's versions), both CPU, 2026-09-18. It found and fixed
  two bugs no test could see:
  - an argument named `mask` broke the Keras 3 build;
  - `tf.keras` 2.13 has no `__version__`.
- [x] The full `--quick` pipeline (00→01→02→Ovid-repro→04→zero-shot→03→05→06) completes on
  the real Corpus A store under both Keras versions. Masking delta was 0.0 before and after
  training, and there were no NaNs.
- [x] Real (non-quick) runs on the cluster or M4, in RUNBOOK order.
  *Done on an M4, CPU only (Metal did not attach). ovid all stages; ovid_modern/T1/T2 stages 01,02,06.*

### C. Modern tasks T1/T2 — code done, data at home
- [x] Corpus M selector: 4k harmful + 4k reviewed-clean, water-filled across quarters.
- [x] Corpus N sampler: random ids inside each quarter's id range, date-matched to M's positives.
  Empty and open changesets are rejected in hash order.
- [x] OSMCha holes refilled with 1-day windows, targeting the missing class per month (osmcha.org
  rate-limits and times out on large queries). Totals are now 91,590 cleared and 41,933 harmful.
  Corpus M has 4,000 per class, balanced over 15 quarters. Corpus N has 4,000.
- [x] 2026 cleared labels are scarce. The T1/T2 temporal cutoff is set to 2025-07-01 so the test
  side holds both classes; report prevalence-corrected metrics.
  *Resolved: holes refilled. Every temporal test side holds both classes (T1/T2 test 50%).*
- [x] N is "not flagged in our harvest", not "verified clean". Disclose it as presumed-negative.
  *Disclosed as presumed-negative in the dataset card and DATA_SOURCES.md.*

### C2. Five years of recent labels — corpus R (added 2026-09-18)
- [x] Measured: OSMCha has no reviewed changesets before 2023, so human labels cannot cover
  4–5 years. Revert-derived labels can, with the same definition as the Ovid benchmark
  (`docs/data/DATA_SOURCES.md`).
- [x] `scripts/mine_changesets.py` reads the 2021-09 onwards byte range of the planet changeset
  dump (multistream bzip2, parallel ranges, nothing stored but the results). A test that splits the
  file 2, 3 and 7 ways counts every changeset exactly once, and it caught a boundary bug before
  the full run.
- [x] Corpus R label rules: reverted by another account within 90 days, whole changeset only,
  90-day censoring at the dump date, at most 5 per author per class, and negatives matched
  per quarter.
- [x] New tasks: `revert` (R only) and `ovid_modern` (train on Ovid 2014–19, test on R 2021–26,
  same label rule, so a drop means time rather than a change of labeller).
- [x] Dump mined: all 246 chunks with 0 failures. That is 78.4 M changesets and 113,559 reverts
  (67k osm-revert, 24k JOSM reverter, 22k comment, 1k simple_revert).
- [x] Corpus R selected: 5,722 positives and 5,722 negatives, matched per quarter over 20
  quarters; 610 self-reverts excluded.
- [x] Corpus R fully fetched: contents, 1.51 M previous versions, accounts and histories.
  The vandalism supplement adds Ovid's rule, reverts that mention vandalism.
- [x] Final store built from A, R, M and N: 46,578 changesets with 100% history. Splits are
  frozen for 6 tasks × (R1b, temporal).
- [x] Corpus A carries adiff geometry and R does not, so `ovid_modern` drops the adiff-only
  columns automatically (`model.exclude_columns=adiff_only`, tested).
- [x] Ovid paper read (arXiv 2203.11087). Its split is **user-disjoint**, so R1b is Ovid's
  protocol; the plan wrongly said R1a, and that is corrected. Its labels are reverts whose comment
  mentions vandalism, so corpus R now has that flag and the task `revert_vandalism`.
- [x] Ovid's remaining features were added: time since the previous version, previous-author
  change, distinct previous authors, imagery use and user active weeks. `ovid_repro.json` is the
  Ovid-equivalent model on R1b. `ovid_table` reports P/R/F1/accuracy beside Ovid's 80.35 / 83.02 /
  81.66 / 81.37.
- [x] Reference numbers on the final store: XGBoost on R1b gives F1 77.7 and accuracy 76.2
  against Ovid's 81.66 / 81.37, so the data reproduces Ovid's range. The neural model has to
  beat 81.66.

### D. Claims that have to match the evidence
- [ ] "Self-supervised": there is no unlabelled Corpus C. Stage 1 pretrains on training-fold
  edits without their labels. The only evidence for the claim is the label-efficiency
  curve, so word it that way.
- [~] "Geometry-aware": geometry covers ~6% of Corpus A today. B1 must be evaluated on
  the geometry-covered subset, or its ablation delta will read as zero.
  *Improved: the previous-version route gives node displacement, tag diffs and node-count change for every modified object in all corpora. Full way shape (adiff) is on 1,006 A changesets; claim B1 way-shape results only on `has_adiff` rows.*
- [x] Ovid comparison: no reproduction of the published protocol exists yet. Without it,
  the paper cannot say "beats Ovid".
  *Protocol reproduced: R1b is user-disjoint like Ovid, `ovid_repro.json` is the Ovid-equivalent model, and `ovid_table` gives P/R/F1/Acc. XGBoost reference: F1 77.7 against Ovid's 81.66.*

### E. Demo and paper — nothing started
- [ ] Phases 3–5 are all open. With 15 days to 2026-10-02, scope the demo to replay mode on
  committed fixtures through the same inference function. Live mode is a stretch goal.

## V2 operating rules

- [x] Treat every gate as a stop/go decision. A red gate changes scope before more
  downstream work is built.
- [ ] Keep research truth separate from product polish: no UI claim is accepted unless
  it is backed by a saved score, model version, input changeset id, and feature schema.
- [ ] Do not claim "real-time" unless the demo shows its polling interval, fetch time,
  inference time, and stale/error state.
- [~] Do not expose raw user names or uid values in screenshots, logs, or hosted output.
  *Release and bundle are pseudonymised (verified: 0 raw uids, 0 usernames). The demo is still to come.*
- [ ] The demo must work with a recorded fixture when external OSM services, Overpass,
  Google Maps, or credentials are unavailable.
- [x] Every result records config, split, seed, model/checkpoint id, git SHA, and data
  window.
  *`train/common/run.py` records config, seed, split file, git SHA, devices and a data summary. It does not record the data date window explicitly.*
  *run.py provenance: config, seed, frozen split file and fingerprint, git SHA (or bundle SHA), data fingerprints and data summary.*

## Phase 0 — reconcile the repo before modelling

- [x] Update the status table in `README.md` from actual `results/*.json` artifacts;
  distinguish verified, running, stale, and not-started work.
- [x] Decide the Gate 1b route in writing: mirror, id-sampled Corpus A, or another
  reproducible route. Record the sampling rule and its coverage consequences.
- [x] Run and save Gate 1c. If the B0 tripwire is high, stop and diagnose leakage before
  training anything.
- [x] Confirm the Python environment is reproducible. The checked-in `.venv311` points
  to a missing Python 3.11 executable; document the supported interpreter and a clean
  setup path.
- [x] Run the no-network test suite in that supported environment and save the command,
  interpreter version, and result.
- [x] Remove or explicitly track stale assumptions in docstrings and docs, especially
  the node-only Corpus A geometry description.
- [x] Reconcile schema names across `build_corpus_a.py`, `build_store.py`,
  `training/dataset.py`, mock Parquet, and the plan. One field name must mean one thing.
  *`train/common/data_contract.py` plus `results/feature_schema.json` define every column. The old `build_corpus_a.py` is superseded and not used by train/.*
- [x] Decide the canonical artifact layout: raw, interim, features, checkpoints,
  fixtures, results. Ensure no large data or secrets enter git.

## Phase 1 — data and feature contract

- [x] Fetch Corpus A metadata and contents with resumability, status accounting, and
  retry/backoff evidence.
  *18,276/18,276 metadata and 18,276/18,276 contents, all status 200.*
- [x] Enrich Corpus A geometry through the selected Gate 1b route and fallback; report
  coverage by object type and by label.
  *1,055 adiff + 65 skipped of 5,972 (the id-hash subset). Averages 122 s per changeset, so the remaining ~4,850 would take ~160 h single-stream. No coverage report yet.*
  *1,512,465 previous versions fetched across all corpora (1.2% moderator-hidden, dropped). adiff on 1,006 A changesets. Coverage per label is in the build report.*
- [x] Build exact all-edit changeset summaries; never erase scale information.
  *Audit 2026-09-17: code and unit tests exist, but it has never produced a feature store from real data. `build_store.py` crashes on the real Corpus A store (queries table `contents`; the table is `changeset`, keyed `id`).*
- [x] Build the K=256 stratified per-edit sample: displacement, area change, text-bearing
  edits, then random fill. Persist `K/N` and truncation status.
  *Same builder - see above.*
- [x] Build point-in-time user history using only changesets strictly earlier than the
  target; add the explicit future-event regression test.
  *Code and test exist. Histories fetched for 20 of 8,768 Corpus A users. `users.sqlite` (account creation dates) was never fetched, so `account_age_days` is null on every row.*
  *100% of 46,578 changesets have a valid point-in-time history.*
- [x] Build text features from the agreed tag keys and document the treatment of casing,
  Unicode, empty values, and personal data.
- [x] Freeze and version the feature schema; write a schema manifest with dtypes, units,
  missing-value policy, and provenance for every field.
  *`train/common/data_contract.py` fixes the required tables and columns and refuses anything else. There is no manifest of dtypes, units or missing-value policy.*
  *Schema v3; `results/feature_schema.json` has 0 undocumented columns.*
- [x] Add a deterministic fixture set containing at least: normal edit, text vandalism,
  geometry displacement, mass change, deleted-account/anonymous case, no-bbox case, and
  a changeset whose geometry requires fallback.
- [x] Verify Corpus C pretraining is disjoint from every Corpus D evaluation changeset
  and uid; save the proof artifact.
  *Corpus C was never fetched. Stage 1 pretrains on training-fold edits minus evaluation authors, and writes `disjointness.json` when it runs.*
  *There is no Corpus C by design. Stage 1 uses training-fold edits only and writes `disjointness.json`; the run aborts on any overlap.*

## Phase 2 — baselines and research model

- [x] Produce R1a random/user-leaky and R1b user-disjoint splits with persisted ids.
- [x] Produce B0, logistic regression, and XGBoost baselines with bootstrap intervals,
  PR-AUC, precision@k, calibration, and prevalence-corrected curves.
  *All 6 tasks: `results/baselines_*_R1b.json`.*
- [x] Reproduce the Ovid feature protocol before claiming improvement; document any
  feature or label mismatch that explains a gap from the published number.
  *Ovid's features added; the differences (no th_e zeroing, no 'valid tags') are documented in `ovid_repro.json`.*
- [x] Implement the smallest end-to-end B5 tabular classifier first and establish a
  working checkpoint/inference contract.
  *`train/03_stage2_ablations/rows/01_base_edits_tabular`. Weights-only checkpoints and a rebuild from `model_config.json`. Wiring check and real-data quick runs pass on Keras 2 and 3.*
- [x] Implement B2 text, B1 geometry, B3 edit sequence, and B4 point-in-time user
  history branches incrementally; test each branch in isolation.
  *B2 now reads the changeset comment plus tag text. All branch sets pass the wiring check.*
- [x] Implement masked attention over sampled edits and verify padded edits cannot
  change the output.
  *Masking delta 0.0 before and after training on real data.*
- [x] Emit the three promised outputs: damage probability, suspicious edit index/weight,
  and branch/signal attribution.
  *Written: `p_damaging`, `edit_attention_weight` and occlusion in `train/common/attribution.py`; all three produced in the quick runs.*
- [x] Train the full supervised model before attempting Stage 1 self-supervision.
  *Script ready: `train/02_stage2_supervised`.*
  *ovid R1b: PR-AUC 0.844, F1 78.1, which ties XGBoost (0.857 / 78.1) and is below Ovid's 81.66.*
- [x] Run the planned ablations and at least two seeds for headline comparisons.
  *Script ready: `train/03_stage2_ablations`, 7 rows x 2 seeds, passes `--dry-run`.*
  *7 rows × 2 seeds. Text is the whole gain (+0.035). Geometry +0.014 and history/sequence +0.016 add nothing on top of text.*
- [x] Run Corpus D temporal evaluation and separate temporal drift from label-source
  drift in both metrics and discussion.
  *Script ready: `train/06_temporal_evaluation`. Temporal splits are frozen for all 6 tasks, and `ovid_modern` separates time from label source (same revert rule on both sides; T1/T2 carry the human-review source).*
  *ovid −0.076, ovid_modern −0.018 (test set is easy, B0 0.839), T1 +0.007, T2 −0.042. Temporal baselines still missing.*
- [x] If Stage 1 survives the compute gate, run corruption contrast, zero-shot transfer,
  and label-efficiency experiments. Otherwise record the cut explicitly and remove
  self-supervised claims from the headline conclusion.
  *Scripts ready: `train/04_stage1_pretrain` (corruptions, pretrain, zero-shot) and `train/05_stage1_label_efficiency`.*
  *Run. Negative result: pretrained minus scratch is within ±0.003 at 10/25/50/100% labels. Zero-shot 0.611. Remove the self-supervised claim from the headline.*

## Phase 3 — visual demo definition

### Product shape

- [ ] Choose a small dashboard surface, preferably a local Streamlit/Gradio or static
  web app with a thin Python inference endpoint; avoid building a second ML pipeline.
- [ ] Support two entry modes:
  - [ ] **Replay mode:** choose a saved fixture or changeset id and reproduce a known
    result deterministically.
  - [ ] **Live mode:** enter a changeset id or poll the OSM changeset replication stream
    for a recent candidate, with visible loading, rate-limit, timeout, and stale states.
- [ ] Make the first screen answer, at a glance: what changed, where, how suspicious it
  is, and why a reviewer should inspect it.

### Map experience

- [ ] Show before/after geometry on an interactive map with a clear legend:
  unchanged, created, modified, deleted, and top-ranked suspicious edit.
- [ ] Add zoom-to-change, fit-to-bounds, object-type filters, and a before/after toggle.
- [ ] On selecting an edit, synchronize the map highlight with the detail panel and
  attention/attribution output.
- [ ] Include a side-by-side provider comparison only if it is legally and technically
  available. Default to an OSM-compatible map layer; make Google Maps an optional
  reference panel requiring a user-supplied key and current attribution/usage review.
- [ ] Never scrape, copy, cache, or merge Google tiles/data into OSM output. The demo
  must remain useful with Google disabled.
- [ ] Display required map/data attribution in the UI and in every exported screenshot.

### Dashboard panels

- [ ] **Decision card:** `p(damaging)`, threshold, confidence/calibration note, model
  version, and timestamp.
- [ ] **Change summary:** changeset id, anonymised author identifier if needed, editor,
  operation mix, object counts, bbox, and fetch status.
- [ ] **Why flagged:** geometry/text/user/scale signal bars tied to actual model outputs,
  not hand-written explanations.
- [ ] **Suspicious edits:** ranked table with object type, action, score/attention weight,
  geometry delta, and a map-selection control.
- [ ] **Text diff:** safe old/new tag display with redaction/truncation rules for hostile
  or sensitive content.
- [ ] **History/evaluation panel:** show OSMCha/revert label only when it exists and label
  it as external review evidence, not as model truth.
- [ ] **System status:** data source, cache/fixture/live mode, request latency, inference
  latency, and failure reason.

## Phase 4 — visual demo implementation and evidence

- [ ] Define the demo API contract: input changeset id, normalized feature payload,
  prediction payload, edit ranking, attribution, geometry payload, and provenance.
- [ ] Make CLI inference and dashboard inference call the same function and checkpoint.
- [ ] Add golden-fixture tests that assert the displayed score, ranked edit, and map
  geometry match saved expected output within defined tolerances.
- [ ] Add an offline/demo mode with no network and no API key; it must launch from a
  clean checkout using committed small fixtures only.
- [ ] Add live-mode safeguards: bounded polling, shared request limiter, timeout,
  retry/backoff, response caching, and a visible "data may be stale" indicator.
- [ ] Include at least three scripted scenarios:
  - [ ] clean ordinary edit;
  - [ ] obvious geometry or mass-change case;
  - [ ] semantic/text case with a clear explanation.
- [ ] Capture a short walkthrough showing input → fetch → map before/after → score →
  suspicious edit → explanation. Keep a recorded version for service outages.
- [ ] Export a reproducible demo evidence bundle: screenshots, fixture ids, model id,
  request timestamps, dependency versions, and a one-command launch instruction.
- [ ] Test desktop and narrow-screen layouts, empty states, slow responses, malformed
  changesets, missing geometry, missing model checkpoint, and disabled Google Maps.
- [ ] Do not call the demo production-ready. Label it as a research prototype and make
  false-positive/false-negative limitations visible.

## Phase 5 — paper, presentation, and handoff

- [ ] Make the visual demo a named contribution/artefact in the paper, not an unlinked
  recording.
- [ ] Add one architecture figure and one end-to-end dashboard figure generated from
  the actual implementation.
- [ ] Report model quality separately from demo responsiveness; do not use UI polish as
  evidence of detection quality.
- [ ] Include limitations: incomplete geometry coverage, relations scope, presumed
  negative contamination, OSMCha/revert label differences, calibration drift, and map
  provider/licensing constraints.
- [ ] Verify ODbL/map attribution, Google Maps key handling, privacy redaction, and no
  account-level claims before publishing screenshots or a URL.
- [ ] Re-run the demo from a clean environment and record the exact commit, checkpoint,
  feature-schema version, and fixture manifest.
- [ ] Final acceptance: a reviewer can launch offline, inspect a known result, then run
  one live changeset if credentials/services are available, and obtain the same score
  from the CLI and dashboard.

## Explicitly deferred unless the core path is green

- [ ] Public hosted service with continuous polling.
- [ ] Google Maps as a required dependency.
- [ ] Full planet-scale live monitoring.
- [ ] Mobile-native client.
- [ ] Extra self-supervised pretexts beyond corruption contrast.
- [ ] Relations-specific modelling.
- [ ] Account-level user-risk rankings.

## Final v2 go/no-go checklist

- [x] One defensible supervised result exists on R1b. Hybrid, ovid: PR-AUC 0.899 [0.887, 0.910],
  F1 0.811, Acc 0.813, one-per-author 0.832; author-bootstrap tests in
  `experiments/rounds/r2b_2026-10-08_altair_h100_weightfix/analysis/deep_dive.json`.
- [ ] The temporal result on Corpus D is reproducible and label-source differences are
  disclosed.
- [ ] No leakage, future information, or pretraining/evaluation overlap remains
  untested.
- [ ] The demo uses the real inference path and exposes the actual evidence behind a
  flag.
- [ ] Offline replay works without secrets; live mode degrades gracefully.
- [ ] Paper, README, results, demo screenshots/recording, and repository status agree.

## Round 2 — proposed after the first training round (2026-09-21, not started)

Deadline 2026-10-02. Ordered by what the results say matters. Details in
`experiments/rounds/r1_2026-09-21_m4/README.md`.

### R2-A. Make the claims defensible — cheap, do first (~1 day)
- [x] Temporal baselines (2026-10-15, `results/temporal_baselines_2026-10-15.json`): hybrid on the
  temporal splits - ovid_modern 0.905, T1 0.813, T2 0.852, ovid 0.824. The 7-year drop is specific
  to the v3 deep model (0.874). Caveat: B0 metadata tripwire ROC 0.83 on ovid_modern temporal.
- [x] (was:) Temporal baselines: XGBoost and B0 on every temporal split. Without them, "the deep model
  holds up over time" has no reference, and ovid_modern's 0.891 may just be an easy test set.
- [ ] Author-robust evaluation on Corpus A: metrics with each author weighted equally, and with
  the 56 mass-revert authors (>20 positives) held out. This measures how much Ovid's label noise
  drives every score, ours and theirs.
- [x] Error analysis (2026-10-15, `experiments/rounds/r4_*/analysis/error_analysis.md`): FPs are
  newcomers adding parks/footpaths in iD (29/30), 2 likely unreverted vandalism; FNs are reverted
  good-faith work by experienced JOSM mappers (mapathons, imports, edit wars).
- [x] (was:) Error analysis: 30 highest-confidence false positives and false negatives on ovid,
  read by hand (mapathons, imports, bots).

### R2-B. Try once, properly, to beat XGBoost (~1–2 days, M4)
- [ ] Regularisation sweep chosen on validation only: dropout 0.3/0.5, AdamW weight decay,
  lr 3e-4, embed 32/64 (about 12 runs × 5 min). Best epoch 5 of 10 says the model overfits.
- [ ] Stronger text branch, since text is the only branch that pays: encode the comment and the
  tag text separately.
- [ ] Hybrid: XGBoost on tabular features plus the deep model's text/edit embeddings. If
  nothing beats XGBoost, report the tie honestly - the ablation insight (text matters, geometry
  does not) is still a result.
- [x] Optional: get Metal working - the M4 sees `GPU:0` with tensorflow-metal (2026-09-22,
  matmul 3.7x CPU; masked BiLSTM 15% slower). Superseded: training moved to the Altair H100.

### R2-C. Demo (~3 days)
- [ ] `score_changeset(cid)`: fetch → build rows with the same builder → saved normaliser →
  model. Replay mode on committed fixtures first, live mode second.
- [ ] Streamlit page: score, before/after map, text diff, branch attribution. Edit attention is
  near-uniform today, so show branch attribution, not "suspicious edit", unless R2-B fixes it.

### R2-D. Paper and presentation (~3 days)
- [ ] Headline framing: a faithful Ovid-protocol comparison, a 5-year modern corpus, leak audit,
  the text-dominance ablation, time drift, and the negative self-supervision result.
- [ ] Limitations: CPU-only training, label noise from mass reverts, B0 0.839 on revert_vandalism,
  presumed-negative N.

## Round 2 — model v2 (2026-09-22, deadline end of October)

Evidence and reasoning: `docs/model/MODEL_V2.md`. Round 1 results: `experiments/rounds/r1_2026-09-21_m4/`.

- [x] Diagnosed round 1. Signal families are complementary (WHO / WHAT / TEXT). Round 1 lacked
  WHAT: it saw counts, not which tag changed.
- [x] Found the benchmark inflation: XGBoost gets 0.858 normally but 0.764 counting one changeset
  per author. Author-robust metrics are now in every report.
- [x] Probe: edit tokens + text + author weighting lift the tree model to 0.898 / 0.836 one-per-author,
  F1 81.1 / Acc 81.3, level with Ovid's 81.66 / 81.37.
- [x] Builder v4: per-edit `diff_tokens` + `old_text`, and a changeset token bag (rebuild running).
- [x] Model v2: B6 edit-meaning branch, gated-attention MIL + max pooling, AdamW, dropout 0.3,
  author-balanced weights.
- [x] Hybrid baseline (trees + out-of-fold token/text models) and stage 07 ensemble.
- [x] Ablation rows redesigned to test each v2 claim (edit meaning, MIL vs mean, weighting).
- [x] Rebuild store v4, rerun baselines + hybrid on all tasks (CPU), re-freeze splits if ids change.
  Done 2026-09-22; the 12 splits did not change (byte-identical on the cluster too).
- [x] Bundle v1.1.0, run round 2 (RUNBOOK training plan), analyse into `experiments/rounds/r2_*`.
  Ran on the Altair H100 instead of the M4 (2026-10-07, 37 min): `r2_2026-10-07_altair_h100/`.
- [ ] If Stage 1 is retried: pretrain on edit tokens (masked-token), not on edit numbers.
- [x] WHERE signal: fetched from the planet changeset dump on the cluster, tested - no gain on ovid.

## Round 2 results and round 2.1 (2026-10-07 → 2026-10-09)

Evidence: `experiments/rounds/r2_2026-10-07_altair_h100/README.md`,
`experiments/rounds/r2b_2026-10-08_altair_h100_weightfix/README.md`.

- [x] Altair job script: one resumable job, logs, STATUS, timing, results zip (`altair/`).
- [x] Round 2 on the H100: all 8 steps exit 0 in 37 min; every masking check passed under float16.
- [x] float16 overflow can no longer leak through padded rows (pools use `where`, not multiply).
- [x] Bug found and fixed: focal loss returned a batch mean, so author weights never reached the
  deep model. Round 2b (fixed): ovid full ablation 0.878 vs 0.863 unweighted; deep one-per-author
  0.763 → 0.801.
- [x] Ablations settle: edit meaning (B6) is the largest single gain; geometry adds nothing;
  MIL vs mean pooling no difference.
- [x] Attention is now saved per edit. Finding: MIL weights exactly uniform (top/(1/n) = 1.00)
  - self-attention had no residual, so edits were blended before pooling.
- [x] v2.1: residual + LayerNorm around edit self-attention; ablation row `full_no_residual`.
- [x] Round 2c on the H100 with v2.1 (2026-10-09): yes. MIL top weight 1.00 → 2.64× uniform,
  sharper on damaging changesets; deep ovid 0.873 → 0.882, F1 0.813 (Ovid 0.817).
- [x] WHERE collected on the cluster (planet dump, 59 min; 93% of changesets have a distance
  from prior history). Probe on the hybrid: no gain (−0.0005 [−0.006, +0.004]). Negative result.
- [ ] Beat Ovid (+2 F1 needed for a claim): object-context features (what the edited object
  was) probed on the hybrid first, CPU.
- [ ] Ensemble weight overfits small validation sets (T1: −0.016 vs hybrid). Constrain or
  cross-validate the blend weight.
- [ ] Final run: 3–5 seeds of the chosen setup for every headline number.
- [ ] Temporal baselines (XGBoost / hybrid on the temporal splits) - still open from R2-A.

## Round 3 — model v3 (built 2026-10-11, demo 2026-10-19..21)

Design and evidence: `docs/model/MODEL_V3.md`. Every item is a config switch measured against v2.1.

- [x] Ensemble blend weight: bagged over 200 author bootstraps of validation (stage 07). Tested on
  round-2c predictions: within ±0.0007 everywhere, T2 +0.003; T1 weight spread 0.0–0.7 shows a
  single weight is unreliable there.
- [x] `split_seed` separate from `seed`: multi-seed runs keep the frozen test set.
- [x] Object context extracted (what each modified/deleted object was) - `C:/osm-data/probes/`.
- [x] Object context probe on the hybrid (CPU, 4 tasks): no gain (ovid −0.002, ovid_modern −0.001,
  T1 −0.004, revert +0.003; all inside 95% intervals). Negative result; real but redundant signal.
  `results/probes_v3_2026-10-11.json`.
- [x] Code: hybrid variants base / ctx / emb / ctx_emb; deep `text_emb`, `context_tabular`,
  `edit_layers`; stage 08 text embeddings; stage 09 stacking (5 author folds, out-of-fold).
- [x] Ablation rows 11–14 (full_text_emb, full_context, full_2layers, full_v3); wiring check v3.
- [x] Job script reads a per-round step plan (`altair/plans/`); zip ships side tables.
- [x] Independent review of the v3 code before the cluster run (two passes, 2026-10-11): pass 1
  found 5 problem groups (incl. ovid_modern stacking columns, silent ablation fallback, split
  seed in 4 stages, job-script parsing); pass 2 confirmed the fixes and found 1 exit-code bug.
  All fixed and retested; zip `r3_v3_explore` built from commit 80dc678.
- [x] Round 3 on the H100 (2026-10-12, `experiments/rounds/r3_2026-10-12_altair_h100_v3explore/`).
  Stack lifts ovid above the hybrid (0.903 vs 0.899, F1 0.821 vs 0.811; Ovid 0.817) - to be
  confirmed with seeds + author bootstrap. edit_layers=2: +0.0034 (~2x seed sd). Object context
  in the deep model: −0.004. Ran twice concurrently (double launch); embeddings failed (numpy).
- [x] Fix round-3 problems (2026-10-13): launch lock with PID + heartbeat (tested: second launch
  refused, lock released at the end); embedding env pinned to numpy-1.22-compatible versions in a
  fresh folder; stage 07 pairs this invocation's own runs; summary works without a base row.
- [x] Adopted from round 3: edit_layers=2 as deep default; object context dropped.
- [x] Round 4 (2026-10-13, `experiments/rounds/r4_2026-10-13_altair_h100_seeds/`): one launch,
  all 24 steps ok, 3 h 43 m; embeddings worked (CUDA, 37 s); 3 seeds × 6 tasks + temporal.
- [x] Embeddings measured: help the trees (ovid +0.0037, T1 +0.0067, revert_vandalism +0.0038,
  significant) and the stack; alone in the deep model −0.002; with 2 layers +0.004.
- [x] Final setup decided on round 4 (3 seeds + author bootstrap): stack + emb for ovid
  (+0.0097 vs hybrid, F1 0.828 vs Ovid 0.817), ensemble for ovid_modern / revert tasks; on T1/T2
  nothing beats the hybrid significantly.
- [x] Final multi-seed round = round 4 (3 seeds). 5 seeds not needed: seed sd ≤ 0.007 vs
  test-set interval ≈ ±0.012.
- [x] v3 deep model transfers worse over 7 years (ovid_modern temporal 0.874 vs 0.908 in v2.1;
  one seed); trees hold (0.905). Decided 2026-10-22: no 3-seed temporal rerun - reported as a
  single-seed observation.
- [x] README rewritten with final results; docs organised into docs/{model,data,running,planning}.
- [x] Dataset release v1.2.0 (side tables: object context, WHERE, dump source).
- [x] Visual / demo (2026-10-22): review console in `visual/` - PR #3 (brickstercodes) merged as the
  base, PR #2's prototype (tanay15) kept in `visual/prototype/` and its ideas folded in (overview
  map, offline mode, CLI). Redesign; overview with round-4 metrics; before/after + type toggles;
  map <-> edit sync; Google Maps panel (key in visual/.env); 27 tests (served scores = round 4).
- [x] Trained checkpoints released (round 4 deep, stack folds, CPU hybrid refit).
- [ ] Report / research paper.
- [ ] Repo cleanup.
