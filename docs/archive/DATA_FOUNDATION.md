# Data foundation plan

**Nothing is built from this yet. It is the design to agree before fetching resumes.**

Written 2026-09-08 after the audit in [`DATA_AUDIT.md`](DATA_AUDIT.md). It replaces the
data half of `CHECKLIST_v2` Phases 0–1 with a sequence that ends in a frozen, versioned,
published dataset.

---

## 0. Why this comes before everything else

The project currently has three partially-fetched corpora, two label sources with
*different meanings*, an empty feature store, and 67 open checklist items against a
four-week deadline. Every modelling hour spent on top of that is an hour that may have to
be thrown away, because a result computed on a dataset you cannot describe is not a result.

The failure mode is specific and common: you train, you get a number, someone asks "what
exactly was the negative class?", and the honest answer changes what the number means. We
already know that question has an uncomfortable answer (§1). Fixing it after modelling
costs the modelling.

So: **finalise, characterise, freeze, and publish the dataset. Then model.** One pass, no
revisiting.

---

## 1. The task definition — the decision everything else hangs on

The audit found that OSMCha's negative class is *reviewed-and-cleared*, not *clean*.
Reviewers look at what a heuristic already flagged, so both classes are drawn from the
suspicious tail. That is not a bug to fix; it is a fact about the data that determines what
question the model is answering.

There are two defensible tasks, and the honest move is to **define both and report both**.

| | **T1 — Triage** | **T2 — Detection** |
|---|---|---|
| Positive | OSMCha `harmful=True` | same |
| Negative | OSMCha `checked=True, harmful=False` | random *unreviewed* changesets |
| Question | "Of the changesets a filter flagged, which deserve a human?" | "Of all changesets, which are damaging?" |
| Matches | how OSMCha is actually used today | what the paper claims |
| Negative quality | human-verified clean | ~99.5% clean, weakly labelled |
| Bias | both classes from the suspicious tail | positives biased, negatives unbiased |

**T2 is the headline. T1 is the operationally useful one.** Reporting only T1 while
claiming T2 would be misrepresentation; reporting only T2 throws away the cleanest labels
in the dataset. The gap between them is informative in exactly the way the R1a/R1b gap is —
it measures how much of the score comes from the flagging heuristic rather than the model.

This mirrors a decision already made elsewhere in the project and should be consistent
with it: report the comparable-but-flawed number *and* the honest one, and label both.

**Decision required from the team.** Everything downstream — corpus sizes, split files,
metric tables, the demo's framing — depends on this being settled now rather than argued
about in week three.

---

## 2. Corpus design

Three corpora, each with one job. Anything that does not serve one of these jobs is out of
scope for the four weeks.

### Corpus M — Modern, labelled (**primary**)

- **Source:** OSMCha reviewed changesets, window fixed by Gate D0.
- **Labels:** `harmful=True` (positive) and `checked=True, harmful=False` (T1 negative).
- **Role:** training and the headline results. Geometry, text, sequence and user branches
  all evaluated here.
- **Why primary:** measured, 2026 alone offers 8,298 harmful + 19,002 reviewed-clean —
  already larger than the whole Ovid benchmark, with human decisions instead of a revert
  proxy, on data that reflects how OSM is edited today.

### Corpus N — Random unreviewed negatives

- **Source:** uniform random sample of changeset ids from the *same window* as M, excluding
  every id that appears in OSMCha at all.
- **Labels:** presumed clean. Real vandalism prevalence among them is well under 1%, so the
  contamination is bounded and must be **stated as a number, not waved at**.
- **Role:** makes T2 possible. Without it there is no unbiased negative class anywhere in
  the project.
- **Size:** matched to M's positive count for the balanced view; the prevalence-corrected
  metrics do the rest without needing millions of rows.

### Corpus A — Ovid historical (**comparability, plus a geometry head-to-head**)

- **Scope:** metadata, contents, and a **~2,000-changeset geometry subset** (decision 2).
  No user histories — Ovid's user features are aggregates, which we already have from
  metadata, and point-in-time histories are a Corpus M concern.
- **Role, two tables:**
  1. reproduce Ovid's own feature set under Ovid's own protocol → "we match published work"
  2. add the geometry branch on the 2,000-changeset subset → **"geometry beats the state of
     the art on the state of the art's own benchmark"**
- **Cost of the second table:** ~10 h of the ~31 h Overpass budget, and a wide confidence
  interval — roughly 400 changesets land in the test fold. The interval is reported, not
  hidden. A wide honest comparison against published work beats no comparison.
- **Why not the full corpus:** the remaining ~21 h goes to Corpus M, where the geometry
  claim is tested on current data at a size that supports a tight interval.

**Dropped entirely:** Corpus B (the S3 bucket window) and Corpus C (bulk pretraining) stay
dropped per `PROJECT_PLAN` §13. Stage 1 remains week-4 conditional.

---

## 3. Sequence — five gates, each with a stop condition

### D0 — Characterise OSMCha before committing to a window

*Nothing is fetched in bulk until this is answered.* The 2026 rate cannot be extrapolated
backwards; OSMCha adoption grew over time, and guessing the window is the exact shortcut
this whole exercise exists to avoid.

Probe, per year from 2019 to 2026:

1. `harmful=True` count and `checked=True, harmful=False` count
2. distinct reviewers, and **the share held by the top 5** — a corpus where one person made
   most decisions is one person's opinion, not a consensus label
3. changesets reviewed more than once, and whether the reviewers agreed — the only route to
   an inter-rater number
4. overlap with Corpus A ids (expected ~0, must be checked)
5. class balance drift by year

**Output:** `results/gate2a_osmcha_coverage.json` and a one-line written decision fixing the
window.
**Stop condition:** if the top 5 reviewers hold more than ~60% of decisions, or a year's
harmful count is under ~500, that year is excluded and the reason recorded.
**Cost:** ~1 hour, count queries only.

### D1 — Fetch, with coverage accounting

One clean run per corpus. Every fetcher already records per-id status; the requirement here
is that **each store can report what it failed to get and why**, because a coverage claim in
the paper has to be measured.

Order matters — cheapest and most reusable first:

| Step | Volume (*est.*) | Rate (measured) | Wall clock |
|---|---|---|---|
| OSMCha labels, full window, windowed pagination | 50k–100k | ~25/s | ~1 h |
| Finish Corpus A contents | 10,021 | 3/s | ~1 h |
| Corpus M changeset metadata | ~50k | 3/s | ~4.6 h |
| Corpus N random ids + metadata | ~20k | 3/s | ~1.9 h |
| Corpus M+N contents | ~70k | 3/s | ~6.5 h |
| User accounts (`account_created`) | ~30k unique | 3/s | ~2.8 h |
| Point-in-time user histories | ~30k users | 3/s | ~2.8 h |
| **Geometry `adiff`, id-sampled subset of M** | 6,000 | **3.17/min** | **~31 h** |

Everything except geometry fits inside a day. Geometry runs in the background across the
week and is the only thing that should ever be split across the team's machines.

**Stop condition:** any store below 95% HTTP 200 gets diagnosed before the next step, not
after.

### D2 — Dataset card

Before any preprocessing, characterise what was actually collected — not what was intended.
`docs/data/DATASET_CARD.md`, generated by a script so it cannot drift from the data:

- exact date windows per corpus, in UTC
- row counts per corpus and per class
- **geographic distribution** by continent and top countries (Corpus A is 43.5% Europe;
  Corpus M will differ and that difference matters)
- temporal distribution by month
- changeset size distribution, including the tail that Ovid zeroes out
- editor distribution
- user concentration: how many changesets come from the top 1% of accounts
- reviewer concentration for M
- known biases, stated plainly
- coverage gaps with counts and causes

**Stop condition:** if the card reveals a distribution that invalidates a planned
experiment, the experiment changes here — not after it has been run.

### D3 — Preprocess to a versioned feature contract

- `SCHEMA_VERSION` bumped and recorded in every Parquet and every results file.
- `docs/SCHEMA.md`: **every field** with dtype, unit, range, missing-value policy, and the
  source it derives from. A field whose provenance cannot be written down does not ship.
- One field name means one thing everywhere. The current tree has `id`/`cid` and
  `vandalism`/`label` for the same concepts across `fetch_corpus_a.py` and `store.py`; that
  gets reconciled here, once.
- Preprocessing is a pure function of (raw store, schema version) → Parquet. Re-running it
  on the same inputs must produce byte-identical output; a test asserts it.

### D4 — Splits, frozen as files

- R1a random (leaky, labelled as such), R1b user-disjoint + class-stratified, and the
  temporal split for T2.
- **Persisted as id lists**, not recomputed at training time. A split that is recomputed is
  a split that silently changes when the split code changes.
- `assert_user_disjoint` runs in CI on the frozen files.

### D5 — Freeze, package, release

Covered in §5.

---

## 4. What "the data is done" means

Acceptance criteria. All must hold before a single model trains.

- [ ] Task definition agreed and written into `PROJECT_PLAN` (T1 and T2, both reported)
- [ ] Gate D0 answered; window fixed in writing with reviewer-concentration numbers
- [ ] Every corpus ≥95% HTTP 200, with failures enumerated by cause
- [ ] Geometry coverage reported **by object type and by class** — if geometry coverage
      correlates with the label, the geometry result is confounded and must be reported that way
- [ ] `DATASET_CARD.md` generated from the data, not written by hand
- [ ] `SCHEMA.md` complete; no field without provenance
- [ ] Preprocessing reproducible byte-for-byte on the same inputs
- [ ] Splits frozen as files; disjointness asserted
- [ ] Corpus N contamination rate estimated and stated
- [ ] 100-changeset manual label audit done for **both** label sources, with inter-rater
      agreement — this is the number that lets the paper say how noisy its labels are
- [ ] Dataset packaged, checksummed, released, and the release URL recorded in the README

---

## 5. Packaging and release

Target: **https://github.com/ayushmgarg/osm_bad**, GitHub Releases.

### What ships

| Asset | Contents | *Est.* size |
|---|---|---|
| `mapguard-dataset-v1.0.0-features.zip` | the three Parquet tables + split files + schema | ~100–300 MB |
| `mapguard-dataset-v1.0.0-raw.zip` | compressed SQLite stores (metadata, labels, geometry) | ~0.5–1.5 GB |
| `DATASET_CARD.md` | rendered in the release body | — |
| `SHA256SUMS` | checksums for every asset | — |

GitHub allows 2 GB per release asset. If raw exceeds it, it splits by corpus rather than by
arbitrary byte ranges, so each part stays independently usable.

**osmChange XML is not released.** It is bulk, reconstructible from the OSM API by id, and
the ids are in the release. Shipping it would multiply the size for no reproducibility gain.

### Release body — required content

Dates, geography and provenance, because "a zip of some OSM data" is not a dataset:

- exact UTC window per corpus
- row counts per corpus and class
- geographic coverage summary
- label provenance per class, and **the task definitions T1/T2 restated**
- schema version and the commit SHA that produced it
- known biases and coverage gaps
- licence and attribution block
- one-command instructions to reproduce from the ids

### Licensing and privacy — non-negotiable

- **ODbL.** OSM data is ODbL; a derived dataset is a Derivative Database and **must** be
  released under ODbL with "© OpenStreetMap contributors". This goes in the release body,
  the dataset card, and a `LICENSE-DATA` file. Releasing it under the code's MIT licence
  would be a licence violation.
- **`uid` and `user` are personal data**, and this dataset labels people's edits as
  damaging. The released tables carry a **salted hash** of uid and **no usernames at all**.
  The salt stays local and unpublished. Changeset ids remain, because they are public and
  the data is reconstructible from them anyway — but nothing we publish should make it
  easier to build a list of "vandals" than OSM already does.
- **No account-level aggregates** in the release. Per-changeset rows only.
- OSMCha reviewer names (`check_user`) are also personal data and get the same treatment.

### Versioning

Semantic, on the *data*: `v1.0.0`. Minor bump for added rows, major for a schema change or
a label-semantics change. The dataset version is recorded in every results file, so any
number in the paper can be traced to the exact data that produced it. **A release is
immutable** — a mistake becomes `v1.0.1`, never an edited asset.

---

## 6. Timeline and what it displaces

Roughly one week of the four, front-loaded.

| | |
|---|---|
| D0 characterise | 2 h |
| D1 fetch (excl. geometry) | ~1 day |
| D1 geometry | ~31 h, background, parallelisable |
| D2 dataset card | 3 h |
| D3 preprocess + schema | 1 day |
| D4 splits | 2 h |
| D5 package + release | 3 h |
| manual label audit | 3 h, two people |

**Displaced:** Stage 1 self-supervision moves from "week 4 conditional" to "almost certainly
cut". That was already the plan's first sacrifice and it stays ablatable — Stage 2 was
designed to stand alone.

**Protected:** the three correctness rules do not get traded away for time — point-in-time
user features, user-disjoint splits, and pretraining/evaluation disjointness. Those are the
difference between a result and a number.

---

## 7. Decisions — SETTLED 2026-09-08

| # | Decision | Chosen | Consequence |
|---|---|---|---|
| 1 | Task framing | **Both T1 and T2** | ~20k random unreviewed changesets fetched as Corpus N (~2 h). Every metric table carries both columns; the gap is reported as a result |
| 2 | Geometry budget | **Split: ~2,000 Corpus A, ~4,000 Corpus M** | Head-to-head against published SOTA *and* a larger modern result. The Corpus A half has ~400 test changesets, so its interval is wide — reported as such, not hidden |
| 3 | Release scope | **Features + raw stores** | Separate assets, both ODbL. Makes it a reusable benchmark rather than a reproducibility appendix |
| 4 | D0 exclusion rule | **Measure first, then set, and disclose** | Concentration is reported per year regardless. The threshold is chosen after seeing the distribution, and the paper says so explicitly |

### On decision 2 — a revision worth recording

The first draft of this document cut geometry from Corpus A entirely, on the reasoning that
Ovid has no geometry so the comparison does not need it. That was right about the mechanics
and wrong about the claim. *"Our geometry branch beats the state of the art on the state of
the art's own benchmark"* is a materially stronger sentence than *"geometry helps on a
corpus we assembled ourselves"*, and giving it up to save fetch time was a bad trade. The
split keeps both claims at the cost of a wide interval on one of them.

### On decision 4 — the honesty condition

Setting the threshold after seeing the distribution is only defensible **because it is
disclosed**. The paper must say that reviewer-concentration limits were chosen after
inspecting per-year coverage, and must publish the per-year numbers so a reader can apply
their own threshold. Choosing it post-hoc and presenting it as pre-registered would be
misrepresentation.
