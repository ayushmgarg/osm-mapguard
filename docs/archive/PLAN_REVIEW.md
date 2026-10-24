# MapGuard — plan review, 2026-09-04

Independent evaluation of `PROJECT_PLAN.md` v1 before any modelling work starts.
Every overturned claim was measured, not argued. Raw output: [`results/gate0d_review_probes.json`](../results/gate0d_review_probes.json).

**Verdict: the project is sound and the contributions are real. Two structural flaws
and one arithmetic impossibility had to be fixed before work starts. `PROJECT_PLAN.md`
has been updated; this file records what changed and why.**

---

## Summary

| # | Flaw | Severity | Status |
|---|---|---|---|
| F1 | Corpus A declared geometry-impossible. It is not — two working routes | **Critical** | fixed, §7/§8 rewritten |
| F2 | "Latest OSM data only reachable one changeset at a time" — false | **Critical** | fixed, Corpus D added |
| F3 | R2 realistic prevalence is arithmetically impossible with 9,138 negatives | **Critical** | fixed, §11 R2 rewritten |
| F4 | R3 says "test on 2026", corpora stop at Mar 2025 | High | fixed by F2 |
| F5 | User-disjoint split silently breaks the claimed Ovid comparison | High | fixed, R1a/R1b |
| F6 | User-history features leak the future | **Critical** | fixed, §5.3 point-in-time rule |
| F7 | G4 (large changesets) claimed but no mechanism specified | High | fixed, §5.3 hierarchical encoder |
| F8 | Pretraining corpus overlaps the evaluation corpus | High | fixed, §7 disjointness rule |
| F9 | Corruption-contrast may learn synthetic artefacts, not vandalism | Medium | fixed, zero-shot transfer check |
| F10 | No leakage tripwire baseline | Medium | fixed, baseline B0 |
| F11 | Bucket density never measured, only its bounds | Medium | measured: 82% |
| F12 | Stage 1 compute never sized | Medium | fixed, throughput gate |
| F13 | Keras/TF locked without a ragged-attention spike | Medium | fixed, Gate 1d |
| F14 | Fetch latency stated as 0.2 s; the repo's own gate recorded 1.2 s | Low | corrected |
| F15 | No ethics / licensing / pseudonymisation section | Low | added §16 |
| F16 | No estimate of label noise | Low | added manual audit of 100 |
| F17 | Relations never handled anywhere | Low | scoped out explicitly |

---

## F1 — Corpus A geometry is *not* impossible (critical)

**The plan said** (§8.3): way/relation geometry on the 2014–2019 labels is `expensive
(N calls/way)`, therefore branch B1 runs only on Corpus B. `src/mapguard/data/osmchange.py`
has already hard-coded this belief in its docstring: *"Corpus A geometry is node-only,
which is exactly why Corpus B exists."*

**Why this mattered more than it looks.** B1 is contribution #1 of the paper. Under v1
it would only ever be evaluated on Corpus B — a corpus that has **no labels yet** and
**no published baseline**. The headline claim would never be tested against Ovid.
That is not a data problem, it is a "the main result is untestable" problem.

**Two working routes, both measured today:**

1. **Overpass `adiff`** — returns `<action><old>…</old><new>…</new></action>` with node
   coordinates already resolved onto the way, at any date back to 2012.

   ```
   [adiff:"2013-12-31T00:00:00Z","2014-01-02T00:00:00Z"];way(202342236);out geom meta;
   -> 200, 4,260 bytes, 2.9 s, full before+after polyline
   ```

   Per-changeset form works too: take the changeset's bbox and `[created_at, closed_at]`
   from the API, run `adiff` over that window, keep elements whose `<new>` carries the
   changeset id. Verified on changeset 52850166 — 5 actions, all correctly attributed.
   Gotcha: `[bbox:...]` in the setting block fails to parse alongside `adiff`; put the
   bbox on each statement instead.

2. **Versioned multi-fetch** — `/api/0.6/ways?ways=<id>v<ver>` and
   `/api/0.6/nodes?nodes=<id>v<ver>` both return 200. Up to 725 versioned ids per
   request, so "previous state" costs a fraction of a call per object, not one each.

**Fix:** B1 runs on Corpus A. Overpass `adiff` is the primary geometry route,
versioned multi-fetch the fallback for changesets whose bbox is too large or absent.
Corpus B survives, but its job changes from "the only place geometry works" to
"the modern half of the temporal study".

---

## F2 — "Latest OSM data" was wrong (critical)

**The plan said** (§7): *"Genuinely current data … is reachable only through the OSM API
or Overpass, one changeset at a time. Fine for a live demo; not for bulk training."*

**Measured, 2026-09-04:**

| Stream | State | Freshness |
|---|---|---|
| `/replication/minute/state.txt` | seq 7,271,812 | `2026-09-04T04:12:47Z` |
| `/replication/changesets/state.yaml` | seq 7,170,065 | `2026-09-04 04:12:51Z` |
| `/replication/day/state.txt` | seq 5,105 | `2026-09-04T00:00:00Z` |

All live to the current minute. Bulk. Free. No account. A changeset-replication minute
file is ~2.9 KB gzipped and carries `id, created_at, closed_at, num_changes, user, uid,
bbox, comments_count, comment, created_by, locale, source` — everything B5 and the
revert miner need, without touching the 8.7 GB dump for recent windows.

**Fix:** new **Corpus D — Live 2026**, built from the changeset-replication stream plus
per-changeset API download. This is what makes R3 an actual 2026 temporal test rather
than a demo, and it is the direct answer to the "use the latest OSM data" requirement.

---

## F3 — R2 cannot be run as written (critical, arithmetic)

**The plan said:** R2 = "negatives upsampled to 1:100, 1:1000".

There are **9,138 negatives**. Against 9,138 positives, 1:100 needs 913,800 negatives and
1:1000 needs 9,138,000. The word "upsampled" is doing impossible work — you cannot
upsample your way to a lower positive rate, and duplicating negatives changes nothing
about PR-AUC beyond inflating the denominator with copies.

**Fix — three parts, all in the rewritten §11:**

1. **Primary: prevalence-corrected PR curves.** Compute precision/recall on the balanced
   test set with importance weights `w = (1-pi)/pi * (n_pos/n_neg)` for the target
   prevalence `pi`. Statistically exact, costs nothing, and is the standard correction.
2. **Confirmatory: a genuinely sampled 1:100 set.** Fetch ~50,000 unlabelled Ovid-era
   changesets as presumed-negatives (prevalence of real vandalism among them is <1%, so
   the contamination is bounded and quantifiable). At 1.2 s/call and 4 concurrent
   workers that is ~4.2 h. Budget it as its own task.
3. **Drop 1:1000 as a sampled regime.** Report it weight-corrected only, and say so.

---

## F6 — User-history features leak the future (critical)

B4 consumes the user's past changesets. Nothing in the plan says *past relative to what*.
If user features are computed from the full changeset dump, then for a 2016 target
changeset the model sees that the account was blocked in 2017 — a perfect and completely
useless predictor.

**Fix:** every user feature and every user-history sequence is computed **strictly from
changesets with `created_at < target.created_at`**. Written into §5.3 as a hard rule,
and into the checklist as a test (`test_user_features_are_point_in_time`).

---

## F7 — G4 is claimed but never engineered (high)

The plan's headline criticism of Ovid is that it zeroes out the edit vector above a size
threshold, so mass vandalism and broken imports fall through. But nowhere does the plan
say how MapGuard encodes a 50,000-edit changeset. Attention over N edits with unbounded N
does not fit in a T4.

**Fix — hierarchical encoding, specified in §5.3:**

- **exact** whole-changeset summary statistics computed over *all* N edits (counts, op
  mix, tag-key histogram, geometry moments) — these never truncate, so scale is always
  visible;
- per-edit encoding over a **stratified sample** of up to K=256 edits (largest
  displacement, largest area change, text-bearing edits, then random fill);
- attention pooling over that sample, with the sampling ratio `K/N` fed in as a feature.

Now the claim "we do not gut large changesets" is defensible: nothing is zeroed, and the
size signal is exact.

---

## F5, F8, F9, F10 — evaluation integrity

- **F5.** R1 uses a user-disjoint split; Ovid does not. Comparing 81.66 against a
  user-disjoint number is comparing a harder task to an easier one and calling it a win.
  Fix: report **R1a** (random, user-leaky) and **R1b** (user-disjoint). The gap between
  them is itself a result.
  **Correction 2026-09-18, from reading the paper:** Ovid's split *is* user-disjoint
  ("training, validation, and test sets are disjoint concerning OSM users", §4.1). So R1b is
  the apples-to-apples comparison with 81.66, and R1a is only a leaky upper bound.
- **F8.** Corpus C (pretraining) is drawn from the same window as Corpus B (evaluation).
  Pretraining on the test set makes R4 — the single most important figure — meaningless.
  Fix: Corpus C excludes every changeset id **and every uid** appearing in Corpus B/D eval splits.
- **F9.** Pretext C trains on synthetic corruptions. A model that separates
  real-vs-synthetic need not separate real-vs-vandalism. Fix: before fine-tuning, score
  the frozen corruption head on the real labelled set and report zero-shot AUC. If it is
  ~0.5, the corruptions are the wrong distribution and must be redesigned — better to
  learn that in week 8 than in the paper.
- **F10.** No tripwire. Fix: baseline **B0** = logistic regression on changeset comment
  length, editor string, and account age only. If B0 reaches F1 ~0.8, the labels are
  leaking (e.g. revert-related text in the record) and every later number is void.

---

## Smaller corrections

- **F11.** Gate 0c pinned the bucket's *bounds* but never its *density*. Measured: 59/72
  = **82%** across 147M–164M, no systematic hole. The 18% of 404s is consistent with
  OSM's normal rate of empty changesets. ~14M usable. Corpus B/C confirmed viable.
- **F12.** Stage 1 was never sized. Added a throughput gate: measure edits/sec on one
  Kaggle P100 at the real batch size, multiply out, and only then fix the pretraining
  token budget. 30 GPU-h/week is a hard ceiling, not a suggestion.
- **F13.** Decision 2 locks Keras 3/TF. Variable-length sets of edits, masked multi-head
  attention and five heterogeneous inputs is exactly where TF's ragged handling gets
  awkward. Added **Gate 1d**: a one-day spike building a toy multi-input masked-attention
  model end-to-end. Pass → keep Keras. Fail → PyTorch, decided by day 5, not month 3.
- **F14.** §8.2 claimed "~0.2 s each" for the API download. The repo's own
  `results/gate0b_fallback.json` records **1.2, 1.17, 1.26 s**, and fresh probes hit
  8.2 s at the tail. Every fetch budget rebuilt on 1.2 s.
- **F15.** Added §16 — ODbL attribution, and `uid`/`user` are personal data. Store the
  raw uid locally, publish only a salted hash. A vandalism-detection paper that names
  accounts is an ethics problem as well as a GDPR one.
- **F16.** Added a manual audit of 100 randomly drawn labelled changesets to estimate the
  label noise rate. Two hours of work; converts "label noise is inherited from Ovid" from
  a hand-wave into a measured number in the limitations section.
- **F17.** Relations are parsed (`n_refs` counts members) but no branch models them.
  Now stated explicitly as out of scope rather than left ambiguous.

---

## What the review did *not* change

The core thesis survives intact and is, if anything, stronger:

- The industry synthesis (§2) is accurate and the Stripe analogy is the right frame.
- The gap analysis (§3) is read from Ovid's feature tables, not its abstract — that is
  the right way to do it.
- The two-stage design, the five branches and the syllabus mapping (§6) all hold.
- The "no outcome leaves us empty-handed" framing (§11) is genuine, not a hedge.
- Gate 0's failure was handled well. The two-corpus response was a good call made on
  incomplete information; it just needed the geometry premise corrected.

## Code that now carries a false premise

Not changed by this review — these are checklist items:

- `src/mapguard/data/osmchange.py` docstring: *"Corpus A geometry is node-only"* — false, see F1.
- `src/mapguard/features/build_corpus_a.py` docstring: same assumption, and
  `changeset_features` computes node-only geometry by design.
- `src/mapguard/data/fetch_corpus_a.py`: no `adiff` enrichment stage, and its
  `--rate 4 --workers 8` default needs re-checking against 1.2 s real latency.
- `osmchange.parse()`: `elem.clear()` empties children but leaves the stripped element
  attached to the root, so a 50,000-object changeset still accumulates 50,000 husks.
  Detach from the parent, or the streaming claim in the docstring is only half true.
