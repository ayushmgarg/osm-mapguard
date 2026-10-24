# Data audit — what is real, what is assumed, what is missing

Written 2026-09-08, before any further fetching. Every number marked **measured** came from
a live call recorded in `results/`; every number marked *estimated* is arithmetic on top of
a measured rate and should be treated as a projection, not a fact.

---

## 1. Provenance — where every row actually came from

| Artefact | Source | Authenticity |
|---|---|---|
| `ovid_labels.tsv` | published Ovid benchmark | third-party, unverified by us. 18,276 ids + boolean |
| Corpus A metadata | `GET /api/0.6/changeset/<id>` | **measured** — 18,276/18,276, HTTP 200 |
| Corpus A contents | `GET /api/0.6/changeset/<id>/download` | **measured** — 8,255 fetched, 0 failures, 200/200 sampled parse clean |
| Corpus A geometry | Overpass `adiff` | **measured** — 11 rows only. 93% yield on a 100-changeset gate sample |
| OSMCha labels | `osmcha.org/api/v1/changesets` | **measured** — 4,654 rows, 226 reviewers |
| Corpus D | `planet.osm.org/replication/changesets` | **measured** — 1,362 changesets, 1,359 in 2026 |

**No synthetic data anywhere.** The generator that produced it was deleted in
`1257fb4`. Fixtures under `data/fixtures/` are *selected* real changesets.

### Verified, not assumed

200 randomly sampled fetched changesets were re-parsed: **200/200 parsed, 0 errors**,
17,453 edits total. The documents carry the `openstreetmap-cgimap` generator string, ODbL
attribution, real object ids, versions, timestamps and inline coordinates. Three of the 200
are 294-byte self-closing `<osmChange/>` documents — those are **empty changesets**, a real
OSM phenomenon, not truncation.

---

## 2. Authenticity problems, worst first

### A1 — The negative class in the OSMCha data is not "clean", it is "reviewed and cleared"

This is the most serious issue and it is not a fetching bug; it is structural.

OSMCha reviewers do not sample changesets at random. They look at things a heuristic already
flagged. So:

- `harmful=True` = a human looked at a *suspicious* changeset and confirmed it
- `checked=True, harmful=False` = a human looked at a *suspicious* changeset and cleared it

Training on those two classes learns **"among changesets that already looked suspicious,
which are actually damaging"**. That is a real and useful task — it is precisely the review
queue triage problem — but it is *not* the task the paper currently claims, which is
"among all changesets, which are damaging".

**This must either be stated as the task, or corrected with a random unreviewed sample as a
third class.** Reporting it as general vandalism detection without saying so would be
misrepresentation. Recommended fix in §4.

### A2 — Ovid labels are revert-derived, and a revert is not a synonym for vandalism

Inherited from the benchmark, not introduced by us. Reverts also happen for import
mistakes, accidental edits, and mapping disputes. Ovid's F1 of 81.66 is measured against
the same proxy, so the comparison stays fair — but the absolute number describes
"predicting reverts", not "predicting vandalism".

Unmeasured by us. The 100-changeset manual audit in the plan is what turns this from a
hand-wave into a number, and it has not been done.

### A3 — The 50/50 balance is constructed

Ovid downsampled negatives to match positives. Real prevalence is well under 1%. Every
balanced-set metric therefore overstates operational performance by a wide margin. This is
handled — `evaluation/metrics.py` does the prevalence correction — but the raw 50/50
numbers must never be presented as operational.

### A4 — Corpus A is Europe-weighted

**Measured**: Europe 43.5%, North America 20.6%, Asia 10.9%, Africa 9.8%, South America
5.7%, Oceania 2.9%. That is OSM's real contributor distribution, not a sampling choice, but
it means the model is best evidenced on European edits. Limitations paragraph.

### A5 — Corpus A is six to twelve years old

2014-01-01 to 2019-12-31. OSM's editor mix, tag conventions and vandalism styles have all
moved since. The top editor in Corpus A is `Potlatch 2`, which is effectively retired. A
model validated only there is validated on a map that no longer exists.

### A6 — Account age was never in Gate 1c

Gate 1c ran with `--no-user`, so **AUC 0.656 is comment length plus editor string only**.
`users.sqlite` was never fetched. The gate result is real but narrower than its own
description says. Must be re-run with the third feature before it is cited.

---

## 3. Coverage gaps — what simply is not there

| | Have | Should have |
|---|---|---|
| Corpus A contents | 8,255 | 18,276 |
| Corpus A geometry | **11** | ~6,000 (id-sampled subset) |
| User histories | **20** | 8,768 |
| `users.sqlite` account_created | **0** | 8,768 |
| OSMCha labels | 4,654, from 2026-06-01 | see §4 |
| Feature store | **empty** | three Parquet tables |

The OSMCha gap has a specific cause: the first harvest used offset pagination, which died
at page 34 with a connection timeout and took the entire reviewed-clean class with it. The
windowed-by-date fix is committed but has not been re-run over the full window.

---

## 4. The pivot: make the modern corpus primary

The current design treats 2014–2019 as the training corpus and 2026 as a temporal test.
Given a four-week budget, that is the wrong way round.

### Why

| | Corpus A (2014–2019) | Modern OSMCha corpus |
|---|---|---|
| Size | 18,276, **fixed forever** | 27,300 for 2026 alone (**measured**) |
| Labels | revert-derived proxy | **human review decisions**, 226 reviewers |
| Geometry | `adiff`, 3.17 q/min, subset only | same rate, but a smaller set to cover |
| Currency | up to 12 years stale | current |
| Growth | none | ~1,000 harmful/month (*estimated* from measured 2026 rate) |

**Measured**: 8,298 harmful + 19,002 reviewed-clean available for 2026-01-01 onward alone.
That is already **larger than the entire Ovid benchmark**, with better labels, on current
data.

### Revised corpus roles

- **Corpus M (modern, primary).** OSMCha-labelled, widest window that Gate 2a shows is
  available. Training, R1, R2, and the demo all run here.
- **Corpus A (historical, comparability only).** Reproduce Ovid's protocol, report the
  number, demonstrate we match published work, then stop. No geometry enrichment needed
  beyond what supports that one table — which frees most of the 31-hour Overpass budget.
- **Corpus N (random negatives).** The fix for A1. A random sample of *unreviewed*
  changesets from the same window, which are overwhelmingly clean at real prevalence. This
  gives a third class and lets both tasks be reported honestly:
  - triage: harmful vs reviewed-clean
  - detection: harmful vs random unreviewed

### What has to be probed before committing (Gate 2a)

Nothing below is measured yet and none of it should be assumed:

1. **How far back does OSMCha review coverage go, and how dense is it per year?** The 2026
   rate cannot be extrapolated backwards — OSMCha adoption has grown over time.
2. **How many distinct reviewers per year**, and does any single reviewer dominate? A
   corpus where one person made most decisions is one person's opinion.
3. **Reviewer agreement.** Are there changesets reviewed more than once, and do reviewers
   agree? Without this there is no inter-rater estimate.
4. **Overlap** between the OSMCha set and Corpus A ids (should be near zero given the
   eras, but must be checked, not assumed).
5. **Class-balance drift by year**, so the training window is chosen deliberately.

### Estimated cost of the corrected fetch

Rates are **measured**; totals are *estimated* from them.

| Job | Volume | Rate (measured) | Wall clock |
|---|---|---|---|
| OSMCha labels, wide window | ~50k–100k rows | ~25 rows/s windowed | ~1 h |
| Changeset metadata for them | ~50k | 3/s | ~4.6 h |
| Contents (osmChange) | ~50k | 3/s | ~4.6 h |
| Random unreviewed negatives | ~20k | 3/s | ~1.9 h |
| User accounts + histories | ~30k users | 3/s | ~2.8 h |
| Geometry `adiff` on a subset | 6,000 | **3.17/min** | **~31 h** |
| Finish Corpus A contents | 10,021 | 3/s | ~0.9 h |

Everything except geometry fits in a day. Geometry remains the binding constraint and the
id-sampling rule in `docs/planning/GATE1B_DECISION.md` still applies — sampled by hash of the
changeset id, never by size or user, because size correlates with the label.

---

## 5. Rules the corrected fetch must follow

1. **Every fetch records its own coverage.** A store that cannot say what it failed to get
   is a store that cannot support a coverage claim in the paper.
2. **Reviewed-clean and random-unreviewed are different classes and never merged.**
3. **Point-in-time stays point-in-time.** User history for a 2026-03 changeset uses only
   pre-2026-03 activity, whatever corpus it belongs to.
4. **No `uid` or username in any published artefact.** Salted hash only.
5. **The label window is stated in the paper as exact dates**, because OSMCha review
   density changes over time and a reader cannot otherwise reproduce the corpus.
6. **Nothing is fetched from the cluster.** The rate limiter assumes one machine and the
   cluster shares an institutional IP.

---

## 6. Honest one-paragraph summary

> The data fetched so far is genuine and verified: 18,276 changeset metadata records and
> 8,255 osmChange documents retrieved from the OpenStreetMap API with zero failures, of
> which a 200-document sample parsed without error. No data was synthesised. However,
> coverage is incomplete — geometry enrichment (11 of ~6,000), user histories (20 of 8,768)
> and account records (0) are effectively unstarted, the OSMCha label harvest covers only
> 2026-06-01 onward because an earlier pagination failure truncated it, and the feature
> store has not been built. Two authenticity issues are structural rather than incidental:
> the OSMCha negative class consists of reviewed-and-cleared changesets rather than random
> ones, which changes the task being learned, and the historical labels are revert-derived
> proxies whose noise rate has not yet been measured.
