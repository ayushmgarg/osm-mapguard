# Gate 1b decision — geometry coverage route

**Decided 2026-09-08. Verdict from `results/gate1b_overpass_capacity.json`: `SUBSET_OR_MIRROR`.**

## What was measured

30 minutes of continuous per-changeset `adiff` against the public `overpass-api.de`,
reading changeset metadata from the local store so the gate measured Overpass rather than
our own contention with the corpus fetch.

| | |
|---|---|
| Queries completed | 95 |
| Rate | **3.17 / minute** |
| Status codes | 91× 200, 3× 504, 1× 429 |
| Latency | median **10.6 s**, p90 42.3 s, max 72.1 s |
| Error rate | 4.2% |
| **Projected for all 18,276** | **96.2 hours** |

Four days of continuous querying against a volunteer-run service, for one corpus, is not
defensible — and the p90 of 42 s says we would be sitting in the queue behind everyone
else while doing it.

## The decision: id-sampled subset, on the public instance

**Enrich a fixed 6,000-changeset subset of Corpus A, sampled by changeset id.**

- 6,000 ÷ 3.17/min ≈ **31.5 hours**, splittable across the team's four machines → under a
  day of wall clock each.
- Rejected a mirror instance: the well-known public mirrors carry the same attic data with
  no throughput guarantee, and adding a second stranger's service to the critical path of a
  four-week project buys uncertainty, not speed.
- Rejected a self-hosted Overpass: building the attic database needs the full history
  planet (162 GB) and days of import. That is the whole project's compute budget.

### The sampling rule

```
selected = { cid : sha256(str(cid)) mod 1000 < 328 }      # ≈ 32.8% → ~6,000 of 18,276
```

Deterministic, reproducible from the id alone, needs no stored list, and **independent of
everything the model reads**.

**Why by id and nothing else.** Sampling by changeset size would put the biggest
changesets in the geometry-enriched half, and size correlates with the label — mass
vandalism and broken imports *are* large. Sampling by user would do the same through
account behaviour. Either would manufacture a correlation between "has geometry" and
"is damaging", and the geometry branch would then appear to work for a reason that has
nothing to do with geometry.

Changeset ids are assigned sequentially by the OSM API in commit order, so a hash of the
id is independent of content, author and size. It does correlate with *time*, which is why
the hash is taken rather than a range: a contiguous id range would have selected a single
period of OSM history.

## What it costs

- **Wider confidence intervals** on every geometry-branch comparison. ~6,000 changesets
  with a ~50/50 class balance still gives ~1,200 in a 20% test fold — enough for a PR-AUC
  point estimate with a usable bootstrap interval, not enough to split three ways by
  vandalism type and keep tight bounds.
- **The ablation is between-model, not between-corpus.** Every ablation row that involves
  B1 must be trained and evaluated on the *same* enriched subset, or "geometry helps" would
  really be measuring "this subset is easier". The `+geometry` and `Ovid-repro` rows are
  therefore both run on the subset as well as on the full corpus, and both numbers are
  reported.
- Corpus D geometry is unaffected: it is a much smaller set and gets full enrichment.

## Honest statement for the paper

> Geometry enrichment was applied to a deterministic 32.8% subset of the labelled corpus
> (n ≈ 6,000), sampled by a hash of the changeset id. Public Overpass throughput was
> measured at 3.17 queries/minute over a 30-minute window, projecting to 96 hours for full
> coverage. The sampling rule is independent of changeset size, author and content;
> ablations involving the geometry branch are trained and evaluated on the enriched subset
> only, with the corresponding no-geometry baselines recomputed on the same subset.

## Revisit if

- geometry turns out to carry the result and the intervals are too wide to claim it, or
- the team gets access to a self-hosted or institutional Overpass instance.

Neither changes the code — only the `--limit` on `scripts/fetch_corpus_a_geometry.py` and
the constant in `mapguard.config.GEOMETRY_SUBSET_PERMILLE`.
