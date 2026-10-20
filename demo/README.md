# MapGuard live map

A local website with a real OpenStreetMap map and two modes:

- **Test set.** The 3,607 held-out changesets from Ovid's benchmark, plotted where they
  happened. Each is coloured by the repo's real round-4 scores (stack + embeddings, mean of
  3 seeds). You can:
  - filter to caught, cleared, false alarms or missed;
  - colour by the model's verdict or by the truth;
  - click one to see its edits, the deep model's attention, and its nodes on the map.
- **Live now.** Every minute the server takes the newest closed changesets from the OSM API.
  For a few of them it rebuilds MapGuard's features with the repo's own feature code, scores
  them, and drops them onto the map.
  - Clicking one shows its edits, its geometry, and which tokens pushed the score up or down.
  - The default threshold flags 5% of random edits and catches about 60% of edits that were
    later reverted (both measured on the held-out set).

## Run

```bash
.venv\Scripts\python live\server.py
```

Then open http://localhost:8765. Options:
- `--per-minute 5`: how many live changesets to score each minute;
- `--no-live`: no background polling (the "Score any changeset" box still works);
- `--offline`: **no network at all**. It uses the committed `fixtures/` (every test point, the 5
  demo scenarios plus 23 more in full detail, and 8 recorded live results) and draws a grid in
  place of map tiles. This is the safe fallback if the room's network fails. If the dataset
  release isn't under `data/`, the server starts in this mode automatically.

At the top of the page, **demo scenarios** open five real test changesets with one click: clean
edit, Pokémon GO text, geometry, mass deletion, and an honest mistake.

## Score one changeset from the command line

```bash
.venv\Scripts\python live\score.py 190804354
.venv\Scripts\python live\score.py 190804354 --json
```

This is the same function (`live/core.py`, `Engine.score_changeset`) that the dashboard's
"Score any changeset" box calls, so both give the same number.

## Tests

```bash
.venv\Scripts\python -m pytest -q tests
```

Golden-fixture tests check that the dashboard reproduces each scenario's score, top-ranked edit,
map points and signal bars, and that recorded live results re-score identically from their
feature rows. They run offline. Set `MAPGUARD_NETWORK_TESTS=1` to also check that the CLI and
the dashboard agree on a live changeset. After changing the model or the data, re-record with
`.venv\Scripts\python live\build_fixtures.py`.

## Rebuild from scratch

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install "numpy>=1.23,<2.2" pandas pyarrow scikit-learn xgboost requests
```

Then:
- download `mapguard-dataset-v1.1.0.zip` from the osm-mapguard GitHub releases and unzip it to
  `data/release_v1.1.0/extracted/`;
- run `build/extract_scores.py .. build` and
  `build/extract_summary.py .. build` (the repository root is `..`);
- run `live/train_scorer.py`, then `live/calibrate_live.py`.

## What the live scorer is

The live scorer is a CPU version of the repo's hybrid model, trained on release v1.1.0. It has:
- logistic regression on edit tokens and editor;
- logistic regression on character n-grams of the comment and tag text;
- XGBoost on the changeset and point-in-time author features plus those two scores.

It leaves out the pretrained text embeddings and the deep network, because the trained
weights are not published. On a user-disjoint held-out set it scores PR-AUC 0.884 overall
(Ovid-era 0.840, 2021–26 reverts 0.922); details are in `live/model/scorer_report.json`.

Live edits have no labels. Red means "worth a reviewer's look", not proven vandalism.

Map data © OpenStreetMap contributors (ODbL). Requests go through the repo's rate-limited
client, at about 1.5 requests per second at most.

`index.html` in this folder is the earlier static page (also published as a private Claude
artifact).
