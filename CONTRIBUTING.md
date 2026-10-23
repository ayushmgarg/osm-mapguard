# Contributing

Contributions are welcome: bug reports, fixes, tests, documentation and new experiments.

## How to contribute

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Fork the repository and work on a branch: `feat/…`, `fix/…` or `docs/…`.
3. Use [Conventional Commits](https://www.conventionalcommits.org) (`feat(data): …`,
   `fix(features): …`, `docs: …`, `test: …`), one logical change per commit.
4. Run the tests, then open a pull request that says what changed and why.

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q -m "not network"
```

## Rules that keep results valid

These have tests. A change that breaks them invalidates results, so keep them passing.

1. **Author features are point-in-time.** Use only changesets created strictly before the one
   being scored. Fields read from the account as it is today (such as account age) leak the
   label and are not used.
2. **Splits are user-disjoint.** No author appears in more than one of train, validation and
   test. The frozen splits in `experiments/splits/` are not regenerated for a new run.
3. **Pretraining excludes the evaluation set**, by changeset id and by author.
4. **Numbers come from files.** A reported metric must trace to a file under `results/` or
   `experiments/rounds/`. Smoke-test (`--quick`) numbers are never reported.

## Talking to OpenStreetMap

The OSM API and Overpass are run by volunteers.

- Send every request through `mapguard.data.http.PoliteSession`: one global rate limit and a
  real User-Agent (`mapguard.config.USER_AGENT`).
- HTTP 429 and 509 mean slow down. Do not work around the back-off.
- Cache to disk so a rerun costs nothing.
- Never make test edits on the real map. Use the OSM development sandbox
  (https://master.apis.dev.openstreetmap.org).

## What does not go in the repository

- **Bulk data.** It lives outside the repo under `MAPGUARD_DATA`. The Ovid label file is the
  one tracked dataset.
- **Secrets.** Tokens go in a git-ignored `.env`; copy `.env.example` to start.
- **Model weights.** They are published as GitHub releases, not committed.

## Licence of contributions

By submitting a contribution you confirm that you have the right to submit it, and you agree
that it is licensed under this repository's [LICENSE](LICENSE). Code derived from
OpenStreetMap data stays subject to the ODbL as described there.
