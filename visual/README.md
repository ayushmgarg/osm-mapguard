# MapGuard demo: a review console on your laptop

A local web app that shows MapGuard at work.

- **Scenarios and the test set.** Round 4's real outputs, replayed: every model's score, which edits the deep model attended to, why the changeset was flagged, and the edit on the map next to satellite imagery.
- **Score a changeset.** Fetch any changeset from OpenStreetMap or the OSM sandbox *now* and score it on the CPU.
- **Live feed.** The newest real changesets, scored as they close.

![The fake park scenario](screenshots/pokemon-park.png)

## Why it runs on the laptop

A static site cannot call the OSM API and run the model, so a small Python server does both (`server.py`, standard library only). Nothing needs TensorFlow:

- **Replayed changesets** show the scores round 4 stored, read from its run folders.
- **Live changesets** are scored by round 4's tree model ("hybrid"). Round 4 saved predictions, not trees, so it is refit on the CPU from the released feature store, with the same code, split and seed. Its test PR-AUC matches round 4's within 0.002 (ovid 0.898 vs 0.899; ovid_modern 0.934 vs 0.936).
- **Features match training exactly.** A live changeset is featurised by the training feature builder itself (`src/mapguard/features/build_store.py`). Re-fetching stored 2021–26 changesets reproduces their stored features and scores exactly (`tests/test_demo.py`, network test).

## Run it

```bash
uv venv --python 3.12 visual/.venv
```

```bash
VIRTUAL_ENV=visual/.venv uv pip install -r visual/requirements.txt
```

macOS only: xgboost needs the OpenMP runtime.

```bash
brew install libomp
```

Download the dataset release (80 MB, pseudonymised feature store) outside the repository:

```bash
mkdir -p ~/osm-data-mac/release && gh release download v1.1.0 -R ayushmgarg/osm-mapguard -p 'mapguard-dataset-v1.1.0.zip' -D ~/osm-data-mac/release
```

```bash
unzip -q ~/osm-data-mac/release/mapguard-dataset-v1.1.0.zip -d ~/osm-data-mac/release/v1.1.0
```

Refit the tree models (about 20 s), then export round 4's outputs (about 1 minute, including map geometry for the scenarios):

```bash
visual/.venv/bin/python visual/tree_model.py --task ovid --task ovid_modern
```

```bash
visual/.venv/bin/python visual/export_demo.py
```

Start the server and open http://127.0.0.1:8765:

```bash
visual/.venv/bin/python visual/server.py
```

Set `MAPGUARD_DATA` if the release is not under `~/osm-data-mac`.

## The live edit, done safely

Never make a test edit on the real OpenStreetMap. It is vandalism: volunteers have to revert it, and the account can be blocked. Use OSM's official sandbox, a separate test copy of the map.

1. Create a free account at https://master.apis.dev.openstreetmap.org. It is separate from any real OSM account.
2. Before the demo, open the sandbox editor (Edit, then iD), draw something MapGuard should dislike, for example a park named "free pokestop here" over a street, and save it with a comment.
3. On the sandbox, open your profile, then Edits, and copy the changeset number.
4. In the demo, choose **Score a changeset**, keep **OSM sandbox** selected, paste the number and press **Score**.

A brand-new sandbox account has no history, so the "who" signal counts against it, as it would for a brand-new real account.

## What is on screen

| Panel | Source |
|---|---|
| Left map: before (dashed amber) and after (magenta) of every edit; the selected edit in blue | OpenStreetMap tiles, and geometry fetched from the OSM API for that changeset |
| Right map: the same spot from above, with the new shapes outlined | Esri World Imagery (display only, attributed) |
| Open in Google Maps | A link to the same spot. Nothing is copied from Google into OSM. |
| Decision: score, threshold, flag or not; the ground truth for replayed changesets | the best round-4 model for the task (stack + embeddings on ovid, ensemble on ovid_modern) |
| Why: who / what / text bars, and the edit patterns that pushed hardest | the tree model's SHAP contributions and its edit-token model |
| Every model, same changeset | all eight round-4 models, mean of 3 seeds, with each model's validation-chosen threshold |
| Which edits look suspicious | replay: the deep model's attention; live: the edit-pattern model per edit (the deep model is not run live) |

Usernames are never shown. Replayed changesets come from the pseudonymised release.

## Files

| File | Job |
|---|---|
| `tree_model.py` | refit the hybrid tree model; score rows; explain a score |
| `osm_live.py` | fetch one changeset (OSM or sandbox) and build its features and map geometry |
| `export_demo.py` | round 4 run folders -> `data/<task>/` (scores, attention, signals), scenarios, geometry |
| `server.py` | local API and static files, bound to 127.0.0.1 |
| `web/` | the page (Leaflet 1.9.4 from cdnjs, no build step) |
| `tests/test_demo.py` | the served score equals round 4's stored output, for every model |
| `screenshots.sh` | regenerate `screenshots/` with headless Chrome while the server runs |

```bash
visual/.venv/bin/python -m pytest visual/tests -m "not network"
```

## Known limits

- **Live geometry features** use the previous-version route only, not Overpass adiff (too slow live). This matches about two-thirds of training rows. `geom_frac_adiff` is 0 live.
- **Way shapes** use today's positions for nodes the changeset did not touch.
- **The live feed** polls OSM once a minute and only while the page is open. It scores at most 5 new changesets per poll.
- **A hosted, replay-only copy** (GitHub Pages) is not built yet. It would need the replay data exported as static JSON.
