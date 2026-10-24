# MapGuard visual demo — plan against CHECKLIST_v2

Source of requirements: `osm-mapguard/docs/planning/CHECKLIST_v2.md`, Phases 3–5 and the
"V2 operating rules". The checklist schedules the demo for 2026-10-19..21.

Starting point: the live map in `mapguard-demo/live/` (Leaflet + OpenStreetMap tiles, a Python
server, test-set mode and live mode).

---

## 1. Where we stand

✅ done · 🟡 partly · ❌ missing

### Product shape
| Requirement | Status | What exists / what's missing |
|---|---|---|
| Small dashboard + thin Python endpoint | ✅ | Leaflet page + stdlib server |
| Replay mode: pick a saved changeset id and get a known result | 🟡 | Test-set mode replays the real round-4 scores for 3,607 changesets. There is no "enter a changeset id" box and no scripted scenarios. |
| Live mode with loading / rate-limit / timeout / stale states | 🟡 | Polls every minute through the rate limiter. No latency shown, no stale warning, no "enter an id" input. |
| First screen answers what, where, how suspicious, why | 🟡 | Score, map and tokens are there, but spread over a long side panel |

### Map
| Requirement | Status | What exists / what's missing |
|---|---|---|
| Before/after geometry with a legend: created / modified / deleted / top edit | 🟡 | Node moves (old → new) and new ways are drawn. The old shape of modified ways isn't drawn, and the legend has no action types. |
| Zoom-to-change, fit-to-bounds | ✅ | |
| Object-type filters, before/after toggle | ❌ | |
| Select an edit ↔ map highlight, both ways | 🟡 | List → map works; map → list doesn't |
| OSM base map, Google optional | ✅ | OSM only (Google deferred, as the checklist allows) |
| Attribution in UI and screenshots | ✅ | |

### Dashboard panels
| Requirement | Status | What exists / what's missing |
|---|---|---|
| Decision card: score, threshold, calibration note, model version, timestamp | 🟡 | Score and threshold only |
| Change summary: id, editor, operation mix, counts, bbox, fetch status | 🟡 | No operation mix, no fetch status |
| Why flagged: geometry/text/user/scale bars from real model output | 🟡 | Live mode shows token pushes and the two sub-scores. Signal bars are missing. |
| Suspicious edits ranked by weight | 🟡 | Edits are listed in commit order with attention bars, not ranked, and without geometry deltas |
| Text diff with redaction rules | 🟡 | Old → new is shown and truncated, with no rule for hostile text |
| History / evaluation panel | 🟡 | Test mode shows the revert label. Live mode has none. |
| System status: source, mode, request and inference latency, failure reason | 🟡 | One status line |

### Phase 4 evidence
| Requirement | Status | What exists / what's missing |
|---|---|---|
| Same function for CLI and dashboard (`score_changeset(cid)`) | ❌ | The server scores inline; there is no CLI |
| Golden-fixture tests | ❌ | |
| Offline mode: no network, committed fixtures | ❌ | Needs map tiles and the API |
| Three scripted scenarios | ❌ | Candidates chosen below |
| Recorded walkthrough + evidence bundle | ❌ | |
| "Research prototype" label, FP/FN limits visible | 🟡 | A note exists; there is no visible label |

### The one real gap
The checklist's acceptance rule is that the demo "scores the same input through the same
inference path as the research results".
- **Test-set mode meets it:** it shows the saved research scores.
- **Live mode does not:** it uses a re-trained CPU scorer, because the trained network weights
  aren't in the repo.

The weights exist on the Altair cluster, in
`<BASE>/rounds/r4_2026-10-13_altair_h100_seeds/app/experiments/runs/`.
You need the `stage2_supervised/*_ovid_R1b_s2026090*` folders, plus `stack/` and `baselines/`.
Each folder holds `model.weights.h5`, `model_config.json` and `normaliser.json`.

Downloading them closes this gap. Until then the demo must say so on screen.

---

## 2. Plan, by priority

### P0 — needed for a presentable demo (about half a day) — ✅ built 2026-10-21

Tested in the browser with all 5 scenarios and live mode:
- Code: `live/explain.py` computes the signal bars and edit push. `live/server.py` adds model
  info, timings, operation mix and bounding box. `live/static/index.html` holds the page.
- Test mode has no "old → new" lines: the release stores only the new node position, so the
  move distance appears in the edit table instead. Ways are drawn in live mode only.
- The signal bars come from the CPU scorer, including in test mode, and the page says so. Near a
  score of 1.0 every bar is small because the signals are redundant (e.g. Pokémon GO: author
  +0.11, everything else ≈ 0).

1. **Scenario bar.** Add one-click scripted scenarios at the top of the page, each from the real
   test set with its real round-4 score:

   | Scenario | Changeset | Score | What it shows |
   |---|---|---|---|
   | Clean ordinary edit | 73656923 "added a service road" | 0.001 | what "clean" looks like |
   | Text / Pokémon GO | 61155395 "i need this for pokemon go pls" | 1.000 | top edit = a park named "w" |
   | Geometry | 25818789 (no comment, 8 edits, a node moved 762 m) | 0.968 | before → after node move on the map |
   | Mass deletion | 22578095 (232 of 235 edits are deletions) | 0.989 | scale signal |
   | Honest mistake | 56564752 "Added missing footpath" | 0.987, never reverted | false alarm: a newcomer looks like vandalism |

2. **Decision card on top**, at a glance:
   - verdict pill, score gauge and threshold;
   - which model produced the score, with its version or run id;
   - label: "research prototype; on real traffic ~4% of flags are vandalism".
3. **"Why flagged" bars.** For each changeset, run one occlusion pass per signal group on the
   scorer: author / text / edit tokens / counts & scale / geometry. Show it as 5 bars. These
   come from real model outputs, not hand-written text.
4. **Ranked suspicious edits.** Sort edits by attention (test mode) or token push (live mode).
   Show action, object type and geometry delta. Clicking a row flashes its feature on the map;
   clicking a map feature selects its row.
5. **Map legend and colours by action:** created green-ink, modified amber, deleted dashed red,
   old position grey, top edit with an amber halo.
6. **System status strip:** mode (replay / live), data source, last poll time, request latency,
   scoring latency, and a stale warning if the last poll is older than 3 minutes.

### P1 — matches the checklist's evidence rules (about half a day) — ✅ items 7–9 built 2026-10-21

- `live/core.py` (`Engine.score_changeset`) is the one scoring path. Both `live/score.py` (CLI)
  and the dashboard's "Score any changeset" box (`/api/score/<cid>`) call it. A real changeset
  scored the same through both, and the networked test checks this.
- `server.py --offline` serves only committed files (`fixtures/`, vendored Leaflet) and makes no
  network requests, checked in the browser's network log. It draws a grid base map in place of
  tiles. If the release is missing, the server starts offline automatically.
- `tests/test_golden.py` has 6 tests. 5 run offline: scenario replay, the offline server, live
  results re-scored from recorded feature rows, no author ids in the fixtures, and fixtures vs
  release. The 6th, with `MAPGUARD_NETWORK_TESTS=1`, checks CLI == dashboard on a live changeset.
- Item 10 (round-4 weights in live mode) still waits on the files from the cluster.

7. **`score_changeset(cid)`, shared by the CLI and the server**, plus a "score any changeset id"
   box in live mode. Command: `python live/score.py 61155395` prints the same JSON the page shows.
8. **Offline replay.** Commit the 5 scenarios plus about 20 examples as fixtures with their
   geometry and saved scores. With `--offline`, the server makes no network calls and the map
   falls back to a plain grid background if tiles fail.
9. **Golden-fixture test.** For each scenario: the displayed score equals the saved one, the
   top-ranked edit is the same, and the geometry point count matches.
10. **Real model in live mode.** Once the weights are pulled from the cluster, load them
    (TensorFlow, Keras 2 config), so live mode uses the round-4 deep model and stack. Keep the
    CPU scorer as a labelled fallback.

### P2 — presentation and handoff (2–3 hours)

11. **Before/after toggle** for modified ways: fetch the previous way's node coordinates, or
    use Overpass adiff for the selected changeset only.
12. **Text-diff safety:** truncate to 80 characters, collapse repeated characters, and add a
    "show raw text" button. Offensive text stays hidden until clicked.
13. **Recorded walkthrough (2–3 minutes):** scenario 1 → 2 → 3 → mistake → live feed. Keep it
    as the backup if the network fails on the day.
14. **Evidence bundle:** screenshots, fixture ids, model ids, dependency versions, and the
    one-line launch command.
15. **Layout checks:** narrow screen, empty feed, slow API, malformed changeset, missing model
    file.

---

## 3. Demo script (5 minutes)

1. **Test-set map, world view (30 s).** 3,607 held-out changesets: red = flagged, teal = clean.
   Show the filter chips: 1,536 caught, 1,430 cleared, 382 false alarms, 258 missed.
2. **Scenario: clean service road (30 s).** Low score, ordinary tokens.
3. **Scenario: Pokémon GO (60 s).** Score 1.000. The top-attended edit is `+leisure=park +name "w"`.
   Point at the "why flagged" bars.
4. **Scenario: geometry or mass deletion (45 s).** Map shows the node moved 762 m, or 232
   deletions. The scale signal fires.
5. **Scenario: honest mistake (45 s).** A newcomer adding a real footpath gets flagged. This is
   why MapGuard ranks a review queue rather than reverting automatically.
6. **Live mode (60 s).** Edits from the last minutes appear. Open the top-scored one and show
   the status strip (latency, last poll).
7. **Close (30 s).** "On real traffic about 4% of flags would be vandalism; recall 86% on
   reverted edits; results in the paper."

---

## 4. Done means
- [ ] `server.py --offline` runs the full scenario script with the network cable unplugged.
- [ ] `score.py <cid>` and the dashboard show the same number for every scenario.
- [ ] The golden-fixture test passes.
- [ ] Screenshots show no usernames or uids, and OSM attribution is visible.
- [ ] A recorded walkthrough exists.
- [ ] The page says which model scored each changeset (research round-4 vs CPU live scorer).
