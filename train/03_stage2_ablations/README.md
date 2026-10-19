# 03 · Stage 2 — ablation grid

Plan stage: **Stage 2 ablations** (PROJECT_PLAN §11). The grid started at 7 rows × 2 seeds
(§13) and has grown to the 15 rows below as the model went from v1 to v3. **Cluster.**

| File | Does |
|---|---|
| `config.json` | shared settings, seeds, which rows to run |
| `rows/*.json` | one file per table row: its name, purpose and config overrides |
| `run_ablations.py` | trains every row × seed, then writes the table |

## The rows

| Row | Answers |
|---|---|
| `base_edits_tabular` | edit numbers + changeset table only - the floor every branch is measured against |
| `plus_geometry` | does geometry add anything on its own? |
| `plus_text` | comment + tag text - the strongest branch in round 1 |
| `plus_sequences` | edit order + point-in-time user history |
| `plus_edit_meaning` | v2: what each edit did (tokens) and its old->new text - the missing WHAT |
| `full` | all six branches, MIL pooling, author-balanced - the v2 headline |
| `full_meanpool` | v2 with round-1 mean pooling - does multiple-instance pooling help? |
| `full_unweighted` | v2 without author-balanced weights - does weighting help the one-per-author score? |
| `full_pretrained` | v2 with the Stage 1 edit encoder - skipped unless a pretrained encoder is given |
| `full_no_residual` | v2.1 without the residual around edit self-attention - does keeping per-edit identity let MIL pooling work? |
| `full_text_emb` | v3: + pretrained multilingual sentence embeddings of comment and tag text (frozen) next to the char-CNN |
| `full_context` | v3: + object context numerics (what the edited objects were before the edit) in the tabular branch |
| `full_2layers` | v3: two transformer encoder blocks over edits instead of one attention layer |
| `full_v3` | v3: text embeddings + object context + two transformer blocks together |
| `full_2layers_text_emb` | round-4 candidate: two transformer blocks + pretrained text embeddings (no object context) |

Each row's exact overrides are in its file under [`rows/`](rows/). Pick a subset with
`--ablation-rows` on `run_pipeline.py`. `full_pretrained` runs only after stage 04; the
embedding and context rows run only when their side tables exist.

**The edit trunk is in every row.** Attention over edits is how the model fuses anything, so
"B1 off" removes the geometry *columns* from edits and tabular features - it does not remove
edits. That is also why the base row is named for what it contains.

## Rules

- **Seeds change initialisation, not the split.** Every row trains on the same frozen split.
  Otherwise a row can look better by drawing an easier test fold.
- **Every row uses the same training loop** as stage 02. Rows differ only by their row file.
- **Attribution is off** here to save time; run `stage2_evaluate.py` on the rows you report.
- Two seeds give a mean and a spread, not a significance test. Say so in the paper.

```bash
python train/03_stage2_ablations/run_ablations.py --dry-run
python train/03_stage2_ablations/run_ablations.py
```

To add a row, add a file to `rows/`. Nothing else changes.
