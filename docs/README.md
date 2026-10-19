# Documentation

New here? Start with the diagram and stage table in the [top-level README](../README.md).

| Folder | Document | What it covers |
|---|---|---|
| `architecture/` | [README.md](architecture/README.md), [mapguard-pipeline.html](architecture/mapguard-pipeline.html) | the end-to-end pipeline diagram, its source JSON, and which code each box maps to |
| `model/` | [MODEL_V2.md](model/MODEL_V2.md) | why the first model plateaued; who/what/text signals; author concentration; multiple-instance pooling |
| | [MODEL_V3.md](model/MODEL_V3.md) | text embeddings, transformer blocks, stacking; what each round decided and why |
| | [mapguard-architecture.html](model/mapguard-architecture.html) | architecture page |
| `data/` | [DATA_SOURCES.md](data/DATA_SOURCES.md) | every data source, what it gives, what was rejected and why |
| | [DATASET_CARD.md](data/DATASET_CARD.md) | the released dataset: contents, labels, privacy, licence |
| | [DATA_FOUNDATION.md](data/DATA_FOUNDATION.md) | label definitions and dataset decisions |
| | [DATA_AUDIT.md](data/DATA_AUDIT.md) | leakage and coverage audits of the feature store |
| `running/` | [RUNBOOK.md](running/RUNBOOK.md) | fetch → build → bundle → train, step by step |
| | [RUN_ALTAIR.md](running/RUN_ALTAIR.md) | the GPU cluster (Altair / PBS) and its constraints; see also [../altair/README.md](../altair/README.md) |
| `planning/` | [PROJECT_PLAN.md](planning/PROJECT_PLAN.md) | the original design |
| | [PLAN_REVIEW.md](planning/PLAN_REVIEW.md) | the review that corrected the plan, with the probes behind each correction |
| | [GATE1B_DECISION.md](planning/GATE1B_DECISION.md) | the geometry data-source decision |
| | [CHECKLIST.md](planning/CHECKLIST.md), [CHECKLIST_v2.md](planning/CHECKLIST_v2.md) | build checklists, ticked as work was verified |

Results live next to the runs that produced them: [`../experiments/rounds/`](../experiments/rounds/)
(one folder per training round, each with a README) and [`../results/`](../results/).
