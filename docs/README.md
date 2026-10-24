# Documentation

New here? Start with the diagram and stage table in the [top-level README](../README.md).

| Folder | Document | What it covers |
|---|---|---|
| `architecture/` | [README.md](architecture/README.md), [mapguard-pipeline.html](architecture/mapguard-pipeline.html) | the end-to-end pipeline diagram, its source JSON, and which code each box maps to |
| `model/` | [MODEL_V2.md](model/MODEL_V2.md) | why the first model plateaued; who/what/text signals; author concentration; multiple-instance pooling |
| | [MODEL_V3.md](model/MODEL_V3.md) | text embeddings, transformer blocks, stacking; what each round decided and why |
| `data/` | [DATA_SOURCES.md](data/DATA_SOURCES.md) | every data source, what it gives, what was rejected and why |
| | [DATASET_CARD.md](data/DATASET_CARD.md) | the released dataset: contents, labels, privacy, licence |
| `running/` | [RUNBOOK.md](running/RUNBOOK.md) | fetch → build → bundle → train, step by step |
| | [../altair/README.md](../altair/README.md) | running a training round on a GPU machine |
| `archive/` | [PROJECT_PLAN.md](archive/PROJECT_PLAN.md) | the original design |
| | [PLAN_REVIEW.md](archive/PLAN_REVIEW.md) | the review that corrected the plan, with the probes behind each correction |
| | [GATE1B_DECISION.md](archive/GATE1B_DECISION.md) | the geometry data-source decision |
| | [CHECKLIST.md](archive/CHECKLIST.md), [CHECKLIST_v2.md](archive/CHECKLIST_v2.md) | build checklists, ticked as work was verified |
| | [DATA_FOUNDATION.md](archive/DATA_FOUNDATION.md) | label definitions and dataset decisions |
| | [DATA_AUDIT.md](archive/DATA_AUDIT.md) | leakage and coverage audits of the feature store |
| | [mapguard-architecture.html](archive/mapguard-architecture.html) | the first architecture page |
| | [RUN_ALTAIR.md](../archive/cluster/RUN_ALTAIR.md) | the notebook-cluster run guide (Altair / PBS) |

Results live next to the runs that produced them: [`../experiments/rounds/`](../experiments/rounds/)
(one folder per training round, each with a README) and [`../results/`](../results/).
