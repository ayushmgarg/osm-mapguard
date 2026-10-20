"""Model-level numbers for the demo page, read from the repo's round-4 analysis and runs."""
import json, sys
from pathlib import Path
import numpy as np

REPO, BUILD = Path(sys.argv[1]), Path(sys.argv[2])
R4 = REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds"
runs = R4 / "raw/experiments/runs"
seeds = json.loads((R4 / "analysis/seeds.json").read_text())
abl = json.loads((R4 / "analysis/ablations_3seeds.json").read_text())
base = json.loads(next((runs / "baselines").glob("*_ovid_R1b_s20260904/results.json")).read_text())
deep = json.loads(next((runs / "stage2_supervised").glob("*_ovid_R1b_s20260904/results.json")).read_text())

tasks = {t: {m: {"pr_auc": v["models"][m]["pr_auc"][0], "one_per_author": v["models"][m].get("one_per_author", [None])[0]}
             for m in v["models"]} | {"_n_test": v["n_test"], "_vs_hybrid": v["vs_hybrid_seed_averaged"]}
         for t, v in seeds["tasks"].items()}

z = np.load(BUILD / "deep_attention.npz")
S = json.loads((BUILD / "scores.json").read_text())
y = dict(zip(S["cids"], S["y"]))
conc = {0: [], 1: []}
for c, a, n in zip(z["cids"], z["att"], z["n"]):
    if n >= 5:
        conc[y[int(c)]].append(float(a[:n].max() * n))
out = {
    "tasks": tasks,
    "tokens": {"damaging": base["models"]["hybrid_emb"]["most_damaging_tokens"],
               "benign": base["models"]["hybrid_emb"]["most_benign_tokens"]},
    "attribution": deep["attribution"]["branches"],
    "ablations": abl,
    "attention_concentration": {str(k): {"median": round(float(np.median(v)), 2), "n": len(v),
                                         "values": [round(x, 2) for x in sorted(v)]} for k, v in conc.items()},
}
(BUILD / "summary.json").write_text(json.dumps(out))
print(json.dumps({k: out["attention_concentration"][k]["median"] for k in ("0", "1")}), json.dumps(abl)[:600])
