"""Seed-averaged test scores (ovid task, round 4) for the demo page. Reads the repo's round-4 runs only."""
import glob, json, sys
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path[:0] = [str(REPO / "train"), str(REPO / "src")]
from common.scoring import best_f1_threshold  # noqa: E402

runs = REPO / "experiments/rounds/r4_2026-10-13_altair_h100_seeds/raw/experiments/runs"

def avg(paths):
    zs = [np.load(p) for p in paths]
    cids = zs[0]["test_cids"]
    for z in zs:
        assert np.array_equal(z["test_cids"], cids) and np.array_equal(z["val_cids"], zs[0]["val_cids"])
    return (cids, zs[0]["test_y"].astype(int), np.mean([z["test_p"] for z in zs], 0),
            zs[0]["val_y"].astype(int), np.mean([z["val_p"] for z in zs], 0), zs)

models = {
    "stack_emb": sorted(glob.glob(str(runs / "stack/*_ovid_R1b_stack_s2026090*/predictions_stack_emb.npz"))),
    "hybrid": sorted(glob.glob(str(runs / "baselines/*_ovid_R1b_s2026090*/predictions_hybrid.npz"))),
    "deep": sorted(glob.glob(str(runs / "stage2_supervised/*_ovid_R1b_s2026090*/predictions.npz"))),
}
out = {"models": {}}
ref = None
for name, paths in models.items():
    assert len(paths) == 3, (name, paths)
    cids, y, p, vy, vp, zs = avg(paths)
    if ref is None:
        ref = cids
        out["cids"] = [int(c) for c in cids]
        out["y"] = [int(v) for v in y]
    assert np.array_equal(cids, ref)
    t, f1 = best_f1_threshold(vy, vp)
    out["models"][name] = {"p": [round(float(v), 4) for v in p], "threshold": round(t, 4),
                           "val_f1": round(f1, 4), "runs": [Path(x).parent.name for x in paths]}
    if name == "deep":
        att = np.mean([z["test_attention"].astype("float32") for z in zs], 0)
        n = zs[0]["test_n_edits"].astype(int)
        np.savez_compressed(OUT / "deep_attention.npz", cids=cids, att=att, n=n)
(OUT / "scores.json").write_text(json.dumps(out))
print({k: (v["threshold"], v["val_f1"]) for k, v in out["models"].items()}, len(out["cids"]))
