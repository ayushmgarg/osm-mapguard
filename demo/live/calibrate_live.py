"""Add a live default threshold to scorer_report.json: the score that flags 5% of random edits
(corpus N, presumed clean) in the held-out user-disjoint test fold, and what it catches there."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import train_scorer as ts
from mapguard.evaluation.splits import user_disjoint_split
sys.modules["__main__"].Scorer = ts.Scorer
cs = pd.read_parquet(ts.STORE / "changesets.parquet")
cs = cs[~((cs["corpus"] == "M") & (cs["label"] == 0))].reset_index(drop=True)
y = cs["label"].to_numpy(int)
te = user_disjoint_split([None if pd.isna(u) else int(u) for u in cs["uid"]], y, seed=ts.SEED).test
sc = pickle.load(open(HERE / "model/scorer.pkl", "rb"))
p = sc.score_frame(cs[te]); c = cs["corpus"].to_numpy()[te]; yt = y[te]
thr = float(np.quantile(p[c == "N"], 0.95))
rep = json.loads((HERE / "model/scorer_report.json").read_text())
rep["live_threshold"] = round(thr, 4)
rep["live_threshold_rule"] = "flags 5% of random edits (corpus N) in the user-disjoint test fold"
rep["live_threshold_catches"] = {k: round(float((p[(c == k) & (yt == 1)] >= thr).mean()), 3) for k in ("A", "R", "M")}
(HERE / "model/scorer_report.json").write_text(json.dumps(rep, indent=1))
print(rep["live_threshold"], rep["live_threshold_catches"])
