"""Inline the demo data into template.html -> ../index.html."""
import json
from pathlib import Path

B = Path(__file__).parent
scores = json.loads((B / "scores.json").read_text())
summary = json.loads((B / "summary.json").read_text())
abl = json.loads((B.parents[1] / "experiments/rounds/r2c_2026-10-09_altair_h100_v21/analysis/ablations_author_robust.json").read_text())
data = {
    "scores": {"y": scores["y"], "models": {m: {"p": v["p"], "threshold": v["threshold"]} for m, v in scores["models"].items()}},
    "examples": json.loads((B / "examples.json").read_text(encoding="utf-8")),
    "summary": {k: summary[k] for k in ("tokens", "attribution", "tasks")},
    "ablation": [[n, abl[k]["pr_auc"][0]] for n, k in (
        ("Base: edit numbers + table", "base_edits_tabular"), ("+ geometry", "plus_geometry"),
        ("+ edit order & history", "plus_sequences"), ("+ comment & tag text", "plus_text"),
        ("+ edit meaning", "plus_edit_meaning"), ("All branches", "full"))],
}
blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (B / "template.html").read_text(encoding="utf-8").replace("__DATA__", blob)
(B.parent / "index.html").write_text(html, encoding="utf-8")
print(f"index.html {len(html)/1e3:.0f} KB, {len(data['examples'])} examples")
