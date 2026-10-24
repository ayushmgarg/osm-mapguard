"""Pack everything a remote machine needs to train into one zip.

    python cluster/bundle.py                       # code + feature store from MAPGUARD_DATA/interim
    python cluster/bundle.py --data D:/fs          # a feature store somewhere else
    python cluster/bundle.py --no-data             # code only (e.g. for 00_wiring_check)

Writes `cluster/_bundles/mapguard_bundle.zip`. The same zip serves the SVKM cluster (upload
it next to any notebook in cluster/) and the M4 (unzip, run `train/setup_env.sh`, then
`train/run_pipeline.py`).

Contents:

    src/ train/ cluster/*/*.ipynb      the code, exactly as committed
    requirements-train.txt RUNBOOK.md
    data/interim/*.parquet            the feature store + build_report.json
    experiments/splits/*.json         frozen splits, so every machine trains on the same rows
    BUNDLE_MANIFEST.json              git SHA, dirty flag, schema version, row counts, hashes

Nothing is fetched on the remote side - ever. Every number a remote run produces can be
traced back to this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "cluster" / "_bundles"
STORE_FILES = ("changesets.parquet", "edits.parquet", "user_seq.parquet")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def code_files() -> list[Path]:
    out = []
    for top in ("src", "train"):
        for f in sorted((REPO / top).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                out.append(f)
    out += sorted((REPO / "cluster").glob("*/*.ipynb"))
    out += sorted((REPO / "cluster").glob("*/README.md"))
    for extra in ("requirements-train.txt", "docs/running/RUNBOOK.md", "docs/running/RUN_ALTAIR.md", "pyproject.toml"):
        if (REPO / extra).exists():
            out.append(REPO / extra)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="feature store dir (default MAPGUARD_DATA/interim)")
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from mapguard.config import interim_dir

    data_dir = Path(args.data) if args.data else interim_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.out) if args.out else OUT / "mapguard_bundle.zip"

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "src", "train", "cluster")),
        "python_packed_with": sys.version.split()[0],
        "data_dir": None if args.no_data else str(data_dir),
        "data": {},
        "feature_schema": None,
        "coverage": None,
        "splits": [],
    }

    if not args.no_data:
        missing = [n for n in STORE_FILES if not (data_dir / n).exists()]
        if missing:
            raise SystemExit(f"feature store at {data_dir} is missing {missing}. Build it first "
                             "(RUNBOOK.md step 2) or pass --no-data for a code-only bundle.")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in code_files():
            z.write(f, f.relative_to(REPO).as_posix())

        splits = REPO / "experiments" / "splits"
        for f in sorted(splits.glob("*.json")) if splits.exists() else []:
            z.write(f, f"experiments/splits/{f.name}")
            manifest["splits"].append(f.name)

        if not args.no_data:
            for name in STORE_FILES:
                src = data_dir / name
                # Parquet is already compressed; storing avoids minutes of pointless deflate.
                z.write(src, f"data/interim/{name}", compress_type=zipfile.ZIP_STORED)
                manifest["data"][name] = {"bytes": src.stat().st_size, "rows": parquet_rows(src),
                                          "sha256": sha256(src)}
            report = data_dir / "build_report.json"
            if report.exists():
                z.write(report, "data/interim/build_report.json")
                rep = json.loads(report.read_text(encoding="utf-8"))
                manifest["feature_schema"] = rep.get("schema_version")
                manifest["coverage"] = rep.get("coverage_by_corpus")

        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))

    print(json.dumps({k: manifest[k] for k in ("generated", "git_sha", "git_dirty",
                                                "feature_schema", "data")}, indent=2))
    print(f"\n{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    if manifest["git_dirty"]:
        print("WARNING: src/, train/ or cluster/ has uncommitted changes - commit first so "
              "git_sha describes this bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
