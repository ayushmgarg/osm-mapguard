#!/bin/bash
# MapGuard on a GPU machine - one training round per zip, plus the WHERE data job.
#
# Start from a Jupyter terminal in the folder holding mapguard_altair.zip:
#   cd <BASE>
#   unzip -o -j mapguard_altair.zip altair/run_round2.sh -d .
#   MAPGUARD_BASE="$(pwd)" setsid nohup bash run_round2.sh > launch.out 2>&1 < /dev/null &
#
# Folder layout (one folder per round = per zip; re-running the same zip resumes it):
#   <BASE>/rounds/<round>/app/                 code + data + experiments/runs (all runs)
#   <BASE>/rounds/<round>/logs/job.log         every attempt, appended, live
#   <BASE>/rounds/<round>/logs/STATUS          what is running now / how it ended
#   <BASE>/rounds/<round>/logs/timing.csv      step, start, end, seconds, exit code
#   <BASE>/rounds/<round>/logs/env.txt         GPU, TensorFlow, packages
#   <BASE>/rounds/<round>/logs/steps/<step>.log
#   <BASE>/rounds/<round>/<round>_results.zip  send this back (runs + logs + WHERE output)
#   <BASE>/where/                              WHERE job: dump download, extract, features
#                                              (shared by all rounds; done once)
#
# Steps come from altair/PLAN.txt inside the zip (one round = one plan), one per line:
#     name | flags | run_pipeline.py arguments
# flags: fatal (stop the job if it fails), install_st (install sentence-transformers first),
#        where (run the WHERE data job instead of the pipeline).
# MAPGUARD_STEPS=a,b limits the run to those step names.

set -uo pipefail

ONLY="${MAPGUARD_STEPS:-}"
ZIP_NAME="mapguard_altair.zip"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

START_DIR="$(pwd)"
JOB_ID="${PBS_JOBID:-local}"; JOB_ID="${JOB_ID%%.*}"

# ---------------------------------------------------------------- 1. where to work
find_base() {
    if [ -n "${MAPGUARD_BASE:-}" ]; then echo "$MAPGUARD_BASE"; return; fi
    for d in /data/*/deepLearning_osm \
             /data/*/*/deepLearning_osm "$HOME/deepLearning_osm"; do
        if [ -d "$d" ] && [ -w "$d" ]; then echo "$d"; return; fi
    done
    echo "$START_DIR"
}
BASE="$(find_base)"

PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
        PY="$(command -v "$c")"; break
    fi
done
if [ -z "$PY" ]; then echo "no working python in this image"; exit 1; fi

ZIP=""
for z in "$BASE/$ZIP_NAME" "$START_DIR/$ZIP_NAME"; do
    if [ -f "$z" ] && { [ -z "$ZIP" ] || [ "$z" -nt "$ZIP" ]; }; then ZIP="$z"; fi
done
if [ -z "$ZIP" ]; then echo "$ZIP_NAME not found in $BASE or $START_DIR"; exit 1; fi

ROUND="$("$PY" - "$ZIP" <<'EOF'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
if "altair/ROUND.txt" in names:
    print(z.read("altair/ROUND.txt").decode().strip())
else:
    print("round_" + json.loads(z.read("BUNDLE_MANIFEST.json"))["generated"][:16].replace(":", "").replace("T", "_"))
EOF
)"
RDIR="$BASE/rounds/$ROUND"
APP="$RDIR/app"
LOGS="$RDIR/logs"
mkdir -p "$LOGS/steps" "$RDIR/.done" "$BASE/where"

# One job per round. Round 3 was launched twice and both copies ran the whole plan at once on one
# GPU (3-5x slower, duplicate runs). A second launch now stops here. mkdir is atomic on NFS.
LOCK="$RDIR/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    other="$(cat "$LOCK/pid" 2>/dev/null)"
    beat="$(cat "$LOCK/heartbeat" 2>/dev/null || echo 0)"
    age=$(( $(date +%s) - ${beat:-0} ))
    # alive if its PID answers here, or (another Jupyter job = other PID namespace) if its
    # heartbeat is fresh: the running job rewrites it every 60 s
    same_host=0; [ "$(cat "$LOCK/host" 2>/dev/null)" = "$(hostname)" ] && same_host=1
    if { [ "$same_host" = 1 ] && [ -n "$other" ] && kill -0 "$other" 2>/dev/null; } || [ "$age" -lt 300 ]; then
        echo "[$(ts)] round $ROUND is already running (PID $other, $(cat "$LOCK/started" 2>/dev/null))."
        echo "Check progress: cat $LOGS/STATUS - nothing was started."
        exit 3
    fi
    echo "[$(ts)] stale lock from PID ${other:-unknown} (not running) - taking over"
    rm -rf "$LOCK"; mkdir "$LOCK" || { echo "could not take the lock"; exit 3; }
fi
echo $$ > "$LOCK/pid"; hostname > "$LOCK/host"; date > "$LOCK/started"; date +%s > "$LOCK/heartbeat"
# rewritten every 60 s while this job lives (temp + mv: a reader never sees an empty file);
# stops by itself if this job is killed, so the lock goes stale after 5 minutes
( while [ -d "$LOCK" ] && kill -0 $$ 2>/dev/null; do
      date +%s > "$LOCK/heartbeat.tmp" 2>/dev/null && mv -f "$LOCK/heartbeat.tmp" "$LOCK/heartbeat" 2>/dev/null
      sleep 60
  done ) &
HEARTBEAT_PID=$!
# until on_exit is installed (below), any early exit still releases the lock
trap 'kill "$HEARTBEAT_PID" 2>/dev/null; rm -rf "$LOCK"' EXIT
exec > >(tee -a "$LOGS/job.log") 2>&1

status() { echo "$(ts)  $*" > "$LOGS/STATUS"; say "STATUS: $*"; }
[ -f "$LOGS/timing.csv" ] || echo "step,start,end,seconds,exit_code,job" > "$LOGS/timing.csv"
JOB_T0=$(date +%s)

echo
echo "=================================================================================="
say "MapGuard | round $ROUND | job $JOB_ID | host $(hostname)"
say "base   : $BASE"
say "zip    : $ZIP"
say "round  : $RDIR"
say "only   : ${ONLY:-all steps in PLAN.txt}"

# ---------------------------------------------------------------- 2. unpack once per round
"$PY" - "$ZIP" "$APP" <<'EOF' || { status "FAILED: could not unpack $ZIP"; exit 1; }
import json, pathlib, sys, zipfile
zpath, app = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zpath) as z:
    fresh = json.loads(z.read("BUNDLE_MANIFEST.json"))
    cur = app / "BUNDLE_MANIFEST.json"
    if cur.exists() and json.loads(cur.read_text()).get("generated") == fresh.get("generated"):
        print("code + data already unpacked for this round")
    else:
        z.extractall(app)                 # experiments/runs is never in the zip: runs survive
        print("unpacked ->", app)
print("bundle", fresh.get("generated"), "git", str(fresh.get("git_sha"))[:12])
EOF
cd "$APP" || exit 1

# ---------------------------------------------------------------- 3. python packages
export PYTHONUSERBASE="$BASE/.pydeps"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PIP_CACHE_DIR="$BASE/.pipcache"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export HF_HOME="$BASE/.hfcache"
NC="${NCPUS:-$(nproc 2>/dev/null || echo 4)}"
export OMP_NUM_THREADS="$NC"

missing() {
    "$PY" - <<'EOF'
import importlib.util
pk = (("pyarrow", "pyarrow"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("xgboost", "xgboost"))
print(" ".join(p for m, p in pk if importlib.util.find_spec(m) is None))
EOF
}
MISSING="$(missing)"
if [ -n "$MISSING" ]; then
    NUMPY="$("$PY" -c 'import numpy; print(numpy.__version__)')"
    say "installing: $MISSING (numpy pinned at $NUMPY)"
    "$PY" -m pip install --user -q $MISSING "numpy==$NUMPY" || true
    MISSING="$(missing)"
    if [ -n "$MISSING" ]; then status "FAILED: could not install $MISSING"; exit 1; fi
fi

{
    echo "== date";   ts
    echo "== host";   hostname
    echo "== cpus";   echo "NCPUS=${NCPUS:-unset} nproc=$(nproc 2>/dev/null)"
    echo "== gpu";    nvidia-smi 2>&1 || echo "nvidia-smi not available"
    echo "== python"; "$PY" --version
    echo "== tensorflow"
    "$PY" -c "import tensorflow as tf; print(tf.__version__, [g.name for g in tf.config.list_physical_devices('GPU')])" 2>&1 | tail -1
    echo "== bundle"; head -12 BUNDLE_MANIFEST.json
    echo "== packages"; "$PY" -m pip list 2>/dev/null
} > "$LOGS/env.txt" 2>&1

# ---------------------------------------------------------------- 4. results packaging
package() {
    "$PY" - "$APP" "$RDIR" "$ROUND" "$BASE/where" <<'EOF' || true
import pathlib, sys, zipfile
app, rdir, name, where = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
dest = rdir / f"{name}_results.zip"
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("experiments/runs", "experiments/pipeline"):
        for f in sorted((app / top).rglob("*")) if (app / top).exists() else []:
            if f.is_file():
                z.write(f, f.relative_to(app).as_posix())
    for f in sorted((rdir / "logs").rglob("*")):
        if f.is_file():
            z.write(f, "logs/" + f.relative_to(rdir / "logs").as_posix())
    for n in ("where_features.parquet", "where.log", "STATUS", "source.json"):
        if (where / n).exists():
            z.write(where / n, f"where/{n}")
print(f"results: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
EOF
}
FINAL="unknown"
on_exit() {
    local total=$(( $(date +%s) - JOB_T0 ))
    say "this job: $((total / 3600))h $(((total % 3600) / 60))m"
    echo "JOB_TOTAL,,,${total},,$JOB_ID" >> "$LOGS/timing.csv"
    [ "$FINAL" = "unknown" ] && status "STOPPED before finishing (killed or error) - run the same command again to resume"
    package
    kill "$HEARTBEAT_PID" 2>/dev/null
    [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"
}
trap on_exit EXIT
trap 'say "received TERM"; exit 143' TERM

# ---------------------------------------------------------------- 5. the steps
FAILED=()
record() { echo "$1,$2,$(ts),$3,$4,$JOB_ID" >> "$LOGS/timing.csv"; }
run_step() {
    local name="$1"; shift
    if [ -f "$RDIR/.done/$name" ]; then
        say "skip $name - finished earlier in this round"
        return 0
    fi
    status "running $name: $*"
    local t0 start rc
    t0=$(date +%s); start="$(ts)"
    "$@" 2>&1 | tee -a "$LOGS/steps/$name.log"
    rc=${PIPESTATUS[0]}
    local secs=$(( $(date +%s) - t0 ))
    record "$name" "$start" "$secs" "$rc"
    if [ "$rc" -eq 0 ]; then
        date > "$RDIR/.done/$name"
        say "done $name in $((secs / 60)) min"
    else
        FAILED+=("$name")
        say "FAILED $name (exit $rc) after $((secs / 60)) min - see logs/steps/$name.log"
    fi
    return "$rc"
}
pipeline() { "$PY" -u train/run_pipeline.py --feature-store data/interim "$@"; }

# torch + sentence-transformers live in their own folder, seen only by steps flagged
# install_st: they must not shadow packages the image's TensorFlow depends on.
# Round 3: the latest transformers (5.x) needs a newer numpy than the image's 1.22.2 and crashed
# on import. The embedding step reads the store with pandas built for numpy 1.22, so numpy stays
# pinned and the libraries are pinned to versions that support it (2024 releases).
# .pydeps_st holds the broken round-3 install and is not used any more.
ST_DIR="$BASE/.pydeps_st2"
ST_PINS=("sentence-transformers==2.7.0" "transformers==4.40.2" "tokenizers<0.20" "huggingface-hub<0.26" "torch==2.5.1")
st_ok() { PYTHONPATH="$ST_DIR" "$PY" -c "import numpy, torch; from sentence_transformers import SentenceTransformer; print('embedding env ok: numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"; }
install_st() {
    if st_ok >/dev/null 2>&1; then st_ok; return 0; fi
    local NUMPY
    NUMPY="$("$PY" -c 'import numpy; print(numpy.__version__)')"
    say "installing ${ST_PINS[*]} into $ST_DIR (numpy pinned at $NUMPY) - several minutes"
    # install beside, move into place only if it imports: an interrupted install never
    # leaves a half-filled $ST_DIR that pip --target would then refuse to repair
    rm -rf "$ST_DIR.tmp"
    if "$PY" -m pip install -q --target "$ST_DIR.tmp" "${ST_PINS[@]}" "numpy==$NUMPY" \
       && PYTHONPATH="$ST_DIR.tmp" "$PY" -c "from sentence_transformers import SentenceTransformer"; then
        rm -rf "$ST_DIR"; mv "$ST_DIR.tmp" "$ST_DIR"; st_ok
    else
        say "WARNING: embedding environment failed to install or import - embedding variants will be skipped"
    fi
}

trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

PLAN="$APP/altair/PLAN.txt"
if [ ! -f "$PLAN" ]; then status "FAILED: no altair/PLAN.txt in this zip"; FINAL="failed"; exit 1; fi
say "plan:"; grep -v '^#' "$PLAN" | grep -v '^[[:space:]]*$' | sed 's/^/    /'
PLAN_N="$LOGS/plan.normalized"          # CRLF removed, trailing newline guaranteed
{ cat "$PLAN"; echo; } | tr -d '\r' > "$PLAN_N"
while IFS='|' read -r name flags pargs <&3; do
    name="$(trim "${name:-}")"; flags="$(trim "${flags:-}")"; pargs="$(trim "${pargs:-}")"
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
    if [[ ",$flags," == *",where,"* ]]; then
        if [ -f "$APP/altair/where_targets.parquet" ]; then
            run_step "$name" "$PY" -u altair/where_job.py \
                --targets altair/where_targets.parquet --workdir "$BASE/where"; rc=$?
        else
            say "skip $name - this zip has no altair/where_targets.parquet"; continue
        fi
    else
        read -r -a ARGV <<< "$pargs"          # plain word split: no quotes, no globbing in plans
        if [[ ",$flags," == *",install_st,"* ]]; then
            install_st
            export PYTHONPATH="$ST_DIR${PYTHONPATH:+:$PYTHONPATH}"
            run_step "$name" pipeline "${ARGV[@]}"; rc=$?
            export PYTHONPATH="${PYTHONPATH#"$ST_DIR"}"; PYTHONPATH="${PYTHONPATH#:}"
            # embeddings are optional for the pipeline (exit 0 without them), but this step
            # only counts as done if they exist - otherwise a rerun retries it
            if [ "$rc" -eq 0 ] && [ ! -f "$APP/data/interim/text_embeddings.json" ]; then
                rm -f "$RDIR/.done/$name"
                say "WARNING: $name finished without text_embeddings.json - embedding variants will be skipped"
                FAILED+=("$name")
            fi
        else
            run_step "$name" pipeline "${ARGV[@]}"; rc=$?
        fi
    fi
    if [ "$rc" -ne 0 ] && [[ ",$flags," == *",fatal,"* ]]; then
        FINAL="failed"
        status "FAILED at $name - nothing after it was run. Send logs/steps/$name.log"
        exit 1
    fi
done 3< "$PLAN_N"

if [ ${#FAILED[@]} -eq 0 ]; then
    FINAL="ok"
    status "FINISHED - all steps ok. Send rounds/$ROUND/${ROUND}_results.zip"
else
    FINAL="partial"
    status "FINISHED with failures: ${FAILED[*]}. Send rounds/$ROUND/${ROUND}_results.zip"
fi
exit 0
