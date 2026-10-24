# Running a training round on a GPU machine

The tools in this folder run a full training round on a remote Linux machine with a GPU: an HPC
cluster, a cloud VM or a university server. Rounds 2–4 of this project ran this way (on an
NVIDIA H100 slice of the Altair HPC cluster, hence the folder name), but nothing here depends
on that cluster.

One zip per round. You build it on your own machine, upload it, and start one job script. The
script unpacks the zip into its own round folder, runs every step of the round's plan, and
packs logs, timings and all results into one zip to download.

## What the remote machine needs

- Linux with `bash`, `unzip` and Python 3 (a TensorFlow 2.12+ image is easiest).
- A GPU is optional. Every step also runs on a CPU, more slowly.
- Internet access only if a plan installs extra packages (`install_st`) or runs the WHERE job.
- A writable working folder, called `<BASE>` below.

## 1. Build the zip (your machine)

```bash
python altair/build_zip.py --round r4_seeds --plan altair/plans/r4_seeds.txt
```

This writes `altair/_dist/mapguard_altair.zip`: code, the feature store and the plan. It is
git-ignored, because it carries data. `--data <dir>` picks a specific feature store.

## 2. Start the job (on the remote machine)

Upload the zip into `<BASE>`, open a terminal there (SSH or a Jupyter terminal), and run:

```bash
cd <BASE>
unzip -o -j mapguard_altair.zip altair/run_round2.sh -d .
MAPGUARD_BASE="$(pwd)" setsid nohup bash run_round2.sh > launch.out 2>&1 < /dev/null &
```

The first command takes the job script out of the zip, so only the zip needs uploading. The
second runs it in the background, detached from the terminal, so you can close the session.
On a scheduler such as PBS or Slurm, put the same two lines in a job script instead.

`MAPGUARD_BASE` sets the working folder. Without it the script looks for a `deepLearning_osm`
folder under `/data` or your home directory, then falls back to the current folder.

## 3. Check progress

```bash
cat <BASE>/rounds/*/logs/STATUS <BASE>/where/STATUS 2>/dev/null
```

When `STATUS` says finished, download `<BASE>/rounds/<round>/<round>_results.zip` and extract
it into `experiments/rounds/<round>/raw/`.

## Folder layout on the remote machine

```
<BASE>/
  mapguard_altair.zip, run_round2.sh, launch.out
  rounds/<round>/                 one per zip; running the same zip again resumes it
    app/                          code + data + experiments/runs
    logs/job.log  STATUS  timing.csv  env.txt  steps/<step>.log
    <round>_results.zip           download this
  where/                          WHERE job, shared by all rounds (done once)
    changesets-latest.osm.bz2     the planet changeset dump (~9 GB)
    author_points.npz             target authors' changeset locations
    where_features.parquet        one row per changeset id; also in the results zip
    where.log  STATUS
```

## Plans: what a round runs

Each zip carries `altair/PLAN.txt`, copied from `altair/plans/<round>.txt`, one step per line:

```
name | flags | run_pipeline.py arguments
```

Flags:
- `fatal`: stop the job if this step fails.
- `install_st`: install torch and sentence-transformers for this step only.
- `where`: run the WHERE data job instead of the pipeline.

Behaviour:
- Running the same zip again skips finished steps.
- Only one copy of a round can run at a time. A second launch exits with "already running"
  (a lock with PID and heartbeat in `rounds/<round>/.lock`).
- `MAPGUARD_STEPS=a,b` limits a run to those step names.
- Training results are packed into the results zip before the WHERE job starts, and again
  at the end.

| Plan | What it runs | Time (one H100 40 GB slice, 1 CPU core) |
|---|---|---|
| `r3_v3_explore.txt` | embeddings, smoke test, v3 ablations (5 rows × 3 seeds), all six tasks with hybrid variants and stacking | ~2 h |
| `r4_seeds.txt` | embeddings, smoke test, ablations (4 rows × 3 seeds), all six tasks × 3 seeds, temporal evaluation | ~3 h 45 min |

To run your own round, copy a plan, edit the steps, and build a zip with `--plan` pointing at it.

## Files

| File | Job |
|---|---|
| `build_zip.py` | pack code, feature store and plan into one zip |
| `run_round2.sh` | the job script: unpack, run each plan step, log, pack results (used for every round since round 2, despite the name) |
| `where_job.py` | the WHERE data job: author locations from the planet changeset dump |
| `plans/` | the step plans of rounds 3 and 4 |
