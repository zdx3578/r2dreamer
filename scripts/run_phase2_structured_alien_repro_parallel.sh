#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-}
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif [ -x "/home/zdx/.micromamba/envs/r2dreamer312/bin/python" ]; then
    PYTHON_BIN="/home/zdx/.micromamba/envs/r2dreamer312/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

GPU_ID=${GPU_ID:-0}
MAX_PARALLEL=${MAX_PARALLEL:-2}
SEEDS=${SEEDS:-"0 1 4 5"}
RUN_NAME=${RUN_NAME:-bench_atari_structured_50k_alien_repro_$(date +%Y%m%d_%H%M%S)}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/$RUN_NAME"}
STRUCTURED_DIR="$BASE_LOGDIR/structured"
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
PROBE_EXTRA_ARGS=${PROBE_EXTRA_ARGS:-}

train_extra=()
probe_extra=()
if [ -n "$TRAIN_EXTRA_ARGS" ]; then
  read -r -a train_extra <<<"$TRAIN_EXTRA_ARGS"
fi
if [ -n "$PROBE_EXTRA_ARGS" ]; then
  read -r -a probe_extra <<<"$PROBE_EXTRA_ARGS"
fi

# Keep per-process CPU thread fan-out under control when multiple runs share one GPU.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$STRUCTURED_DIR"
cd "$ROOT_DIR"

run_seed() {
  local seed="$1"
  local seed_dir="$STRUCTURED_DIR/seed_${seed}"
  mkdir -p "$seed_dir"
  (
    set -euo pipefail
    echo "[$(date '+%F %T')] seed ${seed}: train start"
    "$PYTHON_BIN" train.py \
      env=atari100k \
      +exp=phase2_structured \
      logdir="$seed_dir" \
      seed="$seed" \
      device=cuda:0 \
      buffer.device=cuda:0 \
      buffer.storage_device=cpu \
      batch_size=4 \
      batch_length=32 \
      env.env_num=1 \
      env.eval_episode_num=20 \
      env.steps=50000 \
      env.train_ratio=32 \
      env.task=atari_alien \
      trainer.eval_episode_num=20 \
      trainer.sample_eval_episode_num=5 \
      trainer.eval_every=5000 \
      trainer.save_every=3000 \
      trainer.eval_gap_checkpoint_threshold=100 \
      trainer.eval_drop_checkpoint_ratio=0.5 \
      trainer.eval_drop_checkpoint_sample_ratio=0.75 \
      trainer.pretrain=0 \
      trainer.update_log_every=1000 \
      model.compile=False \
      "${train_extra[@]}" \
      |& tee "$seed_dir/run.out"

    echo "[$(date '+%F %T')] seed ${seed}: train done"
    if [ -f "$seed_dir/latest.pt" ]; then
      "$PYTHON_BIN" scripts/eval_checkpoint_policy_gap.py \
        "$seed_dir/latest.pt" \
        --device cuda:0 \
        --eval-episodes 20 \
        --repeats 3 \
        --output "$seed_dir/latest_mode_vs_sample_eval_20ep_3x.json" \
        "${probe_extra[@]}" \
        |& tee "$seed_dir/probe.out"
      echo "[$(date '+%F %T')] seed ${seed}: probe done"
    else
      echo "[$(date '+%F %T')] seed ${seed}: latest.pt missing, skip probe" | tee "$seed_dir/probe.out"
    fi
  )
}

active_jobs=0
for seed in $SEEDS; do
  run_seed "$seed" &
  active_jobs=$((active_jobs + 1))
  if [ "$active_jobs" -ge "$MAX_PARALLEL" ]; then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done
wait

"$PYTHON_BIN" scripts/summarize_atari_base_50k.py "$BASE_LOGDIR" \
  | tee "$BASE_LOGDIR/summary.out"

"$PYTHON_BIN" - <<'PY' "$STRUCTURED_DIR" | tee "$BASE_LOGDIR/latest_gap_table.tsv"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
print("seed\tbest_eval\tlast_eval\talert_steps\tlatest_mode\tlatest_sample\tlatest_gap")
for seed_dir in sorted(base.glob("seed_*")):
    metrics_path = seed_dir / "metrics.jsonl"
    gap_path = seed_dir / "latest_mode_vs_sample_eval_20ep_3x.json"
    if not metrics_path.exists() or not gap_path.exists():
        continue
    rows = [json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]
    evals = [r for r in rows if "episode/eval_score" in r]
    if not evals:
        continue
    gap = json.loads(gap_path.read_text())
    alerts = [
        str(r["step"])
        for r in evals
        if r.get("episode/eval_gap_triggered") or r.get("episode/eval_zero_collapse_triggered")
    ]
    print(
        seed_dir.name,
        max(r["episode/eval_score"] for r in evals),
        evals[-1]["episode/eval_score"],
        ",".join(alerts) if alerts else "-",
        gap["mode"]["summary"]["score_mean"],
        gap["sample"]["summary"]["score_mean"],
        gap["gap"]["sample_minus_mode_mean"],
        sep="\t",
    )
PY

echo
echo "Done."
echo "BASE_LOGDIR=$BASE_LOGDIR"
echo "Summary: $BASE_LOGDIR/summary.out"
echo "Gap table: $BASE_LOGDIR/latest_gap_table.tsv"
