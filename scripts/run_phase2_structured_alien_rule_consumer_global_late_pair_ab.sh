#!/bin/bash
set -euo pipefail

# Runs the Stage-1 global-only + late-enable rule-consumer smoke A/B for a
# single seed on top of the current replay-off development line:
# 1) no_prio
# 2) no_prio_rule_consumer_global_late
#
# The late-enable schedule is intentionally gentle for the 20k control run:
# - start_updates = 625  (~10k env steps)
# - ramp_updates  = 500  (~10k -> 18k env-step ramp)

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/logdir_naming.sh"
source "$ROOT_DIR/scripts/git_run_metadata.sh"

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

SEED=${SEED:?SEED is required}
GPU_ID=${GPU_ID:-0}
PAIR_PARALLEL=${PAIR_PARALLEL:-2}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
RUN_NAME=${RUN_NAME:-bench_atari_structured_20k_alien_rule_consumer_global_late_pair_seed${SEED}_$(logdir_run_tag)}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/$RUN_NAME"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

COMMON_ARGS=(
  "env.steps=$TRAIN_STEPS"
  buffer.prioritized=False
  model.actor_imagination.mode_mix=0.0
)

run_variant() {
  local variant=$1
  shift
  local train_args=("${COMMON_ARGS[@]}" "$@")
  local train_extra_args="${train_args[*]}"
  local variant_logdir="$BASE_LOGDIR/$variant"

  env \
    PYTHON="$PYTHON_BIN" \
    BASE_LOGDIR="$variant_logdir" \
    SEED="$SEED" \
    TRAIN_EXTRA_ARGS="$train_extra_args" \
    PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1} \
    "$ROOT_DIR/scripts/run_phase2_structured_alien_repro_one.sh"
}

mkdir -p "$BASE_LOGDIR"
cd "$ROOT_DIR"

print_git_run_metadata "$ROOT_DIR" "rule-consumer-global-late pair seed ${SEED} launch"

if [ "$PAIR_PARALLEL" -ge 2 ]; then
  run_variant no_prio &
  pid_a=$!
  run_variant no_prio_rule_consumer_global_late \
    model.use_rule_prediction_consumer=True \
    model.rule_prediction_consumer.apply_to_map=False \
    model.rule_prediction_consumer.apply_to_obj=False \
    model.rule_prediction_consumer.apply_to_global=True \
    model.rule_prediction_consumer.start_updates=625 \
    model.rule_prediction_consumer.ramp_updates=500 &
  pid_b=$!
  wait "$pid_a"
  wait "$pid_b"
else
  run_variant no_prio
  run_variant no_prio_rule_consumer_global_late \
    model.use_rule_prediction_consumer=True \
    model.rule_prediction_consumer.apply_to_map=False \
    model.rule_prediction_consumer.apply_to_obj=False \
    model.rule_prediction_consumer.apply_to_global=True \
    model.rule_prediction_consumer.start_updates=625 \
    model.rule_prediction_consumer.ramp_updates=500
fi

"$PYTHON_BIN" - <<'PY' "$BASE_LOGDIR" | tee "$BASE_LOGDIR/variant_compare.tsv"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
print("variant\tseed\tbest_eval\tlast_eval\traw_mode\tmode\tsample\tsample_minus_mode\tmode_minus_raw")
for variant_dir in sorted(base.iterdir()):
    if not variant_dir.is_dir():
        continue
    structured = variant_dir / "structured"
    if not structured.exists():
        continue
    for seed_dir in sorted(structured.glob("seed_*")):
        metrics = seed_dir / "metrics.jsonl"
        probe = seed_dir / "latest_mode_vs_sample_eval_20ep_3x.json"
        if not metrics.exists() or not probe.exists():
            continue
        rows = [json.loads(x) for x in metrics.read_text().splitlines() if x.strip()]
        evals = [r for r in rows if "episode/eval_score" in r]
        if not evals:
            continue
        gap = json.loads(probe.read_text())
        print(
            variant_dir.name,
            seed_dir.name.replace("seed_", ""),
            max(r["episode/eval_score"] for r in evals),
            evals[-1]["episode/eval_score"],
            gap["raw_mode"]["summary"]["score_mean"],
            gap["mode"]["summary"]["score_mean"],
            gap["sample"]["summary"]["score_mean"],
            gap["gap"]["sample_minus_mode_mean"],
            gap["gap"]["mode_minus_raw_mode_mean"],
            sep="\t",
        )
PY

echo
echo "Done."
echo "BASE_LOGDIR=$BASE_LOGDIR"
echo "Compare table: $BASE_LOGDIR/variant_compare.tsv"
