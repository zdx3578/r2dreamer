#!/bin/bash
set -euo pipefail

# Runs the Stage-1 v9 auxiliary rule-signal smoke A/B for a single seed:
# 1) no_prio
# 2) no_prio_rule_consumer_aux
#
# Compared to v8, this v9 path no longer mutates effect_out["delta_global"].
# The same small rule head is used only as:
# - an auxiliary global-delta prediction target
# - a consistency target for the main delta_global head
#
# The gating schedule stays aligned with v8 so the A/B isolates the injection
# form, not the enable window.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/logdir_naming.sh"
source "$ROOT_DIR/scripts/git_run_metadata.sh"
source "$ROOT_DIR/scripts/structured_alien_defaults.sh"

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
PAIR_PARALLEL=${PAIR_PARALLEL:-${STRUCTURED_ALIEN_PAIR_PARALLEL_DEFAULT}}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
RUN_NAME=${RUN_NAME:-bench_atari_structured_20k_alien_rule_consumer_aux_pair_seed${SEED}_$(logdir_run_tag)}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/$RUN_NAME"}
GATE_THRESHOLD=${GATE_THRESHOLD:-0.045}
START_UPDATES=${START_UPDATES:-625}
RAMP_UPDATES=${RAMP_UPDATES:-500}
LATCH_RAMP_UPDATES=${LATCH_RAMP_UPDATES:-250}
AUX_LOSS_SCALE=${AUX_LOSS_SCALE:-0.02}
CONSISTENCY_LOSS_SCALE=${CONSISTENCY_LOSS_SCALE:-0.01}

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

print_git_run_metadata "$ROOT_DIR" "rule-consumer-aux pair seed ${SEED} launch"

AUX_ARGS=(
  model.use_rule_prediction_consumer=True
  model.rule_prediction_consumer.mode=aux
  model.rule_prediction_consumer.apply_to_map=False
  model.rule_prediction_consumer.apply_to_obj=False
  model.rule_prediction_consumer.apply_to_global=True
  "model.rule_prediction_consumer.start_updates=$START_UPDATES"
  "model.rule_prediction_consumer.ramp_updates=$RAMP_UPDATES"
  model.rule_prediction_consumer.gate_enable_mode=sticky_threshold
  "model.rule_prediction_consumer.gate_threshold=$GATE_THRESHOLD"
  "model.rule_prediction_consumer.latch_ramp_updates=$LATCH_RAMP_UPDATES"
  "model.loss_scales.rule_consumer_global_aux=$AUX_LOSS_SCALE"
  "model.loss_scales.rule_consumer_global_consistency=$CONSISTENCY_LOSS_SCALE"
)

if [ "$PAIR_PARALLEL" -ge 2 ]; then
  run_variant no_prio &
  pid_a=$!
  run_variant no_prio_rule_consumer_aux "${AUX_ARGS[@]}" &
  pid_b=$!
  wait "$pid_a"
  wait "$pid_b"
else
  run_variant no_prio
  run_variant no_prio_rule_consumer_aux "${AUX_ARGS[@]}"
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
