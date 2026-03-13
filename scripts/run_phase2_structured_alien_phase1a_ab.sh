#!/bin/bash
set -euo pipefail

# Runs the 20k Alien Phase1A post-refactor feature A/B matrix on the standard
# phase2_structured Atari runner:
# 1) phase1a_control
# 2) phase1a_structure_change
# 3) phase1a_structure_local
# 4) phase1a_structure_local_direct
#
# Intended default host split:
# - 2080: this script
# - 2070: scripts/run_phase2_structured_alien_deterministic_ab.sh
#
# Typical usage:
#   MACHINE_NAME=$(hostname -s) GPU_ID=0 MAX_PARALLEL=2 SEEDS="3 4 5" \
#   scripts/run_phase2_structured_alien_phase1a_ab.sh

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

GPU_ID=${GPU_ID:-0}
MAX_PARALLEL=${MAX_PARALLEL:-2}
SEEDS=${SEEDS:-"3 4 5"}
RUN_NAME=${RUN_NAME:-bench_atari_structured_20k_alien_phase1a_ab_$(logdir_run_tag)}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/$RUN_NAME"}
MACHINE_NAME=${MACHINE_NAME:-$(hostname -s 2>/dev/null || hostname)}

COMMON_ARGS=(
  env.steps=20000
  model.actor_eval.repeat_calibration=True
  model.actor_eval.repeat_threshold=8
  model.actor_eval.min_top1_prob=0.5
  model.actor_eval.min_margin=0.15
  model.actor_imagination.mode_mix=0.0
  model.actor_training.mode_gap_weight=0.0
  model.actor_training.mode_gap_margin=0.0
  model.phase1a.reach_v2.enabled=False
  model.phase1a.reach_v2.condition_action=False
  model.use_direct_spatial_targets=False
)

run_variant() {
  local variant=$1
  shift
  local train_args=("${COMMON_ARGS[@]}" "$@")
  local train_extra_args="${train_args[*]}"
  local variant_logdir="$BASE_LOGDIR/$variant"

  echo
  echo "[$(date '+%F %T')] variant ${variant}: start"
  GPU_ID="$GPU_ID" \
  MAX_PARALLEL="$MAX_PARALLEL" \
  SEEDS="$SEEDS" \
  BASE_LOGDIR="$variant_logdir" \
  TRAIN_EXTRA_ARGS="$train_extra_args" \
  PYTHON="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/run_phase2_structured_alien_repro_parallel.sh"
  echo "[$(date '+%F %T')] variant ${variant}: done"
}

mkdir -p "$BASE_LOGDIR"
cd "$ROOT_DIR"

print_git_run_metadata "$ROOT_DIR" "phase1a ab launch"

run_variant \
  phase1a_control \
  model.phase1a.use_structure_change_targets=False \
  model.phase1a.use_local_change_targets=False \
  model.phase1a.use_direct_delta_targets=False

run_variant \
  phase1a_structure_change \
  model.phase1a.use_structure_change_targets=True \
  model.phase1a.use_local_change_targets=False \
  model.phase1a.use_direct_delta_targets=False

run_variant \
  phase1a_structure_local \
  model.phase1a.use_structure_change_targets=True \
  model.phase1a.use_local_change_targets=True \
  model.phase1a.use_direct_delta_targets=False

run_variant \
  phase1a_structure_local_direct \
  model.phase1a.use_structure_change_targets=True \
  model.phase1a.use_local_change_targets=True \
  model.phase1a.use_direct_delta_targets=True

"$PYTHON_BIN" - <<'PY' "$BASE_LOGDIR" | tee "$BASE_LOGDIR/variant_compare.tsv"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
print("variant\tseed\tbest_eval\tlast_eval\talert_steps\traw_mode\tmode\tsample\tsample_minus_mode\tmode_minus_raw")
for variant_dir in sorted(base.iterdir()):
    if not variant_dir.is_dir():
        continue
    structured = variant_dir / "structured"
    if not structured.exists():
        continue
    for seed_dir in sorted(structured.glob("seed_*")):
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
            variant_dir.name,
            seed_dir.name.replace("seed_", ""),
            max(r["episode/eval_score"] for r in evals),
            evals[-1]["episode/eval_score"],
            ",".join(alerts) if alerts else "-",
            gap["raw_mode"]["summary"]["score_mean"],
            gap["mode"]["summary"]["score_mean"],
            gap["sample"]["summary"]["score_mean"],
            gap["gap"]["sample_minus_mode_mean"],
            gap["gap"]["mode_minus_raw_mode_mean"],
            sep="\t",
        )
PY

"$PYTHON_BIN" - <<'PY' "$BASE_LOGDIR" "$MACHINE_NAME" | tee "$BASE_LOGDIR/machine_compare.tsv"
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
machine_name = sys.argv[2]
print("machine\tvariant\tseed\tbest_eval\tlast_eval\talert_steps\traw_mode\tmode\tsample\tsample_minus_mode\tmode_minus_raw")
for variant_dir in sorted(base.iterdir()):
    if not variant_dir.is_dir():
        continue
    structured = variant_dir / "structured"
    if not structured.exists():
        continue
    for seed_dir in sorted(structured.glob("seed_*")):
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
            machine_name,
            variant_dir.name,
            seed_dir.name.replace("seed_", ""),
            max(r["episode/eval_score"] for r in evals),
            evals[-1]["episode/eval_score"],
            ",".join(alerts) if alerts else "-",
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
echo "Machine table: $BASE_LOGDIR/machine_compare.tsv"
