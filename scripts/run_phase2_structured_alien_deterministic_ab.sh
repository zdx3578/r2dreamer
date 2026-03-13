#!/bin/bash
set -euo pipefail

# Runs the 20k Alien deterministic actor A/B matrix on the standard
# phase2_structured Atari runner:
# 1) no_prio
# 2) no_prio_mode_mix_small
# 3) no_prio_mode_mix_gap_small
#
# Intended default host split:
# - 2070: this script
# - 2080: scripts/run_phase2_structured_alien_phase1a_ab.sh
#
# Typical usage:
#   MACHINE_NAME=$(hostname -s) GPU_ID=0 MAX_PARALLEL=2 SEEDS="3 4 5" \
#   scripts/run_phase2_structured_alien_deterministic_ab.sh
#
# Useful overrides:
#   MODE_MIX_SMALL=0.2
#   MODE_GAP_WEIGHT_SMALL=0.2
#   MODE_GAP_MARGIN_SMALL=0.05

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
RUN_NAME=${RUN_NAME:-bench_atari_structured_20k_alien_deterministic_ab_$(logdir_run_tag)}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/$RUN_NAME"}
MACHINE_NAME=${MACHINE_NAME:-$(hostname -s 2>/dev/null || hostname)}

# Historical reference:
# - mode_mix was first introduced with config default 0.2 in commit 52015c4.
# - mode_gap shipped with formal config default 0.0/0.0 in commit 8e9d2be;
#   the only in-repo non-zero reference there is the unit-test example 0.2/0.05.
MODE_MIX_SMALL=${MODE_MIX_SMALL:-0.2}
MODE_MIX_START_UPDATES=${MODE_MIX_START_UPDATES:-1500}
MODE_MIX_RAMP_UPDATES=${MODE_MIX_RAMP_UPDATES:-1500}
MODE_GAP_WEIGHT_SMALL=${MODE_GAP_WEIGHT_SMALL:-0.2}
MODE_GAP_MARGIN_SMALL=${MODE_GAP_MARGIN_SMALL:-0.05}

COMMON_ARGS=(
  env.steps=20000
  buffer.prioritized=False
  model.actor_eval.repeat_calibration=True
  model.actor_eval.repeat_threshold=8
  model.actor_eval.min_top1_prob=0.5
  model.actor_eval.min_margin=0.15
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

print_git_run_metadata "$ROOT_DIR" "deterministic ab launch"

run_variant \
  no_prio \
  model.actor_imagination.mode_mix=0.0 \
  model.actor_training.mode_gap_weight=0.0 \
  model.actor_training.mode_gap_margin=0.0

run_variant \
  no_prio_mode_mix_small \
  model.actor_imagination.mode_mix="$MODE_MIX_SMALL" \
  model.actor_imagination.mode_mix_start_updates="$MODE_MIX_START_UPDATES" \
  model.actor_imagination.mode_mix_ramp_updates="$MODE_MIX_RAMP_UPDATES" \
  model.actor_training.mode_gap_weight=0.0 \
  model.actor_training.mode_gap_margin=0.0

run_variant \
  no_prio_mode_mix_gap_small \
  model.actor_imagination.mode_mix="$MODE_MIX_SMALL" \
  model.actor_imagination.mode_mix_start_updates="$MODE_MIX_START_UPDATES" \
  model.actor_imagination.mode_mix_ramp_updates="$MODE_MIX_RAMP_UPDATES" \
  model.actor_training.mode_gap_weight="$MODE_GAP_WEIGHT_SMALL" \
  model.actor_training.mode_gap_margin="$MODE_GAP_MARGIN_SMALL"

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
