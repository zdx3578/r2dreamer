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

# Keep per-process CPU thread fan-out under control when multiple runs share one GPU.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$STRUCTURED_DIR"
cd "$ROOT_DIR"

printf '%s\n' $SEEDS | xargs -n 1 -P "$MAX_PARALLEL" -I '{}' \
  env \
    PYTHON="$PYTHON_BIN" \
    BASE_LOGDIR="$BASE_LOGDIR" \
    SEED='{}' \
    TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS" \
    PROBE_EXTRA_ARGS="$PROBE_EXTRA_ARGS" \
    PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1} \
    "$ROOT_DIR/scripts/run_phase2_structured_alien_repro_one.sh"

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
