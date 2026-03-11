#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-}
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

SEEDS=${SEEDS:-"0 2"}
VARIANTS=${VARIANTS:-"dreamer r2dreamer"}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/bench_atari_base_50k_alien"}
TASK=${TASK:-atari_alien}
STEPS=${STEPS:-50000}
TRAIN_RATIO=${TRAIN_RATIO:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
BATCH_LENGTH=${BATCH_LENGTH:-32}
EVAL_EPISODES=${EVAL_EPISODES:-20}
EVAL_EVERY=${EVAL_EVERY:-5000}
DEVICE=${DEVICE:-cuda:0}
BUFFER_DEVICE=${BUFFER_DEVICE:-$DEVICE}
BUFFER_STORAGE_DEVICE=${BUFFER_STORAGE_DEVICE:-cpu}

cd "$ROOT_DIR"
for variant in $VARIANTS; do
  case "$variant" in
    dreamer|r2dreamer)
      ;;
    *)
      echo "Unsupported VARIANTS entry: $variant" >&2
      exit 1
      ;;
  esac

  for seed in $SEEDS; do
    LOGDIR="$BASE_LOGDIR/$variant/seed_$seed"
    if [ -f "$LOGDIR/latest.pt" ]; then
      echo "Skipping completed run: $LOGDIR"
      continue
    fi
    if [ -f "$LOGDIR/metrics.jsonl" ]; then
      LAST_STEP=$("$PYTHON_BIN" - <<'PY' "$LOGDIR/metrics.jsonl"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
last_step = -1
for line in path.open():
    row = json.loads(line)
    step = row.get("step")
    if step is not None:
        last_step = max(last_step, int(step))
print(last_step)
PY
)
      BACKUP_DIR="${LOGDIR}_aborted_$(date '+%Y%m%d_%H%M%S')"
      echo "Backing up incomplete run: $LOGDIR -> $BACKUP_DIR (last_step=$LAST_STEP)"
      mv "$LOGDIR" "$BACKUP_DIR"
    fi
    "$PYTHON_BIN" train.py \
      env=atari100k \
      logdir="$LOGDIR" \
      seed="$seed" \
      device="$DEVICE" \
      buffer.device="$BUFFER_DEVICE" \
      buffer.storage_device="$BUFFER_STORAGE_DEVICE" \
      buffer.prioritized=False \
      batch_size="$BATCH_SIZE" \
      batch_length="$BATCH_LENGTH" \
      env.env_num=1 \
      env.eval_episode_num="$EVAL_EPISODES" \
      env.steps="$STEPS" \
      env.train_ratio="$TRAIN_RATIO" \
      env.task="$TASK" \
      trainer.eval_episode_num="$EVAL_EPISODES" \
      trainer.eval_every="$EVAL_EVERY" \
      trainer.pretrain=0 \
      trainer.update_log_every=1000 \
      model.compile=False \
      model.rep_loss="$variant"
  done
done
