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

SEEDS=${SEEDS:-"0 1 2"}
BASE_LOGDIR=${BASE_LOGDIR:-"$ROOT_DIR/logdir/bench_phase1b_arc3"}
ENV_DIR=${ARC3_ENV_DIR:-/home/zdx/github/VSAHDC/ARC-AGI-3-Agents/environment_files}
REC_DIR=${ARC3_RECORDINGS_DIR:-/home/zdx/github/VSAHDC/ARC-AGI-3-Agents/recordings}

cd "$ROOT_DIR"
for seed in $SEEDS; do
  LOGDIR="$BASE_LOGDIR/seed_$seed"
  "$PYTHON_BIN" train.py \
    env=arc3_grid \
    +exp=phase1b_arc3 \
    seed="$seed" \
    device=cpu \
    buffer.device=cpu \
    buffer.storage_device=cpu \
    batch_size=1 \
    batch_length=4 \
    env.env_num=1 \
    env.eval_episode_num=0 \
    env.steps=24 \
    env.time_limit=16 \
    env.train_ratio=1 \
    env.operation_mode=offline \
    env.environments_dir="$ENV_DIR" \
    env.recordings_dir="$REC_DIR" \
    trainer.eval_episode_num=0 \
    trainer.eval_every=1000000 \
    trainer.pretrain=1 \
    trainer.update_log_every=1 \
    model.compile=False \
    logdir="$LOGDIR"
  "$PYTHON_BIN" scripts/eval_phase_gate.py "$LOGDIR/metrics.jsonl" --phase phase1b
done
