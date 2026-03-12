#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
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
BASE_LOGDIR=${BASE_LOGDIR:?BASE_LOGDIR is required}
STRUCTURED_DIR="$BASE_LOGDIR/structured"
SEED_DIR="$STRUCTURED_DIR/seed_${SEED}"
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
PROBE_EXTRA_ARGS=${PROBE_EXTRA_ARGS:-}

mkdir -p "$SEED_DIR"
cd "$ROOT_DIR"

train_extra=()
probe_extra=()
if [ -n "$TRAIN_EXTRA_ARGS" ]; then
  read -r -a train_extra <<<"$TRAIN_EXTRA_ARGS"
fi
if [ -n "$PROBE_EXTRA_ARGS" ]; then
  read -r -a probe_extra <<<"$PROBE_EXTRA_ARGS"
fi

train_cmd=(
  "$PYTHON_BIN"
  train.py
  env=atari100k
  +exp=phase2_structured
  "logdir=$SEED_DIR"
  "seed=$SEED"
  device=cuda:0
  buffer.device=cuda:0
  buffer.storage_device=cpu
  batch_size=4
  batch_length=32
  env.env_num=1
  env.eval_episode_num=20
  env.steps=50000
  env.train_ratio=32
  env.task=atari_alien
  trainer.eval_episode_num=20
  trainer.sample_eval_episode_num=5
  trainer.eval_every=5000
  trainer.save_every=3000
  trainer.eval_gap_checkpoint_threshold=100
  trainer.eval_drop_checkpoint_ratio=0.5
  trainer.eval_drop_checkpoint_sample_ratio=0.75
  trainer.pretrain=0
  trainer.update_log_every=1000
  model.compile=False
)
if [ "${#train_extra[@]}" -gt 0 ]; then
  train_cmd+=("${train_extra[@]}")
fi

{
  print_git_run_metadata "$ROOT_DIR" "seed ${SEED} launch"
  echo "[$(date '+%F %T')] seed ${SEED}: train start"
  "${train_cmd[@]}"
} |& tee "$SEED_DIR/run.out"

echo "[$(date '+%F %T')] seed ${SEED}: train done"
if [ -f "$SEED_DIR/latest.pt" ]; then
  "$PYTHON_BIN" scripts/eval_checkpoint_policy_gap.py \
    "$SEED_DIR/latest.pt" \
    --device cuda:0 \
    --eval-episodes 20 \
    --repeats 3 \
    --output "$SEED_DIR/latest_mode_vs_sample_eval_20ep_3x.json" \
    "${probe_extra[@]}" \
    |& tee "$SEED_DIR/probe.out"
  echo "[$(date '+%F %T')] seed ${SEED}: probe done"
else
  echo "[$(date '+%F %T')] seed ${SEED}: latest.pt missing, skip probe" | tee "$SEED_DIR/probe.out"
fi
