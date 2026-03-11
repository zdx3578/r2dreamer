#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/logdir_naming.sh"
LOGDIR=${LOGDIR:-$(default_versioned_logdir "$ROOT_DIR" "phase1b_arc3")}
ENV_DIR=${ARC3_ENV_DIR:-/home/zdx/github/VSAHDC/ARC-AGI-3-Agents/environment_files}
REC_DIR=${ARC3_RECORDINGS_DIR:-/home/zdx/github/VSAHDC/ARC-AGI-3-Agents/recordings}
PYTHON_BIN=${PYTHON:-}
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" train.py \
  env=arc3_grid \
  +exp=phase1b_arc3 \
  env.operation_mode=offline \
  env.environments_dir="$ENV_DIR" \
  env.recordings_dir="$REC_DIR" \
  logdir="$LOGDIR" \
  "$@"
