#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR=${LOGDIR:-"$ROOT_DIR/logdir/phase2_atari"}
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
  env=atari100k \
  +exp=phase2_structured \
  logdir="$LOGDIR" \
  "$@"
