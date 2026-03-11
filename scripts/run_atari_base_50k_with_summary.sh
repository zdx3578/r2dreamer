#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/logdir_naming.sh"
BASE_LOGDIR=${BASE_LOGDIR:-$(default_versioned_logdir "$ROOT_DIR" "bench_atari_base_50k_alien")}
export BASE_LOGDIR
PYTHON_BIN=${PYTHON:-}
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

cd "$ROOT_DIR"
./scripts/benchmark_atari_base_50k.sh
"$PYTHON_BIN" scripts/summarize_atari_base_50k.py "$BASE_LOGDIR"
