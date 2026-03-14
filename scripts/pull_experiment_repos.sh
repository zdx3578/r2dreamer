#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_HOSTS=${REMOTE_HOSTS:-"2080"}

cd "$ROOT_DIR"
echo "[$(date '+%F %T')] local: git pull --ff-only"
git pull --ff-only
echo "[$(date '+%F %T')] local: $(git rev-parse --short HEAD)"

for host in $REMOTE_HOSTS; do
  echo "[$(date '+%F %T')] ${host}: git pull --ff-only"
  ssh "$host" "cd '$ROOT_DIR' && git pull --ff-only && git rev-parse --short HEAD"
done
