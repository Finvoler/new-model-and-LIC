#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

mkdir -p logs

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_LOG="logs/launcher_${STAMP}.log"

nohup bash "$ROOT_DIR/run_lic_lightgcn.sh" > "$RUN_LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true

echo "$PID"
echo "$RUN_LOG"
