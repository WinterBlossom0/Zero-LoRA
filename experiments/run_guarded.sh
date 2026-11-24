#!/usr/bin/env bash
# Run one GPU job under the shared GPU lock, with a RAM watchdog.
# Usage: experiments/run_guarded.sh <log_file> <command> [args...]
# The lock wait is at most 600 s; the job itself has no time limit.
# The watchdog stops the job when available RAM falls below 1.5 GB, because WSL crashes on RAM exhaustion.
set -euo pipefail
LOG=$1; shift
MIN_AVAILABLE_KB=$((1536 * 1024))

mkdir -p "$(dirname "$LOG")"
# setsid makes flock a process-group leader, so the watchdog stops python and all its workers at once.
setsid flock -w 600 "$HOME/.gpu.lock" "$@" >"$LOG" 2>&1 &
JOB_PID=$!

while kill -0 "$JOB_PID" 2>/dev/null; do
  available_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$available_kb" -lt "$MIN_AVAILABLE_KB" ]; then
    echo "WATCHDOG: available RAM ${available_kb} kB < ${MIN_AVAILABLE_KB} kB; stopping job ${JOB_PID}" | tee -a "$LOG"
    kill -TERM -- "-$JOB_PID" || true
    break
  fi
  sleep 5
done
wait "$JOB_PID"
