#!/usr/bin/env bash
set -uo pipefail

# Intended repository path for production systemd startup sync.
REPO_DIR="${REPO_DIR:-/home/pi/workarea/pyPiBot}"
# Allow LOG_FILE override for testing or custom deployments.
LOG_FILE="${LOG_FILE:-/home/pi/workarea/pyPiBot/log/git-sync.log}"

mkdir -p "$(dirname "$LOG_FILE")"

timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
before_sha="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

run_cmd() {
  local label="$1"
  shift

  local output status
  output="$("$@" 2>&1)"
  status=$?

  printf -v "${label}_output" '%s' "$output"
  printf -v "${label}_status" '%s' "$status"
}

run_cmd fetch git -C "$REPO_DIR" fetch --prune origin
run_cmd switch_main git -C "$REPO_DIR" switch main
run_cmd pull_ff git -C "$REPO_DIR" pull --ff-only

after_sha="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

sha_changed="no"
if [[ "$before_sha" != "$after_sha" ]]; then
  sha_changed="yes"
fi

overall_status="ok"
if [[ "$fetch_status" -ne 0 || "$switch_main_status" -ne 0 || "$pull_ff_status" -ne 0 ]]; then
  overall_status="warning"
fi

{
  echo "===== pyPiBot git sync ====="
  echo "timestamp: $timestamp"
  echo "repo_dir: $REPO_DIR"
  echo "before_sha: $before_sha"
  echo "after_sha: $after_sha"
  echo "sha_changed: $sha_changed"
  echo "overall_status: $overall_status"
  echo
  echo "[git fetch --prune origin]"
  echo "exit_code: $fetch_status"
  if [[ -n "$fetch_output" ]]; then
    echo "$fetch_output"
  else
    echo "(no output)"
  fi
  echo
  echo "[git switch main]"
  echo "exit_code: $switch_main_status"
  if [[ -n "$switch_main_output" ]]; then
    echo "$switch_main_output"
  else
    echo "(no output)"
  fi
  echo
  echo "[git pull --ff-only]"
  echo "exit_code: $pull_ff_status"
  if [[ -n "$pull_ff_output" ]]; then
    echo "$pull_ff_output"
  else
    echo "(no output)"
  fi
  echo
} >> "$LOG_FILE"

logger -t pyPiBot-git-sync \
  "status=${overall_status} before_sha=${before_sha} after_sha=${after_sha} sha_changed=${sha_changed} fetch=${fetch_status} switch=${switch_main_status} pull=${pull_ff_status}"

# Preserve best-effort startup behavior.
exit 0
