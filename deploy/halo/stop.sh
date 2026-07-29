#!/usr/bin/env bash
# Stop only services recorded by the local Halo launcher.
set -Eeuo pipefail

STATE_DIR="${AGENTX_HALO_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/agentx/halo}"
STOP_TIMEOUT="${AGENTX_HALO_STOP_TIMEOUT:-15}"

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  local expected_pattern="$3"
  if [[ ! -s "$pid_file" ]]; then
    echo "$label is not recorded as running."
    return 0
  fi

  local pid
  pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$label has no live process; removing stale PID file."
    rm -f "$pid_file"
    return 0
  fi

  local command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ -z "$command_line" || ( "$command_line" != *"$expected_pattern"* ) ]]; then
    echo "$label PID file does not identify the expected launcher; refusing to stop pid $pid." >&2
    return 1
  fi

  echo "Stopping $label (pid $pid)..."
  kill "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + STOP_TIMEOUT))
  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "$label did not stop within ${STOP_TIMEOUT}s; leaving it running." >&2
      return 1
    fi
    sleep 1
  done
  rm -f "$pid_file"
  echo "$label stopped."
}

mkdir -p "$STATE_DIR"
result=0
stop_pid_file "Web UI gateway" "$STATE_DIR/gateway.pid" "vllm-web-gateway.py" || result=1
stop_pid_file "vLLM" "$STATE_DIR/vllm.pid" "agentx-qwen-vllm" || result=1
exit "$result"
