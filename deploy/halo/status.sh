#!/usr/bin/env bash
# Show launcher process state and local OpenAI-compatible readiness.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
STATE_DIR="${AGENTX_HALO_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/agentx/halo}"
ENDPOINT="${AGENTX_HALO_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL="${AGENTX_HALO_MODEL:-Qwen/Qwen3-14B}"

process_status() {
  local label="$1"
  local pid_file="$2"
  local log_file="$3"
  local expected_pattern="$4"
  if [[ ! -s "$pid_file" ]]; then
    echo "$label: stopped"
    echo "  log: $log_file"
    return 1
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$pid_file")"
  local command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && [[ "$command_line" == *"$expected_pattern"* ]]; then
    echo "$label: running (pid $pid)"
    echo "  log: $log_file"
    return 0
  fi
  echo "$label: stale PID file ($pid)"
  echo "  log: $log_file"
  return 1
}

echo "Halo launcher state: $STATE_DIR"
result=0
process_status "vLLM" "$STATE_DIR/vllm.pid" "$STATE_DIR/vllm.log" "agentx-qwen-vllm" || result=1
process_status "Web UI gateway" "$STATE_DIR/gateway.pid" "$STATE_DIR/gateway.log" "vllm-web-gateway.py" || result=1

if command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Endpoint: $ENDPOINT"
  if ! "$PYTHON" "$SCRIPT_DIR/halo_helper.py" probe \
    --endpoint "$ENDPOINT" --model "$MODEL" --timeout 5 --format text; then
    result=1
  fi
else
  echo "Endpoint: not checked (Python '$PYTHON' was not found)." >&2
  result=1
fi

exit "$result"
