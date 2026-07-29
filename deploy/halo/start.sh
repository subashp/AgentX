#!/usr/bin/env bash
# Start the local Qwen service and Web UI gateway, then wait for readiness.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
STATE_DIR="${AGENTX_HALO_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/agentx/halo}"
VLLM_LAUNCHER="${AGENTX_HALO_VLLM_LAUNCHER:-$SCRIPT_DIR/start-qwen3-14b-vllm.sh}"
MODEL="${AGENTX_HALO_MODEL:-Qwen/Qwen3-14B}"
GATEWAY_ENDPOINT="${AGENTX_HALO_ENDPOINT:-http://127.0.0.1:8000/v1}"
STARTUP_TIMEOUT="${AGENTX_HALO_STARTUP_TIMEOUT:-1800}"

VLLM_PID_FILE="$STATE_DIR/vllm.pid"
GATEWAY_PID_FILE="$STATE_DIR/gateway.pid"
VLLM_LOG="$STATE_DIR/vllm.log"
GATEWAY_LOG="$STATE_DIR/gateway.log"

mkdir -p "$STATE_DIR"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python was not found as '$PYTHON'. Run setup.sh or set PYTHON." >&2
  exit 1
fi

pid_is_alive() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

launch_service() {
  local label="$1"
  local pid_file="$2"
  local log_file="$3"
  shift 3
  if pid_is_alive "$pid_file"; then
    echo "$label is already running (pid $(tr -d '[:space:]' < "$pid_file"))."
    return 0
  fi
  rm -f "$pid_file"
  echo "Starting $label; log: $log_file"
  nohup "$@" >"$log_file" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$label exited during startup. See $log_file." >&2
    return 1
  fi
}

launch_service "vLLM" "$VLLM_PID_FILE" "$VLLM_LOG" bash "$VLLM_LAUNCHER"
launch_service "Web UI gateway" "$GATEWAY_PID_FILE" "$GATEWAY_LOG" bash "$SCRIPT_DIR/start-vllm-web-gateway.sh"

echo "Waiting up to ${STARTUP_TIMEOUT}s for $GATEWAY_ENDPOINT to advertise $MODEL..."
READY_JSON="$("$PYTHON" "$SCRIPT_DIR/halo_helper.py" wait \
  --endpoint "$GATEWAY_ENDPOINT" \
  --model "$MODEL" \
  --timeout "$STARTUP_TIMEOUT" \
  --request-timeout 5 \
  --interval 2 \
  --format json)" || {
  echo "Halo services did not become ready. The processes were left running for inspection." >&2
  echo "Run: $SCRIPT_DIR/status.sh" >&2
  exit 1
}

MODEL_ID="$(printf '%s' "$READY_JSON" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["models"][0])')"
SETTINGS_JSON="$($PYTHON "$SCRIPT_DIR/halo_helper.py" settings \
  --endpoint "$GATEWAY_ENDPOINT" \
  --model "$MODEL_ID" \
  --timeout "${AGENTX_HALO_TIMEOUT:-900}")"
echo "Halo services are ready. Model: $MODEL_ID"
echo "Web UI: http://127.0.0.1:8000/"
echo "AgentX endpoint: $GATEWAY_ENDPOINT"
echo "$SETTINGS_JSON" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
action = "created" if payload["created"] else ("updated" if payload["changed"] else "already configured")
print("AgentX settings {}: {}".format(action, payload["path"]))
'
echo "Logs: $VLLM_LOG and $GATEWAY_LOG"
