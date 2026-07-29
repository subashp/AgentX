#!/usr/bin/env bash
# Validate a preconfigured Halo host and prepare external AgentX settings.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
AGENTX_ENDPOINT="${AGENTX_HALO_ENDPOINT:-http://127.0.0.1:8000/v1}"
AGENTX_MODEL="${AGENTX_HALO_MODEL:-Qwen/Qwen3-14B}"
AGENTX_TIMEOUT="${AGENTX_HALO_TIMEOUT:-900}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python was not found as '$PYTHON'. Set PYTHON to the Halo Python executable." >&2
  exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Halo setup requires Python 3.11 or newer." >&2
  exit 1
fi

if [[ "${AGENTX_SKIP_HOST_CHECK:-0}" != "1" ]]; then
  "$SCRIPT_DIR/bootstrap.sh"
else
  echo "Skipping Halo GPU/container checks because AGENTX_SKIP_HOST_CHECK=1."
fi

if ! command -v agentx >/dev/null 2>&1; then
  if [[ "${AGENTX_SKIP_INSTALL:-0}" == "1" ]]; then
    echo "agentx is not installed. Run: $PYTHON -m pip install --editable '$REPO_ROOT'" >&2
    exit 1
  fi
  echo "Installing the local AgentX checkout into the selected Python environment..."
  "$PYTHON" -m pip install --editable --no-build-isolation "$REPO_ROOT"
fi

SETTINGS_JSON="$($PYTHON "$SCRIPT_DIR/halo_helper.py" settings \
  --endpoint "$AGENTX_ENDPOINT" \
  --model "$AGENTX_MODEL" \
  --timeout "$AGENTX_TIMEOUT")"

echo "$SETTINGS_JSON" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
action = "created" if payload["created"] else ("updated" if payload["changed"] else "already configured")
path = payload["path"]
model = payload["model"]
endpoint = payload["endpoint"]
print(f"AgentX settings {action}: {path}")
print(f"Private model: {model} at {endpoint}")
print("Existing public provider settings were preserved.")
'
echo
echo "Setup complete. Start the local model and Web UI with:"
echo "  $SCRIPT_DIR/start.sh"
echo "Then run:"
echo "  agentx"
