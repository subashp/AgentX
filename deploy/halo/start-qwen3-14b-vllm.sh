#!/usr/bin/env bash
# Qwen3-14B on the local ROCm/vLLM server.  The web gateway proxies to 8001.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  MODEL="Qwen/Qwen3-14B" \
  MAX_MODEL_LEN="32768" \
  KV_CACHE_DTYPE="auto" \
  MAX_NUM_SEQS="2" \
  HOST_PORT="8001" \
  VLLM_NO_AUTH="1" \
  ENABLE_REASONING="1" \
  "$SCRIPT_DIR/start-qwen3-vllm.sh"
