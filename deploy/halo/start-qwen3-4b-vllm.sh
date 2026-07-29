#!/usr/bin/env bash
# Small-model ROCm/vLLM smoke test. Reuses the common secure local launcher.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  MODEL="Qwen/Qwen3-4B" \
  MAX_MODEL_LEN="32768" \
  KV_CACHE_DTYPE="auto" \
  MAX_NUM_SEQS="4" \
  HOST_PORT="8001" \
  VLLM_NO_AUTH="1" \
  ENABLE_REASONING="1" \
  "$SCRIPT_DIR/start-qwen3-vllm.sh"
