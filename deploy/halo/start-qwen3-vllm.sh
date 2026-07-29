#!/usr/bin/env bash
# Common ROCm/vLLM launcher. Run this from the Halo host terminal.
set -Eeuo pipefail

ENGINE="${CONTAINER_ENGINE:-docker}"
IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai-rocm:latest}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/models/vllm-huggingface}"
HOST_PORT="${HOST_PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
# Leave headroom for ROCm/vLLM allocations and host desktop usage.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

AUTH_ARGS=()
if [[ "${VLLM_NO_AUTH:-0}" != "1" ]]; then
  : "${VLLM_API_KEY:?Set VLLM_API_KEY to a long random value before starting.}"
  AUTH_ARGS=(--api-key "$VLLM_API_KEY")
fi

REASONING_ARGS=()
if [[ "${ENABLE_REASONING:-0}" == "1" ]]; then
  REASONING_ARGS=(--reasoning-parser deepseek_r1)
fi

for device in /dev/kfd /dev/dri; do
  [[ -e "$device" ]] || { echo "Missing GPU device: $device" >&2; exit 1; }
done

RENDER_GID="$(stat -c '%g' /dev/kfd)"
mkdir -p "$MODEL_CACHE"

exec "$ENGINE" run --rm --name agentx-qwen-vllm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --publish "127.0.0.1:${HOST_PORT}:8000" \
  --volume "$MODEL_CACHE:/root/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  "$IMAGE" \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enforce-eager \
  "${REASONING_ARGS[@]}" \
  "${AUTH_ARGS[@]}"
