#!/usr/bin/env bash
# Verify a Halo host before downloading an image or model.
set -Eeuo pipefail

if [[ ! -e /dev/kfd || ! -e /dev/dri ]]; then
  echo "ROCm GPU devices are missing: expected /dev/kfd and /dev/dri." >&2
  exit 1
fi

if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "The current user is not in the render group." >&2
  exit 1
fi

if ! command -v docker >/dev/null && ! command -v podman >/dev/null; then
  echo "Install Docker or Podman first." >&2
  exit 1
fi

if ! command -v rocminfo >/dev/null; then
  echo "rocminfo is not installed; install ROCm and verify the GPU first." >&2
  exit 1
fi

rocminfo | grep -q 'gfx1151' || {
  echo "ROCm did not report the expected gfx1151 GPU." >&2
  exit 1
}

mkdir -p "$HOME/models/vllm-huggingface"
echo "Halo host checks passed. Model cache: $HOME/models/vllm-huggingface"
echo "Use CONTAINER_ENGINE=podman when Docker is not installed."
