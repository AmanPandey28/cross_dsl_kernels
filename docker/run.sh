\
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CROSSDSL_IMAGE:-crossdsl-kernels:cuda13.2.1}"
GPU_ARGS=()

if [[ "${CROSSDSL_NO_GPU:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

mkdir -p "${HOME}/.cache/crossdsl"
exec docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/home/developer \
  -v "$ROOT:/workspace" \
  -v "${HOME}/.cache/crossdsl:/home/developer/.cache" \
  -w /workspace \
  "$IMAGE" "$@"
