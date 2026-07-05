#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CROSSDSL_IMAGE:-crossdsl-kernels:cuda13.2.1}"
docker build \
  --platform linux/amd64 \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  -t "$IMAGE" "$ROOT"
echo "Built $IMAGE"
docker image inspect "$IMAGE" --format 'Image ID: {{.Id}}'
