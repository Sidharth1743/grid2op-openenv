#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/sidharth1743/grid2op-hf-grpo:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/hf-grpo/Dockerfile}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"

docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_NAME}" .

if [[ "${PUSH_IMAGE}" == "1" ]]; then
  docker push "${IMAGE_NAME}"
fi

echo "Built image: ${IMAGE_NAME}"
