#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_DIR="${1:-/home/azumayudai/Desktop/ws_blender/space_event_blender/datasets/20261110_debris_eventnerf_davis346/scene_000}"
RUN_DIR="${2:-${REPO_ROOT}/runs/1110}"
STEPS="${STEPS:-4000}"
WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_DIR}"
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE:-event_snn_torch:2.8.0}" \
  -t event-nerf:4070 "${REPO_ROOT}"

COMMON=(
  --rm --gpus all
  --user "$(id -u):$(id -g)"
  -e "WANDB_MODE=${WANDB_MODE}"
  -e "WANDB_API_KEY=${WANDB_API_KEY:-}"
  -v "${REPO_ROOT}:/workspace"
  -v "${SCENE_DIR}:/data/scene:ro"
  -w /workspace
)

docker run "${COMMON[@]}" event-nerf:4070 event_nerf.train \
  --scene /data/scene \
  --config /workspace/configs/debris_4070.json \
  --run-dir /workspace/runs/1110 \
  --steps "${STEPS}" \
  --wandb-mode "${WANDB_MODE}"

docker run "${COMMON[@]}" event-nerf:4070 event_nerf.evaluate \
  --scene /data/scene \
  --run-dir /workspace/runs/1110
