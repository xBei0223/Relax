#!/bin/bash
set -euo pipefail

ROOT=/root/task17
NAME="${1:-relax-task17-qwen}"
IMAGE="relaxrl:task17-qwen"

mkdir -p "$ROOT/qwen/experiments" "$ROOT/qwen/tmp" "$ROOT/work/tmp"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME"
fi

docker run -d --name "$NAME" --gpus all --ipc=host --network host \
    -e TMPDIR=/qwen/tmp \
    -e TZ=Asia/Shanghai \
    --entrypoint sleep \
    -v "$ROOT/Relax:/root/Relax" \
    -v "$ROOT/Task17:/root/Task17" \
    -v "$ROOT/qwen/models:/models:ro" \
    -v "$ROOT/data:/data" \
    -v "$ROOT/qwen/experiments:/experiments" \
    -v "$ROOT/qwen:/qwen" \
    -v "$ROOT/work:/work" \
    "$IMAGE" infinity

docker exec "$NAME" nvidia-smi -L
docker exec "$NAME" free -h
