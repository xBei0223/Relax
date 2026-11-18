#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -eo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the worker-visible Qwen-Image-Edit-2509 directory}"
: "${DATA_ROOT:?Set DATA_ROOT to the training image directory}"
: "${METADATA_PATH:?Set METADATA_PATH to paired image JSON metadata}"
: "${DATASET_SIZE:?Set DATASET_SIZE to the exact number of metadata rows}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-lora}"
GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$GPUS}"
if (( GPUS <= 0 || GPUS_PER_NODE <= 0 || GLOBAL_BATCH_SIZE <= 0 || GPUS % GPUS_PER_NODE != 0 || GLOBAL_BATCH_SIZE % GPUS != 0 )); then
    echo "GPUS must be divisible by GPUS_PER_NODE; GLOBAL_BATCH_SIZE must be divisible by GPUS" >&2
    exit 1
fi
case "$MODE" in
    full) LORA_RANK=0; LR="${LR:-1e-6}" ;;
    lora) LORA_RANK="${LORA_RANK:-32}"; LR="${LR:-1e-4}" ;;
    *) echo "MODE must be full or lora" >&2; exit 1 ;;
esac
export DATA_ROOT DATASET_SIZE
I2I_CONFIG=$(python3 "$SCRIPT_DIR/qwen_image_edit_sft_config.py")
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]] && ! timeout 5 ray status >/dev/null 2>&1; then
    source "$SCRIPT_DIR/../../entrypoint/local.sh"
fi
set -u
EXTRA_ARGS=()
if [[ -n "${LOAD_DIR:-}" ]]; then EXTRA_ARGS+=(--load "$LOAD_DIR"); fi
if [[ -n "${SAVE_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-5}")
fi
SUBMIT_ARGS=(--no-wait --address "${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}")
if [[ -n "${JOB_ID:-}" ]]; then SUBMIT_ARGS+=(--submission-id "$JOB_ID"); fi
if [[ -n "${WORKING_DIR:-}" ]]; then SUBMIT_ARGS+=(--working-dir "$WORKING_DIR"); fi
DRIVER_ENV=()
if [[ -n "${DRIVER_PYTHONPATH:-}" ]]; then DRIVER_ENV=(env "PYTHONPATH=$DRIVER_PYTHONPATH"); fi
ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json "${RUNTIME_ENV_JSON:-"{}"}" -- \
    "${DRIVER_ENV[@]}" python3 -m relax.entrypoints.train \
    --train-backend fsdp --loss-type sft --generation-task i2i \
    --model-adapter-path relax.models.qwen_image.sft.QwenImageEditSFTAdapter \
    --model-path "$MODEL_PATH" --prompt-data "$METADATA_PATH" --i2i-sft-config "$I2I_CONFIG" \
    --resource "{\"actor\": [1, $GPUS], \"sft\": [1, 0]}" \
    --actor-num-nodes "$((GPUS / GPUS_PER_NODE))" --actor-num-gpus-per-node "$GPUS_PER_NODE" \
    --num-gpus-per-node "$GPUS_PER_NODE" \
    --fsdp-trainable-mode "$MODE" --fsdp-master-dtype fp32 \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_RANK" \
    --micro-batch-size 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --rollout-batch-size "$GLOBAL_BATCH_SIZE" \
    --n-samples-per-prompt 1 --num-rollout "${STEPS:-5}" --seed "${SEED:-42}" \
    --lr "$LR" --lr-warmup-iters "${LR_WARMUP_ITERS:-200}" --weight-decay 0.01 \
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0 \
    "${EXTRA_ARGS[@]}" "$@"
