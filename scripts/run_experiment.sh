set -euo pipefail
# 用法：bash run_experiment.sh <实验名> <配置文件>；每个实验使用独立目录，已存在时直接失败
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
name="$1"
config="$2"
exp="$EXPERIMENTS/$name"
mkdir "$exp"
cp "$config" "$exp/config.env"
set -a
source "$config"
set +a
: "${DATA_DIR:?}" "${DATASET_SIZE:?}" "${STEPS:?}"
export DATA_ROOT="$DATA_DIR" METADATA_PATH="$DATA_DIR/train.json"
export GPUS=2 GPUS_PER_NODE=2 JOB_ID="$name" DRIVER_PYTHONPATH="$RELAX_ROOT:$MEGATRON"
export RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1
if [[ "${BASELINE_STATS_ENABLED:-0}" == 1 ]]; then
    export BASELINE_STATS_DIR="$exp/baseline_stats"
fi
if [[ "${SAVE_CHECKPOINT:-0}" == 1 ]]; then
    export SAVE_DIR="$exp/checkpoints" SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
fi
cp "$DATA_DIR/selection.json" "$exp/data-selection.json"
git -C "$RELAX_ROOT" rev-parse HEAD > "$exp/base-commit.txt"
git -C "$RELAX_ROOT" status --porcelain > "$exp/git-status.txt"
test ! -s "$exp/git-status.txt"
git -C "$RELAX_ROOT" diff --binary > "$exp/tracked-changes.diff"
sha256sum "$exp/tracked-changes.diff" > "$exp/tracked-changes.sha256"
python -m pip freeze > "$exp/environment.txt"
nvidia-smi > "$exp/nvidia-smi.txt"
env | grep -E '^(MODE|GPUS|GPUS_PER_NODE|GLOBAL_BATCH_SIZE|STEPS|SEED|MAX_PIXELS|FIXED_PIXELS_TIERS|CONDITIONING_IMAGES|ATTENTION_IMPLEMENTATION|DETERMINISTIC_ATTENTION|PREFIX_GRADIENT_CHECKPOINTING|IMAGE_PREFETCH_.*|IMAGE_DP_.*|BASELINE_.*|CONDITION_CACHE_.*|DATA_ROOT|METADATA_PATH|DATASET_SIZE|MODEL_PATH|SAVE_DIR|LOAD_DIR|SAVE_INTERVAL|SFT_EXPERIMENT_DIAGNOSTICS|CANCEL_STEP|EXPECTED_STATUS|PROFILE_.*|RELAX_PROFILE_GZIP_LEVEL|LORA_RANK|LR)=' \
    | sort > "$exp/effective-env.txt"
serve shutdown -y > "$exp/serve-cleanup.log" 2>&1
python "$HERE/check_release.py" "$exp" --baseline
date '+%F %T %Z' > "$exp/start-time.txt"
bash "$HERE/monitor_resources.sh" "$exp/resources.csv" 5 &
monitor=$!
trap 'kill $monitor 2>/dev/null || true' EXIT
cd "$RELAX_ROOT"
extra=()
if [[ "${PROFILE_PREPROCESS:-0}" == 1 ]]; then
    case "${PROFILE_RANK:?}" in
        0|1) ;;
        *) echo "PROFILE_RANK must be 0 or 1" >&2; exit 1 ;;
    esac
    extra+=(--use-pytorch-profiler --profile-ranks "$PROFILE_RANK" --profile-with-stack --profile-step-start "${PROFILE_STEP_START:?}" --profile-step-end "${PROFILE_STEP_END:?}" --profile-target train_overall --profile-output-dir "$exp/profiler")
fi
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh \
    --fsdp-load-wave-size 1 "${extra[@]}" > "$exp/submit.log" 2>&1
python "$HERE/watch_job.py" "$name" --output "$exp" --cancel-step "${CANCEL_STEP:-0}" > "$exp/watch.log" 2>&1 &
watcher=$!
release_watcher=""
if [[ "${EXPECTED_STATUS:-SUCCEEDED}" != SUCCEEDED ]]; then
    python "$HERE/check_release.py" "$exp" --submission-id "$name" --watch-trigger > "$exp/release.log" 2>&1 &
    release_watcher=$!
fi
status=""
while true; do
    status=$(python "$HERE/job_status.py" "$name" --output "$exp/job-status.json")
    case "$status" in SUCCEEDED|FAILED|STOPPED) break ;; esac
    if ! kill -0 "$watcher" 2>/dev/null; then
        wait "$watcher"
    fi
    sleep 5
done
wait "$watcher"
ray job logs "$name" > "$exp/job.log" 2>&1
echo "$status" > "$exp/status.txt"
# MetricsService 的 TensorBoard 事件文件位于容器内 /root/tensorboard_log，复制本次实验新生成的目录
mkdir -p "$exp/tensorboard"
find /root/tensorboard_log -name 'events.out.tfevents*' -newer "$exp/start-time.txt" -exec cp {} "$exp/tensorboard/" \;
date '+%F %T %Z' > "$exp/end-time.txt"
if [[ -n "${BASELINE_STATS_DIR:-}" ]]; then
    python -m relax.engine.sft.condition_tracker "$BASELINE_STATS_DIR" > "$exp/baseline-summary.log" 2>&1
fi
if [[ -n "$release_watcher" ]]; then
    wait "$release_watcher"
else
    python "$HERE/check_release.py" "$exp" --submission-id "$name" > "$exp/release.log" 2>&1
fi
test "$status" = "${EXPECTED_STATUS:-SUCCEEDED}"
if [[ "$status" == FAILED ]]; then
    grep -q 'Task17 injected' "$exp/job.log"
fi
echo "$name $status"
