set -euo pipefail
ROOT=/root/task17
HERE="$ROOT/Task17/bounded_condition_cache/real_run"
IMAGE="${IMAGE_REFERENCE:-${IMAGE_REGISTRY:-ghcr.io}/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6}"
pull_pid="$1"
run_date="$2"
preparation="$ROOT/experiments/a100-preparation-$run_date"
if [[ "$pull_pid" != ready ]]; then
    while kill -0 "$pull_pid" 2>/dev/null; do
        sleep 10
    done
fi
docker image inspect "$IMAGE" > "$preparation/image-inspect.json"
docker tag "$IMAGE" relaxrl:task17
bash "$HERE/start_container.sh" direct > "$preparation/container-direct.log" 2>&1
docker exec relax-task17 python -m pip install --no-index --no-deps \
    /work/ruff-0.15.9-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    > "$preparation/install-ruff.log" 2>&1
docker exec relax-task17 torchrun --standalone --nproc_per_node=2 \
    /root/Task17/bounded_condition_cache/real_run/check_gpu.py \
    > "$preparation/gpu-check-direct.log" 2>&1
docker exec relax-task17 env ARTIFACT_DIR="/experiments/a100-validate-$run_date" \
    bash /root/Task17/bounded_condition_cache/validate.sh
docker exec relax-task17 python -m pytest \
    /root/Task17/bounded_condition_cache/real_run/test_experiment_tools.py -q \
    --basetemp="/work/pytest-tools-$run_date" \
    --junitxml="/experiments/a100-validate-$run_date/experiment-tools.xml" \
    > "$preparation/experiment-tools.log" 2>&1
date -Is > "$preparation/runtime-validated.txt"
