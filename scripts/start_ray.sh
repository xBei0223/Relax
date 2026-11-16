set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export RAY_JOB_STOP_WAIT_TIME_S=60
ray start --head --num-gpus=2 --num-cpus=16 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
    --temp-dir="$RAY_TMP" --disable-usage-stats
ray status
