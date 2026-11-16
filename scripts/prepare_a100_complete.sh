set -euo pipefail
root=/root/task17
here="$root/Task17/bounded_condition_cache/real_run"
bash "$here/download_image_direct.sh"
bash "$here/import_a100_image.sh" > "$root/jobs/import-image.log" 2>&1
IMAGE_REFERENCE=sha256:9b42d559b0e6ec3b20b69fc6b079ea2e256f6c9e8c898c3148082fb8d95ae7e8 \
    bash "$here/prepare_a100_runtime.sh" ready 20261114
