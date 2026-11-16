set -euo pipefail
export TMPDIR=/root/task17/work/tmp
root=/root/task17
original="$root/experiments/a100-preparation-20261114/hosts-before-image-download"
cmp /etc/hosts "$root/experiments/a100-preparation-20261114/hosts-before-download"
cp /etc/hosts "$original"
trap 'cp "$original" /etc/hosts' EXIT
cp "$root/work/a100-hosts-image" /etc/hosts
python3 "$root/Task17/bounded_condition_cache/real_run/download_image_layers.py" \
    --manifest "$root/experiments/a100-preparation-20261114/image-manifest.json" \
    --cache "$root/work/image-layer-cache"
