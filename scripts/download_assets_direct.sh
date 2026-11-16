set -euo pipefail
export TMPDIR=/root/task17/work/tmp
original=/root/task17/experiments/a100-preparation-20261114/hosts-before-download
cp /etc/hosts "$original"
trap 'cp "$original" /etc/hosts' EXIT
cp /root/task17/work/a100-hosts-download /etc/hosts
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    python3 /root/task17/Task17/bounded_condition_cache/real_run/download_assets.py --root /root/task17
