set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
here=/root/Task17/bounded_condition_cache/real_run
preparation="/experiments/a100-preparation-$1"
test -s /models/SenseNova-U1.5-8B-MoT/verified-files.json
test -s /data/MagicBrush/verified-files.json
shards=(/data/MagicBrush/data/train-*.parquet)
test "${#shards[@]}" = 4
python "$here/prepare_magicbrush.py" --shards "${shards[@]:0:1}" --limit 128 --output /data/magicbrush-128 \
    > "$preparation/data-128.log"
python "$here/prepare_magicbrush.py" --shards "${shards[@]:0:2}" --limit 256 --output /data/magicbrush-256 \
    > "$preparation/data-256.log"
python "$here/prepare_magicbrush.py" --shards "${shards[@]}" --limit 512 --output /data/magicbrush-512 \
    > "$preparation/data-512.log"
digest=$(sha256sum /data/magicbrush-128/train.json)
test "${digest%% *}" = 3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9
