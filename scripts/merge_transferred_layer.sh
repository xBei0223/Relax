set -euo pipefail
root=/root/task17
digest=3eb73d67b204224eef46c6949a425329bd8c46669daeff43ed3557e8e527345d
source="$root/work/http-download/sha256:$digest"
target="$root/work/image-layer-cache/sha256:$digest"
test ! -e "$source.aria2"
actual=$(sha256sum "$source")
test "${actual%% *}" = "$digest"
cp "$source" "$target"
rm "$target.aria2"
cmp /etc/hosts "$root/experiments/a100-preparation-20261114/hosts-before-image-download"
