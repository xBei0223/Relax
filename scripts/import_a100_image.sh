set -euo pipefail
export TMPDIR=/root/task17/work/tmp
root=/root/task17
cache="$root/work/image-layer-cache"
preparation="$root/experiments/a100-preparation-20261114"
test -s "$cache/verified-layers.json"
"$root/work/crane" pull \
    ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6 \
    "$root/work/complete-image-oci" --format=oci --cache_path "$cache"
cp "$cache/verified-layers.json" "$preparation/server-image-verified-layers.json"
cp "$root/work/complete-image-oci/index.json" "$preparation/server-image-oci-index.json"
rm -rf "$cache"
skopeo copy "oci:$root/work/complete-image-oci" "docker-archive:$root/work/relaxrl.tar:relaxrl:task17"
rm -rf "$root/work/complete-image-oci"
gzip -1 "$root/work/relaxrl.tar"
docker load -i "$root/work/relaxrl.tar.gz"
image_id=$(docker image inspect relaxrl:task17 --format '{{.Id}}')
test "$image_id" = sha256:9b42d559b0e6ec3b20b69fc6b079ea2e256f6c9e8c898c3148082fb8d95ae7e8
rm "$root/work/relaxrl.tar.gz"
