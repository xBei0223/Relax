set -euo pipefail
export TMPDIR=/root/task17/work/tmp
root=/root/task17
preparation="$root/experiments/a100-preparation-20261114"
cp "$root/work/image-layer-cache/verified-layers.json" "$preparation/server-image-verified-layers.json"
cp "$root/work/complete-image-oci/index.json" "$preparation/server-image-oci-index.json"
rm -rf "$root/work/image-layer-cache" "$root/work/image-transport" "$root/work/image-oci" "$root/work/http-download"
skopeo copy "oci:$root/work/complete-image-oci" "docker-archive:$root/work/relaxrl.tar:relaxrl:task17"
rm -rf "$root/work/complete-image-oci"
docker load -i "$root/work/relaxrl.tar"
image_id=$(docker image inspect relaxrl:task17 --format '{{.Id}}')
test "$image_id" = sha256:9b42d559b0e6ec3b20b69fc6b079ea2e256f6c9e8c898c3148082fb8d95ae7e8
rm "$root/work/relaxrl.tar"
IMAGE_REFERENCE="$image_id" bash "$root/Task17/bounded_condition_cache/real_run/prepare_a100_runtime.sh" ready 20261114
