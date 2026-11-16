set -euo pipefail
# 参数：direct、disable-require 或 host-driver
ROOT=/root/task17
mode="${1:-direct}"
mkdir -p "$ROOT/experiments" "$ROOT/work/tmp"
options=(-e TMPDIR=/work/tmp -e TZ=Asia/Shanghai)
case "$mode" in
    direct) ;;
    disable-require) options+=(-e NVIDIA_DISABLE_REQUIRE=1) ;;
    host-driver)
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        HOST_DRIVER="$ROOT/work/host-driver"
        mkdir "$HOST_DRIVER"
        for soname in libcuda.so.1 libcuda.so libcudadebugger.so.1 libnvidia-nvvm.so.4 libnvidia-ptxjitcompiler.so.1; do
            library="${soname%%.so*}.so.$DRIVER"
            ln -s "/usr/lib/x86_64-linux-gnu/$library" "$HOST_DRIVER/$soname"
        done
        IMAGE_LD=$(docker image inspect relaxrl:task17 --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LD_LIBRARY_PATH=//p')
        options+=(-e NVIDIA_DISABLE_REQUIRE=1 -e LD_LIBRARY_PATH="/work/host-driver:$IMAGE_LD")
        ;;
    *) echo "Unknown container mode: $mode" >&2; exit 1 ;;
esac
docker run -d --name relax-task17 --gpus all --ipc=host --network host \
    "${options[@]}" \
    --entrypoint sleep \
    -v "$ROOT/Relax:/root/Relax" \
    -v "$ROOT/Task17:/root/Task17" \
    -v "$ROOT/models:/models:ro" \
    -v "$ROOT/data:/data" \
    -v "$ROOT/experiments:/experiments" \
    -v "$ROOT/work:/work" \
    relaxrl:task17 infinity
docker exec relax-task17 nvidia-smi -L
