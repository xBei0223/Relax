set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR=/root/task17/work/tmp
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl gnupg2 rsync aria2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /root/task17/work/nvidia-gpgkey
gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /root/task17/work/nvidia-gpgkey
cp "$HERE/nvidia-container-toolkit.list" /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
cp "$HERE/docker-daemon-a100.json" /etc/docker/daemon.json
systemctl restart docker
docker pull ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6
docker tag ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6 relaxrl:task17
