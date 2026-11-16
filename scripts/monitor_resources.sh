set -euo pipefail
# 用法：bash monitor_resources.sh <输出 CSV> <间隔秒数>；按固定间隔记录 GPU、主机内存、共享内存与磁盘
out="$1"
interval="$2"
echo "time,gpu,memory_used_mib,utilization_pct,host_used_bytes,host_available_bytes,ray_worker_rss_kib,shm_used_bytes,experiments_disk_used_bytes" > "$out"
while true; do
    now=$(date +%s.%N)
    host=$(free -b | awk '/^Mem|^内存/ {print $3","$7}')
    rss=$(ps -eo rss,cmd --no-headers | awk '/ray::/ {sum += $1} END {print sum + 0}')
    shm=$(df -B1 --output=used /dev/shm | tail -1 | tr -d ' ')
    disk=$(df -B1 --output=used /experiments | tail -1 | tr -d ' ')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=', ' read -r gpu memory utilization; do
            echo "$now,$gpu,$memory,$utilization,$host,$rss,$shm,$disk"
        done >> "$out"
    sleep "$interval"
done
