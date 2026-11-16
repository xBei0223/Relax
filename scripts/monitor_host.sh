set -euo pipefail
output="$1"
while true; do
    date -Is
    nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
    docker top relax-task17 -eo pid,ppid,comm,args
    for process in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u); do
        if [[ -r "/proc/$process/status" ]]; then
            awk '/^(Name|Pid|PPid|NSpid):/' "/proc/$process/status"
        fi
    done
    sleep 5
done >> "$output" 2>&1
