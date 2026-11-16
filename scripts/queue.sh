set -uo pipefail
# 在服务器上运行：按队列文件顺序执行实验，每行为 "实验名 配置文件名"；单个实验失败时记录并继续
queue="$1"
R=/root/Task17/bounded_condition_cache/real_run
while read -r name config; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    echo "$(date '+%F %T %Z') start $name"
    docker exec relax-task17 bash "$R/run_experiment.sh" "$name" "$R/configs/$config" < /dev/null
    code=$?
    echo "$(date '+%F %T %Z') end $name exit=$code"
done < "$queue"
