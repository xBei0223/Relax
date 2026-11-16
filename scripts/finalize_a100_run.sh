set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
here=/root/Task17/bounded_condition_cache/real_run
name="$1"
experiment="$EXPERIMENTS/$name"
cd "$RELAX_ROOT"
status=$(python "$here/job_status.py" "$name" --output "$experiment/job-status.json")
test "$status" = SUCCEEDED
ray job logs "$name" > "$experiment/job.log" 2>&1
echo "$status" > "$experiment/status.txt"
mkdir -p "$experiment/tensorboard"
find /root/tensorboard_log -name 'events.out.tfevents*' -newer "$experiment/start-time.txt" -exec cp {} "$experiment/tensorboard/" \;
date '+%F %T %Z' > "$experiment/end-time.txt"
if [[ -d "$experiment/baseline_stats" ]]; then
    python -m relax.engine.sft.condition_tracker "$experiment/baseline_stats" > "$experiment/baseline-summary.log" 2>&1
fi
python "$here/check_release.py" "$experiment" --submission-id "$name" > "$experiment/release.log" 2>&1
echo "$name $status"
