set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
here=/root/Task17/bounded_condition_cache/real_run
run_date="$1"
budget="$2"
journal="/experiments/a100-orchestration-$run_date"
stages=("${@:3}")
if [[ ${#stages[@]} == 0 ]]; then
    stages=(D E BC F G)
fi
for stage in "${stages[@]}"; do
    echo "START STAGE $stage"
    python "$here/run_a100.py" --date "$run_date" --budget-gib "$budget" --stage "$stage"
    echo "COMPLETE STAGE $stage"
done
python "$here/audit_a100.py" --root /experiments --date "$run_date" --budget-gib "$budget" \
    --commit "$(git -C "$RELAX_ROOT" rev-parse HEAD)" --output "$journal/experiment-audit.json"
export MPLCONFIGDIR=/work/matplotlib
curves=("/experiments/a100-A3-b$budget-$run_date")
for repeat in 1 2 3; do
    curves+=("/experiments/a100-C1-r$repeat-$run_date")
done
for state in off on; do
    curves+=("/experiments/a100-G3-$state-$run_date")
done
python "$here/plot_resources.py" "${curves[@]}" --output "$journal/resource-curves"
date -Is > "$journal/campaign-completed.txt"
