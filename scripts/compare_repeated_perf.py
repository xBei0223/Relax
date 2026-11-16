import argparse
import json
import math
import statistics
from pathlib import Path


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def compare(online: dict, cached: dict, tracked: dict) -> dict:
    build = cached["epochs"]["epoch1"]["sum_seconds"] - online["epochs"]["epoch1"]["sum_seconds"]
    saving = (online["epochs"]["epoch3"]["mean_step_seconds"] - cached["epochs"]["epoch3"]["mean_step_seconds"]) / 4
    return {
        "runs": {"online": online["run"], "cached": cached["run"], "tracked": tracked["run"]},
        **{f"{epoch}_throughput_gain": online["epochs"][epoch]["mean_step_seconds"] / cached["epochs"][epoch]["mean_step_seconds"] - 1 for epoch in ("epoch1", "epoch2", "epoch3")},
        "epoch1_total_overhead": cached["epochs"]["epoch1"]["sum_seconds"] / online["epochs"]["epoch1"]["sum_seconds"] - 1,
        "epoch1_build_cost_seconds": build,
        "steady_saving_per_sample_seconds": saving,
        "break_even_samples_estimate": math.ceil(build / saving) if build > 0 and saving > 0 else (0 if build <= 0 else None),
        "total_time_reduction": 1 - cached["train_seconds_total"] / online["train_seconds_total"],
        "tracker_total_overhead": tracked["train_seconds_total"] / online["train_seconds_total"] - 1,
        "tracker_epoch3_overhead": tracked["epochs"]["epoch3"]["mean_step_seconds"] / online["epochs"]["epoch3"]["mean_step_seconds"] - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = {run["run"]: run for run in json.loads(args.summary.read_text())}
    groups = json.loads(args.groups.read_text())
    selected = [[indexed[name] for name in groups[group]] for group in ("baseline-plain", "cache-plain", "baseline-tracker")]
    if {len(runs) for runs in selected} != {3}:
        raise ValueError("Expected three complete repetitions per comparison group")
    if any(run["steps"] != 96 or run["status"] != "SUCCEEDED" for runs in selected for run in runs):
        raise ValueError("Expected successful 96-step runs")
    pairs = [compare(*runs) for runs in zip(*selected, strict=True)]
    ranges = {key: describe([pair[key] for pair in pairs]) for key in pairs[0] if key not in {"runs", "break_even_samples_estimate"}}
    independent_ranges = {}
    for epoch in ("epoch1", "epoch2", "epoch3"):
        online = [run["epochs"][epoch]["mean_step_seconds"] for run in selected[0]]
        cached = [run["epochs"][epoch]["mean_step_seconds"] for run in selected[1]]
        independent_ranges[epoch] = {"minimum": min(online) / max(cached) - 1, "maximum": max(online) / min(cached) - 1}
    result = {"pairs": pairs, "paired_descriptive_ranges": ranges, "all_cross_run_throughput_ranges": independent_ranges}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
