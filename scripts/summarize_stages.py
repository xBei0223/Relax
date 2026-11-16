import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("Empty measurement window")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": statistics.quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0],
    }


def summarize(directory: Path, excluded: set[int]) -> dict:
    summary = json.loads((directory / "baseline_stats/baseline_summary.json").read_text())
    records = []
    for path in sorted((directory / "baseline_stats").glob("rank_*_steps.jsonl")):
        with path.open() as stream:
            records.extend(json.loads(line) for line in stream)
    selected = [record for record in records if record["step"] + 1 not in excluded]
    if {record["rank"] for record in selected} != {0, 1}:
        raise ValueError(f"Incomplete ranks: {directory}")
    windows = {rank: {record["step"] for record in selected if record["rank"] == rank} for rank in (0, 1)}
    if windows[0] != windows[1]:
        raise ValueError(f"Different rank windows: {directory}")
    stages = defaultdict(list)
    target_micro = defaultdict(list)
    profiled_micro = []
    for record in records:
        step = record["step"] + 1
        for micro in record["micros"]:
            encoded = micro["encodes"].get("prefix_encode", {}).get("calls", 0)
            if encoded not in (0, 1):
                raise ValueError(f"Unexpected prefix encoding count: {directory}")
            state = "online" if encoded else "reused"
            value = micro["timings_ms"]["target_image_preprocess_ms"]
            if step not in excluded:
                target_micro[state].append(value)
            elif step != 1:
                profiled_micro.append({"rank": record["rank"], "step": step, "micro": micro["micro"] + 1, "sample_id": micro["sample_id"], "state": state, "target_image_preprocess_ms": value})
    for record in selected:
        for stage, value in record["stage_timings_ms"].items():
            stages[stage].append(value)
    stage_means = {stage: describe(values) for stage, values in stages.items()}
    for value in stage_means.values():
        value["share_of_train_time"] = value["mean"] / stage_means["train_ms"]["mean"]
    return {
        "run": directory.name,
        "excluded_steps": sorted(excluded),
        "selected_steps_per_rank": len(windows[0]),
        "sample_stats_all_steps": summary["sample_stats"],
        "encoding_stats_all_steps": summary["encoding_stats"],
        "transfer_stats_all_steps": summary["transfer_stats"],
        "stage_milliseconds_selected_steps": stage_means,
        "target_micro_milliseconds_selected_steps": {state: describe(values) for state, values in target_micro.items()},
        "excluded_micro_measurements": profiled_micro,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--exclude-steps", type=int, nargs="+", default=[1, 63, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(directory, set(args.exclude_steps)) for directory in args.runs]
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps([{key: result[key] for key in ("run", "selected_steps_per_rank", "target_micro_milliseconds_selected_steps")} for result in results], indent=1))


if __name__ == "__main__":
    main()
