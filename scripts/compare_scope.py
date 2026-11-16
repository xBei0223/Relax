import argparse
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from summarize_perf import run_summary


CONDITION_STAGES = ("ref_image_preprocess_ms", "ref_vision_encode_ms", "prefix_encode_ms")
COUNTERS = ("requests", "local_hits", "effective_reuses", "evictions", "inserts")


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def step_times(directory: Path, start: int, end: int) -> list[float]:
    (events,) = list(directory.rglob("events.out.tfevents*"))
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    selected = {event.step: event.value for event in accumulator.Scalars("perf/train_seconds") if start <= event.step <= end}
    if set(selected) != set(range(start, end + 1)):
        raise ValueError(f"Incomplete timing window: {directory}")
    return [selected[step] for step in sorted(selected)]


def tracked_rows(directory: Path) -> dict[int, dict[int, dict]]:
    result = {}
    for path in sorted((directory / "baseline_stats").glob("rank_*_steps.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        (rank,) = {record["rank"] for record in records}
        result[rank] = {record["step"] + 1: record for record in records}
    if set(result) != {0, 1}:
        raise ValueError(f"Expected both tracked ranks: {directory}")
    return result


def condition_fraction(directory: Path, start: int, end: int) -> dict:
    ranks = tracked_rows(directory)
    selected = [records[step] for records in ranks.values() for step in range(start, end + 1)]
    stage_totals = {stage: sum(record["stage_timings_ms"][stage] for record in selected) for stage in CONDITION_STAGES}
    total = sum(record["stage_timings_ms"]["train_ms"] for record in selected)
    return {"run": directory.name, "fraction": sum(stage_totals.values()) / total, "stage_milliseconds": stage_totals, "train_milliseconds": total}


def cache_window(directory: Path, start: int, end: int, steps_per_epoch: int) -> dict:
    if (directory / "baseline_stats").is_dir():
        ranks = tracked_rows(directory)
        counters = {}
        for rank, records in ranks.items():
            before = records[start - 1]["condition_cache"] if start > 1 else {key: 0 for key in COUNTERS}
            after = records[end]["condition_cache"]
            counters[str(rank)] = {key: after[key] - before[key] for key in COUNTERS}
        source = "tracked_step_counters"
    else:
        summary = run_summary(directory, steps_per_epoch)
        if start != steps_per_epoch + 1 or end != summary["steps"]:
            raise ValueError("Untracked cache counters require the full window after epoch one")
        # 此路径仅用于条件全部不同的 128 行数据；第一个 epoch 没有命中或淘汰后的复用。
        if json.loads((directory / "data-selection.json").read_text())["limit"] != 128:
            raise ValueError("Untracked counter window is restricted to the 128-row selection")
        counters = {}
        for rank, values in summary["cache"].items():
            counters[rank] = {key: values[key] for key in COUNTERS}
            counters[rank]["requests"] -= steps_per_epoch * 2
            counters[rank]["inserts"] = None
            counters[rank]["evictions"] = None
        source = "final_hits_after_unique_first_epoch"
    for values in counters.values():
        values["local_hit_rate"] = values["local_hits"] / values["requests"]
        values["group_reuse_rate"] = values["effective_reuses"] / values["requests"]
    return {"source": source, "ranks": counters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, nargs="+", required=True)
    parser.add_argument("--cached", type=Path, nargs="+", required=True)
    parser.add_argument("--tracked-online", type=Path, nargs="+", required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_step < 1 or args.end_step < args.start_step:
        raise ValueError("Invalid comparison window")
    online = describe([statistics.mean(step_times(path, args.start_step, args.end_step)) for path in args.online])
    cached = describe([statistics.mean(step_times(path, args.start_step, args.end_step)) for path in args.cached])
    fractions = [condition_fraction(path, args.start_step, args.end_step) for path in args.tracked_online]
    windows = {path.name: cache_window(path, args.start_step, args.end_step, args.steps_per_epoch) for path in args.cached}
    reuse = statistics.mean(values["group_reuse_rate"] for window in windows.values() for values in window["ranks"].values())
    fraction = statistics.mean(value["fraction"] for value in fractions)
    predicted_reduction = reuse * fraction
    summaries = [run_summary(path, args.steps_per_epoch) for path in [*args.online, *args.cached]]
    report = {
        "start_step": args.start_step,
        "end_step": args.end_step,
        "steps_per_epoch": args.steps_per_epoch,
        "online_step_seconds": online,
        "cached_step_seconds": cached,
        "time_reduction": 1 - cached["mean"] / online["mean"],
        "throughput_gain": online["mean"] / cached["mean"] - 1,
        "condition_fraction": fraction,
        "group_reuse_rate": reuse,
        "predicted_time_reduction": predicted_reduction,
        "predicted_throughput_gain": predicted_reduction / (1 - predicted_reduction),
        "condition_measurements": fractions,
        "cache_windows": windows,
        "runs": summaries,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
