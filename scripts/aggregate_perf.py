import argparse
import json
import math
import statistics
from pathlib import Path


SAMPLES_PER_STEP = 4
STEPS_PER_EPOCH = 32


def group(results: list[dict], prefix: str) -> list[dict]:
    return [result for result in results if result["run"].startswith(prefix)]


def describe(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    args = parser.parse_args()
    results = json.loads(args.summary.read_text())
    groups = {
        name: group(results, name)
        for name in ("baseline-plain", "baseline-tracker", "cache-plain", "cache-tracker")
    }
    if args.groups is not None:
        names = json.loads(args.groups.read_text())
        indexed = {result["run"]: result for result in results}
        groups = {name: [indexed[run] for run in runs] for name, runs in names.items()}
    report: dict = {}
    for name, runs in groups.items():
        report[name] = {
            "runs": [run["run"] for run in runs],
            "launch_to_first_step_seconds": describe([run["launch_to_first_step_seconds"] for run in runs]),
            "first_step_seconds": describe([run["first_step_seconds"] for run in runs]),
            "train_seconds_total": describe([run["train_seconds_total"] for run in runs]),
            **{
                epoch: describe([run["epochs"][epoch]["mean_step_seconds"] for run in runs])
                for epoch in ("epoch1", "epoch2", "epoch3")
            },
        }
    online, cached = report["baseline-plain"], report["cache-plain"]
    tracked = report["baseline-tracker"]
    gains = {
        epoch: online[epoch]["mean"] / cached[epoch]["mean"] - 1 for epoch in ("epoch1", "epoch2", "epoch3")
    }
    # 构建成本：第 1 个 epoch（无复用）缓存路径的额外耗时，含首步
    epoch1_sum = {
        name: statistics.mean(run["epochs"]["epoch1"]["sum_seconds"] for run in groups[name])
        for name in ("baseline-plain", "cache-plain")
    }
    build_cost = epoch1_sum["cache-plain"] - epoch1_sum["baseline-plain"]
    saving_per_sample = (online["epoch3"]["mean"] - cached["epoch3"]["mean"]) / SAMPLES_PER_STEP
    report["comparison"] = {
        "throughput_gain": gains,
        "total_time_reduction": 1 - cached["train_seconds_total"]["mean"] / online["train_seconds_total"]["mean"],
        "tracker_overhead_total": tracked["train_seconds_total"]["mean"] / online["train_seconds_total"]["mean"] - 1,
        "tracker_overhead_epoch3": tracked["epoch3"]["mean"] / online["epoch3"]["mean"] - 1,
        "epoch1_build_cost_seconds": build_cost,
        "steady_saving_per_sample_seconds": saving_per_sample,
        "break_even_samples": (math.ceil(build_cost / saving_per_sample) if saving_per_sample > 0 else None) if build_cost > 0 else 0,
    }
    caches = [run["cache"] for run in groups["cache-plain"]]
    reuse_window = 2 * args.steps_per_epoch * 2
    report["reuse"] = {
        rank: {
            "local_hits": caches[0][rank]["local_hits"],
            "effective_reuses": caches[0][rank]["effective_reuses"],
            "group_forced_online": caches[0][rank]["group_forced_online"],
            "requests": caches[0][rank]["requests"],
            "local_hit_rate_epoch2_3": caches[0][rank]["local_hits"] / reuse_window,
            "effective_reuse_rate_epoch2_3": caches[0][rank]["effective_reuses"] / reuse_window,
            "identical_across_repeats": all(cache[rank] == caches[0][rank] for cache in caches[1:])
            or all(
                cache[rank][key] == caches[0][rank][key]
                for cache in caches[1:]
                for key in ("local_hits", "effective_reuses", "group_forced_online", "inserts", "evictions")
            ),
            "lookup_ms_per_request": caches[0][rank]["lookup_ms"] / caches[0][rank]["requests"],
            "version_check_ms_per_request": caches[0][rank]["version_check_ms"] / caches[0][rank]["requests"],
            "control_ms_per_request": caches[0][rank]["control_ms"] / caches[0][rank]["requests"],
            "restore_ms_per_reuse": caches[0][rank]["restore_ms"] / caches[0][rank]["effective_reuses"] if caches[0][rank]["effective_reuses"] else None,
        }
        for rank in sorted(caches[0])
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
