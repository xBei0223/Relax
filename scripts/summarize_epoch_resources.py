import argparse
import json
import statistics
from pathlib import Path


MEMORY_KEYS = ("step_peak_allocated_bytes", "step_peak_reserved_bytes", "cpu_rss_bytes")
CACHE_COUNTERS = ("requests", "effective_reuses", "local_hits", "evictions", "inserts")


def summarize(run: Path, steps_per_epoch: int, epochs: int, budget_bytes: int) -> dict:
    results = {}
    for path in sorted((run / "baseline_stats").glob("rank_*_steps.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        (rank,) = {record["rank"] for record in records}
        if [record["step"] for record in records] != list(range(steps_per_epoch * epochs)):
            raise ValueError(f"Incomplete training steps: {path}")
        if any(record["epoch"] != record["step"] // steps_per_epoch for record in records):
            raise ValueError(f"Inconsistent epoch identifiers: {path}")
        cached = "condition_cache" in records[0]
        if any(("condition_cache" in record) != cached for record in records):
            raise ValueError(f"Inconsistent cache records: {path}")
        rows = []
        for epoch in range(epochs):
            selected = records[epoch * steps_per_epoch:(epoch + 1) * steps_per_epoch]
            memory = {}
            for key in MEMORY_KEYS:
                values = [record["memory"][key] for record in selected]
                if any(value < 0 for value in values):
                    raise ValueError(f"Negative memory measurement: {path}")
                memory[key] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
            row = {"epoch": epoch + 1, "steps": len(selected), "memory": memory}
            if cached:
                caches = [record["condition_cache"] for record in selected]
                if any(not 0 <= cache["charged_bytes"] <= cache["peak_charged_bytes"] <= budget_bytes for cache in caches):
                    raise ValueError(f"Cache capacity exceeded: {path}")
                before = records[epoch * steps_per_epoch - 1]["condition_cache"] if epoch else dict.fromkeys(CACHE_COUNTERS, 0)
                counters = {key: caches[-1][key] - before[key] for key in CACHE_COUNTERS}
                if any(value < 0 for value in counters.values()):
                    raise ValueError(f"Decreasing cache counter: {path}")
                row["cache"] = {
                    "charged_bytes_min": min(cache["charged_bytes"] for cache in caches),
                    "charged_bytes_max": max(cache["charged_bytes"] for cache in caches),
                    "peak_charged_bytes": caches[-1]["peak_charged_bytes"],
                    "entries_min": min(cache["entries"] for cache in caches),
                    "entries_max": max(cache["entries"] for cache in caches),
                    "group_reuse_rate": counters["effective_reuses"] / counters["requests"],
                    "counters": counters,
                }
            rows.append(row)
        results[str(rank)] = rows
    if set(results) != {"0", "1"}:
        raise ValueError(f"Expected both training ranks: {run}")
    return {"run": run.name, "steps_per_epoch": steps_per_epoch, "epochs": epochs, "budget_bytes": budget_bytes, "ranks": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--budget-gib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps_per_epoch, args.epochs, args.budget_gib) <= 0:
        raise ValueError("Epoch and budget arguments must be positive")
    reports = [summarize(run, args.steps_per_epoch, args.epochs, args.budget_gib * 1024**3) for run in args.runs]
    args.output.write_text(json.dumps(reports, indent=1) + "\n")
    print(json.dumps({"runs": len(reports), "epochs_per_rank": args.epochs, "steps_per_epoch": args.steps_per_epoch, "capacity_checks_passed": True}))


if __name__ == "__main__":
    main()
