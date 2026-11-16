import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GIB = 1024**3


def render(run: Path, output: Path) -> dict:
    with (run / "resources.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    if not samples:
        raise ValueError(f"No resource samples: {run}")
    timestamps = sorted({float(row["time"]) for row in samples})
    origin = timestamps[0]
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    gpu_axis, host_axis, shm_axis, disk_axis, cache_axis, entries_axis = axes.flat
    counts = {}
    for gpu in sorted({row["gpu"] for row in samples}):
        rows = [row for row in samples if row["gpu"] == gpu]
        times = [(float(row["time"]) - origin) / 60 for row in rows]
        memory = [int(row["memory_used_mib"]) / 1024 for row in rows]
        gpu_axis.plot(times, memory, label=f"GPU {gpu}")
        counts[f"gpu_{gpu}"] = len(rows)
    # 主机指标在每张 GPU 的采样行中重复，只选取一个完整序列。
    rows = [row for row in samples if row["gpu"] == samples[0]["gpu"]]
    times = [(float(row["time"]) - origin) / 60 for row in rows]
    host_axis.plot(times, [int(row["ray_worker_rss_kib"]) / 1024**2 for row in rows], label="Ray worker RSS sum")
    host_axis.plot(times, [int(row["host_available_bytes"]) / GIB for row in rows], label="Host available")
    shm_axis.plot(times, [int(row["shm_used_bytes"]) / GIB for row in rows], label="Shared memory")
    initial_disk = int(rows[0]["experiments_disk_used_bytes"])
    disk_axis.plot(times, [(int(row["experiments_disk_used_bytes"]) - initial_disk) / GIB for row in rows], label="Filesystem used delta")
    steps_by_rank = {}
    for path in sorted((run / "baseline_stats").glob("rank_*_steps.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if not records:
            raise ValueError(f"No step records: {path}")
        rank = records[0]["rank"]
        steps = [record["step"] + 1 for record in records]
        if len(set(steps)) != len(steps) or steps != sorted(steps):
            raise ValueError(f"Invalid step order: {path}")
        steps_by_rank[str(rank)] = len(steps)
        if "condition_cache" in records[0]:
            caches = [record["condition_cache"] for record in records]
            cache_axis.plot(steps, [cache["charged_bytes"] / GIB for cache in caches], label=f"Rank {rank}")
            entries_axis.plot(steps, [cache["entries"] for cache in caches], label=f"Rank {rank}")
    for axis, title in zip(axes.flat, ("GPU memory", "Host memory", "Shared memory", "Filesystem change", "Cache charged storage", "Cache entries"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Elapsed minutes" if axis in (gpu_axis, host_axis, shm_axis, disk_axis) else "Training step")
        axis.set_ylabel("Entries" if axis is entries_axis else "GiB")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
        else:
            message = "Cache disabled" if steps_by_rank else "No cache step records"
            axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        for line in axis.lines:
            if not all(math.isfinite(float(value)) for value in line.get_ydata()):
                raise ValueError(f"Non-finite resource series: {run}")
    figure.suptitle(run.name)
    paths = []
    for extension in ("svg", "png"):
        path = output / f"{run.name}-resources.{extension}"
        figure.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(figure)
    return {"run": run.name, "samples_by_gpu": counts, "steps_by_rank": steps_by_rank, "duration_seconds": timestamps[-1] - origin, "artifacts": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [render(run, args.output) for run in args.runs]
    (args.output / "resource-curves.json").write_text(json.dumps(reports, indent=1) + "\n")
    print(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
