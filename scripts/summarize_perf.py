import argparse
import ast
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


STEPS_PER_EPOCH = 32
CACHE_LINE = re.compile(r"i2i condition cache rank=(\d+) stats=(\{.*\})")


def run_summary(run: Path, steps_per_epoch: int = STEPS_PER_EPOCH) -> dict[str, Any]:
    # 保存 checkpoint 的运行把事件文件写在 checkpoints/tensorboard_log 下
    (events,) = list(run.rglob("events.out.tfevents*"))
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    seconds = {event.step: event.value for event in accumulator.Scalars("perf/train_seconds")}
    first_wall = min(event.wall_time for event in accumulator.Scalars("perf/train_seconds"))
    started = datetime.strptime(run.joinpath("start-time.txt").read_text().strip()[:19], "%Y-%m-%d %H:%M:%S")
    steps = sorted(seconds)
    epochs = {}
    for epoch in range(len(steps) // steps_per_epoch):
        window = [seconds[step] for step in steps[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]]
        steady = window[1:] if epoch == 0 else window
        epochs[f"epoch{epoch + 1}"] = {
            "mean_step_seconds": statistics.mean(steady),
            "stdev_step_seconds": statistics.stdev(steady),
            "sum_seconds": sum(window),
        }
    cache = {}
    for line in run.joinpath("job.log").read_text(errors="replace").splitlines():
        match = CACHE_LINE.search(line)
        if match:
            cache[match.group(1)] = ast.literal_eval(match.group(2))
    keys = (
        "requests",
        "lookups",
        "local_hits",
        "effective_reuses",
        "group_forced_online",
        "inserts",
        "evictions",
        "oversized_skips",
        "peak_entries",
        "peak_charged_bytes",
        "lookup_ms",
        "control_ms",
        "restore_ms",
        "version_check_ms",
    )
    return {
        "run": run.name,
        "status": run.joinpath("status.txt").read_text().strip(),
        "steps": len(steps),
        "steps_per_epoch": steps_per_epoch,
        "first_step_seconds": seconds[steps[0]],
        "launch_to_first_step_seconds": first_wall - started.timestamp(),
        "train_seconds_total": sum(seconds.values()),
        "epochs": epochs,
        "cache": {rank: {key: stats.get(key) for key in keys} for rank, stats in sorted(cache.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    args = parser.parse_args()
    if args.steps_per_epoch <= 1:
        raise ValueError("steps_per_epoch must exceed one")
    results = [run_summary(run, args.steps_per_epoch) for run in args.runs]
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
