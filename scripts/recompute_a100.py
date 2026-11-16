import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CACHE_LINE = re.compile(r"i2i condition cache rank=(\d+) stats=(\{.*\})")


def run_stats(run: Path) -> dict:
    (events,) = list(run.rglob("events.out.tfevents*"))
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    seconds = {event.step: event.value for event in accumulator.Scalars("perf/train_seconds")}
    env = dict(
        line.split("=", 1) for line in run.joinpath("effective-env.txt").read_text().splitlines() if "=" in line
    )
    per_epoch = math.ceil(int(env["DATASET_SIZE"]) / int(env["GLOBAL_BATCH_SIZE"]))
    steps = sorted(seconds)
    epochs = []
    for start in range(0, len(steps), per_epoch):
        window = [seconds[step] for step in steps[start : start + per_epoch]]
        epochs.append(statistics.mean(window[1:] if start == 0 else window))
    cache = {}
    for line in run.joinpath("job.log").read_text(errors="replace").splitlines():
        match = CACHE_LINE.search(line)
        if match:
            stats = ast.literal_eval(match.group(2))
            cache[match.group(1)] = {
                key: stats[key]
                for key in ("requests", "local_hits", "effective_reuses", "evictions", "peak_charged_bytes")
            }
    return {
        "run": run.name,
        "status": run.joinpath("status.txt").read_text().strip(),
        "steps": len(steps),
        "first_step": seconds[steps[0]],
        "total": sum(seconds.values()),
        "epochs": epochs,
        "cache": cache,
        "max_bytes": env.get("CONDITION_CACHE_MAX_BYTES"),
        "max_pixels": env.get("MAX_PIXELS"),
        "enabled": env.get("CONDITION_CACHE_ENABLED"),
    }


def main() -> None:
    root = Path(sys.argv[1])
    names = sys.argv[2:]
    results = [run_stats(root / name) for name in names]
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
