import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def completion_intervals(directory: Path, steps_per_epoch: int) -> dict:
    (events,) = list(directory.rglob("events.out.tfevents*"))
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    times = {event.step: event.wall_time for event in accumulator.Scalars("perf/train_seconds")}
    if set(times) != set(range(1, len(times) + 1)):
        raise ValueError(f"Incomplete step completion timestamps: {directory}")
    result = {}
    for epoch in range(len(times) // steps_per_epoch):
        start = max(1, epoch * steps_per_epoch)
        end = (epoch + 1) * steps_per_epoch
        elapsed = times[end] - times[start]
        if elapsed <= 0:
            raise ValueError(f"Invalid completion timestamps: {directory}")
        result[f"epoch{epoch + 1}"] = {"previous_completed_step": start, "last_completed_step": end, "steps": end - start, "elapsed_seconds": elapsed, "mean_interval_seconds": elapsed / (end - start)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for summary in json.loads(args.summary.read_text()):
        directory = args.root / summary["run"]
        job = json.loads((directory / "job-status.json").read_text())
        started = datetime.strptime((directory / "start-time.txt").read_text().strip()[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        if job["status"] != "SUCCEEDED" or job["end_time"] is None:
            raise ValueError(f"Expected completed training: {directory}")
        wall = job["end_time"] / 1000 - started
        train = summary["train_seconds_total"]
        if wall < train:
            raise ValueError(f"Invalid timing boundaries: {directory}")
        results.append({
            "run": directory.name,
            "start_time_seconds": started,
            "ray_start_time_seconds": job["start_time"] / 1000,
            "ray_end_time_seconds": job["end_time"] / 1000,
            "launch_to_terminal_seconds": wall,
            "ray_job_seconds": (job["end_time"] - job["start_time"]) / 1000,
            "train_loop_seconds": train,
            "time_outside_train_loops_seconds": wall - train,
            "launch_to_first_step_completion_seconds": summary["launch_to_first_step_seconds"],
            "first_step_seconds": summary["first_step_seconds"],
            "step_completion_intervals": completion_intervals(directory, summary["steps_per_epoch"]),
        })
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
