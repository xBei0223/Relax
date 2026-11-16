import argparse
import csv
import json
import math
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def expected_runs(budget: int) -> list[tuple[str, str, int, int]]:
    runs = [("A2", "SUCCEEDED", 1, 4), (f"A3-b{budget}", "SUCCEEDED", 1, 64)]
    runs.extend((f"D1-{state}", "SUCCEEDED", 1, 64) for state in ("off", "on"))
    runs.extend((f"E2-{state}", "SUCCEEDED", 33, 64) for state in ("on-on", "off-on", "on-off"))
    runs.extend((f"E1-{state}", "SUCCEEDED", 1, 64) for state in ("off", "on"))
    runs.extend((f"{group}-r{repeat}", "SUCCEEDED", 1, 96) for repeat in range(1, 4) for group in ("B2", "C1", "B1"))
    runs.append(("C2", "SUCCEEDED", 1, 96))
    runs.extend((f"{group}-r{repeat}", "STOPPED" if group == "F1" else "FAILED", 1, 40 if group == "F1" else 39) for group in ("F1", "F2", "F3") for repeat in range(1, 4))
    runs.extend((f"G1-n{size}-{state}", "SUCCEEDED", 1, size // 4 * 3) for size in (256, 512) for state in ("off", "on"))
    runs.extend((f"G2-b{value}", "SUCCEEDED", 1, 96) for value in (2, 4, 8))
    runs.extend((f"{group}-{state}", "SUCCEEDED", 1, steps) for group, steps in (("G3", 192), ("G4", 96)) for state in ("off", "on"))
    return runs


def inspect(run: Path, identifier: str, status: str, first: int, last: int, commit: str) -> dict:
    actual = (run / "status.txt").read_text().strip()
    if actual != status:
        raise ValueError(f"Unexpected status: {run.name}: {actual}")
    if (run / "base-commit.txt").read_text().strip() != commit:
        raise ValueError(f"Unexpected source commit: {run.name}")
    if (run / "git-status.txt").read_text().strip() or (run / "tracked-changes.diff").stat().st_size:
        raise ValueError(f"Source modifications recorded: {run.name}")
    release = json.loads((run / "release.json").read_text())
    if not release["released"] or not all(release["checks"].values()):
        raise ValueError(f"Resource release incomplete: {run.name}")
    trigger = release["job"]["end_time"] / 1000
    if identifier.startswith("F1"):
        trigger = json.loads((run / "cancel.json").read_text())["requested_at"]
    elif identifier.startswith(("F2", "F3")):
        faults = [json.loads(line) for path in (run / "diagnostics").glob("events_rank*.jsonl") for line in path.read_text().splitlines()]
        trigger = min(event["time"] for event in faults if event["event"] == "fault")
    observations = [json.loads(line) for line in (run / "release-samples.jsonl").read_text().splitlines()]
    released = min((sample for sample in observations if all(sample["checks"].values())), key=lambda sample: sample["time"])
    owned_actors = [actor for actor in released["actors"] if actor["job_id"] == release["job_id"] or actor["name"] == f"_ray_internal_job_actor_{run.name}"]
    if not owned_actors or any(actor["state"] != "DEAD" for actor in owned_actors):
        raise ValueError(f"Job actors remain active: {run.name}")
    # 归档文件保留秒级修改时间，下一整秒覆盖完整采样与报告写入。
    release_seconds = math.floor((run / "release.json").stat().st_mtime) + 1 - trigger
    if not 0 <= release_seconds <= 600:
        raise ValueError(f"Resource release observation exceeds 600 seconds: {run.name}: {release_seconds}")
    with (run / "resources.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    if {row["gpu"] for row in samples} != {"0", "1"}:
        raise ValueError(f"Missing GPU resource series: {run.name}")
    log = (run / "job.log").read_text()
    log_steps = sorted({int(value) for value in re.findall(r"i2i SFT step=(\d+)\s", log)})
    if identifier.startswith("F1"):
        cancellation = json.loads((run / "cancel.json").read_text())
        if not cancellation["accepted"] or cancellation["observed_step"] != 40 or 40 not in log_steps:
            raise ValueError(f"Invalid step-40 cancellation: {run.name}")
    elif log_steps != list(range(first, last + 1)):
        raise ValueError(f"Incomplete training steps: {run.name}: {log_steps}")
    if status == "FAILED":
        phase = "forward" if identifier.startswith("F2") else "backward"
        if f"Task17 injected {phase} failure step=40 micro=1 rank=0" not in log:
            raise ValueError(f"Missing injected failure: {run.name}")
    event_steps = []
    if status == "SUCCEEDED":
        (events,) = list(run.rglob("events.out.tfevents*"))
        accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
        accumulator.Reload()
        event_steps = sorted(event.step for event in accumulator.Scalars("perf/train_seconds"))
        if event_steps != list(range(first, last + 1)):
            raise ValueError(f"Incomplete TensorBoard timing: {run.name}")
    return {"run": run.name, "status": status, "logged_steps": len(log_steps), "timed_steps": len(event_steps), "resource_samples": len(samples), "release_checks": release["checks"], "release_time_upper_bound_seconds": release_seconds, "release_bound_source": "release.json modification time rounded up to the next second"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--budget-gib", type=int, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    specifications = expected_runs(args.budget_gib)
    if args.only is not None:
        unknown = set(args.only) - {row[0] for row in specifications}
        if unknown:
            raise ValueError(f"Unknown experiment identifiers: {sorted(unknown)}")
        specifications = [row for row in specifications if row[0] in args.only]
    results = [inspect(args.root / f"a100-{identifier}-{args.date}", identifier, status, first, last, args.commit) for identifier, status, first, last in specifications]
    report = {"commit": args.commit, "budget_gib": args.budget_gib, "complete_plan": args.only is None, "runs_checked": len(results), "runs": results}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=1))


if __name__ == "__main__":
    main()
