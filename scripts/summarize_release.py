import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def summarize(directory: Path) -> dict:
    release = json.loads((directory / "release.json").read_text())
    baseline = json.loads((directory / "resources-before.json").read_text())
    samples = read_jsonl(directory / "release-samples.jsonl")
    final = samples[-1]
    job_id = release["job_id"]
    supervisor = f"_ray_internal_job_actor_{directory.name}"
    actors = [actor for actor in final["actors"] if actor["job_id"] == job_id or actor["name"] == supervisor]
    events = [event for path in sorted((directory / "diagnostics").glob("events_rank*.jsonl")) for event in read_jsonl(path)]
    rank_by_pid = {event["pid"]: event["rank"] for event in events if event["event"] == "initialization"}
    faults = [event for event in events if event["event"] == "fault"]
    cancellation_path = directory / "cancel.json"
    cancellation = json.loads(cancellation_path.read_text()) if cancellation_path.exists() else None
    if faults:
        (fault,) = faults
        trigger = fault["time"]
        expected_status = "FAILED"
    elif cancellation is not None:
        trigger = cancellation["requested_at"]
        expected_status = "STOPPED"
    else:
        trigger = release["job"]["end_time"] / 1000
        expected_status = "SUCCEEDED"
    elapsed = final["observation_completed_at"] - trigger
    training = [actor for actor in actors if actor["class_name"] == "FSDPTrainRayActor"]
    if len(training) != 2 or not actors or any(actor["state"] != "DEAD" for actor in actors):
        raise ValueError(f"Incomplete actor release: {directory}")
    if faults and {rank_by_pid[actor["pid"]] for actor in training} != {0, 1}:
        raise ValueError(f"Incomplete rank identification: {directory}")
    if release["job"]["status"] != expected_status or not release["released"] or not all(final["checks"].values()):
        raise ValueError(f"Unexpected job or release status: {directory}")
    if not 0 <= elapsed <= 600 or abs(elapsed - release["release_time_upper_bound_seconds"]) > 0.001:
        raise ValueError(f"Inconsistent release timing: {directory}")
    resource_checks = set(final["checks"]) - {"job_terminal", "job_actors"}
    resource_observations = [
        sample for sample in samples
        if all(sample["checks"][key] for key in resource_checks)
        and all(actor["state"] == "DEAD" for actor in sample["actors"] if actor["job_id"] == job_id)
    ]
    first_resources = min(sample["observation_completed_at"] for sample in resource_observations)
    resource_elapsed = first_resources - trigger
    if not 0 <= resource_elapsed <= elapsed:
        raise ValueError(f"Inconsistent training resource timing: {directory}")
    initial_gpu = {gpu["uuid"]: float(gpu["memory.used"]) for gpu in baseline["gpus"]}
    gpu_memory = [{"uuid": gpu["uuid"], "before_mib": initial_gpu[gpu["uuid"]], "after_mib": float(gpu["memory.used"])} for gpu in final["gpus"]]
    return {
        "run": directory.name,
        "status": release["job"]["status"],
        "job_id": job_id,
        "driver_exit_code": release["job"]["driver_exit_code"],
        "trigger_time": trigger,
        "observation_completed_at": final["observation_completed_at"],
        "release_time_upper_bound_seconds": elapsed,
        "training_resource_release_upper_bound_seconds": resource_elapsed,
        "first_observation_delay_seconds": release["first_observation_delay_seconds"],
        "release_samples": len(samples),
        "checks": final["checks"],
        "gpu_memory": gpu_memory,
        "compute_processes": final["compute_processes"],
        "new_shared_memory_files": sorted(set(final["shared_memory"]) - set(baseline["shared_memory"])),
        "unexpected_experiment_files": final["unexpected_files"],
        "experiment_disk_bytes": final["experiment_disk_bytes"],
        "cancellation": cancellation,
        "faults": faults,
        "actors": [{**{key: actor[key] for key in ("actor_id", "class_name", "name", "pid", "state", "death_cause")}, "rank": rank_by_pid.get(actor["pid"])} for actor in actors],
        "placement_groups": [{key: group[key] for key in ("placement_group_id", "state")} for group in final["placement_groups"] if group["creator_job_id"] == job_id],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [summarize(directory) for directory in args.runs]
    args.output.write_text(json.dumps(reports, indent=1) + "\n")
    print(json.dumps([{key: report[key] for key in ("run", "status", "release_time_upper_bound_seconds", "checks")} for report in reports], indent=1))


if __name__ == "__main__":
    main()
