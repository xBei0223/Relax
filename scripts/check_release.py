import argparse
import csv
import io
import json
import subprocess
import time
from pathlib import Path

import psutil
from ray.job_submission import JobSubmissionClient
from ray.util.state import list_actors, list_placement_groups


def query_gpu(kind: str, fields: list[str]) -> list[dict]:
    output = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [dict(zip(fields, (cell.strip() for cell in row), strict=True)) for row in csv.reader(io.StringIO(output)) if row]


def inventory(directory: Path) -> dict:
    return {str(path.relative_to(directory)): path.stat().st_size for path in directory.rglob("*") if path.is_file()}


def snapshot(experiment: Path, address: str) -> dict:
    actors = [actor.asdict() for actor in list_actors(address=address, detail=True, limit=10000, timeout=30)]
    processes = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "memory_info"]):
        info = process.info
        name = info["name"] or ""
        if "ray::" in name or "TrainRayActor" in " ".join(info["cmdline"] or []):
            processes.append({"pid": info["pid"], "ppid": info["ppid"], "name": name, "cmdline": info["cmdline"], "rss": info["memory_info"].rss})
    return {
        "time": time.time(),
        "gpus": query_gpu("gpu", ["uuid", "memory.used"]),
        "compute_processes": query_gpu("compute-apps", ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]),
        "actors": actors,
        "placement_groups": [group.asdict() for group in list_placement_groups(address=address, detail=True, limit=10000, timeout=30)],
        "processes": processes,
        "shared_memory": inventory(Path("/dev/shm")),
        "shared_memory_used": psutil.disk_usage("/dev/shm").used,
        "experiment_files": inventory(experiment),
        "experiment_disk_bytes": int(subprocess.run(["du", "-sB1", str(experiment)], check=True, capture_output=True, text=True).stdout.split()[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--submission-id")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--watch-trigger", action="store_true")
    args = parser.parse_args()
    baseline_path = args.experiment / "resources-before.json"
    if args.baseline:
        baseline_path.write_text(json.dumps(snapshot(args.experiment, args.address), indent=1) + "\n")
        return
    baseline = json.loads(baseline_path.read_text())
    client = JobSubmissionClient(args.address)
    if args.watch_trigger:
        while True:
            current = snapshot(args.experiment, args.address)
            with (args.experiment / "before-trigger-samples.jsonl").open("a") as stream:
                stream.write(json.dumps(current) + "\n")
            triggered = (args.experiment / "cancel.json").exists()
            for path in (args.experiment / "diagnostics").glob("events_rank*.jsonl"):
                triggered |= any(json.loads(line)["event"] == "fault" for line in path.read_text().splitlines())
            job = client.get_job_info(args.submission_id)
            if triggered or job.status.value in {"SUCCEEDED", "FAILED", "STOPPED"}:
                break
            time.sleep(5)
    job = client.get_job_info(args.submission_id)
    job_id = job.job_id
    initial_gpu = {gpu["uuid"]: float(gpu["memory.used"]) for gpu in baseline["gpus"]}
    initial_processes = {(item["gpu_uuid"], item["pid"]) for item in baseline["compute_processes"]}
    initial_training_pids = {item["pid"] for item in baseline["processes"]}
    started = time.time()
    trigger = started
    if job.end_time is not None:
        trigger = job.end_time / 1000
    cancellation = args.experiment / "cancel.json"
    if cancellation.exists():
        trigger = json.loads(cancellation.read_text())["requested_at"]
    for path in (args.experiment / "diagnostics").glob("events_rank*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "fault":
                trigger = min(trigger, event["time"])
    while True:
        current = snapshot(args.experiment, args.address)
        supervisor_name = f"_ray_internal_job_actor_{args.submission_id}"
        training_actors = [actor for actor in current["actors"] if (actor["job_id"] == job_id or actor["name"] == supervisor_name) and actor["state"] != "DEAD"]
        pending_processes = {(item["gpu_uuid"], item["pid"]) for item in current["compute_processes"]} - initial_processes
        gpu_released = all(float(gpu["memory.used"]) <= initial_gpu[gpu["uuid"]] + 64 for gpu in current["gpus"])
        shm_new = set(current["shared_memory"]) - set(baseline["shared_memory"])
        training_processes = [item for item in current["processes"] if item["pid"] not in initial_training_pids and "TrainRayActor" in (item["name"] + " ".join(item["cmdline"] or []))]
        artifact_directories = {"baseline_stats", "checkpoints", "diagnostics", "profiler", "tensorboard"}
        artifact_files = {"config.env", "data-selection.json", "base-commit.txt", "git-status.txt", "tracked-changes.diff", "tracked-changes.sha256", "environment.txt", "nvidia-smi.txt", "effective-env.txt", "serve-cleanup.log", "resources-before.json", "start-time.txt", "resources.csv", "submit.log", "watch.log", "job-stream.log", "cancel.json", "job-status.json", "job.log", "status.txt", "end-time.txt", "baseline-summary.log", "release.log", "release-samples.jsonl", "before-trigger-samples.jsonl", "release.json"}
        unexpected_files = [name for name in current["experiment_files"] if name not in artifact_files and Path(name).parts[0] not in artifact_directories]
        job = client.get_job_info(args.submission_id)
        groups = [group for group in current["placement_groups"] if group["creator_job_id"] == job_id and group["state"] != "REMOVED"]
        checks = {"job_terminal": job.status.value in {"SUCCEEDED", "FAILED", "STOPPED"}, "gpu_memory": gpu_released, "compute_processes": not pending_processes, "training_processes": not training_processes, "job_actors": not training_actors, "placement_groups": not groups, "shared_memory_files": not shm_new, "experiment_files": not unexpected_files}
        current["unexpected_files"] = unexpected_files
        current["checks"] = checks
        current["observation_completed_at"] = time.time()
        with (args.experiment / "release-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(current) + "\n")
        elapsed = current["observation_completed_at"] - trigger
        released = all(checks.values()) and elapsed <= args.timeout
        if released or elapsed >= args.timeout:
            report = {"released": released, "release_time_upper_bound_seconds": elapsed, "first_observation_delay_seconds": started - trigger, "checks": checks, "job_id": job_id, "job": job.dict()}
            (args.experiment / "release.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
            print(json.dumps(report, default=str))
            if not released:
                raise RuntimeError("Resources were not released within the observation interval")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
