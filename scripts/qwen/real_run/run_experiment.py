from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
from ray.job_submission import JobStatus, JobSubmissionClient


HERE = Path(__file__).resolve().parent
KINDS = {
    "smoke": (4, True, True, False),
    "parity-off": (64, False, False, True),
    "parity-on": (64, True, False, True),
    "baseline-plain": (96, False, False, False),
    "baseline-tracker": (96, False, True, False),
    "cache-plain": (96, True, False, False),
    "cache-tracker": (96, True, True, False),
}
METADATA_SHA256 = "3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9"


def environment(args: argparse.Namespace) -> dict[str, str]:
    values = json.loads(args.config.read_text())
    steps, cached, tracked, parity = KINDS[args.kind]
    values.update(
        STEPS=str(steps),
        CONDITION_CACHE_ENABLED=str(int(cached)),
        BASELINE_STATS_ENABLED=str(int(tracked)),
        DETERMINISTIC_ATTENTION=str(int(parity)),
        MODEL_PATH=str(args.model.resolve()),
        DATA_ROOT=str(args.data.resolve()),
        METADATA_PATH=str((args.data / "train.json").resolve()),
        JOB_ID=args.output.name,
        RAY_DASHBOARD_ADDRESS=args.address,
        RELAX_ENTRYPOINT_MODE="ray-job",
        DRIVER_PYTHONPATH=f"{args.repo.resolve()}:/root/Megatron-LM",
        TENSORBOARD_DIR=str((args.output / "tensorboard").resolve()),
    )
    if tracked:
        values["BASELINE_STATS_DIR"] = str((args.output / "baseline_stats").resolve())
    if parity:
        values.update(SAVE_DIR=str((args.output / "checkpoints").resolve()), SAVE_INTERVAL=str(steps))
    # Workers receive the same model, cache, logging, and temp directory settings via Ray runtime_env.
    worker_env = {**values, "TMPDIR": os.environ["TMPDIR"], "TZ": "Asia/Shanghai"}
    values["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": worker_env})
    return values


def check_machine(args: argparse.Namespace) -> None:
    gpu_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = list(csv.reader(io.StringIO(gpu_output)))
    if len(gpus) < 2 or any(int(row[1]) < 79000 for row in gpus[:2]) or psutil.virtual_memory().total < 64_000_000_000:
        raise ValueError("Complete experiment requires two 80 GB GPUs and at least 64 GB host memory")
    metadata = args.data / "train.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != METADATA_SHA256:
        raise ValueError("MagicBrush metadata SHA256 differs from the specified 128-row dataset")
    status = subprocess.check_output(["git", "-C", str(args.repo), "status", "--porcelain"], text=True)
    if status:
        raise ValueError("Experiment requires a committed source version")


def run(args: argparse.Namespace) -> None:
    check_machine(args)
    values = environment(args)
    client = JobSubmissionClient(args.address)
    client.list_jobs()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(values, indent=2) + "\n")
    (args.output / "base-commit.txt").write_text(
        subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True)
    )
    (args.output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    (args.output / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (args.output / "start-time.txt").write_text(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%F %T %Z\n"))
    process_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CONDITION_CACHE_", "BASELINE_", "IMAGE_PREFETCH_"))
        and key not in {"SAVE_DIR", "SAVE_INTERVAL", "LOAD_DIR", "RUNTIME_ENV_JSON", "WORKING_DIR"}
    }
    process_env.update(values, TZ="Asia/Shanghai")
    monitor = subprocess.Popen(
        [sys.executable, str(HERE / "monitor_resources.py"), str(args.output / "resources.csv")], env=process_env
    )
    submitted = False
    terminal = False
    try:
        with (args.output / "submit.log").open("w") as stream:
            subprocess.run(
                [
                    "bash",
                    str(args.repo / "scripts/training/diffusion/run-qwen-image-edit-i2i-sft-8xgpu.sh"),
                    "--fsdp-load-wave-size",
                    "1",
                    "--use-tensorboard",
                ],
                cwd=args.repo,
                env=process_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        submitted = True
        while True:
            if monitor.poll() is not None:
                raise RuntimeError(f"Resource monitor terminated with status {monitor.returncode}")
            info = client.get_job_info(args.output.name)
            (args.output / "job-status.json").write_text(json.dumps(info.dict(), indent=2, default=str) + "\n")
            (args.output / "job.log").write_text(client.get_job_logs(args.output.name))
            terminal = info.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
            if terminal:
                break
            time.sleep(5)
        (args.output / "status.txt").write_text(info.status.value + "\n")
        if info.status != JobStatus.SUCCEEDED:
            raise RuntimeError(f"Training finished with status {info.status.value}")
        if values["BASELINE_STATS_ENABLED"] == "1":
            subprocess.run(
                [sys.executable, "-m", "relax.engine.sft.condition_tracker", values["BASELINE_STATS_DIR"]],
                env=process_env,
                cwd=args.repo,
                check=True,
            )
    finally:
        try:
            if submitted and not terminal:
                client.stop_job(args.output.name)
        finally:
            monitor.terminate()
            monitor.wait()
            (args.output / "end-time.txt").write_text(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%F %T %Z\n"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    args = parser.parse_args()
    args.repo, args.model, args.data, args.output = (
        path.resolve() for path in (args.repo, args.model, args.data, args.output)
    )
    if Path(os.environ["TMPDIR"]).resolve() == Path("/tmp"):
        raise ValueError("TMPDIR must point to the experiment workspace")
    run(args)


if __name__ == "__main__":
    main()
