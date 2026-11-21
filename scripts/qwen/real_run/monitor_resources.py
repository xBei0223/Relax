from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from pathlib import Path

import psutil


FIELDS = ("time", "gpu", "memory_used_mib", "utilization_pct", "host_used_bytes", "host_available_bytes",
          "ray_worker_rss_kib", "shm_used_bytes", "experiments_disk_used_bytes")


def sample(directory: Path) -> list[dict]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True,
    )
    host = psutil.virtual_memory()
    rss = sum(process.info["memory_info"].rss for process in psutil.process_iter(["cmdline", "memory_info"])
              if any("ray::" in item for item in (process.info["cmdline"] or [])))
    common = {"time": time.time(), "host_used_bytes": host.used, "host_available_bytes": host.available,
              "ray_worker_rss_kib": rss // 1024, "shm_used_bytes": psutil.disk_usage("/dev/shm").used,
              "experiments_disk_used_bytes": psutil.disk_usage(directory).used}
    return [{**common, "gpu": int(row[0]), "memory_used_mib": int(row[1]), "utilization_pct": int(row[2])}
            for row in csv.reader(io.StringIO(output))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    if args.interval <= 0 or (args.samples is not None and args.samples <= 0):
        parser.error("Sampling interval and count must be positive")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        count = 0
        while args.samples is None or count < args.samples:
            writer.writerows(sample(args.output.parent))
            stream.flush()
            count += 1
            if args.samples is None or count < args.samples:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
