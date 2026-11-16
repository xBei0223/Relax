import argparse
import ast
import csv
import json
import re
from pathlib import Path


CACHE_LINE = re.compile(r"i2i condition cache rank=(\d+) stats=(\{.*\})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        rows = list(csv.DictReader(run.joinpath("resources.csv").open()))
        gpu_peak = {}
        for row in rows:
            gpu_peak[row["gpu"]] = max(gpu_peak.get(row["gpu"], 0), int(row["memory_used_mib"]))
        start, end = int(rows[0]["experiments_disk_used_bytes"]), int(rows[-1]["experiments_disk_used_bytes"])
        cache = {}
        for line in run.joinpath("job.log").read_text(errors="replace").splitlines():
            match = CACHE_LINE.search(line)
            if match:
                stats = ast.literal_eval(match.group(2))
                cache[match.group(1)] = {
                    "peak_charged_bytes": stats["peak_charged_bytes"],
                    "entries_at_close": stats["entries"],
                    "charged_bytes_at_close": stats["charged_bytes"],
                }
        results.append(
            {
                "run": run.name,
                "samples": len(rows),
                "gpu_peak_used_mib": gpu_peak,
                "host_peak_used_bytes": max(int(row["host_used_bytes"]) for row in rows),
                "host_min_available_bytes": min(int(row["host_available_bytes"]) for row in rows),
                "ray_worker_peak_rss_kib": max(int(row["ray_worker_rss_kib"]) for row in rows),
                "shm_peak_used_bytes": max(int(row["shm_used_bytes"]) for row in rows),
                "shm_last_used_bytes": int(rows[-1]["shm_used_bytes"]),
                "disk_used_delta_bytes": end - start,
                "cache": cache,
            }
        )
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
