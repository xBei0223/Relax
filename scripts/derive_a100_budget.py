import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for path in sorted((args.run / "baseline_stats").glob("rank_*_steps.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["step"] for row in rows] != [0, 1, 2, 3]:
            raise ValueError(f"Expected four complete A2 steps: {path}")
        rank = rows[0]["rank"]
        records[rank] = {
            "step_peak_allocated_bytes": max(row["memory"]["step_peak_allocated_bytes"] for row in rows),
            "lifetime_peak_allocated_bytes": max(row["memory"]["lifetime_peak_allocated_bytes"] for row in rows),
            "cpu_peak_rss_bytes": max(row["memory"]["cpu_peak_rss_bytes"] for row in rows),
        }
    if set(records) != {0, 1}:
        raise ValueError("Expected both A100 ranks")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    gpus = [{"uuid": row[0].strip(), "total_bytes": int(row[1].strip()) * 1024**2} for row in csv.reader(io.StringIO(result.stdout))]
    if len(gpus) != 2:
        raise ValueError("Expected two A100 GPUs")
    peak = max(record["step_peak_allocated_bytes"] for record in records.values())
    budget_gib = (min(gpu["total_bytes"] for gpu in gpus) - peak - 4 * 1024**3) // 1024**3
    if budget_gib <= 0:
        raise ValueError(f"No positive cache budget with 4 GiB reserve: {budget_gib}")
    report = {"gpus": gpus, "ranks": records, "M0_bytes": peak, "reserve_bytes": 4 * 1024**3, "candidate_budget_gib": budget_gib, "candidate_budget_bytes": budget_gib * 1024**3}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
