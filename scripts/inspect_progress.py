import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/experiments"))
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    journal = args.root / f"a100-orchestration-{args.date}"
    configuration = max(journal.glob("a100-*.json"), key=lambda path: path.stat().st_mtime_ns)
    name = json.loads(configuration.read_text())["name"]
    run = args.root / name
    status = (run / "status.txt").read_text().strip() if (run / "status.txt").exists() else "RUNNING"
    log = (run / "job-stream.log").read_text() if (run / "job-stream.log").exists() else ""
    steps = [int(value) for value in re.findall(r"i2i SFT step=(\d+)\s", log)]
    report = {"run": name, "status": status, "last_logged_step": max(steps, default=0), "ranks": {}}
    for path in sorted((run / "baseline_stats").glob("rank_*_steps.jsonl")):
        # 读取已经写完的 JSONL 记录，正在写入的末行留待下一次采样。
        contents = path.read_text()
        lines = contents.splitlines()
        if not contents.endswith("\n"):
            lines = lines[:-1]
        if lines:
            record = json.loads(lines[-1])
            cache = record.get("condition_cache", {})
            report["ranks"][str(record["rank"])] = {
                "step": record["step"] + 1,
                "train_seconds": record["train_seconds"],
                "step_peak_allocated_gib": record["memory"]["step_peak_allocated_bytes"] / 1024**3,
                "cache": {key: cache[key] for key in ("charged_bytes", "peak_charged_bytes", "entries", "requests", "effective_reuses", "local_hits", "evictions") if key in cache},
            }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
