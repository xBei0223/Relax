from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shared = Path(__file__).resolve().parents[2] / "bounded_condition_cache" / "real_run"
    campaign = json.loads((args.root / "campaign.json").read_text())
    runs = [
        str(args.root / name)
        for name, kind in campaign
        if kind in {"baseline-plain", "baseline-tracker", "cache-plain", "cache-tracker"}
    ]
    commands = [
        ("summarize_perf.py", [*runs, "--output", str(args.output / "performance.json")]),
        ("summarize_resources.py", [*runs, "--output", str(args.output / "resources.json")]),
        (
            "summarize_walltime.py",
            [
                str(args.output / "performance.json"),
                "--root",
                str(args.root),
                "--output",
                str(args.output / "walltime.json"),
            ],
        ),
        (
            "compare_repeated_perf.py",
            [
                str(args.output / "performance.json"),
                "--groups",
                str(args.root / "groups.json"),
                "--output",
                str(args.output / "repetitions.json"),
            ],
        ),
        (
            "compare_parity.py",
            [
                "--reference",
                str(args.root / "qwen-parity-off"),
                "--candidate",
                str(args.root / "qwen-parity-on"),
                "--work",
                str(args.output / "checkpoint-work"),
                "--output",
                str(args.output / "parity.json"),
            ],
        ),
    ]
    for script, arguments in commands:
        subprocess.run(
            [sys.executable, str(shared / script), *arguments], env={**os.environ, "TZ": "Asia/Shanghai"}, check=True
        )


if __name__ == "__main__":
    main()
