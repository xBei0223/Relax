from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [("qwen-smoke", "smoke"), ("qwen-parity-off", "parity-off"), ("qwen-parity-on", "parity-on")]
    groups = {kind: [] for kind in ("baseline-plain", "cache-plain", "baseline-tracker")}
    for repeat in (1, 2, 3):
        for kind in groups:
            name = f"qwen-{kind}-r{repeat}"
            groups[kind].append(name)
            runs.append((name, kind))
    runs.append(("qwen-cache-tracker-r1", "cache-tracker"))
    (args.output / "campaign.json").write_text(json.dumps(runs, indent=2) + "\n")
    (args.output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    for name, kind in runs:
        subprocess.run([sys.executable, str(Path(__file__).with_name("run_experiment.py")),
                        "--repo", str(args.repo), "--model", str(args.model), "--data", str(args.data),
                        "--output", str(args.output / name), "--kind", kind, "--address", args.address], check=True)


if __name__ == "__main__":
    main()
