import argparse
import json
from pathlib import Path

from check_release import snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = snapshot(args.experiment, "http://127.0.0.1:8265")
    current["cleanup_method"] = "serve shutdown -y"
    current["checks"] = {
        "job_actors": all(actor["state"] == "DEAD" for actor in current["actors"] if actor["job_id"] == args.job_id),
        "placement_groups": all(group["state"] == "REMOVED" for group in current["placement_groups"] if group["creator_job_id"] == args.job_id),
        "compute_processes": not current["compute_processes"],
        "gpu_memory": all(float(gpu["memory.used"]) <= 64 for gpu in current["gpus"]),
    }
    args.output.write_text(json.dumps(current, indent=1) + "\n")
    if not all(current["checks"].values()):
        raise RuntimeError("Cleanup snapshot contains active resources")
    print(json.dumps(current["checks"]))


if __name__ == "__main__":
    main()
