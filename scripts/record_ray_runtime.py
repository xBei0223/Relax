import argparse
import json
from pathlib import Path

import psutil
from ray.util.state import list_actors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actors = [actor.asdict() for actor in list_actors(address="http://127.0.0.1:8265", detail=True, limit=10000)]
    supervisors = [actor for actor in actors if actor["class_name"] == "JobSupervisor" and actor["state"] == "ALIVE"]
    if len(supervisors) != 1:
        raise ValueError(f"Expected one live JobSupervisor: {len(supervisors)}")
    actor = supervisors[0]
    environment = psutil.Process(actor["pid"]).environ()
    expected = {"RAY_JOB_STOP_WAIT_TIME_S": "60", "RELAX_PROFILE_GZIP_LEVEL": "1", "TMPDIR": "/work/tmp"}
    actual = {name: environment.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Unexpected JobSupervisor environment: {actual}")
    report = {"actor_id": actor["actor_id"], "pid": actor["pid"], "name": actor["name"], "environment": actual}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
