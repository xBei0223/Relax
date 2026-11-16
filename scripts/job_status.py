import argparse
import json
from pathlib import Path

from ray.job_submission import JobSubmissionClient


parser = argparse.ArgumentParser()
parser.add_argument("submission_id")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
job = JobSubmissionClient("http://127.0.0.1:8265").get_job_info(args.submission_id)
args.output.write_text(json.dumps(job.dict(), indent=1, default=str) + "\n")
print(job.status.value)
