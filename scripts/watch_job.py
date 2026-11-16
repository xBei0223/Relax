import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from ray.job_submission import JobSubmissionClient


async def monitor(args: argparse.Namespace) -> None:
    client = JobSubmissionClient(args.address)
    pattern = re.compile(r"i2i SFT step=(\d+)\s")
    buffer = ""
    with (args.output / "job-stream.log").open("w") as stream:
        async for chunk in client.tail_job_logs(args.submission_id):
            stream.write(chunk)
            stream.flush()
            buffer += chunk
            lines = buffer.splitlines(keepends=True)
            buffer = ""
            for line in lines:
                if not line.endswith(("\n", "\r")):
                    buffer = line
                    continue
                match = pattern.search(line)
                if args.cancel_step and match and int(match.group(1)) >= args.cancel_step:
                    requested = time.time()
                    stopped = client.stop_job(args.submission_id)
                    (args.output / "cancel.json").write_text(json.dumps({"requested_at": requested, "observed_step": int(match.group(1)), "matched_line": line, "accepted": stopped}, indent=1) + "\n")
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", default="http://127.0.0.1:8265")
    parser.add_argument("--cancel-step", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(monitor(args))


if __name__ == "__main__":
    main()
