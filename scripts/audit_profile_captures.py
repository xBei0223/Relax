import argparse
import hashlib
import json
from pathlib import Path

from audit_a100 import inspect
from select_profile_window import verify_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ranks", type=int, choices=(0, 1), nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    journal = args.root / f"a100-orchestration-{args.date}"
    window = json.loads((journal / "profile-window-single-step.json").read_text())
    captures = json.loads((journal / "profile-samples.json").read_text())
    results = []
    for rank in args.ranks:
        capture = captures[str(rank)]
        run = args.root / f"a100-C2-profile-single-rank{rank}-{args.date}"
        if run.name != capture["run"]:
            raise ValueError("Capture identity differs from the selected rank")
        audit = inspect(run, "C2", "SUCCEEDED", 1, 96, args.commit)
        actual_samples = verify_samples(window, run)
        if actual_samples != capture["samples"]:
            raise ValueError("Capture sample records differ from the archived records")
        (trace,) = list((run / "profiler").rglob("*.pt.trace.json.gz"))
        if f"rank{rank}_dp{rank}_tp0_pp0." not in trace.name or trace.stat().st_size != capture["trace"]["bytes"]:
            raise ValueError("Capture trace differs from the archived record")
        with trace.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        results.append({**audit, "captured_rank": rank, "samples": actual_samples, "trace": str(trace), "trace_bytes": trace.stat().st_size, "trace_sha256": digest})
    report = {"commit": args.commit, "complete_capture_set": set(args.ranks) == {0, 1}, "runs_checked": len(results), "runs": results}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=1))


if __name__ == "__main__":
    main()
