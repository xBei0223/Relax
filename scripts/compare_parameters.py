import argparse
import json
from pathlib import Path

import torch


def records(path: Path) -> dict:
    result = {}
    for file in sorted(path.glob("parameters_rank*.jsonl")):
        rank = file.stem.removeprefix("parameters_rank")
        for line in file.read_text().splitlines():
            record = json.loads(line)
            key = (rank, record["step"], record["kind"], record["name"])
            if key in result:
                raise ValueError(f"Duplicate parameter record: {key}")
            result[key] = record
    if not result:
        raise ValueError(f"Missing parameter records: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = records(args.reference), records(args.candidate)
    if reference.keys() != candidate.keys():
        raise ValueError("Different parameter record keys")
    compared = []
    tensor_files = {}
    for key, expected in reference.items():
        actual = candidate[key]
        rank, step, kind, name = key
        for field in ("present", "shape", "global_shape", "dtype"):
            if expected.get(field) != actual.get(field):
                raise ValueError(f"Different metadata: {key}/{field}")
        identical = expected.get("sha256") == actual.get("sha256")
        result = {"rank": rank, "step": step, "kind": kind, "name": name, "identical": identical}
        if not identical:
            filename = f"{kind}_step{step:07d}_rank{rank}.pt"
            for directory in (args.reference, args.candidate):
                path = directory / filename
                if not path.exists():
                    result["requires_rerun"] = True
                    break
                if path not in tensor_files:
                    tensor_files[path] = torch.load(path, map_location="cpu", weights_only=True)
            else:
                left = tensor_files[args.reference / filename][name].double()
                right = tensor_files[args.candidate / filename][name].double()
                difference = left - right
                norm = float(left.norm())
                error = float(difference.norm())
                result.update(
                    relative_l2=error / norm if norm else None,
                    max_absolute=float(difference.abs().max()) if difference.numel() else 0,
                    finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                    within_tolerance=error <= 0.01 * norm,
                )
        compared.append(result)
    report = {"records": len(compared), "identical": sum(item["identical"] for item in compared), "comparisons": compared}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}))
    if report["identical"] != report["records"]:
        raise ValueError("Parameter parity differs; inspect the saved report")


if __name__ == "__main__":
    main()
