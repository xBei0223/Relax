import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from compare_parameters import records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--parameters", type=int, default=588)
    parser.add_argument("--tensor-steps", nargs="+", type=int, default=[1, 33, 48, 64])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = records(args.diagnostics)
    names = {name for rank, step, kind, name in indexed if rank == "0" and step == 1 and kind == "gradient"}
    if len(names) != args.parameters:
        raise ValueError(f"Unexpected trainable parameter count: {len(names)}")
    expected = {(str(rank), step, kind, name) for rank in range(args.ranks) for step in range(1, args.steps + 1) for kind in ("gradient", "update") for name in names}
    if indexed.keys() != expected:
        raise ValueError(f"Incomplete per-parameter coverage: missing={len(expected - indexed.keys())}, extra={len(indexed.keys() - expected)}")
    present = sum(record["present"] for record in indexed.values())
    for key, record in indexed.items():
        if record["present"] and (not math.isfinite(record["l2"]) or len(record["sha256"]) != 64):
            raise ValueError(f"Invalid parameter digest or norm: {key}")
    files = []
    for rank in range(args.ranks):
        for step in args.tensor_steps:
            for kind in ("gradient", "update"):
                path = args.diagnostics / f"{kind}_step{step:07d}_rank{rank}.pt"
                tensors = torch.load(path, map_location="cpu", weights_only=True)
                expected_names = {name for name in names if indexed[(str(rank), step, kind, name)]["present"]}
                if tensors.keys() != expected_names:
                    raise ValueError(f"Incomplete tensor file: {path}")
                for name, tensor in tensors.items():
                    record = indexed[(str(rank), step, kind, name)]
                    if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
                        raise ValueError(f"Tensor metadata differs: {path}: {name}")
                    digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    if digest != record["sha256"]:
                        raise ValueError(f"Tensor content differs from the recorded digest: {path}: {name}")
                files.append({"path": str(path), "tensors": len(tensors), "bytes": path.stat().st_size})
                del tensors
    report = {"diagnostics": str(args.diagnostics), "steps": args.steps, "ranks": args.ranks, "trainable_parameters": len(names), "records": len(indexed), "present_records": present, "full_tensor_files": files, "verified_full_tensors": sum(file["tensors"] for file in files)}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "full_tensor_files"}, indent=1))


if __name__ == "__main__":
    main()
