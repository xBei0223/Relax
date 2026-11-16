import argparse
import json
from pathlib import Path
from typing import Any

import numpy
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


TAGS = ("train/loss", "train/grad_norm", "train/lr")


def scalars(run: Path, start: int = 1, end: int | None = None) -> dict[str, dict[int, float]]:
    # 保存 checkpoint 的运行把事件文件写在 checkpoints/tensorboard_log 下
    (events,) = list(run.rglob("events.out.tfevents*"))
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {tag: {event.step: event.value for event in accumulator.Scalars(tag) if event.step >= start and (end is None or event.step <= end)} for tag in TAGS}


def flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(child, f"{prefix}/{key}", out)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flatten(child, f"{prefix}/{index}", out)
    else:
        out[prefix] = value


def compare_values(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError(f"Different keys: {sorted(left.keys() ^ right.keys())[:10]}")
    tensors = mismatched = 0
    max_abs = 0.0
    worst_relative = 0.0
    for key in sorted(left):
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            tensors += 1
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError(f"Different tensor metadata at {key}")
            if a.is_floating_point():
                a64, b64 = a.double(), b.double()
                if not (torch.isfinite(a64).all() and torch.isfinite(b64).all()):
                    raise ValueError(f"Non-finite tensor at {key}")
                difference = (a64 - b64).abs()
                max_abs = max(max_abs, float(difference.max()) if difference.numel() else 0.0)
                reference = float(b64.norm())
                relative = float((a64 - b64).norm()) / reference if reference else float((a64 - b64).norm())
                worst_relative = max(worst_relative, relative)
            if not torch.equal(a, b):
                mismatched += 1
        elif isinstance(a, numpy.ndarray):
            if a.dtype != b.dtype or not numpy.array_equal(a, b):
                mismatched += 1
        elif a != b:
            mismatched += 1
    return {"tensors": tensors, "mismatched": mismatched, "max_abs": max_abs, "worst_relative_l2": worst_relative}


def checkpoint(run: Path, work: Path, iteration_number: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if iteration_number is None:
        (iteration,) = sorted((run / "checkpoints" / "i2i_sft").glob("iter_*"))
    else:
        iteration = run / "checkpoints" / "i2i_sft" / f"iter_{iteration_number:07d}"
    output = work / f"{run.name}.pt"
    dcp_to_torch_save(str(iteration / "fsdp"), str(output))
    state: dict[str, Any] = {}
    flatten(torch.load(output, map_location="cpu", weights_only=False), "state", state)
    rng: dict[str, Any] = {}
    for path in sorted(iteration.glob("rng_rank*.pt")):
        flatten(torch.load(path, map_location="cpu", weights_only=False), path.stem, rng)
    output.unlink()
    return state, rng


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    reference, candidate = scalars(args.reference, args.start_step, args.end_step), scalars(args.candidate, args.start_step, args.end_step)
    metrics = {}
    for tag in TAGS:
        if reference[tag].keys() != candidate[tag].keys():
            raise ValueError(f"Different logged steps for {tag}")
        differences = [abs(candidate[tag][step] - reference[tag][step]) for step in sorted(reference[tag])]
        relative = [
            difference / abs(reference[tag][step]) if reference[tag][step] else difference
            for difference, step in zip(differences, sorted(reference[tag]), strict=True)
        ]
        metrics[tag] = {
            "steps": len(differences),
            "identical_steps": sum(difference == 0 for difference in differences),
            "max_abs": max(differences),
            "max_relative": max(relative),
        }
    reference_state, reference_rng = checkpoint(args.reference, args.work, args.iteration)
    candidate_state, candidate_rng = checkpoint(args.candidate, args.work, args.iteration)
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "metrics": metrics,
        "checkpoint": compare_values(candidate_state, reference_state),
        "rng": compare_values(candidate_rng, reference_rng),
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    if any(value["identical_steps"] != value["steps"] for value in metrics.values()) or result["checkpoint"]["mismatched"] or result["rng"]["mismatched"]:
        raise ValueError("Training parity differs; inspect the saved report")


if __name__ == "__main__":
    main()
