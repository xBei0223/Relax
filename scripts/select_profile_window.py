import argparse
import json
from pathlib import Path


def verify_samples(window: dict, run: Path) -> dict:
    samples = {}
    for rank, expected in window["samples"].items():
        rows = [json.loads(line) for line in (run / "baseline_stats" / f"rank_{rank}_steps.jsonl").read_text().splitlines()]
        indexed = {(row["step"] + 1, micro["micro"] + 1): micro for row in rows for micro in row["micros"]}
        samples[rank] = []
        for sample in expected:
            micro = indexed[(sample["step"], sample["micro"])]
            reused = "prefix_encode" not in micro["encodes"]
            if micro["sample_id"] != sample["sample_id"] or reused != sample["reused"]:
                raise ValueError(f"Profile sample differs from calibration: rank={rank}, step={sample['step']}, micro={sample['micro']}")
            samples[rank].append({**sample, "target_preprocess_ms": micro["timings_ms"]["target_image_preprocess_ms"]})
    return samples


def select_window(run: Path, budget_bytes: int, window_steps: int = 2) -> dict:
    if window_steps not in (1, 2):
        raise ValueError("The capture window must contain one or two steps")
    ranks = {}
    for rank in (0, 1):
        path = run / "baseline_stats" / f"rank_{rank}_steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()][:64]
        if [row["step"] for row in rows] != list(range(64)):
            raise ValueError(f"Expected 64 complete calibration steps: {path}")
        if any(row["condition_cache"]["charged_bytes"] > budget_bytes for row in rows):
            raise ValueError(f"Cache storage exceeds the supplied budget: {path}")
        ranks[rank] = rows
    candidates = []
    for start in range(32, 65 - window_steps):
        selected = {rank: rows[start : start + window_steps] for rank, rows in ranks.items()}
        samples = {}
        for rank, rows in selected.items():
            samples[str(rank)] = [
                {"step": row["step"] + 1, "micro": micro["micro"] + 1, "sample_id": micro["sample_id"], "reused": "prefix_encode" not in micro["encodes"], "target_preprocess_ms": micro["timings_ms"]["target_image_preprocess_ms"]}
                for row in rows for micro in row["micros"]
            ]
        counts = [sum(sample["reused"] for sample in values) for values in samples.values()]
        if counts[0] != counts[1]:
            raise ValueError("Different group reuse decisions across ranks")
        for rank, rows in ranks.items():
            observed = rows[start + window_steps - 1]["condition_cache"]["effective_reuses"] - rows[start - 1]["condition_cache"]["effective_reuses"]
            if observed != counts[rank] or len(samples[str(rank)]) != window_steps * 2:
                raise ValueError("Micro-batch encoding records differ from cache reuse counters")
        if 0 < counts[0] < window_steps * 2:
            occupancy = {str(rank): rows[-1]["condition_cache"]["charged_bytes"] / budget_bytes for rank, rows in selected.items()}
            candidates.append({"profile_step_start": start, "profile_step_end": start + window_steps - 1, "training_steps": list(range(start + 1, start + window_steps + 1)), "cache_occupancy_fraction": occupancy, "samples": samples})
    if not candidates:
        raise ValueError("No capture window with online and reused conditions")
    return {"calibration_run": run.name, "budget_bytes": budget_bytes, "window_steps": window_steps, "rule": "latest contiguous window in epoch two with both online and reused conditions", "candidate_windows": len(candidates), **candidates[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--budget-gib", type=int, required=True)
    parser.add_argument("--window-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_window(args.run, args.budget_gib * 1024**3, args.window_steps)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
