import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(run: Path) -> dict:
    dimensions = Counter()
    transfer_bytes = Counter()
    ranks = Counter()
    for path in sorted((run / "baseline_stats").glob("rank_*_steps.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            for micro in record["micros"]:
                height, width = micro["height"], micro["width"]
                if min(height, width) <= 0:
                    raise ValueError(f"Invalid dimensions: {path}")
                dimensions[(height, width)] += 1
                transfer_bytes[micro["transfers"]["target_input_latents"]["bytes"]] += 1
                ranks[record["rank"]] += 1
    if set(ranks) != {0, 1}:
        raise ValueError(f"Expected both training ranks: {run}")
    return {
        "run": run.name,
        "micro_batches_by_rank": dict(ranks),
        "target_dimensions": [{"height": shape[0], "width": shape[1], "count": count} for shape, count in sorted(dimensions.items())],
        "target_transfer_bytes": [{"bytes": size, "count": count} for size, count in sorted(transfer_bytes.items())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [summarize(run) for run in args.runs]
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
