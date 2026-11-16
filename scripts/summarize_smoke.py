import json
import sys
from pathlib import Path


def main() -> None:
    stats = Path(sys.argv[1])
    for path in sorted(stats.glob("rank_*_steps.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            cache = record.get("condition_cache", {})
            print(
                path.name,
                "step",
                record.get("step"),
                "train_seconds",
                round(record.get("train_seconds", 0), 3),
                "memory",
                record.get("memory"),
                "cache",
                {
                    name: cache.get(name)
                    for name in (
                        "entries",
                        "charged_bytes",
                        "peak_charged_bytes",
                        "inserts",
                        "inserted_bytes",
                        "local_hits",
                        "effective_reuses",
                        "oversized_skips",
                        "evictions",
                    )
                },
            )
    for path in sorted(stats.glob("rank_*_summary.json")):
        summary = json.loads(path.read_text())
        print(path.name, "keys", sorted(summary)[:40])


if __name__ == "__main__":
    main()
