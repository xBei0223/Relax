import argparse
import ast
import json
import re
from pathlib import Path

from compare_parity import scalars


def steps(directory: Path, rank: int) -> dict:
    path = directory / "baseline_stats" / f"rank_{rank}_steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return {record["step"] + 1: record for record in records}


def online(record: dict) -> bool:
    return len(record["micros"]) == 2 and all(
        micro["encodes"].get("prefix_encode", {}).get("calls") == 1
        and micro["encodes"].get("ref_vision_encode", {}).get("calls") == 1
        for micro in record["micros"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/experiments"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def run(name: str) -> Path:
        return args.root / f"a100-{name}-{args.date}"

    results = {}
    baseline = scalars(run("D1-off"))
    changed = scalars(run("E1-off"))
    results["encoder_update_preserves_steps_1_to_40"] = all(baseline[tag][step] == changed[tag][step] for tag in ("train/loss", "train/grad_norm") for step in range(1, 41))
    results["encoder_update_changes_loss_at_step_41"] = baseline["train/loss"][41] != changed["train/loss"][41]
    for rank in (0, 1):
        events = [json.loads(line) for line in (run("E1-on") / "diagnostics" / f"events_rank{rank}.jsonl").read_text().splitlines()]
        (event,) = [event for event in events if event["event"] == "encoder_update"]
        results[f"rank{rank}_encoder_cache_cleared"] = event["step"] == 40 and event["before"]["entries"] > 0 and event["after"]["entries"] == 0 and event["after"]["charged_bytes"] == 0
        results[f"rank{rank}_encoder_version_invalidated"] = event["after"]["version_invalidations"] > event["before"]["version_invalidations"]
        updates = steps(run("E1-on"), rank)
        cleared = updates[40]["condition_cache"]
        results[f"rank{rank}_step40_tracks_cleared_cache"] = all(cleared[field] == event["after"][field] for field in ("entries", "charged_bytes", "version_invalidations", "inserts", "effective_reuses"))
        first_updated = updates[41]["condition_cache"]
        results[f"rank{rank}_step41_online"] = online(updates[41]) and first_updated["effective_reuses"] == cleared["effective_reuses"]
        results[f"rank{rank}_step41_rebuilds_cache"] = first_updated["inserts"] - cleared["inserts"] == 2 and first_updated["entries"] == 2 and first_updated["charged_bytes"] > 0
        encoded_after_update = set()
        valid_reuses = True
        for step in range(41, 65):
            for micro in updates[step]["micros"]:
                if "prefix_encode" in micro["encodes"]:
                    encoded_after_update.add(micro["sample_id"])
                else:
                    valid_reuses &= micro["sample_id"] in encoded_after_update
        results[f"rank{rank}_post_update_reuses_have_new_encodes"] = valid_reuses
        for identifier in ("E2-on-on", "E2-off-on"):
            restored = steps(run(identifier), rank)
            first = restored[33]["condition_cache"]
            results[f"{identifier}_rank{rank}_starts_online"] = first["requests"] == 2 and first["effective_reuses"] == 0 and first["inserts"] == 2 and first["entries"] == 2 and online(restored[33])
    caches = {}
    for match in re.finditer(r"i2i condition cache rank=(\d+) stats=(\{.*\})", (run("D1-on") / "job.log").read_text()):
        caches[int(match.group(1))] = ast.literal_eval(match.group(2))
    results["D1_has_group_reuse"] = set(caches) == {0, 1} and all(value["effective_reuses"] > 0 for value in caches.values())
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))
    if not all(results.values()):
        raise ValueError("A100 cache semantic checks failed")


if __name__ == "__main__":
    main()
