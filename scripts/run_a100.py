import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from audit_a100 import inspect as inspect_run
from select_profile_window import verify_samples


HERE = Path(__file__).resolve().parent


def execute(command: list[str], output: Path, env: dict | None = None) -> None:
    with output.open("w") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--budget-gib", type=int, required=True)
    parser.add_argument("--stage", choices=["A2", "A3", "D", "E", "BC", "C2", "F", "G"], required=True)
    parser.add_argument("--root", type=Path, default=Path("/experiments"))
    args = parser.parse_args()
    journal = args.root / f"a100-orchestration-{args.date}"
    journal.mkdir(exist_ok=True)

    def run(identifier: str, **overrides: object) -> Path:
        name = f"a100-{identifier}-{args.date}"
        path = args.root / name
        env = {**os.environ, "CONDITION_CACHE_MAX_BYTES": str(args.budget_gib * 1024**3)}
        diagnostic = overrides.pop("diagnostics", None)
        if diagnostic is not None:
            env["SFT_EXPERIMENT_DIAGNOSTICS"] = json.dumps({"output_dir": str(path / "diagnostics"), **diagnostic})
        env.update({key: str(value) for key, value in overrides.items()})
        (journal / f"{name}.json").write_text(json.dumps({"name": name, "budget_gib": args.budget_gib, "overrides": overrides, "diagnostics": diagnostic}, indent=1, default=str) + "\n")
        print(f"START {name}", flush=True)
        execute(["bash", str(HERE / "run_experiment.sh"), name, str(HERE / "configs/a100.env")], journal / f"{name}.log", env)
        print(f"COMPLETE {name}", flush=True)
        return path

    def parity(reference: Path, candidate: Path, start: int = 1) -> None:
        comparison = f"compare-{reference.name}-vs-{candidate.name}"
        execute(["python", str(HERE / "compare_parity.py"), "--reference", str(reference), "--candidate", str(candidate), "--work", "/work/parity-a100", "--iteration", "64", "--start-step", str(start), "--end-step", "64", "--output", str(journal / f"{comparison}.json")], journal / f"{comparison}.log")

    reference_off = args.root / f"a100-D1-off-{args.date}"
    reference_on = args.root / f"a100-D1-on-{args.date}"
    if args.stage == "A2":
        run("A2", STEPS=4, BASELINE_STATS_ENABLED=1)
    elif args.stage == "A3":
        run(f"A3-b{args.budget_gib}", STEPS=64, BASELINE_STATS_ENABLED=1, CONDITION_CACHE_ENABLED=1)
    elif args.stage == "D":
        for enabled, label in ((0, "off"), (1, "on")):
            run(f"D1-{label}", STEPS=64, CONDITION_CACHE_ENABLED=enabled, DETERMINISTIC_ATTENTION=1, SAVE_CHECKPOINT=1, SAVE_INTERVAL=32, diagnostics={"record_parameters": True, "tensor_steps": [1, 33, 48, 64]})
        parity(reference_off, reference_on)
        execute(["python", str(HERE / "compare_parameters.py"), "--reference", str(reference_off / "diagnostics"), "--candidate", str(reference_on / "diagnostics"), "--output", str(journal / "parameter-parity.json")], journal / "parameter-parity.log")
    elif args.stage == "E":
        for source, enabled, identifier in ((reference_on, 1, "E2-on-on"), (reference_off, 1, "E2-off-on"), (reference_on, 0, "E2-on-off")):
            restore = args.root / f"a100-restore-{identifier}-{args.date}"
            shutil.copytree(source / "checkpoints/i2i_sft/iter_0000032", restore / "i2i_sft/iter_0000032")
            candidate = run(identifier, STEPS=64, CONDITION_CACHE_ENABLED=enabled, DETERMINISTIC_ATTENTION=1, SAVE_CHECKPOINT=1, LOAD_DIR=restore, BASELINE_STATS_ENABLED=1)
            parity(reference_off, candidate, 33)
            parity(reference_on, candidate, 33)
        candidates = []
        for enabled, label in ((0, "off"), (1, "on")):
            candidates.append(run(f"E1-{label}", STEPS=64, CONDITION_CACHE_ENABLED=enabled, DETERMINISTIC_ATTENTION=1, SAVE_CHECKPOINT=1, BASELINE_STATS_ENABLED=1, diagnostics={"encoder_update_step": 40, "encoder_update_scale": 1.01}))
        parity(*candidates)
        execute(["python", str(HERE / "verify_a100_semantics.py"), "--root", str(args.root), "--date", args.date, "--output", str(journal / "cache-semantics.json")], journal / "cache-semantics.log")
    elif args.stage in ("BC", "C2"):
        profile_path = journal / "profile-window-single-step.json"
        execute(["python", str(HERE / "select_profile_window.py"), str(args.root / f"a100-A3-b{args.budget_gib}-{args.date}"), "--budget-gib", str(args.budget_gib), "--window-steps", "1", "--output", str(profile_path)], journal / "profile-window-single-step.log")
        profile = json.loads(profile_path.read_text())
        groups = {"baseline-plain": [], "baseline-tracker": [], "cache-plain": [], "cache-tracker": []}
        for repetition in range(1, 4):
            for identifier, group, enabled, tracked in (("B2", "baseline-plain", 0, 0), ("C1", "cache-plain", 1, 0), ("B1", "baseline-tracker", 0, 1)):
                if args.stage == "BC":
                    path = run(f"{identifier}-r{repetition}", CONDITION_CACHE_ENABLED=enabled, BASELINE_STATS_ENABLED=tracked)
                else:
                    path = args.root / f"a100-{identifier}-r{repetition}-{args.date}"
                    if (path / "status.txt").read_text().strip() != "SUCCEEDED":
                        raise ValueError(f"Incomplete performance reference: {path}")
                groups[group].append(path.name)
        if args.stage == "BC":
            path = run("C2", CONDITION_CACHE_ENABLED=1, BASELINE_STATS_ENABLED=1)
        else:
            path = args.root / f"a100-C2-{args.date}"
            if (path / "status.txt").read_text().strip() != "SUCCEEDED":
                raise ValueError(f"Incomplete stage statistics reference: {path}")
        groups["cache-tracker"].append(path.name)
        commit = (args.root / f"a100-B2-r1-{args.date}" / "base-commit.txt").read_text().strip()
        captures = {}
        for rank in (0, 1):
            path = run(f"C2-profile-single-rank{rank}", CONDITION_CACHE_ENABLED=1, BASELINE_STATS_ENABLED=1, PROFILE_PREPROCESS=1, PROFILE_RANK=rank, PROFILE_STEP_START=profile["profile_step_start"], PROFILE_STEP_END=profile["profile_step_end"])
            actual_samples = verify_samples(profile, path)
            (trace,) = list((path / "profiler").rglob("*.pt.trace.json.gz"))
            if f"rank{rank}_dp{rank}_tp0_pp0." not in trace.name or trace.stat().st_size == 0:
                raise ValueError(f"Invalid profiler trace for rank {rank}: {trace}")
            captures[str(rank)] = {"run": path.name, "samples": actual_samples, "trace": {"path": str(trace), "bytes": trace.stat().st_size}}
            profile_audit = inspect_run(path, "C2", "SUCCEEDED", 1, 96, commit)
            (journal / f"profile-audit-rank{rank}.json").write_text(json.dumps(profile_audit, indent=1) + "\n")
            (journal / "profile-samples.json").write_text(json.dumps(captures, indent=1) + "\n")
        (journal / "performance-groups.json").write_text(json.dumps(groups, indent=1) + "\n")
        runs = [str(args.root / name) for names in groups.values() for name in names]
        summary = journal / "performance-summary.json"
        execute(["python", str(HERE / "summarize_perf.py"), *runs, "--output", str(summary)], journal / "performance-summary.log")
        execute(["python", str(HERE / "aggregate_perf.py"), str(summary), "--groups", str(journal / "performance-groups.json")], journal / "performance-aggregate.json")
    elif args.stage == "F":
        for identifier, phase in (("F1", None), ("F2", "forward"), ("F3", "backward")):
            for repetition in range(1, 4):
                if phase is None:
                    run(f"{identifier}-r{repetition}", STEPS=64, CONDITION_CACHE_ENABLED=1, CANCEL_STEP=40, EXPECTED_STATUS="STOPPED")
                else:
                    run(f"{identifier}-r{repetition}", STEPS=64, CONDITION_CACHE_ENABLED=1, EXPECTED_STATUS="FAILED", diagnostics={"fault": {"step": 40, "micro": 1, "rank": 0, "phase": phase}})
    elif args.stage == "G":
        for size in (256, 512):
            for enabled, label in ((0, "off"), (1, "on")):
                run(f"G1-n{size}-{label}", STEPS=size // 4 * 3, DATASET_SIZE=size, DATA_DIR=f"/data/magicbrush-{size}", CONDITION_CACHE_ENABLED=enabled, BASELINE_STATS_ENABLED=1)
        for budget in (2, 4, 8):
            run(f"G2-b{budget}", CONDITION_CACHE_ENABLED=1, CONDITION_CACHE_MAX_BYTES=budget * 1024**3)
        for enabled, label in ((0, "off"), (1, "on")):
            run(f"G3-{label}", STEPS=192, CONDITION_CACHE_ENABLED=enabled, BASELINE_STATS_ENABLED=1)
            run(f"G4-{label}", MAX_PIXELS=262144, CONDITION_CACHE_ENABLED=enabled, BASELINE_STATS_ENABLED=1)
        for identifier, epoch_steps, total_steps in (("G1-n256", 64, 192), ("G1-n512", 128, 384), ("G3", 32, 192), ("G4", 32, 96)):
            online = args.root / f"a100-{identifier}-off-{args.date}"
            cached = args.root / f"a100-{identifier}-on-{args.date}"
            execute(["python", str(HERE / "compare_scope.py"), "--online", str(online), "--cached", str(cached), "--tracked-online", str(online), "--steps-per-epoch", str(epoch_steps), "--start-step", str(total_steps - epoch_steps + 1), "--end-step", str(total_steps), "--output", str(journal / f"scope-{identifier}.json")], journal / f"scope-{identifier}.log")
        online = [str(args.root / f"a100-B2-r{repetition}-{args.date}") for repetition in range(1, 4)]
        tracked = [str(args.root / f"a100-B1-r{repetition}-{args.date}") for repetition in range(1, 4)]
        for budget in (2, 4, 8):
            identifier = f"G2-b{budget}"
            cached = args.root / f"a100-{identifier}-{args.date}"
            execute(["python", str(HERE / "compare_scope.py"), "--online", *online, "--cached", str(cached), "--tracked-online", *tracked, "--start-step", "33", "--end-step", "96", "--output", str(journal / f"scope-{identifier}.json")], journal / f"scope-{identifier}.log")
        runs = [str(path) for path in sorted(args.root.glob(f"a100-G*-{args.date}"))]
        execute(["python", str(HERE / "summarize_resources.py"), *runs, "--output", str(journal / "scope-resources.json")], journal / "scope-resources.log")


if __name__ == "__main__":
    main()
