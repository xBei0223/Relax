import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from compare_parity import scalars
from summarize_perf import run_summary


def test_parity_step_selection_and_variable_epoch_length(tmp_path):
    writer = EventFileWriter(str(tmp_path))
    for step in range(1, 13):
        values = [Summary.Value(tag=tag, simple_value=float(step)) for tag in ("train/loss", "train/grad_norm", "train/lr", "perf/train_seconds")]
        writer.add_event(Event(wall_time=1800000000 + step, step=step, summary=Summary(value=values)))
    writer.close()
    (tmp_path / "start-time.txt").write_text("2026-11-14 00:00:00 CST\n")
    (tmp_path / "job.log").write_text("")
    (tmp_path / "status.txt").write_text("SUCCEEDED\n")
    selected = scalars(tmp_path, 5, 8)
    assert set(selected["train/loss"]) == {5, 6, 7, 8}
    result = run_summary(tmp_path, 4)
    assert result["epochs"]["epoch1"]["mean_step_seconds"] == 3
    assert result["epochs"]["epoch3"]["sum_seconds"] == 42


@pytest.mark.parametrize("different", [False, True])
def test_parameter_comparison_reports_exact_and_changed_tensors(tmp_path, different):
    import hashlib

    for label, value in (("reference", 1.0), ("candidate", 1.001 if different else 1.0)):
        directory = tmp_path / label
        directory.mkdir()
        tensor = torch.tensor([value], dtype=torch.float32)
        record = {"name": "weight", "step": 1, "kind": "gradient", "present": True, "shape": [1], "global_shape": [1], "dtype": "torch.float32", "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(), "l2": float(tensor.double().norm())}
        (directory / "parameters_rank0.jsonl").write_text(json.dumps(record) + "\n")
        torch.save({"weight": tensor}, directory / "gradient_step0000001_rank0.pt")
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("compare_parameters.py")), "--reference", str(tmp_path / "reference"), "--candidate", str(tmp_path / "candidate"), "--output", str(report)], capture_output=True, text=True)
    assert (result.returncode != 0) == different
    data = json.loads(report.read_text())
    assert data["records"] == 1
    assert data["identical"] == int(not different)
    if different:
        assert data["comparisons"][0]["finite"]
        assert data["comparisons"][0]["within_tolerance"]
