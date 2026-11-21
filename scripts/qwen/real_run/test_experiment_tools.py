from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import run_experiment
import yaml
from ray.runtime_env.runtime_env import _merge_runtime_env
from run_experiment import HERE, check_machine, environment

from relax.utils.utils import post_process_env


@pytest.mark.parametrize(
    "kind,cached,tracked,steps",
    [
        ("smoke", True, True, 4),
        ("parity-off", False, False, 64),
        ("parity-on", True, False, 64),
        ("baseline-plain", False, False, 96),
        ("baseline-tracker", False, True, 96),
        ("cache-plain", True, False, 96),
        ("cache-tracker", True, True, 96),
    ],
)
def test_worker_receives_experiment_configuration(tmp_path, monkeypatch, kind, cached, tracked, steps):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "temp"))
    args = SimpleNamespace(
        config=HERE / "config.json",
        kind=kind,
        model=tmp_path / "model",
        data=tmp_path / "data",
        output=tmp_path / "experiment",
        repo=tmp_path / "Relax",
        address="http://127.0.0.1:8265",
    )
    values = environment(args)
    worker = json.loads(values["RUNTIME_ENV_JSON"])["env_vars"]
    assert worker["STEPS"] == str(steps)
    assert worker["CONDITION_CACHE_ENABLED"] == str(int(cached))
    assert worker["BASELINE_STATS_ENABLED"] == str(int(tracked))
    assert worker["GPUS"] == worker["GPUS_PER_NODE"] == "2"
    assert worker["GLOBAL_BATCH_SIZE"] == "4"
    assert worker["CONDITION_CACHE_MAX_BYTES"] == "2147483648"
    assert worker["TRANSFORMER_GRADIENT_CHECKPOINTING"] == "1"
    assert worker["ATTENTION_IMPLEMENTATION"] == "flash_attention_2"
    assert worker["TMPDIR"] == str(tmp_path / "temp")
    assert worker["TENSORBOARD_DIR"] == str(tmp_path / "experiment" / "tensorboard")
    assert ("SAVE_DIR" in worker) == kind.startswith("parity-")
    assert ("BASELINE_STATS_DIR" in worker) == tracked
    assert worker["DETERMINISTIC_ATTENTION"] == str(int(kind.startswith("parity-")))


def test_machine_requirement_prevents_job_submission(monkeypatch):
    monkeypatch.setattr("run_experiment.subprocess.check_output", lambda *args, **kwargs: "0, 12227\n")
    with pytest.raises(ValueError, match="two 80 GB GPUs"):
        check_machine(SimpleNamespace(data=Path("unused"), repo=Path("unused")))


def test_job_environment_merges_with_training_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    repo = Path(__file__).resolve().parents[3] / "Relax"
    args = SimpleNamespace(
        config=HERE / "config.json",
        kind="smoke",
        model=tmp_path / "model",
        data=tmp_path / "data",
        output=tmp_path / "run",
        repo=repo,
        address="http://127.0.0.1:8265",
    )
    values = environment(args)
    submitted = json.loads(values["RUNTIME_ENV_JSON"])
    driver_args = SimpleNamespace(
        partial_rollout=False, use_dynamic_global_batch_size=False, rollout_batch_size=4, n_samples_per_prompt=1
    )
    with patch.dict(os.environ, {**submitted["env_vars"], "PYTHONPATH": values["DRIVER_PYTHONPATH"]}, clear=True):
        configured = post_process_env(driver_args, yaml.safe_load((repo / "configs/env.yaml").read_text()))
    merged = _merge_runtime_env(submitted, configured)
    assert merged is not None
    assert values["DRIVER_PYTHONPATH"].split(":") == [str(repo), "/root/Megatron-LM"]
    assert str(repo) in merged["env_vars"]["PYTHONPATH"].split(":")
    assert submitted["env_vars"].items() <= merged["env_vars"].items()


def test_monitor_cleanup_survives_job_connection_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    args = SimpleNamespace(
        config=HERE / "config.json",
        kind="smoke",
        model=tmp_path / "model",
        data=tmp_path / "data",
        output=tmp_path / "run",
        repo=tmp_path / "Relax",
        address="http://127.0.0.1:8265",
    )
    monitor = MagicMock()
    monitor.poll.return_value = None
    client = MagicMock()
    client.get_job_info.side_effect = ConnectionError("status endpoint unavailable")
    client.stop_job.side_effect = ConnectionError("stop endpoint unavailable")
    monkeypatch.setattr(run_experiment, "check_machine", lambda args: None)
    monkeypatch.setattr(run_experiment, "JobSubmissionClient", lambda address: client)
    monkeypatch.setattr(run_experiment.subprocess, "check_output", lambda *args, **kwargs: "external boundary\n")
    monkeypatch.setattr(run_experiment.subprocess, "Popen", lambda *args, **kwargs: monitor)
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ConnectionError, match="stop endpoint unavailable"):
        run_experiment.run(args)
    client.stop_job.assert_called_once_with("run")
    monitor.terminate.assert_called_once_with()
    monitor.wait.assert_called_once_with()
    assert (args.output / "end-time.txt").is_file()
