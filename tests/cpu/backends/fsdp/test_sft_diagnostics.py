# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp.sft_diagnostics import SFTDiagnostics, validate_diagnostics


def test_parameter_recording_preserves_gradients_updates_and_rng(tmp_path):
    torch.manual_seed(123)
    reference = torch.nn.Linear(3, 2)
    recorded = copy.deepcopy(reference)
    diagnostics = SFTDiagnostics(
        recorded, {"output_dir": str(tmp_path), "record_parameters": True, "tensor_steps": [1]}, 0, 1, 2
    )
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.01) for model in (reference, recorded)]
    input_values = torch.randn(2, 3)
    histories = []
    for model, optimizer in zip((reference, recorded), optimizers, strict=True):
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        loss = model(input_values).square().mean()
        loss.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        snapshots = diagnostics.before_optimizer(1) if model is recorded else None
        optimizer.step()
        if model is recorded:
            diagnostics.after_optimizer(1, snapshots, None)
        assert torch.equal(rng, torch.get_rng_state())
        histories.append(
            (
                loss.detach(),
                gradients,
                {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()},
            )
        )
    assert torch.equal(histories[0][0], histories[1][0])
    for kind, index in (("gradient", 1), ("update", 2)):
        saved = torch.load(tmp_path / f"{kind}_step0000001_rank0.pt", weights_only=True)
        for name in histories[0][index]:
            assert torch.equal(histories[0][index][name], histories[1][index][name])
            assert torch.equal(saved[name], histories[0][index][name])
    records = [json.loads(line) for line in (tmp_path / "parameters_rank0.jsonl").read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        tensor = histories[1][1 if record["kind"] == "gradient" else 2][record["name"]]
        assert record["sha256"] == hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        assert record["l2"] == float(tensor.double().norm())


def test_encoder_update_changes_condition_weight_and_notifies_cache(tmp_path):
    model = torch.nn.Linear(2, 2)
    embedding = torch.nn.Embedding(5, 2)
    model.pipe = SimpleNamespace(
        dit=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)))
    )
    notifications = []
    trainer = SimpleNamespace(
        actor=SimpleNamespace(model=model),
        condition_cache=None,
        bump_encoder_version=lambda **kwargs: notifications.append(kwargs),
    )
    diagnostics = SFTDiagnostics(
        model, {"output_dir": str(tmp_path), "encoder_update_step": 2, "encoder_update_scale": 1.1}, 0, 1, 1
    )
    previous = embedding.weight.detach().clone()
    diagnostics.after_optimizer(1, None, trainer)
    assert torch.equal(embedding.weight, previous)
    diagnostics.after_optimizer(2, None, trainer)
    assert torch.equal(embedding.weight, previous * 1.1)
    assert notifications == [{"version_tag": "experiment-step-2"}]


@pytest.mark.parametrize(
    "config",
    [
        {"record_parameters": 1},
        {"tensor_steps": [True]},
        {"tensor_steps": [1]},
        {"encoder_update_step": -1},
        {"encoder_update_scale": float("nan")},
        {"fault": {"step": 1, "micro": 1, "rank": 0, "phase": "invalid"}},
    ],
)
def test_diagnostics_rejects_invalid_config(tmp_path, config):
    with pytest.raises(ValueError):
        validate_diagnostics({"output_dir": str(tmp_path), **config})
