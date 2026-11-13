# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor


def validate_diagnostics(config: Any) -> None:
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("experiment_diagnostics must be a dictionary")
    allowed = {
        "output_dir",
        "record_parameters",
        "tensor_steps",
        "encoder_update_step",
        "encoder_update_scale",
        "fault",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown experiment_diagnostics keys: {sorted(set(config) - allowed)}")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"].strip():
        raise ValueError("experiment_diagnostics.output_dir must be a non-empty path")
    if type(config.get("record_parameters", False)) is not bool:
        raise ValueError("record_parameters must be boolean")
    steps = config.get("tensor_steps", [])
    if not isinstance(steps, list) or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("tensor_steps must contain positive integers")
    if steps and not config.get("record_parameters", False):
        raise ValueError("tensor_steps requires record_parameters")
    step = config.get("encoder_update_step", 0)
    if type(step) is not int or step < 0:
        raise ValueError("encoder_update_step must be a nonnegative integer")
    scale = config.get("encoder_update_scale", 1.01)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0 or scale == 1:
        raise ValueError("encoder_update_scale must be positive, finite and different from one")
    fault = config.get("fault")
    if fault is not None:
        if not isinstance(fault, dict) or set(fault) != {"step", "micro", "rank", "phase"}:
            raise ValueError("fault requires step, micro, rank and phase")
        for key, minimum in (("step", 1), ("micro", 1), ("rank", 0)):
            if type(fault[key]) is not int or fault[key] < minimum:
                raise ValueError(f"fault.{key} must be an integer >= {minimum}")
        if fault["phase"] not in ("forward", "backward"):
            raise ValueError("fault.phase must be forward or backward")


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def write_event(directory: Path, rank: int, event: dict[str, Any]) -> None:
    with (directory / f"events_rank{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "pid": os.getpid(), "rank": rank, **event}) + "\n")


def raise_injected_fault(fault: dict[str, Any]) -> None:
    write_event(Path(fault["output_dir"]), fault["rank"], {"event": "fault", **fault})
    raise RuntimeError(
        f"Task17 injected {fault['phase']} failure step={fault['step']} micro={fault['micro']} rank={fault['rank']}"
    )


class SFTDiagnostics:
    def __init__(self, model: torch.nn.Module, config: dict[str, Any], rank: int, world: int, micros: int) -> None:
        validate_diagnostics(config)
        self.config, self.rank = config, rank
        self.directory = Path(config["output_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.tensor_steps = set(config.get("tensor_steps", []))
        write_event(self.directory, rank, {"event": "initialization", "world_size": world, "micro_batches": micros})
        fault = config.get("fault")
        if fault is not None and (fault["rank"] >= world or fault["micro"] > micros):
            raise ValueError("Fault rank/micro exceeds this training configuration")
        if config.get("record_parameters", False):
            if any(parameter.dtype != torch.float32 for parameter in self.parameters.values()):
                raise ValueError("Parameter diagnostics require FP32 master parameters")
            self.records = self.directory / f"parameters_rank{rank}.jsonl"
            self.records.touch(exist_ok=False)

    def fault_for(self, step: int, micro: int) -> dict[str, Any] | None:
        fault = self.config.get("fault")
        if fault is not None and (fault["step"], fault["micro"], fault["rank"]) == (step, micro, self.rank):
            return {**fault, "output_dir": str(self.directory)}
        return None

    def _record(self, step: int, kind: str, tensors: dict[str, torch.Tensor | None]) -> None:
        records = []
        for name, tensor in tensors.items():
            if tensor is None:
                records.append({"step": step, "kind": kind, "name": name, "present": False})
                continue
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite diagnostic tensor: {kind}/{name}")
            records.append(
                {
                    "step": step,
                    "kind": kind,
                    "name": name,
                    "present": True,
                    "shape": list(tensor.shape),
                    "global_shape": list(self.parameters[name].shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "l2": float(tensor.double().norm()),
                }
            )
        with self.records.open("a") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        if step in self.tensor_steps:
            torch.save(tensors, self.directory / f"{kind}_step{step:07d}_rank{self.rank}.pt")

    def before_optimizer(self, step: int) -> dict[str, torch.Tensor] | None:
        if not self.config.get("record_parameters", False):
            return None
        gradients = {
            name: local_tensor(parameter.grad.detach()).to(device="cpu", copy=True).contiguous()
            if parameter.grad is not None
            else None
            for name, parameter in self.parameters.items()
        }
        self._record(step, "gradient", gradients)
        return {
            name: local_tensor(parameter.detach()).to(device="cpu", copy=True)
            for name, parameter in self.parameters.items()
        }

    def after_optimizer(self, step: int, before: dict[str, torch.Tensor] | None, trainer: Any) -> None:
        if before is not None:
            updates = {
                name: (local_tensor(parameter.detach()).cpu() - before[name]).contiguous()
                for name, parameter in self.parameters.items()
            }
            self._record(step, "update", updates)
        if step == self.config.get("encoder_update_step", 0):
            cache = trainer.condition_cache
            previous = cache.stats() if cache is not None else None
            with torch.no_grad():
                weight = trainer.actor.model.pipe.dit.language_model.model.embed_tokens.weight
                local_tensor(weight).mul_(self.config.get("encoder_update_scale", 1.01))
            trainer.bump_encoder_version(version_tag=f"experiment-step-{step}")
            write_event(
                self.directory,
                self.rank,
                {
                    "event": "encoder_update",
                    "step": step,
                    "before": previous,
                    "after": cache.stats() if cache else None,
                },
            )
