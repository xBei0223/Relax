# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from relax.backends.fsdp.checkpoint import _load_dcp_model_state


def _read_contract(checkpoint: Path, output: Path, mode: str) -> dict[str, Any]:
    if torch.distributed.is_initialized():
        raise RuntimeError("Export Qwen checkpoints in a separate CPU process")
    if not (checkpoint / "COMMITTED").is_file():
        raise ValueError("Qwen checkpoint must be committed before export")
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads((checkpoint / "adapter_contract.json").read_text())
    if contract["trainable_mode"] != mode:
        raise ValueError(f"Expected a {mode} checkpoint")
    return contract


def export_adapter(checkpoint: Path, output: Path) -> None:
    from safetensors.torch import save_file

    contract = _read_contract(checkpoint, output, "lora")
    if output.with_suffix(".json").exists():
        raise FileExistsError(output.with_suffix(".json"))
    rank = int(contract["lora"]["rank"])
    if float(contract["lora"]["alpha"]) != rank:
        raise ValueError("Qwen LoRA export requires alpha == rank")
    state = _load_dcp_model_state(str(checkpoint / "fsdp"))
    if not state:
        raise ValueError("Checkpoint has no LoRA parameters")
    pairs: dict[str, set[str]] = {}
    exported = {}
    for name, tensor in state.items():
        kind = next((part for part in ("lora_A", "lora_B") if name.endswith(f".{part}.default.weight")), None)
        if kind is None or not name.startswith("transformer.") or tensor.ndim != 2:
            raise ValueError(f"Unexpected Qwen adapter tensor: {name}")
        if tensor.shape[0 if kind == "lora_A" else 1] != rank:
            raise ValueError(f"Qwen adapter rank mismatch: {name}")
        stem = name.rsplit(f".{kind}.", 1)[0]
        pairs.setdefault(stem, set()).add(kind)
        exported[f"{stem}.{kind}.weight"] = tensor.to(torch.bfloat16).contiguous()
    if any(kinds != {"lora_A", "lora_B"} for kinds in pairs.values()):
        raise ValueError("Incomplete Qwen LoRA A/B pair")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(exported, str(output))
    output.with_suffix(".json").write_text(json.dumps({"contract": contract, "tensors": len(exported)}, indent=2))


def export_full(checkpoint: Path, base: Path, output: Path) -> Path:
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors import safe_open
    from safetensors.torch import save_file

    contract = _read_contract(checkpoint, output, "full")
    source = base / "transformer"
    expected = {}
    for path in sorted(source.glob("diffusion_pytorch_model*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            expected.update({key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()})
    if not expected:
        raise ValueError("Base Qwen transformer has no safetensors weights")
    state = _load_dcp_model_state(str(checkpoint / "fsdp"))
    invalid = [name for name in state if not name.startswith(("transformer.", "text_encoder.", "vae."))]
    if invalid:
        raise ValueError(f"Unexpected full Qwen checkpoint tensors: {invalid[:5]}")
    transformer = {
        name.removeprefix("transformer."): value for name, value in state.items() if name.startswith("transformer.")
    }
    if transformer.keys() != expected.keys():
        raise ValueError("Full Qwen checkpoint does not match the base transformer keyset")
    for name, tensor in transformer.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"Full Qwen checkpoint shape mismatch: {name}")
    transformer = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in transformer.items()}
    shards = split_torch_state_dict_into_shards(
        transformer, filename_pattern="diffusion_pytorch_model{suffix}.safetensors", max_shard_size="4GB"
    )
    destination = output / "transformer"
    destination.mkdir(parents=True)
    for filename, names in shards.filename_to_tensors.items():
        save_file({name: transformer[name] for name in names}, str(destination / filename), metadata={"format": "pt"})
    if shards.is_sharded:
        (destination / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"metadata": shards.metadata, "weight_map": shards.tensor_to_filename}, indent=2)
        )
    shutil.copy2(source / "config.json", destination / "config.json")
    (output / "export.json").write_text(json.dumps({"contract": contract, "tensors": len(transformer)}, indent=2))
    return destination
