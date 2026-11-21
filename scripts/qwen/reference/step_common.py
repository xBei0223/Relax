from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


LORA_TARGETS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
    "attn.to_out.0", "attn.to_add_out", "img_mlp.net.2", "img_mod.1", "txt_mlp.net.2", "txt_mod.1",
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "transformers", "tokenizers", "diffusers", "peft", "Pillow")}


def model_digests(source: Path, fixture: bool) -> dict[str, str]:
    if fixture:
        paths = [source / name for name in ("transformer.safetensors", "transformer_config.json",
                                             "text_encoder.safetensors", "text_encoder_config.json",
                                             "vae.safetensors", "vae_config.json")]
        paths.extend(path for path in (source / "processor").rglob("*") if path.is_file())
    else:
        paths = [path for component in ("transformer", "text_encoder", "vae", "processor")
                 for path in (source / component).rglob("*") if path.is_file()]
    if not paths:
        raise FileNotFoundError(source)
    return {str(path.relative_to(source)): sha256(path) for path in sorted(paths)}


def save_tensors(path: Path, values: dict[str, torch.Tensor]) -> None:
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in values.items()}, str(path))


def rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    values = {"rng_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        values["rng_cuda"] = torch.cuda.get_rng_state(device)
    return values


def restore_rng(values: dict[str, torch.Tensor], device: torch.device) -> None:
    torch.set_rng_state(values["rng_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(values["rng_cuda"], device)


def numerical_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Tensor metadata differs: {actual.shape}/{actual.dtype}, {expected.shape}/{expected.dtype}")
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("Comparison contains a non-finite value")
    delta = actual - expected
    norm = expected.norm().item()
    delta_norm = delta.norm().item()
    return {"max_absolute_error": delta.abs().max().item(),
            "relative_l2": delta_norm / norm if norm else delta_norm,
            "equal": torch.equal(actual, expected)}


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
