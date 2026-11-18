# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import atexit
import json
import tarfile
import tempfile
from collections import UserDict
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from safetensors.torch import load_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from relax.backends.fsdp.lora import inject_lora_adapter
from relax.models.qwen_image.sft import (
    IMAGE_TEMPLATE,
    PROMPT_TEMPLATE,
    TEXT_INPUT_FIELDS,
    QwenImageEditSFTAdapter,
    QwenImageEditSFTModel,
    _digest_text_inputs,
    image_digest,
    load_image,
    preprocess_pixels,
)


_fixture_directory = tempfile.TemporaryDirectory(prefix="qwen-image-edit-reference-")
atexit.register(_fixture_directory.cleanup)
FIXTURES = Path(_fixture_directory.name)
with tarfile.open(Path(__file__).parent / "fixtures" / "qwen_image_edit_diffsynth_f7b7db96.tar.xz") as archive:
    archive.extractall(FIXTURES, filter="data")
CONFIG = {"conditioning_images": 1, "max_pixels": 4096, "seed": 42, "fixed_pixels_tiers": None}


def build_adapter() -> QwenImageEditSFTAdapter:
    adapter = QwenImageEditSFTAdapter()
    adapter.configure_processor(Qwen2VLProcessor.from_pretrained(FIXTURES / "processor"), FIXTURES / "processor")
    return adapter


def build_model(mode: str = "full", dtype: torch.dtype = torch.float32) -> QwenImageEditSFTModel:
    torch.manual_seed(1729)
    transformer = QwenImageTransformer2DModel.from_config(
        json.loads((FIXTURES / "transformer_config.json").read_text())
    )
    transformer.load_state_dict(load_file(FIXTURES / "transformer.safetensors"))
    transformer.set_attention_backend("native")
    text_config = Qwen2_5_VLConfig.from_dict(json.loads((FIXTURES / "text_encoder_config.json").read_text()))
    text_config._attn_implementation = "sdpa"
    text_encoder = Qwen2_5_VLForConditionalGeneration(text_config)
    text_encoder.load_state_dict(load_file(FIXTURES / "text_encoder.safetensors"))
    vae = AutoencoderKLQwenImage.from_config(json.loads((FIXTURES / "vae_config.json").read_text()))
    vae.load_state_dict(load_file(FIXTURES / "vae.safetensors"))
    model = QwenImageEditSFTModel(transformer, text_encoder, vae, training_mode=mode, torch_dtype=dtype).to(dtype)
    if mode == "lora":
        inject_lora_adapter(model, rank=2, alpha=2, target_modules=QwenImageEditSFTAdapter.lora_target_modules)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(std=0.01)
        model.to(dtype)
    model._condition_config = dict(CONFIG)
    model._condition_model_path = str(FIXTURES)
    model.train()
    return model


def small_sample(adapter: QwenImageEditSFTAdapter, index: int = 0, count: int = 1) -> UserDict:
    # Cache training tests use downscaled real images; full-size preprocessing is covered by reference tests.
    target = load_image(FIXTURES / f"input_{index}.png", 4096).resize((32, 32))
    edits = [load_image(FIXTURES / f"input_{i}.png", 4096).resize((32, 32)) for i in range(count)]
    prompt = f"change the color {index}"
    text = PROMPT_TEMPLATE.format("".join(IMAGE_TEMPLATE.format(i + 1) for i in range(count)) + prompt)
    outputs = adapter.processor(text=[text], images=edits, padding=True, return_tensors="pt")
    return UserDict(
        {
            "prompt": prompt,
            "fixed_pixels_budget": None,
            "image": target,
            "edit_image": edits,
            "target_pixels": preprocess_pixels(target),
            "edit_pixels": tuple(preprocess_pixels(image) for image in edits),
            "text_inputs": {name: outputs[name] for name in TEXT_INPUT_FIELDS},
            "text_inputs_digest": _digest_text_inputs(outputs),
            "digests": {
                "target": image_digest(target),
                "vae": [image_digest(x) for x in edits],
                "vl": [image_digest(x) for x in edits],
            },
            "edit_shapes": tuple((1, 2, 2) for _ in edits),
            "tracker_keys": {
                "prompt_key": f"prompt-{index}",
                "ref_vae_image_keys": [f"ref-{i}" for i in range(count)],
                "target_image_key": f"target-{index}",
            },
            "sample_preprocess_ms": 0.1,
        }
    )
