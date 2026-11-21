from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import FlowMatchSFTLoss
from diffsynth.models.general_modules import AdaLayerNorm, RMSNorm, TimestepEmbeddings
from diffsynth.models.qwen_image_dit import QwenEmbedRope, QwenImageDiT, QwenImageTransformerBlock
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.models.qwen_image_vae import QwenImageVAE
from diffsynth.pipelines.qwen_image import QwenImagePipeline, QwenImageUnit_EditImageEmbedder, QwenImageUnit_PromptEmbedder
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2TokenizerFast, Qwen2VLProcessor


def digest(image: Image.Image) -> dict:
    return {"mode": image.mode, "size": list(image.size), "sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def make_processor(output: Path, source: Path) -> Qwen2VLProcessor:
    output.mkdir(parents=True)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                "<|image_pad|>", "<|video_pad|>"]
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()) + specials)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, pad_token="<|endoftext|>", eos_token="<|im_end|>")
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    tokenizer.save_pretrained(output)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy2(source / name, output / name)
    return Qwen2VLProcessor.from_pretrained(output, use_fast=True)


def make_text_encoder(processor: Qwen2VLProcessor) -> Qwen2_5_VLForConditionalGeneration:
    tokenizer = processor.tokenizer
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                     "max_position_embeddings": 4096, "use_cache": False},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": 32, "patch_size": 14, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "fullatt_block_indexes": [0]},
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    config._attn_implementation = "sdpa"
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def make_reference_dit(state: dict[str, torch.Tensor]) -> QwenImageDiT:
    # Build a downscaled model from original DiffSynth modules while keeping the original forward pass.
    with torch.device("meta"):
        model = QwenImageDiT(num_layers=1)
    model.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[4, 6, 6], scale_rope=True)
    model.time_text_embed = TimestepEmbeddings(256, 32, diffusers_compatible_format=True, scale=1000)
    model.txt_norm = RMSNorm(32, eps=1e-6)
    model.img_in = torch.nn.Linear(64, 32)
    model.txt_in = torch.nn.Linear(32, 32)
    model.transformer_blocks = torch.nn.ModuleList([QwenImageTransformerBlock(32, 2, 16)])
    model.norm_out = AdaLayerNorm(32, single=True)
    model.proj_out = torch.nn.Linear(32, 64)
    model.load_state_dict(state, strict=True)
    return model.to(torch.bfloat16)


def pipeline_inputs(image: Image.Image, edits: list[Image.Image], prompt: str) -> tuple[dict, dict, dict]:
    return ({"input_image": image, "edit_image": edits, "height": image.height, "width": image.width,
             "cfg_scale": 1, "rand_device": "cpu", "edit_image_auto_resize": True,
             "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
             "zero_cond_t": False}, {"prompt": prompt}, {"negative_prompt": ""})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-source", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    args.output.mkdir(parents=True)
    processor = make_processor(args.output / "processor", args.processor_source)
    text_encoder = make_text_encoder(processor)
    text_encoder.config.to_json_file(args.output / "text_encoder_config.json")
    save_file(text_encoder.state_dict(), str(args.output / "text_encoder.safetensors"))
    vae_config = {"base_dim": 4, "z_dim": 16, "dim_mult": [1, 2, 2, 2], "num_res_blocks": 1,
                  "temperal_downsample": [False, True, True]}
    vae = AutoencoderKLQwenImage(**vae_config).to(torch.bfloat16).eval()
    vae.save_config(args.output)
    (args.output / "config.json").rename(args.output / "vae_config.json")
    save_file(vae.state_dict(), str(args.output / "vae.safetensors"))
    reference_vae = QwenImageVAE(**vae_config).to(torch.bfloat16).eval()
    reference_vae.load_state_dict(vae.state_dict(), strict=True)
    transformer = QwenImageTransformer2DModel(num_layers=1, attention_head_dim=16, num_attention_heads=2,
                                             joint_attention_dim=32, axes_dims_rope=(4, 6, 6))
    transformer.save_config(args.output)
    (args.output / "config.json").rename(args.output / "transformer_config.json")
    save_file(transformer.state_dict(), str(args.output / "transformer.safetensors"))
    pipe = QwenImagePipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipe.vae = reference_vae
    with torch.device("meta"):
        reference_text = QwenImageTextEncoder()
    reference_text.model, reference_text.lm_head = text_encoder.model, text_encoder.lm_head
    reference_text.config = text_encoder.config
    pipe.text_encoder = reference_text.eval()
    pipe.dit = make_reference_dit(transformer.state_dict())
    pipe.processor = processor
    pipe.scheduler.set_timesteps(1000, training=True)
    prompt_unit = QwenImageUnit_PromptEmbedder()
    edit_unit = QwenImageUnit_EditImageEmbedder()
    cases = [(65, 49, 4096, None), (79, 113, 4096, None), (19, 35, 4096, 2048)]
    pixels, processor_outputs, latent_outputs, records = {}, {}, {}, []
    cropped, edited = [], []
    for index, (width, height, max_pixels, budget) in enumerate(cases):
        rgb = (np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) * (index + 7) % 256).astype(np.uint8)
        image = Image.fromarray(rgb)
        image.save(args.output / f"input_{index}.png")
        if budget is None:
            crop = ImageCropAndResize(max_pixels=max_pixels, height_division_factor=16, width_division_factor=16)
        else:
            scale = (width * height / budget) ** 0.5
            target_h, target_w = round(height / scale) // 16 * 16, round(width / scale) // 16 * 16
            crop = ImageCropAndResize(height=max(16, target_h), width=max(16, target_w))
        target = crop(image)
        edit = edit_unit.edit_image_auto_resize(target)
        visual = prompt_unit.resize_image(edit)
        cropped.append(target)
        edited.append(edit)
        for name, value in (("target", target), ("edit", edit)):
            key = f"case_{index}_{name}"
            pixels[key] = pipe.preprocess_image(value).contiguous()
            with torch.no_grad():
                latent_outputs[key] = pipe.vae.encode(pixels[key]).contiguous()
        outputs = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[visual],
                            padding=True, return_tensors="pt")
        for name in ("pixel_values", "image_grid_thw"):
            processor_outputs[f"case_{index}_{name}"] = outputs[name].contiguous()
        records.append({"input": f"input_{index}.png", "max_pixels": max_pixels, "fixed_pixels_budget": budget,
                        "target": digest(target), "vae": digest(edit), "vl": digest(visual)})
    save_file(pixels, str(args.output / "pixels.safetensors"))
    save_file(processor_outputs, str(args.output / "processor_outputs.safetensors"))
    save_file(latent_outputs, str(args.output / "vae_latents.safetensors"))
    (args.output / "preprocess.json").write_text(json.dumps(records, indent=2) + "\n")
    schedule = pipe.scheduler
    quantized = schedule.timesteps.to(torch.bfloat16)
    indices = torch.stack([torch.argmin((schedule.timesteps - value).abs()) for value in quantized])
    schedule_data = {"timesteps": schedule.timesteps.tolist(), "quantized_timesteps": quantized.float().tolist(),
                     "indices": indices.tolist(), "sigmas": schedule.sigmas[indices].tolist(),
                     "weights": schedule.linear_timesteps_weights[indices].tolist()}
    (args.output / "schedule.json").write_text(json.dumps(schedule_data, indent=2) + "\n")
    text_outputs = {}
    positions = []

    def capture_positions(module: torch.nn.Module, inputs: tuple, kwargs: dict) -> None:
        positions.append(kwargs["position_ids"].detach().clone())

    hook = text_encoder.model.language_model.register_forward_pre_hook(capture_positions, with_kwargs=True)
    with torch.no_grad():
        for count in (1, 2):
            embeds = prompt_unit.encode_prompt_edit_multi(pipe, ["change the color"], edited[:count])
            position = positions[-1]
            if torch.equal(position[1], position[2]):
                raise AssertionError("Reference text encoder did not use image row and column positions")
            text_outputs[f"images_{count}_position_ids"] = position.contiguous()
            text_outputs[f"images_{count}_prompt_embeds"] = torch.stack(embeds).contiguous()
    hook.remove()
    save_file(text_outputs, str(args.output / "text_encoding.safetensors"))
    pipe.text_encoder.to(torch.bfloat16)
    inputs = pipeline_inputs(cropped[0], [cropped[0]], "change the color")
    training = {"rng_before": torch.get_rng_state().clone()}
    with torch.no_grad():
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
    original_target = pipe.scheduler.training_target
    original_model_fn = pipe.model_fn

    def capture_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        training["noise"] = noise.detach().clone()
        training["timestep"] = timestep.detach().clone()
        return original_target(sample, noise, timestep)

    def capture_prediction(**kwargs) -> torch.Tensor:
        prediction = original_model_fn(**kwargs)
        training["prediction"] = prediction.detach().clone()
        return prediction

    pipe.scheduler.training_target = capture_target
    pipe.model_fn = capture_prediction
    loss = FlowMatchSFTLoss(pipe, **inputs[0], **inputs[1])
    loss.backward()
    training["loss"] = loss.detach().clone()
    training["rng_after"] = torch.get_rng_state().clone()
    training.update({f"gradient.{name}": parameter.grad.detach().clone()
                     for name, parameter in pipe.dit.named_parameters() if parameter.grad is not None})
    save_file(training, str(args.output / "training_step.safetensors"))
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in pipe.dit.parameters() if p.grad is not None):
        raise AssertionError("Reference pipeline produced non-finite training values")
    full_gradient_count = sum(p.grad is not None for p in pipe.dit.parameters())
    pipe.dit.zero_grad(set_to_none=True)
    pipe.dit.requires_grad_(False)
    torch.manual_seed(8128)
    lora_config = LoraConfig(r=2, lora_alpha=2, target_modules=[
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_out.0", "attn.to_add_out", "img_mlp.net.2", "img_mod.1", "txt_mlp.net.2", "txt_mod.1",
    ])
    inject_adapter_in_model(lora_config, pipe.dit)
    pipe.dit.to(torch.bfloat16)
    training = {}
    with torch.no_grad():
        for name, parameter in pipe.dit.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
            if "lora_" in name:
                training[f"weight.{name}"] = parameter.detach().clone()
    training["rng_before"] = torch.get_rng_state().clone()
    inputs = pipeline_inputs(cropped[0], [cropped[0]], "change the color")
    with torch.no_grad():
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
    lora_loss = FlowMatchSFTLoss(pipe, **inputs[0], **inputs[1])
    lora_loss.backward()
    training["loss"] = lora_loss.detach().clone()
    training["rng_after"] = torch.get_rng_state().clone()
    training.update({f"gradient.{name}": parameter.grad.detach().clone()
                     for name, parameter in pipe.dit.named_parameters() if parameter.grad is not None})
    save_file(training, str(args.output / "lora_training_step.safetensors"))
    environment = {name: importlib.metadata.version(name) for name in
                   ("torch", "torchvision", "transformers", "tokenizers", "diffusers", "Pillow", "diffsynth")}
    environment["reference_revision"] = "f7b7db96a2572e69693d85e78a1d29d9e8489ef3"
    environment["pipeline_units"] = [type(unit).__name__ for unit in pipe.units]
    environment["pipeline_loss"] = loss.detach().float().item()
    environment["pipeline_gradient_tensors"] = full_gradient_count
    environment["lora_pipeline_loss"] = lora_loss.detach().float().item()
    environment["lora_pipeline_gradient_tensors"] = sum(p.grad is not None for p in pipe.dit.parameters())
    environment["sha256"] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({"environment": environment, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
