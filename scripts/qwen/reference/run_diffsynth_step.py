from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import diffsynth
import torch
from diffsynth.core import ModelConfig
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import FlowMatchSFTLoss
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.models.qwen_image_vae import QwenImageVAE
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from safetensors.torch import load_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from generate_diffsynth_reference import make_reference_dit, pipeline_inputs
from step_common import LORA_TARGETS, model_digests, package_versions, rng_state, save_tensors, sha256, write_json


REFERENCE_REVISION = "f7b7db96a2572e69693d85e78a1d29d9e8489ef3"


def load_pipeline(args: argparse.Namespace) -> QwenImagePipeline:
    if args.fixture is None:
        configs = []
        for component in ("transformer", "text_encoder", "vae"):
            paths = sorted((args.model_path / component).glob("*.safetensors"))
            if not paths:
                raise FileNotFoundError(args.model_path / component)
            configs.append(ModelConfig(path=[str(path) for path in paths]))
        return QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=args.device, model_configs=configs,
            tokenizer_config=None, processor_config=ModelConfig(path=str(args.model_path / "processor")),
        )
    source = args.fixture
    pipe = QwenImagePipeline(device=args.device, torch_dtype=torch.bfloat16)
    config = Qwen2_5_VLConfig.from_dict(json.loads((source / "text_encoder_config.json").read_text()))
    config._attn_implementation = "sdpa"
    text = Qwen2_5_VLForConditionalGeneration(config)
    text.load_state_dict(load_file(source / "text_encoder.safetensors"), strict=True)
    with torch.device("meta"):
        pipe.text_encoder = QwenImageTextEncoder()
    pipe.text_encoder.model, pipe.text_encoder.lm_head = text.model, text.lm_head
    pipe.text_encoder.config = config
    vae_config = json.loads((source / "vae_config.json").read_text())
    pipe.vae = QwenImageVAE(**{name: vae_config[name] for name in
                            ("base_dim", "z_dim", "dim_mult", "num_res_blocks", "temperal_downsample")})
    pipe.vae.load_state_dict(load_file(source / "vae.safetensors"), strict=True)
    pipe.dit = make_reference_dit(load_file(source / "transformer.safetensors"))
    pipe.processor = Qwen2VLProcessor.from_pretrained(source / "processor", use_fast=True)
    return pipe.to(device=args.device, dtype=torch.bfloat16)


def select_samples(args: argparse.Namespace) -> tuple[Path, list[dict]]:
    if args.fixture is not None:
        return args.fixture, [
            {"prompt": "change the color", "image": "input_0.png", "edit_image": ["input_0.png"]},
            {"prompt": "change the color", "image": "input_0.png", "edit_image": ["input_0.png", "input_1.png"]},
        ]
    rows = json.loads(args.metadata.read_text())
    row, second = rows[args.sample_index], rows[args.second_edit_index]
    if len(row["edit_image"]) != 1 or len(second["edit_image"]) != 1:
        raise ValueError("Selected metadata rows must each contain one condition image")
    if sha256(args.data_root / row["edit_image"][0]) == sha256(args.data_root / second["edit_image"][0]):
        raise ValueError("The second condition image must differ from the first")
    return args.data_root, [row, {**row, "edit_image": [row["edit_image"][0], second["edit_image"][0]]}]


def fixed_loss(pipe: QwenImagePipeline, inputs: dict, timestep_id: int, noise: torch.Tensor) -> torch.Tensor:
    # Match the statement order of FlowMatchSFTLoss at the pinned revision, using caller-supplied random inputs.
    timestep = pipe.scheduler.timesteps[torch.tensor([timestep_id])].to(dtype=pipe.torch_dtype, device=pipe.device)
    inputs = {**inputs, "latents": pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)}
    target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    return loss * pipe.scheduler.training_weight(timestep)


def run(args: argparse.Namespace) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(Path(diffsynth.__file__).resolve().parent), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != REFERENCE_REVISION or package_versions()["transformers"] != "4.57.1":
        raise ValueError("Reference requires DiffSynth f7b7db96 and transformers 4.57.1")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    pipe = load_pipeline(args)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.dit.requires_grad_(False).train()
    inject_adapter_in_model(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                      target_modules=list(LORA_TARGETS)), pipe.dit)
    pipe.dit.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, parameter in pipe.dit.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
    save_tensors(args.output / "lora.safetensors", {name: value for name, value in pipe.dit.named_parameters()
                                                   if "lora_" in name})
    pipe.scheduler.set_timesteps(1000, training=True)
    indices = torch.stack([torch.argmin((pipe.scheduler.timesteps - value).abs())
                           for value in pipe.scheduler.timesteps.to(torch.bfloat16)])
    stable = min((index for index in range(200, 800) if indices[index].item() == index),
                 key=lambda index: abs(index - 500))
    root, rows = select_samples(args)
    crop = ImageCropAndResize(max_pixels=args.max_pixels, height_division_factor=16, width_division_factor=16)
    device = torch.device(args.device)
    captured = {}

    def positions(module, inputs):
        captured["position_ids"] = inputs[1].detach().clone()

    hook = pipe.text_encoder.model.language_model.rotary_emb.register_forward_pre_hook(positions)
    original_target, original_model_fn = pipe.scheduler.training_target, pipe.model_fn

    def target(sample, noise, timestep):
        captured["noise"] = noise.detach().clone()
        captured["timestep"] = timestep.detach().cpu()
        index = torch.argmin((pipe.scheduler.timesteps - timestep.cpu()).abs())
        captured.update(index=index, sigma=pipe.scheduler.sigmas[index],
                        weight=pipe.scheduler.linear_timesteps_weights[index])
        return original_target(sample, noise, timestep)

    def predict(**kwargs):
        value = original_model_fn(**kwargs)
        captured["prediction"] = value.detach().clone()
        return value

    pipe.scheduler.training_target, pipe.model_fn = target, predict
    cases = []
    for row in rows:
        count = len(row["edit_image"])
        for timestep_id in (1, 2, stable, None):
            name = f"images-{count}-timestep-{timestep_id if timestep_id is not None else 'random'}"
            directory = args.output / name
            directory.mkdir()
            captured.clear()
            pipe.dit.zero_grad(set_to_none=True)
            torch.manual_seed(args.seed + count)
            before = rng_state(device)
            with Image.open(root / row["image"]) as image:
                image = crop(image.convert("RGB"))
            edits = []
            for path in row["edit_image"]:
                with Image.open(root / path) as edit:
                    edits.append(crop(edit.convert("RGB")))
            inputs = pipeline_inputs(image, edits, row["prompt"])
            inputs[0]["rand_device"] = args.device
            inputs[0]["use_gradient_checkpointing"] = True
            with torch.no_grad():
                for unit in pipe.units:
                    inputs = pipe.unit_runner(unit, pipe, *inputs)
            merged = {**inputs[0], **inputs[1]}
            captured["target_latents"] = merged["input_latents"]
            captured["prompt_embeds"] = merged["prompt_emb"]
            for index, latent in enumerate(merged["edit_latents"]):
                captured[f"edit_latents_{index}"] = rearrange(latent, "b c (h p) (w q) -> b (h w) (c p q)", p=2, q=2)
            if timestep_id is None:
                loss = FlowMatchSFTLoss(pipe, **merged)
            else:
                noise = torch.randn_like(merged["input_latents"])
                loss = fixed_loss(pipe, merged, timestep_id, noise)
            loss.backward()
            captured["loss"] = loss.detach()
            captured.update({f"before.{name}": value for name, value in before.items()})
            captured.update({f"after.{name}": value for name, value in rng_state(device).items()})
            parameters = {name: parameter for name, parameter in pipe.dit.named_parameters() if parameter.requires_grad}
            for parameter_name, parameter in parameters.items():
                if parameter.grad is not None:
                    captured[f"gradient.{parameter_name}"] = parameter.grad
            save_tensors(directory / "values.safetensors", captured)
            cases.append({"name": name, "row": row, "timestep_id": timestep_id,
                          "unused_gradients": [name for name, parameter in parameters.items() if parameter.grad is None]})
            print(json.dumps({"case": name, "loss": loss.item()}), flush=True)
    hook.remove()
    manifest = {"reference_revision": revision, "packages": package_versions(), "device": args.device,
                "seed": args.seed, "lora_rank": args.lora_rank, "max_pixels": args.max_pixels,
                "fixture": str(args.fixture) if args.fixture else None,
                "model_path": str(args.model_path) if args.model_path else None,
                "base_sha256": model_digests(args.fixture or args.model_path, args.fixture is not None),
                "data_root": str(root.resolve()), "cases": cases,
                "images": {path: sha256(root / path) for row in rows for path in [row["image"], *row["edit_image"]]},
                "sha256": {str(path.relative_to(args.output)): sha256(path)
                           for path in sorted(args.output.rglob("*.safetensors"))}}
    write_json(args.output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--second-edit-index", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.fixture is None and any(value is None for value in (args.model_path, args.data_root, args.metadata)):
        parser.error("Provide --model-path, --data-root and --metadata for complete model validation")
    if args.lora_rank <= 0 or args.max_pixels <= 0:
        parser.error("--lora-rank and --max-pixels must be positive")
    run(args)


if __name__ == "__main__":
    main()
