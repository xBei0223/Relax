from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from safetensors.torch import load_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from relax.backends.fsdp.lora import inject_lora_adapter
from relax.models.qwen_image.sft import QwenImageEditSFTAdapter, QwenImageEditSFTModel

from step_common import model_digests, numerical_error, package_versions, restore_rng, rng_state, save_tensors, sha256, write_json


def load_model(args: argparse.Namespace, manifest: dict):
    adapter = QwenImageEditSFTAdapter()
    config = {"attention_implementation": args.attention_implementation,
              "transformer_gradient_checkpointing": True, "max_pixels": manifest["max_pixels"],
              "fixed_pixels_tiers": None, "seed": manifest["seed"]}
    if args.fixture is None:
        model = adapter.load_train_model(SimpleNamespace(model_path=str(args.model_path),
                                                         fsdp_trainable_mode="lora", i2i_sft_config=config))
    else:
        source = args.fixture
        transformer = QwenImageTransformer2DModel.from_config(json.loads((source / "transformer_config.json").read_text()))
        transformer.load_state_dict(load_file(source / "transformer.safetensors"), strict=True)
        transformer.set_attention_backend("native")
        transformer.enable_gradient_checkpointing()
        text_config = Qwen2_5_VLConfig.from_dict(json.loads((source / "text_encoder_config.json").read_text()))
        text_config._attn_implementation = "sdpa"
        text = Qwen2_5_VLForConditionalGeneration(text_config)
        text.load_state_dict(load_file(source / "text_encoder.safetensors"), strict=True)
        vae = AutoencoderKLQwenImage.from_config(json.loads((source / "vae_config.json").read_text()))
        vae.load_state_dict(load_file(source / "vae.safetensors"), strict=True)
        model = QwenImageEditSFTModel(transformer, text, vae, training_mode="lora")
        adapter.configure_processor(Qwen2VLProcessor.from_pretrained(source / "processor"), source / "processor")
    inject_lora_adapter(model, rank=manifest["lora_rank"], alpha=manifest["lora_rank"],
                        target_modules=adapter.lora_target_modules)
    model.to(device=args.device, dtype=torch.bfloat16)
    adapter.on_device_ready(model, torch.device(args.device))
    weights = load_file(args.reference / "lora.safetensors")
    parameters = {name: parameter for name, parameter in model.transformer.named_parameters() if "lora_" in name}
    if parameters.keys() != weights.keys():
        raise ValueError("LoRA parameter names differ between implementations")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])
    return adapter, model, config


def run(args: argparse.Namespace) -> None:
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if model_digests(args.fixture or args.model_path, args.fixture is not None) != manifest["base_sha256"]:
        raise ValueError("Base model or processor differs from the reference")
    for path, digest in manifest["sha256"].items():
        if sha256(args.reference / path) != digest:
            raise ValueError(f"Reference digest differs: {path}")
    root = args.data_root or Path(manifest["data_root"])
    for path, digest in manifest["images"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Input image digest differs: {path}")
    if package_versions()["transformers"] != "5.16.0":
        raise ValueError("Candidate requires transformers 5.16.0")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    adapter, model, config = load_model(args, manifest)
    captured = {}

    def positions(module, inputs):
        captured["position_ids"] = inputs[1].detach().clone()

    hook = model.text_encoder.model.language_model.rotary_emb.register_forward_pre_hook(positions)
    original_encode, original_predict = model.encode_conditions, model.predict
    original_target = model.schedule.training_target

    def encode(data):
        values = original_encode(data)
        captured.update({name: values[name] for name in ("prompt_embeds", "target_latents")})
        captured.update({f"edit_latents_{index}": value for index, value in enumerate(values["edit_latents"])})
        return values

    def predict(noisy, condition, timestep):
        value = original_predict(noisy, condition, timestep)
        captured["prediction"] = value.detach().clone()
        return value

    def target(sample, noise, timestep):
        captured["noise"] = noise.detach().clone()
        captured["timestep"] = timestep
        index = model.schedule.lookup_index(timestep)
        captured.update(index=index, sigma=model.schedule.sigmas[index],
                        weight=model.schedule.linear_timesteps_weights[index])
        return original_target(sample, noise, timestep)

    model.encode_conditions, model.predict, model.schedule.training_target = encode, predict, target
    results = []
    for case in manifest["cases"]:
        expected = load_file(args.reference / case["name"] / "values.safetensors")
        data = adapter.prepare_sample(case["row"], root, {**config, "conditioning_images": len(case["row"]["edit_image"])})
        model.zero_grad(set_to_none=True)
        captured.clear()
        device = torch.device(args.device)
        restore_rng({name.removeprefix("before."): value for name, value in expected.items()
                     if name.startswith("before.")}, device)
        timestep_id = case["timestep_id"]
        if timestep_id is None:
            loss = model(data)
        else:
            loss = model(data, timestep_id=torch.tensor([timestep_id]), noise=expected["noise"].to(device))
        loss.backward()
        captured["loss"] = loss.detach()
        parameters = {name: parameter for name, parameter in model.transformer.named_parameters() if parameter.requires_grad}
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                captured[f"gradient.{name}"] = parameter.grad
        expected_gradients = {name for name in expected if name.startswith("gradient.")}
        actual_gradients = {name for name in captured if name.startswith("gradient.")}
        if expected_gradients != actual_gradients or set(case["unused_gradients"]) != {
            name for name, parameter in parameters.items() if parameter.grad is None
        }:
            raise ValueError(f"Gradient parameter names differ: {case['name']}")
        comparisons = {name: numerical_error(value, expected[name]) for name, value in captured.items()}
        required = ("position_ids", "index", "sigma", "weight", "timestep", "noise")
        if timestep_id is None:
            state = rng_state(device)
            for name, value in state.items():
                comparisons[f"after.{name}"] = numerical_error(value, expected[f"after.{name}"])
            required = (*required, *(f"after.{name}" for name in state))
        passed = all(comparisons[name]["equal"] for name in required) and comparisons["loss"]["relative_l2"] <= 0.01
        directory = args.output / case["name"]
        directory.mkdir()
        save_tensors(directory / "values.safetensors", captured)
        write_json(directory / "comparison.json", comparisons)
        results.append({"case": case["name"], "passed": passed,
                        "loss_relative_error": comparisons["loss"]["relative_l2"],
                        "gradient_max_relative_l2": max(comparisons[name]["relative_l2"] for name in actual_gradients),
                        "gradient_tensors_above_one_percent": sum(comparisons[name]["relative_l2"] > 0.01 for name in actual_gradients)})
        print(json.dumps(results[-1]), flush=True)
    hook.remove()
    write_json(args.output / "summary.json", {"packages": package_versions(), "reference": str(args.reference),
                                              "device": args.device, "cases": results,
                                              "gradient_policy": "report_standard_diffusers_difference"})
    if not all(case["passed"] for case in results):
        raise ValueError("Reference comparison failed; inspect comparison.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-implementation", choices=("torch", "flash_attention_2"), default="torch")
    args = parser.parse_args()
    if args.fixture is None and args.model_path is None:
        parser.error("Provide --model-path or --fixture")
    run(args)


if __name__ == "__main__":
    main()
