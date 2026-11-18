# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from safetensors.torch import load_file

from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import validate_i2i_sft_config
from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.engine.sft.condition_cache import condition_key
from relax.engine.sft.condition_tracker import ConditionTracker, summarize_baseline_dir
from relax.models.qwen_image.sft import (
    QWEN_IMAGE_EDIT_TRACKER_SCHEMA,
    QwenImageEditSFTAdapter,
    image_digest,
    load_image,
    preprocess_pixels,
    resize_condition,
)
from relax.models.qwen_image.sft_export import export_adapter, export_full
from relax.models.qwen_image.training_schedule import QwenImageTrainingSchedule
from tests.cpu.models.qwen_image_edit_helpers import CONFIG, FIXTURES, build_adapter, build_model, small_sample


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_reference_files_and_schedule() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest
    expected = json.loads((FIXTURES / "schedule.json").read_text())
    schedule = QwenImageTrainingSchedule()
    torch.testing.assert_close(schedule.timesteps, torch.tensor(expected["timesteps"]), rtol=0, atol=0)
    quantized = schedule.timesteps.to(torch.bfloat16)
    assert quantized.float().tolist() == expected["quantized_timesteps"]
    for index, timestep in enumerate(quantized):
        selected = schedule.lookup_index(timestep)
        assert selected.item() == expected["indices"][index]
        assert schedule.sigmas[selected].item() == expected["sigmas"][index]
        torch.testing.assert_close(
            schedule.training_weight(timestep),
            torch.tensor(expected["weights"][index], dtype=torch.float32),
            rtol=1e-6,
            atol=1e-6,
        )
    assert sum(index != actual for index, actual in enumerate(expected["indices"])) == 624


def test_preprocessing_and_processor_match_reference() -> None:
    adapter = build_adapter()
    pixels = load_file(FIXTURES / "pixels.safetensors")
    processor_outputs = load_file(FIXTURES / "processor_outputs.safetensors")
    for index, case in enumerate(json.loads((FIXTURES / "preprocess.json").read_text())):
        target = load_image(FIXTURES / case["input"], case["max_pixels"], case["fixed_pixels_budget"])
        edit = resize_condition(target, 1024**2)
        visual = resize_condition(edit, 384**2)
        for field, image in (("target", target), ("vae", edit), ("vl", visual)):
            actual = image_digest(image)
            actual["size"] = list(actual["size"])
            assert actual == case[field]
        assert torch.equal(preprocess_pixels(target), pixels[f"case_{index}_target"])
        assert torch.equal(preprocess_pixels(edit), pixels[f"case_{index}_edit"])
        outputs = adapter.processor(
            text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[visual], padding=True, return_tensors="pt"
        )
        for field in ("pixel_values", "image_grid_thw"):
            assert torch.equal(outputs[field], processor_outputs[f"case_{index}_{field}"])


def test_first_parallel_prepare_sample() -> None:
    adapter = build_adapter()
    rows = [{"prompt": f"change {i}", "image": f"input_{i}.png", "edit_image": [f"input_{i}.png"]} for i in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda row: adapter.prepare_sample(row, FIXTURES, CONFIG), rows))
    for row, actual in zip(rows, results, strict=True):
        expected = adapter.prepare_sample(row, FIXTURES, CONFIG)
        assert actual["tracker_keys"] == expected["tracker_keys"]
        for field in actual["text_inputs"]:
            assert torch.equal(actual["text_inputs"][field], expected["text_inputs"][field])
        assert torch.equal(actual["target_pixels"], expected["target_pixels"])


@pytest.mark.parametrize("count", [1, 2])
def test_text_encoding_uses_reference_image_positions(count: int, record_property) -> None:
    adapter, model = build_adapter(), build_model()
    config = {**CONFIG, "conditioning_images": count}
    data = adapter.prepare_sample(
        {"prompt": "change the color", "image": "input_0.png", "edit_image": [f"input_{i}.png" for i in range(count)]},
        FIXTURES,
        config,
    )
    positions = []

    def capture(module, args):
        positions.append(args[1].clone())

    hook = model.text_encoder.model.language_model.rotary_emb.register_forward_pre_hook(capture)
    with torch.no_grad():
        actual = model.text_encoder.model(**data["text_inputs"], use_cache=False).last_hidden_state[:, 64:]
        expected = load_file(FIXTURES / "text_encoding.safetensors")
        assert torch.equal(positions[-1], expected[f"images_{count}_position_ids"])
        reference = expected[f"images_{count}_prompt_embeds"]
        error = (actual - reference).norm() / reference.norm()
        record_property("text_relative_l2", error.item())
        record_property("text_max_absolute_error", (actual - reference).abs().max().item())
        assert error <= 1e-5
        model.text_encoder.model.rope_deltas = None
        model.text_encoder.model(
            **{k: v for k, v in data["text_inputs"].items() if k != "mm_token_type_ids"}, use_cache=False
        )
        assert not torch.equal(positions[-1], positions[-2])
    hook.remove()


def test_vae_encoding_matches_reference(record_property) -> None:
    model = build_model(dtype=torch.bfloat16)
    pixels = load_file(FIXTURES / "pixels.safetensors")
    reference = load_file(FIXTURES / "vae_latents.safetensors")
    with torch.no_grad():
        for key, value in pixels.items():
            actual = model.encode_vae(value)
            error = (actual.float() - reference[key].float()).norm() / reference[key].float().norm()
            record_property(f"{key}_relative_l2", error.item())
            record_property(f"{key}_max_absolute_error", (actual.float() - reference[key].float()).abs().max().item())
            assert error <= 1e-2
        latent = model.vae.encode(pixels["case_0_target"].unsqueeze(2)).latent_dist.mode().squeeze(2)
        mean = torch.tensor(model.vae.config.latents_mean).to(torch.bfloat16).view(1, -1, 1, 1)
        inverse = (1 / torch.tensor(model.vae.config.latents_std)).to(torch.bfloat16).view(1, -1, 1, 1)
        assert torch.equal(model.encode_vae(pixels["case_0_target"]), (latent - mean) * inverse)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_training_step_preserves_diffsynth_rng_and_reports_numerics(mode: str, record_property) -> None:
    adapter, model = build_adapter(), build_model(mode, dtype=torch.bfloat16)
    data = adapter.prepare_sample(
        {"prompt": "change the color", "image": "input_0.png", "edit_image": ["input_0.png"]}, FIXTURES, CONFIG
    )
    reference = load_file(
        FIXTURES / ("lora_training_step.safetensors" if mode == "lora" else "training_step.safetensors")
    )
    if mode == "lora":
        with torch.no_grad():
            for name, parameter in model.transformer.named_parameters():
                if "lora_" in name:
                    parameter.copy_(reference[f"weight.{name}"])
    predictions = []
    original_predict = model.predict

    def capture(noisy, condition, timestep):
        prediction = original_predict(noisy, condition, timestep)
        predictions.append(prediction.detach())
        return prediction

    model.predict = capture
    torch.set_rng_state(reference["rng_before"])
    loss = model(data)
    loss.backward()
    assert torch.equal(torch.get_rng_state(), reference["rng_after"])
    relative_loss = ((loss.detach() - reference["loss"]) / reference["loss"]).abs().item()
    record_property("loss_relative_error", relative_loss)
    assert relative_loss <= 0.01
    error = (predictions[0].float() - reference["prediction"].float()).norm() / reference["prediction"].float().norm()
    record_property("prediction_relative_l2", error.item())
    assert error <= 0.01
    if mode == "full":
        return
    actual_gradients = {
        f"gradient.{name}": parameter.grad
        for name, parameter in model.transformer.named_parameters()
        if parameter.grad is not None
    }
    assert actual_gradients.keys() == {name for name in reference if name.startswith("gradient.")}
    max_error = 0.0
    above_one_percent = 0
    for name, gradient in actual_gradients.items():
        expected = reference[name].float()
        relative = (gradient.float() - expected).norm() / expected.norm().clamp_min(1e-20)
        assert torch.isfinite(gradient).all(), name
        max_error = max(max_error, relative.item())
        above_one_percent += int(relative > 0.01)
    record_property("gradient_max_relative_l2", max_error)
    record_property("gradient_tensors_above_one_percent", above_one_percent)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cache_continuous_training_preserves_gradients_updates_rng(mode: str) -> None:
    histories = []
    for enabled in (False, True):
        adapter, model = build_adapter(), build_model(mode)
        samples = [small_sample(adapter, i) for i in (0, 1)]
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert all(p.dtype == torch.float32 for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {}) if enabled else None
        torch.manual_seed(1991)
        history = []
        for step in range(4):
            data = samples[step % 2]
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in parameters]
            if cache is not None:
                cache.decide([cache.prepare(data)])
            loss = model(data)
            loss.backward()
            gradients = [p.grad.clone() if p.grad is not None else None for p in parameters]
            optimizer.step()
            history.append(
                [
                    loss.detach(),
                    *gradients,
                    *(p.detach() - old for p, old in zip(parameters, before, strict=True)),
                    torch.get_rng_state(),
                ]
            )
            if cache is not None:
                cache.end_micro()
                if step == 1:
                    saved_stats = cache.stats()
                    stats_copy = copy.deepcopy(saved_stats)
        if cache is not None:
            assert saved_stats == stats_copy
            stats = cache.stats()
            assert stats["effective_reuses"] == 2
            assert stats["restore_ms"] > 0 and stats["build_host_ms"] > 0
            for counter in adapter.condition_skip_counters:
                assert stats[counter] == 2 and saved_stats[counter] == 0
            cache.close()
        histories.append(history)
    for online, cached in zip(*histories, strict=True):
        for expected, actual in zip(online, cached, strict=True):
            if expected is None:
                assert actual is None
                continue
            assert torch.isfinite(actual).all()
            assert torch.equal(expected, actual)


def test_cache_identity_versions_and_shared_tensors() -> None:
    adapter, model = build_adapter(), build_model()
    data = small_sample(adapter)
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})

    def identity(sample):
        request = adapter.prepare_condition_request(model, sample)["edit_inputs"]
        return condition_key(cache.specs[0], request.metadata, 0)

    original = identity(data)
    assert original == identity(copy.deepcopy(data))
    for field in ("prompt", "fixed_pixels_budget"):
        changed = copy.deepcopy(data)
        changed[field] = "new prompt" if field == "prompt" else 2048
        assert identity(changed) != original
    for field in ("target", "vae", "vl"):
        changed = copy.deepcopy(data)
        digest = changed["digests"][field]
        (digest if isinstance(digest, dict) else digest[0])["sha256"] = "changed"
        assert identity(changed) != original
    for field, value in (("max_pixels", 8192), ("deterministic_attention", True)):
        model._condition_config[field] = value
        assert identity(data) != original
        model._condition_config = dict(CONFIG)
    model.torch_dtype = torch.bfloat16
    assert identity(data) != original
    model.torch_dtype = torch.float32
    model.text_encoder.config._attn_implementation = "eager"
    assert identity(data) != original
    model.text_encoder.config._attn_implementation = "sdpa"
    cache.version_tags["edit_inputs"] = "encoder-v2"
    assert identity(data) != original
    cache.version_tags["edit_inputs"] = None
    adapter.processor.tokenizer.add_tokens(["extra-token"])
    cache.bump_encoder_version()
    cache.decide([cache.prepare(data)])
    assert identity(data) != original
    model(data).backward()
    cache.end_micro()
    cache.decide([cache.prepare(data)])
    stored = cache.values["edit_inputs"]
    consumed = []
    predict = model.predict

    def capture(noisy, condition, timestep):
        consumed.append(condition)
        return predict(noisy, condition, timestep)

    model.predict = capture
    model(data).backward()
    assert consumed[0]["prompt_embeds"] is stored["prompt_embeds"]
    assert consumed[0]["target_latents"] is stored["target_latents"]
    assert consumed[0]["edit_latents"][0] is stored["edit_latents"][0]
    cache.end_micro()
    stored["prompt_embeds"].add_(1)
    with pytest.raises(ValueError, match="changed"):
        cache.prepare(data)
    assert cache.stats()["entries"] == 0
    cache.close()


def test_prefetched_processor_inputs_keep_distinct_cache_identity() -> None:
    adapter, model = build_adapter(), build_model()
    row = {"prompt": "change the color", "image": "input_0.png", "edit_image": ["input_0.png"]}
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})
    prepared = adapter.prepare_sample(row, FIXTURES, CONFIG)
    adapter.processor.image_processor.rescale_factor *= 0.5
    cache.bump_encoder_version()
    cache.decide([cache.prepare(prepared)])
    old_condition = model.encode_conditions(prepared)
    cache.put("edit_inputs", old_condition)
    cache.end_micro()

    current = adapter.prepare_sample(row, FIXTURES, CONFIG)
    expected = model.encode_conditions(current)
    assert not torch.equal(prepared["text_inputs"]["pixel_values"], current["text_inputs"]["pixel_values"])
    assert not torch.equal(old_condition["prompt_embeds"], expected["prompt_embeds"])
    cache.decide([cache.prepare(current)])
    assert not cache.values
    model(current).backward()
    cache.end_micro()
    cache.decide([cache.prepare(current)])
    actual = cache.values["edit_inputs"]
    assert torch.equal(actual["prompt_embeds"], expected["prompt_embeds"])
    assert torch.equal(actual["target_latents"], expected["target_latents"])
    cache.end_micro()
    cache.close()


def test_native_attention_mask_and_training_boundaries() -> None:
    model = build_model()
    inputs = {
        "hidden_states": torch.randn(1, 8, 64),
        "encoder_hidden_states": torch.randn(1, 11, 32),
        "timestep": torch.tensor([0.5]),
        "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
        "return_dict": False,
    }
    one = model.transformer(**inputs, encoder_hidden_states_mask=None)[0]
    two = model.transformer(**inputs, encoder_hidden_states_mask=torch.ones(1, 11, dtype=torch.bool))[0]
    assert torch.equal(one, two)
    actor = SimpleNamespace(
        model=model,
        adapter=QwenImageEditSFTAdapter(),
        _trainable_mode="full",
        args=Namespace(loss_type="sft", generation_task="i2i"),
    )
    FSDPTrainRayActor._assert_trainable_boundary(actor)
    parameter = next(model.text_encoder.parameters())
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen modules"):
        FSDPTrainRayActor._assert_trainable_boundary(actor)
    parameter.requires_grad_(False)
    next(model.transformer.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="all transformer params trainable"):
        FSDPTrainRayActor._assert_trainable_boundary(actor)
    lora = build_model("lora")
    assert all(
        name.startswith("transformer.") and ".lora_" in name for name, p in lora.named_parameters() if p.requires_grad
    )
    assert not lora.text_encoder.training and not lora.vae.training


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_checkpoint_export_roundtrip(tmp_path: Path, mode: str) -> None:
    model = build_model(mode, torch.bfloat16)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    contract = {"trainable_mode": mode, "lora": {"rank": 2, "alpha": 2}}
    (checkpoint / "adapter_contract.json").write_text(json.dumps(contract))
    (checkpoint / "COMMITTED").touch()
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if mode == "full" or ".lora_" in name
    }
    dcp.save({"model": state}, checkpoint_id=checkpoint / "fsdp")
    if mode == "lora":
        output = tmp_path / "adapter.safetensors"
        export_adapter(checkpoint, output)
        fresh = build_model("full", torch.bfloat16)
        pipeline = QwenImageEditPlusPipeline(
            transformer=fresh.transformer,
            vae=fresh.vae,
            text_encoder=fresh.text_encoder,
            tokenizer=None,
            processor=None,
            scheduler=None,
        )
        pipeline.load_lora_weights(str(output))
        inputs = {
            "hidden_states": torch.randn(1, 4, 64, dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn(1, 5, 32, dtype=torch.bfloat16),
            "timestep": torch.tensor([0.5], dtype=torch.bfloat16),
            "img_shapes": [[(1, 2, 2)]],
            "return_dict": False,
        }
        assert torch.equal(model.transformer(**inputs)[0], pipeline.transformer(**inputs)[0])
    else:
        base = tmp_path / "base"
        model.transformer.save_pretrained(base / "transformer")
        output = export_full(checkpoint, base, tmp_path / "export")
        fresh = QwenImageTransformer2DModel.from_pretrained(output, torch_dtype=torch.bfloat16)
        assert fresh.state_dict().keys() == model.transformer.state_dict().keys()
        for name, tensor in fresh.state_dict().items():
            assert torch.equal(tensor, model.transformer.state_dict()[name])


def test_configuration_validation() -> None:
    def arguments(config):
        return Namespace(
            loss_type="sft",
            generation_task="i2i",
            train_backend="fsdp",
            model_path="/model",
            prompt_data="/data.json",
            i2i_sft_config={"data_root": "/data", "dataset_size": 2, **config},
            global_batch_size=2,
            rollout_batch_size=2,
            fsdp_trainable_mode="full",
            num_rollout=2,
            num_epoch=None,
            seed=42,
            model_adapter_path="relax.models.qwen_image.sft.QwenImageEditSFTAdapter",
        )

    validate_i2i_sft_config(arguments({"transformer_gradient_checkpointing": True}))
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="transformer_gradient_checkpointing"):
            validate_i2i_sft_config(arguments({"transformer_gradient_checkpointing": value}))
    with pytest.raises(ValueError, match="unknown|Unknown|Unsupported"):
        validate_i2i_sft_config(arguments({"unknown": True}))


def test_adapter_loads_component_directories(tmp_path: Path) -> None:
    source = build_model()
    source.transformer.save_pretrained(tmp_path / "transformer")
    source.text_encoder.save_pretrained(tmp_path / "text_encoder")
    source.vae.save_pretrained(tmp_path / "vae")
    shutil.copytree(FIXTURES / "processor", tmp_path / "processor")
    adapter = QwenImageEditSFTAdapter()
    loaded = adapter.load_train_model(
        Namespace(
            model_path=str(tmp_path),
            fsdp_trainable_mode="full",
            i2i_sft_config={**CONFIG, "attention_implementation": "torch"},
        )
    )
    adapter.prepare_for_sharding(loaded)
    adapter.on_device_ready(loaded, torch.device("cpu"))
    assert loaded.transformer.is_gradient_checkpointing
    assert all(parameter.dtype == torch.bfloat16 for parameter in loaded.parameters())
    assert all(
        parameter.requires_grad == name.startswith("transformer.") for name, parameter in loaded.named_parameters()
    )
    data = small_sample(adapter)
    assert torch.isfinite(loaded(data))


def test_tracker_qwen_schema_across_ranks_and_cli(tmp_path: Path) -> None:
    adapter, model = build_adapter(), build_model()
    data = small_sample(adapter, count=2)
    config = {**CONFIG, "conditioning_images": 2}
    trackers = [
        ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=tmp_path,
            warmup_steps=0,
            schema=QWEN_IMAGE_EDIT_TRACKER_SCHEMA,
        )
        for rank in (0, 1)
    ]
    for tracker in trackers:
        model.tracker = tracker
        for step in range(2):
            tracker.begin_step(step, torch.device("cpu"))
            tracker.begin_micro(0, tracker.rank, {"_prepared": data}, config, "full", 32, 32)
            condition = model.encode_conditions(data)
            tracker.end_step(step, 0.001, 0.01)
        result = tracker.save_summary()
        for kind in ("prompt_encode", "target_vae_encode"):
            assert result["encoding_stats"][kind]["repeated_calls"] == 1
        assert result["encoding_stats"]["ref_vae_encode"]["repeated_calls"] == 2
        assert result["encoding_stats"]["ref_vae_encode"]["encoded_images"] == 4
        assert result["feature_tensor_bytes"]["prompt_embeds_bytes"]["max_bytes"] == condition["prompt_embeds"].nbytes
        assert result["feature_tensor_bytes"]["edit_latents_bytes"]["max_bytes"] == sum(
            tensor.nbytes for tensor in condition["edit_latents"]
        )
        assert (
            result["feature_tensor_bytes"]["target_latents_bytes"]["max_bytes"] == condition["target_latents"].nbytes
        )
        with pytest.raises(ValueError, match="Unknown encode"):
            tracker.record_encode("undefined", 1)
        with pytest.raises(ValueError, match="Unknown feature"):
            tracker.record_encode("prompt_encode", 1, {"undefined": 1})
    summary = summarize_baseline_dir(tmp_path)
    assert summary["tracker_schema"] == QWEN_IMAGE_EDIT_TRACKER_SCHEMA.name
    assert summary["encoding_stats"]["prompt_encode"]["global_repeated_calls"] == 3
    assert summary["encoding_stats"]["ref_vae_encode"]["global_repeated_calls"] == 6
    assert summary["repeat_evidence"]["global_repeat_count"] == 3
    expected = (tmp_path / "baseline_summary.json").read_bytes()
    subprocess.run(
        [sys.executable, "-m", "relax.engine.sft.condition_tracker", str(tmp_path)], check=True, capture_output=True
    )
    assert (tmp_path / "baseline_summary.json").read_bytes() == expected
