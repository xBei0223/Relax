# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import json
import os


def boolean(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def configuration() -> dict:
    tiers = os.environ.get("FIXED_PIXELS_TIERS", "").strip()
    config = {
        "data_root": os.environ["DATA_ROOT"],
        "dataset_size": int(os.environ["DATASET_SIZE"]),
        "max_pixels": int(os.environ.get("MAX_PIXELS", "1048576")),
        "conditioning_images": int(os.environ.get("CONDITIONING_IMAGES", "1")),
        "attention_implementation": os.environ.get("ATTENTION_IMPLEMENTATION", "flash_attention_2"),
        "deterministic_attention": boolean("DETERMINISTIC_ATTENTION", "0"),
        "transformer_gradient_checkpointing": boolean("TRANSFORMER_GRADIENT_CHECKPOINTING", "1"),
        "fixed_pixels_tiers": [int(value.strip()) for value in tiers.split(",")] if tiers else None,
        "seed": int(os.environ.get("SEED", "42")),
        "image_prefetch_depth": int(os.environ.get("IMAGE_PREFETCH_DEPTH", "2")),
        "image_prefetch_workers": int(os.environ.get("IMAGE_PREFETCH_WORKERS", "2")),
        "dp_tier_micro_balance": boolean("IMAGE_DP_TIER_MICRO_BALANCE", "0"),
        "dp_tier_rank_rotation": boolean("IMAGE_DP_TIER_RANK_ROTATION", "0"),
        "condition_cache_enabled": boolean("CONDITION_CACHE_ENABLED", "0"),
        "condition_cache_max_entries": int(os.environ.get("CONDITION_CACHE_MAX_ENTRIES", "256")),
        "condition_cache_max_bytes": int(os.environ.get("CONDITION_CACHE_MAX_BYTES", "2147483648")),
    }
    if os.environ.get("BASELINE_STATS_DIR", "").strip():
        config["baseline_stats_dir"] = os.environ["BASELINE_STATS_DIR"].strip()
        config["baseline_warmup_steps"] = int(os.environ.get("BASELINE_WARMUP_STEPS", "5"))
    return config


if __name__ == "__main__":
    print(json.dumps(configuration()))
