#!/usr/bin/env python3
"""Run the Python MLX Qwen vision tower on a deterministic small image grid.

The Flash-Next vision weights are compatible with the Qwen3.5/Qwen3-VL vision
implementation shipped by mlx-vlm.  Only the shard containing vision_tower is
opened; the 113 GB language group is never loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import mlx_vlm.models.qwen3_vl.vision as qwen_vision
from mlx_vlm.models.qwen3_vl.config import VisionConfig


def merge_order_pixels(height: int, width: int, patch: int, merge: int) -> mx.array:
    import numpy as np

    rng = np.random.default_rng(3808)
    pixels = rng.standard_normal((1, height, width, 3), dtype=np.float32) * 0.1
    pixels = mx.array(pixels.astype(np.float32)).astype(mx.bfloat16)
    grid_h, grid_w = height // patch, width // patch
    patches = []
    for block_h in range(grid_h // merge):
        for block_w in range(grid_w // merge):
            for inner_h in range(merge):
                for inner_w in range(merge):
                    h0 = (block_h * merge + inner_h) * patch
                    w0 = (block_w * merge + inner_w) * patch
                    patch_pixels = pixels[0, h0:h0 + patch, w0:w0 + patch]
                    temporal = mx.stack([patch_pixels, patch_pixels], axis=0)
                    patches.append(temporal.transpose(3, 0, 1, 2))
    return pixels, mx.stack(patches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_data = json.loads((args.model_dir / "config.json").read_text())
    vision_data = dict(config_data["vision_config"])
    vision_data["model_type"] = "qwen3_5"
    accepted = {field.name for field in fields(VisionConfig)}
    config = VisionConfig(**{
        key: value for key, value in vision_data.items() if key in accepted
    })
    model = qwen_vision.VisionModel(config)

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted({
        shard for name, shard in index["weight_map"].items()
        if name.startswith("vision_tower.")
    })
    if len(shard_names) != 1:
        raise RuntimeError(f"expected one vision shard, found {shard_names}")
    loaded = mx.load(str(args.model_dir / shard_names[0]))
    weights = {
        name.removeprefix("vision_tower."): value
        for name, value in loaded.items()
        if name.startswith("vision_tower.")
    }
    model.load_weights(list(weights.items()), strict=True)

    pixels, patches = merge_order_pixels(
        height=32, width=64, patch=config.patch_size, merge=config.spatial_merge_size)
    grid = mx.array([[1, 2, 4]], dtype=mx.int32)
    hidden = model.patch_embed(patches)
    position = model.fast_pos_embed_interpolate(grid)
    trace = {
        "patch_embed": hidden,
        "position_embed": position,
    }
    hidden = hidden + position
    trace["pre_block"] = hidden
    rotary = model.rot_pos_emb(grid)
    for index, block in enumerate(model.blocks):
        normalized = block.norm1(hidden)
        attention_output = block.attn(
            normalized, cu_seqlens=mx.array([0, 8], dtype=mx.int32),
            rotary_pos_emb=rotary)
        after_attention = hidden + attention_output
        mlp_output = block.mlp(block.norm2(after_attention))
        hidden = after_attention + mlp_output
        trace[f"block_{index}_attention"] = attention_output
        trace[f"block_{index}_after_attention"] = after_attention
        trace[f"block_{index}_mlp"] = mlp_output
        trace[f"block_{index}"] = hidden
    output = model.merger(hidden)
    trace["output"] = output
    mx.eval(pixels, *trace.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace["pixels"] = pixels
    mx.save_safetensors(str(args.output), trace, metadata={
        "kind": "qwen4_exp_vision_real_checkpoint",
        "height": "32",
        "width": "64",
        "patch_size": str(config.patch_size),
        "merge_size": str(config.spatial_merge_size),
        "shard": shard_names[0],
    })
    print(f"vision fixture écrite : {args.output}")
    print(f"pixels: {pixels.shape} · patches: {patches.shape} · output: {output.shape}")


if __name__ == "__main__":
    main()
