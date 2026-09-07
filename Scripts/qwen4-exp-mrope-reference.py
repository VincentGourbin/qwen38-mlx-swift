#!/usr/bin/env python3
"""Export a deterministic multimodal MRoPE reference fixture.

The fixture deliberately contains a small image sequence followed by text.
It validates both the three-row multimodal clock and the interleaved partial
RoPE table without loading the Flash-Next checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx


def multimodal_positions(
    input_ids: list[int], image_token_id: int, vision_start_token_id: int,
    image_height: int, image_width: int, spatial_merge_size: int = 2,
) -> mx.array:
    grid_h = image_height // spatial_merge_size
    grid_w = image_width // spatial_merge_size
    image_count = grid_h * grid_w
    rows = [[], [], []]
    start = input_ids.index(vision_start_token_id)
    marker = input_ids.index(image_token_id, start + 1)
    text_prefix = marker
    for axis in range(3):
        rows[axis].extend(range(text_prefix))
    image_start = text_prefix
    for h in range(grid_h):
        for w in range(grid_w):
            rows[0].append(image_start)
            rows[1].append(image_start + h)
            rows[2].append(image_start + w)
    maximum = image_start + max(grid_h - 1, grid_w - 1)
    after = marker + image_count
    for position in range(len(input_ids) - after):
        value = maximum + 1 + position
        for axis in range(3):
            rows[axis].append(value)
    if any(len(row) != len(input_ids) for row in rows):
        raise RuntimeError("position fixture has an invalid length")
    return mx.array(rows, dtype=mx.int32)[:, None, :]


def rope_tables(position_ids: mx.array, rotary_dim: int = 64):
    half = rotary_dim // 2
    inv_freq = 1.0 / mx.power(
        mx.array(10_000_000.0),
        (2.0 * mx.arange(half, dtype=mx.float32)) / rotary_dim,
    )
    frequencies = (
        inv_freq.reshape(1, 1, half, 1)
        * position_ids.reshape(3, 1, 1, -1).astype(mx.float32)
    ).transpose(0, 1, 3, 2)
    interleaved = frequencies[0]
    sections = [11, 11, 10]
    for axis in (1, 2):
        for section_index in range(sections[axis]):
            index = axis + section_index * 3
            if index < half:
                interleaved = mx.concatenate(
                    [interleaved[..., :index], frequencies[axis][..., index:index + 1],
                     interleaved[..., index + 1:]], axis=-1)
    full = mx.concatenate([interleaved, interleaved], axis=-1)
    return mx.cos(full), mx.sin(full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image_token_id = 99
    vision_start_token_id = 98
    # 4x6 pre-merge patches with merge=2 -> six image markers.
    input_ids = [10, vision_start_token_id] + [image_token_id] * 6 + [11, 12, 13]
    position_ids = multimodal_positions(
        input_ids, image_token_id, vision_start_token_id, image_height=4, image_width=6)
    cos, sin = rope_tables(position_ids)
    arrays = {
        "input_ids": mx.array(input_ids, dtype=mx.int32)[None, :],
        "position_ids": position_ids,
        "cos": cos,
        "sin": sin,
    }
    mx.eval(*arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), arrays, metadata={
        "kind": "qwen4_exp_multimodal_mrope",
        "image_token_id": str(image_token_id),
        "vision_start_token_id": str(vision_start_token_id),
        "image_height": "4",
        "image_width": "6",
        "spatial_merge_size": "2",
    })
    print(f"MRoPE fixture écrite : {args.output}")


if __name__ == "__main__":
    main()
