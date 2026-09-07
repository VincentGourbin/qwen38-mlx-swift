#!/usr/bin/env python3
"""Export a deterministic, weight-free QSA reference fixture.

The fixture starts after ``index_qk_proj``.  This isolates the numerically
fragile QSA path (RMSNorm, interleaved partial MRoPE, block pooling, float32
scores and discrete mask selection) from checkpoint loading and quantized
matmul.  The resulting safetensors can be consumed by a future Swift parity
command without loading Flash-Next's full checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx


def rope_tables(position_ids: mx.array, rotary_dim: int = 64):
    if position_ids.ndim == 2:
        zeros = mx.zeros(position_ids.shape, dtype=mx.int32)
        position_ids = mx.stack([position_ids, zeros, zeros])
    batch, sequence = position_ids.shape[1:]
    half = rotary_dim // 2
    inv_freq = 1.0 / mx.power(
        mx.array(10_000_000.0),
        (2.0 * mx.arange(half, dtype=mx.float32)) / rotary_dim,
    )
    frequencies = (
        inv_freq.reshape(1, 1, half, 1)
        * position_ids.reshape(3, batch, 1, sequence).astype(mx.float32)
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


def apply_rope(x: mx.array, position_ids: mx.array):
    rotary_dim = 64
    cos, sin = rope_tables(position_ids, rotary_dim)
    rotated = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    half = rotary_dim // 2
    rotate_half = mx.concatenate([-rotated[..., half:], rotated[..., :half]], axis=-1)
    rotated = rotated * cos[:, None, ...] + rotate_half * sin[:, None, ...]
    return mx.concatenate([rotated, passthrough], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3802)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--compress-ratio", type=int, default=4)
    args = parser.parse_args()
    if args.head_dim < 64 or args.head_dim % 2:
        parser.error("--head-dim doit être pair et >= 64")

    # Deterministic values are generated on the host so this probe remains
    # stable across MLX random-generator changes.
    import numpy as np

    rng = np.random.default_rng(args.seed)
    projected = mx.array(rng.standard_normal(
        (args.batch, args.sequence, args.n_heads + 1, args.head_dim),
        dtype=np.float32).astype(np.float16))
    positions = mx.arange(args.sequence, dtype=mx.int32).reshape(1, args.sequence)
    zeros = mx.zeros((1, args.sequence), dtype=mx.int32)
    position_ids = mx.stack([positions, zeros, zeros])

    q = projected[..., :args.n_heads, :]
    raw_keys = projected[..., args.n_heads:, :].squeeze(2)
    q = mx.fast.rms_norm(q, mx.ones((args.head_dim,)), 1e-6).transpose(0, 2, 1, 3)
    q = apply_rope(q, position_ids)
    pooled = raw_keys[:, :args.sequence // args.compress_ratio * args.compress_ratio]
    pooled = pooled.reshape(
        args.batch, -1, args.compress_ratio, args.head_dim).astype(mx.float32).mean(axis=2)
    pooled = mx.expand_dims(mx.fast.rms_norm(
        pooled.astype(mx.float16), mx.ones((args.head_dim,)), 1e-6), 1)
    block_positions = position_ids[..., ::args.compress_ratio]
    pooled = apply_rope(pooled, block_positions)
    scores = mx.maximum(
        q.astype(mx.float32) @ pooled.astype(mx.float32).transpose(0, 1, 3, 2), 0
    ).sum(axis=1) / math.sqrt(args.head_dim)

    key_len = args.sequence
    block_count = key_len // args.compress_ratio
    top_k = args.budget // args.compress_ratio
    query_ends = mx.arange(1, args.sequence + 1, dtype=mx.int32)
    visible = mx.arange(block_count, dtype=mx.int32)[None, None, :] < (
        query_ends[None, :, None] // args.compress_ratio)
    masked_scores = mx.where(visible, scores, -mx.inf)
    selected_blocks = mx.argpartition(masked_scores, kth=block_count - top_k, axis=-1)[
        ..., -top_k:]
    token_indices = (
        selected_blocks[..., None] * args.compress_ratio
        + mx.arange(args.compress_ratio, dtype=mx.int32)[None, None, None]
    ).reshape(args.batch, args.sequence, -1)
    valid = token_indices < key_len
    selected = mx.zeros((args.batch, args.sequence, key_len + 1), dtype=mx.bool_)
    selected = mx.put_along_axis(selected, token_indices, valid, axis=-1)[..., :key_len]
    key_indices = mx.arange(key_len, dtype=mx.int32)[None, None, :]
    tail = key_indices >= ((query_ends // args.compress_ratio) * args.compress_ratio)[None, :, None]
    tail = tail & (key_indices < query_ends[None, :, None])
    causal = key_indices < query_ends[None, :, None]
    use_sparse = (query_ends // args.compress_ratio) > top_k
    mask = mx.where(use_sparse[None, :, None], selected | tail, causal)[:, None]

    arrays = {
        "projected": projected,
        "position_ids": position_ids,
        "queries": q,
        "raw_keys": raw_keys,
        "pooled_keys": pooled,
        "scores": scores,
        "mask": mask.astype(mx.uint8),
    }
    mx.eval(*arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), arrays, metadata={
        "kind": "qwen4_exp_qsa_after_projection",
        "seed": str(args.seed),
        "budget": str(args.budget),
        "compress_ratio": str(args.compress_ratio),
    })
    print(f"QSA fixture écrite : {args.output}")


if __name__ == "__main__":
    main()
