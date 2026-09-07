#!/usr/bin/env python3
"""Export a real first-layer Flash-Next language reference.

Only layer 0 is instantiated and loaded.  This keeps the parity probe small
while exercising the checkpoint's quantized linears, Gated DeltaNet,
four-stream hyper-connections and the 512-expert MoE branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_qwen4_language(scratch: Path):
    """Load the local qwen4_exp source despite an older installed mlx-vlm."""
    from mlx_lm.models.cache import dynamic_roll
    import mlx_vlm.models.cache as cache
    import mlx_vlm.models.qwen3_5.language as qwen35_language
    from mlx_vlm.models.qwen3_5.config import TextConfig

    # The scratchpad source is newer than the installed qwen3.5 package.  The
    # compatibility shims are not used by the singleton layer probe, but make
    # the module importable without changing the reference implementation.
    cache.dynamic_roll = dynamic_roll
    if not hasattr(qwen35_language, "_restore_batch_padding_metadata"):
        qwen35_language._restore_batch_padding_metadata = lambda cache, *args: cache

    verifier = types.ModuleType("mlx_vlm.models.qwen3_5.speculative_verifier")
    verifier.Qwen3_5ExactSpeculativeVerifier = type(
        "Qwen3_5ExactSpeculativeVerifier", (object,), {}
    )
    sys.modules[verifier.__name__] = verifier

    package = types.ModuleType("mlx_vlm.models.qwen4_exp")
    package.__path__ = []
    sys.modules[package.__name__] = package
    config_module = types.ModuleType("mlx_vlm.models.qwen4_exp.config")
    config_module.ModelConfig = object
    config_module.TextConfig = TextConfig
    sys.modules[config_module.__name__] = config_module

    spec = importlib.util.spec_from_file_location(
        "mlx_vlm.models.qwen4_exp.language", scratch
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {scratch}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _install_gdn_call_compat(module)
    return module, TextConfig


def _install_gdn_call_compat(language):
    """Make the installed old Qwen3.5 base call Qwen4's L2 hook.

    Older mlx-vlm releases inline the Qwen3.5 RMSNorm scaling in
    ``Qwen3_5GatedDeltaNet.__call__`` and never call the Qwen4 override's
    ``_normalize_qk`` method. The local qwen4_exp source expects a newer base
    class, so patch only the loaded reference class instead of changing the
    installed package.
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    def call(self, inputs, mask=None, cache=None, gdn_sink=None):
        batch, sequence, _ = inputs.shape
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, sequence, -1, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            if mask.shape[0] != batch:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)

        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            tensor.reshape(batch, sequence, heads, dimension)
            for tensor, heads, dimension in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        q, k = self._normalize_qk(q, k)
        state = cache[1] if cache is not None else None
        if state is not None and state.shape[0] != batch:
            state = None
        if gdn_sink is not None:
            gdn_sink.append(
                (q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                 conv_input, self.conv_kernel_size)
            )
        output, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(sequence)
        raw_output = output
        norm_output = self.norm(raw_output, z)
        language._gdn_last_capture = {
            "qkv": mixed_qkv,
            "z": z,
            "a": a,
            "b": b,
            "conv_input": conv_input,
            "conv_out": conv_out,
            "q": q,
            "k": k,
            "v": v,
            "q_normed": q,
            "k_normed": k,
            "g": mx.exp(
                -mx.exp(self.A_log.astype(mx.float32))
                * nn.softplus(a + self.dt_bias)
            ),
            "beta": mx.sigmoid(b).astype(mx.float32),
            "gdn_output": raw_output,
            "norm_output": norm_output,
        }
        return self.out_proj(norm_output.reshape(batch, sequence, -1))

    language.Qwen4ExpGatedDeltaNet.__call__ = call


def make_config(config_path: Path, text_config_type):
    raw = json.loads(config_path.read_text())["text_config"]
    fields = {field.name for field in dataclasses.fields(text_config_type)}
    kwargs = {key: value for key, value in raw.items() if key in fields}
    # Qwen3.5's compatibility dataclass requires this unused dense-M​​LP field.
    kwargs["model_type"] = "qwen3_5"
    kwargs["intermediate_size"] = raw.get(
        "intermediate_size", raw["moe_intermediate_size"]
    )
    kwargs["rope_parameters"] = raw["rope_parameters"]
    config = text_config_type(**kwargs)
    # qwen4_exp adds fields consumed by the scratchpad implementation.
    for key, value in raw.items():
        setattr(config, key, value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3802)
    parser.add_argument("--sequence", type=int, default=3)
    args = parser.parse_args()

    language, text_config_type = load_qwen4_language(args.scratch)
    config = make_config(args.model_dir / "config.json", text_config_type)

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefix = "language_model.model.layers.0."
    selected = {key: shard for key, shard in index.items() if key.startswith(prefix)}
    quantized_paths = {
        key[len(prefix) : -len(".scales")]
        for key in selected
        if key.endswith(".scales")
    }

    layer = language.Qwen4ExpDecoderLayer(config, 0)
    nn.quantize(
        layer,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=lambda path, module: path in quantized_paths,
    )

    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(args.model_dir / shard))
        for key, current_shard in selected.items():
            if current_shard == shard:
                weights[key[len(prefix) :]] = arrays[key]
    layer.load_weights(list(weights.items()), strict=True)
    mx.eval(layer.parameters())

    import numpy as np

    rng = np.random.default_rng(args.seed)
    hidden = mx.array(
        rng.standard_normal(
            (1, args.sequence, config.hidden_size * config.hc_count),
            dtype=np.float32,
        ).astype(np.float16)
    )
    input_ids = mx.array(
        [[10 + index for index in range(args.sequence)]], dtype=mx.int32
    )
    cache = language.ArraysCache(size=4)
    # Keep the branch boundaries in the fixture.  A large final error is not
    # actionable without knowing whether it starts in a packed matmul, GDN,
    # hyper-connection injection or the MoE reduction.
    attn_normed = layer.attn_hyper_connection.hc_norm(hidden)
    attn_down = layer.attn_hyper_connection.input_mix_weight_down(attn_normed)
    attn_down_silu = nn.silu(attn_down / config.hc_count)
    attn_up = layer.attn_hyper_connection.input_mix_weight_up(attn_down_silu)
    attn_mix = mx.sigmoid(attn_up).reshape(
        *hidden.shape[:-1], config.hc_count, config.hidden_size
    )
    attn_streams = attn_normed.reshape(
        *hidden.shape[:-1], config.hc_count, config.hidden_size
    )
    attn_mixed = mx.mean(attn_mix * attn_streams, axis=-2)
    attn_injection = 2 * mx.sigmoid(
        layer.attn_hyper_connection.block_inject_weight(attn_normed)
        / config.hc_count
    )
    attn_hyper_input = hidden
    attn_weights = attn_injection
    attn_branch = layer.linear_attn(attn_mixed, mask=None, cache=cache)
    state_after_attention = attn_hyper_input + (
        attn_branch[..., None, :] * attn_weights[..., None]
    ).reshape(attn_hyper_input.shape)
    mlp_mixed, mlp_hyper_input, mlp_weights = layer.mlp_hyper_connection(
        state_after_attention
    )
    mlp_branch = layer.mlp(mlp_mixed)
    output = mlp_hyper_input + (
        mlp_branch[..., None, :] * mlp_weights[..., None]
    ).reshape(mlp_hyper_input.shape)
    mx.eval(
        output,
        attn_normed,
        attn_down,
        attn_down_silu,
        attn_up,
        attn_mix,
        attn_mixed,
        attn_branch,
        state_after_attention,
        mlp_mixed,
        mlp_branch,
        *[value for value in cache.state if value is not None],
    )

    arrays = {
        "hidden": hidden,
        "input_ids": input_ids,
        "attn_normed": attn_normed,
        "attn_down": attn_down,
        "attn_down_silu": attn_down_silu,
        "attn_up": attn_up,
        "attn_mix": attn_mix,
        "attn_mixed": attn_mixed,
        "attn_branch": attn_branch,
        "state_after_attention": state_after_attention,
        "mlp_mixed": mlp_mixed,
        "mlp_branch": mlp_branch,
        "output": output,
    }
    for index, value in enumerate(cache.state):
        if value is not None:
            arrays[f"cache_{index}"] = value
    mx.eval(*arrays.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        arrays,
        metadata={
            "kind": "qwen4_exp_first_language_layer",
            "layer": "0",
            "seed": str(args.seed),
            "sequence": str(args.sequence),
            "quantization": "4-bit affine group 32",
        },
    )
    print(f"Language fixture écrite : {args.output}")


if __name__ == "__main__":
    main()
