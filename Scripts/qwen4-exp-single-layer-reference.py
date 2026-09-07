#!/usr/bin/env python3
"""Export a one-layer end-to-end Flash-Next reference.

This joins the real quantized embedding, decoder layer 0, final hyper mixer
and lm-head. It is intentionally not a quality benchmark: its purpose is to
prove that the Swift streaming seams agree with Python before all 48 layers
are exposed as a public generator.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_language(scratch: Path):
    path = Path(__file__).with_name("qwen4-exp-language-reference.py")
    spec = importlib.util.spec_from_file_location("qwen4_language_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger le loader de compatibilité")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_qwen4_language(scratch)


def make_config(config_path: Path, text_config_type):
    raw = json.loads(config_path.read_text())["text_config"]
    fields = {field.name for field in dataclasses.fields(text_config_type)}
    kwargs = {key: value for key, value in raw.items() if key in fields}
    kwargs["model_type"] = "qwen3_5"
    kwargs["intermediate_size"] = raw.get(
        "intermediate_size", raw["moe_intermediate_size"]
    )
    kwargs["rope_parameters"] = raw["rope_parameters"]
    config = text_config_type(**kwargs)
    for key, value in raw.items():
        setattr(config, key, value)
    return config


class SingleLayer(nn.Module):
    def __init__(self, config, language):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer = language.Qwen4ExpDecoderLayer(config, 0)
        self.hyper_connection_mixer = language.Qwen4ExpGatedResidual(
            config, use_combine=False
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3804)
    parser.add_argument("--sequence", type=int, default=3)
    args = parser.parse_args()

    language, text_config_type = load_language(args.scratch)
    from mlx_lm.models.gated_delta import gated_delta_update
    config = make_config(args.model_dir / "config.json", text_config_type)
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefixes = (
        "language_model.model.embed_tokens.",
        "language_model.model.layers.0.",
        "language_model.model.hyper_connection_mixer.",
        "language_model.lm_head.",
    )
    selected = {
        key: shard for key, shard in index.items() if key.startswith(prefixes)
    }
    quantized_paths = {
        "embed_tokens",
        "lm_head",
        "layer.linear_attn.in_proj_qkv",
        "layer.linear_attn.in_proj_z",
        "layer.linear_attn.in_proj_b",
        "layer.linear_attn.in_proj_a",
        "layer.linear_attn.out_proj",
        "layer.mlp.shared_expert.gate_proj",
        "layer.mlp.shared_expert.up_proj",
        "layer.mlp.shared_expert.down_proj",
        "layer.mlp.shared_expert_gate",
        "layer.mlp.switch_mlp.gate_proj",
        "layer.mlp.switch_mlp.up_proj",
        "layer.mlp.switch_mlp.down_proj",
        "layer.mlp.experts.gate_up_proj",
        "layer.mlp.experts.down_proj",
        "layer.attn_hyper_connection.input_mix_weight_down",
        "layer.attn_hyper_connection.input_mix_weight_up",
        "layer.attn_hyper_connection.block_inject_weight",
        "layer.mlp_hyper_connection.input_mix_weight_down",
        "layer.mlp_hyper_connection.input_mix_weight_up",
        "layer.mlp_hyper_connection.block_inject_weight",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
    }
    model = SingleLayer(config, language)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=lambda path, module: path in quantized_paths,
    )

    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(args.model_dir / shard))
        for key, current_shard in selected.items():
            if current_shard != shard:
                continue
            if key.startswith("language_model.model."):
                local_key = key[len("language_model.model.") :]
            else:
                local_key = key[len("language_model.") :]
            if local_key.startswith("layers.0."):
                local_key = "layer." + local_key[len("layers.0.") :]
            weights[local_key] = arrays[key]
    model.load_weights(list(weights.items()), strict=True)
    # The Swift runtime is an inference path and uses the fused Metal GDN
    # kernel. MLX modules default to training mode, which would select the
    # Python recurrence fallback and make cache parity compare unlike paths.
    model.eval()
    mx.eval(model.parameters())

    input_ids = mx.array(
        [[10 + index for index in range(args.sequence)]], dtype=mx.int32
    )
    embedded = model.embed_tokens(input_ids)
    hidden = mx.tile(embedded, (1, 1, config.hc_count))
    cache = language.ArraysCache(size=4)
    layer = model.layer
    normed = layer.attn_hyper_connection.hc_norm(hidden)
    down = layer.attn_hyper_connection.input_mix_weight_down(normed)
    down_silu = nn.silu(down / config.hc_count)
    up = layer.attn_hyper_connection.input_mix_weight_up(down_silu)
    mix = mx.sigmoid(up).reshape(
        *hidden.shape[:-1], config.hc_count, config.hidden_size
    )
    streams = normed.reshape(
        *hidden.shape[:-1], config.hc_count, config.hidden_size
    )
    mixed = mx.mean(mix * streams, axis=-2)
    injection = 2 * mx.sigmoid(
        layer.attn_hyper_connection.block_inject_weight(normed)
        / config.hc_count
    )
    linear = layer.linear_attn
    qkv = linear.in_proj_qkv(mixed)
    z = linear.in_proj_z(mixed).reshape(
        mixed.shape[0], mixed.shape[1], linear.num_v_heads, linear.head_v_dim
    )
    b = linear.in_proj_b(mixed)
    a = linear.in_proj_a(mixed)
    conv_state = mx.zeros(
        (mixed.shape[0], linear.conv_kernel_size - 1, linear.conv_dim),
        dtype=mixed.dtype,
    )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))
    q, k, v = [
        tensor.reshape(mixed.shape[0], mixed.shape[1], heads, dimension)
        for tensor, heads, dimension in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]
    q_normed, k_normed = linear._normalize_qk(q, k)
    state = cache[1]
    g = mx.exp(
        -mx.exp(linear.A_log.astype(mx.float32))
        * nn.softplus(a + linear.dt_bias)
    )
    beta = mx.sigmoid(b).astype(mx.float32)
    gdn_output, state = gated_delta_update(
        q_normed, k_normed, v, a, b, linear.A_log, linear.dt_bias,
        state, None, use_kernel=not linear.training,
    )
    cache[1] = state
    cache.advance(args.sequence)
    norm_output = linear.norm(
        gdn_output.reshape(
            mixed.shape[0], mixed.shape[1], linear.num_v_heads, linear.head_v_dim
        ), z
    )
    attn_branch = linear.out_proj(
        norm_output.reshape(mixed.shape[0], mixed.shape[1], -1)
    )
    state_after_attention = hidden + (
        attn_branch[..., None, :] * injection[..., None]
    ).reshape(*hidden.shape)
    mlp_mixed, mlp_hyper_input, mlp_injection = layer.mlp_hyper_connection(
        state_after_attention
    )
    mlp_branch = layer.mlp(mlp_mixed)
    layer_output = mlp_hyper_input + (
        mlp_branch[..., None, :] * mlp_injection[..., None]
    ).reshape(*hidden.shape)
    reduced = model.hyper_connection_mixer(layer_output)
    logits = model.lm_head(reduced)
    mx.eval(
        embedded, hidden, layer_output, reduced, logits,
        normed, down, down_silu, up, mix, mixed, attn_branch,
        state_after_attention, mlp_mixed, mlp_branch,
        qkv, z, b, a, conv_input, conv_out, q, k, v, q_normed, k_normed,
        g, beta, gdn_output, norm_output,
        *[value for value in cache.state if value is not None],
    )

    # Continue with one token on the same layer/cache. This validates the
    # recurrent cache seam without loading the complete 48-layer Python model.
    prefill_cache = {
        index: value for index, value in enumerate(cache.state) if value is not None
    }
    decode_input_ids = mx.array([[13]], dtype=mx.int32)
    decode_embedded = model.embed_tokens(decode_input_ids)
    decode_hidden = mx.tile(decode_embedded, (1, 1, config.hc_count))
    decode_layer_output = model.layer(
        decode_hidden, input_ids=decode_input_ids, mask=None, cache=cache,
        position_ids=None
    )
    mx.eval(
        decode_embedded, decode_hidden, decode_layer_output,
        *[value for value in cache.state if value is not None],
    )
    for name, value in language._gdn_last_capture.items():
        mx.eval(value)

    arrays = {
        "input_ids": input_ids,
        "embedded": embedded,
        "hidden": hidden,
        "layer_output": layer_output,
        "reduced": reduced,
        "logits": logits,
        "attn_normed": normed,
        "attn_down": down,
        "attn_down_silu": down_silu,
        "attn_up": up,
        "attn_mix": mix,
        "attn_mixed": mixed,
        "attn_branch": attn_branch,
        "qkv": qkv,
        "z": z,
        "b": b,
        "a": a,
        "conv_input": conv_input,
        "conv_out": conv_out,
        "q": q,
        "k": k,
        "v": v,
        "q_normed": q_normed,
        "k_normed": k_normed,
        "g": g,
        "beta": beta,
        "gdn_output": gdn_output,
        "norm_output": norm_output,
        "state_after_attention": state_after_attention,
        "mlp_mixed": mlp_mixed,
        "mlp_branch": mlp_branch,
        "decode_input_ids": decode_input_ids,
        "decode_embedded": decode_embedded,
        "decode_hidden": decode_hidden,
        "decode_layer_output": decode_layer_output,
    }
    for name, value in language._gdn_last_capture.items():
        arrays[f"decode_{name}"] = value
    for index, value in prefill_cache.items():
        arrays[f"prefill_cache_{index}"] = value
    for index, value in enumerate(cache.state):
        if value is not None:
            arrays[f"cache_{index}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        arrays,
        metadata={
            "kind": "qwen4_exp_single_layer_reference",
            "seed": str(args.seed),
            "sequence": str(args.sequence),
            "quantization": "4-bit affine group 32",
        },
    )
    print(f"Single-layer fixture écrite : {args.output}")


if __name__ == "__main__":
    main()
