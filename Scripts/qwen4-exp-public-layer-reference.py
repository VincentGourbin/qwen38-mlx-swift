#!/usr/bin/env python3
"""Export a public Flash-Next decoder-layer reference for one real layer."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import scaled_dot_product_attention


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


class PublicLayer(nn.Module):
    def __init__(self, config, language, layer_index):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer = language.Qwen4ExpDecoderLayer(config, layer_index)


def local_key(key: str, layer_index: int) -> str:
    prefix = "language_model.model."
    if key.startswith(prefix):
        key = key[len(prefix) :]
    layer_prefix = f"layers.{layer_index}."
    if key.startswith(layer_prefix):
        key = "layer." + key[len(layer_prefix) :]
    marker = ".ngram_embedding.shard_"
    if marker in key:
        head, suffix = key.split(marker, 1)
        key = f"{head}.ngram_embedding.shards.{suffix}"
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument(
        "--input-fixture",
        type=Path,
        default=None,
        help="Fixture selected-layers : utilise layer_{i-1} Python comme entrée E2",
    )
    parser.add_argument(
        "--dequantized",
        action="store_true",
        help="Déquantifier les poids de la couche en FP32 pour E3",
    )
    args = parser.parse_args()

    language, text_config_type = load_language(args.scratch)
    config = make_config(args.model_dir / "config.json", text_config_type)
    # The released Flash-Next config omits the deterministic PLE hash seed;
    # mlx-vlm's constructor default is zero.
    if not hasattr(config, "seed"):
        config.seed = 0
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefixes = (
        "language_model.model.embed_tokens.",
        f"language_model.model.layers.{args.layer_index}.",
    )
    selected = {key: shard for key, shard in index.items() if key.startswith(prefixes)}

    local_keys = {local_key(key, args.layer_index) for key in selected}
    quantized_paths = {
        key[: -len(".weight")]
        for key in local_keys
        if key.endswith(".weight")
        and key[: -len(".weight")] + ".scales" in local_keys
    }

    model = PublicLayer(config, language, args.layer_index)
    if not args.dequantized:
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
            if current_shard == shard:
                weights[local_key(key, args.layer_index)] = arrays[key]
    if args.dequantized:
        floating_weights = {}
        raw_quant = json.loads((args.model_dir / "config.json").read_text())["quantization"]
        for key, value in weights.items():
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                scales = weights.get(base + ".scales")
                if scales is not None:
                    value = mx.dequantize(
                        value,
                        scales=scales,
                        biases=weights.get(base + ".biases"),
                        group_size=raw_quant["group_size"],
                        bits=raw_quant["bits"],
                        mode=raw_quant.get("mode", "affine"),
                        dtype=mx.float32,
                    )
            floating_weights[key] = value
        weights = floating_weights
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())

    input_ids = mx.array([[10, 11, 12]], dtype=mx.int32)
    input_fixture = None
    if args.input_fixture is not None:
        input_fixture = mx.load(str(args.input_fixture))
        input_ids = input_fixture["input_ids"]
    embedded = model.embed_tokens(input_ids)
    if input_fixture is not None and args.layer_index > 0:
        hidden = input_fixture[f"layer_{args.layer_index - 1}"]
    else:
        hidden = mx.tile(embedded, (1, 1, config.hc_count))
    if config.layer_types[args.layer_index] == "linear_attention":
        cache = language.ArraysCache(size=4)
    else:
        cache = language.QSAKVCache()
    layer_output = model.layer(
        hidden, input_ids=input_ids, mask=None, cache=cache, position_ids=None
    )
    # Rebuild the exact state entering attention for diagnostics. The public
    # call already advanced its PLE cache, so use a separate cache here.
    stage_hidden = hidden
    if "ple" in model.layer:
        stage_cache = language.ArraysCache(size=4)
        stage_hidden = stage_hidden + model.layer.ple(
            stage_hidden, input_ids, stage_cache, None
        )
    cache_values = []
    if isinstance(cache, language.ArraysCache):
        cache_values = [value for value in cache.state if value is not None]
    mx.eval(embedded, hidden, layer_output, *cache_values)

    result = {
        "layer_index": mx.array(args.layer_index, dtype=mx.int32),
        "input_ids": input_ids,
        "embedded": embedded,
        "hidden": hidden,
        "layer_output": layer_output,
    }
    if config.layer_types[args.layer_index] == "full_attention":
        # Reconstruct the public QSA path without touching the cache a second
        # time.  For this short fixture the indexer is below its sparse
        # boundary, so the upstream implementation deliberately leaves
        # ``mask=None`` for dense SDPA.
        attention_mix, attention_hyper_input, attention_injection = (
            model.layer.attn_hyper_connection(stage_hidden)
        )
        attention_normed = model.layer.attn_hyper_connection.hc_norm(stage_hidden)
        attention = model.layer.self_attn
        q_projected = attention.q_proj(attention_mix)
        q_reshaped = q_projected.reshape(
            stage_hidden.shape[0], stage_hidden.shape[1], attention.num_attention_heads,
            attention.head_dim * 2
        )
        queries, output_gate = mx.split(q_reshaped, 2, axis=-1)
        normalized_queries = attention.q_norm(queries)
        queries = normalized_queries.transpose(0, 2, 1, 3)
        normalized_keys = attention.k_norm(
            attention.k_proj(attention_mix).reshape(
                stage_hidden.shape[0], stage_hidden.shape[1], attention.num_key_value_heads,
                attention.head_dim
            )
        )
        keys = normalized_keys.transpose(0, 2, 1, 3)
        values = attention.v_proj(attention_mix).reshape(
            stage_hidden.shape[0], stage_hidden.shape[1], attention.num_key_value_heads,
            attention.head_dim
        ).transpose(0, 2, 1, 3)
        positions = mx.tile(
            mx.arange(stage_hidden.shape[1], dtype=mx.int32)[None], (3, 1, 1)
        )
        queries, keys = attention.rotary_emb.apply_rotary(
            queries, keys, positions, unsqueeze_dim=1
        )
        attended = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attention.scale, mask=None
        )
        attended_flat = attended.transpose(0, 2, 1, 3).reshape(
            stage_hidden.shape[0], stage_hidden.shape[1], -1
        )
        output_gate = output_gate.reshape(
            stage_hidden.shape[0], stage_hidden.shape[1], -1
        )
        gated_attended = attended_flat * mx.sigmoid(output_gate)
        attention_branch = attention.o_proj(gated_attended)
        state_after_attention = attention_hyper_input + (
            attention_branch[..., None, :] * attention_injection[..., None]
        ).reshape(*stage_hidden.shape)
        mlp_mixed, mlp_hyper_input, mlp_injection = (
            model.layer.mlp_hyper_connection(state_after_attention)
        )
        mlp_normed = model.layer.mlp_hyper_connection.hc_norm(state_after_attention)
        mlp_branch = model.layer.mlp(mlp_mixed)
        full_output = mlp_hyper_input + (
            mlp_branch[..., None, :] * mlp_injection[..., None]
        ).reshape(*hidden.shape)
        result.update({
            "attn_normed": attention_normed,
            "attn_mixed": attention_mix,
            "attn_injection": attention_injection,
            "q_normed": normalized_queries,
            "k_normed": normalized_keys,
            "v": values,
            "q_rope": queries,
            "k_rope": keys,
            "output_gate": output_gate,
            "attended": attended_flat,
            "gated_attended": gated_attended,
            "attn_branch": attention_branch,
            "state_after_attention": state_after_attention,
            "mlp_normed": mlp_normed,
            "mlp_mixed": mlp_mixed,
            "mlp_injection": mlp_injection,
            "mlp_branch": mlp_branch,
            "full_output": full_output,
        })
    for name, value in getattr(language, "_gdn_last_capture", {}).items():
        if name in {"q_normed", "k_normed", "v", "gdn_output", "norm_output"}:
            result[name] = value
    if config.layer_types[args.layer_index] == "linear_attention":
        result["attn_branch"] = model.layer.linear_attn.out_proj(
            language._gdn_last_capture["norm_output"].reshape(
                stage_hidden.shape[0], stage_hidden.shape[1], -1
            )
        )
    # Recompute the MoE boundaries from the same post-attention state. The
    # public layer call above is still the source of layer_output; these
    # tensors only make routing/order discrepancies observable in Swift.
    if config.layer_types[args.layer_index] == "linear_attention":
        attention_mix, attention_hyper_input, attention_injection = (
            model.layer.attn_hyper_connection(stage_hidden)
        )
        attention_normed = model.layer.attn_hyper_connection.hc_norm(stage_hidden)
        attention_branch = result["attn_branch"]
        attention_state = attention_hyper_input + (
            attention_branch[..., None, :] * attention_injection[..., None]
        ).reshape(*stage_hidden.shape)
        mlp_mixed, mlp_hyper_input, mlp_injection = (
            model.layer.mlp_hyper_connection(attention_state)
        )
        mlp_normed = model.layer.mlp_hyper_connection.hc_norm(attention_state)
        moe = model.layer.mlp
        probabilities = mx.softmax(moe.gate(mlp_mixed), axis=-1, precise=True)
        top_k = moe.top_k
        indices = mx.argpartition(probabilities, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        flat_mixed = mlp_mixed.reshape(-1, mlp_mixed.shape[-1])
        flat_indices = indices.reshape(-1, top_k)
        flat_scores = scores.reshape(-1, top_k)
        routed = (
            moe.switch_mlp(flat_mixed, flat_indices) * flat_scores[..., None]
        ).sum(axis=-2).reshape(mlp_mixed.shape)
        shared = mx.sigmoid(moe.shared_expert_gate(mlp_mixed)) * moe.shared_expert(
            mlp_mixed
        )
        for name, value in {
            "attn_normed": attention_normed,
            "attn_mixed": attention_mix,
            "attn_injection": attention_injection,
            "moe_input": mlp_mixed,
            "mlp_normed": mlp_normed,
            "mlp_mixed": mlp_mixed,
            "mlp_injection": mlp_injection,
            "moe_probabilities": probabilities,
            "moe_indices": indices,
            "moe_scores": scores,
            "moe_routed": routed,
            "moe_shared": shared,
            "moe_output": routed + shared,
        }.items():
            result[name] = value
    if isinstance(cache, language.ArraysCache):
        for index, value in enumerate(cache.state):
            if value is not None:
                result[f"cache_{index}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output), result, metadata={"layer_index": str(args.layer_index)}
    )
    print(f"Public-layer fixture écrite : couche {args.layer_index} → {args.output}")


if __name__ == "__main__":
    main()
