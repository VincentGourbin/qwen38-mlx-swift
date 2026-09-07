#!/usr/bin/env python3
"""Export a teacher-forced Flash-Next reference on exact token IDs.

The decoder layers are loaded one at a time, like the Swift streaming
runtime.  This is intentional: constructing the complete Python module would
also construct the 51B n-gram table and defeat the memory budget of the
reference machine.

The fixture is the counterpart of ``flash-teacher-forced-score``.  It keeps
the complete logits for the short sequence, plus the target log-probability,
target rank, argmax and top-1/top-2 margin for each continuation token.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ids(value: str, option: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"{option} doit contenir des IDs entiers séparés par des virgules") from error
    if not values:
        raise ValueError(f"{option} ne peut pas être vide")
    if any(value < 0 for value in values):
        raise ValueError(f"{option} ne peut pas contenir d'ID négatif")
    return values


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
    if not hasattr(config, "seed"):
        config.seed = 0
    if not hasattr(config, "ple_storage"):
        config.ple_storage = None
    return config


def python_local_key(key: str) -> str:
    """Match checkpoint shard_N names to the Python list-backed module."""
    marker = ".ngram_embedding.shard_"
    if marker in key:
        head, suffix = key.split(marker, 1)
        return f"{head}.ngram_embedding.shards.{suffix}"
    return key


NORM_SUFFIXES = (
    "hc_norm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "q_layernorm.weight",
    "k_layernorm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)


def shift_zero_centered_norms(weights: dict, shift: float) -> dict:
    """Undo the legacy +1 applied by the Vontra qwen4_exp conversion."""
    if shift == 0:
        return weights
    return {
        key: value + mx.array(shift, dtype=value.dtype)
        if key.endswith(NORM_SUFFIXES) else value
        for key, value in weights.items()
    }


def load_weights_for_prefix(model_dir: Path, index: dict, prefix: str, local_key):
    selected = {
        key: shard for key, shard in index.items() if key.startswith(prefix)
    }
    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(model_dir / shard))
        for key, current_shard in selected.items():
            if current_shard == shard:
                weights[local_key(key[len(prefix) :])] = arrays[key]
    return selected, weights


def load_global(model_dir: Path, index: dict, config, language, global_ref, norm_shift: float):
    global_model = global_ref.GlobalWeights(config, language)
    prefixes = (
        "language_model.model.embed_tokens.",
        "language_model.model.hyper_connection_mixer.",
        "language_model.lm_head.",
    )
    selected = {
        key: shard for key, shard in index.items() if key.startswith(prefixes)
    }
    quantized_paths = {
        "embed_tokens",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
        "lm_head",
    }
    nn.quantize(
        global_model,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=lambda path, module: path in quantized_paths,
    )
    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(model_dir / shard))
        for key, current_shard in selected.items():
            if current_shard != shard:
                continue
            if key.startswith("language_model.model."):
                local = key[len("language_model.model.") :]
            else:
                local = key[len("language_model.") :]
            weights[local] = arrays[key]
    weights = shift_zero_centered_norms(weights, norm_shift)
    global_model.load_weights(list(weights.items()), strict=True)
    global_model.eval()
    mx.eval(global_model.parameters())
    return global_model


def score_logits(logits, continuation_ids: list[int], prompt_count: int):
    continuation = mx.array(continuation_ids, dtype=mx.int32)
    rows = logits[0, prompt_count - 1 : -1, :].astype(mx.float32)
    if rows.shape[0] != len(continuation_ids):
        raise RuntimeError(
            f"Nombre de lignes incohérent : {rows.shape[0]} != {len(continuation_ids)}"
        )
    log_probs = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    target_logprob = mx.take_along_axis(
        log_probs, continuation[:, None], axis=-1
    ).squeeze(-1)
    target_logits = mx.take_along_axis(
        rows, continuation[:, None], axis=-1
    ).squeeze(-1)
    argmax = mx.argmax(rows, axis=-1).astype(mx.int32)
    target_rank = mx.sum(rows > target_logits[:, None], axis=-1).astype(mx.int32) + 1

    # argpartition is enough for the two largest values and avoids sorting a
    # 248K-wide vocabulary for every target position.
    top2_indices = mx.argpartition(rows, kth=rows.shape[-1] - 2, axis=-1)[..., -2:]
    top2_values = mx.take_along_axis(rows, top2_indices, axis=-1)
    margin = mx.max(top2_values, axis=-1) - mx.min(top2_values, axis=-1)
    return target_logprob, argmax, target_rank, margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(__file__).parent / "references/vlm_q4_language.py",
        help="Source mlx-vlm qwen4_exp (défaut: référence vendorisée du repo)",
    )
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--continuation-ids", required=True)
    parser.add_argument("--layers", default=None, help="Sous-ensemble diagnostique, ex. 0,1,2")
    parser.add_argument("--confident-margin", type=float, default=0.5)
    parser.add_argument(
        "--norm-shift",
        type=float,
        default=-1.0,
        help="Décalage ajouté aux normes ciblées (Vontra: -1, officiel: 0)",
    )
    args = parser.parse_args()

    prompt_ids = parse_ids(args.prompt_ids, "--prompt-ids")
    continuation_ids = parse_ids(args.continuation_ids, "--continuation-ids")
    all_ids = prompt_ids + continuation_ids
    input_ids = mx.array([all_ids], dtype=mx.int32)

    language_ref = load_module(
        "qwen4_exp_language_reference",
        Path(__file__).with_name("qwen4-exp-language-reference.py"),
    )
    language, text_config_type = language_ref.load_qwen4_language(args.scratch)
    config = make_config(args.model_dir / "config.json", text_config_type)
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    global_ref = load_module(
        "qwen4_exp_global_reference",
        Path(__file__).with_name("qwen4-exp-global-reference.py"),
    )
    global_model = load_global(
        args.model_dir, index, config, language, global_ref, args.norm_shift
    )

    hidden = mx.tile(global_model.embed_tokens(input_ids), (1, 1, config.hc_count))
    mx.eval(hidden)
    layer_indices = (
        [int(value) for value in args.layers.split(",") if value]
        if args.layers is not None
        else list(range(config.num_hidden_layers))
    )
    if not layer_indices:
        raise ValueError("--layers ne peut pas être vide")

    for layer_index in layer_indices:
        prefix = f"language_model.model.layers.{layer_index}."
        selected, weights = load_weights_for_prefix(
            args.model_dir, index, prefix, python_local_key
        )
        weights = shift_zero_centered_norms(weights, args.norm_shift)
        local_keys = {python_local_key(key[len(prefix) :]) for key in selected}
        quantized_paths = {
            key[: -len(".weight")]
            for key in local_keys
            if key.endswith(".weight")
            and key[: -len(".weight")] + ".scales" in local_keys
        }
        layer = language.Qwen4ExpDecoderLayer(config, layer_index)
        nn.quantize(
            layer,
            group_size=32,
            bits=4,
            mode="affine",
            class_predicate=lambda path, module: path in quantized_paths,
        )
        layer.load_weights(list(weights.items()), strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        cache = (
            language.ArraysCache(size=4)
            if config.layer_types[layer_index] == "linear_attention"
            else language.QSAKVCache()
        )
        layer_mask = (
            None if config.layer_types[layer_index] == "linear_attention" else "causal"
        )
        hidden = layer(
            hidden,
            input_ids=input_ids,
            mask=layer_mask,
            cache=cache,
            position_ids=None,
        )
        mx.eval(hidden, *[value for value in cache.state if value is not None])
        print(f"layer {layer_index} terminé · {hidden.shape}")
        del layer, cache, selected, weights, local_keys, quantized_paths
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    reduced = global_model.hyper_connection_mixer(hidden)
    logits = global_model.lm_head(reduced)
    target_logprob, argmax, target_rank, margin = score_logits(
        logits, continuation_ids, len(prompt_ids)
    )
    mx.eval(reduced, logits, target_logprob, argmax, target_rank, margin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        {
            "input_ids": input_ids,
            "continuation_ids": mx.array([continuation_ids], dtype=mx.int32),
            "reduced": reduced,
            "logits": logits,
            "target_logprob": target_logprob,
            "argmax": argmax,
            "target_rank": target_rank,
            "argmax_margin": margin,
        },
        metadata={
            "kind": "qwen4_exp_teacher_forced_reference",
            "prompt_count": str(len(prompt_ids)),
            "continuation_count": str(len(continuation_ids)),
            "layers": ",".join(str(value) for value in layer_indices),
            "quantization": "4-bit affine group 32",
            "confident_margin": str(args.confident_margin),
            "norm_shift": str(args.norm_shift),
        },
    )
    target_values = target_logprob.tolist()
    argmax_values = argmax.tolist()
    target_rank_values = target_rank.tolist()
    margin_values = margin.tolist()
    agreement = sum(
        int(actual == target)
        for actual, target in zip(argmax_values, continuation_ids)
    ) / len(continuation_ids)
    confident = [
        index for index, value in enumerate(margin_values)
        if value >= args.confident_margin
    ]
    confident_agreement = (
        sum(int(argmax_values[index] == continuation_ids[index]) for index in confident)
        / len(confident)
        if confident else 0.0
    )
    print(f"fixture teacher-forced écrite : {args.output}")
    print(f"mean logprob : {sum(target_values) / len(target_values):.6f}")
    print(f"accord argmax : {agreement * 100:.1f}%")
    print(f"accord confiant : {confident_agreement * 100:.1f}% · {len(confident)} tokens")
    print(f"rang cible moyen : {sum(target_rank_values) / len(target_rank_values):.2f}")


if __name__ == "__main__":
    main()
