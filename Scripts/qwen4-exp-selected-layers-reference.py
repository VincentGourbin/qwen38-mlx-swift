#!/usr/bin/env python3
"""Export a short real-checkpoint Flash-Next multi-layer reference."""

from __future__ import annotations

import argparse
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


def python_local_key(key: str) -> str:
    """Match checkpoint shard_N names to the list-backed Python module."""
    marker = ".ngram_embedding.shard_"
    if marker in key:
        head, suffix = key.split(marker, 1)
        return f"{head}.ngram_embedding.shards.{suffix}"
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--sequence", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt naturel rendu par le chat template local du checkpoint",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Demander le mode thinking au chat template",
    )
    args = parser.parse_args()

    language_ref = load_module(
        "qwen4_exp_language_reference",
        Path(__file__).with_name("qwen4-exp-language-reference.py"),
    )
    language, text_config_type = language_ref.load_qwen4_language(args.scratch)
    raw = json.loads((args.model_dir / "config.json").read_text())["text_config"]
    config = language_ref.make_config(args.model_dir / "config.json", text_config_type)
    # The released qwen4_exp config omits the hash seed; mlx-vlm's reference
    # uses the constructor default (zero) for the PLE n-gram table.
    if not hasattr(config, "seed"):
        config.seed = 0
    # Newer mlx-vlm references optionally externalize the huge PLE table.
    # The converted checkpoint used by this project keeps it in safetensors;
    # explicitly select that legacy/in-memory path when the older config
    # object does not declare the optional field.
    if not hasattr(config, "ple_storage"):
        config.ple_storage = None
    layer_indices = [int(value) for value in args.layers.split(",") if value]
    if not layer_indices:
        raise ValueError("--layers ne peut pas être vide")

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]

    # Keep the global weights resident just as the Swift streaming model does.
    global_ref = load_module(
        "qwen4_exp_global_reference",
        Path(__file__).with_name("qwen4-exp-global-reference.py"),
    )
    global_model = global_ref.GlobalWeights(config, language)
    global_prefixes = (
        "language_model.model.embed_tokens.",
        "language_model.model.hyper_connection_mixer.",
        "language_model.lm_head.",
    )
    global_selected = {
        key: shard for key, shard in index.items()
        if key.startswith(global_prefixes)
    }
    global_quantized_paths = {
        "embed_tokens",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
        "lm_head",
    }
    nn.quantize(
        global_model, group_size=32, bits=4, mode="affine",
        class_predicate=lambda path, module: path in global_quantized_paths,
    )
    global_weights = {}
    for shard in sorted(set(global_selected.values())):
        arrays = mx.load(str(args.model_dir / shard))
        for key, current_shard in global_selected.items():
            if current_shard != shard:
                continue
            local = (
                key[len("language_model.model."):]
                if key.startswith("language_model.model.")
                else key[len("language_model."):]
            )
            global_weights[local] = arrays[key]
    global_model.load_weights(list(global_weights.items()), strict=True)
    mx.eval(global_model.parameters())

    if args.prompt is None:
        input_ids = mx.array(
            [[10 + offset for offset in range(args.sequence)]], dtype=mx.int32
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=True
        )
        template_kwargs = {
            "enable_thinking": args.thinking,
            "reasoning_effort": "low",
        }
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError:
            # Keep the probe useful with an older local transformers release;
            # the checkpoint's template remains the source of truth.
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        input_ids = mx.array([rendered], dtype=mx.int32)
        print(f"Prompt naturel rendu : {len(rendered)} tokens")
    embedded = global_model.embed_tokens(input_ids)
    hidden = mx.tile(embedded, (1, 1, config.hc_count))
    result = {"input_ids": input_ids, "embedded": embedded}

    for layer_index in layer_indices:
        prefix = f"language_model.model.layers.{layer_index}."
        selected = {
            key: shard for key, shard in index.items() if key.startswith(prefix)
        }
        local_keys = {
            python_local_key(key[len(prefix):]) for key in selected
        }
        quantized_paths = {
            key[:-len(".weight")]
            for key in local_keys
            if key.endswith(".weight")
            and key[:-len(".weight")] + ".scales" in local_keys
        }
        layer = language.Qwen4ExpDecoderLayer(config, layer_index)
        nn.quantize(
            layer, group_size=32, bits=4, mode="affine",
            class_predicate=lambda path, module: path in quantized_paths,
        )
        weights = {}
        for shard in sorted(set(selected.values())):
            arrays = mx.load(str(args.model_dir / shard))
            for key, current_shard in selected.items():
                if current_shard == shard:
                    weights[python_local_key(key[len(prefix):])] = arrays[key]
        layer.load_weights(list(weights.items()), strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        if config.layer_types[layer_index] == "linear_attention":
            cache = language.ArraysCache(size=4)
        else:
            cache = language.QSAKVCache()
        # The real Qwen4ExpModel supplies the regular causal attention mask
        # to full-attention layers.  Passing None here makes the first QSA
        # layer use unmasked dense SDPA while Swift correctly uses causal
        # SDPA, producing a false parity failure before sparse selection is
        # even active.  GDN ignores this string; keep its recurrent path
        # unchanged and mirror the model's layer-specific mask contract.
        layer_mask = "causal" if config.layer_types[layer_index] != "linear_attention" else None
        hidden = layer(
            hidden, input_ids=input_ids, mask=layer_mask, cache=cache, position_ids=None
        )
        mx.eval(hidden, *[value for value in cache.state if value is not None])
        result[f"layer_{layer_index}"] = hidden

    reduced = global_model.hyper_connection_mixer(hidden)
    logits = global_model.lm_head(reduced)
    mx.eval(reduced, logits)
    result["reduced"] = reduced
    result["logits"] = logits
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output), result,
        metadata={
            "kind": "qwen4_exp_selected_layers_reference",
            "layers": ",".join(str(value) for value in layer_indices),
            "quantization": "4-bit affine group 32",
            "sequence": str(args.sequence),
        },
    )
    print(f"Selected-layer fixture écrite : couches {layer_indices} → {args.output}")


if __name__ == "__main__":
    main()
