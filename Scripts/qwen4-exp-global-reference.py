#!/usr/bin/env python3
"""Export a small real-checkpoint Flash-Next global-weight reference.

The fixture isolates the three resident pieces around the streamed decoder:
the quantized token embedding, the final four-stream hyper-connection mixer,
and the quantized language head. It deliberately does not instantiate the
48 decoder layers or the n-gram table.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_qwen4_language(scratch: Path):
    """Reuse the compatibility loader from the first-layer reference."""
    loader_path = Path(__file__).with_name("qwen4-exp-language-reference.py")
    spec = importlib.util.spec_from_file_location(
        "qwen4_exp_language_reference", loader_path
    )
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


class GlobalWeights(nn.Module):
    def __init__(self, config, language):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hyper_connection_mixer = language.Qwen4ExpGatedResidual(
            config, use_combine=False
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3803)
    parser.add_argument("--sequence", type=int, default=3)
    args = parser.parse_args()

    language, text_config_type = load_qwen4_language(args.scratch)
    config = make_config(args.model_dir / "config.json", text_config_type)
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]

    prefixes = (
        "language_model.model.embed_tokens.",
        "language_model.model.hyper_connection_mixer.",
        "language_model.lm_head.",
    )
    selected = {
        key: shard
        for key, shard in index.items()
        if key.startswith(prefixes)
    }
    quantized_paths = {
        "embed_tokens",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
        "lm_head",
    }

    model = GlobalWeights(config, language)
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
            weights[local_key] = arrays[key]
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

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
    embedded = model.embed_tokens(input_ids)
    reduced = model.hyper_connection_mixer(hidden)
    logits = model.lm_head(reduced)
    mx.eval(embedded, reduced, logits)

    arrays = {
        "hidden": hidden,
        "input_ids": input_ids,
        "embedded": embedded,
        "reduced": reduced,
        "logits": logits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        arrays,
        metadata={
            "kind": "qwen4_exp_global_reference",
            "seed": str(args.seed),
            "sequence": str(args.sequence),
            "quantization": "4-bit affine group 32",
        },
    )
    print(f"Global fixture écrite : {args.output}")


if __name__ == "__main__":
    main()
