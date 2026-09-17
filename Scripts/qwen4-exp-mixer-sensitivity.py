#!/usr/bin/env python3
"""Sensibilité du mixer hyper-connection Flash-Next à une perturbation amont.

Étude 2026-08-31 (dérive multi-couches, cf. PLAN.md — RÉPONSE diagnostic).
Charge les poids globaux réels (embed/mixer/lm_head, quantifiés comme la
référence), vérifie que la réduction Python du `layer_3` du fixture est
bit-exacte contre le fixture, imprime les statistiques des frontières internes
du mixer, puis mesure l'effet d'un bruit gaussien calibré sur `reduced`, les
logits et l'argmax. C'est ce probe qui établit que le gate token/logit sur le
fixture dégénéré n'est pas atteignable en bf16 (σ=0,0005 rms suffit à faire
basculer l'argmax).

Usage :
  python3 Scripts/qwen4-exp-mixer-sensitivity.py \
    --model-dir "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP" \
    --scratch <chemin du language.py qwen4_exp de référence> \
    --fixture parity/qwen4-exp-selected-layers-reference.safetensors
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3803)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    scripts = Path(__file__).parent
    language_ref = load_module(
        "qwen4_exp_language_reference", scripts / "qwen4-exp-language-reference.py"
    )
    language, text_config_type = language_ref.load_qwen4_language(args.scratch)
    config = language_ref.make_config(args.model_dir / "config.json", text_config_type)
    if not hasattr(config, "seed"):
        config.seed = 0

    global_ref = load_module(
        "qwen4_exp_global_reference", scripts / "qwen4-exp-global-reference.py"
    )
    global_model = global_ref.GlobalWeights(config, language)
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefixes = (
        "language_model.model.embed_tokens.",
        "language_model.model.hyper_connection_mixer.",
        "language_model.lm_head.",
    )
    selected = {k: s for k, s in index.items() if k.startswith(prefixes)}
    raw_quant = json.loads((args.model_dir / "config.json").read_text())["quantization"]
    quantized_paths = {
        "embed_tokens",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
        "lm_head",
    }
    nn.quantize(
        global_model,
        group_size=raw_quant["group_size"],
        bits=raw_quant["bits"],
        mode=raw_quant.get("mode", "affine"),
        class_predicate=lambda path, module: path in quantized_paths,
    )
    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(args.model_dir / shard))
        for key, cur in selected.items():
            if cur != shard:
                continue
            local = (
                key[len("language_model.model."):]
                if key.startswith("language_model.model.")
                else key[len("language_model."):]
            )
            weights[local] = arrays[key]
    global_model.load_weights(list(weights.items()), strict=True)
    mx.eval(global_model.parameters())

    fx = mx.load(str(args.fixture))
    h = fx["layer_3"]
    mixer = global_model.hyper_connection_mixer

    def forward(hidden):
        reduced = mixer(hidden)
        logits = global_model.lm_head(reduced)
        mx.eval(reduced, logits)
        return reduced, logits

    red0, log0 = forward(h)
    for name, mine, ref in (
        ("reduced", red0, fx["reduced"]),
        ("logits", log0, fx["logits"]),
    ):
        delta = mx.abs(mine.astype(mx.float32) - ref.astype(mx.float32))
        print(f"sanité {name}: max|Δ|={delta.max().item():.6f}")

    normed = mixer.hc_norm(h)
    pre_down = mixer.input_mix_weight_down(normed) / mixer.hc_count
    pre_up = mixer.input_mix_weight_up(nn.silu(pre_down))
    gate = mx.sigmoid(pre_up)
    mx.eval(normed, pre_down, pre_up, gate)
    for name, tensor in (
        ("h(layer_3)", h),
        ("normed", normed),
        ("pre_down", pre_down),
        ("pre_up", pre_up),
        ("gate", gate),
        ("reduced", red0),
    ):
        values = tensor.astype(mx.float32)
        print(
            f"{name:12s} absmax={mx.abs(values).max().item():9.4f} "
            f"rms={mx.sqrt(mx.mean(values * values)).item():9.4f}"
        )

    last = log0.astype(mx.float32)[0, -1]
    top0 = mx.argmax(last).item()
    order = mx.argsort(last)[::-1][:4]
    print(f"\nargmax de référence : {top0}")
    print("top-4 :", [(int(t), round(last[t].item(), 4)) for t in order])

    mx.random.seed(args.seed)
    for sigma in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        flips = 0
        red_err, log_err, log_rms = 0.0, 0.0, 0.0
        for _ in range(args.trials):
            noise = mx.random.normal(h.shape).astype(mx.float32) * sigma
            perturbed = (h.astype(mx.float32) + noise).astype(h.dtype)
            red, log = forward(perturbed)
            red_err = max(
                red_err,
                mx.abs(red.astype(mx.float32) - red0.astype(mx.float32)).max().item(),
            )
            delta_logits = log.astype(mx.float32) - log0.astype(mx.float32)
            log_err = max(log_err, mx.abs(delta_logits).max().item())
            log_rms = max(
                log_rms, mx.sqrt(mx.mean(delta_logits * delta_logits)).item()
            )
            if mx.argmax(log.astype(mx.float32)[0, -1]).item() != top0:
                flips += 1
        print(
            f"sigma={sigma:7.4f}  max|Δreduced|={red_err:8.4f}  "
            f"max|Δlogits|={log_err:7.4f}  rms|Δlogits|={log_rms:7.4f}  "
            f"argmax flips: {flips}/{args.trials}"
        )


if __name__ == "__main__":
    main()
