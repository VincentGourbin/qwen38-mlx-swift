#!/usr/bin/env python3
"""Stabilité du routage MoE Flash-Next au bruit de noyau (étude 2026-08-31).

Deux mesures sur le fixture naturel couche 4 et les poids réels du checkpoint :

1. Distribution des marges du routeur à la frontière top-10/top-11 : si la
   marge est comparable au bruit numérique inter-noyaux, l'appartenance des
   experts n'est pas un invariant inter-runtime.
2. Même runtime Python, même version MLX : le routeur évalué sur le backend
   CPU puis GPU. Tout écart d'appartenance ici prouve que la divergence
   Python/Swift est une variation inter-noyaux banale, pas un bug de portage.

Usage :
  python3 Scripts/qwen4-exp-moe-routing-stability.py \
    --model-dir /Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --fixture parity/qwen4-exp-e5-natural-public-layer-4-reference.safetensors \
    --layer 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    fixture = mx.load(str(args.fixture))
    probabilities = fixture["moe_probabilities"].astype(mx.float32)[0]
    moe_input = fixture["moe_input"][0]
    k = args.top_k

    # 1. Marges à la frontière top-k.
    ordered = mx.sort(probabilities, axis=-1)
    gap = np.array((ordered[:, -k] - ordered[:, -(k + 1)]).tolist())
    rel_gap = np.array((ordered[:, -k] / mx.maximum(ordered[:, -(k + 1)], 1e-12) - 1).tolist())
    positions = gap.shape[0]
    print(
        "frontière top-%d/top-%d : gap min %.2e  médiane %.2e  max %.2e"
        % (k, k + 1, gap.min(), np.median(gap), gap.max())
    )
    print(
        "positions gap<1e-3 : %d/%d ; gap<1e-4 : %d/%d ; gap==0 : %d/%d"
        % (
            (gap < 1e-3).sum(), positions,
            (gap < 1e-4).sum(), positions,
            (gap == 0).sum(), positions,
        )
    )

    # 2. Routeur CPU vs GPU dans le même runtime.
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    gate_key = f"language_model.model.layers.{args.layer}.mlp.gate.weight"
    weight = mx.load(str(args.model_dir / index[gate_key]))[gate_key]
    print(f"routeur {gate_key} : {weight.shape} {weight.dtype} (non quantifié)")

    def route(device):
        with mx.stream(device):
            logits = moe_input @ weight.T
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            indices = mx.argpartition(probs, kth=-k, axis=-1)[:, -k:]
            mx.eval(indices, probs)
        return indices, probs

    gpu_indices, gpu_probs = route(mx.gpu)
    cpu_indices, cpu_probs = route(mx.cpu)
    gpu_sets = [set(row) for row in np.array(gpu_indices.tolist())]
    cpu_sets = [set(row) for row in np.array(cpu_indices.tolist())]
    swaps = [len(a ^ b) // 2 for a, b in zip(gpu_sets, cpu_sets)]
    print("experts différents par position (CPU vs GPU, même MLX) :", swaps)
    print(
        "positions avec appartenance différente : %d/%d"
        % (sum(1 for s in swaps if s), positions)
    )
    print(
        "max|Δprobas| CPU vs GPU : %.2e ; GPU vs fixture : %.2e"
        % (
            mx.abs(gpu_probs - cpu_probs).max().item(),
            mx.abs(gpu_probs - probabilities).max().item(),
        )
    )


if __name__ == "__main__":
    main()
