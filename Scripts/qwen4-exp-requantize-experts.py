#!/usr/bin/env python3
"""Requantize the Flash-Next `switch_mlp` experts from 4-bit g32 to 3-bit g64.

PLAN.md, "G-4bis levée — décision Vincent du 2026-09-08", tâche Q3.1.

Streams the converted Vontra checkpoint
(`$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) shard by
shard. For every `*.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.weight`
tensor (and its `.scales`/`.biases` companions):

    dequantize(4-bit, group_size=32) -> quantize(3-bit, group_size=64)

Every other tensor (n-gram table, attention, shared expert, gate, norms,
vision, MTP non-expert weights) is copied unchanged: read, `mx.eval`, kept
in its original dtype/shape, and written back into the *same* output shard
index it came from (one output shard per input shard, same filename).

Memory discipline (PLAN.md §0 / §6.3 piège 8): `mx.load` on a safetensors
file is lazy (arrays reference the mmap'd file until evaluated — verified:
loading a 5.3 GB shard costs ~5 ms and no RSS), so this script opens every
selected shard's lazy handle up front (metadata only, effectively free) but
only ever evaluates the handful of tensors needed for the *current* output
shard, plus at most the few companion tensors of the checkpoint's 9
expert tensors whose `.weight`/`.scales`/`.biases` triplet is split across a
shard boundary (verified against the real index). Each evaluated tensor is
written to its output shard and released before the next one is touched.

Usage:
    venv617/bin/python Scripts/qwen4-exp-requantize-experts.py --dry-run \
        --src "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"

    venv617/bin/python Scripts/qwen4-exp-requantize-experts.py \
        --src "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP" \
        --limit-shards 1 --dst /path/to/test-dir

    caffeinate -dimsu venv617/bin/python \
        Scripts/qwen4-exp-requantize-experts.py \
        --src "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"

`$QWEN38_MODELS_DIR` defaults to `~/models`; `--dst` defaults to
`$QWEN38_MODELS_DIR/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

MODELS_DIR = os.environ.get("QWEN38_MODELS_DIR") or os.path.expanduser("~/models")
DEFAULT_DST = os.path.join(MODELS_DIR, "local", "Qwen3.8-Flash-Next-MLX-e3bit-MTP")
INDEX_NAME = "model.safetensors.index.json"
EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.weight$")
SUFFIXES = (".weight", ".scales", ".biases")

# Q3.1 report probe: (layer index, expert index) pairs chosen in PLAN.md.
REPORT_TARGETS = [(0, 0), (24, 100), (47, 511)]


def human_bytes(n: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} Po"


def base_key(key: str) -> str | None:
    for suffix in SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def target_layer_of(base: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.mlp\.switch_mlp\.", base)
    if match is None:
        return None
    return int(match.group(1))


@dataclass
class ExpertAccumulator:
    """Running totals for the conversion report."""

    tensor_families_converted: int = 0
    expert_bytes_in: int = 0
    expert_bytes_out: int = 0
    other_tensor_count: int = 0
    reconstruction: dict[tuple[int, int, str], dict[str, float]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Checkpoint Flash-Next source (4-bit g32)")
    parser.add_argument("--dst", type=Path, default=Path(DEFAULT_DST), help="Répertoire de sortie")
    parser.add_argument("--bits", type=int, default=3, help="Bits des experts en sortie")
    parser.add_argument("--group-size", type=int, default=64, help="group_size des experts en sortie")
    parser.add_argument("--dry-run", action="store_true", help="Lister les tenseurs sans rien écrire")
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="Ne traiter que les N premiers shards (par ordre alphabétique) — pour un test rapide")
    return parser.parse_args()


def load_index(src: Path) -> dict[str, Any]:
    index_path = src / INDEX_NAME
    with open(index_path, "r") as handle:
        return json.load(handle)


def read_source_quantization(src: Path) -> tuple[int, int, str]:
    with open(src / "config.json", "r") as handle:
        config = json.load(handle)
    quant = config.get("quantization") or config.get("quantization_config")
    if quant is None:
        raise SystemExit("config.json source ne déclare aucune quantification globale.")
    return quant["group_size"], quant["bits"], quant.get("mode", "affine")


def patch_config(config: dict[str, Any], old: tuple[int, int, str], new_group_size: int, new_bits: int) -> dict[str, Any]:
    old_group_size, old_bits, old_mode = old
    experts_block = {"group_size": new_group_size, "bits": new_bits, "mode": "affine"}
    quantization_block = {"group_size": old_group_size, "bits": old_bits, "mode": old_mode, "experts": experts_block}
    patched = dict(config)
    if "quantization" in patched:
        patched["quantization"] = quantization_block
    if "quantization_config" in patched:
        patched["quantization_config"] = quantization_block
    return patched


def copy_side_files(src: Path, dst: Path) -> list[str]:
    """Copy every non-safetensors, non-index, non-AppleDouble file as-is."""
    copied = []
    for entry in sorted(src.iterdir()):
        name = entry.name
        if not entry.is_file():
            continue
        if name.startswith("._"):
            continue  # ExFAT AppleDouble side-car, not real content.
        if name == INDEX_NAME:
            continue
        if name.endswith(".safetensors"):
            continue
        if name == "config.json":
            continue  # patched separately
        shutil.copy2(entry, dst / name)
        copied.append(name)
    return copied


def dry_run_report(index: dict[str, Any], limit_shards: int | None) -> None:
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    if limit_shards is not None:
        shard_names = shard_names[:limit_shards]
    allowed = {k for k, v in weight_map.items() if v in shard_names}

    expert_bases = {b for k in weight_map if (b := base_key(k)) and EXPERT_WEIGHT_RE.search(k)}
    expert_keys = {b + s for b in expert_bases for s in SUFFIXES} & set(weight_map)
    selected_expert_keys = expert_keys & allowed

    print(f"shards sélectionnés : {len(shard_names)} / {len(set(weight_map.values()))}")
    print(f"tenseurs totaux (sélection) : {len(allowed)}")
    print(f"familles d'experts (checkpoint complet) : {len(expert_bases)}")
    print(f"clés d'experts dans la sélection : {len(selected_expert_keys)}")
    print("premières clés sélectionnées :")
    for key in sorted(allowed)[:20]:
        marker = " [expert]" if key in selected_expert_keys else ""
        print(f"  {key} -> {weight_map[key]}{marker}")


def main() -> None:
    args = parse_args()
    src: Path = args.src
    dst: Path = args.dst

    index = load_index(src)
    weight_map: dict[str, str] = index["weight_map"]

    if args.dry_run:
        dry_run_report(index, args.limit_shards)
        return

    old_group_size, old_bits, old_mode = read_source_quantization(src)
    new_group_size, new_bits = args.group_size, args.bits

    shard_names = sorted(set(weight_map.values()))
    if args.limit_shards is not None:
        shard_names = shard_names[: args.limit_shards]
    selected_shards = set(shard_names)
    selected_keys = {k for k, v in weight_map.items() if v in selected_shards}

    expert_bases_all = {b for k in weight_map if (b := base_key(k)) and EXPERT_WEIGHT_RE.search(k)}
    # In --limit-shards test mode, an expert base whose weight/scales/biases
    # triplet spans a shard outside the current selection cannot be fully
    # reconstructed; skip requantizing it (its raw 4-bit tensors are copied
    # unchanged instead) rather than fail the smoke test.
    expert_bases = {
        b for b in expert_bases_all
        if all((b + s) in weight_map and weight_map[b + s] in selected_shards for s in SUFFIXES)
    }
    expert_keys = {b + s for b in expert_bases for s in SUFFIXES}

    print(f"Shards à traiter : {len(shard_names)}")
    print(f"Familles d'experts à requantifier : {len(expert_bases)} / {len(expert_bases_all)}")
    if expert_bases != expert_bases_all:
        skipped = expert_bases_all - expert_bases
        print(f"  (--limit-shards : {len(skipped)} famille(s) hors sélection ignorée(s), copiées en 4-bit)")

    dst.mkdir(parents=True, exist_ok=True)

    print("Ouverture paresseuse des shards sélectionnés…")
    t0 = time.time()
    shard_lazy: dict[str, dict[str, mx.array]] = {}
    for shard_name in shard_names:
        shard_lazy[shard_name] = mx.load(str(src / shard_name))
    print(f"  {len(shard_lazy)} shard(s) ouverts en {time.time() - t0:.2f} s (paresseux, aucune donnée lue)")

    def fetch(key: str) -> mx.array:
        shard = weight_map[key]
        return shard_lazy[shard][key]

    accum = ExpertAccumulator()
    # base -> (new_weight, new_scales, new_biases); populated on first touch,
    # drained (and freed) once all three of its output keys have been written.
    computed: dict[str, tuple[mx.array, mx.array, mx.array]] = {}
    delivered_count: dict[str, int] = {}

    def compute_expert(base: str) -> tuple[mx.array, mx.array, mx.array]:
        if base in computed:
            return computed[base]
        old_weight = fetch(base + ".weight")
        old_scales = fetch(base + ".scales")
        old_biases = fetch(base + ".biases")
        mx.eval(old_weight, old_scales, old_biases)
        deq4 = mx.dequantize(
            old_weight, scales=old_scales, biases=old_biases,
            group_size=old_group_size, bits=old_bits, mode=old_mode)
        mx.eval(deq4)
        new_weight, new_scales, new_biases = mx.quantize(
            deq4, group_size=new_group_size, bits=new_bits, mode="affine")
        mx.eval(new_weight, new_scales, new_biases)

        layer = target_layer_of(base)
        target_layers = {layer for layer, _ in REPORT_TARGETS}
        if layer in target_layers:
            deq3 = mx.dequantize(
                new_weight, scales=new_scales, biases=new_biases,
                group_size=new_group_size, bits=new_bits, mode="affine")
            mx.eval(deq3)
            proj_name = base.rsplit(".", 1)[-1]
            for report_layer, expert_index in REPORT_TARGETS:
                if report_layer != layer:
                    continue
                row4 = deq4[expert_index].astype(mx.float32)
                row3 = deq3[expert_index].astype(mx.float32)
                diff = mx.abs(row4 - row3)
                mean_abs = float(mx.mean(diff))
                max_abs = float(mx.max(diff))
                denom = float(mx.sqrt(mx.mean(row4 * row4)))
                rms = float(mx.sqrt(mx.mean(diff * diff)))
                relative_rms = rms / denom if denom > 0 else float("nan")
                accum.reconstruction[(report_layer, expert_index, proj_name)] = {
                    "mean_abs": mean_abs, "max_abs": max_abs, "relative_rms": relative_rms,
                }
            del deq3

        accum.expert_bytes_in += old_weight.nbytes + old_scales.nbytes + old_biases.nbytes
        accum.tensor_families_converted += 1
        del deq4, old_weight, old_scales, old_biases
        computed[base] = (new_weight, new_scales, new_biases)
        delivered_count[base] = 0
        return computed[base]

    suffix_index = {".weight": 0, ".scales": 1, ".biases": 2}
    total_in_bytes = 0
    total_out_bytes = 0

    for shard_name in shard_names:
        shard_start = time.time()
        output_tensors: dict[str, mx.array] = {}
        keys_here = sorted(k for k in selected_keys if weight_map[k] == shard_name)
        for key in keys_here:
            base = base_key(key)
            if base is not None and base in expert_bases and key in expert_keys:
                triple = compute_expert(base)
                array = triple[suffix_index[key[len(base):]]]
                output_tensors[key] = array
                delivered_count[base] += 1
                if delivered_count[base] == 3:
                    accum.expert_bytes_out += sum(a.nbytes for a in triple)
                    del computed[base]
                    del delivered_count[base]
            else:
                array = fetch(key)
                mx.eval(array)
                output_tensors[key] = array
                accum.other_tensor_count += 1

        out_path = dst / shard_name
        mx.save_safetensors(str(out_path), output_tensors, metadata={"format": "mlx"})
        in_size = (src / shard_name).stat().st_size
        out_size = out_path.stat().st_size
        total_in_bytes += in_size
        total_out_bytes += out_size
        del output_tensors
        mx.clear_cache()
        elapsed = time.time() - shard_start
        print(
            f"  {shard_name} : {len(keys_here)} tenseurs, "
            f"{human_bytes(in_size)} -> {human_bytes(out_size)} ({elapsed:.1f} s)")

    assert not computed, f"familles d'experts jamais livrées entièrement : {list(computed)}"

    index_out = dict(index)
    index_out["weight_map"] = {k: v for k, v in weight_map.items() if k in selected_keys}
    index_out["metadata"] = {"total_size": total_out_bytes}
    with open(dst / INDEX_NAME, "w") as handle:
        json.dump(index_out, handle, indent=2)

    with open(src / "config.json", "r") as handle:
        source_config = json.load(handle)
    patched_config = patch_config(source_config, (old_group_size, old_bits, old_mode), new_group_size, new_bits)
    with open(dst / "config.json", "w") as handle:
        json.dump(patched_config, handle, indent=2)

    copied = copy_side_files(src, dst)

    print()
    print("=== Rapport Q3.1 ===")
    print(f"Tenseurs experts requantifiés (familles gate/up/down) : {accum.tensor_families_converted}")
    print(f"Autres tenseurs recopiés tels quels : {accum.other_tensor_count}")
    print(f"Fichiers annexes copiés : {len(copied)} ({', '.join(copied)})")
    print(f"Taille totale entrée (shards sélectionnés) : {human_bytes(total_in_bytes)}")
    print(f"Taille totale sortie (shards sélectionnés) : {human_bytes(total_out_bytes)}")
    print(f"Taille experts avant (poids+scales+biases) : {human_bytes(accum.expert_bytes_in)}")
    print(f"Taille experts après (poids+scales+biases) : {human_bytes(accum.expert_bytes_out)}")
    if accum.expert_bytes_in > 0:
        ratio = accum.expert_bytes_out / accum.expert_bytes_in
        print(f"Ratio experts sortie/entrée : {ratio:.3f}")
    print()
    print(
        "Erreur de reconstruction (ATTENTION : la référence '4-bit' est déjà "
        "une quantification du poids original en bf16/fp — cette mesure est "
        "l'écart 4-bit -> 3-bit, pas l'écart au poids plein précision) :")
    for (layer, expert, proj), stats in sorted(accum.reconstruction.items()):
        print(
            f"  couche {layer:>2} expert {expert:>3} {proj:<10} : "
            f"mean|Δ|={stats['mean_abs']:.6f}  max|Δ|={stats['max_abs']:.6f}  "
            f"RMS relative={stats['relative_rms']:.4%}")
    if len(accum.reconstruction) < len(REPORT_TARGETS) * 3:
        print(
            "  (probe partiel : certaines couches cibles sont hors de la "
            "sélection --limit-shards)")


if __name__ == "__main__":
    main()
