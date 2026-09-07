#!/usr/bin/env python3
"""Analyse d'une trace `flash-layer-bench --trace` (swift-mlx-profiler).

Usage : python3 Scripts/bench-layers.py <trace.json>

Pour chaque type de couche benché (`Bench couche gdn`, `Bench couche qsa`) :
médiane, p10, p90, min, max de la durée par pas, plus la moyenne des
compteurs `Utilization` (CPU %, GPU %) et `Memory` échantillonnés pendant
chaque pas. Sert de verdict chiffré pour P0 (PLAN.md, RÉPONSE "chantier P
avant G-8", 2026-09-07, tableau R1, ligne P0) : adapté de
`Scripts/trace-layers.py`, qui fait la même chose pour les phases
`Flash couche N` d'un run résident réel.
"""
import json
import statistics
import sys


def percentile(sorted_values, p):
    if not sorted_values:
        return float("nan")
    index = round((len(sorted_values) - 1) * p)
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data

    names = sorted({
        e["name"] for e in events
        if e.get("ph") in ("B", "E") and str(e.get("name", "")).startswith("Bench couche")
    })
    if not names:
        print("aucune phase `Bench couche *` dans la trace")
        return 1

    counters = [e for e in events if e.get("ph") == "C" and "args" in e and "ts" in e]
    util = [e for e in counters if e.get("name") == "Utilization"]
    mem = [e for e in counters if e.get("name") == "Memory"]

    def mean_in(samples, key, t0, t1):
        vals = [e["args"][key] for e in samples if t0 <= e["ts"] < t1 and key in e["args"]]
        return statistics.mean(vals) if vals else float("nan")

    print(f"{path}")
    for name in names:
        begins = sorted(
            (e["ts"] for e in events if e.get("ph") == "B" and e.get("name") == name))
        ends = sorted(
            (e["ts"] for e in events if e.get("ph") == "E" and e.get("name") == name))
        spans = list(zip(begins, ends))
        if not spans:
            continue
        durs_ms = [(b - a) / 1e3 for a, b in spans]
        sorted_ms = sorted(durs_ms)
        cpu = [mean_in(util, "CPU (%, all threads)", a, b) for a, b in spans] if util else []
        gpu = [mean_in(util, "GPU (%)", a, b) for a, b in spans] if util else []
        mlx = [mean_in(mem, "MLX Active (MB)", a, b) for a, b in spans] if mem else []

        def safe_mean(values):
            clean = [v for v in values if v == v]  # drop NaN
            return statistics.mean(clean) if clean else float("nan")

        print(f"\n{name} : {len(spans)} pas")
        print(
            f"  ms/pas   médiane {statistics.median(sorted_ms):7.2f}  "
            f"p10 {percentile(sorted_ms, 0.1):7.2f}  "
            f"p90 {percentile(sorted_ms, 0.9):7.2f}  "
            f"min {min(sorted_ms):7.2f}  max {max(sorted_ms):7.2f}"
        )
        print(
            f"  CPU % moyen {safe_mean(cpu):5.1f}   "
            f"GPU % moyen {safe_mean(gpu):5.1f}   "
            f"MLX actif moyen {safe_mean(mlx):7.0f} MB"
        )

    for e in events:
        if e.get("name") == "Session Info" and "args" in e:
            a = e["args"]
            keys = [
                "warmup_steps", "measured_steps", "layer_kind",
                "bench_gdn_median_ms", "bench_gdn_cpu_percent_mean", "bench_gdn_gpu_percent_mean",
                "bench_qsa_median_ms", "bench_qsa_cpu_percent_mean", "bench_qsa_gpu_percent_mean",
            ]
            print("\nsession :", {k: a[k] for k in keys if k in a})
    return 0


if __name__ == "__main__":
    sys.exit(main())
