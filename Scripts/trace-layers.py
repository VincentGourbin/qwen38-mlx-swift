#!/usr/bin/env python3
"""Analyse d'une trace Chrome `flash-*-probe --trace` (swift-mlx-profiler).

Usage : python3 Scripts/trace-layers.py <trace.json> [--layers 48]

Pour chaque passe de N couches (préfill = passe 0, puis un token par passe) :
somme, médiane, min, max des durées `Flash couche N`, plus la moyenne des
compteurs `Utilization` (CPU %, GPU %) et `Memory` pendant la passe.
Le régime établi commence à la passe 2 (la passe 1 est le warm-up du premier
token décodé). Sert de verdict chiffré pour P0/P1 (PLAN.md, RÉPONSE du
2026-09-07).
"""
import json
import statistics
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    layers_per_pass = 48
    if "--layers" in sys.argv:
        layers_per_pass = int(sys.argv[sys.argv.index("--layers") + 1])
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data

    begins = sorted(
        (e for e in events if e.get("ph") == "B" and str(e.get("name", "")).startswith("Flash couche")),
        key=lambda e: e["ts"],
    )
    ends: dict[str, list[float]] = {}
    for e in events:
        if e.get("ph") == "E" and str(e.get("name", "")).startswith("Flash couche"):
            ends.setdefault(e["name"], []).append(e["ts"])
    for name in ends:
        ends[name].sort()
    seen: dict[str, int] = {}
    spans = []
    for b in begins:
        i = seen.get(b["name"], 0)
        seen[b["name"]] = i + 1
        lst = ends.get(b["name"], [])
        if i < len(lst):
            spans.append((b["name"], b["ts"], lst[i]))
    if not spans:
        print("aucune phase `Flash couche N` dans la trace")
        return 1

    counters = [e for e in events if e.get("ph") == "C" and "args" in e and "ts" in e]
    util = [e for e in counters if e.get("name") == "Utilization"]
    mem = [e for e in counters if e.get("name") == "Memory"]

    def mean_in(samples, key, t0, t1):
        vals = [e["args"][key] for e in samples if t0 <= e["ts"] < t1 and key in e["args"]]
        return statistics.mean(vals) if vals else float("nan")

    t_origin = spans[0][1]
    passes = len(spans) // layers_per_pass
    print(f"{path}: {len(spans)} phases couche, {passes} passes de {layers_per_pass}")
    header = "passe  début(s)  somme(s)  méd(ms)  min(ms)  max(ms)  couche_max  CPU%  GPU%  MLX(MB)"
    print(header)
    steady = []
    for p in range(passes):
        chunk = spans[p * layers_per_pass:(p + 1) * layers_per_pass]
        durs = [(b - a) / 1e3 for _, a, b in chunk]
        t0, t1 = chunk[0][1], chunk[-1][2]
        worst = chunk[durs.index(max(durs))][0].split()[-1]
        cpu = mean_in(util, "CPU (%, all threads)", t0, t1) if util else float("nan")
        gpu = mean_in(util, "GPU (%)", t0, t1) if util else float("nan")
        mlx = mean_in(mem, "MLX Active (MB)", t0, t1) if mem else float("nan")
        print(
            f"{p:5d}  {(t0 - t_origin) / 1e6:8.1f}  {sum(durs) / 1e3:8.2f}  "
            f"{statistics.median(durs):7.0f}  {min(durs):7.0f}  {max(durs):7.0f}  "
            f"{worst:>10s}  {cpu:4.0f}  {gpu:4.0f}  {mlx:7.0f}"
        )
        if p >= 2:
            steady.append((sum(durs) / 1e3, statistics.median(durs), cpu, gpu))
    if steady:
        print(
            "régime établi (passes ≥ 2) : "
            f"{statistics.mean(s[0] for s in steady):.2f} s/token, "
            f"médiane {statistics.mean(s[1] for s in steady):.0f} ms/couche, "
            f"CPU {statistics.mean(s[2] for s in steady):.0f} %, "
            f"GPU {statistics.mean(s[3] for s in steady):.0f} %"
        )
        per_layer: dict[str, list[float]] = {}
        for name, a, b in spans[2 * layers_per_pass:]:
            per_layer.setdefault(name, []).append((b - a) / 1e3)
        slow = sorted(per_layer.items(), key=lambda kv: -statistics.median(kv[1]))[:5]
        print("couches les plus lentes (médiane ms) :", [(n.split()[-1], round(statistics.median(v))) for n, v in slow])
    for e in events:
        if e.get("name") == "Session Info" and "args" in e:
            a = e["args"]
            keys = ["decoded", "max_new_tokens", "layer_load_seconds", "layer_forward_seconds", "ngram_cache_hit_rate", "ngram_cache_misses", "mtp"]
            print("session :", {k: a[k] for k in keys if k in a})
    return 0


if __name__ == "__main__":
    sys.exit(main())
