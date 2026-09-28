#!/usr/bin/env python3
"""Bilan du banc pi-codebench : une ligne par (modèle, découpage), à partir
de results.jsonl.

    ./report.py                 # tous les runs
    ./report.py --label v1      # un seul libellé
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


PRICES = {k: v for k, v in json.loads((HERE / "prices.json").read_text()).items()
          if not k.startswith("_")}


def price_of(model: str):
    """Tarif Ollama Cloud du modèle (`gemma4:cloud`, `gpt-oss:20b-cloud`…),
    `None` pour un modèle local."""
    for suffix in (":cloud", "-cloud"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
    return PRICES.get(model)


def cost(run) -> float | None:
    p = price_of(run["model"])
    if p is None:
        return None
    return (run["input_tokens"] * p["input"] + run["cached_tokens"] * p["cached"]
            + run["output_tokens"] * p["output"]) / 1e6


MAC = json.loads((HERE / "mac_costs.json").read_text())


def mac_cost(run) -> dict | None:
    """Coût local d'un run mesuré (`--measure-power`), en euros : électricité
    totale, électricité au-dessus du repos (marginale), amortissement du Mac
    au prorata de la durée."""
    if "energy_wh" not in run:
        return None
    hours = run["wall_s"] / 3600
    idle_wh = run.get("idle", {}).get("mean_w", 0) * hours
    kwh_price = MAC["electricity_eur_per_kwh"]
    per_hour = MAC["mac_price_eur"] / (MAC["amortization_years"] * 365 * MAC["amortization_hours_per_day"])
    return {"energy_wh": run["energy_wh"], "mean_w": run["mean_w"],
            "idle_w": run.get("idle", {}).get("mean_w", 0),
            "electricity": run["energy_wh"] / 1000 * kwh_price,
            "marginal": max(run["energy_wh"] - idle_wh, 0) / 1000 * kwh_price,
            "amortization": hours * per_hour}


def med(xs):
    return statistics.median(xs) if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--file", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()
    runs = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    if args.label is not None:
        runs = [r for r in runs if r.get("label") == args.label]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["plan"])].append(r)

    print("| Modèle | Découpage | Runs | Succès | Tâches (moy.) | Durée méd. | Tours méd. "
          "| Contexte max | Jetons entrée méd. | dont cache | Compactions | Échecs de fiche |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (model, plan), rs in sorted(groups.items()):
        ok = sum(1 for r in rs if r["success"])
        tasks = sum(r["tasks_passed"] for r in rs) / len(rs)
        prompt = med([r["input_tokens"] + r["cached_tokens"] for r in rs])
        cached = med([r["cached_tokens"] for r in rs])
        fails = []
        for r in rs:
            for f in r["fiches"]:
                g = f["grade"]
                missed = [t for t in f["tasks"] if not g.get(t)]
                if f.get("timed_out"):
                    fails.append(f"F{f['fiche']} délai")
                elif missed:
                    fails.append(f"F{f['fiche']} {'+'.join(missed)}")
        print(f"| {model} | {plan} | {len(rs)} | {ok}/{len(rs)} | {tasks:.1f}/4 "
              f"| {med([r['wall_s'] for r in rs]) / 60:.1f} min | {med([r['turns'] for r in rs]):.0f} "
              f"| {max(r['max_context'] for r in rs)} | {prompt:.0f} "
              f"| {100 * cached / prompt if prompt else 0:.0f} % "
              f"| {sum(r['compactions'] for r in rs)} | {', '.join(fails) or '—'} |")

    print("\nCoût Ollama Cloud (tarif standard, heures creuses -50 %) :\n")
    print("| Modèle | Découpage | Runs | Succès | Coût médian par run | Coût total | Coût par run réussi "
          "| Jetons sortie méd. |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (model, plan), rs in sorted(groups.items()):
        costs = [cost(r) for r in rs]
        if None in costs:
            continue
        ok = sum(1 for r in rs if r["success"])
        per_success = f"{sum(costs) / ok * 100:.2f} ¢" if ok else "—"
        print(f"| {model} | {plan} | {len(rs)} | {ok}/{len(rs)} | {med(costs) * 100:.2f} ¢ "
              f"| {sum(costs) * 100:.2f} ¢ | {per_success} | {med([r['output_tokens'] for r in rs]):.0f} |")

    measured = [r for r in runs if "energy_wh" in r]
    if measured:
        print(f"\nCoût local mesuré ({MAC['electricity_eur_per_kwh']} €/kWh ; Mac {MAC['mac_price_eur']} € "
              f"amorti sur {MAC['amortization_years']} ans à {MAC['amortization_hours_per_day']} h/jour) :\n")
        print("| Modèle | Découpage | Runs | Succès | Durée méd. | Puissance moy. | Repos | Énergie méd. "
              "| Électricité | dont au-dessus du repos | Amortissement | Total par run réussi |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        mgroups = defaultdict(list)
        for r in measured:
            mgroups[(r["model"], r["plan"])].append(r)
        for (model, plan), rs in sorted(mgroups.items()):
            cs = [mac_cost(r) for r in rs]
            ok = sum(1 for r in rs if r["success"])
            total = sum(c["electricity"] + c["amortization"] for c in cs)
            print(f"| {model} | {plan} | {len(rs)} | {ok}/{len(rs)} | {med([r['wall_s'] for r in rs]) / 60:.1f} min "
                  f"| {med([c['mean_w'] for c in cs]):.0f} W | {med([c['idle_w'] for c in cs]):.0f} W "
                  f"| {med([c['energy_wh'] for c in cs]):.1f} Wh "
                  f"| {med([c['electricity'] for c in cs]) * 100:.2f} c€ | {med([c['marginal'] for c in cs]) * 100:.2f} c€ "
                  f"| {med([c['amortization'] for c in cs]) * 100:.1f} c€ "
                  f"| {total / ok * 100:.1f} c€ ≈ {total / ok * MAC['usd_per_eur'] * 100:.1f} ¢ |" if ok else "| — |")

    print("\nPar fiche (médianes) :\n")
    print("| Modèle | Découpage | Fiche | Tâches | Durée | Tours | Contexte max | Réussite |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    per = defaultdict(list)
    for r in runs:
        for f in r["fiches"]:
            per[(r["model"], r["plan"], f["fiche"], "+".join(f["tasks"]))].append(f)
    for (model, plan, i, tasks), fs in sorted(per.items()):
        ok = sum(1 for f in fs if all(f["grade"].get(t) for t in f["tasks"]))
        print(f"| {model} | {plan} | F{i} | {tasks} | {med([f['wall_s'] for f in fs]) / 60:.1f} min "
              f"| {med([f.get('turns', 0) for f in fs]):.0f} | {max(f.get('max_context', 0) for f in fs)} "
              f"| {ok}/{len(fs)} |")


if __name__ == "__main__":
    main()
