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
