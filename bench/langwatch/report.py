#!/usr/bin/env python3
"""Bilan d'une campagne LangWatch par cible : taux de réussite, grille scénario
× cible, durées.

LangWatch ne dit pas, dans son API, quelle cible a produit un run d'un plan à
plusieurs cibles. Le rattachement passe par le journal local de l'agent
(`runs.jsonl`, une ligne par tour : fil, modèle, réponse) : chaque réponse de
l'assistant d'un run est cherchée dans ce journal.

    set -a; . bench/langwatch/.env; set +a
    bench/langwatch/.venv/bin/python bench/langwatch/report.py suite_var_gQLGYCdCkoUJ1oCtm
    … --since 2026-09-27T17:00:00Z     # seulement les runs postérieurs
    … --markdown                        # tableaux prêts à coller
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fetch_runs(plan_id: str, limit: int) -> list[dict]:
    env = {**os.environ, "LANGWATCH_NO_DAEMON": "1"}
    raw = subprocess.run(
        ["npx", "-y", "langwatch", "simulation-run", "list",
         "--scenario-set-id", f"__internal__{plan_id}__suite", "--limit", str(limit), "-o", "json"],
        capture_output=True, text=True, env=env, check=True).stdout
    data = json.loads(raw[raw.index("{"):] if "{" in raw else "[]")
    return data.get("runs", data) if isinstance(data, dict) else data


def load_journal() -> dict[str, str]:
    """Réponse (texte exact) → modèle."""
    by_reply: dict[str, str] = {}
    path = HERE / "runs.jsonl"
    if not path.exists():
        return by_reply
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_reply.setdefault(entry["reply"].strip(), entry["model"])
    return by_reply


def timestamp(run: dict) -> datetime:
    value = run.get("timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_id")
    parser.add_argument("--since")
    parser.add_argument("--limit", type=int, default=100)  # plafond du CLI
    parser.add_argument("--markdown", action="store_true")
    options = parser.parse_args()

    runs = fetch_runs(options.plan_id, options.limit)
    if options.since:
        since = datetime.fromisoformat(options.since.replace("Z", "+00:00"))
        runs = [r for r in runs if timestamp(r) >= since]
    journal = load_journal()

    table: dict[str, list[dict]] = collections.defaultdict(list)
    unmatched = 0
    for run in runs:
        models = {journal.get((m.get("content") or "").strip())
                  for m in run.get("messages", []) if m.get("role") == "assistant"}
        models.discard(None)
        if len(models) != 1:
            unmatched += 1
            model = "non attribué" if not models else "ambigu"
        else:
            model = models.pop()
        table[model].append(run)

    scenarios = sorted({r["name"] for r in runs})
    models = sorted(table, key=lambda m: (m.startswith("non") or m == "ambigu", m))

    def rate(items):
        ok = sum(1 for r in items if r["status"] == "SUCCESS")
        return ok, len(items)

    lines = ["| Cible | Réussite | Erreurs | Durée médiane |", "|---|---:|---:|---:|"]
    for model in models:
        items = table[model]
        ok, n = rate(items)
        errors = sum(1 for r in items if r["status"] == "ERROR")
        durations = sorted((r.get("durationInMs") or 0) / 1000 for r in items)
        median = durations[len(durations) // 2] if durations else 0
        lines.append(f"| {model} | {ok}/{n} = {100 * ok / max(n, 1):.0f} % | {errors} | {median:.0f} s |")
    lines += ["", "| Scénario | " + " | ".join(models) + " |",
              "|---|" + "---:|" * len(models)]
    for scenario in scenarios:
        cells = []
        for model in models:
            ok, n = rate([r for r in table[model] if r["name"] == scenario])
            cells.append(f"{ok}/{n}" if n else "—")
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"{len(runs)} runs, {unmatched} non attribués ou ambigus.")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
