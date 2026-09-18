#!/usr/bin/env python3
"""Chiffre une session pi (harnais d'agent) servie par `qwen38 serve` au tarif
de modèles du marché.

Lit un ou plusieurs journaux de session pi (`.pi/sessions/*.jsonl`, ou un
dossier), additionne le bloc `usage` que le serveur renvoie sur chaque tour
assistant (entrée, sortie, part en cache — voir `ChatCompletionUsage` dans
Qwen38Server.swift) et affiche ce que la même session aurait coûté chez
plusieurs fournisseurs.

    Scripts/pi-session-cost.py ~/Developpements/YuE2-mlx-swift/.pi/sessions
    Scripts/pi-session-cost.py session.jsonl --prices mes-tarifs.json

Les tours enregistrés AVANT que le serveur ne renvoie `usage` n'ont que des
zéros : ils sont alors estimés en caractères/4 (comme pi le fait lui-même) et
signalés « estimé ». Sur un serveur qui renvoie `usage`, `input` est la part
du prompt NON servie par le cache et `cacheRead` la part servie par le cache :
la somme des deux est la longueur totale du prompt à chaque tour.

Tarifs ($ par million de jetons) : table intégrée ci-dessous, datée ; à
vérifier avant toute comparaison sérieuse, ou à remplacer par `--prices`
(même structure JSON). Les lectures de cache Anthropic sont facturées 10 % du
tarif d'entrée, l'écriture 125 % ; OpenAI facture le cache à 10 % de l'entrée
et n'a pas de surcoût d'écriture.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

# $ / 1M jetons — relevé du 2026-09-17 (Anthropic : tarifs API première partie ;
# OpenAI : grille GPT-5.6 du 2026-08-21). `cache_read` / `cache_write`
# absents ⇒ 10 % / 125 % (Anthropic) ou 10 % / 100 % (OpenAI) de `input`.
DEFAULT_PRICES = {
    "claude-opus-5":    {"input": 5.00,  "output": 25.00, "vendor": "anthropic"},
    "claude-sonnet-5":  {"input": 2.00,  "output": 10.00, "vendor": "anthropic"},
    "claude-haiku-4-5": {"input": 1.00,  "output": 5.00,  "vendor": "anthropic"},
    "gpt-5.6-sol":      {"input": 5.00,  "output": 30.00, "cache_read": 0.50, "vendor": "openai"},
    "gpt-5.6-terra":    {"input": 2.00,  "output": 12.00, "cache_read": 0.20, "vendor": "openai"},
    "gpt-5":            {"input": 1.25,  "output": 10.00, "cache_read": 0.125, "vendor": "openai"},
}


def unit_prices(p):
    inp = p["input"]
    vendor = p.get("vendor", "anthropic")
    read = p.get("cache_read", inp * 0.10)
    write = p.get("cache_write", inp * (1.25 if vendor == "anthropic" else 1.0))
    return inp, p["output"], read, write


def estimate_tokens(message):
    """Même barème que pi (chars/4) pour les tours sans `usage`."""
    role, content = message.get("role"), message.get("content")
    if role == "assistant":
        chars = 0
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                chars += len(block.get("text", ""))
            elif kind == "thinking":
                chars += len(block.get("thinking", ""))
            elif kind == "toolCall":
                chars += len(block.get("name", "")) + len(json.dumps(block.get("arguments", {})))
        return chars // 4
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict)) // 4
    return 0


def analyse(path):
    entries = [json.loads(line) for line in open(path) if line.strip()]
    messages = [e["message"] for e in entries if e.get("type") == "message"]
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    turns = estimated_turns = 0
    context_estimate = 0  # pour les tours sans usage : taille du prompt ≈ tout ce qui précède
    for message in messages:
        context_estimate += estimate_tokens(message)
        if message.get("role") != "assistant":
            continue
        turns += 1
        usage = message.get("usage") or {}
        reported = sum(usage.get(k, 0) or 0 for k in totals)
        if reported > 0:
            for k in totals:
                totals[k] += usage.get(k, 0) or 0
        else:
            estimated_turns += 1
            own = estimate_tokens(message)
            totals["input"] += max(context_estimate - own, 0)
            totals["output"] += own
    stamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    minutes = 0.0
    if len(stamps) > 1:
        fmt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        minutes = (fmt(stamps[-1]) - fmt(stamps[0])).total_seconds() / 60
    compactions = sum(1 for e in entries if e.get("type") == "compaction")
    return {"path": path, "turns": turns, "estimated_turns": estimated_turns,
            "compactions": compactions, "minutes": minutes, **totals}


def cost(row, price):
    inp, out, read, write = unit_prices(price)
    return (row["input"] * inp + row["output"] * out + row["cacheRead"] * read
            + row["cacheWrite"] * write) / 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="fichier(s) .jsonl ou dossier(s) de sessions pi")
    parser.add_argument("--prices", help="JSON {modèle: {input, output, cache_read?, cache_write?, vendor?}}")
    parser.add_argument("--json", action="store_true", help="sortie JSON (une ligne par session)")
    args = parser.parse_args()

    prices = DEFAULT_PRICES
    if args.prices:
        prices = json.load(open(args.prices))

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            files.append(p)
    if not files:
        sys.exit("aucune session trouvée")

    rows = [analyse(f) for f in files]
    if args.json:
        for row in rows:
            row["costs"] = {name: round(cost(row, p), 4) for name, p in prices.items()}
            print(json.dumps(row, ensure_ascii=False))
        return

    names = list(prices)
    header = f"{'session':<22} {'tours':>5} {'min':>5} {'compact':>7} {'entrée':>8} {'cache':>8} {'sortie':>7} "
    header += " ".join(f"{n:>14}" for n in names)
    print(header)
    total = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "turns": 0, "minutes": 0.0, "compactions": 0}
    for row in rows:
        name = os.path.basename(row["path"])[:19]
        flag = "~" if row["estimated_turns"] else " "
        line = f"{name:<22}{flag}{row['turns']:>4} {row['minutes']:>5.0f} {row['compactions']:>7} {row['input']:>8} {row['cacheRead']:>8} {row['output']:>7} "
        line += " ".join(f"{cost(row, prices[n]):>13.3f}$" for n in names)
        print(line)
        for k in total:
            total[k] += row[k]
    line = f"{'TOTAL':<22} {total['turns']:>5} {total['minutes']:>5.0f} {total['compactions']:>7} {total['input']:>8} {total['cacheRead']:>8} {total['output']:>7} "
    line += " ".join(f"{cost(total, prices[n]):>13.3f}$" for n in names)
    print(line)
    if any(r["estimated_turns"] for r in rows):
        print("~ : session avec des tours sans `usage` serveur, estimés en caractères/4 "
              "(entrée = tout le contexte précédent, sans part en cache).")
    print("Tarifs $/M jetons — " + " · ".join(
        f"{n} {p['input']}/{p['output']} (cache {unit_prices(p)[2]:g})" for n, p in prices.items()))


if __name__ == "__main__":
    main()
