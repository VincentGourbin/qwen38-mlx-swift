#!/usr/bin/env python3
"""Aides de `run.sh`, en fichier séparé : un programme Python inséré dans une
chaîne bash à guillemets simples ne peut contenir ni apostrophe ni guillemet
échappé, et la première version de `infra_error` était pour cette raison un
SyntaxError permanent (constaté par pi le 2026-09-24).

    run_helpers.py transport-error < <sortie JSON de `run-plan run … -o json`>
        code de sortie 0 si au moins un run de la sortie est en ERROR pour une
        cause de transport (agent_relay_unreachable, agent_disconnected), 1 sinon.
"""
import json
import os
import subprocess
import sys

TRANSPORT_MARKERS = ("agent_relay_unreachable", "agent_disconnected", "relay", "unreachable")


def last_json_line(text: str):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_error_text(run_id: str) -> str:
    env = dict(os.environ, LANGWATCH_NO_DAEMON="1")
    detail = subprocess.run(
        ["npx", "-y", "langwatch", "simulation-run", "get", run_id, "-o", "json"],
        capture_output=True, text=True, env=env).stdout
    payload = last_json_line(detail)
    return json.dumps(payload.get("results") or {}) if payload else detail


def transport_error(payload) -> bool:
    for result in (payload or {}).get("results", []):
        if result.get("status") != "ERROR":
            continue
        text = run_error_text(result["scenarioRunId"])
        if any(marker in text for marker in TRANSPORT_MARKERS):
            print(f"ERROR de transport sur {result['scenarioRunId']} : relance", file=sys.stderr)
            return True
    return False


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "transport-error":
        sys.exit(0 if transport_error(last_json_line(sys.stdin.read())) else 1)
    sys.exit(f"commande inconnue : {command}")
