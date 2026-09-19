#!/usr/bin/env bash
# Crée ou met à jour la suite « Agent de code » et ses scénarios (voir scenarios.py).
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
exec .venv/bin/python scenarios.py
