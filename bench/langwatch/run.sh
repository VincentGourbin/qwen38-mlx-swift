#!/usr/bin/env bash
# Lance la suite « Agent de code » contre l'agent connecté, en comparaison.
#   bench/langwatch/run.sh                       # local vs gpt-5-mini, 3 répétitions
#   bench/langwatch/run.sh local gpt-5           # cibles au choix (valeurs du paramètre model)
#   REPEAT=1 bench/langwatch/run.sh local        # une seule cible, une passe
# Prérequis : agent.py connecté (« Online » dans `langwatch agent list`), et un
# fournisseur de modèle activé sur LangWatch pour le juge et le simulateur.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${LANGWATCH_API_KEY:?LANGWATCH_API_KEY manquante (bench/langwatch/.env)}"
ENV_NAME="${LANGWATCH_AGENT_ENVIRONMENT:-development}"
SUITE="Agent de code"
REPEAT="${REPEAT:-3}"
# Juge et simulateur : le fournisseur « custom » (Ollama Cloud, https://ollama.com/v1)
# enregistré sur le projet le 2026-09-19. deepseek-v4.1-flash et glm-5.3-flash rendent des
# verdicts cohérents ; gpt-oss:120b non (raisonnement « tout est satisfait », verdict
# « échec », tous les critères classés non satisfaits). Surcharger avec JUDGE_MODEL=….
JUDGE="${JUDGE_MODEL:-custom/deepseek-v4.1-flash}"
targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(local gpt-5-mini)
args=()
for t in "${targets[@]}"; do args+=(--target "connected:qwen38-bench@${ENV_NAME}?model=${t}"); done
[ -n "$JUDGE" ] && args+=(--judge-model "$JUDGE" --simulator-model "$JUDGE")
note="$(git -C ../.. log -1 --pretty=%s | cut -c1-200)"
exec npx -y langwatch test-suite run "$SUITE" "${args[@]}" --repeat "$REPEAT" \
  --name "Agent de code : $(IFS=' vs '; echo "${targets[*]}")" --note "$note" --wait 45
