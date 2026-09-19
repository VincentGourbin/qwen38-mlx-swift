#!/usr/bin/env bash
# Lance la suite « Agent de code » contre l'agent connecté, scénario par
# scénario (le modèle local ne sert qu'une conversation à la fois : lancer la
# suite d'un bloc fait échouer les runs en `agent_busy`).
#   bench/langwatch/run.sh                       # cible local, 3 passes
#   bench/langwatch/run.sh local gpt-5-mini      # cibles au choix (valeurs du paramètre model)
#   REPEAT=1 bench/langwatch/run.sh local        # une seule passe
# Tous les runs d'une même cible rejoignent le plan « Agent de code : <cibles> »
# (onglet Results). Prérequis : agent.py connecté, fournisseur de modèle actif.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
: "${LANGWATCH_API_KEY:?LANGWATCH_API_KEY manquante (bench/langwatch/.env)}"
ENV_NAME="${LANGWATCH_AGENT_ENVIRONMENT:-development}"
REPEAT="${REPEAT:-3}"
# Juge et simulateur : le fournisseur « custom » (Ollama Cloud, https://ollama.com/v1)
# enregistré sur le projet le 2026-09-19. deepseek-v4.1-flash et glm-5.3-flash rendent des
# verdicts cohérents ; gpt-oss:120b non (raisonnement « tout est satisfait », verdict
# « échec », tous les critères classés non satisfaits). Surcharger avec JUDGE_MODEL=….
JUDGE="${JUDGE_MODEL:-custom/deepseek-v4.1-flash}"
export LANGWATCH_NO_DAEMON=1   # le démon du CLI abandonne après 25 s

targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(local)
args=()
for t in "${targets[@]}"; do args+=(--target "connected:qwen38-bench@${ENV_NAME}?model=${t}"); done
name="Agent de code : $(IFS=' vs '; echo "${targets[*]}")"
note="$(git -C ../.. log -1 --pretty=%s | cut -c1-200)"
ids=$(.venv/bin/python scenarios.py --ids)
[ -n "$ids" ] || { echo "aucun scénario : lancer scenarios.sh d'abord"; exit 1; }

failed=0; total=0
for pass in $(seq 1 "$REPEAT"); do
  for id in $ids; do
    total=$((total+1))
    echo "=== passe $pass · scénario $id"
    if ! npx -y langwatch run-plan run --scenario "$id" "${args[@]}" \
        --judge-model "$JUDGE" --simulator-model "$JUDGE" \
        --name "$name" --note "$note" --wait 15 -o json | tail -1; then
      failed=$((failed+1))
    fi
  done
done
echo "=== $total runs, $failed avec au moins un échec (un scénario raté par un modèle est un résultat, pas une erreur du banc)"
echo "Résultats : $(npx -y langwatch open --dry-run 2>/dev/null || echo 'https://app.langwatch.ai')  → Agent Testing → Results → « $name »"
