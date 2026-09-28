#!/bin/zsh
# Campagne v1, volet cloud : des modèles Ollama Cloud plus modestes, sur les
# mêmes découpages que le 27B local (mono et split4). Attend la fin de
# campaign.sh pour ne pas mêler les builds Swift aux mesures locales.
cd "$(dirname "$0")"
until grep -q CAMPAGNE_FINIE runs/campaign-v1.log 2>/dev/null; do sleep 30; done

# Un modèle n'entre dans la campagne que s'il répond sur le démon local.
available() {
  curl -s -m 60 http://127.0.0.1:11434/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":8}" \
    | grep -q '"choices"'
}

for model in gpt-oss:20b-cloud gemma4:cloud gpt-oss:120b-cloud; do
  if ! available $model; then echo "== $model indisponible, ignoré"; continue; fi
  for plan in mono split4; do
    ./run.py --provider ollama --model $model --plan $plan --fiche-timeout 1200 --label v1
  done
done
# glm-5.3-flash a déjà son mono dans campaign.sh : on complète avec split4.
./run.py --provider ollama --model glm-5.3-flash:cloud --plan split4 --fiche-timeout 1200 --label v1
echo CAMPAGNE_CLOUD_FINIE
