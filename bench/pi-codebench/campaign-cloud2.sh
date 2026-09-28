#!/bin/zsh
# Deuxième run cloud par combinaison, pour consolider la synthèse (aucun GPU local).
cd "$(dirname "$0")"
for model in glm-5.3-flash:cloud gemma4:cloud gpt-oss:120b-cloud gpt-oss:20b-cloud; do
  for plan in mono split4; do
    ./run.py --provider ollama --model $model --plan $plan --fiche-timeout 1200 --label v1
  done
done
echo CAMPAGNE_CLOUD2_FINIE
