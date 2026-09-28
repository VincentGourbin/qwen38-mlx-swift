#!/bin/zsh
# Campagne v1 : 27B 4 bits local sur les quatre découpages, puis repère cloud.
cd "$(dirname "$0")"
for plan in split4 mono split2 split4-continue; do
  ./run.py --plan $plan --label v1
done
./run.py --provider ollama --model glm-5.3-flash:cloud --plan mono --label v1
echo CAMPAGNE_FINIE
