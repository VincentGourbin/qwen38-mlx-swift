#!/bin/zsh
# Campagne v2 : 27B 4 bits local, serveur corrigé (appels d'outil dans une
# réflexion non fermée), deux runs par découpage.
cd "$(dirname "$0")"
for rep in 1 2; do
  for plan in mono split2 split4-continue; do
    ./run.py --plan $plan --label v2
  done
done
echo CAMPAGNE_V2_FINIE
