# Banc LangWatch

Comparer `qwen38 serve` (Bonsai 2, Flash-Next) aux modèles du marché sur les
mêmes scénarios d'agent de code, juge et simulateur d'utilisateur étant des
modèles du marché. Le plan d'exécution complet, prévu pour une session pi,
est dans `docs/langwatch-bench/plan.md`.

- `agent.py` : l'agent connecté (harnais commun, paramètre de run `model`).
- `fixture/` : le paquet Swift sur lequel les scénarios travaillent.
- `scenarios.py` / `scenarios.sh` : la suite « Agent de code » et ses scénarios (API REST).
- `run.sh [cibles…]` : une comparaison, par exemple `run.sh local gpt-5-mini`.
- `.env` (non versionné) : `LANGWATCH_API_KEY`, `LANGWATCH_AGENT_ENVIRONMENT`, clés des fournisseurs.
