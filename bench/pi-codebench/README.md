# pi-codebench : création de code par pi.dev, mesurée

Un banc court et reproductible pour répondre à deux questions :

1. **Capacité** : un modèle servi par `qwen38 serve` et piloté par
   [pi](https://pi.dev) mène-t-il à bien une petite fonctionnalité Swift, de
   bout en bout, sans aide ?
2. **Découpage** : pour un modèle donné, vaut-il mieux une fiche qui contient
   tout, deux fiches moyennes ou quatre fiches courtes, chacune dans une session
   pi neuve ? Ces mesures alimentent la compétence de découpage de plans.

## Ce qui est testé

La fixture `fixture/` est un paquet Swift autonome, `AgentKit` (Foundation
seul, build et tests en quelques secondes). C'est une copie du module
`Qwen38Agent` de ce dépôt, avec ses 30 tests. La fonctionnalité demandée,
« consommation de jetons », est découpée en quatre tâches (`tasks/`) :

| Tâche | Contenu | Fichiers touchés |
|---|---|---|
| T1 | `AgentTokenUsage`, lecture du bloc `usage` de la réponse | AgentWireFormat |
| T2 | cumul des jetons dans `AgentStats` à chaque tour | AgentLoopEngine |
| T3 | `cacheHitRate` et résumé au format exact (virgule décimale, `·`) | AgentLoopEngine |
| T4 | nouvel outil `count_lines` (catalogue, schéma, exécuteur, garde de chemin) ; un test existant à mettre à jour | AgentToolCatalog, AgentToolExecutor, tests |

Chaque tâche a un **test d'acceptation caché** (`hidden/`), que l'agent ne voit
jamais. Le banc le copie dans le paquet après la fiche, l'exécute seul, puis le
retire. `reference/solution.patch` est une solution de référence ;
`./run.py --dry-check` vérifie que la notation donne 0/4 sur la fixture brute et
4/4 sur la référence.

## Découpages

| Plan | Fiches | Sessions pi |
|---|---|---|
| `mono` | F1 = T1+T2+T3+T4 | 1 |
| `split2` | F1 = T1+T2, F2 = T3+T4 | 2, neuves |
| `split4` | une fiche par tâche | 4, neuves |
| `split4-continue` | une fiche par tâche | 1, poursuivie (`--continue`) |

## Ce qui est mesuré

Pour chaque fiche, le banc mesure :

- **l'état réel du paquet** : build, suite de tests de l'agent, test caché de
  chaque tâche déjà demandée ;
- **l'effort** : durée, tours, appels d'outils, erreurs d'outils, nombre de
  `swift test` lancés par l'agent ;
- **la consommation de jetons** : entrée, dont cache, sortie. Ces chiffres
  viennent du bloc `usage` du serveur, repris par pi dans sa session.
- **le contexte** : contexte maximal atteint et nombre de compactions ;
- **la fin de fiche** : arrêt (`stop`, `length`, erreur) ou délai dépassé.

Un run réussit quand les quatre tests cachés, le build et la suite de l'agent
passent à la fin.

## Lancer

```bash
# 1. le serveur, sur le modèle à évaluer (pas de clé : boucle locale)
caffeinate -i .xcodebuild/Build/Products/Release/qwen38 serve \
  --model-path /Volumes/Lexar/models/mlx-community/Qwen3.8-27B-4bit --enable-thinking

# 2. les runs
cd bench/pi-codebench
./run.py --plan split4 --label v1
./run.py --plan mono --label v1
./run.py --provider ollama --model glm-5.3-flash:cloud --plan mono --label v1   # repère cloud

# 3. le bilan
./report.py --label v1
```

pi tourne avec un dossier de configuration **isolé**, `runs/<run>/pi-agent`,
via `PI_CODING_AGENT_DIR` ; `~/.pi/agent` n'est jamais modifié. Ce dossier
reprend le réglage retenu à l'issue #1 :

- contexte 65 536, `maxTokens` 12 288 ;
- `thinkingFormat qwen` et `thinkingLevelMap` ;
- compaction **imbriquée** (`reserveTokens` 16 384, `keepRecentTokens` 12 000) ;
- `--thinking low` ;
- sans extensions, compétences ni gabarits.

Chaque run garde, sous `runs/<run>/` (ignoré par git) :

- l'espace de travail (`ws/`, un dépôt git, pour `git diff`) ;
- les sessions pi ;
- le flux d'événements JSON de chaque fiche ;
- `result.json`.

Toutes les lignes s'ajoutent à `results.jsonl`.

## Résultats

### v1 — 2026-09-28, un run par combinaison

Conditions :
- M3 Max 96 Go, pi 0.87.1, `--thinking low` ;
- modèle local : `qwen38 serve` sur `mlx-community/Qwen3.8-27B-4bit` ;
- modèles cloud : Ollama Cloud via le démon local.

| Modèle | Découpage | Tâches | Durée | Tours | Contexte max | Entrée en cache | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-4bit | mono | 4/4 | 31,9 min | 15 | 25 k | 90 % | 0 |
| Qwen3.8-27B-4bit | split2 | 2/4 ¹ | 18,1 min | 13 | 17 k | 75 % | 0 |
| Qwen3.8-27B-4bit | split4 | 4/4 | 52,2 min ² | 26 | 15 k | 77 % | 0 |
| Qwen3.8-27B-4bit | split4-continue | 4/4 | 29,2 min | 34 | 34 k | 96 % | 0 |
| glm-5.3-flash | mono | 4/4 | 2,4 min | 14 | 29 k | 60 % | 0 |
| glm-5.3-flash | split4 | 4/4 | 2,9 min | 35 | 18 k | 37 % | 0 |
| gemma4 | mono | 4/4 | 2,3 min | 37 | 41 k | 96 % | 0 |
| gemma4 | split4 | 4/4 | 2,5 min | 46 | 19 k | 84 % | 0 |
| gpt-oss:120b | mono | 0/4 ³ | 1,4 min | 64 | 41 k | 98 % | 0 |
| gpt-oss:120b | split4 | 4/4 | 2,9 min | 100 | 43 k | 96 % | 0 |
| gpt-oss:20b | mono | 0/4, ne compile plus | 6,3 min | 83 | 46 k | 93 % | 1 |
| gpt-oss:20b | split4 | 3/4 (T4 ratée) | 12,8 min | 134 | 44 k | 91 % | 0 |

1. Appels `edit` écrits dans une réflexion jamais fermée : le serveur les
   rangeait en `reasoning_content`, et pi s'arrêtait sans rien modifier.
   Corrigé dans le serveur (voir CHANGELOG).
2. Dont 25 min sur une seule réponse : une boucle de réflexion à vérifier un
   arrondi (« 66,7 ») pourtant juste, coupée à `maxTokens`.
3. Flux Ollama interrompu (« Stream ended without finish_reason »), mais aussi
   15 appels à un outil `search` inexistant et des `edit` ambigus.

Lecture :
- Glm-5.3-flash, gemma4 et le 27B local réussissent la fonctionnalité en une
  seule fiche.
- La famille gpt-oss n'y arrive qu'avec une tâche par fiche.
- En local, poursuivre la même session bat les sessions neuves : le cache est
  conservé et le code n'est pas relu.
- Le contexte n'a jamais été limitant.

Les règles de découpage qui en découlent sont dans la compétence
`pi-plan-decoupage` (`~/.claude/skills/`).

### v2 — 2026-09-28, Qwen3.8-27B-4bit, serveur corrigé, deux runs par découpage

| Découpage | Run 1 | Run 2 | Bilan avec la v1 |
|---|---|---|---|
| mono | 4/4 · 27,7 min · 15 tours | **0/4** ¹ · 25 min · 5 tours | 2 réussites sur 3 |
| split2 | 4/4 · 36,9 min · 25 tours | 4/4 · 43,4 min · 21 tours | 2/2 depuis le correctif |
| split4-continue | 4/4 · 21,5 min · 27 tours | 4/4 · 25,0 min · 29 tours | **3/3**, le plus rapide |

1. Le modèle a rédigé toute l'implémentation des quatre tâches dans sa
   réflexion, soit 40 000 caractères. Il a atteint `maxTokens` (12 288) avant
   d'émettre un seul appel. Aucun appel ne se trouvait dans la réflexion,
   donc le correctif n'avait rien à récupérer.

Le décodage a ralenti au fil de l'après-midi : 13-14 tok/s le matin,
6,5-9 tok/s en v2. Les durées v2 sont donc pessimistes.

Conclusion pour le 27B local :
- une tâche par fiche, dans une session pi poursuivie ;
- la fiche unique est plus exposée à une réflexion qui déborde.
