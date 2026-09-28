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

### Synthèse — deux runs par modèle cloud, trois par découpage pour le 27B local

Réussite = runs où les quatre tâches passent. Durée = médiane.

| Modèle | Fiche unique | Une tâche par fiche | Découpage conseillé |
|---|---|---|---|
| glm-5.3-flash (cloud) | ✅ 2/2 · 2,5 min · 12 tours | ✅ 2/2 · 3,3 min · 40 tours | fiche unique |
| gemma4 (cloud) | ✅ 2/2 · 2,0 min · 34 tours | ✅ 2/2 · 2,1 min · 47 tours | fiche unique |
| gpt-oss:120b (cloud) | ⚠️ 1/2 ¹ · 1,6 min · 70 tours | ✅ 2/2 · 2,6 min · 90 tours | une tâche par fiche |
| gpt-oss:20b (cloud) | ❌ 0/2 · 7,2 min · 84 tours | ⚠️ 1/2 · 13 min · 156 tours | à éviter |
| Qwen3.8-27B-4bit (local) | ⚠️ 2/3 · 26-32 min | ✅ 3/3 · 21-29 min, même session ; ✅ 1/1 · 52 min, sessions neuves | une tâche par fiche, même session |

1. L'échec vient d'un flux Ollama coupé en cours de réponse.

### Coût des modèles cloud

Tarifs Ollama Cloud du 2026-09-28 (`prices.json`, source
<https://ollama.com/pricing>), appliqués aux jetons réellement consommés par
run : entrée non servie par le cache, entrée servie par le cache, sortie
(réflexion comprise). Tarif standard ; les heures creuses divisent le coût
par deux.

| Modèle | $/M entrée · cache · sortie | Fiche unique : coût par run réussi | Une tâche par fiche : coût par run réussi |
|---|---|---:|---:|
| gemma4 | 0,14 · 0,05 · 0,40 | 4,5 ¢ | **2,5 ¢** |
| glm-5.3-flash | 0,15 · 0,03 · 0,50 | **2,6 ¢** | 4,9 ¢ |
| gpt-oss:120b | 0,15 · 0,014 · 0,60 | 7,6 ¢ (1 run réussi sur 2) | 3,7 ¢ |
| gpt-oss:20b | 0,07 · 0,035 · 0,30 | — (aucun run réussi) | 30 ¢ (1 run réussi sur 2) |

Le coût par run réussi divise tout ce qui a été dépensé, échecs compris, par
le nombre de runs réussis.

Lecture :
- **Le modèle le moins cher au jeton n'est pas le moins cher à la tâche.**
  gpt-oss:20b a le tarif le plus bas, mais il consomme 1,9 à 2,5 millions de
  jetons d'entrée par run, avec un cache peu efficace (≈ 44 %), et il échoue.
  C'est de loin le plus cher par tâche réussie.
- glm-5.3-flash en fiche unique et gemma4 en une tâche par fiche livrent la
  fonctionnalité pour **2,5 à 2,6 ¢**.
- Pour glm, découper double le coût : plus de tours, et un cache moins bien
  servi (32 % contre 62 %).
- **L'ordre de grandeur** : une fonctionnalité d'environ 300 lignes coûte
  quelques centimes. Le forfait Pro d'Ollama (20 $/mois pour 60 $ de crédits)
  couvre plus de 2 000 fonctionnalités de cette taille.

### Coût local (MacBook Pro M3 Max) comparé au cloud

Méthode :
- **Durée par run réussi** : durée totale des runs, échecs compris, divisée
  par le nombre de réussites. Pour le 27B en `split4-continue`, trois runs
  réussis sur trois font 25,2 min par fonctionnalité livrée.
- **Électricité** : Tarif Bleu EDF, 0,2001 €/kWh au 1er août 2026.
  - Au repos, le Mac consomme 22 W, mesurés (`ioreg`, `SystemLoad`), modèle
    chargé.
  - La puissance en pleine inférence n'a pas pu être mesurée proprement : le
    GPU était pris par une autre tâche. On prend donc une fourchette de 60 à
    100 W. Le résultat en dépend peu, puisque l'électricité pèse moins de
    1 centime.
- **Amortissement** : MacBook Pro 16" M3 Max 96 Go à 5 169 € (dernier prix
  LDLC), sur 4 ans, soit 44 c€ de l'heure à 8 h par jour et 15 c€ de l'heure
  à 24 h par jour.
- **Change** : 1 € ≈ 1,10 $. Toutes les hypothèses sont dans `mac_costs.json`.

| Option | Réussite | Durée par fonctionnalité | Coût par fonctionnalité livrée |
|---|---|---:|---:|
| gemma4, Ollama Cloud, une tâche par fiche | 2/2 | 2,1 min | **2,5 ¢** |
| glm-5.3-flash, Ollama Cloud, fiche unique | 2/2 | 2,5 min | **2,6 ¢** |
| 27B local, électricité seule (Mac déjà acheté) | 3/3 | 25 min | **0,6 à 0,9 ¢** |
| 27B local, amorti 24 h/24 sur 4 ans | 3/3 | 25 min | ≈ 7 à 7,5 ¢ |
| 27B local, amorti 8 h/jour sur 4 ans | 3/3 | 25 min | ≈ 21 ¢ |
| 27B local en fiche unique, amorti 8 h/jour | 2/3 | 42 min | ≈ 35 ¢ |

Lecture :
- **Si le Mac est déjà là**, le local est le moins cher : moins d'un centime
  d'électricité par fonctionnalité. Mais il est dix fois plus lent que le
  cloud, qui livre en 2 à 3 min.
- **Si le Mac est acheté pour ça**, l'amortissement domine tout. Selon le
  taux d'usage, le local coûte 3 à 8 fois plus cher que glm-5.3-flash ou
  gemma4 dans le cloud.
- **Le local rattrape le cloud** quand il sert de toute façon : données qui
  ne doivent pas sortir, travail hors ligne, Mac déjà rentabilisé par
  d'autres usages.
- **Le découpage compte aussi en euros.** En local, la fiche unique échoue une
  fois sur trois, ce qui fait passer le coût par fonctionnalité de 21 à 35 c€.
