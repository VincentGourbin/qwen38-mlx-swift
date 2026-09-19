# Plan d'exécution — banc LangWatch : notre serveur et Bonsai 2 face aux modèles du marché

> Plan autonome pour une session pi (« exécute `docs/langwatch-bench/plan.md` »).
> Rédigé le 2026-09-19. Tout ce qui est décrit ici existe déjà dans le dépôt ;
> la session exécute, vérifie, mesure et consigne. Elle ne conçoit rien.

## 0. Mode d'exécution

- **Une fiche à la fois, dans l'ordre.** Une fiche est terminée quand sa
  *porte de sortie* a été observée dans la sortie d'une commande. Recopie la
  ligne observée dans le journal (§7).
- **Aucune décision d'architecture.** En cas d'ambiguïté : STOP, écris un ASK
  (§7), termine par `FICHE L-x BLOQUÉE`.
- **Secrets** : les clés vivent dans `bench/langwatch/.env` (ignoré par git).
  Ne les affiche jamais, ne les copie nulle part, ne les passe pas en argument
  de commande. `set -a; . bench/langwatch/.env; set +a` les charge.
- **Interdits** : modifier `Sources/`, `Vendor/`, `Package.swift` ; toucher au
  serveur autrement que par `qwen38 serve` ; supprimer des runs ou des
  scénarios sur LangWatch ; `git push`.
- **Outils** : le CLI `langwatch` (via `npx -y langwatch`, version ≥ 1.15) et
  le SDK Python (`bench/langwatch/.venv`). Pas de MCP dans pi.
- **Commit** : un commit par fiche validée, première ligne `L-x : <titre>`.

## 1. Ce que le banc mesure, et comment il reste impartial

LangWatch « Agent Testing » fait jouer des **scénarios** (une situation pour
un utilisateur simulé, des critères pour un juge) contre une **cible**. Une
cible est ici toujours le **même agent** (`bench/langwatch/agent.py`, une
boucle outils → modèle → outils minimale), dont le paramètre de run `model`
choisit le modèle : `local` (ce que `qwen38 serve` a chargé, Bonsai 2 ou
Flash-Next), `gpt-5-mini`, `gpt-5`, `claude-sonnet-5`. Une **comparaison** est
un run avec plusieurs cibles : mêmes scénarios, mêmes outils, même prompt
système, seule la case modèle change. LangWatch affiche par cible le taux de
réussite, la latence et le coût.

Impartialité :
- le **juge** et le **simulateur d'utilisateur** sont des modèles du marché
  configurés sur LangWatch (jamais le modèle local) ;
- les critères sont vérifiés sur les **traces** (appels d'outils réels : « a
  lancé les tests avant de répondre »), pas seulement sur le texte ;
- `--repeat 3` : trois passes par scénario et par cible, pour ne pas conclure
  sur un coup de chance.

Ce que le banc ne mesure pas : la qualité brute du modèle hors harnais, et le
coût du modèle local (LangWatch ne connaît pas de prix pour lui ; la ligne
`usage` du serveur et `Scripts/pi-session-cost.py` couvrent ce point).

## 2. Ce qui existe déjà (vérifié le 2026-09-19)

| Élément | Où | État |
|---|---|---|
| Agent connecté (id `agent_16a72fa0423f42a4bc4e1`, suite `suite__bVlq4-HITTubWoHYmkqC`) | `bench/langwatch/agent.py` | testé en direct sur Bonsai 2 (corrige le bug de la fixture en 6 appels modèle, 165 s, tests verts) **et par le relais LangWatch** (`agent run` : réponses correctes en 34 à 62 s, trace unique par tour avec spans `llm` et `tool` emboîtés) |
| Fixture | `bench/langwatch/fixture/` (paquet Swift `Calc`, 5 tests dont 1 en échec voulu) | `swift test` → `Executed 5 tests, with 1 failure` |
| Création de la suite et des 8 scénarios | `bench/langwatch/scenarios.sh` (→ `scenarios.py`, API REST) | exécuté une fois le 2026-09-19 ; relançable |
| Lancement d'une comparaison | `bench/langwatch/run.sh [cibles…]` | à exécuter (L-4) |
| Environnement Python | `bench/langwatch/.venv` (`requirements.txt`) | créé |
| Clé projet LangWatch | `bench/langwatch/.env` → `LANGWATCH_API_KEY` | présente |
| Environnement d'agent | `.env` → `LANGWATCH_AGENT_ENVIRONMENT=bench-m3max` | l'agent apparaît comme `qwen38-bench · bench-m3max` |

Ce que Vincent doit fournir avant L-3 : **un modèle du marché joignable par
LangWatch** pour le juge et le simulateur, activé dans Settings → Model
Providers. Trois voies, toutes sans carte bancaire :
- **Ollama Cloud** — **c'est la voie en place** : fournisseur « Custom
  (OpenAI-compatible) » enregistré le 2026-09-19 (base URL
  `https://ollama.com/v1`, clé ollama.com/settings/keys). `run.sh` utilise par
  défaut `custom/deepseek-v4.1-flash` comme juge et simulateur, vérifié sur un
  scénario réel : verdict cohérent, 3 critères sur 3, raisonnement qui cite la
  trace. `glm-5.3-flash` convient aussi. **Ne pas utiliser `gpt-oss:120b`** :
  son raisonnement disait « tout est satisfait » et son verdict classait les
  trois critères en non satisfaits. Éviter tout juge de la famille Qwen
  (candidat testé). Les noms valides sont ceux de
  `curl https://ollama.com/api/tags` (`deepseek-v4.1-flash`, `kimi-k3`,
  `mistral-large-3:675b`, `glm-5.3`, `gemma4:31b`…).
- **Groq** ou **Google AI Studio** : clé gratuite, fournisseur natif
  (`JUDGE_MODEL=groq/openai/gpt-oss-120b` ou `gemini/gemini-2.5-flash`).
- Une clé OpenAI/Anthropic payante, si elle existe un jour.
Les cibles du marché (`gpt-*`, `claude-*`) demandent en plus la clé
correspondante dans `.env` ; sans elle, on ne compare que `local`, ce qui
suffit pour noter Bonsai 2 puis Flash-Next. Consommation côté juge :
30 à 50 k jetons par conversation, soit 300 à 400 k pour une passe de la
suite ; commencer par `REPEAT=1` et lire la page usage du fournisseur.

## 3. Fiches

### L-0 — Serveur et agent en ligne

1. Serveur, en Release, sur le pack à mesurer (Bonsai 2 par défaut) :
   ```bash
   caffeinate -i .xcodebuild/Build/Products/Release/qwen38 serve \
     --model-path /Volumes/Lexar/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
     --port 8848 --enable-thinking
   ```
   dans un terminal dédié (ou `nohup … &`). Attendre `"model_loaded":true`
   sur `curl -s http://127.0.0.1:8848/healthz`.
2. Agent, dans un second terminal :
   ```bash
   cd bench/langwatch && set -a && . ./.env && set +a && .venv/bin/python agent.py
   ```
   La ligne `qwen38-bench : connecté (bench-m3max)` s'affiche.
3. Vérifier : `npx -y langwatch agent list` montre `qwen38-bench` en ligne
   dans l'environnement `bench-m3max`.

**Porte de sortie** : la ligne de `agent list` recopiée, avec l'état en ligne.

### L-1 — Un tour par la plateforme

`langwatch agent run <id> --message "Que fait la fonction slugify ?"` où
`<id>` vient de `langwatch agent list -o json`. La réponse doit citer
`Sources/Calc/Slug.swift` et décrire minuscules, accents, tirets. Ouvrir
ensuite la trace : `langwatch trace search` (ou l'onglet Traces de l'app) montre
un span `read_file` sous l'appel. Préfixer les commandes longues par
`LANGWATCH_NO_DAEMON=1` : le démon du CLI abandonne après 25 s, un tour du
modèle local prend plus.

**Porte de sortie** : la réponse recopiée, et `langwatch trace get <id>` (id de
la trace la plus récente dans `trace search`) qui montre un span racine
`qwen38-bench` avec, dessous, des spans `llm chat.completions` et
`tool read_file`. Un appel `agent run` manuel ne porte pas de contexte de
trace de plateforme : le lien trace ↔ run ne se vérifie qu'en L-3, quand le
juge cite un span.

### L-2 — La suite et ses scénarios

`bench/langwatch/scenarios.sh` crée ou met à jour la suite « Agent de code »
et ses huit scénarios (lecture, correction de bug, ajout de fonction,
prudence, multi-tours, recherche, hors dépôt, ambigu) par l'API REST — pas par
`langwatch scenario create`, qui découpe les critères sur les virgules.
Relancer le script ne duplique rien : il retrouve chaque scénario par son nom.
Les textes vivent dans `scenarios.py` ; pour en changer un, édite le fichier
et relance.

**Porte de sortie** : la dernière ligne du script,
`Suite « Agent de code » (<id>) : 8 scénarios, 28 critères`, et la même
liste dans l'app (Agent Testing → Scenarios → Agent de code).

### L-3 — Juge et simulateur (nécessite la clé de Vincent)

1. Vérifier qu'un fournisseur est actif : `langwatch status` ou l'app
   (Settings → Model Providers). Si aucun ne l'est : STOP, ASK à Vincent.
2. Choisir le juge **d'un autre fournisseur que les cibles du marché
   comparées** quand c'est possible (par exemple juge Anthropic si la cible
   est `gpt-5-mini`, et inversement). Noter le choix dans le journal. Le
   passer à `run.sh` par `JUDGE_MODEL=<fournisseur/modèle>` ; sans cette
   variable, LangWatch prend le modèle par défaut du projet.
3. Passe de fumée : `REPEAT=1 bench/langwatch/run.sh local` (une seule cible,
   une passe). Un scénario isolé se relance avec
   `langwatch run-plan run --scenario <id> --target 'connected:qwen38-bench@bench-m3max?model=local' --judge-model custom/deepseek-v4.1-flash --simulator-model custom/deepseek-v4.1-flash --wait 15`
   (`scenario run` n'accepte pas le choix du juge). Lire le résultat dans l'app (lien imprimé par la commande) ou
   par `langwatch simulation-run list`.

**Porte de sortie** : 8 runs terminés (aucun `stalled`), au moins un critère
de trace passé (le juge cite un span dans son raisonnement — déjà observé le
2026-09-19 sur le scénario « Hors dépôt »), et pour chaque
scénario échoué, la raison recopiée en une ligne.

### L-4 — La comparaison

```bash
bench/langwatch/run.sh local                   # 8 scénarios × 3 passes, un scénario à la fois
```
Avec une clé du marché dans `.env` : `bench/langwatch/run.sh local gpt-5-mini`.
Le script lance **un scénario à la fois** (`run-plan run --scenario`, `--wait`) :
lancer la suite d'un bloc (`test-suite run`) fait tourner les 8 scénarios en
parallèle, le modèle local n'en sert qu'un, et les autres échouent en
`agent_busy` après 90 s de réessais (observé le 2026-09-19 : 7 runs sur 8 en
ERROR). Tous les runs d'une cible rejoignent le plan « Agent de code : local ».
Un scénario raté par un modèle est un résultat, pas une erreur du banc ; un
run en **ERROR** en est une (lire `simulation-run get <id>`).

Pendant le run, noter les lignes `qwen38 serve · usage` du serveur : elles
donnent les jetons et le débit réels du modèle local, que LangWatch ne
chiffre pas.

**Porte de sortie** : sur la page Results, le plan « Agent de code : local »
avec ses 24 runs (8 scénarios × 3 passes), aucun en ERROR ; taux de
réussite, latence moyenne et scénarios ratés recopiés dans le journal.

### L-5 — Flash-Next, même banc

Arrêter le serveur, le relancer sur Flash-Next 3 bits
(`/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`), relancer
`agent.py` (il relit `/v1/models`), puis `bench/langwatch/run.sh local` avec
`--name` distinct : éditer `run.sh` n'est pas nécessaire, la variable
`LANGWATCH_AGENT_ENVIRONMENT=bench-m3max-flashnext` dans l'environnement du
shell suffit pour créer une cible distincte (`qwen38-bench · bench-m3max-flashnext`).

**Porte de sortie** : sur la page Results, groupée par cible, trois lignes
comparables : Bonsai 2, Flash-Next, et le modèle du marché.

### L-6 — Bilan

Une entrée dans `docs/knowledge/log.md` (« 2026-09-xx — Banc LangWatch ») avec
le tableau taux de réussite / latence / coût par cible, les scénarios que le
modèle local rate systématiquement (3/3) et pourquoi (raison du juge), et
une ligne de conclusion. Pas de recommandation d'optimisation avant cette
ligne.

## 4. Ordre et dépendances

L-0 → L-1 → L-2 → L-3 (bloquante sans la clé) → L-4 → L-5 → L-6.
L-2 peut se faire avant L-0 (elle ne touche pas au serveur).

## 5. Pièges connus

1. **Le juge n'est pas configuré** : les runs restent « running » puis
   `stalled`. Vérifier Model Providers avant L-3.
2. **`agent_offline`** au lancement : `agent.py` n'est plus connecté (Ctrl-C,
   veille du Mac). Relancer, revérifier `agent list`.
3. **`agent_environment_unresolved`** : deux environnements en ligne pour le
   même nom. `run.sh` nomme toujours l'environnement (`@bench-m3max`), garder
   cette forme.
4. **Un tour dépasse 300 s** (`timeout` de l'agent) : `swift test` compile
   à froid à chaque nouvel espace de travail (~20 s), plus le modèle. Si ça
   arrive avec Flash-Next, réduire `max_steps` par `--param max_steps=8`
   plutôt que d'augmenter le timeout (300 s est le plafond de la plateforme).
5. **Espaces de travail** : `bench/langwatch/workspaces/<thread_id>` s'accumulent
   (une copie de la fixture par conversation, `.build` compris). Les supprimer
   entre deux campagnes : `rm -rf bench/langwatch/workspaces`.
6. **Cache de préfixe** : les lignes `usage` montrent `dont 0 en cache` sur
   les tours outillés du chemin 27B. C'est une mesure, pas une erreur du banc ;
   à consigner, pas à corriger ici.
7. **Concurrence** : le modèle local ne sert qu'une conversation à la fois
   (verrou `LOCAL_TURN` dans `agent.py`) ; `run.sh` lance donc les scénarios
   en série. Ne jamais lancer `test-suite run` sur la suite entière avec une
   cible locale. Compter environ 2 à 4 minutes par conversation locale.
8. **Après toute modification de `agent.py`**, relancer le processus : la
   plateforme lit ses paramètres et sa concurrence à la connexion.

## 6. Commandes de référence

```bash
npx -y langwatch --help                      # tout le CLI
npx -y langwatch agent list                  # agents et état en ligne
npx -y langwatch test-suite get "Agent de code" -o json
npx -y langwatch scenario list --test-suite "Agent de code"
npx -y langwatch test-suite run --help       # flags : --target, --repeat, --judge-model, --wait
npx -y langwatch simulation-run list          # runs récents
npx -y langwatch trace search                 # traces récentes (spans d'outils)
npx -y langwatch simulation-run get <runId>   # transcription, verdict, coûts
npx -y langwatch open                        # ouvre le projet dans le navigateur
```

## 7. Journal d'exécution (à remplir, à la fin du fichier)

```
## L-x — <titre> — <AAAA-MM-JJ> — validée|bloquée
- Fait : <une ligne>
- Porte de sortie observée : `<commande>` → `<ligne exacte>`
- Écart au plan : <aucun, ou quoi et pourquoi>
```

```
## ASK — L-x — <AAAA-MM-JJ>
- Contexte : <2 lignes>
- Ce que j'ai essayé : <3 lignes max>
- Question : <fermée si possible>
- Options : A) … B) …
```

### Journal

## L-0 — Serveur et agent en ligne — 2026-09-19 — validée
- Fait : serveur Release sur Bonsai 2 (`Ternary-Bonsai-2-27B-mlx-2bit`, port 8848, `--enable-thinking`) via `nohup caffeinate`, puis `agent.py` connecté dans `bench-m3max` avec le venv du banc.
- Porte de sortie observée : `LANGWATCH_NO_DAEMON=1 npx -y langwatch agent list` → `qwen38-bench  bench-m3max  online  connected  agent_16a72fa0423f42a4bc4e1`
- Écart au plan : aucun. `curl /healthz` a répondu `"model_loaded":true` dès le premier sondage ; `/v1/models` confirme `Ternary-Bonsai-2-27B-mlx-2bit` avec `loaded:true`.

## L-1 — Un tour par la plateforme — 2026-09-19 — validée
- Fait : `agent run` sur `agent_16a72fa0423f42a4bc4e1` (question « Que fait la fonction slugify ? ») ; réponse en 41 588 ms citant `Sources/Calc/Slug.swift:9-23`, minuscules/accents (`folding`, ligne 10), tirets (lignes 13-21), troncature à `maxLength` (lignes 25-30) et les deux exemples de tests.
- Porte de sortie observée : `langwatch trace get 54cef514bbcf75a40000` → `[2ce41f2e] qwen38-bench (41.50s)` racine avec, dessous, `chat.completions (8.43s)` (span.type: llm), `search`, `chat.completions`, `read_file (0ms)` (span.type: tool, `Sources/Calc/Slug.swift`) et `chat.completions (24.20s)`.
- Écart au plan : aucun. `trace search` affichait `{"value":"\n"}` pour la sortie, mais l'output réel du span racine est bien la réponse complète (l'agent renvoie `choice.content` qui commence par un saut de ligne). Préfixe `LANGWATCH_NO_DAEMON=1` utilisé comme prescrit.

## L-2 — La suite et ses scénarios — 2026-09-19 — validée
- Fait : `bench/langwatch/scenarios.sh` a mis à jour les 8 scénarios de la suite « Agent de code » (`suite__bVlq4-HITTubWoHYmkqC`) via l'API REST, chacun avec sa situation + « Tu écris toujours en français. » (idempotent, retrouvés par nom).
- Porte de sortie observée : `bench/langwatch/scenarios.sh` → `Suite « Agent de code » (suite__bVlq4-HITTubWoHYmkqC) : 8 scénarios, 28 critères` ; `npx -y langwatch test-suite get "Agent de code" -o json` liste Lecture, Correction, Ajout, Prudence, Multi-tours, Recherche, Hors dépôt, Ambigu.
- Écart au plan : `langwatch scenario list --test-suite` (commande de référence §6) n'existe pas dans le CLI 1.15 (`error: unknown option '--test-suite'`) ; remplacé par `langwatch test-suite get "Agent de code" -o json`, qui montre la même liste.
