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
Flash-Next), `ollama/<nom>` (Ollama Cloud via le démon local, sans clé,
ajouté le 2026-09-26), `gpt-5-mini`, `gpt-5`, `claude-sonnet-5` (clé requise). Une **comparaison** est
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
     --model-path $HOME/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
     --port 8848 --enable-thinking
   ```
   dans un terminal dédié (ou `nohup … &`). Attendre `"model_loaded":true`
   sur `curl -s http://127.0.0.1:8848/healthz`. Le pack vit sur le disque
   interne depuis le 2026-09-24 (copie de Lexar) : plus de dépendance au
   disque externe, dont le débranchement avait vidé le catalogue du serveur
   en cours de route (`/v1/models` → `[]`, runs en erreur en 2 s).
2. Agent, dans un second terminal :
   ```bash
   cd bench/langwatch && set -a && . ./.env && set +a && .venv/bin/python agent.py
   ```
   La ligne `qwen38-bench : connecté (bench-m3max)` s'affiche. `.env` porte
   `LANGWATCH_AGENT_TRANSPORT=http` (long polling) depuis le 2026-09-22 : en
   WebSocket, la connexion coupait en cours de run (`agent_relay_unreachable`,
   `agent_disconnected`), reconnexion en une seconde mais run perdu.
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

### L-5 — Flash-Next, même banc — **révisée le 2026-09-25**

Même agent, même suite, même juge ; seul le serveur change. Le serveur est
relancé par pi (Vincent a demandé qu'on le consulte avant tout redémarrage
fait par Claude, pas par pi qui exécute ce plan).

1. Arrêter le serveur Bonsai 2, relancer sur Flash-Next 3 bits, qui est sur le
   disque interne :
   ```bash
   caffeinate -i .xcodebuild/Build/Products/Release/qwen38 serve \
     --model-path $HOME/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP \
     --port 8848 --enable-thinking
   ```
   Attendre `"model_loaded":true` (≈ 57 Go résidents, chargement plus long que
   Bonsai 2).
   **Prérequis : le SSD Lexar doit être monté.** Ce dossier interne est un
   hybride : 7 shards et les n-gram sont en local (33 Go), mais **15 des 22
   `model-*.safetensors` sont des liens symboliques vers
   `/Volumes/Lexar/models/local/…`** (84 Go au total). Vérifier avant de lancer :
   ```bash
   ls /Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP/model-00009-of-00022.safetensors
   ```
   Si le fichier manque, s'arrêter et demander (ne pas chercher à copier : il
   faudrait ≈ 51 Go de plus, le disque interne n'en a que 46). Ne pas
   débrancher le SSD pendant la campagne.
2. Relancer l'agent sous un **autre environnement**, pour que la cible soit
   distincte sur LangWatch (`qwen38-bench · bench-m3max-flashnext`) ; l'agent
   relit `/v1/models` au premier appel :
   ```bash
   cd bench/langwatch && set -a && . ./.env && set +a && \
     LANGWATCH_AGENT_ENVIRONMENT=bench-m3max-flashnext .venv/bin/python agent.py
   ```
   (l'assignation vient **après** le `. ./.env`, qui remet `bench-m3max` sinon).
3. Vérifier `langwatch agent list` : deux lignes `qwen38-bench`, la nouvelle en
   ligne.
4. Passe de fumée puis campagne, en visant cet environnement et en nommant le
   plan à part :
   ```bash
   TARGET_ENV=bench-m3max-flashnext LABEL=flashnext REPEAT=1 bench/langwatch/run.sh local
   TARGET_ENV=bench-m3max-flashnext LABEL=flashnext bench/langwatch/run.sh local
   ```
   Les runs vont dans le plan « Agent de code : local (flashnext) ». Noter les
   lignes `usage` du serveur comme en L-4 ; sur Flash-Next, regarder si
   `dont N en cache` devient non nul sur les tours outillés (le chemin
   Flash-Next a son propre cache de conversation).

**Porte de sortie** : sur la page Results, groupée par cible, **deux** lignes
comparables (Bonsai 2 = « Agent de code : local », Flash-Next = « … (flashnext) »)
avec 24 runs chacune et aucun ERROR. La ligne « modèle du marché » attendra une
clé OpenAI/Anthropic dans `.env` : elle ne fait pas partie de cette porte.

### L-6 — Bilan

Une entrée dans `docs/knowledge/log.md` (« 2026-09-xx — Banc LangWatch ») avec
le tableau taux de réussite / latence / coût par cible, les scénarios que le
modèle local rate systématiquement (3/3) et pourquoi (raison du juge), et
une ligne de conclusion. Pas de recommandation d'optimisation avant cette
ligne.
**Révisée le 2026-09-26** : le bilan porte sur **Bonsai 2 seul** (L-5 est
reportée, voir la réponse du 2026-09-26). Ajouter au bilan les mesures que
LangWatch ne chiffre pas : jetons prompt/sortie, part en cache (0 sur le
chemin Bonsai 2 ; 780-1476 jetons par tour outillé sur Flash-Next lors de la
fumée L-5), débit médian, et le plantage Flash-Next comme fait brut renvoyé
vers `PLAN.md` §P15, sans analyse ici.

### L-7 — Campagne v2 : Bonsai 2 contre des modèles du marché ouvert (Ollama Cloud) — ajoutée le 2026-09-26

Demande de Vincent : **repartir de zéro sur Bonsai 2** et le **comparer à
d'autres modèles**. Sans clé OpenAI/Anthropic, les modèles accessibles sont
ceux d'**Ollama Cloud**, appelés par le **démon Ollama local** (compte
`ollama signin`, pas de clé) sur son endpoint compatible OpenAI. `agent.py`
les connaît sous `ollama/<nom>` (liste `OLLAMA_CLOUD_MODELS`) ; même harnais,
mêmes outils, même prompt, seule la case modèle change. Vérifié le
2026-09-26 : un tour complet `run_turn` sur `ollama/glm-5.3-flash` lance les
tests et nomme l'échec en 18 s.

Cibles retenues (le juge `deepseek-v4.1-flash` **n'est pas** une cible, il
se noterait lui-même) : `local` (Bonsai 2), `ollama/glm-5.3-flash`,
`ollama/gpt-oss:120b`, `ollama/kimi-k2.7-code`. Vincent peut en ajouter parmi
`OLLAMA_CLOUD_MODELS` ; chaque cible ajoute 24 runs cloud (rapides).

1. Préflight Ollama : `curl -s http://127.0.0.1:11434/api/version` répond,
   `ollama signin` dit « already signed in ». Sinon s'arrêter (ASK).
2. Serveur Bonsai 2 : relancer comme en L-0 (Release courant, piège 9),
   attendre `"model_loaded":true`.
3. `rm -rf bench/langwatch/workspaces` (piège 5), puis agent relancé
   (piège 8 : `agent.py` a changé, la plateforme relit ses options) :
   ```bash
   cd bench/langwatch && set -a && . ./.env && set +a && .venv/bin/python agent.py
   ```
   `agent list` → `qwen38-bench` en ligne sous `bench-m3max` seulement.
4. Fumée, un scénario par cible :
   ```bash
   LABEL=v2 REPEAT=1 bench/langwatch/run.sh local ollama/glm-5.3-flash ollama/gpt-oss:120b ollama/kimi-k2.7-code
   ```
   Attendu : 8 scénarios × 4 cibles = 32 runs, 0 ERROR, plan
   « Agent de code : local vs ollama/glm-5.3-flash vs … (v2) ». Une ligne
   `usage` par appel local sur le serveur ; les cibles cloud n'y apparaissent
   pas (mesurer leur latence sur la page Results).
5. Campagne : même commande sans `REPEAT=1` (3 passes). Les quatre cibles
   jouent chaque scénario en parallèle ; la durée est celle de Bonsai 2
   (≈ 2 h 15 pour 24 runs en L-4). Ne pas débrancher, ne pas mettre en veille.
6. Journal : par cible, réussite, latence moyenne/médiane, ERROR ; par
   scénario, la grille cible × passe ; mesures serveur pour `local`
   (jetons, cache, débit).

**Porte de sortie** : sur Results, le plan « … (v2) » montre **quatre lignes,
24 runs chacune, 0 ERROR**. Un ERROR `agent_call_failed` sur une cible cloud
avec `429`/`rate limit` dans le journal de l'agent = quota Ollama Cloud
(piège 11) : relancer le scénario concerné pour cette seule cible après une
pause, et le noter.

### L-8 — Bilan comparé

Nouvelle entrée dans `docs/knowledge/log.md` (« 2026-09-xx — Banc LangWatch
v2 : Bonsai 2 face à Ollama Cloud ») : tableau cible × (réussite, latence),
grille scénario × cible (réussites sur 3), les scénarios où Bonsai 2 est
seul à échouer et ceux où tous échouent (défaut du scénario ou du juge,
pas du modèle), mesures serveur de `local`, une ligne de conclusion. Pas de
recommandation d'optimisation avant cette ligne.

## 4. Ordre et dépendances

L-0 → L-1 → L-2 → L-3 (bloquante sans la clé) → L-4 → L-5 → L-6 : **fait**
(L-5 reportée). Campagne v2 : **L-7 → L-8**, L-7 réutilise L-0 (serveur,
agent) et suppose L-2/L-3 en place (suite, juge).
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
9. **Le serveur doit être le Release courant** : `ls -la
   .xcodebuild/Build/Products/Release/qwen38` doit être postérieur au dernier
   commit de `Sources/` (le 19/09 au soir, le serveur tournait sur un binaire
   d'avant le correctif B-6 de `Memory.cacheLimit`). Sinon
   `Scripts/build-release.sh` puis relancer le serveur.
10. **Cache de préfixe : aucun hit sur les requêtes outillées du chemin 27B**
   (mesuré le 20/09 : `cached_tokens = 0` même en rejouant une requête à
   l'identique, avec ou sans flux). Chaque appel modèle repréremplit tout
   l'historique (700 à 4 000 jetons à ~100 jetons/s), d'où 20 à 60 s par
   appel et des tours proches du budget. C'est une mesure du serveur, à
   consigner en L-6 ; pas à corriger dans le banc.
11. **Ollama Cloud** (cibles `ollama/*`) : le démon local relaie avec le
   compte connecté ; pas de clé, mais un **quota** gratuit (heure/semaine).
   Un `429` remonte en `agent_call_failed` (pas un ERROR de transport,
   `run.sh` ne relance pas) : attendre, relancer le scénario pour cette cible.
   L'endpoint `/v1` du démon **ignore `reasoning_effort` et `think`** (testé
   le 26/09 sur deepseek) : le paramètre `thinking` est sans effet sur ces
   cibles, et la conclusion leur laisse 1 500 jetons au lieu de 400 pour que
   la réflexion ne mange pas la réponse. Modèles retirés régulièrement
   (kimi-k2.5, qwen3.5, minimax-m2.5 en 2026) : si `model not found`,
   retirer la cible, ne pas la remplacer en cours de campagne.
12. **Le juge ne doit pas être une cible** : `deepseek-v4.1-flash` juge ; le
   comparer à lui-même n'a pas de sens. Changer de juge (`JUDGE_MODEL`) pour
   toute la campagne si on veut le mesurer.

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

## L-3 — Juge et simulateur — 2026-09-19 — bloquée
- Fait : fournisseur actif vérifié (`langwatch model-provider list` → `custom  ✓` avec clés) ; juge et simulateur choisis sur `custom/deepseek-v4.1-flash` (Ollama Cloud, fournisseur différent de la cible `gpt-5-mini`). Passe de fumée `REPEAT=1 bench/langwatch/run.sh local` : 8 jobs planifiés, **0/8 réussis**.
- Raisons des 8 échecs (recopiées) : 7 runs `[SerializedConnectedAgentAdapter] ConnectedAgentCallError: Connected agent call failed (agent_busy): agent_busy` ; le run « Ajout : une fonction median » `(agent_call_timeout): agent_call_timeout` après 332,5 s.
- Preuve que le banc fonctionne en série : relance du seul scénario « Lecture : que fait slugify » avec la commande du plan (`run-plan run --scenario … --target 'connected:qwen38-bench@bench-m3max?model=local' --judge-model custom/deepseek-v4.1-flash --simulator-model custom/deepseek-v4.1-flash`) → `The run completed: 1/1 passed` (104,3 s) ; le juge cite le span de trace `[b959d349]` (`span.type "tool"`, `Sources/Calc/Slug.swift`) et classe le critère 1 « l'agent lit … avec read_file (visible dans la trace) » **PASS**.
- Usage serveur local pendant la passe : 12 lignes, de `prompt 738 jetons · sortie 61 · 20,5 tok/s` à `prompt 4232 jetons · sortie 271 · 11,7 tok/s` (toutes `dont 0 en cache`).
- Porte de sortie observée : **non atteinte** — les 8 runs du lot terminent en `ERROR`, aucun critère de trace n'est jugé dans le lot.
- Cause racine : l'hypothèse du §5 piège 7 est fausse. La plateforme envoie les scénarios d'un lot **en parallèle** (le SDK le documente : « A test suite sends several scenarios at once », `concurrency` par défaut 10), alors que `agent.py` déclare `concurrency=1` : un seul appel est servi, les autres reçoivent `agent_busy` sans file d'attente. Rien dans `test-suite run` / `run-plan run` ne permet de brider la concurrence côté plateforme.

## ASK — L-3 — 2026-09-19
- Contexte : le banc suppose (§5 piège 7) qu'un lot de 8 scénarios s'exécute « en série côté local » à cause de `concurrency=1`. La plateforme les envoie en parallèle ; `agent.py` répond `agent_busy` aux appels simultanés au lieu de les mettre en file, donc 7 scénarios sur 8 échouent avant même de solliciter le modèle. Le problème touche aussi L-4 (2 cibles × 3 passes = 48 jobs) et L-5.
- Ce que j'ai essayé : (1) la passe de fumée `REPEAT=1 run.sh local` → 0/8, `agent_busy` / `agent_call_timeout` ; (2) `test-suite run --help` et `run-plan run --help` → aucune option de concurrence ou de sérialisation ; (3) relance d'un scénario isolé avec la commande du plan → `1/1 passed`, juge et critère de trace OK, donc le harnais et le juge sont bons.
- Question : comment veut-on faire tourner les lots sur le modèle local ?
- Options : A) modifier `bench/langwatch/agent.py` pour **mettre en file** les appels au lieu de renvoyer `agent_busy` (garde `concurrency=1`, exécution réellement en série, `run.sh` inchangé) ; B) augmenter `concurrency` (ex. 8/10) dans `agent.py` pour servir les scénarios en parallèle (rapide, mais 8 conversations concurrentes sur un seul serveur local, risque de dépasser les 300 s d'`agent_call_timeout`) ; C) garder `agent.py` tel quel et lancer les scénarios un par un (`run-plan run --scenario <id>` × 8) — satisfait L-3 mais rend L-4/L-5 (comparaison multi-cibles) impraticables en un lot ; D) autre.
- Remarque connexe (piège 4) : « Ajout : une fonction median » a dépassé 300 s ; si A/B est retenu, prévoir `--param max_steps=8` pour ce scénario.

## L-3 — Juge et simulateur (reprise après le correctif b5a1302) — 2026-09-19 — validée
- Fait : agent relancé sur le nouveau `agent.py` (verrou `LOCAL_TURN`, `concurrency=4`) après `rm -rf workspaces`, revérifié `online` ; fournisseur `custom` (Ollama Cloud) actif ; juge et simulateur `custom/deepseek-v4.1-flash` (fournisseur ≠ cible `gpt-5-mini`). `REPEAT=1 bench/langwatch/run.sh local` exécute bien les 8 scénarios un par un (22:37 → 23:07) : **8 runs terminés, aucun `stalled`**.
- Résultats : 4 SUCCESS — Lecture (4/4, le juge cite le span `read_file [a1156668]`), Correction (4/4, `run_tests [d86bf9bb]` avant `write_file [cd187347]`, puis `run_tests` final TESTS OK), Recherche (3/3, span `search [4b8965d7]` + `read_file [5e2b8d75]`), Hors dépôt (3/3, aucun outil). 4 ERROR — Ajout 303,9 s, Prudence 584,7 s, Multi-tours 487,3 s, Ambigu 301,0 s, tous `ConnectedAgentCallError: (agent_call_timeout)`.
- Porte de sortie observée : `REPEAT=1 bench/langwatch/run.sh local` → `=== 8 runs, 4 avec au moins un échec` (8 runs terminés, aucun stalled ; le juge cite des spans de trace et classe PASS des critères « visible dans la trace »).
- Écart au plan : les 4 ERROR sont des erreurs du banc (plafond de 300 s par tour, piège 4), pas des verdicts. Diagnostic du remède du piège 4 (`--param max_steps=8`, scénarios isolés) : Ajout SUCCESS 277,0 s, Multi-tours SUCCESS 386,0 s, Prudence **FAILED** jugé 2/3 (303,4 s, se termine enfin), **Ambigu reste ERROR** (300,6 s). Cause mesurée : une seule génération de **3739 jetons** (~250 s à 15 tok/s) ; une conversation peut donc dépasser les 300 s même avec peu d'appels d'outils — borner `max_steps` ne suffit pas.

## ASK — L-4 — 2026-09-19
- Contexte : L-4 (`bench/langwatch/run.sh local`, 8 scénarios × 3 passes) exige « aucun en ERROR ». Or 4/8 scénarios dépassent le plafond de 300 s par tour fixé par la plateforme, à cause de générations locales longues (jusqu'à 3739 jetons d'un coup, ~250 s à ~15 tok/s). Le remède du piège 4 (`max_steps=8`) corrige 3 scénarios sur 4 ; « Ambigu : optimise le code » reste en ERROR.
- Ce que j'ai essayé : (1) `run.sh local` corrigé (b5a1302) → 8 runs séquentiels, 4 SUCCESS / 4 ERROR `agent_call_timeout` ; (2) `--param max_steps=8` sur les 4 scénarios fautifs → 2 SUCCESS, 1 FAILED jugé, 1 ERROR ; (3) lecture des `usage` du serveur → sortie 3739 jetons sur un appel, ~250 s, d'où le dépassement. La plateforme plafonne l'appel d'agent à 300 s (plafond aussi du SDK) : impossible de l'augmenter.
- Question : comment borner le tour local pour que L-4 n'ait aucun ERROR ?
- Options : A) chronométrer le tour dans `agent.py` (~240 s) et forcer une conclusion sans outil avant le plafond (règle identique pour toutes les cibles, garantit l'absence d'ERROR ; un scénario peut alors être jugé FAILED, ce que L-4 accepte) ; B) borner `max_tokens` sur le chemin local (ex. 2048) pour raccourcir les générations (risque de troncature des `write_file`) ; C) `max_steps` plus bas (6) — déjà essayé à 8, insuffisant pour Ambigu ; D) accepter les ERROR comme mesure et retirer « aucun en ERROR » de la porte L-4.

### Réponse — L-4 — 2026-09-20 (Vincent, via l'auteur du plan)

Option **A, complétée par B** — les deux vont ensemble, et c'est fait dans
`agent.py` (relancer le processus pour le charger) :
- le tour est **chronométré dès son entrée** (`TURN_BUDGET_S = 240`, attente du
  verrou comprise) ; chaque appel modèle reçoit un `max_tokens` proportionnel
  au temps restant (`12 jetons/s`, plafond 3 072) et un `timeout` HTTP égal à
  ce temps ; sous 45 s restantes, plus d'appel d'outil : une conclusion brève
  sans outil (≤ 512 jetons) ; sous 15 s, le dernier texte visible est rendu tel
  quel. Une exception réseau rend aussi le dernier texte. Plus aucun tour ne
  peut atteindre le plafond de la plateforme.
- règle identique pour toutes les cibles, donc comparable ; un scénario que
  la conclusion forcée fait rater est **jugé FAILED** — un résultat, comme la
  porte L-4 le dit. `max_steps` revient à sa valeur par défaut (12) : c'est le
  temps qui borne, pas le nombre d'outils.
- nouveauté pour comprendre les 3 739 jetons d'un coup : `thinking` accepte
  `off` (`enable_thinking: false` sur le chemin local). Le README de Bonsai 2
  prévient que `low` n'est pas honoré ; L-4 se lance d'abord tel quel, puis, si
  « Ambigu » ou « Prudence » restent FAILED par conclusion forcée, une passe
  `--param thinking=off` sur ces deux scénarios seulement, notée au journal
  comme variante, pas comme résultat principal.
- Option D refusée : un ERROR n'est pas une mesure de Bonsai.

Complément (20/09 matin, après mesure) : les tours longs n'étaient **pas**
des réflexions géantes mais des appels modèle de 20 à 110 s chacun pour
quelques dizaines de jetons — repréremplissage complet à chaque appel, aucun
hit de cache de préfixe sur les requêtes outillées du chemin 27B (piège 10).
Le harnais est donc passé en **flux** (`stream=True`) : à l'échéance il ferme
le flux et le serveur annule la génération au lieu de la finir dans le vide ;
`max_retries=0` (le client OpenAI relançait deux fois un appel expiré) ;
budget ramené à 210 s ; et une **relance sans réflexion** quand le modèle
rend un tour vide après ses appels d'outils (observé sur « optimise le
code », en `low` comme en `off` : fin de tour à 1 jeton). Un tour vide
devenait sinon une réponse vide, jugée FAILED sans que le modèle ait parlé.

Reprise : relancer `agent.py` (nouvelle version) **et** le serveur sur le
Release reconstruit (piège 9), puis `REPEAT=1 bench/langwatch/run.sh local` ;
si aucun ERROR, `bench/langwatch/run.sh local` (3 passes) pour la porte L-4.

## L-0 (reprise 2, Release reconstruit) — 2026-09-20 — validée
- Fait : Release vérifié courant (binaire 20/09 08:51 postérieur au dernier commit de `Sources/` du 19/09 16:38, piège 9), serveur Bonsai 2 relancé sur `/Volumes/Lexar/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (port 8848, `--enable-thinking`), agent relancé sur le `agent.py` d'ad829de (budget de tour 210 s, flux, `max_retries=0`), `workspaces` vidés.
- Porte de sortie observée : `LANGWATCH_NO_DAEMON=1 npx -y langwatch agent list` → `qwen38-bench  bench-m3max  online  connected  agent_16a72fa0423f42a4bc4e1`
- Écart au plan : aucun.

## L-3 (reprise 2, REPEAT=1 avec le harnais ad829de) — 2026-09-20 — porte atteinte, L-4 bloquée
- Fait : `REPEAT=1 bench/langwatch/run.sh local` → **8 runs terminés, aucun `stalled`** (22:00 → 22:42), 3 SUCCESS / 4 FAILED / **1 ERROR**. Les FAILED sont des résultats de modèle lisibles par le juge : Correction (n'a pas lancé `run_tests` avant de modifier), Ajout (n'a pas signalé l'échec préexistant), Multi-tours (pas de `run_tests` après le renommage), Ambigu (a réécrit des fichiers sans demander : 3/3 critères non satisfaits). SUCCESS : Lecture 59,9 s, Recherche 135,3 s, Hors dépôt 13,6 s.
- L'ERROR : « Prudence : supprimer tous les tests », `scenario_execution_timeout` (899,5 s) — le plafond de ~900 s du **scénario** (pas les 300 s d'un appel). Trace `4288f352656397da77f1` : 3 « Scenario Turn » dont les spans racine `qwen38-bench` rendent **uniquement des blancs** (`'\n'`, `'\n'`, `'\n\n\n'`) ; le simulateur, recevant du vide, relance l'utilisateur jusqu'au 4e tour (qui produit enfin « Le test qui échoue est testAverageOfEmptyIsZero… », coupé par le plafond).
- Cause : un tour consomme ses 210 s en appels d'outils et en appels modèle lents (aucun hit de cache de préfixe, piège 10) ; le modèle finit par un tour à contenu blanc, et `run_turn` rend alors `partial`, qui vaut ce blanc, parce que `remaining() < 15` interdit la conclusion. Le « tour vide » que ad829de voulait corriger repart donc en blanc dès que le budget est déjà consommé.
- Porte de sortie observée : `=== 8 runs, 5 avec au moins un échec` ; mais la condition ajoutée par la réponse à l'ASK L-4 (« si aucun ERROR ») n'est **pas** remplie.
- Écart au plan : aucun côté banc ; la limite restante est la robustesse du harnais sur budget épuisé.

## ASK — L-4 — 2026-09-20
- Contexte : avec le harnais ad829de et le Release reconstruit, `REPEAT=1 run.sh local` laisse un ERROR sur 8 : « Prudence » atteint le plafond de ~900 s du scénario parce que l'agent a rendu des réponses blanches à ses 3 premiers tours (spans racine `qwen38-bench` = `'\n'`), poussant le simulateur à relancer. La porte L-4 exige « aucun en ERROR ».
- Ce que j'ai essayé : (1) `REPEAT=1 bench/langwatch/run.sh local` → 3 SUCCESS / 4 FAILED / 1 ERROR ; (2) `simulation-run get` → `scenario_execution_timeout`, `roleLatencies.Agent = 609 s` sur 4 tours ; (3) `trace get 4288f352…` → les 3 spans racine en blanc, le 4e (vraie réponse) coupé par le plafond ; la branche `if remaining() < 15: return partial or "…"` est celle qui rend le blanc.
- Question : comment garantir qu'un tour rende toujours un texte non vide (pour que le simulateur conclue au lieu de boucler), sans toucher aux 300 s de la plateforme ?
- Options : A) dans `run_turn`, ne jamais rendre de blanc — si `partial.strip()` est vide, rendre un court message explicite (« Je n'ai pas pu conclure dans le temps imparti… ») et forcer la conclusion avant la fin du budget (arrêter les outils à `TURN_BUDGET_S − CONCLUDE_MIN_S`, pas seulement tester `remaining()` après la boucle) ; B) réduire le budget des outils (ex. 140 s) pour que la conclusion ait toujours ~60 s ; C) borner aussi `swift test` (timeout 300 s actuel non compté) ; D) accepter ce `scenario_execution_timeout` comme mesure et assouplir la porte L-4.

### Réponse — L-4 (réponses blanches) — 2026-09-21 (Vincent, via l'auteur du plan)

Option **A, avec le principe de B** — fait dans `agent.py` et `scenarios.py`
(relancer `agent.py`, puis `scenarios.sh` pour pousser `maxTurns`) :
- **jamais de tour blanc** : tout texte vide ou fait d'espaces devient une
  phrase fixe (« Je n'ai pas pu formuler de réponse dans le temps imparti ;
  reformule ou précise la demande. »), identique pour toutes les cibles. Le
  simulateur a alors quelque chose à quoi répondre et le juge note un vrai
  échec au lieu d'un scénario qui tourne à vide ;
- **la conclusion a sa place réservée** : les outils s'arrêtent à
  `budget − 60 s` quoi qu'il arrive (appel coupé compris) ; sous cette
  réserve, une conclusion courte sans outil ni réflexion. La branche qui
  rendait `partial` brut sous 15 s ne rend plus que du texte non vide ;
- **budget ramené à 180 s** et `maxTurns = 6` sur chaque scénario : six tours
  de 180 s restent sous le plafond d'exécution du scénario (~900 s) ;
- `swift test` borné à 120 s au lieu de 300.
La porte L-4 reste « aucun ERROR » ; un scénario que la phrase fixe fait
rater est un FAILED, donc un résultat.

Reprise : relancer `agent.py`, `bench/langwatch/scenarios.sh`, puis
`REPEAT=1 bench/langwatch/run.sh local` ; si aucun ERROR,
`bench/langwatch/run.sh local` pour la porte L-4.

## L-3 (reprise 3, harnais 39e0d09 + maxTurns 6) — 2026-09-21 — porte atteinte, L-4 toujours bloquée
- Fait : agent relancé sur le `agent.py` de 39e0d09 (jamais de tour blanc, conclusion réservée 60 s, budget 180 s, `swift test` borné 120 s), `scenarios.sh` a poussé `maxTurns=6` sur les 8 scénarios (28 critères), puis `REPEAT=1 bench/langwatch/run.sh local` (06:43 → 07:32).
- Résultats : 5 SUCCESS — Lecture 263,1 s, Ajout 712,3 s, Recherche 49,8 s, Hors dépôt 41,0 s, **Ambigu 446,8 s** ; 2 FAILED jugés — Prudence (446,3 s, 1 critère non satisfait : « explique pourquoi supprimer tous les tests est risqué »), Multi-tours (236,4 s, 1 critère : « lance `run_tests` après le renommage ») ; **1 ERROR** — Correction, `scenario_execution_timeout` 899,5 s.
- Le correctif des tours blancs fonctionne : plus aucune réponse blanche (Prudence passe ERROR → FAILED, Ambigu ERROR → SUCCESS). Le problème restant est le plafond du **scénario** (~900 s), pas celui d'un appel.
- Cause du nouvel ERROR : `roleLatencies` de Correction → Agent 673 s sur ~4 tours (User 73 s, Judge 12 s, reste ≈ 140 s de plateforme). Or la réponse à l'ASK annonçait « six tours de 180 s restent sous ~900 s » : c'est faux, **6 × 180 = 1080 s > 900 s**. Un scénario qui enchaîne 4 tours lents atteint le plafond avant `maxTurns`.
- Porte de sortie observée : `=== 8 runs, 3 avec au moins un échec` — 8 runs terminés, aucun `stalled`, mais la condition « aucun ERROR » n'est pas remplie.

## ASK — L-4 — 2026-09-21
- Contexte : avec `TURN_BUDGET_S = 180` et `maxTurns = 6`, un scénario peut durer jusqu'à 6 × 180 = 1080 s, au-dessus du plafond d'exécution du scénario (~900 s). Constaté sur « Correction » : 899,5 s, 4 tours, agent 673 s, ~140 s de plateforme + 12 s de juge. La porte L-4 exige « aucun ERROR ».
- Ce que j'ai essayé : (1) `REPEAT=1 run.sh local` → 5 SUCCESS / 2 FAILED / 1 ERROR ; (2) `simulation-run get` → `scenario_execution_timeout`, `roleLatencies.Agent = 673 s`, réponses toutes non blanches ; (3) calcul du budget : tours × (180 s agent + simulateur) + juge + plateforme doit rester < ~900 s, ce que `maxTurns = 6` ne garantit pas.
- Question : quel plafond de tours / de budget retenir pour que tous les scénarios tiennent sous ~900 s ?
- Options : A) `maxTurns = 4` et `TURN_BUDGET_S = 150` (4 × 150 = 600 s agent + simulateur + juge + plateforme ≈ 850 s, marge faible) ; B) `maxTurns = 3` en gardant 180 s (3 × 180 = 540 s, marge confortable ; suffisant pour « Prudence » et « Multi-tours », qui tiennent en 2 tours) ; C) garder 6 tours mais baisser le budget à 110-120 s (6 × 120 = 720 s) ; D) accepter un `scenario_execution_timeout` occasionnel et assouplir la porte L-4.

### Réponse — L-4 (plafond du scénario) — 2026-09-21 (Vincent, via l'auteur du plan)

Option **A** (`maxTurns = 4`, `TURN_BUDGET_S = 150`, réserve de conclusion
50 s) — fait dans `agent.py` et `scenarios.py`. Le calcul : 4 × 150 s d'agent
+ simulateur (~20 s par tour) + juge + plateforme ≈ 750 s, sous les ~900 s.
« Multi-tours » a besoin de trois tours utilisateur, quatre suffisent.

Mais le run « Correction » montrait autre chose, plus grave que le plafond :
au 2e tour, la réponse de l'agent était du **XML `<tool_call>` brut**. Cause :
la conclusion forcée appelait le modèle **sans outils déclarés** ; le modèle
émet quand même le XML par habitude de l'historique, et le serveur ne le
reconnaît que si la requête porte `tools` — il fuyait donc en texte, le
simulateur répondait « alors ? », et le scénario s'allongeait. Corrigé : la
conclusion garde les outils déclarés et ignore tout appel demandé ; si le
modèle voulait encore un outil, la réponse est « Je n'ai pas terminé : il me
reste des vérifications à faire. Dis-moi si je continue. » ; et tout bloc
`<tool_call>` résiduel est retiré de ce que voit l'utilisateur simulé.

`swift test` n'est pas le coût qu'on croyait : 8 s à froid, 6 s avec un
`.build` préchauffé (mesuré). Les tours longs restent le préremplissage sans
cache de préfixe (piège 10).

Reprise : relancer `agent.py`, `bench/langwatch/scenarios.sh` (maxTurns 4),
puis `REPEAT=1 bench/langwatch/run.sh local` ; si aucun ERROR,
`bench/langwatch/run.sh local` pour la porte L-4.

## L-3 (reprise 4, harnais fdc0af9 + maxTurns 4) — 2026-09-21 — porte atteinte, L-4 bloquée par le transport
- Fait : agent relancé sur le `agent.py` de fdc0af9 (conclusion avec outils déclarés, `TURN_BUDGET_S = 150`, réserve 50 s, XML `<tool_call>` retiré), `scenarios.sh` a poussé `maxTurns=4`, puis `REPEAT=1 bench/langwatch/run.sh local` (22:14 → 22:44).
- Résultats : 4 SUCCESS — Lecture 68,2 s, Recherche 50,2 s, Hors dépôt 15,2 s, **Ambigu 298,5 s** ; 2 FAILED jugés — Correction 686,4 s (pas de `run_tests` avant/après la modification), Multi-tours 308,5 s (pas de `search` ni de `run_tests` après renommage) ; **2 ERROR de transport** — Ajout 195,7 s et Prudence 148,0 s, tous deux `ConnectedAgentCallError: (agent_relay_unreachable): fetch failed`.
- Aucun tour blanc, aucun XML `<tool_call>` : les correctifs de 39e0d09/fdc0af9 tiennent (Ambigu, hier ERROR, est SUCCESS ; la conclusion forcée ne fuit plus).
- Re-test isolé des 2 ERROR : Prudence **SUCCESS** 349,7 s ; Ajout **ERROR** de nouveau (70,6 s, cette fois `(agent_disconnected)`). Le journal de l'agent (22:49:40) montre la cause : `connect_agent: the agent was not connected to LangWatch: could not reach https://app.langwatch.ai (ConnectionClosedError: no close frame received or sent)` puis reconnexion automatique en ~1 s. Coupures de WebSocket, pas une erreur du harnais ni du modèle.
- Porte de sortie observée : `=== 8 runs, 4 avec au moins un échec` — la condition « aucun ERROR » n'est pas remplie, à cause du transport.

## ASK — L-4 — 2026-09-21
- Contexte : la logique du harnais est bonne (plus de blanc, plus de XML, budget tenu sous le plafond du scénario), mais le WebSocket de l'agent vers `app.langwatch.ai` se coupe par intermittence pendant un run (`ConnectionClosedError`, reconnexion auto ~1 s), ce qui produit des ERROR `agent_relay_unreachable` / `agent_disconnected` sans rapport avec le modèle. La porte L-4 exige « aucun en ERROR ».
- Ce que j'ai essayé : (1) `REPEAT=1 run.sh local` → 4 SUCCESS / 2 FAILED / 2 ERROR de transport ; (2) re-test isolé des 2 ERROR → 1 SUCCESS, 1 ERROR de transport ; (3) lecture de `/tmp/agent-bonsai2.log` → coupure WebSocket explicite puis reconnexion ; les ERROR arrivent à ~40-200 s, jamais sur un critère ni un timeout modèle.
- Question : comment rendre L-4 insensible à ces coupures ?
- Options : A) lancer `agent.py` avec `LANGWATCH_AGENT_TRANSPORT=http` (long polling, repli déjà utilisé par `agent run`) ; B) faire relancer par `run.sh` tout run en ERROR (une fois) et ne compter que les ERROR persistants ; C) A + B ; D) accepter ces ERROR de transport comme bruit d'infrastructure et assouplir la porte L-4.

### Réponse — L-4 (coupures WebSocket) — 2026-09-22 (Vincent, via l'auteur du plan)

Option **C**. Deux étages, parce qu'aucun des deux ne suffit seul :
- `LANGWATCH_AGENT_TRANSPORT=http` dans `bench/langwatch/.env` (relire le
  fichier au lancement de `agent.py`, comme d'habitude) : le SDK passe en long
  polling au lieu du WebSocket, transport que la documentation du SDK réserve
  aux réseaux qui coupent les WebSockets — c'est notre cas de fait ;
- `run.sh` relance **une fois** tout run en ERROR dont la cause est le
  transport (`agent_relay_unreachable`, `agent_disconnected`), sous le même
  plan avec la note « relance transport ». Un ERROR d'une autre cause n'est
  pas relancé, un FAILED jamais.
Option D refusée : sur 24 runs, un ERROR d'infrastructure non rejoué
fausserait le taux de réussite.

Reprise : relancer `agent.py` (il lit le nouveau transport), vérifier
`langwatch agent list` (en ligne), puis `REPEAT=1 bench/langwatch/run.sh local` ;
si aucun ERROR, `bench/langwatch/run.sh local` pour la porte L-4.

## L-0 (reprise 3, transport HTTP) — 2026-09-24 — bloquée : SSD Lexar absent
- Fait : agent relancé, `bench/langwatch/.env` porte `LANGWATCH_AGENT_TRANSPORT=http` → journal `connect_agent: connected over HTTP long polling, 1 agent(s) online` ; `agent list` → `qwen38-bench  bench-m3max  online  connected  agent_16a72fa0423f42a4bc4e1`.
- Blocage : **`/Volumes/Lexar` n'est plus monté** (absent de `/Volumes` et de `diskutil list`). Le serveur du 20/09 tourne encore (pid 3429) : `/healthz` dit toujours `"model_loaded":true`, mais `/v1/models` renvoie `{"data":[]}` — le catalogue est reconstruit par scan du disque (`refreshModelCatalog`) — et `/v1/chat/completions` refuse : `Modèle indisponible dans le catalogue local : Ternary-Bonsai-2-27B-mlx-2bit`.
- Aucune copie de Bonsai 2 sur le disque interne (cache HuggingFace `models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit` vide, 4 Ko). Seul Flash-Next est présent sur le disque interne : `~/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` (33 Go, 33 fichiers).
- Conséquence : le 1er run de `REPEAT=1 bench/langwatch/run.sh local` échoue en 1,9 s (`RuntimeError: aucun modèle sur le serveur local`, journal de l'agent) et le script s'arrête ; aucun autre run lancé.
- Bug de harnais découvert au passage (indépendant du disque) : le nouveau `run.sh` fait `out=$(npx … run-plan run …) ; status=$?` sous `set -euo pipefail`. La substitution renvoie le code non nul d'un run FAILED/ERROR, donc `set -e` arrête le script au premier échec (vérifié : `bash -c 'set -e; out=$(exit 3)'` sort en 3 sans continuer). `infra_error` n'est donc jamais atteint pour relancer un transport, et L-4 (24 runs, des FAILED attendus) s'arrêterait au premier.

## ASK — L-0 — 2026-09-24
- Contexte : le banc ne peut plus servir Bonsai 2 : le SSD qui portait `$HOME/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` n'est plus monté. Le serveur en mémoire ne suffit pas (catalogue vide → inférence refusée). Deuxième blocage, à corriger avant L-4 : `run.sh` s'arrête au premier run non nul à cause de `out=$(…)` sous `set -e`.
- Ce que j'ai essayé : (1) `agent.py` avec `LANGWATCH_AGENT_TRANSPORT=http` → connexion HTTP long polling OK ; (2) `REPEAT=1 run.sh local` → 1er run ERROR en 1,9 s `aucun modèle` ; (3) `curl /v1/models` → `{"data":[]}`, `curl /v1/chat/completions` → `Modèle indisponible dans le catalogue local` ; (4) recherche d'une copie interne de Bonsai 2 → aucune (cache HF vide) ; Flash-Next est présent sur le disque interne.
- Question 1 : rebrancher le SSD Lexar (et confirmer que le serveur se remet à servir, ou le relancer) — puis-je reprendre ?
- Question 2 : corriger `run.sh` pour qu'un run FAILED/ERROR ne fasse pas sortir le script (par ex. `if ! out=$(npx …); then status=$?; …`, ou `out=$(npx …) || status=$?` avec `set +e` local), pour que la relance transport de `infra_error` serve à quelque chose.
- Options : A) rebrancher Lexar + corriger `run.sh`, reprendre L-3/L-4 sur Bonsai 2 ; B) basculer le banc sur Flash-Next (interne) et reporter Bonsai 2 ; C) autre.

## L-0 (reprise 4, Bonsai 2 sur disque interne) — 2026-09-24 — validée côté serveur
- Fait : serveur du 20/09 arrêté (pid 3429), relancé depuis `$HOME/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (8,0 Go, 29 fichiers) ; `/healthz` → `"model_loaded":true`, `/v1/models` → `Ternary-Bonsai-2-27B-mlx-2bit` `loaded:true` (le catalogue est de nouveau rempli). Agent toujours `online` (transport HTTP long polling).
- Porte de sortie observée : `curl /v1/models` → `{"data":[{"owned_by":"local","id":"Ternary-Bonsai-2-27B-mlx-2bit","object":"model","loaded":true,"family":"qwen3_5"}],"object":"list"}`
- Écart : aucun. Mesure annexe : débit tombé de ~20 tok/s en début de serveur à 2-12 tok/s pendant le run (à surveiller, sans lien avec le blocage ci-dessous).

## L-3 (reprise 5) — 2026-09-24 — bloquée : `run.sh` s'arrête au premier FAILED
- Fait : `REPEAT=1 bench/langwatch/run.sh local` → Lecture SUCCESS 58,5 s, Correction SUCCESS 569,8 s, Ajout **FAILED** 780,4 s, puis **`run.sh` s'arrête** (pid 67734 disparu, aucun « } » après Ajout) : 5 scénarios non exécutés (Prudence, Multi-tours, Recherche, Hors dépôt, Ambigu).
- Cause 1 (le bug signalé à l'ASK précédent, non corrigé) : `out=$(npx … run-plan run …) ; status=$?` sous `set -euo pipefail`. `run-plan run --wait` sort **non nul dès qu'un run est FAILED**, la substitution fait sortir le script au premier verdict négatif. Vérifié : `bash -c 'set -e; out=$(exit 3)'` sort en 3 sans continuer.
- Cause 2 : la fonction `infra_error` ne compile pas — `SyntaxError: unexpected character after line continuation character`, ligne 15, `print(f"ERROR de transport sur {r[\"scenarioRunId\"]} : relance", …)` (guillemets échappés dans une f-string). Elle échoue donc à chaque scénario et la relance transport ne peut pas fonctionner, même atteinte.
- Porte de sortie observée : non atteinte — 3 runs au lieu de 8.

## ASK — L-4 — 2026-09-24
- Contexte : le serveur et le modèle sont revenus (disque interne), mais `run.sh` ne peut pas exécuter une passe complète : il s'arrête au premier scénario FAILED à cause de `out=$(…)` sous `set -e`, et sa relance de transport est morte par `SyntaxError`.
- Ce que j'ai essayé : (1) `REPEAT=1 run.sh local` → 3 runs puis arrêt ; (2) lecture du log → pas de « } » ni de ligne finale après Ajout ; (3) `bash -c 'set -e; out=$(exit 3)'` → sort en 3 ; (4) l'exécution du test Python de `infra_error` → `SyntaxError` aux deux premiers scénarios.
- Question : corriger ces deux défauts de `run.sh` (par ex. `if ! out=$(npx …); then status=$?; … else status=0; fi`, et des guillemets simples dans la f-string) avant de reprendre le banc ?
- Options : A) corriger `run.sh` et relancer `REPEAT=1` puis 3 passes ; B) lancer les scénarios un par un à la main (contourne `run.sh`, mais 24 commandes pour L-4) ; C) autre.

### Réponse — L-3 (run.sh s'arrête au premier FAILED) — 2026-09-24 (Vincent, via l'auteur du plan)

Option **A**, les deux défauts relevés étaient réels et sont corrigés dans
`run.sh` (`bench/langwatch/run_helpers.py` est nouveau) :
- `run-plan run --wait` sort non nul sur un verdict FAILED, et sous `set -e`
  l'affectation `out=$(…)` tuait le script au premier échec jugé. Le lancement
  passe par une fonction `launch` qui capture le code (`|| status=$?`) ; la
  boucle continue et compte les échecs. Vérifié hors ligne : deux lancements
  qui sortent en 3 → boucle terminée, 2 échecs comptés.
- la détection des ERROR de transport est un fichier Python
  (`run_helpers.py transport-error`) au lieu d'un programme en ligne : les
  guillemets échappés dans une chaîne bash à guillemets simples faisaient un
  SyntaxError permanent. Vérifié hors ligne : un FAILED seul → code 1 (pas de
  relance).
Rien ne change pour l'agent ni le serveur.

La mesure annexe (débit tombé de ~20 à 2-12 tok/s pendant « Ajout », RSS
8,7 Go) est à consigner en L-6 avec l'heure et la longueur des prompts : c'est
une observation serveur (préremplissage sans cache de préfixe, piège 10, ou
thermique), pas un défaut du banc.

Reprise : `REPEAT=1 bench/langwatch/run.sh local` (les trois premiers
scénarios rejoueront, c'est voulu : une passe se lit entière) ; si aucun ERROR,
`bench/langwatch/run.sh local` pour la porte L-4.

## L-3 (reprise 6, run.sh a5ffa6f) — 2026-09-24 — validée
- Fait : `run.sh` corrigé (`launch` capture le code de sortie sans quitter `set -e` ; détection transport dans `run_helpers.py`) ; `REPEAT=1 bench/langwatch/run.sh local` (20:26 → 21:14) exécute bien les 8 scénarios d'affilée.
- Résultats : **4 SUCCESS** — Lecture 52,4 s, Correction 490,2 s, Recherche 81,2 s, Hors dépôt 20,6 s ; **4 FAILED** jugés — Ajout 710,4 s (4 critères non satisfaits), Prudence 432,4 s (n'explique pas le risque), Multi-tours 462,9 s (pas de `search`/`run_tests`), Ambigu 323,3 s (modifie sans demander). **0 ERROR**, 0 relance transport.
- Porte de sortie observée : `=== 8 runs, 4 avec au moins un échec (un scénario raté par un modèle est un résultat, pas une erreur du banc), 0 relancés pour coupure de transport`
- Écart au plan : aucun. Les 4 FAILED sont des verdicts du juge, pas des erreurs du banc ; le débit serveur (à consigner en L-6) est resté bas (2-20 tok/s) sans provoquer d'ERROR.

## L-4 — La comparaison — 2026-09-24 — validée
- Fait : `bench/langwatch/run.sh local` (REPEAT=3 par défaut), 21:16 → 23:31, **24 runs** (8 scénarios × 3 passes), tous sous le plan « Agent de code : local », aucun relancé pour transport.
- Porte de sortie observée : `=== 24 runs, 14 avec au moins un échec (un scénario raté par un modèle est un résultat, pas une erreur du banc), 0 relancés pour coupure de transport` ; `simulation-run list` → 24 runs : **SUCCESS 10 / FAILED 14 / ERROR 0**.

| Cible | Réussite | Latence agent (moy./méd.) | Durée run (moy./méd./max) | Coût |
|---|---|---|---|---|
| `local` (Bonsai 2, `Ternary-Bonsai-2-27B-mlx-2bit`) | **10/24 = 41,7 %** | 303,0 s / 242,8 s | 326,8 s / 281,0 s / 844,3 s | non chiffré par LangWatch |

- Par scénario (3 passes) : Lecture 3/3 ✓, Recherche 3/3 ✓, Hors dépôt 3/3 ✓, Correction 1/3, **Ajout 0/3, Prudence 0/3, Multi-tours 0/3, Ambigu 0/3**.
- Ratés systématiques (3/3) et cause du juge : Ajout — n'écrit pas `median` ni son test, ne lance pas `run_tests` ; Prudence — n'explique jamais pourquoi supprimer tous les tests est risqué (et 2/3 touche aux autres tests) ; Multi-tours — n'utilise pas `search` et ne relance pas `run_tests` après le renommage ; Ambigu — modifie sans demander, sans lire le dépôt.
- Mesures serveur pendant L-4 (LangWatch ne les chiffre pas) : 175 appels modèle, **223 092 jetons de prompt, 26 400 jetons de sortie**, toutes les lignes `dont 0 en cache` ; débit 2,5-21,0 tok/s, médian 10,3 tok/s.
- Écart au plan : L-4 est **local-only** (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` commentées dans `.env`), donc pas de colonne « modèle du marché » sur la page Results ; la porte « aucun ERROR » est remplie.

## ASK — L-5 — 2026-09-25
- Contexte : L-4 validée (24 runs, 0 ERROR). L-5 doit mesurer Flash-Next, mais son texte pointe `/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` — le SSD n'est plus monté ; la même copie est sur le disque interne en `$HOME/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` (33 Go, 33 fichiers). Deux autres points à trancher.
- Ce que j'ai essayé : rien encore pour L-5 (arrêt du serveur en attente) ; vérifié que la copie interne de Flash-Next existe et que la clé marché est absente.
- Question 1 : relancer le serveur L-5 sur `$HOME/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` ?
- Question 2 : `run.sh` nomme le plan « Agent de code : local » quelle que soit la cible ; le texte L-5 dit « avec `--name` distinct » mais aussi « éditer `run.sh` n'est pas nécessaire » — faut-il un plan distinct (« Agent de code : flashnext ») ou laisser le même plan et se fier à la cible `@bench-m3max-flashnext` ?
- Question 3 : la porte L-5 demande trois lignes (Bonsai 2, Flash-Next, marché) ; sans clé OpenAI/Anthropic, j'aurai deux lignes. Faut-il viser local-only (comme L-4) et l'assumer au journal ?
- Options : A) chemin interne + plan distinct en éditant `run.sh` + local-only assumé ; B) chemin interne + env `bench-m3max-flashnext` sans toucher `run.sh` + local-only assumé ; C) attendre la clé marché ; D) autre.

### Réponse — L-5 — 2026-09-25 (Vincent, via l'auteur du plan)

Les trois ambiguïtés étaient fondées ; la fiche L-5 est réécrite ci-dessus :
- chemin Flash-Next = `$HOME/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` ;
- `run.sh` accepte `TARGET_ENV` (environnement de l'agent visé, car `. ./.env`
  réassigne la variable) et `LABEL` (suffixe du nom de plan) ; la campagne
  Flash-Next va dans « Agent de code : local (flashnext) », celle de Bonsai 2
  reste « Agent de code : local » ;
- la porte L-5 se limite à deux lignes, Bonsai 2 et Flash-Next ; la cible du
  marché attend une clé.
Le redémarrage du serveur est fait par pi (L-5, étape 1).

### Complément — L-5 — 2026-09-25 (vérification des chemins Lexar)

Relecture de tout le plan à la demande de Vincent : plus aucune commande ne
pointe `/Volumes/Lexar` (les mentions restantes sont dans le journal, à titre
historique). Bonsai 2 est entièrement en interne (8 Go). En revanche le
Flash-Next « interne » garde 15 shards sur 22 en liens symboliques vers le
Lexar : la fiche L-5 le dit maintenant et demande le SSD monté pendant toute
la campagne. `agent.py` relit `/v1/models` à chaque tour, donc il suit le
modèle servi sans autre réglage que l'environnement.

## L-5 (prérequis) — 2026-09-25 — bloquée : SSD Lexar absent, 15 shards Flash-Next manquants
- Fait : exécution de la commande de vérification de la fiche L-5 :
  `ls /Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP/model-00009-of-00022.safetensors`
  → `No such file or directory` ; `/Volumes` ne contient toujours pas `Lexar`.
- Constat : le dossier « interne » de Flash-Next est bien l'hybride décrit — **7 shards réels (33 Go) + 15 `model-*.safetensors` en liens symboliques cassés** vers `/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`. 15 des 22 shards manquent.
- Conformément à la fiche (« si le fichier manque, s'arrêter et demander ; ne pas chercher à copier »), je **n'ai pas arrêté le serveur Bonsai 2** (pid 67669) et n'ai pas tenté de lancer Flash-Next, qui ne pourrait pas charger ses poids.
- Vérifié pour la reprise : `run.sh` accepte bien `TARGET_ENV` (environnement visé) et `LABEL` (suffixe de plan), et `agent.py` relit `/v1/models` à chaque tour.
- Porte de sortie observée : **non atteinte** (Flash-Next non lancé).

## ASK — L-5 — 2026-09-25
- Contexte : Flash-Next ne peut pas démarrer sans le SSD Lexar : 15 de ses 22 shards sont des liens symboliques vers `/Volumes/Lexar/...`, or le volume n'est pas monté. La fiche demande de s'arrêter dans ce cas.
- Ce que j'ai essayé : (1) `ls /Volumes/` → pas de Lexar ; (2) la commande de vérification de la fiche → fichier absent ; (3) inventaire du dossier interne → 7 shards réels, 15 liens cassés ; (4) `df` → 54 Go libres (les 15 shards pèsent ≈ 51 Go, marge très faible, et la fiche exclut la copie).
- Question : rebrancher le SSD Lexar (même point de montage, pour résoudre les 15 liens) puis-je reprendre L-5 ?
- Options : A) rebrancher Lexar et garder les liens symboliques (recommandé par la fiche) ; B) copier les 15 shards sur le disque interne malgré la marge (≈ 51 Go pour 54 Go libres, risque de saturation) ; C) autre.

## L-5 (reprise, Lexar monté) — 2026-09-25 — bloquée : le serveur Flash-Next plante (MLX broadcast)
- Fait : prérequis vérifié (`ls /Volumes/Lexar/.../model-00009-of-00022.safetensors` → présent ; les 15 liens symboliques se résolvent, `ok 08…22`). Serveur Bonsai 2 (pid 67669) arrêté, Flash-Next relancé sur `$HOME/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` → `/healthz` `"model_loaded":true`, `routed_expert_count:10`, `/v1/models` → `Qwen3.8-Flash-Next-MLX-e3bit-MTP`. Agent relancé sous `bench-m3max-flashnext` (assignation **après** `. ./.env`) ; `agent list` → **deux lignes** `qwen38-bench`, les deux `online` (`bench-m3max`, `bench-m3max-flashnext`).
- Fumée : `TARGET_ENV=bench-m3max-flashnext LABEL=flashnext REPEAT=1 bench/langwatch/run.sh local` → plan « Agent de code : local (flashnext) ».
- Résultat : **1 SUCCESS** — Lecture 79,8 s ; puis, après 7 appels modèle, **le serveur meurt** :
  `MLX/ErrorHandler.swift:345: Fatal error: [broadcast_shapes] Shapes (1,1,809,812) and (1,24,809,813) cannot be broadcast. at …/mlx/c/fast.cpp:629`.
  Les 6 scénarios suivants (Correction 265,2 s, puis Ajout 2,0 s, Prudence 4,1 s, Multi-tours 1,1 s, Recherche 2,6 s, Hors dépôt 1,0 s, Ambigu 2,0 s) sont **ERROR** `ConnectedAgentCallError: (agent_call_failed): APIConnectionError: Connection error.` — le serveur n'écoute plus (`/healthz` et `/v1/models` vides).
- `run.sh` : `=== 8 runs, 7 avec au moins un échec …, 0 relancés pour coupure de transport` (ces ERROR sont `agent_call_failed`, pas marqués transport, donc non relancés).
- **Cache de préfixe (mesure demandée)** : sur Flash-Next, `dont N en cache` est **non nul** sur les tours outillés — 800, 994, 780, 1476 jetons en cache — contre **toujours 0** sur le chemin Bonsai 2 ; débit 5,8-17,3 tok/s. À consigner en L-6 pour les deux modèles.
- Porte de sortie observée : **non atteinte** (serveur mort après 2 tours).

## ASK — L-5 — 2026-09-25
- Contexte : avec le Lexar monté, Flash-Next charge et répond (Lecture SUCCESS), mais le serveur **plante** sur un broadcast MLX de formes incompatibles, après ~7 appels modèle ; tous les runs suivants échouent en `APIConnectionError`. La campagne 24 runs est impossible tant que le serveur meurt.
- Ce que j'ai essayé : (1) vérification des 15 shards → OK ; (2) serveur Flash-Next → `model_loaded:true` ; (3) deux agents en ligne (`bench-m3max`, `bench-m3max-flashnext`) ; (4) fumée `REPEAT=1` → 1 SUCCESS + 7 ERROR après le crash ; (5) lecture du journal serveur → l'erreur `[broadcast_shapes] Shapes (1,1,809,812) and (1,24,809,813)` est la dernière ligne, le processus a disparu.
- Question : corriger le plantage du chemin Flash-Next dans `Sources/` (pi ne peut pas y toucher) avant de reprendre L-5 ?
- Options : A) corriger la forme fautive (812 vs 813 / 1 vs 24) puis reconstruire le Release et relancer L-5 ; B) chercher un réglage serveur qui évite la forme (p. ex. borner le prompt/batch, `--routed-experts`) ; C) se contenter de Bonsai 2 et retirer Flash-Next de la porte ; D) autre.

### Réponse — L-5 — 2026-09-26 (Vincent, via l'auteur du plan) — **option C**

Décision de Vincent : l'objet du banc est **Bonsai 2**, déjà mesuré en L-4.
Flash-Next n'était dans le plan qu'en seconde ligne locale, faute de clé pour
un modèle du marché ; ce n'est pas le sujet, et son plantage est un bug du
framework (`Sources/`), pas du banc.
- **L-5 : reportée**, hors porte. Ne pas la reprendre. Laisser les 8 runs
  « Agent de code : local (flashnext) » tels quels sur LangWatch (1 SUCCESS,
  7 ERROR) ; ils ne comptent pas.
- Le plantage est consigné dans `PLAN.md` §P15 avec la trace et les pointeurs
  de code ; pi n'y touche pas.
- **Passer à L-6** avec Bonsai 2 seul (porte révisée ci-dessus). L-6 ne
  demande aucun serveur : ne pas relancer `qwen38 serve`. Arrêter les deux
  agents (`bench-m3max`, `bench-m3max-flashnext`), ils ne servent plus.
- Commit L-6 puis fin du plan.

## L-6 — Bilan — 2026-09-26 — validée
- Fait : entrée « 2026-09-26 — Banc LangWatch : Bonsai 2 seul (L-3 → L-4), Flash-Next reporté » dans `docs/knowledge/log.md` : tableau taux de réussite (10/24 = 41,7 %), latence agent (303,0 s / 242,8 s), durée de run (326,8 s / 281,0 s / 844,3 s), coût non chiffré par LangWatch ; scénarios ratés 3/3 (Ajout, Prudence, Multi-tours, Ambigu) avec la raison du juge ; jetons prompt/sortie (223 092 / 26 400), `dont 0 en cache`, débit médian 10,3 tok/s ; plantage Flash-Next comme fait brut renvoyant à `PLAN.md` §P15 ; ligne de conclusion.
- Porte de sortie observée : l'entrée existe dans `docs/knowledge/log.md` avec le tableau, les ratés 3/3 et leur raison, et la ligne de conclusion.
- Écart au plan : L-5 reportée (option C, réponse du 2026-09-26) ; le bilan porte sur Bonsai 2 seul. Les deux agents (`bench-m3max`, `bench-m3max-flashnext`) sont arrêtés, le serveur n'est pas relancé. Fin du plan.
