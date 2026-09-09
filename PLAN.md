# qwen38-mlx-swift — Plan d'implémentation

> **Statut : rév. 4 du 2026-09-06 — Jalon 3 réanalysé et réorienté (§6-§7 réécrits, §0.1 et §11 amendés).** Rév. 3 (cadrage acté par Vincent le 2026-08-28) reste valable pour §1-§5. Document de référence **local** (pas d'artifact). Le journal d'exécution (REQUEST/RÉPONSE/Étapes V1…V54) suit le §11.
> Roadmap : **Jalon 1 (27B + GUI) → Jalon 2 (serveur d'inférence LAN) → Jalon 3 (portage Flash-Next)**. À chaque jalon, Vincent teste lui-même via la GUI de bench.
> Cible matérielle : MacBook Pro M3 Max, 96 Go RAM unifiée, macOS 15+, disque externe Lexar (`/Volumes/Lexar/models`, **~384 Go libres** après nettoyage).

---

## 0. Mode d'exécution — ce plan sera exécuté par un agent (modèle plus petit)

Le « réalisateur » n'est pas une équipe humaine mais un agent IA de capacité inférieure au rédacteur de ce plan. Conséquences sur la façon d'écrire et d'exécuter :

- **Aucune décision d'architecture n'est laissée à l'exécutant.** Chaque tâche nomme le fichier source à copier/adapter, le fichier cible, et le critère de sortie vérifiable par une commande. En cas d'ambiguïté : STOP et question à Vincent, jamais d'improvisation.
- **Une tâche = un commit = un critère vérifiable** (test qui passe, parité sous tolérance, build vert). Interdiction d'enchaîner deux étapes sans avoir validé la première.
- **Les tests de parité sont le harnais de sécurité** de l'exécutant : ils sont écrits/dumpés AVANT le portage de chaque module (§6.1), pas après. Un module sans dump de référence Python ne se code pas.
- **La checklist de pièges (§6.3) est un contrat**, pas une lecture recommandée : chaque revue de PR interne coche les 10 points explicitement.
- **Le profiler est l'outil de chasse aux bugs par défaut** : toute anomalie de perf ou de mémoire se diagnostique d'abord avec `swift-mlx-profiler` (rapport console + Chrome Trace), pas en ajoutant des `print`.
- **Interdictions absolues** (car les erreurs sont silencieuses) : `eval(model.parameters())` global sur Flash-Next (matérialiserait la table n-gram) ; `swift build` (metallib) ; `branch:` dans Package.swift ; suppression de fichiers sur le Lexar ; publication (push, HF upload) sans validation de Vincent ; exposer le serveur au-delà du LAN.
- L'ordre des étapes du Jalon 3 (A→H) est **strict** : chaque étape ne dépend que des précédentes.
- **Fin de chaque jalon = démo à Vincent via la GUI** : le jalon n'est pas terminé tant que Vincent n'a pas testé.

### 0.1 Points d'arrêt obligatoires (GATES)

Les questions encore ouvertes ne sont PAS laissées à l'appréciation de l'exécutant : chacune est ancrée à un point précis du plan. Quand l'exécutant atteint une ligne marquée **⛔ GATE**, il s'arrête et pose la question à Vincent, même si une réponse lui semble évidente. Il ne franchit jamais une GATE de sa propre initiative.

| GATE | Où | Question à poser |
|---|---|---|
| G-0 | Avant le premier commit | « Le plan rév. 3 est-il validé ? Des changements de dernière minute ? » |
| G-1 | Fin Jalon 1 (démo) | Démo GUI + résultats de parité ; « on passe au Jalon 2 ? » |
| G-2 | Tâche 2.6 (config serveur) | « api-key obligatoire ou optionnelle par défaut ? » (proposé : optionnelle) |
| G-3 | Fin Jalon 2 (démo) | Démo depuis l'autre machine ; « on passe au Jalon 3 ? » |
| G-4 | Début §7 (avant tout téléchargement Flash-Next) | « Quant : option A (conversion existante — laquelle) ou option B (outil streaming) ? » |
| G-5 | ~~Avant l'étape G~~ — **requalifiée rév. 4** : l'étape G a été réalisée sans la poser (MTP opt-in, désactivé par défaut). La question devient « P-MTP maintenu ou gelé ? », posée à G-8 | voir §6.1 point 2 |
| G-6 | Avant tout `git push` / upload HF / publication | Toujours — jamais de publication sans accord explicite (rappel : licence `qwen-community-1.0` à vérifier avant un upload de quant) |
| G-7 | Toute déviation du plan (dépendance à ajouter, étape à réordonner, API à changer) | Décrire la déviation proposée et attendre l'accord |
| G-8 | **Fin de l'étape H (rév. 4)** — démo Flash-Next GUI + serveur | « H validé ? G-1/G-3 actées rétroactivement ? Suite : P (débit) ou §7 (quant maison, G-4bis) ? P-MTP maintenu ? » |
| G-4bis | Posée à G-8 (§7 rév. 4) | « Lancer la quant maison (option B, ≤ 70 Go sur SSD interne) au vu des mesures P1 ? » |

Entre deux GATES, l'exécutant ne pose PAS de question pour des choix déjà tranchés dans ce document — il applique le plan. S'il rencontre une ambiguïté réelle non couverte, c'est une GATE G-7.

---

## 1. Objectif et périmètre

Porter/intégrer la famille **Qwen3.8** (août 2026) pour l'inférence locale MLX Swift sur Apple Silicon, avec un **serveur d'inférence intégré** accessible depuis les autres machines du réseau local, dans la lignée des projets existants (`gemma-4-swift-mlx`, `h3-swift-mlx`, `flux-2-swift-mlx`).

| Modèle | Type | Licence | Verdict M3 Max 96 Go |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` | VLM dense 27B, hybride Gated DeltaNet | Apache-2.0 | ✅ **Jalon 1** — 4-bit = 16 Go |
| `Qwen/Qwen3.8-Flash-Next` | VLM MoE **125B total / 6B actifs** + 51B n-gram + 4B MTP (`qwen4_exp`, preview archi Qwen4) | qwen-community-1.0 | ⚠️ **Jalon 3** — quant mixte ~65-75 Go |
| `Qwen/Qwen3.8-2.4T-A95B` | MoE texte 2.4T | qwen3.8-max | ❌ Hors périmètre |

**Fait structurant** : le 27B (`model_type: qwen3_5`) est supporté par `mlx-swift-lm` 3.31.4 pour l'inférence standard (`MLXVLM/Models/Qwen35.swift`, kernel Metal `GatedDelta.swift`). Le drafter MTP Qwen et le rewind hybride arrivent sur `main` après #351, mais ne sont pas dans la release 3.31.4 ; le Jalon 1 doit donc intégrer une révision upstream épinglée et le checkpoint drafter séparé. En revanche **`qwen4_exp` (Flash-Next) n'existe dans aucune lib Swift** — seule référence : `mlx-vlm` Python (`mlx_vlm/models/qwen4_exp/`). C'est la valeur ajoutée réelle du projet.

### 1.1 Position upstream (état vérifié le 2026-08-28) — stratégie VALIDÉE par Vincent

- **PR ml-explore/mlx-swift-lm#545 (ouverte, active)** : « Add Qwen 3.8 support for text, VLM, and MTP speculative decoding » — le 27B est en cours de finalisation upstream par un tiers. **Ne pas dupliquer** ; tester la branche, bugs remontés en PR (par Vincent).
- **PR #510 (ouverte)** : « Make MambaCache rewindable » — exactement le verrou de l'étape G (rollback GDN pour la vérif MTP). Si elle merge, l'étape G devient triviale.
- **Aucune PR/issue `qwen4_exp`/Flash-Next dans mlx-swift-lm** : le terrain Swift est libre.
- `mlx-vlm` Python optimise activement `qwen4_exp` (PR #2037 mergée, équipe Qwen) — la référence de portage est vivante.

**Avancement local M1 (2026-08-29)** : le rollback GDN upstream a été corrigé
dans le checkout épinglé et validé par `xcodebuild`; la parité greedy est
identique sur le premier tour image. Les tours suivants contenant l'image
restent volontairement en fallback standard tant que le cache M-RoPE complet
n'est pas transporté. La couture technique M2 est détaillée dans
`docs/knowledge/investigations/mtp-m2-persistent-conversation.md`.

**Stratégie retenue (validée)** : *portage local d'abord, upstream ensuite*. Développer `qwen4_exp` dans ce repo, **écrit dès le premier commit dans les conventions upstream** (protocoles `LanguageModel`, caches MLXLMCommon, `MTPDrafterModelFactory`, style de `Qwen35.swift`, tests façon `Qwen35GDNDecodeBitwiseTests`) pour que la PR upstream finale soit un déplacement de fichiers. La PR upstream sera pilotée par Vincent, pas par l'agent. Le serveur, la GUI, le downloader et l'outil de quant restent dans ce repo (pas leur place upstream).

---

## 2. Rappel d'architecture (ce que l'exécutant doit connaître)

### 2.1 Commun aux deux modèles
- **Hybride 3:1** : motif `3 × (Gated DeltaNet → FFN)` puis `1 × (attention pleine → FFN)` (`full_attention_interval: 4`).
- **Gated DeltaNet** : 48 têtes V / 16 têtes QK, head_dim 128, conv1d causale kernel 4, **état récurrent float32**. Projections d'entrée **séparées en 4** (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`). β = `sigmoid(b)`, décroissance `g = -exp(A_log)·softplus(a + dt_bias)` en float32, q/k L2-normalisés dans le kernel.
- **Attention pleine « gated »** : gate sigmoid en sortie, q/k RMSNorm, head_dim 256, **RoPE partiel dim 64** (`partial_rotary_factor 0.25`), **MRoPE entrelacé** (`mrope_section [11,11,10]`, θ 1e7).
- **Vision** : ViT 27 blocs, hidden 1152, patch 16, spatial_merge 2, temporal_patch 2 — même encodeur pour les deux (seul `out_hidden_size` change : 5120 vs 2560).
- **Vocab 248 320**, ChatML, thinking par défaut (`<think>`) ; kwargs de template `enable_thinking`, `preserve_thinking` (défaut true), `reasoning_effort` ∈ {xhigh, medium, low}. Tool calling style Hermes (`<tool_call>`).
- **MTP 1 couche** (speculative decoding), sans embeddings dédiés. Contexte natif **262 144** (1M via YaRN ×4).
- Sampling : thinking `temp 1.0 / top_p 0.95 / top_k 20` ; instruct `temp 0.7 / top_p 0.80 / top_k 20 / presence 1.5`. EOS `[248046, 248044]`.

### 2.2 Spécifique Flash-Next (`qwen4_exp`) — les 5 briques à porter
1. **MoE** : 512 experts (dim 640), 10 routés + 1 partagé, 48 couches, hidden 2560.
2. **QSA (Qwen Sparse Attention)** : micro-blocs de 4 tokens, **indexer** MQA 4 têtes Q + 1 tête K (head_dim 128), budget 512 blocs = 2048 tokens. Attention principale 24 Q / 2 KV, head_dim 256.
3. **N-gram Embedding** : table hashée ~20M d'entrées × 8 têtes (vocabs premiers par tête), hash multiplicatif-XOR conscient des segments EOS, **51B paramètres**, injectée couche 2 via PLE + conv kernel 4, shardée ×128.
4. **Hyper-connections** : 4 flux résiduels, bottleneck rang 320, gate de lecture data-dépendant + gate scalaire d'écriture. Hidden tuilé ×4 à l'embedding ; le MTP voit `[B, L, 4·hidden]`.
5. **MTP hybride** (`mtp.hybrid: true`, 1 couche full_attention).

Référence ligne-à-ligne : `mlx_vlm/models/qwen4_exp/language.py` (1776 lignes) : `Qwen4ExpQSAIndexer`, `QSAKVCache`, `Qwen4ExpNGramEmbedding` (le hash exact), `Qwen4ExpGatedResidual`, vérificateur spéculatif exact.

### 2.3 Noms de tenseurs (27B, pour le sanitizer)
- 48× `model.language_model.layers.N.linear_attn.{A_log, dt_bias, conv1d.weight, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, norm, out_proj}`
- 16× `model.language_model.layers.N.self_attn.{q,k,v,o}_proj` + `{q,k}_norm`
- `mtp.{fc, norm, pre_fc_norm_embedding, pre_fc_norm_hidden, layers.0.*}`
- `model.visual.blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2, norm1, norm2}` + `patch_embed.proj`, `pos_embed`, `merger.*`
- Pièges connus (mlx-lm) : norms zéro-centrées dans les checkpoints récents (+1.0 pour les anciens) ; conv1d sanitizée par `moveaxis(2,1)` ; `A_log` hors des casts bf16.

---

## 3. Budget mémoire M3 Max 96 Go

Limite wired GPU par défaut ≈ 72 Go, ajustable : `sudo sysctl iogpu.wired_limit_mb=85000` (à documenter dans le README, pas dans le code).

| Config | Poids | KV/état @ 32K ctx | Verdict |
|---|---|---|---|
| 27B 4-bit (`mlx-community/Qwen3.8-27B-4bit`) | 16,1 Go | ~2 Go | ✅ Large |
| 27B 8-bit | ~29 Go | ~2 Go | ✅ Confortable |
| 27B bf16 | ~54 Go | ~2 Go | ✅ Possible, sans intérêt |
| Flash-Next mixed ~2-bit (Sawfwair) | 73,1 Go | faible | ⚠️ Passe, qualité à valider |
| Flash-Next mixed 4/8-bit (orcarouter) | ~88,7 Go | — | ❌ Trop juste |
| Flash-Next quant maison ~3.5-4 bpw | 65-75 Go | faible | 🎯 Cible §7 |

Atout : 6B actifs + attention bornée par QSA → débit décode Flash-Next attendu excellent, coût mémoire du long contexte quasi plat (16 couches full sur le 27B, 12 sur Flash-Next).

---

## 4. Jalon 1 — Qwen3.8-27B + GUI de bench

**Objectif** : Vincent chatte avec le 27B (texte + image) dans une petite app SwiftUI qui affiche les temps, MTP spéculatif activable, profiler branché. **3-6 jours** (assemblage de briques existantes + GUI).

### 4.1 Squelette du package
```
Sources/Qwen38Core/       // lib : modèles, pipeline, registration
Sources/Qwen38Server/     // lib : serveur HTTP (Jalon 2, cible créée vide dès maintenant)
Sources/Qwen38CLI/        // qwen38 : ArgumentParser, AsyncParsableCommand
Sources/Qwen38BenchUI/    // app SwiftUI de test/bench (executableTarget)
Tests/Qwen38Tests/
Scripts/run-tests.sh      // copie gemma-4 (deadlock xcodebuild, watchdog)
docs/knowledge/           // convention OKF : index.md, log.md, pitfalls/, benchmarks/
```
Dépendances (pinning strict — mlx-swift casse des APIs en patch release) :
```swift
.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
.package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.4.0"),
// Jalon 2 : .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0")
```
⚠️ Jamais de `branch:` sur les deps mlx. Build via `xcodebuild`, jamais `swift build`. macOS `.v15`.

### 4.2 Tâches
| # | Tâche | Source à copier / adapter |
|---|---|---|
| 1.1 | Downloader HF sans SDK (actor + URLSession, resume par fichier, vitesse fenêtre glissante) + `--models-dir` → Lexar par défaut | `gemma-4 Sources/Gemma4Swift/Download/*`, `Gemma4ModelCache.swift` ; listing via `/api/models/{id}/tree` (le downloader `Hub` stalle sur les redirects LFS) |
| 1.2 | Chargement 27B : **tester d'abord la PR #545** (checkout branche) ; valider les quants `mlx-community` 4/8-bit et `-MTP-4bit` ; dépendre de la branche en local jusqu'au merge, bugs remontés par Vincent | `loadModelContainer` + `ChatSession` ; PR mlx-swift-lm#545 |
| 1.3 | `enable_thinking` / `reasoning_effort` via `additionalContext` → **bypass ChatSession** (TokenIterator + `generateTask`) | `Gemma4Pipeline.chatStreamBypassingSession` |
| 1.4 | Presets sampling (§2.1) + filtre de flux `<think>` (affiché à part dans la GUI) | `Gemma4TokenFilter` |
| 1.5 | Image : smart_resize + injection — vérifier d'abord ce que `MLXVLM/Qwen35.swift` fait déjà ; ne réécrire que si nécessaire | `flux-2 Qwen35ImageProcessor.swift`, vectorisé façon `Gemma4UnifiedImageProcessor` |
| 1.6 | MTP spéculatif stock + toggle + stats accept rate | `QwenMTP.swift` ; stats façon `Gemma4MTPPipeline.Stats` |
| 1.7 | **Intégration profiler systématique** : `LLMMetrics` (tokenization/prefill/génération, tok/s, TTFT) autour de chaque génération ; `--trace` (CLI) et bouton « Exporter trace » (GUI) → Chrome Trace pour Perfetto | `swift-mlx-profiler LLMProfiling.swift`, `ChromeTraceExporter.swift` ; convention `@_exported import MLXProfiler` comme h3 |
| 1.8 | KV quant natif : `GenerateParameters(kvBits: 4, kvGroupSize: 64, quantizedKVStart: 5000)` — pas de TurboQuant maison | verdict gemma-4 |
| 1.9 | **GUI v1 `Qwen38BenchUI`** — voir §4.3 | `gemma-4 Sources/Gemma4BenchUI` (structure onglets) |
| 1.10 | `BENCHMARKS.md` vs mlx-vlm Python (protocole bench-guard : GPU froid, A/B/B/A, `LC_ALL=C`) | `gemma-4 BENCHMARKS.md`, `h3 scripts/bench-guard.sh` |

### 4.3 GUI v1 — spécification
Une seule app qui grandit à chaque jalon (pas trois apps). Modèle : `gemma4-bench-ui`.
- **Onglet Chat** : sélecteur de modèle (variantes présentes sur le disque, taille affichée), zone de chat streaming, drop d'image, toggles thinking / reasoning_effort / MTP, presets sampling.
- **Panneau Mesures (toujours visible)** : TTFT, prefill tok/s, decode tok/s, tokens prompt/générés, accept rate MTP, mémoire MLX (active/peak via `MLX.GPU`), durée totale. Source unique : `LLMMetrics` du profiler — la GUI ne re-mesure rien elle-même.
- **Onglet Historique** : table des runs (modèle, quant, params, métriques) pour comparer A/B à l'œil ; export CSV.
- **Bouton « Exporter trace »** : Chrome Trace du dernier run (chasse aux bugs de perf dans Perfetto).

### 4.4 Critères d'acceptation — ⛔ GATE G-1 (démo Vincent)
- Sortie greedy **identique token-à-token** à `mlx_vlm.generate` Python sur 5 prompts fixes (texte + 1 image), même checkpoint 4-bit.
- Template vérifié **id-à-id** contre le rendu HF de `chat_template.jinja` (pièges swift-jinja : réparer sur les ids, pas les strings).
- Débit décode ≥ 90 % de mlx-vlm Python ; accept rate MTP affiché dans la GUI.
- **Vincent teste** : chat + image dans la GUI, timings cohérents avec le CLI, export trace ouvrable dans Perfetto.

---

## 5. Jalon 2 — Serveur d'inférence LAN intégré

**Objectif** : un autre ordinateur du réseau local se connecte au M3 Max et utilise le 27B via une **API compatible OpenAI**. **3-5 jours** (le modèle et le pipeline existent déjà ; c'est du transport).

### 5.1 Choix techniques (décidés, pas à rediscuter par l'exécutant)
- **Framework HTTP : Hummingbird 2** (`hummingbird-project/hummingbird`, from: "2.0.0") — léger, Swift concurrency natif, SSE simple. Si blocage réel documenté : repli Vapor, après accord de Vincent.
- **API : compatible OpenAI** — n'importe quel client existant (SDK openai, Open WebUI, scripts curl) marche sans code custom :
  - `POST /v1/chat/completions` (stream SSE `data: {...}` + `data: [DONE]`, et non-stream) ; vision via `image_url` en `data:` base64 ; paramètres `temperature/top_p/max_completion_tokens` mappés sur `GenerateParameters` (`max_tokens` accepté comme alias historique) ; extensions maison sous `"extra": {}` (reasoning_effort, enable_thinking, mtp).
  - `GET /v1/models` : variantes présentes sur le disque.
  - `GET /healthz` ; `GET /metrics` : JSON des métriques `LLMMetrics` par requête (l'onglet Serveur de la GUI le consomme).
- **Concurrence** : UN modèle résident dans un `actor` (le `ModelContainer`), requêtes **sérialisées en FIFO** (MLX = un seul GPU ; le continuous batching est explicitement hors périmètre v1). File d'attente visible dans `/metrics`.
- **Réseau** : bind `0.0.0.0`, port par défaut `8848`, **token Bearer optionnel** (`--api-key`), log des IP clientes. Jamais exposé au-delà du LAN (pas d'UPnP, rien). Annonce Bonjour `_qwen38._tcp` = nice-to-have, pas bloquant.
- **Stateless v1** : pas de cache de prompt inter-requêtes (le `preserve_thinking` du template limite déjà la recomputation côté client qui renvoie l'historique). Prompt-cache persistant = amélioration ultérieure (attention issue upstream #443 : `savePromptCache` perd l'état M-RoPE des VLM).
- Pendant le serve : `Memory.cacheLimit` fixé, et recommander le skill `mac-awake` dans le README pour les longues sessions.

### 5.1.1 Contrat de sélection dynamique des modèles — ajouté 2026-08-29

- `modelsDirectory` est le catalogue autorisé du serveur. Chaque sous-dossier
  dont `config.json` annonce `model_type: qwen3_5` devient un identifiant
  `model` ; aucun chemin arbitraire fourni par le client n'est accepté.
- Une requête sans `model` réutilise le modèle résident courant. Si plusieurs
  modèles sont catalogués et qu'aucun n'est résident, `model` est obligatoire.
- Lorsqu'un identifiant change, la file FIFO prend le verrou avant la
  transition : décharger le `ModelContainer` et le drafter, purger la mémoire
  MLX, charger le nouveau modèle et seulement ensuite lancer le préfill. Le
  premier appel d'une variante porte donc son coût de chargement ; les appels
  suivants restent chauds.
- `/v1/models` publie tous les identifiants valides et marque celui qui est
  résident avec `loaded: true`. `/metrics` et l'onglet Serveur exposent aussi
  le modèle associé à chaque session.
- Champs standards acceptés : `model`, `messages`, `stream`, `temperature`,
  `top_p`, `max_completion_tokens` ; `max_tokens` reste accepté comme alias
  historique. Thinking : `reasoning.effort` et `reasoning_effort` sont des
  alias de compatibilité ; `enable_thinking` et `extra.mtp` restent les
  extensions Qwen explicites. Ces paramètres ne sont pas tous universels dans
  l'ancien endpoint Chat Completions et doivent être documentés comme tels.

### 5.2 Tâches
| # | Tâche | Critère |
|---|---|---|
| 2.1 | Cible `Qwen38Server` : routes healthz/models/metrics | `curl localhost:8848/healthz` = 200 |
| 2.2 | `/v1/chat/completions` non-stream, texte seul | réponse JSON conforme schéma OpenAI, validée avec le SDK Python `openai` pointé sur `base_url` |
| 2.3 | Streaming SSE + arrêt propre sur déconnexion client (annulation de la génération, pas de fuite de tâche) | `curl -N` streame ; Ctrl-C client → génération annulée, loggée |
| 2.4 | File FIFO multi-clients | 2 requêtes simultanées : la 2ᵉ attend, aucune corruption ; test XCTest |
| 2.5 | Vision (image_url base64) | image depuis le client → même réponse que la GUI locale |
| 2.6 | ⛔ **GATE G-2** (api-key ?) puis `qwen38 serve --model ... --port --api-key` + intégration GUI (§5.3) | démarrage/arrêt depuis CLI et GUI |
| 2.7 | Métriques par requête via `LLMMetrics` → `/metrics` + log | tok/s visibles pour chaque requête |

### 5.2.1 Implémentation locale en cours — 2026-08-29

- Dépendance Hummingbird 2 ajoutée dans `Package.swift`, résolue par Xcode en 2.26.0.
- `Qwen38Server` implémente déjà le bind LAN `0.0.0.0`, l'arrêt par annulation du service, la file FIFO, l'authentification Bearer optionnelle, les routes de santé/métriques/modèles, le JSON non-stream et le SSE.
- Le serveur appelle `Qwen38Runtime.generateStateless(...)` avec l'historique complet `messages`, afin de ne pas partager le cache KV de la conversation GUI avec les clients LAN.
- La GUI contient l'onglet **Serveur** : port, clé optionnelle, start/stop, URL locale/LAN, compteurs FIFO et journal des sessions actualisé toutes les 400 ms avec dernier fragment reçu, tokens, TTFT et tok/s.
- Vérification effectuée : `xcodebuild` GUI et CLI verts ; `/healthz` et `/v1/models` validés sur le modèle 4-bit ; une génération SSE locale de quatre tokens a retourné les événements `data: ...` puis `[DONE]`, et l'arrêt a libéré le port.
- Reste avant G-3 : tests dédiés de déconnexion/FIFO/vision, validation depuis une autre machine et mode client distant de la GUI. Aucun téléchargement supplémentaire n'est requis pour ce jalon.

### 5.3 GUI v2 — onglet Serveur
- Start/stop du serveur, port, api-key, affichage de l'URL LAN (`http://<ip-locale>:8848/v1`).
- Journal des requêtes en direct : client, modèle, tokens, TTFT, tok/s, statut file.
- **Mode client distant** : la GUI peut pointer vers une URL de serveur au lieu du modèle in-process — c'est comme ça que Vincent teste depuis son autre Mac avec la même app (et ça valide l'API au passage).

### 5.4 Critères d'acceptation — ⛔ GATE G-3 (démo Vincent)
- Depuis un **autre ordinateur du LAN** : `curl` streaming OK, SDK `openai` OK, GUI en mode client distant OK (texte + image).
- Surcoût serveur vs in-process < 5 % sur le débit décode (mesuré via `/metrics` vs onglet Chat local).
- Deux clients simultanés : sérialisation propre, pas de crash, pas de mélange de réponses.

---

## 6. Jalon 3 — Portage `qwen4_exp` (Flash-Next) — **rév. 4 du 2026-09-06**

> Cette section remplace la rév. 3 du Jalon 3 (§6-§7). Le journal d'exécution (V1…V54 plus bas dans ce fichier, et `docs/knowledge/log.md`) reste la source des mesures citées. La numérotation A→H de la rév. 3 est conservée ; seule la suite (H, P, P-MTP, §7) est réécrite. L'exécutant applique §0 (contrat d'exécution) et §0.1 (GATES) sans changement.

### 6.0 Réanalyse — état du portage vérifié dans le code le 2026-09-06

| Étape rév. 3 | État | Preuve |
|---|---|---|
| A Config + sanitizer | ✅ | `Qwen4ExpConfiguration.swift`, `Qwen4ExpWeightSanitizer.swift`, tests « contrat de configuration », « sanitizer » |
| B Gated DeltaNet | ✅ | `Qwen4ExpGatedDeltaNet.swift`, parité publique GDN (D3t-D3u) |
| C MoE | ✅ | `Qwen4ExpSparseMoE.swift` ; la divergence 4-bit inter-runtime est qualifiée chaotique et **close** (RÉPONSE du 2026-08-31) |
| D1-D2 QSA (fallback dense + cache) | ✅ | `Qwen4ExpQSAIndexer/Attention/Mask/Cache.swift`, parité D2j. **D3 (kernel Metal) non fait et non nécessaire** : sous 2 048 tokens `makeMask` retourne `nil` (chemin dense) |
| E N-gram + PLE | ✅ | `Qwen4ExpPLE.swift`, table mmap lazy + LRU 4 096 lignes, parité `delta=0` (V40-V46) |
| F Hyper-connections | ✅ | `Qwen4ExpHyperConnection.swift`, parité single-layer et multi-couches |
| G MTP hybride | ⚠️ fait **sans poser G-5**, opt-in, correct (3/5 acceptés, sortie identique au greedy) mais **1 618 s / 8 tokens** (V54) | `Qwen4ExpFlashMTP.swift`, `Qwen4ExpFlashMTPGenerator.swift` |
| H Vision + E2E + GUI/serveur | ⚠️ vision, fusion et E2E **validés en CLI seulement** (`flash-generate-probe`, `QUALITY_GATE=PASS` V53) ; **rien** côté catalogue, runtime, GUI, serveur | `Qwen38ModelValidator.validate` refuse tout sauf `qwen3_5` |

Ce qui manque pour qu'un utilisateur voie Flash-Next :

1. **Catalogue** : `Qwen38ModelValidator` et `Qwen38ModelCatalog.discover` rejettent `qwen4_exp` ; la GUI (`selectVariant`) code en dur trois chemins 27B.
2. **Générateur utilisable** : `Qwen4ExpGreedyGenerator` est greedy uniquement, non streamé (le texte n'est rendu qu'à la fin), sans presets de sampling (§2.1).
3. **Adaptateur runtime** : `Qwen38Runtime` ne connaît que `ChatSession`/`ModelContainer` upstream ; aucun chemin Flash-Next dans `generate` ni `generateStateless`.
4. **Qualification** : un seul prompt de référence validé ; `</think>` jamais observé fermé.
5. **Débit** : greedy résident 28 s / 8 tokens (≈ 0,3 tok/s) contre ~10 tok/s pour le 27B 4-bit, alors que Flash-Next n'a que 6B actifs. Un facteur > 30 d'overhead d'implémentation reste à trouver (chantier P).
6. **Bench** : aucune ligne Flash-Next dans `BENCHMARKS.md`.

**Contraintes matérielles actualisées** (elles remplacent §3 pour Flash-Next) :

- Checkpoint Vontra 4-bit : **113 Go** sur le Lexar, dont 32 Go de table n-gram et 0,9 Go de vision. SSD interne : **88 Go libres** → **la copie du checkpoint sur le SSD interne est impossible ; ne pas l'essayer.**
- Résidence partielle (tout sauf n-gram ; n-gram en mmap + LRU) : **pic MLX 79-82 Go**. Incompatible avec un 27B chargé en même temps (17-54 Go) → un seul modèle résident par process (contrat §5.1.1, déjà en place côté serveur).
- Chargement one-shot depuis le Lexar : **~95-105 s**. C'est le TTFT du premier tour d'un process neuf. Dans un process résident (GUI, serveur), ce coût est payé **une seule fois** ; il ne bloque donc pas l'étape H. Ce qui reste lent **par token** relève du chantier P.
- Le Bash tool de Claude Code dispose du device Metal (mémoire `project-flashnext-terminal-metal-access`) : exécuter les probes directement, en tâche de fond, avec des timeouts longs (un run résident 8 tokens ≈ 2-3 min après correctif V54, mais 15-25 min si l'I/O Lexar est contendue).
- **Avant chaque run résident**, vérifier : Lexar monté (`ls /Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/config.json`), pas d'autre charge GPU lourde (`top -l 1 | grep PhysMem`, `ps aux | grep -c '[c]laude'`). Deux runs ont été tués pour mémoire basse à cause de charges concurrentes (V53-V54).

### 6.1 Réorientation — décisions rév. 4

1. **H avant P.** Intégrer Flash-Next dans le catalogue, la GUI et le serveur avec le chemin greedy/sampling résident **tel quel, même lent**. Raison : la lenteur se décompose en un chargement one-shot (payé une fois par process) et un débit par token à optimiser séparément. Vincent peut donc tester la qualité en conditions réelles dès la fin de H, et la qualification élargie devient un test GUI au lieu de runs CLI de 20 minutes.
2. **MTP Flash-Next = chantier séparé (P-MTP), désactivé par défaut.** GATE G-5 est requalifiée : la question n'est plus « MTP requis v1 ? » (réponse de fait : non, le générateur existe en opt-in et le plan acceptait déjà de livrer sans) mais « le goulot de ~880 s hors profiler (V54) vaut-il l'investissement avant la quant maison ? ». Elle est posée à G-8.
3. **Le critère « sortie greedy identique à mlx-vlm Python » (§6.4 rév. 3) est abandonné.** (a) La chaîne 4-bit est chaotique inter-runtime (décision du 2026-08-31, close). (b) mlx-vlm 0.6.17 applique lui-même la mauvaise convention de normes sur ce checkpoint (mémoire `project-flashnext-norm-convention`) : sa sortie native est fausse, ce n'est plus une référence. Remplacé par : sanité absolue teacher-forced (Q-B, V32) + qualification qualitative sur 5 prompts (H6).
4. **§7 (quant maison) reste ouvert et devient une décision post-démo (G-4bis).** Une quant ≤ 70 Go tiendrait sur le SSD interne et supprimerait le goulot Lexar ; la décision dépend des mesures P1.
5. **Nouvelle GATE G-8** = démo Flash-Next dans la GUI + serveur (fin de H). C'est la première démo formelle du projet ; elle vaut aussi acte de franchissement rétroactif de G-1 et G-3 si Vincent le décide.
6. **Versionnement obligatoire dès la première tâche** (H0.1) : le dossier n'est pas un dépôt git et cinq fichiers de `Vendor/mlx-swift-lm` portent des modifications locales non sauvegardées nulle part.

### 6.2 Étapes restantes (ordre strict)

Conventions : une tâche = un commit = un critère vérifiable. Build : `Scripts/build.sh` (xcodebuild, jamais `swift build`). Tests : `Scripts/run-tests.sh`. Binaire : `./.xcodebuild/Build/Products/Debug/qwen38`. Checkpoint : `/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, noté `$FLASH` ci-dessous. Prompt de référence : « Explique en français qui est le président de la Chine et quel est son rôle. » Image de référence : `/Users/vincent/Downloads/licensed-image-2.jpeg`. Sortie greedy attendue sur le prompt de référence (8 tokens, sans thinking) : `"Le président de la Chine est Xi Jinping"`. Les probes (`flash-generate-probe`, `flash-chat-probe`), le bench `flash-layer-bench`, les runs H6 et la démo G-8 utilisent le binaire **Release** (`Scripts/build-release.sh`, wrapper `QWEN38_CONFIGURATION=Release Scripts/build.sh` ; binaire `.xcodebuild/Build/Products/Release/qwen38`) ; Debug (`Scripts/build.sh` par défaut) reste réservé aux tests (`Scripts/run-tests.sh`).

**H0 — Préalables (0,5 j)**

| # | Tâche | Critère |
|---|---|---|
| H0.1 | `git init` à la racine. Compléter `.gitignore` : ajouter `.xcodebuild-*/`, `.build/`, `*.trace.json`, `results/*.trace.json`, `Vendor/mlx-swift-lm/` y est déjà. Le checkout `Vendor/mlx-swift-lm` (git séparé, `1a562aa`) a 5 fichiers modifiés (`GatedDelta.swift`, `KVCache.swift`, `MTPDrafterModel.swift`, `SwitchLayers.swift`, `Qwen35MTP.swift`) : `git -C Vendor/mlx-swift-lm diff > Vendor/mlx-swift-lm-local.patch`, committer le patch dans le dépôt principal, avec un `Vendor/README.md` (3 lignes : révision de base, commande d'application). Premier commit « rév. 4 baseline ». | `git log --oneline` montre 1 commit ; `git -C Vendor/mlx-swift-lm stash && git -C Vendor/mlx-swift-lm apply --check ../mlx-swift-lm-local.patch && git -C Vendor/mlx-swift-lm stash pop` passe |
| H0.2 | `Scripts/run-tests.sh` : suite verte (54 tests `@Test` attendus). | `** TEST SUCCEEDED **` |
| H0.3 | Baseline 27B avant toute modification du runtime : 27B 4-bit, prompt de référence, greedy, 16 tokens, sans image, via la CLI ou la GUI → texte dans `results/baseline-27b-rev4.txt`. | fichier présent (non-régression H3.4) |

**H1 — Catalogue (0,5 j)** — `Sources/Qwen38Core/Qwen38ModelValidation.swift`, `Sources/Qwen38Server/Qwen38Server.swift`

| # | Tâche | Critère |
|---|---|---|
| H1.1 | Ajouter `public enum Qwen38ModelFamily: String, Sendable { case qwen35 = "qwen3_5"; case qwen4Exp = "qwen4_exp" }` et `public var family: Qwen38ModelFamily?` sur `Qwen38ModelInfo`. `validate` accepte les deux familles ; pour `qwen4Exp`, appeler en plus `try Qwen4ExpConfiguration.load(from: directory).validate()` (existe déjà) pour rejeter un checkpoint incomplet. Message d'erreur inchangé pour les autres types. | tests existants « validator accepte qwen3_5 » et « lit les dimensions » verts + 2 nouveaux : `qwen4_exp` accepté sur un `config.json` minimal (réutiliser le fixture du test « contrat de configuration Flash-Next ») ; `qwen3` refusé |
| H1.2 | `Qwen38ModelCatalog.discover` publie les deux familles. Étendre le test « Le catalogue serveur ne publie que les modèles Qwen3.8 valides » avec un dossier `qwen4_exp` factice. | test vert |
| H1.3 | `Qwen38ModelCatalog.sizeOnDisk(_ url: URL) -> Int64` : somme des tailles des `*.safetensors` (attributs fichier, aucune lecture). | test sur un dossier temporaire à 2 fichiers |

**H2 — Générateur Flash-Next streamé avec sampling (1-2 j)** — nouveau fichier `Sources/Qwen38Core/FlashNext/Qwen4ExpStreamingGenerator.swift`. Modèle : la boucle de `Qwen4ExpGreedyGenerator.generate` (préfill puis un `forward` par token). **Ne pas modifier `Qwen4ExpGreedyGenerator`** : il reste l'oracle greedy des tests.

| # | Tâche | Critère |
|---|---|---|
| H2.1 | Sampling : `GenerateParameters(temperature:topP:topK:…).sampler()` de MLXLMCommon (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift:267`, retourne `ArgMaxSampler` / `TopPSampler` / `CategoricalSampler`) appliqué à `logits[0..., -1, 0...]`. **Interdit** : réécrire un sampler. Presets §2.1 exposés comme `Qwen4ExpSamplingPreset` (`.thinking` = 1.0/0.95/20, `.instruct` = 0.7/0.80/20). | test : température 0, prompt de référence, 8 tokens ⇒ IDs identiques à `Qwen4ExpGreedyGenerator` dans le même process |
| H2.2 | Streaming : `func generate(...) -> AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>` avec `.token(Int32)` émis après chaque `forward`, puis `.finished(Qwen4ExpGenerationSummary)` (prompt tokens, TTFT, prefill, decode, `layerVisitCount`, `layerLoadTime`, mémoire active/pic, `ngramCacheStats`). Le premier `.token` part dès que le préfill a produit un ID. Matérialiser le token à chaque pas (`.item(Int32.self)`) : aucun graphe différé (piège 11). | test `maxNewTokens: 3` ⇒ 3 `.token` puis `.finished` ; `.finished.decode > 0` |
| H2.3 | Multi-tour : lire ce que fait `Qwen4ExpStreamingTextModel.forward` avec `logicalOffset` quand on l'appelle avec les seuls tokens du tour N+1 ; les caches par couche (GDN + QSA) doivent continuer, pas repartir de zéro. Si `Qwen4ExpGreedyGenerator` remet à zéro (`resetConversation`) en début de `generate`, le nouveau générateur prend `continueConversation: Bool`. Le rendu du suffixe de tour (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` + préfixe thinking) suit V51. | probe H2.4 à 2 tours : le `promptTokenCount` du tour 2 = tokens du suffixe uniquement ; réponse cohérente avec le tour 1 |
| H2.4 | CLI : commande `flash-chat-probe $FLASH --prompt … [--second-prompt …] [--image …] [--preset thinking\|instruct] [--temperature --top-p --top-k] [--thinking] [--max-new-tokens] [--trace]`, texte affiché en streaming. Extraire la préparation prompt/image de `FlashGenerateProbe` (`Sources/Qwen38CLI/Qwen38CLI.swift:480-640`) dans un type partagé `Qwen4ExpPromptBuilder` (nouveau fichier Core) que H3 réutilisera ; `flash-generate-probe` l'utilise aussi (pas de code dupliqué). | greedy 8 tokens sur le prompt de référence ⇒ sortie attendue ; `flash-generate-probe` inchangé en sortie |
| H2.5 | Annulation : si la `Task` consommatrice est annulée, la boucle s'arrête au token suivant sans forward orphelin. | test : annuler après 1 token ⇒ `layerVisitCount` ≤ préfill + 2 × 48 |

**H3 — Adaptateur runtime (2 j)** — nouveau fichier `Sources/Qwen38Core/FlashNext/Qwen38FlashNextEngine.swift` + modifications minimales de `Qwen38Runtime.swift`

| # | Tâche | Critère |
|---|---|---|
| H3.1 | `Qwen38Runtime.load(from:)` lit `validate(directory).family`. Pour `.qwen4Exp` : `unload()` du résident (27B ou Flash, `Memory.clearCache()` déjà dans `unload`), puis `Qwen38FlashNextEngine(directory:)` qui possède `Qwen4ExpStreamingTextModel(directory:, layerLoadingMode: .resident, residentEvaluationInterval: 1)`, le tokenizer, et le loader vision (`Qwen4ExpVisionCheckpointLoader.load`) chargé à la demande au premier tour image. `isLoaded`, `loadedDirectory`, `unload`, `resetConversation`, `decode` couvrent les deux familles. `mtpState` = `.fallback("Flash-Next : MTP local en chantier P-MTP")`, jamais d'exception. Injecter un protocole `Qwen38FlashNextEngineFactory` pour que les tests de dispatch n'aient pas à charger 80 Go. | test unitaire : dossier factice `qwen4_exp` ⇒ `load` prend la branche Flash via une factory mock |
| H3.2 | `generate(prompt:systemPrompt:imageURLs:options:)` et `generateStateless(messages:options:)` : si le moteur Flash est résident, prompt via `Qwen4ExpPromptBuilder` (texte : `tokenizer.applyChatTemplate(messages:tools:additionalContext: ["enable_thinking": …, "reasoning_effort": …])`, validé id-à-id en V51 ; image : rendu manuel du CLI + `Qwen4ExpMRoPE.multimodalPositionIDs`). Émettre les mêmes `Qwen38GenerationEvent` : `.chunk` après `Qwen38VisibleTokenFilter`, `.metrics` final avec un `Qwen38RunMetrics` construit depuis le summary H2.2 et `LLMMetrics` du profiler ; `acceptRate` nil, `mtpStatus` indisponible. `options.mtp` ignoré (un log, pas d'erreur). Sampling : `options.temperature/topP/topK` → H2.1. | `flash-chat-probe` et la GUI donnent le même texte greedy sur le prompt de référence |
| H3.3 | Mémoire : `Memory.cacheLimit` fixé pendant la résidence Flash (pattern netflix-void, valeur de départ 8 Go, à mesurer) ; à `unload` : `decoder.unloadResidentLayers()` + `Memory.clearCache()`. | probe dans un même process : load Flash → unload → `Memory.activeMemory` < 2 Go → load 27B 4-bit → 16 tokens sans erreur |
| H3.4 | Non-régression 27B : `Scripts/run-tests.sh` vert ; smoke 27B 4-bit greedy 16 tokens identique à `results/baseline-27b-rev4.txt` ; `mtpState` 27B toujours `.active`. | diff vide |

**H4 — GUI (1 j)** — `Sources/Qwen38BenchUI/Qwen38BenchUIApp.swift`

| # | Tâche | Critère |
|---|---|---|
| H4.1 | Remplacer les trois chemins codés en dur de `selectVariant` par la découverte du répertoire parent de `modelPath` via `Qwen38ModelCatalog.discover` (H1.2) : liste = nom, famille, taille (`sizeOnDisk`). | Flash-Next visible dans le sélecteur avec sa taille |
| H4.2 | Pendant `loadModel` d'un Flash-Next : statut « Chargement Flash-Next (résident, ~100 s depuis le Lexar) » et progression par couche (exposer un `AsyncStream<Int>` 0…48 depuis `Qwen38FlashNextEngine`). | progression visible, `loadDuration` affiché |
| H4.3 | Toggles MTP désactivés, `mtpHelp` = raison du `.fallback` ; presets sampling, thinking / reasoning_effort et drop d'image actifs. Panneau Mesures alimenté uniquement par `.metrics` (TTFT, prefill/decode tok/s, mémoire) ; la GUI ne mesure rien. | chat texte + image dans la GUI |
| H4.4 | Le bouton « Exporter trace » du dernier run Flash produit une trace contenant les phases `Flash couche N` et les compteurs `Flash n-gram cache`. | ouvrable dans Perfetto |

**H5 — Serveur (1 j)** — `Sources/Qwen38Server/Qwen38Server.swift`

| # | Tâche | Critère |
|---|---|---|
| H5.1 | `/v1/models` liste Flash-Next avec `loaded` et un champ `family`. Sélection `model` Flash-Next : la FIFO prend le verrou, décharge le résident, charge Flash (~100 s), puis préfill (contrat §5.1.1 inchangé). | `curl :8848/v1/models` montre les deux familles |
| H5.2 | `curl -N` streaming SSE avec `"model": "Qwen3.8-Flash-Next-MLX-4bit-MTP"` : texte, puis image en `data:` base64 ; `reasoning_content` séparé (parser thinking existant) ; `"mtp": true` ignoré sans erreur. | 3 requêtes vertes, `data: [DONE]` reçu |
| H5.3 | Keep-alive : le premier chunk peut arriver après ~100 s (chargement) + préfill. Émettre un commentaire SSE `: loading` toutes les 10 s pendant le chargement pour que les proxies/clients ne coupent pas à 60 s. En non-stream, documenter dans le README que le timeout client doit être ≥ 300 s. | `curl -N` ne coupe pas ; SDK `openai` (Python) reçoit la réponse en stream |
| H5.4 | Test XCTest catalogue mixte : dossier temporaire avec un `qwen3_5` et un `qwen4_exp` factices ⇒ `/v1/models` retourne les deux ; `model` inconnu ⇒ `modelNotFound`. | test vert |

**H6 — Qualification élargie (1 j de runs, en parallèle de H4-H5, via `flash-chat-probe`)** — résultats dans `results/flash-qualification-rev4.tsv` (prompt, mode, tokens, texte, TTFT, decode, pic mémoire, verdict). Greedy, résident, sans MTP, un run à la fois.

| # | Tâche | Critère |
|---|---|---|
| H6.1 | Prompt de référence, `--thinking`, 200 tokens. | `</think>` observé **fermé**, réponse visible après, en français, sans balise ChatML |
| H6.2 | 4 prompts nouveaux, 48 tokens, sans thinking : « Explique en une phrase ce qu'est la photosynthèse. » ; « Écris une fonction Swift qui inverse une chaîne. » ; « Quelle est la capitale de l'Australie et pourquoi pas Sydney ? » ; « Traduis en anglais : Le chat dort sur le canapé. » | 4/4 cohérents, sans répétition ni dégénérescence |
| H6.3 | Image de référence, 48 tokens. | nom complet « Emmanuel Macron » et son rôle |
| H6.4 | 2 tours texte (H2.3) : prompt de référence puis « Et son prédécesseur ? ». | la réponse du tour 2 nomme Hu Jintao |
| H6.5 | Garde de régression Q-B : `flash-teacher-forced-score` sur le prompt et les IDs de V32. **À rejouer après tout changement numérique dans `Qwen4Exp*`.** | hit-rate ≥ 10/28, logprob moyen ≥ −4,5 |

⛔ **GATE G-8 — démo Flash-Next à Vincent (fin de H)** : GUI (texte, image, thinking, 2 tours), serveur depuis un client LAN, tableau H6, mesures TTFT/decode/mémoire. Questions posées : (1) H est-il validé, et G-1/G-3 rétroactivement ? (2) Priorité suivante : P (débit greedy) ou §7 (quant maison, G-4bis) ? (3) P-MTP : maintenu ou gelé ?

**P — Débit greedy résident (3-5 j, après G-8)**

Constat (corrigé le 2026-09-07 par relecture de la trace V54, voir RÉPONSE « chantier P avant G-8 ») : régime établi **1,6 s par token ≈ 25 ms par couche**, CPU 98 % (un cœur), **GPU 0-5 %** ; le premier token décodé coûte 17 s de warm-up (le « 28 s / 8 tokens » mélangeait les deux). Pour 6B paramètres actifs, > 15× plus lent que le 27B dense 4-bit ; hypothèse dominante : pagination/décompression des 77 Go non wirés (H-A), à départager par P0/P1. Méthode : profiler d'abord, **une hypothèse à la fois**, chaque correctif mesuré par la même commande (`flash-chat-probe $FLASH --prompt <référence> --max-new-tokens 32 --trace …`) et consigné dans `BENCHMARKS.md`.

| # | Tâche | Critère |
|---|---|---|
| P1 | Profil par couche dans Perfetto sur un decode de 32 tokens : ventiler chaque couche en GDN / QSA / MoE (routage, gather des experts, expert partagé) / hyper-connections / PLE-n-gram / `eval`. Ajouter des sous-phases profiler dans `Qwen4ExpDecoderLayer` si elles manquent, derrière un flag pour ne pas ralentir le chemin normal. | tableau « ms par sous-bloc, médiane sur 32 tokens » dans `log.md` |
| P2 | Hypothèses à clore **dans cet ordre**, chacune par une mesure : (a) MoE : `Qwen4ExpSparseMoE` doit passer par le gather quantifié par lot (`gatherQMM` / `SwitchGLU` upstream, `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift`) et non par une boucle Swift sur les 10 experts routés ; (b) n-gram : `ngram_cache_misses` > 0 en régime établi ⇒ lectures mmap sur le Lexar par token ⇒ préchauffer les lignes du prompt et agrandir le LRU ; (c) `eval` : ne pas rebatcher (V54), mais vérifier qu'il n'y a ni plusieurs `eval` par couche ni `.item()` intermédiaires ; (d) allocations/casts par token (`asType`, `reshaped`, `concatenated` sur les 4 flux) ; (e) résidence : aucune couche rechargée du disque pendant le decode (`layerLoadTime` = 0 après le préfill). | chaque hypothèse : « confirmée / écartée + chiffre » |
| P3 | Jauge d'utilisabilité (pas un critère de succès du plan) : ≥ 5 tok/s en decode sur prompt court. Consigner TTFT hors chargement, prefill tok/s, decode tok/s, pic mémoire dans une section Flash-Next de `BENCHMARKS.md`. | ligne `BENCHMARKS.md` |

**P-MTP — Goulot du générateur MTP (2 j, seulement si G-8 le maintient)** — `Sources/Qwen38Core/FlashNext/Qwen4ExpFlashMTPGenerator.swift`, point de départ V54 (~880 s non profilés sur 1 618 s)

| # | Tâche | Critère |
|---|---|---|
| PM1 | Phases profiler explicites autour de `engine.draftBlock`, `model.snapshot()`, le forward de vérification, la boucle `targetIDs` (`.item()` par position), `restore` + replay, `engine.commit`. | trace avec 6 phases dont la somme ≈ decode total (écart < 5 %) |
| PM2 | Corriger dans l'ordre de coût mesuré : (a) `targetIDs` : un seul `argMax(axis: -1)` + `asArray(Int32.self)` au lieu d'un `.item()` par position ; (b) `greedyToken` du `DraftEngine` (ligne ~265) sans `eval` ⇒ matérialiser le token retourné ; (c) `model.snapshot()` à chaque round : mesurer `copy()` des 48 caches ; si dominant, snapshotter seulement les caches QSA (les GDN ont `GDNStateSnapshot`) ou copier paresseusement. | decode MTP ≤ 1,2 × decode greedy pour le prompt de référence, 8 tokens, ≥ 3/5 acceptés |
| PM3 | Si PM2 atteint son critère : MTP Flash-Next opt-in dans la GUI/serveur (`mtpState = .active`, toggle réactivé). Sinon : reste hors catalogue, le plan livre sans MTP (dégradé accepté en rév. 3). | décision consignée dans `log.md` |

**Statut (2026-09-09, fait)** : PM1/PM2/PM3 exécutées sur le checkpoint
3-bit `e3bit-MTP`. PM1 a résorbé la question de V54 : le decode instrumenté
n'a plus de temps fantôme (somme des 6 phases ≈ decode total, écart
≈0,01 %). PM2 (a)/(b) appliquées et mesurées (effet dans le bruit, coût
déjà <3 % du budget) ; (c)/(d) mesurées/vérifiées sans changement de code
nécessaire (snapshot 0,05 % du budget, chemin asyncEval déjà partagé avec
le greedy) ; (e) partiellement atteint (2 syncs/round, pas 1 — assumé).
PM3 : **MTP non branché** — decode MTP reste 1,5-2× plus lent que le
greedy résident (5,2-5,3s vs 8,08-10,71s pour 32 tokens, blocs 2/3/4)
malgré des IDs identiques au greedy à chaque bloc testé ; cause racine :
taux d'acceptation du drafter 1 couche trop bas sur ce prompt/checkpoint
(24 %/12 %/8 % pour blocs 2/3/4), pas un problème de bookkeeping hôte.
Détail et tableau : `docs/knowledge/log.md` 2026-09-09 « P-MTP ».

### 6.3 Checklist de pièges (contrat de revue — cocher les 14)

1. `QuantizedLinear.weight.dtype` = uint32 packé → caster les activations via `computeDType` (scales).
2. `Module.update(parameters:)` ne vérifie rien par défaut → `verify:` + assertions sur les **valeurs** ; une clé contenant « . » ne charge jamais, sans erreur.
3. Clés numériques `@ModuleInfo(key: "0")` → vrais arrays Swift.
4. MLX avale les erreurs de lecture safetensors différées → vérifier taille vs `8 + header + max(data_offsets)` au download.
5. `quantize()` saute les modules custom → conformer à `Quantizable` (pattern `ScaledLinear`/`QuantizedScaledLinear`) pour gates MoE, mix hyper-connections, projections PLE.
6. Tables RoPE/MRoPE en float32 même en bf16 ; état GDN float32.
7. SDPA : head_dim non standard → NaN sur lignes tout-masquées ; padder à 64/80/128/256.
8. Éval per-tensor dans les boucles de transformation de poids (OOM silencieux sinon — leçon LoRA h3).
9. `Scripts/run-tests.sh` obligatoire (deadlock ABBA compile/eval) ; `xcrun xctest` pour l'exit code fiable.
10. Un `LogitProcessor` custom désactive le pipelining `asyncEval` — mesurer avant d'en ajouter.
11. **Pas de batching d'`eval` sur plusieurs couches** : un graphe MLX différé de 8 couches a coûté 30-50× (V54). `residentEvaluationInterval` reste à 1.
12. **Convention des normes du checkpoint** : les normes zéro-centrées de Vontra sont stockées déjà décalées de +1 ; le loader détecte (moyenne ≈ 1) et retire le décalage (tests « loader retire / conserve »). Tout nouveau checkpoint (quant maison §7, conversion HF) doit passer ces deux tests avant tout run.
13. **Table n-gram jamais matérialisée** : mmap + LRU uniquement ; interdiction de `eval(model.parameters())` et de tout `.asArray` sur la table.
14. **Charge machine** : un run résident pèse ~80 Go ; vérifier avant chaque run qu'aucune autre charge GPU lourde ne tourne (§6.0).

### 6.4 Critères d'acceptation rév. 4 — démo G-8, puis démo finale après P

- Flash-Next sélectionnable dans la GUI et dans `/v1/models` ; chat texte + image + thinking + multi-tour ; mesures affichées ; servi sur le LAN sans modification côté client.
- Sanité absolue : Q-B (H6.5) dans les seuils. Qualification H6 : 5 prompts, image, 2 tours PASS, `</think>` fermé.
- Débit et mémoire consignés dans `BENCHMARKS.md` (TTFT hors chargement, decode tok/s, pic), comparés au 27B 4-bit — pas de cible imposée, mais l'écart est expliqué par P1.
- **Abandonné** : identité token-à-token avec mlx-vlm Python (§6.1 point 3).

---

## 7. Quantification Flash-Next — rév. 4 : décision différée à G-4bis

GATE G-4 a été levée le 2026-08-29 (option A : conversion Vontra 4-bit-MTP, 113 Go). Ce qui change en rév. 4 :

- **Motivation nouvelle pour l'option B** : le SSD interne a 88 Go libres. Une quant maison ≤ 70 Go tiendrait **en local**, supprimerait le chargement de ~100 s depuis le Lexar et les misses n-gram sur USB. Recette rév. 3 inchangée : experts 3-4 bit affine g64 ; attention/QSA/shared/gates 8 bit ; n-gram 3-4 bit ; embeddings/lm_head 4-6 bit ; **non quantifiés** : normes, `A_log`, `dt_bias`, conv1d, gates scalaires.
- **Coût** : les shards BF16 HF (~340 Go) ne tiennent pas sur le Lexar → outil streaming shard-par-shard obligatoire (télécharger → quantifier → écrire → supprimer), format `PrequantizedCheckpoint` de h3 (`Qwen4ExpPrequantized.swift` existe déjà côté lecture), éval per-tensor (piège 8). Les normes HF sont zéro-centrées **sans** décalage (piège 12) : le loader doit prendre la branche « conserve ».
- **⛔ GATE G-4bis (posée à G-8)** : lancer l'option B ? Éléments de décision : résultats P1 (si le goulot est l'I/O Lexar, B est le levier ; si c'est le calcul, B n'apporte que la mémoire), qualité H6 du 4-bit Vontra, temps disponible (estimation B : 3-4 j de dev + ~10 h de téléchargement/quant).
- Inchangé : qualité à valider par Q-B (H6.5) + prompts H6 sur le nouveau checkpoint ; licence `qwen-community-1.0` à lire avant toute publication (G-6).

---

## 8. Écosystème — intégrations transverses (tous jalons)

| Brique | Usage dans ce projet |
|---|---|
| `swift-mlx-profiler` | **Partout** : `LLMMetrics` = source unique des mesures (GUI, `/metrics`, CLI) ; Chrome Trace = outil de chasse aux bugs par défaut ; convention `@_exported import MLXProfiler` + typealias comme h3 |
| `gemma-4-swift-mlx` | Downloader, cache modèles, bypass ChatSession, TokenFilter, structure BenchUI, `run-tests.sh` |
| `h3-swift-mlx` | `PrequantizedCheckpoint`, presets quant, `ParityCommand` + `parity_reference.py`, `bench-guard.sh`, `docs/knowledge/` (convention OKF à répliquer) |
| `flux-2-swift-mlx` | ViT Qwen + MRoPE + smart_resize, lm_head tied quantifié 248K, pitfalls docs |
| Skill `mac-awake` | Sessions longues (serve, quant streaming) — à mentionner dans le README |
| `netflix-void-swift-mlx` | `Memory.cacheLimit` par phase, presets RAM |

---

## 9. Stockage (Lexar)

État après nettoyage (2026-08-28) : **384 Go libres**. Contrainte levée.
- Jalon 1 : 27B 4-bit (16 Go) + `-MTP-4bit` + 8-bit (29 Go).
- Jalon 3 : deux conversions Flash-Next comparées (~150 Go) + quant maison — tout tient.
- ⚠️ `MiniMax-H3-turbo4/8` sont cassés (symlinks vers `MiniMax-H3/` supprimé) — hors périmètre, ne pas y toucher.

---

## 10. Risques

| Risque | Impact | Mitigation |
|---|---|---|
| Perf QSA sans kernel Metal fusionné | Débit long-contexte décevant | Fallback dense correct d'abord ; kernel en D3 ; sous 2K tokens le problème n'existe pas |
| RAM n-gram (51B) si le lazy-mmap déçoit | OOM / swap | Mesurer tôt (étape E) au profiler ; plan B : quant 2-3 bit + résidence partielle |
| Rollback d'état GDN pour la vérif MTP | MTP inutilisable sur Flash-Next | Suivre PR upstream #510 ; livrer sans MTP d'abord |
| PR #545 upstream pas mergée à temps | Dépendance branche | Dépendance locale temporaire ; le reste du Jalon 1 n'en dépend pas |
| SSE/annulation mal gérées côté serveur | Fuites de tâches, GPU occupé pour un client parti | Tâche 2.3 dédiée + test explicite de déconnexion |
| Licence `qwen-community-1.0` | Redistribution des quants ? | À lire avant toute publication HF |
| Quant 2-bit communautaire décevante | Fausse impression sur le modèle | Toujours comparer à Python au même bpw |

---

## 11. Décisions actées & questions restantes

**Acté (2026-08-28)** : portage local (repo perso, conventions upstream, PR finale pilotée par Vincent) · ordre 27B → serveur → Flash-Next · GUI de bench à chaque jalon, mesures via le profiler · document de référence local (ce fichier) · Lexar : 384 Go libres, réglé.

**Restant à trancher — chaque question est ancrée à une GATE (§0.1), l'exécutant la posera au moment où elle devient bloquante, pas avant** :
1. **Quant Flash-Next** : ~~option A ou B → G-4~~ G-4 levée le 2026-08-29 (option A, Vontra 4-bit) ; option B redevient une question → **GATE G-4bis** (§7 rév. 4, posée à G-8).
0. **Rév. 4 (2026-09-06)** : G-1 et G-3 jamais formellement franchies, à acter à **G-8** ; versionnement git obligatoire (H0.1).
2. **MTP Flash-Next** : ~~requis v1 ?~~ réalisé en opt-in sans G-5 ; « P-MTP maintenu ? » → **GATE G-8** (rév. 4).
3. **Publication HF** : → **GATE G-6** (avant tout upload, licence vérifiée d'abord).
4. **Api-key serveur** : → **GATE G-2** (tâche 2.6 ; défaut proposé : optionnelle, LAN de confiance).

<!-- REQUEST: étude et implémentation MTP Qwen3.8 -->
## REQUEST — MTP Qwen3.8-27B

### Objectif

Étudier puis implémenter le MTP intégré de Qwen3.8-27B, d’abord pour le texte, puis pour le VLM et les conversations multi-tour avec cache persistant. Ne pas commencer par un modèle externe de speculative decoding.

### État à prendre en compte

- `mlx-swift-lm 3.31.4` fournit `MTPDrafterModel`, `MTPSpeculativeTokenIterator` et la boucle générique draft → verify → accept/reject.
- Le checkout utilisé ne contient pas de drafter Qwen3.8 ; seul le support MTP générique et le drafter Gemma4 sont présents.
- Le runtime actuel utilise une `ChatSession` standard sans configuration MTP.
- Les checkpoints locaux 4-bit et 8-bit ne contiennent aucune clé `mtp.*`, malgré `mtp_num_hidden_layers: 1` et `mtp_use_dedicated_embeddings: false` dans leur configuration.

### Travail demandé à l’équipe de planification

1. Vérifier la référence Python Qwen3.8 et l’état des travaux upstream, notamment les PR relatives au support Qwen3.8/MTP et au rollback de `MambaCache`/GDN.
2. Identifier une conversion contenant réellement les poids MTP, notamment les familles de clés :

   ```text
   mtp.fc.*
   mtp.norm.*
   mtp.pre_fc_norm_embedding.*
   mtp.pre_fc_norm_hidden.*
   mtp.layers.0.*
   ```

3. Définir le contrat exact du drafter Qwen3.8 conforme à `MTPDrafterModel`, y compris embeddings partagés, hidden states, tête de sortie et couche MTP full-attention.
4. Définir la stratégie de rollback synchronisé pour les `KVCache`, les états récurrents GDN et l’état MTP après rejet partiel de tokens : snapshot/restauration ou replay déterministe.
5. Vérifier la compatibilité du chemin VLM : préfill image unique, MRoPE, cache visuel conservé entre tours, aucune réinjection de l’image dans le drafter.
6. Proposer les fichiers à créer/modifier sans modifier directement le checkout SwiftPM.

### Intégration attendue

- Options runtime : `mtpEnabled`, `mtpDraftTokens`.
- Chargement du drafter une seule fois et réutilisation entre les tours.
- Fallback explicite et mesuré si les poids MTP sont absents ou si le drafter n’est pas compatible.
- GUI : état actif/fallback, tokens proposés, acceptés, taux d’acceptation, cycles, raison du fallback.
- `MLXProfiler` comme source unique des mesures : TTFT, débit standard, débit MTP, mémoire et traces Chrome.

### Validation obligatoire

- Parité Python/Swift d’un pas MTP.
- Même sortie en greedy avec MTP activé et désactivé.
- Rejet partiel et rollback GDN vérifiés.
- Premier tour VLM, tour texte suivant, nouvelle image au tour suivant et reset conversationnel.
- Tests 4-bit/8-bit et fallback sans `mtp.*`.
- Benchmark identique avec/sans MTP, incluant taux d’acceptation et débit effectif.

### Hors périmètre de cette demande

Flash-Next, serveur LAN, modèle externe de speculative decoding et optimisation Metal dédiée avant validation fonctionnelle.

---

<!-- ANSWER: étude MTP Qwen3.8 — rédigée le 2026-08-28, faits vérifiés (checkout local 3.31.4, mlx-vlm 0.6.2 installé, HF, PRs GitHub) -->
## RÉPONSE — étude MTP Qwen3.8-27B (2026-08-28)

**Résumé exécutif** : tout ce qui est demandé existe déjà côté upstream, mais **sur `main` non releasé** (la dernière release est 3.31.4, du 30 juin — antérieure au merge du MTP Qwen). Sur 3.31.4 le MTP Qwen3.8 est **impossible sans fork** (deux verrous, détaillés en R1). Le travail de l'équipe : (1) basculer la dépendance sur une révision épinglée de `main` (ou la branche PR #545), (2) télécharger le checkpoint drafter séparé `mlx-community/Qwen3.8-27B-MTP-4bit` — **intégré à l'API des modèles**, voir R2, (3) valider d'abord le chemin upstream `generate(mtpDrafter:blockSize:)` tel quel (baseline M1), puis (4) écrire **notre pipeline MTP maison** sur le modèle de `Gemma4MTPPipeline` pour lever la limite upstream d'un token spéculé par round (R4-bis). Le drafter reste 100 % upstream — on ne réécrit que la boucle.

### R1 — Référence Python et état upstream (vérifié le 2026-08-28)

**Côté Swift (`ml-explore/mlx-swift-lm`)** :

- **Release 3.31.4 (30 juin 2026) = notre pin actuel.** Elle contient le support MTP *générique* (`MTPDrafterModel`, `MTPSpeculativeTokenIterator`, `MTPDrafterModelFactory`) et le seul drafter Gemma4. Le MTP Qwen3.8 y est **bloqué par construction**, pour deux raisons vérifiées dans le checkout :
  1. `MTPSpeculativeTokenIterator.init` exige `canTrimPromptCache(mainCache)` ; or `MambaCache` (couches GDN) hérite de `isTrimmable = false` en 3.31.4 → **l'init `throw` immédiatement** sur tout modèle hybride Qwen.
  2. `Qwen35.swift` (LLM et VLM) en 3.31.4 **n'émet pas** `mtpLastHiddenStatesKey`/`mtpSharedKVStatesKey` et **filtre les poids `mtp.*`** dans son sanitize (`weights.filter { !$0.key.contains("mtp.") }`).
- **PR #351 « Add Qwen3.5 MTP speculative decoding » — MERGÉE sur `main` le 13 août 2026.** C'est elle qui apporte tout : `Qwen35MTP.swift` (variantes MLXLLM *et* MLXVLM), le protocole raffiné `StatefulMTPDrafterModel` + `MTPDrafterState`, `MambaCache.saveSpeculativeCheckpoint/restoreSpeculativeCheckpoint`, `SpeculativeCacheRewindModel`, l'émission du hidden dans `Qwen35`, et l'API publique `generate(input:context:mtpDrafter:blockSize:)` dans `Evaluate.swift`. NB : le fichier que la tâche 1.6 nomme « QwenMTP.swift » s'appelle en réalité **`Qwen35MTP.swift`**.
- **PR #506 — mergée** : passthrough avant le wrap d'un sliding cache (protège le cas `maxKVSize`).
- **PR #510 « Make MambaCache rewindable » — ouverte mais de facto OBSOLÈTE** : un contributeur l'a déclarée « effectively superseded by #351 » le 25 août. L'approche retenue upstream est le **checkpoint** de #351, pas le `rewindDepth` de #510. → **Proposition d'amendement (pour Vincent)** : la GATE G-5 du Jalon 3 ne doit plus surveiller #510 mais poser la question « l'approche checkpoint de #351 (rewind natif limité à 1 token) suffit-elle pour l'étape G Flash-Next ? ».
- **PR #545 « Add Qwen 3.8 support » — ouverte, active (dernier push le 28 août)** : basée sur `main` post-#351, elle *adapte* l'existant à la famille 3.8 : registrations `qwen3_8`, `qwen3_8_text`, `qwen3_8_moe`, `qwen3_8_mtp`, nouveau `HybridAttentionSchedule.swift` (décodage dynamique des `layer_types`), durcissement du sanitize. Review davidkoski : *changes requested*, il reste des conflits `ssmIdx`/`faIdx` à résoudre. **Important** : notre 27B local a `model_type: "qwen3_5"` (vérifié dans `config.json`) — il est donc couvert par les registrations de #351 déjà sur `main` ; #545 n'est nécessaire que pour des checkpoints étiquetés `qwen3_8*` et apporte des correctifs de robustesse.

**Côté Python (source de vérité pour la parité)** — `mlx-vlm 0.6.2`, installé localement (`~/Library/Python/3.12/lib/python/site-packages/mlx_vlm/`) :

| Fichier | Rôle |
|---|---|
| `speculative/mtp.py` | Boucle de rounds `_mtp_rounds` (draft → verify → walk → rollback), `_slice_shared_kv_after_reject`, block size adaptatif |
| `speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py` | `Qwen3_5MTPDraftModel` : `draft_block`, `accept_verified_tokens`, `set_shared_kv`, `bind(target)` |
| `speculative/drafters/qwen3_5_mtp/split.py` | L'outil qui a produit les checkpoints drafter séparés publiés sur HF |
| `models/qwen3_5/language.py` | `rollback_speculative_cache` (l. 1919) : la référence exacte du rollback GDN ; `_gated_delta_update_verify_decode` |

### R2 — Conversion contenant les poids MTP

**Aucune conversion MLX « full model » ne contient les clés `mtp.*`** — vérifié : le 4-bit local a 0 clé `mtp.*` sur 2180 (idem 8-bit), et le `mlx_vlm.convert` de la communauté les droppe. Les familles de clés citées dans la demande (`mtp.fc.*`, `mtp.norm.*`, …) n'existent que dans le checkpoint source `Qwen/Qwen3.8-27B` bf16. L'écosystème publie le drafter **séparément**, via l'outil `split.py` ci-dessus :

- **`mlx-community/Qwen3.8-27B-MTP-4bit`** (vérifié sur HF, index + config téléchargés) : `model_type: "qwen3_5_mtp"`, `block_size: 3`, affine 4-bit g64, **31 tenseurs, clés SANS préfixe `mtp.`** : `fc.*`, `norm.*`, `pre_fc_norm_embedding.*`, `pre_fc_norm_hidden.*`, `layers.0.{self_attn.{q,k,v,o}_proj + {q,k}_norm, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}`. **Pas d'`embed_tokens` ni de `lm_head`** : empruntés à la cible au runtime (`mtp_use_dedicated_embeddings: false`). Poids légers (< 1 Go). Existe aussi en `-MTP-8bit`, `-MTP-bf16`, `-MTP-mxfp4/mxfp8/nvfp4`. Révision source : `1d4bf0f`.
- **Règle d'appairage stricte** (README du repo) : drafter et cible issus du même checkpoint de base et de la même famille de quant → `Qwen3.8-27B-4bit` ↔ `Qwen3.8-27B-MTP-4bit`, `-8bit` ↔ `-MTP-8bit`. Le loader Swift (`qwenMTPSanitizeWeights`, `preconvertedNorms: true` pour `model_type` `*_mtp`) attend exactement ce format — c'est le format natif de `MTPDrafterModelFactory` (répertoire modèle autonome avec son `config.json`).
- **Action** : ajouter ces deux repos drafter aux variantes du downloader (tâche 1.1) et au sélecteur de la GUI (affichés comme « accessoire » du modèle cible, pas comme modèle chattable).

### R3 — Contrat exact du drafter

**Ne PAS coder contre le protocole `MTPDrafterModel` de 3.31.4** (sans état, `sharedKV` obligatoire) : le contrat réel du drafter Qwen est le protocole raffiné de `main`, **`StatefulMTPDrafterModel`**, déjà implémenté par `Qwen35VLMNextNDraftModel` (MLXVLM) / `Qwen35NextNDraftModel` (MLXLLM). À connaître pour l'intégration :

- **Constantes de contrat** (lues dans le source `main`) : `maximumBlockSize = 2`, `requiresSharedTargetKV = false` (le drafter Qwen a **son propre** `KVCacheSimple` par couche MTP, contrairement à Gemma4 qui réutilise le K/V de la cible), `requiresPromptPrefill = true` (le drafter est primé sur le prompt à partir du hidden de la cible), `requiresGreedySampling = true`.
- **Cycle de vie** : `makeState(parameters:)` → `MTPDrafterState` (cache KV drafter, `nextPosition`, `seedToken`/`seedHidden`) ; `prepareDrafterState(target:promptTokens:targetHidden:firstBonus:positionDeltas:state:sampler:)` au prefill ; `draftBlock(...)` par round ; `commitDrafterState(...)` après verify (c'est lui qui trime le cache drafter sur rejet).
- **Architecture du module** (`Qwen35VLMNextNPredictor`, clé de module `mtp`) : entrée = `concat(pre_fc_norm_embedding(embed(token)), pre_fc_norm_hidden(hidden_cible))` → `fc` `Linear(2·hidden → hidden, sans biais)` → `layers.0` = `DecoderLayer` **full_attention forcée** (`fullAttentionInterval = 1`, MRoPE partiel identique à la cible) → `norm` → **`lm_head` de la CIBLE** (embeddings et tête récupérés via `target as? Qwen35` ; `isCompatible(with:)` vérifie le type). Cohérent avec `mtp_num_hidden_layers: 1`.
- **Côté cible** : `Qwen35` (`main`) émet `mtpLastHiddenStatesKey` (hidden post-norm, pré-lm_head) quand `mtpEmitFlagKey` est posé ; pas de partage K/V.
- **Registration** : registry `MTPDrafterTypeRegistry` — `qwen3_5_mtp` couvre notre checkpoint drafter (et `qwen3_8_mtp` via #545). Sur la branche #545 c'est le registry `visionLanguage` qui sert les drafters M-RoPE, même si `vision_config` est vide dans le config du drafter. À appeler une fois avant tout chargement (même pattern last-write-wins que §6.1-A).

**Conséquence** : notre travail dans `Qwen38Core` = enregistrer, charger, brancher. Zéro couche de modèle à écrire pour le 27B.

### R4 — Stratégie de rollback après rejet partiel

**Décision : snapshot/restauration (l'approche upstream mergée), pas de replay déterministe.** Mécanique exacte, par type d'état :

- **KV full-attention (cible)** : `trimPromptCache(cache, numTokens: rejetés)` — inchangé.
- **États récurrents GDN (cible)** : pendant la passe verify, chaque `GatedDeltaNet` sauvegarde un checkpoint (fenêtre conv + état récurrent float32) au dernier token committé inconditionnellement : `MambaCache.saveSpeculativeCheckpoint(convState:recurrentState:advancedBy:)` ; sur rejet partiel, l'itérateur déclenche `restoreSpeculativeCheckpoint()`. ⚠️ Ces méthodes sont **`package`-scoped** : inaccessibles depuis `Qwen38Core`. Ne pas tenter de piloter le rollback nous-mêmes — c'est l'affaire de l'itérateur upstream, on consomme l'API `generate(mtpDrafter:blockSize:)`.
- **État MTP (drafter)** : son cache KV propre est trimé par `commitDrafterState` via `MTPDrafterState`.
- **Limite structurelle upstream** : `Qwen35` déclare `maximumNativeTargetCacheRewind = 1` et le drafter `maximumBlockSize = 2` → **1 seul token spéculé par round** dans l'itérateur upstream (bonus + 1 draft ; rejet max = 1 token = ce que le checkpoint unique sait rembobiner). Python, lui, capture les états intermédiaires **par pas** (`intermediate_states[:, accepted]` + `conv_input[:, a+1 : a+K]` dans `rollback_speculative_cache`) et tourne à `block_size: 3`. Cette limite ne s'applique qu'au chemin upstream (baseline M1) : **on la lève avec le pipeline maison — voir R4-bis**.
- **Interactions déjà gérées upstream (ne pas re-blinder, mais tester)** : quantification KV en cours de stream → passthrough sticky (tests upstream `MTPQuantizationOnsetTests`) ; wrap d'un `RotatingKVCache` → stand-down (#506). Conséquence pratique de la tâche 1.8 : avec `kvBits: 4, quantizedKVStart: 5000`, **le MTP passe en passthrough à 5000 tokens** — c'est attendu ; la GUI doit afficher la raison (`passthroughReason`). Pour les benchs MTP purs, désactiver le KV quant.

### R4-bis — Lever la limite : pipeline MTP maison (le précédent gemma-4)

**Le précédent** : sur `gemma-4-swift-mlx` on a résolu exactement cette situation (boucle upstream inadaptée à nos besoins) en **écrivant la boucle chez nous** : `Sources/Gemma4Swift/Pipeline/Gemma4MTPPipeline.swift` — un actor `mtpStream(...)` qui possède tout le round (prefill → draft → verify → walk → rollback), upstream ne fournissant que les briques (drafter, caches). Résultat : blockSize paramétrable, `SpeculativeWalk` à nous, diagnostics à nous (`sequentialVerify`), contrat de sortie identique à `ChatSession.streamResponse`. **On applique le même pattern ici** — c'est déjà la philosophie du plan (« portage local d'abord »).

**Faisabilité vérifiée hors package** (lecture du source 3.31.4 et `main`) : `MambaCache` (via `ArraysCache`) expose **publiquement** le subscript get/set de ses 2 tenseurs (`cache[0]` fenêtre conv, `cache[1]` état récurrent), `state` get/set, `copy()`, et `offset` est `public var` sur `KVCache`. Les caches full-attention sont trimmables. Un snapshot/restauration complet des états GDN se pilote donc depuis `Qwen38Core` **sans toucher au checkout ni aux méthodes `package`-scoped**.

**Mécanique du rollback multi-token (V1 — replay déterministe, API publique uniquement)** :
1. Avant chaque verify : snapshot des 48 `MambaCache` (2 tenseurs + offset par couche ; état récurrent float32 [B, 48, 128, 128] ≈ 3 Mo/couche → **≈ 150 Mo par snapshot**, un seul vivant à la fois — coût de copie à mesurer au profiler, attendu ~1-2 ms/round).
2. Verify du bloc `[bonus | d_1 … d_K-1]` en un seul forward.
3. Acceptation totale → discard du snapshot, zéro coût.
4. Acceptation partielle → restore des MambaCache + `trimPromptCache` des caches KV au point pré-verify, puis **re-forward du préfixe accepté** (≤ blockSize tokens ; décode memory-bound → coût ≈ un pas de décode). Exactitude par construction : replay déterministe, aucun état approximé.

**Séquencement (précise le plan d'intégration de R6)** :
- **M1 — baseline upstream (1-2 j)** : `generate(mtpDrafter:blockSize: 2)` tel quel. Objectif : valider drafter/checkpoint/registration, chiffres de référence, suite de tests upstream au vert. Ce chemin reste ensuite le **fallback de secours** du pipeline maison.
- **M2 — pipeline maison (3-5 j)** : `Qwen38MTPPipeline` actor (structure calquée sur `Gemma4MTPPipeline`, adaptée au drafter stateful Qwen), `blockSize` paramétrable **de 2 à 9** (1 à 8 tokens draftés — le drafter est autorégressif, `maximumBlockSize = 2` n'est que la limite de l'itérateur upstream, pas du modèle ; le rollback V1 rembobine un rejet de n'importe quelle profondeur), **défaut 3** (= `block_size` du checkpoint, comme Python), rollback V1 ci-dessus. Critère : parité token-à-token greedy avec M1 ET avec mlx-vlm Python ; accept rate ≥ celui de Python au même blockSize.
- **M2+ — mode adaptatif (0,5-1 j, après le bench en grille)** : portage de la politique `_effective_mtp_block_size` de Python (le bloc demandé est un plafond ; on ne dépasse la profondeur configurée que si le préfixe complet est accepté ≥ 65 % des 32 derniers rounds). Exposé comme `mtpDraftTokens = auto` — c'est le bon défaut utilisateur une fois validé.
- **M3 — optimisation conditionnelle (sur mesure profiler uniquement)** : si le snapshot+replay coûte > ~5 % du débit décode → capture per-step des états GDN pendant le verify (parité `intermediate_states` Python : sélection `[:, accepted]`, zéro replay), en **patch propre sur le clone épinglé**, conventions upstream, candidate à une PR pilotée par Vincent. Ne pas la faire « au cas où » : mesurer d'abord.

**Sampling** : le pipeline maison est **greedy-only en v1** (équivalence exacte triviale ; c'est aussi le protocole de parité §4.4). Le sampling spéculatif exact (rejection sampling, cf. `_SpeculativeSamplerRNG` Python) devient une extension ultérieure possible puisque la boucle est à nous — pas une contrainte upstream subie.

### R5 — Compatibilité du chemin VLM

- **Le drafter ne voit jamais l'image** — garanti par construction : `Qwen35VLMNextNDraftModel` n'a aucun module vision ; il ne consomme que (embedding du token via `embed_tokens` de la cible, hidden de la cible). Pas de re-résolution `smart_resize`, pas de `<|image_pad|>` réinjecté. Rien à faire de notre côté, mais l'affirmer dans un test (prompt image + MTP : compter les appels vision via le profiler = 1 seul encodage).
- **MRoPE** : les `positionDeltas` de la cible sont threadés au drafter (`qwen35MTPPositionIds(offset:length:batchSize:positionDeltas:)`) — la couche MTP full-attention voit les mêmes positions M-RoPE que la couche 64 de la cible.
- **Multi-tour avec cache persistant** : persistance **en mémoire du process uniquement** — conserver dans notre actor pipeline, par conversation : le `[KVCache]` cible **et** l'état drafter (cache KV MTP + position). Sur le chemin M1, `requiresPromptPrefill = true` implique que `prepareDrafterState` re-prime le drafter à chaque génération ; **avec le pipeline maison (M2) la question disparaît** : la gestion de l'état drafter entre tours est chez nous, on prime le drafter sur le suffixe du tour courant uniquement (coût ≈ 1 forward drafter sur le suffixe). **Interdit** : `savePromptCache` disque (issue upstream #443 : perd l'état M-RoPE des VLM — déjà noté §5.1).
- Le scénario de validation demandé (tour 1 image → tour 2 texte → tour 3 nouvelle image → reset) est repris tel quel dans le plan de tests R7, avec assertion sur les offsets M-RoPE après chaque tour.

### R6 — Dépendance et fichiers (sans modifier le checkout SwiftPM)

**Décision de dépendance (couverte par la tâche 1.2, pas une nouvelle GATE)** : le MTP Qwen n'existe dans **aucune release**. Deux options conformes à l'interdit « jamais de `branch:` » :
1. **Recommandée** : clone local de `ml-explore/mlx-swift-lm` figé sur un commit (branche PR #545 si les conflits `ssmIdx`/`faIdx` sont résolus au moment de démarrer, sinon `main` post-#351 qui suffit pour `model_type: qwen3_5`) + dépendance `path:` dans `Package.swift`. Le commit épinglé est noté dans `docs/knowledge/log.md` ; bugs remontés par Vincent sur la PR.
2. Alternative : `.package(url: ..., revision: "<sha>")` — épinglage exact, pas de `branch:`.

À la release upstream qui embarque #351/#545, revenir à un pin de version standard (un commit dédié).

**Fichiers à créer (tous dans ce repo)** :

| Fichier | Contenu |
|---|---|
| `Sources/Qwen38Core/MTP/MTPRegistration.swift` | Appel unique des registrations drafter upstream (`qwen3_5_mtp`/`qwen3_8_mtp`), avant tout load |
| `Sources/Qwen38Core/MTP/MTPDrafterProvider.swift` | Actor : résolution du repo drafter apparié (4bit↔MTP-4bit), chargement **une seule fois** via `MTPDrafterModelFactory.shared`, réutilisation entre tours, `isCompatible` |
| `Sources/Qwen38Core/MTP/MTPOptions.swift` | `mtpEnabled: Bool`, `mtpDraftTokens: .fixed(1…8)` ou `.auto` (M2+) → `blockSize = draftTokens + 1` sur le pipeline maison ; plafonné à `drafter.maximumBlockSize` seulement sur le chemin M1 |
| `Sources/Qwen38Core/MTP/MTPRunStatus.swift` | `active(blockSize)` / `fallback(raison)` / `indisponible` + compteurs (proposés, acceptés, rounds) — alimentés par notre pipeline (M2) ou par `GenerateCompletionInfo.{proposedDraftTokens, acceptedDraftTokens, passthroughReason}` (M1) |
| `Sources/Qwen38Core/MTP/Qwen38MTPPipeline.swift` | **M2 — le cœur** : actor calqué sur `Gemma4MTPPipeline` (gemma-4), boucle prefill → draft → verify → walk → rollback V1, blockSize paramétrable, sortie streaming identique au bypass 1.3 |
| `Sources/Qwen38Core/MTP/GDNStateSnapshot.swift` | Snapshot/restore des `MambaCache` via l'API publique (subscript + offset) ; utilisé par le rollback V1 de R4-bis |
| `Sources/Qwen38Core/MTP/SpeculativeWalk.swift` | Walk greedy pur (copie adaptée de `gemma-4 SpeculativeWalk.swift`) — logique sans MLX, testable unitairement |
| `Sources/Qwen38Core/Pipeline/…` (extension du bypass 1.3) | Sélection pipeline MTP maison (M2) → chemin upstream (M1, secours) → génération standard ; **fallbacks explicites et loggés** : pas de repo drafter sur disque → standard ; échec de load / `isCompatible == false` → standard ; init qui `throw` → standard ; passthrough en cours de stream → fin de stream normale, raison remontée |
| `Tests/Qwen38Tests/MTP*.swift` | Voir R7 |
| `Scripts/parity_reference.py` (+ module `mtp-step` de `qwen38 parity`) | Dump Python : `draft_block` de `Qwen3_5MTPDraftModel` sur (hidden, token, position) fixés seedés ; comparaison Swift sous tolérance |

**Poids MTP dans l'API des modèles (demande Vincent 2026-08-28)** — le drafter est un citoyen du catalogue, pas un fichier annexe :
- **Downloader / catalogue (tâche 1.1)** : les repos `-MTP-*` sont déclarés comme **artefact lié** de leur variante cible (paire `4bit ↔ MTP-4bit`, `8bit ↔ MTP-8bit`), avec taille, présence sur disque, resume — même mécanique que les modèles.
- **GUI (gestionnaire de modèles + sélecteur)** : chaque variante affiche son statut MTP (« drafter présent » / « télécharger (< 1 Go) » / « pas de drafter publié ») ; le téléchargement du drafter se fait depuis la GUI, pas à la main.
- **Jalon 2, `GET /v1/models` (§5.1)** : chaque entrée expose `"extra": {"mtp_available": bool, "mtp_repo": "mlx-community/…-MTP-4bit"}` ; `extra.mtp` de `/v1/chat/completions` (déjà prévu) n'est honoré que si `mtp_available` — sinon réponse avec `"mtp": "fallback:<raison>"` dans les métadonnées de fin de stream.

### R6-bis — Impacts GUI (précis)

Le panneau et l'onglet Chat (§4.3) s'enrichissent ainsi — source unique des chiffres : `LLMMetrics` du profiler, comme partout :
- **Toggle MTP à 3 états** : *Actif* (badge vert + blockSize effectif) ; *Fallback* (badge orange + raison lisible : « poids MTP absents », « drafter incompatible », « sampling non-greedy », « KV quant actif », « cache wrap ») ; *Indisponible* (drafter pas sur le disque → bouton « Télécharger » branché sur le downloader, cf. API des modèles ci-dessus).
- **Greedy-only v1** : quand un preset sampling à température > 0 est actif (thinking/instruct §2.1), le toggle MTP passe en « greedy requis » (grisé, tooltip). Un preset « **Bench MTP (greedy)** » est ajouté à la liste des presets ; il force greedy + désactive le KV quant (tâche 1.8) le temps du run.
- **Réglage `mtpDraftTokens`** : stepper 1 → 8 + position « auto » (M2+) directement dans l'onglet Chat, à côté du toggle MTP — pour reproduire à l'identique les comparaisons du type « avec 5 tokens de prédiction je passe à X t/s » et vérifier le chiffre chez nous.
- **Panneau Mesures** : lignes supplémentaires quand MTP actif — tokens proposés / acceptés, accept rate, rounds, **débit effectif vs débit standard** (le delta est la seule mesure qui compte pour l'utilisateur), raison du passthrough si survenu en cours de stream, et **histogramme d'acceptation par profondeur de draft** (position 1…K) — c'est lui qui dit si drafter profond rapporte ou coûte sur ce prompt.
- **Onglet Historique** : colonnes `mtp` (on/off/fallback), `blockSize`, `accept_rate` — pour les comparaisons A/B à l'œil et l'export CSV.
- **Onglet Serveur (Jalon 2)** : `/metrics` expose les mêmes compteurs par requête ; le journal des requêtes affiche l'état MTP de chacune.

### R7 — Plan de validation (reprend les exigences de la demande)

Dans l'ordre, chaque point = un test committé :
1. **Suite upstream d'abord** : exécuter les tests MTP du checkout épinglé (`MTPSpeculativeTokenIteratorTests`, `Qwen35MTPTests`, …) via `Scripts/run-tests.sh` — c'est le harnais gratuit qui valide la révision choisie.
2. **Parité d'un pas MTP** : `qwen38 parity mtp-step` vs dump mlx-vlm (checkpoint MTP-4bit, mêmes entrées seedées).
3. **Équivalence greedy** : 5 prompts fixes (dont 1 image), MTP on vs off → identité token-à-token (garantie par conception de la vérif spéculative ; le test protège contre une régression d'intégration).
4. **Rejet partiel + rollback GDN** : run greedy long (≥ 512 tokens) avec accept rate < 100 % observé, sortie comparée token-à-token au run sans MTP — un rollback GDN cassé diverge en quelques tokens. Pour M2, test unitaire dédié de `GDNStateSnapshot` : snapshot → forward de N tokens → restore → re-forward → états bit-identiques.
4bis. **Parité M2 vs M1** : mêmes prompts greedy, pipeline maison (blockSize 2) vs itérateur upstream → identité token-à-token ; puis blockSize 3 vs Python mlx-vlm → identité token-à-token et accept rate comparé.
5. **VLM multi-tour** : tour 1 image → tour 2 texte (cache persistant) → tour 3 nouvelle image → reset ; à chaque tour, sortie identique au même historique rejoué sans MTP ; un seul encodage vision par image (profiler).
6. **4-bit et 8-bit appariés** + **fallback sans drafter** (supprimer/renommer le répertoire MTP → génération standard, raison correcte, zéro crash).
7. **Bench A/B/B/A** avec/sans MTP (greedy, KV quant désactivé), accept rate + débit effectif → `BENCHMARKS.md`, comparé au même bench mlx-vlm Python (`--draft-model`).
8. **Grille de profondeur** : `mtpDraftTokens ∈ {1, 2, 3, 4, 5}` sur les mêmes prompts (protocole bench-guard), avec par point : débit effectif, accept rate global et par profondeur → tableau dans `BENCHMARKS.md`. C'est cette grille qui départage « 5 tokens de prédiction » vs le défaut 2, et qui calibre le seuil du mode auto (M2+).

**Ce qui change dans le plan si Vincent valide cette réponse** : tâche 1.2 = pin `main`/#545 par `path:` ou `revision:` (au lieu d'attendre la release) ; tâche 1.6 devient le séquencement M1 → M2 (→ M3 sur mesure) de R4-bis, avec `Qwen35MTP.swift` upstream comme drafter et le checkpoint `-MTP-4bit` apparié ; tâche 1.1 et §5.1 intègrent les repos drafter à l'API des modèles (R6) ; §4.3 (GUI) est amendé par R6-bis ; GATE G-5 requalifiée (checkpoint #351 au lieu de PR #510, et l'expérience M2/M3 alimente directement l'étape G Flash-Next). Le reste du Jalon 1 est inchangé.

<!-- EXECUTION-STATUS: factual state; G-1 remains pending until Vincent validates the GUI. -->
## ÉTAT D’EXÉCUTION M1 — 2026-08-29

### Réalisé

- Checkout local de `mlx-swift-lm` épinglé sur `1a562aa` (post-#351, base
  #545) branché par dépendance `path:` ; aucun `branch:` SwiftPM.
- Checkpoints drafter appariés présents sur le Lexar :
  `Qwen3.8-27B-MTP-4bit` et `Qwen3.8-27B-MTP-8bit`.
- Chargement MTP robuste pour le format publié séparément : les clés plates
  (`fc.*`, `layers.*`, …) sont préfixées avant le sanitize upstream puis
  chargées strictement avec la quantification déclarée.
- Baseline M1 branchée dans le runtime, la CLI et la GUI : chargement unique
  du drafter, état actif/fallback/indisponible, propositions, acceptations,
  rounds, raison de passthrough et TTFT mesuré depuis le début réel du tour.
- Smokes réels validés : texte 4-bit avec MTP, premier tour VLM 4-bit avec
  image (572 tokens de préfill, 12 proposés, 10 acceptés), et texte 8-bit.
- Build d’exécution MLX validé avec `Scripts/build.sh` (`xcodebuild`) ; la
  sortie contient `mlx-swift_Cmlx.bundle` et `default.metallib` à côté du
  binaire. La suite locale passe avec 9 tests. `swift build` reste uniquement
  un secours de compilation et ne constitue jamais un build exécutable MLX.
- Protocole final exécuté sur l’image Xi Jinping : trois tours continus,
  thinking `low`, `maxTokens=2048`, 4-bit/8-bit/bf16, MTP désactivé puis activé.
  Les six rapports sont conservés dans `results/final-*.tsv`; les probes
  mémoire correspondantes sont dans `results/memory-*.tsv`.
- Les trois variantes MTP chargent maintenant leur drafter apparié, y compris
  `Qwen3.8-27B-MTP-bf16`, et remontent propositions, acceptations, accept rate,
  TTFT et mémoire MLX dans les rapports.

### Limites connues à ne pas masquer

- L’API upstream limite M1 à `blockSize = 2`, donc un seul token drafté par
  round ; le réglage GUI 1…8 est déjà préparé mais plafonné tant que M2 n’est
  pas implémenté.
- M1 re-prépare l’historique complet pour le chemin direct MTP. Après le
  premier tour MTP, les tours suivants restent sur ce chemin même si MTP est
  désactivé ; les métriques indiquent `conversationReplayed` et n’annoncent
  pas une réutilisation KV. Le cache target + drafter persistant est M2.
- La parité token-à-token Python/Swift, le rollback GDN multi-token et la
  grille de profondeur restent à faire. G-1 n’est donc pas franchie.

### Avancement M2 local — 2026-08-29

- Une première boucle locale `Qwen38MTPPipeline` existe derrière la commande
  diagnostique `qwen38 mtp-probe`. Elle possède son cache target, son état de
  drafter, la vérification `[bonus + drafts]`, le walk greedy, le rollback GDN
  et la reprise du préfixe accepté.
- Sonde réelle 4-bit avec l’image Xi Jinping, `thinking low`, 16 tokens,
  largeur demandée 3 : sortie identique au chemin target standard avec les
  mêmes paramètres ; 5 rounds, 10 propositions, 9 acceptations, 1
  restauration GDN (90 %). Le passage multi-token est donc fonctionnel sur ce
  premier cas multimodal.
- Le drafter upstream prépare un seed et annonce `maximumBlockSize = 2`.
  La boucle locale consomme ce seed puis le réinjecte avec son hidden déjà
  calculé afin de prolonger le bloc ; `proposalAppended` reste compté selon
  les entrées réellement écrites dans le cache privé. La largeur 3 n’est donc
  plus plafonnée par la limite de l’itérateur upstream.
- Le budget terminal est complété par un pas target ordinaire lorsqu’il ne
  reste qu’un slot, afin de respecter exactement `maxTokens`.
- `Scripts/build.sh` utilise désormais le scheme package
  `Qwen38MLXSwift-Package`, qui reconstruit à la fois le CLI et la GUI ; le
  scheme GUI seul pouvait laisser le CLI obsolète. Les builds et tests restent
  exclusivement effectués par `xcodebuild` et embarquent
  `default.metallib`.
- Suite locale : 11 tests passés par `Scripts/run-tests.sh` via `xcodebuild`.

### Intégration M2 runtime/GUI — 2026-08-29

- Le M2 local est maintenant sélectionnable dans le chemin de production du
  runtime et de la mini-GUI via `Qwen38MTPEngine.local` ; le choix reste
  expérimental et le défaut demeure M1 upstream.
- `Qwen38MTPPipeline` émet les tokens acceptés/corrigés au fil de l'eau. Le
  runtime les transforme en chunks, mesure le TTFT au premier token réellement
  généré, alimente `MLXProfiler` (préfill, génération, décodage) et remonte les
  compteurs MTP dans `GenerateCompletionInfo` puis `Qwen38RunMetrics`.
- La GUI propose désormais `M1 upstream` / `M2 local`, conserve la
  conversation target + drafter entre les tours et affiche le statut/cache
  correspondant. Le M2 reste greedy-only ; une image ajoutée après le premier
  tour M2 n'est pas acceptée par ce chemin et doit rester explicitement traitée
  comme un cas de fallback à finaliser.
- Validation fonctionnelle courte : le binaire CLI compilé par `xcodebuild` a
  exécuté le premier tour VLM avec l'image Xi Jinping en 4-bit, thinking `low`,
  16 tokens, M2 local : 1 279 tokens de préfill, TTFT 12,24 s, 11 tokens
  proposés, 9 acceptés, sans crash ni perte de flux.
- La suite `Scripts/run-tests.sh` passe par `xcodebuild` avec 11 tests. Le build
  package reconstruit CLI + GUI et fournit `mlx-swift_Cmlx.bundle` ainsi que
  `default.metallib`.
- Les longueurs de sortie et de prompt divergent encore entre les runs
  standard et MTP ; les chiffres M1 ne doivent donc pas être présentés comme
  une accélération validée à sortie identique tant que le test d’identité
  greedy de R7 n’est pas vert.
- Le rollback GDN M1 a révélé et corrigé un défaut d’alignement : le checkpoint
  restaurait l’état récurrent avec l’offset pré-vérification au lieu de
  l’offset après le bonus committé. L’assertion d’offset a été ajoutée au test
  upstream local.
- La parité M1 contrôlée sur un prompt texte (256 tokens) est identique après
  ce correctif. Sur le scénario VLM trois tours, le tour 1 est identique avec
  MTP actif; les tours 2–3 sont désormais identiques grâce à un fallback
  explicite, car le replay M1 ne transmet pas la table M-RoPE complète des
  images au cache privé du drafter. Ce fallback est temporaire et doit être
  remplacé par l'état persistant/les positions du pipeline M2.
- Les premières primitives M2 sont en place et testées : walk greedy sans
  dépendance MLX (`Qwen38SpeculativeWalk`) et snapshot/restauration strict des
  `MambaCache` (`Qwen38GDNStateSnapshot`), avec refus explicite d'une topologie
  incohérente.
- Le squelette exécutable du pipeline local M2 est compilé :
  `Qwen38MTPPipeline.run` réalise préfill → draft → verify → walk → rollback
  et accepte une largeur supérieure à celle du baseline upstream. Il n'est
  pas encore sélectionné par le runtime de production : l'intégration
  streaming, les états entre tours et le test réel sur checkpoint restent la
  prochaine sous-étape.
- La parité brute M2/M1 est maintenant validée sur le cas multimodal réel :
  4-bit, image Xi Jinping, thinking `low`, `maxTokens=16`, `blockSize=2`.
  `qwen38 mtp-parity` obtient 16 tokens de chaque côté, séquence d'IDs
  `IDENTIQUE` et texte décodé identique. La comparaison ne repose donc plus
  sur une simple égalité de chaînes.
- L'append multi-tour ne re-render plus les anciens messages assistant. Le
  template Qwen3.8 encode séparément `reasoning_content`, alors que le cache
  M2 contient les tokens bruts après l'ouverture `<think>` ; retransformer la
  réponse décodée provoquait une divergence à la frontière du premier tour.
  `Qwen38MTPConversation.append(suffixTokens:)` ajoute maintenant seulement
  le suffixe structurel exact (`im_end`, nouveau user, nouveau assistant) et
  conserve le ledger target/drafter au token près.
- La sonde trois tours a passé le scénario réel 4-bit avec l'image Xi Jinping,
  thinking `low`, `maxTokens=8`, `blockSize=3` : image → texte → texte,
  avec réutilisation M2 continue. Résultats : tour 1 = 4 propositions/4
  acceptées, tour 2 = 8/2, tour 3 = 6/4 ; aucune divergence ni replay du
  préfixe image.
- `Scripts/run-tests.sh` repasse via `xcodebuild` avec 11 tests, dont walk
  greedy, snapshot/restauration GDN et résolution des pairs MTP.
- Le pipeline M2 connaît désormais les EOS déclarés par `config.json`, le
  token EOS du tokenizer et les `extra_eos_tokens`. Il retire le token d'arrêt
  de la sortie et invalide l'état conversationnel M2, ce qui interdit de
  repartir accidentellement derrière un cache positionné après EOS. La
  validation trois tours reste identique après ce changement.

### Prochaine séquence

1. Faire valider la mini-GUI avec le scénario image → texte → texte, M2 local,
   puis tester le bouton reset et le fallback après arrêt EOS.
2. Ajouter la parité M2/M1 et l'identité greedy sur plusieurs prompts, puis
   comparer les sorties M2 avec le chemin standard sur le scénario complet
   image → texte → texte.
3. Valider le traitement EOS sur une génération longue et documenter le
   comportement de reset après arrêt ; le probe reste borné par `maxTokens`
   pour les comparaisons reproductibles.
4. Une fois l'intégration métrique validée, lancer la grille de profondeur
   `mtpDraftTokens` 1…5 en 4-bit, 8-bit et bf16, puis seulement décider du
   mode `auto` et du portage Flash-Next.

<!-- REQUEST: BF16 dans la mini-GUI + correction M2 premier tour + nettoyage des tokens structuraux -->
## REQUEST — retour de test GUI M2 — 2026-08-29

Observations remontées après un test réel :

- La mini-GUI ne propose actuellement que les boutons 4-bit et 8-bit. Le
  prochain round doit ajouter la variante **BF16** et sélectionner
  `/Volumes/Lexar/models/mlx-community/Qwen3.8-27B-bf16`, avec son drafter
  apparié `Qwen3.8-27B-MTP-bf16`. L'état de disponibilité MTP et les mesures
  mémoire doivent être recalculés après chaque changement de variante.
- Avec M2 activé, un second tour peut afficher `La conversation M2 n'a pas
  encore été initialisée.` alors qu'un premier tour a déjà été affiché comme
  terminé. Le prochain round doit instrumenter et verrouiller la transition
  **chargement → sélection M2 → premier tour → état M2 prêt pour le second** :
  l'engine choisi doit être capturé au début du tour, l'état ne doit être
  publié comme réutilisable qu'après son initialisation complète, et toute
  transition impossible doit donner une raison de fallback explicite plutôt
  qu'une erreur interne.
- Un token structural `<|im_start|>` apparaît parfois à la fin du message
  assistant. Le flux GUI doit distinguer les tokens de contrôle du texte
  visible : filtrer les marqueurs ChatML (`im_start`, `im_end`, rôles et
  séparateurs) dans le detokenizer de sortie, sans casser la gestion du
  thinking ni le ledger brut utilisé par M2. Le nettoyage doit être appliqué
  au flux et au texte final conservé pour le tour suivant ; il ne doit jamais
  modifier les IDs servant au cache.

Critère de validation obligatoire du round : scénario GUI reproductible
**BF16 / image au premier tour / M2 local / thinking low / texte au second
tour**, sans erreur d'initialisation et sans token ChatML visible, avec TTFT,
accept rate, compteurs MTP et mémoire affichés séparément pour chaque tour.

### Round exécuté — 2026-08-29

- Ajout du bouton **BF16** dans la mini-GUI, avec sélection du chemin cible et
  résolution automatique du drafter `Qwen3.8-27B-MTP-bf16`. Les quatre
  répertoires attendus sont présents sur le Lexar ; le BF16 cible occupe
  environ 51 Go et son drafter environ 831 Mo.
- Correction de la transition après EOS : le runtime ne considère plus une
  session M2 arrêtée comme continuable (`isStarted`). Il utilise alors un
  replay contrôlé et expose une raison de fallback au lieu de déclencher
  `conversation M2 non initialisée`.
- Ajout de `Qwen38VisibleText` : les tokens ChatML structuraux sont retirés
  des chunks et du texte affiché/conservé, tandis que les IDs bruts restent
  inchangés pour le ledger target/drafter.
- Build `xcodebuild` réussi ; suite `Scripts/run-tests.sh` réussie avec 12
  tests. Smoke test réel 4-bit VLM + image + M2 local réussi : 1 279 tokens
  de préfill, TTFT 10,93 s, 11 proposés, 9 acceptés, aucun marker ChatML
  dans la sortie.
- La GUI fraîchement reconstruite est lancée. La validation interactive du
  scénario complet BF16 reste le contrôle utilisateur de G-1 ; la nouvelle
  variante est prête à être sélectionnée puis chargée.

### Correctif M2 après EOS — 2026-08-29

- Diagnostic confirmé par la capture utilisateur : le tour 1 M2 était actif,
  puis `Qwen38MTPConversation` supprimait son état dès l'émission d'un token
  EOS, ce qui imposait au tour 2 un replay contrôlé.
- Correction : la frontière cible/drafter est conservée après EOS avec le
  token d'arrêt en attente ; le tour suivant le consomme et n'ajoute pas un
  second `<|im_end|>` lorsque l'EOS est déjà ce marqueur.
- Validation `xcodebuild` : build réussi et suite de 12 tests réussie.
- Validation réelle 4-bit, image Xi Jinping, thinking low, M2 local,
  `maxTokens=1024`, deux tours : tour 1 `58 rounds / 116 proposés / 87
  acceptés`, tour 2 `352 rounds / 704 proposés / 484 acceptés`. Le second
  tour est resté sur M2 local ; aucun fallback/replay n'a été déclenché.

### Diagnostic trace MTP — 2026-08-29

- La trace fournie (`qwen38-run.trace.json`) correspond au dernier tour d'une
  conversation de trois tours. Elle contient un préfill de 2 408 tokens et
  18,3 s avant le premier token : ce tour a bien suivi le replay contrôlé,
  conformément au statut `MTP fallback` affiché. Elle ne permettait pas de
  conclure sur le premier tour, car le panneau ne montrait que le statut du
  dernier run.
- Le fallback observé est cohérent avec une session M2 arrêtée par EOS après
  le tour précédent ; il ne signifie pas que M2 n'a jamais été exécuté.
- Le panneau affiche désormais le moteur et le bilan MTP de **chaque tour**.
  La trace embarque `mtpRequested`, `mtpEngine` et `mtpSelectedPath` dans
  `Session Info`, afin de rendre le chemin réellement choisi vérifiable.
- Validation réelle complémentaire : image + 4-bit + M2 local, 16 tokens,
  `MTP actif`, 10 tokens proposés et 10 acceptés ; la trace de contrôle porte
  `mtpEngine=local`, `mtpRequested=true`, `mtpSelectedPath=local`.

### Matrice de qualification locale — 2026-08-29

- La commande `conversation-benchmark` accepte maintenant
  `--mtp-engine`, `--mtp-draft-tokens`, et écrit pour chaque tour le TSV, la
  réponse visible et la trace Chrome du profiler. Le benchmark extrait le
  `content` du flux thinking avec `Qwen38ThinkingStreamParser`; les IDs bruts
  du runtime et du ledger M2 restent inchangés.
- Matrice exécutée avec `xcodebuild`, thinking `low`, plafond 2 048 tokens,
  température greedy, image `/Users/vincent/Downloads/licensed-image-2.jpeg`
  (Macron, car l'image Xi demandée n'était pas présente au moment du run),
  puis les deux questions textuelles de suivi. Six rapports sont disponibles
  sous `results/qualification-*.tsv`, avec 18 réponses et 18 traces sidecar.
- Les références sans MTP ont terminé les trois tours avec cache réutilisé aux
  tours 2 et 3. Pics MLX observés : 17 484,7 MiB en 4-bit, 32 238,9 MiB en
  8-bit et 53 538,6 MiB en BF16.
- M2 local avec un bloc de largeur 3 (`--mtp-draft-tokens 2`) est actif dans
  les trois précisions. Le cache est réellement réutilisé aux tours 2 et 3,
  avec une entrée réduite à environ 19–26 tokens. Taux d'acceptation moyens
  observés sur les trois tours : 65,9 % en 4-bit, 65,4 % en 8-bit et 66,7 %
  en BF16. Pics MLX : 18 247,1 MiB, 32 238,9 MiB et 55 572,0 MiB.
- Toutes les traces contiennent les métadonnées profiler/MTP et tous les
  sidecars visibles ont été contrôlés sans `<think>`, `</think>`,
  `<|im_start|>` ni `<|im_end|>`. `xcodebuild test` est passé après le
  correctif du benchmark.
- Correction de qualification : les runs ont été exécutés séquentiellement et
  la charge GPU observée correspondait aux inférences de cette matrice ; aucune
  charge externe n'est à retenir dans l'interprétation. Les temps constituent
  donc une baseline valable sur le M3 Max 96 Go. La variation entre tours doit
  plutôt être lue comme un effet de chauffe, de longueur de contexte et de
  chemin d'inférence (standard contre cache M2), à confirmer par une campagne
  A/B dédiée si l'on veut isoler précisément chaque facteur.

### Sweep profondeur M2 et profil retenu — 2026-08-29

- Sweep complet exécuté avec `xcodebuild` : 15/15 runs, moteur M2 local,
  thinking `low`, trois tours du scénario image puis texte, plafond de 512
  tokens pour rendre la comparaison reproductible, profondeurs 1 à 5 en
  4-bit, 8-bit et BF16. Les rapports sont sous
  `results/sweep-20260829-{4bit,8bit,bf16}-draft{1..5}.tsv`.
- Le réglage `draft=1` est retenu pour le profil par défaut : sur les tours
  2–3, il obtient le meilleur taux d'acceptation moyen et le meilleur débit
  ou un débit équivalent dans les trois précisions. Les largeurs supérieures
  dégradent nettement l'acceptation et n'apportent aucun gain de débit sur ce
  Mac avec ce checkpoint.
- Défauts appliqués : mini-GUI en 4-bit, M2 local activé, un token drafté,
  plafond 2 048 ; l'API accepte aussi `mtp_engine` et `mtp_draft_tokens` et
  active ce profil par défaut si ces champs sont absents.

### Profil final 27B — 2026-08-29

- Run complet :
  `results/final-profile-4bit-m2-local-2048.tsv`, avec image
  `/Users/vincent/Downloads/licensed-image-2.jpeg` (Macron, utilisée comme
  média disponible ; le fichier Xi n'était pas présent au moment de cette
  exécution).
- Résultats : tour 1 image, prompt 1 296 tokens, TTFT 10 990,1 ms, 222
  tokens générés, 17,08 tok/s, M2 `119/104` acceptés/proposés ; tour 2,
  TTFT 303,5 ms, 641 tokens, 12,33 tok/s, `365/277` ; tour 3, TTFT 361,9 ms,
  568 tokens, 12,85 tok/s, `307/261`. Le cache est réutilisé aux tours 2 et 3.
- Pic MLX : 18 247,1 MiB. Les trois sidecars texte et leurs traces Chrome
  ont été inspectés : aucune balise `<think>`, `</think>`, `<|im_start|>` ou
  `<|im_end|>` visible.

### Passage au jalon Flash-Next — état de préparation — 2026-08-29

- Vérification locale effectuée avant toute mutation : le Lexar contient les
  six variantes 27B et les trois drafters correspondants, mais aucune
  conversion `Qwen3.8-Flash-Next`/`qwen4_exp`.
- Le checkout de référence Swift est `Vendor/mlx-swift-lm` sur `pr-545`,
  commit `1a562aa` ; il contient le support Qwen3.8/3.5 et MTP, pas
  `qwen4_exp`. L'installation Python locale de `mlx-vlm` ne fournit pas non
  plus les sources `qwen4_exp` attendues par l'étude initiale.
- La suite est donc arrêtée proprement au **GATE G-4** (§7) : il faut choisir
  l'option A (conversion MLX Flash-Next existante) ou l'option B
  (quantification streaming à développer). Aucun modèle Flash-Next n'a été
  téléchargé et aucun code Swift incomplet n'a été enregistré ; le jalon 27B
  reste buildé et testable indépendamment.

### GATE G-4 levé — conversion de qualification retenue — 2026-08-29

- Option A retenue : `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, conversion
  communautaire MLX issue du checkpoint BF16 officiel, 4-bit affine g32,
  vision + architecture `qwen4_exp` + bloc MTP natif inclus, environ 113 Go.
  Cette conversion est choisie pour qualifier l'architecture complète avant
  d'envisager notre propre quantification streaming ; elle ne constitue pas
  une validation de licence ou de qualité finale.
- Le premier essai via `qwen38 download` a échoué sur `NSURLError -1005`
  après avoir reçu une partie du premier shard ; le fichier temporaire a été
  supprimé par `URLSession`. Pour ne pas perdre les gros fichiers, le secours
  est `Scripts/download-hf-resumable.sh`, qui télécharge directement les URLs
  HF avec `curl --continue-at -`, retries et validation de taille, dans
  `/Volumes/Lexar/models/Vontra/`. Le nouveau téléchargement est actif.
- Référence de conversion :
  `https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`. Le runtime
  Swift actuel ne sait pas encore charger `qwen4_exp` ; la prochaine tâche
  vérifiable est l'ajout du contrat config/sanitizer et du probe de détection,
  avant toute tentative de chargement des shards.

### Étape A — contrat de configuration Flash-Next — 2026-08-29

- `Sources/Qwen38Core/FlashNext/Qwen4ExpConfiguration.swift` ajoute le
  décodage typé de `qwen4_exp` : topologie hybride 48 couches, paramètres
  GDN/QSA, MoE, hyper-connections, n-gram/PLE, MTP et vision.
- Les invariants bloquants sont contrôlés avant chargement : nombre de
  `layer_types`, présence des couches full-attention, quatre flux résiduels,
  128 shards n-gram, n-grammes de taille 3 et paramètres QSA positifs.
- Deux tests sans checkpoint couvrent la configuration réelle et le rejet
  d'une topologie incohérente. Build et tests `xcodebuild` sont verts.
- Le téléchargement direct est en cours : le premier shard est reçu dans le
  fichier temporaire système avant déplacement atomique sur le Lexar. La
  suite du portage attendra le modèle complet pour écrire les règles de
  sanitizer contre l'index réel, sans exposer de modèle non chargeable dans
  la GUI ou le serveur.
- Pour réduire la durée totale, le script accepte désormais un fichier de
  départ précis ; un second worker a été lancé sur
  `model-00012-of-00022.safetensors` tandis que le premier poursuit les
  shards précédents. Les deux workers partagent la bande passante et gardent
  chacun leur reprise `.part` indépendante.
- Le premier lancement direct était anonyme. Le script réutilise maintenant
  automatiquement le token de la session `hf auth token` (ou `HF_TOKEN`) via
  un fichier netrc temporaire privé, jamais via la ligne de commande ; les
  deux workers ont été relancés authentifiés avec reprise des shards.
- Téléchargement terminé le 2026-08-29 : 22/22 shards, index safetensors,
  tokenizer et fichiers de configuration présents. L'index référence 3 747
  tenseurs répartis sur 22 fichiers, sans référence manquante ; poids sur
  disque : 113 209 682 735 octets (105,43 GiB). Le `config.json` réel annonce
  `qwen4_exp`, `qwen4_exp_text`, 48 couches et un MTP à une couche.
- Préflight réel exécuté après téléchargement : 36 couches
  `linear_attention`, 12 couches full-attention/QSA, 128 shards n-gram,
  hyper-connections présentes sur les 48 couches, et poids MTP sous
  `language_model.mtp.*`. Le CLI `qwen38 info` affiche désormais ces valeurs
  et confirme : « configuration valide, runtime qwen4_exp non implémenté ».
  Compilation validée avec `xcodebuild` ; aucune inférence Flash-Next n'est
  lancée tant que QSA, PLE/n-gram et flux résiduels ne sont pas portés.
- Le preflight `Qwen4ExpCheckpointPreflight` lit uniquement les headers
  safetensors : il vérifie les 3 747 clés contre les 22 shards, les offsets,
  les 1 020 triplets quantifiés `weight/scales/biases`, la topologie 36/12,
  les 128 shards n-gram et la présence du MTP. Le probe réel `qwen38 info`
  passe ; la compilation et les tests passent avec `xcodebuild`.

### Étape D2 — cache QSA et premier sélecteur de masque — 2026-08-29

- `Qwen4ExpCachePlan` encode le choix par couche : `MambaCache`/état récurrent
  pour les 36 couches `linear_attention`, cache QSA pour les 12 couches
  `full_attention`. Le cache QSA conserve séparément les K/V principaux, les
  clés brutes de l'indexeur en `[B,S,D]` (format réel après suppression de la
  tête KV unique) et les positions multimodales ; `trim`, `copy` et la
  sérialisation de l'état sont couverts sans checkpoint.
- `Qwen4ExpQSAMask` porte la partie discrète indépendante des poids : scores
  de blocs en float32, top-k de blocs, conservation du tail causal et
  court-circuit causal lorsque le contexte visible reste sous
  `budget / compress_ratio`. Le constructeur produit un masque booléen
  `[B, 1, Q, K]` utilisable par SDPA.
- Correction importante rencontrée pendant la compilation : `BaseKVCache` et
  `KVCacheSimple` exposent des membres non-`open` ; le cache client implémente
  donc directement le protocole public `KVCache`, ce qui le rend réellement
  consommable depuis ce package.
- Validation : build et suite de tests sérialisée réussis par `xcodebuild`, puis
  25 tests confirmés directement par `xcrun xctest`. Tests ajoutés pour le plan
  hybride, le cycle cache/indexeur et le fallback dense puis la sélection d'un
  bloc. Le modèle Flash-Next n'est toujours pas chargé : l'étape suivante est
  la projection apprise de l'indexeur, puis la parité scores/indices contre
  `mlx-vlm`.

### Étape D2b — projection apprise de l’indexeur QSA — 2026-08-30

- `Qwen4ExpQSAIndexer` porte la projection `index_qk_proj` `[2560 → 640]`,
  compatible avec un remplacement `Linear`/`QuantizedLinear` avant chargement
  des poids 4-bit du checkpoint. Il sépare les quatre têtes Q et l’unique tête
  K, normalise Q, et conserve K brut pour le pooling de blocs ultérieur.
- La forme est alignée sur la référence : sorties Q `[B,4,S,128]`, clés brutes
  `[B,S,128]`, puis RMSNorm K après moyenne des micro-blocs. RoPE reste
  volontairement dans l’étape suivante car il dépend de l’horloge MRoPE du
  décodeur et des positions multimodales.
- Validation : build `xcodebuild` réussi ; 26 tests exécutés et confirmés par
  `xcrun xctest`, incluant formes et absence de NaN sur la projection.
- Prochaine étape : ajouter le pooling `[B,S,D] → [B,Blocks,D]`, appliquer la
  MRoPE de l’indexeur et produire les scores/indices de référence avant toute
  tentative de chargement complet du modèle.

### Étape D2c — pooling QSA — 2026-08-30

- `Qwen4ExpQSAIndexer.poolCompleteKeys` moyenne en float32 les groupes de
  `compressRatio` clés brutes et exclut explicitement le tail incomplet ;
  `pooledKeys` applique ensuite le RMSNorm K et retourne `[B,1,Blocks,D]` pour
  le scoreur QSA.
- Le test couvre un contexte dont la longueur n’est pas multiple de 4 et
  vérifie que le tail n’est pas absorbé dans le dernier bloc.
- Validation : build `xcodebuild` réussi ; suite de tests puis confirmation
  directe `xcrun xctest`, 26 tests passés. Le prochain bloc est MRoPE indexeur
  (positions texte et multimodales), puis la parité des scores/indices.

### Étape D2d — MRoPE indexeur QSA — 2026-08-30

- `Qwen4ExpMRoPE` porte les positions texte `[B,S]` et multimodales `[3,B,S]`,
  les fréquences intercalées `[11,11,10]` et la rotation partielle des 64
  premières dimensions. Les tables sont construites en float32 ; la queue
  `[64..head_dim]` est garantie inchangée.
- Le module est partagé par l’indexeur QSA et la future attention complète,
  afin d’éviter deux implémentations de la même horloge MRoPE. Il reste
  indépendant des poids et du chargement des 113 Go.
- Validation : build `xcodebuild` réussi ; 27 tests exécutés et confirmés par
  `xcrun xctest`. Le prochain jalon est l’assemblage projection → MRoPE →
  pooling → scores/indices, puis une référence Python minimale pour vérifier
  les valeurs et non seulement les formes.

### Étape D2e — assemblage du chemin indexeur QSA — 2026-08-30

- `Qwen4ExpQSAIndexer.makeMask` relie désormais projection apprise, cache des
  clés brutes/positions, MRoPE, pooling float32 et sélection de blocs dans un
  appel unique. Le masque retourné respecte la forme SDPA `[B,1,Q,K]`.
- Tant qu’aucun micro-bloc complet n’est disponible, l’API retourne `nil` afin
  que la couche d’attention conserve le masque causal ordinaire. Le tail
  incomplet reste ensuite directement visible ; il n’est jamais mélangé à un
  bloc pooled.
- Une assertion de fixture a été réalignée sur la dimension QSA réelle
  `indexer_head_dim=128` après le premier run de test ; ce n’était pas une
  correction de runtime.
- Validation finale : `xcodebuild test` réussi, puis `xcrun xctest` direct
  confirmé avec 27 tests passés. Les avertissements CoreData/XPC observés
  viennent de l’environnement de test et n’affectent pas les assertions.
- Aucun chargement des 113 Go ni aucune inférence Flash-Next n’est encore
  engagé. Prochaine étape : produire un probe Python/MLX-VLM de scores et
  indices QSA, comparer les valeurs avec Swift, puis intégrer ce masque dans
  une couche full-attention minimale avant le décodeur complet.

### Étape D2f — frontière de sélection alignée sur mlx-vlm — 2026-08-30

- `makeMask` applique maintenant exactement la condition de la référence :
  tant que `max_complete_blocks <= budget / compress_ratio`, il retourne
  `nil` et laisse l’attention causale prendre le relais. Cela évite un coût de
  sélection inutile au début du contexte et fixe la frontière discrète QSA.
- Le test d’assemblage couvre désormais les deux régimes : contexte court
  sans masque QSA, puis 12 tokens avec trois blocs complets et sélection de
  deux blocs selon le budget. La forme et la causalité du masque sont vérifiées.
- Validation : `xcodebuild test` réussi et `xcrun xctest` direct confirmé avec
  27 tests passés.

### Étape D2g — première couche d’attention QSA isolée — 2026-08-30

- `Qwen4ExpQSAAttention` assemble désormais, dans une couche testable seule,
  les projections Q/K/V/O, le query output gate, les RMSNorm Q/K, le MRoPE,
  l’indexeur QSA et la mise à jour du cache K/V principal.
- Le chemin sans assez de blocs complets utilise un masque causal explicite ;
  le chemin sparse transmet le masque QSA à SDPA. La couche n’instancie
  toujours ni MoE, ni n-gram/PLE, ni hyper-connections : elle sert de couture
  de parité et de garde-fou mémoire avant le décodeur complet.
- Validation : `xcodebuild test` réussi ; `xcrun xctest` direct confirme 28
  tests passés, dont un forward MLX réel de la couche QSA sur une fixture
  réduite. Les avertissements CoreData/XPC restent propres à l’environnement
  de test.
- Prochaine étape : dumper les tenseurs intermédiaires d’une couche QSA
  Python/MLX-VLM (projection, MRoPE, scores, masque, sortie SDPA), puis
  comparer numériquement cette couture Swift avant d’ajouter GDN et MoE.

### Étape D2h — couture de parité QSA — 2026-08-30

- `Qwen4ExpQSAIndexer` expose maintenant un chemin `fromProjected` : la
  projection `index_qk_proj` peut être fournie depuis un fixture, ce qui
  sépare les erreurs de chargement/quantification des erreurs d’algorithme.
- `Scripts/qwen4-exp-qsa-reference.py` exporte un fixture déterministe
  safetensors (projection, positions, Q/K MRoPE, clés pooled, scores et
  masque) avec des dimensions réduites mais les constantes Flash-Next réelles
  (`rotary_dim=64`, sections `[11,11,10]`, scores float32, budget et ratio
  configurables). Le script est syntaxiquement validé ; son exécution MLX
  nécessite une session macOS disposant du device Metal.
- Validation Swift : build et `xcodebuild test` réussis ; `xcrun xctest`
  confirme 28 tests passés, dont le forward QSA isolé.
- Étape suivante : ajouter le lecteur Swift de ce fixture et les tolérances
  max/mean pour les cinq sorties discrètes/continues, puis lancer la parité
  sur le Mac avec Metal avant de brancher QSA au décodeur hybride.

### Étape D2i — correction batch du masque QSA — 2026-08-30

- Le retour de `Qwen4ExpQSAMask.tokenMask` insère maintenant l’axe SDPA après
  le batch (`[B,1,Q,K]`). L’ancienne insertion en tête était invisible en
  `B=1`, mais incorrecte pour un préremplissage batché.
- Une régression `B=2` est ajoutée et vérifie la forme attendue par SDPA.
- Validation : `xcodebuild test` réussi et `xcrun xctest` direct confirme 29
  tests passés.

### Étape D2j — première parité numérique Python/Swift QSA — 2026-08-30

- Le probe Python a généré un fixture MLX safetensors réel avec projection,
  positions, Q/K après MRoPE, clés pooled, scores et masque.
- Le lecteur Swift `Qwen4ExpQSAParity` relit ce fixture et compare chaque
  étape. Résultat sur la fixture 12 tokens : `max=0` et `mean=0` pour les
  sorties continues comme pour le masque discret.
- La divergence initiale du masque a permis de corriger un vrai bug de
  portage : la division des tokens visibles devait être un floor explicite
  avant conversion en nombre de blocs complets.
- Validation propre : `xcodebuild test` réussi ; `xcrun xctest` direct avec
  le fixture Python confirme 30 tests passés. Les avertissements CoreData/XPC
  restent sans incidence.
- Ce jalon valide la couture QSA uniquement ; il ne constitue pas encore une
  inférence Flash-Next complète. Prochaine étape : ajouter GDN/état récurrent,
  puis les hyper-connections et une couche MoE réduite avant l’assemblage du
  décodeur 48 couches.

### Étape D3a — wrapper GDN Flash-Next et état récurrent — 2026-08-30

- `Qwen4ExpGatedDeltaNet` porte le contrat `linear_attn` réel : convolution
  causale depthwise, projections séparées `in_proj_qkv/z/b/a`, normalisation
  Q/K, appel à `gatedDeltaUpdate` public de `mlx-swift-lm`, état récurrent
  conservé en float32, RMSNorm gated et projection de sortie.
- Le wrapper utilise `MambaCache` pour la fenêtre convolutionnelle et l’état
  récurrent ; un test exerce deux appels successifs et vérifie les formes,
  l’état à deux tenseurs et le dtype float32.
- Une hypothèse a été retirée du test : `MambaCache.advance()` ne modifie pas
  `offset` (contrairement au cache KV), il gère les métadonnées SSM. Le runtime
  devra donc garder l’horloge logique du décodeur séparément pour MRoPE/QSA.
- Validation : `xcodebuild test` réussi ; `xcrun xctest` direct confirme 31
  tests passés.
- Prochaine étape : parité Python/Swift du wrapper GDN sur un fixture à poids
  synthétiques, puis intégration de l’horloge logique et des hyper-connections.

### Étape D3b — RMSNorm zéro-centrée et hyper-connections — 2026-08-30

- `Qwen4ExpRMSNorm` applique le contrat Flash-Next des poids zéro-centrés :
  la valeur chargée est utilisée comme `1 + weight`, avec prise en charge du
  groupement par hidden size.
- `Qwen4ExpGatedResidual` porte le mélange des quatre flux, le bottleneck
  low-rank et les poids d’injection du bloc ; un test réduit vérifie les
  formes, l’absence de NaN et la largeur `hc_count × hidden_size`.
- Validation : `xcodebuild test` puis `xcrun xctest` direct, 33 tests passés.

### Étape D3c — n-gram embedding et PLE isolés — 2026-08-30

- `Qwen4ExpNGramEmbedding` calcule les bigrammes/trigrammes, la séparation en
  shards et l’état de contexte EOS ; `Qwen4ExpPLELayer` porte les projections,
  le gate, la convolution dilatée et son état temporel dans `ArraysCache`.
- Un vrai bug de portage a été corrigé : la tranche de cache convolutionnel
  ciblait la dernière dimension au lieu de l’axe séquence. Pour Flash-Next,
  sa longueur est `(ple_conv_kernel_size - 1) × ngram_size`.
- Le composant est testé sur deux appels successifs avec vérification du
  contexte n-gram, des formes PLE et de l’état `[batch, 9, hc_count × hidden]`
  de la fixture réduite ; 33 tests passent avec `xcodebuild` et XCTest direct.
- Limite explicite avant intégration checkpoint : le réordonnancement des
  sorties shardées utilise encore une copie hôte bornée au batch. Il est
  acceptable pour la couture de parité, mais doit être remplacé par un gather
  MLX/Metal groupé avant de charger les 128 shards et de mesurer le runtime.
- Prochaine étape : mapper les noms réels `ngram_embedding.shard_N`, vérifier
  les formes/poids du checkpoint par préflight strict, puis écrire la couture
  PLE complète avant MoE et décodeur.

### Étape D3d — hiérarchie PLE réelle et MoE routé — 2026-08-30

- L’arbre Swift PLE contient maintenant `ple_embedding.ngram_embedding` avec
  les 128 tables et les deux tenseurs de métadonnées de vocabulaire. Le
  sanitizer convertit les clés checkpoint `shard_N` vers la représentation
  Swift `shards.N` ; une assertion couvre le shard 127.
- Le préflight Release a été exécuté sur
  `/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` sans charger
  les poids : `3747` tenseurs, `22` shards, `1020` poids quantifiés, `128`
  shards n-gram, `1` couche MTP, `113,21 GB` de poids.
- `Qwen4ExpSparseMoE` porte le routeur top-k ascendant, les projections
  `SwitchGLU` groupées et l’expert partagé SwiGLU avec gate sigmoïde, en
  réutilisant les primitives publiques MLXLMCommon. Une fixture réduite
  valide le forward dense et l’absence de NaN.
- Validation après ces changements : `xcodebuild test` terminé avec succès,
  puis `xcrun xctest` direct confirme 34 tests passés. Les messages
  CoreData/XPC restent des warnings de l’environnement XCTest.
- Limites restantes avant l’inférence : le décodeur Flash-Next complet,
  vision, chargement strict des poids quantifiés, horloge de position et MTP
  natif ne sont pas encore assemblés ; la copie hôte PLE reste un point à
  remplacer par un gather MLX/Metal avant tout benchmark de performance.

### Étape D3e — couche et modèle texte hybrides réduits — 2026-08-30

- `Qwen4ExpDecoderLayer` assemble maintenant la topologie réelle de la
  référence : PLE éventuelle, hyper-connexion attention, branche GDN ou QSA
  sélectionnée par `layer_types`, injection dans les quatre flux, puis
  hyper-connexion MoE et seconde injection.
- `Qwen4ExpTextModel` fournit la couture textuelle minimale : tuilage ×4 à
  l’entrée, tableau de couches, réduction finale des flux et cache choisi par
  couche (`MambaCache` pour `linear_attention`, `Qwen4ExpQSAKVCache` pour
  `full_attention`). Le constructeur accepte une plage de couches pour les
  tests/parités tronqués, afin de ne pas instancier inutilement toute la
  table n-gram de 51B paramètres.
- Un défaut de test a été corrigé : `MambaCache.offset` ne représente pas
  l’horloge de tokens. Le runtime devra conserver cette horloge séparément
  pour les positions MRoPE, QSA et les continuations MTP ; le cache QSA, lui,
  avance bien son offset KV.
- Validation : compilation et `xcodebuild test` sérialisé réussis ;
  `xcrun xctest` direct confirme 36 tests passés, dont un forward MLX réel
  traversant successivement GDN/MoE et QSA/MoE.
- Limite conservée : le chargement quantifié du checkpoint réel, le modèle
  complet, la vision et le gather PLE sans copie hôte restent à faire avant
  toute inférence Flash-Next.

### Étape D3f — réassemblage PLE sur MLX — 2026-08-30

- `Qwen4ExpNGramEmbedding.lookup` conserve uniquement le routage des IDs vers
  les shards sur l’hôte. Les lignes d’embedding sont maintenant réassemblées
  par `MLXArray.at[...].add(...)`, donc sans matérialisation `Float32` ni copie
  hôte des vecteurs sélectionnés à chaque appel.
- Cette couture reprend la primitive scatter déjà utilisée par `mlx-swift-lm`.
  Elle reste un prototype de débit : le calcul des shards et l’évaluation des
  IDs nécessitent encore une synchronisation, et un kernel hash+gather fusionné
  pourra être étudié après la première mesure réelle.
- Validation : compilation `xcodebuild` réussie et XCTest direct sérialisé,
  36 tests passés. Aucun poids du checkpoint Flash-Next de 113 Go n’a été
  matérialisé.
- Étape suivante : introduire un chargeur de tranche du checkpoint (couche
  réduite, quantification appliquée avant `update`, vérification stricte des
  clés/formes), puis seulement ouvrir le chemin du modèle complet.

### Étape D3g — stockage préquantifié et garde de formes — 2026-08-30

- `Qwen4ExpPrequantized.swift` fournit des conteneurs empaquetés pour les
  `Linear`, `Embedding` et `SwitchLinear` Flash-Next. Ils créent directement
  les formes `U32`/échelles attendues, sans construire puis quantifier une
  matrice flottante de 512 experts.
- `SwitchGLU` dans le vendor local `mlx-swift-lm` expose maintenant un
  initialiseur préquantifié direct. Le changement est générique et candidat à
  une contribution upstream ; il évite la même pointe mémoire pour tout MoE
  très large.
- Tous les sous-modules Flash-Next acceptent ce mode : GDN, QSA/indexeur,
  hyper-connections, MoE, PLE et embedding de tokens. Le mode flottant reste
  la valeur par défaut des fixtures et des tests réduits.
- Le préflight conserve les validations header-only puis vérifie les ancres
  de forme du vrai checkpoint (largeurs packées embedding/lm-head, GDN, QSA et
  deux projections d’experts). Sur
  `/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, ces contrôles
  passent sur les 22 shards et 3 747 tenseurs.
- Validation finale : compilation et `xcodebuild test` sérialisé réussis,
  puis `xcrun xctest` direct avec **37 tests passés**. `qwen38 info` valide le
  checkpoint réel sans matérialiser ses 113 Go.
- Prochaine étape : chargeur de tranche réel (poids sélectionnés, sanitizer,
  `update(verify: .all)` et forward d’une couche réelle), avant l’instanciation
  du décodeur complet, de la vision et de l’adaptateur runtime/GUI.

### Étape D3h — chargement réel des couches GDN et QSA — 2026-08-30

- `Qwen4ExpCheckpointSliceLoader` ajoute un probe de chargement partiel piloté
  par `model.safetensors.index.json`. Il sélectionne les tenseurs globaux et
  ceux d’une seule couche, ouvre uniquement les shards concernés, construit le
  module directement dans ses formes quantifiées, puis applique
  `Module.update(..., verify: [.all])`. Les poids MTP, vision, n-gram et les
  autres couches ne sont ni instanciés ni matérialisés.
- La commande `qwen38 flash-slice-probe` permet de refaire ce contrôle ;
  `--lazy` isole le chargement et `--run-forward` ajoute un forward de fumée.
  Le mode `--layer 0` valide GDN/MoE et `--layer 3` valide QSA/indexeur/MoE.
- Le premier essai réel a corrigé deux contrats silencieux du checkpoint : le
  routeur `mlp.gate` est flottant et ne possède pas de compagnons quantifiés ;
  les fréquences RoPE dérivées ne sont pas des paramètres safetensors. Elles
  ne doivent donc respectivement pas recevoir le conteneur 4-bit et ne doivent
  pas être exposées par la réflexion `Module`.
- Sur le checkpoint réel
  `/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, les deux
  couches chargent chacune 71 tenseurs depuis 2/3 shards ; les forwards réels
  renvoient `[1, 4, 2560]`. La couche 0 matérialise environ 2,02 Go.
- Validation de round : compilation et tests MLX via `xcodebuild`, puis bundle
  XCTest direct sérialisé. Aucun poids complet de 113 Go n’a été chargé.
- Limite suivante : généraliser le plan de quantification par clé à tous les
  sous-modules, puis assembler le modèle texte complet avec la vision et le
  lm-head avant l’adaptateur `LanguageModel`/runtime.

### Étape D3i — validation PLE et contrainte mémoire du checkpoint — 2026-08-30

- Le probe accepte maintenant la couche 1, qui contient la PLE et les 128
  shards n-gram. Le chargement strict réel passe avec 468 tenseurs répartis sur
  8 shards, toujours sans matérialiser les tableaux n-gram.
- Un forward PLE réel sur quatre tokens produit `[1, 4, 2560]`. Le cache de
  cette couche est désormais un `ArraysCache(size: 4)` : slots 0/1 pour GDN,
  slot 2 pour la convolution PLE et slot 3 pour l’historique n-gram. Les
  couches linéaires sans PLE gardent `MambaCache()`.
- La mesure header-only du checkpoint indique environ 79,58 Go pour le groupe
  langage et 32,00 Go pour les tables n-gram (113,21 Go de fichiers au total,
  hors petites métadonnées). Un modèle Flash-Next 4-bit complet ne peut donc
  pas être considéré comme résident dans 96 Go avec marge pour activations et
  cache ; il faudra choisir entre chargement/exécution par couches, un
  checkpoint plus compact (2-bit/mixte) ou une politique d’offload explicite.
- Les trois types de couches réelles sont maintenant couverts isolément :
  couche 0 GDN/MoE, couche 1 PLE/GDN/MoE et couche 3 QSA/indexeur/MoE. Chacun
  a passé `update(verify: [.all])` et un forward MLX réel.
- Prochaine étape : décider et implémenter le modèle complet sous contrainte
  mémoire (avec vision et lm-head), puis seulement le brancher sur
  `LanguageModel` et la GUI ; ne pas lancer une matérialisation naïve des 22
  shards.

### Étape D3j — exécution séquentielle couche par couche — 2026-08-30

- `Qwen4ExpCheckpointLayerLoader` charge uniquement les paramètres d’une
  couche, sans dupliquer embedding, lm-head ou hyper-connexion finale. Le
  module est créé dans ses formes packées, mis à jour avec vérification stricte,
  puis matérialisé à la demande.
- `Qwen4ExpStreamingDecoder` exécute une liste de couches en conservant les
  caches GDN/PLE/QSA par index, mais libère le module de couche entre deux
  itérations. `eval(output)` sépare les activations de la couche précédente et
  `Memory.clearCache()` est appelé avant l’ouverture du shard suivant.
- La commande `qwen38 flash-stream-probe` couvre les trois familles dans un
  seul passage :
  `--layers 0,1,3 --sequence-length 4`.
- Sur le checkpoint réel, le passage a réussi avec sortie `[1, 4, 10240]` :
  couche 0 1,62 Go, couche 1 PLE 33,64 Go, couche 3 1,62 Go ; pic MLX
  mesuré 30,3 Go et mémoire active finale 8,9 Mo. Les caches survivent au
  changement de couche, ce qui est la base requise pour les tours suivants.
- Validation : compilation `xcodebuild`, `xcodebuild test` sérialisé et
  `xcrun xctest` direct, **37 tests passés**.
- Limite restante : ce probe ne possède encore ni embedding global, ni vision,
  ni lm-head, et recharge chaque couche. L’étape suivante doit assembler ces
  composants autour de l’exécuteur tout en conservant le budget mémoire ; une
  optimisation ultérieure pourra garder les couches chaudes ou préparer un
  cache de poids local.

### Étape D3k — embedding, réduction hyper et lm-head globaux — 2026-08-30

- `Qwen4ExpGlobalTextModel` regroupe les 13 tenseurs globaux nécessaires au
  chemin texte : embedding quantifié, hyper-connexion finale et `lm_head`
  quantifié. `Qwen4ExpGlobalCheckpointLoader` ne lit que les deux shards qui
  les contiennent et applique chaque groupe avec `update(verify: [.all])`.
- `Qwen4ExpStreamingTextModel` assemble ces poids résidents avec le décodeur
  séquentiel. Il tuile l’embedding sur les quatre flux, exécute les couches
  sélectionnées, réduit les flux puis produit les logits de vocabulaire ; les
  caches restent attachés au décodeur entre deux appels.
- `qwen38 flash-global-probe` valide le chemin embedding → réduction → tête :
  `[1,1,2560]` → `[1,1,248320]`, 798,7 MB matérialisés et 719,7 MB de pic.
  `qwen38 flash-text-probe --layers 0,1,3` produit les logits réels avec un
  pic de 31,01 GB (la couche PLE), puis revient à 728,3 MB actifs.
- `--repeat-count 2 --layers 0` exécute deux appels sur le même objet et
  confirme que le second passage réutilise le cache sans augmenter la mémoire
  résidente ; ce probe mesure désormais une vraie couture multi-tour, pas un
  simple rechargement indépendant.
- Validation du round : `xcodebuild` réussi, puis `xcrun xctest` direct avec
  **37 tests passés**. Le modèle Flash-Next complet et la vision restent
  volontairement hors du runtime public tant que le forward 48 couches et la
  parité de fusion multimodale ne sont pas validés.

### Étape D3l — tour vision isolée et prétraitement local — 2026-08-30

- `Qwen4ExpVisionEncoder` porte la tour BF16 complète : patch embedding
  `Conv3d` temporel, positions apprises interpolées, RoPE 2D, 27 blocs
  d’attention/MLP et merger spatial 2×2 vers 2 560 dimensions.
- `Qwen4ExpVisionCheckpointLoader` sélectionne les 333 tenseurs `vision_tower`
  du shard 1 et les injecte strictement sans ouvrir le groupe langage ni les
  128 tables n-gram.
- `Qwen4ExpImageProcessor` accepte une image locale, la redimensionne sur le
  facteur Qwen patch×merge (32), la convertit en RGB NHWC et normalise en
  BF16 dans `[-1,1]`. Le probe `qwen38 flash-vision-probe --image ...`
  permet maintenant de tester une vraie photo.
- Validations réelles : image synthétique 224×224 → `[1,49,2560]`, pic
  1,96 GB ; image locale → `[1,950,2560]`, 914 MB actifs et pic 4,64 GB.
  Aucun poids langage de 79,58 Go ni n-gram de 32 Go n’est chargé par ce
  chemin.
- Prochaine étape : parité vision Python, fusion des embeddings avec les
  marqueurs image (`<|image_pad|>`), puis adaptateur de génération. Le
  runtime public 48 couches reste texte uniquement à ce stade.

### Étape D3m — fusion multimodale et forward Flash-Next complet — 2026-08-30

- `Qwen4ExpMRoPE.multimodalPositionIDs` reprend le contrat multimodal de
  `mlx-vlm` : coordonnées temporelles/hauteur/largeur pour les tokens image,
  division par le merge spatial, horloge texte avant/après l'image et sortie
  `[3, 1, sequence]`. Le test couvre la reprise correcte de l'horloge après
  les marqueurs visuels.
- `Qwen4ExpInputMerger` vérifie le nombre de `<|image_pad|>` avant le scatter,
  puis remplace uniquement ces positions par les embeddings de la vision.
  Les délimiteurs ChatML/vision sont conservés dans la séquence et reçoivent
  leurs positions texte ; aucune image n'est réinjectée implicitement.
- `flash-multimodal-probe` assemble désormais preprocessing → vision →
  MRoPE → fusion → embedding global → 48 couches streamées → réduction
  hyper-connexion → lm-head. Sur l'image locale `licensed-image-2.jpeg`,
  le chemin réel produit `[1, 953, 248320]` (950 tokens image) en 99,91 s,
  avec un pic MLX de 4,12 Go et un pic processus de 32,70 Go.
- Le probe accepte `--trace`. `MLXProfiler` instrumente la vision, les
  globaux, chaque couche et le forward multimodal ; la trace Chrome/Perfetto
  `/private/tmp/qwen4-exp-multimodal.trace.json` a bien été produite et
  contient les événements des 48 couches.
- Validation du round : compilation `xcodebuild`, tests unitaires directs
  `xcrun xctest`, **40 tests passés**, puis forward réel multimodal complet
  sous `caffeinate`. Cette étape valide l'assemblage et le budget mémoire,
  mais pas encore un tokenizer/chat-template, une génération autoregressive,
  le runtime public, la GUI, le serveur LAN ni le MTP `qwen4_exp`.

### Étape D3n — prochain adaptateur public Flash-Next

- Construire un adaptateur de génération qui respecte le contrat existant du
  runtime : tokenizer/chat-template Qwen, positions multimodales par tour,
  cache persistant par conversation et stream de tokens visibles avec TTFT,
  prefill/decode et trace profiler.
- Conserver le chemin séquentiel couche par couche tant que la table n-gram
  complète n'a pas une politique de résidence mesurée ; ne jamais appeler
  `eval(model.parameters())` sur le modèle Flash complet.
- Ajouter une commande/probe texte puis image qui émet réellement du texte,
  avant d’exposer Flash-Next dans le catalogue GUI/serveur. Le MTP Flash reste
  une étape séparée après la validation de ce chemin de génération de base.

### Étape D3o — premier chemin autoregressif réel — 2026-08-30

- Qwen4ExpGreedyGenerator ajoute le contrat minimal de génération greedy :
  prefill d'un prompt déjà rendu, choix argmax, puis appels mono-token qui
  réutilisent les caches GDN/PLE/QSA et l'offset M-RoPE du modèle streamé.
  Les embeddings image sont acceptés uniquement au prefill ; un couple
  incomplet image/token est rejeté explicitement.
- flash-generate-probe charge le tokenizer HF local, applique le template
  texte Qwen ou construit le préfixe ChatML + vision pour une image, puis
  expose --max-new-tokens, --thinking et --trace.
- Validation réelle texte sur le checkpoint 4-bit : 22 tokens de prompt,
  premier token produit (13845, décodé "utral"), TTFT 125,066 s, pic
  processus 29,87 Go. Validation réelle image : 971 tokens après insertion
  de 950 marqueurs image, premier token produit (24, décodé "9"), TTFT
  117,676 s, vision comprise, pic processus 31,72 Go. Les deux traces
  profiler ont été écrites dans /private/tmp/.
- Ces sorties prouvent la couture technique tokenizer → prefill → 48 couches
  → lm-head → token suivant, pas encore la qualité ni la parité Python :
  les tokens isolés ne constituent pas un benchmark, et l'implémentation
  Flash-Next n'est pas encore enregistrée comme LanguageModel/VLM public.
  La prochaine étape est donc le dump Python de référence et la comparaison
  token/logit avant toute intégration GUI ou serveur.

### Étape D3p — parité MRoPE Python/Swift — 2026-08-30

- Le script `scripts/qwen4-exp-mrope-reference.py` produit un fixture
  déterministe sans charger le checkpoint : une séquence ChatML avec six
  marqueurs image issus d'une grille 4×6 pré-merge et trois tokens texte après
  l'image. Il exporte les positions `[3, 1, S]`, les tables cosinus et sinus.
- `Qwen4ExpMRoPEParity` reconstruit les positions avec le même contrat
  multimodal que le runtime, puis compare les tables interleavées partielles
  en float32. La commande est `qwen38 flash-mrope-parity <fixture>`.
- Fixture local :
  `parity/qwen4-exp-mrope-reference.safetensors`. Résultat réel :
  `position_ids` max 0, cosinus max `2.38419e-7`, sinus max `1.78814e-7`.
  La parité MRoPE est donc validée à l'erreur d'arrondi float32.
- `xcodebuild` puis `xcrun xctest` passent avec les fixtures QSA et MRoPE,
  soit **41 tests**. Le prochain module à isoler est la vision Qwen4-exp
  (patch embedding, RoPE 2-D, merger et poids du shard), avant d'interpréter
  les logits ou les tokens produits par le chemin autoregressif.

### Étape D3q — parité vision sur checkpoint réel — 2026-08-30

- Le script `scripts/qwen4-exp-vision-reference.py` charge uniquement le shard
  vision du checkpoint Flash-Next via la référence Python `mlx-vlm`, construit
  une entrée BF16 déterministe 32×64 et exporte les activations de chaque
  jalon : patch embedding, position interpolée, 27 blocs (attention, résidu,
  MLP) et merger.
- Une première divergence importante a été remontée au fixture : l'ordre des
  axes de patch était incorrect. La correction a ramené l'écart du patch
  embedding à 0 et l'écart de la sortie finale à `max 0,00556403`, moyenne
  `0,00106566`.
- `Qwen4ExpVisionParity` et `flash-vision-parity` comparent tous les jalons,
  en tenant compte de la différence de forme `[tokens, hidden]` Python contre
  `[1, tokens, hidden]` Swift. Le dernier bloc montre une amplification
  numérique (max ~80 sur son activation interne), mais la sortie merger reste
  sous la tolérance d'intégration et le point est conservé comme diagnostic,
  pas masqué par une moyenne globale.
- `xcodebuild` puis `xcrun xctest` passent avec les trois fixtures : **42 tests**.
  La prochaine parité prioritaire est le premier bloc langage complet / logits
  sur un fixture à poids, puis la validation token-à-token ; la sortie greedy
  actuelle reste explicitement non qualifiée.

### Étape D3r — parité du premier bloc langage réel — 2026-08-30

- `scripts/qwen4-exp-language-reference.py` instancie seulement la couche 0
  du checkpoint avec les mêmes linears affines 4-bit (groupe 32), Gated
  DeltaNet, hyper-connexions quatre voies et MoE 512 experts que la référence
  Python. Le fixture contient l'entrée, les frontières attention/MoE et les
  deux états de cache.
- `Qwen4ExpLanguageParity` et `flash-language-parity` relisent ce fixture et
  chargent uniquement les shards nécessaires à la couche 0. La comparaison
  est déterministe après matérialisation des tenseurs safetensors ; l'écart
  observé sur la sortie complète est `max 0,28219`, moyenne `0,0276238`.
- Une cause de non-déterminisme a été corrigée dans les constructeurs
  préquantifiés Swift et dans `SwitchGLU` local : `scales` et `biases` ne
  doivent jamais partager la même allocation. Le loader matérialise aussi
  chaque tenseur sélectionné avant de libérer le dictionnaire du shard.
- La parité de ce bloc est bornée à une tolérance de 1,0, mais elle ne qualifie
  pas encore les logits ni les tokens. La prochaine étape est d'ajouter les
  sorties du modèle complet (embedding/PLE, couches 1..47, mixer final et
  lm-head) et une comparaison token-à-token avant de réactiver la génération
  Flash-Next dans le chemin public.

### Étape D3s — parité des globaux et couture single-layer — 2026-08-30

- `scripts/qwen4-exp-global-reference.py` exporte un fixture déterministe du
  checkpoint réel avec seulement l'embedding quantifié, le mixer final des
  quatre flux et le `lm_head`. `flash-global-parity` mesure séparément les
  trois sorties ; le résultat observé est exact (`max 0`, `mean 0` pour les
  trois).
- `scripts/qwen4-exp-single-layer-reference.py` assemble ces globaux avec la
  couche 0 Python et exporte les sorties embedding, couche, réduction et
  logits. `flash-single-layer-parity` relit le chemin Swift streamé et ferme
  la couture sans charger les 48 couches simultanément.
- Résultat réel 4-bit groupe 32 : embedding `max 0`, couche 0 `max 0,879883`,
  réduction finale `max 14,6406` (moyenne `1,16275`), logits `max 6,01953`
  (moyenne `0,852625`). La mesure est reproductible mais reste une parité
  **bornée et diagnostique**, pas une qualification token/logit : la
  normalisation GDN et l'amplification du mixer final doivent encore être
  expliquées avant de déclarer le chemin Flash-Next équivalent à Python.
- `QWEN38_SINGLE_LAYER_FIXTURE` et `QWEN38_GLOBAL_FIXTURE` couvrent ces deux
  comparaisons dans la suite de tests directe. La suite doit continuer à être
  exécutée avec `xcodebuild`, puis `xcrun xctest` en largeur de parallélisme 1.
- Le harness Python a aussi été corrigé pour appeler réellement la
  normalisation L2 Qwen4 dans la référence GDN lorsque `mlx-vlm` installé est
  plus ancien ; il ne modifie pas le paquet Python installé.
- Validation finale de ce round : `xcodebuild test` puis `xcrun xctest` direct,
  **45 tests passés**.
- Suite immédiate : instrumenter les frontières GDN/MoE de la couture
  single-layer pour localiser l'écart, puis produire une référence
  token-à-token sur le chemin complet avant toute intégration MTP ou
  réactivation publique Flash-Next.

### Étape D3t — parité GDN publique et couture de continuation — 2026-08-31

- Le fixture single-layer couvre désormais le préremplissage de trois tokens,
  la poursuite d'un token avec le même état récurrent, les deux états de cache
  GDN et l'appel public de `Qwen4ExpDecoderLayer`.
- Les projections, la convolution, les tenseurs `q/k/v` normalisés, la sortie
  GDN et l'état récurrent sont alignés avec Python à l'erreur flottante près.
  La sortie de couche manuelle et l'appel public sont tous deux à `max
  0,00219727`; l'état GDN de continuation est à `max 2,38419e-7`.
- Cause corrigée dans `Qwen4ExpGatedDeltaNet` : le scale `1/sqrt(Dk)` est
  appliqué à `q` après la normalisation L2, mais pas à `k`. Le scale sur `k`
  produisait une dérive publique de `max 0,135498` malgré un probe manuel
  trompeusement correct.
- Les sorties du mixer final et du `lm_head` restent une mesure bornée de la
  couture (réduction `max 0,3125`, logits `max 0,179688`) et ne constituent
  toujours pas une qualification de génération complète.
- La régression est maintenant stricte sur GDN, les caches et les deux chemins
  de couche dans `qwen4ExpSingleLayerPythonFixtureParity`. La prochaine étape
  est une parité token/logit sur plusieurs couches sélectionnées, puis le
  raccordement au générateur public Flash-Next ; MTP reste séparé.

### Étape D3u — parité des appels publics GDN/QSA — 2026-08-31

- Le probe public générique `flash-public-layer-parity` couvre maintenant une
  couche GDN réelle (couche 2) et une couche QSA réelle (couche 3), chacune
  depuis l'embedding partagé et avec les poids affines 4-bit du checkpoint.
- Les frontières internes sont comparées : hyper-connexion, q/k/v, RoPE,
  sortie GDN ou SDPA, gate, projection et MoE isolé. La couche 2 donne
  `max 0,0949707` sur la sortie complète ; l'appel GDN et le MoE isolé restent
  exacts à l'erreur flottante près. La couche 3 donne `max 0,0377916` après
  correction du chemin QSA.
- Deux divergences réelles ont été corrigées : le masque causal implicitement
  ajouté en préfill court alors que la référence laisse `mask=nil`, et les
  positions texte MRoPE qui doivent être dupliquées sur les trois axes, pas
  remplacées par deux axes nuls. Les frontières q/k/v avant RoPE sont désormais
  exactes ; l'écart RoPE résiduel est inférieur à `0,03` sur ce fixture.
- `xcodebuild build-for-testing`, puis `xcrun xctest` direct avec le checkpoint
  et les fixtures réelles passent : **46 tests**. La tolérance de couche 2 est
  explicitement bornée à `0,15` car l'amplification BF16 du mixer après le
  chemin quantifié n'est pas une preuve de parité logits.
- Étape suivante : fixture multi-couches et token/logit sur une courte séquence,
  avant d'autoriser le générateur Flash-Next public. Le MTP `qwen4_exp` reste
  volontairement hors de ce jalon.

### Étape D3v — chaîne multi-couches Flash-Next : gate token/logit ouvert — 2026-08-31

- Le script `scripts/qwen4-exp-selected-layers-reference.py` et le probe Swift
  `Qwen4ExpSelectedLayersParity` comparent maintenant une chaîne réelle des
  couches 0, 1, 2 et 3, puis la réduction des quatre flux et le `lm_head`.
- Sur le checkpoint 4-bit réel, les erreurs de sortie des couches restent
  bornées (`max 0,015625`, `0,015625`, `0,0352783`, `0,0724167`), mais la
  réduction hyper-connection atteint `max 17,1635` et les logits `max 4,36233`.
  Le token suivant Swift/Python est différent (`86394` contre `70608`).
- Désactiver le kernel Metal GDN ne change pas le diagnostic (`réduction
  17,1375`, logits `4,55719`, token toujours différent) : l'écart n'est pas
  attribuable au seul kernel GDN, mais à l'amplification de la dérive dans le
  mélange quatre voies après plusieurs couches.
- La régression reste volontairement un probe diagnostique borné (`<20` sur la
  réduction et `<10` sur les logits), et non une qualification de génération.
  La génération Flash-Next publique et le MTP `qwen4_exp` restent bloqués par
  ce gate token/logit jusqu'à localisation de la dérive ou définition d'une
  tolérance de qualité justifiée.
- `xcodebuild build-for-testing`, puis `xcrun xctest` direct avec le checkpoint
  et tous les fixtures réels passent : **47 tests**. Les messages CoreData/XPC
  du runner sont du bruit système sans impact sur le résultat.

<!-- ASK: étude approfondie de la dérive multi-couches Flash-Next -->
## ASK — analyse approfondie du gate token/logit Flash-Next

À transmettre à l’agent de planification :

Le portage Flash-Next est suffisamment cohérent pour que les briques isolées
soient alignées, mais la chaîne réelle des couches 0→3 n’est pas encore
qualifiée. Avec le checkpoint `Qwen3.8-Flash-Next-MLX-4bit-MTP` et la même
séquence courte, les erreurs de sortie de couches sont respectivement
`0,015625`, `0,015625`, `0,0352783` et `0,0724167`, puis la réduction des quatre
flux atteint `max 17,1635`, les logits `max 4,36233`, et le token suivant
diverge (`Swift 86394`, `Python 70608`). Le fallback GDN donne pratiquement
les mêmes chiffres (`réduction 17,1375`, logits `4,55719`) : le kernel Metal
GDN seul n’explique donc pas l’écart.

Demander une investigation dédiée qui doit :

1. Ajouter ou exploiter des captures Python/Swift à chaque frontière des
   couches 0→3 : entrée, `hc_norm`, `input_mix`, injection, sortie GDN/QSA,
   sortie MoE, état des quatre flux et entrée du mixer final.
2. Comparer séparément erreur absolue, erreur relative, norme, cosinus,
   top-k/logit margin et token suivant, afin de distinguer une dérive de
   précision normale d’une erreur structurelle amplifiée par le mixer.
3. Auditer explicitement l’ordre des opérations et les casts BF16/FP32, les
   reshape/transpositions des quatre flux, le chargement quantifié affine
   (poids/scales/biases), les caches `MambaCache`/`ArraysCache`, les positions
   MRoPE/QSA et le chemin de réduction final.
4. Rejouer le probe avec une couche à la fois, puis avec les couches 0→3 en
   désactivant le kernel GDN et, si la mémoire le permet, avec les projections
   non quantifiées ou en 8-bit. Le test doit identifier la première frontière
   où l’écart devient disproportionné.
5. Produire une décision exploitable : correctif nécessaire, ou tolérance
   justifiée ; seuils de parité retenus ; conditions pour autoriser le
   générateur public et le MTP. Tant que cette décision n’est pas étayée, ne
   pas présenter Flash-Next comme équivalent à la référence Python.

Le résultat attendu est une note de diagnostic et, si nécessaire, un plan de
correctif minimal avec fixtures/tests reproductibles. Toute validation MLX
doit utiliser `xcodebuild` puis `xcrun xctest`, jamais `swift build`.

### Étape D3w — E0/E1/E2 rebasés : mixer isolé validé — 2026-08-31

- E0 est maintenant instrumenté dans `Qwen4ExpSelectedLayersParity` et
  `Qwen4ExpPublicLayerParity` : max absolu, RMS, RMS relatif, cosinus, marge
  top-1 et rangs croisés des tokens. Les fixtures
  `parity/qwen4-exp-rebased-layer-{0,1,2,3}-reference.safetensors` réinjectent
  l'état Python exact de la couche précédente.
- E1 est concluant : le mixer hyper-connection et le `lm_head` Swift, alimentés
  par l'état Python de la dernière couche, sont bit-exacts (`max 0`, cosinus 1)
  et produisent le même token. Le chargement quantifié des globaux n'est donc
  pas la cause de la dérive de la chaîne.
- E2 rebasé donne les erreurs de sortie suivantes : couche 0 `max 0,015625`,
  `rel 0,01746` ; couche 1 `max 0,00390625`, `rel 0,00560` ; couche 2
  `max 0,0220947`, `rel 0,03472` ; couche 3 `max 0,0156886`, `rel 0,02475`.
  Les captures internes montrent que GDN/QSA et le MoE isolé restent exacts ;
  les premiers écarts visibles sont la projection de branche quantifiée puis
  le chemin `mlp_hyper_connection`/MoE, avec amplification de petits écarts
  BF16 dans les quatre flux. La PLE de la couche 1 ne produit pas de divergence
  structurelle une fois le fixture correctement rebasé.
- Le résultat ne justifie pas encore un correctif de production, mais les
  seuils proposés par l'analyse (`rel ≤ 3e-3`) ne sont pas atteints par la
  chaîne 4-bit. Le gate de génération/MTP reste donc ouvert ; la prochaine
  étape est l'ablation E3 en fp32/dequantifié sur les mêmes couches, puis E5
  sur des prompts naturels avec marges de logits mesurées.
- Un oubli du harness E2 a été corrigé : les captures Python de la couche PLE
  doivent être calculées après l'ajout PLE, dans un cache séparé. La suite
  `xcodebuild build-for-testing` puis `xcrun xctest` passe avec **47 tests**.

---

<!-- ANSWER: diagnostic dérive multi-couches Flash-Next — 2026-08-31, mesures nouvelles reproductibles -->
## RÉPONSE — diagnostic de la dérive multi-couches Flash-Next (2026-08-31)

**Verdict en deux phrases.** Les mesures faites aujourd'hui (probe `Scripts/qwen4-exp-mixer-sensitivity.py`, reproductible) montrent que le mixer hyper-connection est un **amplificateur structurel ~×50-100 sur ce fixture** et qu'un bruit gaussien de **rms 0,0005 sur `layer_3` suffit à faire basculer l'argmax** : les chiffres observés (`réduction 17,16`, `logits 4,36`, token rang 44) sont **entièrement compatibles avec une dérive amont de rms ~0,002-0,005 amplifiée par le mixer**, et le gate token/logit actuel est **mathématiquement inatteignable en bf16 sur ce fixture** — la référence Python elle-même le raterait contre une variante de son propre ordre d'opérations. Ce qui reste à prouver (et que le harnais actuel ne mesure pas) : que chaque couche Swift, **à entrée égale**, est au niveau du bruit bf16 — c'est l'expérience E2 ci-dessous, et c'est elle qui décide s'il y a un correctif à faire ou non.

### D1 — Mesures nouvelles (faites ce jour, avec les poids réels du checkpoint Vontra)

Probe : `Scripts/qwen4-exp-mixer-sensitivity.py` (committé, validé), qui recharge embed/mixer/lm_head exactement comme la référence (4-bit affine g32, conforme au `config.json` du checkpoint) et rejoue la réduction sur le `layer_3` du fixture. La source Python de référence est maintenant **vendorée** dans `Scripts/references/vlm_q4_language.py` (sha256 `c301a425…`, elle ne vivait que dans un scratchpad de session temporaire — c'était un risque de reproductibilité).

1. **Sanité : la chaîne Python est bit-exacte.** Réduction et logits reconstruits depuis le checkpoint vs fixture : `max|Δ| = 0,000000` sur les deux. Le fixture et son script de génération sont sains.
2. **Le mixer est un amplificateur géant sur ce fixture.** Statistiques des frontières internes sur le vrai `layer_3` (absmax / rms) : `h = 0,72 / 0,049` → **`hc_norm` = 159 / 4,83** (le RMSNorm groupé divise par un rms minuscule puis multiplie par `1+w`) → `pre_up = 61,8 / 8,0` → `reduced = 34,3 / 2,32`. Gain rms amont→aval ≈ ×50. Une erreur absolue « bornée » de 0,07 sur `h` n'est donc **pas petite** : c'est ~10 % de l'absmax du signal, projetée à travers un gain de ×50.
3. **Sensibilité mesurée** (bruit gaussien sur `layer_3`, 5 tirages/σ) :

   | σ (rms bruit) | max Δreduced | max Δlogits | rms Δlogits | flips argmax |
   |---|---|---|---|---|
   | 0,0005 | 5,97 | 0,78 | 0,12 | 4/5 |
   | 0,001 | 11,03 | 1,63 | 0,22 | 4/5 |
   | 0,002 | 13,53 | 3,14 | 0,47 | 5/5 |
   | 0,005 | 26,84 | 4,59 | 0,83 | 5/5 |

   Les valeurs Swift observées (17,16 / 4,36) tombent entre σ = 0,002 et 0,005 — c'est-à-dire **quelques ulps bf16 d'écart d'ordre d'opérations accumulés sur 4 couches**.
4. **Le fixture est dégénéré et son gate token n'a pas de sens statistique.** Prompt = tokens 10-12 (ids arbitraires, `embedded` absmax 0,02), logits Python quasi plats : top-4 dans un intervalle de 0,16, **marge top-1 = 0,031**. Le token Swift 86394 est au rang 44 Python (Δ 1,22 du top) — un déplacement compatible avec le bruit σ≈0,002 mesuré au point 3 (rms Δlogits 0,47, max 3,1 sur 248 320 coordonnées).
5. **La croissance ×2/couche des erreurs chaînées** (0,0156 → 0,0156 → 0,0353 → 0,0724) est le motif d'une **amplification par rétroaction** (chaque couche relit les flux via `hc_norm` à gain ~×20-100 et réécrit avec un gate d'injection ≤ 2), pas celui d'un bruit indépendant (qui croîtrait en √N). Cela n'incrimine ni n'innocente Swift : un biais systématique minuscule par couche ET du bruit d'ordre d'opérations produisent tous deux ce motif une fois amplifiés. D'où E2.

**Conséquence immédiate** : le harnais `Qwen4ExpSelectedLayersParity` chaîne les sorties Swift (les erreurs se composent — c'est voulu et c'est bien) mais ne rapporte que le **max absolu**, la seule métrique qui ne permet PAS de conclure ici. Et il n'exonère pas encore le chemin mixer/lm_head **Swift** (ma sanité n'exonère que le Python ; le code Swift du mixer est identique ligne à ligne à la référence — vérifié — mais son chargement quantifié doit être testé par E1).

### D2 — Protocole d'investigation (répond aux points 1-4 de l'ASK ; ordre strict, chaque étape conclut)

- **E0 — Métriques d'abord (0,5 j)** : étendre le harnais pour rapporter, à chaque frontière : `max|Δ|`, `rms(Δ)`, **`rel = rms(Δ)/rms(ref)`**, cosinus, et pour les logits : marge top-1 Python, rang du token Swift chez Python et réciproquement. Sans cela, aucune décision n'est possible. (ASK point 2.)
- **E1 — Mixer Swift isolé (0,5 j, discriminant)** : passer le `layer_3` **Python** du fixture dans `reduceHyperStreams` + `logits` **Swift** (3 lignes de variante dans le harnais). Attendu si le chargement quantifié Swift du mixer/lm_head est sain : `rel ≤ 1e-3`. Sinon : le bug est dans le chargement affine g32 des poids globaux côté Swift — corriger là, et tout le reste se réévalue.
- **E2 — Parité re-basée par couche (1 j, LE test décisif)** : pour chaque couche i ∈ {0..3}, entrée = `layer_{i-1}` **Python** (le fixture actuel contient déjà tout — aucun nouveau dump), sortie Swift comparée à `layer_i` Python. À entrée égale, l'amplification par rétroaction disparaît. Attendu si le portage est sain : `rel ~1e-3` **stable d'une couche à l'autre** (pas de croissance). Une couche à `rel ≥ 1e-2` = **première frontière fautive** → descendre dans ses captures internes (hc_norm, input_mix, injection, GDN/QSA, MoE — les captures publiques existent depuis le probe public-layer) pour isoler la sous-frontière. (ASK points 1 et 4.)
- **E3 — Ablations de précision (1 j, seulement si E2 montre des `rel` élevés uniformes)** : rejouer E2 avec les flux castés fp32 avant/après chaque couche, des deux côtés (harnais uniquement, pas le code de prod). Si les `rel` s'effondrent → bruit bf16 d'ordre d'opérations, **pas de correctif à faire** ; s'ils persistent → erreur structurelle, retour à E2 ciblé. Le volet « BF16/8-bit » de l'ASK est adapté aux moyens réels : il n'existe **aucun checkpoint Flash-Next bf16/8-bit sur le disque** (le bf16 ferait ~250 Go) ; l'équivalent honnête est le harnais E3 en dequant-fp32 des couches 0-3 (faisable en RAM), même comparaison. (ASK points 3 et 4.)
- **E4 — Audit ciblé des casts/ordre des ops** : à ne lancer que si E1/E2 désignent une frontière. Suspects par ordre de vraisemblance, vérifiés contre `Scripts/references/vlm_q4_language.py` : (1) chargement affine g32 des poids globaux Swift (E1 le teste) ; (2) dtype de portage des 4 flux entre couches (bf16 attendu, pas de cast fp32 intermédiaire qui n'existe pas en Python) ; (3) `hc_norm` groupé : fp32 interne puis re-cast bf16 — l'ordre `(1+w)` après `rsqrt` (vérifié identique aujourd'hui) ; (4) division par `hc_count` **avant** silu/sigmoid (vérifié identique) ; (5) `mean(axis:-2)` vs somme/4 (ordre de réduction) ; (6) reshape `[.., hc_count, hidden]` — ordre des flux dans le tuilage initial. Les points 3-6 sont déjà alignés ligne à ligne (revue faite ce jour) ; ils ne redeviennent suspects que si E1 est vert et E2 rouge sur le mixer d'une couche.
- **E5 — Gate réaliste (0,5 j)** : nouveau fixture « vrai prompt » (template chat appliqué, 16-32 tokens naturels, plus tard +1 variante image) et continuation greedy de 32 tokens, générés par le script de référence existant étendu. Comparaison : tokens identiques **là où la marge top-1 Python > 0,25** ; les positions sous la marge sont rapportées comme ex-aequo (avec marge et rang), pas comme des échecs. Rapport `rel` des logits à chaque pas.

### D3 — Décision de qualification proposée (ASK point 5)

- **Seuils de parité retenus** (re-basés, par frontière) : `rel ≤ 3e-3` et cosinus ≥ 0,9999 par couche (E2) ; `rel ≤ 1e-3` sur le mixer isolé (E1) ; logits globaux re-basés `rel ≤ 2e-2`. Ces seuils correspondent au bruit bf16 d'ordre d'opérations mesuré (D1.3) avec une marge ×2-3.
- **Le gate token/logit actuel (fixture dégénéré, seuils <20/<10) est requalifié en diagnostic borné** — c'est déjà son statut dans le log — et **remplacé comme gate de déblocage par E5**. Un test de token sur un prompt à marge 0,03 ne peut servir de critère à rien.
- **Déblocage du générateur public et du MTP Flash-Next** : E1 vert + E2 vert (4 couches) + E5 vert sur 3 prompts réels → lever le blocage. E2 vert signifie qu'étendre aux couches 4-47 est une formalité (même harnais, fixture `--layers` élargi par tranches).
- **Si E1 ou E2 est rouge** : correctif nécessaire, localisé par la frontière fautive (E4) ; le blocage reste, et la correction se valide en rejouant E0→E2 uniquement.
- **Tant que E2 n'a pas tourné : le blocage demandé par Vincent reste en place tel quel.** Rien dans le présent diagnostic ne « blanchit » le portage — il établit seulement que les chiffres actuels ne prouvent **ni** un bug **ni** son absence, et fournit l'expérience qui tranche.

**Livrables déposés ce jour** : `Scripts/qwen4-exp-mixer-sensitivity.py` (probe, validé contre le checkpoint réel), `Scripts/references/vlm_q4_language.py` + `README.md` (référence vendorée, sha256 consigné). À l'équipe : E0→E2 d'abord (2 jours), décision ensuite. Validation via `Scripts/run-tests.sh` (`xcodebuild` + `xcrun xctest`), jamais `swift build`.

### Étape D3x — E3 déquantifié FP32 : la dérive est quantificationnelle — 2026-08-31

- E3 a été exécuté avec un protocole symétrique : Python et Swift utilisent les
  mêmes poids du checkpoint 4-bit, déquantifiés avec les mêmes paramètres affine
  g32, puis évalués en FP32. Comparer uniquement un côté déquantifié à l'autre
  côté 4-bit aurait été invalide et a été écarté.
- Résultats E2 rebasés, appel public Swift contre la référence Python FP32 :
  couche 0 `max 2,38e-7`, `rel 1,16e-7`, cosinus `1` ; couche 2
  `max 2,86e-6` dans les frontières intermédiaires, `rel 2,0e-7`, cosinus
  `1` ; couche 3 `max 9,54e-6` dans le chemin MLP, `rel 2,0e-7`, cosinus
  `1`. Les sorties finales sont respectivement `2,38e-7`, `7,45e-8` et
  `7,45e-8` au maximum.
- Les frontières GDN/QSA, RoPE, mixer hyper-connection, routage MoE et sortie
  restent exactes ou au bruit FP32. L'erreur 4-bit rebasée de D3w
  (`rel 0,0175` à `0,0347`) disparaît donc avec les poids déquantifiés. La
  cause qualifiée est l'amplification des arrondis/erreurs de quantification
  dans le chemin 4-bit, pas une divergence structurelle du portage Swift.
- La couche 1 n'est pas incluse dans cette ablation : sa table PLE/n-gram est
  trop volumineuse pour être déquantifiée de manière pratique dans ce harnais
  local. Elle reste couverte par E2 4-bit (`rel 0,00560`) et par les tests de
  chargement. Cela ne remet pas en cause la conclusion sur les projections
  quantifiées des couches 0, 2 et 3.
- Décision : aucun correctif structurel E4 ne doit être lancé sur la base de
  D3w. Le gate de génération/MTP passe à E5 : prompts naturels, marges de
  logits, comparaison 4-bit réelle et qualification de l'impact MTP. Le
  chemin déquantifié reste un oracle de diagnostic, pas un mode de production.

### Étape D3y — E5 naturel : première divergence localisée au MoE de la couche 4 — 2026-08-31

- Le script de référence accepte désormais `--prompt` et rend le chat
  template du checkpoint localement. Un prompt naturel de 29 tokens a été
  évalué sur les 48 couches avec le checkpoint 4-bit réel.
- Le premier smoke test du générateur Swift complet fonctionne : 48 couches,
  un token produit, pic MLX `34,49 Go`. Sur ce chemin couche-par-couche, le
  TTFT mesuré est `85,4 s`; ce chiffre est un diagnostic de correction et non
  une mesure de performance finale.
- La référence Python donne le token `11` avec une marge top-1 de `1,75`,
  tandis que Swift donne `16`. L'E1 final reste sain sur l'état Python :
  `rel 0,00195`, cosinus `0,999993`, et le mixer Swift choisit bien `11`.
- Le probe 48 couches indique une première hausse rebasée à la couche 4
  (`rel 0,0360`). Le diagnostic public couche 4 montre que les projections
  sont proches, mais que le top-k MoE diverge (`moe_indices rel 0,132`,
  `moe_output rel 0,0751`). L'ablation symétrique FP32 de cette même couche
  retombe à `rel 1,37e-7` sur `mlp_mixed`, indices identiques, et `rel
  6,13e-7` sur la sortie MoE.
- Conclusion E5 intermédiaire : le mismatch naturel est reproductible et
  vient de la sensibilité du routage MoE au chemin 4-bit entre les deux
  runtimes, pas d'une erreur structurelle de la couche 4. Le gate « token
  identique » reste rouge en 4-bit malgré une marge Python confortable. Il
  faut maintenant mesurer plusieurs prompts et comparer les variantes 4-bit,
  8-bit et BF16 quand elles sont disponibles avant d'activer le MTP comme
  accélération qualifiée.
- Correction harness déposée : le diagnostic public ne suppose plus une
  séquence de longueur 3 (`reshape(batch, sequence, -1)`).
- Contrôle de version effectué : Python MLX `0.31.2` puis `0.32.0` donnent
  les mêmes indices et les mêmes écarts couche 4 face à Swift MLX `0.31.6`.
  La disponibilité PyPI ne permet pas d'installer Python MLX `0.31.6`, mais
  le résultat stable sur deux backends Python écarte un simple artefact de
  version mineure. La qualification finale doit donc traiter le chemin
  quantifié Swift comme une variante à comparer par qualité, et non supposer
  une identité bit-à-bit Python.

<!-- REQUEST: analyse divergence quantifiée MoE Flash-Next -->
## REQUEST — validation de la divergence 4-bit inter-runtime du MoE Flash-Next

À partir des résultats E3/E5 ci-dessus, analyser la divergence observée sur la
couche 4 entre Python MLX et Swift MLX : `moe_indices rel 0,132` en 4-bit,
contre des indices identiques et une sortie quasi exacte en FP32 déquantifié.

Déterminer si l'écart peut venir de `gatherQuantizedMM`, du dtype/cast, de
l'ordre du top-k/argPartition, du chemin `SwitchLinear` quantifié ou d'une
différence de backend MLX. Proposer le protocole minimal permettant de trancher
entre :

1. corriger le chemin quantifié Swift ;
2. considérer cette différence comme une variation inter-runtime acceptable ;
3. modifier le critère de qualification par qualité plutôt que par identité
   bit-à-bit.

Inclure une recommandation concrète pour la matrice 4-bit/8-bit/BF16 et le gate
MTP, en tenant compte des variantes déjà présentes dans
`/Volumes/Lexar/models/mlx-community/` et de la contrainte de validation
`xcodebuild`, puis `xcrun xctest`.

---

<!-- ANSWER: divergence MoE 4-bit inter-runtime — 2026-08-31, tranché par mesures nouvelles -->
## RÉPONSE — divergence MoE 4-bit : verdict, matrice et gate MTP (2026-08-31)

**Verdict : option 2 + 3.** La divergence d'indices MoE est une **variation inter-noyaux banale, pas un bug du chemin quantifié Swift**, et la qualification Flash-Next doit passer **par la qualité, pas par l'identité bit-à-bit**. Le protocole minimal demandé a été **exécuté aujourd'hui** (pas seulement proposé) ; il clôt la question. Aucun correctif E4 à lancer.

### V1 — Les deux mesures qui tranchent (probe `Scripts/qwen4-exp-moe-routing-stability.py`, committé et validé)

1. **La frontière top-10/top-11 du routeur est plus fine que tout bruit numérique.** Sur les 29 positions du prompt naturel (fixture couche 4, `moe_probabilities` Python) : gap absolu **médian 2,4e-4**, 27/29 positions sous 1e-3, 5/29 sous 1e-4, et **4/29 positions à exactement 0** — des ex-aequo parfaits en bf16. À ces positions, même des logits bit-identiques avec un ordre de tie-break différent changent l'appartenance des experts. L'invariance inter-runtime du top-k est donc **mathématiquement impossible** sur ce modèle (512 experts, probas top-10 ~0,008).
2. **Python diverge déjà d'avec lui-même entre ses propres backends.** Même runtime, même version MLX, mêmes poids (routeur couche 4, **non quantifié** — vérifié : pas de scales/biases), même entrée (`moe_input` du fixture) : le backend CPU et le backend GPU donnent **des ensembles d'experts différents à 2 positions sur 29** (max|Δprobas| = 8,8e-4). La divergence Python↔Swift observée est donc de même nature et de même ordre que la divergence Python↔Python. Il n'y a rien à « corriger » côté Swift.

### V2 — Les suspects de l'ASK, un par un

| Suspect | Verdict | Preuve |
|---|---|---|
| `gatherQuantizedMM` / `SwitchLinear` quantifié | **Innocenté** | Ablation FP32 symétrique (D3x/D3y) : indices identiques, sortie MoE `rel 6,1e-7` |
| dtype/cast du routeur | **Innocenté** | Routeur non quantifié des deux côtés ; `softmax(precise: true)` fp32 identique ligne à ligne (revue Swift `Qwen4ExpSparseMoE` vs `Qwen3_5MoeSparseMoeBlock` faite ce jour) |
| Ordre du top-k / `argPartition` | **Non-problème, mais métrique à corriger** | L'ordre interne de la tranche `argpartition` est non spécifié dans les deux runtimes ; seule l'**appartenance** compte (les scores sont renormalisés sur l'ensemble sélectionné). Corollaire : `moe_indices rel 0,132` compare des **ids** numériquement — métrique invalide. Remplacer par le comptage d'experts différents (différence d'ensembles), comme dans le probe déposé |
| Backend/version MLX | **Cause confirmée** | CPU vs GPU du même Python MLX : 2/29 positions d'appartenance différente ; Python 0.31.2/0.32.0 vs Swift 0.31.6 = noyaux Metal différents, même classe d'écart |

Mécanisme complet : les noyaux quantifiés (versions MLX différentes) produisent un bruit ~1e-3 rel sur le hidden pré-routeur → les probas du routeur bougent de ~1e-4-1e-3 → les experts au bord du top-10 basculent (frontière médiane 2,4e-4) → `moe_output rel 0,075` → amplifié ensuite par les hyper-connections (mécanisme déjà quantifié dans la RÉPONSE précédente, gain ~×50).

### V3 — Matrice 4-bit / 8-bit / BF16 recommandée

- **27B (Jalon 1)** : matrice complète — les six variantes sont déjà sur le Lexar (`4bit`, `8bit`, `bf16` + les trois drafters `-MTP-*`). Le 27B n'a **ni MoE routé ni hyper-connections** : les deux amplificateurs de chaos sont absents, le gate token-à-token greedy vs Python (§4.4) y **reste le critère**, avec le rapport de marge en garde-fou.
- **Flash-Next : qualification sur le 4-bit Vontra uniquement.** Le reste de la matrice est **matériellement impossible et inutile** : 8-bit complet ≈ 130 Go > RAM ; BF16 ≈ 250 Go > les 192 Go libres du Lexar ; et surtout **l'oracle FP32-déquantifié symétrique (E3) est supérieur à un checkpoint BF16** pour la question posée — il compare les deux runtimes à poids strictement identiques, ce qu'un BF16 original ne ferait pas mieux. Ne rien télécharger. (La qualité intrinsèque du quant 4-bit vs l'original est une autre question — c'est la GATE G-4/§7, hors périmètre ici.)
- **Métriques à corriger dans le harnais** (héritées de E0) : différence d'ensembles pour les indices MoE ; `rel`/cosinus partout ; marge top-1 sur les logits.

### V4 — Gate MTP et générateur public (redéfinition)

Découpler deux choses que le gate actuel confond :

1. **Correction du MTP = auto-cohérence Swift, pas parité Python.** Le décodage spéculatif doit être **token-identique à la génération greedy Swift sans MTP** — c'est sa définition mathématique, et ça ne dépend en rien de l'écart inter-runtime. Gate MTP : (a) greedy MTP-on ≡ MTP-off sur N prompts naturels (texte + image), (b) rejet partiel + rollback GDN exercés et vérifiés, (c) accept rate et débit consignés. Ce gate peut passer au vert **indépendamment** de la qualification Python.
2. **Qualification du générateur public = qualité vs Python, marge en tête.** Trois volets, tous sur le 4-bit Vontra :
   - **Oracle structurel** : E3 FP32 symétrique — **déjà vert** (rel ~1e-7, indices identiques).
   - **Accord teacher-forced sous marge** : sur 10 prompts naturels (dont 2 image), forward Swift et Python sur les mêmes positions ; exiger l'accord argmax **uniquement où la marge top-1 Python > 0,5** (proposé : ≥ 99,5 % de ces positions) ; sous la marge, rapporter la distribution (marge, rang croisé) sans en faire un critère. Un forward par prompt, pas de génération : compatible avec le TTFT diagnostic actuel de 85 s.
   - **Qualité de tâche** : sous-ensemble MMLU-Pro (~50-100 questions) + les prompts qualitatifs du protocole §7, Swift vs Python même checkpoint : écart ≤ 2-3 points. C'est le critère du plan §7 appliqué au portage.
3. **Déblocage** : générateur public quand les trois volets du point 2 sont verts ; MTP quand le point 1 est vert **et** que le générateur est débloqué. Le gate « token identique vs Python sur la chaîne 4-bit » est **retiré** (impossible par construction, cf. V1) ; les tests actuels restent en diagnostic borné.

**Livrable déposé** : `Scripts/qwen4-exp-moe-routing-stability.py` (marges de frontière + routeur CPU/GPU, validé sur le checkpoint réel). Validation des volets Swift via `Scripts/run-tests.sh` (`xcodebuild` puis `xcrun xctest`), jamais `swift build`.

### Étape V5 — gate 27B et MTP conversationnel — 2026-08-31

- Le correctif de métrique MoE est intégré dans
  `Sources/Qwen38Core/FlashNext/Qwen4ExpPublicLayerParity.swift` : les
  indices `argPartition` ne sont plus comparés par leur ordre arbitraire ; le
  rapport compte désormais les positions dont l'ensemble d'experts diffère.
  Le probe couche 4 donne `3/29` positions différentes et `3` affectations,
  en cohérence avec l'analyse inter-runtime du plan.
- Validation Swift après ce changement : `xcodebuild build-for-testing`, puis
  `xcrun xctest` direct avec les fixtures réelles, **47 tests passés**.
- Gate 27B 4-bit exécuté avec l'image disponible
  `/Users/vincent/Downloads/licensed-image-2.jpeg`, greedy, 16 tokens/tour,
  sur les trois prompts de référence : standard et MTP produisent des textes
  **identiques sur les trois tours**. Le M1 upstream est actif au tour 1 avec
  `7/8` tokens acceptés ; les tours 2 et 3 passent en fallback documenté car
  le replay upstream ne transporte pas les positions MRoPE du préfixe image.
- Gate M2 local exécuté sur la même conversation : le drafter reste actif sur
  les trois tours, avec `9/11`, `9/12` et `8/13` tokens acceptés. C'est le
  premier résultat positif du gate d'auto-cohérence conversationnelle locale
  avec image, et il confirme la réutilisation du cache/état M2 au tour suivant.
- Le gate MTP du **27B local** est donc vert pour cette matrice smoke. Le
  fallback M1 multimodal reste un point d'intégration à traiter séparément
  si l'on veut que le serveur conserve M1 sur les tours textuels après image ;
  il ne doit pas être confondu avec un échec de parité greedy.
- La même matrice smoke passe aussi en **8-bit** : trois sorties
  standard/MTP identiques, M1 actif au premier tour (`7/8` acceptés), puis le
  fallback MRoPE attendu sur les tours suivants. Le smoke BF16 a été lancé
  avec les mêmes paramètres mais n'a pas produit de résultat après environ
  13 minutes ; le processus a été arrêté proprement. BF16 est donc **non
  conclu** pour ce gate, et non déclaré en échec. Il faudra le mesurer avec un
  probe dédié à un seul tour/quelques tokens avant de l'inclure dans la
  matrice de performance.
- La qualification Flash-Next reste séparée : la divergence MoE 4-bit est
  qualifiée par qualité/inter-runtime, mais aucun générateur Flash-Next public
ni MTP Flash-Next n'est débloqué par ce test 27B.

### Étape V6 — probe BF16 borné — 2026-08-31

- Un probe BF16 d'un seul tour image et d'un token donne le même résultat que
  le chemin standard (`L`). Le probe à deux tokens donne également une sortie
  identique (`L'utilisateur`).
- Ces limites sont volontairement trop courtes pour déclencher un round M2 :
  les deux exécutions rapportent `0/0` tokens proposés/acceptés. BF16 est
  donc validé ici pour la cohérence du chemin court, mais pas pour une mesure
  de taux MTP. Les taux MTP exploitables restent ceux des gates 4-bit/8-bit et
  du probe M2 conversationnel.

### Étape V7 — décision sur le replay M1 multimodal — 2026-08-31

- Le fallback M1 observé aux tours texte suivant un premier tour image est
  confirmé comme une limitation de contrat, pas comme un défaut de calcul :
  `MLXLMCommon.generate` peut rejouer le prompt cible mais le drafter M1
  upstream ne reçoit pas la table de positions M-RoPE 3 axes associée aux
  embeddings vision.
- Aucun correctif local opportuniste ne doit être appliqué au chemin M1 :
  forcer le drafter à poursuivre avec un préfixe texte incomplet invaliderait
  le cache privé et pourrait produire une accélération silencieusement fausse.
- Décision d’architecture : après une conversation multimodale, le chemin
  persistant officiel est **M2 local**, qui conserve cible, drafter, états GDN,
  positions et ledger au même endroit. M1 reste le chemin rapide de secours
  pour les conversations texte ou le premier tour image froid ; son fallback
  est exposé dans les métriques et l’API.
- Prochaine implémentation : vérifier que le serveur utilise ce choix avec une
  session persistante explicite, sans partager par erreur le cache d’un client
  avec un autre, puis lancer la matrice de qualification Flash-Next par
qualité. La modification d’API upstream M1 est hors périmètre du portage.

### Étape V8 — session persistante et métriques serveur — 2026-08-31

- Le serveur accepte désormais `conversation_id` au niveau de la requête, ou
  dans `extra.conversation_id`. Une continuation est reconnue uniquement si
  l'identifiant, le modèle, les paramètres compatibles et le préfixe complet
  des messages correspondent ; sinon le runtime est réinitialisé et le replay
  reste contrôlé.
- Le serveur n'héberge toujours qu'un cache cible à la fois. Un changement de
  conversation ou de modèle ne peut donc pas exposer le contexte d'un autre
  client : il invalide explicitement le cache actif.
- `Qwen38ServerSession` expose maintenant `conversationID`, `cacheReused`,
  `conversationReplayed`, le statut MTP et les compteurs proposés/acceptés,
  ainsi que le taux d'acceptation dans `/metrics`.
- Smoke HTTP réel sur le port isolé `8859`, modèle 4-bit : `/v1/models` a
  publié les variantes 4/8-bit/BF16 ; deux tours avec `conversation_id` ont
  donné `cacheReused=false` puis `cacheReused=true`, sans replay au second
  tour. Le serveur de test a été arrêté après validation.
- Validation après le changement : `xcodebuild build-for-testing`, puis
  `xcrun xctest` direct, **47 tests passés**. Le prochain jalon est la
  qualification qualité Flash-Next, pas une modification opportuniste de M1.

### Étape V9 — smoke Flash-Next multimodal — 2026-08-31

- Le binaire `flash-generate-probe` a été rejoué avec le checkpoint réel
  `Qwen3.8-Flash-Next-MLX-4bit-MTP` et l'image locale
  `/Users/vincent/Downloads/licensed-image-2.jpeg`.
- La chaîne technique complète fonctionne : vision `1216x800`, `950`
  marqueurs image, prompt `971` tokens, globals, 48 couches, réduction et
  premier token généré (`2005`, décodé `“`). Le profiler a écrit
  `/private/tmp/qwen38-flash-smoke.trace.json`.
- TTFT mesuré `100,524 s`, dont `100,521 s` de prefill ; pic MLX `2,58 Go`
  et pic processus `36,28 Go`. Le chemin est donc exécutable dans le budget
  du Mac, mais la performance actuelle est celle d'un adaptateur de
  validation couche-par-couche, pas celle d'un runtime optimisé.
- Le token isolé n'est pas une mesure de qualité. Le générateur Flash-Next et
  son MTP restent bloqués jusqu'au protocole qualité multi-prompts prévu dans
  V4 ; aucun déblocage n'est inféré de ce smoke.

### Étape V10 — séparation préfill / décodage Flash-Next — 2026-08-31

- Le micro-benchmark multimodal à deux tokens a été exécuté sur le checkpoint
  réel `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, avec la même image locale.
  Résultat : 971 tokens de prompt, TTFT `129,487 s`, préfill `129,485 s`,
  décodage `105,998 s` pour le second token, pic MLX `37,81 Go`.
- Le premier smoke à un token mesurait `100,524 s`. Le second token coûte donc
  environ `106 s` : le décodeur recharge les 48 couches à chaque appel
  autoregressif. Il s'agit d'une limite connue de l'adaptateur de validation
  couche-par-couche, pas d'une mesure du débit attendu d'un runtime public.
- Le Flash-Next reste exécutable dans le budget mémoire du Mac, mais il ne doit
  pas encore être exposé comme générateur utilisateur. Le prochain travail est
  un choix explicite entre (a) résidence persistante de toutes les couches,
  si le budget réel le permet, (b) cache/LRU configurable, ou (c) réécriture
  d'un forward compilé avec chargement groupé. Ce choix doit être benchmarké
  sur le pic processus, pas seulement `Memory.activeMemory`.
- Une tentative depuis le terminal isolé a révélé un défaut de robustesse du
  probe profilé : un processus neuf peut initialiser le profiler avant le
  device Metal et provoquer un `NSRangeException` MLX. Le correctif ajouté
  dans `FlashGenerateProbe` force `Device.defaultDevice()` avant les phases
  profiler ; la validation de compilation doit être rejouée via le projet
  Xcode généré/ouvrable localement, car ce checkout ne contient actuellement
  aucun `.xcodeproj` et le `xcodebuild` direct ne reconnaît pas `Package.swift`.
- Le prochain gate reste la qualité Flash-Next (prompts naturels, marge des
  logits, comparaison Python/Swift), puis seulement l'optimisation de
  résidence. Aucun MTP Flash n'est activé par ce benchmark.

### Étape V11 — affichage GUI du canal de réponse — 2026-08-31

- La première exécution depuis Xcode confirme le gate MTP 27B local : bloc 2,
  `46/49` tokens acceptés (`93,9 %`), `14,8 tok/s` de décodage, TTFT `2,506 s`,
  pic mémoire `17,07 Go`.
- Le serveur séparait déjà `reasoning_content` et `content`, mais la mini-GUI
  ajoutait les chunks bruts dans la bulle. La GUI passe maintenant par
  `Qwen38ThinkingStreamParser` et n'affiche que le canal `content`, tout en
  laissant le raisonnement disponible dans le runtime/API pour la conversation.
- Avec thinking activé, le parseur est amorcé dans `<think>` ; avec thinking
  désactivé, il démarre directement dans le canal de réponse. Le flush final
  est réalisé sur l'événement de métriques afin de ne pas perdre un marqueur
  coupé entre deux chunks.

### Étape V12 — thinking visible mais escamotable — 2026-08-31

- La mini-GUI conserve maintenant deux champs par message assistant : la
  réponse visible et le raisonnement. Le raisonnement est affiché dans une
  bulle séparée `DisclosureGroup`, repliée par défaut et extensible au clic.
- Le flux tokenisé alimente le parseur thinking pendant la génération ; les
  fragments `reasoning` et `content` sont routés vers leurs bulles respectives,
  y compris le flush final d'un marqueur coupé entre deux chunks.
- Le serveur/API conserve son comportement inchangé : `content` et
  `reasoning_content` restent deux champs distincts.

### Étape V13 — validation des blocs MTP 2/4 et compteur thinking — 2026-08-31

- Le test GUI en bloc 4 donne `256/498` tokens acceptés (`51,4 %`) et
  `3,0 tok/s` sur le troisième tour. Comparé au bloc 2 précédent
  (`869/1177`, `73,8 %`, `9,7 tok/s`), augmenter la profondeur ne paie pas
  encore sur cette conversation ; le bloc 2 reste le preset recommandé en
  attendant une mesure bloc 3 contrôlée.
- Le libellé de la bulle thinking affichait littéralement
  `message.reasoning.count` à cause d'une interpolation Swift manquante. Il
  utilise maintenant `\(message.reasoning.count)` et affiche le nombre réel de
  caractères.

### Étape V14 — E5 naturel complet et préparation du chemin rapide Flash-Next — 2026-08-31

- Le fixture `parity/qwen4-exp-e5-natural-full-reference.safetensors` contient
  bien les 48 couches et 29 tokens du prompt naturel :
  « Explique en français qui est le président de la Chine et quel est son rôle. ».
  Le harnais Swift a été exécuté sur le checkpoint réel 4-bit Vontra, avec le
  chemin chaîné et le chemin E2 rebasé par couche.
- E1 reste sain : mixer Swift alimenté par l'état Python `rel=0,00121662`,
  cosinus `0,999993`; logits Swift sur état Python `rel=0,00194756`,
  cosinus `1,00011`, et top-1 identique (`11/11`, rang 1 des deux côtés).
- E2 rebasé confirme la première frontière sensible à la couche 4 :
  `rel=0,0360479`; les couches suivantes restent affectées par la dérive
  chaînée. La sortie de chaîne choisit `16` contre `11` pour Python, malgré
  une marge Python de `1,75`. Le résultat confirme que le strict token gate
  4-bit est trop exigeant face au routage MoE quantifié ; il ne révèle pas un
  défaut du mixer ou de la tête Swift.
- Le générateur public Flash-Next et son MTP restent donc bloqués. Il faut
  compléter E5 avec deux prompts naturels supplémentaires ou une qualification
  par tâche avant de conclure sur la qualité utilisateur.
- Optimisation préparatoire intégrée dans
  `Sources/Qwen38Core/FlashNext/Qwen4ExpCheckpointLayerLoader.swift` et
  `Qwen4ExpStreamingDecoder.swift` : la table couche → shard est construite
  une seule fois par décodeur au lieu de reparcourir l'index JSON à chaque
  couche et à chaque token. Cette optimisation ne conserve aucun poids en
  mémoire et ne modifie pas les opérations du modèle ; elle prépare le futur
  benchmark de résidence des couches.
- La validation de cette dernière modification n'a pas pu être relancée par
  `xcodebuild` en CLI : le checkout ne contient pas de `.xcodeproj` ni de
  `.xcworkspace`, et la version de `xcodebuild` présente ne sait pas prendre
  directement `Package.swift`. Le binaire Xcode précédent a servi au harnais
  E5 ; la prochaine validation doit être faite depuis le projet Package ouvert
  dans Xcode, ou après génération d'un projet Xcode reproductible. `swift build`
  reste volontairement interdit. Le contrôle syntaxique `xcrun swiftc -parse`
  des deux fichiers modifiés passe ; il ne remplace pas le build Xcode/MLX.

### Étape V15 — build Xcode de l’optimisation Flash-Next — 2026-08-31

- Vincent a reconstruit le Package depuis Xcode et a confirmé que la mini-GUI
  démarre et fonctionne.
- Le binaire produit dans DerivedData contient les symboles
  `Qwen4ExpCheckpointLayerIndex` et la nouvelle signature du loader avec index
  injecté. L’optimisation de parcours de l’index couche→shard est donc bien
  compilée dans le produit Xcode/MLX.
- Aucun calcul Flash-Next long n’est relancé dans ce sous-jalon afin de garder
  le GPU disponible. Suite : compléter E5 multi-prompts, puis mesurer le coût
  réel d’une résidence persistante des couches avant toute activation publique.

### Étape V16 — E5 multi-prompts Flash-Next — 2026-08-31

- Deux nouveaux fixtures naturels 4-bit ont été générés avec le template du
  checkpoint et les 48 couches : transition énergétique (`51` tokens) et
  fonction Swift (`56` tokens). Le fixture présidentiel existant complète la
  série de trois prompts.
- Les trois comparaisons E5 gardent E1 dans la même zone : mixer Swift sur
  état Python `rel=0,00121662`, `0,001785`, `0,00190269`; logits sur état
  Python `rel=0,00194756`, `0,00253786`, `0,00266371`; le top-1 E1 est
  identique dans chaque cas et de rang 1 des deux côtés.
- Les trois chaînes complètes divergent après la couche 4, avec des sorties
  `16/11`, `1710/11` et `258/16837` (Swift/Python). La première frontière
  sensible est donc reproductible, mais le strict token gate n’est pas adapté
  à ce checkpoint 4-bit : le mixer et la tête ne sont pas en cause, le
  routage MoE amplifie la différence de quantification inter-runtime.
- Les données brutes sont conservées dans
  `results/flash-e5-natural-2026-08-31.tsv`. E5 qualité n’est pas vert : le
  générateur public et le MTP Flash-Next restent bloqués. La suite doit être
  une qualification par tâche (réponses complètes Swift/Python) ou une étude
  d’un checkpoint Flash moins sensible, pas un relâchement silencieux du gate.

### Étape V17 — mode de résidence expérimentale Flash-Next — 2026-08-31

- `Qwen4ExpLayerLoadingMode` propose maintenant `streamed` (valeur par défaut)
  et `resident`. En mode résident, les couches déjà chargées sont conservées
  entre les appels autoregressifs : le coût de rechargement par token peut donc
  être mesuré directement.
- Le CLI Flash ajoute `--resident-layers`. Le mode est explicitement opt-in,
  car le checkpoint complet peut dépasser le budget mémoire selon le Mac et la
  variante quantifiée. `unloadResidentLayers()` permet de libérer les poids
  expérimentaux sans détruire les caches de conversation.
- Le contrôle syntaxique des trois fichiers modifiés passe. Il faut maintenant
  reconstruire depuis Xcode avant d’exécuter ce mode ; aucun essai résident
  n’est lancé automatiquement pour éviter un dépassement mémoire non sollicité.

### Étape V18 — compilation Xcode du mode résident — 2026-08-31

- Vincent a reconstruit le Package depuis Xcode.
- Le binaire GUI produit dans DerivedData contient `Qwen4ExpLayerLoadingMode`
  et le stockage interne `residentLayers`. Le mode expérimental est donc
  effectivement compilé dans le produit MLX.
- La mesure n’est pas encore lancée : le prochain essai doit utiliser le
  target CLI Flash-Next avec `--resident-layers`, sur un seul token et avec le
  pic mémoire observé, avant tout essai autoregressif plus long.

### Étape V19 — mesure streaming contre résidence complète — 2026-08-31

- Le nouveau binaire CLI Xcode a été vérifié avec `flash-generate-probe
  --resident-layers`.
- Sur le même prompt texte de 23 tokens et le même checkpoint 4-bit, le mode
  résident termine un premier forward : TTFT `136,106 s`, préfill
  `136,089 s`, mémoire MLX active `111,14 Go`, pic `111,56 Go`, token produit
  `47106` (`"itre"`). Le chemin streaming termine avec le même token : TTFT
  `121,846 s`, préfill `121,844 s`, mémoire active `0,98 Go`, pic `34,49 Go`.
- La résidence complète est donc trop coûteuse pour être activée par défaut
  sur une machine de 96 Go et ne bat pas le streaming sur ce premier forward.
  Elle reste un mode expérimental utile pour diagnostiquer le coût de recharge,
  mais `decode` sur un run limité à un token ne prouve pas encore un gain de
  token 2 : aucun second forward autoregressif n’a eu lieu.
- Un essai résident à deux tokens a été interrompu après environ cinq minutes
  dans `eval` sous pression mémoire ; il n’est pas retenu comme benchmark et
  aucune accélération autoregressive ne doit être déduite de ce run.
- Décision : ne pas généraliser la résidence complète. La suite performance
  doit étudier une résidence groupée/lazy ou un forward fusionné/batché qui
  réduit les rechargements sans conserver les 48 couches. Le générateur public
  et le MTP Flash-Next restent par ailleurs bloqués par la qualification E5
  4-bit ; ce résultat mémoire ne change pas ce gate qualité.
- Les chiffres reproductibles sont dans
  `results/flash-residency-2026-08-31.tsv`.

### Étape V20 — instrumentation du coût de rechargement Flash-Next — 2026-08-31

- `Qwen4ExpGreedyGenerationResult` expose maintenant trois métriques de
  diagnostic : le nombre de visites de couches, le temps cumulé de chargement
  des tenseurs depuis les shards et le temps cumulé de forward MLX. Cela évite
  de confondre le coût I/O/instanciation du mode `streamed` avec le coût des
  noyaux Flash-Next eux-mêmes.
- Ces valeurs sont imprimées par `flash-generate-probe` et ajoutées aux
  métadonnées de `swift-mlx-profiler` lorsqu'une trace est demandée. Elles
  permettront de comparer proprement un futur forward batché/MTP avec le
  chemin couche-par-couche actuel.
- Validation : `Scripts/build.sh` via `xcodebuild` termine par `BUILD
  SUCCEEDED`; `Scripts/run-tests.sh` puis `xcrun xctest` passent avec **48
  tests**.
- Le premier essai sans sandbox a été nécessaire uniquement parce que SwiftPM
  compile le manifeste et ses modules auxiliaires dans les caches utilisateur
  macOS. Aucun modèle ni calcul Flash long n'a été relancé pendant ce jalon.
- Décision : conserver `streamed` par défaut et `resident` en diagnostic. Le
  prochain vrai chantier Flash reste le drafter MTP `qwen4_exp` et son forward
  spéculatif compatible avec les caches GDN/QSA/PLE; l'instrumentation est
  maintenant suffisante pour en mesurer séparément le gain et le coût.

### Étape V21 — premier round MTP réel Flash-Next — 2026-08-31

- Le predictor MTP natif du checkpoint réel
  `Qwen3.8-Flash-Next-MLX-4bit-MTP` charge ses 76 tenseurs dans un seul shard
  et passe un forward réel : état quatre-flux `[1,1,10240]`, projection tête
  `[1,1,2560]`, offset MTP `1`.
- Le seam local `Qwen4ExpFlashMTPDraftEngine` implémente maintenant un round
  complet borné : préfill cible, préparation du cache MTP, draft, vérification
  cible `[bonus | draft]`, walk greedy, restauration du snapshot cible, rejeu du
  préfixe accepté et commit du cache privé MTP. La restauration couple bien les
  caches GDN/QSA et l'horloge M-RoPE ; elle n'est pas réduite à un trim KV.
- Exécution réelle avec un prompt texte de 22 tokens et `blockSize=2` : draft
  `19`, prédictions cible `25,15`, acceptation `0/1`, correction `25`, rejeu
  cible d'un token, offset drafter final `23`, durée `335,491 s`. Le résultat
  valide le cycle de contrôle et de rollback ; il ne qualifie pas la qualité
  4-bit ni la rentabilité du MTP, et le strict gate E5 reste fermé.
- Le probe est reproductible avec :
  `qwen38 flash-mtp-probe <checkpoint> --round --block-size 2 --prompt ...`.
  Les chiffres bruts sont dans
  `results/flash-mtp-round-2026-08-31.tsv`.
- Validation intermédiaire : build `xcodebuild` réussi ; `Scripts/run-tests.sh`
  sérialisé puis XCTest direct passent avec **50 tests**. Le prochain jalon
  est d'adapter ce seam en générateur Flash opt-in (statistiques MTP et
  rollback), puis seulement de brancher une boucle publique/GUI après une
  qualification par tâche et une mesure du coût contre le greedy streaming.

### Étape V22 — générateur MTP Flash opt-in et CLI — 2026-08-31

- Le cycle validé au jalon V21 est encapsulé dans
  `Qwen4ExpGreedyGenerator.generateMTP(...)`. Le générateur conserve la cible
  comme autorité, vérifie chaque bloc en un forward, restaure puis rejoue le
  préfixe accepté après rejet, et commite ensuite l'état privé du drafter.
  Aucun rollback approximatif ni simple trim KV n'est utilisé.
- Le résultat expose `rounds`, `proposedTokens`, `acceptedTokens`,
  `targetVerifiedTokens`, `rollbacks`, `replayedTokens`, le taux d'acceptation,
  TTFT et les temps cumulés de chargement/forward des couches. Les phases
  prefill/generation continuent d'être instrumentées par `swift-mlx-profiler`.
- `flash-generate-probe --mtp --mtp-block-size N` permet désormais de lancer
  ce chemin explicitement. Le chemin greedy existant reste inchangé et l'option
  MTP refuse les prompts image tant que les deltas M-RoPE de continuation ne
  sont pas intégrés au drafter ; cette restriction est volontaire et visible.
- Validation : build `xcodebuild` réussi, CLI vérifié avec `--help`,
  `Scripts/run-tests.sh` terminé par `** TEST SUCCEEDED **` et **50 tests**
  passés. Aucun nouveau run Flash long n'est lancé après cette compilation.
- Suite : exécuter un court run CLI `--mtp` (1 à 2 tokens) si nécessaire pour
  valider l'enveloppe publique, puis brancher les statistiques dans la GUI et
  le serveur seulement après qualification qualité/multimodale ; le gate E5
  4-bit reste fermé.

### Étape V23 — smoke du générateur MTP Flash public — 2026-08-31

- Le premier contrôle avec `max-new-tokens=2` a révélé une borne trop stricte :
  le bonus initial était émis mais aucun draft n'était demandé. La formule a
  été corrigée pour utiliser tout le budget restant après le bonus ; ce cas
  aurait sinon donné un faux smoke MTP sans round.
- Après rebuild Xcode, le CLI réel a exécuté deux rounds sur le checkpoint
  `Qwen3.8-Flash-Next-MLX-4bit-MTP`, prompt texte de 22 tokens, bloc 2 et
  budget de 3 tokens : IDs `[17,25,16]`, `2` propositions, `0` acceptées,
  `4` tokens vérifiés, `2` rollbacks et `4` tokens rejoués.
- TTFT `107,419 s`, génération `420,236 s`, `240` visites de couches,
  `515,802 s` cumulés de chargement contre `9,894 s` de forward. Le cycle
  MTP est fonctionnel, mais le chemin streamed couche-par-couche rend cette
  configuration non rentable ; aucun gain de débit ne doit être annoncé.
- Les chiffres bruts sont dans
  `results/flash-mtp-generator-smoke-2026-08-31.tsv`. Le compteur de tokens
  rejoués est désormais imprimé par `flash-generate-probe --mtp`.
- Validation finale : build `xcodebuild` réussi et suite sérialisée `xcodebuild`
  réussie avec **51 tests**. Le gate E5 qualité Flash reste fermé ; la suite
  utile est l'intégration des métriques dans GUI/serveur et la définition du
  contrat multimodal M-RoPE, pas un benchmark MTP plus long sur ce chemin.

### Étape V24 — qualification utilisateur Flash-Next : gate qualité maintenu fermé — 2026-08-31

- Un smoke Swift streamed texte, borné à 8 tokens, produit `var\\n<|im_end|>`
  pour « Réponds en une phrase : qui est Xi Jinping ? ». Le forward complet
  fonctionne (`24` tokens de prompt, TTFT `128,457 s`, pic `34,66 Go`), mais la
  sortie n'est pas une réponse exploitable.
- Le même smoke avec l'image locale
  `/Users/vincent/Downloads/licensed-image-2.jpeg` consomme bien la tour
  vision (`1216×800`, `950` marqueurs, `978` tokens de prompt) et traverse les
  `48` couches. Il produit néanmoins `y\\n\\n2user...\\n<|im_end|>` : la
  multimodalité est techniquement branchée, mais sa qualité n'est pas validée.
- La référence Python `mlx-vlm 0.6.17` reconnaît `qwen4_exp`, mais son chargeur
  résident dépasse la mémoire Metal sur le checkpoint local de `105 Go` avant
  de produire un token (`Insufficient Memory`). Ce n'est donc pas une référence
  de qualité utilisable telle quelle sur ce Mac ; E5 reste le seul diagnostic
  inter-runtime disponible.
- Données brutes : `results/flash-quality-smoke-2026-08-31.tsv`. Décision : ne
  pas ouvrir le générateur Flash-Next dans le catalogue public ni poursuivre
  l'intégration MTP/serveur sur la base de ces sorties. La prochaine étape est
  une investigation ciblée du premier écart de génération (prompt/template,
  ordre des flux hyper-connections et/ou routage MoE), avec un fixture Python
  léger ou une exécution sans charge concurrente. Le smoke multimodal ne doit
  pas être présenté comme une validation de qualité.

### Étape V25 — parité causale complète et diagnostic de dérive 4-bit — 2026-08-31

- Le fixture multi-couches a été corrigé : les couches `full_attention` sont
  désormais évaluées avec `mask="causal"`, comme dans `Qwen4ExpModel`. Le
  précédent `mask=None` comparait un SDPA dense non masqué Python au chemin
  causal Swift et fabriquait un faux écart dès la couche QSA 3.
- Le fixture causal des 48 couches a été généré en mode Python MLX streamé,
  sans charger le modèle Flash-Next résident. La comparaison Swift a été
  effectuée avec le binaire construit par `xcodebuild`.
- La première QSA est maintenant saine à entrée Python : erreur re-ancrée
  `0,8037 %` RMS relative. Sur l'ensemble des couches, aucune frontière ne
  présente une rupture structurelle isolée ; la pire erreur re-ancrée est
  `3,848 %` au layer 31, tandis que le mixer Swift appliqué à l'état Python
  reste à `0,128 %` et conserve le top-1 Python.
- En revanche, la chaîne Swift quantifiée amplifie les écarts de couche en
  couche : le top-1 final devient `917` côté Swift contre `332` côté Python.
  Cela confirme que le smoke utilisateur non cohérent est un problème de
  qualification qualité du chemin 4-bit inter-runtime, pas un problème de
  masque QSA ni une preuve suffisante d'un défaut du mixer.
- Le résultat détaillé est dans
  `results/flash-selected-layers-causal-2026-08-31.tsv`. Le gate qualité et
  l'exposition publique Flash-Next restent fermés. Aucun nouveau run image ou
  MTP long ne serait informatif avant une décision sur ce comportement
  quantifié (référence FP32 symétrique, checkpoint 8-bit/BF16 réellement
  disponible, ou critère qualité indépendant de l'identité token).
- Validation : le binaire Swift utilisé est issu du dernier `xcodebuild` et
  la suite sérialisée `Scripts/run-tests.sh` reste verte à 51 tests ; la
  modification du fixture passe également `python3 -m py_compile`.

<!-- ASK: décision de qualification Flash-Next après parité 48 couches -->
## ASK — décision ciblée pour la dérive qualité Flash-Next 4-bit

À transmettre à l’agent de planification :

La parité causale complète du checkpoint local
`Qwen3.8-Flash-Next-MLX-4bit-MTP` est maintenant exécutée. Le masque QSA est
aligné avec Python ; aucune frontière isolée ne présente de rupture nette en
re-ancrage, le mixer Swift appliqué à l’état Python reste proche (`rel
0,00128`) et conserve le top-1 Python. En revanche, la chaîne 4-bit amplifie
les petites différences de couche jusqu’à produire le top-1 `917` côté Swift
contre `332` côté Python, ce qui rend le smoke utilisateur non exploitable.

Merci de trancher les points suivants :

1. Cette dérive doit-elle être considérée comme une variation inter-runtime
   acceptable pour ce checkpoint 4-bit, ou faut-il encore rechercher un bug
   spécifique dans le chemin Swift ?
2. Quel protocole minimal permet de qualifier la qualité sans exiger une
   identité token-à-token (référence FP32 déquantifiée symétrique, marges de
   logits, prompts naturels, ou autre) ?
3. Faut-il investir maintenant dans une variante Flash-Next 8-bit/BF16, ou
   reprendre le chantier GUI/serveur/MTP seulement après un critère qualité
   indépendant de cette comparaison inter-runtime ?

Contrainte : toute validation MLX doit rester effectuée via `xcodebuild`, puis
`xcrun xctest`; ne pas relancer un long benchmark MTP avant cette décision.

---

<!-- ANSWER: décision de qualification Flash-Next — 2026-08-31 -->
## RÉPONSE — décision de qualification Flash-Next 4-bit (2026-08-31, soir)

Les trois points, tranchés. Le socle de preuve est la RÉPONSE précédente (V1-V4) **plus** les faits nouveaux du log (parité causale 48 couches sans rupture isolée, mixer sur état Python `rel 0,00128` conservant le top-1, smokes qualité dégénérés, résidence complète à 111 Go non viable, streaming à ~120 s/token) et une mesure faite ce soir : **la table n-gram pèse 32,0 Go des 113 Go du checkpoint** (lecture des headers safetensors ; vision 0,9 Go, reste ≈ 80,3 Go).

### Point 1 — Variation inter-runtime : OUI, la décision V1/V2 est confirmée et close

La parité causale 48 couches renforce le verdict précédent : aucune frontière isolée en rupture, mixer exonéré sur état Python, et les mécanismes chiffrés (frontière top-k médiane 2,4e-4 avec ex-aequo exacts ; CPU vs GPU du même Python qui change déjà l'appartenance des experts) restent valables sur toute la chaîne. **Arrêter définitivement la chasse au bug du chemin quantifié Swift.** Le top-1 `917` vs `332` après 48 couches de chaîne 4-bit est l'issue attendue d'un système chaotique — pas un symptôme.

**Mais attention à ne pas sur-étendre ce verdict** : « pas de bug numérique » ne signifie pas « générateur validé ». Les smokes dégénérés (`var\n<|im_end|>` au 2ᵉ token, `2user` dans la sortie image) ne ressemblent pas à du bruit inter-runtime — ils ressemblent à des bugs **d'intégration du générateur** (rendu du template, gestion `<think>`/`enable_thinking`, EOS, conditionnement du prompt — les pièges déjà listés en §4.4) et/ou à la **qualité intrinsèque de ce quant communautaire** (risque déjà au §10). Ces deux pistes se vérifient sans aucun forward de 48 couches : template id-à-id vs rendu HF, et inspection des 20 premiers logits/rangs au premier token de génération (les fixtures full-chain déposées ce jour suffisent). À faire **avant** toute conclusion qualité.

### Point 2 — Protocole qualité minimal (sans identité token-à-token), dans cet ordre

- **Q-A — Sanité d'intégration du générateur (0,5 j, aucun gros forward)** : template rendu id-à-id contre HF ; vérification EOS/thinking ; au premier token de génération des deux prompts capturés (`…swift-full…` / `…photosynthesis…`), rapporter top-20, marges et rangs croisés Swift/Python. Si le top Swift est un token absurde de rang profond chez Python **avec une grande marge Python**, chercher le bug d'intégration ; si les deux distributions sont plates ou d'accord sous la marge, passer à Q-B.
- **Q-B — Cross-scoring teacher-forced (le critère central, 1 forward par côté et par prompt)** : générer UNE fois côté Python (mlx-vlm 0.6.17 en mode par-couches, pas résident — il OOM en résident) une continuation greedy de 32-64 tokens pour 5 prompts naturels (~une nuit au rythme actuel). Puis, en un seul forward teacher-forced par côté : (a) logprob moyen par token de la continuation sous Swift vs sous Python — **seuil proposé : |Δ| ≤ 0,15 nat/token** ; (b) accord argmax aux positions où la marge Python > 0,5 — **seuil : ≥ 99 %** ; (c) rangs croisés moyens. Aucune génération Swift n'est requise : compatible avec les 120 s/token actuels et `xcodebuild`/`xcrun xctest`.
- **Q-C — Qualité de tâche (MMLU-Pro subset, prompts §7) : explicitement DIFFÉRÉ** jusqu'à un chemin d'inférence utilisable (voir point 3). Ne pas bloquer le verdict dessus ; Q-A + Q-B suffisent pour qualifier le portage.

### Point 3 — Ni 8-bit ni BF16 ; le chantier prioritaire est la résidence partielle

- **8-bit/BF16 : non, définitivement** (inchangé, V3) : ~130 Go RAM / ~250 Go disque, et l'oracle FP32-déquantifié symétrique — déjà vert sur les frontières testées — répond mieux à la question que tout checkpoint supplémentaire.
- **Le fait nouveau qui commande la suite** : résident complet = 111 Go (> 96 Go physiques, impossible) ; streaming = ~120 s/token (inutilisable, et rend le bench MTP dénué de sens — les 0/2 acceptés sur 3 tokens ne mesurent rien). Or **n-gram = 32 Go sur 113**. La résidence « tout sauf n-gram » ≈ 81 Go + activations : ça passe avec `iogpu.wired_limit_mb` relevé (~88-90 Go), serré mais viable — et c'est **exactement ce que le plan exigeait depuis le départ** (§2.2 brique 3 et étape E : table n-gram 51 B « JAMAIS matérialisée », 128 shards mmap lazy ; interdiction absolue §0). Le probe de résidence a matérialisé la table — c'est ça, la cause de l'inviabilité, pas la taille du modèle.
- **Ordre de reprise recommandé** : (1) Q-A immédiatement ; (2) résidence partielle n-gram-mmap (le vrai déverrouilleur : générateur utilisable, Q-B/Q-C rapides, bench MTP mesurable) ; (3) Q-B ; (4) alors seulement bench MTP Flash-Next (la brique fonctionne — rounds/rollback/rejeu validés — inutile de la re-benchmarker avant), puis GUI/serveur/catalogue pour Flash-Next. **Le chantier 27B (GUI/serveur/MTP) n'est pas concerné par ce gate et continue en parallèle.**

**Gate final inchangé dans sa forme** (V4) : générateur public Flash-Next quand Q-A et Q-B sont verts ; MTP Flash-Next quand l’auto-cohérence Swift (MTP-on ≡ MTP-off) est verte **et** que le générateur est débloqué. Pas de benchmark MTP long avant la résidence partielle.

### Étape V26 — Q-A template validé et résidence lazy n-gram — 2026-08-31

- La nouvelle sonde `flash-template-probe` inspecte le tokenizer sans charger
  de poids. Un JSON produit par `Scripts/flash-template-reference.py` est
  comparé id-à-id par Swift.
- Sur le prompt naturel présidentiel, le rendu Swift est **identique** à la
  référence Python en thinking activé (`57` tokens) comme désactivé (`29`
  tokens). La frontière est correcte : thinking actif se termine sur
  `<think>\\n`, thinking désactivé sur `<think>\\n\\n</think>\\n\\n`.
  Les IDs `<|im_start|> = 248045`, `<|im_end|> = 248046` et
  `<|endoftext|> = 248044` sont présents et cohérents.
- Le loader de couche ne matérialise plus les paramètres contenant
  `ngram_embedding` ; les autres poids restent matérialisés. Le probe réel de
  la couche PLE 2 (`sequenceLength=1`) réussit avec `61` tenseurs, `2` shards,
  `1,62 Go` matérialisés et `1,63 Go` de pic MLX, sortie `[1,1,10240]`.
- Validation intermédiaire : `xcodebuild` réussi, puis suite sérialisée
  `xcodebuild` réussie avec **51 tests**. Q-A template est vert ; le prochain
  contrôle est un appel PLE multi-token pour vérifier la continuité de l’état
  n-gram avant de reprendre Q-B/résidence autoregressive.
- Le probe `flash-stream-probe --layers 2 --sequence-length 1
  --repeat-count 2` réussit sur deux appels successifs avec le cache conservé.
  Chaque appel garde `1,62 Go` de poids matérialisés et le pic du processus
  reste `1,63 Go`, sortie `[1,1,10240]`. La continuité de cache n-gram est donc
  validée au niveau couche ; le prochain test est la résidence autoregressive
  de toutes les couches ordinaires avec n-gram lazy.

### Étape V27 — Résidence n-gram réellement partielle — 2026-08-31

- Le premier essai « résident complet » n'était pas concluant : malgré le
  filtre initial, l'accès MLX à une ligne de la table matérialisait les
  `ngram_embedding` entiers. Le processus montait à `109,59 Go` actifs et
  `110,32 Go` de pic ; ce chemin ne convient pas à la machine 96 Go.
- Le loader utilise maintenant `Qwen4ExpLazyNGramStorage`. Il lit uniquement
  les en-têtes safetensors, conserve l'index `(shard, weight, scales, biases)`
  et lit les lignes demandées par le hash courant avant une déquantification
  MLX compacte. Les gros tenseurs n-gram ne sont plus injectés dans le
  `Module`; les petits paramètres de hash restent chargés normalement.
- La couche PLE réelle (couche 2 du checkpoint, index Swift 1) a été testée
  sur deux appels successifs avec cache conservé : sortie `[1,1,10240]`,
  `1,64 Go` de poids ordinaires et `1,65 Go` de pic. Cela valide le chemin
  lazy et la continuité du contexte n-gram au niveau couche.
- Le parcours borné des 48 couches réelles avec deux tokens termine désormais
  à `78,84 Go` actifs / `79,81 Go` de pic (`86,492 s` de chargement cumulé,
  `129,082 s` de forward cumulé), soit environ 30 Go de moins que l'ancien
  chemin. La porte mémoire de la résidence partielle est donc verte.
- La sortie bornée observée (`alfa营商`) n'est **pas** une validation de
  qualité : elle ne contient que deux tokens et le générateur Flash-Next
  reste soumis à Q-B (cross-scoring teacher-forced). Il faut maintenant
  comparer quelques lignes lazy à une lecture eager de référence, puis
  reprendre Q-A logits / Q-B avant toute conclusion sur le modèle ou MTP.

### Étape V28 — Parité des lignes n-gram et correction BF16 — 2026-08-31

- La sonde `flash-ngram-parity` compare désormais, sur le checkpoint réel, la
  sortie du reader row-wise à la déquantification directe des mêmes lignes
  safetensors. Sur les lignes `[0, 1, 12345]` du shard `0`, les deux sorties
  sont `[3,160]` avec `max |delta| = 0` et `mean |delta| = 0`.
- L'écart initial (`max |delta| ≈ 737280`) venait d'une conversion numérique
  des bits BF16 (`asType`) au lieu d'une réinterprétation binaire (`view`).
  Cette correction est dans `Qwen4ExpLazyNGramStorage` et concerne les
  échelles comme les biais quantifiés.
- Le build et la suite sérialisée restent verts avec `xcodebuild` : **51 tests
  passés**. La résidence partielle est donc à la fois memory-safe et
  numériquement exacte sur le chemin de lookup testé.
- Le gate qualité reste volontairement fermé : la prochaine étape est Q-A
  (premiers logits/rangs puis EOS/thinking), suivie de Q-B teacher-forced.
  Aucun benchmark MTP long ni conclusion de qualité ne doit être tiré du
  probe borné `alfa营商`.
- Après correction BF16, un nouveau probe résident de deux tokens reste
  stable en mémoire (`78,84 Go` actifs / `79,81 Go` de pic) mais produit
  `var\\n`. La correction a donc bien changé le chemin numérique, sans rendre
  la génération qualitativement exploitable ; elle n'est pas à elle seule la
  cause de la dégénérescence observée. Cela confirme la priorité de Q-A
  logits/EOS-thinking et de Q-B, plutôt qu'un benchmark MTP prématuré.

### Étape V29 — Q-A : premiers logits Flash-Next — 2026-08-31

- Le probe `flash-generate-probe --max-new-tokens 1 --report-top-k 20`
  capture désormais les candidats du premier token sans refaire de forward.
  Sur `Réponds en une phrase : qui est Xi Jinping ?`, le top-20 commence par
  `917 (12,463096)`, `70 (12,337608)`, `846 (12,073127)` ; la marge top-1 / 
  top-2 vaut seulement `0,125488`. Le token structural `<|im_start|>` (`248045`)
  est même au rang 9.
- Ce profil est plat et ne permet pas de diagnostiquer seul template contre
  quantification : il faut le comparer à la référence Python sur les mêmes
  logits, puis rapporter rangs et marges. Il justifie le passage à Q-B
  teacher-forced si aucun écart franc de distribution n'est observé.
- Coût et résidence restent cohérents : `TTFT 100,224 s`, `79,14 Go` actifs,
  `79,81 Go` de pic, 48 visites de couches. Le rapport top-k est donc un outil
  de diagnostic, pas un changement du chemin d'inférence.

### Étape V30 — primitive Q-B teacher-forced côté Swift — 2026-09-01

- `Qwen4ExpStreamingTextModel.scoreTeacherForced` évalue maintenant une
  continuation fixe en un seul forward causal, sans sampler ni gestion EOS.
  Il rapporte le logprob moyen par token, l'accord argmax, l'accord sur les
  positions dont la marge dépasse `0,5`, le rang moyen de la cible et les
  observations token par token.
- Le CLI expose `flash-teacher-forced-score`, avec une continuation texte
  tokenisée localement ou une liste d'IDs exacte (`--continuation-ids`). Le
  prompt passe par le template ChatML et le mode thinking est explicite.
  Cette commande est la moitié Swift de Q-B ; elle est volontairement
  indépendante de la génération et du MTP.
- Build `xcodebuild` réussi et suite `xcodebuild`/`xcrun xctest` sérialisée
  réussie avec `51` tests. Les parités Python/Swift précédentes sont déjà
  validées pour QSA, MRoPE, vision, globaux, couches publiques/sélectionnées,
  n-gram lazy et l'identité M2/M1. Le gate qualité Flash-Next reste fermé
  uniquement pour la nouvelle comparaison Q-B : il manque la fixture de
  continuation et le score Python teacher-forced sur les mêmes IDs. Aucun
  benchmark MTP long ne doit précéder cette comparaison.
- `Scripts/qwen4-exp-teacher-forced-reference.py` complète maintenant le
  protocole côté Python : IDs prompt/continuation explicites, couches chargées
  séquentiellement, logits et métriques cibles exportés en safetensors. Le
  script est syntaxiquement validé ; son exécution MLX doit être faite sur le
  Mac avec Metal, comme les fixtures Python précédentes.
- `Qwen4ExpTeacherForcedParity.compare` et l'option CLI `--python-fixture`
  complètent la moitié Swift : après production de la fixture sur Metal, le
  CLI compare les IDs, logprobs, marges, rangs et accords argmax sans refaire
  un second forward. Le gate Q-B reste donc en attente de cette exécution,
  pas en attente d'une parité Python générale.
- Clarification sur « quand peut-on inférer ? » : techniquement, le chemin
  Flash-Next autoregressif existe depuis V27 (`--resident-layers`, résidence
  partielle n-gram) et peut déjà produire des tokens. Il reste toutefois
  expérimental et trop lent/dégénéré pour être déclaré utilisable : le statut
  public/qualité est ouvert seulement après Q-A puis Q-B. Le 27B, lui, est
  déjà un chemin d'inférence utilisable et indépendant de ce gate Flash.
- Tentative de production de la fixture avec la séquence exacte du fixture E5
  (29 tokens de prompt + 27 de continuation) : le processus `python3` lancé
  depuis la session Codex a échoué à initialiser Metal (`No Metal device
  available`). Ce diagnostic est limité à cette session d'exécution et ne
  décrit ni le Mac de Vincent ni son environnement Python interactif. Ce
  n'est pas une absence de parité antérieure ; la commande est prête à être
  relancée dans son terminal avec la référence vendorisée par défaut.

<!-- ASK: vérifier le contexte Metal de la nouvelle fixture Q-B -->
## ASK — relance Q-B dans l’environnement Python MLX déjà validé

La session Codex utilisée pour tenter la nouvelle fixture a échoué dès
l’import de `mlx.nn` avec :

```text
RuntimeError: [metal::load_device] No Metal device available.
This typically occurs in headless, sandboxed, or virtualized macOS sessions
where the GPU is not accessible.
```

Ce processus utilisait `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`,
avec `mlx 0.31.2` et `mlx-vlm 0.6.2`. Le message est limité au contexte
Codex : les parités Python MLX antérieures ont bien été exécutées et validées.
Merci de vérifier :

1. quel interpréteur/contexte a produit les fixtures Python déjà validées ;
2. pourquoi il diffère du processus Codex actuel (sandbox/headless, session
   GUI, environnement ou variable) ;
3. la commande exacte permettant de produire la fixture Q-B complète avec la
   référence vendorisée et les IDs E5 déjà extraits.

---

<!-- ANSWER: revue indépendante + relance Q-B — 2026-09-02 -->
## RÉPONSE — revue de ce qui a été fait, et cause racine de la dégénérescence Flash-Next (2026-09-02)

### R0 — L'ASK Metal, en trois lignes

1. Les fixtures validées ont été produites par `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (mlx 0.31.2, mlx-vlm 0.6.2 + référence vendorée), celui-là même que Codex a utilisé.
2. Depuis un terminal normal du Mac (et depuis la session de revue), `mx.metal.is_available()` est `True` pour cet interpréteur. `No Metal device available` est un artefact du sandbox Codex, pas de la machine.
3. **Ne relancez pas la fixture Q-B telle quelle** : la suite montre que la référence Python contre laquelle Q-B allait comparer Swift n'a jamais produit de logits cohérents. Q-B mesurait l'accord entre deux sorties fausses.

### R1 — Ce qui tourne en rond, et pourquoi

Aucune étape depuis D3v n'a testé la **sanité absolue** de la référence Python ; toutes les mesures sont des parités **relatives** Swift↔Python. Le test le plus simple manquait : sur ses propres fixtures E5 (`parity/qwen4-exp-e5-natural-*-full-reference.safetensors`, logits Python des 29/51/56 positions), la référence prédit le token suivant du prompt à **3/28, 1/50 et 3/55 positions**, logprob moyen ≈ −8,5 nat/token. Elle prédit `2` après `<|im_start|>`, `<|im_end|>` après `assistant`, `分析` après `<think>\n\n`, et `,` comme premier token de réponse sur deux prompts sur trois. Un modèle de 125 B ne fait pas cela : **la référence est cassée**, et Swift la reproduit fidèlement. Les verdicts « variation inter-runtime acceptable », « système chaotique », « routage MoE trop sensible » et le retrait du gate token-à-token (V1–V4, réponse du 31 août soir) reposaient sur cette référence et sont **révoqués**.

### R2 — Cause racine : convention des normes du checkpoint Vontra (prouvée tenseur par tenseur)

Comparaison directe du checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` avec les shards **BF16 officiels** `Qwen/Qwen3.8-Flash-Next` (requêtes HTTP partielles, `Scripts/qwen4-exp-checkpoint-vs-hf.py`, cache dans `Scripts/hf-cache/`) :

| Tenseur | (mlx − hf) moyenne | écart-type | max\|Δ\| |
|---|---|---|---|
| `hyper_connection_mixer.hc_norm` | **+1,000** | 6,7e-3 | 1,031 |
| `layers.0.attn_hyper_connection.hc_norm` | **+1,000** | 1,9e-3 | 1,016 |
| `layers.0.mlp_hyper_connection.hc_norm` | **+1,000** | 2,2e-3 | 1,016 |
| `layers.3.self_attn.q_norm` / `k_norm` | **+1,000** | 2,4e-3 | 1,004 |
| `layers.3.self_attn.indexer.q/k_layernorm` | **+1,000** | 1,5e-3 | 1,004 |
| `layers.1.ple.norm_key` / `norm_query` / `norm_conv` | **+1,000** | 1,4e-3 | 1,004 |
| `layers.*.linear_attn.norm` (RMSNormGated) | 0 | 0 | 0 |
| `A_log`, `dt_bias`, `conv1d`, `mlp.gate`, normes vision | 0 | 0 | 0 |
| projections 4-bit (qkv, z, out, hyper, experts, shared) | rel_rms ≈ 0,085 | — | cos ≈ 0,996 (bruit de quantification normal) |

Le convertisseur tiers a appliqué le `+1.0` du sanitize `qwen3_5` à **toutes** les normes zéro-centrées de `qwen4_exp`. Or mlx-vlm 0.6.17, la référence vendorée (`vlm_q4_language.py:576-597`) et Swift (`Qwen4ExpRMSNorm`, `Sources/Qwen38Core/FlashNext/Qwen4ExpHyperConnection.swift:107,110`) appliquent `x·(1 + w)` : l'échelle effective est `2 + w_hf` sur 100 % des hyper-connexions, des normes q/k QSA, de l'indexeur et de la PLE. Parité Swift/Python parfaite, sortie fausse des deux côtés.

**Preuve par correction** (`Scripts/qwen4-exp-official-teacher-forced.py`, paquet PyPI mlx-vlm 0.6.17 officiel, sans shim, prompt « président de la Chine », 48 couches, 2 s/couche) :

| Variante | hits teacher-forced sur le prompt | logprob moyen | premier token de réponse |
|---|---|---|---|
| officiel 0.6.17 tel quel | 2/28 | −10,06 | `Dr` / `err` |
| idem, `seed=0` | 2/28 (identique au bit) | −10,06 | idem |
| idem, gate/up experts inversés | 0/28 | −13,58 | garbage |
| **idem, normes `w − 1`** | **10/28** (tous les tokens ChatML à perte ≈ 0) | **−4,43** | **`Le` (marge 1,5), `Je`, `La`** |
| normes `w − 1`, continuation forcée « Le président de la Chine est Xi Jinping, qui occupe ce poste depuis 2013. Son rôle est de diriger » | **20/27** | **−0,68** | `Xi`→`Jinping` à −0,00 |

Remarques annexes : le `seed` n'a aucun effet parce que `layer_multipliers` est chargé depuis le checkpoint (Swift aussi, `@ParameterInfo("layer_multipliers")`) ; l'ordre gate/up des experts et la disposition `down_proj` sont corrects (cos 0,996 contre HF) ; la référence vendorée est un instantané légitime de `main` postérieur à 0.6.17 (numériquement équivalent en batch 1 < 2052 tokens) ; le harnais Python repose toutefois sur les classes de base `qwen3_5` de mlx-vlm **0.6.2** avec des shims, ce qui donne des logits différents (max|Δ| = 31) de ceux du paquet 0.6.17 : à abandonner au profit d'un venv 0.6.17.

### R3 — Ce qui reste valide de l'existant

Toute l'infrastructure de parité (QSA masque/indexeur, MRoPE, vision, GDN, globaux, lazy n-gram, E0–E3, metrics `rel`/cosinus), le chargeur préquantifié, la résidence partielle n-gram et le cycle MTP (rounds/rollback) restent utiles : ce sont des parités **relatives** et elles ne changent pas. Seules les **conclusions qualité** (V1–V4, V24–V25, réponse du 31 au soir) et les fixtures Python E5/causal (générées avec les mauvaises normes) sont à jeter.

### R4 — Plan de reprise (ordre strict)

1. **Correctif Swift (1 ligne de contrat, une demi-journée)** : au chargement de ce checkpoint, soustraire `1.0` aux poids dont la clé finit par `hc_norm.weight`, `q_norm.weight`, `k_norm.weight`, `q_layernorm.weight`, `k_layernorm.weight`, `norm_key.weight`, `norm_query.weight`, `norm_conv.weight` (langage uniquement ; **pas** `linear_attn.norm.weight`, pas la vision, pas le MTP tant qu'il n'est pas vérifié de la même façon). Détection de convention à la lecture des headers : moyenne de `layers.0.attn_hyper_connection.hc_norm.weight` ≈ +0,94 (décalé) contre ≈ −0,06 (HF). Consigner le choix dans le préflight (`qwen38 info`).
2. **Même correctif dans `Scripts/qwen4-exp-*-reference.py`** (les `load_weights_for_prefix` / `load_global`), puis migrer le harnais vers un venv `mlx-vlm==0.6.17` (`python3 -m venv venv617 && venv617/bin/pip install mlx-vlm==0.6.17`), ou réutiliser directement `Scripts/qwen4-exp-official-teacher-forced.py --norm-shift -1`.
3. **Gate de sanité absolu, obligatoire avant toute parité** : hit-rate teacher-forced sur le prompt ≥ 30 % et tokens ChatML (`<|im_end|>`, `<|im_start|>`, `assistant`, `<think>`, `</think>`) prédits à logprob > −0,5. Puis smoke `flash-generate-probe --max-new-tokens 8` : attendu « Le président de la Chine est Xi Jinping… ».
4. **Rétablir le gate token-à-token** (§4.4/§6.4) : greedy identique là où la marge Python > 0,25, avec fixtures régénérées ; Q-B (cross-scoring) redevient un simple contrôle, pas le critère central.
5. **Performance (après la qualité, pas avant)**, d'après l'audit du chemin Swift : (a) le mode streamed relit ~80 Go depuis le Lexar (USB, ~0,8 Go/s) à chaque token — 515 s de chargement pour 9,9 s de calcul dans `results/flash-mtp-generator-smoke-2026-08-31.tsv` ; (b) le mode résident tourne sans wired limit (`iogpu.wired_limit_mb = 0`, aucun `WiredMemoryTicket` dans `Sources/`) avec 48 `eval` + 48 `Memory.clearCache()` par token (`Qwen4ExpStreamingDecoder.swift:81,131`) et des captures de parité retenues sur chaque module ; (c) MoE, cache KV/GDN et n-gram lazy sont sains. Ordre : relever la wired limit (~88 Go), supprimer eval/clearCache par couche en résident, rendre `lastParityCapture` opt-in, déplacer le checkpoint sur le SSD interne (76 Go libres, à vérifier). Objectif réaliste ensuite : > 15 tok/s en résident partiel.
6. Seulement après 3–5 : MTP Flash-Next, GUI, serveur, catalogue.

### R5 — Reproduction

```bash
# venv officiel (déjà créé dans le scratchpad de la session de revue, sinon :)
python3 -m venv venv617 && venv617/bin/pip install mlx-vlm==0.6.17
# sanité absolue, normes corrigées (≈ 2 min, pic ≈ 35 Go)
venv617/bin/python Scripts/qwen4-exp-official-teacher-forced.py \
  --model-dir /Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --norm-shift -1 --output /tmp/qb-normfix.safetensors \
  --ids 248045,846,198,42498,2295,644,52543,7528,1725,501,85648,401,1147,183085,1778,24232,1725,4292,176517,13,248046,198,248045,74455,198,248068,271,248069,271
# comparaison checkpoint ↔ HF BF16 (couche 0 + globaux, expert 0)
venv617/bin/python Scripts/qwen4-exp-checkpoint-vs-hf.py
```

### Étape V31 — correction du contrat de normes Vontra — 2026-09-02

- Le loader Swift corrige désormais au chargement les normes zéro-centrées
  dont le checkpoint Vontra contient la version décalée de `+1` :
  `hc_norm`, `q_norm`, `k_norm`, normes de l'indexeur et normes PLE. La
  primitive `Qwen4ExpRMSNorm` reste inchangée (`1 + weight`) afin de préserver
  le contrat des fixtures synthétiques et des checkpoints déjà corrigés.
- La correction est appliquée aux loaders globaux, couches et tranches, avec
  détection par moyenne de l'ancre `hc_norm`; `linear_attn.norm`, vision et
  MTP ne sont pas modifiés. `qwen38 info` annonce explicitement la politique.
- Le dumper Q-B applique le même décalage `-1` par défaut et l'inscrit dans
  les métadonnées de fixture. Le test de norme couvre le cas Vontra et le cas
  déjà zéro-centré.
- `xcodebuild` build réussi ; `Scripts/run-tests.sh` a passé **53 tests**.
  Il faut maintenant relancer la sanité absolue avec `mlx-vlm==0.6.17` et
  `--norm-shift -1`, puis seulement régénérer les fixtures E5/greedy. Les
  anciennes conclusions qualité et fixtures E5/causal restent révoquées,
  tandis que l'infrastructure de parité, le lazy n-gram et MTP restent valides.

### Étape V32 — sanité absolue Q-B validée sur Metal — 2026-09-02

- Le venv local `venv617` a été créé avec `mlx-vlm==0.6.17`. Depuis le
  terminal GPU, `metal True` est confirmé ; l'erreur précédente était bien
  limitée au sandbox Codex.
- Exécution réelle de `Scripts/qwen4-exp-official-teacher-forced.py` sur le
  checkpoint Vontra 4-bit, 48 couches, avec `--norm-shift -1` : `124,4 s`,
  `teacher-forced hits 10/28`, logprob moyenne `-4,426`. Les RMS cachés
  restent bornés jusqu'à la couche 47 (`1,0870`) ; aucun crash ni anomalie de
  chargement.
- La mesure reproduit la preuve de la revue et valide le correctif de contrat
  des normes comme prérequis qualité. Elle ne constitue pas encore une
  validation de génération libre : prochaine action, smoke greedy Swift sur
  le même prompt, puis régénération des fixtures token-à-token.

### Étape V33 — préflight et tranche Swift validés, smoke complet à optimiser — 2026-09-02

- Le binaire Debug issu du dernier build `xcodebuild` expose bien
  `qwen4_exp`, 48 couches, 1 couche MTP, 22 shards et la politique de
  correction Vontra. Le poids déclaré est `113,21 GB`.
- La validation Swift bornée sur le shard réel est passée : couche 0,
  71 tenseurs, 2 shards lus, `2,02 GB` matérialisés, forward `[1, 4, 2560]`
  OK. Cela vérifie le chargement corrigé sans engager tout le modèle.
- Le smoke greedy complet résident a été lancé avec le prompt Q-B et
  `--max-new-tokens 8`. Après plus de dix minutes, le processus était encore
  dans `Qwen4ExpStreamingDecoder.forward → mlx_eval`, avec ~74 Go de footprint
  et ~60 % d'utilisation GPU ; il a été arrêté proprement. Ce n'est pas un
  verdict de qualité : c'est le coût du chemin actuel sur le checkpoint
  `113 GB` stocké sur Lexar. Il faut traiter la résidence/les synchronisations
  avant de relancer un smoke complet.
- Le décodeur ne synchronise et ne vide maintenant le cache par couche qu'en
  mode streamed ; le résident reporte la matérialisation au forward texte
  complet. Le rebuild `Scripts/build.sh` (donc `xcodebuild`) et les **53 tests**
  sérialisés sont verts.
- Après ce changement, un run thinking de 1 token a terminé : token initial
  `The` (début du raisonnement), TTFT `244,333 s`, 48 visites, `79,59 GB`
  actifs et `80,92 GB` peak. Le run no-thinking lancé immédiatement après a
  rencontré une pression page-cache : à 9 min le GPU était retombé à ~5 % et
  le scheduler attendait ; il a été arrêté. Ces deux résultats ne suffisent
  pas à conclure sur la qualité libre no-thinking ; ils bornent le coût et
  confirment que l'optimisation suivante doit viser le chargement/résidence
  et l'exécution répétée, pas les normes.
- Prochaine étape stricte : faire un run greedy no-thinking isolé (machine
  GPU au repos, cache stabilisé) ou utiliser une fixture de logits Swift,
  puis régénérer les fixtures greedy et rétablir le gate token-à-token. MTP et
  Flash GUI restent après cette validation absolue.

### Étape V34 — premier token greedy Swift aligné — 2026-09-02

- Après stabilisation du GPU et du page-cache, le smoke no-thinking résident
  isolé a terminé sur le prompt Q-B exact. Le runtime Swift a généré
  `id 2229`, décodé `Le`, qui est le premier token attendu par la référence
  officielle corrigée (`mlx-vlm 0.6.17`, normes `w - 1`).
- Mesure : TTFT `272,643 s`, prefill `272,631 s`, decode `0,011 s`, 48 visites
  de couches, `79,22 GB` actifs et `80,52 GB` peak. Le premier run long n'était
  donc pas un échec numérique ; il exposait surtout le coût du chargement et
  de la résidence du checkpoint `113 GB` sur Lexar.
- Ce jalon valide ensemble le correctif de normes, le rendu ChatML
  no-thinking et la couture du chemin greedy jusqu'au premier token. Il ne
  valide pas encore une réponse libre multi-token ni le MTP.
- Un run de 8 tokens avec profiler a été lancé ensuite, mais a été arrêté après
  7 minutes lorsque le GPU est retombé à ~6 % dans l'attente page-cache/I/O ;
  la trace n'est exportée qu'en fin de run et n'a donc pas été produite. Il
  faut éviter de confondre cette attente du stockage USB avec une régression
  du modèle.
- Prochaine étape stricte : régénérer la fixture greedy token-à-token avec ce
  prompt exact, comparer plusieurs positions au teacher-forced officiel,
  puis reprendre l’optimisation de résidence avant les benchmarks MTP.

### Étape V35 — graphe résident borné et option configurable — 2026-09-02

- Le mode résident ne fait plus un `eval` par couche et par token. Il
  matérialise désormais le graphe toutes les `8` couches par défaut, avec un
  dernier `eval` garanti en fin de passe : sur 48 couches, cela représente six
  points de synchronisation au lieu de 48, sans `Memory.clearCache()` ni
  rechargement des poids entre ces points.
- L’intervalle est configurable par `--resident-eval-interval` sur
  `flash-generate-probe` et `flash-teacher-forced-score`. La valeur doit être
  strictement positive ; `1` permet de reproduire le comportement fortement
  synchronisé et une valeur supérieure à 48 limite la passe à son dernier
  point.
- Le paramètre est propagé jusqu’à
  `Qwen4ExpStreamingDecoder.residentEvaluationInterval`, donc le serveur et
  les appels bibliothèque peuvent l’utiliser sans dépendre de la CLI.
- `Scripts/build.sh` (xcodebuild) et `Scripts/run-tests.sh` passent toujours,
  avec **53 tests**. L’aide du binaire Debug expose bien l’option.
- Ce jalon est une optimisation d’exécution/mémoire ; il ne modifie pas les
  poids, les normes Vontra, le cache récurrent/QSA ou le protocole MTP. La
  prochaine mesure utile est un run multi-token résident profilé après
  stabilisation du page-cache, puis la reprise du gate greedy et des essais
  MTP.

### Étape V36 — smoke résident multi-token et trace profiler — 2026-09-02

- Le run réel sur le checkpoint Vontra 4-bit avec
  `--resident-eval-interval 8 --max-new-tokens 2 --trace` s’est terminé sans
  erreur et a exporté `/private/tmp/qwen38-flash-v35.trace.json` (165 Ko).
- Génération obtenue : IDs `[2229, 85648]`, texte `Le président`, cohérent
  avec le premier token validé par la sanité absolue corrigée.
- Mesure de ce run : TTFT `102,026 s`, génération `131,569 s`, 96 visites de
  couches, chargement cumulé `95,448 s`, forward cumulé `135,508 s`, mémoire
  MLX peak `75,58 GB` (process peak `76,47 GB`). Les temps restent dominés
  par la résidence/lecture du checkpoint sur Lexar ; ce run ne doit pas être
  comparé à une machine GPU concurrente.
- La trace confirme les six checkpoints de la passe résidentielle (couches 7,
  15, 23, 31, 39 et 47), tandis que les couches intermédiaires restent de
  l’ordre de la milliseconde sur le second token. Le profiler est bien
  intégré au chemin optimisé et exporte la timeline mémoire.
- Le chemin greedy multi-token corrigé est maintenant exécutable. La suite
  est la régénération/gate token-à-token puis la validation MTP sur texte ; le
  multimodal et Flash-Next complet restent après ce gate.

### Étape V37 — gate libre 8 tokens reporté pour cause d’I/O — 2026-09-02

- Un run greedy de 8 tokens a été lancé sur la même machine, sans autre
  lancement GPU concurrent, avec le mode résident et l’intervalle 8. Après
  plus de 21 minutes, le processus restait vivant mais alternait calcul et
  attente de pages du checkpoint sur Lexar ; il a été interrompu proprement
  (`SIGINT`, code 130) sans sortie ni erreur MLX.
- Ce résultat ne permet ni d’ouvrir ni de fermer le gate qualité à 8 tokens.
  Le smoke résident de 2 tokens reste le dernier résultat libre concluant :
  `Le président`, trace profiler exportée, 75,58 GB peak MLX.
- La suite doit déplacer/copier le checkpoint sur un stockage local plus
  rapide, ou produire un runner de gate avec des poids déjà résidents, avant
  de relancer 8 tokens. Aucun benchmark MTP long ne doit être lancé sur le
  chemin I/O actuel.

### Étape V38 — seconde tentative du gate 8 tokens — 2026-09-03

- Le gate greedy 8 tokens a été relancé avec la machine dégagée, le binaire
  Debug déjà compilé et `--resident-eval-interval 8`. Après environ 19 minutes,
  le processus restait actif mais alternait calcul et attente du checkpoint
  externe ; il a été interrompu proprement par SIGINT (code 130).
- La réduction de la charge concurrente n’a donc pas supprimé le problème :
  le chemin résident actuel reste trop lent pour un gate libre de 8 tokens sur
  Lexar. Il n’y a ni crash ni erreur de qualité observée dans ce run, car il
  n’a pas atteint l’impression du résultat.
- Le prochain jalon doit traiter le coût structurel (poids réellement
  résidents, graphe MLX, stockage interne ou cache de shards) avant de relancer
  le même scénario, puis seulement ouvrir la validation MTP longue.

### Étape V39 — captures de parité rendues opt-in — 2026-09-03

- Les dictionnaires `lastParityCapture` de GDN, QSA, MoE et des deux
  hyper-connections retenaient auparavant des tenseurs intermédiaires à chaque
  appel, y compris pendant une génération normale. Sur le chemin résident,
  cela pouvait conserver des sous-graphes MLX et gonfler le coût de
  matérialisation.
- Les captures sont maintenant désactivées par défaut. Le layer public expose
  `setParityCapture(true)` et le probe de parité l’active explicitement avant
  son forward ; la couverture des stages de parité est donc conservée sans
  polluer les inférences et le serveur.
- `Scripts/build.sh` avec `xcodebuild` et `Scripts/run-tests.sh` passent : **53
  tests**. Aucun run libre long n’est relancé avant une mesure courte post-
  optimisation ; le gate qualité 8 tokens et MTP restent ouverts.
- Prochaine mesure : smoke résident de 2 tokens avec trace profiler, puis
  comparaison des visites de couches et de la mémoire avec V36. Si le coût
  reste dominé par Lexar, déplacer le checkpoint ou ajouter une stratégie de
  résidence/cache de shards avant toute campagne de qualité.

### Étape V40 — table n-gram POSIX-mapped et smoke résident — 2026-09-03

- `Qwen4ExpLazyNGramStorage` mappe maintenant les fichiers Safetensors de la
  table n-gram via `mmap(MAP_PRIVATE)`. Les lignes sont lues à la demande
  depuis les pages du fichier ; si le mapping échoue, le lecteur revient à
  `FileHandle` sans changer le format ni la sortie. Cette implémentation évite
  la copie complète que Foundation peut effectuer avec `mappedIfSafe` sur
  ExFAT, utilisé par le Lexar (~105 GB pour le checkpoint complet).
- La première vérification a détecté une hypothèse d’alignement invalide sur
  les offsets Safetensors. Les lecteurs mapped UInt32/UInt16 assemblent
  désormais explicitement les octets little-endian, comme le chemin classique.
- Vérification réelle : `flash-ngram-parity --shard 0 --rows 0,1,12345`
  retourne les formes `[3,160] / [3,160]`, `max |delta|: 0`,
  `mean |delta|: 0`, et `parité n-gram lazy: IDENTIQUE`.
- Le smoke résident réel de 2 tokens a terminé avec `[2229, 85648]`, soit
  `Le président`, TTFT `112,426 s`, préremplissage `112,381 s`, décodage
  `236,941 s`, chargement cumulé `96,996 s`, forward cumulé `249,601 s`,
  mémoire MLX active/peak `78,83/80,19 GB`. Le mapping ne provoque plus le
  SIGKILL 137 précédent, mais n’a pas encore rendu le stockage USB assez
  rapide pour un gate libre de 8 tokens.
- La trace profiler est exportée dans
  `/private/tmp/qwen38-flash-v40.trace.json`. Elle montre que les poids restent
  résidents pendant la génération ; le coût restant est principalement le
  chargement/évaluation du chemin MLX sur le volume externe.
- `Scripts/build.sh` via `xcodebuild`, le smoke et la parité n-gram passent.
  Le gate qualité libre 8 tokens et le benchmark MTP restent volontairement
  ouverts. Prochaine étape : vérifier la suite complète de tests puis décider
  entre stockage interne/cache de shards et reprise du gate qualité.

### Étape V41 — intervalle `eval` 48 non rentable sur le Lexar — 2026-09-03

- La suite complète `Scripts/run-tests.sh` a été rejouée via `xcodebuild` et
  reste verte avec **53 tests** après l’introduction de `mmap`.
- Un smoke de 2 tokens avec `--resident-eval-interval 48` a été lancé pour
  remplacer les six synchronisations intermédiaires par une seule. Après plus
  de neuf minutes, il n’avait pas atteint la sortie ; le processus a été
  interrompu proprement (`SIGINT`, code 130), sans crash ni fixture exploitable.
- Le réglage `8` reste le dernier réglage résident mesuré jusqu’au bout
  (`Le président`, trace V40). Repousser `eval` à 48 n’est donc pas retenu par
  défaut : la matérialisation d’un graphe de 48 couches est trop coûteuse ou
  trop risquée sous la contrainte actuelle.
- Conclusion opérationnelle : la qualité et MTP ne sont toujours pas à
  relancer en campagne longue sur ce volume. Il faut prioriser un cache de
  shards/poids réellement résident ou un stockage plus rapide ; le SSD interne
 ne dispose que de 49 GiB libres contre ~105 GiB pour le checkpoint.

### Étape V42 — cache borné de lignes n-gram — 2026-09-04

- Le lecteur POSIX-mapped possède maintenant un cache LRU borné à 4096
  lignes brutes, avec une clé incluant le fichier, l’offset Safetensors,
  la ligne et le type de donnée. Il conserve uniquement les valeurs brutes
  empaquetées/ BF16, jamais des MLXArray ou un graphe Metal.
- Le probe flash-ngram-parity --shard 0 --rows 0,1,12345 retourne
  max delta 0 et parité n-gram lazy IDENTIQUE. La seconde lecture donne
  hits=18, misses=9, entrées=9 et max delta répétition=0.
- La collision initiale entre weight et scales a été détectée par le probe
  puis corrigée en ajoutant l’offset Safetensors dans la clé du cache.
- Scripts/build.sh et Scripts/run-tests.sh via xcodebuild passent avec
  53 tests. Aucun smoke long supplémentaire n’est lancé à ce stade : le
  gain doit être mesuré sur une conversation réelle répétant les mêmes hashes.
- Prochaine étape : exposer ces compteurs dans le rapport profiler Flash-Next,
 puis reprendre le gate qualité lorsque la résidence/stockage sera stable.

### Étape V43 — compteurs n-gram dans le profiler Flash-Next — 2026-09-04

- Le décodeur expose les compteurs cumulés du cache de lignes n-gram :
  hits, misses et entrées. En mode résident, les visites répétées sont
  différenciées par couche pour éviter de compter deux fois les mêmes hits ;
  en mode streamed, chaque lecteur de couche est compté séparément.

- Les générateurs greedy et MTP publient après chaque forward un événement
  Counter nommé Flash n-gram cache dans la session swift-mlx-profiler.
  Les mêmes valeurs sont ajoutées aux métadonnées finales de la trace :
  ngram_cache_hits, ngram_cache_misses, ngram_cache_entries et
  ngram_cache_hit_rate.
- Un test unitaire couvre le calcul lookups/hitRate ; le parseur Swift des
  sources modifiées passe sans erreur.
- La compilation complète CLI n’a pas pu être exécutée depuis ce terminal :
  Xcode 26.6 ne reconnaît ni Package.swift seul ni le workspace SwiftPM
  généré via xcodebuild en ligne de commande dans ce checkout. La validation
  MLX complète doit donc être relancée depuis le Package ouvert dans Xcode,
  conformément à la contrainte default.metallib ; aucun résultat de build
  ou de test n’est revendiqué pour ce sous-jalon.
- Prochaine étape : lancer ce build Xcode, vérifier une trace réelle avec la
  courbe Flash n-gram cache, puis reprendre le gate qualité avant de mesurer
  le gain conversationnel du cache.

### Étape V44 — reset cohérent des compteurs n-gram — 2026-09-04

- La revue du raccord profiler a confirmé que `ProfilingSession.addCounterEvent`
  produit bien des événements Chrome Trace `ph: "C"` et que
  `ChromeTraceExporter` les exporte sur la piste des métriques d'entraînement.
- Correction d'un défaut de remise à zéro en mode résident : les snapshots
  utilisés pour calculer les deltas par couche sont maintenant effacés avec
  les compteurs publics. Après un nouveau tour, le premier forward n'est donc
  plus sous-compté.
- `xcrun swiftc -parse` passe sur toutes les sources et le test touché. La
  validation MLX complète reste à faire via le Package ouvert dans Xcode,
  puis avec une trace réelle contenant plusieurs forwards n-gram.
- Prochaine étape : build/test Xcode, vérifier les métadonnées et la courbe
  `Flash n-gram cache`, puis reprendre le gate qualité Flash-Next.

### Étape V45 — artefact Xcode vérifié, runtime terminal sans device Metal — 2026-09-04

- Après le build Xcode, les binaires Debug ont été régénérés à 19:32 dans
  DerivedData. Le binaire CLI contient bien les chaînes
  `Flash n-gram cache` et `ngram_cache_hit_rate`, ce qui confirme que le
  raccord profiler a été compilé dans le produit utilisé.
- La commande courte `flash-ngram-parity` lancée depuis le terminal s'arrête
  avant l'ouverture du checkpoint avec une exception MLX native
  `NSRangeException: NSArray0 objectAtIndex: index 0 beyond bounds`, dans
  `mlx::core::metal::load_device`. Ce contexte terminal ne publie donc pas de
  device Metal utilisable ; ce résultat ne constitue pas un échec du lecteur
  n-gram, du checkpoint ou de la parité Python.
- La vérification runtime doit être faite depuis l'exécution Xcode/GUI déjà
  autorisée sur la machine, avec une trace réelle. Ne pas utiliser cet abort
  de contexte pour révoquer la parité Python ou la qualité Flash-Next.
- Prochaine étape : exécuter `flash-ngram-parity` ou un smoke Flash-Next depuis
  Xcode, puis inspecter `Flash n-gram cache` dans Perfetto avant le gate qualité.

### Étape V46 — probe Flash et parité n-gram exécutés depuis Xcode — 2026-09-05

- `flash-generate-probe` a terminé avec le code 0 et a exporté
  `/Users/vincent/Downloads/qwen38-flash.trace.json`. Le chemin d'exécution
  Flash-Next et l'écriture de trace sont donc fonctionnels dans le contexte
  Xcode/Metal.
- `flash-ngram-parity --shard 0 --rows 0,1,12345` a terminé avec le code 0 :
  `max |delta| = 0`, `mean |delta| = 0`, parité lazy `IDENTIQUE`, puis
  `hits = 18`, `misses = 9`, `entrées = 9` et delta de répétition nul.
- La parité du lecteur n-gram est validée. La présence des événements `C`
  `Flash n-gram cache` dans la trace exportée reste à contrôler dans Perfetto
  (le terminal Codex ne peut pas lire directement le fichier dans Downloads).
- Prochaine étape : vérifier la courbe et les métadonnées de la trace, puis
  lancer le gate qualité Flash-Next avant les comparaisons de performance.

### Étape V47 — trace profiler Flash-Next vérifiée — 2026-09-05

- La trace `/Users/vincent/Downloads/qwen38-flash.trace.json`, copiée dans le
  dépôt après export Xcode, contient 608 événements et identifie le device
  `applegpu_g15s`.
- Les événements `C` `Flash n-gram cache` sont présents avec deux points
  métriques : après le préremplissage (`hits=720`, `misses=720`, `entries=720`)
  puis après le forward suivant (`hits=768`, `misses=768`, `entries=768`). Les
  métadonnées finales concordent (`hit_rate=0.5000`). Le profiler et le cache
  n-gram sont donc validés sur une trace réelle.
- La même trace donne 96 visites de couche, `94.448259 s` de chargement cumulé
  et `112.431608 s` de forward cumulé. Elle correspond à un probe greedy de
  2 tokens sans image et sans option `--mtp` : le chemin MTP Flash-Next reste
  à valider séparément.
- Le lecteur n-gram conserve sa validation exacte indépendante : delta eager/
  lazy nul et cache double lecture `18 hits / 9 misses`.
- Prochaine étape : lancer le même probe depuis Xcode avec `--mtp
  --mtp-block-size 2`, puis exécuter le gate qualité Flash-Next. La trace
  actuelle constitue la baseline greedy, pas la mesure MTP.

### Étape V48 — tentative MTP directe depuis le terminal — 2026-09-05

- Le binaire CLI fraîchement reconstruit a été lancé directement avec
  `--mtp --mtp-block-size 2`, en écrivant la sortie attendue dans le dépôt.
- Il s'arrête avant le premier chargement avec l'exception native MLX
  `NSRangeException` dans `mlx::core::metal::load_device` (`NSArray0
  objectAtIndex: index 0`). Le terminal Codex ne publie pas de device Metal,
  alors que le même binaire exécuté depuis Xcode a déjà produit la trace
  greedy sur `applegpu_g15s`.
- Aucune trace MTP n'a été écrite pendant cette tentative ; ce n'est pas un
  résultat sur la qualité ou l'acceptance MTP. Le probe MTP doit être lancé
  depuis Xcode avec le scheme CLI et les arguments validés.

### Étape V49 — probe MTP Flash-Next réel et trace analysée — 2026-09-05

- Le probe exécuté depuis Xcode avec `--mtp --mtp-block-size 2` a terminé
  correctement et la trace `qwen38-flash-mtp.trace.json` a été copiée dans le
  dépôt. Les métadonnées indiquent `mtp=true`, device `applegpu_g15s`, bloc 2,
  un round, un token proposé et deux tokens vérifiés.
- Résultat MTP : `0/1` accepté, un rollback et un token rejoué. Le chemin de
  vérification, restauration et replay est donc exercé ; le taux d'acceptation
  n'est pas encore interprétable avec un budget total de seulement 2 tokens,
  qui ne laisse qu'une proposition.
- La génération produit `I am`. Durées observées : TTFT `101.416 s`,
  préremplissage `101.075 s`, décodage `330.357 s`, 144 visites de couche,
  `97.118 s` de chargement et `323.378 s` de forward cumulés, avec un pic MLX
  de `81.89 GB`.
- Les compteurs n-gram de la trace sont cohérents : après préremplissage
  `720/720`, après vérification `816/816`, puis après replay `864/816`, soit
  `864 hits`, `816 misses`, `816 entrées` et un hit-rate final de `0.5143`.
  Le profiler capture donc également les forwards spécifiques au cycle MTP.
- Prochaine étape : refaire depuis Xcode un probe de 8 tokens avec le même
  bloc, afin d'obtenir plusieurs propositions et une mesure d'acceptance
  MTP exploitable ; ne pas comparer la vitesse de ce run au 27B, le coût est
  dominé par le checkpoint Flash-Next streamé sur le Lexar.

### Étape V50 — probe MTP Flash-Next multi-round vérifié — 2026-09-05

- La trace Xcode `qwen38-flash-mtp.trace.json` contient 2 638 événements,
  identifie `applegpu_g15s` et confirme `max_new_tokens=8`, `mtp=true` et un
  bloc MTP de taille 2.
- Le cycle est désormais exercé sur 5 rounds : 5 tokens proposés, 2
  acceptés, donc un taux d'acceptation MTP de 40 %. La présence de plusieurs
  rounds rend cette mesure interprétable, contrairement au probe à 2 tokens.
- Le profil comptabilise 432 visites de couche, `93.124212 s` de chargement
  cumulé et `2109.018566 s` de forward cumulé. Le pic MLX est d'environ
  `78.1 GB` (`77 992 MB`), ce qui confirme que ce run reste dominé par le
  modèle Flash-Next streamé depuis le Lexar et ne constitue pas encore un
  benchmark de vitesse.
- Les événements `Flash n-gram cache` évoluent sur les phases MTP jusqu'à
  `hits=1344`, `misses=1200`, `entries=1200`, `hit_rate=0.5283`. Le profiler
  capture donc bien le cache pendant les forwards multi-round, pas seulement
  le préremplissage.
- Validation acquise : chemin MTP local multi-round, compteurs n-gram et
  trace profiler. Reste à faire : comparer à un run greedy équivalent et
  mesurer le gain MTP sur un checkpoint résident ; la qualité token-à-token
  et le multimodal Flash-Next restent des gates séparés.

### Étape V51 — contrôle du template ChatML Flash-Next — 2026-09-05

- Le binaire Xcode actuel exécute `flash-template-probe` sans initialiser
  Metal ; le rendu peut donc être contrôlé sans payer le forward de 48
  couches. Le prompt thinking activé se termine par
  `<|im_start|>assistant\\n<think>\\n`, ce qui est le préfixe attendu pour
  générer la réflexion.
- Le mode thinking désactivé rend toutefois un bloc vide
  `<think>\\n\\n</think>\\n\\n` après le tour assistant. Ce comportement est
  cohérent avec le template chargé, mais doit rester explicite dans le gate :
  il ne faut pas confondre ce bloc de contrôle vide avec une réflexion
  produite, ni supprimer les balises avant le forward.
- Les IDs ChatML sont stables sur le checkpoint : `<|im_start|>=248045`,
  `<|im_end|>=248046`, `<think>=248068` et `</think>=248069`. Le contrôle de
  l'intégration texte est donc passé ; la prochaine vérification est le gate
  absolu teacher-forced/greedy depuis Xcode, puis le multimodal avec image.

### Étape V52 — trace qualité auto-descriptive — 2026-09-05

- La trace `qwen38-flash-quality.trace.json` confirme un run greedy de 8
  tokens sur Metal (`applegpu_g15s`), sans image ni MTP, avec 384 visites de
  couche et les compteurs n-gram finaux `1728/1608` hits/misses. Elle ne
  contenait toutefois pas le texte généré : un fichier profiler seul ne
  permettait donc pas de trancher le gate sémantique.
- Le probe CLI ajoute désormais aux métadonnées de session `thinking`,
  `mtp`, `prompt_tokens`, `generated_ids` et `decoded`, pour que la prochaine
  trace soit auto-suffisante et vérifiable sans recoller la sortie console.
- `xcrun swiftc -parse Sources/Qwen38CLI/Qwen38CLI.swift` passe. Une nouvelle
  exécution Xcode est nécessaire pour produire la trace enrichie et décider
  le gate qualité ; aucune conclusion sémantique n'est tirée du run V52
  actuel.

### HANDOFF — reprise autonome après V52 — 2026-09-05

État précis au moment de la reprise :

- Validé sur exécution Xcode/Metal : build du chemin Flash-Next, rendu ChatML,
  parité du lecteur n-gram (`delta=0`), trace profiler réelle, résidence
  partielle et cycle MTP multi-round (5 rounds, 5 propositions, 2
  acceptations, rollback/rejeu).
- Validé techniquement mais pas sémantiquement : le probe greedy de 8 tokens
  a terminé avec `exit code 0` et 384 visites de couches. La trace
  `qwen38-flash-quality.trace.json` contient les métriques Metal/n-gram mais a
  été produite avant l'ajout du texte généré dans les métadonnées ; elle ne
  permet donc pas de confirmer le contenu de la réponse.
- Modification en attente de validation Xcode : le CLI exporte maintenant
  dans `ProfilingSession.metadata` `thinking`, `mtp`, `prompt_tokens`,
  `generated_ids` et `decoded`. Le parse Swift du fichier modifié passe.

Reprise recommandée, dans cet ordre :

1. Dans Xcode, rebuilder le scheme `Qwen38MLXSwift-Package` en Debug avec
   `Product > Clean Build Folder`, puis `Cmd+B`. Ne pas utiliser `swift build`.
2. Relancer le scheme CLI sans MTP avec les arguments suivants :

   ```text
   flash-generate-probe
   /Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
   --prompt
   Explique en français qui est le président de la Chine et quel est son rôle.
   --max-new-tokens
   8
   --resident-layers
   --resident-eval-interval
   8
   --trace
   /Users/vincent/Developpements/qwen38-mlx-swift/qwen38-flash-quality-enriched.trace.json
   ```

3. Inspecter dans la nouvelle trace `Session Info.decoded` et
   `Session Info.generated_ids`. Le gate greedy est ouvert seulement si la
   continuation commence par `Le président` et ne contient pas de balises
   ChatML parasites dans la réponse visible. Vérifier également que
   `thinking=false` et `mtp=false` sont bien présents.
4. Si le gate greedy passe, refaire le même contrôle avec `--thinking` et
   vérifier que la réflexion est bornée par `<think>`/`</think>` et retirée
   de la réponse visible. Ensuite seulement lancer le test multimodal avec
   une image.
5. Ne pas comparer les temps de ces probes au 27B : le Flash-Next est encore
   streamé depuis le Lexar. Le benchmark MTP résident et le travail GUI/serveur
   viennent après la qualité et nécessitent un stockage/cache de poids adapté.

Critère de sortie de cette reprise : une trace enrichie lisible et un verdict
explicite `QUALITY_GATE=PASS` ou `QUALITY_GATE=FAIL` basé sur le texte/les IDs,
pas uniquement sur `exit code 0` ou la parité inter-runtime.

### Étape V53 — HANDOFF exécuté : QUALITY_GATE=PASS — 2026-09-05

Reprise effectuée directement depuis le terminal Bash de cette session : à la
différence de tous les essais précédents, ce terminal initialise correctement
un device Metal (`applegpu_g15s`) et peut exécuter les probes `qwen38`
directement, sans passer par une session Xcode manuelle (voir mémoire
`project-flashnext-terminal-metal-access`). Après rebuild propre
(`Scripts/build.sh`), les trois contrôles prescrits ont été exécutés sur le
checkpoint Vontra 4-bit réel :

1. **Greedy, sans thinking, 8 tokens** : `"Le président de la Chine est Xi
   Jinping"` — commence par `Le président`, aucune balise ChatML parasite,
   `thinking=false`/`mtp=false` confirmés. **PASS.**
2. **Thinking, 8 puis 48 tokens** : raisonnement anglais cohérent et factuel
   (« The current President of China is Xi Jinping. The role of the
   President of China is largely ceremonial, but as the General Secretary of
   the Communist Party of China... »), sans dégénérescence sur 48 tokens.
   `</think>` non encore atteint à ce budget — pas un signal d'échec, juste
   un budget insuffisant. **PASS (sanity), fermeture `</think>` à confirmer
   avec un budget plus large.**
3. **Multimodal (image `licensed-image-2.jpeg`, portrait Macron)** : `"La
   personne sur cette image est **Em"` — français correct, début exact du nom
   attendu. **PASS.**

**`QUALITY_GATE=PASS`** — c'est la première confirmation de qualité du
portage Flash-Next depuis le début du Jalon 3 ; le blocage de principe sur la
génération dégénérée (cause : convention des normes, corrigée le 2026-09-02)
est levé. Traces : `qwen38-flash-quality-enriched.trace.json`,
`qwen38-flash-quality-thinking.trace.json`,
`qwen38-flash-quality-thinking-long.trace.json`,
`qwen38-flash-quality-multimodal.trace.json` (racine du dépôt).

**Ce qui reste ouvert avant de considérer l'étape H (§6.1) terminée** :
- Confirmer la fermeture `</think>` en clair sur un budget de tokens élargi
  (~100-150 tokens) et vérifier que la réponse visible post-`</think>` est
  propre.
- Élargir la qualification à au moins 2-3 prompts supplémentaires (pas
  seulement le prompt de référence Xi Jinping) avant de considérer le gate
  robuste et pas un coup de chance sur un seul prompt.
- Le chemin résident streamé depuis le Lexar reste très lent et à charge
  variable (forward cumulé observé entre ~90 s et ~940 s pour 8 tokens selon
  les runs, probablement contention I/O) : ce n'est pas un blocage qualité,
  mais un chantier de performance séparé (résidence groupée, cache de shards,
  ou stockage plus rapide) avant tout débit publiable dans `BENCHMARKS.md`.
- Intégration GUI/serveur (§6.1 étape H, §6.4) : Flash-Next n'apparaît pas
  encore dans le sélecteur GUI ni `/v1/models` — c'est la suite naturelle une
  fois la qualification élargie ci-dessus faite.
- ⛔ **Point de reprise recommandé pour Vincent** : ce jalon de qualité est un
  bon moment pour une démo/décision sur la priorité de la suite (élargir la
  qualification vs. performance résidence vs. intégration GUI/serveur), sans
  qu'aucune GATE formelle du §0.1 ne l'impose.

### Étape V54 — performance résidente : cause racine trouvée (décodeur), second goulot MTP non résolu — 2026-09-05/06

**Fait, corrigé, mesuré deux fois** : le chemin résident (`--resident-layers`)
différait l'évaluation MLX de 8 couches avant de forcer le calcul
(`residentEvaluationInterval` par défaut = 8). Ce n'est pas une optimisation :
ça produit des paliers de calcul énormes et très instables (forward cumulé
observé entre 89s et 943s pour 8 tokens selon les runs, y compris après avoir
relevé la limite GPU wired à 85000 Mo sur suggestion du plan §3 — sans effet,
un run a même été plus lent). Passer l'intervalle à **1** (`eval()` après
chaque couche) donne un forward cumulé stable de **28s** pour la même charge
(8 tokens greedy, gain **~30-50×**), sortie identique (`"Le président de la
Chine est Xi Jinping"`). QSA a été écarté comme cause : sur des prompts courts
`Qwen4ExpQSAIndexer.makeMask` retourne toujours `nil` (fallback dense), pas de
gather top-k coûteux. Correctif appliqué : défaut `residentEvaluationInterval`
1 partout (`Qwen4ExpStreamingDecoder`, `Qwen4ExpStreamingTextModel`, les deux
commandes CLI), commentaire inline dans le code, `xcodebuild` vert. Détails
dans `docs/knowledge/log.md` (2026-09-05, section « cause racine de la
lenteur/instabilité résidente »).

**Non résolu, transmis pour analyse experte** : le même run mais avec `--mtp
--mtp-block-size 2` (8 tokens, 5 rounds, 3/5 acceptés, sortie identique et
correcte) reste très lent : `decode: 1618.100s`, alors que la somme des
durées de couches individuellement profilées (`Flash couche N`, le même code
que le chemin greedy déjà corrigé) ne totalise que **~736s**. Il manque donc
**~880s** qui ne sont couverts par AUCUNE phase du profiler — ce temps est
dépensé entre les appels à `model.forward(...)`, dans
`Qwen4ExpFlashMTPGenerator.generateMTP` / `Qwen4ExpFlashMTPDraftEngine`
(`Sources/Qwen38Core/FlashNext/Qwen4ExpFlashMTP.swift` et
`Qwen4ExpFlashMTPGenerator.swift`), et n'a pas encore été localisé
précisément. Deux suspects identifiés par lecture de code, ni confirmés ni
infirmés par une mesure :

1. `model.snapshot()` (ligne ~141 de `Qwen4ExpFlashMTPGenerator.swift`) copie
   les caches des 48 couches (`caches.mapValues { $0.copy() }`) à **chaque
   round**, pas seulement en cas de rollback — coût potentiellement non
   négligeable si `.copy()` force une réévaluation plutôt qu'un clone léger.
2. `Qwen4ExpFlashMTPDraftEngine.greedyToken` (méthode privée, ligne ~265 du
   même fichier — à ne pas confondre avec la méthode homonyme de
   `Qwen4ExpGreedyGenerator` qui, elle, appelle `eval()`) **n'appelle jamais
   `eval()`** sur le token/logits qu'elle retourne. `state.seedToken` reste
   donc un graphe paresseux non matérialisé jusqu'à sa consommation dans le
   round suivant, ce qui peut reporter le coût du `lm_head` 248K-vocab
   quantifié (`target.global.logits(...)`, appelé plusieurs fois par round
   dans `draftBlock`/`commit`) au moment le moins prévisible, sans jamais être
   compté dans une phase profilée.

**Pour la prochaine analyse (agent expert)** : instrumenter
`Qwen4ExpFlashMTPGenerator.generateMTP` avec des phases `MLXProfiler`
explicites autour de `engine.draftBlock`, `model.snapshot()`,
`engine.commit`, et la boucle `targetIDs = (0..<verifyTokens.dim(1)).map {
... .item(Int32.self) }` (qui fait une synchronisation CPU par token vérifié,
potentiellement coûteuse elle aussi). Rebuild `xcodebuild` puis rejouer
exactement la commande suivante (déjà utilisée pour cette mesure, checkpoint
et prompt identiques au reste de la campagne V53/V54) :

```text
flash-generate-probe
/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
--prompt "Explique en français qui est le président de la Chine et quel est son rôle."
--max-new-tokens 8
--mtp
--mtp-block-size 2
--resident-layers
--trace <chemin>.trace.json
```

Ce goulot est spécifique au chemin MTP local (`Qwen4ExpFlashMTPGenerator`) ;
le chemin greedy/standard (le plus utilisé — GUI, serveur, CLI par défaut)
est déjà corrigé et validé (~28s de decode pour 8 tokens). Le MTP Flash-Next
reste une étape « dégradé acceptable sans MTP » selon le plan (§6.1 étape G),
donc ce n'est pas un blocage pour la suite (élargir la qualification,
intégration GUI/serveur), juste un chantier de performance à part entière.
Attention machine : la session du 2026-09-05/06 a subi deux `kill` pour
mémoire basse à cause de nombreuses sessions Claude Code concurrentes sur la
même machine (jusqu'à 26 processus `claude` simultanés) — vérifier la charge
(`top -l 1 | grep PhysMem`, `ps aux | grep -c claude`) avant de relancer un
run résident à ~80 Go de pic.

### Rév. 4 — réanalyse et réorientation du Jalon 3 — 2026-09-06

Revue faite après V54, en croisant ce fichier, `docs/knowledge/log.md` et le
code (`Sources/`, `Tests/`, `Package.swift`, `.gitignore`, `Vendor/`). Les
sections §6 et §7 ont été **remplacées** (l'ancien texte rév. 3 est dans
l'historique de ce fichier, sauvegarde du jour dans le scratchpad de session) ;
§0.1 (GATEs G-5, G-8, G-4bis) et §11 ont été amendés. Constats qui ont motivé
la réorientation :

- Étapes A-F et vision/E2E faites et validées (gate qualité PASS le 05/09) ;
  étape G faite sans G-5 ; étape H (catalogue/runtime/GUI/serveur) pas
  commencée : `Qwen38ModelValidator` n'accepte que `qwen3_5`.
- Le dossier n'est pas un dépôt git et `.gitignore` exclut
  `Vendor/mlx-swift-lm/`, dont cinq fichiers portent des patchs locaux non
  sauvegardés → H0.1.
- Checkpoint 113 Go, SSD interne 88 Go libres : la copie locale est impossible ;
  la résidence partielle (pic ~80 Go) fonctionne, le chargement one-shot
  (~100 s) n'est pas un blocage en process résident → H avant P.
- Débit greedy résident ≈ 0,3 tok/s après V54, pour 6B actifs : overhead
  d'implémentation > 30× à ventiler au profiler (chantier P).
- Le critère « greedy identique à mlx-vlm » est abandonné (chaos 4-bit +
  référence Python fausse sur ce checkpoint) au profit de Q-B + H6.

**Point de départ pour l'agent d'implémentation** : §6.2, tâche H0.1, puis
strictement dans l'ordre H0 → H1 → H2 → H3 → H4 → H5 (H6 en parallèle de H4-H5)
→ ⛔ G-8. Ne pas commencer P, P-MTP ni §7 avant G-8.

---

<!-- ASK: perf/mémoire Flash-Next bloque H6.1 — 2026-09-07 après-midi -->
## ASK — chantier P à prioriser : le GPU semble inactif, et le mode résident sature la mémoire système

À transmettre à l'agent de planification.

### État d'avancement (2026-09-07)

H0 → H6.5 sont faits et commités dans le dépôt git local créé en H0.1
(`8f4a7ed`…`60cc4a2`) : catalogue double-famille (H1), générateur
streamé avec sampling/continuation (H2), adaptateur runtime Flash-Next
dans `Qwen38Runtime` (H3), GUI (catalogue dynamique, warm-up couche par
couche, MTP/trace génériques — H4), serveur LAN (`family` sur
`/v1/models`, keep-alive SSE pendant le chargement — H5), garde de
régression Q-B (H6.5 : la séquence de 29 IDs de V32 rejouée sur le
checkpoint réel, PASS en 94,6 s, hit-rate et logprob dans les seuils de
V32). 62 tests verts, dont plusieurs rejoués sur les deux checkpoints
réels (27B et Flash-Next) dans le même process. Reste avant G-8 :
H6.1-H6.4 (qualification qualitative élargie), actuellement **bloqués**
— voir ci-dessous.

### Ce qui bloque H6.1 (prompt de référence, thinking, 200 tokens, résident)

Deux échecs distincts, à ne pas confondre l'un avec l'autre :

1. **Mode streamed** (sans `--resident-layers`) : le process avance mais
   très lentement — `ps -o etime,time` a montré **1h48m40s de temps réel
   pour seulement 24m13s de temps CPU cumulé**, avant que je ne l'arrête
   (aucune chance d'atteindre 200 tokens avant plusieurs heures). Le
   ratio ETIME/TIME ≈ 4,5 confirme que le process attend l'essentiel du
   temps plutôt qu'il ne calcule : chaque token recharge les 48 couches
   depuis le Lexar (I/O disque, pas calcul GPU). Vincent confirme ne pas
   entendre le ventilateur pendant ces phases, cohérent avec de
   l'attente I/O.
2. **Mode résident** (`--resident-layers`) : **tué deux fois de suite par
   pression mémoire système** (« stopped because the system is running
   low on memory »), à chaque fois pendant le chargement, avant même
   d'atteindre le décodage en régime établi. Le compresseur mémoire
   macOS est monté à 27 Go juste avant le second kill (contre ~600 Mo au
   repos). Mesure de la mémoire de base **sans** Flash-Next sur cette
   machine à ce moment : **~66 Go déjà utilisés par le reste (Arc, Teams,
   ChatGPT/Codex, Xcode…) sur 96 Go**, soit ~30 Go de marge réelle —
   insuffisant pour le pic résident documenté au §6.0 (« 79-82 Go »).
   Rien dans les commits H2-H6 ne touche l'allocation mémoire de
   `Qwen4ExpStreamingDecoder`/`Qwen4ExpCheckpointLayerLoader` (H4.2 n'a
   ajouté qu'un callback optionnel `onLayerVisited`, sans changer ce qui
   est chargé) : l'hypothèse la plus probable est un écart entre le pic
   documenté et la marge réellement disponible en usage courant de la
   machine, pas une régression introduite aujourd'hui — mais ça reste à
   vérifier, pas à supposer.

### Le doute de fond que Vincent veut faire trancher

Au-delà du diagnostic I/O du mode streamed (point 1), Vincent n'a perçu
**aucune activité ventilateur notable** pendant les tentatives, y
compris quand le process était censé calculer plutôt qu'attendre sur de
l'I/O. Ça rejoint directement le chantier P déjà scopé au §6.2 (« Rien
dans l'architecture ne justifie ce ratio : c'est de l'overhead
d'implémentation ») mais avec un doute plus fort : est-ce que le GPU
dispatch vraiment du travail Metal en régime établi, ou est-ce qu'il y a
quelque chose qui sérialise/attend sans réel calcul (boucle CPU sur les
experts MoE — hypothèse P2(a) déjà notée au §6.2 —, synchronisation
host/device excessive par couche, appel bloquant) ? Le plan reportait ce
chantier après G-8 ; Vincent souhaite qu'un agent de planification
regarde le dossier dès maintenant, avant de relancer d'autres runs
longs qui risquent de re-consommer du temps machine pour rien.

### Demande

1. Décider si P doit être avancé avant G-8 (dérogation à §6.1 point 1 de
   la rév. 4), et si oui prioriser **P1** (profilage Perfetto par
   couche — déjà instrumenté : phases `Flash couche N`, compteur `Flash
   n-gram cache`, `flash-chat-probe --trace` disponible) sur un run
   **court** (résident, 4-8 tokens) plutôt qu'un run de qualification de
   200 tokens, pour obtenir un diagnostic rapidement sans re-déclencher
   les kills mémoire.
2. Vérifier si le pic mémoire résident (§6.0, « 79-82 Go ») est toujours
   exact avec le code actuel (H2-H6 n'y ont normalement pas touché), ou
   s'il a dérivé — et pourquoi le chargement lui-même (avant tout
   décodage) suffit à déclencher la pression mémoire aujourd'hui.
3. Mettre à jour la checklist « avant tout run résident » du §6.0 : la
   marge réelle sur cette machine en usage courant (~30 Go libres avant
   Flash-Next) est plus faible que ce que §6.0 supposait ; préciser un
   seuil chiffré (Go libres minimum) à vérifier avant de lancer, pas
   seulement une vérification qualitative.
4. Une fois P1 fait : statuer sur l'ordre — reprendre H6.1-H6.4 avec les
   paramètres corrigés (résident, RAM libérée au préalable), ou
   d'abord corriger ce que P2 aura trouvé avant de consommer plus de
   temps machine sur des runs de qualification lents.

Contrainte : ne pas relancer de run résident sans vérifier la mémoire
libre au préalable (`top -l 1 | grep PhysMem`) ; viser une marge large
avant de lancer (la pression peut monter en cours de chargement, pas
seulement au démarrage) ; rester sur `xcodebuild`/`Scripts/*.sh`, jamais
`swift build`/`swift test`.

<!-- ANSWER: chantier P avant G-8 — 2026-09-07 -->
## RÉPONSE — chantier P avant G-8 : le GPU est bien inactif, la cause première est la mémoire, protocole court (2026-09-07)

Socle de preuve : les quatre traces de la campagne V53/V54 déjà sur disque
(`qwen38-flash-perf-interval1-8tok`, `-mtp-interval1`, `-wiredlimit85000`,
`qwen38-flash-quality-thinking-long`), relues avec le nouveau
`Scripts/trace-layers.py` ; le code de `Sources/Qwen38Core/FlashNext/` et du
Vendor à `HEAD` (`3c80cae`) ; les sources de mlx-swift 0.31.6 dans
`.xcodebuild/SourcePackages/checkouts/` ; les en-têtes safetensors du
checkpoint ; l'état mémoire de la machine ce jour. Aucun run n'a été lancé.

### R0 — Ce que les traces existantes disent déjà (le doute de Vincent est tranché)

Trace greedy résidente, intervalle 1, 8 tokens (celle du « 28 s » de V54) :

| Passe | Contenu | Durée | Médiane / couche | CPU (tous threads) | GPU |
|---|---|---|---|---|---|
| 0 | préfill = chargement des 48 couches | 93,4 s | 1 876 ms | 58 % | 14 % |
| 1 | premier token décodé (warm-up) | 17,4 s | 200 ms (max 3,9 s couche 24) | 94 % | 1 % |
| 2-7 | **régime établi** | **1,36-1,67 s / token** | **22-28 ms** | **98 %** | **5 % (médiane 0)** |

Le compteur GPU du profiler est `Device Utilization %` lu dans le driver AGX
(swift-mlx-profiler `SystemMetrics.swift`), le compteur CPU vient de `rusage`.
Trois faits en découlent :

1. **En régime établi le process sature exactement un cœur CPU (médiane
   103 %) et le GPU est à 0-5 %.** Vincent a raison : il n'y a pas de calcul
   GPU pendant le décodage. Le ventilateur silencieux est cohérent.
2. **Le « 28 s / 8 tokens » de V54 et du §6.2 P mélangeait deux régimes** :
   17,4 s de warm-up du premier token décodé, puis ~1,6 s par token (≈ 0,6
   tok/s, 25 ms par couche). Le §6.2 P est corrigé en conséquence ci-dessous.
3. Les traces en intervalle 8 (`thinking-long`, `wiredlimit85000`) montrent
   2-3 ms par couche pour les couches **sans** `eval` : c'est le coût CPU de
   construction du graphe Swift/MLX par couche (≈ 120 ms par token sur 48
   couches, réel mais secondaire). Le reste du temps (jusqu'à 19 s par token
   dans ces runs) est passé **dans `eval`, CPU à 93-98 %, GPU à 0-1 %**.

Ce qui est écarté par lecture du code, pas par supposition :
- P2(a) « boucle Swift sur les experts » : `Qwen4ExpSparseMoE` passe par
  `SwitchGLU` upstream (`gatherQMM`) ; le patch Vendor de `SwitchLayers.swift`
  n'ajoute que des initialiseurs empaquetés.
- GDN : kernel Metal `gatedDeltaUpdate` upstream, l'escape hatch
  `MLX_GDN_KERNEL=0` du patch Vendor n'est pas activé.
- Aucun op sur stream CPU hors chargement (`stream: .cpu` uniquement dans
  `loadArraysAndMetadata`) ; le `dequantize → float32` du loader ne sert qu'à
  l'oracle E3, pas au chemin résident.

Restent deux hypothèses, non exclusives, qu'un seul run court départage :

- **H-A (dominante, à confirmer) — pagination et décompression.** Le process
  résident pèse 76,4 Go (MLX actif 75,2 Go, trace V54). MLX **ne wire aucun
  buffer par défaut** : `wired_limit_{0}` dans `MetalAllocator`
  (`allocator.h:70`), et rien dans le code n'appelle `mlx_set_wired_limit`.
  Le `sysctl iogpu.wired_limit_mb=85000` de V54 ne pouvait donc **rien
  changer** (il relève un plafond que personne n'utilise) : le « aucun effet »
  observé était attendu. Conséquence : 76 Go de pages anonymes non wirées +
  la mémoire réelle des autres apps (mesurée ce jour à ~27 Go : 18,3 Go
  anonymes + 6,7 Go dans le compresseur + 2,3 Go de swap, machine au repos
  sans Flash-Next) + 4,5 Go wirés par le noyau ≈ 108 Go > 96 Go. macOS
  compresse donc les pages froides — et 98 % des experts sont froids à chaque
  token (10 experts routés sur 512 par couche). Chaque token touche des pages
  compressées → fautes de page + décompression, comptées en temps CPU du
  thread MLX, mono-thread, GPU en attente. Cette hypothèse explique à elle
  seule : CPU 100 % un cœur, GPU 0 %, la variance de 89 s à 1 436 s entre
  runs identiques (elle dépend de l'état du compresseur au moment du run), le
  warm-up de 17 s au premier token, et les deux kills du 2026-09-07.
- **H-B — coût CPU intrinsèque du chemin Swift/MLX.** Mesuré : 2-3 ms par
  couche de graphe + la couche 1 (PLE n-gram) à 59-69 ms par token (lookup
  hôte `asArray` + lectures mmap sur le Lexar, 52 % de hits, 1 608 misses
  sur 8 tokens). Réel, mais insuffisant pour expliquer 25 ms par couche sur
  les 46 autres couches.

### R1 — Point 1 : oui, avancer un P **borné** avant G-8, avec ce protocole

Dérogation à §6.1 point 1 accordée, parce que H6.1 (200 tokens) est de toute
façon impossible tant que H-A n'est pas neutralisée (kill ou plusieurs heures)
et que le correctif probable est peu coûteux. Le P complet reste après G-8 ;
seules les tâches suivantes passent devant, **dans cet ordre, une journée
machine au plus** :

| # | Tâche | Critère |
|---|---|---|
| P0 | **Micro-bench synthétique sans checkpoint** (zéro risque mémoire, zéro Lexar). Nouvelle commande CLI `flash-layer-bench` : construire une couche GDN+MoE et une couche QSA+MoE aux dimensions réelles (hidden 2 560, 512 experts dim 640, 4 flux, 4-bit g32) avec des poids aléatoires **directement empaquetés** (le test « Les modules Flash-Next peuvent naître directement empaquetés » montre comment), 1 token, 20 pas de warm-up puis 200 pas, `eval` à chaque pas, compteurs `Utilization`/`Memory` du profiler activés, `--trace`. Rapporter ms/pas médian par type de couche et CPU/GPU moyens. | Verdict écrit dans `log.md` : **≤ 5 ms/couche et GPU > 30 %** ⇒ le chemin de calcul est sain, H-A domine, passer à P1. **≥ 15 ms/couche et GPU ≈ 0** ⇒ H-B domine : `sample <pid> 10 -file results/p0-sample.txt` pendant le bench, corriger sur le bench (pas de run résident) avant P1 |
| P1 | **Un seul run résident court** : `flash-chat-probe $FLASH --prompt <référence> --max-new-tokens 6 --trace results/p1.trace.json` (6 tokens : le régime établi commence au 3ᵉ, un run de 200 tokens n'apprendrait rien de plus). Préconditions : `Scripts/preflight-resident.sh` = OK (R3), `Scripts/sample-system.sh results/p1-system.tsv &` lancé avant, et **pendant les tokens 3-6** : `sample $(pgrep -x qwen38) 10 -file results/p1-sample.txt`. Analyse : `Scripts/trace-layers.py results/p1.trace.json`. | Trois chiffres consignés : ms/couche en régime établi, GPU %, et dans `p1-sample.txt` la part des frames `vm_fault`/`_vm_page_decompress`/`kernel` vs `mlx::core`/Swift. **> 50 % noyau ⇒ H-A confirmée**, sinon H-B |
| P2-mem | Si H-A : (1) wirer les poids résidents après le chargement : `mlx_set_wired_limit` via l'API `WiredMemoryManager`/`Memory` de mlx-swift 0.31.6 (`Source/MLX/WiredMemory.swift`, `Memory.swift:312`) à ~78 Go, dans `Qwen38FlashNextEngine` après `Qwen4ExpStreamingTextModel` ; cela **exige** `sudo sysctl iogpu.wired_limit_mb=88000` au préalable (l'API refuse toute valeur > `max_recommended_working_set_size`, ≈ 72 Go par défaut ; le sysctl retombe à 0 à chaque reboot, il vaut 0 ce jour) ; (2) rejouer P1 tel quel. Wirer 80 Go sans la précondition R3 ne fait que déplacer la pression sur les autres apps (swap, puis jetsam possible du process) — les deux vont ensemble. | P1 rejoué : ms/couche et GPU % avant/après dans `BENCHMARKS.md` ; objectif de jauge : GPU > 30 %, < 10 ms/couche |
| P2-code | Si H-B (ou en complément, **après G-8** sinon) : (a) couche 1 : préchauffer les lignes n-gram du prompt et agrandir le LRU (1 608 misses / 8 tokens sur USB) ; (b) coût de graphe 2-3 ms/couche : compter les ops par couche, viser `compile` des sous-blocs stables (hyper-connections, gate MoE) ; (c) `.item()`/`asArray` par token dans `Qwen4ExpPLE.lookup` et le générateur. | mesure sur le bench P0 avant/après, puis P1 |

### R2 — Point 2 : le pic n'a pas dérivé ; le chargement atteint le pic par construction

- `git diff 8f4a7ed..HEAD` sur `Qwen4ExpStreamingDecoder.swift` et
  `Qwen4ExpStreamingTextModel.swift` : uniquement le paramètre
  `onLayerVisited` (H4.2). `Qwen4ExpCheckpointLayerLoader.swift` et
  `Qwen4ExpPLE.swift` sont inchangés depuis la baseline. Aucune dérive de code.
- Pic mesuré (trace V54) : MLX actif **75,2 Go**, process **76,4 Go** — la
  fourchette « 79-82 Go » du §6.0 est celle des runs thinking/image (caches
  plus gros). Le chiffre à retenir pour le budget : **~77 Go de process, ~82
  Go avec image**.
- Pourquoi le chargement seul tue : en mode résident, les 48 couches sont
  matérialisées (`eval(residentWeights)`) **pendant la passe 0**, donc le pic
  est atteint à la fin du chargement, avant le premier token. Le kill « au
  chargement » est simplement le moment où 77 Go rencontrent le reste. Avec
  les ~27 Go réels des autres apps ce jour, le total dépasse 96 Go de ~12 Go :
  le compresseur monte (27 Go observés par Vincent), puis jetsam.
- Le « 66 Go déjà utilisés » de l'ASK vient de `top` (« PhysMem used »), qui
  compte le **cache de fichiers** : ce jour, 67 Go « used » pour 68 Go de
  pages file-backed (le checkpoint lu la veille), `memory_pressure` à 94 %
  libre, plus gros process 1 Go. Ce chiffre ne mesure pas la marge ; la bonne
  mesure est celle de R3.
- Le checkpoint est **uniformément 4-bit g32** (experts `U32 [512, 640, 320]`
  + scales `[512, 640, 80]`, n-gram `[2 500 012, 20]` + `[.., 5]`) : experts
  77,1 Go (61,7 packés + 15,4 scales/biases), n-gram 32,0 Go, reste 4,1 Go.
  Le pic résident de ~77 Go est intrinsèque à ce quant : il n'y a pas de
  « fuite » à chercher dans le loader. Entrée pour G-4bis : une quant maison
  4-bit **g64** ne gagnerait que ~11 Go (scales) ; seul un passage des
  experts en **3-bit g64** (≈ 54 Go d'experts) ramène le pic sous ~60 Go et
  le checkpoint sous 88 Go, avec un risque qualité à mesurer par Q-B.

### R3 — Point 3 : checklist « avant tout run résident », chiffrée

Remplace la vérification qualitative du §6.0 par `Scripts/preflight-resident.sh`
(committé avec cette réponse), qui refuse le run tant que :

- **mémoire anonyme + pages stockées dans le compresseur + swap utilisé >
  9 Go** (dérivation : 96 − 77 process − 4,5 noyau wired − ~5 marge ≈ 9,5 ;
  seuil ajustable par `QWEN38_PREFLIGHT_LIMIT_GB`, à porter à **5 Go** pour
  un run avec image, pic 82 Go). Ce jour : 27,3 Go → REFUS, conforme aux deux
  kills.
- checkpoint absent (Lexar non monté) ; et il affiche `iogpu.wired_limit_mb`,
  le GPU % instantané et les process `qwen38`/python déjà actifs.

En pratique, atteindre 9 Go impose de fermer Arc, Teams, ChatGPT/Codex,
Xcode et toutes les sessions Claude sauf celle qui pilote — le mode résident
4-bit de ce checkpoint n'est viable que sur une machine dédiée au run. Le
sampler `Scripts/sample-system.sh` (toutes les 2 s : anonyme, compresseur,
décompressions, pageins, swap, GPU %, RSS qwen38) doit tourner pendant le
run : si le compresseur dépasse 4 Go pendant le chargement, arrêter le run
soi-même (`kill -INT`) avant que macOS ne le fasse. Les deux chiffres
`Decompressions` et `Pageins` sont cumulatifs ; leur delta par intervalle en
régime établi est la mesure directe de H-A.

### R4 — Point 4 : ordre après P0/P1

1. **P0 immédiatement** (aucun risque, ~1 h de dev, minutes d'exécution).
2. **P1 une fois le préflight vert** (un run, ~5 min hors chargement).
3. Si H-A : **P2-mem**, rejouer P1. Si H-B : **P2-code sur le bench P0**
   jusqu'à < 10 ms/couche, puis P1.
4. **H6.1-H6.4 dans un seul process résident** pour ne payer les ~100 s de
   chargement qu'une fois : `qwen38 serve` (H5) + un script de 5 `curl`
   (H6.1 thinking 200 tokens, H6.2 × 4, H6.3 image, H6.4 deux tours via
   `conversation_id`), préflight vert et sampler actif. Même sans P2, à
   ~1,6 s/token le lot H6 tient en ~15 min si H-A est neutralisée par la
   seule précondition mémoire.
5. **G-8.** Puis P2-code, P-MTP et §7 selon la décision.

Interdit d'ici là : tout run de qualification de 200 tokens avant le verdict
P1 ; tout run résident sans préflight vert ; le mode streamed pour autre
chose qu'une parité (par construction il relit ~80 Go par token depuis le
Lexar : le ratio ETIME/TIME de 4,5 observé est attendu, ce n'est pas une
anomalie à investiguer).

### P0 exécuté — verdict et correction de trajectoire — 2026-09-07 (soir)

Bench synthétique livré (`flash-layer-bench`, commit `9ebb35a`) et rejoué en
**Release** (détail dans `docs/knowledge/log.md`, entrée « P0 rejoué en
Release ») :

| Couche | Debug, calcul seul | Release, calcul seul | GPU |
|---|---|---|---|
| GDN + MoE | 6,5 ms | **5,5 ms** | ~10 % |
| QSA + MoE | 10,3 ms | **5,8 ms** | ~10 % |

`sample` CPU sur le bench : aucune faute de page, temps passé dans le
bookkeeping hôte de MLX (`eval_impl`, caches de buffers/pipelines,
`Concatenate` → `copy_gpu_inplace`, refcounting). Deux faits nouveaux
changent la lecture de la RÉPONSE ci-dessus :

1. **Tous les runs depuis le 29 août étaient des builds Debug** (défaut de
   `Scripts/build.sh`) : C++ de MLX non optimisé. Release divise le coût hôte
   par ~1,2 (GDN) à ~1,8 (QSA).
2. **Le profiler coûte ≈ 4,7 ms par phase** (IOKit + rusage à chaque
   `start`/`end`), soit ~225 ms par token dans les traces résidentes qui
   enveloppent chaque couche. Les « 22-28 ms/couche » de V54 en contiennent
   ~5 ; le reliquat (Debug) est cohérent avec le bench. **H-B (coût hôte)
   est majoritaire ; H-A (mémoire) n'est plus qu'un reliquat à mesurer.**

Conséquences immédiates (remplacent l'ordre R4 pour les points 1-3) :

| # | Tâche | Critère |
|---|---|---|
| P0-b | `Scripts/build.sh` : le défaut reste Debug pour les tests ; ajouter `Scripts/build-release.sh` (wrapper `QWEN38_CONFIGURATION=Release`) et faire pointer les probes/H6/G-8 sur `.xcodebuild/Build/Products/Release/qwen38`. Documenter dans `PLAN.md` §6.2 conventions et dans le README. | binaire Release à jour, GUI et CLI |
| P0-c | Profiler par couche **opt-in** : dans `Qwen4ExpStreamingDecoder.forward`, n'appeler `profiler.start/end("Flash couche N")` que si un flag (`Qwen4ExpStreamingDecoder.profileLayers`, défaut `false`, activé par `--profile-layers` en CLI) est levé ; garder les phases `Prefill`/`Generation` et les compteurs n-gram. | bench P0 et trace résidente sans phase par couche : `--trace` ne coûte plus ~225 ms/token |
| P2-code | Sur le bench P0 (Release, sans checkpoint, boucle de minutes) : (a) compter les ops par pas (nombre d'appels `eval_gpu` via `sample` ou instrumentation) ; (b) `MLX.compile` du forward de couche (entrées : hidden 4 flux, cache) si les formes sont stables en décodage ; (c) supprimer les `concatenated`/`split` des 4 flux hyper-connections (vues sur un seul tenseur) ; (d) `asyncEval` : lancer la couche N+1 pendant que le GPU exécute N, un seul `eval` par token, à re-mesurer en Release (le verdict V54 sur l'intervalle 8 est contaminé par Debug + profiler + mémoire). Jauge : **≤ 2 ms/couche, GPU ≥ 40 %** sur le bench. | tableau avant/après par levier dans `log.md` |
| P1 | Inchangé (préflight vert obligatoire), mais en **Release** et avec `--profile-layers` désactivé : mesure du reliquat H-A par le sampler (`Decompressions`/`Pageins` par intervalle) et `sample` CPU (part `vm_fault`). | verdict chiffré H-A |
| Suite | P2-mem seulement si P1 montre > 20 % du temps en fautes de page ; puis H6 en un seul process serveur (Release) ; puis G-8. | — |

### P2-code exécuté — verdict, et protocole P1 définitif — 2026-09-07 (soir, suite)

Leviers mesurés sur le bench P0 en Release (`docs/knowledge/log.md`, entrée
« P2-code ») : `compile` sans effet (−2 % GDN, +13 % QSA) ; hyper-connections
déjà écrites sans découpage ; masque QSA superflu supprimé en décodage
(production, effet marginal) ; **`asyncEval` avec un seul `eval` bloquant
par groupe : −13 à −17 % par couche et GPU occupé 82 % (`ioreg`) au lieu de
0 %**. Le comptage des ops montre un coût réparti sur ~100 petits ops par
couche (copies, binaires, matmuls quantifiés, réductions) : c'est le nombre
de lancements de noyaux qui plafonne à ~4,5 ms par couche, pas un op
coupable. Plafond structurel actuel : 48 × 4,5 ≈ 215 ms par token ≈ 4,6
tok/s, suffisant pour H6 et G-8 ; la fusion d'ops (moins de noyaux par
couche) est un chantier post-G-8.

**Changement de production ajouté** : option `residentAsyncEval` (défaut
`false`) sur `Qwen4ExpStreamingDecoder` → `Qwen4ExpStreamingTextModel` →
`Qwen38FlashNextEngine`, flag `--resident-async` sur `flash-chat-probe` et
`flash-generate-probe` : en résident, `asyncEval` sur chaque couche
intermédiaire, `eval` bloquant sur la dernière seulement. 65 tests verts,
Release construit.

**P1 — protocole définitif (Release, préflight vert obligatoire)** : trois
runs de 6 tokens sur le prompt de référence, sans `--profile-layers`, avec
`Scripts/sample-system.sh results/p1-system.tsv &` lancé avant et arrêté
après, et pendant les tokens 3-6 du run (i) un `sample $(pgrep -x qwen38) 10
-file results/p1-sample.txt` :

| Run | Commande (binaire `.xcodebuild/Build/Products/Release/qwen38`) | Ce qu'il mesure |
|---|---|---|
| (i) | `flash-chat-probe $FLASH --prompt <référence> --max-new-tokens 6 --resident-layers --trace results/p1-i.trace.json` | régime établi de référence, Release, `eval` par couche |
| (ii) | idem + `--resident-async` | recouvrement CPU/GPU en résidence réelle |
| (iii) | idem + `--resident-eval-interval 48` (sans `--resident-async`) | un seul `eval` par token, pour clore le verdict V54 en conditions propres |

Verdict attendu, chiffré dans `log.md` : s/token en régime établi (phase
`Generation` de la trace et horloge du générateur), GPU % (`ioreg` du
sampler), delta `Decompressions`/`Pageins` par intervalle (part H-A), et pour
(i) la part noyau dans `p1-sample.txt`. Décision : la variante la plus rapide
devient le défaut de `Qwen38FlashNextEngine` (GUI, serveur) avant H6. Si le
sampler montre des décompressions soutenues pendant le décodage, P2-mem
(wiring MLX + sysctl 88000) passe avant H6 ; sinon on enchaîne H6 en un seul
process serveur, puis G-8.

### P1 exécuté — 2026-09-08

Détail dans `docs/knowledge/log.md` (entrée « P1 »). Trois faits :

1. **Le Mac s'endormait pendant les runs** (`sleep 1` sur batterie et
   secteur ; Idle Sleep journalisés pendant V53, V54 et les essais H6.1).
   Correctif : assertion anti-veille posée par `Qwen38FlashNextEngine`
   pendant la résidence ; préflight étendu (alimentation, assertions).
2. **Mémoire** : sur machine propre (7 Go d'anonyme après reboot) le mode
   résident passe sans compression, pic 75,2 Go. En usage courant (19-27 Go)
   macOS compresse le modèle dès 57 Go de RSS : c'est le décodage CPU-bound
   d'hier. Plafond structurel, pas un bug ; sortie par §7 (3-bit) ou experts
   mappés en fichier — à décider à G-8.
3. **Variante retenue : `asyncEval` par couche** (0,47 s/token contre 0,60
   avec `eval` par couche et 2,05 avec un `eval` différé par token), défaut
   du moteur GUI/serveur. GPU 32 % pendant la génération : plafond du chemin
   à ~100 noyaux par couche, fusion d'ops = chantier post-G-8.

**Suite immédiate** : H6 en un seul process serveur (`qwen38 serve`,
Release, machine propre, cinq `curl`), puis ⛔ G-8.

### H6 — deux tentatives arrêtées par la mémoire, protocole et tâches — 2026-09-08 (soir)

Détail dans `log.md` (« H6 : deux tentatives »). Fait établi : **le mode
résident (76 Go de process) n'a de marge que si les autres applications
occupent ≤ ~8 Go de mémoire anonyme** ; avec 24 Go (VM UTM comprise) le noyau
compresse 60 Go du modèle en dix secondes, même cache de fichiers purgé.

| # | Tâche | Critère |
|---|---|---|
| H6 (protocole) | Reboot, Terminal seul, préflight ≤ 8 Go, `Scripts/h6-qualification.sh` sans délai. Filet à 30 Go de compresseur uniquement. | `results/flash-qualification-rev4.tsv` complet, verdict par item |
| P2-mem-a | `Qwen4ExpCheckpointLayerLoader` (et le loader global/vision) : lire les tenseurs via un `FileHandle` avec `fcntl(F_NOCACHE)` + `pread` à l'offset du header safetensors, puis `MLXArray(buffer, shape, dtype)` ; ne plus passer par `loadArraysAndMetadata` pour les couches résidentes. Garde : parité bit-exacte avec l'ancien chemin sur une couche (test), `flash-teacher-forced-score` inchangé. | sampler : `file_gb` reste < 3 Go pendant tout le chargement ; marge apps mesurée ≈ 16 Go au lieu de ~3 |
| P2-mem-a — **fait, 2026-09-09** | `Qwen4ExpUncachedTensorReader` + `Qwen4ExpSafetensorsHeader` (parseur d'en-tête partagé avec la table n-gram) ; `uncachedIO` (défaut `true`) propagé aux 4 loaders et jusqu'à `--cached-io` en CLI. Détail et tableau de mesures : `log.md` 2026-09-09. | Bit-exact confirmé sur 3-bit et 4-bit (mêmes `generated ids`, mêmes pic MLX) et Q-B V32 inchangé (10/28 · −4,80). Mécanisme F_NOCACHE **prouvé correct au niveau syscall** (0,00-0,01 Go de croissance du cache pour un shard de 5,3 Go relu à froid, dans les deux sens de lecture) mais **le critère sampler `file_gb < +3 Go` n'a pas pu être démontré proprement** de bout en bout dans cette session (mesures bruitées par l'enchaînement de runs sans reboot, `file_gb` étant une métrique système entière) — voir `log.md` pour l'analyse complète. Gardé à `true` par défaut : aucune régression mesurée, correction prouvée à la source. Un protocole H6 (reboot, un seul run par variante) resterait à faire pour trancher le sampler. |
| G-4bis (à G-8) | Choisir la réduction d'empreinte : (B) quant maison 3-bit g64 des experts depuis les shards HF BF16 (streaming, ~54 Go d'experts, process ≈ 60 Go) ou (C) requantification 4-bit → 3-bit du checkpoint Vontra (pas de téléchargement, qualité à valider par Q-B, faisable en heures). | décision Vincent |

### G-4bis levée — décision Vincent du 2026-09-08 : tenter le 3-bit (option C), puis viser le déchargement disque

Décision : requantifier les **experts** du checkpoint Vontra de 4-bit g32 vers
**3-bit g64** (option C, sans téléchargement HF), valider par Q-B et un greedy
de référence, mesurer le pic mémoire. Attendu : experts 77 → ~53 Go, process
résident ≈ 60 Go, ce qui laisse ~30 Go au reste de la machine. Vincent
anticipe que la solution de fond sera ensuite le **déchargement disque des
experts** (experts mappés en fichier sur le SSD interne, seuls les 10 experts
routés par couche et par token copiés vers le GPU) : inscrit comme chantier
P3 post-G-8. Contraintes techniques vérifiées : MLX 0.32.2 (`venv617`)
quantifie en 3-bit (`[.., 240]` pour 2560 entrées) ; côté Swift,
`Qwen4ExpQuantizationSpec` exige `32 % bits == 0` (à relaxer : la contrainte
réelle est `in × bits % 32 == 0`) ; Lexar : 192 Go libres.

| # | Tâche | Critère |
|---|---|---|
| Q3.1 | `Scripts/qwen4-exp-requantize-experts.py` (Python `venv617`) : shard par shard, pour chaque tenseur `*.mlp.switch_mlp.{gate,up,down}_proj.weight` avec ses `.scales`/`.biases` : `mx.dequantize(4, g32)` → `mx.quantize(bits=3, group_size=64)` ; tous les autres tenseurs recopiés tels quels (n-gram, attention, shared expert, gates, normes) ; `mx.eval` par tenseur, un shard en mémoire à la fois ; nouvel index et `config.json` avec `"quantization": {"group_size": 32, "bits": 4, "mode": "affine", "experts": {"group_size": 64, "bits": 3, "mode": "affine"}}`. Sortie : `/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`. Rapport : taille totale, taille experts, erreur de reconstruction moyenne/max sur 3 experts (4-bit vs 3-bit vs les deux). | dossier complet, `index.json` cohérent, taille experts ≈ 53 Go |
| Q3.2 | Swift : `Qwen4ExpQuantization` lit l'override `experts` ; `Qwen4ExpQuantizationSpec` relaxe la précondition ; `Qwen4ExpSparseMoE` construit `SwitchGLU` avec le spec experts ; loader résident et `dequantize` (oracle E3) choisissent le spec selon la clé ; `Qwen4ExpLayerBench --expert-bits 3` pour mesurer le débit 3-bit. Tests : parse de l'override, module empaqueté 3-bit, bench réduit. | 65+ tests verts, Release construit |
| Q3.3 | Validation sur le nouveau checkpoint (machine propre, préflight, `caffeinate`) : (a) `flash-teacher-forced-score` sur la séquence V32 — rapporter hit-rate et logprob face aux 10/28 et −4,4 du 4-bit ; (b) `flash-chat-probe … --temperature 0 --max-new-tokens 8 --resident-layers` sur le prompt de référence ; (c) pic mémoire et s/token via le sampler. | tableau 4-bit vs 3-bit dans `log.md` ; décision « qualité acceptable ? » posée à Vincent |
| P3 (post-G-8) | Déchargement disque des experts : experts sur SSD interne, mmap, gather des 10 experts routés vers un tenseur `[10, out, in]` par couche et par token, LRU d'experts chauds ; à concevoir après Q3. | étude puis prototype sur le bench |

### H6 PASS sur le 3-bit — fiche de démo ⛔ G-8 — 2026-09-08 (nuit)

H6 : 8/8 PASS (`results/flash-qualification-rev4.tsv`, `log.md` « H6 »),
`BENCHMARKS.md` section Flash-Next ajoutée. Correctif serveur : image
acceptée sur le dernier message utilisateur en mode stateless.

**Démo G-8 (Vincent)** — checkpoint `local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`,
binaire Release (`Scripts/build-release.sh`), machine dans son état normal :
1. GUI : sélectionner le 3-bit dans le catalogue (`/Volumes/Lexar/models/local`),
   chargement ~60 s avec progression, puis texte / thinking / image / deux tours.
2. Serveur : onglet Serveur ou `qwen38 serve --model-path …`, puis depuis
   l'autre Mac : `curl -N` streaming, SDK `openai`, image en `data:` base64.
3. Comparer au 4-bit si la machine est propre (reboot) : même sortie greedy,
   0,47 s/token contre 0,22.

**Questions à trancher à G-8** :
- (a) H validé ; G-1 et G-3 actées rétroactivement ?
- (b) Checkpoint de référence pour la suite : 3-bit (tient avec les apps
  ouvertes, 2× plus rapide, −0,42 nat de logprob) ou 4-bit (qualité maximale,
  machine dédiée) ?
- (c) Ordre des chantiers post-G-8 : P3 déchargement disque des experts
  (retour au 4-bit sans la contrainte mémoire, selon l'intuition de Vincent) ;
  P2-fusion (moins de noyaux par couche, GPU > 32 %) ; P-MTP ; 3-bit direct
  depuis BF16 (option B) si le 3-bit devient la référence.
- (d) Réglage de veille du Mac (`sleep 1` sur secteur) : le moteur pose une
  assertion, mais les probes CLI et les builds longs restent exposés.

### P2-fusion — réduire le nombre de noyaux par couche — lancé le 2026-09-09

Constat (P0/P2-code) : ~4,5 ms par couche en `asyncEval` pour ~0,5 ms de
calcul utile, GPU 32 % en génération ; le comptage par `sample` donne, par
famille, Copy ≈ 260, binaire ≈ 270, QuantizedMatmul+qmv ≈ 77, Concatenate
62, Reduce 39, unaire 36, SDPA 33 échantillons : les copies et les petites
opérations élémentaires coûtent plus que le matmul quantifié. Objectif :
≤ 2,5 ms par couche sur le bench (`flash-layer-bench --async-interval 8`,
Release), soit ~8-10 tok/s en génération, à sortie **bit-identique** sur le
prompt de référence et Q-B inchangé. Méthode inchangée : un levier = une
mesure avant/après = un commit, même négatif.

| # | Levier | Critère |
|---|---|---|
| F1 | GDN : fusionner `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` en un seul `QuantizedLinear` (concaténation des poids packés, scales et biases sur l'axe de sortie **au chargement**, découpage par `split` après le matmul) ; idem pour tout autre groupe de projections partageant la même entrée (QSA q/k/v, indexeur). | ms/pas bench, IDs identiques, parité GDN publique verte |
| F2 | Normes : toute RMSNorm zéro-centrée (`hc_norm`, q/k norm, indexeur, PLE) passe par `MLXFast.rmsNorm` avec un poids `1 + w` (ou `w` pour Vontra, cf. piège 12) **précalculé au chargement**, plus aucune addition ni cast par appel ; vérifier que `Qwen4ExpRMSNorm` n'upcaste pas en float32 quand ce n'est pas nécessaire. | idem |
| F3 | Hyper-connections : relire `Qwen4ExpHyperConnection.swift` et `Qwen4ExpDecoderLayer.inject` ; remplacer les `expandedDimensions` + broadcast + `mean` + `reshaped` par un ou deux matmuls batchés sur `[B, S, 4, hidden]` et supprimer les copies (`Copy`/`Concatenate` dominants en (a)). | idem, test « mélangent quatre flux » vert |
| F4 | MoE : `softmax(precise:)` → mesurer sans `precise` (perte de précision acceptable ? comparer les indices routés sur 200 tokens synthétiques) ; `argPartition` + `takeAlong` + normalisation + `weightedExpertSum` : éliminer reshapes et casts intermédiaires ; expert partagé : gate sigmoid fusionnée. | idem, test MoE vert |
| F5 | Casts : inventorier tous les `asType` par pas (GDN gating en float32, MRoPE, masques) et ne garder que ceux exigés par la numérique (état GDN float32, tables RoPE float32 — piège 6). | idem |
| F6 | `MLX.compile` **par sous-graphe élémentaire stable** (gating GDN, mix hyper-connections, routage MoE), pas sur la couche entière (P2-code (c) : +13 % en QSA). | idem |
| F7 | Validation finale sur le checkpoint 3-bit : `flash-chat-probe … --temperature 0 --max-new-tokens 8 --resident-layers --resident-async` (IDs identiques `[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171]`), garde Q-B (`xcodebuild … '-only-testing:Qwen38Tests/flashTeacherForcedRegressionGuardV32()'` avec `TEST_RUNNER_QWEN38_FLASH_MODEL`, attendu 10/28 −4,80), s/token avant/après dans `BENCHMARKS.md`. | tableau final dans `log.md` |

**Statut — 2026-09-09** : F1 (fusion GDN/QSA input projections), F2 (normes
`1+w` précalculées, `MLXFast.rmsNorm` non groupé) et F4 (`softmax(precise:
false)`, routage vérifié sur 200 tokens synthétiques) implémentés,
parité-validés (`flash-layer-bench --check-parity`, 74 tests verts), mais
**aucun gain ms/pas mesurable** au protocole (`--steps 300 --async-interval
8` : ~4,5 ms/pas GDN/QSA à tous les niveaux 0→4, GPU 45-46 % inchangé) —
conservés en option (`--fusion-level`/`Qwen4ExpFusionLevel`, défaut `.none`
partout, comportement de production inchangé) par analogie avec P2-code
(e), pas par gain démontré. F3 : rien à changer (re-confirme P2-code (b)).
F5 : audit, rien au-delà de F2. F6 : non implémenté (gating GDN dans le
paquet vendu épinglé ; faisceau de preuves F1/F2/F4 + P2-code (c) rendant un
gain improbable). **F7 non exécutée** : préflight REFUS (43,9 Go à évincer,
`qwen38-bench-ui` actif à 24 Go — machine jugée occupée par Vincent),
commandes prêtes (`--fusion-level` câblé jusqu'à `flash-chat-probe`) pour la
prochaine session avec la machine libre. Détail complet, tableau de mesures
et écarts assumés : `docs/knowledge/log.md`, « 2026-09-09 — P2-fusion :
leviers F1-F7 ».

### P-MTP (suite) — PM4 : vérification sans rejeu — 2026-09-09

État : normes de la tête MTP corrigées (loader + suffixes `pre_fc_norm_*`,
audit HF tenseur par tenseur, `log.md` « P-MTP (suite) ») → acceptation
24 % → **47,6 %** en bloc 2 sur le 3-bit, IDs identiques au greedy. Le MTP
reste 20 % plus lent que le greedy parce que chaque rejet rejoue un forward
sur le préfixe accepté (`Qwen4ExpFlashMTPGenerator.swift:249-254`). La mesure
d'acceptation sur le 4-bit a été tuée par la mémoire (23 Go d'apps) : à
refaire sur machine propre, elle dit si le 3-bit limite le drafter.

| # | Tâche | Critère |
|---|---|---|
| PM4.1 | États GDN par token : variante `gatedDeltaUpdateWithStates` qui rend, en plus de la sortie, l'état récurrent **après chaque token** (`[B, T, Hv, Dv, Dk]` float32). Deux implémentations : (a) boucle ops (copie de `gatedDeltaOps` upstream, dans notre code, T ≤ 4) ; (b) extension du kernel Metal upstream (buffer de sortie optionnel) si (a) coûte > 1 ms par couche GDN sur le bench. Utilisée uniquement par le forward de vérification MTP. | test : états intermédiaires = états obtenus par T forwards à un token (1e-5) |
| PM4.2 | Rollback sans rejeu : pendant la vérification, chaque couche conserve ses états par token ; sur rejet à la position k, `MambaCache` reprend l'état après k tokens acceptés et `Qwen4ExpQSAKVCache.trim(rejetés)` tronque K/V + clés indexeur + positions (vérifier que `trim` couvre bien les trois, `Qwen4ExpCache.swift:138`). Plus de `model.restore` + `model.forward` de rejeu ; `snapshot` devient inutile. Le prédicteur MTP (1 couche QSA) subit le même `trim`. Cache n-gram/PLE : vérifier qu'il n'a pas d'état séquentiel à rembobiner (contexte n-gram = fenêtre sur les IDs → recalculé à partir des IDs acceptés). | IDs identiques au greedy sur 32 tokens, blocs 2/3/4 ; `stats.replayedTokens == 0` |
| PM4.3 | Mesure 3-bit, 32 et 128 tokens : decode MTP bloc 2/3 vs greedy ; cible : **MTP bloc 2 ≤ 0,8 × greedy** à 48 % d'acceptation. Si atteint : brancher (PM3 : `Qwen38FlashNextEngine`, `mtpState = .active`, streaming des tokens acceptés, GUI/serveur honorent le toggle), défaut **off** tant que Vincent n'a pas validé en GUI. | tableau `log.md`, ligne `BENCHMARKS.md` |
| PM4.4 | Sur machine propre (Vincent) : acceptation bloc 2 sur le 4-bit. | chiffre dans `log.md` |
