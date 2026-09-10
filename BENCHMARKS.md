# Qwen3.8 benchmarks

Chaque run conserve : modèle, quantification, version des dépendances, prompt
tokens, tokens générés, TTFT, prefill tok/s, decode tok/s, mémoire MLX et trace
Chrome. Les lignes `conversation-benchmark` correspondent au même processus
sur trois tours ; `cache historique rejoué` est le comportement M1 MTP attendu.

| Date | Modèle | Quant | Prompt | Générés | TTFT | Prefill | Decode | Peak mémoire | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 68 | 16 | 2.50 s | 27 tok/s | 10.4 tok/s | GUI | text, greedy, limit 16 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 319 | 24 | 3.67 s | 87 tok/s | 10.3 tok/s | GUI | local PNG, greedy, limit 24 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 67 | 24 | 1.19 s | 57 tok/s | 10.7 tok/s | — | ChatSession, greedy, limit 24 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-8bit | 8-bit | 69 | 24 | 5.85 s | 12 tok/s | 6.9 tok/s | — | ChatSession, greedy, limit 24 |
| 2026-08-29 | mlx-community/Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 59 | 2 | 0.522 s | 118 tok/s | 21.4 tok/s | — | M1 upstream, greedy, MTP actif, 1 drafter apparié |
| 2026-08-29 | mlx-community/Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 572 | 24 | 5.669 s | 102 tok/s | 18.3 tok/s | — | M1 upstream, première inférence VLM, image, 12 proposés / 10 acceptés |
| 2026-08-29 | mlx-community/Qwen3.8-27B-8bit + MTP-8bit | 8-bit + MTP | 59 | 2 | 0.743 s | 82 tok/s | 10.9 tok/s | — | M1 upstream, greedy, MTP chargé, propositions non observées sur ce run court |
| 2026-08-29 | Qwen3.8-27B-4bit | 4-bit | 1279 → 1441 → 2491 | 140 → 1022 → 1031 | 12.504 → 18.183 → 56.484 s | 103 → 79 → 44 tok/s | 10.29 → 2.87 → 8.37 tok/s | 17,075 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 1279 → 1441 → 2385 | 140 → 916 → 884 | 13.403 → 15.098 → 23.409 s | 96 → 96 → 102 tok/s | 12.12 → 12.70 → 10.89 tok/s | 17,885 MiB MLX | M1, 78/63 · 508/408 · 480/404 proposés/acceptés, historique rejoué ; probe mémoire |
| 2026-08-29 | Qwen3.8-27B-8bit | 8-bit | 1279 → 1415 → 2263 | 114 → 820 → 936 | 22.256 → 17.794 → 22.986 s | 58 → 80 → 99 tok/s | 4.35 → 6.20 → 7.22 tok/s | 31,483 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-8bit + MTP-8bit | 8-bit + MTP | 1279 → 1419 → 2326 | 118 → 879 → 646 | 30.991 → 27.335 → 29.829 s | 41 → 52 → 78 tok/s | 3.41 → 4.53 → 4.56 tok/s | 31,483 MiB MLX | M1, 64/55 · 502/378 · 363/284 proposés/acceptés, historique rejoué ; probe mémoire |
| 2026-08-29 | Qwen3.8-27B-bf16 | bf16 | 1279 → 1415 → 2280 | 114 → 837 → 934 | 14.324 → 11.663 → 19.805 s | 90 → 122 → 115 tok/s | 4.84 → 4.85 → 3.96 tok/s | 53,093 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-bf16 + MTP-bf16 | bf16 + MTP | 1279 → 1415 → 2186 | 114 → 743 → 859 | 11.900 → 11.883 → 29.924 s | 108 → 119 → 73 tok/s | 3.64 → 3.80 → 2.22 tok/s | 54,332 MiB MLX | M1, 62/53 · 413/330 · 471/388 proposés/acceptés, historique rejoué ; probe mémoire |

Les pics ci-dessus sont les valeurs `MLX.Memory.peakMemory` des probes
`maxTokens=1` (un probe par variante et mode MTP), et non une estimation RSS.
Pendant les longs runs bf16, l’allocation driver observée a atteint environ
66,8 GiB sans MTP et 78,0 GiB avec MTP ; cette différence est conservée dans
le journal pour distinguer mémoire MLX suivie par le runtime et allocation
driver macOS.

Important : M1 est la baseline de l’itérateur upstream (`blockSize=2`, donc un
seul token drafté par round). Les compteurs d’acceptation sont bien mesurés,
mais les premiers runs historiques ont montré une divergence sur les tours
avec replay d’image. Après correction du rollback GDN et ajout du fallback
M-RoPE, le contrôle trois tours 4-bit à 512 tokens est identique : tour 1 MTP
actif (63/78 acceptés), tours 2–3 en fallback standard (0 proposition), car
M1 ne sait pas transporter la table M-RoPE image vers le cache privé du
drafter. Ce contrôle valide la sécurité de sortie, mais pas encore le gain
net multi-tour; celui-ci attend le pipeline M2 persistant.

## Flash-Next (`qwen4_exp`) — mode résident, Release, `asyncEval` par couche — 2026-09-08

Prompt de référence « Explique en français qui est le président de la Chine et
quel est son rôle. » (29 tokens), greedy, `flash-chat-probe --resident-layers
--resident-async`, checkpoint streamé depuis le Lexar (USB, ExFAT). TTFT =
chargement one-shot des 48 couches (payé une fois par process ; GUI et
serveur le gardent résident). Détails : `docs/knowledge/log.md` (P1, Q3.3, H6).

| Date | Checkpoint | Experts | Taille | TTFT (chargement) | Décodage | Pic MLX | Q-B V32 (hits · logprob) | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| 2026-09-08 | Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP | 4-bit g32 | 113 Go | 90,9 s | 0,47 s/token (2,1 tok/s) | 75,2 Go | 10/28 · −4,38 | ne tient qu'avec ≤ ~8 Go d'autres apps ; `eval` par couche : 0,60 s/token ; un `eval` différé par token : 2,05 s/token |
| 2026-09-08 | local/Qwen3.8-Flash-Next-MLX-e3bit-MTP (Q3.1) | **3-bit g64** | 84 Go | 60,2 s | **0,22 s/token (4,6 tok/s)** | **56,6 Go** | 10/28 · −4,80 | sortie greedy identique au 4-bit ; H6 8/8 PASS via le serveur (48 tokens ≈ 10 s, image 24 tokens 4,8 s après vision) |

| 2026-09-09 | local/Qwen3.8-Flash-Next-MLX-e3bit-MTP — **GUI** (Release, lancée depuis le terminal), thinking élevé, 2048 max | 3-bit g64 | 84 Go | 48,4 s | **5,4 tok/s** (1 072 tokens en 200 s) · prefill 24,7 tok/s, TTFT 2,39 s (59 tokens) | 52,9 Go | — | démo G-8 : premier tour GUI Flash-Next, MTP en fallback attendu |

### MTP local Flash-Next — branchement opt-in (PM4.3, 2026-09-09)

Même prompt de référence, checkpoint 3-bit, `--resident-layers --resident-async`,
IDs identiques au greedy dans les deux colonnes (`docs/knowledge/log.md`
« PM4 : vérification MTP sans rejeu »). « Avant » = P-MTP PM3 (rejeu
`model.snapshot()`/`restore()` par rejet) ; « après » = PM4.1/PM4.2 (rollback
sans rejeu) + PM4.3 branché dans `Qwen38FlashNextEngine`
(`options.mtp.enabled`, opt-in, défaut off).

| Variante | tokens | avant (PM3, avec rejeu) | après (PM4, sans rejeu) | ratio après/greedy |
|---|---:|---:|---:|---:|
| Greedy (référence) | 128 | 21,16 s | 21,16 s | 1,00 |
| MTP bloc 2 | 128 | ~1,19-1,6× le greedy (plus lent) | **18,13 s** | **0,86** |

Validation du branchement (serveur, `qwen38 serve --model-path
.../Qwen3.8-Flash-Next-MLX-e3bit-MTP`) : deux `curl` 32 tokens
`"temperature":0` sur le même prompt, `"mtp":false` puis `"mtp":true` —
`content` identique dans les deux réponses, 7,95 s contre 7,42 s
(`mtpAccepted: 10`, `mtpProposed: 21`, `mtpAcceptRate: 0.476` dans
`/metrics`). Continuation multi-tour testée par `conversation_id` : le
second tour MTP répond correctement à partir du premier (`cacheReused:
true`, `mtp: "actif"`), sans rejeu du préfixe.

Bench synthétique d'une couche (`flash-layer-bench`, Release, poids aléatoires,
sans checkpoint) : GDN+MoE 5,5 ms/pas, QSA+MoE 5,8 ms/pas en eager ; 4,5 ms
avec `asyncEval` (GPU 82 % `ioreg`). Debug : 6,5 / 10,3 ms. Le profiler par
couche coûte ~4,7 ms par phase (opt-in `--profile-layers` depuis P0-c).

### P4 — débit de décodage : `residentAsyncInterval` corrige un bug de bookkeeping (2026-09-09)

Même checkpoint/prompt/réglages ; `residentAsyncEval` était un no-op en
production (`residentEvaluationInterval == 1` forçait un `eval` bloquant sur
chaque couche quel que soit ce flag). `residentAsyncInterval` (nouveau,
défaut 8) répare l'asynchronie sans toucher `residentEvaluationInterval`
(piège 11 inchangé). Détail, sweep N=1-12 et méthodologie Metal System
Trace : `docs/knowledge/log.md` « 2026-09-09 — P4 : débit de décodage ».

| Variante | tokens | avant P4 | après P4 (N=8) | Δ | IDs = référence |
|---|---:|---:|---:|---:|---|
| Greedy | 32 | 0,166 s/token (6,0 tok/s) | **0,1405 s/token (7,1 tok/s)** | **−15,4 %** | oui |
| MTP bloc 2 | 32 | 0,135 s/token (7,4 tok/s) | **0,1328 s/token (7,5 tok/s)** | −1,6 % | oui |

GPU actif pendant le décodage (Metal System Trace, union des intervalles,
pas une moyenne d'échantillons `ioreg`) : 14,2 % avant P4.1 → 82-97 % après
(`ioreg`, cross-check). Fin de token (`lm_head`+`sampler.sample`+`.item()`) :
2,2 % du décodage, sous le seuil de 10 % — rien à changer (P4.2). Aucune
fusion de noyau custom retenue (P4.3, décision documentée) : P2-fusion avait
déjà montré un gain nul sur des leviers structurellement comparables, et
P4.1 seul a refermé l'essentiel de l'écart GPU visé par ces fusions.

### P5.4 — préfill 2 543 tokens : la couche PLE domine, routage shard O(n) au lieu de O(shards×n) (2026-09-10)

Deux tours `flash-chat-probe --resident-layers --resident-async --profile-layers
--trace` (tour 1 court force la résidence, tour 2 mesure un préfill pur, sans
coût de chargement) ; checkpoint 3-bit, prompt répété, greedy. Détail complet
et hypothèse initiale réfutée : `docs/knowledge/log.md` 2026-09-10 « P5.4 ».

| Variante | tokens | préfill total (tour 2, pur) | couche PLE (n-gram) | % du préfill | IDs |
|---|---:|---:|---:|---:|---|
| Avant | 2 543 | 9,914 s | 2,746 s | 29,2 % | `[78768]` |
| Après lecture par plages (retenu, sans gain net) | 2 543 | — | 2,746 → 3,101 s | ~31 % | `[78768]` |
| Après routage shard O(n) (retenu) | 2 543 | **9,753 s** (−1,6 %) | **2,568 s** (−6,5 %) | **27,8 %** | `[78768]` |

La couche PLE reste au-dessus du seuil de 15 % de la consigne après les deux
correctifs ; le coût résiduel probable (dispatch GPU par shard n-gram touché,
jusqu'à 128) n'a pas été attaqué — chantier distinct, hors du correctif ciblé
demandé.

### P5.6 — serveur multi-conversations : validation dialogue A/B 20 min (2026-09-10)

`qwen38 serve --trace` (sans Metal System Trace), checkpoint 3-bit,
`Scripts/agent-dialogue.py --max-minutes 20` (T=0,7, top_p 0,8, 120 tokens
max, aucun champ de pénalité — défaut serveur `presence 1.5`). Détail complet,
y compris le correctif P5.2 nécessaire en cours de route et l'écart non
atteint sur la similarité : `docs/knowledge/log.md` 2026-09-10 « P5.6 ».

| | Avant (référence, 38 tours) | Après (P5.1-P5.5, 76 tours) |
|---|---:|---:|
| TTFT tour 1 | 71,6 s | 0,93 s |
| TTFT au dernier tour | 23,89 s (tour 38, 2 691 tok d'historique) | 2,65 s (tour 76) |
| TTFT max sur le dialogue | 23,89 s | **2,81 s** (0 tour > 3 s) |
| tok/s décodage | 4,9-6,0 | 4,9-6,4 |
| Pic process | non mesuré | **57,58 Go** |
| `cacheRestored` | métrique inexistante | 74/76 tours |
| Similarité Jaccard médiane (i vs i-2) | 0,116 | 0,145 (non amélioré — voir log) |

TTFT et mémoire : critère largement dépassé (76 tours plats sous 3 s contre
l'objectif de 38 sous 3 s ; 57,6 Go contre le plafond de 75 Go). Similarité :
critère « aucune boucle » non atteint — `presencePenalty` est remis à zéro à
chaque tour (contrat P5.3), donc structurellement incapable d'empêcher une
dérive thématique inter-tours ; une boucle quasi verbatim apparaît aux tours
68-73 malgré la pénalité par défaut. Un correctif nécessaire en cours de
route (P5.2 : « nouvel état après un rejeu », partie manquée du premier
passage — sans lui, un agent dont le tout premier message contient déjà un
tour assistant restait bloqué en rejeu complet toute la conversation).

### P6.6 — validation finale : dialogue A/B 20 min **sans** `conversation_id` (2026-09-10)

`qwen38 serve --trace` (checkpoint 3-bit, P6.1-P6.5 tous appliqués),
`Scripts/agent-dialogue.py --max-minutes 20 --no-conversation-id` (le cas
Open WebUI / SDK openai ordinaire : aucun agent n'envoie `conversation_id`,
chaque tour renvoie tout l'historique). Comparé à P5.6 (76 tours, avec
`conversation_id`) et à la validation dédiée de P6.3 (61 tours, avec
`conversation_id`, `docs/knowledge/log.md` 2026-09-10 « P6.3 »). Détail
complet : `docs/knowledge/log.md` 2026-09-10 « P6.6 ».

| | P5.6 (avec id, 76 tours) | P6.3 (avec id, 61 tours) | **P6.6 (sans id, 58 tours)** |
|---|---:|---:|---:|
| TTFT tour 1 | 0,93 s | — | 1,13 s |
| TTFT médian | 2,16 s | 2,29 s | **2,34 s** |
| TTFT max | 2,81 s (0 tour > 3 s) | 3,09 s | 4,43 s (4 tours consécutifs, 18-21, transitoire) |
| tours > 3 s | 0/76 | 0/61 | 4/58 |
| tok/s décodage | 4,9-6,4 | — | 2,5-6,6 |
| Pic process | 57,58 Go | non mesuré | **58,32 Go** |
| Restaurations / total | 74/76 (`cacheRestored`, id explicite) | — | 56/58 (`prefixHits`, préfixe implicite P6.1) |
| Misses | `cacheMisses` 2 | — | `prefixMisses` 2 |
| Jaccard médian (i vs i-2, même agent) | 0,145 | 0,114 | **0,143** |
| Jaccard max | boucle 1,0 aux tours 68-73 | 0,184 | **0,261** |
| Paires Jaccard > 0,6 | ≥ 1 (boucle) | 0 | **0** |

**Tous les critères P6.6 atteints** : TTFT médian 2,34 s < 3 s sans
`conversation_id` (contre un rejeu complet à chaque tour avant P6.1,
TTFT croissant comme en référence P5.6) ; aucune boucle (0 paire > 0,6,
Jaccard max 0,261, cohérent avec la validation dédiée de P6.3) ; pic
process 58,32 Go < 75 Go. Le cache de préfixe implicite (P6.1) restaure
56/58 tours (les 2 misses sont les deux tours de démarrage à froid, un par
agent) — la trace confirme que le serveur reconnaît la continuation sans
`conversation_id` exactement comme avec. Écart mineur documenté : 4 tours
consécutifs (18-21) au-dessus de 3 s (jusqu'à 4,43 s) alors qu'aucun miss
n'est enregistré à ce moment — un pic transitoire de pression mémoire
système (compresseur actif, ratio 35× en fin de run) est le candidat le
plus probable, non isolé plus précisément.

Fichiers : `results/p66/dialogue.jsonl`, `results/p66/server.log`
(trace `.trace.json` non versionnée, gitignore).
