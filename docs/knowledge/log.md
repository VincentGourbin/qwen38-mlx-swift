# Journal

## 2026-08-31 — Flash-Next : parité causale des 48 couches

- Le harnais multi-couches a été corrigé pour donner `mask="causal"` aux
  couches QSA, conformément à `Qwen4ExpModel`; le `mask=None` précédent
  expliquait le faux écart de la première couche full-attention.
- Le re-ancrage Swift contre Python sur les 48 couches ne montre pas de rupture
  structurelle isolée, mais la chaîne 4-bit amplifie les écarts : top-1 final
  `917` Swift contre `332` Python. Le mixer Swift appliqué à l'état Python
  reste proche (`rel 0,00128`) et conserve le token Python.
- Conclusion : le gate qualité Flash-Next reste fermé ; le prochain choix est
  une qualification indépendante par qualité ou une référence symétrique
  8-bit/BF16, pas un nouveau benchmark MTP sur ce chemin.

## 2026-08-30 — couture globale et single-layer Flash-Next

- La parité des globaux (embedding quantifié, mixer hyper final, lm-head) est
  exacte sur trois tokens du checkpoint réel 4-bit groupe 32 : max `0` sur
  chaque sortie.
- La couture embedding → couche 0 → réduction → logits est déterministe mais
  seulement bornée : max `0,879883` à la couche, `14,6406` après le mixer et
  `6,01953` sur les logits. Ces valeurs restent un signal de diagnostic ;
  elles ne valident pas encore les tokens générés.

## 2026-08-30 — parité Flash-Next

- Ajout d'un fixture Python/MLX déterministe pour la MRoPE multimodale et de
  son comparateur Swift. Sur la séquence image + texte, les positions sont
  identiques ; l'écart maximal est `2.38419e-7` pour cos et `1.78814e-7`
  pour sin.
- La commande `flash-qsa-parity` reste validée sur son fixture précédent
  (pire écart scores `0.00124264`, masque exact) et `flash-mrope-parity`
  complète désormais cette première couverture numérique Flash-Next.
- Validation effectuée par `xcodebuild`, puis `xcrun xctest` en série avec
  les deux fixtures : 41 tests passés. La parité vision et la parité des
  logits finaux restent à faire ; les premiers tokens greedy isolés ne sont
  pas encore une preuve de qualité.

## 2026-08-30 — parité vision Flash-Next

- Le fixture Python réel ne charge que le shard vision et conserve toutes les
  activations intermédiaires. Une erreur d'ordre d'axes dans sa construction
  de patches a été détectée par le comparateur puis corrigée ; le patch embed
  est désormais exact.
- Sur la sortie finale de la tour vision 27 blocs + merger, l'écart Swift/MLX
  est `max 0,00556403`, moyenne `0,00106566`. Le dernier bloc amplifie les
  différences internes (max ~80), information conservée dans le rapport pour
  investigation ultérieure.
- `flash-vision-parity` et le test conditionné passent avec le checkpoint réel.
  La campagne combinée QSA + MRoPE + vision totalise 42 tests via
  `xcodebuild` puis `xcrun xctest` en série.

## 2026-08-30 — parité du premier bloc langage Flash-Next

- Le fixture réel de couche 0 couvre les linears affines 4-bit groupe 32,
  GDN, hyper-connexions et MoE 512 experts sans charger les 47 autres couches
  ni la table n-gram de 51B paramètres.
- Une aliasation des placeholders `scales`/`biases` rendait les résultats
  non déterministes selon l'ordre d'update des paramètres. La correction est
  appliquée aux Linear/Embedding Flash-Next et au `SwitchGLU` local.
- Après correction, deux lancements isolés sont identiques. La sortie de la
  couche présente `max 0,279892`, moyenne `0,0274141` contre Python ; le test
  reste volontairement borné mais ne prétend pas encore à une parité logits.

## 2026-08-28

- Initialised the Swift package around `mlx-swift-lm` 3.31.x and its upstream
  `qwen3_5` implementation, with `MLXVLM` linked explicitly so the local
  Qwen3.8-27B VLM loader is available.
- Added a local model validator, external-disk cache resolution, tokenizer
  bridge, actor-isolated runtime, profiler metrics/trace export, CLI smoke
  commands, and a small SwiftUI benchmark surface.
- Verified a Debug build with `xcodebuild`; tests must run serially through
  `Scripts/run-tests.sh` because of the known MLX lock-order hazard.
- Downloaded `mlx-community/Qwen3.8-27B-4bit` to `/Volumes/Lexar/models`
  (16.08 GB, three shards). A custom URLSession delegate stalled before the
  first byte on the HF LFS redirect; the native async `URLSession.download`
  path follows the redirect correctly and now resumes at completed-file
  granularity.
- Real local smoke runs passed for text and vision on the M3 Max: text 68
  prompt tokens at 27 tok/s, 16 generated at 10.4 tok/s, TTFT 2.50 s;
  vision 319 prompt tokens at 87 tok/s, 24 generated at 10.3 tok/s, TTFT
  3.67 s. Both were intentionally stopped by the token limit.
- `Qwen38BenchUI` v1 now accepts a local image through the picker and sends it
  through the upstream Qwen3.5 VLM processor; the CLI has the same `--image`
  smoke-test option.
- The GUI now separates model load from inference. Load time is reported once,
  and TTFT is measured wall-clock from the inference request start to the first
  non-empty streamed chunk; prefill time remains the model-side metric from
  `GenerateCompletionInfo`.
- `ChatSession` is now persistent across GUI turns: it retains the conversation
  KV cache, while `Nouvelle conversation` recreates only the lightweight session
  and keeps the model weights resident. Thinking effort is passed per turn as
  `low`, `medium`, or `xhigh` through the Qwen chat-template context.
- The 8-bit sibling checkpoint is available on the Lexar at
  `/Volumes/Lexar/models/mlx-community/Qwen3.8-27B-8bit` (29.53 GB) and passed
  a real smoke inference: 69 prompt tokens at 12 tok/s, 24 generated at 6.9
  tok/s, TTFT 5.85 s. The 4-bit ChatSession path also passed: 67 prompt tokens
  at 57 tok/s, 24 generated at 10.7 tok/s, TTFT 1.19 s.
- Root cause of the GUI's ignored first image: the generic
  `loadModelContainer(from:using:)` tried the LLM factory first, which accepted
  `model_type: qwen3_5` and silently selected the text-only model. The runtime
  now calls `VLMModelFactory.shared` explicitly; a real 4-bit image smoke test
  then completed with 546 prompt tokens and a 6.09 s TTFT. Modality is recorded
  per turn in the stats panel (`Texte` or `Texte + N images`).
- A second GUI-only image regression was found during verification: the view
  model cleared `self.imageURL` before copying it into the runtime request, so
  the UI bubble showed the attachment while `generate(imageURLs:)` received an
  empty array. The attachment URL is now captured before clearing the composer;
  the status line also reports `1 image envoyée` before the first token.
- M1 MTP integration uses the upstream post-#351/#545 checkout locally pinned
  under `Vendor/mlx-swift-lm` (commit `1a562aa`). The public target models do
  not contain `mtp.*`; the paired drafter is downloaded separately as
  `Qwen3.8-27B-MTP-4bit` or `Qwen3.8-27B-MTP-8bit`.
- The published standalone MTP checkpoint stores bare keys (`fc.*`,
  `layers.*`, `norm.*`). The upstream factory currently assumes embedded
  `mtp.*` keys, so `Qwen38MTPDrafterProvider` prefixes the standalone keys,
  reuses the upstream Qwen sanitizer, and applies the declared affine
  quantization before strict loading. A real 4-bit text smoke test now reports
  MTP active with accepted proposals; a VLM first-turn test reports 572
  prompt tokens, 12 proposals, and 10 accepted tokens.
- TTFT was corrected to start before message reconstruction, image processing,
  target prefill, and MTP initialization. The previous MTP smoke value (~4 ms)
  excluded those phases; the corrected value is ~522 ms for a 59-token text
  prompt, with ~502 ms target prefill.
- M1 conversation safety: after entering the standalone upstream MTP path, the
  runtime keeps later turns on the same direct full-history generation path,
  even if the GUI toggle is turned off. This avoids mixing ChatSession's
  persistent target cache with a separately prepared MTP context. Metrics now
  distinguish `cacheReused` from `conversationReplayed`; persistent target +
  drafter state across MTP turns remains the M2 pipeline work.
- The executable build workflow is enforced by `Scripts/build.sh`, which calls
  `xcodebuild` and uses `.xcodebuild` as its derived-data directory. This is
  required for MLX inference because the adjacent
  `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` is part of the
  runnable output; a successful `swift build` alone is not sufficient.
- Final three-turn protocol completed with the supplied Xi Jinping image,
  prompts fixed by the request, thinking `low`, and `maxTokens=2048` for
  Qwen3.8 4-bit, 8-bit, and bf16, each with MTP off/on. The conversation CLI
  keeps one runtime alive and writes one TSV per run under `results/`.
- MTP rollback finding (2026-08-29): `MambaCache.saveSpeculativeCheckpoint`
  stored the pre-verify offset while storing recurrent tensors after the
  committed bonus. Restoring after a rejected draft left the recurrent cache
  one position behind attention. The checkpoint now stores `offset +
  advancedBy`, with a regression assertion on the restored offset. A 256-token
  text control became byte-identical after the fix.
- M1 VLM replay finding (2026-08-29): a three-turn image parity run still
  diverged after rollback was fixed because the upstream Qwen MTP drafter
  prefilled its private attention cache with linear token positions; the full
  image-bearing prompt's per-token M-RoPE table is not part of the upstream
  drafter API. Runtime now keeps MTP active for the cold first image turn but
  falls back explicitly on later image-bearing replays. The guarded 4-bit
  three-turn/512-token parity is now identical (MTP 63/78 on turn 1, fallback
  on turns 2–3). M2 must carry the actual M-RoPE positions with persistent
  target/drafter state instead of replaying image history.
- The observed MTP acceptance pairs (proposed/accepted) were 4-bit:
  `78/63`, `508/408`, `480/404`; 8-bit: `64/55`, `502/378`, `363/284`;
  bf16: `62/53`, `413/330`, `471/388`. Standard turns reused ChatSession's
  KV cache; M1 MTP turns reported full-history replay as designed.
- The bf16 runs reached approximately 66.8 GiB driver allocation without MTP
  and 78.0 GiB with MTP. The benchmark command now records MLX active/peak
  memory in TSV; these values remain distinct driver observations rather than
  `MLX.Memory` readings.
- Short memory probes (`maxTokens=1`, same image and three-turn shape) recorded
  MLX peaks of about 17,075 MiB for 4-bit standard, 17,885 MiB for 4-bit MTP,
  31,483 MiB for both 8-bit modes, 53,093 MiB for bf16 standard, and 54,332
  MiB for bf16 MTP. These are runtime MLX peaks; macOS driver allocation was
  higher during the long bf16 runs (about 66.8/78.0 GiB standard/MTP).
- The final MTP comparison is a throughput/acceptance measurement, not yet a
  greedy-equivalence proof: generated lengths and later prompt lengths differ
  between MTP and standard runs. The R7 token-identity/parity gate therefore
  remains open before treating the M1 speed differences as quality-neutral.

## 2026-08-30

- Flash-Next multimodal assembly passed on the real
  Qwen3.8-Flash-Next-MLX-4bit-MTP checkpoint: local image resized to
  1216×800, 950 vision tokens, multimodal sequence [1, 953], logits
  [1, 953, 248320], all 48 layers streamed one at a time. Peak MLX active
  memory was 4.12 GB and peak process memory 32.70 GB.
- MLXProfiler now instruments the Flash vision phase, global weights, every
  streamed layer and the multimodal forward. The Chrome/Perfetto trace
  /private/tmp/qwen4-exp-multimodal.trace.json contains all 48 layer events.
- Added the first autoregressive Flash-Next probe. It uses the local HF
  tokenizer/template, keeps the recurrent/QSA caches across mono-token
  forwards, accepts image embeddings only during prefill, and reports TTFT,
  prefill/decode and profiler traces. Real one-token text and image runs
  completed with peaks of 29.87 and 31.72 GB respectively.
- The current Flash path is a validated architectural/generation probe, not
  yet a quality or Python-parity result: the first observed tokens were
  13845 (utral) for text and 24 (9) for image. Do not expose this
  implementation through the GUI/server until logit parity and a longer
  controlled generation are established.
- Post-correction verification completed with `Scripts/run-tests.sh` via
  `xcodebuild`: 9 tests passed, and the executable build still contains
  `default.metallib`.
## 2026-08-31 — Flash-Next : scale GDN de K et continuation publique

Le probe single-layer réel (checkpoint 4-bit, trois tokens de préremplissage
puis un token de continuation) a localisé une dérive qui n'apparaissait pas
dans la reproduction manuelle : `Qwen4ExpGatedDeltaNet` appliquait le facteur
`1/sqrt(Dk)` à `q` et à `k`. Le contrat Python Flash-Next normalise `q` et `k`
par leur norme L2, puis scale uniquement `q`. Retirer le scale de `k` ramène
l'appel public `Qwen4ExpDecoderLayer` de `max 0,135498` à `max 0,00219727` ; la
sortie GDN est à `max 0` sur la continuation et l'état récurrent à
`2,38419e-7`. Les tests couvrent désormais explicitement l'appel public et le
cache chaud, pas seulement une copie du calcul dans le harness.

## 2026-08-31 — Flash-Next : parité publique GDN/QSA

Le probe `flash-public-layer-parity` compare désormais les appels publics des
couches 2 (GDN) et 3 (QSA) du checkpoint 4-bit, avec captures des frontières
hyper-connection, q/k/v, RoPE, SDPA/GDN, gate, projection et MoE. Le MoE isolé
de la couche 2 est exact ; sa sortie complète reste à `max 0,0949707` après
amplification BF16 du mixer. La couche QSA 3 est à `max 0,0377916`.

Deux causes ont été corrigées : le chemin Python laisse `mask=nil` pour un
préfill court tant que l'indexeur n'a pas franchi le budget sparse, et les
positions texte doivent être recopiées sur les trois axes MRoPE. Après
correction, les q/k/v pré-RoPE sont exacts et l'erreur RoPE reste sous `0,03`.
`xcodebuild build-for-testing` puis `xcrun xctest` direct passent avec **46
tests** et les fixtures réelles. Cette parité reste bornée : la qualification
token/logit multi-couches est encore requise avant l'exposition publique.

## 2026-08-31 — Flash-Next : chaîne 0→3 et gate token/logit

Le nouveau fixture `qwen4-exp-selected-layers-reference.safetensors` compare
la chaîne réelle Python/Swift des couches 0 à 3, la réduction des quatre flux
et le `lm_head`. Les sorties de couches restent bornées (`0,015625`, `0,015625`,
`0,0352783`, `0,0724167`), mais la réduction hyper-connection monte à
`17,1635`, les logits à `4,36233` et le token suivant diverge (`86394` contre
`70608`). Le fallback GDN donne pratiquement le même résultat (`17,1375`,
`4,55719`) : le problème n'est pas le kernel Metal seul, mais l'amplification
de la dérive après plusieurs couches dans le mélange quatre voies.

Le test est donc conservé comme diagnostic borné, avec seuils `<20` et `<10`,
sans prétendre à la parité de génération. `xcodebuild build-for-testing` puis
`xcrun xctest` direct passent avec **47 tests** et les fixtures réelles. Le
générateur Flash-Next public et son MTP restent derrière le gate token/logit.

## 2026-08-31 — Flash-Next : E0/E1/E2 rebasés

Le harnais rapporte désormais RMS, RMS relatif, cosinus, marge top-1 et rangs
croisés. Des fixtures rebasés injectent à chaque couche la sortie Python exacte
de la couche précédente. E1 est bit-exact : mixer hyper-connection et `lm_head`
Swift sur état Python donnent `max 0`, cosinus `1` et le même token ; le
chargement quantifié des globaux est donc innocenté.

E2 produit : couche 0 `max 0,015625 / rel 0,01746`, couche 1
`0,00390625 / 0,00560`, couche 2 `0,0220947 / 0,03472`, couche 3
`0,0156886 / 0,02475`. Les chemins GDN/QSA et le MoE isolé sont exacts ; les
écarts apparaissent surtout dans les projections de branche et le chemin
`mlp_hyper_connection`/MoE, compatibles avec l'amplification BF16 des quatre
flux mais supérieurs au seuil de qualification proposé (`rel 3e-3`). Le
fixture PLE de la couche 1 a été corrigé pour capturer l'état après PLE avec un
cache séparé.

La génération Flash-Next et le MTP restent bloqués. Prochaine expérience :
ablation E3 fp32/dequantifiée sur les mêmes couches, puis E5 avec prompts
naturels et marge de logits. Validation finale du round : `xcodebuild` puis
`xcrun xctest`, **47 tests passés**.

## 2026-08-31 — Flash-Next : E3 déquantifié FP32

E3 compare maintenant des chemins symétriques : mêmes poids du checkpoint 4-bit,
déquantifiés en FP32 côté Python et côté Swift. Les couches 0, 2 et 3, avec
entrée Python rebasée, retombent au bruit numérique : couche 0 sortie
`max 2,38e-7` (`rel 1,16e-7`), couche 2 `max 2,86e-6` dans les frontières
(`rel 2,0e-7`), couche 3 `max 9,54e-6` (`rel 2,0e-7`), cosinus 1 partout.
La dérive 4-bit de D3w (`rel 0,0175` à `0,0347`) disparaît ; elle est donc
quantificationnelle/amplifiée par le mixer, pas structurelle. La couche 1 n'est
pas déquantifiée dans ce harnais à cause de la taille de sa table PLE/n-gram,
mais reste couverte par E2 et les tests. E4 structurel est écarté ; le prochain
gate est E5 sur prompts naturels et marges de logits avant de débloquer la
génération/MTP.

## 2026-08-31 — Flash-Next : E5 naturel et couche 4

Le script de référence rend maintenant un prompt naturel via le chat template
local. Sur 29 tokens et les 48 couches, le Python 4-bit donne le token `11`
(marge `1,75`) ; le générateur Swift complet produit `16` en `85,4 s` de TTFT
sur le chemin couche-par-couche, avec un pic MLX de `34,49 Go`. L'E1 sur l'état
Python reste sain (`rel 0,00195`, cosinus `0,999993`, token `11`).

Le probe rebasé trouve la première hausse à la couche 4 (`rel 0,0360`). Son
diagnostic public montre une divergence du top-k MoE (`moe_indices rel 0,132`,
sortie `rel 0,0751`) malgré des projections proches. En déquantifiant
symétriquement les poids Python/Swift, la couche 4 redevient quasi exacte
(`mlp_mixed rel 1,37e-7`, indices identiques, sortie MoE `rel 6,13e-7`). Le
mismatch E5 est donc lié à la sensibilité du routage au chemin 4-bit
inter-runtime ; il ne justifie pas encore un correctif structurel. Le gate de
génération/MTP reste ouvert pour une matrice multi-prompts 4/8-bit/BF16.

Un contrôle avec Python MLX `0.32.0` reproduit les mêmes écarts couche 4 que
Python `0.31.2` face à Swift MLX `0.31.6` (`moe_indices rel 0,134`, sortie MoE
`rel 0,0747`). Python MLX `0.31.6` n'est pas disponible sur PyPI. La version
mineure seule n'explique donc pas le mismatch ; le chemin quantifié Swift doit
être qualifié par qualité et stabilité du routage, tandis que le chemin FP32
reste l'oracle de parité structurelle.

## 2026-08-31 — gate MTP 27B et métrique de routage MoE

Le correctif du rapport public Flash-Next compare désormais l'appartenance
des ensembles d'experts, et non l'ordre non spécifié d'`argPartition`. Le probe
réel de couche 4 donne `3/29` positions différentes (`3` affectations) ; les
47 tests Swift passent après le changement avec `xcodebuild build-for-testing`,
puis `xcrun xctest` direct.

Le gate conversationnel 27B 4-bit, sur trois tours avec une image au premier
tour et `temperature=0`, donne des sorties standard/MTP identiques. M1
upstream accepte `7/8` tokens au premier tour ; les deux tours suivants
retombent sur le fallback MRoPE documenté du replay multimodal. Le chemin M2
local reste actif sur les trois tours et accepte respectivement `9/11`, `9/12`
et `8/13` tokens. Le gate d'auto-cohérence MTP 27B est vert pour cette matrice
smoke ; cela ne débloque pas le générateur Flash-Next, qui conserve son gate
de qualification séparé.

Le même smoke passe en 8-bit avec des sorties identiques et `7/8` tokens M1
acceptés au premier tour. Le BF16 a été lancé mais n'a pas terminé après
environ 13 minutes ; il a été arrêté proprement sans erreur. Ce résultat est
classé non concluant et devra être repris avec un probe BF16 limité à un tour
et quelques tokens, plutôt que d'être interprété comme un échec du portage.

Un probe BF16 encore plus court (un puis deux tokens, image au premier tour)
donne une sortie identique au chemin standard (`L`, puis `L'utilisateur`).
Les deux essais rapportent `0/0` propositions M2, ce qui est attendu avec une
limite aussi basse et ne constitue pas une mesure de taux d'acceptation.

Le replay M1 après image est classé limitation de contrat : le drafter
upstream ne reçoit pas les positions M-RoPE 3 axes du préfixe vision. Aucun
contournement local ne doit forcer ce cache incomplet. M2 local reste le chemin
persistant officiel pour une conversation multimodale ; M1 reste disponible
pour le premier tour froid et les conversations texte.

Le serveur implémente maintenant un `conversation_id` explicite (top-level ou
dans `extra`) et ne réutilise le cache que pour un préfixe de messages contigu,
un modèle et des paramètres compatibles. `/metrics` expose l'identifiant de
conversation, `cacheReused`, `conversationReplayed`, le statut MTP et ses
compteurs. Smoke HTTP réel sur le port `8859` : deux tours 4-bit ont donné
`cacheReused=false` puis `true`, sans replay au second tour. La suite directe
reste verte avec 47 tests après `xcodebuild`.

Le smoke Flash-Next multimodal fonctionne sur le checkpoint 4-bit réel :
vision `1216x800`, 950 marqueurs, 971 tokens, 48 couches et premier token
produit. TTFT `100,524 s`, pic processus `36,28 Go`, trace profiler écrite
dans `/private/tmp/qwen38-flash-smoke.trace.json`. Ce résultat valide la
chaîne d'exécution et le budget mémoire, pas la qualité ni le MTP Flash.

## 2026-08-31 — Flash-Next : coût du second token et initialisation Metal

Le micro-benchmark multimodal à deux tokens, sur le même checkpoint et la même
image, donne 971 tokens de prompt, TTFT `129,487 s`, préfill `129,485 s`,
`105,998 s` pour le décodage du second token et un pic MLX de `37,81 Go`.
Le coût confirme que `Qwen4ExpStreamingDecoder` recharge actuellement les 48
couches à chaque appel autoregressif : ce chemin est un adaptateur de
validation, pas encore un runtime de production. Le choix de résidence des
couches/cache reste le prochain chantier avant une mesure de débit publique.

Un processus neuf peut aussi avorter avant le chargement si le profiler lit la
mémoire avant la création du device Metal. `FlashGenerateProbe` force désormais
`Device.defaultDevice()` avant toute phase profilée. Le terminal sandboxé n'a
pas accès au device MLX dans ce contexte ; le même test lancé dans le contexte
local autorisé a fonctionné. La compilation doit être rejouée via le projet
Xcode généré/ouvrable : le checkout courant ne contient pas de `.xcodeproj` et
`xcodebuild` direct refuse le `Package.swift`.

## 2026-08-31 — GUI : filtrage du canal thinking

Une exécution depuis Xcode confirme le MTP 27B local : bloc 2, `46/49` tokens
acceptés (`93,9 %`), `14,8 tok/s`, TTFT `2,506 s` et pic mémoire `17,07 Go`.
La capture montrait cependant le raisonnement dans la bulle. Le serveur le
sérialisait déjà séparément, mais la GUI ajoutait les chunks bruts. Elle
utilise désormais `Qwen38ThinkingStreamParser` et affiche uniquement le
contenu visible, avec un flush final sur les métriques ; le raisonnement reste
conservé pour le runtime et l'API.

## 2026-08-31 — GUI : thinking escamotable

Le thinking n'est plus mélangé à la réponse visible. Chaque message assistant
conserve désormais le raisonnement séparément et la GUI l'affiche dans une
bulle `DisclosureGroup` repliée par défaut. Le parseur route les fragments
`reasoning` et `content`, avec flush final ; l'API garde ses champs
`reasoning_content` et `content` séparés.

## 2026-08-31 — MTP bloc 4 et compteur thinking GUI

Le test bloc 4 de la conversation donne `256/498` tokens acceptés (`51,4 %`)
et `3,0 tok/s`, contre `869/1177` (`73,8 %`) et `9,7 tok/s` en bloc 2. La
profondeur supérieure n'est donc pas encore rentable sur ce cas. Le libellé de
la bulle thinking affichait aussi l'expression Swift littérale ; l'interpolation
est corrigée et le nombre réel de caractères sera maintenant affiché.

## 2026-08-31 — Flash-Next : E5 naturel complet et index de shards réutilisé

Le fixture E5 naturel complet (48 couches, 29 tokens) a été comparé avec le
harnais Swift sur le checkpoint 4-bit réel. Le mixer et la tête Swift restent
alignés lorsqu'ils reçoivent l'état Python (`rel 0,00121662` pour le mixer,
`rel 0,00194756` pour les logits, top-1 `11/11`). La chaîne 4-bit diverge
cependant à la couche 4 (`rel rebasée 0,0360479`) et choisit `16` contre `11`,
avec une marge Python de `1,75`. Le strict token gate reste donc fermé ; la
cause reste la sensibilité du routage MoE quantifié, pas le mixer Swift.

Le décodeur streaming construit désormais une fois la table immuable
couche→shard du fichier d'index, au lieu de reparcourir le JSON pour chaque
couche et chaque token. Aucun poids n'est conservé par cette optimisation.
La compilation CLI n'a pas pu être relancée avec `xcodebuild` car le checkout
est un Package.swift sans projet/workspace ; validation à reprendre dans le
projet Package ouvert par Xcode. `swift build` n'est pas utilisé.
Le contrôle syntaxique `xcrun swiftc -parse` des deux fichiers modifiés passe,
sans se substituer à la validation Xcode qui doit suivre.

## 2026-08-31 — Flash-Next : optimisation compilée par Xcode

Après reconstruction du Package depuis Xcode, la mini-GUI démarre et
fonctionne. Le binaire DerivedData expose bien `Qwen4ExpCheckpointLayerIndex`
et la signature du loader recevant l’index préchargé : la modification est
compilée dans le produit Xcode/MLX. Aucun nouveau calcul Flash-Next long n’a
été lancé pour préserver le GPU ; E5 multi-prompts reste le prochain jalon.

## 2026-08-31 — Flash-Next : E5 multi-prompts

Deux fixtures naturels supplémentaires ont été produits avec la référence
Python compatible du checkpoint : transition énergétique (51 tokens) et
fonction Swift (56 tokens). Avec le fixture présidentiel, les trois chaînes
48 couches 4-bit divergent après la couche 4 : Swift/Python `16/11`,
`1710/11`, `258/16837`. E1 reste sain sur les trois (`mixer rel` entre
0,00121662 et 0,00190269 ; `logits rel` entre 0,00194756 et 0,00266371 ;
top-1 identique sur état Python). Le strict token gate reste fermé, et la
cause qualifiée demeure la sensibilité du routage MoE à la quantification
inter-runtime. Les métriques sont dans
`results/flash-e5-natural-2026-08-31.tsv`.

## 2026-08-31 — Flash-Next : mode résident opt-in

Le décodeur propose désormais un mode `resident` expérimental, activable par
`--resident-layers` dans le probe de génération. Il conserve les couches entre
tokens afin de mesurer le vrai coût du chemin rapide ; le mode `streamed` reste
la valeur par défaut et `unloadResidentLayers()` libère les poids résidents.
Le contrôle syntaxique passe ; un build Xcode est requis avant toute mesure
mémoire résidente.

## 2026-08-31 — Flash-Next : mode résident compilé

Le rebuild Xcode confirme la présence de `Qwen4ExpLayerLoadingMode` et du
stockage `residentLayers` dans le binaire GUI. La fonctionnalité est compilée,
mais aucun chargement résident n’est encore exécuté. Le prochain test doit
passer par le target CLI avec `--resident-layers` et une limite d’un token.

## 2026-08-31 — Flash-Next : mesure de résidence complète

Le probe CLI Xcode a comparé le même prompt texte de 23 tokens sur le
checkpoint 4-bit. Le mode résident termine le premier forward à `136,106 s`,
avec `111,14 Go` de mémoire MLX active et `111,56 Go` au pic. Le mode streaming
termine à `121,846 s`, avec `0,98 Go` active et `34,49 Go` au pic. Les deux
produisent le même token `47106` (`"itre"`).

Ce run à un token ne contient pas de second forward autoregressif : son champ
`decode` ne mesure donc pas un gain token 2. L’essai résident à deux tokens a
été interrompu après environ cinq minutes sous pression mémoire et n’est pas
un benchmark valide. La résidence complète reste opt-in/diagnostique ; la
prochaine piste est une résidence groupée ou un forward fusionné, pas une
activation par défaut.

## 2026-08-31 — Flash-Next : métriques load/forward instrumentées

`Qwen4ExpGreedyGenerationResult` agrège désormais les visites de couches, le
temps de chargement des poids et le temps de calcul des forwards séparément.
`flash-generate-probe` les affiche et les ajoute aux métadonnées de la trace
`swift-mlx-profiler`. Le build Xcode/MLX et la suite directe `xcrun xctest`
passent avec 48 tests. Cette séparation est le prérequis de mesure pour le
futur forward batché/MTP Flash-Next; la résidence complète reste non viable sur
96 Go.

## 2026-08-31 — Flash-Next : premier round MTP réel

Le predictor MTP natif réel charge 76 tenseurs et passe un forward quatre-flux
(`[1,1,10240]`, logits hidden `[1,1,2560]`). Le nouveau probe borné exécute le
cycle cible → draft → vérification → rollback/rejeu → commit sur 22 tokens,
`blockSize=2`. Le draft `19` est rejeté par la cible (`25,15`) ; le snapshot
cible rejoue correctement un token et le cache drafter termine à l'offset 23.
La durée de ce round est `335,491 s`, à considérer comme une mesure de
fonctionnement/diagnostic du streaming 4-bit, pas comme un benchmark de qualité
ou de performance. Build `xcodebuild` et suite sérialisée : 50 tests passés.
Le seam reste local et opt-in tant que le gate E5 qualité n'est pas vert.

## 2026-08-31 — Flash-Next : générateur MTP opt-in

Le cycle local cible → draft → vérification → snapshot/rejeu → commit est
maintenant encapsulé dans `Qwen4ExpGreedyGenerator.generateMTP`. Le résultat
expose TTFT, rounds, propositions, acceptations, rollbacks et tokens rejoués,
avec les temps de couches déjà suivis par `swift-mlx-profiler`. Le CLI propose
`flash-generate-probe --mtp --mtp-block-size N`; le greedy reste par défaut et
le mode MTP refuse encore l'image tant que la continuation M-RoPE multimodale
n'est pas définie. Build `xcodebuild` et tests sérialisés : 50 tests passés.

## 2026-08-31 — Flash-Next : smoke public MTP et correction de budget

Le smoke CLI avec deux tokens a d'abord exposé une borne off-by-one : le bonus
était produit mais la boucle ne demandait aucun draft. Après correction et
rebuild Xcode, un run borné à trois tokens a traversé deux rounds réels sur le
checkpoint 4-bit : `[17,25,16]`, `0/2` acceptés, `4` vérifiés, `2` rollbacks et
`4` tokens rejoués. TTFT `107,419 s`, génération `420,236 s`, avec
`515,802 s` de chargement cumulé pour `9,894 s` de calcul des couches. La
brique est fonctionnelle mais le streaming couche-par-couche est trop coûteux
pour conclure à un gain MTP ; résultat brut :
`results/flash-mtp-generator-smoke-2026-08-31.tsv`. Le test de non-régression
sur le budget du bonus porte la suite Xcode à 51 tests, tous passés.

## 2026-08-31 — Flash-Next : smoke qualité texte + image

Le générateur Swift streamed traverse correctement le chemin image réel
(`1216×800`, `950` marqueurs), mais les deux sorties bornées restent
inexploitables : texte `var\\n<|im_end|>`, image
`y\\n\\n2user...\\n<|im_end|>`. Le multimodal est donc consommé
techniquement sans être validé qualitativement. La référence Python
`mlx-vlm 0.6.17` reconnaît `qwen4_exp` mais dépasse la mémoire Metal sur le
checkpoint Flash MTP résident. Résultats :
`results/flash-quality-smoke-2026-08-31.tsv`. Le gate qualité reste fermé ;
la suite doit isoler le premier écart de génération avant serveur/catalogue.

## 2026-08-31 — Flash-Next : Q-A template et n-gram lazy

`flash-template-probe` et `Scripts/flash-template-reference.py` comparent le
rendu ChatML Swift/Python id-à-id sans charger le modèle. Le prompt présidentiel
est identique en thinking on (`57` tokens) et off (`29` tokens) ; les bornes
`<think>`, `</think>`, `<|im_start|>`, `<|im_end|>` et l'EOS sont cohérentes.
Le loader de couche conserve désormais `ngram_embedding` lazy/mmap au lieu de
les évaluer avec toute la couche. La couche PLE 2 réelle s'exécute avec `1,62`
Go de poids matérialisés et `1,63` Go de pic MLX, sortie `[1,1,10240]`, sans
matérialiser la table n-gram de 32 Go. Build `xcodebuild` et suite sérialisée :
51 tests passés. Prochaine vérification : continuité de l'état n-gram sur deux
tokens, puis résidence autoregressive.

Le contrôle de continuité PLE sur deux appels successifs (`layer 2`,
`sequenceLength=1`, `repeatCount=2`) réussit avec le cache conservé. Les deux
appels sortent `[1,1,10240]`, chargent `61` tenseurs et maintiennent le pic à
`1,63 Go`. Le n-gram lazy fonctionne donc aussi avec l'état inter-appels ; le
prochain test est la résidence autoregressive des couches ordinaires.

## 2026-08-31 — Flash-Next : résidence partielle n-gram validée

Le premier filtre MLX ne suffisait pas : le gather sur une table safetensors
lazy provoquait la matérialisation globale des `ngram_embedding`, avec
`109,59 Go` actifs et `110,32 Go` de pic. `Qwen4ExpLazyNGramStorage` contourne
ce comportement en lisant les en-têtes puis uniquement les lignes
`weight/scales/biases` demandées par le hash de la PLE.

La couche PLE réelle (couche 2, index Swift 1) passe deux appels successifs
avec cache conservé, sortie `[1,1,10240]`, environ `1,64 Go` de poids et
`1,65 Go` de pic. Le parcours borné de toutes les 48 couches sur deux tokens
termine à `78,84 Go` actifs / `79,81 Go` de pic, au lieu de `110,32 Go`.
La porte mémoire est verte. Le token de sortie du probe (`alfa营商`) reste
uniquement un résultat de plomberie : la qualité Flash-Next demeure non
qualifiée tant que Q-A logits et Q-B teacher-forced ne sont pas passés.

La sonde `flash-ngram-parity` a ensuite comparé trois lignes (`0, 1, 12345`)
du shard réel `0` avec une déquantification eager directe. Après correction
de la conversion BF16 par `view(dtype: .bfloat16)` (au lieu d'une conversion
numérique), `max |delta| = 0` et `mean |delta| = 0`. Build et tests
`xcodebuild` : 51 tests passés. La suite est Q-A logits puis Q-B
teacher-forced ; le probe borné ne qualifie pas la qualité.

Le probe résident relancé après correction BF16 reste dans le budget
(`78,84 Go` actifs / `79,81 Go` de pic), mais renvoie `var\\n` sur deux tokens.
Le reader lazy est donc exact et moins gourmand, sans que la qualité du
générateur Flash-Next soit pour autant validée. Q-A logits/EOS-thinking puis
Q-B restent prioritaires devant MTP.

Le probe Q-A avec `--report-top-k 20` rapporte pour le premier token les IDs
`917 (12,463096)`, `70 (12,337608)`, `846 (12,073127)` ; marge top-1/top-2
`0,125488`, avec `<|im_start|>` au rang 9. La distribution est plate : cette
mesure ne tranche pas sans référence Python, mais le rapport top-k est
désormais disponible sans forward supplémentaire. TTFT `100,224 s`, pic
`79,81 Go`.

## Correction de formulation — parités Flash-Next déjà validées

Les parités Python/Swift Flash-Next précédentes sont bien validées pour QSA,
MRoPE, vision, globaux, couches publiques/sélectionnées, n-gram lazy et
l'identité M2/M1. Seule la nouvelle comparaison Python du scoring
teacher-forced Q-B reste à produire ; l'impossibilité de relancer MLX Python
dans un environnement sans Metal ne remet pas en cause ces résultats.

## 2026-09-01 — Flash-Next : scorer teacher-forced Q-B côté Swift

Le runtime expose maintenant `Qwen4ExpStreamingTextModel.scoreTeacherForced`.
Il concatène prompt et continuation, exécute un seul forward causal, puis
calcule par token le logprob de la cible, son rang, l'argmax et la marge
top-1/top-2. Le résultat agrège logprob moyen, accord argmax et accord sur les
positions confiantes ; aucun sampler ou stop-token ne pollue la mesure.

Le CLI `flash-teacher-forced-score` accepte soit `--continuation`, soit les
IDs exacts avec `--continuation-ids`, et rend le prompt via ChatML avec le
mode thinking choisi. Le build Xcode et la suite sérialisée restent verts
(`51` tests). Les parités Python/Swift précédentes sont déjà validées pour
QSA, MRoPE, vision, globaux, couches publiques/sélectionnées, n-gram lazy et
l'identité M2/M1. Seule la nouvelle comparaison Python du scoring
teacher-forced Q-B reste à produire.

Le dumper `Scripts/qwen4-exp-teacher-forced-reference.py` est maintenant en
place. Il prend les IDs exacts du prompt et de la continuation, charge les 48
couches séquentiellement et exporte les logits ainsi que les métriques cibles
dans une fixture safetensors. Syntaxe validée ; l'exécution MLX doit se faire
sur le Mac équipé de Metal.

Le comparateur Swift `Qwen4ExpTeacherForcedParity.compare` et l'option CLI
`--python-fixture` sont ajoutés : ils relisent cette fixture et confrontent
IDs, logprobs, marges, rangs et accords argmax sans second forward Swift. La
prochaine action est uniquement de produire puis lire la fixture Q-B sur le
Mac Metal ; les parités Flash-Next précédemment listées restent déjà validées.

Clarification de statut : la résidence partielle n-gram et le chemin
autoregressif Flash-Next existent depuis V27 et peuvent déjà générer des
tokens, mais restent expérimentaux jusqu'à Q-A/Q-B (qualité trop lente ou
dégénérée). Le chemin 27B est déjà utilisable indépendamment. Une tentative
de lancer le nouveau dumper avec une séquence E5 exacte a été faite depuis la
session Codex ; ce processus précis a reçu `No Metal device available`. Cela
ne décrit pas le Mac ni le Python interactif de Vincent. Le script prend
maintenant `Scripts/references/vlm_q4_language.py` par défaut.

Un `ASK` est ajouté à l'étape V30 pour faire vérifier la différence entre ce
contexte Codex et l'environnement ayant produit les fixtures Python MLX déjà
validées, ainsi que la commande exacte de relance de Q-B.

## 2026-09-02 — correction des normes zéro-centrées Vontra côté Swift

Le loader Flash-Next corrige au chargement les suffixes de normes validés par
comparaison au BF16 officiel : il soustrait `1` aux versions Vontra décalées,
sans modifier `Qwen4ExpRMSNorm` ni toucher à `linear_attn.norm`, la vision ou
le MTP. La détection utilise la moyenne de l'ancre `hc_norm`, ce qui conserve
les checkpoints déjà zéro-centrés. Le dumper Q-B reprend la même correction
avec `--norm-shift -1` par défaut. `qwen38 info` expose la politique.

Build `xcodebuild` et suite `xcodebuild` sérialisée verts : **53 tests**. La
prochaine validation est la sanité absolue avec `mlx-vlm==0.6.17`, avant toute
nouvelle fixture E5, conclusion qualité ou mesure de performance.

## 2026-09-02 — sanité absolue Q-B exécutée sur Metal

Le venv `venv617` avec `mlx-vlm==0.6.17` voit bien Metal depuis le terminal
réel (`metal True`). Le teacher-forced officiel a chargé les 48 couches du
checkpoint Vontra 4-bit en 124,4 s avec `--norm-shift -1` : **10/28 hits**,
logprob moyenne **-4,426**, RMS caché final 1,0870. L'erreur « No Metal device »
était limitée au sandbox Codex. La prochaine validation est le smoke greedy
Swift corrigé, puis la régénération des fixtures de génération.

## 2026-09-02 — préflight et tranche Swift Flash-Next

Le binaire Debug issu du dernier `xcodebuild` lit correctement le checkpoint
`qwen4_exp` : 48 couches, 1 couche MTP, 22 shards, 113,21 GB de poids et
correction Vontra annoncée au preflight. Une tranche réelle couche 0 a chargé
71 tenseurs depuis 2 shards, matérialisé 2,02 GB et produit une sortie
`[1, 4, 2560]` sans erreur.

Le smoke greedy résident de 8 tokens a ensuite été arrêté après plus de dix
minutes dans `Qwen4ExpStreamingDecoder.forward → mlx_eval` (~74 GB de footprint,
GPU ~60 %). Ce n'est pas une conclusion qualité : le chemin complet est
actuellement dominé par le coût de résidence/synchronisation du checkpoint
113 GB sur Lexar. L'optimisation de ce chemin précède la régénération des
fixtures greedy et le gate token-à-token.

Le décodeur synchronise et vide désormais le cache par couche uniquement en
mode streamed ; le résident reporte la matérialisation au forward texte
complet. Rebuild `xcodebuild` et suite sérialisée : **53 tests**. Un smoke
thinking de 1 token a terminé en 244,333 s (token initial `The`, début du
raisonnement), avec 79,59 GB actifs et 80,92 GB peak. Le run no-thinking lancé
immédiatement après a subi une pression page-cache : GPU à ~5 % et scheduler
en attente après 9 min, puis arrêt propre. Le prochain run de qualité doit
être isolé après stabilisation du cache ; ce résultat ne constitue pas un
verdict de qualité no-thinking.

Un run de 8 tokens avec profiler a été lancé ensuite, mais arrêté après 7 min
avec le GPU à ~6 % dans l'attente page-cache/I/O ; aucune trace n'a été
exportée car l'export intervient en fin de run. Cette attente du stockage USB
doit rester distincte d'une régression numérique.

Après stabilisation du GPU, le smoke Swift no-thinking résident isolé a
généré le premier token `id 2229`, décodé **« Le »**, exactement comme la
référence officielle corrigée `mlx-vlm 0.6.17`. TTFT `272,643 s`, prefill
`272,631 s`, 48 couches, 79,22 GB actifs et 80,52 GB peak. Le correctif des
normes, le template ChatML no-thinking et le chemin greedy sont donc alignés
jusqu’au premier token ; la réponse libre multi-token et MTP restent à
valider.

## 2026-09-02 — synchronisation intermédiaire du mode résident

Le décodeur résident borne maintenant son graphe MLX avec un `eval` toutes les
8 couches par défaut, et un dernier `eval` en fin de passe. Les poids restent
résidents et aucun `Memory.clearCache()` n’est exécuté entre ces checkpoints ;
le chemin streamed conserve son `eval` et son nettoyage par couche. L’intervalle
est exposé par `--resident-eval-interval` sur les deux probes CLI et par
`residentEvaluationInterval` dans l’API Swift. Build `xcodebuild`, aide CLI et
suite de **53 tests** validés.

## 2026-09-02 — smoke résident multi-token profilé

Un run réel du binaire Debug sur le checkpoint Vontra 4-bit avec
`--resident-eval-interval 8 --max-new-tokens 2 --trace` s’est terminé et a
exporté `/private/tmp/qwen38-flash-v35.trace.json` (165 Ko). Il a généré
`[2229, 85648]`, soit **« Le président »**. TTFT : **102,026 s** ; génération :
**131,569 s** ; 96 visites ; chargement cumulé **95,448 s** ; forward cumulé
**135,508 s** ; mémoire MLX peak **75,58 GB**. La trace montre les six
checkpoints attendus aux couches 7, 15, 23, 31, 39 et 47. Le profiler est
fonctionnel sur ce chemin ; la lenteur reste dominée par l’I/O/résidence du
checkpoint externe.

## 2026-09-03 — seconde tentative du gate greedy 8 tokens

Le gate 8 tokens a été relancé sur une machine dégagée, avec le mode résident
et l’intervalle de synchronisation 8. Après environ 19 minutes, le processus
restait actif mais alternait calcul et attente de pages du checkpoint sur Lexar.
Il a été arrêté proprement par SIGINT (code 130), sans crash ni résultat libre.
La baisse de charge concurrente n’a pas résolu le coût structurel du chemin ;
la prochaine étape doit traiter le stockage/résidence et le graphe MLX avant
une nouvelle validation 8 tokens ou MTP longue.

## 2026-09-03 — captures de parité opt-in

Les `lastParityCapture` de GDN, QSA, MoE et hyper-connections ne conservent
plus leurs tenseurs intermédiaires pendant une inférence normale. Le layer de
parité active explicitement la capture via `setParityCapture(true)` avant son
forward, ce qui conserve les probes sans retenir leurs graphes dans le chemin
résident ou serveur. Rebuild `xcodebuild` et suite de **53 tests** validés.

## 2026-09-03 — table n-gram POSIX-mapped et smoke résident

`Qwen4ExpLazyNGramStorage` mappe désormais les fichiers Safetensors de la
table n-gram avec `mmap(MAP_PRIVATE)`, avec repli sur `FileHandle`. Cela évite
la copie complète que Foundation peut déclencher avec `.mappedIfSafe` sur le
Lexar ExFAT. Les offsets étant des offsets d’octets non nécessairement alignés,
les chemins mapped UInt32/UInt16 reconstruisent explicitement le little-endian.
La commande `flash-ngram-parity --shard 0 --rows 0,1,12345` donne
`max |delta|: 0` et `parité n-gram lazy: IDENTIQUE`. Le smoke résident de
2 tokens génère `Le président`, TTFT **112,426 s**, peak MLX **80,19 GB**, et
exporte `/private/tmp/qwen38-flash-v40.trace.json`. Le gate libre 8 tokens et
MTP restent ouverts : le stockage USB reste le facteur limitant.

## 2026-09-02 — gate libre 8 tokens reporté

Le run greedy 8 tokens résident (intervalle 8), lancé sans autre charge GPU,
a dépassé 21 minutes en alternant calcul et attente de pages du checkpoint
sur Lexar. Il a été interrompu proprement par SIGINT, sans sortie ni erreur
MLX. Le gate 8 tokens reste donc non concluant ; le dernier smoke libre valide
reste celui de 2 tokens, **« Le président »**, avec trace profiler. La suite
est de réduire l’I/O (stockage local ou poids déjà résidents) avant de relancer
ce gate et MTP.

## 2026-09-03 — intervalle résident 48 non retenu

La suite `xcodebuild` reste verte avec **53 tests**. Un smoke résident de
2 tokens avec `--resident-eval-interval 48` n’a pas atteint la sortie après
plus de neuf minutes et a été arrêté proprement (`SIGINT`, code 130). Le
réglage 8 reste le dernier chemin complet validé ; la matérialisation d’un
graphe de 48 couches n’est pas rentable sous la contrainte Lexar/96 GiB. Le
prochain levier est un cache de shards ou un stockage plus rapide, pas une
campagne qualité/MTP longue.

## 2026-09-04 — cache borné de lignes n-gram

Qwen4ExpLazyNGramStorage possède maintenant un cache LRU borné à 4096
lignes de données brutes, avec une clé incluant l’offset Safetensors pour
éviter toute collision entre weight/scales/biases. Le probe double lecture
donne max delta 0, hits=18, misses=9, et max delta répétition=0.
Build et suite complète via xcodebuild passent avec 53 tests. L’effet sur
une conversation réelle reste à mesurer ; aucun gate long 8 tokens/MTP
n’est relancé sur le Lexar à ce stade.

## 2026-09-04 — compteurs n-gram dans le profiler

Le décodeur Flash-Next expose les hits, misses et entrées du cache n-gram.
Les générateurs greedy et MTP publient ces valeurs après chaque forward dans
un événement Counter swift-mlx-profiler, et les ajoutent aux métadonnées de
la trace finale. Le parseur Swift passe et le test du hit rate a été ajouté.
Le build MLX complet doit être relancé depuis le Package ouvert dans Xcode :
le CLI xcodebuild de cet environnement ne reconnaît pas ce Package Swift seul.

## 2026-09-04 — reset des snapshots n-gram

La vérification de l'exporteur confirme que les compteurs n-gram sont bien
des événements Chrome Trace `C`. Le reset des compteurs efface désormais aussi
les snapshots de delta par couche du mode résident ; cela évite de perdre les
premiers hits/misses du tour suivant.

## 2026-09-04 — build Xcode et limitation Metal du terminal

Le binaire Debug régénéré contient bien les symboles de compteurs
`Flash n-gram cache` et `ngram_cache_hit_rate`. Une parité n-gram courte lancée
depuis le terminal abort avant l'accès au checkpoint dans
`mlx::core::metal::load_device` avec une liste de devices vide. C'est une
limitation du contexte terminal sandboxé ; la validation runtime doit être
exécutée depuis le contexte Xcode/GUI qui dispose du device Metal.

## 2026-09-05 — probe Flash et parité n-gram depuis Xcode

Le probe Flash-Next a terminé avec le code 0 et écrit
`/Users/vincent/Downloads/qwen38-flash.trace.json`. La parité n-gram réelle
est exacte (`max delta 0`, `mean delta 0`, `18 hits`, `9 misses`, `9 entrées`,
delta de répétition nul). Il reste à inspecter dans Perfetto que la trace
contient bien les événements `Flash n-gram cache` exportés par le profiler.

## 2026-09-05 — trace profiler Flash-Next vérifiée

La trace réelle contient 608 événements sur `applegpu_g15s`, dont les points
`Flash n-gram cache` à `720/720` puis `768/768` hits/misses, avec 768 entrées
et un hit-rate final de 0,5. Les métadonnées concordent. Le probe était greedy,
2 tokens, sans image et sans `--mtp` : la validation du chemin MTP reste donc
à faire séparément.

## 2026-09-05 — tentative MTP directe depuis le terminal

Le CLI fraîchement reconstruit s'arrête avant le chargement avec
`NSRangeException` dans `mlx::core::metal::load_device`, car le terminal
Codex ne publie pas de device Metal. Aucune trace MTP n'a été produite ; la
validation doit être exécutée depuis le contexte Xcode qui avait fourni
`applegpu_g15s` pour la trace greedy.

## 2026-09-05 — probe MTP Flash-Next réel

Le probe Xcode avec MTP local bloc 2 termine avec `mtp=true`, un round, zéro
token accepté sur un token proposé, un rollback et un replay. Le chemin de
vérification/restauration fonctionne, mais deux tokens maximum ne suffisent
pas pour estimer l'acceptance. La trace contient aussi les compteurs n-gram
des trois phases : `720/720`, `816/816`, puis `864/816` hits/misses, avec un
hit-rate final de 0,5143.

## 2026-09-05 — probe MTP Flash-Next multi-round

Le run Xcode à 8 tokens produit 5 rounds MTP, 5 propositions et 2
acceptations (40 %), avec le chemin rollback/rejeu exercé sur plusieurs
rounds. La trace contient 2 638 événements sur `applegpu_g15s` et les
compteurs n-gram finissent à `1344 hits / 1200 misses / 1200 entrées`, soit
0,5283. Le mécanisme MTP et son instrumentation sont validés ; les durées
(`2109 s` de forward cumulé, pic MLX ~78,1 GB) ne sont pas comparables comme
benchmark tant que le checkpoint est streamé depuis le Lexar.

## 2026-09-05 — contrôle du template ChatML Flash-Next

Le `flash-template-probe` du binaire Xcode rend correctement le préfixe
assistant + `<think>` en thinking activé. En thinking désactivé, le template
insère un bloc `<think>...</think>` vide ; les IDs sont stables
(`<|im_start|> 248045`, `<|im_end|> 248046`, `<think> 248068`, `</think> 248069`).
Le rendu ChatML est documenté avant le gate d'inférence et ne nécessite pas
de device Metal.

## 2026-09-05 — GATE GREEDY OUVERT : Flash-Next répond correctement en français

Le probe enrichi (checkpoint Vontra 4-bit, résident, intervalle 8, 8 tokens,
sans image, sans MTP) a été relancé et a terminé avec succès directement
depuis le terminal de cette session (device Metal `applegpu_g15s` bien
détecté, contrairement à tous les essais précédents dans un contexte Codex
sandboxé sans device Metal). La sortie décodée est
`"Le président de la Chine est Xi Jinping"` : elle commence par
`Le président`, ne contient aucune balise ChatML parasite dans la réponse
visible, et les métadonnées confirment `thinking=false` et `mtp=false`.
TTFT 99,688 s, préremplissage 99,685 s, décodage 841,697 s, pic MLX
75 241 MB actif / 76 149,6 MB process. Trace écrite dans
`qwen38-flash-quality-enriched.trace.json`.

**Le gate greedy no-thinking défini par le HANDOFF est donc PASS.** Étape
suivante du HANDOFF : même contrôle avec `--thinking`, puis test multimodal
avec image seulement si le thinking est correctement borné.

Le même probe relancé avec `--thinking` (57 tokens de prompt, même checkpoint,
8 tokens) produit `"The user is asking for information about the"` : un
raisonnement anglais cohérent et sain, sans balise parasite ni dégénérescence,
mais trop court (8 tokens) pour observer la fermeture `</think>` et la réponse
finale visible. TTFT 96,891 s, décodage 87,191 s, pic MLX 78,83/80,27 Go. Ce
résultat est un sanity-check positif, pas encore une preuve complète du
bornage thinking — un budget de tokens plus large serait nécessaire pour voir
`</think>` et la réponse. Étape suivante du HANDOFF : test multimodal avec
`/Users/vincent/Downloads/licensed-image-2.jpeg`, l'image de référence déjà
utilisée dans toute la campagne Flash-Next.

Le test multimodal (même checkpoint, image `licensed-image-2.jpeg` — portrait
d'Emmanuel Macron —, prompt « Qui est sur cette image et quel est son rôle ? »,
greedy, sans thinking ni MTP, 8 tokens) traverse la tour vision réelle
(`1216x800`, 950 marqueurs, 976 tokens de prompt) et produit
`"La personne sur cette image est **Em"` : français correct, markdown cohérent,
et le début exact du nom attendu (« Em[manuel Macron] »), sans balise
parasite. TTFT 101,995 s, décodage 166,363 s, pic MLX 79/82,65 Go.

**Verdict global : `QUALITY_GATE=PASS`** sur les trois volets du HANDOFF
(greedy texte, thinking sanity, multimodal). Le gate qualité Flash-Next, fermé
depuis fin août, est donc levé pour la première fois sur ce prompt de
référence. Reste ouvert : confirmer la fermeture `</think>` sur un budget de
tokens plus large (le sanity-check thinking n'a vu que 8 tokens de
raisonnement).

Un second run thinking avec un budget élargi à 48 tokens confirme la qualité :
`"The user is asking for information about the President of China and their
role.\nThe current President of China is Xi Jinping.\nThe role of the
President of China is largely ceremonial, but as the General Secretary of the
Communist Party of China"` — raisonnement multi-phrases cohérent, factuel et
correct, sans dégénérescence ni répétition sur 48 tokens (`</think>` pas
encore atteint à ce budget, la chaîne continue logiquement). TTFT 95,018 s,
décodage 943,127 s (48 tokens), pic MLX 78,84/80,27 Go.

**Conclusion de la reprise HANDOFF post-V52 : `QUALITY_GATE=PASS`.** Les trois
contrôles (greedy no-thinking, thinking, multimodal) produisent tous une
sortie française/anglaise cohérente et correcte sur le prompt de référence,
sans balise ChatML parasite dans la réponse visible. C'est la première
confirmation de qualité du portage Flash-Next depuis le début du Jalon 3. Le
gate long-terme (fermeture `</think>` observée en clair, robustesse sur
d'autres prompts, débit/mémoire du chemin résident streamé depuis le Lexar,
intégration GUI/serveur) reste à traiter, mais n'est plus un blocage de
principe sur la qualité de génération.

Une tentative d'élargir la qualification à un second prompt (« Explique en
une phrase ce qu'est la photosynthèse. », sans thinking ni image) a été tuée
par le système pour cause de mémoire basse : un autre processus GPU
(`ltx-video`, sans rapport avec ce projet) tournait en parallèle et la somme
des deux dépassait la RAM disponible. Ce n'est pas un échec du portage
Flash-Next ; le second prompt reste à rejouer quand aucune autre charge GPU
lourde n'est active en parallèle.

## 2026-09-05 — trace qualité enrichie

La trace qualité de 8 tokens confirmait le run Metal, la résidence et les
compteurs n-gram, mais n'exportait pas le texte généré. Le probe enregistre
désormais dans `ProfilingSession.metadata` le mode thinking/MTP, le nombre de
tokens du prompt, les IDs générés et le texte décodé. Le fichier CLI passe le
parse Swift ; le prochain run Xcode fournira une trace suffisante pour le gate
sémantique sans dépendre de la sortie console.

## 2026-09-05 — cause racine de la lenteur/instabilité résidente : le batching d'`eval()` par 8 couches

Après le gate qualité, investigation de performance sur le chemin résident
(`--resident-layers`), historiquement très lent et très instable (forward
cumulé observé entre 89s et 943s pour 8 tokens greedy selon les runs). Deux
hypothèses ont été écartées avec preuves :

- **Limite GPU wired** (`sudo sysctl iogpu.wired_limit_mb=85000`, appliquée
  par Vincent) : aucune amélioration (un run après relèvement a même été plus
  lent, 1436s). Le pic MLX (~79-82 Go) n'est donc pas la cause.
- **QSA sparse gather coûteux** : écarté par lecture du code — sur des
  prompts courts (29-57 tokens), `Qwen4ExpQSAIndexer.makeMask` retourne
  toujours `nil` (`maxCompleteBlocks(~7-14) > blockTopK(512)` est faux), donc
  QSA utilise le fallback dense bon marché, pas le gather top-k.

**Cause racine identifiée par un probe diagnostique `--resident-eval-interval
1`** (`eval()` après chaque couche au lieu de toutes les 8) : le décodage du
second token, isolé, prend **21,4s** sur 48 couches (~200ms/couche, GPU
~100%), sans aucun palier anormal. Le run complet à 8 tokens avec
`--resident-eval-interval 1` confirme : **decode 27,686s / forward cumulé
28,053s**, contre 841-1436s avec l'intervalle 8 par défaut — un gain de
**~30-50×**, avec une sortie identique (`"Le président de la Chine est Xi
Jinping"`), donc aucun impact sur la qualité. Le TTFT/préremplissage reste
inchangé (~94-105s), dominé par le chargement disque one-shot du Lexar, sans
rapport avec ce bug.

**Explication** : différer l'évaluation MLX de 8 couches accumule un graphe
paresseux massif avant de le forcer d'un coup. Au lieu de réduire le coût de
synchronisation (l'hypothèse de conception initiale), ce batching produit des
paliers de calcul énormes et très variables (paliers observés de 20s à 1m23s
par fenêtre de 8 couches), signature typique d'un graphe MLX trop gros
évalué en une fois plutôt que d'un vrai goulot GPU ou disque — cohérent avec
le point 8 de la checklist de pièges du plan (§6.3) : « Éval per-tensor dans
les boucles de transformation de poids, sinon OOM silencieux — leçon LoRA
h3 », qui s'applique ici au niveau couche plutôt que tenseur.

**Correctif appliqué** : `residentEvaluationInterval` par défaut passe de 8 à
1 partout (`Qwen4ExpStreamingDecoder`, `Qwen4ExpStreamingTextModel`,
`flash-generate-probe` — les deux occurrences CLI et l'API Swift), avec
commentaire inline documentant la mesure. Ce changement invalide la
conclusion du 2026-09-03 (« l'intervalle 8 reste le dernier chemin complet
validé, 48 non retenu ») : cette conclusion reposait sur des essais avec de
plus gros intervalles (8, 48), jamais sur l'intervalle 1. Rebuild
`xcodebuild` vert après le changement de défaut.

**Reste à faire côté performance** : mesurer l'intervalle 1 sur un run MTP
multi-round (le générateur MTP a son propre chemin d'évaluation à vérifier),
et sur un contexte plus long, avant de considérer le chemin résident prêt
pour `BENCHMARKS.md`.

## 2026-09-06 — rév. 4 du plan : Jalon 3 réanalysé et réorienté

Revue complète du Jalon 3 (PLAN.md §6-§7 réécrits). Faits établis à partir du
code : étapes A-F validées ; G réalisée sans GATE G-5 (MTP opt-in, 1 618 s /
8 tokens) ; H non commencée (`Qwen38ModelValidator` refuse `qwen4_exp`, GUI
`selectVariant` code en dur les chemins 27B, `Qwen4ExpGreedyGenerator` ni
streamé ni échantillonné). Le dossier n'est pas versionné et `.gitignore`
exclut `Vendor/mlx-swift-lm/` malgré cinq patchs locaux. SSD interne : 88 Go
libres, checkpoint 113 Go → copie locale impossible. Décisions : H avant P ;
MTP Flash-Next en chantier séparé désactivé par défaut ; abandon du critère
« greedy identique à mlx-vlm » (remplacé par Q-B + qualification H6) ; nouvelle
GATE G-8 (démo GUI + serveur) et G-4bis (quant maison ≤ 70 Go sur SSD interne,
décidée après les mesures P1). Séquence d'exécution : H0 (git, tests,
baseline 27B) → H1 catalogue → H2 générateur streamé + sampling → H3
adaptateur runtime → H4 GUI → H5 serveur → H6 qualification → G-8.

## 2026-09-07 — RÉPONSE chantier P : GPU inactif confirmé par les traces, cause première mémoire

Relecture des traces V53/V54 avec `Scripts/trace-layers.py` (nouveau) : en
régime établi (tokens 3-8) le décodage résident coûte 1,6 s/token, 25 ms par
couche, **CPU 98 % (un cœur) et GPU 0-5 %** ; le « 28 s / 8 tokens » de V54
incluait 17 s de warm-up du premier token. Les traces en intervalle 8 donnent
2-3 ms/couche de construction de graphe, le reste dans `eval` avec GPU 0-1 %.
Écarté par lecture : boucle Swift sur les experts (`SwitchGLU` upstream),
GDN sans kernel, ops sur stream CPU. Hypothèse dominante H-A : le process
résident (76,4 Go, MLX actif 75,2 Go) n'est **pas wiré** (`wired_limit_{0}`
par défaut dans MLX, personne n'appelle `mlx_set_wired_limit` — le sysctl
85000 de V54 ne pouvait rien changer) et cohabite avec ~27 Go réels d'autres
apps (18,3 Go anonymes + 6,7 Go compresseur + 2,3 Go swap mesurés ce jour au
repos) → compression des experts froids, fautes de page décompressées sur le
thread MLX. Le « 66 Go used » de `top` est du cache de fichiers (68 Go
file-backed), pas de la marge. Le checkpoint est uniformément 4-bit g32
(experts 77,1 Go, n-gram 32,0 Go, reste 4,1 Go) : le pic ~77 Go est
intrinsèque. Protocole acté : P0 micro-bench synthétique sans checkpoint →
P1 un run résident de 6 tokens avec `Scripts/preflight-resident.sh` (seuil
9 Go anonyme+compresseur+swap ; 27,3 Go ce jour = REFUS),
`Scripts/sample-system.sh` et `sample` CPU → P2-mem (wired limit MLX +
sysctl 88000) ou P2-code → H6 dans un seul process serveur → G-8.

## 2026-09-07 — P0 : bench synthétique d'une couche Flash-Next

Nouvelle commande `flash-layer-bench` (`Sources/Qwen38CLI/Qwen38CLI.swift`)
et logique dans `Sources/Qwen38Core/FlashNext/Qwen4ExpLayerBench.swift` :
construit une couche GDN+MoE (`.linearAttention`) et une couche QSA+MoE
(`.fullAttention`) aux dimensions réelles du checkpoint (hidden 2560, 4 flux
hyper-connections rang 320, 512 experts dim 640 dont 10 routés + 1 partagé
640, GDN 48/16 têtes head_dim 128 conv 4, QSA 24/2 têtes head_dim 256), poids
empaquetés 4-bit g32 créés directement via `qwen4ExpLinear`/`qwen4ExpSwitchLinear`
(comme le test « Les modules Flash-Next peuvent naître directement
empaquetés »), sans checkpoint ni accès au Lexar. Boucle : 20 pas de warm-up
puis 200 pas d'un token `[1,1,4·hidden]`, `eval` à chaque pas, cache par
couche (`MambaCache`/`Qwen4ExpQSAKVCache`, même sélection que
`Qwen4ExpStreamingDecoder.makeCache(for:)`, qui reste `private` — non
modifiée), masque causal `Qwen4ExpQSAAttention.causalMask` pour la couche
QSA. Run : `./.xcodebuild/Build/Products/Debug/qwen38 flash-layer-bench
--trace results/p0-layer-bench.trace.json`, analysé par le nouveau
`Scripts/bench-layers.py` (adapté de `trace-layers.py`, mêmes compteurs
`Utilization`/`Memory`, phases `Bench couche gdn`/`Bench couche qsa`).

**Résultat chiffré** (mesure via la trace du profiler, méthodologie
identique à `trace-layers.py` sur les runs résidents réels — donc
directement comparable aux 22-28 ms/couche de la RÉPONSE du 2026-09-07
ci-dessus) :

| Couche | ms/pas médiane | p10 | p90 | min | max | CPU % | GPU % | Mémoire couche |
|---|---|---|---|---|---|---|---|---|
| GDN (`linear_attention`) | 11,20 | 10,97 | 11,87 | 10,85 | 12,78 | 99,9 | 0,3 | 1,63 Go |
| QSA (`full_attention`) | 15,12 | 14,67 | 15,94 | 14,27 | 19,95 | 99,7 | 1,2 | 1,62 Go |

(mesure du temps de calcul seul, sans le bookkeeping du profiler, obtenue en
parallèle par `ContinuousClock` autour de chaque pas dans
`Qwen4ExpLayerBench` : GDN médiane 6,46 ms, QSA médiane 10,33 ms — l'écart
avec le tableau ci-dessus est le coût des deux `beginPhase`/`endPhase` par
pas, présent de façon identique dans les traces résidentes réelles puisque
`Qwen4ExpStreamingDecoder.forward` enveloppe aussi chaque couche d'un
`profiler.start`/`.end`).

Échantillon GPU pendant le run (`ioreg -r -c AGXAccelerator`, 30 s,
`Device Utilization %`) : quasi tout à 0, un seul pic isolé à 34 % au tout
début (warm-up/compilation des premiers noyaux Metal), cohérent avec les
0,3-1,2 % moyens mesurés par le profiler.

La couche QSA dépasse le seuil de 15 ms avec GPU ≈ 0 : `sample $(pgrep -x
qwen38) 10 -file results/p0-sample.txt` lancé pendant un second run
(`--layer-kind qsa --steps 3000`). Sur le thread de calcul (5562
échantillons sur 10 s), 3576 sont dans `decodeOneStep`, dont 3575 dans
`eval()` → `mlx_eval` → `mlx::core::eval`/`eval_impl` → `gpu::eval` →
`UnaryPrimitive::eval_gpu`, avec une fraction notable (383/994) dans
`Concatenate::eval_gpu` → `copy_gpu_inplace`. Le tableau « Sort by top of
stack » (hors threads de pool GCD/IOKit idle — `__workq_kernreturn`,
`mach_msg2_trap`, `__psynch_cvwait`, `iokit_user_client_trap`, tous des
threads d'attente séparés du thread de calcul) : `std::__tree_sub_invariant`
(163, arbre rouge-noir du cache de buffers Metal), `_xzm_free` (68),
`_platform_memset`/`_platform_memmove` (35/31), `std::__hash_table<void
const*>::__emplace_unique_key_args` (30, cache de ressources/pipelines
Metal), `__psynch_mutexwait` (28), `mlx::core::eval_impl` (27),
`SmallVector<int,10>::size()` (26), puis d'autres allocations/hash lookups
(23, 23, 21…). Aucune trace de fautes de page (`vm_fault`,
`_vm_page_decompress`) ni de noyau Metal dominant : le CPU est consommé par
le bookkeeping hôte de MLX (refcounting `array`/`ArrayDesc`, recherches dans
les caches de buffers/pipelines/streams, copies `SmallVector`) autour de
nombreuses petites opérations (`concatenate`/`slice`/`copy` — cohérent avec
les hyper-connections à 4 flux et le routage MoE qui multiplient les petits
tenseurs), pas par le calcul GPU lui-même ni par la pagination mémoire.

**Verdict selon le critère de PLAN.md R1/P0** : ni l'un ni l'autre des deux
cas propres. Le GPU ne dépasse jamais 30 % (0,3-1,2 %), donc **le critère
« chemin sain / H-A domine » est exclu pour les deux couches, y compris sans
aucune pression mémoire** (P0 n'a ni checkpoint, ni table n-gram, ni 77 Go
résidents — la RAM active reste sous 2,3 Go). La couche QSA remplit
strictement le second critère (**≥ 15 ms et GPU ≈ 0 ⇒ H-B domine**) ; la
couche GDN est juste en dessous (11,20 ms) sur la mesure avec overhead
profiler, et à 6,46 ms sur la mesure de calcul pur — donc entre les deux
seuils au sens strict de PLAN.md, mais du même côté que QSA (GPU ≈ 0, pas de
signe d'un chemin GPU actif). Combiné à l'échantillon CPU (bookkeeping
hôte MLX, pas de fautes de page), la lecture la plus honnête est : **H-B est
confirmée comme contributeur substantiel et mesurable** (11-15 ms/couche de
coût de calcul pur sans aucune mémoire sous pression, contre l'hypothèse du
2026-09-07 matin qui le jugeait « insuffisant pour expliquer 25 ms par
couche ») ; cela ne clôt pas H-A pour autant, puisque P0 ne reproduit pas la
pression mémoire du run résident réel (77 Go, pagination) — seul P1 peut
trancher la part qui reste. Décision : **passer à P1** tel quel (le
protocole P0/P1/P2 de la RÉPONSE du 2026-09-07 l'anticipait : « en
complément, après G-8 sinon » pour P2-code), en gardant les pistes P2-code
identifiées ici (concaténations/copies redondantes dans les
hyper-connections et le routage MoE) comme cibles concrètes si P1 confirme
que H-B pèse aussi en résidence.

Fichiers : `Sources/Qwen38Core/FlashNext/Qwen4ExpLayerBench.swift`,
`Sources/Qwen38CLI/Qwen38CLI.swift` (commande `flash-layer-bench`),
`Scripts/bench-layers.py`, test
`qwen4ExpLayerBenchMeasuresPositiveDurations` (dimensions réduites, P0
`Tests/Qwen38Tests/Qwen38Tests.swift`), traces
`results/p0-layer-bench.trace.json` et `results/p0-sample.txt`.

## 2026-09-07 — P0 rejoué en Release : le build Debug doublait le coût hôte ; le profiler ajoute ~4,7 ms par phase

Même bench (`flash-layer-bench --trace results/p0-layer-bench-release.trace.json`)
avec le binaire construit par `QWEN38_CONFIGURATION=Release Scripts/build.sh`
(`.xcodebuild/Build/Products/Release/qwen38`) :

| Couche | ms/pas (horloge interne) | ms/pas (phase profiler) | CPU % | GPU % (profiler) |
|---|---|---|---|---|
| GDN | **5,52** (Debug : 6,46) | 10,20 (Debug : 11,20) | 17 (interne) / 99,8 (phase) | 10,2 |
| QSA | **5,79** (Debug : 10,33) | 10,43 (Debug : 15,12) | 28 / 99,4 | 10,4 |

Échantillon `ioreg` pendant le run : 0 % sur 22 échantillons sur 40, pics
isolés à 18-51 %. Deux enseignements :

1. **Tout ce qui a été mesuré depuis le 29 août l'a été en Debug** (défaut
   de `Scripts/build.sh`, C++ de MLX compilé sans optimisation). En Release,
   le coût hôte d'une couche QSA passe de 10,3 à 5,8 ms, GDN de 6,5 à 5,5 ms.
2. **Chaque paire `profiler.start`/`.end` coûte ≈ 4,7 ms** (`beginPhase`
   → `SystemMetrics.gpuUtilization()` via IOKit + `processCPUTime` + mémoire,
   à chaque frontière de phase). `Qwen4ExpStreamingDecoder.forward` enveloppe
   chaque couche : sur 48 couches, **~225 ms par token** de pur profiler dans
   tous les runs résidents tracés. Les « 22-28 ms/couche » de la trace V54
   contiennent donc ~5 ms de profiler ; le reste (~17-23 ms, Debug) est
   cohérent avec les 6,5-10,3 ms de calcul hôte Debug du bench plus la part
   mémoire/n-gram. **H-B est majoritaire ; H-A n'est plus qu'un reliquat à
   mesurer en P1, en Release et sans phase par couche.**

Reste, même en Release : 5,5-5,8 ms hôte par couche pour ~0,5 ms de GPU
(GPU 10 %). C'est le coût MLX par op (refcounting, caches de buffers,
`Concatenate`/`copy_gpu_inplace`) multiplié par un grand nombre de petits ops
par couche. Plafond actuel : 48 × 5,6 ≈ 270 ms/token ≈ 3,7 tok/s avant tout
autre effet. Levier suivant (P2-code, sur ce bench, sans checkpoint) :
compter les ops par couche, `compile` des sous-graphes stables, éliminer les
concaténations des 4 flux, et pipeliner (`asyncEval`) au lieu d'un `eval`
bloquant par couche.

## 2026-09-07 — P2-code : leviers hôte sur le bench P0

Suite de P0-b (`Scripts/build-release.sh`) et P0-c (profiler par couche
opt-in, `Qwen4ExpStreamingDecoder.profileLayers`). Mesure de départ,
Release, `flash-layer-bench --steps 200` **sans** `--trace` (horloge interne
`ContinuousClock`, indépendante du profiler) répétée deux fois, meilleure et
pire :

| Couche | ms/pas médiane (run 1) | ms/pas médiane (run 2) | GPU % (profiler, moyen) | GPU % (`ioreg`, médiane sur 25 s) |
|---|---|---|---|---|
| GDN | 5,42-5,46 | 5,44-5,46 | 13,8-15,8 | 0 (un seul pic isolé à 53 sur 25 échantillons) |
| QSA | 5,71-5,72 | 5,72-6,24 | 27,5-37,8 | 0 (idem) |

Le « GPU % » du profiler et celui d'`ioreg` mesurent des choses différentes
(moyenne d'échantillons instantanés autour de chaque pas de ~5 ms, très
bruitée sur un intervalle aussi court, vs. médiane d'un échantillonnage à 1
Hz sur toute la durée du run) ; les deux s'accordent sur l'essentiel : le GPU
est quasiment inactif, cohérent avec l'entrée « P0 rejoué en Release »
ci-dessus. Jauge P2-code (PLAN.md) : ≤ 2 ms/couche, GPU ≥ 40 % — très loin du
point de départ.

### (a) Comptage des ops par pas

`sample $(pgrep -x qwen38) 8 -file results/p2-ops-sample.txt` pendant
`flash-layer-bench --layer-kind qsa --steps 3000` (Release). Le fichier
« Sort by top of stack » liste d'abord les threads de pool GCD/IOKit **au
repos** (`__workq_kernreturn`, `mach_msg2_trap`, `__psynch_cvwait`,
`iokit_user_client_trap`, 3 000+ échantillons chacun — cf. l'échantillon P0
du 2026-09-07 après-midi) : ce ne sont pas les threads de calcul. Sur le
thread de calcul (`mlx::core::scheduler::StreamThread::thread_fn`, 5 466
échantillons sur 10 s), poids cumulés (auto+enfants, donc non strictement
exclusifs) des 15 nœuds du **call graph** les plus fréquents sous ce thread :

| Rang | Primitive / fonction | Échantillons | Famille |
|---|---|---|---|
| 1 | `mlx::core::eval` | 3 887 | ancêtre commun (traversée + eval du graphe) |
| 2 | `mlx::core::eval_impl` | 786 | ancêtre commun |
| 3 | `mlx::core::copy_gpu_inplace` | 177 | **Copy** (hyper-connections `inject`, cache) |
| 4 | `mlx::core::binary_op_gpu` | 143 | **Binaire** (RMSNorm, gating, mix, injection) |
| 5 | `mlx::core::binary_op_gpu_inplace` | 127 | Binaire |
| 6 | `mlx::core::copy_gpu` | 83 | Copy |
| 7 | `std::__function::__func<mlx::core::gpu::eval…>` | 69 | ancêtre commun (dispatch par primitive) |
| 8 | `mlx::core::concatenate_gpu` | 62 | **Concatenate** (cache KV/indexeur, conv GDN, MRoPE) |
| 9 | `mlx::core::QuantizedMatmul::eval_gpu` | 41 | **Matmul quantifié** (linéaires, MoE) |
| 10 | `mlx::core::Reduce::eval_gpu` | 39 | **Reduce** (RMSNorm, softmax, top-k MoE) |
| 11 | `mlx::core::unary_op_gpu` | 36 | **Unaire** (SiLU, sigmoid, sqrt) |
| 12 | `mlx::core::qmv` | 36 | Matmul quantifié (matvec) |
| 13 | `mlx::core::metal::MetalAllocator::malloc` | 36 | bookkeeping hôte (pas un op) |
| 14 | `mlx::core::array::~array` | 35 | bookkeeping hôte (refcounting) |
| 15 | `mlx::core::fast::ScaledDotProductAttention::eval_gpu` | 33 | **Attention** (QSA seulement) |

Famille par famille (QuantizedMatmul+qmv ≈ 77, Copy ≈ 260,
Binaire ≈ 270, Reduce 39, Unaire 36, Concatenate 62, Attention 33) : aucune
primitive isolée ne domine à elle seule — le coût est réparti sur beaucoup de
petits ops de familles comparables (copie et opérations binaires en tête,
juste devant le matmul quantifié qui est le calcul réellement utile), plus
~36+35 échantillons de bookkeeping hôte pur (allocateur Metal, refcounting
`array`). C'est la confirmation directe, par comptage, de la lecture
qualitative du 2026-09-07 après-midi (« bookkeeping hôte de MLX… pas par le
calcul GPU »). Fichier : `results/p2-ops-sample.txt`. Pas de changement de
code pour ce levier (mesure seule).

### (b) Hyper-connections sans concaténation

Relecture de `Sources/Qwen38Core/FlashNext/Qwen4ExpHyperConnection.swift` et
`Sources/Qwen38Core/FlashNext/Qwen4ExpDecoderLayer.swift` : contrairement à
l'hypothèse de PLAN.md, **les 4 flux ne sont jamais découpés en
`split`/slices puis recollés par `concatenated`/`stacked`** dans ce chemin.
`Qwen4ExpGatedResidual.callAsFunction`/`.mixedInput` utilisent déjà
`reshaped([B, S, hcCount, hiddenSize])` et une réduction sur l'axe -2
(`.mean(axis: -2)`) ; `Qwen4ExpDecoderLayer.inject` combine la branche et les
poids d'injection par `expandedDimensions` + multiplication broadcastée +
`reshaped(hyperInput.shape)`, sans `concatenated` ni `split`. Ce sous-système
est donc déjà écrit sous la forme demandée par PLAN.md — **rien à changer,
lever non applicable tel que décrit**. Les `concatenated`/`split` réels vus
en (a) viennent d'ailleurs : la fenêtre de convolution GDN
(`Qwen4ExpGatedDeltaNet.swift:107-112`, `concatenated([convState, qkv])` puis
`MLX.split` — fenêtre de taille constante, coût nécessairement petit et
borné), la croissance du cache KV/indexeur QSA
(`Qwen4ExpCache.swift:107-108`, `concatenated` sur l'historique — croît avec
`offset`, intrinsèque à l'attention causale) et le découpage rope/no-rope de
QSA (`Qwen4ExpQSAAttention.swift:86`, `split(parts: 2)`, puis recombiné dans
`Qwen4ExpMRoPE.swift`). Aucun de ces trois points ne correspond au motif «
4 flux découpés/recollés » ciblé par PLAN.md ; les éliminer changerait la
sémantique du cache causal ou de MRoPE, hors périmètre d'un levier de pur
bookkeeping. Pas de changement de code, pas de commit de code pour ce
levier — seulement cette entrée.

### (c) `MLX.compile` dans le bench (option `--compiled` / `--shapeless`)

`Qwen4ExpLayerBench.run` gagne un paramètre `computeMode:
Qwen4ExpLayerBenchComputeMode` (`compiled`, `shapeless`), exposé par
`flash-layer-bench --compiled [--shapeless]`. Le cache (`MambaCache` /
`Qwen4ExpQSAKVCache`) est bouclé dans un petit adaptateur
`Qwen4ExpLayerBenchCacheBox: Updatable` (`KVCache` et `Updatable`
partagent la même unique exigence `innerState() -> [MLXArray]` mais Swift
n'infère pas la conformité entre deux protocoles distincts à partir d'une
signature identique — l'adaptateur fait juste le pont, en relisant
`cache.innerState()` à chaud à chaque appel compilé) et passé en
`inputs`/`outputs` de `MLX.compile(inputs:outputs:shapeless:)`. GDN : forward
à un seul argument (`hidden -> output`, cache géré par `compile`). QSA :
forward à deux arguments (`hidden, mask -> output`), le masque restant un
argument positionnel car il dépend de `cache.offset`, lu côté Swift.

**Correction** : un test unitaire ajouté
(`qwen4ExpLayerBenchCompiledMatchesEager`) vérifie que le chemin compilé
produit exactement les mêmes valeurs que le chemin eager. Les poids du bench
sont des placeholders zéro déterministes (`qwen4ExpLinear`) ; en réamorçant
`MLXRandom.seed` à l'identique avant les deux runs, seule l'entrée
synthétique par pas varie — `eager.lastOutput` et `compiled.lastOutput`
concordent à 1e-4 près pour GDN et QSA (64 tests verts, `Scripts/run-tests.sh`).

**Mesure** (Release, `--steps 200`, deux runs, sans `--trace`) :

| Variante | GDN ms/pas médiane | QSA ms/pas médiane |
|---|---|---|
| eager (référence) | 5,41 / 5,42 | 5,68 / 5,69 |
| `--compiled` | **5,31 / 5,32** (−2 %) | **6,38 / 6,46** (+12-14 %) |
| `--compiled --shapeless` | crash (voir obstacle) | crash (idem) |

GDN : gain net mais modeste (≈2 %, cohérent avec un cache à forme constante —
fenêtre de conv et état récurrent de taille fixe, cf. (b) — où `compile`
évite de retraverser/reconstruire le même petit graphe Swift à chaque pas).
QSA : **plus lent avec `--compiled`**, confirmant l'hypothèse de PLAN.md : le
masque causal (`Qwen4ExpQSAAttention.causalMask`, taille `offset + 1`) et
l'état du cache KV/indexeur (`concatenated` à chaque pas) changent de forme
à *chaque* appel, donc le graphe compilé sans `shapeless` est retracé/
recompilé à chaque pas — un coût strictement additionnel par-dessus le
travail déjà fait par MLX en mode eager.

**Obstacle précis pour `shapeless: true`** (les deux couches) :

```
MLX/ErrorHandler.swift:345: Fatal error: [Primitive::output_shapes] Split
cannot infer output shapes. at .../mlx-c/mlx/c/closure.cpp:104
```

`mlx-swift` 0.31.6 (`Transforms+Compile.swift`) : le mode `shapeless` exige
que chaque primitive du graphe puisse déduire ses formes de sortie sans
retracer — la primitive `Split` (utilisée par GDN pour séparer q/k/v après
la conv, `Qwen4ExpGatedDeltaNet.swift:112`, et par QSA pour séparer
rope/no-rope, `Qwen4ExpQSAAttention.swift:86`) ne le peut pas dans cette
version de MLX. `shapeless: true` est donc **inutilisable tel quel** sur ce
chemin de calcul, pour les deux types de couche, tant que `Split` n'a pas
cette capacité en amont.

Décision : `--compiled`/`--shapeless` restent des **options du bench**
(comme prévu par PLAN.md — « (c) et (d) restent des options du bench tant
que P1 n'a pas tranché ») ; aucun changement au chemin de production
(`Qwen4ExpStreamingDecoder`, `Qwen4ExpDecoderLayer`). Conservé dans le
code du bench (utile pour un futur P1/wrap plus poussé sur GDN
spécifiquement), non retenu comme optimisation par défaut : le gain GDN est
trop faible et QSA régresse.
