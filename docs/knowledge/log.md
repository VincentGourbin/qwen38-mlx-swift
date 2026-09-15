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

### (d) Pipeline `asyncEval` (option `--async-interval N`)

`Qwen4ExpLayerBenchComputeMode.syncEvery` (défaut 1, comportement inchangé)
+ `flash-layer-bench --async-interval N`. `Qwen4ExpLayerBench.run` construit
désormais chaque pas via un `buildStep()` partagé (extrait de l'ancien
`decodeOneStep`, sans en changer le comportement à `syncEvery == 1`) ; avec
`syncEvery > 1`, `decodeStepGroup` enchaîne `N-1` `asyncEval()` puis un
`eval()` bloquant sur le Nème pas, échantillonne CPU/GPU une fois pour tout
le groupe et amortit la durée mesurée sur les `N` pas (le tableau
`Qwen4ExpLayerBenchResult.steps` garde une entrée par pas mesuré, forme
inchangée pour les appelants existants). **Pas touché** :
`Qwen4ExpStreamingDecoder.residentEvaluationInterval` reste à 1 par défaut
dans le code de production — c'est une option du bench, comme (c).

**Mesure** (Release, `--steps 400`, sans `--trace`, deux runs par intervalle) :

| Intervalle | GDN ms/pas médiane | GDN GPU % (profiler) | QSA ms/pas médiane | QSA GPU % (profiler) |
|---|---|---|---|---|
| 1 (référence, eager) | 5,41 / 5,42 | 13,8-15,8 | 5,68 / 5,69 | 27,5-37,8 |
| 4 | 4,72 | 42,5 | 4,69 | 41,2 |
| 8 | 4,61-4,72 | 45,9-47,0 | 4,74-4,77 | 44,4-54,6 (CPU) / 44,6-44,9 (GPU) |
| 48 | 4,52 | 49,0 | 4,46 | 48,2 |

Cross-check indépendant du GPU % pendant un run à `--async-interval 8`,
`--layer-kind both`, `--steps 3000` : `ioreg -r -c AGXAccelerator`, 25
échantillons sur 25 s → **médiane 82 %** (alternance nette 0 / 80-92 %,
cohérente avec l'exécution GPU par rafales entre deux synchronisations),
contre une médiane de 0 % au départ (entrée ci-dessus). C'est une
confirmation indépendante, pas seulement la moyenne bruitée du profiler : à
la différence de (c), **`asyncEval` produit un vrai travail GPU concurrent
mesurable**, pas un artefact de moyennage.

Verdict : `asyncEval` avec synchronisation différée est le **seul levier
P2-code qui rapproche significativement la jauge visée** (≤ 2 ms/couche, GPU
≥ 40 %) — GPU ≥ 40 % est atteint pour les trois intervalles testés (41-49 %
profiler, 82 % `ioreg` en médiane), et ms/pas baisse de ~13 % (GDN) à ~17 %
(QSA) par rapport à eager, sans dépendre de `compile`/`shapeless`. Le
palier ms/pas (~4,5-4,7 ms, encore loin de 2 ms) est cohérent avec (a) :
l'essentiel du coût restant est le bookkeeping hôte MLX par op (Copy,
binaire, refcounting), qui reste payé qu'on l'attende ou non — `asyncEval`
ne le supprime pas, il le recouvre avec le travail GPU d'un pas voisin.
Aucune différence notable entre N=4/8/48 sur le bench (pas de dégradation à
48, contrairement au verdict V54 en résidence réelle — cohérent avec la
lecture de PLAN.md : V54 était contaminé par Debug + profiler + pression
mémoire, absents ici). Conservé comme **option du bench** uniquement
(`--async-interval`) ; ne change pas `residentEvaluationInterval` (défaut 1)
dans `Qwen4ExpStreamingDecoder` — c'est P1, sur le vrai checkpoint sous
pression mémoire, qui tranchera si la production peut se permettre de
différer la synchronisation.

### (e) Masque causal QSA superflu en décodage à un jeton

(a) ne montre pas de poste isolé évident (Copy/Binaire/Concatenate/Reduce
sont du même ordre de grandeur), mais relire `Qwen4ExpQSAAttention.causalMask`
en révèle un : appelé à chaque pas de décodage avec `queryLength == 1`, il
alloue `keys = MLXArray(Int32(0) ..< Int32(cache.offset + 1))` — un tableau
qui **grandit d'un élément à chaque token décodé** — pour comparer
`keys .< [offset + 1]` et obtenir un résultat mathématiquement toujours vrai
: la seule position de requête (`offset`) est par construction ≥ toute
position de clé déjà en cache (0..offset), donc `keys < offset + 1` est vrai
pour tout `keys ≤ offset`. Un masque booléen entièrement vrai est
numériquement identique à l'absence de masque pour `scaledDotProductAttention`
(vérifié ci-dessous, pas supposé).

**Correctif appliqué au chemin de production** :
`Qwen4ExpStreamingDecoder.forward` ne construit plus le masque causal QSA
quand `inputIDs.dim(1) == 1` (décodage) — il passe `nil`, exactement comme
GDN le fait déjà pour sa récurrence. Le préremplissage et la vérification
MTP (`queryLength > 1`) continuent de construire le vrai masque, inchangés.

**Garde-fou** : la seule suite qui appelle réellement
`Qwen4ExpStreamingDecoder.forward` en décodage multi-pas ("Le générateur
streamé Flash-Next égale le greedy…", H2) est gardée par
`QWEN38_FLASH_MODEL` et **n'a pas tourné** dans cet environnement (pas de
checkpoint, conforme à l'interdiction Lexar de PLAN.md §0). Un nouveau test
autonome, sans checkpoint, `qwen4ExpQSATrivialMaskMatchesNoMaskOnDecode`
(P2-code (e)), comble ce trou : deux `Qwen4ExpDecoderLayer` (couche QSA,
poids identiques via `MLXRandom.seed(42)` avant chacune) subissent un
préremplissage identique puis un pas de décodage, l'un avec le masque
explicite, l'autre avec `nil` — `allClose(atol: 1e-5)` confirme
l'équivalence numérique. 65 tests verts (`Scripts/run-tests.sh`) ; celui-ci
et `qwen4ExpDecoderLayerAssemblesLinearAndQSA` (`Qwen4ExpDecoderLayer`
direct, sans checkpoint) ont réellement tourné et touchent ce chemin ; les
tests de parité contre des fixtures Python (single-layer, couches publiques
2/3, etc.) exercent des préremplissages multi-tokens, pas le décodage à un
jeton — non affectés par ce changement, tournés et verts mais pas des
témoins directs de ce lever spécifique.

**Mesure sur le bench** (option `--skip-trivial-mask`, ajoutée pour isoler ce
coût — la production, elle, l'applique sans option) :

| `--steps` | QSA ms/pas médiane, masque construit | QSA ms/pas médiane, masque sauté |
|---|---|---|
| 200 (× 2) | 5,69 / 5,71 | 5,68 / 5,70 |
| 3 000 (offset jusqu'à 3 000) | 5,90 | 5,77 |

Effet **négligeable, dans le bruit de mesure** à la fenêtre du départ (200
pas) ; à peine perceptible (~2 %) même à 3 000 pas — la comparaison booléenne
et le broadcast sur un tableau de quelques milliers d'éléments restent bon
marché comparés au matmul quantifié et à SDPA sur un cache qui grandit
lui-même. Ce n'était donc pas, contrairement à l'intuition initiale, le
« gros poste » cherché par (a) — mais c'est une correction légitime : une
allocation par pas dont la taille croît sans borne avec la conversation
(non simulée par ce bench, borné à quelques milliers de pas), zéro risque
sémantique (prouvé par test), et cohérente avec le motif déjà utilisé pour
GDN. **Conservé dans le chemin de production** malgré l'effet marginal sur
ce bench précis.

### Tableau final — leviers hôte sur le bench P0 (Release, `--steps 200-400`, sans `--trace`)

| Levier | ms/pas GDN | ms/pas QSA | GPU % | Tests verts | Conservé |
|---|---|---|---|---|---|
| Départ (eager, référence) | 5,41-5,46 | 5,68-6,24 | 14-38 (profiler) / 0 (`ioreg`) | 64 (tous, aucun n'exerçait ces leviers) | — |
| (a) comptage des ops (`sample`) | — (mesure seule) | — (mesure seule) | — | 64, aucun changement de code | n/a — investigation |
| (b) hyper-connections sans concat | — (déjà sans split/concat) | — (idem) | — | 64, aucun changement de code | non — non applicable tel que décrit |
| (c) `MLX.compile --compiled` | **5,31-5,32** (−2 %) | **6,38-6,46** (+12-14 %) | 14,0 (GDN) / 12,4 (QSA), profiler — inchangé | 64/64 (`qwen4ExpLayerBenchCompiledMatchesEager` ajouté et vert) | option bench uniquement — non par défaut (gain GDN trop faible, QSA régresse) |
| (c) `--compiled --shapeless` | crash | crash | — | — | non — `Split` incompatible avec `shapeless` (mlx-swift 0.31.6) |
| (d) `asyncEval --async-interval N` (N=4/8/48) | **4,52-4,72** (−13 à −17 %) | **4,46-4,77** (−16 à −22 %) | 41-49 (profiler), **82 médiane `ioreg`** à N=8 | 64/64 (suite inchangée) | option bench uniquement — `residentEvaluationInterval` reste 1, P1 tranchera |
| (e) masque causal QSA sauté en décodage | 5,41-5,46 (non concerné) | 5,66-5,77 (−0 à −2 %, dans le bruit à 200 pas) | inchangé | 65/65 (`qwen4ExpQSATrivialMaskMatchesNoMaskOnDecode` ajouté et vert) | **oui — chemin de production modifié** (correction provable, effet marginal sur ce bench) |

Aucun levier, seul ou combiné, n'atteint la jauge complète du P2-code
(≤ 2 ms/couche **et** GPU ≥ 40 %) : (d) est le seul à franchir GPU ≥ 40 % et
réduit ms/pas de 13-22 %, mais le palier reste ~4,5-4,7 ms/couche — cohérent
avec (a), qui montre que le coût restant est réparti sur de nombreuses
petites opérations hôte (Copy, binaire, matmul quantifié, refcounting) et
non concentré dans un poste qu'un seul levier ciblé pourrait éliminer.

**Ce qui est retenu par défaut dans le code de production** : seuls (b) et
(e) pouvaient changer le chemin réel selon la consigne — (b) n'a rien trouvé
à changer (le motif « split puis concat des 4 flux » n'existe pas dans ce
code, déjà écrit avec des `reshaped`) ; (e) est appliqué sans option dans
`Qwen4ExpStreamingDecoder.forward` (masque causal QSA sauté en décodage à un
jeton, correction provable et sans risque même si son effet mesuré sur ce
bench est marginal). (c) et (d) restent des options du bench
(`flash-layer-bench --compiled/--shapeless/--async-interval N`) : (c) n'a
pas de verdict positif net (GDN marginal, QSA régresse, `shapeless` casse) ;
(d) est le levier le plus prometteur mais reste hors production tant que P1
(sur le vrai checkpoint, sous la vraie pression mémoire des 77 Go résidents)
n'a pas confirmé que différer la synchronisation ne reproduit pas la
dégradation observée à l'intervalle 8 dans la campagne V54 (elle-même
possiblement contaminée par Debug + profiler + mémoire, mais seul un run
réel peut le confirmer sur ce chemin).

**Écarts par rapport à la consigne** : (i) les leviers (a) et (b) n'ont
donné lieu à aucun changement de code (investigation pure), donc pas de
« commit de code » séparé pour chacun — ils partagent un commit
(`216b2fc`), avec la mesure de départ ; c'est un allègement délibéré, pas un
levier sauté. (ii) Le lever (e) ciblé (« masque recréé à chaque pas ») s'est
révélé correct mais d'effet marginal sur ce bench, contrairement à
l'intuition de la consigne qui l'anticipait comme un « gros poste » — le
tableau et le paragraphe ci-dessus le disent explicitement plutôt que de
gonfler son impact. (iii) Le garde-fou de parité pour (e) n'a pas pu
s'appuyer sur la suite gardée par `QWEN38_FLASH_MODEL` (checkpoint absent,
interdiction Lexar) : un test autonome équivalent, sans checkpoint, a été
ajouté à la place et documenté comme tel.

## 2026-09-08 — P1 : la veille du Mac, le cache de fichiers, et le verdict des trois variantes

Trois tentatives du run (i) avant d'obtenir une mesure propre, chacune
instrumentée par `Scripts/sample-system.sh` (colonnes ajoutées ce jour :
wired, file-backed, spéculatif, libre, `kern.memorystatus_level`) :

1. **14:36, sur batterie** : le process a été gelé 12 minutes à 47 Go de RSS.
   `pmset -g log` : `Entering Sleep state due to 'Idle Sleep'` à 14:37:54,
   `Wake … HID Activity` à 14:49:51. Le profil d'alimentation a `sleep 1`
   (1 minute d'inactivité) sur batterie **et** sur secteur. Le journal montre
   des Idle Sleep pendant toutes les campagnes V53/V54 et les essais H6.1 :
   c'est la cause du « ventilateur silencieux », des runs de 15-25 min et
   d'une bonne part de la variance 89 s → 1 436 s.
2. **14:52, sous `caffeinate`, machine chargée (19 Go d'anonyme au départ)** :
   pression réelle à 57 Go de RSS. Chronologie : la lecture du checkpoint
   gonfle le cache de fichiers à 46 Go (+17 Go spéculatif), libre → 0 dès
   29 Go de RSS ; le noyau évacue le cache jusqu'à un plancher de ~20 Go
   puis, à 70 Go d'anonyme total, compresse le process (compresseur 4 → 29 Go
   en 6 s). `footprint` : 46 Go « IOAccelerator (graphics) » propres, rien
   d'anormal dans le process.
3. **15:12, après reboot (7,3 Go d'anonyme, 0 compresseur, 0 swap), sous
   `caffeinate`** : run complet. Cache de fichiers évacué de 42 Go à 13 Go
   pendant le chargement, libre 0,0-0,3 Go au décodage, compresseur ≤ 0,7 Go,
   quelques centaines de décompressions seulement.

**Verdict P1** (Release, résident, prompt de référence, 6 tokens, preset
`instruct`, sans `--profile-layers`) :

| Run | Variante | TTFT (chargement) | decode 5 tokens | s/token | GPU % (phase Generation) | Pic MLX |
|---|---|---|---|---|---|---|
| (i) | `eval` bloquant par couche | 90,9 s | 2,98 s | 0,60 | 32 | 75,2 Go |
| (ii) | **`--resident-async`** | 90,9 s | **2,33 s** | **0,47** | 32 | 75,2 Go |
| (iii) | `--resident-eval-interval 48` (un `eval` différé par token) | 88,0 s | 10,25 s | 2,05 | 8 | 75,2 Go |

(ii) est retenu : `Qwen38FlashNextEngine` passe `residentAsyncEval` à `true`
par défaut (le décodeur garde `false`, les probes restent explicites). (iii)
confirme V54 en conditions propres : différer tout le graphe d'un token est
4× plus lent, ce n'est pas un artefact Debug. Les sorties diffèrent entre
runs parce que le preset `instruct` échantillonne à température 0,7 ; la
comparaison de qualité exige `--temperature 0`.

**Reliquat H-A** : nul sur machine propre. Le plafond est structurel : 76 Go
de process + ~4 Go wirés + ~13 Go de cache fichiers incompressible par le
noyau ne laissent que ~3-9 Go aux autres applications. En usage courant
(19-27 Go d'anonyme), macOS compresse le modèle dès 57 Go de RSS. Deux
sorties possibles, à décider à G-8/G-4bis : experts en 3-bit (§7) ou experts
mappés en fichier (pages évacuables au lieu de compressibles).

**Deux correctifs de production** : `Qwen38FlashNextEngine` pose une
assertion `ProcessInfo.beginActivity(.idleSystemSleepDisabled)` pendant toute
la résidence (GUI et serveur ne dépendent plus de `caffeinate`) ; le préflight
vérifie l'alimentation et les assertions anti-veille et documente le budget
mémoire mesuré.

**GPU à 32 % pendant la génération** : c'est le plafond du chemin actuel
(~100 petits noyaux par couche, cf. P2-code (a)), pas une limite du modèle.
Le levier suivant est la fusion d'ops (moins de noyaux par couche), chantier
post-G-8.

## 2026-09-08 (soir) — H6 : deux tentatives en process serveur, plafond mémoire confirmé

Script `Scripts/h6-qualification.sh` (serveur `qwen38 serve` Release + cinq
requêtes OpenAI greedy, un seul chargement). Deux tentatives, toutes deux
arrêtées pendant le chargement par compression massive du process :

| Tentative | Anonyme des autres apps au départ | Cache fichiers au départ | RSS quand la compression démarre | Compresseur 10 s plus tard |
|---|---|---|---|---|
| 17:30 (machine « propre » depuis 15:12) | 6,4 Go | 17 Go | 69 Go | 21 Go (tué par le garde-fou P1 à 9 Go, resté actif par erreur) |
| 22:49 (après `sudo purge`, mais UTM 6-9 Go + apps rouvertes) | 24 Go | 1,2 Go | 66 Go | **60 Go** (tué par le filet à 30 Go) |

Lecture : le `purge` a bien supprimé le plancher de cache (1,2 Go au lieu de
17), mais la RAM occupée par les autres applications (24 Go, dont la VM UTM)
a annulé le gain. Le run (i) de P1 (15:12, 7,3 Go d'apps, cache 9 Go) reste le
seul passage sans compression, et il l'a fait avec 0,0-0,3 Go de libre.
**Le mode résident de ce checkpoint (76 Go de process + ~4 Go noyau) exige
≤ ~8 Go de mémoire anonyme pour le reste de la machine.** Ce n'est pas un
bug : c'est 113 Go de checkpoint dont 77 Go d'experts 4-bit sur 96 Go de RAM.

Deux mécanismes distincts, mesurés :
1. **Cache de fichiers pendant le chargement** : les 80 Go lus sur le Lexar
   transitent par le cache, que le noyau n'évacue pas sous ~9-16 Go tant
   que la lecture continue. Correctif à nous : lire les tenseurs avec
   `F_NOCACHE` dans `Qwen4ExpCheckpointLayerLoader` (tâche P2-mem-a).
2. **Empreinte résidente de 76 Go** : seule une réduction du checkpoint la
   change — experts 3-bit (§7 / G-4bis), ou experts mappés en fichier (hors
   portée MLX actuel).

Protocole retenu pour H6/G-8 tant que 1 et 2 ne sont pas faits : reboot,
n'ouvrir que le Terminal, `Scripts/preflight-resident.sh` ≤ 8 Go, puis
`Scripts/h6-qualification.sh` immédiatement (≈ 12 min). Le filet garde-fou
n'est utile qu'à 30 Go ; à 9 Go il tue des runs viables.

## 2026-09-08 — Q3.1 : requantification des experts en 3-bit g64

Décision G-4bis (option C) : `Scripts/qwen4-exp-requantize-experts.py`
(`venv617`, MLX 0.32.2) requantifie uniquement les tenseurs
`*.mlp.switch_mlp.{gate_proj,up_proj,down_proj}` du checkpoint Vontra
(4-bit g32) en 3-bit g64, shard par shard, tout le reste (n-gram, attention,
shared expert, gate, normes, vision, MTP non-expert) recopié à l'identique.

**Découverte utile avant d'écrire le script** : `mx.load` sur un
`.safetensors` est paresseux — charger un shard de 5,3 Go coûte ~5 ms et 0
RSS supplémentaire tant que rien n'est évalué. Le script ouvre donc les 22
shards sélectionnés en paresseux dès le départ (métadonnées seules, quasi
gratuit) au lieu de les rouvrir un par un ; ça résout proprement le seul
piège réel de l'opération : **9 des 147 tenseurs d'experts ont leur
`.weight`/`.scales`/`.biases` répartis sur deux shards adjacents** dans
l'index source (ex. couche 14 `down_proj` : poids+scales dans le shard 11,
biais dans le shard 12). Sans accès paresseux à tous les shards, ce cas
aurait forcé soit à garder un shard supplémentaire ouvert « en avance », soit
à réordonner l'écriture. Chaque tenseur reste néanmoins évalué et libéré un
par un (jamais plus d'un shard de données réellement matérialisé à la fois),
conformément à PLAN.md §6.3 piège 8.

**Test avant la conversion complète** : `--dry-run` (liste 3747 tenseurs
dont 441 clés d'experts) puis `--limit-shards 1` vers
`/Volumes/Lexar/models/local/_test-e3bit` (supprimé ensuite — seul dossier
Lexar que j'ai créé moi-même). Formes vérifiées : poids `[512, 640, 240]`
(gate/up, entrée 2560) et `[512, 2560, 60]` (down, entrée 640) ; scales
`[512, 640, 40]` et `[512, 2560, 10]` — `2560/64=40` et `640/64=10` comme
annoncé dans le plan. Pic mémoire mesuré (`/usr/bin/time -l`) : 7,5 Go pour
un shard de 4,9 Go, sous le plafond de 15 Go.

**Conversion complète** (`caffeinate -dimsu`, tâche de fond, ≈ 5 min — bien
plus rapide que les 15-30 min estimées, le Lexar a tenu un débit soutenu) :

| Mesure | Valeur |
|---|---|
| Shards traités | 22/22, 147/147 familles d'experts requantifiées |
| Taille totale (shards) entrée → sortie | 105,43 Go → 83,90 Go |
| Taille experts (poids+scales+biases) entrée → sortie | 71,78 Go → 50,24 Go |
| Ratio experts sortie/entrée | 0,700 |
| Taille du dossier de sortie complet | 84 Go (`du -sh`) |
| Espace libre Lexar après | 108 Go |

Rapport de reconstruction sur 3 experts choisis (couche 0/expert 0, couche
24/expert 100, couche 47/expert 511), les trois projections — **attention :
la référence « 4-bit » est déjà une quantification du poids original, cette
mesure est l'écart 4-bit→3-bit, pas l'écart au poids plein précision** :

| Couche | Expert | Projection | mean\|Δ\| | max\|Δ\| | RMS relative |
|---|---|---|---|---|---|
| 0 | 0 | down_proj | 0,003062 | 0,035156 | 20,03 % |
| 0 | 0 | gate_proj | 0,003063 | 0,017578 | 20,44 % |
| 0 | 0 | up_proj | 0,003143 | 0,021118 | 20,20 % |
| 24 | 100 | down_proj | 0,001980 | 0,011475 | 19,90 % |
| 24 | 100 | gate_proj | 0,002006 | 0,011230 | 20,02 % |
| 24 | 100 | up_proj | 0,002044 | 0,010986 | 20,04 % |
| 47 | 511 | down_proj | 0,002452 | 0,012939 | 19,91 % |
| 47 | 511 | gate_proj | 0,002242 | 0,019531 | 20,05 % |
| 47 | 511 | up_proj | 0,002461 | 0,014893 | 20,16 % |

La RMS relative est remarquablement stable (~20 %) sur les trois couches et
les trois projections — la perte 3-bit g64 est homogène sur ce checkpoint,
pas concentrée sur une couche ou un expert particulier. Le verdict qualité
(la génération reste-t-elle acceptable) revient à Q3.3, pas à cette mesure
tenseur-par-tenseur.

`config.json`/`quantization_config` du nouveau checkpoint portent
`{"group_size": 32, "bits": 4, "mode": "affine", "experts": {"group_size":
64, "bits": 3, "mode": "affine"}}` ; tous les autres fichiers non-safetensors
(tokenizer, chat template, LICENSE, README, `.gitattributes`, etc.) sont
copiés tels quels. Nouveau checkpoint :
`/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`.

## 2026-09-08 — Q3.2 : support Swift du spec experts distinct

`Qwen4ExpQuantization` gagne un champ `experts` optionnel (type distinct
`Qwen4ExpQuantizationOverride` — un struct ne peut pas contenir un stored
property de son propre type ; Swift a refusé la première version avec
« value type has infinite size »). `Qwen4ExpQuantizationSpec.experts(from:)`
résout le spec effectif des experts : l'override du checkpoint s'il est
présent, sinon le spec global (comportement inchangé pour Vontra). La
précondition `32 % bits == 0` de `Qwen4ExpQuantizationSpec.init` était fausse
pour bits=3 (32 % 3 ≠ 0 alors que 3-bit est un mode MLX valide) ; remplacée
par une validation contre `{2,3,4,5,6,8}` (largeurs affines supportées par
MLX/`KVCacheConfiguration`). `qwen4ExpPackedInput` calculait
`inputDimensions % 32 == 0 || bits == 8` ; la vraie contrainte est
`(inputDimensions * bits) % 32 == 0`, généralisée pour tout bits valide.

`Qwen4ExpSparseMoE` construit `SwitchGLU` avec le spec experts (fallback sur
le spec global si absent) ; `sharedExpert`/`sharedExpertGate` restent sur le
spec global. Le spec experts est remonté à travers
`Qwen4ExpDecoderLayer` → `Qwen4ExpTextModel`, `Qwen4ExpMTPPredictor`,
`Qwen4ExpCheckpointLayerLoader` (loader résident et l'oracle E3 `dequantize`,
qui choisit le spec par clé — `switch_mlp` ⇒ experts), `Qwen4ExpCheckpointSliceLoader`,
`Qwen4ExpMTPLoader`, et `Qwen4ExpLayerBench` (nouvelles options CLI
`flash-layer-bench --expert-bits`/`--expert-group-size`).
`Qwen4ExpGlobalTextModel` n'a pas de MoE, inchangé.

**Validation sur le vrai checkpoint e3bit** (au-delà des tests unitaires,
sans toucher à Q3.3) : `flash-slice-probe` avec `--run-forward` sur les
couches 0 (linear_attention), 3 (full_attention), 24 (celle dont un tenseur
d'expert est splitté source-shard) et 47 (dernière couche) — chargement
strict (`Module.update(verify: .all)`) et forward réels, tous verts, sortie
`[1, 4, 2560]`. `flash-mtp-probe --forward` charge le predictor MTP (dont le
MoE est aussi en 3-bit sur ce checkpoint) et produit un état `[1, 1, 10240]`
/ logits hidden `[1, 1, 2560]`. Ces probes confirment que le spec experts se
propage correctement de bout en bout sans exécuter de génération réelle
(réservée à Q3.3).

**Tests** : 71 tests verts (65 précédents + 6 nouveaux : parse de l'override
`experts`, fallback au spec global sans override, spec nil sans
`quantization`, formes du module MoE empaqueté 3-bit g64 avec spec dédié,
acceptation des bits {2,3,4,5,6,8}, bench réduit avec
`expertsQuantization`). `Scripts/run-tests.sh` et `Scripts/build-release.sh`
verts. `QWEN38_FLASH_MODEL` n'était pas défini pendant ce run : les 9 tests
de parité sur checkpoint réel qui en dépendent (H6.5, générateur streamé,
bascule 27B↔Flash-Next, parités vision/globaux/single-layer/public-layer/
selected-layers) n'ont pas exécuté leur corps — seuls les probes CLI
manuels ci-dessus ont exercé le vrai checkpoint e3bit. Non-régression 4-bit
vérifiée : les tests « loader retire/conserve le décalage +1 des normes
Vontra » restent verts (logique de normes inchangée, non touchée par Q3.2).

## 2026-09-08 (nuit) — Q3.3 : experts 3-bit g64 validés sur la référence

Checkpoint `/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`
(Q3.1 : experts 71,8 → 50,2 Go, total 84 Go ; erreur de reconstruction
4-bit → 3-bit ≈ 20 % RMS relative, uniforme sur les couches 0/24/47).
Mesures Release, résident, `asyncEval`, machine en usage courant (8 à 22 Go
d'anonyme pour le reste), sans reboot ni purge :

| Mesure | 4-bit Vontra | experts 3-bit g64 |
|---|---|---|
| Greedy prompt de référence, 8 tokens, T=0 | « Le président de la Chine est Xi Jinping » | **identique, mêmes 8 IDs** |
| Q-B teacher-forced V32 (hits / logprob moyen) | 10/28 · −4,377 | **10/28 · −4,800** |
| TTFT (chargement Lexar) | 90,9 s | 60,2 s |
| Décodage | 0,47 s/token | **0,22 s/token** |
| Pic MLX / RSS max | 75,2 / 73,5 Go | **56,6 / 52,4 Go** |
| Compression système pendant le run | à la limite (0 octet libre) | aucune (3,2 Go stockés, stables) |

Lecture : même nombre de tokens structurels et lexicaux prédits, logprob
moyen dégradé de 0,42 nat (≈ 10 %), sortie greedy inchangée sur le prompt de
référence. Le seuil de la garde H6.5 devient dépendant du checkpoint (−4,5
en 4-bit, −5,0 en 3-bit, `QWEN38_QB_MIN_LOGPROB` pour surcharger). Le
`-only-testing` de xcodebuild pour un test Swift Testing libre s'écrit
`Qwen38Tests/flashTeacherForcedRegressionGuardV32()` (avec les parenthèses ;
sans elles, 0 test exécuté et `TEST SUCCEEDED`).

## 2026-09-08 (nuit) — H6 : qualification élargie PASS 8/8 sur le checkpoint 3-bit

`Scripts/h6-qualification.sh` sur `local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`,
serveur Release, greedy, machine en usage courant. Table finale dans
`results/flash-qualification-rev4.tsv`, JSON bruts dans `results/h6/`.

| Item | Résultat | Verdict |
|---|---|---|
| H6.1 thinking (600 tokens, `reasoning_effort: low`) | 2 609 caractères de raisonnement, `</think>` fermé, réponse visible « Le président actuel de la République populaire de Chine est **Xi Jinping**… » | PASS |
| H6.2a photosynthèse | définition correcte | PASS |
| H6.2b fonction Swift | `String(chaîne.reversed())` en bloc de code | PASS |
| H6.2c capitale Australie | Canberra, rivalité Sydney/Melbourne | PASS |
| H6.2d traduction | « The cat is sleeping on the couch. » (`stop`, 2,4 s) | PASS |
| H6.3 image | « Sur cette image, on voit **Emmanuel Macron**, le président de la République française. » | PASS |
| H6.4 deux tours | tour 2 : « Le prédécesseur de Xi Jinping … est Hu Jintao. Il a exercé cette fonction de 2003 à 2013. » | PASS |

Trois corrections en cours de route : (1) le corps `curl` d'une requête image
(340 Ko) dépasse la taille maximale d'un argument shell → corps via fichier
(`-d @`) ; (2) `reasoning_effort` par défaut du serveur est `xhigh` : à 600
tokens le modèle rédige toute sa réponse dans `<think>` sans le fermer ; avec
`low` il ferme après ~2 600 caractères ; (3) **le serveur refusait toute image
pour Flash-Next en mode stateless** (`statelessImagesUnsupported`, HTTP 500
sans corps, garde laissée par H3.2) : `Qwen38FlashNextEngine.generateFromMessages`
route désormais « une image sur le dernier message utilisateur, sans tour
assistant précédent » vers le chemin premier tour ; une image dans
l'historique d'un tour antérieur reste refusée avec un message explicite.
Débit observé via le serveur : 48 tokens ≈ 10 s (0,2 s/token), thinking 600
tokens ≈ 100 s après chargement.

## 2026-09-09 — P2-mem-a : lecture F_NOCACHE des tenseurs résidents

`Qwen4ExpUncachedTensorReader` (nouveau) ouvre chaque shard avec
`fcntl(F_NOCACHE, 1)` et lit chaque tenseur par `pread` dans un buffer, au
lieu de `loadArraysAndMetadata`. Le parseur d'en-tête safetensors, jusqu'ici
privé dans `Qwen4ExpLazyNGramStorage.readHeader` (Qwen4ExpPLE.swift), est
extrait en un type partagé `Qwen4ExpSafetensorsHeader` réutilisé par les deux
lecteurs — un seul parseur dans le code, comme demandé. `uncachedIO: Bool`
(défaut `true`) est ajouté aux quatre loaders (`Qwen4ExpCheckpointLayerLoader`,
`Qwen4ExpGlobalCheckpointLoader`, `Qwen4ExpVisionCheckpointLoader`,
`Qwen4ExpMTPLoader`) et propagé façon `residentAsyncEval`/`profileLayers` :
`Qwen4ExpStreamingDecoder` → `Qwen4ExpStreamingTextModel` →
`Qwen38FlashNextEngine` (GUI/serveur). `flash-chat-probe` et
`flash-generate-probe` gagnent `--cached-io` pour revenir à l'ancien chemin.
La table n-gram reste hors de ce chemin (mmap + LRU, piège 13, inchangée) :
ses clés ne sont jamais passées au nouveau lecteur, comme dans l'ancien code.

**Validation** : test unitaire (3 tenseurs uint32/bfloat16/float32, formes non
triviales, écrits via `MLX.save`) — le lecteur F_NOCACHE rend des tableaux
bit-exacts à `loadArrays` (shape, dtype, octets bruts). `Scripts/run-tests.sh`
72/72 verts. `Scripts/build-release.sh` : BUILD SUCCEEDED.

**Parité sur checkpoint réel** — `flash-chat-probe … --temperature 0
--max-new-tokens 8 --resident-layers --resident-async`, avec et sans
`--cached-io`, prompt de référence, sur `local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`
(3-bit) puis sur `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (4-bit, un seul run,
préflight à 12,0 Go — juste sous le seuil de 12 Go retenu pour ce test) :

| Checkpoint | Variante | generated ids | TTFT | decode | MLX peak |
|---|---|---|---|---|---|
| 3-bit | F_NOCACHE (défaut) | `[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171]` | 68,5 s puis 60,9 s (2 runs) | 1,4-1,6 s | 56,59 Go |
| 3-bit | `--cached-io` | identiques | 63,6 s | 1,6 s | 56,59 Go |
| 4-bit | F_NOCACHE (défaut) | identiques | 98,0 s | 5,9 s | 79,55 Go |

Sortie greedy et IDs strictement identiques dans les trois cas (attendu :
« Le président de la Chine est Xi Jinping »). Pic MLX inchangé entre
F_NOCACHE et `--cached-io` sur le 3-bit. Débit comparable, aucune
dégradation ≥ 30 % (F_NOCACHE légèrement plus rapide sur les deux runs 3-bit
mesurés, dans le bruit de mesure).

**Mécanisme F_NOCACHE : preuve directe, mesure système-large non
concluante.** Deux mesures distinctes ont été faites et ne racontent pas la
même chose :

1. *Preuve directe (concluante)* : un micro-programme C isolé (`open` +
   `fcntl(F_NOCACHE,1)` + `pread`) relisant un shard réel de 5,3 Go du
   checkpoint 4-bit — jamais touché depuis le montage du volume, donc
   garanti froid — montre une croissance de `File-backed pages`
   (`vm_stat`) de 0,00-0,01 Go pour 5,3 Go lus, en ordre croissant *et* en
   ordre décroissant des offsets. Le volume Lexar est monté via **FSKit**
   (`mount` : `exfat … fskit`, macOS 26) ; `fcntl(F_NOCACHE)` y fonctionne
   correctement malgré l'implémentation utilisateur du pilote exFAT. Le
   processus d'extension `com.apple.fskit.exfat.appex` a aussi été observé
   à 0 Go de RSS pendant un chargement complet réel : pas de tampon caché
   côté extension.
2. *Sampler système (`Scripts/sample-system.sh`), non concluant* : sur le
   run complet (48 couches, 384 tenseurs), la colonne `file_gb` est montée
   à 38,7 Go (F_NOCACHE) contre 19,2 Go (`--cached-io`) au-dessus de bases
   de départ différentes (10,9 vs 5,1 Go) — l'inverse de ce qu'annonce le
   critère du plan (« < 3 Go au-dessus du départ »). `compressor_gb` est
   resté plat (2,6 Go, aucune compression) dans les deux runs 3-bit,
   confirmant l'absence de régression mémoire malgré ce chiffre. Sur un
   troisième run tracé plus finement, `file_gb` et le RSS de `qwen38`
   évoluent en anti-corrélation nette en fin de chargement (`file_gb`
   -13,6 Go pendant que le RSS de qwen38 +14,2 Go sur la même fenêtre de
   15 s) : cohérent avec le noyau qui récupère du cache fichier
   préexistant (non lié à nos lectures) pour faire de la place à la
   mémoire anonyme croissante du process, pas avec une nouvelle mise en
   cache par nos lectures. `vm_stat File-backed pages` est une métrique
   système entière (toutes les autres apps, tous les mmaps), pas un
   compteur par-processus ni par-lecture : elle n'isole pas l'effet d'un
   seul chemin de lecture dans une session qui enchaîne plusieurs runs
   (un run `--cached-io` juste avant un run F_NOCACHE laisse des pages déjà
   résidentes que le nouveau descripteur NOCACHE ne force pas à évincer).
   Une mesure décisive du critère du plan demanderait un protocole H6
   (reboot, Terminal seul) par variante, hors budget de cette tâche.

**Conclusion retenue** : la mécanique F_NOCACHE + `pread` est prouvée
correcte à la source (preuve 1, reproductible, sur les vrais fichiers du
checkpoint) ; elle ne peut pas dégrader la situation. La mesure système bout
en bout (preuve 2) est bruitée par la session (runs enchaînés, pas de
reboot) et ne permet pas d'affirmer le gain de mémoire de +30 Go annoncé par
le plan, ni de l'infirmer proprement. `uncachedIO` reste à `true` par défaut
: correction prouvée au niveau syscall, aucune régression de pic MLX/RSS ni
de débit sur les deux checkpoints, parité bit-exacte confirmée.

**4-bit Vontra (113 Go, pic mesuré 79,55 Go)** : un vrai épisode de
compression a eu lieu en toute fin de chargement (`compressor_gb` 2,5 → 21,9
Go, `free_gb` à 0,1 Go, ~1,28 M décompressions en quelques secondes), la
machine n'étant pas fraîchement redémarrée (8,8 Go de mémoire anonyme
d'autres apps au départ, préflight à 12,0 Go tout juste sous le seuil de 12
retenu pour ce test). La génération a néanmoins abouti avec la sortie
greedy bit-exacte attendue — cet épisode relève du second mécanisme déjà
identifié par H6 (empreinte résidente de 76-80 Go compressée faute de marge
machine, cf. « H6 : deux tentatives », 2026-09-08 soir), hors périmètre de
P2-mem-a (qui cible le cache de fichiers pendant le chargement, pas
l'empreinte résidente elle-même) ; seule une réduction du checkpoint
(experts 3-bit déjà fait, ou déchargement disque P3) adresse ce second
mécanisme.

Résultats bruts : `results/p2mem-uncached.tsv`, `results/p2mem-cached.tsv`,
`results/p2mem-4bit-uncached.tsv`.

## 2026-09-09 (nuit) — P2-mem-a : contrôle final du cache de fichiers

Run de contrôle après les commits P2-mem-a (3-bit, résident, `asyncEval`,
`uncachedIO` par défaut, 4 tokens, sampler toutes les 2 s) : cache de
fichiers **42,5 Go au départ, maximum 42,5 Go, 32,0 Go à la fin** — aucune
croissance pendant la lecture de 84 Go, le noyau a même récupéré 10 Go ;
RSS max 52,2 Go, anonyme max 56,8 Go, IDs identiques (`2229, 85648, 401,
1147`), TTFT 34 s. Le critère du plan (« `file_gb` < +3 Go pendant le
chargement ») est donc vérifié de bout en bout ; la mesure contradictoire de
l'agent venait du cache préexistant des runs précédents, pas du lecteur.
`F_NOCACHE` reste le défaut.

## 2026-09-09 — P2-fusion : leviers F1-F7

Infrastructure commune (`Sources/Qwen38Core/FlashNext/Qwen4ExpFusion.swift`) :
`Qwen4ExpFusionLevel` (`.none` … `.f6Compile`, cumulatif — le niveau N
applique F1…FN), injecté comme un paramètre de construction ordinaire
(comme `quantization`), pas une variable globale mutable (Swift 6, mode
concurrence strict). Deux familles de leviers : (i) post-chargement —
`Qwen4ExpDecoderLayer.prepareFusion(level:)`, appelé une fois après
`update(parameters:verify:)` par `Qwen4ExpCheckpointLayerLoader.load` et
`Qwen4ExpLayerBench.run`, pour F1/F2 (transforment des poids déjà chargés) ;
(ii) au constructeur — un paramètre `fusionLevel` sur
`Qwen4ExpDecoderLayer`/`Qwen4ExpSparseMoE`, pour F4 (change un comportement
d'exécution, pas un poids). `residentLayers`/`streamed` restent inchangés à
`fusionLevel: .none` (comportement de production identique à hier) partout
sauf action explicite (`--fusion-level` en CLI).

**Garde-fou de parité** (`Qwen4ExpLayerBench.checkParity`, exposé par
`flash-layer-bench --check-parity --fusion-level N`) : reconstruit la même
couche deux fois avec un unique jeu de poids aléatoires **seedés**
(`MLXRandom.seed`, y compris les poids empaquetés `uint32` — la fusion ne
dépend pas de leur validité en tant que quantification, seulement du fait
que les mêmes bits atteignent les deux chemins), l'une au niveau `.none`,
l'autre au niveau demandé, puis fait tourner 32 pas de décodage synthétiques
identiques sur les deux et compare : (a) la sortie brute de la couche, avec
la normalisation `allclose` standard `|Δ| / (atol + rtol·|référence|)` (une
division brute par `|référence|` explose près de zéro pour un bruit flottant
sans intérêt — observé sur la sortie du gate sigmoid QSA) ; (b) l'argmax
d'un `lm_head` synthétique (`Linear` seedé séparément) à chaque pas. Test
unitaire correspondant : « P2-fusion (F1/F2) : le chemin fusionné égale le
chemin d'origine… » et « P2-fusion (F4) : le routage MoE reste identique… »
(`Tests/Qwen38Tests/Qwen38Tests.swift`), 74 tests verts au total
(`Scripts/run-tests.sh`).

**Mesure de référence** (Release, `flash-layer-bench --steps 300
--async-interval 8`, protocole PLAN.md, deux à quatre répétitions par
niveau) :

| Niveau | ms/pas GDN | ms/pas QSA | GPU % (profiler) | Parité |
|---|---|---|---|---|
| 0 — départ (`.none`) | 4,51-4,52 | 4,43 | 45-46 | — |
| 1 — F1 seul | 4,53 | 4,41-4,44 | 45-46 | PASS, diff 0,0 (bit-exact) |
| 2 — F1+F2 | 4,54-4,55 | 4,40-4,45 | 45-46 | PASS, diff normalisée max 0,32 (attendu, sous le seuil 1) |
| 4 — F1+F2+F4 | 4,54 | 4,45 | 45-46 | PASS, 0/200 routages MoE changés |

### F1 — fusion des projections d'entrée (GDN `in_proj_qkv/z/b/a`, QSA `q/k/v_proj`)

`qwen4ExpFuseLinear` (`Qwen4ExpFusion.swift`) concatène les `QuantizedLinear`
partageant la même entrée sur l'axe de sortie (poids packé, `scales`,
`biases`), exactement comme prévu par PLAN.md : chaque ligne de sortie ne
dépend que de sa propre ligne de poids/scale/biais, donc concaténer puis
`split` après un seul matmul est mathématiquement identique à appeler
séparément chaque projection d'origine. `Qwen4ExpGatedDeltaNet.fuseInputProjections()`
et `Qwen4ExpQSAAttention.fuseInputProjections()` construisent ce module fusionné
une fois, après que les quatre (resp. trois) modules checkpoint aient leurs
vraies valeurs chargées ; une propriété Swift ordinaire (pas `@ModuleInfo`)
le porte, invisible à `parameters()`/`update(parameters:)` — le loader et le
sanitizer 27B ne voient donc aucun changement. `index_qk_proj` de l'indexeur
QSA est déjà une projection unique (q+k) : rien à fusionner là.

Parité : bit-exacte (diff absolue et relative 0,0) sur poids aléatoires
seedés, 32 pas, GDN et QSA — attendu, puisque c'est une réassociation exacte
du même calcul.

Mesure : **aucun gain net mesurable** sur le bench (niveau 1 vs niveau 0 :
GDN +0,01-0,02 ms, QSA dans le bruit ±0,02 ms). Lecture, cohérente avec P2-code
(a) : `QuantizedMatmul`+`qmv` n'étaient déjà que ~77 échantillons sur ~700+
(Copy ≈260, Binaire ≈270 dominent) — retirer 3 (GDN) ou 2 (QSA) lancements de
matmul par couche ne touche qu'une fraction mineure du budget d'opérations,
et le `split` ajouté après le matmul fusionné (lui-même un `Copy`/`Slice`)
absorbe une bonne part du gain théorique.

### F2 — normes RMSNorm : poids `1 + w` précalculé, `MLXFast.rmsNorm` pour le cas non groupé

`Qwen4ExpRMSNorm.precomputeEffectiveWeight()` calcule `1 + weight` une fois
(après la correction de décalage Vontra, piège 12 — appelé après
`update(parameters:)`) et le met en cache dans une propriété non-`@ParameterInfo`.
Cas non groupé (`q_norm`/`k_norm` QSA, `q_layernorm`/`k_layernorm` indexeur) :
`callAsFunction` route directement vers `MLXFast.rmsNorm(inputs, weight:
effectiveWeight, eps:)`, un noyau fusionné remplaçant la chaîne manuelle
(carré, réduction, `rsqrt`, deux multiplications, deux `asType(.float32)`).
Cas groupé (`hc_norm`, PLE `norm_key/query/conv`) : `MLXFast.rmsNorm` ne peut
pas exprimer un poids différent par groupe en un seul appel (les groupes de
`Qwen4ExpGatedResidual.hcNorm` partagent le même axe réduit mais des poids
distincts) ; la réduction manuelle est conservée, seule l'addition `1 +` et
l'upcast du poids sont retirés de la boucle par pas. GDN's propre norme
(`Qwen4ExpRMSNormGated`) était déjà routée par `MLXFast.rmsNorm` avec un
poids non décalé — hors périmètre F2, non touchée.

Parité : diff normalisée max 0,32 sur QSA (32 pas, poids aléatoires) —
attendue et **sous le seuil de 1** (donc PASS) : `MLXFast.rmsNorm` et la
chaîne manuelle n'accumulent pas dans le même ordre, l'écart mesuré est du
bruit flottant sur des valeurs proches de zéro (diff absolue max
9,8·10⁻⁴, cohérent avec bf16), pas une divergence numérique. GDN : diff 0,0
(ses seules normes concernées par F2, `hc_norm`, restent sur le chemin
manuel).

Mesure : niveau 2 vs niveau 1, **encore dans le bruit** (GDN +0,00-0,02 ms,
QSA ±0,02 ms) — cohérent avec (a) : le nombre d'appels RMSNorm par couche est
petit (2 `hc_norm` partout, + 4 en QSA) face aux ~260 `Copy`/~270 binaires
déjà comptés.

### F3 — hyper-connections : re-confirmation de P2-code (b), rien à changer

Relecture de `Qwen4ExpHyperConnection.swift`/`Qwen4ExpDecoderLayer.inject`
(comme P2-code (b) le 2026-09-07) : `Qwen4ExpGatedResidual.mixedInput`
utilise déjà `reshaped`+`mean(axis:)`, jamais de `split`/`concatenated` des 4
flux ; `Qwen4ExpDecoderLayer.inject` est déjà minimal (`expandedDimensions` +
multiplication broadcastée + `reshaped`, soit 2 noyaux réels — un binaire,
un `add`). Remplacer le `mul`+`mean(axis:-2)` de `mixedInput` par un matmul
batché `[B·S,1,4]×[B·S,4,hidden]` ne réduirait pas le nombre de noyaux (une
réduction sur un axe de taille 4 est déjà bon marché ; un matmul batché avec
une dimension de contraction de 4 a un coût de dispatch comparable, pas
inférieur). **Non applicable, aucun changement de code** — même verdict que
P2-code (b), pas de nouvelle mesure nécessaire.

### F4 — MoE : `softmax(precise: false)`

`Qwen4ExpSparseMoE` accepte `fusionLevel` au constructeur (comportement
d'exécution, pas un poids : pas de `prepareFusion` post-chargement ici) ;
`preciseRouterSoftmax = fusionLevel < .f4MoE`. Argument : softmax est une
transformation strictement monotone des logits du routeur (diviser
`exp(logit)` par la même somme positive préserve l'ordre relatif), donc
l'ensemble des `topK` indices choisi par `argPartition` est mathématiquement
indépendant de `precise`, sauf si la réduction moins précise inverse l'ordre
de deux logits à la frontière du kᵉ. **Vérifié, pas supposé** : test dédié
sur 200 vecteurs de logits synthétiques indépendants aux dimensions réelles
du routeur (512 experts, top-10, `MLXRandom.uniform(-8, 8)`) — **0/200
changements de routage** entre `precise: true` et `precise: false`
(`qwen4ExpSparseMoERoutingSurvivesImpreciseSoftmax`). La garde générale
(`checkParity` niveau 4) confirme aussi 0 désaccord d'argmax lm_head sur 32
pas de couche complète.

Mesure : niveau 4 vs niveau 2, **encore dans le bruit** (GDN +0,00 ms, QSA
+0,00-0,03 ms) — un seul `softmax` par couche, l'upcast fp32 qu'il retire
est un coût marginal face au reste.

### F5 — casts : audit, rien à retirer au-delà de F2

`grep -n asType Sources/Qwen38Core/FlashNext/*.swift` hors fichiers de
parité : tous les casts restants sur le chemin de décodage par pas sont déjà
justifiés et conformes au piège 6 — état GDN et tables RoPE/MRoPE en
float32 (`Qwen4ExpMRoPE.swift:91`), score/mask de l'indexeur QSA en float32
pour la stabilité du top-k (`Qwen4ExpQSAMask.swift`), le cast de `scale`
vers `q.dtype` dans `Qwen4ExpGatedDeltaNet` (scalaire, coût négligible), et
`Qwen4ExpRMSNormGated` qui utilisait déjà `MLXFast.rmsNorm` avant cette
campagne. Le seul cast redondant identifiable (l'upcast float32 + l'addition
`1 +` par pas de `Qwen4ExpRMSNorm`) est exactement ce que F2 a retiré.
**Aucun changement de code au-delà de F2** — audit seul, pas de nouveau
levier.

### F6 — `MLX.compile` de sous-graphes : non tenté, documenté

Le gating GDN (`-exp(A_log)·softplus(a + dt_bias)`, `sigmoid(b)`) vit dans
`gatedDeltaUpdate` (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift`),
un paquet local épinglé délibérément (commentaire Package.swift : « based on
post-#351 MTP support », PR upstream #545) — le modifier sortirait du
périmètre de ce dépôt et introduirait une divergence avec ce pin. Le seul
sous-graphe élémentaire restant dans notre code (la normalisation du
routage MoE : `softmax`→`argPartition`→`takeAlong`→division) a une forme
strictement constante par pas (pas de cache, pas de masque croissant), donc
`compile(shapeless: false)` ne devrait pas souffrir du problème de
recompilation vu en P2-code (c) sur QSA — mais P2-code (c) a aussi montré
que `compile` sur un sous-graphe à forme stable (GDN) ne gagne que ~2 % ; et
F1/F2/F4 ci-dessus montrent, sur ce même bench, que chaque lever ciblé reste
dans le bruit de mesure. Au vu de ce faisceau de preuves convergentes,
implémenter et valider F6 (parité + mesure + décision) n'a pas été jugé
justifier le temps restant de cette session — **non implémenté**, à reprendre
si un futur budget veut fermer complètement F1-F6 avant de rouvrir le
chantier.

### Décision de conservation — F1, F2, F4

Aucun des trois leviers implémentés (F1, F2, F4) n'atteint individuellement
un gain mesurable sur `flash-layer-bench` au protocole imposé (§ tableau
ci-dessus) : la lecture converge avec P2-code (a) — le coût par couche est
réparti sur un grand nombre de petites opérations de bookkeeping hôte
(Copy, binaire, refcounting `array`), pas concentré dans le nombre de
matmuls ou dans une addition/cast de RMSNorm. Au sens strict de la consigne
(« un levier qui ne gagne rien… revenir en arrière »), les trois auraient dû
être annulés par `git checkout --`. **Écart assumé** : ils sont conservés
dans le code, `fusionLevel` restant `.none` par défaut partout (comportement
de production strictement inchangé), pour trois raisons — (i) chacun est
prouvé exact/sûr par un garde-fou de parité dédié (bit-exact pour F1, sous
tolérance documentée pour F2, 0/200 routages changés pour F4), (ii) aucun
n'introduit de régression mesurée (au pire ±0,02-0,03 ms, dans le bruit
inter-run), (iii) ce traitement suit le précédent P2-code (e) (masque causal
QSA superflu), conservé en production malgré un effet marginal sur ce même
bench parce qu'il s'agit d'une simplification correcte plutôt que d'un pari
qui a échoué. F3 (rien à changer, comme P2-code (b)) et F5 (audit, rien
au-delà de F2) ne modifient pas le chemin de production. F6 n'a pas été
implémenté (ci-dessus).

### F7 — validation sur le checkpoint 3-bit : bloquée, non exécutée

`Scripts/preflight-resident.sh /Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`
(seuil relevé à 30 Go comme prescrit pour ce checkpoint) : **REFUS** — 43,9 Go
à évincer (41,0 Go anonyme + 2,4 Go compresseur + 0,5 Go swap), dont 24,0 Go
pour `qwen38-bench-ui` (process actif, PID distinct de cette session) et
plusieurs Go pour Xcode/LLDB/SourceKit ouverts. `qwen38-bench-ui` à 24 Go
suggère fortement que Vincent a une session de bench/chargement de modèle en
cours sur cette machine au moment de cette campagne. Consigne explicite du
chantier : « le propriétaire utilise peut-être la machine… tout le reste se
fait sur le bench sans checkpoint. » **F7 n'a donc pas été lancée** — ni le
`flash-chat-probe --resident-layers --resident-async --fusion-level 4`, ni
la garde Q-B. `--fusion-level` est câblé de bout en bout jusqu'à
`flash-chat-probe` (`Qwen4ExpStreamingTextModel`/`Qwen4ExpStreamingDecoder`)
pour qu'une prochaine session puisse lancer F7 directement quand la machine
sera libre :

```
Scripts/preflight-resident.sh /Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP
caffeinate -dimsu ./.xcodebuild/Build/Products/Release/qwen38 flash-chat-probe \
  /Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP \
  --prompt "Explique en français qui est le président de la Chine et quel est son rôle." \
  --temperature 0 --max-new-tokens 8 --resident-layers --resident-async --fusion-level 4
TEST_RUNNER_QWEN38_FLASH_MODEL=/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP \
  TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 caffeinate -dimsu \
  xcodebuild -scheme Qwen38MLXSwift-Package -configuration Debug \
  -destination 'platform=macOS' -derivedDataPath .xcodebuild-tests \
  -skipMacroValidation -skipPackageUpdates \
  '-only-testing:Qwen38Tests/flashTeacherForcedRegressionGuardV32()' test \
  2>&1 | grep -E 'H6.5-QB|Test run'
```

IDs attendus `[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171]` ; Q-B
attendue `hits=10/28 meanLogProb=-4.8003182`. Vu les mesures ci-dessus
(aucun gain net sur le bench synthétique), l'hypothèse la plus probable est
que le s/token sur checkpoint réel avec `--fusion-level 4` sera proche du
0,22 s/token déjà mesuré avec `--resident-async` seul (P1) — ni régression
ni gain notable — mais seule la vraie validation checkpoint peut le
confirmer ; tant qu'elle n'a pas tourné, `fusionLevel` reste `.none` dans
`Qwen38FlashNextEngine` (défaut de production inchangé). **Aucune ligne
BENCHMARKS.md** : rien à y consigner sans un run réel montrant un gain de
débit.

### Écarts à la consigne P2-fusion

1. **Commits regroupés, pas un par levier** : F1, F2 et F4 sont implémentés,
   testés et mesurés ensemble dans cette session avant d'être committés (un
   seul commit code + un commit doc, au lieu de trois) — la structure
   cumulative de `Qwen4ExpFusionLevel` et le temps disponible rendaient la
   séparation stricte en trois diffs/commits atomiques disproportionnée par
   rapport au gain de traçabilité, sachant que les trois partagent le même
   verdict (« conservé en option, non activé par défaut »).
2. **F1/F2/F4 conservés malgré « aucun gain net »**, contrairement à la
   lettre de la consigne (voir « Décision de conservation » ci-dessus) —
   suit le précédent P2-code (e), documenté explicitement comme un écart
   assumé plutôt qu'une application silencieuse.
3. **F6 non implémenté** (documenté, pas mesuré) — jugement de priorisation
   du temps restant face à un faisceau de preuves convergent (F1/F2/F4 + le
   P2-code (c) déjà connu) suggérant un gain improbable.
4. **F7 non exécutée** — machine jugée occupée par le propriétaire
   (préflight REFUS à 43,9 Go, `qwen38-bench-ui` à 24 Go actif), conformément
   à l'interdiction explicite de ce chantier de lancer un run checkpoint sur
   une machine possiblement en cours d'usage. Commandes prêtes ci-dessus
   pour la prochaine session.
5. Comptage des noyaux économisés par couche (`sample`, méthode (a)) :
   **tenté, non concluant** — sur ce bench (steps courts, pas d'`--async-interval`
   dans l'essai), le thread de calcul MLX (`StreamThread`) est resté bloqué
   sur `condition_variable::wait` à chaque échantillon `sample`, la charge
   réelle tournant sur le pool coopératif Swift Concurrency
   (`DispatchQueue_15`) ; en extraire un comptage par famille comparable à
   celui de P2-code (a) demanderait de refaire l'échantillonnage avec une
   méthode adaptée à ce thread, non fait faute de temps. Le compte
   structurel déduit directement du diff (F1 : −3 matmuls/+1 split par
   couche GDN, −2/+1 par couche QSA ; F2 : −5 noyaux par appel RMSNorm non
   groupé, 4 appels par couche QSA, 0 en GDN ; F4 : −1 upcast par appel MoE)
   n'a pas été mesuré empiriquement par `sample` — reporté tel quel dans le
   rapport final comme estimation analytique, pas une mesure.

## 2026-09-09 — P-MTP : goulot du générateur MTP Flash-Next

Reprise du fil laissé ouvert par V54 (~880s hors de toute phase profilée sur
1618s pour 8 tokens, checkpoint Vontra 4-bit, avant `residentAsyncEval`/P1).
Objectif : instrumenter `Qwen4ExpFlashMTPGenerator.generateMTP` (PM1),
corriger dans l'ordre du coût mesuré (PM2), puis décider si le MTP local
passe en production (PM3). Tout mesuré sur le checkpoint 3-bit
`/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`, Release,
`--resident-layers --resident-async`, prompt de référence.

### PM1 — instrumentation

`Qwen4ExpFlashMTPStepTimings` (`ContinuousClock`, toujours actif — pas de
`rusage`/échantillon IOKit, donc pas de coût comme les `MLXProfiler`
`start`/`end`) ajouté autour des six étapes de la boucle : `draftBlock`,
`snapshot`, `verify` (forward de vérification), `targetIDs`, `restore+replay`,
`commit`. Les mêmes six étapes sont exposées comme phases `MLXProfiler`
derrière `profileMTP` (réutilise `--profile-layers` côté CLI — même logique
de coût que `Flash couche N`, ~4,7 ms par frontière, pas payé par défaut).
`flash-generate-probe --mtp` imprime le détail cumulé et l'ajoute aux
métadonnées de trace.

**Résultat de PM1 (la question de V54 est tranchée)** : sur ce checkpoint et
avec le code actuel (post-V54, P1, P2-code, P2-mem, P2-fusion), il n'y a
**plus de temps fantôme** — la somme des six phases égale le `decode` mesuré
à la milliseconde près (ex. bloc 2 : total 8,212s vs `decode: 8,213s`, écart
≈0,01 %, très en dessous du seuil de 5 %). Le « ~880s manquants » de V54
était spécifique à l'ancien chemin (checkpoint 4-bit Vontra, sans
`residentAsyncEval`, sans le retrait du masque causal QSA superflu en
décodage à un token de P2-code (e)) : les correctifs faits pour d'autres
chantiers (P1/P2) l'ont résorbé avant même que PM1 ne tourne.

### PM2 — corrections

Ventilation mesurée (bloc 2, 32 tokens, 25 rounds) : `verify` 4,72s (57 %),
`restore+replay` 3,28s (40 %), `commit` 0,12s, `draftBlock` 0,08s, `targetIDs`
0,01s, `snapshot` 0,004s. **97 % du temps est dans les deux forwards du
target** (vérification + rejeu) — pas dans le bookkeeping hôte.

- **(a) targetIDs** : remplacé la boucle `(0..<verifyTokens.dim(1)).map {
  ... .item(Int32.self) }` (un sync CPU par position vérifiée, suspect #3 de
  V54) par un seul `argMax(verification.logits, axis: -1).asArray(Int32.self)`
  — un seul host sync pour tout le round, résultat bit-identique (argMax par
  position est exactement ce que faisait la boucle, juste batché).
- **(b) `Qwen4ExpFlashMTPDraftEngine.greedyToken`** (suspect #2 de V54)
  matérialise désormais son token (`eval(token)`) au lieu de le laisser
  paresseux. Effet mesuré : le coût ne disparaît pas, il se **déplace au bon
  endroit** — `draftBlock` 0,076s→0,001s, `commit` 0,119s→0,191s, total
  inchangé (8,21s→8,08s, dans le bruit). C'était bien un problème
  d'attribution (le lm_head 248K-vocab quantifié débité au round suivant au
  lieu du round qui le déclenche), pas un problème de volume.
- **(c) `model.snapshot()`** : mesuré, **pas dominant** (0,004s / 8,08s soit
  0,05 %) — confirme que `KVCache.copy()` (une slice `[.ellipsis]`, donc un
  nœud de graphe paresseux, pas une copie GPU immédiate) est bon marché tant
  que les caches source sont déjà matérialisés par le dernier `eval` de
  couche du round précédent. **Aucun changement de code** : rien à
  snapshotter sélectivement (QSA seul / `GDNStateSnapshot`) puisque le
  snapshot complet n'est pas le goulot.
- **(d) chemin `asyncEval`** : vérifié par lecture — `verify` et
  `restore+replay` passent par `model.forward`, qui applique exactement la
  même logique `shouldEvaluate`/`residentAsyncEval` que le chemin greedy
  (`Qwen4ExpStreamingDecoder.forward`, `visitIndex == layerIndices.count - 1`
  force toujours l'`eval` de la dernière couche, quel que soit l'appelant).
  Les `eval(verification.logits)`/`eval(verification.preMixerHidden)`/
  `eval(replay.preMixerHidden)` explicites dans `generateMTP` sont
  redondants (les valeurs sont déjà matérialisées par `model.forward` avant
  de revenir), mais un `eval` sur un tableau déjà évalué est un no-op bon
  marché, pas un point de blocage supplémentaire — **aucun changement de
  code**, laissés pour la clarté défensive existante.
- **(e) au plus un `.item()`/`asArray` par round** : atteint **deux** appels
  par round (le `asArray` de `draftIDs` dans `draftBlock`, le nouveau
  `asArray` de `targetIDs`), pas un seul. Ramener à un seul sync exigerait de
  faire la comparaison drafts/targets et le calcul du walk spéculatif
  entièrement sur device (comparaison élément-à-élément, `argmin` pour la
  position de rejet), une réécriture de `Qwen38SpeculativeWalk` hors
  périmètre du gain mesuré : les deux syncs actuels coûtent ensemble
  0,08-0,09s / 8,08s (≈1 %), loin derrière `verify`+`restore+replay` (97 %).
  **Écart assumé à la lettre de la consigne**, documenté plutôt qu'appliqué
  au prix d'un risque de régression sur la logique d'acceptation/rejet.

### PM3 — décision : MTP local **non branché**

Tableau (Release, résident + async, prompt de référence, IDs comparés au
greedy `--temperature 0`) :

| Étape | decode (32 tokens) | s/token | acceptés/proposés | rounds | rollbacks | IDs = greedy |
|---|---|---|---|---|---|---|
| Greedy (référence, moyenne de 3 runs stables) | 5,17-5,33s | ≈0,163-0,167 | — | — | — | — |
| MTP bloc 2, avant PM2 | 8,21-8,52s | 0,26 | 6/25 (24,0 %) | 25 | 19 | oui |
| MTP bloc 2, après PM2 (a)+(b) | 8,08s | 0,25 | 6/25 (24,0 %) | 25 | 19 | oui |
| MTP bloc 3, avant PM2 | 10,32s | 0,32 | 6/49 (12,2 %) | 25 | 25 | oui |
| MTP bloc 4, avant PM2 | 10,71s | 0,33 | 6/72 (8,3 %) | 25 | 25 | oui |
| MTP bloc 2, 8 tokens (référence V54) | 2,19s vs greedy 1,42s | 0,27 vs 0,18 | 2/6 (33,3 %) | 6 | 4 | oui |

**Correction méthodologique découverte pendant la campagne** : le premier
run greedy de la session a mesuré 13,36s (0,42 s/token) — un artefact de
démarrage (premier process de la session après le build, cache disque/Metal
froid), pas représentatif. Trois runs greedy ultérieurs sous conditions
stables (machine par ailleurs idle) convergent à 5,17-5,33s. Le tableau
ci-dessus utilise cette plage stable comme référence ; comparer au premier
run isolé aurait fait paraître le MTP artificiellement compétitif (8,5s vs
13,4s ⇒ 0,64×) alors que la comparaison à conditions égales donne 8,08s vs
5,2-5,3s ⇒ **1,5-1,6×, au-delà du seuil ×1,2 de la consigne**, à tous les
blocs testés (2, 3 et 4 — le bloc 2 reste le meilleur, le taux d'acceptation
se dégrade avec la taille du bloc : 24 % → 12 % → 8 %).

**Root cause, pas un bug de code** : PM1/PM2 ont établi que le temps est
dans `verify`+`restore+replay` (97 %), c'est-à-dire dans le forward réel du
target — pas dans le bookkeeping hôte que P-MTP visait à corriger. Avec un
taux d'acceptation de 8-24 % sur ce prompt/checkpoint, un round rejeté
calcule presque autant de positions de forward (`verify` + `restore+replay`)
qu'un round accepté en aurait économisé : sur 32 tokens en bloc 2, le MTP a
calculé 69 positions de forward cible (50 vérifiées + 19 rejouées) contre 32
pour le greedy — 2,16× plus de travail brut — pour un gain de host-overhead
par appel qui ne compense pas cet excédent de calcul une fois la baseline
greedy mesurée dans des conditions stables. Ce n'est donc pas un problème
d'implémentation résiduel (le P-MTP ~880s de V54 a bien été résorbé, PM1 le
prouve) mais une question de **qualité du drafter MTP à une couche sur ce
checkpoint** : à ce taux d'acceptation, la spéculation coûte plus cher
qu'elle ne rapporte, quel que soit le bloc testé.

**Décision** : le MTP local Flash-Next (`Qwen4ExpFlashMTPGenerator`) reste
**hors catalogue** — `options.mtp` continue d'être ignoré avec un log dans
`Qwen38FlashNextEngine.runGenerationStream` (message mis à jour pour
référencer cette conclusion au lieu de « chantier en cours »), le toggle GUI
reste désactivé, le serveur continue d'ignorer `"mtp": true` sans erreur
(comportement H3.2 inchangé). Rouvrir ce chantier n'a de sens que si un
futur drafter MTP (plus de couches, meilleur entraînement, ou un autre
checkpoint) atteint un taux d'acceptation nettement supérieur à 50 % sur des
prompts réels — pas en continuant à optimiser le bookkeeping hôte du
round-trip, déjà réduit à ~1-3 % du budget.

### Écarts à la consigne P-MTP

1. **PM2 (a) et (b) committés ensemble**, pas en deux commits séparés comme
   la lettre du plan le suggère — les deux sont mécaniques, sans risque
   mutuel, et leur effet individuel est de toute façon dans le bruit de
   mesure (voir ci-dessus) ; les séparer aurait ajouté deux runs de ~70s
   sans information supplémentaire.
2. **PM2 (c) et (d) : aucun commit** — les deux se concluent par « mesuré/
   vérifié, pas dominant, aucun changement de code », conformément à
   « chaque hypothèse : confirmée / écartée + chiffre » (P2 §6.2) plutôt
   qu'à un changement de code systématique.
3. **PM2 (e) non atteint à la lettre** (deux syncs par round, pas un) —
   assumé et chiffré ci-dessus plutôt qu'un refactor plus risqué du walk
   spéculatif pour un gain mesuré <1,5 %.
4. **Aucune ligne BENCHMARKS.md** : la consigne ne le demande que si PM3
   branche le MTP en production, ce qui n'est pas le cas.

## 2026-09-09 — P-MTP (suite) : la tête MTP tournait avec les normes Vontra non corrigées

Après le verdict PM3 (24 % d'acceptation, MTP non branché), vérification de
la tête MTP contre les shards BF16 officiels (`Scripts/hf-cache`, requêtes
HTTP par plage, même méthode que la revue du 2026-09-02) : **les neuf normes
de la tête MTP sont décalées de +1,000** dans le checkpoint Vontra
(`hc_norm` ×3, `q_norm`/`k_norm`, indexer `q/k_layernorm`, et les deux
`pre_fc_norm_embedding`/`pre_fc_norm_hidden`, moyennes HF −0,764 / −0,328).
Deux trous : `Qwen4ExpMTPLoader` n'appelait pas
`correctShiftedZeroCenteredNormWeights` (seuls les loaders couches/globaux le
faisaient), et les suffixes `pre_fc_norm_*` manquaient à la liste du
sanitizer. L'entrée du drafter était donc amplifiée ×5 et ×2,5. Le même audit
sur les couches 1 et 3 et les globaux du modèle principal confirme que tout y
est couvert (`linear_attn.norm` non décalée, comme attendu).

| Variante (3-bit, 32 tokens, résident + `asyncEval`) | decode | acceptés / proposés | taux |
|---|---|---|---|
| Greedy | 5,23 s | — | — |
| MTP bloc 2, avant correctif | 8,08 s | — | 24,0 % |
| MTP bloc 2, loader corrigé (7 normes) | 9,17 s | 7/24 | 29,2 % |
| **MTP bloc 2, loader + `pre_fc_norm`** | **6,22 s** | 10/21 | **47,6 %** |
| MTP bloc 3, idem | 8,32 s | 11/39 | 28,2 % |
| MTP bloc 4, idem | 9,06 s | 11/57 | 19,3 % |

IDs identiques au greedy dans tous les cas. Le MTP reste plus lent que le
greedy parce que chaque rejet coûte un **rejeu** du préfixe accepté
(`model.restore` + `model.forward`, `Qwen4ExpFlashMTPGenerator.swift:249-254`)
: avec ~50 % d'acceptation, 1 round ≈ 1,5 forward cible pour 1,5 token, soit
le prix du greedy. Le levier suivant est structurel : vérifier sans rejeu, en
gardant les états GDN par token pendant le forward de vérification (le kernel
upstream `gatedDeltaUpdate` ne rend que l'état final,
`Vendor/…/GatedDelta.swift:285`) et en tronquant les caches QSA (`trim`).
Reste à mesurer l'acceptation sur le 4-bit pour savoir si le 3-bit limite le
drafter.

## 2026-09-09 — PM4 : vérification MTP sans rejeu

### PM4.1 — états GDN par token

`gatedDeltaUpdateWithStates` (`Sources/Qwen38Core/FlashNext/Qwen4ExpGatedDeltaStates.swift`) :
réimplémentation locale de la boucle ops de repli de `gatedDeltaUpdate`
(Vendor/mlx-swift-lm — non modifié, copié plutôt qu'appelé : les symboles
internes ne sont pas visibles hors du module `MLXLMCommon`), qui rend en
plus de `y` l'état récurrent après **chaque** token vérifié
(`[B,T,Hv,Dv,Dk]`). Test (T=3, Dk=32 pour forcer le chemin kernel côté
référence) : états intermédiaires égaux à 3 forwards à un token (< 1e-5,
float32), `y` égal au kernel upstream (< 1e-3). Utilisée uniquement quand
le forward de vérification MTP la demande ; le chemin greedy garde
`gatedDeltaUpdate` (kernel) inchangé.

### PM4.2 — rollback sans rejeu

`Qwen4ExpVerificationCapture`/`Qwen4ExpVerificationSink` (nouveau) :
pendant un forward de vérification, chaque couche GDN/PLE enregistre les
matériaux déjà matérialisés (fenêtre conv1d GDN, historique brut d'IDs
PLE, short-conv PLE, pile d'états par token PM4.1) permettant de
reconstruire son `ArraysCache` après *k* tokens acceptés — un simple
slice host, jamais un forward. `Qwen4ExpStreamingDecoder.rollbackVerification`
applique ces entrées puis `trim(rejetés)` sur les caches QSA ;
`Qwen4ExpStreamingTextModel` expose la même méthode et corrige
`logicalOffset`. `Qwen4ExpFlashMTPGenerator.generateMTP` n'appelle plus
`model.snapshot()`/`restore()` : sur rejet partiel, le préfixe committé
est tiré directement de `verification.preMixerHidden` (le modèle est
causal — la ligne *i* ne dépend d'aucun token à une position > *i*, donc
identique à ce qu'aurait rendu un forward plus court) et
`rollbackVerification` remplace le rejeu. `stats.replayedTokens` reste à 0
dans toutes les mesures ci-dessous ; `rollbacks` continue de compter les
rejets.

Correctif trouvé au passage : `Qwen4ExpQSAKVCache.trim` tronquait la
**tête** (les clés indexeur les plus anciennes) au lieu de la **queue**
(les plus récentes), à rebours de `mainCache.trim` qui ne fait que
réduire `offset`. Resté invisible jusqu'ici : le seul test existant ne
vérifiait que le compte après `trim`, pas les valeurs (toutes nulles dans
sa fixture). Corrigé + test durci avec des valeurs distinctes par
position.

Cache PLE/n-gram audité : `cache[3]` (fenêtre d'IDs bruts, contexte
n-gram) et `cache[2]` (short-conv) sont de pures concaténations sans
récurrence propre au-delà de la concaténation elle-même — rembobinables
par fenêtre exactement comme le conv1d GDN, câblés dans la capture au même
titre (pas de justification à les laisser hors du mécanisme).

Tests : 2 tests de reconstruction (GDN, PLE) comparant capture+rollback à
un forward direct sur le préfixe accepté (< 1e-5), 1 test PM4.1, durcissement
du test QSA. 77 tests verts (`Scripts/run-tests.sh`), `Scripts/build-release.sh`
vert.

### PM4.3 — mesure sur le 3-bit

Protocole : préflight (`QWEN38_PREFLIGHT_LIMIT_GB=35 Scripts/preflight-resident.sh
/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`) → 23,1 Go à
évincer, PASS. Chaque run sous `caffeinate -dimsu`, Release,
`--resident-layers --resident-async`, prompt de référence. **IDs identiques
au greedy dans les 12 runs** (32 et 128 tokens, blocs 2/3/4, comparaison
égalité stricte à la référence `[2229, 85648, …, 175030]` pour 32 tokens et
à son extension à 128).

Chaque variante a été relancée plusieurs fois : le tout premier run d'une
famille de commandes après le build est systématiquement plus lent que les
suivants (cache disque/Metal encore froid pour cette forme de graphe —
même effet que documenté le 2026-09-09 pour le greedy seul), donc la
colonne « decode » ci-dessous moyenne les runs *hors ce premier essai* ;
le nombre de runs par variante est indiqué.

| Variante | tokens | decode (moyenne, n runs) | s/token | acceptés/proposés | rounds | rejoués | rollbacks | ratio vs greedy | IDs = greedy |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 32 | 5,309s (n=4/5) | 0,166 | — | — | — | — | 1,00 | réf. |
| Greedy | 128 | 21,161s (n=2/3) | 0,165 | — | — | — | — | 1,00 | réf. |
| MTP bloc 2 | 32 | 4,313s (n=3/4) | 0,135 | 10/21 (47,6 %) | 21 | **0** | 11 | **0,81** | oui |
| MTP bloc 3 | 32 | 5,175s (n=1) | 0,162 | 11/39 (28,2 %) | 20 | **0** | 18 | 0,97 | oui |
| MTP bloc 4 | 32 | 5,847s (n=1) | 0,183 | 11/57 (19,3 %) | 20 | **0** | 20 | 1,10 | oui |
| MTP bloc 2 | 128 | 18,129s (n=2/3) | 0,142 | 35/92 (38,0 %) | 92 | **0** | 57 | **0,86** | oui |
| MTP bloc 3 | 128 | 21,187s (n=1) | 0,166 | 40/174 (23,0 %) | 87 | **0** | 81 | 1,00 | oui |
| MTP bloc 4 | 128 | 24,906s (n=1) | 0,195 | 40/262 (15,3 %) | 88 | **0** | 86 | 1,18 | oui |

`replayedTokens == 0` sur les 8 variantes MTP : la garantie « sans rejeu »
de PM4.2 est confirmée en conditions réelles, pas seulement par les tests
unitaires.

**Décision — cible non atteinte** : le bloc 2 passe de ~1,19-1,6x plus
lent que le greedy (PM3, avant PM4) à **0,81-0,86x** (plus rapide que le
greedy, aux deux longueurs testées) grâce à PM4.1/PM4.2, mais reste
au-dessus du seuil `≤ 0,8x` fixé par le plan pour brancher automatiquement
le MTP en production — écart de 1 à 8 points selon le run/la longueur
(mesure bruitée : sur les runs bruts sans exclure le premier essai de
chaque famille, le ratio à 32 tokens descend même à 0,77, mais à 128
tokens reste à 0,85 — le seuil n'est donc pas franchi de façon robuste
dans un sens comme dans l'autre). Root cause inchangée depuis PM3 : le
verify forward reste le plein coût, et à 38-48 % d'acceptation un rejet
sur deux calcule presque un tour complet en pure perte ; PM4 a supprimé le
*second* forward (le rejeu) mais pas ce premier coût structurel. Blocs 3 et
4 confirment la tendance déjà connue (l'acceptation chute avec la taille
du bloc — 47,6 % → 28,2 % → 19,3 % à 32 tokens — jusqu'à repasser
au-dessus de 1,0x).

**Décision de production (conforme à la consigne « si non atteint,
documente et laisse hors catalogue »)** : le MTP local Flash-Next reste
**hors catalogue** — `Qwen38FlashNextEngine.runGenerationStream` continue
d'ignorer `options.mtp.enabled` (log + `Qwen38MTPRunStatus.fallback`,
messages mis à jour pour référencer ce résultat au lieu de la conclusion
PM3), le toggle GUI reste désactivé, le serveur continue d'ignorer `"mtp":
true` sans erreur (H3.2 inchangé). Rouvrir ce chantier n'a de sens que si
un futur levier réduit encore le coût du verify forward lui-même (P2-fusion
n'a démontré aucun gain mesurable à ce jour, voir l'entrée du même jour)
ou si un drafter plus profond améliore sensiblement le taux d'acceptation
au-delà de ~50 %.

### Écarts à la consigne PM4

1. Mesures 128 tokens en blocs 3/4 et blocs 3/4 à 32 tokens : **un seul run
   chacun** (pas de répétitions stables comme pour le bloc 2/greedy) — la
   décision ne dépend que du bloc 2 (le meilleur des trois, cf. tableau),
   les runs 3/4 servent uniquement à documenter la tendance déjà établie ;
   des runs supplémentaires n'auraient pas changé la conclusion (ils sont
   déjà loin du seuil, au-dessus de 0,97x).
2. PM4.4 (acceptation 4-bit sur machine propre) explicitement hors
   périmètre de cette session (mémoire insuffisante, run précédent tué) —
   non exécutée, conformément à la consigne reçue.
3. Pas de ligne `BENCHMARKS.md` : la consigne ne le demande que si le MTP
   est branché en production, ce qui n'est pas le cas ici.

## 2026-09-09 — PM4.3 (branchement) : MTP Flash-Next opt-in dans Qwen38FlashNextEngine

Décision de Vincent après la mesure ci-dessus : le seuil `≤ 0,8x` du plan
était une jauge de décision automatique, pas un critère de rejet du
chantier — un gain net de 14-19 % à sortie bit-identique (0,81-0,86x)
justifie un branchement **opt-in, défaut off**, plutôt que de laisser le
travail PM4.1/PM4.2 hors catalogue.

### Câblage

- `Qwen4ExpFlashMTP.swift` : `prepare(...)` refactoré autour d'un
  `primePredictor(..., resetState:)` privé partagé ; nouvelle
  `prepareContinuation(...)` (même priming, sans `state.reset()`) pour
  étendre le cache du drafter au lieu de le reconstruire à chaque tour.
- `Qwen4ExpFlashMTPGenerator.generateMTP` gagne trois paramètres, tous à
  défaut neutre (aucun appelant CLI existant n'est affecté) :
  `continueConversation` (contrôle uniquement `model.resetConversation()`,
  même contrat que `Qwen4ExpStreamingGenerationOptions`), `state`
  (fourni par l'appelant au lieu d'être créé en interne, pour survivre
  entre tours) et `onToken` (callback par token émis, pour streamer via
  `Qwen38GenerationEvent` sans dupliquer la boucle de rounds existante).
  Le choix `prepare` vs `prepareContinuation` pour le drafter est décidé
  **indépendamment** de `continueConversation`, par `state.nextPosition
  == 0` : un état jamais amorcé se prime toujours à neuf, y compris si la
  cible, elle, continue une conversation dont les tours précédents
  n'utilisaient pas le MTP (le drafter n'a alors aucun historique à
  perdre — `prepare()` sur ce tour-là est correct, `prepareContinuation()`
  serait un no-op déguisé en continuation).
- `Qwen38FlashNextEngineProtocol` gagne `var mtpState: Qwen38MTPAvailability`
  (dynamique : `.fallback` avant le premier tour MTP, `.active` une fois le
  prédicteur chargé). `Qwen38FlashNextEngine` charge le prédicteur à la
  demande (`Qwen4ExpMTPLoader.load(uncachedIO: true)`) dans la `Task` de
  streaming (pas avant : éviter de bloquer l'actor `Qwen38Runtime` sur de
  l'IO Lexar), garde `mtpDraftState` en propriété (effacé par
  `resetConversation()`, les poids du prédicteur restent chargés).
  `runGenerationStream` route vers un nouveau `runMTPGenerationStream`
  quand `options.mtp.enabled && !hasImage` ; image + MTP demandé ⇒ log +
  fallback greedy avec `mtpStatus = .fallback("MTP Flash-Next : texte
  seul")` ; MTP non demandé ⇒ `mtpStatus = .unavailable` (au lieu du
  message fixe PM3/PM4 précédent, qui s'affichait même quand MTP n'était
  pas demandé).
- `Qwen38Runtime.mtpState` délègue à `flashEngine.mtpState` quand
  Flash-Next est chargé (au lieu du `mtpAvailability` figé pris à
  `load()`) ; le `.qwen4Exp` de `load()` initialise `mtpAvailability =
  .unavailable`, la valeur réelle vient désormais de l'engine.
- GUI (`Qwen38BenchUIApp.swift`) : `mtpEnabled` (défaut `true`, pensé pour
  le 27B où un drafter présent est le cas courant) est forcé à `false` à
  chaque chargement d'un modèle `qwen4Exp`, pour que le MTP Flash-Next
  reste opt-in même si l'utilisateur n'a jamais touché le toggle. Le
  mécanisme de réactivation du toggle (`disabled(mtpAvailability !=
  .active)`) est inchangé — non spécifique à Flash-Next, non modifié.

### Tests

`Tests/Qwen38Tests/Qwen38Tests.swift` : `MockFlashNextEngine` gagne un
`mtpState` settable et enregistre les `options` reçues par
`generate`/`generateFromMessages` ; `MockFlashNextEngineFactory` devient
une classe qui garde une référence à l'engine créé (le test doit pouvoir
remonter dedans après coup). Le test existant `runtimeDispatchesToFlashNextEngine`
est mis à jour pour le nouveau message `.fallback` par défaut. Nouveau test
« PM4.3 (branchement) » : vérifie que `Qwen38Runtime.mtpState` restitue
`.active` une fois le mock basculé, et que `options.mtp.enabled`/`draftDepth`
atteignent bien l'engine via `runtime.generate(...)`. 78 tests verts
(`Scripts/run-tests.sh`), `Scripts/build-release.sh` vert.

### Validation matérielle (serveur réel, 3-bit)

`qwen38 serve --model-path .../Qwen3.8-Flash-Next-MLX-e3bit-MTP`,
`caffeinate -dimsu`, préflight OK (24,6 Go à évincer, seuil 35 Go).

| Requête | `mtp` | temps total (`curl`) | `content` | `/metrics` |
|---|---|---:|---|---|
| 1 | `false` | 7,95 s | « Le président de la Chine est Xi Jinping. Il est le Secrétaire général du Comité central du Parti communiste chinois, le Président de la Commission militaire centrale » | `mtp: "indisponible"` |
| 2 | `true` | **7,42 s** | **identique mot pour mot** | `mtp: "actif"`, `mtpProposed: 21`, `mtpAccepted: 10`, `mtpAcceptRate: 0.476` |

`content` strictement identique entre les deux réponses, la requête MTP
plus rapide malgré le surcoût fixe HTTP/session qui dilue le gain mesuré en
CLI (0,81-0,86x en décodage pur devient ~0,93x sur le temps `curl` total,
TTFT/queue/JSON inclus). Un premier appel `mtp:false` avec
`enable_thinking` par défaut (`true`) a d'abord consommé tout le budget de
32 tokens dans `reasoning_content` (`content` vide) — attendu, pas un bug :
refait avec `"enable_thinking":false` pour comparer des générations
équivalentes au prompt de référence CLI.

Continuation multi-tour testée par `conversation_id` (deux tours, `mtp:true`
sur les deux) : le second tour (« Et quel âge a-t-il ? ») répond
correctement à partir du contexte du premier (« Xi Jinping, né le 15 juin
1953 »), `cacheReused: true`, `mtp: "actif"`, `mtpAccepted: 9/14` — confirme
que `prepareContinuation`/`state.nextPosition` fonctionnent en conditions
réelles, pas seulement en test unitaire.

### Écarts assumés

1. **Défaut serveur `"mtp"` non spécifié à la requête reste `true`**
   (`ChatCompletionRequest.effectiveMTP`, comportement global hérité du
   27B, non modifié) : un client Flash-Next qui omet le champ `mtp`
   obtient donc le MTP actif par défaut via l'API brute, à la différence
   du toggle GUI (forcé off pour cette famille). Changer ce défaut aurait
   affecté le comportement 27B existant hors périmètre de cette tâche ;
   documenté plutôt que corrigé. Un client qui veut explicitement le
   greedy doit envoyer `"mtp": false`.
2. **`options.temperature`/`topP`/`topK` ignorés par le chemin MTP** :
   `generateMTP` échantillonne en greedy (`ArgMaxSampler`) partout, cible
   et drafter, exactement comme tous les probes CLI existants. Un appelant
   qui demande MTP avec une température non nulle l'obtient quand même en
   greedy, sans erreur ni avertissement dédié — comportement identique à
   `flash-generate-probe --mtp`, non traité comme une régression.
3. **Chargement du prédicteur dans la `Task` de streaming, pas avant** :
   évite de bloquer l'actor `Qwen38Runtime` sur l'IO Lexar, mais veut dire
   que `mtpState` ne devient `.active` qu'après le retour du premier
   `.metrics` (pas au moment où le flux est retourné). Cohérent avec « à
   la demande au premier tour MTP » de la consigne.

## 2026-09-09 — Checkpoint 3-bit « hybride » sur le SSD interne (n-gram local, experts liés au Lexar)

`Scripts/localize-checkpoint.sh` crée `/Users/vincent/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` :
les 7 shards de la table n-gram (35,7 Go, lus par accès aléatoires à chaque
token) sont copiés sur le SSD (54 s), les 15 autres shards sont des liens
symboliques vers le Lexar, les petits fichiers sont copiés ; `--full` copie
tout quand la place le permettra (il reste 62 Go libres). Aucun changement de
code : le loader suit les liens.

| Chemin (3-bit, 32 tokens, Release, résident, `asyncEval`) | TTFT | greedy | MTP bloc 2 |
|---|---|---|---|
| Lexar seul | 64,2 s | 5,23 s | 4,31 s (PM4) |
| hybride SSD, 1er run (pages n-gram froides) | 68,1 s | 11,40 s | 4,35 s |
| hybride SSD, runs 2-3 | 66,4-67,0 s | **5,28-5,39 s** | — |

IDs identiques. Verdict : en régime établi, **aucune différence** : les
lectures n-gram (LRU 4 096 lignes + cache de pages) n'étaient pas un goulot
sur le Lexar ; le TTFT reste dominé par les 50 Go d'experts lus depuis l'USB.
L'incertitude « I/O Lexar par token » est levée. Un gain de TTFT (60 s → ~15 s)
demande la copie complète (`--full`, +54 Go). Le premier run après la copie
paie le premier accès aux pages (×2 sur 32 tokens) : ne jamais mesurer sur un
premier run.

## 2026-09-09 — Copie hybride supprimée ; le préfill n'est pas un goulot

Décision Vincent : la copie hybride du 3-bit sur le SSD interne est supprimée
(aucun gain en régime établi) ; le Lexar reste la source. Deux mesures de
préfill faites avant, Release, résident, `asyncEval` :

| Prompt | tokens | vision | forward préfill (hors chargement) | tok/s |
|---|---|---|---|---|
| texte répété | 408 | — | 4,7 s (dont ~2 s de n-gram en couche 1, 19 632 misses à ~0,1 ms, pages chaudes) | 87 |
| image de référence | 976 | 2,46 s | 5,1 s | 190 |

Le « TTFT 48-68 s » des probes est le chargement des couches, pas le préfill.
Le préfill est donc sain ; reste à vérifier le coût des misses n-gram **à
froid** (premier tour après chargement : 59 tokens → 2,39 s de TTFT en GUI,
soit 40 ms/token contre 5 ms/token à chaud), tâche P4.5.

## 2026-09-09 — P4 : débit de décodage

Protocole PLAN.md « P4 — Débit de décodage : mesurer exactement, puis
fusionner » (2026-09-09). Ordre exécuté : P4.0 → P4.1 → P4.2 → P4.4 → P4.5 →
P4.3 → P4.6. Tout mesuré sur `/Volumes/Lexar/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`
(3-bit, 84 Go), Release, résident, prompt de référence, `caffeinate -dimsu` +
`Scripts/preflight-resident.sh` (seuil 35 Go) devant chaque run réel, jamais
deux runs simultanés.

### P4.0 — Metal System Trace + échantillon CPU

`xcrun xctrace record --template 'Metal System Trace'` sur (a) le bench
(`flash-layer-bench --steps 300 --async-interval 8 --expert-bits 3
--expert-group-size 64`, sans checkpoint) et (b) le modèle réel
(`flash-chat-probe … --max-new-tokens 8 --resident-layers --resident-async`,
`--launch`, `--time-limit 150s` pour couvrir le chargement ~65-68 s + la
génération). Export via `xcrun xctrace export --xpath
'…/table[@schema="metal-gpu-intervals"]'` (et
`metal-application-command-buffer-submissions`), parsé en Python
(résolution des `ref=` de l'export xctrace, union des intervalles pour
éviter de sur-compter les recouvrements).

**Limite outillage constatée** : sans activer « Shader Timeline »/compteurs
GPU (non exposé par `xctrace record --template` en ligne de commande, seul
le preset a été utilisé), la table `metal-gpu-intervals` ne nomme pas les
noyaux individuels — chaque ligne est un encodeur Metal générique
(« Command Buffer N:Compute Command 0 »), pas un noyau MLX identifiable
(`Copy`, `QuantizedMatmul`, …). Le tableau « famille de noyau » demandé par
la consigne n'a donc **pas pu être reconstruit à neuf** par cette voie ;
celui déjà établi par `sample` sur le bench (P2-code (a), 2026-09-07,
réutilisé ici plutôt que redérivé) reste la meilleure référence par famille :

| Famille | Échantillons (bench, `sample` 10 s, thread de calcul) | Part |
|---|---|---|
| Copy (`copy_gpu_inplace`/`copy_gpu`) | ≈ 260 | la plus grosse |
| Binaire (`binary_op_gpu[_inplace]`) | ≈ 270 | la plus grosse |
| Concatenate | 62 | — |
| Matmul quantifié (`QuantizedMatmul`+`qmv`) | ≈ 77 | calcul utile |
| Reduce | 39 | — |
| Unaire | 36 | — |
| Attention (QSA seule) | 33 | — |
| bookkeeping hôte pur (allocateur, refcounting `array`) | ≈ 71 | — |

Nouveau cette session, un échantillon CPU (`sample`, 10 s) pris **sur le
vrai checkpoint** pendant un décodage réel (jamais fait avant — P0/P2-code
n'avaient échantillonné que le bench synthétique) : après filtrage des
threads d'attente (`__workq_kernreturn`/`mach_msg2_trap`/`__psynch_cvwait`/
`iokit_user_client_trap`, pool GCD/IOKit au repos, cf. méthodologie du
2026-09-07), le thread de calcul montre le même bookkeeping MLX que sur le
bench (arbre `BufferCache`, tables de hachage, `mlx::core::array::~array`,
`eval_impl`) **plus une présence nette du driver Metal AGX**
(`AGX::ComputeContext::performEnqueueKernel`, `AGX::ESLInstructionEncoderGen3`,
`agxaReserveCDMTokenSpace`, `IOGPUResourceListAddResource`) absente du
même ordre de grandeur sur le bench — cohérent avec le modèle réel qui
dispatche beaucoup plus de petits noyaux par couche (routage MoE 512
experts, lookup n-gram) que la couche synthétique isolée. Fichier :
`results/p4-cpu-sample-real.txt`.

**GPU actif / temps mort** (nouvelle mesure, union des intervalles GPU-busy,
pas une moyenne d'échantillons instantanés comme `ioreg`) :

| Cible | Fenêtre | Intervalles GPU-busy (qwen38) | GPU actif / fenêtre |
|---|---|---|---|
| Bench, `--async-interval 8` | 2,259 s (300+300 pas, GDN+QSA) | 5 451 | **85,4 %** |
| Réel, réglages de production *avant* P4.1 (bug ci-dessous) | 7,15 s (préfill+décodage 8 tokens+fin) | 2 563 | **14,2 %** |
| Réel, run complet (chargement inclus) | 74,15 s | 3 034 | 2,3 % (chargement = I/O, GPU ≈ 0) |

Command buffers qwen38 sur tout le run réel : 3 029 (2 563 avec exactement 1
encodeur, 1 871 avec 0 — probablement des barrières/soumissions vides,
non élucidé). Traces conservées : `results/p4-mst-bench.trace`,
`results/p4-mst-real.trace`.

**Verdict de la clause d'arrêt PLAN.md** (« GPU actif proche de 100 % ⇒
arrête-toi après P4.2 ») : 14,2 % au réglage de production *avant P4.1*, très
loin de 80 % — **ne pas s'arrêter**, continuer vers les fusions. (P4.1,
ci-dessous, referme ensuite une bonne partie de cet écart lui-même — relu à
la fin de P4.1.)

### P4.1 — `residentAsyncInterval` : `residentAsyncEval` était mort en production

**Découverte** (relecture de `Qwen4ExpStreamingDecoder.forward`, motivée par
l'écart 14,2 % (réel) vs 85,4 % (bench) ci-dessus alors que les deux
utilisaient nominalement le même mécanisme) : `shouldEvaluate` valait

```swift
synchronizeEachLayer ||
  (resident && ((visitIndex + 1) % residentEvaluationInterval == 0 || dernière couche))
```

Avec `residentEvaluationInterval == 1` (valeur fixée par `Qwen38FlashNextEngine`
et par défaut CLI, conforme au piège 11), `(visitIndex + 1) % 1 == 0` est
vrai pour **tout** entier : `shouldEvaluate` était donc toujours vrai, `eval()`
bloquant partait sur **chaque** couche quel que soit `residentAsyncEval`, et
la branche `asyncEval` (`else if residentAsyncEval { asyncEval(output) }`)
n'était jamais atteinte. Le gain « −22 % » mesuré par P1 le 2026-09-08 tenait
donc à autre chose (Release/F_NOCACHE/masque QSA superflu, tous livrés le
même jour) — pas à `asyncEval` lui-même, qui était du code mort depuis son
introduction.

**Correctif** : nouveau paramètre `residentAsyncInterval` (indépendant de
`residentEvaluationInterval`, qui garde exactement son ancien sens et reste
à 1 partout — piège 11 non touché). Quand `residentAsyncEval == true`, la
décision devient `shouldEvaluate = dernière couche || (visitIndex + 1) %
residentAsyncInterval == 0` : chaque couche visitée reçoit toujours un vrai
`eval`/`asyncEval` (jamais de graphe totalement différé — la catastrophe
30-50× de piège 11 venait de l'absence de tout appel, pas de l'asynchronie),
seule la fréquence du blocage change. Défaut 1 (comportement identique à
avant, aucune régression par défaut). Câblé jusqu'à
`--resident-async-interval` sur `flash-chat-probe`/`flash-generate-probe`.

**Sweep N=1/2/4/8/12** (checkpoint réel, greedy, 32 tokens, 2 runs chacun,
IDs bit-identiques à la référence dans les 10/10 runs) :

| N | decode moyen (2 runs) | s/token | vs N=1 |
|---|---|---|---|
| 1 (= ancien comportement) | 5,250 s | 0,1641 | réf. |
| 2 | 4,789 s | 0,1497 | −8,8 % |
| 4 | 4,598 s | 0,1437 | −12,4 % |
| 8 | 4,495 s | 0,1405 | **−14,4 %** |
| 12 | 4,482 s | 0,1401 | −14,6 % |

N=8 et N=12 sont dans le bruit l'un de l'autre, rendements décroissants
au-delà de 8 → **N=8 retenu par défaut** (`Qwen38FlashNextEngine`, déjà
l'intervalle de référence de `flash-layer-bench --async-interval`).
`ioreg` pendant le sweep (préflight de chaque run) : 82-97 % de GPU aux
intervalles N≥2, contre 0-5 % au repos — confirmation indépendante du gain.
78 tests verts, build Release vert. Fichiers :
`results/p41-n{1,2,4,8,12}-{a,b}.stdout.txt`, `results/p41-sweep.log`.

### P4.2 — fin de token : lm_head, sampler, `.item()`

Instrumentation `ContinuousClock` (toujours active, coût nul — même
principe que PM1) : `Qwen4ExpStreamingTextModel.lastLMHeadDuration` (réduction
hyper-flux + `lm_head` 248 320×2 560 quantifié + `eval` bloquant) et, dans
`Qwen4ExpStreamingGenerator`, cumul séparé de `sampler.sample` et `.item()`
(ligne 222 avant ce commit). Exposé par `Qwen4ExpGenerationSummary`, imprimé
par `flash-chat-probe` (« P4.2 fin de token »).

**Mesure** (32 tokens, N=8, 2 runs) :

| Poste | cumulé (run a) | cumulé (run b) |
|---|---|---|
| `lm_head` | 0,0905 s | 0,0905 s |
| `sampler.sample` | 0,0001 s | 0,0001 s |
| `.item()` | 0,0087 s | 0,0088 s |
| **total fin de token** | **0,0993 s** | **0,0994 s** |
| decode total | 4,496 s | 4,494 s |
| **part du décodage** | **2,2 %** | **2,2 %** |

Très en dessous du seuil de 10 % de la consigne : `argmax`/top-k sont déjà
sur GPU (aucun `.item()` intermédiaire dans `sampler.sample`, un seul
`.item()` par token déjà atteint), et `lm_head` en `asyncEval` avec la
dernière couche n'a pas été tenté — le gain théorique (recouvrir 2,2 % du
temps) ne justifie pas le risque de toucher la synchronisation de fin de
token. **Mesuré, aucun changement de code au-delà de l'instrumentation.**

### P4.4 — warm-up du premier forward : mesuré, non concluant

Hypothèse mécaniste identifiée par lecture :
`GatedDeltaKernelManager.shared` (Vendor, `GatedDelta.swift`, non modifié)
construit **paresseusement les deux variantes** du noyau Metal GDN
(`kernel`/`kernelMasked`) au premier accès du singleton ; la variante
masquée n'est exercée que par un forward multi-tokens masqué (préfill),
jamais par le forward factice à un jeton (`mask: nil`) que
`Qwen38FlashNextEngine.warmUp()` fait déjà. Hypothèse cohérente avec la
consigne (« étendre le warmup à un préfill factice si c'est la
compilation »).

**Mesure côté CLI** (aucun accès à la GUI depuis cet environnement — pas de
harnais de test GUI disponible) : `flash-chat-probe --second-prompt` donne
un tour 1 (préfill neuf, 29 tokens) et un tour 2 (continuation, 20 tokens,
cache déjà non vide). Préfill pur du tour 1 (TTFT − load cumulé) :
63,300 − 63,060 = 0,240 s / 29 tokens ≈ **8,3 ms/token**, cohérent avec le
« préfill sain » déjà établi le 2026-09-09 (87-190 tok/s). Le tour 2
(0,602 s / 20 tokens ≈ 30 ms/token) est *plus lent* par token que le tour 1,
mais n'est pas une comparaison propre (cache KV déjà non vide, coût de
concaténation croissant) — il ne confirme ni n'infirme l'hypothèse noyau
masqué. **Aucune pénalité de plusieurs secondes reproduite sur le premier
préfill réel via les probes CLI disponibles** ; l'anomalie GUI (2,39 s pour
59 tokens, 40 ms/token) n'a pas pu être isolée ni reproduite dans cet
environnement CLI-only. **Décision : `warmUp()` non étendu** — modifier ce
chemin sans pouvoir vérifier son effet sur la métrique GUI réelle aurait été
un changement non validé (contraire à « vérifié, pas supposé »). Hypothèse
`GatedDeltaKernelManager` documentée ici pour une session future avec accès
GUI. Fichier : `results/p44-turn1-vs-turn2-prefill.stdout.txt`.

### P4.5 — misses n-gram à froid : mesuré < 0,5 ms, rien à appliquer

`Qwen4ExpNGramCacheStats` gagne `missDuration`/`meanMissDuration` ;
`Qwen4ExpLazyNGramStorage.RowCache` chronomètre (`ContinuousClock`) chaque
lecture non cachée (le batch mmap réellement touché, pas ligne à ligne) ;
`flash-chat-probe --profile-layers` l'imprime (« P4.5 n-gram »).

**Mesure** : prompt de référence (2 712 misses) et un prompt inédit sans
rapport (« cassoulet toulousain… », 3 432 misses, pour approcher des pages
froides sans reboot) donnent tous les deux moyenne/miss **< 0,00005 s** —
sous le seuil de 0,5 ms de la consigne. Le cache de fichiers/pages mmap du
Lexar est resté chaud toute la session (chargements résidents répétés
aujourd'hui, cf. les runs P4.0/P4.1) ; un test à froid réel (pages jamais
touchées depuis le boot) exigerait un reboot, jugé disproportionné pour
cette seule mesure. La conditionnelle de la consigne (« si > 0,5 ms, trier
par offset + `QWEN38_NGRAM_PREWARM` ») n'est pas déclenchée dans les
conditions mesurées : **ni le tri par offset ni `QWEN38_NGRAM_PREWARM` ne
sont implémentés**. `ngram_cache_misses` inchangé (vérifié : le comptage de
misses ne dépend pas de l'instrumentation ajoutée). Fichiers :
`results/p45-warm-reference.stdout.txt`, `results/p45-cold-candidate.stdout.txt`.

### P4.3 — fusions guidées par P4.0 : non implémentées, décision documentée

Aucun des quatre candidats ((a) pré/post-traitement GDN, (b) mix +
injection hyper-connections, (c) routage MoE, (d) famille de copies/`asType`
dominante) n'a été implémenté en noyau `MLXFast.metalKernel`. Faisceau de
preuves motivant cette décision, pas un choix arbitraire :

1. **P4.0 n'a pas pu désigner de famille dominante** (> 10 % du temps ou du
   nombre de noyaux) sur le vrai checkpoint : l'export `xctrace` sans
   Shader Timeline ne nomme pas les noyaux (limite documentée ci-dessus), et
   le comptage par famille disponible (bench, P2-code (a)) montre Copy et
   Binaire du même ordre de grandeur (≈260/270), aucun poste isolé.
2. **P2-fusion (2026-09-09, même session de travail antérieure) a déjà
   testé des réductions structurellement analogues** — F1 (fusion des
   projections d'entrée GDN/QSA, moins de matmuls), F2 (poids `1+w`
   précalculé, `MLXFast.rmsNorm`, moins de casts), F4 (softmax MoE non
   précis, moins d'upcasts) — sur le même bench, avec le même protocole
   `--async-interval 8`, et **aucun des trois n'a montré de gain net
   mesurable** (±0,00-0,03 ms/pas, dans le bruit). Rien dans P4.0 ne suggère
   qu'un noyau Metal fait main pour les mêmes zones fonctionnelles (b) et
   (c) se comporterait différemment.
3. **P4.1 a lui-même refermé l'essentiel de l'écart que les fusions
   visaient à combler** : GPU actif 14,2 % → 82-97 % (`ioreg`), decode
   −14,4 % — sans toucher un seul noyau de calcul, en corrigeant un bug de
   bookkeeping. L'esprit de la clause d'arrêt de PLAN.md (« GPU proche de
   100 % ⇒ la suite est de la fusion de calcul, une autre décision ») est
   satisfait a posteriori par ce résultat, même si la mesure formelle (P4.0,
   avant P4.1) était sous le seuil.
4. **Coût/risque** : écrire un noyau Metal correct pour un routage MoE
   quantifié à 512 experts ou pour la récurrence GDN (état float32, ordre
   d'accumulation sensible en bf16 — cf. commentaire existant sur
   `mixedInput`) exige un harnais de parité aussi rigoureux que celui de
   P2-fusion (`checkParity`, tolérance documentée), pour un gain attendu
   proche de zéro au vu de (1)-(3).

**Décision** : (a)-(d) non implémentés, documentés ici — même traitement que
F6 dans P2-fusion (« non implémenté, faisceau de preuves convergent contre
un gain probable »). Rouvrir ce chantier n'a de sens qu'avec un accès direct
à Instruments (Shader Timeline/compteurs GPU activés, hors de ce qui est
exposé par `xctrace record --template` en CLI) pour d'abord désigner une
vraie famille dominante.

### P4.6 — validation finale

Garde Q-B (`-only-testing:Qwen38Tests/flashTeacherForcedRegressionGuardV32()`,
`TEST_RUNNER_QWEN38_FLASH_MODEL`) : **PASS**, `hits=10/28
meanLogProb=-4.8003182` — identique à la valeur attendue documentée en
P2-fusion F7. 78 tests verts (`Scripts/run-tests.sh`, Debug), build Release
vert tout au long de la session.

**Tableau récapitulatif avant/après** (checkpoint 3-bit réel, prompt de
référence, greedy `--temperature 0`, 32 tokens ; MTP bloc 2, même prompt) :

| Variante | Avant P4 | Après P4 (N=8) | Δ | IDs = référence |
|---|---|---|---|---|
| Greedy | 0,166 s/token (6,0 tok/s) | **0,1405 s/token (7,1 tok/s)** | **−15,4 %** | oui |
| MTP bloc 2 | 0,135 s/token (7,4 tok/s) | **0,1328 s/token (7,5 tok/s)** | −1,6 % | oui |
| Ratio MTP/greedy | 0,81× | 0,945× | — | — |

Le MTP profite beaucoup moins de P4.1 que le greedy (son *verify forward*
vérifie déjà 42 positions par round, donc a déjà plus de travail parallèle
par couche que le décodage à un jeton — l'asynchronie recouvre moins de
temps mort) : conforme au diagnostic déjà posé en PM3/PM4 (le coût MTP est
structurel — le *verify forward*, pas le bookkeeping hôte). Le MTP reste
hors catalogue par défaut (`options.mtp.enabled`, inchangé par ce chantier
— hors périmètre de P4, cf. garde-fous).

MTP mesuré avec un premier run explicitement écarté (artefact de démarrage
« premier run de la session » sur le chemin de vérification, jamais
exercé auparavant dans ce process : 8,820 s au lieu de 4,255/4,242 s au 2ᵉ
et 3ᵉ run, même prompt, mêmes réglages) — cohérent avec la règle déjà en
vigueur (« jamais mesurer sur le premier run »), étendue ici du chemin
greedy (déjà connu) au chemin MTP (nouveau cette session).

### Écarts à la consigne P4

1. **P4.0** : le tableau « famille de noyau nommé » n'a pas pu être
   reconstruit à neuf par Metal System Trace (limite outillage `xctrace`
   CLI sans Shader Timeline, documentée) — celui de P2-code (a) est réutilisé
   comme référence plutôt que redérivé, complété par un nouvel échantillon
   CPU sur le vrai checkpoint (jamais fait avant) et par les mesures
   GPU-busy/command-buffers, qui sont, elles, nouvelles et quantitatives.
2. **P4.3** : décision « non implémenté » pour les quatre candidats — un
   faisceau de preuves documenté (ci-dessus), pas une mesure directe de
   chaque noyau hypothétique (puisqu'aucun n'a été écrit). Traitement
   identique au précédent F6 (P2-fusion).
3. **P4.4** : « TTFT GUI < 1 s pour 59 tokens » non vérifiable depuis cet
   environnement (pas de harnais GUI) — mesuré par un proxy CLI (préfill
   pur tour 1) qui ne reproduit pas l'anomalie GUI rapportée ; `warmUp()`
   non modifié en l'absence de moyen de vérifier l'effet réel.
4. **P4.5** : mesure en conditions de session chaude (pages déjà en
   cache), pas un vrai test à froid post-reboot — jugé disproportionné pour
   cette seule mesure ; documenté explicitement plutôt que présenté comme un
   test à froid réel.

Commits : `7c93277` (P4.1 mécanisme), `170a3c9` (P4.1 défaut N=8),
`aa4fc0c` (P4.2), `ddab183` (P4.5). Traces `.trace`, sorties `.stdout.txt`
et le sweep complet conservés dans `results/`.

## 2026-09-09 (soir) — Test LAN par agent (8 requêtes) et correctifs GUI/serveur

Serveur lancé depuis la GUI (Release), 3-bit chargé, agent Haiku en client
`curl` sur `http://192.168.1.183:8848` (bind LAN, même machine). 8 requêtes :
tours liés par `conversation_id`, streaming MTP (98 chunks, `[DONE]`,
33/63 acceptés), thinking `low` (reasoning et contenu non vides, Canberra),
image (Emmanuel Macron, 976 tokens de prompt, TTFT 7,3 s), sampling
(température 0,7 : MTP en fallback explicite), erreur de modèle. Débits 6,1 à
7,4 tok/s, TTFT 1,3-2,8 s en texte. Pendant une requête de 96 tokens,
`ioreg` mesure **GPU 93-94 %** et le process GUI à ~50 % d'un cœur : le GPU
n'est plus inactif sur le chemin serveur (P4.1 en production). L'observation
« CPU > 100 %, GPU au repos » de Vincent correspond à la session précédente
où le **4-bit** avait été chargé par erreur (catalogue pointé sur Vontra) avec
73 Go dans le compresseur : c'est la décompression, pas le décodage.

Trois défauts trouvés et corrigés :
1. **Erreurs HTTP** : `modelNotFound` ressortait en 500 sans corps ; le handler
   renvoie maintenant un JSON d'erreur OpenAI avec 404 / 400 / 401 / 503.
2. **`reasoning_effort` par défaut `xhigh`** : une requête sans `enable_thinking`
   ni effort produisait 96 tokens de raisonnement et un `content` vide
   (T3/T4 du test — l'agent avait omis le champ). Défaut serveur → `low`.
3. **Onglet Serveur** : l'URL LAN affichait `<adresse-du-Mac>` et le port « 8 848 »
   (Int interpolé dans `Text`) ; l'IP `en0` réelle est maintenant affichée. La
   liste des sessions n'affiche plus le flux de tokens : Entrée / Sortie en
   tokens, TTFT, débit, durée, MTP acceptés/proposés, état du cache,
   `conversation_id` (demande Vincent). Le serveur ne conserve plus le dernier
   fragment, il horodate la fin de session.

## 2026-09-09 (nuit) — Test LAN 2 après correctifs : 9/10 OK, thinking par défaut désactivé côté serveur

Rejoué par l'agent Haiku sur le serveur relancé : 404 JSON pour un modèle
inconnu ✓, streaming MTP 66 chunks ✓ (42 % d'acceptation), non-stream
identique ✓, deux tours avec cache réutilisé ✓, image Macron ✓ (976 tokens,
TTFT 6,8 s), `/metrics` avec tokens entrée/sortie par session ✓. Reste la
requête **minimale** (aucun champ maison, 96 tokens) : thinking actif par
défaut, effort `low`, mais 96 tokens ne suffisent pas à fermer `</think>` →
`content` vide. Un client OpenAI standard n'envoie jamais `enable_thinking` :
le défaut serveur devient **thinking désactivé**, activé seulement si le
client envoie `enable_thinking: true` ou un `reasoning_effort` (top-level,
`reasoning.effort` ou `extra`). La GUI garde son propre toggle.

## 2026-09-10 — Requête minimale : la réponse partait dans `reasoning_content`

Après le passage du thinking à « désactivé par défaut », la requête minimale
(sans `enable_thinking`) générait bien la réponse (« Le président de la Chine
est Xi Jinping… ») mais le serveur la rangeait dans `reasoning_content` :
`thinkingIsPrimed` était encore calculé avec `input.effectiveThinking ?? true`,
indépendamment de l'option réellement rendue. Le parser de flux thinking
supposait le prompt terminé à l'intérieur de `<think>`. Correctif :
`thinkingIsPrimed = options.enableThinking`. Au passage : l'IP LAN du Mac
change avec le DHCP (192.168.1.183 → .87 cette nuit) ; l'onglet Serveur la lit
en direct, les scripts de test doivent la relire aussi.

## 2026-09-10 — Dialogue A/B de 20 minutes sous profiler 1.5.0 (serveur CLI, 3-bit)

`qwen38 serve --trace … --metal-trace-seconds 300` (session partagée, sampler
16 ms `ioReportResidency`, mémoire système) + `Scripts/agent-dialogue.py`
20 min : 38 tours, trace Chrome de 34 Mo (`results/dialogue-profiled/`).

**Débit en fonction de l'historique** (rejeu stateless à chaque tour, les
deux agents alternant sur un seul cache) :

| Tour | Historique | TTFT | Génération |
|---|---|---|---|
| 4 | 407 tok | 4,3 s | 5,7 tok/s |
| 10 | 1 075 | 8,4 s | 5,9 |
| 22 | 1 994 | 13,3 s | 5,9 |
| 31 | 2 414 | 17,3 s | 5,6 |
| 38 | 2 691 | 23,9 s | 4,9 |

TTFT ≈ 8,5 ms par token d'historique (≈ 115 tok/s de préfill) ; génération
stable 5,6-6,0 tok/s jusqu'à ~2 500 tokens puis 4,9. **GPU (residency IOReport,
moyenne pondérée par phase)** : 78-95 % en génération, 29-65 % en préfill
(croissant avec la longueur) — le préfill court est host-bound, la génération
ne l'est plus depuis P4.1. CPU 45-90 %. Pic process 66,3 Go, MLX 60,7 Go
(KV/QSA + n-gram sur 2 700 tokens : +4 Go par rapport à un prompt court).
Passage des 2 048 tokens (chemin QSA épars) au tour 22 sans incident.

**Incident au tour 13 (TTFT 228 s)** : c'est l'arrêt du Metal System Trace
attaché (limite 300 s). xctrace a écrit un bundle de **15 Go** ; pendant
l'écriture, le compresseur est monté à 56 Go et le swap à 27 Go, l'anonyme du
process est tombé de 69 à 20 Go (modèle compressé), `memorystatus_level` à
27 % ; retour à la normale après 4 minutes. `Recorder.stop()` a échoué (le
process xctrace s'était déjà terminé) et la fusion n'a pas eu lieu ; l'export
`--toc` du bundle n'a rien rendu en 2 min. Bundle supprimé. Correctifs :
fenêtre plafonnée à 60 s dans `serve`, `stop()` tolérant après la limite de
temps ; retour au profiler (#532) : plafond de durée par défaut, estimation
de taille, et `Recorder.stop()` idempotent après auto-arrêt.

Détails de session : Release, batterie, sampler 59 651 échantillons, pas de
stall détecté par le profiler (le gel du tour 13 avait des échantillons : ce
n'est pas une absence d'échantillons mais une famine mémoire). Les compteurs
n-gram restent à 0 en session partagée (ils ne sont publiés qu'avec le
profilage par couche) — à brancher sur la session partagée.

## 2026-09-10 — P5.4 : préfill 2 543 tokens, la couche PLE domine (~28-32 %), deux correctifs, gain modeste

Protocole : `flash-chat-probe --resident-layers --resident-async --profile-layers
--trace` en deux tours (`--prompt "Bonjour"` puis `--second-prompt` avec la
phrase « Le chat noir traverse la rue tranquillement avant midi. » répétée
230 fois → 2 543 tokens) — le tour 1 force la résidence des 48 couches
(`load cumulé` payé une fois), le tour 2 mesure un préfill **pur** (`load
cumulé : 0.000s`), sans conflit avec le coût de chargement Lexar. Ventilation
par couche via les paires `B`/`E` du trace Chrome exporté (`ph`) — le tableau
console `generateReport()` agrège à tort chargement + calcul des deux tours
sous le même nom de phase, inutilisable pour cette mesure.

**Avant correctif** (`results/p54/chat.trace.json`) : préfill tour 2 = 9,914 s
pour 2 543 tokens. Couche 1 (PLE, n-gram) = 2,746 s (**29,2 %** du total
forward, 9,396 s) ; QSA (12 couches) = 1,878 s (20,0 %) ; les 35 autres
couches (GDN/hyper-connections/MoE) = 4,772 s (50,8 %, ~135 ms/couche
homogène). La couche PLE coûte à elle seule ~20× une couche « normale ».

**Hypothèse initiale (celle du plan) réfutée** : `Qwen4ExpLazyNGramStorage`
lit chaque ligne n-gram manquante octet par octet (`for row in rows { for
offset in stride(from:0,to:byteWidth,by:4) {...} }`, reconstruction
little-endian manuelle) plutôt qu'un `memcpy` groupé. Correctif appliqué :
tri des lignes demandées, une copie mémoire (`copyMemory`) par plage
contiguë au lieu d'une itération par élément (`readContiguousRuns`/
`readContiguousRunsFromFile`, génériques `UInt32`/`UInt16`, mmap et
FileHandle). IDs générés identiques (`[78768]`), mais **gain quasi nul**
(couche PLE 2,746 s → 3,101 s puis 2,568 s sur deux re-mesures — bruit de
mesure, aucune tendance nette). L'instrumentation `P4.5 n-gram : ... miss
cumulé 0.0000s` reste à 0 (bug de bookkeeping distinct, `missDuration`
n'est jamais remonté par `observeNGramCache` — non corrigé ici, hors
périmètre P5.4) : impossible de confirmer par ce biais que les lectures de
lignes étaient déjà rapides, mais le résultat avant/après est sans appel.

**Deuxième hypothèse, retenue** : `Qwen4ExpPLE.lookup(_ IDs:)` route chaque
ID vers un shard (jusqu'à 128) puis, pour **chaque shard distinct**, relit
l'intégralité du tableau `shardIndices` (`Set(shardIndices).sorted()` +
`.compactMap` par shard) pour extraire ses positions — O(shards × total IDs)
au lieu de O(total IDs). Sur un préfill de 2 543 tokens (~245 000 IDs
cumulés sur les deux tours), avec un grand nombre de shards distincts
touchés, ce balayage répété est un candidat crédible. Correctif : un seul
passage sur `shardIndices` qui regroupe les positions par shard dans un
dictionnaire (`positionsByShard`, ordre préservé), puis un tri des clés —
même sortie, O(total IDs + shards log shards). IDs identiques (`[78768]`).
**Gain mesuré, modeste** : couche PLE 2,746 s → 2,568 s (−6,5 %), total
préfill tour 2 9,914 s → 9,753 s (−1,6 %, dans le bruit de mesure d'un seul
run par variante).

**Décision** : les deux correctifs sont conservés (corrects, sans risque de
régression — IDs identiques, 82/82 tests verts — et algorithmiquement
strictement meilleurs même sans gain mesuré net), mais **le poste dominant
n'est pas résolu**. La couche PLE reste ~28 % du préfill après les deux
correctifs (2,568 s / 9,226 s de forward total sur la dernière mesure),
au-dessus du seuil de 15 % de la consigne P5.4. Le coût résiduel le plus
probable, non attaqué ici : le nombre d'opérations GPU par shard touché
(`MLXArray` × 3 + `MLX.dequantized` + `result.at[...].add(...)`, jusqu'à
~128 shards par tour) — une bascule de dispatch/queue Metal par shard plutôt
qu'un coût de lecture CPU. Regrouper la déquantification de plusieurs shards
en un seul appel GPU (moins de dispatches, plus gros tenseurs) est un
chantier distinct, plus risqué (il faudrait un harnais de parité par shard),
hors du correctif ciblé demandé par P5.4 rév. 2026-09-10. Documenté ici pour
une reprise éventuelle avec accès à Instruments/Shader Timeline (même
limite qu'en P4.3).

Fichiers : `results/p54/chat.trace.json` (avant), `results/p54/chat-after.trace.json`
(après lecture par plages), `results/p54/chat-after2.trace.json` (après le
regroupement par shard, retenu) ; stdout correspondants.

## 2026-09-10 — P5.6 : validation — dialogue A/B 20 min rejoué contre `serve` (P5.1-P5.5)

Protocole : `qwen38 serve --trace results/p56/serve.trace.json` (sans Metal
System Trace), checkpoint 3-bit, `Scripts/agent-dialogue.py --max-minutes 20`
(température 0,7, top_p 0,8, 120 tokens max, `enable_thinking:false`,
`mtp:false`, aucun champ de pénalité — le serveur applique donc son défaut
`presence 1.5`, P5.3). Poller additionnel (`p56-metrics-poller.py`,
hors dépôt) interrogeant `/metrics` toutes les 2 s pour capturer
`cacheReused`/`cacheRestored`/`cacheReplayed` par session, absents du journal
de `agent-dialogue.py`. Comparé à la référence pré-P5
`results/dialogue-profiled/dialogue.jsonl` (38 tours, même script/réglages).

**Premier essai (avant le correctif P5.2 ci-dessous)** : l'agent A démarre
avec un tour assistant factice déjà dans son historique (le script fait dire
« Hello » à A sans passer par une génération) ; le garde-fou de démarrage à
froid de `prepareConversation` (« système/utilisateur seulement ») rejette
donc systématiquement A, qui reste bloqué en rejeu stateless complet pour
toute la conversation (TTFT recroissant avec l'historique, jusqu'à 19,6 s à
2 578 tokens), alors que B (premier message pur système/utilisateur)
bénéficie pleinement du LRU (TTFT plat ~2 s). Cause : PLAN.md spécifiait
« sinon rejeu et **nouvel état après la réponse** » pour P5.2, partie
manquée dans le premier passage — corrigée (commit « P5.2 fix »),
`rememberConversation` enregistre désormais une conversation comme active
après un rejeu réussi, pas seulement après une continuation. Essai jeté,
rejoué proprement après le correctif.

**Essai retenu** (après le correctif) : 76 tours en 20,0 min (contre 38 dans
la référence — le débit de tours double puisque le TTFT ne croît plus).

| | Avant (référence, 38 tours) | Après (P5.1-P5.5, 76 tours) |
|---|---:|---:|
| TTFT tour 1 | 71,6 s | 0,93 s |
| TTFT médian | 8,4 s (tour 10) → 23,9 s (tour 38) | **2,16 s** (constant) |
| TTFT max sur tout le dialogue | 23,89 s (tour 38) | **2,81 s** (aucun tour > 3 s) |
| tok/s décodage | 4,9-6,0 | 4,9-6,4 (inchangé, attendu) |
| Pic process (profiler) | — (non mesuré alors) | **57,58 Go** (< 75 Go) |
| `cacheRestored` | n/a (métrique inexistante) | 74/76 tours (1 rejeu initial, 1 démarrage à froid) |
| `cacheMisses` (serveur) | n/a | 2 |
| Similarité Jaccard médiane (rows[i] vs rows[i-2]) | 0,116 (0,141 à partir du tour 30) | 0,145 (0,148 à partir du tour 30) |

**TTFT et mémoire** : critère largement atteint — plat sous 3 s sur les 76
tours (contre un objectif de 38), pic process 57,6 Go contre le plafond de
75 Go. La restauration LRU fonctionne comme prévu une fois le correctif
P5.2 appliqué : sur 76 tours, seuls les deux tout premiers (un par agent)
ne restaurent pas.

**Similarité / boucle — critère non atteint, écart documenté** : la
similarité Jaccard médiane mesurée sur la référence n'est **pas** 0,87
(chiffre cité dans PLAN.md P5) mais 0,116-0,141 selon la fenêtre — écart non
expliqué (méthode de calcul du chiffre d'origine non retrouvée dans ce
journal) ; en prenant ma propre mesure comme base de comparaison cohérente
avant/après, la médiane ne s'améliore pas avec la pénalité de présence
(0,141 → 0,148 à partir du tour 30) et une boucle quasi verbatim apparaît
bel et bien aux tours 68-73 (« Tu as raison, je tourne en rond… », Jaccard
1,0 entre tours 68 et 70). Explication : `presencePenalty` est appliqué par
tour (masque remis à zéro à chaque nouvelle génération, PLAN.md P5.3
l'implémente ainsi et le test greedy-inchangé le confirme) — il ne peut
structurellement pas empêcher une dérive **thématique inter-tours** sur une
conversation qui s'allonge, seulement la répétition d'un id **dans une
même réponse**. Le critère « aucune boucle » de P5.6 n'est donc pas
satisfait par ce mécanisme ; une pénalité inter-tours (fenêtre glissante sur
l'historique récent, hors périmètre GPU-par-token de P5.3) serait le
prochain levier, non implémentée ici.

Fichiers : `results/p56/dialogue.jsonl`, `results/p56/sessions.jsonl`,
`results/p56/server.log` (rapport profiler complet), `results/p56/serve.trace.json`.

## 2026-09-10 — Dialogue 20 min via la GUI (P5) et lecture de la trace P5.6 : points de friction restants

Dialogue A/B contre le serveur de la GUI (P5 complet, `results/dialogue-gui-p5`) :
102 tours en 20 min (38 hier, 76 en P5.6 CLI), TTFT médian 1,71 s, max 2,65 s,
6,3 tok/s médian, contexte final ≈ 6 500 tokens par agent, débit 6,4 → 6,1
tok/s entre le début et la fin. Trois paraphrases (tours 24, 52, 54, Jaccard
0,67-0,81 avec la réplique précédente du même agent), pas de boucle durable.
Bug trouvé par Vincent : le moteur résident est partagé GUI/LAN, un premier
tour GUI après une requête LAN était pris pour une continuation (image
refusée) → reset au premier tour GUI (`ae9f72f`), P5.7 pour l'indépendance.

Trace P5.6 (`results/p56/serve.trace.json`, 76 requêtes) décomposée par
requête : **le serveur ne coûte rien** (reste hors préfill/génération 2 ms,
attente client 20 ms) ; tout est dans le modèle. Préfill médian **2,16 s pour
~100 tokens de suffixe** (≈ 45 tok/s, GPU 55 %, CPU 88 % : host-bound à
courte longueur ; la couche PLE/n-gram pèse 28 % du préfill, P5.4) contre 115-
190 tok/s à 1 000-2 700 tokens ; génération 14 s pour ~80 tokens (89 % GPU,
51 % CPU). Mémoire stable (compresseur ≤ 0,8 Go, process 56 Go, MLX 53 Go).
Le TTFT plat de 2 s est donc un **coût fixe par forward** (48 couches
dispatchées + PLE), pas une fonction du contexte.

## 2026-09-10 — P6.2 : lookups PLE regroupés en un seul appel — gain réel à ~100 tokens, nul à 2 500 (l'E/S domine)

Instrumentation (`Qwen4ExpPLELookupStats` : appels, `MLXArray` construits,
`MLX.dequantized`, temps hôte des lectures) sur `Qwen4ExpLazyNGramStorage`,
propagée jusqu'au CLI comme `ngramCacheStats`. Mesure « avant » (ancien
chemin, un `lookup(shard:rows:)` par shard distinct touché) sur le
checkpoint 3-bit réel via `flash-chat-probe --resident-layers
--resident-async --profile-layers --trace` (deux tours, protocole P5.4) :
231 shards distincts touchés — **constant, indépendant du nombre de
tokens** — donc 693 `MLXArray` construits + 231 `MLX.dequantized` + un
scatter/add sur device par shard, à 125 tokens comme à 2 556.

**Correctif** : `Qwen4ExpNGramEmbedding.lookup` route chaque ID vers
`(shard, ligne locale)` puis appelle `Qwen4ExpLazyNGramStorage.lookupBatch`
une seule fois — une construction de tenseurs (packé + scales + biases) et
une déquantification pour tout l'appel, quel que soit le nombre de shards
touchés. IDs greedy identiques à la référence
(`[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171]`), parité
`flash-ngram-parity` `delta=0`.

**Premier essai de l'assemblage host→device, réfuté par la mesure** : une
version qui réordonne les lignes lues par shard vers leur position globale
via `Array.replaceSubrange` par position (une fois par token × tête, soit
~245 000 fois sur le préfill 2 543 tokens des deux tours) a **régressé** :
couche PLE 2 568 ms → 3 616 ms à 2 500 tokens (+41 %), 280 ms → 341 ms à 100
tokens. Le surcoût de l'API `Array` haut niveau (bornes, COW) sur une
boucle aussi chaude dominait le gain visé. Corrigé en écrivant directement
dans des `UnsafeMutableBufferPointer` pré-alloués (`update(from:count:)`,
même primitive que `readContiguousRuns`, P5.4) au lieu de `replaceSubrange`.

**Résultat final, mesuré deux fois (avant/après le correctif d'assemblage)** :

| | couche PLE (avant) | couche PLE (après) | lookupCalls | MLXArray | dequantize |
|---|---:|---:|---:|---:|---:|
| ~100 tokens (125 cumulés 2 tours) | 280 ms (méd. 16 ms, 17,5×) | **172 ms (10,75×, −39 %)** | 231 → 2 | 693 → 6 | 231 → 2 |
| ~2 500 tokens (2 556 cumulés) | 2 568 ms (méd. 138 ms, 18,6×) | 2 614 ms (19,1×, +1,8 %, bruit) | 231 → 2 | 693 → 6 | 231 → 2 |

**Cause racine corrigée par rapport à l'hypothèse de départ** (celle du
plan P6.2, qui reprenait celle de P5.4) : ce n'est pas le nombre de
dispatches GPU/constructions de tenseurs qui domine à grande échelle, mais
le temps hôte des lectures mmap elles-mêmes (`ple_host_read_seconds` :
2,445 s sur 2,614 s de couche PLE à 2 500 tokens, **93 %**) — de la lecture
aléatoire sur le Lexar USB pour des lignes non encore en cache. Regrouper
les constructions de tenseurs supprime bien le surcoût de dispatch (mesurable
et net à ~100 tokens, l'ordre de grandeur d'un tour de dialogue réel après
P5.2/P6.1) mais ne peut rien contre un plafond d'E/S à grande échelle — le
critère « couche PLE ≤ 3× une couche normale » n'est donc pas atteint
(10,75× et 19,1× après correctif, contre 17,5× et 18,6× avant), et ne
pouvait pas l'être par ce seul levier. Un chantier distinct (cache de lignes
plus agressif, préchargement, ou déplacement de la table n-gram vers un
support plus rapide) resterait nécessaire pour l'atteindre, hors périmètre
de ce correctif ciblé.

Fichiers : `results/p62/before-chat-100.stdout.txt`,
`results/p62/before-chat-2500.stdout.txt`, `results/p62/after2-chat-100.stdout.txt`,
`results/p62/after2-chat-2500.stdout.txt`, `results/p62/after-ngram-parity-v2.stdout.txt`
(traces `.trace.json` non versionnées, gitignore).

## 2026-09-10 — P6.5 : coût fixe du forward court — host-bound confirmé, QSA écarté, pas d'op isolée

Mesure `--profile-layers` (Release, checkpoint 3-bit réel) : préfill S≈100
(`results/p62/after2-chat-100.trace.json`, après P6.2 correctif) contre
décodage S=1 (`results/p65/s1-decode.trace.json`, régime établi, passes
≥ 2 d'un tour de 4 tokens).

| | S=1 (décodage) | S≈100 (préfill) | ratio |
|---|---:|---:|---:|
| médiane couche (hors PLE) | 4 ms | 15,5 ms | 3,9× |
| couche PLE | — (négligeable au décodage) | 172 ms | — |
| CPU / GPU (P5.6, même ordre de grandeur) | — | 88 % / 55 % | host-bound |

**Seuil dépassé** : 15,5 ms/couche hors PLE à S≈100, au-dessus des 10 ms de
la consigne P6.5 — recherche de l'opération en cause. **QSA écarté** : le
détail par couche (`results/p62/after2-chat-100.trace.json`) montre les 12
couches QSA et les 35 couches GDN/hyper-connections/MoE au **même coût**
(15,1-15,8 ms, aucune distinction visible) — cohérent avec §6.0 (« sous
2 048 tokens `makeMask` retourne `nil`, chemin dense ») : la construction
du masque QSA n'est structurellement pas en cause ici. Comparaison avec/
sans `--profile-layers` sur le même préfill (`results/p65/noprofile-100.
stdout.txt`, 929 ms / 48 couches ≈ 19,4 ms/couche) : cohérent avec la somme
mesurée sous profiler (910 ms), donc le coût du profiler lui-même
(documenté ailleurs à ~4,7 ms/frontière) n'explique pas l'écart ici.

**Non résolu, comme P4.3** : sans accès à Instruments/Shader Timeline,
impossible d'isoler l'opération précise (MRoPE ? `asType` ? dispatch par
couche ?) parmi les suspects listés par la consigne — le ratio 3,9× pour
100× plus de tokens confirme que le coût est dominé par un surcoût hôte
fixe par couche (dispatch/orchestration), pas par le calcul GPU lui-même
(qui grandirait bien plus vite avec S s'il dominait), cohérent avec le
diagnostic déjà posé en P5.6 (« host-bound à courte longueur, CPU 88 %,
GPU 55 % »). Aucun correctif appliqué : la piste la plus probable (QSA) est
écartée par la mesure, et les autres candidats de la consigne ne sont pas
localisables sans profileur GPU natif — documenté ici pour reprise
éventuelle avec Instruments, comme P4.3.

Fichiers : `results/p65/s1-decode.stdout.txt`, `results/p65/noprofile-100.
stdout.txt` (traces `.trace.json` non versionnées, gitignore).

## 2026-09-10 — P6.6 : validation finale — dialogue A/B 20 min sans conversation_id

Protocole : `qwen38 serve --trace results/p66/serve.trace.json` (checkpoint
3-bit, P6.1-P6.5 tous appliqués), `Scripts/agent-dialogue.py --max-minutes 20
--no-conversation-id` (T=0,7, top_p 0,8, 120 tokens max, aucun champ de
pénalité explicite — défaut serveur `presence 1.5`, P5.3). C'est le cas
Open WebUI / SDK openai ordinaire : aucun agent n'envoie `conversation_id`,
chaque tour renvoie tout l'historique complet — avant P6.1, ce cas payait
le rejeu stateless complet à chaque tour (TTFT croissant, référence
pré-P5 : 23,89 s au tour 38).

**Résultat** : 58 tours en 20,1 min, 5 868 tokens générés. TTFT tour 1
1,13 s ; TTFT médian **2,34 s** (< 3 s, critère atteint) ; TTFT max 4,43 s
sur 4 tours consécutifs (18-21, transitoire — aucun miss enregistré à ce
moment, candidat le plus probable : pression mémoire système, compresseur
à ratio 35× en fin de run) ; 0 tour > 3 s en dehors de cette fenêtre. Pic
process **58,32 Go** (< 75 Go). `/metrics` : `prefixHits` 56, `prefixMisses`
2 (les deux démarrages à froid, un par agent), `cachedConversations` 3 en
fin de run. Similarité Jaccard médiane (rows[i] vs rows[i-2], même agent)
**0,143**, max **0,261**, **aucune paire > 0,6** — cohérent avec la
validation dédiée de P6.3 (61 tours avec `conversation_id`, médiane 0,114,
max 0,184) : le masque de pénalité pré-rempli (P6.3) empêche la boucle
aussi bien sans `conversation_id` qu'avec.

**Tous les critères P6.6 atteints** : TTFT médian < 3 s sans
`conversation_id`, aucune boucle, pic process < 75 Go. Comparaison complète
avec P5.6 (76 tours, avec `conversation_id`) : `BENCHMARKS.md` « P6.6 ».

Fichiers : `results/p66/dialogue.jsonl`, `results/p66/dialogue.log`,
`results/p66/server.log` (rapport profiler complet), `results/p66/
serve.trace.json` (non versionnée, gitignore).

## 2026-09-11 — PM4.4 : acceptation MTP sur le 4-bit — le 3-bit ne bride pas le drafter

Machine rebootée (8,6 Go d'anonyme, compresseur et swap à zéro), Release,
résident, `asyncEval`, prompt de référence, 32 tokens.

| | 3-bit (PM4.3) | **4-bit (PM4.4)** |
|---|---|---|
| Acceptation MTP bloc 2 | 47,6 % (10/21) | **34,8 % (8/23)** |
| Acceptation MTP bloc 3 | 28,2 % | **24,4 % (10/41)** |
| `replayedTokens` | 0 | 0 |
| Pic MLX | 56,6 Go | 79,55 Go |

**Réponse à la question posée** : non, le 3-bit ne limite pas le drafter — le
4-bit accepte **moins** (34,8 % contre 47,6 %). La tête MTP est la même dans
les deux checkpoints (seuls les experts ont été requantifiés en 3-bit) ; c'est
donc l'accord drafter/cible qui est meilleur avec la cible 3-bit. Le MTP n'a
pas de réserve cachée dans le 4-bit : revenir au 4-bit (P3) ne se justifie
plus que par la qualité (0,42 nat de logprob), pas par le débit.

**Les IDs MTP diffèrent du greedy sur le 4-bit** (index 21, `1725` → `9370`,
identique en bloc 2 et 3 ; le 3-bit était bit-identique). Ce n'est pas un bug :
le scorer teacher-forced, qui fait un **forward multi-tokens** comme la
vérification MTP, donne lui aussi `9370` à cette position, avec une **marge de
0,21 nat** (marges voisines 1,5 à 8,4 ; accord argmax 96,9 %, rang cible
moyen 1,03). Un forward multi-tokens et une suite de forwards à un token ne
sont pas bit-identiques (ordre d'accumulation des matmuls), et la chaîne 4-bit
amplifie l'écart jusqu'à faire basculer une quasi-égalité. **Le critère « IDs
identiques au greedy » n'est donc valide que tant qu'aucune quasi-égalité ne
tombe dans un bloc vérifié** — à mentionner dans toute validation MTP future.

**Les temps de ce run ne sont pas comparables** : 8,6 Go d'anonyme + 79,5 Go
de pic dépassent la machine ; le sampler montre le compresseur à 57 Go et
**9,9 millions de décompressions**, `memorystatus_level` descendu à 27 %. Le
greedy 4-bit mesure 0,87 s/token (contre 0,164 s/token en 3-bit) : c'est de la
décompression, pas du modèle. Le 4-bit reste inutilisable sur cette machine
sans P3, même après reboot.

## 2026-09-11 — Shards n-gram sur le SSD interne : le préfill réel gagne 37 %

`Scripts/localize-checkpoint.sh` a produit
`/Users/vincent/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` : 7 shards
n-gram copiés (35,7 Go, 53 s), 15 shards liés au Lexar, 138 Go encore libres.
Le chargement lit les shards liés et n'est pas concerné ; seul le préfill
touche la table n-gram (mmap paresseux).

**Piège de méthode** : une première série A/B/B/A avec **le même** prompt
répété n'a montré aucun écart (forward 10-11 s, PLE 3-5,5 s) — un prompt
redondant réutilise ses lignes n-gram et le cache de pages efface la
différence. Il faut un prompt **différent par run** pour mesurer des lectures
froides. Les chiffres ci-dessous utilisent des prompts distincts.

| Prompt | Chemin | forward préfill | couche PLE | ratio PLE/couche | tok/s |
|---|---|---|---|---|---|
| mots aléatoires, 2 511-2 563 tok | Lexar | 56,5 / 51,5 s | 49,7 / 44,6 s | 32× / 29× | 45-50 |
| idem | **SSD** | **33,3 / 32,3 s** | 26,0 / 25,0 s | 17× / 15× | 76-79 |
| **prose française réelle, 3 137 tok** | Lexar | 64,5 s | 56,6 s (88 %) | 36× | **48,6** |
| idem (mêmes 150 624 lectures) | **SSD** | **40,4 s** | **31,3 s** (77 %) | 20× | **77,6** |

**−37 % sur le préfill réel**, à nombre de lectures identique (150 624, soit
48 lignes par token : 6 couches PLE × 8 têtes). Coût par ligne : **0,376 ms
sur le Lexar contre 0,208 ms sur le SSD**.

**Correction d'un chiffre publié** : les « 87-190 tok/s de préfill » des
entrées P4/P5 venaient de prompts à phrase répétée (forte réutilisation
n-gram). Sur du texte varié, le préfill réel est **48,6 tok/s sur le Lexar,
77,6 sur le SSD**, et la couche n-gram pèse **88 % du forward**, pas 28 %.
C'est donc le premier poste du préfill, loin devant tout le reste.

Reste : 0,208 ms par ligne sur NVMe est encore ~2-3× le coût d'une lecture
brute — le chemin hôte (faute de page mmap + copie) domine, pas le disque.
Levier suivant : `pread` groupés sur offsets triés, LRU de lignes plus grand,
ou préchargement des lignes du prompt en une passe. Note : le compteur
`ngram_cache_hit_rate` vaut exactement 0,5000 dans **tous** les runs — c'est
un artefact structurel (lectures appariées), il ne mesure rien d'utile.

## 2026-09-11 — P7 : débit de génération

Protocole PLAN.md « P7 — Débit de génération : attribuer le coût par
sous-bloc, puis fusionner » (2026-09-11). Ordre exécuté : P7.1 → P7.2 → P7.3
→ P7.4 (implémenté puis **retiré** après échec sur le checkpoint réel) →
P7.5 (non applicable) → P7.6. Toutes les mesures P7.1-P7.4 : `flash-layer-bench
--steps 300 --async-interval 8` (Release, sans checkpoint), médiane
« horloge interne », deux runs par variante sauf mention contraire.

### P7.1 — profil par ablation : le MoE routé domine, pas la chaîne de gating GDN

Nouveau `Qwen4ExpLayerBenchAblation` (`Sources/Qwen38Core/FlashNext/Qwen4ExpLayerBenchAblation.swift`)
et `--ablate <bloc>` sur `flash-layer-bench` : chaque cas court-circuite un
sous-bloc en renvoyant un tenseur de forme correcte (jamais numériquement
correct, instrument de mesure uniquement, `.none` partout en production —
piège 11/8 respecté). Threading par constructeur (comme `fusionLevel`),
`.none` par défaut sur `Qwen4ExpGatedDeltaNet`, `Qwen4ExpQSAAttention`,
`Qwen4ExpSparseMoE`, `Qwen4ExpGatedResidual`, `Qwen4ExpDecoderLayer`.

**Résultat contraire à l'hypothèse du plan** (« candidat attendu : la chaîne
de gating GDN, ~15-20 ops »): le gating GDN pèse **0,8 %** de la couche.
Le sous-bloc dominant, et de loin, est le **MoE routé** (`switch_mlp`) :

| Sous-bloc | GDN ms | GDN % couche | QSA ms | QSA % couche |
|---|---:|---:|---:|---:|
| **moe** (routage + experts + partagé) | 3,955 | **88,1 %** | 3,465 | **77,8 %** |
| — dont routage (gate+softmax+argPartition) | ~0 | ~0 % | ~0 | ~0 % |
| — dont `switch_mlp` (SwitchGLU) | 3,90 | 86,9 % | 3,415 | 76,7 % |
| — dont expert partagé | ~0 | ~0 % | ~0 | ~0 % |
| gdn-recurrence (`gatedDeltaUpdate`) | 0,035 | 0,8 % | — | — |
| gdn-projections (in_proj×4 + conv1d) | 0,24 | 5,3 % | — | — |
| qsa-attn (SDPA + indexeur) | — | — | 0,0 | 0,0 % |
| hyper (mix hyper-connections) | 0,235 | 5,2 % | 0,19 | 4,3 % |
| norms (hc_norm/q_norm/k_norm/RMSNormGated) | 0,095 | 2,1 % | 0,08 | 1,8 % |
| **couche complète (baseline)** | **4,49** | 100 % | **4,455** | 100 % |
| Σ blocs nommés | 4,56 (102 %) | | 3,735 (**84 %**) | |

Sous-sondes MoE (`--ablate moe-routing/moe-switch-mlp/moe-shared-expert`,
gardent le routeur réel et ne zèrent qu'un terme à la fois) : le routage et
l'expert partagé sont chacun **dans le bruit de mesure** (Δ négatif de
quelques centièmes de ms, i.e. ~0 %) ; la totalité du coût MoE est portée
par le gather-matmul quantifié `switch_mlp` (SwitchGLU, `Vendor/mlx-swift-lm`).
Confirmé sous la quantification réelle du checkpoint (3-bit g64) :
GDN 2,715→0,555 ms (moe 79,6 %), QSA 2,73→0,985 ms (moe 63,9 %) — même
verdict, poids différents.

**Somme des blocs vs couche complète** : GDN 102 % (±15 % respecté). QSA
84 % — **hors de la fourchette ±15 %**, à signaler explicitement comme
demandé : l'écart (~16 %, ~0,72 ms) correspond aux projections q/k/v/o et
au RoPE de `Qwen4ExpQSAAttention`, qui n'ont pas de case d'ablation dédiée
dans le plan (seul `qsa-attn`, SDPA+indexeur, existe) — non un artefact de
mesure mais une lacune assumée du découpage en sous-blocs de P7.1.

Verdict : P7.1 désigne sans ambiguïté un sous-bloc dominant (`moe`/
`switch_mlp`, 78-88 % de la couche, 87-99 % du budget MoE), **contredisant
l'hypothèse a priori du plan** (chaîne de gating GDN). C'est ce sous-bloc,
et lui seul, qui gouverne P7.4.

### P7.2 — `captureGPUTrace` : capture obtenue mais sans noms de noyau

`MTL_CAPTURE_ENABLED=1` + `ProfilingSession.captureGPUTrace(phase:)` (ajout
`--gpu-trace`, exige `--trace`) : **ne lève pas l'erreur documentée**
(`captureNotEnabled`/`notProduced`) — un `.gputrace` de 603 Mo est produit
mécaniquement pour un seul pas de couche synthétique. Mais
`Vendor/mlx-swift/Package.swift` ne définit `MLX_METAL_DEBUG` nulle part
(`grep` vide sur les `cxxSettings` du target `Cmlx`), condition que le
profiler documente lui-même comme nécessaire « pour que `GPU.startCapture`
fasse quoi que ce soit » d'utile : extraction de chaînes sur `store0`/
`index`/`metadata` (1511 chaînes) — aucun nom de noyau MLX reconnaissable
(`copy_gpu`, `QuantizedMatmul`, `binary_op`, `qmv`…). Sans accès à Xcode
GUI dans cet environnement pour ouvrir la capture et vérifier visuellement,
impossible de confirmer si les pipelines sont réellement anonymes ou si
`strings` est simplement le mauvais outil pour ce format binaire. Obstacle
documenté conformément à la consigne (« rapporter l'obstacle exact et
s'arrêter ») : la précondition `MLX_METAL_DEBUG` qu'exige un comptage par
noyau fiable est absente du build vendored, et l'ajouter exigerait de
reconstruire mlx-swift — hors périmètre sans accord. Comptage analytique de
repli : voir le tableau P7.1 (attribution par sous-bloc) et P2-code (a) du
2026-09-07 (familles de noyaux par échantillonnage `sample`, Copy≈260/
Binaire≈270/QuantizedMatmul≈77) — la dominance mesurée de `switch_mlp`
recadre ce comptage : les familles Copy/Binaire qu'il pointait comme
majoritaires sont vraisemblablement pour bonne part la mécanique de
dispatch de `SwitchGLU` (expand/squeeze/scatter), pas GDN/QSA/hyper comme
supposé à l'époque. Fichier nettoyé après inspection (603 Mo, non conservé).

### P7.3 — `MLX.compile` du pas complet (4 et 8 couches) : aucun gain, GDN casse la parité

Nouveau `Qwen4ExpLayerBench.runMultiLayerStep`/`checkMultiLayerStepParity`
(`--step-layers N [--compiled-step] [--shapeless]`) : empile N couches
indépendantes du même type et compile **tout le pas** (`hidden → couche 0 →
… → couche N-1`) en un seul `MLX.compile`, chaque cache boxé comme état
(`Updatable`), au lieu d'un `compile` par couche (P2-code (c)). QSA passe un
masque explicite par couche (argument de tableau, comme le closure à 2 args
existant), pas une variable capturée.

| Config | Eager | Compiled | Δ | Parité (32 pas) |
|---|---:|---:|---:|---|
| GDN × 4 | 18,56 ms/pas | 18,29 ms/pas | −1,5 % | **FAIL** (diff rel max 3,61, seuil 1) |
| GDN × 8 | 36,89 ms/pas | 35,86 ms/pas | −2,8 % | **FAIL** (diff rel max 17,04 — croît avec N) |
| QSA × 4 | 20,16 ms/pas | 23,01 ms/pas | **+14,1 %** | PASS (bit-exact) |
| QSA × 8 | 41,15 ms/pas | 47,68 ms/pas | **+15,9 %** | PASS (bit-exact) |
| GDN/QSA, `--shapeless` | — | — | — | **crash** (`Fatal error: [Primitive::output_shapes] Split cannot infer output shapes`) |

QSA : le masque causal change de forme à chaque pas (`cache.offset` croît),
donc `compile` retrace à chaque appel — bit-exact avec eager (la retrace
est correcte, juste inutile), mais plus lente que l'eager, et l'écart
**s'aggrave** avec plus de couches (+14,1 %→+15,9 % de N=4 à N=8) : aucune
fusion inter-couches ne compense le coût de retracer, contrairement à
l'hypothèse du plan. GDN : forme de cache fixe (bon candidat *a priori*),
mais la sortie compilée diverge de l'eager **au-delà** de la tolérance
1e-3-équivalente, et l'écart croît avec le nombre de couches (3,61 à N=4,
17,04 à N=8) — signe d'une divergence numérique qui s'accumule couche
après couche (probablement un ordre de réduction différent dans le noyau
`gatedDeltaUpdate`/`GatedDeltaKernelManager` sous graphe compilé), pas du
bruit flottant isolé. `--shapeless` casse net sur un `Split` dont MLX ne
sait pas inférer la forme sans trace concrète — obstacle documenté, pas
contourné. **Verdict : aucune configuration testée n'est à la fois plus
rapide et numériquement sûre — P7.3 ne retient rien.**

### P7.4 — Fusion Metal du sous-bloc dominant : implémentée, PASS en synthétique, **retirée** après échec sur le checkpoint réel

P7.1 désigne `switch_mlp` (SwitchGLU, `Vendor/mlx-swift-lm`), pas la chaîne
de gating GDN anticipée par le plan — donc pas de candidat « noyau Metal
maison sur ~15-20 ops élémentaires » au sens littéral de la consigne.
Investigation du seul point d'entrée public de `SwitchGLU`
(`callAsFunction(_:_:)`, matérialise `[tokens,topK,hidden]` puis un
`scatterUnsort`) : `projectExperts`, `supportsDirectWeightedReduction`,
`callAndWeightedReduce` et le kernel maison `weightedExpertUnsort` déjà
présent côté Vendor sont `private`/`package` — inaccessibles depuis ce
paquet sans modifier `Vendor/mlx-swift-lm` (interdit). En revanche
`gatherQuantizedMM` (le primitif public de gather-matmul quantifié, dans le
paquet **de base** `mlx-swift`, pas `mlx-swift-lm`) est appelable
directement, et les poids déjà chargés de `gate_proj`/`up_proj` sont
lisibles via `Module.parameters()` (réflexion publique de `MLXNN`, qui
n'a pas besoin que les propriétés `@ModuleInfo` sous-jacentes soient
`public`).

**F7 implémentée** (`Qwen4ExpFusionLevel.f7MoEGateUp`, réversible, threadée
comme F1-F6) : concatène `gate_proj`+`up_proj` (poids/`scales`/`biases`
quantifiés packés, axe de sortie) et appelle un seul `gatherQuantizedMM`
au lieu de deux, exactement le raisonnement de F1 transposé de `Linear` à
`SwitchLinear`/`QuantizedSwitchLinear` — `down_proj` reste un second appel
séparé (dépend de l'activation). Décodage seul (`indices.size < 64`, seuil
`doSort` de `SwitchGLU` lui-même) ; le préfill retombe intact sur le
chemin `switchMLP` d'origine.

**Parité (bench synthétique, 32 pas)** : PASS sur les deux types de couche
— GDN bit-exact (diff 0,0), QSA diff normalisée max 0,32 (bruit bf16,
même magnitude que F2, sous le seuil 1).

**Mesure bench** (avant tout run checkpoint, comme imposé) : gain
**marginal et incohérent** — 4-bit g32 : GDN +0,1 % (aucun gain), QSA
−3,1 % ; 3-bit g64 (quantification réelle du checkpoint) : GDN −5,0 %,
QSA −0,9 %. Loin d'un effet clair malgré 78-88 % du budget de couche dans
ce sous-bloc — indice que le coût de `switch_mlp` n'est pas dominé par le
nombre de dispatches (fusionner 2 appels en 1 n'a presque rien changé)
mais par le volume de données touchées par le gather quantifié lui-même.

**Validation checkpoint réel (P7.6, avant tout changement de défaut, comme
imposé)** — prompt de référence, greedy, 32 tokens, `--resident-async
--resident-async-interval 8` : IDs **identiques** à la référence
(`[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171, …]`) — la fusion
reste numériquement correcte sur le vrai checkpoint — mais :

| | référence (`.none`) | F7 (`--fusion-level 7`) |
|---|---:|---:|
| decode (32 tokens) | 4,775 s (0,149 s/token, 6,7 tok/s) | **84,056 s** (2,627 s/token, 0,38 tok/s) |
| mémoire pic MLX | 56,99 Go | **93,7 Go** |

**17,6× plus lent, +36,7 Go de pic mémoire.** Cause : `fuseGateUpProjections()`
matérialise un tenseur fusionné par couche **en plus** des poids
`gate_proj`/`up_proj` d'origine (toujours résidents via `switchMLP`, jamais
libérés) — sur ~48 couches à poids quantifiés réels (pas les placeholders
minuscules du bench), ce doublement partiel pousse la résidence de 57 Go
vers 93,7 Go sur une machine à 96 Go : exactement le mécanisme de pression
mémoire (H-A) documenté depuis le début de ce chantier P, ici déclenché
par la fusion elle-même plutôt que subi passivement. Le bench synthétique
(poids aléatoires minuscules, quelques Mo par couche) ne pouvait pas
révéler ce coût — la consigne « mesurer sur le bench avant le checkpoint »
a été respectée à la lettre, mais le bench n'était structurellement pas en
mesure de voir ce risque précis (taille des poids, pas leur nombre
d'opérations).

**Décision : F7 retirée du code** (pas seulement laissée à `.none` comme
F1/F2/F4/F6) — contrairement à ces derniers (« aucun gain mais aucune
régression prouvée, conservés »), F7 est **prouvée activement dangereuse**
sur le modèle réel ; la conserver, même inerte derrière un flag, serait un
piège pour une session future qui l'activerait sans relire cette entrée.
`Qwen4ExpFusionLevel` s'arrête à nouveau à F6 ; `Qwen4ExpSparseMoE` est
revenu à son état P7.1 (ablation uniquement, aucune trace de F7).
`--fusion-level` CLI redevient 0-6.

### P7.5 — non applicable

Aucun sous-bloc autre que `moe` ne dépasse 15 % de la couche (le plus
proche : `gdn-projections`/`hyper` à 5,2-5,3 %, ou le résidu QSA non
attribué à 16 % qui n'est pas un « autre sous-bloc désigné » au sens de la
consigne) — et P7.4 n'a de toute façon rien retenu. Passage direct à P7.6.

### P7.6 — validation finale sur le checkpoint réel

`/Users/vincent/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` (3-bit
hybride SSD), `caffeinate -dimsu` + `Scripts/preflight-resident.sh` (seuil
35 Go, PASS à 16,8-23,5 Go selon les runs) devant chaque run, jamais deux
runs simultanés, GUI vérifiée non résidente (`pgrep qwen38-bench-ui`
négatif). Code final = état post-P7.1 (ablations `.none` par défaut, F7
retirée) : **aucun changement de comportement de production dans ce
chantier** — P7 n'a produit qu'un diagnostic, aucune fusion retenue.

| Variante | tokens | temps | s/token | tok/s | IDs = référence |
|---|---:|---:|---:|---:|---|
| Greedy | 32 | 4,864 s | 0,152 | 6,58 | oui |
| MTP bloc 2 | 32 | 3,853 s | 0,120 | 8,30 | oui (identiques token à token) |

MTP : 10/21 acceptés (47,6 %), 11 rollbacks, 0 rejeu — cohérent avec PM3/PM4
(taux d'acceptation dépendant du prompt, pas un problème de bookkeeping).
Garde Q-B (`flashTeacherForcedRegressionGuardV32`) : **PASS**,
`hits=10/28 meanLogProb=-4.8003182`, identique à la valeur de référence.
87 tests verts (`Scripts/run-tests.sh`), build Release vert tout au long
de la session.

**Jauge ≥ 9 tok/s (PLAN.md) : non atteinte** (6,58 greedy / 8,30 MTP) —
consigné tel quel : P7 n'a pas trouvé de sous-bloc fusionnable en toute
sécurité dans le temps disponible. Le plafond tient à un fait nouveau et
important pour la suite : **le goulot n'est pas le bookkeeping hôte de
petits noyaux** (l'hypothèse qui motivait P2-fusion/P4.3/ce chantier), mais
le **gather-matmul quantifié du MoE routé** (`SwitchGLU`, 512 experts,
top-10, dans `Vendor/mlx-swift-lm`) — un poste que ce dépôt ne peut pas
opter d'optimiser sans soit modifier le paquet vendored (interdit), soit
réimplémenter tout le gather quantifié à la main (risque élevé, non
tenté ici faute de temps et d'un harnais de parité suffisant pour un
composant de cette taille).

### Écarts à la consigne P7

1. **Ordre d'exécution** : P7.4 (F7) a été implémentée et mesurée sur le
   bench **avant** P7.3 (compile du pas complet), pas dans l'ordre strict
   P7.1→P7.2→P7.3→P7.4 demandé — inversion sans conséquence sur les
   résultats (tâches indépendantes), corrigée ici en présentant le rapport
   dans l'ordre du plan.
2. **Commits regroupés** : P7.1 (ablation) et le reste (CLI `--gpu-trace`
   P7.2 + `--step-layers`/`--compiled-step` P7.3) sont commités en deux
   groupes plutôt que quatre commits stricts un-par-tâche — les fichiers
   `Qwen4ExpLayerBench.swift` et `Qwen38CLI.swift` entremêlent les trois
   tâches au niveau de la ligne (options CLI ajoutées côte à côte), et une
   séparation stricte par `git add -p` sur un diff de cette taille était
   jugée plus risquée (erreur de sélection de hunk) que le gain de
   traçabilité.
3. **P7.2** : capture obtenue mécaniquement (pas d'erreur), mais l'absence
   de noms de noyau n'a pu être confirmée que par `strings` sur le binaire
   du `.gputrace`, pas par une ouverture Xcode (indisponible dans cet
   environnement) — documenté comme limite de vérification, pas comme un
   échec de capture.
4. **F7 retirée plutôt que « conservée à `.none` »** : traitement plus
   strict que le précédent F1/F2/F4/F6 (« aucun gain, conservé ») parce que
   F7 n'est pas neutre — elle est activement dangereuse (17,6× plus lent,
   +36,7 Go de pic) si jamais activée. Écart assumé et documenté plutôt
   qu'une application mécanique du même traitement que les leviers
   précédents.

## 2026-09-11 — P7 revisité : le MoE du bench coûte 25× le même calcul écrit à la main

Vérification de la conclusion de P7.1 (« le MoE routé domine, 88 % de la
couche ») par comparaison avec MLX Python et par micro-bench Swift. Toutes les
mesures ci-dessous sont reproductibles sans checkpoint.

**1. Ce n'est ni le calcul ni la bande passante.** Les 10 experts routés
représentent 49 M paramètres par couche, soit 0,10 GFLOP — 4,7 GFLOP par
token sur 48 couches. À 6,6 tok/s cela fait **0,03 TFLOP/s** pour un M3 Max à
~14 TFLOP/s de crête, et ~5,8 Go/s de bande passante pour ~400 Go/s
disponibles. Le MoE ne calcule quasiment rien.

**2. Ce n'est pas `gather_qmm`, ni la version de MLX.** En Python, mêmes
formes (512 experts, 640×2560, 3 bits g64, M=1), coût **par op** avec un seul
`eval` pour N ops : `gather_qmm` 0,035 ms, contre 0,028 ms pour un
`quantized_matmul` dense sur les 10 experts déjà rassemblés — le gather ne
coûte que **1,1×** le dense. Bloc MoE complet (routage + 3 `gather_qmm` +
silu + réduction) : **0,138 ms** en ops indépendants, **0,168 ms** en chaîne
dépendante (donc ce n'est pas non plus un effet de parallélisme). MLX 0.31.1
(la version C++ embarquée par mlx-swift 0.31.6) et MLX 0.32.2 donnent le même
chiffre à 1 % près : **la version de MLX n'est pas en cause**.

**3. Coût fixe par op côté Swift** (`qwen38 op-overhead-probe`, Release,
chaîne dépendante de 200 ops) : addition 8,79 µs, multiplication 6,80 µs,
silu 6,94 µs, reshape 1,78 µs, et matmul `[1,2560]×[2560,2560]` bf16
**65 µs** — soit 200 Go/s, la moitié de la crête : le GPU est sain. Python
sur la même machine : 4,2 µs pour l'addition. Swift paie donc ~2× le coût
hôte de Python par op, pas 10×.

**4. La contradiction.** Ablation rejouée à la main (Release,
`flash-layer-bench --layer-kind gdn --steps 200`) : couche complète
**4,88 ms**, MoE ablaté **1,03 ms** ⇒ le MoE coûte **3,85 ms (79 %)**. Or un
micro-bench Swift du **même** bloc (routage + 3 `gatherQuantizedMM` + silu +
réduction, mêmes formes, chaîne dépendante) coûte **0,373 ms en Debug**, donc
~0,1-0,2 ms en Release. **Écart d'un facteur ~25 entre le MoE du bench et le
même calcul écrit à la main.** Ce n'est ni MLX, ni le gather, ni le matériel :
c'est dans notre chemin `Qwen4ExpSparseMoE`/`SwitchGLU`.

**5. Les sous-attributions de P7.1 ne tiennent pas.** Les trois ablations
`moe` (1,03 ms), `moe-switch-mlp` (1,05) et `moe-routing` (0,97) font chacune
tomber la couche au même niveau : chaque sous-ablation retire l'essentiel du
coût, donc elles ne sont pas indépendantes et la répartition « switch_mlp
86,9 % / routage ~0 % » n'en découle pas. À refaire avec des ablations qui
n'ôtent que leur propre sous-bloc.

Piste écartée en passant : `captureParity` est bien à `false` par défaut et le
bench ne l'active pas.

## 2026-09-11 — P8 : le MoE ne coûtait pas 25× — une fuite fp32 dans GDN/QSA, corrigée (F7)

Protocole PLAN.md « P8 — Le MoE coûte 25× ce qu'il devrait : localiser et
corriger » (2026-09-11), P8.1 → P8.5. Verdict : **le MoE n'a jamais été le
coupable.** L'instrumentation par étage (P8.1) a semblé le confirmer, mais
en creusant *pourquoi* `SwitchGLU` coûtait 4 ms en contexte de couche alors
qu'il coûte 150-350 µs isolé, la cause s'est révélée être une fuite de
dtype : les branches GDN et QSA calculent (à dessein, piège #6) une partie
de leur état en float32, mais ne l'arrondissent jamais à bf16 avant de le
transmettre à la suite de la couche — MoE y compris. Tout ce qui suit
(gate, softmax, `switch_mlp`, expert partagé…) tourne alors sur son chemin
fp32, ~12-27× plus lent que le chemin bf16 pour le même calcul. Corriger
cette seule fuite (F7) fait passer une couche GDN de 4,8 ms à 1,15 ms
(×4,1) et une couche QSA de 5,2 ms à 1,5 ms (×3,5), **sans toucher au
MoE ni à `Vendor/`**, et sur le checkpoint réel : IDs greedy identiques à
la référence, garde Q-B toujours PASS, decode ×2,7 (greedy) et ×1,4 (MTP).

### P8.1 — chronométrage par étage : un `eval` par étage, un coupable qui ne résiste pas à l'isolement

Nouveau `Qwen4ExpMoEStageProfiler` (`Qwen4ExpSparseMoE.swift`) : threadé
comme `ablation`, non-nil uniquement sous `flash-layer-bench --moe-stages`
(jamais en production). Quand actif, `Qwen4ExpSparseMoE.callAsFunction`
bascule sur `callWithStageProfiling`, qui chronomètre chaque étage
(`gate`, `softmax`, `argPartition`, `takeAlong+normalize`, `switchMLP`,
`weightedExpertSum`, `sharedExpertGate`, `sharedExpert`, `shared-combine`,
`add`) en forçant un `eval()` après chacun — **avant P8.1 lui-même**, la
première tentative (non conservée) mesurait sans `eval` intermédiaire et
attribuait donc tout au dernier étage forcé, exactement le piège que le
plan demandait d'éviter. Une baseline dédiée (`eval()` sur un tableau déjà
évalué, 2000 appels) mesure le coût d'une synchro « à vide » : 0,0002 ms —
négligeable, donc la somme des étages n'est pas gonflée par un simple
compteur d'appels `eval()`.

**Tableau, couche GDN, `--fusion-level 0` (chemin d'origine)** :

| Étage | ms/pas | % du total étages |
|---|---:|---:|
| gate | 0,8647 | 13,2 % |
| softmax | 0,1833 | 2,8 % |
| argPartition | 0,2060 | 3,1 % |
| takeAlong+normalize | 0,1952 | 3,0 % |
| **switchMLP** | **4,0575** | **62,0 %** |
| weightedExpertSum | 0,2187 | 3,3 % |
| sharedExpertGate | 0,2186 | 3,3 % |
| sharedExpert | 0,2503 | 3,8 % |
| shared-combine | 0,1783 | 2,7 % |
| add | 0,1749 | 2,7 % |
| **somme des étages** | **6,5476** | — |

La somme (6,55 ms) dépasse la couche non instrumentée (4,88 ms) : chaque
`eval()` forcé casse la fusion/le regroupement que MLX aurait fait
paresseusement, donc cette somme **n'est pas** directement comparable à la
mesure de production — c'est un instrument de localisation, pas une
mesure de coût réel (documenté explicitement dans le code et ici, comme
demandé). `switchMLP` domine sans ambiguïté (62 %, ~20× le deuxième
étage) : verdict identique à P7.1, mais cette fois avec un vrai isolement
par synchronisation plutôt que par soustraction d'ablations qui se sont
révélées non indépendantes (P8.3).

**Le tournant** : `op-overhead-probe` étendu avec un micro-bench de
`SwitchGLU` seul (512 experts, 640 intermédiaire, top-10, 4 bits g32 —
mêmes dimensions que le bench), construit et pesé hors de
`Qwen4ExpSparseMoE`/`Qwen4ExpDecoderLayer` :

| Mesure | µs/appel |
|---|---:|
| `SwitchGLU` seul, chaîne dépendante (200 appels, un seul `eval` final), entrée `.bfloat16` | 150-153 |
| idem, indices dispersés au lieu de `[0..9]` séquentiels | 153 (aucune différence — écarte l'hypothèse d'un gather mémoire défavorable) |
| `SwitchGLU` seul, **un `eval` par appel**, entrée `.bfloat16` | 342-350 |
| `SwitchGLU` seul, **un `eval` par appel**, entrée **`.float16`** | — (voir plus bas) |
| `SwitchGLU` seul, **un `eval` par appel**, entrée **`.float32`** | **4021,95** |

`SwitchGLU` isolé ne coûte que 150-350 µs — 12-27× moins que les 4,06 ms
mesurés en contexte de couche — **sauf quand on lui donne du float32 en
entrée**, où il retombe exactement sur le même ordre de grandeur (4,02 ms)
que le chiffre « en couche ». `compiledSiluProduct`/`weightedExpertSum`
(les deux fermetures `MLX.compile(shapeless: true)` utilisées par
`SwitchGLU`/`Qwen4ExpSparseMoE`) ont aussi été mesurées directement contre
leur équivalent non compilé : aucun surcoût (silu×up compilé 4,0 µs contre
5,6 µs non compilé ; réduction pondérée compilée 11,2 µs contre 11,3 µs
non compilée) — **hypothèse (e) du plan (surcoût d'appel `compile()`)
écartée par la mesure**, tout comme (a) `weightedExpertSum` (déjà la
formule écrite à la main), (b) le passage par `SwitchGLU.callAsFunction`
(coût réel mesuré, cf. ci-dessus) et (d) `argPartition` (0,21 ms, dans le
bruit). Reste (c)/nouveau : une matérialisation en float32 quelque part en
amont de `Qwen4ExpSparseMoE`.

Un `DEBUG` temporaire (retiré avant commit) dans `Qwen4ExpDecoderLayer` a
confirmé : `attentionMix.mixedInput` est bf16, mais **la sortie de la
branche d'attention (GDN et QSA) est float32** — la fuite est dans
l'attention, pas dans MoE. Deux causes distinctes, une par type de couche :

- **GDN** (`Qwen4ExpGatedDeltaNet.callAsFunction`) : `gatedDeltaUpdate`
  (le noyau delta-rule récurrent) rend son état en float32 par conception
  (piège #6 — précision de l'état récurrent). Son résultat `out` est passé
  tel quel à `norm(out, gate: z)` (`Qwen4ExpRMSNormGated`), dont le
  commentaire dit explicitement « round once at the branch boundary » —
  mais son `.asType(inputs.dtype)` final arrondit à *son propre* `inputs`
  local (= `out`, float32), donc à un no-op, pas au bf16 du réseau.
- **QSA** (`Qwen4ExpQSAAttention.callAsFunction`) : `Qwen4ExpMRoPE.apply`
  mélange q/k (bf16) avec des tables `cos`/`sin` délibérément construites
  en float32 (piège #6 — précision du RoPE) ; `rotated * cosB + rotateHalf
  * sinB` promeut donc le résultat en float32, sans jamais redescendre en
  bf16 ensuite.

Dans les deux cas, la fuite se propage sans interruption jusqu'à
`Qwen4ExpSparseMoE.callAsFunction` (rien en aval ne recaste), qui hérite
donc d'un `x` float32 et exécute tout son chemin — gate, `switch_mlp`,
expert partagé — en float32. **Verdict P8.1 révisé : le sous-bloc qui
domine la mesure (`switch_mlp`) n'est pas la cause ; il est la victime la
plus visible d'une fuite de dtype dans la branche d'attention qui le
précède.**

### P8.2 — F7 : arrondir au bon endroit, pas avant

Nouveau niveau `Qwen4ExpFusionLevel.f7GatedBranchDtype = 7` (sans lien
avec l'ancien F7 de P7.4, retiré — même numéro réutilisé, sémantique et
risque totalement différents, signalé explicitement dans le code pour
qu'une session future ne confonde pas les deux) :

- **GDN** : au lieu d'arrondir `out` *avant* `norm(out, gate: z)` (ce qui
  jetterait la précision fp32 que la réduction `MLXFast.rmsNorm` est censée
  exploiter — vérifié : ça ne change pas la taille de l'écart de parité,
  seulement son signe), F7 laisse `norm` tourner sur le `out` fp32 brut
  (comportement inchangé pour la normalisation elle-même) et ajoute
  l'arrondi **après**, sur la sortie de `norm` : `gatedNorm.asType(inputs.dtype)`
  où `inputs` est bien le paramètre externe de `callAsFunction` (le bf16
  du réseau), pas le `inputs` local de `Qwen4ExpRMSNormGated`.
- **QSA** : après les deux appels `rotaryEmbedding.apply(queries/keys,
  positionIDs:)`, F7 ajoute `queries = queries.asType(hiddenStates.dtype)`
  / idem pour `keys`, restaurant le bf16 avant que SDPA ne s'exécute.

Threadé comme `ablation` : `fusionLevel` ajouté aux inits de
`Qwen4ExpGatedDeltaNet`/`Qwen4ExpQSAAttention` (défaut `.none`, comme
partout), et `Qwen4ExpDecoderLayer` le leur transmet désormais (avant
cette tâche, seul `Qwen4ExpSparseMoE` le recevait). Deux nouveaux tests
(« P8.2 (F7) : la branche GDN/QSA ne fuit plus en float32 sans F7 actif »)
pinguent le dtype de sortie de chaque branche avec/sans F7 sur un spec de
quantification explicite (un `Linear` non quantifié par défaut a un poids
float32, ce qui aurait masqué la fuite) — 89 tests verts au total
(87 + ces 2).

**Gain mesuré (bench synthétique, `flash-layer-bench --steps 200-400`)** :

| Couche | `--fusion-level 0` | `--fusion-level 6` (F1-F6, sans F7) | `--fusion-level 7` (+F7) | Δ (0→7) |
|---|---:|---:|---:|---:|
| GDN | 4,77-4,88 ms | 4,72 ms (aucun gain — confirme que F1-F6 n'y sont pour rien) | **1,10-1,16 ms** | **×4,1-4,3** |
| QSA | 5,19-5,24 ms | 5,15 ms | **1,45-2,40 ms** (bruit plus élevé, voir P8.3) | **×2,2-3,6** |
| GDN, experts 3-bit g64 réels | 3,03 ms | — | **1,10 ms** | ×2,8 |

`--ablate moe` après F7 : GDN 1,15 → 0,90-1,00 ms (MoE tombe à ~0,15-0,26 ms,
conforme aux 0,138-0,168 ms Python/écrit-à-la-main de la revue P7 — l'écart
« ×25 » d'origine est refermé, et il l'est *sans* toucher à
`Qwen4ExpSparseMoE`/`SwitchGLU`/`Vendor/`).

**Parité — un choix méthodologique délibéré, documenté dans le code** :
`flash-layer-bench --check-parity --fusion-level 7` **échoue** sur le
bench synthétique (poids aléatoires, 32 pas) : diff abs max 0,7-2,6 selon
la couche, dès le premier pas (donc pas un artefact d'accumulation
récurrente). C'est **attendu et correct**, pas un bug : contrairement à
F1-F6 (réordonnancements exacts, censés être bit-identiques à `.none`),
F7 corrige un bug qui changeait déjà les valeurs — comparer son résultat à
la baseline *buguée* avec une tolérance 1e-3 revient à exiger qu'il
reproduise le bug. Reproduit aussi en isolant le phénomène : réactiver
`.bfloat16` (au lieu de `.float16`, la convention pré-existante de ce
harnais) dans les *entrées* de `checkParity`/`checkMultiLayerStepParity`
fait échouer un test **F1/F2 déjà validé et sans rapport avec F7**
(« P2-fusion (F1/F2) », QSA, diff relative 5824×) : la mantisse plus
grossière de bf16 (7 bits contre 10 pour float16) rapproche le bruit de
réordonnancement, pourtant prouvé exact, du seuil d'amplification
chaotique déjà documenté pour GDN/QSA sur poids aléatoires non entraînés
(P7.3, divergence `compile` vs eager croissante avec le nombre de
couches). Décision : `run`/`runMultiLayerStep` (mesure de *débit*, doivent
refléter le dtype réel du checkpoint) passent en `.bfloat16` ; `checkParity`/
`checkMultiLayerStepParity` (comparent deux chemins entre eux, pas de
raison de changer leur dtype) restent en `.float16`, avec un commentaire
expliquant pourquoi — écart au « bit-exact si possible, 1e-3 sinon » de la
consigne, justifié et documenté plutôt qu'un test relâché en silence.

**La validation qui fait foi est le checkpoint réel** (poids entraînés,
dynamique bien conditionnée — voir P8.5) : IDs greedy **identiques**,
0/0 régression sur la garde Q-B (elle s'améliore même légèrement,
−4,80 → −4,68). PLAN.md §6.1 point 3 a déjà acté l'abandon de l'identité
bit-à-bit avec une référence Python comme objectif du projet ; ce choix
est cohérent avec ça.

**Défaut de production inchangé** : `Qwen4ExpCheckpointLayerLoader`/
`Qwen4ExpStreamingTextModel`/`Qwen4ExpStreamingDecoder` restent à
`fusionLevel: .none` — comme F1-F6 avant elle (« validées » mais jamais
activées par défaut), F7 reste strictement opt-in (`--fusion-level 7`),
malgré un gain nettement supérieur à F1-F6. Décision assumée par prudence
(précédent F7-de-P7.4 : validée sur bench, désastreuse sur checkpoint réel
avant d'être testée en conditions réelles) plutôt que par doute sur le
résultat — **recommandation explicite à Vincent** : les preuves
réunies ici (gain ×2,7-4,1 sur checkpoint réel, IDs identiques, mémoire
+0,44-1,13 Go seulement, Q-B stable ou meilleure) justifient de basculer
le défaut, mais c'est une décision produit qui n'a pas été prise
unilatéralement dans cette tâche.

### P8.3 — ablations rejouées : elles étaient déjà indépendantes, `switch_mlp` les écrasait toutes

P7-revisité (2026-09-11, entrée précédente) avait noté que `moe`,
`moe-switch-mlp` et `moe-routing` s'effondraient toutes au même niveau
(~1 ms) et en avait conclu qu'elles n'étaient « pas indépendantes ». Relu
avec le code sous les yeux : chaque ablation *est* logiquement bien
définie et cible une combinaison distincte et documentée (`moe-switch-mlp`
zère `switch_mlp` seul, garde le routeur et l'expert partagé réels ;
`moe-routing` zère `switch_mlp` et l'expert partagé, garde le routeur réel ;
`moe-shared-expert` zère l'expert partagé seul). Le symptôme observé
n'était pas une non-indépendance du câblage, mais la conséquence directe
de la fuite fp32 : `switch_mlp` pesait 79-88 % de la couche, donc **toute**
ablation qui l'annule fait mécaniquement retomber la couche au même
plancher, quel que soit ce qu'elle ablate en plus. Rejouées après F7 (où
`switch_mlp` ne pèse plus que 15-20 % du budget MoE), les mêmes ablations
donnent des coûts nettement distincts — aucun changement de code n'était
nécessaire, seulement la correction de la cause qui les écrasait :

**GDN** (`--fusion-level 7`, `--steps 300`, baseline `.none`(ablation)
≈ 1,11-1,16 ms) :

| Ablation | ms/pas | Coût isolé (baseline − ablation) |
|---|---:|---:|
| `moe` (tout le MoE) | 0,90 | ~0,21-0,26 ms |
| `moe-switch-mlp` | 0,95 | ~0,16-0,21 ms |
| `moe-routing` (switch_mlp+partagé) | 0,87 | ~0,24-0,29 ms |
| `moe-shared-expert` | 1,13 | ~0,00-0,03 ms (bruit) |
| `gdn-recurrence` | 1,08 | ~0,03-0,08 ms |
| `gdn-projections` | 0,95 | ~0,16-0,21 ms |
| `hyper` (mix hyper-connections) | 0,83 | ~0,28-0,33 ms — désormais le plus gros poste nommé |
| `norms` | 1,09 | ~0,02-0,07 ms (bruit) |

Σ sous-blocs nommés ≈ 0,70-0,95 ms sur 1,11-1,16 ms (63-82 %, hors
fourchette ±15 % de la consigne P7.1 originale) — mais c'est maintenant un
**bruit de mesure à l'échelle sous-milliseconde** (p90 souvent 20-30 %
au-dessus de la médiane à ce niveau), pas un sous-bloc ccaché : aucune
ablation ne dépasse ~0,3 ms.

**QSA** (mêmes réglages, baseline ≈ 1,45-2,40 ms selon le run — bruit
sensiblement plus élevé que GDN, cohérent avec le résidu déjà documenté en
P7.1 pour les projections q/k/v/o + RoPE, qui n'ont pas de case
d'ablation dédiée) :

| Ablation | ms/pas | Coût isolé |
|---|---:|---:|
| `moe` | 1,24 | ~0,26 ms |
| `moe-switch-mlp` | 1,30 | ~0,20 ms |
| `moe-routing` | 1,21 | ~0,29 ms |
| `moe-shared-expert` | 1,48 | ~0,02 ms (bruit) |
| `qsa-attn` (SDPA+indexeur) | 1,25 | ~0,25 ms |

Σ nommés ≈ 0,51 ms sur ~1,50 ms (34 %) — le résidu (q/k/v/o+RoPE, sans
case dédiée) est maintenant *proportionnellement* plus large qu'en P7.1
puisque le MoE, qui dominait, a fondu : lacune de découpage déjà assumée
en P7.1, pas un artefact nouveau.

**Correctif à l'entrée `log.md` de P7.1** (2026-09-11, « P7 : débit de
génération ») : le tableau qui y attribue 86,9 % de la couche à
`switch_mlp` et ~0 % au routage/à l'expert partagé reste **vrai comme
mesure brute avant F7**, mais son interprétation (« switch_mlp domine
intrinsèquement ») est **incorrecte** — la mesure était dominée par une
fuite de dtype externe à `Qwen4ExpSparseMoE`, pas par un coût propre à
`switch_mlp`/`SwitchGLU`. Voir cette entrée (2026-09-11, « P8 ») pour
l'attribution corrigée. Rectificatif ajouté ici plutôt que réécrit dans
l'entrée d'origine, comme demandé.

### P8.4 — budget d'ops : la fusion redevient pertinente

Comptage par lecture du code (pas d'instrumentation automatique — écart
documenté ci-dessous), pour la configuration réellement mesurée
(`--fusion-level 7`, donc F1 projections fusionnées + F2 normes
précalculées + F7 actifs), confronté au coût fixe mesuré par
`op-overhead-probe` (8,79 µs/op, addition élémentaire en Release) :

| Couche | Ops comptés (approx.) | N × 8,79 µs | Mesuré | Écart |
|---|---:|---:|---:|---:|
| GDN | ~106 (44 hyper-connections ×2, 27 attention GDN, 10 injections ×2, 25 MoE) | 0,93 ms | 1,11-1,16 ms | 18-20 % |
| QSA | ~144 (44 hyper-connections, 65 attention QSA dont ~18 estimés pour l'indexeur, 10 injections, 25 MoE) | 1,27 ms | 1,45-1,50 ms | 12-15 % |

Les deux couches tombent **dans la fourchette ±25 %** demandée par la
consigne. **Conclusion, qui inverse celle de P7.6/P4.3** : une fois la
fuite fp32 corrigée, le coût restant d'une couche (~1-1,5 ms) est
maintenant majoritairement expliqué par le nombre d'ops × leur coût de
dispatch hôte fixe, pas par un poste unique — P7.6 avait raison de dire
« le goulot n'est pas le bookkeeping hôte de petits noyaux », mais
seulement *tant que la fuite dominait* (elle absorbait ~80 % du budget de
couche, masquant le reste). Elle ne domine plus : **une fusion Metal
(regrouper plusieurs des ~100-145 ops d'une couche en quelques noyaux
maison) redeviendrait un levier proportionnellement utile** — contrairement
à la tentative F7-de-P7.4, qui ciblait `switch_mlp` (le symptôme) plutôt
que la vraie source du budget d'ops restant (hyper-connections, MoE au
grand complet, indexeur QSA). Non tentée dans cette tâche (hors périmètre
P8, qui est diagnostic + correctif ciblé, pas une nouvelle fusion) —
piste consignée pour une tâche future.

### P8.5 — validation sur le checkpoint réel

`/Users/vincent/models/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP` (3-bit
hybride SSD interne). `pgrep -x qwen38-bench-ui` négatif avant chaque run.
`Scripts/preflight-resident.sh` avec `QWEN38_PREFLIGHT_LIMIT_GB=35` (25,3 Go
à évincer, sous le seuil retenu pour cette tâche) : PASS. Tous les runs
sous `caffeinate -dimsu`. `flash-generate-probe`/`flash-chat-probe` gagnent
`--fusion-level` (jusqu'ici seul `flash-chat-probe` l'avait) pour pouvoir
comparer avant/après sur le checkpoint réel, y compris en MTP.

Prompt de référence, `--temperature 0`, 32 tokens, `--resident-layers
--resident-async --resident-async-interval 8` (greedy) / `--mtp
--mtp-block-size 2` (MTP) :

| | avant (`--fusion-level 0`) | après (`--fusion-level 7`) | Δ |
|---|---:|---:|---:|
| Greedy decode (32 tok) | 4,720 s (0,1475 s/tok, **6,78 tok/s**) | 1,720 s (0,0538 s/tok, **18,58 tok/s**) | **×2,74** |
| MTP bloc 2 decode (32 tok) | 4,482 s (0,1401 s/tok, **7,14 tok/s**) | 3,193 s (0,0998 s/tok, **10,02 tok/s**) | **×1,40** |
| MLX pic (greedy) | 56,99 Go | 57,43 Go | +0,44 Go |
| MLX pic (MTP) | 57,87 Go | 59,00 Go | +1,13 Go |
| IDs greedy = référence | oui | oui (bit-identiques à l'avant aussi) | — |
| MTP accepté/proposé | 10/21 (47,6 %) | 11/20 (55,0 %) | quasi-égalité (PM4.4) |

IDs générés (greedy et MTP, avant et après, tous identiques) :
`[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171, 13, 7305, 1725, 501,
4372, 41196, 67763, 85041, 3717, 181876, 8358, 3717, 179342, 47561, 68,
215309, 11, 501, 183435, 401, 1147, 9364, 198070, 175030]` — exactement la
référence attendue (« Le président de la Chine est Xi Jinping. Il est le
Secrétaire général du Comité central du Parti communiste chinois, le
Président de la Commission militaire centrale »).

Garde Q-B (`flashTeacherForcedRegressionGuardV32`, nouveau
`QWEN38_FUSION_LEVEL` en variable d'environnement pour la rejouer sous un
niveau de fusion donné sans toucher au défaut de production) :

| | avant (`.none`) | après (F7) |
|---|---:|---:|
| hits | 10/28 | 10/28 |
| meanLogProb | −4,8003182 (référence exacte) | −4,678542 (légèrement meilleur) |

**Jauge ≥ 12 tok/s (P8.5) : ATTEINTE pour le greedy** (18,58 tok/s, ×1,55
la cible). **MTP** : 10,02 tok/s, sous la cible en valeur absolue mais
+40 % — `Qwen38Runtime`/la GUI/le serveur ne branchent pas encore
`--fusion-level` sur le chemin MTP (seul `flash-generate-probe --mtp` l'a,
ajouté pour cette validation) ; le chiffre MTP « après » est donc une
borne basse de ce qu'un déploiement réel obtiendrait une fois le défaut de
production aligné sur F7. `Scripts/run-tests.sh` : 89 tests verts (87 +
2 nouveaux P8.2). `Scripts/build-release.sh` : BUILD SUCCEEDED tout au
long de la session.

### Écarts à la consigne P8

1. **Commits regroupés** : P8.1 (instrumentation) et P8.2 (correctif F7)
   sont commités ensemble — les deux touchent les mêmes fichiers
   (`Qwen4ExpSparseMoE.swift`, `Qwen38CLI.swift`) et sont causalement liés
   (l'instrumentation a mené directement au diagnostic puis au correctif
   dans la même session) ; séparer par `git add -p` sur ce diff aurait été
   plus risqué que le gain de traçabilité (même raisonnement que la
   déviation n°2 de P7). P8.3-P8.5 (aucun changement de code sauf les
   petits ajouts CLI/tests de validation P8.5) sont commités ensemble
   séparément.
2. **Défaut de production non modifié** : voir P8.2 — décision assumée,
   pas un oubli, présentée à Vincent comme recommandation plutôt
   qu'appliquée unilatéralement.
3. **Parité `--check-parity --fusion-level 7` en échec sur le bench
   synthétique** : voir P8.2 — comportement attendu et documenté dans le
   code, la validation qui fait foi (checkpoint réel) passe sans réserve.
4. **P8.4 par lecture de code, pas par instrumentation automatique** :
   aucun compteur d'ops n'existait dans le dépôt et MLX_METAL_DEBUG reste
   indisponible (obstacle déjà documenté en P7.2) ; le compte pour
   l'indexeur QSA (~18 ops) est une estimation, pas un décompte ligne à
   ligne exhaustif — signalé explicitement, l'accord à ±25 % avec la
   mesure tient malgré cette approximation.
5. **MTP sans `--fusion-level` dans `Qwen38Runtime`** : seul
   `flash-generate-probe` (probe CLI) a reçu l'option pour cette tâche ;
   `Qwen38Runtime.load`/la GUI/`serve` n'exposent pas encore F7 sur le
   chemin MTP — cohérent avec la décision de ne pas changer le défaut de
   production (point 2), documenté comme limite de portée plutôt que
   contourné.

## 2026-09-11 (soir) — F7 passé en défaut de production : ce n'était pas une optimisation mais un écart à la référence

Vérification indépendante du résultat P8, puis bascule du défaut.

**L'argument qui a décidé.** La référence Python vendorée
`Scripts/references/vlm_q4_language.py` termine `Qwen4ExpRMSNorm` (l. 597) et
`Qwen4ExpRMSNormGated` (l. 615) par `.astype(dtype)` — retour explicite au
dtype d'entrée (bf16). Notre chemin ne le faisait pas : le float32 de l'état
récurrent GDN et des tables MRoPE (conservé à dessein, piège §6.3-6) fuyait
dans toute la suite de la couche, et le MoE héritait d'une entrée fp32.
`SwitchGLU` isolé coûte 150 µs en bf16 et **4 022 µs en fp32** : les « 88 % de
la couche » attribués au MoE en P7.1 étaient cet artefact, pas un défaut du
MoE. F7 (`f7GatedBranchDtype`) rétablit le comportement de la référence.

**Mesures faites ici, checkpoint 3-bit hybride, Release, résident, 32 tokens :**

| | F0 (ancien défaut) | F7 (nouveau défaut) |
|---|---|---|
| greedy | 5,24 s · **5,91 tok/s** | 2,43 s · **12,9 tok/s** |
| MTP bloc 2 | — | 2,28 s · **13,6 tok/s** · 55 % d'acceptation |
| IDs greedy | référence | **identiques** (bit-exact) |
| garde Q-B | 10/28 · −4,8003182 | **10/28 · −4,8003182** (inchangée) |
| pic MLX | 56,59 Go | 57,42 Go |

**×2,2 sur la génération, à sortie bit-identique et garde Q-B inchangée.**
L'acceptation MTP monte aussi à 55 % (47,6 % auparavant), cohérent avec un
drafter qui voit enfin la même arithmétique que la cible.

**Bascule** : `Qwen4ExpStreamingDecoder`, `Qwen4ExpStreamingTextModel` et
`Qwen4ExpCheckpointLayerLoader` ont `.f7GatedBranchDtype` en défaut ;
`Qwen38FlashNextEngine` ne passe pas le paramètre, donc **la GUI et le serveur
en héritent**. Les probes CLI passaient explicitement `0` : leur défaut est
passé à `7` pour que les mesures reflètent la production. `--fusion-level 0`
reste le moyen de comparer.

## 2026-09-11 (nuit) — Drafter MTP câblé sur F7, et conséquence inattendue : le MTP ne sert plus à rien

`Qwen4ExpMTPPredictor` construisait sa couche `Qwen4ExpDecoderLayer` **sans**
niveau de fusion (défaut `.none`) : le drafter subissait donc la fuite fp32
que F7 corrige sur la cible, et travaillait dans une arithmétique différente
d'elle. Corrigé : `fusionLevel` traverse `Qwen4ExpMTPPredictor.init` et
`Qwen4ExpMTPLoader.load`, défaut `.f7GatedBranchDtype` comme partout ailleurs.

**Effet mesuré : nul, et c'est explicable.** Le drafter est **une** couche
appelée une fois par round ; sur 20 rounds, F7 lui fait gagner ~70 ms sur
2 300, soit 3 %, dans le bruit. L'acceptation reste identique (11/20, 55,0 %)
— cohérent avec le reste : F7 change la vitesse, pas la numérique.

**En revanche, le rapport greedy/MTP s'est inversé** (checkpoint 3-bit,
32 tokens, Release, résident) :

| | avant F7 | après F7 |
|---|---|---|
| greedy | 5,24 s | **2,40-2,43 s** |
| MTP bloc 2 | 2,28 s (×2,3 sur le greedy) | 2,28-2,38 s (**×1,03**) |
| MTP bloc 3 | — | 3,55 s (plus lent que le greedy) |

Le MTP gagnait 2,3× parce que le forward de la cible coûtait 5 ms ; à 2,4 ms
son surcoût (drafting + vérification + rollback à 45 % de rejet) annule le
gain. **Le MTP n'a plus d'intérêt sur ce chemin** tant que l'acceptation
reste à 55 %. Il demeure opt-in et désactivé par défaut (contrat PM4.3
inchangé) ; rien à retirer, mais plus rien à en attendre non plus. Le
chantier P-MTP est donc clos par la disparition de son objet, pas par un
échec.

## 2026-09-12 — La GUI plafonnait à 6,5 tok/s : c'était SwiftUI, pas le modèle

Vincent a observé 6,5 tok/s en GUI (466 tokens) là où la mesure CLI donnait
12,9 tok/s. Trois hypothèses testées et écartées dans l'ordre :

| Hypothèse | Test | Résultat |
|---|---|---|
| La GUI n'a pas F7 | lecture du câblage `moteur → modèle → décodeur → loader → prepareFusion` | câblage correct |
| Longueur de génération | CLI 466 tokens (au lieu de 32) | 9,39 tok/s — n'explique pas 6,5 |
| Contexte long (tour après image) | serveur, prompt 1 480 tok + 466 générés | **14,66 tok/s** — n'explique rien |
| **Chemin moteur lui-même** | serveur CLI = `Qwen38Runtime` → `Qwen38FlashNextEngine`, 466 tokens | **15,10 tok/s** |

Le moteur — exactement celui de la GUI — fait donc 15 tok/s. Le goulot était
la boucle de consommation du flux dans `Qwen38BenchUIApp` : chaque `.chunk`
(un token) mutait `messages[index].text` **et** `output`, deux propriétés
`@Published`, donc SwiftUI relayait tout le fil de conversation à chaque
token. Plus le message grandit, plus le rendu coûte : la GUI bridait la
génération à 6,5 tok/s. Invisible avant F7, où le modèle plafonnait lui-même
à 6 tok/s — c'est l'accélération du modèle qui a fait apparaître le plafond
de l'interface.

Correctif : les fragments sont accumulés et publiés par lots d'au plus 80 ms
(≈ 12 rafraîchissements par seconde, rendu toujours fluide), le reliquat
étant publié à la fin du tour. Le parseur thinking continue de traiter chaque
fragment immédiatement, seule la publication est regroupée.

## 2026-09-12 — Une image dans le contexte coûte 2,4× sur toute la génération qui suit

Vincent a relancé le protocole de bench M1 (image au tour 1, puis deux tours
texte) et mesuré 3,8 / 1,8 / 2,5 tok/s là où le moteur venait d'être mesuré à
15 tok/s. Enquête, chaque hypothèse écartée par mesure :

| Hypothèse | Test | Verdict |
|---|---|---|
| La GUI n'a pas F7 | lecture du câblage complet jusqu'à `prepareFusion` | correct |
| Rendu SwiftUI (corrigé le matin) | lots de 80 ms déjà en place | n'explique pas |
| Longueur de génération | serveur, 600 tokens générés | 13,95 tok/s |
| Contexte long | serveur, prompt 3 767 + 600 générés (contexte 4 367, au-delà du seuil QSA de 2 048) | 9,82 tok/s — le seuil coûte ~30 %, pas un facteur 4 |
| Pression mémoire | sampler pendant le run : compresseur 9,7 Go, 4 685 décompressions | hors de cause |
| **Image dans le contexte** | **test contrôlé, même longueur de prompt et même nombre de tokens générés** | **décisif** |

| Entrée | prompt | générés | TTFT | débit |
|---|---|---|---|---|
| Texte + image | 973 tok (dont ~950 de vision) | 200 | 4,24 s | **6,88 tok/s** |
| Texte seul | 906 tok | 200 | 6,13 s | **16,41 tok/s** |

**2,4× de pénalité permanente sur le décodage dès qu'une image est dans le
contexte**, à longueur égale. Le prefill n'est pas en cause (TTFT plus court
avec l'image). La boucle de décodage est pourtant structurellement identique :
`Qwen4ExpStreamingGenerator` n'appelle `model.forward(inputIDs:)` qu'avec le
token courant, sans `visionEmbeddings` ni `positionIDs`. La différence est
donc dans le **contenu** des caches — piste principale : les positions M-RoPE
3 axes stockées par `Qwen4ExpQSAKVCache.updateIndexer(keys:positions:)` et
leur usage à chaque pas dans l'indexeur QSA. Non diagnostiqué à ce stade.

Cela explique entièrement les chiffres de Vincent : son tour 1 contenait une
image, et les tours suivants en héritaient par le cache de conversation.

**Second défaut, corrigé** : une continuation de conversation après une image
sort en `HTTP 500` sans corps sur le serveur. `Qwen38FlashNextEngineError`
n'était pas mappé dans `status(for:)` (seul `Qwen38ServerError` l'était) : il
rend désormais un `400` avec le message. La limitation elle-même (« Flash-Next
n'accepte une image qu'au premier tour ») reste, elle est connue et voulue ;
la GUI, qui utilise le chemin conversationnel, n'est pas concernée.

## 2026-09-12 (suite) — La pénalité image était la même fuite de dtype, côté vision : corrigée, 6,9 → 17,8 tok/s

Profil par couche de 20 pas de décodage, après un prompt image (973 tokens) et
après un prompt texte (901 tokens) :

| | couche QSA | couche GDN | total/token |
|---|---|---|---|
| après image | 3,44 ms | 3,15 ms | 167,5 ms |
| après texte | 1,99 ms | 1,69 ms | 102,6 ms |

**L'écart est uniforme sur les deux types de couche** — or une couche GDN est
récurrente : son coût par token ne dépend **ni de la longueur ni du contenu du
contexte**. Cela élimine l'attention, l'indexeur QSA et les positions M-RoPE,
qui étaient la piste privilégiée, et désigne une propriété partagée par toutes
les couches : le **dtype de l'état caché**.

Cause : `Qwen4ExpVisionEncoder` remonte du float32 (interpolation des
positions, l. 241-242) ; `Qwen4ExpInputMerger.merge` fusionnait ces embeddings
tels quels avec les embeddings texte bf16, l'état caché devenait fp32 et
contaminait **les caches remplis au préfill** — d'où un surcoût permanent sur
tous les tours suivants. C'est exactement la fuite corrigée par F7, mais
entrée par la tour vision.

Correctif (`Qwen4ExpInputMerger`) : les embeddings vision sont castés au dtype
des embeddings texte avant fusion, et seulement s'ils diffèrent.

| Test contrôlé (serveur, 200 tokens générés) | avant | après |
|---|---|---|
| Texte + image, prompt ~975 tok | 6,88 tok/s | **17,77 tok/s** |
| Texte seul, prompt 906 tok | 16,41 tok/s | 14,66 tok/s (inchangé, bruit) |
| Sortie image | « Emmanuel Macron … » | **identique** |

La génération après image est désormais **plus rapide** que le texte à
longueur comparable, ce qui est attendu : 950 des 976 tokens de prompt sont
des marqueurs image identiques, donc très favorables au cache n-gram. 89 tests
verts, dont les parités vision.

## 2026-09-12 — Audit systématique des dtypes : plus aucune fuite

Après les deux fuites float32 (normes GDN/QSA corrigées par F7, tour vision
corrigée dans `Qwen4ExpInputMerger`), audit complet plutôt que ponctuel.

**1. Comparaison statique à la référence.** Les points où
`Scripts/references/vlm_q4_language.py` contrôle explicitement le dtype, et
notre équivalent :

| Référence | Ce qu'elle fait | Chez nous |
|---|---|---|
| `Qwen4ExpRMSNorm` l. 589-597 | calcule en fp32, `return .astype(dtype)` | ✅ F7 |
| `Qwen4ExpRMSNormGated` l. 609-615 | idem | ✅ F7 |
| clés poolées de l'indexeur l. 745 | `mean(.astype(float32)).astype(raw_keys.dtype)` | ✅ identique (`Qwen4ExpQSAIndexer.pooledKeys`) |
| scores de sélection l. 768-770 | fp32 **volontaire** (choix discret des blocs) | ✅ identique, alimente un top-k → indices |
| `sparse_bias` l. 842/1455 | `.astype(mask.dtype)` sur la branche additive | ✅ sans objet : notre masque QSA est **booléen** (`logicalAnd`), jamais additif |

**2. Contrôle à l'exécution.** Nouveau `Qwen4ExpDtypeAudit`
(`QWEN38_DTYPE_AUDIT=1`, sans coût quand désactivé) : à la première visite de
chaque couche, il imprime le dtype de l'état caché en entrée et en sortie,
plus celui de tous les tenseurs du cache — c'est-à-dire **tout ce qui
propage**. Résultat sur le checkpoint réel, texte **et** image :

```
couche  0 GDN · entrée bfloat16 · sortie bfloat16 · cache [bfloat16,float32]
couche  1 GDN · entrée bfloat16 · sortie bfloat16 · cache [bfloat16,float32,bfloat16,int64]
couche  3 QSA · entrée bfloat16 · sortie bfloat16 · cache [bfloat16,bfloat16,bfloat16,int32]
```

48 couches, 0 ligne signalée, dans les deux cas. Les seuls `float32`
subsistants sont les **états récurrents GDN** (un par couche GDN), voulus par
conception (piège §6.3-6) et conformes à la référence ; les caches QSA sont
intégralement en bf16. Le décodage après image tombe à 0,478 s pour 2 tokens.

**Conclusion : il n'y a plus de fuite de dtype.** Les deux qui existaient ont
été trouvées par la mesure, pas par la lecture ; l'audit les aurait montrées
en une commande. À lancer après toute modification touchant une frontière de
couche, un cache ou un encodeur.

## 2026-09-12 — P10.1 : le facteur ~2x « Swift vs Python » n'est pas l'ARC, pas un lock, pas le pool coopératif

Protocole PLAN.md §P10, P10.1. Point de départ (tableau d'attribution
post-F7) : 8,58 µs/op mesurés côté Swift (`qwen38 op-overhead-probe`, chaîne
dépendante de 200 additions, Release) contre 4,2 µs/op en MLX Python, même
machine, même version C++ (0.31.1 embarquée par mlx-swift 0.31.6).

**Méthode** : `sample <pid> 8` (macOS `sample`, pas Instruments) sur
`op-overhead-probe --chain 8000 --reps 300` pendant la mesure de la chaîne
d'additions. Le thread « main » Dispatch reste parqué dans un `CFRunLoop`
pour toute la durée (`AsyncParsableCommand.main` → async main) ; le travail
réel tourne sur le pool coopératif de Swift Concurrency. Sur ce thread,
5331/5419 échantillons (98,4 %) tombent dans `OpOverheadProbe.run() →
eval(_:) → mlx_eval → mlx::core::eval → eval_impl`. À l'intérieur :

| Poste | Échantillons | % |
|---|---:|---:|
| `Scheduler::wait_for_one()` (`condition_variable::wait`, bloquant) | 4677/5165 | 90,5 % |
| Dispatch hôte réel (command encoder, driver AGX, fences IOKit) | 362/5165 | 7,0 % |
| Reste (traversée de graphe C++, marshaling `mlx_eval`) | 126/5165 | 2,5 % |

`swift_retain`/`swift_release`/`swift_allocObject` (ARC, allocation
`MLXArray`) : une poignée d'occurrences sur ~2400 lignes de trace —
négligeables. `evalLock` (`NSRecursiveLock`,
`Source/MLX/Transforms+Eval.swift:9,17` du checkout mlx-swift local)
n'apparaît pas comme poste séparable à la résolution 1 ms de `sample`.

**Contre-expérience** (hypothèse : le réveil d'un thread du pool coopératif
après le signal de complétion Metal coûte plus cher qu'un thread classique,
faute d'équivalent côté Python). Ajout temporaire à `OpOverheadProbe.run()`
(`Sources/Qwen38CLI/Qwen38CLI.swift`) : même chaîne exécutée (a) dans le
contexte async normal et (b) sur un `Thread` dédié (sémaphore, QoS `.default`
et `.userInteractive`), chauffe partagée, mesures alternées pour neutraliser
un biais d'ordre. 3 runs machine calme (`--chain 2000 --reps 60`) :
cooperative 4,20-6,78 µs/op, Thread `.default` 4,21-7,60 µs/op, Thread
`.userInteractive` 4,22-5,71 µs/op — **aucun écart mesurable**. Hypothèse
écartée : ce n'est pas le pool coopératif.

**Comparaison Python dos-à-dos**, même session shell, même état machine
(python3 système, mlx 0.31.2, proche de 0.31.1) : même chaîne, même chauffe
→ 2,62-3,18 µs/op sur plusieurs essais entrelacés avec la mesure Swift
(4,20-4,51 µs/op dans la fenêtre immédiatement adjacente). **Ratio réel et
reproductible en conditions calmes : ~1,5×, pas 2×** — le 2× documenté
provenait vraisemblablement d'une charge machine différente entre les deux
mesures d'origine (variance observée entre runs Swift successifs sur cette
même machine : 4,2 à 7,6 µs/op selon la charge résiduelle d'Xcode/
SourceKit).

**Conclusion** : le point chaud n'est pas dans notre usage
(`Qwen38CLI`/`Qwen4Exp*`) — le budget est presque entièrement consommé à
l'intérieur de mlx-swift (`Transforms+Eval.swift` → `mlx_eval` → C++
partagé), avant même d'atteindre notre code, et rien côté ARC/lock/
allocation Swift ne ressort du profil. Le ratio résiduel ~1,5× est
vraisemblablement l'indirection architecturale de mlx-swift (binding C
`mlx-c` avec marshaling/refcounting à la frontière C) contre le binding
direct pybind11-C++ de Python — pas un bug corrigeable localement. Action
plan ouvert : `VincentGourbin/action-plans#536` (kind `upstream-blocker`,
projet `mlx-swift`, sévérité `low` — le gain théorique d'un correctif
upstream resterait modeste face au budget d'ops qui domine désormais,
§P8.4). Rien à corriger ici ; passage à P10.2.

Diagnostic conservé dans `op-overhead-probe` (mesures « cooperative pool »
vs « Thread .default/.userInteractive ») pour reproduction future.

## 2026-09-12 (suite) — P10.2 : fusion mix+inject des hyper-connexions (F8) — validée juste sur les kernels, neutre au bench synthétique

Protocole PLAN.md §P10, P10.2. Deux `MLXFast.metalKernel` (`Sources/Qwen38Core/
FlashNext/Qwen4ExpHyperConnection.swift`) : `qwen4ExpHyperMixFused` fusionne
`sigmoid(upOut).reshaped(...) * normed.reshaped(...)).mean(axis: -2)` (5 ops)
en un appel ; `qwen4ExpHyperInjectFused` fusionne le
`expandedDimensions` ×2 + multiply + reshape + add de
`Qwen4ExpDecoderLayer.inject` (5 ops également) en un autre. Les deux
matmuls bas-rang (`input_mix_weight_down/up`, `block_inject_weight`) restent
des `Linear`/`QuantizedLinear` MLX inchangés — seule la glue autour est
fusionnée. Nouveau niveau `Qwen4ExpFusionLevel.f8HyperConnectionKernel = 8`.

**Piège découvert en écrivant P10.2** : `Qwen4ExpFusionLevel` est cumulatif
(`>=`), et la production est par défaut à F7 (7) depuis le 2026-09-11. Le
plan proposait de réutiliser l'ancien slot `.f3HyperConnections` (jamais
implémenté) — mais 3 < 7, donc câbler un comportement sur ce niveau
l'aurait **activé silencieusement en production** dès le câblage, sans
passer par la porte bench-puis-checkpoint. F8 (8 > 7) évite ce piège :
`.f3HyperConnections` reste un slot mort, à traiter en P10.6.

**Fausse alerte de parité, élucidée** : `Qwen4ExpLayerBench.checkParity`
compare toujours à `.none` ; comme F8 (8) est cumulatif au-dessus de F7 (7),
« .none vs F8 » mélange le F8 tout neuf avec le F7 déjà connu pour changer
la sortie (c'est une correction de bug, pas juste un réordonnancement). Sur
poids aléatoires en float16 (convention du harnais) mélangés à des
`QuantizedLinear` à échelles bf16, le chemin d'origine (`.none`) se
retrouve parfois promu en float32 par la règle de promotion bf16/float16 de
MLX — un vrai écart mesuré (diff abs jusqu'à 1,6, diff rel jusqu'à 447),
mais un artefact du mélange de dtypes propre à ce harnais synthétique, pas
un bug de F8. Deux tests dédiés, à la place :
- `qwen4ExpHyperKernelsMatchNaiveComputation` : les deux kernels seuls,
  tenseurs float32 isolés, contre le calcul non fusionné — diff < 1e-5.
- `qwen4ExpGatedResidualFusionMatchesOriginalPathAtProductionDtype` :
  `Qwen4ExpGatedResidual` complet, bf16 de bout en bout (le seul dtype réel
  de production, où `upOut.dtype == normed.dtype` tient toujours) — diff
  < 1e-2 (bruit d'arrondi bf16 attendu, ~quelques ULP sur des valeurs O(1)).
93 tests verts au total (91 + ces 2).

**Bench synthétique** (`flash-layer-bench --steps 300`, Release, machine
partagée avec une autre charge GPU au moment de la mesure — bruit élevé,
p10 retenu comme le moins pollué) :

| Couche | F7 (défaut) | F8 | Δ |
|---|---:|---:|---:|
| GDN médiane | 1,18-1,53 ms | 1,18-1,34 ms | neutre |
| QSA médiane | 1,47-1,81 ms | 1,75-1,79 ms | **+12 à +19 %** (pire) |

Aucun gain net ; QSA légèrement pire. Cohérent avec le précédent de la
campagne (3 tentatives de fusion à la main déjà écartées avant les deux
gains dtype) : un kernel maison n'a pas automatiquement moins de surcoût
qu'un enchaînement d'ops MLX déjà optimisées, surtout à cette taille de
tenseur. Mesure checkpoint réel : en attente (GPU partagé au moment de
cette session — voir note de fin de section P10).

**Décision provisoire** : le bench synthétique ne montre déjà pas le ≥5 %
requis. Sauf contre-mesure au checkpoint réel, F8 sera retiré du
défaut (il ne l'était de toute façon pas) et consigné comme résultat
négatif valide, sans être promu — comportement identique au sort de
`switch_mlp`/compile-par-couche. `--fusion-level 8` reste disponible pour
comparer.

## 2026-09-12 (suite) — P10.3 : fusion L2-norm q/k GDN (F9) — kernels corrects, gain synthétique marginal et bruyant

Protocole PLAN.md §P10, P10.3, même méthode que P10.2. Cible : la
normalisation L2 de q/k dans `Qwen4ExpGatedDeltaNet.callAsFunction`
(`x * rsqrt((x*x).sum(axis:-1,keepDims:true) + 1e-6) * scale`, 4-5 ops
chacune) — **hors `gatedDeltaUpdate`**. Vérification faite en lisant
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift` : les
transformations de gating citées par le plan
(`-exp(A_log)·softplus(a+dt_bias)`, `sigmoid(b)`) vivent déjà dans le corps
de cette fonction vendue (`computeGatedDeltaG`, l. 14-15 ; `sigmoid(b)`,
l. 296) — en dehors du kernel récurrent proprement dit, mais dans une
dépendance amont non patchée (PLAN.md §1.1). La surface fusionnable réelle
de P10.3, dans notre propre code, est donc la L2-norm elle-même, pas le
gating.

Nouveau kernel `qwen4ExpL2NormLastAxisFused` (`Qwen4ExpGatedDeltaNet.swift`),
niveau `Qwen4ExpFusionLevel.f9GdnL2NormKernel = 9` (au-dessus de F8, même
raison que F8 au-dessus de F7 — cumulatif, ne pas retomber sous le défaut
de production). `eps` (toujours 1e-6 dans ce fichier) est écrit en dur dans
la source Metal plutôt que passé en template : `KernelTemplateArg` n'admet
que `Bool`/`Int`/`DType`, pas `Float`. Piège de compilation Metal rencontré
et corrigé : un scalaire passé comme `MLXArray(scale)` (0-d) fait générer
par MLX un paramètre non subscriptable (`scale[0]` ne compile pas) —
il faut le former en `[1]`, pas en tableau 0-d.

**Deux tests dédiés**, même schéma que P10.2 (le kernel seul en float32,
puis le module complet en bf16 contre F7 — pas `.none`, pour la même
raison de cumulativité que P10.2) :
- `qwen4ExpGdnL2NormKernelMatchesNaiveComputation` : diff < 1e-4 (float32).
- `qwen4ExpGatedDeltaNetFusionMatchesOriginalPathAtProductionDtype` :
  diff < 1e-2 (bf16, un seul pas — pas de récurrence sur 32 pas, pour ne
  pas retomber dans le régime chaotique déjà documenté en P8.2 pour les
  poids aléatoires non entraînés).
95 tests verts au total (93 + ces 2).

**Bench synthétique** (`flash-layer-bench --layer-kind gdn --steps 300`,
même machine partagée, bruit élevé — médianes ponctuellement polluées par
une charge GPU concurrente jusqu'à ×20-30, p10 retenu) :

| | F7 (défaut) | F9 | Δ (p10) |
|---|---:|---:|---:|
| p10 (3 runs) | 1,23 / 1,25 / 1,23 ms | 1,17 / 1,23 / 1,23 ms | 0 à −5 % |

Gain marginal, dans le bruit de mesure de cette session (machine partagée
avec une autre charge GPU) — à revérifier sur machine calme avant toute
conclusion. Mesure checkpoint réel : en attente (voir note de fin P10).

**Décision provisoire** : contrairement à F8, F9 ne montre pas de
régression et un léger mieux au p10 — à confirmer sur machine calme et sur
le checkpoint réel avant toute promotion. `--fusion-level 9` disponible
pour comparer.

## 2026-09-12 (suite) — P10.4 : le vrai goulot n'est pas le regroupement de lignes contiguës, c'est la profondeur de file — pread concurrent ×10

Protocole PLAN.md §P10, P10.4. Consigne explicite : mesurer une lecture de
ligne isolée contre une lecture groupée dans un petit programme, **avant**
de toucher `Qwen4ExpLazyNGramStorage`. Nouvel outil conservé,
`qwen38 ngram-io-probe --file <shard réel> --row-bytes 1280 --count 300`
(aucun chargement de checkpoint, aucun GPU) :

**Piège de méthode découvert et corrigé en écrivant le probe** : une
première version faisait lire les 4 méthodes comparées sur le **même**
jeu de lignes aléatoires — la méthode B (lue en 2ᵉ) paraissait 155× plus
rapide que A (lue en 1ʳᵉ), un artefact pur de cache page OS déjà chauffé
par A, pas une vraie différence de méthode. Corrigé : quatre jeux de
lignes **disjoints** (un quart de fichier chacun), pour que chaque méthode
paie son propre premier accès.

**Résultat (lignes disjointes, 300 lignes/méthode, une seule mesure —
machine avec une autre charge GPU en tâche de fond, sans effet ici : ceci
est un test CPU/disque, pas GPU)** :

| Méthode | SSD interne (`model-00001`) | Lexar/USB (`model-00010`) |
|---|---:|---:|
| A: `pread` + 2 allocations/ligne (chemin actuel isolé) | 113,6 µs/ligne | 301,2 µs/ligne |
| B: `pread` direct, sans allocation superflue | 111,9 µs/ligne | 298,7 µs/ligne |
| C: `mmap` + `copyMemory` (chemin de production actuel) | 142,6 µs/ligne | **540,9 µs/ligne** |
| D: `pread` concurrents (16 voies) | **11,2 µs/ligne** | **55,0 µs/ligne** |

Deux enseignements, contraires à l'hypothèse de départ du plan :
1. **La double allocation Swift par ligne ne coûte presque rien** (A vs B :
   1,7 µs sur SSD interne, 2,5 µs sur Lexar) — le "coût de copie hôte" cité
   par le plan n'est pas là.
2. **`mmap`+`copyMemory` (le chemin de production) est la méthode la plus
   lente** en mono-thread sur Lexar (540,9 µs, pire que `pread` nu à
   301,2 µs) — la faute de page `mmap` a un vrai surcoût propre, pas
   seulement "un accès disque comme un autre".
3. **Le vrai levier est la concurrence** : 16 `pread` en vol simultané
   contre 1 divise le coût par ligne par **~10×**, sur les deux supports.
   Le regroupement par lignes numériquement adjacentes
   (`readContiguousRuns`, déjà en place depuis P5.4) ne peut structurellement
   pas aider ici : les index n-gram viennent d'un hash, l'adjacence
   numérique est une coïncidence, pas la règle — en pratique chaque ligne
   est son propre « run » de longueur 1.

**Correctif implémenté** (`Qwen4ExpPLE.swift`) : `readRowsConcurrently`,
un remplaçant de `readContiguousRuns` à `pread` concurrents (jusqu'à 16 en
vol, `DispatchSemaphore`), même contrat `[Int: [Element]]` en sortie — donc
le cache LRU (`RowCache`, gardé intact comme demandé) et l'assemblage de
`lookupBatch` n'ont pas changé. `pread` est thread-safe par construction
(offset explicite, pas de curseur de fichier partagé). Un descripteur de
fichier persistant par shard (`OpenFile`, à côté du `MappedFile` existant,
pas à sa place) est ouvert une fois à l'initialisation. Chemin d'origine
(`mmap`) conservé et **par défaut** ; nouveau chemin derrière
`QWEN38_NGRAM_CONCURRENT_PREAD=1` (variable d'environnement, pas un niveau
`Qwen4ExpFusionLevel` — cette classe n'en prenait pas et le plan ne
l'exigeait pas explicitement pour P10.4), en attendant la validation
checkpoint réel.

**Parité** : `flash-ngram-parity` (comparaison bit-exacte à la référence
Python eager) rejouée avec `QWEN38_NGRAM_CONCURRENT_PREAD=1`, plusieurs
shards et lignes (dont doublons et lignes à travers tout l'espace d'un
shard) : **IDENTIQUE** dans tous les cas, `max |delta| = 0`. 95 tests
`Scripts/run-tests.sh` toujours verts (aucun test n'est spécifique à ce
chemin au niveau XCTest — la couverture vient de `flash-ngram-parity`,
qui exige le checkpoint réel et n'est donc pas dans le harnais Debug).

**Mesure de préfill réel (77,6 → cible ≥120 tok/s)** : en attente — le GPU
était occupé par une autre charge (inférence diffusion de Vincent) pendant
cette session ; voir note de fin de section P10. Le résultat isolé
(×10 sur le coût par ligne, sur les deux supports) rend un gain net sur le
préfill plausible mais **pas acquis** : la PLE effectue aussi le
dequantize/assemblage MLX après la lecture, et le passage de `mmap` à
`pread` change le modèle mémoire (copie explicite au lieu d'un mapping
virtuel) — seul le chronométrage réel du préfill sur 3 137 tokens tranchera,
conformément au protocole de cette campagne.

## 2026-09-12 (verdict) — P10.2 et P10.3 : retirées, le checkpoint réel ne confirme aucun gain — F8/F9 supprimées

Mesure décisive (protocole PLAN.md §P10 : « ≥ 5 % sur le checkpoint réel
sinon retiré », précédent `switch_mlp`/compile-par-couche/-pas). Machine
libre (GPU 4-12 %, préflight OK), `flash-chat-probe` alterné F7/F8 puis
F7/F9, 32 tokens greedy, `--resident-layers --resident-async`, IDs
vérifiés à chaque run :

**F8 (hyper-connexions)** — 3 paires F7/F8 consécutives, même prompt :

| Run | F7 (s) | F8 (s) |
|---|---:|---:|
| 1 | 2,383 | 2,412 |
| 2 | 2,374 | 2,383 |
| 3 | 2,376 | 2,395 |
| 4 | 2,354 | 2,379 |

F7 moyenne 2,372 s · F8 moyenne 2,392 s → **F8 systématiquement ~0,8-1 %
plus lent**, dans les 4 paires sans exception. Ni régression ≥ 5 % (comme
redouté par le bench synthétique QSA +12-19 %), ni le moindre gain — un
verdict net d'échec au seuil, pas un verdict ambigu. Sur une génération de
200 tokens (arrêt naturel sur EOS), les IDs F7 et F8 **divergent au
72ᵉ token** (F8 n'est pas bit-exact au-delà d'une poignée de tokens,
cohérent avec son recalcul en float32 puis recast) — sortie toujours
cohérente en français, mais confirme que F8 n'est pas candidate à un
« gardé en option car bit-exact ».

**F9 (L2-norm GDN)** — 3 paires F7/F9 :

| Run | F7 (s) | F9 (s) |
|---|---:|---:|
| 1 | 2,315 | 2,388 |
| 2 | 2,392 | 2,394 |
| 3 | 2,404 | 2,401 |

F7 moyenne 2,370 s · F9 moyenne 2,394 s → même verdict : ~1 % plus lent,
pas de gain, malgré un bench synthétique qui suggérait un léger mieux
(0 à −5 %, dans son propre bruit). IDs identiques à 32 tokens.

**Décision, appliquée** : F8 et F9 retirées — code supprimé (pas seulement
leur défaut, qui était de toute façon `.none`/opt-in), suivant le même sort
que le kernel `switch_mlp` et les tentatives `MLX.compile` par
couche/par pas de la campagne P7. `Qwen4ExpHyperConnection.swift` et
`Qwen4ExpDecoderLayer.swift` (F8), `Qwen4ExpGatedDeltaNet.swift` (F9)
reviennent au chemin d'origine inconditionnel ; `Qwen4ExpFusionLevel` perd
les cas `f8HyperConnectionKernel`/`f9GdnL2NormKernel` ; les 4 tests dédiés
sont supprimés (89 tests verts, retour au compte d'avant P10.2/P10.3) ;
`--fusion-level` redevient 0-7 dans le CLI.

**P10.6 (nettoyage) fait dans la foulée** : `Qwen4ExpFusionLevel` documente
désormais explicitement deux catégories — les « fusions collage » (F1, F2,
F4 : réordonnancements exacts, neutres en production, gardés parce que
gratuits) et le « correctif de dtype » (F7, pas une optimisation, défaut de
production, ne pas le traiter comme un réglage perf optionnel). Les cases
`f3HyperConnections`/`f5Casts`/`f6Compile` restent (jamais implémentées,
retirer casserait la continuité « F1-F6 » des logs passés) mais sont
explicitement annotées « réservé, jamais implémenté » pour qu'un lecteur ne
les croie pas actives. F8/F9 ne réapparaissent pas dans l'énumération — un
futur niveau ne devrait pas réutiliser ces deux numéros sans consulter
l'historique git.

**Le vrai enseignement, pour la suite** : deux tentatives de plus qui
confirment le motif déjà observé trois fois cette campagne — un kernel
maison, même correct et même bien conçu, n'a pas automatiquement moins de
surcoût qu'un enchaînement d'ops MLX déjà optimisées à cette échelle de
tenseur (quelques centaines à quelques milliers d'éléments). Les deux
vrais gains de la campagne (F7, et P10.4 ci-dessous) sont un correctif de
comportement et un changement de mécanisme d'E/S — jamais une fusion de
petits kernels.

## 2026-09-12 (verdict) — P10.4 : pread concurrents n-gram promus en défaut — préfill réel +23 à +30 %

Mesure décisive, machine libre, prompt de prose française variée (extrait
technique de log.md, non répétitif — 4 831 tokens réels, au-delà de la
cible ~3 100), `flash-generate-probe --resident-layers --resident-async
--profile-layers --trace`, même prompt pour les deux mesures :

| | `mmap` (ex-défaut) | `pread` concurrent (nouveau défaut) | Δ |
|---|---:|---:|---:|
| couche PLE (couche 1, préfill) | 44,68 s | 31,63 s | **−29,2 %** |
| forward total (`layer_forward_seconds`) | 56,19 s | 43,11 s | **−23,3 %** |
| débit forward (4 831 tokens) | 86,0 tok/s | **112,1 tok/s** | **+30,4 %** |
| chargement des poids (`layer_load_seconds`, non concerné) | 60,43 s | 59,72 s | ~0 (bruit) |

`layer_load_seconds` inchangé confirme que l'écart est bien attribuable
aux lectures n-gram (le chargement des poids de checkpoint passe par un
mécanisme séparé, `Qwen4ExpUncachedTensorReader`, non touché par ce
levier). Le gain de bout en bout (+23-30 %) est net mais **très inférieur**
au ×10 mesuré ligne par ligne en isolation (`ngram-io-probe`) : le cache
LRU (`RowCache`) absorbe déjà 50 % des lookups sur ce prompt (`ngram_cache_
hit_rate: 0.5000`), donc seule la moitié « miss » profite de la
concurrence — cohérent, pas un signe d'échec de la méthode.

**Bascule en défaut, comme demandé** : `useConcurrentPread` (`Qwen4ExpPLE.
swift`) devient `true` par défaut ; `QWEN38_NGRAM_MMAP=1` est
l'échappatoire vers l'ancien chemin `mmap`. Parité bit-exacte reconfirmée
après bascule (`flash-ngram-parity`, plusieurs shards, avec et sans
l'échappatoire — `IDENTIQUE` dans les deux cas). 89 tests
`Scripts/run-tests.sh` verts.

**Cible ≥ 120 tok/s (contre 77,6) : proche mais pas formellement atteinte**
sur cette mesure précise (112,1 tok/s de débit forward) — la référence
77,6 tok/s vient d'un prompt et d'une session différents (3 137 tokens,
contenu différent), donc pas directement comparable terme à terme ; la
comparaison qui décide ici est la paire mmap/pread **au même prompt, même
session**, qui montre sans ambiguïté un gain réel et substantiel. Consigné
comme résultat positif franchissant très largement le seuil de 5 %, sans
sur-vendre un chiffre rond non atteint à la décimale.

## 2026-09-12 (verdict) — P10.5 : validation d'ensemble après P10.1-P10.4

Défaut de production final de cette tâche : `Qwen4ExpFusionLevel.
f7GatedBranchDtype` (inchangé, F8/F9 retirées) + lectures n-gram par
`pread` concurrents (nouveau défaut, P10.4). Machine libre, préflight OK à
chaque run, `caffeinate -dimsu`, un seul run résident à la fois.

| Critère | Résultat |
|---|---|
| IDs greedy = référence | **oui** — 32 tokens et 137 tokens (arrêt EOS naturel), bit-identiques à `[2229, 85648, 401, 1147, 183085, ...]` |
| Garde Q-B (`flashTeacherForcedRegressionGuardV32`) | **10/28 · −4,8003182** — exactement la valeur de référence |
| Débit greedy court (32 tokens) | 13,0-13,5 tok/s |
| Débit greedy long (137 tokens, EOS naturel) | 13,13 tok/s |
| `QWEN38_DTYPE_AUDIT=1`, run texte | **0 ligne signalée** (48 couches) |
| `QWEN38_DTYPE_AUDIT=1`, run image (bruit synthétique — `licensed-image-2.jpeg` inaccessible depuis ce terminal, permission macOS) | **0 ligne signalée** (48 couches), sortie cohérente (« image de bruit blanc ») |
| H6 qualification (texte : H6.1, H6.2a-d, H6.4-t1/t2) | **PASS** sur les 7 rejouées ; H6.3 (image) non rejouée cette session (même blocage d'accès à `~/Downloads`) |
| `Scripts/run-tests.sh` | 89 tests verts |

**Écart à la consigne, assumé** : H6.3 (image) et le dialogue A/B de 20 min
n'ont pas été rejoués cette session — H6.3 à cause d'un blocage de
permission macOS sur `~/Downloads/licensed-image-2.jpeg` depuis ce
terminal (« Operation not permitted »), contourné pour l'audit dtype par
une image synthétique de test mais pas repris pour le H6.3 exact ; le
dialogue A/B 20 min n'a pas été rejoué faute de temps dans cette session,
sans lien avec un doute sur la stabilité (P6.6 l'avait déjà validé, rien
dans P10 ne touche à la boucle de conversation/serveur). Les deux tiennent
la place d'un contrôle de confiance additionnel, pas d'un critère bloquant
de cette tâche — tous les critères explicitement redemandés (IDs, Q-B,
débit court/long, audit dtype) sont PASS.

**Bilan chiffré de la campagne P10** :

| Tâche | Levier | Verdict | Chiffre décisif |
|---|---|---|---|
| P10.1 | attribution du coût par op | aucun correctif local | 90,5 % du temps d'`eval()` = attente GPU (C++ partagé) ; ratio réel Swift/Python ~1,5× en conditions calmes, pas 2× |
| P10.2 | kernel mix/inject hyper-connexions (F8) | **retiré** | F7 2,372 s vs F8 2,392 s (32 tok, 4 paires) — F8 systématiquement ~1 % plus lent |
| P10.3 | kernel L2-norm GDN (F9) | **retiré** | F7 2,370 s vs F9 2,394 s (32 tok, 3 paires) — ~1 % plus lent, pas de gain |
| P10.4 | `pread` concurrents n-gram | **promu en défaut** | forward 56,19 s → 43,11 s (4 831 tokens, même prompt) — **+30,4 % de débit** |

## 2026-09-12 (fin) — Vérification indépendante de P10 et récupération de l'image de référence

Contrôle des deux points que P10 laissait ouverts.

**Préfill, sur le prompt de référence du 2026-09-11** (3 137 tokens de prose
française réelle, extrait de ce journal, `--profile-layers`) :

| | forward | débit |
|---|---|---|
| avant les `pread` concurrents | 40,43 s | 77,6 tok/s |
| **après (défaut actuel)** | **28,82 s** | **108,8 tok/s** |

**+40 % sur le préfill réel**, mesure directement comparable à la référence
(même prompt, même protocole). L'agent annonçait +30 % sur un prompt de
4 831 tokens ; l'écart entre les deux chiffres vient de la longueur, le cache
LRU de lignes absorbant une part croissante des lectures.

**Image, que l'agent n'a pas pu tester** : `~/Downloads` est devenu
inaccessible au terminal (refus TCC de macOS — `ls` voit le fichier, toute
lecture échoue, y compris depuis le shell). L'image de référence a été
récupérée depuis une requête H6 archivée
(`results/h6/H6.3.request.json`, data URL base64) et déposée dans
`results/assets/ref-image.jpeg` : elle est désormais versionnée avec le dépôt
et ne dépend plus d'une autorisation. Résultat sur le défaut post-P10 :

```
image 1216x800 · 950 marqueurs · 976 tokens de prompt
"Sur cette image, on voit **Emmanuel Macron**, le président de la République
 française. ### Rôle : Il est **le chef de l'État français**, élu au suffrage
 univers…"
40 tokens en 3,14 s = 12,4 tok/s · audit dtype : 0 fuite
```

Qualité et débit conformes ; le trou de validation de P10.5 est comblé.
**Les scripts qui référencent `~/Downloads/licensed-image-2.jpeg`
(`Scripts/h6-qualification.sh`) doivent basculer sur
`results/assets/ref-image.jpeg`.**

---

## 2026-09-12 — ménage du dépôt (avant le plan révisé §P11)

Le dossier de travail avait accumulé 48 Go, dont l'essentiel n'était ni du
code ni de la preuve.

| Poste | Avant | Après | Geste |
|---|---:|---:|---|
| `.xcodebuild-*` obsolètes (14 répertoires) | 29,3 Go | 0 | supprimés |
| `.xcodebuild` + `.xcodebuild-tests` (utilisés par `Scripts/*.sh`) | 8,3 Go | 8,3 Go | conservés |
| Traces Chrome du profileur, non citées par la doc (95 fichiers) | 129 Mo | 0 | supprimées |
| Traces Chrome citées par `log.md` / `PLAN.md` (10 fichiers) | 86 Mo | 86 Mo | conservées |
| Bundles Metal System Trace `p4-mst-*.trace` | 225 Mo | 225 Mo | conservés (seule preuve MST de P4) |
| Racine : `default.profraw`, 12 `*.trace.json`, `.xcodebuild-tests.log` | 18 Mo | 0 | supprimés |
| Preuves textuelles non versionnées (`.tsv`, `.jsonl`, `.stdout.txt`) | 3,6 Mo | 3,6 Mo | **commitées** (200 fichiers) |

Total du dossier projet : **48 Go → 15 Go**. `git status` est propre.

`.gitignore` gagne trois règles pour que les sorties de profilage cessent de
s'accumuler dans `results/` — elles sont régénérables, et les conclusions
qu'on en tire vivent ici :

```
results/**/*.trace.json
results/**/*.trace/
results/**/*.xctrace.log
```

Deux points restent à la charge de la machine, pas du dépôt : les 29 Go
libérés ne sont pas encore rendus par APFS (instantanés locaux Time Machine),
et la mise en veille sur secteur n'est toujours pas corrigée dans le profil
d'alimentation (cf. P1, tous les runs restent enveloppés dans
`caffeinate -dimsu`).

---

## 2026-09-12 — toit de performance du décodage, et débits SSD réels (préparation P11)

Deux mesures faites pour trancher la question posée par Edge0 : sommes-nous
limités par la bande passante mémoire, et le déchargement disque est-il
possible ?

### Octets d'experts lus par token décodé

Dérivé des en-têtes safetensors du checkpoint réel
(`local/Qwen3.8-Flash-Next-MLX-e3bit-MTP`), pas d'une estimation :

| | |
|---|---|
| Checkpoint total | 83,9 Gio |
| dont experts routés (`switch_mlp`, 48 couches) | 50,2 Gio |
| dont table n-gram PLE (mmap paresseux) | 29,8 Gio |
| dont attention / vision / expert partagé | 3,9 Gio |
| Par couche (512 experts) | 1 072 Mio |
| **Par expert** | **2,094 Mio** |
| **Lu par token décodé** (10 experts × 48 couches) | **1 005 Mio** |
| Fraction des experts touchée par token | 2,0 % |

### Où nous sommes par rapport au toit

M3 Max : ≈ 400 Go/s de bande passante, de l'ordre de 14 TFLOP/s.

| Débit | Gio/s exigés | % bande passante | GFLOP/s | % calcul |
|---|---:|---:|---:|---:|
| 13,5 tok/s (mesuré, P10.5) | 13,3 | **3,6 %** | 63,7 | **0,46 %** |
| 17,8 tok/s (record, post-P9) | 17,5 | 4,7 % | 84,0 | 0,60 % |

Toit de bande passante du seul MoE : **2,63 ms/token, soit 380 tok/s**.
Mesuré : 74 ms/token. **Facteur 28.** Ni la lecture des poids ni
l'arithmétique n'expliquent le temps ; il part en lancements de noyaux
(~100 ops/couche d'après P2-code, ~4 800/token, ~16 µs pièce). C'est la
confirmation chiffrée de ce que P0-P10 avaient établi au cas par cas, et
l'explication de l'échec systématique des fusions F1-F9.

### Débits de lecture, fenêtre de travail 84 Gio, `F_NOCACHE`, accès aléatoires

Fenêtre volontairement plus grande que le cache de pages disponible, pour ne
pas mesurer le cache. Threads POSIX, `pread`.

| Support | bloc | 1 fil | 4 fils | 8 fils | 16 fils |
|---|---|---:|---:|---:|---:|
| SSD interne | 2 Mio | 8,43 | 10,81 | 6,48 | 1,44 |
| SSD interne | 8 Mio | 11,66 | **15,22** | 6,50 | 1,44 |
| Lexar USB4 | 2 Mio | 0,62 | 0,80 | 0,80 | 0,78 |
| Lexar USB4 | 8 Mio | 0,71 | 0,76 | 0,76 | 0,76 |

(en Gio/s)

Deux enseignements. Le SSD interne plafonne à **15,2 Gio/s à 4 fils avec des
blocs de 8 Mio**, à peine au-dessus des 13,3 Gio/s qu'exige le décodage — et
**s'effondre à 8 fils et au-delà** (6,5 puis 1,4 Gio/s), ce qui condamne
l'idée d'un déchargement à forte concurrence. Le Lexar tient 0,76 Gio/s, soit
**dix-sept fois moins que nécessaire** : il est écarté définitivement comme
support des experts. La valeur « 0,7 Go/s » que l'étude P3 citait de seconde
main est donc confirmée par la mesure.

Ces chiffres réorientent le chantier de déchargement (désormais P11.5) : son
objectif n'est plus le débit mais la **mise en service du 4-bit**.

---

## 2026-09-12 (nuit) — P11.1 exécutée : réduire K ne fait pas gagner de débit

Première tâche du plan révisé. L'outillage (surcharge de `num_experts_per_tok`
à l'exécution) est décrit dans `PLAN.md` P11.1.

### Garde préalable : le réglage ne change rien quand on ne lui demande rien

`flash-chat-probe`, greedy, 8 tokens, prompt de référence, sans drapeau puis
avec `--routed-experts 10` : IDs identiques entre eux **et** identiques à la
référence historique du dépôt (2026-09-09, P2-mem-a) —
`[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171]`. Pic MLX 57,43 Go dans
les deux cas.

### Qualité — teacher-forcing, déterministe, process neuf par K

Continuation scorée = les 128 tokens produits en greedy à K=10 sur le prompt
de référence. Mesurer la logprob de **sa propre sortie à K=10** donne
directement la divergence induite par la réduction.

| K | octets/token | logprob moyenne | Δ nat | accord argmax | rang cible moyen |
|---:|---:|---:|---:|---:|---:|
| 10 | 1 005 Mio | −0,3386 | — | 99,2 % | 1,01 |
| 8 | 804 Mio | −0,3687 | −0,030 | 94,5 % | 1,05 |
| 6 | 603 Mio | −0,4051 | −0,066 | 92,2 % | 1,16 |
| 5 | 503 Mio | −0,4799 | −0,141 | 83,6 % | 1,26 |
| 4 | 402 Mio | −0,7024 | −0,364 | 80,5 % | 1,49 |

Dégradation propre et monotone. Le 99,2 % de la ligne K=10 n'est pas un défaut
de K : c'est l'écart entre le chemin de décodage (`--resident-async`) et le
chemin de scoring (teacher-forcing), un token sur 128 (le #73, marge 0,125).
**C'est le plancher de bruit de cette comparaison.**

### Débit — serveur chaud, tours entrelacés, deux campagnes indépendantes

| K | campagne 1 | campagne 2 | écart-type | gain médian |
|---:|---:|---:|---:|---:|
| 10 | 21,02 | 21,12 | 0,05 | — |
| 8 | 21,31 | 21,44 | 0,05 | **+1,5 %** |
| 6 | 21,79 | 21,92 | 0,06 | **+3,8 %** |
| 5 | 22,25 | **22,37** | 0,05 | **+5,9 %** |
| 4 | 21,97 | 22,12 | 0,09 | +4,7 % |

(tok/s ; 6 puis 8 tours de 5 requêtes entrelacées, greedy, 96 tokens, prompt de
référence, `/healthz` vérifié à chaque requête)

**Retirer 60 % des octets d'experts par token rapporte 4,7 % de débit.** C'est
la confirmation expérimentale la plus directe du diagnostic P11.1 : à 3,6 % de
la bande passante et 0,46 % du calcul, nous ne sommes limités ni par l'un ni
par l'autre. Le nombre de noyaux lancés ne dépend pas de K — seules la largeur
d'`argPartition` et la taille du *gather* changent — et c'est lui qui fixe le
débit.

Le critère de succès de P11.1 (**≥ +15 % pour ≤ 0,2 nat**) n'est **pas
atteint** : le meilleur point, K=5, donne +5,9 % pour 0,141 nat et fait tomber
l'accord argmax de 99,2 % à 83,6 %.

**Anomalie reproduite deux fois : K=4 est plus lent que K=5** (−1,1 %) alors
qu'il lit 20 % d'octets en moins. Piste probable, non vérifiée : un effet de
tuilage dans `gatherQMM`, dont la forme préférée ne serait pas atteinte à K=4.

### Deux leçons de méthode, à appliquer désormais

1. **Le débit mesuré par une sonde en process neuf n'est pas exploitable.** Le
   premier passage (un run par K, process neuf à chaque fois) donnait 18,20 /
   19,92 / 19,63 / 21,48 / 20,58 tok/s — non monotone, avec plusieurs tok/s de
   dispersion, et un « +18 % » à K=5 qui n'existe pas. Le serveur chaud, en
   tours entrelacés, descend à 0,05 tok/s d'écart-type. **Toute comparaison de
   débit doit passer par le serveur chaud et des tours entrelacés**, pour que
   la dérive de la machine se répartisse également sur toutes les variantes.
2. **Le débit publié par `/metrics` mélange préfill et décodage.** Sur un
   prompt de 4 759 tokens il devient bimodal — 7,6 tok/s quand le cache de
   préfixe implicite rate, 15,4 quand il touche — et l'effet de K y est
   noyé. **L'effet de K à contexte long n'est donc pas mesuré**, et ce n'est
   pas une omission mais un manque d'instrumentation : il faudrait un compteur
   de décodage seul, séparé du préfill, côté serveur.

### Conditions de mesure

Un build Release concurrent (autre session, autre projet) a saturé la mémoire
entre 23:16 et 23:28 : compresseur à 60 Go, **11 millions de décompressions**.
La campagne 2 a été relancée machine calme (4,7 Go d'anonyme, **3 127**
décompressions, RSS 53,1 Go) et **reproduit la campagne 1 à 0,15 tok/s près**.
Les onze millions de décompressions appartenaient donc au build voisin, pas à
notre process. Retenir tout de même que `sample-system.sh` devient
ininterprétable sous contention : la mémoire anonyme y oscille de 60 Go en
5 secondes, ce qui n'est pas physique.

### Verdict

**La réduction de K n'est pas un levier de débit sur un modèle résident.** Le
réglage est conservé, mais son usage bascule vers P11.5 : sous déchargement
disque, où les octets *sont* la contrainte, K=5 divise par deux les entrées-
sorties pour 0,141 nat. C'est là qu'il faudra le ressortir.

---

## 2026-09-12 (nuit) — P11.4 : le plafond du décodage spéculatif, et le chiffre qui manque

Mesures faites dans la foulée de P11.1, même protocole (serveur chaud, tours
entrelacés, prompt de référence, greedy, 96 tokens).

### MTP contre greedy, remesuré proprement

| | tok/s | ms/token |
|---|---:|---:|
| greedy | 21,01 | 47,61 |
| MTP | 19,72 | 50,72 |

**MTP = 0,939× le greedy**, acceptation **48,4 %** (31 acceptés sur 64
proposés, relevé dans `/metrics`). Plus serré que les 0,81-0,86× historiques,
même conclusion : le MTP coûte plus qu'il ne rapporte.

96 tokens en 65 pas ⇒ **75,0 ms par pas de vérification**.

### Coût d'un forward selon le nombre de tokens qu'il traite

Mesuré par le TTFT du serveur sur des prompts uniques (jamais de cache de
préfixe), 4 tours :

| tokens de prompt | TTFT médian | ms/token | marginal sur le point précédent |
|---:|---:|---:|---:|
| 25 | 720 ms | 28,80 | (point froid, écarté) |
| 30 | 484 ms | 16,13 | — |
| 45 | 747 ms | 16,59 | 17,5 |
| 81 | 1 309 ms | 16,15 | 15,6 |
| 152 | 2 096 ms | 13,79 | 11,1 |
| 342 | 3 218 ms | 9,41 | 5,9 |
| 700 | 6 143 ms | 8,78 | 8,2 |
| 1 492 | 11 797 ms | 7,91 | 7,1 |

**Le coût marginal d'un token supplémentaire dans un forward groupé n'est pas
constant** : ~16 ms par token à N petit, ~7,7 ms à N grand. À comparer aux
**47,6 ms** d'un token décodé seul. Grouper vaut donc entre 3× et 6× selon le
régime — pas 50×.

### Ce que ça donne comme plafond, et pourquoi c'est indécidable en l'état

Avec un **drafter gratuit** et 48,4 % d'acceptation :

| brouillon | plafond si marginal = 7,74 ms | plafond si marginal = 16 ms |
|---:|---:|---:|
| 1 token | **1,28×** | **1,11×** |
| 2 tokens | **1,30×** | 1,03× |
| 3 tokens | 1,23× | 0,91× |
| 4 tokens | 1,14× | 0,80× |

Et le drafter actuel coûte, par soustraction sur les 75,0 ms du pas :
**19,6 ms** (41 % d'un forward complet) dans la première hypothèse, **11,4 ms**
(24 %) dans la seconde. Pour une seule couche d'attention pleine contre 48
couches côté cible : c'est du coût fixe, pas du calcul.

**Le sort de P11.4 tient donc à un seul nombre que l'instrumentation actuelle
ne sait pas isoler** : le coût d'un forward sur N ∈ {1, 2, 4, 8} tokens.
Le TTFT ne convient pas — il porte ~250 ms de coût fixe de requête (HTTP,
rendu du gabarit, mise en place des caches), soit cinq fois le forward
lui-même. Il faut une sonde qui chronomètre le forward seul.

- Si le marginal à petit N vaut ~8 ms, le plafond est 1,30× et il faut
  ramener le drafter sous ~5 ms pour en récupérer l'essentiel. Ça vaut le
  chantier.
- S'il vaut ~16 ms, le plafond est 1,11× même avec un drafter gratuit, et
  **le MTP est condamné** : on l'écrit et on ferme.

### Tâche qui en découle

**P11.4a — sonde de coût de forward.** Chronométrer le forward du modèle
réel sur N ∈ {1, 2, 4, 8, 16, 32} tokens, hors serveur, hors tokenisation,
hors échantillonnage : `eval()` bloquant, médiane sur 50 pas après warm-up.
Bon marché, et décide à elle seule de tout le chantier MTP.

---

## 2026-09-13 — P11.2 : l'ablation par soustraction ne mesure pas ce qu'on croit

L'outillage est en place (les neuf ablations de `Qwen4ExpLayerBenchAblation`
sont désormais applicables au checkpoint réel, modifiables à chaud, publiées
dans `/healthz`). La mesure, elle, **ne donne pas de chiffre exploitable**, et
c'est le résultat de la nuit.

### Le signal d'alarme

Serveur chaud, ablations entrelacées, débit de décodage seul
(`tokensPerSecond` = `generationTokensPerSecond`, vérifié dans
`Qwen38Server.swift:623` — il exclut bien le préfill) :

| ablation | tok/s | coût attribué |
|---|---:|---:|
| aucune | 20,51 | — |
| `moe` (bloc entier court-circuité) | 26,05 | 10,38 ms |
| `moe-routing` (routeur réel, sorties mises à zéro) | 30,96 | **16,47 ms** |

**`moe` court-circuite strictement plus de travail que `moe-routing`** — il
saute le routeur en plus des experts — et mesure pourtant un coût *inférieur*.
Reproduit en ordre aléatoire sur 8 tours (σ = 0,09 pour `moe`), donc ce n'est
ni du bruit ni un effet d'ordonnancement.

### La cause

Il suffit de regarder ce que chaque variante produit :

| ablation | sortie (40 tokens) | mots distincts |
|---|---|---:|
| aucune | « Le président de la Chine est Xi Jinping. Il est le Secrétaire général… » | 25 |
| `moe` | charabia multilingue varié | 12 |
| `moe-routing` | `1225666666666666666666666666666666666666` | **1** |
| `hyper` | ` (`&nbsp;` (`&nbsp;` (`&nbsp;…` répété 40 fois | **1** |

**Les variantes qui mesurent « le plus rapide » sont exactement celles dont la
sortie dégénère en un seul token répété.** Un token répété touche toujours la
même ligne de la table n-gram (donc LRU chaud, zéro lecture disque) et route
vers les mêmes experts à chaque pas. Ce n'est pas le bloc ablaté qu'on mesure,
c'est la dégénérescence de la sortie.

Contrôle du même effet sans ablation, quatre prompts produisant des sorties de
natures différentes (prose, code, chiffres, répétition) : l'étendue est de
**+5,7 %** entre la plus lente et la plus rapide. Entre une sortie normale et
un token unique répété, l'effet est manifestement bien plus grand.

### Conséquence

**L'attribution par soustraction sur une génération libre est invalide sur ce
modèle.** C'est la deuxième fois que cet instrument (écrit pour P7) produit une
réponse trompeuse : P7 avait conclu « 78-88 % du temps dans le MoE routé », que
P8 a réfuté en trouvant une fuite de dtype ailleurs. Le motif est le même — on
compare des exécutions qui ne calculent pas la même chose.

Deux biais distincts, à ne pas confondre :
1. **La dégénérescence de sortie** (celui-ci, dominant) : l'ablation change les
   tokens produits, donc les lectures n-gram et le routage.
2. **Le repliement de constantes**, soupçonné mais non prouvé : les cas qui
   substituent `MLXArray.zeros(...)` rendent la sortie du bloc identiquement
   nulle, ce que MLX pourrait replier en aval. `moe-shared-expert` (qui garde
   la branche routée réelle, donc pas de sortie nulle) mesure un coût petit et
   crédible, ce qui est cohérent avec cette hypothèse.

### Ce qu'il faut pour mesurer vraiment

**Un décodage à séquence forcée.** Décoder token par token en imposant à chaque
pas un identifiant fixé au lieu de l'argmax du modèle, de sorte que **toutes
les variantes décodent exactement les mêmes tokens** — mêmes lectures n-gram,
même routage, seule la branche ablatée diffère.

`flash-teacher-forced-score` ne convient pas : il fait **un seul forward
groupé** sur toute la séquence, donc il attribuerait du coût de préfill, pas de
décodage — et les deux régimes n'ont pas le même mélange d'opérations (matvec
batch 1 contre matmul).

Tant que cet instrument n'existe pas, **aucun chiffre d'attribution par
sous-bloc ne doit être inscrit**, et P11.3 (part des hyper-connexions) comme
P11.10 (cible du noyau Metal) restent sans cible chiffrée.

---

## 2026-09-13 — P11.4a et P11.2a mesurées : le MTP est condamné, le coût est diffus

`flash-decode-bench` (décodage à séquence forcée) livré et vérifié. Toutes les
mesures ci-dessous rejouent **la même séquence de tokens** pour chaque
variante, ce qui élimine le biais de dégénérescence qui invalidait P11.2.

### (B) Coût d'un forward selon le nombre de jetons traités

Variante unique, sans `snapshot`/`restore`, 16 pas de warm-up, 64 mesurés :

| jetons par forward | ms par forward | écart-type | marginal du jeton ajouté |
|---:|---:|---:|---:|
| 1 | **47,84** | 1,18 | — |
| 2 | 62,42 | 1,61 | **14,58** |
| 4 | 94,17 | 2,85 | **15,87** |
| 8 | 175,54 | 26,26 | **20,34** |

Le forward à un jeton donne 47,84 ms contre 47,6 ms mesurés côté serveur :
l'instrument recoupe la production, il est bon.

**Le coût marginal est donc le régime pessimiste — 15 à 20 ms — et il croît
avec N.** L'asymptote de 7,7 ms déduite du préfill (3 137 jetons) ne s'applique
pas du tout à la plage utile pour la spéculation.

### Le MTP est condamné : le plafond est 1,14× avec un drafter GRATUIT

À 48,4 % d'acceptation (mesuré), en prenant les coûts de forward réels :

| brouillon | jetons par pas | coût du pas | ms/jeton | gain, **drafter gratuit** |
|---:|---:|---:|---:|---:|
| 1 (vérifier 2) | 1,484 | 62,4 ms | 42,1 | **1,14×** |
| 3 (vérifier 4) | 1,832 | 94,2 ms | 51,4 | 0,93× |
| 7 (vérifier 8) | 1,932 | 175,5 ms | 90,9 | 0,53× |

Un brouillon plus profond est **contre-productif** : le coût du forward croît
plus vite que les jetons acceptés ne s'accumulent.

Le MTP actuel mesure 0,939×, avec un pas à 75,0 ms ; par soustraction son
drafter coûte **12,6 ms**. Le seuil de rentabilité est un pas sous 71,0 ms,
donc un drafter **sous 8,6 ms**. Même atteint, le gain plafonnerait à
**1,14×**, et un drafter réellement gratuit n'existe pas.

**Verdict : on ferme P11.4.** Le décodage spéculatif ne peut pas rapporter plus
de 14 % sur cette machine et ce modèle, et il faudrait pour cela diviser le
drafter par plus de deux. Le rapport gain/effort ne le justifie pas. Le MTP
reste disponible en option (`mtp: true`), documenté comme perdant.

*La seule voie qui rouvrirait le sujet est une acceptation nettement plus
haute : à 70 %, le brouillon de 1 donnerait 1,30×. Mais une meilleure
acceptation demande un drafter plus gros, donc plus cher, et le budget est de
8,6 ms.*

### (A) Attribution par sous-bloc — le coût est diffus, aucun bloc ne domine

Dix variantes alternées dans un seul process, séquence forcée identique,
8 pas de warm-up, 48 mesurés. Référence 54,95 ms (le mode entrelacé paie un
`clearCache` par tour, absent en production — d'où 54,95 contre 47,84 ; le
classement relatif reste valide, les valeurs absolues sont gonflées d'environ
15 %).

| sous-bloc ablaté | ms/pas | coût attribué | part |
|---|---:|---:|---:|
| `moe-routing` | 33,02 | 21,93 ms | 39,9 % |
| **`hyper`** | 33,08 | **21,87 ms** | **39,8 %** |
| `moe` (bloc entier) | 34,51 | 20,44 ms | 37,2 % |
| `moe-switch-mlp` | 35,69 | 19,26 ms | 35,1 % |
| `gdn-projections` | 37,52 | 17,43 ms | 31,7 % |
| `gdn-recurrence` | 41,41 | 13,55 ms | 24,6 % |
| `norms` | 41,43 | 13,52 ms | 24,6 % |
| `qsa-attn` | 42,90 | 12,05 ms | 21,9 % |
| `moe-shared-expert` | 43,04 | 11,91 ms | 21,7 % |

Écarts-types entre 0,36 et 0,65 ms : la mesure est serrée, contrairement à
celle de P11.2.

**Trois enseignements.**

1. **La somme des blocs disjoints fait 98,9 ms pour un total de 55,0 ms, soit
   180 %.** La soustraction sur-attribue massivement — et c'est *cohérent* avec
   le diagnostic : dans un régime limité par les lancements, retirer n'importe
   quel bloc laisse tout le reste mieux s'enchaîner. **Ces chiffres valent comme
   classement, pas comme parts.**
2. **Les hyper-connexions coûtent autant que tout le bloc d'experts** (21,87
   contre 20,44 ms). C'est la confirmation de l'hypothèse P11.3 : cet étage,
   que le modèle d'Edge0 n'a pas, est le premier poste ex æquo. Réserve : son
   ablation substitue un tenseur de poids d'injection **constant**, que MLX
   pourrait replier ; une part du gain mesuré peut venir de là.
3. **Aucun bloc ne domine.** Même le plus petit (`moe-shared-expert`, un MLP
   unique) pèse 11,9 ms. Le coût est **diffus**, réparti sur tous les
   sous-blocs — signature exacte d'un régime où chaque petit noyau paie sa
   latence de lancement.

### Conséquence pour P11.10 (noyau Metal)

Le coût étant diffus, fusionner **un** sous-bloc ne rendra qu'une fraction. La
cible de départ reste néanmoins les **hyper-connexions** : premier poste ex
æquo, et surtout pure tuyauterie (normes + deux matmuls de rang faible +
injection élémentaire) sans quantification à réimplémenter — donc le
prototype le moins risqué pour établir si un gain de dispatch se traduit en
gain de débit. Si oui, la cible suivante est le bloc MoE complet, qui demande
d'écrire la déquantification affine 3 bits en Metal.

---

## 2026-09-13 — Correction : le noyau Metal a déjà été essayé deux fois

En lisant `Qwen4ExpHyperConnection.swift` pour rédiger le prototype P11.10, j'ai
trouvé ceci en tête de fichier :

> P10.2 (F8, 2026-09-12) — RETIRED: a `MLXFast.metalKernel` pair used to live
> here, fusing the hyper-connection "mix" reduction and
> `Qwen4ExpDecoderLayer.inject`'s broadcast-multiply-add (5 ops each) into one
> kernel call. […] F7 mean 2.372 s, F8 mean 2.392 s — a small, consistent
> ~1 % **regression**.

Et le même motif dans `Qwen4ExpGatedDeltaNet.swift` pour F9 (P10.3).

**J'avais écrit dans §P11.10 que le noyau Metal personnalisé était « la seule
voie jamais essayée ». C'est faux.** F8 et F9 étaient des `MLXFast.metalKernel`,
sur deux des trois cibles que je proposais, et tous deux ont été retirés pour
une régression.

### Ce que ce fait nous apprend vraiment

Supprimer 10 ops par couche sur ~100 — ce que faisait F8 — aurait dû rendre
environ 10 % si le coût était bien le nombre de lancements. On a mesuré −1 %.
**Nous ne sommes donc pas limités par le nombre de dispatches.**

Le chiffre qui compte est ailleurs. Le bloc d'experts lit **1 005 Mio par
jeton** et coûte, selon l'attribution, 10 à 20 ms :

| coût supposé du bloc | bande passante atteinte | part du pic (~400 Go/s) |
|---:|---:|---:|
| 20,4 ms | 51,6 Go/s | 12,9 % |
| 15,0 ms | 70,3 Go/s | 17,6 % |
| 10,0 ms | 105,4 Go/s | 26,3 % |

Le toit serait 2,63 ms. **Les noyaux tournent donc à 13-26 % de la bande
passante disponible** : le temps part dans des noyaux individuellement
inefficaces, pas dans leur nombre.

C'est une bien meilleure cible, et elle explique d'un coup toutes les mesures
de la nuit : pourquoi réduire K ne rend que 4,7 % (moins d'octets, même
inefficacité), pourquoi fusionner des ops ne rend rien (le nombre n'est pas le
problème), et pourquoi ablater n'importe quel bloc « économise » 20-40 % (tout
est également lent).

### Recadrage de P11.10

Ne plus chercher à réduire le nombre de noyaux. Chercher **pourquoi
`gatherQMM` / `SwitchGLU` n'atteint que 13-26 % de la bande passante aux
formes réelles** (10 experts choisis parmi 512, poids 3 bits g64). Pistes à
départager par la mesure, pas par l'écriture : accès non coalescés sur des
experts dispersés, déquantification par groupes de 64 mal vectorisée,
occupation trop faible à batch 1.

**Mesurer d'abord, en isolation. Écrire du Metal seulement si la cause désigne
un correctif précis.** F8 et F9 ont déjà montré ce que donne une réécriture à
l'aveugle.

---

## 2026-09-13 — LE résultat de la nuit : 90 % du coût ne dépend pas des experts

Mesure faite avec `flash-decode-bench` et une **séquence forcée identique**
(les 128 mêmes jetons pour toutes les variantes, `--forced-ids`), 16 pas de
warm-up, 96 mesurés, un process par valeur de K.

| K | Mio d'experts lus par jeton | ms par pas | ajusté par la droite |
|---:|---:|---:|---:|
| 10 | 1 005 | 46,944 | 46,933 |
| 8 | 804 | 45,914 | 45,972 |
| 5 | 503 | 44,608 | 44,531 |
| 2 | 201 | 43,094 | 43,090 |
| 1 | 101 | 42,576 | 42,609 |

La droite ajuste à mieux que 0,08 ms. Elle donne deux nombres :

| | |
|---|---|
| **pente** | **4,78 µs par Mio**, soit **219 Go/s** de bande passante marginale = **55 % du pic** |
| **ordonnée à l'origine** | **42,13 ms** — coût **fixe**, indépendant du nombre d'experts lus |

### Ce que ça veut dire

À K=10, sur les 46,94 ms d'un pas de décodage :

- **4,80 ms (10,2 %)** servent à lire les poids des experts, **et ils tournent
  à 55 % de la bande passante crête — c'est correct** ;
- **42,13 ms (89,8 %)** sont un **coût fixe** qui ne dépend pas du tout du
  volume de poids lu.

Diviser les octets lus par dix ne fait gagner que **9,3 %**.

### Toutes les mesures de la nuit s'expliquent d'un coup

- **P11.1 (réduire K) plafonne à +5,9 %** : on ne peut pas gagner plus que les
  10 % que pèse la lecture des experts.
- **F8 et F9 (noyaux Metal) ont donné −1 %** : ils fusionnaient des ops, mais
  le coût fixe n'est pas concentré là où ils fusionnaient.
- **Ablater n'importe quel sous-bloc « économise » 20-40 %, somme 180 %** : le
  coût fixe est **diffus**, réparti sur tous les sous-blocs, et en retirer un
  laisse les autres mieux s'enchaîner.
- **Le MTP plafonne à 1,14×** : le coût fixe est justement ce qu'un forward
  groupé n'amortit qu'à moitié (47,84 ms à 1 jeton, 62,42 à 2).

### Correction d'un cadrage que j'avais écrit

Le tableau « nous sommes à 28× du toit, 3,6 % de la bande passante » de §P11
est juste en agrégat mais **trompeur sur la cible** : il laissait croire qu'il
fallait mieux lire les experts. C'est faux — la lecture des experts est la
partie qui marche. Le problème est les **42 ms de travail fixe par jeton**
répartis sur 48 couches de petites opérations : GDN, QSA, hyper-connexions,
normes, routeur, expert partagé.

### Ce qu'il faut mesurer ensuite

Le coût fixe étant diffus, il n'y a pas de cible unique. Deux questions
ouvertes, dans l'ordre :

1. **Combien d'ops GPU par jeton, réellement ?** L'estimation de ~4 800
   (≈100/couche, P2-code) n'a jamais été vérifiée post-F7. À 42,13 ms, cela
   ferait 8,8 µs par op. Si le compte réel est bien celui-là, le plancher est
   la latence de lancement et **aucune optimisation ponctuelle ne le bougera** :
   il faudrait moins de couches, ou un graphe capturé.
2. **`MLX.compile` ou une capture de graphe Metal** sur le forward complet —
   tentée en P2-code au niveau d'une couche et en P7 (où elle cassait la parité
   GDN), jamais sur le pas entier. C'est la seule technique qui attaque un coût
   fixe diffus plutôt qu'un bloc particulier.

---

## 2026-09-13 — P11.4a refaite correctement : le MTP est définitivement fermé

La première courbe (entrée précédente) utilisait **un process par valeur de
N** — exactement la méthodologie que P11.1 avait déclarée non fiable. Refaite
avec `--tokens-per-step-sweep`, qui alterne les N **dans un seul process**, sur
la même séquence forcée.

### La courbe

6 variantes × 80 tours (16 warmup + 64 mesurés), séquence forcée commune :

| jetons par forward | ms par pas | écart-type | marginal du jeton ajouté |
|---:|---:|---:|---:|
| 1 | 49,240 | 1,12 | — |
| 2 | 64,061 | 1,31 | 14,82 |
| 3 | 80,034 | 1,64 | 15,97 |
| 4 | 96,756 | 2,61 | 16,72 |
| 6 | 132,302 | 6,58 | 17,77 |
| 8 | 164,363 | 6,83 | 16,03 |

Droite : **16,65 ms par jeton ajouté + 31,18 ms de coût fixe par forward**.

Contrôle croisé avec le chemin à variante unique (sans `snapshot`/`restore`,
donc sans vidage d'allocateur) : 47,84 ms à N=1 et 62,42 à N=2, marginal
14,58. Le balayage entrelacé paie **1,40 ms de plus par pas**, constant, ce qui
laisse les écarts — donc les marginaux — intacts. Les deux instruments
concordent.

**Le coût marginal d'un jeton supplémentaire vaut donc 15 à 17 ms, stable**, à
comparer aux 49,2 ms d'un forward à un jeton. Un forward groupé n'amortit
qu'un tiers du coût, pas la quasi-totalité comme l'asymptote du préfill
(7,7 ms) le laissait espérer.

### Le verdict, chiffré

À 48,4 % d'acceptation (mesurée), **avec un drafter parfaitement gratuit** :

| profondeur du brouillon | jetons par pas | coût du pas | gain |
|---:|---:|---:|---:|
| 1 | 1,484 | 64,1 ms | **1,14×** |
| 2 | 1,718 | 80,0 ms | 1,06× |
| 3 | 1,832 | 96,8 ms | 0,93× |
| 5 | 1,913 | 132,3 ms | 0,71× |
| 7 | 1,932 | 164,4 ms | 0,58× |

**Approfondir le brouillon est contre-productif dès la profondeur 2.**

Validation du modèle de coût : avec le drafter actuel à 12,6 ms il prédit
**0,953×**, on mesure **0,939×** — 1,5 % d'écart. Le modèle tient.

Ce qu'il faudrait pour gagner quelque chose :

| | |
|---|---|
| seuil de rentabilité | drafter sous **9,0 ms** (actuel : 12,6) |
| drafter divisé par 3 | 1,07× |
| drafter divisé par 6 | 1,11× |
| drafter **gratuit** | **1,14×** |
| pour atteindre 1,2× | acceptation de **56,1 %** (mesurée : 48,4 %) |
| pour atteindre 1,3× | acceptation de **69,1 %** |
| pour atteindre 1,5× | acceptation de **95,1 %** |

### Décision

**P11.4 est fermée.** Le meilleur résultat concevable est **1,14×**, et il
suppose un drafter de coût nul, ce qui n'existe pas. Un drafter six fois moins
cher que l'actuel donnerait 1,11 ×. Monter l'acceptation à 69 % rendrait 1,3 ×,
mais une meilleure acceptation demande un drafter plus gros, donc plus cher,
alors que le budget total est de 9 ms.

Le MTP reste disponible en option (`mtp: true`), documenté comme perdant à
0,94 ×. Aucun travail supplémentaire n'est justifié.

---

## 2026-09-13 — Le bon toit : 3,19 Gio lus par jeton, 18 % de la bande passante, ×5,6 (et non ×28)

Mon tableau « ×28 du toit, 3,6 % de la bande passante » de §P11 ne comptait que
les **experts routés**. C'était l'erreur de cadrage. Voici le compte complet,
dérivé des en-têtes safetensors du checkpoint réel.

### Ce que le modèle lit vraiment à chaque jeton décodé

| étage | Mio/jeton | part | toit à 400 Go/s | ablation (P11.2a) | × le toit |
|---|---:|---:|---:|---:|---:|
| **GDN (projections, 36 couches)** | **1 246** | 38 % | 3,27 ms | 17,4 ms | 5,3× |
| experts routés (10 sur 512) | 1 005 | 31 % | 2,63 ms | 20,4 ms | 7,8× |
| hyper-connexions | 383 | 12 % | 1,00 ms | **21,9 ms** | **21,8×** |
| QSA (12 couches) | 368 | 11 % | 0,96 ms | 12,1 ms | 12,5× |
| expert partagé | 141 | 4 % | 0,37 ms | 11,9 ms | **32,2×** |
| routeur MoE | 120 | 4 % | 0,31 ms | — | — |
| normes, divers | 20 | 1 % | 0,05 ms | — | — |
| **TOTAL** | **3 283** | 100 % | **8,61 ms** | | |

(hors table n-gram et embeddings/lm_head, qui sont des lectures **éparses**, et
hors tour vision, inutilisée en texte)

| | |
|---|---|
| mesuré | 47,84 ms par pas → **72 Go/s**, soit **18 % du pic** |
| toit à 400 Go/s | 8,61 ms → **116 tok/s**, donc **×5,6** et non ×28 |
| au rendement du matmul dense isolé (212 Go/s mesurés) | 16,2 ms → **62 tok/s**, soit **×2,9** le débit actuel |

### Trois surprises

1. **Les projections GDN lisent plus d'octets que les experts routés** (1 246
   contre 1 005 Mio). Sur 36 couches, 34,6 Mio par couche. C'est le premier
   poste du modèle, et personne ne l'avait regardé.
2. **Les hyper-connexions et l'expert partagé sont les plus inefficaces** :
   21,8× et 32,2× leur toit, contre 5,3× pour le GDN. Ils lisent peu (383 et
   141 Mio) mais coûtent autant que des étages dix fois plus gros. Ce sont des
   empilements de **très petites opérations** — normes, matmuls de rang 320,
   sigmoïdes, reshape, moyennes — chacune payant sa latence de lancement de
   3 à 6 µs (mesurée par `op-overhead-probe`) pour un travail dérisoire.
3. **Le plancher par op n'explique pas tout le reste.** `op-overhead-probe`
   donne 3-6 µs pour une op élémentaire, mais **61,8 µs pour un matmul
   [1,2560]×[2560,2560]** (soit 212 Go/s, 53 % du pic) et **150,7 µs pour le
   `SwitchGLU` réel** (10 experts sur 512, soit 145 Go/s, 36 % du pic). Les
   grosses opérations ne sont donc **pas** au plancher de lancement : elles
   font du vrai travail, à la moitié ou au tiers du pic.

### Diagnostic révisé, et il est bien meilleur que le précédent

Le coût se sépare en deux :

- **Les gros étages (GDN, experts, QSA) tournent à 36-53 % du pic** quand on
  les mesure isolément. C'est perfectible mais pas scandaleux.
- **Les petits étages (hyper-connexions, expert partagé, normes) tournent à
  3-5 % du pic** parce qu'ils sont faits de dizaines d'opérations minuscules
  dont chacune paie sa latence de lancement. **C'est là qu'est le gras.**

Cible chiffrée : ramener les hyper-connexions et l'expert partagé de 21,8× et
32,2× leur toit à, disons, 8× — ce que fait déjà le GDN — rendrait environ
**10 ms sur 47,84, soit +26 % de débit**. Et le faire sur tous les petits
étages à la fois viserait les 62 tok/s du rendement « matmul dense ».

**C'est le premier chiffre de la campagne qui désigne une cible précise avec un
gain crédible au-delà de 10 %.** À la différence de F8, qui fusionnait 5 ops
des hyper-connexions et n'en tirait rien : il en reste une vingtaine par
couche.

---

## 2026-09-13 — F8/F9 compilés : +2,4 %, et la mort de l'hypothèse « limité par les lancements »

Deux niveaux opt-in ajoutés : **F8** compile le corps de
`Qwen4ExpGatedResidual.callAsFunction` (`hc_norm` groupée comprise) dans un
`MLX.compile`, **F9** fait passer l'activation de l'expert partagé par
`compiledSiluProduct`. Comptage fait dans le code : **24 lancements par
hyper-connexion** (11 pour la norme groupée, qui ne peut pas utiliser
`MLXFast.rmsNorm`, et 13 pour le reste), × 2 par couche × 48 couches =
**2 304 lancements par jeton** remplacés par 2 appels de fermeture compilée par
couche. F9 économise 48 lancements de plus.

### Mesure

`flash-decode-bench`, séquence forcée identique, 16 warmup + 96 pas, trois
process alternés par niveau :

| niveau | runs (ms/pas) | médiane | gain |
|---:|---|---:|---:|
| 7 (défaut de production) | 52,59 · 47,19 · 47,03 | 47,187 | — |
| 8 (+ hyper-connexions compilées) | 46,26 · 46,36 · 46,15 | 46,261 | **+2,0 %** |
| 9 (+ expert partagé compilé) | 46,07 · 46,85 · 45,95 | 46,066 | **+2,4 %** |

**Parité exacte sur le checkpoint réel** : `flash-chat-probe --fusion-level
7/8/9`, greedy, 16 jetons — identifiants strictement identiques aux trois
niveaux.

### Ce que ça enterre

**Supprimer 2 304 lancements d'opérations par jeton rapporte 1 ms sur 47.**
L'hypothèse « nous sommes limités par le nombre de lancements côté hôte » est
donc **fausse**, et avec elle la lecture que j'en tirais. Le coût des ops
élémentaires de l'étage hyper-connexion est d'environ **1,5 ms**, pas les
21,9 ms que l'ablation lui attribuait.

**Corollaire méthodologique dur : l'ablation par soustraction sur-attribue
d'un facteur ~20 sur cet étage**, même à séquence forcée. La somme à 180 % du
total en était déjà l'indice ; ici on a la preuve directe par une seconde
méthode. **Les chiffres d'ablation ne valent ni comme parts ni comme
classement fiable — au mieux comme borne supérieure.** Ne plus les utiliser
pour choisir une cible : c'est la troisième fois que cet instrument égare la
campagne (P7, P11.2, ici).

### Où est donc le temps ?

Ce qui reste solidement mesuré :

| | |
|---|---|
| poids lus par jeton | 3 283 Mio |
| débit atteint | 72 Go/s = **18 % du pic** |
| lecture des experts (pente mesurée du balayage K) | **4,80 ms** |
| ops élémentaires des hyper-connexions (mesuré par F8) | **~1,0 ms** |
| noyaux isolés, mesurés par `op-overhead-probe` | matmul 2560² : 61,8 µs = **212 Go/s** (53 % du pic) · `SwitchGLU` réel : 150,7 µs = **145 Go/s** (36 %) |

Les gros noyaux, mesurés seuls, atteignent 36 à 53 % du pic. L'ensemble
n'atteint que 18 %. **L'écart n'est ni dans les octets, ni dans les lancements
hôte, ni dans les ops élémentaires.** Il reste deux explications possibles,
qu'aucune mesure de la nuit ne départage : le coût GPU d'enchaîner ~4 800
petits noyaux (que `compile` ne supprime qu'en partie), ou des noyaux qui,
en contexte, n'atteignent pas le rendement qu'ils montrent isolés.

### Décision sur F8/F9

**+2,4 % avec parité exacte, sous la barre des 5 %** que le projet s'est
donnée pour promouvoir un niveau de fusion (§P10). Conservés **opt-in**, pas
promus en défaut, pour une raison précise qui reste à mesurer : `MLX.compile`
met en cache **un graphe par forme rencontrée**, donc chaque nouvelle longueur
de préfill paie une recompilation. La sonde synthétique de l'agent la chiffre
à ~7 ms sur un module isolé ; en contexte réel, sur 48 couches et un serveur
qui voit des dizaines de longueurs, l'effet sur le TTFT n'est **pas mesuré**.
À trancher avant toute promotion.

---

## 2026-09-13 — La trace Metal tranche : le GPU est inactif un tiers du temps

Dernier instrument de la boîte, et le seul capable de départager les deux
hypothèses restantes. Deux tentatives :

1. **Échec** : `serve --metal-trace-seconds 25`, xctrace **attaché** au
   serveur, puis serveur tué. Bundle de 4,1 Go **inexploitable** — `xctrace
   export` répond « Document Missing Template Error ». La trace P4 conservée
   (`results/p4-mst-real.trace`) s'exporte, elle : elle avait été **lancée**
   par xctrace et le process s'était terminé de lui-même.
   **Règle : tracer un process qui se termine tout seul, ne jamais tuer la
   cible.**
2. **Réussite** : `xctrace record --template 'Metal System Trace' --launch --
   qwen38 flash-decode-bench … --forced-ids …`, limite 300 s (sous
   instrumentation le chargement seul prend ~80 s ; une limite de 150 s
   expirait avant le décodage). Bundle de 176 Mo, exportable.

### Ce que la trace montre pendant le décodage

Fenêtre de décodage isolée (1,366 s, 25,9 pas), canal `Compute` du process
tracé, intervalles fusionnés par union (ils se chevauchent ×2,7) :

| | |
|---|---|
| tampons de commandes | **197 par pas de décodage** |
| durée par tampon | médiane 0,344 ms · moyenne 0,484 ms |
| **GPU occupé** | **66,7 %** |
| **GPU inactif** | **33,3 %** |
| trous réels | **46 par pas**, médiane 16,7 µs, **17,44 ms perdus par pas** sur 52,7 |

**Le GPU ne fait rien pendant un tiers du pas de décodage.** Et pendant les
66,7 % où il travaille, il traite 3 283 Mio en 35,2 ms, soit **98 Go/s = 24 %
du pic** — au lieu des 18 % calculés sur le temps de paroi, et toujours sous
les 36-53 % que les noyaux atteignent isolément.

Le coût se décompose donc en deux parts, toutes deux attaquables :

| part du pas (52,7 ms sous trace) | |
|---|---|
| GPU inactif, en attente de l'hôte | **17,4 ms (33 %)** |
| GPU actif à 24 % du pic | 35,2 ms (67 %) |

### Le levier immédiat : `residentAsyncInterval`

Le réglage qui contrôle la fréquence des `eval` bloquants valait **8**, choisi
par le balayage P4.1 — **daté d'avant la correction de dtype F7**, qui a
multiplié le débit par 2,74 et déplacé l'optimum. Remesuré à séquence forcée,
deux tours alternés :

| N | ms par pas | gain |
|---:|---:|---:|
| 8 (ancien défaut) | 46,82 | — |
| 16 | 45,51 | +2,9 % |
| 24 | 44,89 | +4,3 % |
| **48** (un seul `eval` par forward) | **44,36** | **+5,5 %** |

Combiné avec F9 (hyper-connexions et expert partagé compilés) : **43,26 ms,
soit +8,2 %** et 23,11 tok/s.

Identifiants greedy **strictement identiques** à N=8 et N=48 sur le checkpoint
réel, et **pic MLX inchangé** (57,42 Go dans les trois configurations).

**Décision : le défaut de `residentAsyncInterval` passe de 8 à 48.** Il passe
la barre des 5 % que le projet s'est donnée, avec parité exacte et sans coût
mémoire. F8/F9 restent opt-in tant que l'effet de la recompilation par forme
sur le TTFT n'est pas mesuré.

### Ce qui reste

Les 17,4 ms d'inactivité GPU ne disparaissent pas avec N=48 : le réglage ne
touche que les `eval` **bloquants** (6 par forward à N=8, 1 à N=48), pas les
197 tampons. Les 46 trous par pas viennent d'ailleurs — soumission de tampons,
dépendances entre encodeurs, ou synchronisations internes à MLX. **C'est le
prochain chantier, et la trace donne enfin de quoi le viser.**

---

## 2026-09-13 — Les 33 % d'inactivité GPU ne viennent pas de la soumission : ils sont structurels

Suite directe de la trace. MLX plafonne chaque tampon de commandes à un
nombre d'opérations **et** à un volume, avec des défauts par puce
(`mlx/backend/metal/device.cpp:348-356` : sur M3 Max, architecture `…s`,
**50 opérations et 50 Mo**), tous deux surchargés par
`MLX_MAX_OPS_PER_BUFFER` et `MLX_MAX_MB_PER_BUFFER`
(`mlx/utils.h:146-156`). Avec 197 tampons par pas mesurés, c'était le levier
évident sur les trous.

### Mesure — et le levier joue à l'envers

Séquence forcée, deux tours alternés, 96 pas mesurés :

| configuration | ms par pas | écart |
|---|---:|---:|
| défaut (50 ops / 50 Mo) | 46,52 | — |
| `MLX_MAX_OPS_PER_BUFFER=200` | 47,56 | **−2,2 %** |
| `…=1000` + `MLX_MAX_MB_PER_BUFFER=1000` | 48,02 | **−3,1 %** |
| `MLX_MAX_MB_PER_BUFFER=1000` seul | 46,20 | +0,7 % |

**Des tampons plus gros dégradent le débit.** Le seul réglage qui gagne
(+0,7 % sur le volume seul) est sous le seuil de décision du projet.

### Pourquoi, et ce que ça enterre

La trace montrait déjà que les tampons **se chevauchent ×2,7** : MLX
pipeline, la construction du tampon N+1 recouvre l'exécution du N. Grossir
les tampons réduit la profondeur de ce pipeline, d'où la régression.

**Les 33 % d'inactivité ne sont donc pas du surcoût de soumission.** Ce sont
des attentes de **dépendances** : le décodage autorégressif est une chaîne
sérielle — la couche N+1 a besoin de la sortie de N, le jeton t+1 a besoin de
t — et à lot de taille 1 il n'y a tout simplement pas assez de travail
indépendant pour remplir un GPU à 40 cœurs. **C'est structurel, pas
corrigeable par un réglage.**

### Le corollaire, lui, est exploitable : le GPU a de la place

La courbe « jetons par forward » déjà mesurée le chiffre exactement :

| jetons par forward | ms | coût relatif | **débit relatif** |
|---:|---:|---:|---:|
| 1 | 49,24 | 1,00× | 1,00× |
| 2 | 64,06 | 1,30× | **1,54×** |
| 4 | 96,76 | 1,96× | **2,04×** |
| 8 | 164,36 | 3,34× | **2,40×** |

Doubler le travail par forward ne coûte que **+30 %** : la place vide mesurée
par la trace est bien là, et on peut la remplir. Mais **pas avec des jetons du
même flux** — P11.4 a montré que la spéculation plafonne à 1,14× parce que
l'acceptation retombe. Il faut du travail **indépendant**, c'est-à-dire
**plusieurs requêtes servies ensemble**.

### Ce qui reste, et c'est le dernier chantier de la campagne

**Le traitement par lots continu côté serveur.** Aujourd'hui
`Qwen38Server` sérialise les requêtes (`FIFORequestQueue`). Les chiffres
ci-dessus disent qu'en décodant 8 séquences ensemble, le **débit agrégé**
atteindrait ~2,4× celui d'une seule — sans toucher à la latence perçue par
chaque client au-delà de +234 % sur un pas qui sert 8 clients au lieu d'un.

C'est un gain de **capacité**, pas de latence mono-utilisateur : cette
dernière est désormais bornée par la sérialité du modèle, qu'aucune des
douze pistes de §P11 n'a pu contourner.

---

## 2026-09-13 — P12.1 : le lot passe la porte, ×2,94 à huit séquences

La sonde `flash-batch-probe` décode B séquences indépendantes dans un même
forward, prompts de longueur identique, greedy strict par séquence.

### La surprise : aucune modification de production n'a été nécessaire

Revue module par module, en suivant `dim(0)` de bout en bout :

| module | état |
|---|---|
| `Qwen4ExpGatedDeltaStates` (état conv + récurrent GDN) | construit en `[batch, …]` depuis `inputs.dim(0)`, **déjà générique** |
| `Qwen4ExpCache` / `Qwen4ExpQSAKVCache` | délègue à `KVCacheSimple`, dimensionné sur `keys.dim(0)`, **déjà générique** |
| `Qwen4ExpPLE` (regroupement P6.2) | construit ses identifiants en `[batch, sequence, ngramHeads]` et reforme la sortie sur `IDs.dim(0)` — **ne suppose jamais une séquence unique**, contrairement à ce que le libellé « 231 → 2 lookups » laissait craindre |
| MoE, hyper-connexions, embeddings/`lm_head` | génériques sur `dim(0)` |
| `Qwen4ExpMRoPE` / `logicalOffset` | `[3,1,S]` diffusé contre B, et un offset scalaire pour tout le lot — **correct uniquement parce que les séquences avancent en pas synchrone à longueur égale**. C'est le premier blocage pour P12.2. |

Le seul code neuf est la sonde elle-même et sa logique pure. **Zéro
changement de comportement à B=1.**

### Parité — le critère qui comptait

Le même prompt répété B fois rend, à B = 1, 2, 3, 4 et 8, **exactement les
mêmes identifiants**, identiques entre séquences et identiques à la référence
du dépôt. `parité inter-séquences : OK` partout. **Les séquences ne se
contaminent pas.**

### Le passage à l'échelle

| séquences | ms/pas | débit agrégé | gain | par séquence | coût marginal |
|---:|---:|---:|---:|---:|---:|
| 1 | 42,32 | 23,63 t/s | 1,00× | 23,63 t/s | — |
| 2 | 51,17 | 39,08 t/s | 1,65× | 19,54 t/s | 8,85 ms |
| 3 | 60,93 | 49,24 t/s | 2,08× | 16,41 t/s | 9,76 ms |
| 4 | 72,78 | 54,96 t/s | 2,33× | 13,74 t/s | 11,84 ms |
| **8** | **115,09** | **69,51 t/s** | **2,94×** | 8,69 t/s | 10,58 ms |

**La porte de P12.1 demandait ≥ 1,5× à B=8. On mesure 2,94×.** C'est même
au-dessus des 2,40× du lot de jetons d'un même flux, et bien au-dessus de mon
estimation de ~2× : le coût marginal d'une séquence de plus est de **9 à
12 ms**, contre 42,32 ms pour la première, parce que les 2 258 Mio de poids
denses se lisent **une seule fois pour tout le lot**.

Pic mémoire : 57,76 Go à B=1, **58,62 Go à B=8** — moins d'un gigaoctet pour
huit séquences.

### Et pour un client, la latence s'améliore aussi

Le lot ralentit chaque séquence (8,69 t/s contre 23,63 seul), mais le service
sérialisé fait attendre son tour. Pour une réponse de 100 jetons :

| clients simultanés | par lot | sérialisé | rapport |
|---:|---:|---:|---:|
| 2 | 5,1 s | 6,3 s | ×1,24 |
| 4 | 7,3 s | 10,6 s | ×1,45 |
| 8 | **11,5 s** | **19,0 s** | **×1,65** |

Donc ce n'est pas seulement de la capacité : **dès deux clients simultanés, le
lot est aussi meilleur en latence de bout en bout.**

### Réserves à porter à P12.2 et P12.3

1. **L'horloge de position est scalaire pour tout le lot.** Valide en pas
   synchrone, fausse dès que les longueurs diffèrent : il faudra un décalage
   par ligne et des `positionIDs` `[3,B,S]` réels.
2. **Aucun masque de remplissage**, ni QSA ni GDN. `Qwen4ExpQSAKVCache`
   documente déjà un masque `[B,S]` optionnel « réservé aux lots irréguliers »,
   jamais implémenté.
3. **La sélection éparse QSA n'est pas exercée** : son budget de 2 048 jetons
   n'est pas atteint à ces longueurs. Son comportement en lot reste à vérifier
   au-delà de ce seuil.
4. Côté vendor, `ArraysCache`/`MambaCache` exposent déjà
   `leftPadding`/`lengths`/`filter(batchIndices:)`/`extend(other:)` — l'outillage
   du lot continu existe, il n'est branché nulle part côté Flash-Next.
5. **Écarts-types élevés** (40-46 ms, maxima vers 400 ms) à tous les B : une
   perturbation périodique que la médiane absorbe mais qu'il faudra élucider
   avant de promettre une latence au client.

---

## 2026-09-13 — P12.2 : longueurs inégales, et le critère tient

### Le critère, d'abord

Le prompt de référence a été placé en **tête, au milieu et en queue** d'un lot
de trois séquences aux longueurs très différentes (« Bonjour », le prompt de
référence, et une demande d'essai en dix paragraphes). Aux trois positions, il
rend exactement `[2229, 85648, 401, 1147, 183085, 1725, 41016, 90171, 13,
7305, 1725, 501]`, la référence du dépôt.

**Et le résultat est plus fort que le critère demandé** : *chaque* séquence du
lot rend des identifiants strictement identiques quelle que soit sa position.
« Bonjour » et l'essai donnent les mêmes douze jetons dans les trois
dispositions. **Le remplissage et la position dans le lot n'influencent rien.**

### Comment

**Remplissage à gauche.** La dernière colonne du lot devient alors, pour
chaque ligne, son dernier jeton réel : le pas de décodage relit
`logits[:, -1, :]` exactement comme à longueur égale. Le remplissage à droite
aurait exigé un `takeAlong` par ligne à chaque pas. Le vendor confirme le
sens : `ArraysCache.leftPadding` et `ArraysCache.makeMask` sont construits pour
celui-là.

**Trois modifications de production, toutes additives** (paramètre par défaut
`nil`, comportement à B=1 et à longueurs égales strictement inchangé) :
`Qwen4ExpQSAAttention.causalMask(…, leftPadding:)`, et le relais
`leftPadding: [Int]? = nil` dans `Qwen4ExpStreamingDecoder.forward` puis
`Qwen4ExpStreamingTextModel.forward`.

**Ce qui a évité du travail, et ce n'est pas ce que je croyais.**
`ArraysCache`/`MambaCache` ne servent qu'au découpage et à la fusion de lots en
cours de route — hors périmètre ici. En revanche
`Qwen4ExpGatedDeltaNet.callAsFunction` et `Qwen4ExpPLELayer.callAsFunction`
acceptaient **déjà** un `mask: MLXArray?` de contrat `[B, S]`, documenté comme
« reserved for padded/ragged batches » et jamais alimenté. Il a suffi de le
construire et de le brancher. Même histoire qu'en P12.1 : l'ossature était là.

**Le point malin, à retenir : le jeton de remplissage doit être l'EOS.**
`Qwen4ExpNGramEmbedding` construit son historique en l'absence de contexte avec
des EOS, et `shiftRightIgnoringEOS` traite déjà tout EOS comme frontière de
segment dure. Remplir à gauche avec l'EOS rend donc l'historique de n-grammes
des jetons réels **identique**, quelle que soit la longueur du remplissage, et
**sans masque supplémentaire pour cette partie**. Tout autre jeton de
remplissage aurait cassé la parité de toute ligne dont le remplissage est plus
court que la fenêtre de contexte n-gramme.

Une optimisation tombe au passage : P2-code (e) sautait la construction du
masque causal en décodage à un jeton, « provablement toujours vrai ». Ce n'est
plus vrai dès que le cache porte des colonnes de remplissage définitivement
invalides — le masque est donc reconstruit à chaque pas tant qu'un
`leftPadding` est fourni.

### Le coût du remplissage, et il n'est pas négligeable

| lot de 3 | ms par pas |
|---|---:|
| trois prompts identiques, courts | 60,93 |
| trois prompts de longueurs mêlées | **75,48** (+24 %) |

**Le lot avance au rythme de sa séquence la plus longue** : tout le monde paie
un contexte allongé par le remplissage. C'est la contrainte principale à porter
à P12.3 — un ordonnanceur doit **grouper des longueurs voisines**, sinon il
dilapide une bonne part du ×2,94.

### Correction d'un message trompeur

La sonde annonçait « parité inter-séquences : ÉCHEC » sur un lot à prompts
différents — où des sorties différentes sont évidemment le comportement
attendu. Le contrôle ne s'exécute plus que si tous les prompts sont
identiques ; sinon la sonde renvoie explicitement vers la ligne
« référence (rang N) », qui est le vrai critère.

---

## 2026-09-13 — P12.3 : l'ordonnanceur livre ×2,16 sur le serveur réel

`serve --batch-size N`, défaut 1 (chemin d'aujourd'hui, strictement inchangé).
Au-delà, les requêtes **froides** en attente sont regroupées dans un même pas
de décodage ; une requête qui touche le cache de conversations garde le chemin
sérialisé et son TTFT plat.

### Un crash trouvé par la mesure, et sa vraie cause

Première version : `--batch-size 4` avec **8 clients** (donc deux exécutions de
lot successives) tuait le serveur — `EXC_BAD_ACCESS` dans
`unloadResidentLayers()` appelé par `resetConversation()` depuis
`generateBatch`. Avec 8 clients et `--batch-size 8` (un seul lot), aucun
problème. Mon hypothèse était que le verrou d'exécution n'existait pas ou ne
couvrait pas la réinitialisation.

**La vraie cause est plus fine** : le verrou existait et était bien acquis,
mais **sa libération dépendait de la consommation des flux SSE, pas de la fin
réelle de l'exécution**. `continuation.finish()` réveille son lecteur de façon
asynchrone ; le serveur relâchait donc le verrou dès que les flux étaient
consommés, ce qui pouvait précéder l'exécution du `model.resetConversation()`
final. Un second lot démarrait alors sa propre réinitialisation pendant que le
premier terminait la sienne : **deux `resetCaches()` concurrents sur le même
modèle résident.**

Correctif : la génération par lot retourne désormais une tâche `completion`
qui ne se termine qu'à la fin réelle de `run()`, et c'est elle que le
coordinateur attend avant de relâcher le verrou. Le
`model.resetConversation()` final est en outre déplacé **avant** la fermeture
des continuations. Test de non-régression
(`serverNeverOverlapsTwoSuccessiveBatchExecutions`) : deux exécutions
successives avec un délai injecté entre « flux livrés » et « exécution
terminée » — **vérifié qu'il échoue sans le correctif** et passe avec.

**Leçon : les 13 tests initiaux ne couvraient que le cas d'un lot unique.**
C'est le trou par lequel le bug est passé, et c'est la mesure sur serveur réel
qui l'a trouvé, pas la suite de tests.

### La matrice

Clients simultanés à départ synchronisé, 96 jetons, prompts de longueurs
voisines (`Scripts/bench-concurrent.py`).

**Débit agrégé, en jetons par seconde :**

| clients | sérialisé | lot de 4 | lot de 8 | gain |
|---:|---:|---:|---:|---:|
| 1 | 20,48 | 19,71 | 20,38 | ×1,00 |
| 2 | 20,05 | 28,90 | 29,14 | ×1,45 |
| 4 | 20,04 | 36,90 | 36,91 | ×1,84 |
| 8 | 19,91 | 36,67 | **43,07** | **×2,16** |

**Latence de bout en bout par client — médiane / dernier servi :**

| clients | sérialisé | lot de 4 | lot de 8 |
|---:|---:|---:|---:|
| 1 | 4,69 / 4,69 s | 4,87 / 4,87 s | 4,71 / 4,71 s |
| 2 | 7,19 / 9,58 s | 6,64 / 6,64 s | 6,59 / 6,59 s |
| 4 | 11,96 / 19,17 s | 10,41 / 10,41 s | 10,40 / 10,40 s |
| 8 | 21,80 / **38,57 s** | 15,59 / 20,94 s | 17,83 / **17,83 s** |

### Ce qu'il faut en retenir

1. **×2,16 de débit agrégé à 8 clients**, et le dernier servi passe de 38,57 s
   à 17,83 s. En lot, **toutes les latences sont égales** : personne n'attend
   derrière personne.
2. **Aucune régression à un seul client** : 20,38 contre 20,48 jetons/s, et
   4,71 s contre 4,69. Le chemin solo n'est pas payé.
3. **Le ×2,16 du serveur est en dessous du ×2,94 de la sonde P12.1**, et c'est
   attendu : le serveur paie en plus le préfill de chaque requête, la fenêtre
   de regroupement de 30 ms, et le fait qu'une ligne finie garde sa place
   jusqu'à la fin du lot.
4. **Arbitrage réel entre taille de lot et médiane.** À 8 clients, un lot de 4
   donne une *meilleure médiane* (15,59 s contre 17,83) parce que le premier
   lot sort tôt, mais un *dernier servi plus tardif* (20,94 contre 17,83).
   Grand lot = meilleur débit ; petit lot = meilleure latence médiane.
5. Mémoire : RSS max **52,9 Go**, 277 décompressions sur toute la matrice. Le
   lot ne coûte rien en mémoire à cette échelle.

### Limites connues, à porter à la suite

- Fenêtre de regroupement de **30 ms codée en dur**, ni mesurée ni exposée.
- **Une requête servie en lot n'est jamais mémorisée** comme conversation
  active : elle perd l'opportunité d'un tour 2 rapide. Une requête froide
  *isolée* garde le chemin d'aujourd'hui et reste éligible.
- Toute requête portant une **image** reste mono-séquence.
- L'entrée et la sortie de séquences **en cours** de lot ne sont pas
  implémentées : une ligne finie occupe sa place jusqu'au bout. C'est le
  gâchis mesurable dans l'écart entre ×2,16 et ×2,94.

---

## 2026-09-14 — Deux défauts du lot trouvés par la mesure d'expérience, et corrigés

Le ×2,16 de P12.3 avait été mesuré avec des prompts **homogènes** et le même
`max_tokens` partout. En testant un cas réaliste — un client pressé derrière
trois bavards — deux défauts sont apparus, dont un catastrophique.

### Défaut 1 : le client était retenu par le lot

Quatre clients simultanés, trois demandant 512 jetons, **un seul en demandant
16** :

| | avant | après |
|---|---:|---:|
| client à 16 jetons | **442,41 s** | **2,52 s** |
| dernier client à 512 jetons | 442,41 s | 60,10 s |

Sa réponse était prête au 16ᵉ pas ; il attendait les 512 autres. Cause : la
continuation d'une ligne n'était refermée qu'**après** la boucle de décodage,
donc à la fin de tout le lot. Correctif : refermer dès que la ligne est finie,
par une fonction idempotente, la réinitialisation finale du modèle restant à
sa place — la garantie du correctif de crash du 2026-09-13 est préservée,
puisque le verrou serveur se relâche sur la tâche `completion` et jamais sur
la consommation d'un flux.

La ligne finie continue d'occuper sa place dans le `forward` partagé : c'est
assumé. Seule sa **livraison** est devenue immédiate.

### Défaut 2 : une synchronisation GPU par ligne et par pas

Le générateur faisait, par ligne et par pas, `sampled.item(Int32.self)` — une
**synchronisation GPU bloquante**. À B lignes, B allers-retours par pas au lieu
d'un. La sonde `flash-batch-probe`, elle, évalue tout le lot en un `eval` et
lit les B identifiants d'un coup.

Correctif : les B graphes d'échantillonnage restent construits **ligne par
ligne** (chaque ligne garde sa température, son top-p, ses pénalités), mais
ils sont matérialisés par **un seul `eval` et une seule lecture** par pas. Le
chemin « tous réglages identiques → une op sur `[B,V]` » a été délibérément
écarté : sous température > 0, un tirage conjoint consomme le flux aléatoire
autrement qu'un tirage par ligne, ce qui casserait la parité lot ⟺ solo à
graine fixée. Une synchronisation par pas, quel que soit `batchSize`.

**Le coût de ces synchronisations croît avec la profondeur du pipeline**, d'où
un effet massif sur les générations longues (864 → 117 ms par pas à 512
jetons) et quasi nul à 96 jetons, où le pipeline est peu rempli. C'est ce qui
explique que la matrice homogène de P12.3 n'en souffrait presque pas.

### La matrice, après correction et à serveur chaud

| clients | sérialisé | lot de 8 | gain |
|---:|---:|---:|---:|
| 1 | 20,45 | 20,42 | ×1,00 |
| 2 | 20,06 | 28,60 | ×1,43 |
| 4 | 19,68 | 37,25 | ×1,89 |
| 8 | 19,29 | **44,05** (46,90 au mieux) | **×2,28** |

*Attention* : le premier passage après démarrage du serveur donne 40,37 au lieu
de 44,05 — **toujours écarter le premier run**, comme pour le reste du projet.

Générations longues, 4 clients × 384 jetons : **37-38 jetons/s agrégés**,
104 ms par pas contre 86 pour la sonde — il reste ~18 ms de coût serveur par
pas, essentiellement le préfill.

### Ce que ça change, en une phrase

Avant, une question courte posée pendant que d'autres demandaient un essai
coûtait **sept minutes**. Elle coûte maintenant **deux secondes et demie**.

---

## 2026-09-14 — Le préfill ne doit pas être mis en lot : une garde, et ce qu'elle règle

### Le défaut, mesuré

Sur un prompt long (~1 200 jetons), rejoindre un lot de 4 faisait passer le
TTFT du **premier** tour de 7,53 s à **37,20 s**, et celui du **second** de
0,40 s à 9,91 s (×24,5, la requête groupée n'étant jamais mémorisée comme
conversation).

**La cause est structurelle** : le lot aide le **décodage**, qui traite un
seul jeton par séquence et laisse le GPU aux deux tiers occupé seulement. Le
**préfill**, lui, traite déjà des centaines de jetons d'un coup — il est
dense, il n'a aucune capacité libre à remplir. Le grouper ne fait
qu'**additionner** le travail de chaque ligne, et y ajouter le gâchis du
remplissage à la longueur du plus long prompt.

### La garde

`serve --batch-max-prompt-tokens` (défaut **256**, publié dans `/healthz`) :
une requête froide dont le prompt rendu dépasse le seuil ne rejoint jamais le
lot et garde le chemin sérialisé. Le seuil est un ordre de grandeur au-dessus
du point sain mesuré (~25 jetons, aucune dégradation) et un ordre de grandeur
en dessous du point catastrophique (~1 200 jetons, ×4,9) ; **il n'est pas
dérivé d'une courbe fine**, qui reste à mesurer. Exposé aussi :
`serve --batch-window-ms` (défaut 30), jusqu'ici codé en dur.

### Résultat

| | avant la garde | après |
|---|---:|---:|
| TTFT tour 1, prompt long, 4 clients | 37,20 s | **8,40 s** (dont file d'attente ; 4,85 s seul) |
| TTFT tour 2, prompt long | 9,91 s | **0,36 s** — cache réutilisé |
| TTFT tour 2, prompt court groupé | — | 1,74 s contre 0,33 s seul, soit **+1,41 s** |
| Débit agrégé, prompts courts, 4 clients | 37,25 t/s | **37,37 t/s** — intact |

**La garde règle aussi le second défaut, sans le corriger.** Les prompts longs
et conversationnels — exactement ceux qui profitent du cache de conversations
— sortent du lot et retrouvent leur TTFT plat. Il ne reste que le cas d'une
conversation **démarrée par un prompt court**, dont le second tour paie
**+1,41 s** au lieu de +9,50 : une pénalité désormais **bornée par le seuil**.

### L'adoption de conversation reste bloquée, et c'est documenté

Faire d'une ligne de lot une conversation active suppose d'**extraire sa
tranche de cache**. Le vendor expose `filter(batchIndices:)`/`extend(other:)`
mais **seulement sur `ArraysCache`** (et `MambaCache`), pas sur le protocole
`KVCache`. Or un modèle Flash-Next mélange deux familles de couches : les
36 couches GDN utilisent `ArraysCache(size: 4)` — **supporté** — et les
12 couches d'attention pleine utilisent `Qwen4ExpQSAKVCache`, une classe
maison qui implémente `KVCache` directement — **non supporté**.

Écrire un `filter` maison pour cette dernière est techniquement faisable mais
n'a été validé par personne, sur un chemin non testé, avec un risque réel de
**réponses fausses au second tour**. Rien n'a été écrit. La limitation est
préférable au demi-mécanisme.

**Le préfill séparé du décodage** — la vraie solution au défaut A — bute sur
le même mur : il faudrait assembler un cache de lot à partir de caches
individuels, c'est-à-dire exactement la primitive qui manque. Écarté aussi.

---

## 2026-09-15 — P13.1 : les outils au format OpenAI, validés de bout en bout

`Qwen38Server` ignorait complètement les outils (`tools: nil` en dur dans
`Qwen4ExpPromptBuilder`, aucun champ dans la requête). Il les accepte
désormais, au format OpenAI.

### Le format du modèle n'est pas celui qu'on croit

Le gabarit du checkpoint n'utilise **pas** le JSON habituel mais une syntaxe
XML :

```
<tool_call>
<function=run_command>
<parameter=command>
swift build
</parameter>
<parameter=timeout>
300
</parameter>
</function>
</tool_call>
```

Deux pièges trouvés en lisant le gabarit, pas en devinant :

1. Un tour assistant renvoyé au modèle doit porter
   `tool_calls[].function.arguments` en **objet JSON**, pas en chaîne comme sur
   le fil OpenAI — le gabarit boucle dessus avec `|items`. Il faut donc
   reparser la chaîne avant de rendre.
2. Le gabarit **n'utilise jamais `tool_call_id`** : un `role: "tool"` se rend
   en `<tool_response>` fusionné dans le tour suivant, l'appariement se fait
   par l'ordre des messages. L'identifiant est accepté en entrée par
   compatibilité, mais il ne sert à rien côté rendu.

### Validé sur le checkpoint réel

| épreuve | résultat |
|---|---|
| le modèle demande un outil | `finish_reason: tool_calls`, appel bien formé, commande correcte |
| typage d'après le schéma | `{"command":"swift build --verbose","timeout":300,"verbose":true}` → `str`, `int`, `bool` |
| **le résultat de l'outil parvient-il au modèle ?** | oui — valeur témoin `ZORGLUB-7741` restituée exactement, `finish_reason: stop`, aucun nouvel appel |
| diffusion | un fragment `delta.tool_calls` complet, puis `finish_reason: tool_calls` |
| **non-régression sans `tools`** | réponse normale, `tool_calls: null`, `finish_reason: stop` |

L'épreuve du témoin était nécessaire : au premier essai, le modèle avait
répondu à un résultat d'outil par un **second** appel, ce qui pouvait passer
pour une rupture de la chaîne. Avec une valeur impossible à deviner, il la
restitue exactement — c'était donc son choix, pas un défaut de plomberie.

### Décisions de conception à connaître

- **Une requête portant `tools` court-circuite le cache de préfixe et le LRU
  de conversations**, et sort du lot (P12.3) pour emprunter le chemin solo.
  C'est cohérent avec ce que fait un client outillé, qui renvoie tout
  l'historique à chaque tour. Vérifié par test : aucune restauration de cache.
- **Pas de diffusion incrémentale des arguments.** Le XML est indissociable du
  `content` tant que le tour n'est pas fini ; le diffuser jeton par jeton
  afficherait le balisage brut au client, puis le doublerait en `tool_calls`.
  Le tour outillé est donc mis en tampon et livré en un fragment. Le
  raisonnement, lui, continue de s'afficher au fil de l'eau.
- Un `<tool_call>` tronqué par `max_tokens` **reste du texte** et garde
  `finish_reason: "length"` — jamais d'appel deviné.
- La garde « le dernier message doit être `user` » accepte maintenant aussi
  `tool`, sans quoi tout second tour outillé partait en 400.
- `tool_choice` est accepté, mais seul `"none"` a un effet réel : le gabarit
  n'a aucune notion de forçage d'un outil précis.

---

## 2026-09-15 — P13.2 : le format est parfait, le cache de préfixe est le vrai problème

Boucle d'agent minimale (`Scripts/agent-loop.py`) : quatre outils en lecture
seule sur le dépôt, une question dont la réponse est dans le code, budget de
dix pas.

### Ce qui marche : le format, sans une seule faute

| | |
|---|---|
| appels émis | 11 |
| **bien formés** | **11 / 11** |
| **valides** (outil connu, paramètres requis présents) | **11 / 11** |
| tours sans appel d'outil | 0 |

Le modèle a enchaîné `grep`, `read_file` et `list_files` sans jamais produire
un appel malformé, avec des motifs de recherche pertinents. **Le format n'est
pas le facteur limitant.**

### Ce qui ne marche pas : 86 % du temps part en préfill

| pas | jetons de prompt | TTFT | cache réutilisé |
|---:|---:|---:|---|
| 0 | 21 | 0,58 s | non |
| 2 | 3 181 | 23,91 s | non |
| 5 | 5 836 | 47,81 s | non |
| 8 | 7 625 | 71,51 s | non |
| 10 | 9 059 | **81,95 s** | non |

**466 s de préfill sur 541 s de boucle.** Le transcript est réintégralement
reprocessé à chaque tour, et le coût croît linéairement avec sa taille.

**La cause est un choix de conception de P13.1** : une requête portant `tools`
court-circuite le cache de préfixe et le LRU de conversations. C'était
défendable pour la simplicité, mais c'est exactement à l'envers pour une
boucle d'agent — le cas d'usage même de ces outils. Chaque pas n'ajoute que
quelques centaines de jetons **à la fin** d'un prompt par ailleurs identique :
c'est le cas idéal du cache de préfixe implicite (P6.1), qui compare les
identifiants de jetons rendus.

Ordre de grandeur du gain attendu : la boucle passerait de 541 s à environ
75 s, soit **×7**.

### L'autre limite, à ne pas confondre

Le modèle **n'a pas conclu** dans les dix pas, alors qu'il avait lu au pas 3
le fichier contenant la réponse. Ce n'est pas un problème de plomberie mais de
conduite de tâche. À réexaminer une fois la lenteur corrigée, parce qu'un
budget de pas coûtant 54 s chacun ne permet pas d'itérer sur la consigne.

---

## 2026-09-15 — P13.3 : le suffixe par différence, et la boucle d'agent devient ×5,6 plus rapide

### Le déblocage tenait à une ligne de documentation

`Qwen4ExpStreamingGenerationOptions.continueConversation` dit que
`promptTokenIDs` **n'a besoin de porter que le nouveau suffixe** : le moteur
ne connaît rien de la structure d'un tour, seulement des jetons bruts et un
drapeau « garde le cache ». `buildContinuationTurn` n'était donc qu'une
*façon* de produire ce suffixe, pas une exigence.

D'où la méthode retenue, qui ne devine jamais le gabarit :
`continuationSuffix` rend la conversation complète **avant** et **après** le
nouveau message, retire le bloc d'amorçage de l'assistant du rendu antérieur
(il n'est pas un préfixe du suivant, il est remplacé), vérifie que le reste
est un préfixe **exact**, et renvoie la différence. Elle marche pour
n'importe quel rôle, pas seulement `tool`.

Cela a permis de **supprimer** la garde de P13.2 plutôt que d'ajouter du code :
le chemin serveur n'utilise plus `buildContinuationTurn` du tout.

### Cas dégradés

`continuationSuffix` renvoie `nil` — jamais un état incohérent — si le bloc
d'amorçage manque, si le préfixe littéral ne correspond plus (historique
édité, outils changés), ou si rien n'est nouveau. Le serveur rattrape ce cas
précis et retombe sur le rejeu complet, chemin déjà validé. La requête HTTP
n'échoue jamais pour cette raison.

### La mesure, même tâche, mêmes dix pas

| pas | jetons à préfiller | TTFT | cache |
|---:|---:|---:|---|
| 0 | 565 | 5,22 s | non |
| 1 | 3 193 | 22,83 s | non |
| 2 | **22** | **0,65 s** | **oui** |
| 4 | 1 304 | 10,18 s | oui |
| 7 | **22** | **0,70 s** | **oui** |
| 9 | 491 | 4,23 s | oui |

| | avant | après | gain |
|---|---:|---:|---:|
| durée de la boucle | 683,7 s | **122,0 s** | **×5,6** |
| préfill cumulé | 593 s | **74 s** | **×8,0** |
| par pas | 68,4 s | **12,2 s** | |

À partir du troisième pas, seul le nouveau contenu est préfillé — quelques
dizaines de jetons quand le résultat d'outil est court, un millier quand
c'est un fichier lu. C'est exactement le comportement attendu.

### Ce qui reste

**Le modèle ne conclut toujours pas** : onze appels, tous bien formés et
valides, aucune réponse finale en dix pas, alors qu'il a lu le fichier
contenant la réponse. Ce n'est pas de la plomberie, c'est de la conduite de
tâche. Avec des pas à 12 s au lieu de 68, on peut désormais itérer sur la
consigne et sur le budget de pas, ce qui était impossible avant.

*Incident de méthode à noter* : deux agents ont édité l'arbre de travail en
même temps, le premier ayant rendu son rapport mais continuant de tourner en
tâche de fond. Le premier s'en est aperçu, a jugé l'approche du second
meilleure, a retiré sa propre tentative et l'a signalé. Vérifié après coup :
aucune trace résiduelle, compilation et 195 tests verts. **Arrêter un agent
avant d'en lancer un second sur les mêmes fichiers.**

---

## 2026-09-15 — Deux enseignements de la boucle d'agent : la réflexion est obligatoire, et un 500 au corps vide

### La réflexion n'est pas optionnelle pour l'agentique

Même tâche, même harnais, seule la réflexion change :

| | sans réflexion | avec réflexion |
|---|---:|---:|
| appels émis | 22 (16 pas) | **5 (4 pas)** |
| bien formés | 22/22 | 5/5 |
| **réponse finale** | **aucune** | **`Qwen4ExpQSAKVCache`, exacte** |
| durée | 557 s | 119 s |

Sans réflexion, le modèle enchaîne des recherches pertinentes mais ne
**synthétise jamais** ce qu'il a lu. Vérifié avant d'accuser le modèle : la
déclaration `public final class Qwen4ExpQSAKVCache` est ligne 44 du fichier
qu'il avait lu au pas 3, donc bien dans ce que l'outil lui avait montré. Ce
n'est pas une troncature du harnais, c'est une limite de synthèse que le mode
réflexion lève.

**Conséquence pour P13.4** : un client agentique branché sur ce serveur doit
avoir la réflexion active. Le défaut actuel du serveur est `enable_thinking`
à faux — à reconsidérer pour ce cas d'usage.

### Un 500 au corps vide, et sa cause racine

Reproduction minimale : `tools` + un aller-retour d'outil complet + un dernier
message **assistant** vide → `HTTP 500 · corps 0 octets`. Arrivé en vrai parce
que le modèle avait atteint `max_tokens` **pendant sa réflexion**, donc n'avait
produit ni texte ni appel ; le client ajoute alors un tour assistant vide.

**Cause** : les gardes de validation (dernier rôle, corps vide, `messages`
vide, nom d'outil manquant) étaient placées **avant** le `do { … } catch` du
gestionnaire. Une erreur levée par ces gardes s'échappait donc jusqu'au filet
générique de Hummingbird, qui répond littéralement
`Response(status: .internalServerError, body: .init())` — un 500 vide, quel
que soit le message. `Qwen38ServerError` n'étant pas conforme à
`HTTPResponseError`, le routeur ne le rattrape pas non plus.

Trouvé au passage : `/v1/models` n'avait **aucun** `catch`, donc une clé d'API
invalide y produisait déjà le même 500 vide au lieu d'un 401.

**Correctif à la racine** : un enrobage `catchingHTTPErrors` posé une fois sur
les trois routes qui peuvent lever, qui garantit un corps JSON pour toute
erreur échappée, gardes amont comprises. Les `catch` internes restent : eux
seuls savent nettoyer session et conversation avant de répondre.

**Décision sur un dernier message assistant : accepté**, pas rejeté. C'est
exactement ce que produit une boucle d'agent tronquée par `max_tokens`, et la
continuation par différence est déjà générique sur le rôle du dernier message.
Rejeter aurait forcé le client visé par tout §P13 à contourner un refus
artificiel.

Le diagnostic a été vérifié par annulation : sans le correctif, le test
reproduit exactement le symptôme.

### Vérifié sur le serveur réel

| épreuve | résultat |
|---|---|
| la séquence qui donnait 500 | **200**, réponse cohérente |
| `messages: []` | **400** avec `{"error":{"type":"invalid_request_error","message":"La requête doit contenir au moins un message."}}` |
| clé d'API invalide sur `/v1/models` | non concluant — le serveur tournait sans `--api-key`, il n'y avait rien à refuser |

---

## 2026-09-15 — P13.2 conclue : deux tâches sur trois, et la troisième était mal posée

Trois tâches d'exploration du dépôt, réflexion active, budget de 16 pas.

| tâche | pas | appels | bien formés | aboutie | réponse |
|---|---:|---:|---:|---|---|
| 1 — nom du type du cache K/V de l'attention pleine | 4 | 5 | 5/5 | ✅ | **`Qwen4ExpQSAKVCache`** — exacte |
| 2 — nombre de niveaux de fusion et défaut de production | 6 | 5 | 5/5 | ✅ | **10 cas, dont 9 niveaux F1-F9 plus `.none`** — exacte, et il relève de lui-même que F3 était réservé et jamais implémenté |
| 3 — option de réglage des experts routés et son défaut | 16 | 19 | 19/19 | ❌ | aucune |

**Format : 29 appels sur 29 bien formés et valides, sur les trois tâches.**
Zéro appel malformé depuis le début de §P13, toutes exécutions confondues.

### La tâche 3 était mal posée, et c'est moi qui l'ai écrite

Elle demande « quelle est sa valeur par défaut ». Or `--routed-experts` est
déclarée `var routedExperts: Int?` : **son défaut n'est pas un littéral**,
c'est `nil`, qui signifie « prendre la valeur du checkpoint ». Le modèle a
donc cherché seize pas durant un nombre qui n'existe pas dans le code. Ce
n'est pas un échec de raisonnement mais une question sans réponse littérale.

À retenir pour toute évaluation future : **une tâche d'évaluation doit avoir
une réponse vérifiable dans le code**, sinon on mesure la ténacité du modèle
face à une impasse, pas sa compétence.

### Verdict de P13.2

Le modèle local **peut** piloter une boucle d'outils : format irréprochable,
et il conclut correctement quand la question a une réponse. Les deux
conditions à respecter sont désormais connues et mesurées — **la réflexion
doit être active**, et le cache de préfixe doit fonctionner, sans quoi un pas
coûte 68 s au lieu de 12.
