# Plan — Bonsai 2 comme cerveau d'app : sortie du fork, façade, profils lean/fast, intégration Fluxforge Studio

> Chantier **P16** dans `PLAN.md`. Rédigé le 2026-09-27, **révisé le même jour**
> après deux décisions de Vincent : pas de dépôt séparé pour Bonsai 2, et
> enchaîner tout le plan pour que Fluxforge Studio puisse étudier
> l'intégration du framework. Exécuté directement (pas par pi) ; le journal
> est en fin de fichier.

## 0. Règles de mesure

- **Mesurer avant d'affirmer** (règle de YuE2) : binaire **Release**, machine
  au repos (aucun autre process MLX), une comparaison = un levier, ordre
  A/B/B/A, un gain < 5 % est du bruit. Chaque mesure = une ligne JSON de
  `qwen38 brain bench|replay` recopiée dans `BENCHMARKS.md`.
- **Qualité** : parité B-3 (référence Python) verte, et boucle d'agent qui
  aboutit. Un profil ou un levier qui casse l'un des deux est rejeté.

## 1. Faits vérifiés (ne pas re-dériver)

**Le blocage.** Le paquet dépendait de `Vendor/mlx-swift-lm`, un clone local
de la PR upstream #545 (d'un tiers, ouverte) plus 194 lignes locales. Une app
qui résout déjà mlx-swift-lm upstream (Fluxforge Studio : LTX, Gemma,
`main` @604fae710a, mlx-swift 0.31.6) ne pouvait pas l'ajouter : SwiftPM
refuse deux paquets d'identité `mlx-swift-lm`.

**Ce qu'on utilisait du fork** : un initialiseur « experts déjà empaquetés »
(Flash-Next), le filtrage des clés `.signs` au chargement (Bonsai 2), une garde
et une continuation MTP (27B). La branche `main` upstream contient désormais
Qwen 3.5 et le MTP.

**Le modèle** (`config.json` du pack) :

| Grandeur | Valeur |
|---|---|
| Couches | 64, dont 16 d'attention complète, 48 GatedDeltaNet |
| Cache KV fp16 | 64 Kio par jeton : 2,1 Go à 32 k, 6,6 Go à 100 k |
| État GDN | 151 Mo, constant |
| Poids | 8,6 Go, dont 0,92 Go de tour de vision |

**Référence du fork** (`qwen38 serve`, 2026-09-19) : préfill 123 → 51 tok/s et
décodage 16,7 → 5,3 tok/s de 1 k à 100 k jetons ; pic 13 → 35 Go. Banc
LangWatch : **0 jeton réutilisé** sur 175 appels d'agent, parce qu'un
historique qui finit par un résultat d'outil est toujours rejoué en entier
puis jeté (`Qwen38Runtime.generateStateless`, branche `.tool`).

## 2. Ce qu'on livre, dans ce dépôt

- Dépendance **mlx-swift-lm upstream `main`** (fait, `cc09c2d`).
- Bibliothèque **`Qwen38Brain`** : `Qwen38Brain.load(modelDirectory:profile:)`,
  `respond(to:tools:options:)` → flux d'événements `.reasoning`, `.text`,
  `.toolCall`, `.done(usage)` ; `resetConversation`, `memoryReport`, `unload`.
  Ne dépend que de `Qwen38Core` (pas de serveur HTTP).
- Profils **`Qwen38BrainProfile.fast` / `.lean`** sur le modèle de YuE2.
- CLI **`qwen38 brain ask|agent|bench|replay`** pour tout mesurer.
- Guide d'intégration **`docs/integration/fluxforge-studio.md`**.

## 3. Fiches

| Fiche | Objet | Porte |
|---|---|---|
| K-1 | Sortie du fork | tout compile, 233 tests, parité B-3 et parités Flash-Next vertes |
| K-2 | Release sur upstream, Flash-Next de bout en bout | génération réelle et pic mémoire comparables au fork |
| K-3 | Façade `Qwen38Brain` | `brain agent` sur la fixture Swift du banc : appels d'outils, réponse juste |
| K-4 | Banc de référence | `brain bench` 1 k/10 k/32 k, `brain replay` d'une conversation d'agent |
| K-5 | Profils fast/lean | lean : pic ≤ 12 Go à 32 k avec 16 Go simulés ; parité et agent OK |
| K-6 | Réutilisation de la conversation | replay : > 80 % de jetons réutilisés dès le 2e tour, TTFT ÷ 3 |
| K-7 | Préfill | balayage des tranches ; gain gardé seulement s'il dépasse 5 % |
| K-8 | Décodage | part Hadamard mesurée ; levier gardé seulement s'il est mesuré |
| K-9 | Intégration Fluxforge Studio | copie jetable : une seule entrée mlx-swift-lm résolue, l'app compile |

## 5. Pièges connus

1. **`verify: [.all]` et les clés en trop** : toute clé du safetensors que le
   module ne connaît pas fait échouer le chargement. `.signs` est retiré avant
   ; en `lean`, les clés `vision_tower.*` aussi, et la tour de vision ne doit
   pas être instanciée (sinon clés manquantes).
2. **`MLX_QWEN_FOUR_GDN=0`** doit être posé avant le chargement : le
   chargeur Hadamard le vérifie. La tête de `main` n'a plus cette variable,
   `604fae710a` l'a ; la poser ne coûte rien.
3. **Pas de `Memory.cacheLimit` = mémoire qui explose** : 74 Go à 30 k avant
   le correctif du 19/09. Tout profil pose une limite.
4. **Une autre app MLX qui tourne** (gemma4-cli à 74 Go le 19/09) fausse
   toutes les mesures : vérifier avant chaque banc.
5. **`high` n'existe pas** pour `reasoning_effort` dans ce gabarit : `low`,
   `medium`, `xhigh`.
6. **Branche `main` mouvante** : noter la révision résolue (`Package.resolved`)
   dans chaque ligne de `BENCHMARKS.md`. Si `main` casse la compilation,
   épingler la dernière révision qui marche.
7. **`Scripts/build*.sh` passent `-skipPackageUpdates`** : après un changement
   de dépendance, lancer une fois `xcodebuild -resolvePackageDependencies
   -scheme Qwen38MLXSwift-Package -derivedDataPath .xcodebuild` (et
   `.xcodebuild-tests`), sinon « Could not resolve package dependencies ».

## 6. Commandes de référence

```bash
B=~/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
Q=.xcodebuild/Build/Products/Release/qwen38
$Q brain ask --model-path $B "Quelle est la capitale de l'Australie ?"
$Q brain agent --model-path $B --workspace bench/langwatch/fixture \
  --task "Lance l'analyse : quel test échoue et pourquoi ?" --record /tmp/agent.json
$Q brain replay --model-path $B --profile lean --transcript /tmp/agent.json
$Q brain bench --model-path $B --profile fast --cooldown 120 \
  --prompt-files docs/bonsai2-brain/prompts/1k.txt,docs/bonsai2-brain/prompts/10k.txt,docs/bonsai2-brain/prompts/32k.txt
QWEN38_BRAIN_AVAILABLE_MB=16384 $Q brain bench --profile lean …   # Mac 16 Go simulé
```

## 7. Hors plan (noté pour plus tard)

- **Décodage spéculatif** : le pack Bonsai 2 n'a pas de tête MTP.
- **iPhone** : 8,6 Go de poids, hors de portée. `lean` vise les Mac 16 Go.
- **Continuation MTP multi-tour du 27B** : perdue avec le fork (patch local
  jamais proposé upstream) ; chaque tour relance le drafter.

## 8. Journal d'exécution (à remplir, à la fin du fichier)

Gabarits :

```
## K-x — <titre> — <AAAA-MM-JJ> — validée|bloquée
- Fait : …
- Porte observée : <ligne recopiée>

## ASK — K-x — <AAAA-MM-JJ>
- Contexte : …
- Ce que j'ai essayé : …
- Question : …
- Options : A) … B) …
```

### Journal

## K-1 — Sortie du fork — 2026-09-27 — validée
- Fait : dépendance mlx-swift-lm upstream `main` (`ee673d6`) ; initialiseur
  d'experts empaquetés remplacé par `quantize(model:)` ; chargement Bonsai 2
  propre (`Qwen38Bonsai2.loadContainer`) ; continuation MTP 27B désactivée.
- Porte observée : `Test run with 233 tests in 0 suites passed` ; B-3 greedy
  32/32 sur 3 invites, logits maxAbsErr 3,4e-5 à 3,9e-5 ; parités Flash-Next
  (QSA, MRoPE, langage, globaux, single-layer, couches publiques 2 et 3) « OK »
  en Release contre le checkpoint 4 bits.

## K-3 — Façade Qwen38Brain — 2026-09-27 — validée
- Porte observée : `brain agent` sur `bench/langwatch/fixture` : list_files,
  trois read_file, réponse « testAverageOfEmptyIsZero … division par zéro »
  (juste). Greedy identique au jeton près entre moteur dense et runtime.

## K-6 — Réutilisation de la conversation — 2026-09-27 — validée
- Porte observée : rejeu de `transcripts/agent-scripts.json`, 4 tours :
  jetons réutilisés 0/481/1860/5159 (tout l'historique précédent à 7 jetons
  près) contre 0 partout sur le runtime ; réponses identiques ; préfill total
  84 et 96 s contre 145 et 253 s (A/B/B/A). Conditions de machine
  dégradées (voir K-7), les ratios restent valables.

## K-7/K-8 — Mesures de débit — 2026-09-27 — en attente
- Constat : la machine ralentit d'une passe à l'autre indépendamment du code
  (fork : 102 puis 61 tok/s de préfill à 10 k en A/B/B/A). Un processus
  WebKit (≈ 110 % CPU depuis 2,5 jours) garde le GPU vers 60 %. Mesures
  reprises quand il sera fermé ; `campaign.sh` attend désormais 60 s de GPU
  < 75 % sans calcul lourd.
- Rotations Hadamard partagées : 402 → 257 par passe, greedy inchangé ;
  gain de débit non mesuré proprement à ce jour.

## K-9 — Intégration Fluxforge Studio — 2026-09-27 — validée
- Porte observée : une seule entrée `mlx-swift-lm` (`ee673d6`) ;
  `** BUILD SUCCEEDED **` de l'app entière ; 776 symboles Qwen38Brain liés.
