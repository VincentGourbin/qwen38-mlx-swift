# Plan d'implémentation — Bonsai 2 comme cerveau d'app, sans fork, profils lean/fast

> Plan d'exécution autonome pour un agent (pi, cerveau cloud ou Sonnet).
> Rédigé le 2026-09-27 après une sonde qui **charge et fait répondre Bonsai 2
> contre mlx-swift-lm upstream**, sans `Vendor/mlx-swift-lm`
> (`Scripts/bonsai2-upstream-probe/`). Chantier **P16** dans `PLAN.md`.

## 0. Mode d'exécution

- **Une fiche à la fois, dans l'ordre.** Une fiche n'est terminée que lorsque
  sa *porte de sortie* a été observée dans la sortie d'une commande pendant la
  session. Recopie la ligne observée dans le journal (§8).
- **Blocage** : STOP, écris la question dans le journal (gabarit ASK), termine
  ta réponse par `FICHE K-x BLOQUÉE`. Ne contourne pas une porte.
- **Mesurer avant d'affirmer** (règle de YuE2) : tout chiffre de débit ou de
  mémoire vient du binaire **Release**, machine au repos (aucun autre process
  MLX : `pgrep -fl "qwen38|gemma|yue2|Bonsai2"` vide), refroidissement de
  120 s entre deux mesures, ordre **A/B/B/A** pour toute comparaison, une
  ligne par mesure dans `BENCHMARKS.md` du nouveau dépôt. Un levier = une
  fiche = une comparaison. Un gain < 5 % est du bruit et ne se garde pas.
- **Interdits** : modifier `Vendor/mlx-swift-lm` ou un fichier de ce dépôt
  hors `docs/bonsai2-brain/plan.md` (journal) ; dépendre du fork depuis le
  nouveau dépôt ; `git push`, créer un dépôt GitHub, publier une release (Vincent
  le fait) ; commiter dans Fluxforge Studio ; relancer `qwen38 serve` sans
  que la fiche le demande ; lire un fichier de plus de 300 lignes d'un seul
  appel.
- **Construction** : `xcodebuild` (les shaders Metal de MLX ne se compilent
  pas avec `swift build`), toujours avec `-derivedDataPath .dd`.
- **Langue** : documents en français ; code, noms et commentaires de code en
  anglais.
- **Commit** : dans le nouveau dépôt, un commit par fiche validée, première
  ligne `K-x : <titre>`. Le journal de ce plan se commite dans ce dépôt-ci.

## 1. Faits vérifiés (ne pas re-dériver)

**Le conflit.** Ce dépôt dépend de `.package(path: "Vendor/mlx-swift-lm")`,
un clone local de la PR upstream #545 (toujours ouverte le 2026-09-27) plus
194 lignes de correctifs non poussés. Une app ne peut pas avoir deux paquets
d'identité `mlx-swift-lm` : dès qu'elle dépend de ce paquet, SwiftPM refuse la
résolution face à la version upstream qu'utilisent LTX et Gemma. **Même si
l'app n'utilise qu'un produit**, toutes les dépendances du paquet sont
résolues. D'où un **dépôt séparé**, pas un produit de plus ici.

**Ce que l'app résout** (Fluxforge Studio, `Package.resolved`, 2026-09-27) :

| Paquet | Résolu |
|---|---|
| mlx-swift | 0.31.6 |
| mlx-swift-lm | branche `main`, `604fae710a` |
| swift-transformers | 1.3.4 |
| swift-jinja | 2.5.0 |
| swift-mlx-profiler | 1.5.0 |

**Ce que Bonsai 2 demande au fork : presque rien.** Le modèle est un
`Qwen35` (famille `qwen3_5`, présente upstream dans `MLXVLM` depuis la 3.31.4
et dans `main`). Le seul apport du fork au chemin Bonsai 2 est un filtrage de
6 lignes des clés `.signs` dans `loadWeights`. La sonde le remplace par un
chargement qui refait les étapes publiques de `VLMModelFactory._load` et
retire `.signs` avant `update(verify: .all)`.

**Sonde** (Release, M3 Max, 2026-09-27), sur `604fae710a` puis sur la tête
de `main` :

```
chargé en 2.8 s · actif 8203 Mo
réponse : La capitale de l'Australie est Canberra.
génération 3.7 s · pic 8850 Mo · actif 8385 Mo
```

**Le modèle** (`config.json` du pack, `~/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) :

| Grandeur | Valeur |
|---|---|
| Couches | 64, dont 16 d'attention complète (1 sur 4), 48 GatedDeltaNet |
| Attention | 24 têtes, 4 têtes KV, `head_dim` 256 |
| Cache KV fp16 | **64 Kio par jeton** : 2,1 Go à 32 k, 6,6 Go à 100 k, 17,2 Go à 262 k |
| État GDN | 151 Mo en fp32, constant, quelle que soit la longueur |
| Poids | 8,6 Go, dont **0,92 Go de tour de vision** |
| Contexte max | 262 144 jetons |

**Référence actuelle** (`qwen38 serve`, fork, après le correctif
`Memory.cacheLimit`, `docs/knowledge/log.md` 2026-09-19) :

| Prompt | Préfill | Décodage | Pic mémoire |
|---|---:|---:|---:|
| 1 k | 123 tok/s | 16,7 tok/s | 13 Go |
| 10 k | 104 tok/s | 11,4 tok/s | 20 Go |
| 30 k | 55 tok/s | 8,7 tok/s | 23 Go |
| 100 k | 51 tok/s | 5,3 tok/s | 35 Go |

Et, sur le banc LangWatch (L-4) : **0 jeton réutilisé** en cache sur 175
appels outillés. Chaque tour d'agent repréremplit tout l'historique.

**Toits théoriques** (M3 Max, 400 Go/s) : le décodage lit ~8 Go de poids par
jeton, soit un toit de ~50 tok/s à contexte court (24 tok/s mesurés à
14 jetons). À 100 k, poids + KV ≈ 14,6 Go par jeton, toit ~27 tok/s contre
5,3 mesurés. Il y a de la marge des deux côtés.

**Le modèle d'organisation à imiter : YuE2** (`~/Developpements/YuE2-mlx-swift`,
`Sources/YuE2Core/Configuration/ReferenceProfiles.swift` et
`Sources/YuE2Core/Memory/YuE2MemoryManager.swift`). Des profils nommés
`fast`/`lean`, chacun un jeu de réglages existants, applicables d'un appel,
remesurés par un script ; des limites mémoire calculées depuis la mémoire
disponible (`cache = min(1 Go, dispo/6)`, `memoryLimit = dispo − 1,25 Go`,
jamais sous 3 Go) plutôt que figées.

## 2. Ce qu'on construit

Un dépôt **`bonsai2-swift-mlx`**, local dans `~/Developpements/bonsai2-swift-mlx`
(Vincent le poussera), sur le modèle de `gemma-4-swift-mlx` :

- produit bibliothèque **`Bonsai2Brain`** : chargement, profils, conversation,
  appels d'outils, rapport mémoire ;
- produit exécutable **`bonsai2`** : `info`, `chat`, `bench`, `parity` ;
- dépendances **exactement alignées sur l'app** : mlx-swift
  `.upToNextMinor(from: "0.31.6")`, mlx-swift-lm `branch: "main"` (comme LTX
  dans l'app), swift-transformers `from: "1.3.3"`, swift-jinja
  `from: "2.5.1"`, swift-mlx-profiler `from: "1.5.0"` (cible CLI seulement).

API visée (à tenir, les noms peuvent être affinés en K-3 sans changer la forme) :

```swift
public actor Bonsai2Brain {
    public static func load(from directory: URL, profile: Bonsai2Profile) async throws -> Bonsai2Brain
    public func respond(to messages: [Bonsai2Message], tools: [Bonsai2Tool],
                        options: Bonsai2Options) -> AsyncThrowingStream<Bonsai2Event, Error>
    public func resetConversation()
    public func memoryReport() -> Bonsai2MemoryReport
    public func unload()
}
public enum Bonsai2Event: Sendable {
    case reasoning(String), text(String)
    case toolCall(name: String, argumentsJSON: String)
    case done(Bonsai2Usage)   // prompt, dont en cache, sortie, tok/s
}
```

Ce dépôt-ci garde le fork pour Flash-Next (MTP). Le code Bonsai 2 est
**dupliqué** le temps que ce dépôt sorte lui aussi du fork (hors plan, §7).

## 3. Fiches

### K-0 — Référence mesurée sur la machine du jour

Les chiffres de §1 datent du 19/09. Les remesurer avant tout gain annoncé.

1. Release courant de ce dépôt (`Scripts/build-release.sh` si le binaire est
   plus ancien que le dernier commit de `Sources/`).
2. Trois prompts de 1 k, 10 k, 30 k jetons construits comme en B-6
   (`docs/bonsai2/plan.md`, fiche B-6 : concaténation de fichiers du dépôt,
   greedy, 128 jetons générés). Écrire les trois prompts dans
   `docs/bonsai2-brain/prompts/{1k,10k,30k}.txt` pour que K-4 rejoue
   **exactement** les mêmes. Relever préfill, TTFT, décodage, pic mémoire.
3. Régénérer la référence de parité Python de B-3 si elle manque
   (`parity/bonsai2-reference.safetensors`, non commitée ; script
   `Scripts/references/bonsai2_reference.py`, venv `venv-bonsai2`, voir
   `docs/parity-method.md`, section Bonsai 2). C'est l'oracle de K-2 :
   indépendant du fork comme d'upstream.

**Porte** : une ligne par prompt dans le journal, les trois fichiers de
prompts présents, et `parity/bonsai2-reference.safetensors` présent (sha256
de ses 1 Mo initiaux dans le journal).

### K-1 — Le dépôt et le paquet, depuis la sonde

1. `git init ~/Developpements/bonsai2-swift-mlx`, `Package.swift` selon §2,
   `Sources/Bonsai2Brain/` avec les trois fichiers copiés de la sonde
   (`Qwen38HadamardModules.swift`, `Qwen38Bonsai2Loader.swift`,
   `Qwen38TokenizerLoader.swift`) **renommés** en `Bonsai2*` (types compris)
   et `Bonsai2Loading.swift` qui reprend `loadBonsai2(directory:)` de
   `Scripts/bonsai2-upstream-probe/Sources/Bonsai2Probe/main.swift`.
2. `Sources/Bonsai2CLI/` : `bonsai2 info <dir>` (config, mémoire après
   chargement) et `bonsai2 chat <dir> "<prompt>"`.
3. README court : ce que c'est, dépendances, pourquoi pas de fork.

**Porte** : `xcodebuild … build` sans avertissement dans `Sources/`, puis
`bonsai2 chat ~/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit "Quelle est la
capitale de l'Australie ?"` répond « Canberra ».

### K-2 — Parité avec le chemin du fork

Même modèle, deux piles : les jetons doivent être les mêmes.

1. `bonsai2 parity <dir> --reference <chemin>/parity/bonsai2-reference.safetensors` :
   pour chacune des 4 invites de B-3, mêmes `prompt_ids_i` (les comparer
   d'abord : un prompt différent invalide tout le reste), logits de la
   dernière position contre `last_logits_i` avec la tolérance de B-3, puis
   32 ids greedy contre `greedy_ids_i`. Porter la logique du test Swift de
   B-3 (`Tests/Qwen38Tests/Qwen38Tests.swift`, gardé par
   `QWEN38_BONSAI_FIXTURE`), pas la réinventer.
2. En cas d'écart : premier jeton divergent et écart de logits à cette
   position (méthode `docs/parity-method.md`). Au-delà de la tolérance : ASK
   (le `Qwen35` upstream diffère du fork de 570 lignes sur
   `MLXVLM/Models/Qwen35.swift`).

**Porte** : `4/4 invites : prompt identique, logits sous tolérance, greedy
32/32`, le même résultat que B-3 sur le fork.

### K-3 — L'API cerveau : flux, réflexion, outils

1. `Bonsai2Brain` selon §2, par-dessus `ModelContainer` et le générateur
   upstream (`generate(input:parameters:context:)`), en flux.
2. Réflexion : `enable_thinking` et `reasoning_effort` (`low`/`medium`/`xhigh`
   seulement, le gabarit refuse `high`) passés au gabarit ; le contenu entre
   `<think>` et `</think>` sort en `.reasoning`, le reste en `.text`.
3. Outils : reprendre de ce dépôt `Qwen38OrderedJSON.swift` et la mise en
   forme ordonnée des specs (`Qwen38ToolCalling.swift`, `orderedSpec`,
   `templateValue`), sans quoi `tojson` ne rend pas le prompt vu à
   l'entraînement (`docs/knowledge/log.md`, 2026-09-18). Les appels sortent en
   `.toolCall`, jamais en XML dans `.text`.
4. `.done` porte l'usage : jetons de prompt, dont réutilisés, sortie, débit.

**Porte** : un test (`swift test` via `xcodebuild test`) qui déroule une
boucle outil sur le pack réel : question → `.toolCall(list_files)` → résultat
injecté → réponse `.text` qui cite un fichier. Plus un test sans modèle du
rendu `tojson` ordonné (octets identiques à la référence Python de ce dépôt).

### K-4 — Banc de mesure intégré

1. `bonsai2 bench <dir> --profile <p> --context 1k,10k,32k` : préfill, TTFT,
   décodage sur 256 jetons, pic et actif (`Memory.snapshot()`), `phys_footprint`.
   Une ligne JSON par mesure, ajoutée à `BENCHMARKS.md`.
2. `--agent-replay <fichier>` : rejoue une conversation outillée enregistrée
   (6 à 10 tours, prendre une trace du banc LangWatch L-4 exportée en JSON) et
   mesure le TTFT de chaque tour et les jetons réutilisés. C'est la mesure qui
   compte pour un cerveau d'agent.
3. Instrumentation `swift-mlx-profiler` optionnelle (`--profile-trace`).

**Porte** : K-0 rejoué avec `bonsai2 bench` sur `docs/bonsai2-brain/prompts/`
donne des chiffres à ±10 % de `qwen38 serve` (même modèle, même machine), et `--agent-replay` affiche une
ligne par tour.

### K-5 — Profils `fast` et `lean`

Sur le modèle de `YuE2ReferenceProfile`. Chaque champ correspond à un réglage
existant ; `applyGlobalPolicy()` pose les réglages de processus.

| Réglage | `fast` | `lean` |
|---|---|---|
| Tour de vision | chargée si `vision: true` | **jamais** (−0,92 Go ; filtrer `vision_tower.*`, `verify` adapté) |
| Cache KV | fp16 | **8 bits** sur les 16 couches d'attention (`QuantizedKVCache` upstream) |
| Contexte max | 262 k | 32 k par défaut, réglable |
| `Memory.cacheLimit` | 4 Go | `min(1 Go, dispo/6)` |
| `Memory.memoryLimit` | non posé | `max(4 Go, dispo − 2 Go)` |
| Préfill par tranches | valeur retenue en K-7 | 512 (pic transitoire bas) |
| `clearCache()` | jamais | après chaque réponse |

« dispo » = `os_proc_available_memory()` sur iOS, `ProcessInfo.physicalMemory
− 8 Go` sur macOS, surchargeable par `BONSAI2_AVAILABLE_MB` pour simuler un
Mac 16 Go sur le M3 Max.

**Porte** : `bench` des deux profils à 1 k/10 k/32 k. Objectif `lean` :
**pic ≤ 12 Go à 32 k** avec `BONSAI2_AVAILABLE_MB=16384`, soit un Mac 16 Go.
Et **qualité** : `parity` en `lean` garde prompt et logits sous tolérance
sur 4/4 et au moins 3/4 greedy 32/32 (le KV 8 bits peut faire diverger
tard), et la boucle outil de K-3 passe.

### K-6 — Levier 1 : réutiliser la conversation (le plus gros gain attendu)

Le cerveau d'une app d'agent renvoie à chaque tour tout l'historique. Sans
réutilisation, le TTFT croît avec la conversation (20 à 60 s par appel en
L-4). Les caches GDN ne se **rognent pas** : on ne peut pas revenir en
arrière dans un état récurrent.

1. Après le préfill de chaque prompt, **avant** de générer, garder un
   instantané : l'état des 48 caches GDN (151 Mo), l'offset, et les jetons du
   prompt.
2. Au tour suivant, rendre le prompt complet. S'il commence exactement par
   les jetons gardés : restaurer l'état GDN, **rogner le KV** des 16 couches
   d'attention à l'offset gardé, préremplir seulement le suffixe. Sinon :
   tout repréremplir. (Le gabarit Qwen retire la réflexion des tours passés :
   le point de reprise fiable est la fin du prompt précédent, pas la fin de
   la réponse.)
3. **Invariant vérifié à chaque reprise** : pour chaque couche d'attention,
   nombre de clés stockées == offset. Un écart lève une erreur Swift, jamais
   un `fatalError` MLX. C'est exactement le défaut qui tue le serveur
   Flash-Next (`PLAN.md` §P15 : masque de 812 clés pour 813 stockées).
4. Test : conversation interrompue au milieu d'une génération (tâche
   annulée), puis tour suivant : pas de plantage, reprise ou repréfill propre.

**Porte** : `--agent-replay` en A/B/B/A sans/avec : jetons réutilisés > 80 %
du prompt à partir du 2e tour, TTFT médian divisé par au moins 3, et les
réponses greedy identiques avec et sans réutilisation.

### K-7 — Levier 2 : préfill

1. Balayer la taille de tranche de préfill (512, 1024, 2048, 4096) à 10 k et
   32 k, profil `fast` : débit et pic.
2. Vérifier avec le profiler que le noyau GDN upstream est bien pris
   (`Dk % 32 == 0`, ici 128) et que le `sdpa` des couches d'attention ne
   matérialise pas un masque plein à 32 k.

**Porte** : taille retenue pour `fast` si gain ≥ 5 %, sinon on garde le
défaut ; ligne A/B/B/A dans `BENCHMARKS.md`.

### K-8 — Levier 3 : décodage (Hadamard, décodage compilé, mémoire câblée)

Mesurer d'abord, choisir ensuite, un levier par comparaison :

1. **Part de la rotation Hadamard** dans un pas de décodage (profiler) : si
   elle dépasse 15 %, essayer de fusionner la multiplication par les signes
   dans la transformée (un passage mémoire de moins par projection). En
   dessous, ne rien faire et le noter.
2. **Décodage compilé** upstream (`Qwen35CompiledDecode…` dans `main`) :
   vérifier s'il s'active avec les modules Hadamard ; si non, dire pourquoi.
3. **Mémoire câblée** (`WiredMemoryTicket` upstream) pour les 8 Go de poids :
   variance du TTFT et du débit sur 10 réponses, avec et sans.

**Porte** : décodage `fast` à 1 k et 32 k, avant/après chaque levier retenu,
en A/B/B/A. Objectif indicatif ×1,3 à 1 k ; aucun levier n'est gardé sans
gain mesuré.

### K-9 — Preuve d'intégration dans l'app

1. Dans une **copie de travail jetable** de Fluxforge Studio
   (`git worktree add ../fluxforge-bonsai2-probe`), ajouter le paquet par
   chemin local (`../bonsai2-swift-mlx`).
2. `xcodebuild -resolvePackageDependencies` : **une seule** entrée
   `mlx-swift-lm` dans le `Package.resolved`, aucune erreur de conflit.
3. `xcodebuild build` de l'app sans erreur.
4. Supprimer la copie de travail. Rien n'est commité dans l'app.

**Porte** : les deux commandes réussissent ; journal avec la ligne
`mlx-swift-lm` résolue. Puis README complet, `CHANGELOG.md`, tag local
`v0.1.0` (pas de push).

## 4. Ordre

K-0 → K-1 → K-2 → K-3 → K-4 → K-5 → K-6 → K-7 → K-8 → K-9.
K-9 peut être fait juste après K-1 si Vincent veut la preuve d'intégration
tôt ; il suffit alors de le refaire à la fin.

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
6. **Branche `main` mouvante** : noter la révision résolue dans chaque ligne de
   `BENCHMARKS.md`. Si `main` casse la compilation, épingler la dernière
   révision qui marche et le dire (ASK si la seule issue est de modifier
   mlx-swift-lm).

## 6. Commandes de référence

```bash
# sonde d'origine
cat Scripts/bonsai2-upstream-probe/README.md
# construction du nouveau dépôt
cd ~/Developpements/bonsai2-swift-mlx
xcodebuild -scheme bonsai2 -destination 'platform=macOS' -derivedDataPath .dd -configuration Release build -quiet
./.dd/Build/Products/Release/bonsai2 info ~/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
# aucune autre app MLX
pgrep -fl "qwen38|gemma|yue2|Bonsai2|ltx"
```

## 7. Hors plan (noté pour plus tard)

- **Sortir ce dépôt-ci du fork.** Upstream `main` contient désormais
  `Qwen35MTP` et ses enregistrements ; l'écart restant est la PR #545 et
  194 lignes locales (décalage d'offset MTP, `SwitchLayers`). À évaluer
  séparément ; alors ce dépôt pourra consommer `bonsai2-swift-mlx` au lieu de
  dupliquer le code.
- **Décodage spéculatif** : le pack Bonsai 2 n'a pas de tête MTP. Un drafter
  externe serait un autre chantier.
- **iPhone** : 8,6 Go de poids, hors de portée d'un iPhone. `lean` vise les
  Mac 16 Go.

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
