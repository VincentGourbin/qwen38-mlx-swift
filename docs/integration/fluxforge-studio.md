# Intégrer Bonsai 2 comme cerveau de Fluxforge Studio

> Pour l'équipe Fluxforge Studio. Rédigé le 2026-09-27 à partir de mesures sur
> un MacBook Pro M3 Max 96 Go (`docs/bonsai2-brain/plan.md`, journal K-1 à K-9).
> Tous les chiffres viennent de `qwen38 brain bench|replay` en Release.

## En bref

- Le paquet `qwen38-mlx-swift` dépend désormais de **mlx-swift-lm upstream,
  branche `main`**, comme LTX et Gemma dans l'app. Plus de fork : SwiftPM
  résout une seule copie de mlx-swift-lm (preuve en fin de document).
- La bibliothèque à importer est **`Qwen38Brain`** : un acteur qui charge le
  modèle, prend une conversation au format OpenAI et rend un flux
  d'événements typés (réflexion, texte, appels d'outils, usage).
- Deux profils, `fast` et `lean`, à la manière de YuE2 ; `lean` tient dans
  environ 12 Go jusqu'à 32 000 jetons de contexte, vision comprise.
- Le modèle conseillé est **Bonsai 2** (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`,
  8,6 Go sur disque) : un Qwen3.8-27B ternaire, 2 bits, contexte 262 k.

## Qualité mesurée en boucle d'agent

Banc LangWatch « Agent de code » (8 scénarios × 3 passes, juge indépendant,
serveur sur le même moteur que `Qwen38Brain`, 2026-09-27) :

| Modèle | Réussite | Poids | Pic mémoire (10k) |
|---|---:|---:|---:|
| Qwen3.8-27B 4 bits (`mlx-community/Qwen3.8-27B-4bit`) | 67 % | 16 Go | 17 Go |
| Bonsai 2 (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) | 54 % | 8,6 Go | 12,5 Go |
| Références cloud : glm-5.3-flash, kimi-k2.7-code / gpt-oss:120b | ≈ 77 % / ≈ 62 % | — | — |

Choix conseillé : le 27B 4 bits quand la machine a de la marge (32 Go et
plus), Bonsai 2 quand la mémoire est la contrainte (Mac 16 Go, cohabitation
avec Flux ou LTX). Pour évaluer vos propres usages :
[`evaluation-fluxforge.md`](evaluation-fluxforge.md).

## Ajouter le paquet

Dans Xcode : *File › Add Package Dependencies…*, puis produit **`Qwen38Brain`**
pour la cible de l'app.

- Tant que les commits du 27 septembre ne sont pas poussés, ajouter le paquet
  **par chemin local** (`../qwen38-mlx-swift`).
- Ensuite : `https://github.com/VincentGourbin/qwen38-mlx-swift`, branche
  `master`.

Ce que la résolution impose à l'app :

| Paquet | Exigence du paquet | Fluxforge Studio aujourd'hui |
|---|---|---|
| mlx-swift | `exact: 0.31.6` | 0.31.6 |
| mlx-swift-lm | `branch: main` | `main` |
| swift-transformers | `from: 1.3.3` | 1.3.4 |
| swift-jinja | `from: 2.5.1` | 2.5.0, monte en 2.5.1 |
| swift-mlx-profiler | `from: 1.5.0` | 1.5.0 |

La montée de swift-jinja en 2.5.1 est voulue : elle aligne `tojson` sur
Python, sans quoi le prompt outillé diffère de celui de l'entraînement.
Le paquet déclare aussi Hummingbird (serveur HTTP) et ArgumentParser : SwiftPM
les résout pour tout le paquet, mais le produit `Qwen38Brain` ne les lie pas.

Cible minimale : macOS 15 (l'app est en 26.2).

## Utilisation

```swift
import Qwen38Brain
import Qwen38Core   // Qwen38ChatMessage, Qwen38ToolSpec, Qwen38ToolCall

let brain = try await Qwen38Brain.load(
    modelDirectory: URL(fileURLWithPath: "/…/Ternary-Bonsai-2-27B-mlx-2bit"),
    profile: .lean)

let messages: [Qwen38ChatMessage] = [
    .init(role: .system, content: "Tu es l'assistant de Fluxforge Studio."),
    .init(role: .user, content: "Propose trois prompts pour une vidéo de plage au coucher du soleil."),
]
for try await event in await brain.respond(to: messages) {
    switch event {
    case .reasoning(let text): break                 // si options.enableThinking
    case .text(let text): print(text, terminator: "")
    case .toolCall(let call): break                  // voir ci-dessous
    case .done(let usage): print("\n", usage.tokensPerSecond, "tok/s")
    }
}
```

**Toujours renvoyer toute la conversation**, comme à une API HTTP. Le cerveau
reconnaît qu'elle prolonge la précédente et ne recalcule que la fin : c'est ce
qui garde des réponses rapides dans une boucle d'agent.

### Outils

```swift
let tools = [Qwen38ToolSpec(
    name: "generate_image", description: "Génère une image avec Flux 2.",
    parameters: .object([
        "type": .string("object"),
        "properties": .object(["prompt": .object(["type": .string("string")])]),
        "required": .array([.string("prompt")]),
    ]))]

var conversation = messages
var calls: [Qwen38ToolCall] = []
var answer = ""
for try await event in await brain.respond(to: conversation, tools: tools) {
    if case .text(let t) = event { answer += t }
    if case .toolCall(let call) = event { calls.append(call) }
}
if !calls.isEmpty {
    conversation.append(.init(role: .assistant, content: answer, toolCalls: calls))
    for call in calls {
        let result = runTool(call.name, call.argumentsJSON)   // à vous
        conversation.append(.init(role: .tool, content: result))
    }
    // puis `respond(to: conversation, tools: tools)` à nouveau
}
```

`argumentsJSON` est typé d'après le schéma (un nombre reste un nombre). Le
texte d'une réponse outillée ne contient jamais le XML `<tool_call>`.

### Images

Joindre les images au message : `.init(role: .user, content: "Décris.",
imageURLs: [url])`. Une image déjà vue dans la conversation n'est jamais
recalculée : seules les images nouvelles passent dans la tour de vision.

Par défaut, chaque image suit le budget du checkpoint, environ 1 270 jetons de
vision pour une photo : le plus de détail, environ 12 s de préfill sur un M3
Max. `Qwen38BrainOptions(imageResize: CGSize(width: 512, height: 512))`
descend à environ 200 jetons, environ 2 s.

Mesure, dialogue de 3 tours avec 2 photos, budget du checkpoint : le 3e tour
ne préremplit que 32 jetons nouveaux et reprend 2 573 jetons du cache, soit
0,6 s au lieu d'environ 25 s.

### Mémoire et cohabitation avec les autres modèles de l'app

- `await brain.memoryReport()` : mémoire active, cache, pic.
- `await brain.unload()` libère tout ; recharger coûte ~2 s (fichier en cache
  disque) à ~30 s (à froid).
- Le profil `lean` vide le cache MLX après chaque réponse : les autres
  modèles (Flux, LTX) retrouvent la mémoire entre deux réponses.
- Les réglages mémoire de MLX (`Memory.cacheLimit`, `memoryLimit`) sont
  **globaux au processus** : le profil les pose au chargement. Si l'app en
  pose d'autres pour Flux ou LTX, le dernier qui écrit gagne.

## Profils mesurés

M3 Max, Release, 2026-09-27 ; lignes brutes et conditions dans `BENCHMARKS.md`
(section P16). Le Mac 16 Go est simulé par `QWEN38_BRAIN_AVAILABLE_MB=16384`.

| Profil | Contexte | Préfill | Décodage | Pic mémoire MLX |
|---|---:|---:|---:|---:|
| `lean` | 1 k | 119 tok/s | 17,8 tok/s | 10,0 Go |
| `lean` | 10 k | 119 tok/s | 17,6 tok/s | 10,4 Go |
| `lean` | 32 k | 70 tok/s | 8,3 tok/s | 12,2 Go |
| `fast` | 1 k | 97 tok/s | 13,8 tok/s | 10,8 Go |
| `fast` | 10 k | 108 tok/s | 14,5 tok/s | 12,6 Go |
| `fast` | 32 k | 71 tok/s | 9,2 tok/s | 16,1 Go |

Les débits entre les deux profils ne sont pas directement comparables (la
machine dérive d'une passe à l'autre) ; les pics mémoire le sont.

- **`lean`** : cache KV 8 bits, tranches de préfill de 256, limites mémoire
  calculées depuis la mémoire disponible, cache MLX vidé après chaque
  réponse. Vision chargée. Pour une app qui n'envoie jamais d'image,
  `Qwen38BrainProfile.lean.textOnlyVariant()` retire la tour de vision
  (−0,9 Go) et refuse alors les images.
- **`fast`** : cache KV fp16, tranches de 512, cache MLX de 4 Go.

**Boucle d'agent** (4 tours, 16 000 jetons de prompt cumulés) : le cerveau
réutilise tout l'historique d'un tour à l'autre ; temps de préfill total
78-83 s contre 138-148 s sans réutilisation, 4e tour 30-33 s contre 76 s.

## Limites connues

- Un seul modèle résident par `Qwen38Brain` ; les requêtes sont servies une à
  la fois.
- `textOnlyVariant()` charge le modèle sans vision : une image est alors refusée.
- La réutilisation de conversation vaut pour Bonsai 2 et la famille Qwen 3.5
  dense, images comprises ; Flash-Next passe par le chemin historique du
  runtime.
- mlx-swift-lm est suivi sur `main` tant qu'aucune version publiée ne contient
  Qwen 3.5 : noter la révision résolue.

## Preuve d'intégration (K-9, 2026-09-27)

Sur une copie jetable de Fluxforge Studio (`git archive` de `b8cf2932`, le
dépôt de l'app n'a pas été touché), avec le paquet ajouté par chemin local, le
produit `Qwen38Brain` lié à la cible de l'app et un fichier qui appelle
`Qwen38Brain.load(…, profile: .lean)` puis `respond(to:)` :

- `xcodebuild -resolvePackageDependencies` : **une seule** entrée
  `mlx-swift-lm` (`main`, `ee673d6`), mlx-swift 0.31.6, swift-jinja 2.5.1,
  swift-transformers 1.3.4 ;
- `xcodebuild build` (Debug, arm64, sans signature) : **BUILD SUCCEEDED**,
  LTX, Flux 2, Gemma 4, Voxtral et Qwen38Brain compilés ensemble ;
- `Fluxforge Studio.debug.dylib` contient 776 symboles `Qwen38Brain`.

À savoir : la résolution fait avancer la révision de mlx-swift-lm de l'app de
`604fae710a` (11 septembre) à `ee673d6` (22 septembre, tête de `main`) ; toute
l'app compile avec. Tester LTX et Gemma sur cette révision avant de livrer.
