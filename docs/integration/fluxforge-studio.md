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
- Deux profils, `fast` et `lean`, à la manière de YuE2 ; `lean` vise un Mac
  16 Go.
- Le modèle conseillé est **Bonsai 2** (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`,
  8,6 Go sur disque) : un Qwen3.8-27B ternaire, 2 bits, contexte 262 k.

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

(Remplis par K-5 ; voir `BENCHMARKS.md` pour les lignes brutes.)

## Limites connues

- Un seul modèle résident par `Qwen38Brain` ; les requêtes sont servies une à
  la fois.
- `lean` charge le modèle sans vision : une image est refusée.
- La réutilisation de conversation vaut pour Bonsai 2 et la famille Qwen 3.5
  dense ; Flash-Next passe par le chemin historique du runtime.
- mlx-swift-lm est suivi sur `main` tant qu'aucune version publiée ne contient
  Qwen 3.5 : noter la révision résolue.

## Preuve d'intégration

(Remplie par K-9.)
