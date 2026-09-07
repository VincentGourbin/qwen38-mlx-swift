# Qwen3.8 MLX Swift

Inférence locale Qwen3.8 (27B et Flash-Next) sur Apple Silicon via MLX Swift,
avec CLI, GUI de bench et serveur d'inférence LAN compatible OpenAI.

## Build et tests

```sh
Scripts/build.sh       # xcodebuild, jamais `swift build` (voir docs/knowledge/)
Scripts/run-tests.sh   # suite de tests, derived data dédié
```

Binaires : `.xcodebuild/Build/Products/Debug/qwen38` (CLI),
`.xcodebuild/Build/Products/Debug/qwen38-bench-ui` (GUI).

## Mémoire GPU (wired limit)

Le M3 Max limite par défaut la mémoire wired GPU à ~72 Go. Pour la résidence
Flash-Next (pic mesuré 79-82 Go), augmenter la limite avant de lancer une
session longue :

```sh
sudo sysctl iogpu.wired_limit_mb=85000
```

Ce réglage revient à la valeur système au redémarrage — à refaire à chaque
session si nécessaire.

## Serveur LAN (`qwen38 serve`)

- API compatible OpenAI (`/v1/models`, `/v1/chat/completions`, `/healthz`,
  `/metrics`). Un seul modèle résident par process ; les requêtes sont
  sérialisées en FIFO (pas de continuous batching).
- **Timeout client** : sur Flash-Next, le premier tour d'un modèle
  fraîchement sélectionné peut prendre ~100 s (chargement des couches
  depuis le disque externe) avant même de commencer le préfill. En mode
  **streaming**, le serveur envoie un commentaire SSE `: loading` toutes
  les 10 s pendant cette attente pour éviter qu'un proxy ou un client ne
  coupe la connexion sur un timeout d'inactivité (souvent 60 s). En mode
  **non-stream**, il n'y a pas d'équivalent possible (la réponse HTTP est
  atomique) : configurer un timeout client d'**au moins 300 s** sur ce
  modèle, ou préférer le streaming.
- Pour une session longue (téléchargement, quantification, serveur en
  continu), utiliser le skill `mac-awake` pour empêcher la mise en veille
  sans forcer l'écran à rester allumé.

## Documentation technique

Voir `docs/knowledge/index.md` (parités, régressions, pièges) et `PLAN.md`
(plan d'implémentation, journal d'exécution).
