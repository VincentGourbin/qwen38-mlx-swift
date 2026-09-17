# Qwen3.8 MLX Swift — knowledge base

Les observations durables, parités et régressions sont documentées ici. Les
mesures d'inférence utilisent `swift-mlx-profiler` et doivent être reproductibles
sur le même Mac avec une GPU calme.

## Contenu

- [`log.md`](log.md) — le journal d'exécution daté : chaque mesure, chaque
  hypothèse réfutée, chaque correctif, dans l'ordre où ils sont arrivés. C'est
  la source de vérité des chiffres cités dans `BENCHMARKS.md` et le README.
- [`investigations/p3-expert-offload.md`](investigations/p3-expert-offload.md) —
  déchargement disque des experts : étude d'architecture et pourquoi la piste
  a été abandonnée.
- [`investigations/mtp-m2-persistent-conversation.md`](investigations/mtp-m2-persistent-conversation.md) —
  le pipeline MTP persistant multi-tour.
- [`../reference/claude-code-wire-format.md`](../reference/claude-code-wire-format.md) —
  ce que Claude Code envoie réellement sur le fil.

## Ailleurs dans `docs/`

- [`../parity-method.md`](../parity-method.md) — la méthode de parité
  Python/Swift et la régénération des fixtures *(anglais)*.
- [`../architecture.md`](../architecture.md) — les cinq modules du dépôt
  *(anglais)*.
