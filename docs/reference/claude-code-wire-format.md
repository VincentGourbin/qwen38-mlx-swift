# Ce que Claude Code envoie réellement — capture du 2026-09-15

Capturé en pointant `ANTHROPIC_BASE_URL` vers un serveur local qui enregistre
puis répond 500. **Aucune supposition** : tout ce qui suit vient d'une requête
réelle. Le contenu (prompts, identifiants de session) n'est pas reproduit ici,
seule la **structure**.

## Transport

```
POST /v1/messages?beta=true
anthropic-version: 2023-06-01
anthropic-beta: claude-code-20250219,interleaved-thinking-2025-05-14,…
anthropic-dangerous-direct-browser-access: true
Content-Type: application/json
Accept-Encoding: gzip, deflate, br, zstd
```

L'authentification passe par un en-tête `Authorization`, alimenté par
`ANTHROPIC_AUTH_TOKEN`. Huit requêtes sont parties pour un seul prompt : le
client **réessaie** après une erreur.

## Champs du corps

| champ | forme observée |
|---|---|
| `model` | `"claude-sonnet-5"` — c'est `ANTHROPIC_DEFAULT_SONNET_MODEL` qui le remplace |
| `max_tokens` | `64000` |
| `stream` | **`true` — toujours**, la diffusion n'est pas optionnelle |
| `system` | **liste de 3 blocs** `{type:"text", text:…}`, **pas une chaîne** |
| `messages` | liste de blocs de contenu typés, jamais du texte nu |
| `tools` | **liste de 59 outils**, clés `['description', 'input_schema', 'name']` |
| `thinking` | `{"type": "adaptive", "display": "omitted"}` |
| `output_config` | `{"effort": "high"}` |
| `context_management` | `{"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}` |
| `metadata` | identifiants de session et d'appareil |

Un outil porte `name`, `description`, et `input_schema` (un schéma JSON avec
`['$schema', 'additionalProperties', 'properties', 'required', 'type']`) — **pas** l'emboîtement
`{"type":"function","function":{…}}` du dialecte OpenAI.

## Le chiffre qui commande la faisabilité

**126775 caractères de schémas d'outils, soit environ
31693 jetons, dans *chaque* requête.**

C'est le coût de préfill à payer au premier tour d'une session. Le cache de
préfixe l'amortit ensuite, tant que la liste d'outils ne change pas — mais
elle change dès qu'un serveur MCP est branché ou débranché.

## Ce qu'il faut en conclure pour la traduction

- La route doit accepter `?beta=true` et les champs inconnus sans broncher
  (`context_management`, `output_config`, `metadata` n'ont pas d'équivalent
  chez nous, il faut les ignorer proprement).
- `system` en liste de blocs → à concaténer vers notre message système.
- Les outils Anthropic → notre format interne : `input_schema` devient
  `parameters`.
- `stream: true` étant systématique, la diffusion SSE à événements nommés
  n'est pas un raffinement, c'est le chemin principal.
- `thinking` est présent : cohérent avec la mesure du 2026-09-15, où le mode
  réflexion s'est révélé **nécessaire** pour que le modèle conclue une tâche
  agentique.
