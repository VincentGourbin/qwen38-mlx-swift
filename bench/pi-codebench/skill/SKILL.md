---
name: pi-plan-decoupage
description: Découper un plan d'implémentation en fiches que pi.dev exécutera seul (modèle local via qwen38 serve, ou modèle Ollama Cloud). À utiliser dès qu'on écrit ou révise un plan, des fiches tasks/T-x.y.md, un AGENTS.md d'exécutant, ou qu'on choisit le modèle qui exécutera un plan — la granularité dépend du modèle, mesurée par le banc pi-codebench.
---

# Découper un plan pour pi.dev

Règles mesurées, pas intuitives : banc `bench/pi-codebench/` du dépôt
`qwen38-mlx-swift` (campagnes v1 et v2 du 2026-09-28 ; cloud : un run par
combinaison ; Qwen3.8-27B local : trois runs par découpage). Même
fonctionnalité Swift (≈ 300 lignes, 4 tâches, 5 fichiers), même harnais
pi 0.87.1, notation par tests d'acceptation cachés.

| Modèle | 1 fiche pour tout | 1 tâche par fiche | Profil |
|---|---|---|---|
| glm-5.3-flash (cloud) | 4/4 · 2,4 min · 14 tours | 4/4 · 2,9 min · 35 tours | solide |
| gemma4 (cloud) | 4/4 · 2,3 min · 37 tours | 4/4 · 2,5 min · 46 tours | solide |
| Qwen3.8-27B 4 bits (local) | **2 réussites / 3** · 26-32 min | **3/3** · 21-29 min (session poursuivie) ; 52 min en sessions neuves | fiable en fiches courtes |
| gpt-oss:120b (cloud) | 0/4 | 4/4 · 100 tours | a besoin de découpage |
| gpt-oss:20b (cloud) | 0/4, paquet cassé | 3/4 (rate la tâche multi-fichiers) | à la limite |

## 1. Choisir la granularité selon le modèle

- **Modèle cloud solide** (glm-5.3-flash, gemma4) : une fiche = une
  fonctionnalité cohérente (jusqu'à ~4 tâches liées, ~5 fichiers, ~300 lignes).
  Découper plus fin n'augmente pas la réussite et coûte des tours.
- **Qwen3.8-27B local** : **une tâche par fiche, dans la même session**.
  C'est à la fois le plus fiable (3/3) et le plus rapide (21-29 min). La
  fiche unique échoue une fois sur trois : le modèle rédige toute
  l'implémentation dans sa réflexion et dépasse `maxTokens` (12 288) avant
  d'avoir émis un seul appel ; pi s'arrête, rien n'est modifié. Deux tâches
  par fiche passent (2/2 après correctif serveur) mais coûtent ~40 min.
- **Modèle faible ou inconnu** (famille gpt-oss, petits modèles) : **une tâche
  par fiche**, une session neuve par fiche. C'est ce qui fait passer gpt-oss de
  0/4 à 4/4 (120b) ou 3/4 (20b).
- **Modèle jamais mesuré** : ne pas deviner, le passer au banc
  (`./run.py --provider ollama --model <id> --plan mono` puis `--plan split4`,
  ~5 min en cloud). Une réussite en `mono` ⇒ profil solide.
- La fenêtre de contexte n'a **jamais** été le facteur limitant (max 45 k
  jetons sur 65 k, une seule compaction, chez un modèle qui tournait en rond).
  On découpe pour le modèle, pas pour le contexte.

## 2. Local (qwen38 serve) : poursuivre la session

Sur un modèle local, préférer des fiches courtes **dans la même session pi**
(`pi -p --continue`, ou une session interactive) à des sessions neuves : le
cache de conversation du serveur sert 90-96 % de l'entrée, une session neuve
repart de zéro. Mesuré : 29 min (poursuivie) contre 52 min (neuves) pour les
mêmes 4 fiches. Couper la session seulement si le contexte approche le seuil
de compaction (fenêtre − `reserveTokens`, soit ~49 k avec le réglage retenu).

## 3. Isoler la tâche la plus risquée

La tâche qui tombe en premier chez un modèle faible : **plusieurs fichiers
touchés + un test existant à mettre à jour** (catalogue + schéma + exécuteur +
test du nombre d'outils). Dans un plan, la mettre dans sa propre fiche et
dire explicitement quel test existant va changer et pourquoi.

## 4. Écrire la fiche

- Un **exemple exact** pour toute sortie formatée (chaîne attendue complète,
  cas limites `0`, `100`, séparateur décimal) : sans lui, le Qwen3.8-27B a
  bouclé 25 min dans sa réflexion à vérifier un arrondi déjà juste.
- Signatures publiques exactes (nom, type, valeur par défaut, place du
  paramètre), et « les appels existants doivent continuer de compiler ».
- Critère de fin **exécutable** : « `swift build` et `swift test` passent,
  lance-les toi-même avant de t'arrêter ». Plusieurs modèles ont conclu sans
  jamais lancer les tests.
- Une règle dans AGENTS.md : « si un test existant contredit le comportement
  demandé, mets-le à jour ; ne supprime ni ne désactive aucun test ».
- Pas de hors-sujet : ce que la fiche ne demande pas va dans `ASK.md`.
- Pour un modèle local qui réfléchit (Qwen3.8) : dans AGENTS.md, « modifie un
  fichier à la fois ; ne rédige pas tout le code dans ta réflexion avant
  d'agir ». Augmenter `maxTokens` ne ferait que repousser la coupure.

## 5. Signaux d'alerte à lire dans la session pi

Si l'on voit ces motifs en cours de route, le modèle est au-delà de sa
capacité pour cette granularité : redécouper ou changer de modèle.

- appel à un outil inexistant (gpt-oss invente `search` : « Tool search not
  found » répété) ;
- `edit` qui échoue (« Could not find the exact text », « Found N
  occurrences ») plusieurs fois de suite ;
- plus de ~40 tours pour une seule tâche ;
- une fin de fiche sans aucun `swift test` ;
- côté local : une réponse coupée à `maxTokens` (`stopReason: length`) = boucle
  de réflexion ou implémentation entière rédigée en réflexion (fiche trop
  grosse pour ce modèle) ; un tour « texte seul » sans modification alors que la fiche
  n'est pas faite = appel d'outil resté dans la réflexion (corrigé dans
  qwen38 serve le 2026-09-28, vérifier que le binaire est à jour).

## 6. Réglages pi à ne pas oublier (issue #1)

`contextWindow 65536`, `maxTokens 12288`, `compat.thinkingFormat "qwen"`,
`thinkingLevelMap` (low/medium/xhigh), compaction **imbriquée** sous
`compaction` (`reserveTokens 16384`, `keepRecentTokens 12000`),
`--thinking low`. Le banc écrit ce réglage dans un dossier pi isolé
(`PI_CODING_AGENT_DIR`) ; pour un projet, vérifier le script qui réécrit
`~/.pi/agent/models.json`.
