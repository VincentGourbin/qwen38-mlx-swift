# Évaluer le cerveau sur les usages de Fluxforge Studio

> Pour l'équipe Fluxforge Studio : un point de contrôle externe. Vous écrivez
> les situations et les critères qui comptent pour l'app ; le même banc que
> celui du framework les joue contre le modèle local et contre des modèles
> cloud, et un juge indépendant tranche.

## Le principe

- **Un scénario** = une situation jouée par un utilisateur simulé (un modèle
  cloud) et une liste de critères vérifiés par un juge (un autre modèle cloud),
  sur la conversation **et** sur la trace des appels d'outils.
- **Un agent unique** (`bench/langwatch/agent.py`) répond : même boucle, mêmes
  outils, même invite système pour toutes les cibles ; seule la case « modèle »
  change. La cible `local` est le serveur `qwen38 serve`, qui passe par le même
  moteur que `Qwen38Brain` (conversation réutilisée, outils, images).
- **Des outils simulés** propres à l'app : l'agent les déclare au modèle et
  renvoie des réponses factices réalistes. On évalue les **décisions** du
  modèle (quel outil, quels arguments, quelle réponse), pas la génération
  d'image elle-même.
- **Le juge et le simulateur** sont des modèles d'Ollama Cloud
  (`deepseek-v4.1-flash`), jamais le modèle testé.

## Ce que vous fournissez

Deux fichiers JSON, à déposer dans `bench/langwatch/` :

1. **Un jeu d'outils**, `toolsets/<nom>.json` : l'invite système de l'assistant
   de l'app, les outils (format OpenAI `tools`) et une réponse simulée par
   outil. `{argument}` est remplacé par la valeur envoyée par le modèle, un
   argument facultatif absent s'affiche « défaut ».
2. **Une suite de scénarios**, `suites/<nom>.json` :
   `{"suite": "<nom>", "scenarios": [{"name", "labels", "situation", "criteria"}]}`.

Un exemple complet est fourni : `toolsets/fluxforge-exemple.json` (lister les
ressources, générer une image, générer une vidéo, détourer) et
`suites/fluxforge-exemple.json` (visuel produit à partir d'une idée floue,
animer une image existante, détourer puis réutiliser).

### Écrire de bons critères

- **Vérifiables sur la trace** : « appelle `generate_image` avec une largeur
  égale à la hauteur », plutôt que « fait une belle image ».
- **Un fait par critère**, formulé comme une affirmation vraie ou fausse.
- **Au moins un critère « ne pas »** par scénario : ne pas inventer une
  ressource, ne pas générer avant d'avoir compris, ne pas écraser sans
  demander.
- La situation dit ce que l'utilisateur simulé sait et répond s'il est
  questionné ; sinon il improvise.

## Lancer une évaluation

Prérequis : le serveur tourne sur le modèle à évaluer, l'agent est connecté
(voir `docs/langwatch-bench/plan.md`, fiche L-0), la clé LangWatch est dans
`bench/langwatch/.env`.

```bash
cd bench/langwatch && set -a && . ./.env && set +a
# 1. créer ou mettre à jour la suite sur LangWatch
SUITE_FILE=suites/fluxforge-exemple.json .venv/bin/python scenarios.py
# 2. la jouer : 3 passes, modèle local contre deux modèles cloud
cd ../.. && SUITE_FILE=bench/langwatch/suites/fluxforge-exemple.json TOOLSET=fluxforge-exemple \
  LABEL=fluxforge bench/langwatch/run.sh local ollama/glm-5.3-flash ollama/gpt-oss:120b
# 3. le bilan par cible et par scénario
bench/langwatch/.venv/bin/python bench/langwatch/report.py <id du plan> --markdown
```

L'identifiant du plan s'affiche dans LangWatch (Agent Testing → Run plans) ;
les conversations, traces et raisons du juge y sont consultables run par run.

## Lire les résultats

- **Réussite par cible** : la part de runs où le juge valide tous les critères.
  Comparer le modèle local aux modèles cloud sur la même suite ; un scénario
  raté par toutes les cibles désigne souvent un critère mal posé, pas un
  modèle.
- **Grille scénario × cible** : où le modèle local décroche.
- **Durées** : la cible locale est plus lente (préfill sur un Mac) ; c'est une
  information, pas un critère.

## Limites

- Les outils sont simulés : un bon choix d'outil avec de bons arguments est
  mesuré, pas la qualité de l'image produite.
- Le juge est un modèle : ses verdicts sont cohérents mais pas infaillibles ;
  lire les raisons sur les runs surprenants.
- Trois passes par scénario lissent la part d'aléa (température 0,7).
