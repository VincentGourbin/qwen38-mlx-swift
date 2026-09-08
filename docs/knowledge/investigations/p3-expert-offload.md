# P3 — Déchargement disque des experts Flash-Next (étude d'architecture)

Contexte : checkpoint Vontra 4-bit g32, 512 experts × 48 couches, 10 routés +
1 partagé par token, chaque expert = 3 matrices 640×2560 (gate/up entrée
2560 sortie 640 ; down entrée 640 sortie 2560). Poids des experts ≈ 3,1 Mo
(packé+scales+biases) par expert, ≈ 77,1 Go pour les 48 couches (61,7 Go
packés + 15,4 Go scales/biases — `PLAN.md`, « R2 » 2026-09-07). Par token,
au pire 48 × 10 = 480 experts distincts touchés ≈ 1,5 Go. Décodage actuel :
0,47 s/token en 4-bit résident (77 Go de process), 0,22 s/token en 3-bit
g64 résident (52,4 Go de RSS, `docs/knowledge/log.md` « Q3.3 »), host-bound
(~100 noyaux/couche, GPU 32 %, `PLAN.md` « P1 exécuté »). But de P3 :
retrouver le 4-bit sans les 77 Go résidents, en laissant les experts sur le
SSD interne (~5 Go/s séquentiel, ~100 k IOPS, 88 Go libres) et en ne
transférant vers le GPU que les experts routés par token.

## 1. Options d'architecture

Fait vérifié à la base de toute l'étude — **`MLXArray` a deux chemins de
construction depuis de la mémoire externe, un seul évite la copie** :

| Chemin Swift | Chemin C++ | Comportement |
|---|---|---|
| `MLXArray(_ value: [T], shape)` (utilisé par `Qwen4ExpPLE.swift:277-284` pour envoyer les lignes n-gram lues en mmap) | `mlx_array_new_data` (`array.cpp:237`) → constructeur itérateur `array(It data, Shape, Dtype)` (`array.h:40-43`) → `array::init` (`array.h:588`) : `set_data(allocator::malloc(...))` puis `std::copy(...)` | **Copie toujours**, prouvé par le code (pas d'heuristique) |
| `MLXArray(rawPointer:_:dtype:finalizer:)` (`MLXArray+Init.swift:78`) | `mlx_array_new_data_managed_payload` (`array.cpp:253`) → constructeur `array(void* data, Shape, Dtype, deleter)` (`array.h:65-69`), commentaire du header : *"The constructor will attempt to use the input data without a copy… deleter called after the array is destroyed in the no-copy case **and after the copy otherwise**"* | **Tentative de zéro-copie, non garantie** — la doc Swift ajoute : *"the raw pointer must be compatible with the computational backing, e.g. a Metal stream requires something compatible with an MTLBuffer"*. Un pointeur mmap CPU brut n'est pas nativement un `MTLBuffer` ; sur mémoire unifiée Apple Silicon c'est plausible mais **pas prouvé par la seule lecture du contrat** — à vérifier empiriquement (P3.1). |

Le seul mécanisme de « déchargement disque » déjà en production
(`Qwen4ExpLazyNGramStorage`, `Qwen4ExpPLE.swift:34-266`) tranche déjà cette
question dans les faits : il mmap les shards (`mmap(..., MAP_PRIVATE, ...)`,
ligne 73), memcpy les lignes demandées dans des `[UInt32]`/`[UInt16]` Swift
(`readPackedRowsUncached`, ligne 324), **puis appelle `MLXArray(packed)`**
(ligne 277) — donc **le chemin copiant**, jamais `rawPointer:`. C'est le
précédent direct le plus fiable pour estimer le coût réel.

### Option A — Store disque mmap + gather-copie par token vers `[10, out, in]`

**Mécanisme** : un nouveau `Qwen4ExpExpertDiskStore`, calqué sur
`Qwen4ExpLazyNGramStorage`, mmap un fichier « experts » par couche
(§2). Pour les `k` indices routés d'un token (`k=10` décodage, `k≤40`
vérification MTP), il memcpy les blocs contigus (poids+scales+biases) de
chaque expert depuis la région mmap vers des buffers Swift, construit
`MLXArray([k, out, in])` (+ scales/biases) via le chemin copiant ci-dessus,
puis exécute un matmul quantifié **dense** (`MLX.quantizedMatmul`, indices
locaux 0..k-1) au lieu de `gatherQMM` sur `[512, out, in]`.

**Effet sur `SwitchGLU`/`gatherQMM`** : `QuantizedSwitchLinear.callAsFunction`
appelle `MLX.gatherQuantizedMM(..., rhsIndices: indices, ...)`
(`SwitchLayers.swift:528-540`) — ce noyau ne multiplie déjà que les lignes
gatherées, pas les 512 experts : le FLOP compté est identique à un matmul
dense sur `[10,...]`. **Le gain de bascule vers un tenseur dense n'est donc
pas computationnel, il est mémoire** : `gatherQMM` exige que les 512
experts existent déjà en RAM adressable par MLX (77 Go) ; le dense `[10,...]`
n'exige que 10 experts en mémoire, au prix de la copie/lecture disque. Pour
`M=1` (décodage) ou `M=2..4` (MTP), `SwitchLayers.swift:210` et `:375`
(`let doSort = indices.size >= 64`) montrent que le chemin de tri/lot de
`SwitchGLU` (utile en préfill, `indices.size ≥ 64`) **ne s'active de toute
façon jamais** à ces tailles — gatherQMM et un matmul dense sur `[10,...]`
font un travail équivalent, non trié, ligne par ligne. Verdict attendu : à
compute égal, **la variante disque est neutre-à-légèrement-plus-lente** que
gatherQMM/résident, la différence se jouant entièrement sur le coût
IO+copie ajouté (§4), pas sur le matmul.

**Coût par token** (10 experts, 4-bit g32) : ≈ 31 Mo à lire/copier par
couche (10 × 3,1 Mo), 480 Mo/token cumulés sur 10 % des couches
équivalent-échantillon, 1,5 Go/token au pire sur 48 couches. Appels système :
1 `pread`/mmap-touch par expert si le layout est contigu par expert (§2) →
10/couche, 480/token (décodage) ; le SSD annoncé (~100 k IOPS) absorbe ce
volume sans être le facteur limitant (§4). Noyaux MLX supplémentaires : la
construction `MLXArray([k,out,in])` est un travail hôte pur (pas de kernel
GPU avant `eval`), donc l'ajout net de noyaux Metal par rapport au chemin
résident est nul — le coût est un memcpy CPU + une éventuelle synchronisation
(pas un noyau GPU en plus).

### Option B — `MTLBuffer.makeBuffer(bytesNoCopy:)` sur mmap page-alignée

**Mécanisme** : mmap le fichier expert avec des offsets alignés sur la page
VM du système (**16 384 octets sur Apple Silicon**, pas 4096 — à lire via
`sysconf(_SC_PAGESIZE)`, jamais codé en dur), puis
`device.makeBuffer(bytesNoCopy: pointeur, length:, options: .storageModeShared, deallocator:)`
pour obtenir un `MTLBuffer` qui référence directement les pages mmap sans
copie côté Metal (la mémoire unifiée le permet en théorie). Il faudrait
ensuite faire adopter ce buffer par `MLXArray` — `MLXArray+Metal.swift:28`
montre le sens inverse déjà supporté (`asMTLBuffer(noCopy:)`, un `MLXArray`
existant expose son buffer sans copie), mais **aucune API mlx-swift lue
dans `.xcodebuild/SourcePackages/checkouts/mlx-swift/Source/MLX/` ne prend
un `MTLBuffer` externe en entrée** pour construire un `MLXArray` — le seul
point d'entrée externe reste `MLXArray(rawPointer:...)` (Option A/zéro-copie
non garantie). **Conséquence : Option B, telle que décrite dans la
consigne, n'a pas de point d'entrée mlx-swift vérifié** ; elle se réduirait
en pratique à la même tentative `rawPointer:` que l'Option A avec un soin
d'alignement supplémentaire (utile *si* le zéro-copie s'avère réel, sans
gain sinon). Non retenue comme option distincte tant que P3.1 n'a pas
confirmé qu'un `MTLBuffer` `bytesNoCopy` change quelque chose que
`rawPointer:` n'offre pas déjà.

### Option C — Cache LRU d'experts résidents en `MLXArray`, gather par indices

**Mécanisme** : garder un sous-ensemble d'experts « chauds » par couche déjà
matérialisés en `MLXArray` (donc déjà évalués, résidents), organisés comme
un pool `[cache_size, out, in]` avec une table `expertID → position locale`
tenue par une structure LRU proche de `RowCache` (`Qwen4ExpPLE.swift:118-191`,
capacité 4096 lignes aujourd'hui). Sur un hit, `gatherQMM` s'exécute sur ce
pool réduit avec des indices remappés ; sur un miss, l'expert manquant est
lu depuis le store disque (Option A), copié en `MLXArray`, inséré dans le
pool (évincant le moins récent). **Risque de fond, propre à cette option**
: un `MLXArray` mis en cache est *résident* — c'est exactement le problème
que P3 cherche à éliminer, à une échelle réduite. Si le cache est
dimensionné trop grand ou si le taux de rotation est élevé, on recrée
progressivement le pic mémoire des 77 Go (chaque insertion est un `eval`
qui matérialise et wire indirectement de la mémoire anonyme, contrairement
aux pages de fichier mmap qui restent évincables par le noyau — cf. §5, le
constat H6 sur compresseur vs pages fichier). Coût par token : nul en cas
de hit (gatherQMM déjà en place), sinon identique à l'Option A + coût
d'insertion dans le pool (une concat/scatter supplémentaire).

### Option D — Préchargement asynchrone « couche N+1 pendant N » (corrigée)

La consigne suggère de précharger les experts de la couche N+1 pendant le
calcul de N, par analogie avec `residentAsyncEval`
(`Qwen4ExpStreamingDecoder.swift:56-63,200-208`, qui recouvre le calcul GPU
de la couche N avec la construction hôte du graphe de N+1). **Cette
analogie ne tient pas pour le routage MoE** : les indices routés de la
couche N+1 dépendent de son état d'entrée `gate(x)` où `x` est la sortie
*déjà injectée* de la couche N (`Qwen4ExpDecoderLayer.callAsFunction`,
`Qwen4ExpDecoderLayer.swift:104-114`, puis `hidden = output` réinjecté au
tour suivant de la boucle, `Qwen4ExpStreamingDecoder.swift:210) — la couche
N+1 ne peut pas savoir quels experts elle routera avant que N ait fini son
`forward`. **Le préchargement à un jeton d'avance sur des indices exacts
est donc causalement impossible entre couches successives d'un même
token.** Ce qui reste possible :
- **intra-couche** : `Qwen4ExpSparseMoE.callAsFunction` calcule `indices`
  (`Qwen4ExpSparseMoE.swift:107-109`, `argPartition`) *avant* d'appeler
  `switchMLP(flatX, flatIndices)` (ligne 119-120) — la fenêtre entre ces
  deux lignes est le seul instant où lancer la lecture disque sans bloquer
  le calcul en cours, mais elle est courte (le gate est un matmul léger) ;
- **inter-token** : précharger de façon spéculative les experts routés au
  *jeton précédent* pour la même couche, sous l'hypothèse « routage stable
  par sujet » (§3, à valider), avec vérification/repli si le nouveau
  routage diffère.

Ni l'une ni l'autre ne constitue un levier de la même ampleur que l'asyncEval
GPU existant ; le recouvrement réel de P3 vient surtout du cache (Option C)
et de la marge entre le temps de calcul par couche (5,5-9,8 ms mesurés,
§4) et le temps de lecture (≈6,1 ms à 0 % de succès), pas d'un pipeline
explicite de préchargement inter-couche.

**Décision de conception** : Option A (store disque façon
`Qwen4ExpLazyNGramStorage`) comme brique de base, avec une variante Option C
(petit cache LRU, borné et mesuré) en complément optionnel une fois que
l'instrumentation §3 aura chiffré le taux de succès — jamais un cache non
borné.

## 2. Layout disque

Un fichier par couche, `experts_layer_{L}.bin` (48 fichiers), écrit une
seule fois par un script de conversion analogue à
`Scripts/qwen4-exp-requantize-experts.py` (Q3.1, `docs/knowledge/log.md`
« Q3.1 »), qui a déjà validé le motif « ouvrir tous les shards source en
paresseux, évaluer et libérer tenseur par tenseur » nécessaire ici aussi
(9 des 147 familles d'experts ont leurs `.weight`/`.scales`/`.biases`
répartis sur deux shards adjacents dans le checkpoint Vontra — même piège
que Q3.1).

| Champ | Valeur |
|---|---|
| Page VM Apple Silicon | 16 384 octets (`sysconf(_SC_PAGESIZE)`, pas 4096) |
| Bloc par expert | poids `[640,320]` U32 (819 200 o) + scales `[640,80]` BF16 (102 400 o) + biases idem pour gate/up ; poids `[2560,80]` U32 + scales `[2560,20]` BF16 ×2 pour down ; total ≈ 3,1 Mo, **arrondi à la page supérieure** (3 145 728 o = 192 pages de 16 Ko) pour que chaque expert commence à une frontière de page — condition nécessaire si l'Option B (mmap+`MTLBuffer bytesNoCopy`) est un jour retenue, gratuite sinon (+ < 2 % de gonflement) |
| Taille par fichier de couche | 512 × 3 Mo (arrondi page) ≈ 1,64 Go |
| Taille totale (48 couches) | ≈ 78,7 Go (77,1 Go utiles + arrondi page) — **tient dans les 88 Go libres du SSD interne avec ~9 Go de marge**, ne tiendrait pas à côté du reste du checkpoint (113 Go) |
| En-tête par fichier | table fixe de 512 triplets `(offset packé, offset scales, offset biases)` en tête de fichier, elle-même page-alignée |

**`model.safetensors.index.json`** : les clés `*.mlp.switch_mlp.*` restent
listées (pour la compatibilité d'un chargement complet, ex. `flash-slice-probe`
sans stockage externe) mais un nouveau champ `config.json` —
`"expertsStorage": {"path": "experts/", "pageSize": 16384}` — signale à
`Qwen4ExpCheckpointLayerLoader` de **ne pas** matérialiser ces clés depuis
les shards Lexar quand il est présent, symétrique de `isNGramTableKey`
(`Qwen4ExpCheckpointLayerLoader.swift:196-200`) qui fait déjà exactement ce
tri pour la table n-gram — même endroit du code, même patron, nouvelle
prédicate `isExpertTableKey`.

**Cohabitation** : le reste du checkpoint (globaux, attention, hyper-
connections, gate MoE, shared expert, table n-gram) reste sur le Lexar
inchangé — `Qwen4ExpLazyNGramStorage` continue de mmaper le Lexar pour le
n-gram (32 Go, non concerné par P3). Seuls les ≈77 Go d'experts migrent
vers le SSD interne ; le Lexar héberge alors ≈36 Go (113 − 77), largement
sous ses capacités actuelles.

## 3. Cache d'experts chauds

**Mesure directe impossible avec l'existant** : vérifié, `results/h6/*.json`
ne contient que les réponses complètes du serveur (`content`,
`reasoning_content`, métadonnées HTTP) — aucun indice de routage n'est
journalisé nulle part aujourd'hui. `Qwen4ExpSparseMoE` calcule `indices`
(`Qwen4ExpSparseMoE.swift:109`) et l'expose déjà via
`lastParityCapture["moe_indices"]` (ligne 127) mais uniquement quand
`captureParity` est actif (coûteux, pensé pour les probes de parité en
développement, retient d'autres tenseurs volumineux) — inadapté à un run de
qualification de 600 tokens.

**Instrumentation minimale proposée** (à écrire, hors périmètre de ce
document) : un booléen indépendant `routingLog` sur `Qwen4ExpSparseMoE`
qui, activé, fait un seul `indices.asArray(Int32.self)` (host sync,
coût comparable à un `beginPhase`/`endPhase` de profiler, ~qq ms) et
l'ajoute à un journal `[couche: [tokenIndex: [Int32]]]` ; remonté par un
callback parallèle à `onLayerVisited`
(`Qwen4ExpStreamingDecoder.swift:96-103`, ex. `onExpertsRouted: (Int, [Int32]) -> Void`).
Coût nul quand désactivé (comme `profileLayers`, `Qwen4ExpStreamingDecoder.swift:46-55`),
jamais activé en production.

**Protocole de mesure** (à exécuter dans un run futur, hors ce document) :
rejouer les prompts H6 (`Scripts/h6-qualification.sh`) ou un jeu plus long
et thématiquement varié avec `routingLog` actif, puis analyser hors ligne
(pas besoin de rejouer le moteur) : (a) nombre d'experts distincts routés
par couche sur une fenêtre glissante de N tokens ; (b) simulation d'un
cache LRU/LFU de taille K par couche rejouée sur la trace journalisée,
courbe taux-de-succès vs K ; (c) plusieurs prompts thématiquement disjoints
enchaînés, pour mesurer la perturbation du cache au changement de sujet et
le temps de restabilisation. **Hypothèse « routage stable par sujet »**
(non vérifiée) : dans un modèle MoE entraîné, le gate correlé au contenu
sémantique tend à router vers un sous-ensemble d'experts par domaine
lexical/sémantique — plausible mais non mesuré sur ce checkpoint ; c'est
précisément ce que (a)/(b)/(c) doivent trancher.

**Dimensionnement proposé (hypothèse de départ, à ajuster)** : un cache de
15-20 % des experts par couche (~75-100/512) ≈ 12-15 Go résidents au total,
LRU en premier choix (motif déjà éprouvé, `RowCache`,
`Qwen4ExpPLE.swift:118-191`, plutôt qu'une politique par fréquence plus
coûteuse à tenir à jour) — mais voir §5 : **un cache d'`MLXArray` matérialisés
recrée un pic mémoire résident**, contrairement à un simple mmap laissé à la
gestion du cache de pages du noyau (déjà le choix de
`Qwen4ExpLazyNGramStorage`, sans réplique applicative de second niveau
au-delà de `RowCache` qui ne cache que des octets bruts, pas des
`MLXArray`). Recommandation : mesurer d'abord le taux de succès du seul
cache de pages du noyau (mmap sans `F_NOCACHE`, aucun cache applicatif)
avant d'ajouter un cache d'`MLXArray` — le gain net d'un cache applicatif
n'est justifié que si le cache noyau, sous la contrainte mémoire mesurée en
H6 (≤ 8 Go d'anonyme dispo), s'avère insuffisant.

## 4. Budget de latence

Hypothèse de disque : SSD interne ≈ 5 Go/s séquentiel, ~100 k IOPS (donné) ;
Lexar USB ExFAT ≈ 0,7 Go/s (donné, mesuré ailleurs dans `PLAN.md`). Volume
par couche pour 10 experts routés : 10 × 3,1 Mo ≈ 31 Mo.

| Taux de succès cache | Octets à lire/couche | SSD interne (temps) | Lexar (temps, référence — non retenu §2) |
|---|---|---|---|
| 0 % (à froid) | 31 Mo | 6,1 ms | 44,3 ms |
| 50 % | 15,5 Mo | 3,0 ms | 22,1 ms |
| 80 % | 6,2 Mo | 1,2 ms | 8,9 ms |
| 95 % | 1,55 Mo | 0,3 ms | 2,2 ms |

Budget de calcul par couche disponible pour recouvrir cette lecture :
**5,5-5,8 ms** (GDN/QSA, Release, calcul pur, `docs/knowledge/log.md` « P0
rejoué en Release ») en isolation synthétique, ou **≈9,8 ms/couche en
moyenne réelle** (0,47 s / 48 couches, résident 4-bit + `asyncEval`,
`PLAN.md` « P1 exécuté ») en pratique côté production, contre **≈4,6 ms/couche**
pour la référence 3-bit (0,22 s / 48).

- **Si la lecture disque est parfaitement recouverte** par le calcul (un
  thread dédié au memcpy pendant que le thread hôte MLX calcule la couche
  courante, sans contention) : le temps de couche devient `max(calcul,
  lecture)`. Contre le 4-bit résident (9,8 ms), même 0 % de succès (6,1 ms)
  reste sous le budget → **neutre par construction**, à condition que le
  memcpy ne contende pas avec le thread de calcul hôte déjà saturé à ~100 %
  d'un cœur (constat P0 : le CPU est déjà occupé par le bookkeeping MLX,
  pas de marge évidente sur ce même thread — **hypothèse à valider par
  P3.1**, pas acquise). Contre le 3-bit (4,6 ms), il faut un taux de succès
  ≥ ~25 % pour rester sous le budget (`(1−h)×6,1 ≤ 4,6 ⇒ h ≥ 24,6 %`).
- **Si la lecture est sérielle** (pas de recouvrement, hypothèse
  pessimiste) : temps ajouté = `48 × (1−h) × 6,1 ms`. Table :

| Taux de succès | Temps ajouté/token | Nouveau s/token (base 4-bit 0,47 s) | Variation |
|---|---|---|---|
| 0 % | +293 ms | 0,76 s | +62 % |
| 50 % | +146 ms | 0,62 s | +32 % |
| 80 % | +59 ms | 0,53 s | +12 % |
| 95 % | +15 ms | 0,49 s | +3 % (quasi neutre) |
| 99 % | +3 ms | 0,47 s | neutre |

**Conclusion chiffrée** : le déchargement disque est neutre à partir d'un
taux de succès ≈ 95 % *sans* recouvrement, ou dès 0 % de succès *avec*
recouvrement parfait contre la référence 4-bit (mais ≥ 25 % contre la
référence 3-bit). L'écart entre ces deux bornes est entièrement determiné
par la faisabilité du recouvrement CPU (P3.1 doit trancher), pas par le
débit du SSD lui-même qui n'est jamais le facteur limitant à cette échelle.

## 5. Interactions

- **Préfill (29-1000 tokens)** : avec 10 tirages/token sur 512 experts, le
  nombre attendu de tirages par expert après *n* tokens est `10n/512` — dès
  n≈150 (1 500 tirages), chaque expert reçoit en moyenne ~2,9 tirages :
  la quasi-totalité des 512 experts d'une couche sont statistiquement
  touchés bien avant la fin d'un préfill de 200+ tokens. **Le chemin
  « décharger seulement les routés » n'a aucun intérêt en préfill** ; il
  faut un chemin batch qui charge la totalité des 512 experts d'une couche
  depuis le store disque, exécute le forward, puis libère — c'est
  exactement le comportement déjà implémenté par le mode `.streamed`
  existant (`Qwen4ExpLayerLoadingMode.streamed`, `Qwen4ExpStreamingDecoder.swift:10-13`,
  `Qwen4ExpCheckpointLayerLoader.load(materialize: true)`), à rebrancher
  simplement sur le nouveau store au lieu des shards Lexar pour les clés
  `switch_mlp`.
- **MTP (vérification de 2-4 tokens)** : un round de vérification est un
  forward multi-position (`M=2..4`), donc jusqu'à `M×10=40` indices par
  couche — l'union de ces indices (dédupliquée avant de toucher le disque)
  doit être calculée une fois par couche, pas `M` fois. Pour `M≤4`,
  `SwitchLayers.swift:210,375` (`doSort = indices.size >= 64`) confirme que
  le chemin tri/lot de `SwitchGLU` reste inactif à cette taille — aucune
  optimisation de `SwitchGLU` n'est perdue en passant par un tenseur
  disque-gathered de taille ≤40, mais la déduplication doit être faite en
  amont, au niveau du store, pas dans `SwitchGLU`.
- **`asyncEval` par couche / `residentEvaluationInterval`** : voir Option D
  §1 — le recouvrement possible n'est pas inter-couche (dépendance causale
  du routage) mais (i) intra-couche, entre le calcul de `indices`
  (`Qwen4ExpSparseMoE.swift:109`) et l'appel à `switchMLP` (ligne 120), et
  (ii) inter-token, sous l'hypothèse de stabilité du routage (§3). Le flag
  `residentAsyncEval` (`Qwen4ExpStreamingDecoder.swift:63`) gouverne le
  recouvrement GPU existant et resterait indépendant ; un futur
  recouvrement IO nécessiterait son propre point de synchronisation
  (`await` juste avant le matmul d'experts), pas une réutilisation directe
  de ce flag.
- **Loader `F_NOCACHE` en cours (P2-mem-a)** : ce chantier, actuellement en
  cours sur `Qwen4ExpCheckpointLayerLoader.swift`/`Qwen4ExpUncachedTensorReader.swift`
  (non touchés par ce document), vise à **éviter** le double tampon
  RAM+cache-fichier pour les tenseurs *non-experts* qui restent résidents
  toute la session. **P3 devrait faire l'inverse pour les experts** : ne
  pas ouvrir le store d'experts avec `F_NOCACHE`, pour laisser le cache de
  pages du noyau agir comme un second niveau de cache gratuit entre tokens
  — ces pages, contrairement à de la mémoire anonyme `MLXArray`, sont
  évincées en douceur sous pression mémoire plutôt que compressées
  (constat H6, `PLAN.md` « H6 : deux tentatives », le cache fichier du
  checkpoint est évacué avant que le compresseur ne s'active). C'est le
  même choix que `Qwen4ExpLazyNGramStorage`, qui n'utilise jamais
  `F_NOCACHE`.
- **Mémoire, pic visé ≤ 30 Go** : composantes du process avec P3 « pur »
  (aucun cache applicatif) — poids non-experts résidents (~4 Go, mesure
  R2), table n-gram mmap non comptée en RSS propre (pages de fichier,
  32 Go virtuels), plus le cache de pages du noyau pour le store d'experts
  (taille variable, évincable, pas garanti dans le budget process mais
  compte dans la RAM système globale). Avec un cache applicatif borné en
  MLXArray (Option C) de 12-15 Go (§3), le pic process resterait ≈ 4 + 15
  = 19 Go, sous les 30 Go visés — mais chaque Go ajouté au cache applicatif
  reproduit à cette échelle le problème diagnosé en H6 (mémoire anonyme non
  évincable en douceur) ; à dimensionner strictement après mesure §3, pas
  par confort de code.

## 6. Plan d'exécution recommandé

Option retenue : **A (store disque mmap, calqué sur `Qwen4ExpLazyNGramStorage`)
sans cache applicatif dans une première version**, cache LRU borné (Option C)
ajouté seulement si la mesure §3 montre que le cache de pages du noyau est
insuffisant sous la contrainte mémoire H6.

| # | Tâche | Fichier cible | Critère vérifiable | Ordre |
|---|---|---|---|---|
| P3.1 | Prototype sur `flash-layer-bench` : nouveau mode `--disk-experts` construisant un fichier expert synthétique page-aligné (512 experts, dims réelles), mmap, et à chaque pas simulé : (a) `gatherQMM` sur `[512,...]` résident (référence, chemin actuel du bench) ; (b) mmap+memcpy 10 indices → `MLXArray([10,...])` via le chemin copiant `MLXArray(_:)` → matmul dense ; (c) idem via `MLXArray(rawPointer:...)` pour trancher empiriquement si la copie est évitée sur ce matériel/OS ; option d'exécuter (b)/(c) sur une `DispatchQueue` séparée du thread MLX pour mesurer la contention CPU (§4) | `Sources/Qwen38Core/FlashNext/Qwen4ExpLayerBench.swift`, `Sources/Qwen38CLI/Qwen38CLI.swift` (option CLI) | Tableau ms/pas médiane + GPU/CPU % pour (a)/(b)/(c), verdict écrit sur la contention et sur le zéro-copie réel | 1 |
| P3.2 | `Qwen4ExpExpertDiskStore` (nouveau fichier, calqué sur `Qwen4ExpLazyNGramStorage`, `Qwen4ExpPLE.swift:34-266`) : mmap par couche, `lookup(layerIndex:, expertIndices:) -> (packed, scales, biases)`, sans `F_NOCACHE` | nouveau `Sources/Qwen38Core/FlashNext/Qwen4ExpExpertDiskStore.swift` | Tests unitaires sur fixture synthétique (pas de checkpoint réel), parité octet-à-octet avec un tenseur de référence | 2 |
| P3.3 | Script de conversion `Scripts/qwen4-exp-export-experts-to-ssd.py` (calqué sur `Scripts/qwen4-exp-requantize-experts.py`, Q3.1) : écrit les fichiers page-alignés depuis le checkpoint Lexar existant, met à jour `config.json`/index | `Scripts/qwen4-exp-export-experts-to-ssd.py` | Dossier de sortie complet, taille ≈78,7 Go, vérification octet-à-octet sur 3 experts (méthode Q3.1) | 3 |
| P3.4 | Branchement décodage : `isExpertTableKey` dans `Qwen4ExpCheckpointLayerLoader` (coordination requise avec P2-mem-a en cours — ne pas éditer en parallèle), `Qwen4ExpSparseMoE` route vers le store disque en décodage (`M<64`), garde le chemin `SwitchGLU`/résident pour le préfill batch (§5) | `Qwen4ExpCheckpointLayerLoader.swift` (après P2-mem-a), `Qwen4ExpSparseMoE.swift` | Parité bit-exacte décodage disque vs résident sur 2-3 couches (méthode `flash-slice-probe --run-forward`, Q3.2) | 4 |
| P3.5 | Instrumentation `routingLog` + mesure du taux de succès (§3), décision de cache | `Qwen4ExpSparseMoE.swift`, `Qwen4ExpStreamingDecoder.swift` | Courbe succès vs taille de cache, décision documentée | 5 |
| P3.6 | Intégration complète derrière un flag, run P1-style (latence + mémoire) sur le vrai checkpoint 4-bit | `Qwen38FlashNextEngine`, probes CLI | s/token et pic RSS mesurés, comparés à 0,47 s/77 Go (4-bit résident) et 0,22 s/52 Go (3-bit résident) | 6 |

**Risques et invalidants** : (i) si `MLXArray(rawPointer:...)` copie
systématiquement sur ce backend Metal (P3.1 le mesurera), le gain restant
vient uniquement de l'empreinte mémoire, pas d'un noyau économisé — n'invalide
pas l'option mais réduit l'attrait vs un simple cache LRU d'`MLXArray` déjà
alloués ; (ii) si le thread hôte MLX ne peut pas recouvrir la lecture disque
(contention avec le bookkeeping déjà saturé à 100 % d'un cœur, constat P0),
la latence retombe sur le régime sériel du tableau §4 — à ≥ 95 % de succès
cache cela reste acceptable, sinon P3 perd son intérêt face au 3-bit déjà
qualifié (H6 PASS 8/8, `PLAN.md` « H6 PASS sur le 3-bit ») ; (iii) si
l'hypothèse « routage stable par sujet » est fausse (mesure §3), le taux de
succès reste bas et seul le recouvrement CPU (risque ii) peut encore sauver
l'option ; (iv) toute dérive de la taille du cache applicatif (Option C)
vers plusieurs dizaines de Go reproduirait le problème mémoire que P3 est
censé résoudre — à surveiller à chaque itération de P3.5/P3.6, pas
seulement au dimensionnement initial.
