# Plan d'implémentation — servir Bonsai 2 27B dans `qwen38 serve`

> Plan d'exécution autonome pour un agent. Rédigé le 2026-09-18 après vérification
> du pack (en-tête du safetensors, `config.json`, `hadamard.json`, `runtime/*.py`)
> et de notre code. Le chantier est référencé **P14** dans `PLAN.md`.
> Ce fichier vit sous `docs/bonsai2/` parce qu'un `plan.md` à la racine entrerait
> en collision avec `PLAN.md` sur APFS insensible à la casse.

## 0. Mode d'exécution

- **Une fiche à la fois, dans l'ordre.** Une fiche n'est terminée que lorsque sa
  *porte de sortie* a été observée dans la sortie d'une commande pendant la
  session. Recopie la ligne observée dans le journal (§9).
- **Aucune décision d'architecture n'est à prendre.** Chaque fiche nomme le
  fichier, la fonction, la ligne d'ancrage et le code attendu. En cas
  d'ambiguïté réelle : STOP, écris la question dans le journal (§9, gabarit
  ASK), termine ta réponse par `FICHE B-x BLOQUÉE`.
- **Rien n'est accepté sur la qualité apparente d'une sortie.** Un texte
  plausible ne prouve rien (voir `docs/parity-method.md`). Seule la fiche B-3
  (parité) autorise à continuer.
- **Interdits** : modifier `Vendor/mlx-swift-lm` au-delà des deux lignes
  prévues en B-1 ; changer une version épinglée dans `Package.swift` ; toucher
  au chemin Flash-Next (`Sources/Qwen38Core/FlashNext/`) ; `git push` ;
  supprimer ou réécrire un fichier sous `/Volumes/Lexar/models` ; lire un
  fichier de plus de 300 lignes d'un seul appel (utilise `sed -n a,bp`).
- **Compilation** : `swift build --product qwen38` (rapide, Debug) pour
  vérifier qu'un fichier compile. **Exécution** : `Scripts/build.sh` (Debug)
  ou `Scripts/build-release.sh` ; binaires dans
  `.xcodebuild/Build/Products/{Debug,Release}/qwen38`. Pour les mesures de
  débit (B-6) toujours Release.
- **Zéro avertissement** dans `Sources/` : le projet est à zéro, il doit le
  rester (`swift build 2>&1 | grep "warning:.*Sources/"` doit être vide).
- **Langue** : documents et commentaires de plan en français ; code, noms et
  commentaires de code en anglais, sauf dans `Qwen38Server.swift` et
  `Qwen38CLI.swift` où les commentaires existants sont en français — respecte
  le style du fichier touché.
- **Commit** : un commit par fiche validée, message en français, première
  ligne `B-x : <titre>`. Pas de commit d'une fiche bloquée.

## 1. Ce que le pack est (faits vérifiés, ne pas re-dériver)

Pack : `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (Hugging Face), 8,6 Go,
`model.safetensors` unique, licence Apache-2.0.

| Fait | Valeur | Conséquence |
|---|---|---|
| `config.json` → `model_type` | `prism_hadamard_qwen35` ; `base_model_type: qwen3_5` ; `schema_version: 2` | inconnu de notre `Qwen38ModelFamily` et du registre VLM |
| `text_config` | Qwen3.8-27B : 64 couches, `layer_types` 3 × `linear_attention` puis 1 × `full_attention`, `hidden_size` 5120, `intermediate_size` 17408, 24 têtes q / 4 kv, `head_dim` 256, `linear_num_value_heads` 48, `linear_num_key_heads` 16, `vocab_size` 248320, `tie_word_embeddings: false` | **exactement notre famille `qwen35`** (`Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift`) |
| `vision_config` | tour Qwen3.8-27B, 27 blocs, F16, **non quantifiée, non transformée** | passe telle quelle |
| Tenseurs | 2 390 ; racines `language_model.*` (2 057) et `vision_tower.*` (333) ; métadonnées `format: mlx` | mêmes noms que les nôtres ; `Qwen35.sanitize` renvoie les poids inchangés quand `format == mlx` |
| Modules empaquetés | **402** : `weight` U32 `[out, in/16]`, `scales` F16 `[out, in/128]`, `biases` F16 `[out, in/128]` (== `-scales`), **`signs` F32 `[in]`** | `quantization: {bits 2, group_size 128, mode affine}` → le chargeur vendorisé crée déjà `QuantizedLinear`/`QuantizedEmbedding` 2 bits partout où `.scales` existe |
| Lesquels | `embed_tokens` (inverse), `lm_head`, `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `mlp.{gate,up,down}_proj` — la liste exacte est `config.json` → `modules[]` (`path` relatif à `language_model`, `block`, `embedding`, `dtype`) | liste à parcourir en B-2 |
| Restent flottants | `linear_attn.{in_proj_a,in_proj_b}` F32 `[48, 5120]`, `conv1d.weight` F32 `[10240, 4, 1]`, `A_log`, `dt_bias`, normes, toute la tour vision | la fusion GDN 4-en-1 est **inéligible** (voir B-1) |
| `hadamard.json` | `block_size 1024`, `transform normalized-sylvester-walsh-hadamard`, `axis input-last-dimension`, `sign_mode explicit`, `sign_widths [5120, 6144, 17408]`, `sign_values` (28 672 valeurs ±1, concaténées dans l'ordre des largeurs), `weight_names` (401), `inverse_weight_names` (`language_model.model.embed_tokens.weight`), `gdn_v_grouped: true` | les `.signs` du safetensors sont les mêmes vecteurs, copiés par module |
| Transformée (runtime Python de référence, `runtime/runtime.py`) | projection : `x = x.astype(float32); x = x * signs; x = hadamard_transform(x.reshape(-1, 1024), scale = 1/sqrt(1024)).reshape(shape); x = x.astype(dtype)` puis `quantized_matmul(x, weight, scales, biases, transpose=True, group_size=128, bits=2)`. Embedding : `out = dequantize(weight[idx], scales[idx], biases[idx], 128, 2)` puis `hadamard_transform(out.reshape(-1,1024), 1/sqrt(1024))` puis `out * signs` (l'inverse applique les signes **après**) | B-2 reproduit ceci à l'identique, float32 compris |
| mlx-swift 0.31.6 | `hadamardTransform(_:scale:stream:)` dans `Source/MLX/Ops.swift` ; `quantizedMM` accepte `bits: 2` ; `QuantizedLinear` et `QuantizedEmbedding` sont `open class` dans `Source/MLXNN/Quantized.swift` | sous-classer, pas réécrire |
| Gabarit | `chat_template.jinja` séparé (swift-transformers le lit via `AutoTokenizer.from(modelFolder:)`), identique en structure à Flash-Next : `enable_thinking`, `reasoning_effort ∈ {low, medium, xhigh}` (sinon `raise_exception`), outils en XML `<tool_call>\n<function=…>\n<parameter=…>` | `Qwen38ToolCallParser` (Sources/Qwen38Core/Qwen38ToolCalling.swift) est réutilisable tel quel |
| Fin de séquence | `generation_config.json` → `eos_token_id 248046` (`<|im_end|>`) | lu par le chargeur vendorisé |
| Réflexion | le README du pack : `low` n'est pas honoré, comportement proche de `xhigh` | à mesurer en B-6, pas à corriger |

## 2. Ce qui coince dans notre pile (à corriger, dans cet ordre)

1. `Sources/Qwen38Core/Qwen38ModelValidation.swift` : `family` = `Qwen38ModelFamily(rawValue: modelType)` → `nil` pour ce type → `unsupportedModelType`.
2. `Vendor/mlx-swift-lm/Libraries/MLXVLM/VLMModelFactory.swift` ligne 89 : `VLMTypeRegistry.shared` ne connaît pas `prism_hadamard_qwen35`.
3. `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift` `loadWeights` (≈ ligne 368) : `model.update(parameters:verify: [.all])` refusera les 402 clés `.signs`.
4. Fusion GDN 4-en-1 : `Qwen35.swift` ligne ≈ 1068 `prepare()` appelle `prepareFusedInputProjection()` pendant `materializeModelForInference`, **avant** notre remplacement de modules. Avec `in_proj_a`/`in_proj_b` en `Linear` flottant et `in_proj_qkv`/`in_proj_z` en `QuantizedLinear`, `fuseQuantizedLinearProjections` renvoie `nil` → état `.ineligible`, pas de fusion, pas d'erreur (vérifié dans `FusedQuantizedLinear.swift` lignes 81-102). On force quand même `MLX_QWEN_FOUR_GDN=0` par sécurité (B-1) ; la propriété `hasFusedInputProjection` est interne à MLXVLM, donc la vérification passe par la variable d'environnement et par la parité (B-2, B-3).
5. `Sources/Qwen38Server/Qwen38Server.swift` lignes ≈ 651 et ≈ 797 : `tools` refusé en 400 si `!runtime.isFlashNextLoaded`.
6. Chemin 27B (`Qwen38Runtime.swift` ≈ 1596-1775, `ChatSession`) : le cache KV n'est réutilisé qu'en continuation stricte ; pi renvoie tout l'historique à chaque tour.

## 3. Fiches

### B-0 — Préparation (pas de code)

1. `export QWEN38_MODELS_DIR=/Volumes/Lexar/models` puis
   `.xcodebuild/Build/Products/Debug/qwen38 download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`
   (construire d'abord avec `Scripts/build.sh` si le binaire manque). Si la
   commande `download` ne prend pas ce dépôt, utiliser
   `hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir "$QWEN38_MODELS_DIR/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"`.
2. `export BONSAI="$QWEN38_MODELS_DIR/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"` ;
   vérifier `ls -la "$BONSAI"` : `model.safetensors` 8 595 477 990 octets,
   `hadamard.json`, `chat_template.jinja`, `runtime/`.
3. Environnement Python **séparé** pour la référence (les versions diffèrent
   de `venv617`) :
   `python3 -m venv venv-bonsai2 && venv-bonsai2/bin/pip install -r "$BONSAI/runtime/requirements.txt"`.
   Vérifier : `venv-bonsai2/bin/python -c "import mlx.core as mx, mlx_vlm; print(mx.__version__, mlx_vlm.__version__)"` → `0.32.0 0.6.3`.
4. Vérifier que la référence tourne :
   ```bash
   cd "$BONSAI" && "$HOME/Developpements/qwen38-mlx-swift/venv-bonsai2/bin/python" - <<'EOF'
   import sys; sys.path.insert(0, "runtime")
   from vision_artifact import load_vl_model, chat_config
   from mlx_vlm import generate
   from mlx_vlm.prompt_utils import apply_chat_template
   model, processor, config = load_vl_model(".")
   prompt = apply_chat_template(processor, chat_config(config), "Dis bonjour en un mot.", num_images=0)
   print(generate(model, processor, prompt, max_tokens=32, temperature=0.0))
   EOF
   ```
   (le point important est `load_vl_model`, pas `artifact.load_model` qui
   refuse `schema_version: 2`).

**Porte de sortie** : la commande 4 imprime une réponse en français cohérente.
Note la sortie exacte dans le journal : elle sert de premier repère en B-2.

### B-1 — Charger le pack sans Hadamard (sortie fausse attendue)

**Fichiers** : `Sources/Qwen38Core/Qwen38ModelValidation.swift`,
`Sources/Qwen38Core/Qwen38Runtime.swift`, `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift`,
nouveau `Sources/Qwen38Core/Bonsai2/Qwen38Bonsai2Registration.swift`.

1. **Validation.** Dans `Qwen38ModelInfo`, remplacer la propriété calculée
   `family` par :
   ```swift
   /// `prism_hadamard_qwen35` (Bonsai 2, Prism ML) is Qwen3.8-27B with a
   /// blockwise Hadamard rotation folded into 2-bit weights — same module
   /// topology, so it is served by the `qwen35` family plus the Hadamard
   /// modules installed by `Qwen38Bonsai2Loader` (docs/bonsai2/plan.md).
   public var family: Qwen38ModelFamily? {
       if modelType == Qwen38Bonsai2.modelType { return .qwen35 }
       return Qwen38ModelFamily(rawValue: modelType)
   }
   public var isBonsai2: Bool { modelType == Qwen38Bonsai2.modelType }
   ```
2. **Registre VLM.** Nouveau fichier `Qwen38Bonsai2Registration.swift` sur le
   modèle de `Vendor/mlx-swift-lm/Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift`
   (même mécanisme `await VLMTypeRegistry.shared.registerModelType(_:creator:)`,
   le registre est un `actor`) :
   ```swift
   import MLXLMCommon
   import MLXVLM

   public enum Qwen38Bonsai2 {
       public static let modelType = "prism_hadamard_qwen35"
       /// Idempotent; call before `VLMModelFactory.shared.loadContainer`.
       /// Mirrors the private `create(Qwen35Configuration.self, Qwen35.init)`
       /// entry for "qwen3_5" in VLMModelFactory.swift (line ≈ 54-64 and 95):
       /// same configuration type, same model class, same validation hook.
       public static func register() async {
           await VLMTypeRegistry.shared.registerModelType(modelType) { data in
               let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
               if let validating = configuration as? ModelConfigurationValidating {
                   try validating.validateModelConfiguration()
               }
               return Qwen35(configuration)
           }
       }
   }
   ```
   `ModelTypeRegistry` est un `actor` (`Vendor/…/MLXLMCommon/Registries/ModelTypeRegistry.swift`) :
   `registerModelType(_:creator:)` prend une fermeture `(Data) throws -> LanguageModel`.
   `Qwen35Configuration` (struct publique, `Qwen35.swift` ligne 26) et
   `Qwen35` (`public class`, ligne 1110, `public init(_ config:)`) sont
   accessibles depuis `Qwen38Core`. Si `ModelConfigurationValidating` n'est
   pas visible, omets ce bloc `if` : `Qwen35Configuration` n'y est pas
   conforme aujourd'hui.
3. **Runtime.** Dans `Qwen38Runtime.load(...)`, cas `.qwen35` (ligne ≈ 437),
   juste avant `container = try await VLMModelFactory.shared.loadContainer(...)` :
   ```swift
   if info.isBonsai2 {
       await Qwen38Bonsai2.register()
       // Belt and braces: the fused 4-way GDN projection cannot fuse mixed
       // packed/float projections anyway (FusedQuantizedLinear returns
       // ineligible), but never let it try on this checkpoint.
       setenv("MLX_QWEN_FOUR_GDN", "0", 1)
   }
   ```
   `qwen35FourGDNEnabled` (`FusedQuantizedLinear.swift` ligne 8) est un `let`
   global évalué à la première lecture : `setenv` doit précéder le premier
   chargement d'un modèle 27B dans le processus, ce qui est le cas ici.
4. **Chargeur vendorisé** — la seule modification autorisée dans `Vendor/` :
   dans `loadWeights` (`Load.swift`), juste après
   `weights = model.sanitize(weights: weights, metadata: metadata)` :
   ```swift
   // Prism Hadamard packs (Bonsai 2) ship a `.signs` vector next to each
   // packed module; it is consumed by the host application after loading
   // (Qwen38Bonsai2Loader), not by the module tree, so keep `verify: [.all]`
   // honest by dropping it here.
   weights = weights.filter { !$0.key.hasSuffix(".signs") }
   ```
5. Compiler : `swift build --product qwen38` sans erreur ni avertissement.
6. Construire et lancer : `Scripts/build.sh` puis
   `.xcodebuild/Build/Products/Debug/qwen38 generate --model-path "$BONSAI" --prompt "Dis bonjour." --max-tokens 16`
   (vérifier le nom exact des options avec `qwen38 generate --help` ; si la
   sous-commande s'appelle autrement, `qwen38 --help`).

**Porte de sortie** : le chargement aboutit sans exception, la mémoire
résidente rapportée est ≈ 9 Go (pas 54), et des jetons sont produits — leur
contenu est faux, c'est attendu et il faut le noter tel quel dans le journal.
Si le chargement échoue sur une clé inattendue, la porte n'est pas franchie :
le nom de la clé va dans le journal.

### B-2 — Les modules Hadamard et leur installation

**Fichiers** : nouveaux `Sources/Qwen38Core/Bonsai2/Qwen38HadamardModules.swift`
et `Sources/Qwen38Core/Bonsai2/Qwen38Bonsai2Loader.swift` ; `Qwen38Runtime.swift`.

1. **Transformée commune** (fidèle au Python, float32) :
   ```swift
   /// x · signs, then a normalized Walsh-Hadamard transform over blocks of
   /// `block` along the last axis (forward); the inverse applies the
   /// transform first and the signs after. Computed in float32 like the
   /// reference runtime, cast back to the input dtype.
   func hadamardRotate(_ x: MLXArray, signs: MLXArray, block: Int, inverse: Bool) -> MLXArray {
       precondition(x.dim(-1) % block == 0, "Hadamard block does not divide activation width")
       let shape = x.shape
       var y = x.asType(.float32)
       if !inverse { y = y * signs }
       y = hadamardTransform(y.reshaped([-1, block]), scale: 1 / Float(block).squareRoot())
           .reshaped(shape)
       if inverse { y = y * signs }
       return y.asType(x.dtype)
   }
   ```
2. **Projection** :
   ```swift
   public final class Qwen38HadamardQuantizedLinear: QuantizedLinear {
       let signs: MLXArray   // float32, [inputDim]
       let block: Int
       public init(_ source: QuantizedLinear, signs: MLXArray, block: Int) {
           self.signs = signs; self.block = block
           super.init(weight: source.weight, bias: source.bias, scales: source.scales,
                      biases: source.biases, groupSize: source.groupSize,
                      bits: source.bits, mode: source.mode)
       }
       public override func callAsFunction(_ x: MLXArray) -> MLXArray {
           super.callAsFunction(hadamardRotate(x, signs: signs, block: block, inverse: false))
       }
   }
   ```
   L'initialiseur désigné existe exactement sous cette forme
   (`.build/checkouts/mlx-swift/Source/MLXNN/Quantized.swift` ligne ≈ 317,
   « Initializer meant for subclasses to provide arrays directly »). Vérifie
   que `signs` n'est **pas** enregistré comme paramètre entraînable (le
   déclarer en `let` simple suffit : `Module` n'introspecte que les propriétés
   `@ModuleInfo`/`@ParameterInfo`). Si `update(parameters:)` se plaint plus
   tard d'un paramètre `signs` manquant, c'est qu'il a été déclaré avec un
   wrapper : retire-le.
3. **Embedding.** `QuantizedEmbedding` n'a **pas** d'initialiseur qui accepte
   des tableaux déjà quantifiés (ses `init` re-quantifient un poids flottant,
   `Quantized.swift` lignes 168-200) : ne le sous-classe pas. Sous-classe
   `Embedding` (`init(weight:)`, `Embedding.swift` ligne 29) et reproduis
   `QuantizedEmbedding.callAsFunction` (lignes 204-212) :
   ```swift
   public final class Qwen38HadamardQuantizedEmbedding: Embedding, Quantized {
       public let groupSize: Int
       public let bits: Int
       public let mode: QuantizationMode
       @ParameterInfo(key: "scales") public var scales: MLXArray
       @ParameterInfo(key: "biases") public var biases: MLXArray
       let signs: MLXArray
       let block: Int
       public init(_ source: QuantizedEmbedding, signs: MLXArray, block: Int) {
           groupSize = source.groupSize; bits = source.bits; mode = source.mode
           _scales.wrappedValue = source.scales
           _biases.wrappedValue = source.biases!   // Bonsai packs always carry biases
           self.signs = signs; self.block = block
           super.init(weight: source.weight)
           freeze()
       }
       public override func callAsFunction(_ x: MLXArray) -> MLXArray {
           let shape = x.shape
           let flat = x.flattened()
           let out = dequantized(
               weight[flat], scales: scales[flat], biases: biases[flat],
               groupSize: groupSize, bits: bits, mode: mode
           ).reshaped(shape + [-1])
           return hadamardRotate(out, signs: signs, block: block, inverse: true)
       }
   }
   ```
   Les clés de paramètres restent `weight`, `scales`, `biases` : les chemins
   ne changent pas. `asLinear` n'est jamais appelé (`tie_word_embeddings: false`).
4. **Chargeur** `Qwen38Bonsai2Loader` :
   - lit `config.json` → `modules[]` (`path`, `block`, `embedding`) et
     `quantization` ; refuse (erreur explicite) si `block` ∉ {512, 1024, 2048, 4096}
     ou `dtype != "float16"` ;
   - lit **uniquement** les vecteurs de signes : `loadArrays(url: model.safetensors)`
     est paresseux, ne garde que les clés `language_model.<path>.signs`, et
     vérifie que chaque vecteur ne contient que ±1 (`all(abs(s) == 1)`) — sinon
     erreur ;
   - pour chaque enregistrement, chemin de module = `"language_model." + path`
     (le VLM `Qwen35` expose `@ModuleInfo(key: "language_model")`), retrouve le
     module courant via `model.leafModules().flattened()` (même mécanisme que
     `quantize(model:filter:apply:)` dans `Quantized.swift` ≈ ligne 106-130),
     exige un `QuantizedLinear` (ou `QuantizedEmbedding` si `embedding: true`)
     avec `bits == 2 && groupSize == 128`, construit le remplaçant, puis
     `model.update(modules: ModuleChildren.unflattened(updates))` en une seule
     fois ;
   - compte : 402 remplacements attendus, sinon erreur avec le compte réel ;
   - `hasFusedInputProjection` et `linearAttn` sont **internes** à MLXVLM,
     inaccessibles depuis `Qwen38Core` : la garde est donc double, côté
     processus (`MLX_QWEN_FOUR_GDN=0` posé en B-1, à revérifier ici avec
     `ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"] == "0"`, sinon
     erreur explicite) et côté résultat (une fusion qui aurait quand même eu
     lieu ignorerait nos modules `in_proj_qkv`/`in_proj_z` dans le forward et
     ferait échouer B-3). Ne cherche pas à exposer ces propriétés.
5. **Branchement** dans `Qwen38Runtime.load`, juste après `loadContainer` :
   ```swift
   if info.isBonsai2 {
       let installer = try Qwen38Bonsai2Loader(directory: directory) // reads config + signs
       await container!.update { context in installer.install(into: context.model) }
       // `install` reports via a counter; assert the 402 afterwards.
   }
   ```
   (`ModelContainer.update(_:)` est `async` non-`throws` : fais toute
   l'analyse pouvant échouer **avant**, dans l'initialiseur, et garde
   `install` non-`throws` avec un résultat lu après.)
6. Compiler, `Scripts/build.sh`, relancer la commande de B-1.

**Porte de sortie** : `qwen38 generate` en greedy (`--temperature 0`, ou
l'option équivalente) répond de façon cohérente aux trois invites suivantes,
et la première coïncide mot pour mot avec la sortie notée en B-0 :
`Dis bonjour en un mot.` ; `Écris une fonction Swift qui renvoie le carré d'un entier.` ;
`Quelle est la capitale de la France ? Réponds en un mot.`
Recopie les trois sorties dans le journal.

### B-3 — Parité contre le runtime Python du pack

**Fichiers** : nouveau `Scripts/references/bonsai2_reference.py`,
`Tests/Qwen38Tests/Qwen38Tests.swift`, `docs/parity-method.md` (une section).

1. **Script de référence** (venv-bonsai2). Pour chacune des trois invites de
   B-2, plus une quatrième **outillée** (une fonction `read_file(path)` déclarée
   dans `tools`, question « Lis le fichier README.md ») :
   - construit le prompt avec `apply_chat_template(processor, chat_config(config), …)`
     et `enable_thinking=False` (on compare le modèle, pas la réflexion) ;
   - tokenise, exécute `model.language_model` sur les ids, sauve les **logits
     de la dernière position** (`float32`, `[vocab]`) et les **32 premiers ids
     greedy** (boucle simple : argmax, ré-injection) ;
   - écrit `parity/bonsai2-reference.safetensors` avec, par invite `i`,
     `prompt_ids_i` (int32), `last_logits_i` (float32), `greedy_ids_i` (int32),
     et une métadonnée `model_sha256` = sha256 des 1 Mo initiaux de
     `model.safetensors` (garde-fou contre un mauvais checkpoint).
   Le fichier généré **n'est pas commité** (voir `docs/parity-method.md`,
   `parity/` est ignoré sauf les fixtures tiny).
2. **Test Swift** gardé par `QWEN38_BONSAI_MODEL` et `QWEN38_BONSAI_FIXTURE`
   (même schéma que les tests gardés par `QWEN38_FLASH_MODEL`, ex.
   `flashTeacherForcedRegressionGuardV32` ≈ ligne 2296) : charge le pack par
   `Qwen38Runtime.load` (pour passer par B-1 et B-2, pas par un chemin
   parallèle), rend **le même prompt** (mêmes ids d'entrée : compare
   `prompt_ids_i` aux ids produits par notre tokenizer + gabarit, et échoue
   d'abord là-dessus si ça diffère), puis :
   - `greedy_ids_i` identiques 32/32 pour les quatre invites ;
   - `last_logits_i` : erreur absolue max ≤ 2e-2 et cosinus ≥ 0,9999 (les
     deux implémentations font la transformée en float32 puis un matmul 2 bits
     identique ; une différence plus grande signale un vrai écart, pas du bruit).
3. Documenter dans `docs/parity-method.md` (section « Bonsai 2 ») la commande
   de génération et la commande de test :
   `TEST_RUNNER_QWEN38_BONSAI_MODEL="$BONSAI" TEST_RUNNER_QWEN38_BONSAI_FIXTURE="$PWD/parity/bonsai2-reference.safetensors" Scripts/run-tests.sh   # préfixe TEST_RUNNER_ obligatoire, sinon le test saute`.

**Porte de sortie** : `TESTS OK` avec le test Bonsai 2 exécuté (pas sauté),
ids greedy 32/32 sur 4/4 invites. Si une invite échoue : STOP, journal, la
suite du plan n'a pas de sens sans cette porte.

### B-4 — Outils sur la famille 27B — **révisée le 2026-09-18 après l'ASK**

**Décision (réponse à l'ASK B-4)** : pas de revue intermédiaire. L'incertitude
« `ChatSession`/`UserInput` exposent-ils `tools` ? » est levée par lecture du
code vendorisé, ci-dessous. Exécute la fiche telle quelle ; les écarts vont
dans le journal comme d'habitude.

**Ce que le code vendorisé offre déjà (vérifié)** :
- `UserInput(chat:tools:additionalContext:)` porte `tools: [ToolSpec]?` avec
  `public typealias ToolSpec = [String: any Sendable]`
  (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Tool.swift` ligne 5).
- Le processeur du type `qwen3_5` (`VLMProcessorLoadingRegistry.swift` ligne
  146, même processeur que Qwen3-VL) transmet `input.tools` au gabarit :
  `Qwen3VL.swift` lignes 112-114, `tokenizer.applyChatTemplate(messages:…, tools: input.tools, …)`.
- `Chat.Message` (`MLXLMCommon/Chat.swift`) sait représenter l'historique
  outillé : `Chat.Message(role:content:images:tool:)` avec
  `tool: .calls([ToolCall])` sur un tour assistant et `.result(id:name:)` sur
  un tour `role: .tool` ; la conversion en dictionnaire pour le gabarit rend
  `tool_calls` (ligne 156) et `tool_call_id` (ligne 170). `ToolCall.Function`
  = `name: String`, `arguments: [String: JSONValue]` (`Tool/ToolCall.swift`).
- Le chemin que le serveur emprunte pour la famille 27B est
  `Qwen38Runtime.generateStateless` (≈ ligne 1472) → `generate(prompt:…,
  forceConversationReplay: true)` → branche `useDirectConversation`
  (≈ ligne 1699) → `UserInput(chat: messages, additionalContext: …)` (≈ ligne
  1774) → `context.processor.prepare(input:)` → `MLXLMCommon.generate`.
  **C'est le seul endroit à modifier pour le rendu.** `Qwen38GenerationOptions.tools`
  (`[Qwen38ToolSpec]`, ≈ ligne 67) et `Qwen38ToolSpec.toolSpecDictionary`
  (`Qwen38ToolCalling.swift` ≈ ligne 174) existent déjà ; `Qwen38ChatMessage`
  a déjà `role: .tool` et `toolCalls: [Qwen38ToolCall]` (≈ lignes 226-245).
- Le parseur de sortie `Qwen38ToolCallParser` est appelé côté serveur sur le
  texte produit dès que `options.tools` est non vide, quelle que soit la
  famille (`makeJSONResponse` et le flux) : **rien à faire** de ce côté.

**Fichiers** : `Sources/Qwen38Core/Qwen38Runtime.swift`,
`Sources/Qwen38Server/Qwen38Server.swift`.

1. **Serveur — lever la garde.** Aux deux endroits (≈ lignes 651 et ≈ 797)
   remplacer `guard await runtime.isFlashNextLoaded` par une garde « un
   modèle est chargé et sa famille rend les outils », c'est-à-dire n'importe
   quel modèle chargé aujourd'hui (Flash-Next, Qwen3.8-27B ordinaire,
   Bonsai 2 partagent le même gabarit outillé). Garde le 400 uniquement pour
   « aucun modèle chargé », et mets à jour le commentaire P13.1 au-dessus
   (« la famille 27B n'a rien d'équivalent » n'est plus vrai : même
   `chat_template.jinja`). Ne touche pas au routage autour du cache de
   conversation Flash-Next qui suit : sur la famille 27B, la requête part
   déjà vers `generateStateless` (≈ lignes 998 et 1012).
2. **Runtime — porter les outils dans l'historique rejoué.** Étendre
   `Qwen38ConversationTurn` (≈ ligne 288, `private struct`) de deux champs :
   `toolCalls: [Qwen38ToolCall]` (tours assistant) et `toolName: String?`
   (tours `tool`, nom de la fonction dont c'est le résultat, `nil` si
   inconnu). Dans `generateStateless`, `priorTurns` les remplit depuis
   `Qwen38ChatMessage` ; le rôle `.tool` doit devenir `Chat.Message.Role.tool`
   (le `Chat.Message.Role(rawValue:)` actuel fonctionne : les deux enums ont
   la valeur brute `"tool"`).
3. **Runtime — rendre `Chat.Message` outillé.** Dans la branche
   `useDirectConversation` (≈ lignes 1764-1781), construire chaque message
   ainsi :
   ```swift
   func chatMessage(_ turn: Qwen38ConversationTurn) throws -> Chat.Message {
       let images = turn.imageURLs.map(UserInput.Image.url)
       switch turn.role {
       case .assistant where !turn.toolCalls.isEmpty:
           let calls = try turn.toolCalls.map { call in
               ToolCall(function: .init(
                   name: call.name,
                   arguments: try JSONDecoder().decode(
                       [String: JSONValue].self, from: Data(call.argumentsJSON.utf8))))
           }
           return Chat.Message(role: .assistant, content: turn.text, images: images,
                               tool: .calls(calls))
       case .tool:
           return Chat.Message(role: .tool, content: turn.text, images: [],
                               tool: .result(id: "", name: turn.toolName))
       default:
           return Chat.Message(role: turn.role, content: turn.text, images: images)
       }
   }
   ```
   Vérifie les initialiseurs exacts dans `Chat.swift` (lignes 23-60 et
   95-105 : `Chat.Message.tool(...)` existe aussi comme constructeur
   statique) et dans `Tool/ToolCall.swift` ; adapte les étiquettes sans
   changer le sens. Le gabarit du checkpoint n'utilise pas `tool_call_id`
   (réponses appariées par ordre), donc un identifiant vide est sans effet.
4. **Runtime — déclarer les outils du tour courant.** Même endroit, dans
   `UserInput(chat: messages, additionalContext: …)`, remplacer le
   dictionnaire `additionalContext` par
   `Qwen4ExpPromptBuilder.templateContext(thinking: options.enableThinking, reasoningEffort: options.reasoningEffort, tools: options.tools.isEmpty ? nil : options.tools)`
   et ajouter `tools: options.tools.isEmpty ? nil : options.tools.map(\.toolSpecDictionary)`.
   `templateContext` (2026-09-18, voir `docs/knowledge/log.md`) place dans
   `additionalContext["tools"]` une `Jinja.Value` **ordonnée** que
   swift-transformers applique après son propre paramètre `tools:` — c'est
   ce qui rend les outils dans l'ordre du client, comme transformers.
   Limite connue sur ce chemin : les arguments d'un `tool_calls` rejoué
   passent par `MLXLMCommon.ToolCall` (`[String: JSONValue]`, non ordonné)
   et sortent donc à clés triées ; acceptable pour B-4, à noter au journal.
5. **Les trois autres constructions de `Chat.Message`** (≈ lignes 1154, 1194,
   1313 : premier tour GUI et chemin M2/MTP) ne sont pas sur le chemin serveur
   de cette fiche ; laisse-les, mais fais-les passer par la même fonction
   `chatMessage(_:)` si c'est une substitution triviale (un tour sans outils
   donne exactement le même `Chat.Message` qu'avant).
6. Compiler (`swift build --product qwen38`, zéro avertissement dans les
   fichiers touchés), `Scripts/build.sh`, puis relancer le serveur sur le pack.

**Porte de sortie** : serveur lancé par
`.xcodebuild/Build/Products/Debug/qwen38 serve --model-path "$BONSAI" --port 8848 --enable-thinking`,
deux `curl` enchaînés sur `/v1/chat/completions` avec un outil
`read_file(path)` : le premier (question « Lis le fichier README.md ») renvoie
`finish_reason: "tool_calls"` et un `tool_calls[0].function.name == "read_file"` ;
le second renvoie l'historique complet (assistant avec `tool_calls`, puis un
message `role: "tool"` contenant un extrait) et obtient une réponse finale
en texte. Rejouer les deux en `stream: true` (le `tool_calls` arrive en un
seul fragment, puis `finish_reason`). Recopier `tool_calls` et la réponse
finale dans le journal, pour le non-stream et pour le stream.

### B-5 — Cache de préfixe sur le chemin 27B

**Fichiers** : `Sources/Qwen38Core/Qwen38Runtime.swift` (≈ 640-720 et ≈ 780-900 :
comparaison de préfixe P6.1, LRU P5.2), `Sources/Qwen38Server/Qwen38Server.swift`.

1. Lire les commentaires P5.2 / P6.1 dans `Qwen38Runtime.swift` : ils décrivent
   comment le chemin Flash-Next compare l'historique reçu au `ledger` d'une
   conversation en cache et ne préremplit que le suffixe. Le chemin 27B
   (`ChatSession`) ne fait cela qu'en continuation stricte (`cacheReused`).
2. Objectif minimal, sans généraliser le LRU : quand la requête reçue est
   `ledger + [assistant précédent] + [nouveaux messages]` de la conversation
   `ChatSession` courante, continuer la session (préremplir uniquement le
   suffixe) au lieu de la recréer. Le rendu du suffixe suit la même logique
   que `continuationSuffix` dans `Qwen4ExpPromptBuilder` (rendu de l'historique
   complet moins rendu de l'historique précédent) ; réutilise cette fonction
   si elle est indépendante de Flash-Next, sinon copie-la dans
   `Sources/Qwen38Core/Bonsai2/` avec un commentaire qui pointe l'original.
3. Renseigner `cachedPromptTokens` dans `Qwen38RunMetrics` sur ce chemin
   (voir comment `Qwen38FlashNextEngine` tient `conversationTokenCount`) pour
   que `usage.prompt_tokens_details.cached_tokens` soit exact.

**Porte de sortie** : deux requêtes successives de la même conversation ; la
seconde renvoie `usage.prompt_tokens_details.cached_tokens > 0` et la ligne
stderr `qwen38 serve · usage · … (dont N en cache)` avec N > 0. Recopier la
ligne.

### B-6 — Mesures et décision

**Fichiers** : `docs/knowledge/log.md` (une entrée), aucun code.
Toujours en **Release** (`Scripts/build-release.sh`), serveur lancé sous
`caffeinate -i`.

1. Décodage et préremplissage à 1 k, 10 k, 30 k et 100 k jetons de prompt
   (prompt = concaténation de fichiers du dépôt, greedy, 128 jetons générés) :
   noter `prompt N jetons`, TTFT et tok/s tels que le serveur les journalise.
   Comparer à Flash-Next 3 bits sur les mêmes prompts (12,7 tok/s décodage en
   boucle d'agent, 44 tok/s préremplissage à 30 k, mesures du 2026-09-17).
2. Mémoire à 100 k : `activeMemoryBytes`/pic rapportés par le serveur ; la
   prévision est ≈ 64 Kio/jeton de cache KV (16 couches d'attention pleine ×
   4 têtes kv × 256 × 2 × 2 octets) soit ≈ 6,4 Go à 100 k, en plus des 8,6 Go
   de poids. Un écart ×2 ou plus est à expliquer avant de conclure.
3. Une fiche réelle du port YuE2 (`~/Developpements/YuE2-mlx-swift`,
   `Scripts/launch-with-local.sh T-2.1` après avoir relancé le serveur sur le
   pack), chiffrée par `Scripts/pi-session-cost.py ~/Developpements/YuE2-mlx-swift/.pi/sessions`
   : durée, tours, jetons de réflexion par tour (la session pi enregistre
   `usage` par tour), issue `VALIDÉE`/`BLOQUÉE`.
4. Réflexion : `pi --thinking low` envoie `reasoning_effort: low` ; le pack dit
   ne pas l'honorer. Mesurer les jetons de réflexion par tour contre la
   session Flash-Next équivalente. C'est le critère qui décide.
5. Entrée dans `docs/knowledge/log.md` : tableau Bonsai 2 vs Flash-Next
   (décodage, préremplissage, mémoire, réflexion/tour, durée de fiche, issue).

**Porte de sortie** : le tableau existe et se termine par une ligne
« Go » ou « No-go » argumentée en trois phrases. Pas de recommandation
d'optimisation avant cette ligne.

### B-7 (optionnel, seulement si B-6 dit « Go » et décodage < 20 tok/s)

Transformée en float16 (mesurer la parité B-3 après : la tolérance peut
bouger, la décision est à Vincent), fusion de `x · signs` dans la transformée,
noyaux du fork `PrismML-Eng/mlx-swift` (branche `prism`). Une mesure
avant/après par changement ; retirer tout changement dont le gain est < 5 %.

## 4. Ordre imposé et dépendances

B-0 → B-1 → B-2 → B-3 (**bloquant**) → B-4 et B-5 (indépendantes) → B-6 → B-7.
Aucune fiche ne commence tant que la précédente n'a pas sa porte de sortie
recopiée dans le journal, sauf B-4/B-5 qui peuvent s'enchaîner dans l'ordre
qu'on veut après B-3.

## 5. Pièges connus (cocher à chaque fiche)

1. `verify: [.all]` : toute clé du safetensors non consommée fait échouer le
   chargement — les `.signs` sont la seule famille de clés à retirer.
2. `hadamardTransform` exige une dernière dimension de taille exactement
   `block` : toujours `reshaped([-1, block])` avant, `reshaped(shape)` après.
3. L'embedding applique les signes **après** la transformée ; les projections
   **avant**. Inverser les deux donne du texte plausible mais faux — c'est
   exactement ce que B-3 attrape.
4. `biases == -scales` est une propriété du pack, pas une hypothèse à coder :
   on lit `biases` comme n'importe quel tenseur.
5. Le gabarit refuse `reasoning_effort` hors `{low, medium, xhigh}` : le
   serveur transmet la valeur reçue telle quelle, le `thinkingLevelMap` de pi
   (`~/.pi/agent/models.json`) fait déjà la correspondance.
6. `setenv("MLX_QWEN_FOUR_GDN", "0")` doit précéder le **premier** chargement
   27B du processus : un serveur qui a déjà servi un 27B classique et change
   de modèle vers Bonsai a déjà figé la valeur à `true`. Ce cas n'est pas
   détectable depuis `Qwen38Core` (propriété interne) : pose la variable
   d'environnement **avant de lancer le processus** pour les mesures
   (`MLX_QWEN_FOUR_GDN=0 qwen38 serve …`) et considère « changement de modèle
   27B → Bonsai dans un serveur déjà lancé » comme non pris en charge tant
   que B-3 n'a pas été rejoué dans cette configuration.
7. Ne jamais mesurer en Debug.
8. La sortie de B-1 est fausse **par construction** : ne pas « corriger » le
   chargeur à ce stade.

## 6. Commandes de référence

```bash
swift build --product qwen38                       # compilation rapide
swift build 2>&1 | grep "warning:.*Sources/"       # doit être vide
Scripts/build.sh                                   # Debug, .xcodebuild/Build/Products/Debug/qwen38
Scripts/build-release.sh                           # Release
Scripts/run-tests.sh                               # suite complète (les tests à checkpoint sautent sans variables)
TEST_RUNNER_QWEN38_BONSAI_MODEL="$BONSAI" TEST_RUNNER_QWEN38_BONSAI_FIXTURE="$PWD/parity/bonsai2-reference.safetensors" Scripts/run-tests.sh   # préfixe TEST_RUNNER_ obligatoire, sinon le test saute
caffeinate -i .xcodebuild/Build/Products/Release/qwen38 serve --model-path "$BONSAI" --port 8848 --enable-thinking
curl -s http://127.0.0.1:8848/healthz
Scripts/pi-session-cost.py ~/Developpements/YuE2-mlx-swift/.pi/sessions
```

## 7. Ce que ce plan ne couvre pas

MTP (le pack n'en a pas), images (la tour vision est chargée mais non
validée ici — hors périmètre tant que B-6 n'a pas tranché), la route
`/v1/messages` (P13.3), et le traitement par lots (P12) sur ce chemin.

## 8. Références

- Pack : https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
  (`README.md`, `PACK-RUNTIME.md`, `runtime/runtime.py`, `runtime/vision_artifact.py`,
  `hadamard.json`).
- Démo et source de vérité d'exécution : https://github.com/PrismML-Eng/Bonsai-demo
  (son `start_mlx_server.sh` **refuse** Bonsai 2 : les serveurs MLX Python
  standard n'appliquent pas la transformée — c'est précisément ce que ce plan
  ajoute chez nous).
- Notre méthode de parité : `docs/parity-method.md`.
- Historique et mesures : `PLAN.md` § P14, `docs/knowledge/log.md`.

## 9. Journal d'exécution (à remplir par l'agent, à la fin du fichier)

Gabarit d'entrée :
```
## B-x — <titre> — <AAAA-MM-JJ> — validée|bloquée
- Fait : <fichiers créés/modifiés, une ligne>
- Porte de sortie observée : `<commande>` → `<ligne exacte recopiée>`
- Écart au plan : <aucun, ou ce qui a dû être adapté et pourquoi>
- Pas d'agent : <n> · appels d'outils : <n>
```

Gabarit ASK (fiche bloquée) :
```
## ASK — B-x — <AAAA-MM-JJ>
- Contexte : <2 lignes>
- Ce que j'ai essayé : <3 lignes max>
- Question : <une question fermée si possible>
- Options : A) … B) …
```

## B-0 — Préparation — 2026-09-18 — validée

- Fait : téléchargement du pack via `qwen38 download` (les fichiers modèle
  standard) puis `hf download --include "runtime/*"` en complément (le
  premier ne récupère pas le dossier `runtime/`, non listé dans son
  manifeste) ; création de `venv-bonsai2` et installation de
  `runtime/requirements.txt` (identique au fichier récupéré à l'avance
  depuis HF, diff vide).
- Porte de sortie observée : script Python de la fiche →
  `GenerationResult(text='Bonjour', token=248046, ..., prompt_tokens=19,
  generation_tokens=2, ..., peak_memory=9.250450707, ...)` — réponse
  française cohérente, mémoire pic ≈ 9,25 Go (attendu ≈ 8,6 Go de poids,
  pas 54 Go). `mx.__version__, mlx_vlm.__version__` → `0.32.0 0.6.3`
  (conforme). `model.safetensors` : 8 595 477 990 octets (conforme).
- Écart au plan : `qwen38 download` ne rapatrie pas `runtime/` (dossier hors
  de son manifeste de fichiers modèle) — complété par `hf download
  prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --include "runtime/*" --local-dir
  "$BONSAI"` plutôt que de relancer tout le dépôt en fallback complet comme
  suggéré au point 1 (aurait retéléchargé les 8,6 Go déjà en place).
- Pas d'agent : 1 (session courante) · appels d'outils : ~12.

## B-1 — Charger le pack sans Hadamard (sortie fausse attendue) — 2026-09-18 — validée

- Fait : `Qwen38ModelValidation.swift` (`family`/`isBonsai2`) ; nouveau
  `Sources/Qwen38Core/Bonsai2/Qwen38Bonsai2Registration.swift` ; branchement
  dans `Qwen38Runtime.load` (cas `.qwen35`, `Qwen38Bonsai2.register()` +
  `setenv("MLX_QWEN_FOUR_GDN", "0", 1)`) ; `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift`
  (filtre des clés `.signs` après `sanitize`).
- Porte de sortie observée : `qwen38 generate --model-path "$BONSAI" --prompt
  "Dis bonjour." --max-tokens 16` → charge sans exception, produit
  `ึกplin兼职惜{'êtburgo为人民_HOR田市ikteINSTchow财.Nilconti` (faux, attendu :
  Hadamard pas encore appliqué) ; `/usr/bin/time -l` → `8890056704 maximum
  resident set size` (≈ 8,3 Gio), `9789562272 peak memory footprint`
  (≈ 9,1 Gio) — conforme à « ≈ 9 Go, pas 54 ».
- Écart au plan : aucun. `swift build --product qwen38` propre ; grep ciblé
  des fichiers touchés (`Qwen38ModelValidation`, `Qwen38Runtime`, `Bonsai2`)
  sans avertissement (des avertissements Sendable préexistants dans
  `Qwen38CLI.swift`, non touché par cette fiche, apparaissent sur un build
  complet — hors périmètre B-1, à traiter séparément si besoin).
- Pas d'agent : 1 (session courante) · appels d'outils : ~15.

## B-2 — Les modules Hadamard et leur installation — 2026-09-18 — validée

- Fait : nouveaux `Sources/Qwen38Core/Bonsai2/Qwen38HadamardModules.swift`
  (`hadamardRotate`, `Qwen38HadamardQuantizedLinear`,
  `Qwen38HadamardQuantizedEmbedding`) et
  `Sources/Qwen38Core/Bonsai2/Qwen38Bonsai2Loader.swift` (lecture de
  `config.json` → `modules[]`, lecture paresseuse des `.signs` via
  `loadArrays`, validation ±1, remplacement des 402 modules via
  `model.update(modules:)`) ; branchement dans `Qwen38Runtime.load` juste
  après `loadContainer`, via `container!.perform { context in
  installer.install(into: context.model) }` (pas `container!.update` : le
  modèle est une référence de classe, `perform` suffit et évite de capturer
  un `var` mutable dans la fermeture `@Sendable`).
- Porte de sortie observée : `qwen38 generate --model-path "$BONSAI"
  --temperature 0` sur les trois invites, avec `--max-tokens` relevé à
  200–400 pour laisser la réflexion se dérouler (voir écart ci-dessous) :
  1. « Dis bonjour en un mot. » → texte de réflexion cohérent puis
     `Bonjour` — identique au mot produit par la référence en B-0.
  2. « Écris une fonction Swift qui renvoie le carré d'un entier. » →
     réflexion cohérente puis code Swift correct
     (`func carré(_ nombre: Int) -> Int { return nombre * nombre }`).
  3. « Quelle est la capitale de la France ? Réponds en un mot. » →
     réflexion cohérente puis `Paris`.
  Les trois réponses sont cohérentes et correctes — contraste net avec le
  charabia de B-1, signe que la transformée Hadamard et l'ordre
  signes/transformée (avant pour les projections, après pour l'embedding)
  sont corrects. La fiche B-3 (parité logits) reste la seule preuve
  quantitative ; ceci n'est qu'un premier repère qualitatif comme prévu par
  le plan (§0 : « un texte plausible ne prouve rien »).
- Écart au plan : la porte de sortie demandait une coïncidence mot pour mot
  avec la sortie B-0 dès `--max-tokens 16`. La sous-commande CLI `generate`
  (`Qwen38CLI.swift`, `struct Generate`) force `enableThinking: true`
  inconditionnellement, alors que le script de référence B-0
  (`chat_config` de `vision_artifact.py`) ne positionne pas
  `enable_thinking` et obtient donc le défaut du gabarit (pas de réflexion
  visible, 2 jetons générés). Les deux ne sont donc pas directement
  comparables à budget de jetons égal : `--max-tokens 16` ne laissait que
  la réflexion démarrer, sans jamais atteindre la réponse. Relancé à
  200–400 jetons pour laisser `</think>` puis la réponse finale
  apparaître ; la réponse finale de l'invite 1 coïncide bien avec la sortie
  B-0 (`Bonjour`). Aucun changement de code motivé par cet écart — c'est un
  problème de comparabilité des bancs d'essai, pas du chargeur Bonsai 2.
- Pas d'agent : 1 (session courante) · appels d'outils : ~40.

## B-3 — Parité contre le runtime Python du pack — 2026-09-18 — validée

- Fait : nouveau `Scripts/references/bonsai2_reference.py` (venv-bonsai2,
  charge via `runtime/vision_artifact.py`, rend les 4 invites avec
  `enable_thinking=False`, sauve `prompt_ids_i`/`last_logits_i`/`greedy_ids_i`
  dans `parity/bonsai2-reference.safetensors`, non commité — git-ignoré
  comme les fixtures Flash-Next) ; nouveau test Swift
  `bonsai2ParityAgainstPythonReference` (`Tests/Qwen38Tests/Qwen38Tests.swift`)
  gardé par `QWEN38_BONSAI_MODEL`/`QWEN38_BONSAI_FIXTURE` ; nouvel accesseur
  `Qwen38Runtime.performRaw` (expose `(any LanguageModel, Tokenizer)` du
  container chargé — seule façon d'atteindre les logits bruts, aucune API
  publique de `ChatSession` ne les expose) ; section « Bonsai 2 » ajoutée à
  `docs/parity-method.md`.
- Méthode : le test charge le pack par `Qwen38Runtime.load` (donc B-1+B-2,
  pas un chemin parallèle), rend les 4 invites avec le même tokenizer +
  gabarit + `tools` que le serveur utiliserait, **compare d'abord les ids de
  prompt** à ceux de Python (échoue immédiatement sinon, avant tout calcul
  numérique — piège explicitement prévu par le plan), puis construit un
  `TokenIterator` (la même machinerie que `ChatSession` en production) avec
  un `LogitProcessor` qui capture les logits du premier appel (= dernière
  position du prompt) et un `ArgMaxSampler` pour 32 jetons greedy.
- Porte de sortie observée : `TESTS OK` — `Test run with 231 tests in 0
  suites passed after 30.218 seconds` (`** TEST SUCCEEDED **`), test B-3
  exécuté (pas sauté) et vert sur 4/4 invites :
  ```
  B-3 invite 0 : greedy 32/32 identiques, logits maxAbsErr=3.4809113e-05, cosine=1.0000001
  B-3 invite 1 : greedy 32/32 identiques, logits maxAbsErr=3.385544e-05, cosine=1.0000006
  B-3 invite 2 : greedy 32/32 identiques, logits maxAbsErr=3.6239624e-05, cosine=1.0000002
  B-3 invite 3 : greedy 32/32 identiques, logits maxAbsErr=5.2452087e-05, cosine=1.0000001
  ```
  Trois ordres de grandeur sous les tolérances (2e-2 / cosinus ≥ 0,9999) —
  quasi bit-exact, cohérent avec « transformée float32 + matmul 2 bits
  identiques des deux côtés ».
- Écart au plan : deux allers-retours avant le vert, aucun des deux dans le
  port Hadamard lui-même.
  1. **Exécution des tests** : `xcodebuild test` sans
     `TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1` (mis
     par `Scripts/run-tests.sh`, oublié dans un premier essai manuel avec
     `-only-testing`) fait tourner les tests en parallèle et bloque
     indéfiniment sur le verrou MLX (0 % CPU, RSS ~200 Mo après plus de 2
     minutes — jamais un chargement de modèle normal). Toujours passer par
     `Scripts/run-tests.sh`, jamais `xcodebuild test` nu.
  2. **Rendu JSON des outils** : l'invite 4 (« Lis le fichier README.md »,
     outil `read_file`) échouait sur la comparaison d'ids (piège attrapé
     comme prévu), pour deux raisons cumulées, aucune liée au modèle :
     (a) `swift-jinja` (`Filters.swift`) trie toujours les clés
     (`JSONEncoder.sortedKeys`) alors que le `tojson` Python
     (`transformers/utils/chat_template_utils.py`) préserve l'ordre
     d'insertion du dict — corrigé en écrivant `READ_FILE_TOOL` en ordre
     alphabétique côté Python (le rendu Swift, lui, ne peut pas faire
     autrement) ; (b) `swift-jinja` sérialise compact (`JSONEncoder` sans
     `.prettyPrinted` n'ajoute jamais d'espace) alors que `json.dumps`
     Python par défaut ajoute un espace après `:`/`,` — corrigé par un
     patch temporaire de `json.dumps` (`compact_tojson()` dans le script de
     référence, `separators=(",", ":")` quand l'appelant passe
     `separators=None`) pour la durée du rendu. Les deux écarts sont des
     caractéristiques partagées par tout le pipeline Jinja du dépôt (donc
     par Flash-Next aussi, déjà en production) — hors périmètre B-3, notés
     ici pour B-4/une fiche future si la qualité d'appel d'outils Bonsai 2
     s'avère sensible à cet ordre/formatage.
- Pas d'agent : 1 (session courante) · appels d'outils : ~90.

## ASK — B-4 — 2026-09-18

- Contexte : B-0 à B-3 validées et committées (`3918c3f`, `ef68393`,
  `b6c1715`, `d78cb4f`) ; B-3, la fiche bloquante, est verte (4/4 invites,
  greedy 32/32, logits ~3-5e-5 sous tolérance). B-4 (outils sur la famille
  27B) touche `Qwen38Server.swift` et `Qwen38Runtime.swift`, qui portent
  chacun un chantier local non commité sans rapport avec Bonsai 2
  (`cachedPromptTokens`/`usage.prompt_tokens_details.cached_tokens`) —
  laissé de côté jusqu'ici via `git add -p` sélectif par hunk.
- Ce que j'ai essayé : lancé un agent Plan pour relire l'incertitude propre
  à B-4 (le plan ne sait pas si `ChatSession`/`UserInput` de MLXLMCommon
  exposent `tools`) et proposer une méthode de rendu de prompt avant
  d'écrire du code. Arrêté sur demande de Vincent : les revues/agents sont
  lancés par lui, pas par l'agent d'exécution du plan.
- Question : qui relance la revue de B-4 (et quand) ?
- Options : A) Vincent lance lui-même la revue (agent Plan ou autre) puis
  redonne le feu vert pour B-4. B) Vincent tranche directement
  l'incertitude `ChatSession`/`tools` et l'agent d'exécution reprend B-4
  sans revue intermédiaire. C) L'agent d'exécution documente son
  investigation dans ce fichier au fil de B-4 (pas de revue à part) et
  Vincent relit après coup, au commit.

FICHE B-4 BLOQUÉE

### Réponse — B-4 — 2026-09-18 (Vincent, via l'auteur du plan)

Option **B**. L'incertitude est tranchée par lecture du code vendorisé (voir
la fiche B-4 révisée ci-dessus, § « Ce que le code vendorisé offre déjà ») :
`UserInput` accepte `tools`, le processeur `qwen3_5` les transmet au gabarit,
et `Chat.Message` représente les appels et résultats d'outils de
l'historique. Aucune revue intermédiaire : reprends B-4 avec la fiche
révisée, journal comme d'habitude. Les modifications non commitées de
`Qwen38Server.swift` / `Qwen38Runtime.swift` / `Qwen38FlashNextEngine.swift`
(bloc `usage`, `cachedPromptTokens`, 400 sur `max_tokens`) appartiennent à
un chantier séparé (issue #1) : commitées à part le 2026-09-18 (`10e9c10`),
l'arbre est propre.

**Complément du 2026-09-18 (rendu `tojson`)** : l'écart Jinja relevé en B-3
était réel et a été corrigé dans la pile (swift-jinja 2.5.1 +
`Qwen38OrderedJSON`, voir `docs/knowledge/log.md`). La compensation Python
(`compact_tojson`, clés alphabétiques) a été **retirée** de
`Scripts/references/bonsai2_reference.py`, la fixture régénérée et la parité
B-3 rejouée telle quelle. Pour B-4, suis l'étape 4 révisée ci-dessus.

## ASK — B-4 — 2026-09-18 (reprise)

- Contexte : fiche reprise avec la révision ci-dessus (réponse à l'ASK
  précédente, option B). Code écrit et compilé (zéro avertissement dans les
  fichiers touchés) : garde serveur levée (`Qwen38Server.swift`, les deux
  occurrences ≈ 651/797, `runtime.isLoaded` au lieu de
  `runtime.isFlashNextLoaded`) ; `Qwen38ConversationTurn` étendue
  (`toolCalls`, `toolName`) ; nouvelle fonction partagée `chatMessage(_:)`
  (`Qwen38Runtime.swift`, remplace les 2 constructions triviales de
  `Chat.Message` concernées, ≈ lignes 1764 et 1349 avant mes ajouts) ;
  branche `useDirectConversation` : `UserInput(tools:...)` +
  `Qwen4ExpPromptBuilder.templateContext(thinking:reasoningEffort:tools:)`.
- Ce que j'ai testé, avec le serveur réellement lancé sur le pack
  (`qwen38 serve --model-path "$BONSAI" --port 8848 --enable-thinking`) :
  1. **Bug trouvé et corrigé, indépendant de B-4 mais jamais déclenché avant
     elle.** Dès que `tools` est non vide, `MLXLMCommon.generate` (via son
     `TokenStreamDecoder`, `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Tool/
     TokenStreamDecoder.swift`) intercepte tout span `<tool_call>…` et
     l'émet comme `Generation.toolCall`/`.rejectedToolCall` — jamais comme
     `.chunk`. Le code existant de `Qwen38Runtime.generate` (avant B-4, donc
     déjà présent, jamais exercé) faisait `case .toolCall,
     .rejectedToolCall: break`, perdant silencieusement tout le texte :
     une requête outillée renvoyait `content: ""`, aucun `tool_calls`,
     `finish_reason: "stop"`, sans erreur. Corrigé : les deux cas
     reconstruisent le texte XML (`reconstructedToolCallXML(_:)` pour un
     appel complet déjà validé par MLXLMCommon, `rejection.rawTextPreview`
     tel quel pour un appel rejeté) et le réinjectent comme `.chunk`, pour
     que `Qwen38ToolCallParser` (déjà testé, déjà utilisé par Flash-Next)
     reste la seule source de vérité, comme le plan le voulait
     (« rien à faire de ce côté »). **Vérifié** : `curl` non-stream ET
     stream sur « Lis le fichier README.md » avec l'outil `read_file`
     renvoient tous les deux `finish_reason: "tool_calls"` et
     `tool_calls[0].function.name == "read_file"`,
     `arguments: {"path":"README.md"}` — exactement la première moitié de
     la porte de sortie.
  2. **Bloquant, non résolu.** Le second `curl` (historique se terminant
     par `role: "tool"`, sans nouveau message utilisateur — exactement la
     forme que la porte de sortie demande) échoue systématiquement avec
     `La conversation doit se terminer par un message utilisateur.`
     (`Qwen38RuntimeError.missingUserMessage`). Cause : `Qwen38Runtime.
     generateStateless` (≈ ligne 1555) a `guard let lastUserIndex =
     messages.lastIndex(where: { $0.role == .user }) ... guard
     lastUserIndex == messages.count - 1 else { throw missingUserMessage }`
     — une contrainte **antérieure à B-4**, jamais mentionnée par la fiche
     (qui ne parle que de remplir `priorTurns`, en supposant implicitement
     que le dernier message reste `.user`). Le serveur, lui, accepte déjà
     `tool`/`assistant` comme dernier rôle au niveau HTTP (P13.1/P13.3) —
     l'incohérence est entre les deux couches.
     - **Vérifié que ce n'est pas un problème de gabarit** : j'ai rendu le
       même historique tri-tours (user → assistant+tool_calls → tool,
       aucun nouveau user) directement via
       `processor.tokenizer.apply_chat_template(...)` côté Python
       (`chat_template.jinja`, celui du pack) : le gabarit **synthétise
       lui-même** un tour `<|im_start|>user\n<tool_response>…
       </tool_response><|im_end|>` autour du message `tool` puis ajoute
       `<|im_start|>assistant\n<think>\n\n</think>\n\n` normalement — voir
       le commentaire déjà présent sur `Qwen38ChatMessage.Role` (« rendu
       … comme un `<tool_response>` fusionné dans le tour utilisateur
       suivant »). Rien à faire côté gabarit ; `chatMessage(_:)` construit
       déjà `Chat.Message(role: .tool, …)` correctement pour ce cas
       (testé isolément dans le rendu, pas encore dans `generateStateless`
       puisqu'on n'y arrive pas).
     - **Ce qui manque réellement** : `generateStateless` sépare
       aujourd'hui `priorTurns` (tout ce qui précède le dernier message
       `user`) d'un nouveau « prompt » `String`, transmis à `generate(
       prompt:…)` qui ajoute lui-même **un nouveau tour `.user`** avec ce
       texte (≈ ligne 1723, `Qwen38ConversationTurn(role: .user, text:
       prompt, …)`) — ce mécanisme n'a pas d'équivalent pour « il n'y a
       pas de nouveau texte utilisateur, continue depuis l'historique tel
       quel ». Deux façons de combler ça, aucune anodine :
       A) Ajouter un chemin dans `generateStateless` qui, quand le dernier
       message n'est pas `.user`, place l'intégralité de `messages` dans
       `conversationTurns` (via `chatMessage(_:)`) et appelle la
       génération SANS passer par `generate(prompt:…)` (qui suppose
       toujours un nouveau tour `.user`) — nécessite une nouvelle petite
       fonction parallèle à `generate`, pas juste changer une garde.
       B) Assouplir `generate(prompt:…)` pour qu'un `prompt` vide
       n'ajoute pas de tour `.user` — plus court, mais `generate` est
       partagé par des appelants qui ne s'attendent pas à ce
       comportement (risque de régression silencieuse ailleurs).
     Je n'ai pas tranché : c'est exactement le type de décision
     d'architecture que le plan me dit de ne pas prendre seul.
- Question : comment `generateStateless` doit-il gérer un historique déjà
  terminé par `tool`/`assistant` (aucun nouveau texte utilisateur) sur le
  chemin 27B ?
- Options : A) option A ci-dessus (nouveau chemin dédié, plus sûr, plus de
  code). B) option B ci-dessus (`generate(prompt: "")` n'ajoute rien, plus
  court, à auditer pour ne rien casser ailleurs). C) limiter le tour 2 du
  test à un historique se terminant par un message utilisateur de relance
  (« Continue. ») — contourne le blocage sans toucher au runtime, mais
  s'écarte de la porte de sortie telle qu'écrite (le second `curl` doit
  précisément renvoyer l'historique avec le message `tool`).

FICHE B-4 BLOQUÉE

### Réponse — B-4 (reprise) — 2026-09-18 (Vincent, via l'auteur du plan)

Le bug `.toolCall`/`.rejectedToolCall` jeté en silence est un vrai bug,
antérieur à B-4 ; le correctif (réinjection du XML en `.chunk`, parseur
unique) est le bon : garde-le, et ajoute-lui un test unitaire sans
checkpoint si `Generation.toolCall` se construit à la main (sinon, note-le
comme couvert par la porte de sortie serveur).

**Décision : ni A ni B tels quels — « A-bis », un seul chemin, sans
chaîne vide magique et sans fonction parallèle.** `generate(prompt:…)` fait
deux choses : construire le nouveau tour `.user`, puis générer depuis
`conversationTurns`. Sépare-les :

1. Renomme le corps actuel en fonction privée
   `generate(newUserTurn: Qwen38ConversationTurn?, systemPrompt: String?,
   options: Qwen38GenerationOptions, forceConversationReplay: Bool)`.
   La fonction publique `generate(prompt:systemPrompt:imageURLs:options:
   forceConversationReplay:)` devient un simple emballage qui construit
   `Qwen38ConversationTurn(role: .user, text: prompt, imageURLs: imageURLs)`
   et appelle le corps. Les sept appelants (`Qwen38Server.swift` ≈ 1007,
   `Qwen38CLI.swift` × 4, `Qwen38BenchUIApp.swift` ≈ 406, et
   `generateStateless`) ne changent pas ; aucun comportement ne bouge pour
   eux, c'est le point.
2. Dans le corps, `prompt` apparaît à quatre endroits (≈ 1649, 1664, 1789,
   1868 dans ta version) :
   - ≈ 1789, le seul qui compte ici : `if let userTurn = newUserTurn {
     conversationTurns.append(userTurn) }` — le tour système reste ajouté
     comme avant, avant ce `if`.
   - ≈ 1868 (`chatSession.streamDetails(to: prompt, …)`, chemin
     `ChatSession` non rejoué) et la branche Flash-Next (≈ 1649/1664) :
     `guard let newUserTurn else { throw Qwen38RuntimeError.missingUserMessage }`
     en tête de branche, puis `newUserTurn.text` / `newUserTurn.imageURLs`.
     Ces branches exigent un nouveau texte utilisateur, on le dit
     explicitement au lieu de le supposer. `generateStateless` ne les
     atteint jamais (il force `forceConversationReplay: true`, donc la
     branche `useDirectConversation`), et la branche MTP/M2 non plus pour
     Bonsai 2 (pas de drafter) — même garde là aussi si `canUseMTP` est
     vrai avec `newUserTurn == nil`.
3. Dans `generateStateless` (≈ 1555) : remplace les deux `guard` par une
   règle en trois cas sur `messages.last?.role` —
   - `.user` : comportement actuel (`priorTurns` = tout sauf le dernier,
     `generate(prompt: last.content, …)`) ;
   - `.tool` : `conversationTurns = messages.map(chatMessageTurn)` (**tous**
     les messages, via la même conversion que `priorTurns`, `toolName`
     compris), `conversationTurnCount = nombre de tours .user`, puis
     `generate(newUserTurn: nil, systemPrompt: nil, options: options,
     forceConversationReplay: true)` ;
   - tout autre dernier rôle (`.assistant`, `.system`) : `missingUserMessage`
     comme aujourd'hui. Un historique fini par `assistant` serait un
     préremplissage de réponse, hors périmètre ; le gabarit rend le tour
     `tool` en `<tool_response>` fusionné dans un tour utilisateur, comme
     tu l'as vérifié, donc `.tool` est le seul cas légitime à ouvrir.
   Le serveur acceptait déjà `tool` en dernier rôle : l'incohérence entre
   les deux couches disparaît sans rien changer côté HTTP.
4. `conversationTurnCount`/`turnIndex` : `turnIndex` compte les requêtes,
   pas les tours utilisateur ; laisse-le s'incrémenter. `cacheReused` reste
   `false` sur ce chemin (rejoué), ce qui est vrai.

Porte de sortie inchangée : le second `curl` renvoie l'historique complet
terminé par le message `role: "tool"` et obtient une réponse finale en
texte, en non-stream **et** en stream. L'option C (message « Continue. »)
est refusée : elle ne teste pas la forme que pi envoie réellement.

Écarts attendus au journal : la liste des sept appelants vérifiés
(`grep -n "runtime.generate(\|try await generate(" Sources/**/*.swift`),
et la ligne exacte de la garde ajoutée dans chaque branche.

## B-4 — Outils sur la famille 27B — 2026-09-18 — validée

- Fait :
  1. **Garde serveur** (`Qwen38Server.swift`, deux occurrences) : `guard
     await runtime.isFlashNextLoaded` → `guard await runtime.isLoaded`, avec
     un message d'erreur générique et le commentaire P13.1 mis à jour.
  2. **Bug `.toolCall`/`.rejectedToolCall` jeté en silence** (antérieur à
     B-4, jamais déclenché avant elle) : `Qwen38Runtime.swift`, boucle
     d'événements de `generate` — dès que `tools` est déclaré,
     `MLXLMCommon.generate` intercepte tout span `<tool_call>…` via son
     `TokenStreamDecoder` et l'émet comme `Generation.toolCall`/
     `.rejectedToolCall`, jamais `.chunk`. Corrigé : les deux cas
     reconstruisent le texte XML (`reconstructedToolCallXML(_:)` pour un
     appel complet, `rejection.rawTextPreview` tel quel pour un appel rejeté)
     et le réinjectent en `.chunk`, pour que `Qwen38ToolCallParser` (déjà
     testé, déjà utilisé par Flash-Next) reste la seule source de vérité.
  3. **Architecture « A-bis » (réponse à l'ASK)** pour le second tour d'un
     round-trip d'outil (historique fini par `role: "tool"`, aucun nouveau
     texte utilisateur) : `generate(prompt:…)` est redevenu un emballage
     public construisant un `Qwen38ConversationTurn` puis appelant un corps
     privé `generate(newUserTurn: Qwen38ConversationTurn?, …)`. `newUserTurn
     == nil` signifie « rien de neuf à ajouter, génère depuis
     `conversationTurns` tel quel » ; les trois branches qui exigent un
     nouveau texte (Flash-Next, MTP local, `ChatSession` non rejoué) le
     disent explicitement (`guard newUserTurn != nil else { throw
     Qwen38RuntimeError.missingUserMessage }`). `generateStateless` porte
     désormais une règle à trois cas sur `messages.last?.role` : `.user`
     (comportement historique), `.tool` (tout l'historique devient
     `conversationTurns` via la nouvelle fonction partagée
     `conversationTurn(_:)`, `newUserTurn: nil`), tout autre rôle
     (`missingUserMessage`, inchangé). Les sept appelants de
     `generate(prompt:…)` (`Qwen38Server.swift:1007`,
     `Qwen38BenchUIApp.swift:406`, `Qwen38CLI.swift:3469/3598/3714/3911`,
     `generateStateless` lui-même) ne changent pas de signature ni de
     comportement — vérifié par grep, confirmé par la suite de tests.
  4. Pas de test unitaire séparé pour `reconstructedToolCallXML(_:)` /
     `Generation.toolCall` construit à la main (la fonction est `private`,
     minuscule, un pur formatage de chaîne) : couvert par la porte de
     sortie serveur ci-dessous, sur les quatre scénarios réels — l'option
     que la réponse à l'ASK autorisait explicitement en repli.
- Porte de sortie observée : serveur lancé
  (`.xcodebuild/Build/Products/Debug/qwen38 serve --model-path "$BONSAI"
  --port 8848 --enable-thinking`), quatre `curl` (non-stream tour 1 et 2,
  stream tour 1 et 2) sur « Lis le fichier README.md » avec l'outil
  `read_file` :
  - Non-stream tour 1 : `finish_reason: "tool_calls"`,
    `tool_calls[0].function.name == "read_file"`,
    `arguments: {"path":"README.md"}`.
  - Non-stream tour 2 (historique avec le message `role: "tool"`) :
    `finish_reason: "stop"`, réponse finale cohérente en français
    reprenant le contenu du fichier.
  - Stream tour 1 : `tool_calls` arrive en un seul fragment `delta`, puis
    `finish_reason: "tool_calls"`.
  - Stream tour 2 : réponse finale en `delta.content`, puis
    `finish_reason: "stop"`.
  Suite complète rejouée après coup (`Scripts/run-tests.sh` avec
  `QWEN38_BONSAI_MODEL`/`QWEN38_BONSAI_FIXTURE`) : `Test run with 233 tests
  in 0 suites passed` / `** TEST SUCCEEDED **` — aucune régression.
- Écart au plan : fiche rouverte deux fois (le journal ci-dessus détaille
  les deux ASK et leurs réponses) ; le code final suit la réponse
  « A-bis », pas les options A/B initialement proposées dans la première
  ASK. `swift build --product qwen38` propre, zéro avertissement dans les
  fichiers touchés (`Qwen38Runtime.swift`, `Qwen38Server.swift`).
- Pas d'agent : 1 (session courante, plus l'agent Plan de l'auteur du plan
  entre les deux reprises) · appels d'outils : ~150 au total sur B-4.

## B-5 — Cache de préfixe sur le chemin 27B — 2026-09-18 — validée

- Fait :
  1. **Découverte clé** qui a simplifié la fiche : `GenerateCompletionInfo`
     (MLXLMCommon/Evaluate.swift) porte déjà `cachedPromptTokenCount` — «
     the number of prompt tokens served by a reused KV-cache prefix …
     `ChatSession` attributes it from its cache reuse decision. » Le chemin
     `ChatSession` de `Qwen38Runtime.generate` (utilisé par le GUI) ne lisait
     simplement jamais ce champ. Corrigé en un endroit (construction de
     `Qwen38RunMetrics` dans `generate`) : `cachedPromptTokens:
     info.cachedPromptTokenCount`. `Qwen38Server.swift` savait déjà
     construire `usage.prompt_tokens_details.cached_tokens` et la ligne
     stderr depuis ce champ (commit `10e9c10`, antérieur) — rien à y
     toucher.
  2. **`generateStatelessContinuation(priorTurns:newUserTurn:options:)`**
     (nouvelle fonction privée, `Qwen38Runtime.swift`) : le chemin rapide
     pour un historique se terminant par `.user`. État dédié,
     délibérément séparé de `chatSession`/`conversationTurns` (le GUI) pour
     ne jamais laisser une requête LAN interférer avec une conversation GUI
     interactive : `statelessSession: ChatSession?`,
     `statelessLedger: [Qwen38ConversationTurn]`,
     `statelessLedgerKey: Qwen38StatelessCacheKey?` (le sous-ensemble
     d'options qui change le rendu du gabarit — `enableThinking`,
     `reasoningEffort`, `tools` — pas `temperature`/`maxTokens`).
     - Extension stricte détectée (`priorTurns == statelessLedger` et même
       clé de cache) : réutilise `statelessSession` tel quel — son cache KV
       est intact, `streamDetails` ne préremplit que le nouveau tour.
     - Sinon : reconstruit une session via l'initialiseur MLXLMCommon de
       « Prompt Re-hydration » (`ChatSession.init(_:instructions:history:…)`,
       `Cache.history([Chat.Message])`, préremplissage différé au premier
       usage) — toujours correct, jamais qu'une optimisation manquée en cas
       de non-correspondance.
     - `Qwen38ConversationTurn` est devenu `Equatable` (tous ses champs le
       sont déjà) pour permettre la comparaison stricte du ledger.
  3. `generateStateless` : le cas `.user` (auparavant fondu dans la règle à
     trois cas de B-4) route maintenant vers
     `generateStatelessContinuation` au lieu du rejeu complet
     systématique ; le cas `.tool` (round-trip d'outil) garde le chemin B-4
     inchangé — combiner cache et continuation d'outil n'est pas couvert
     par cette fiche.
- Porte de sortie observée : serveur réel, deux requêtes successives de la
  même conversation (« Dis bonjour en un mot. » puis, historique étendu,
  « Et maintenant dis au revoir en un mot. ») :
  ```
  qwen38 serve · usage · client LAN · prompt 19 jetons (dont 0 en cache) · sortie 1 jetons · 9.3 tok/s
  qwen38 serve · usage · client LAN · prompt 43 jetons (dont 21 en cache) · sortie 2 jetons · 8.7 tok/s
  ```
  `usage.prompt_tokens_details.cached_tokens` : 0 puis 21 (> 0). Vérifié en
  plus (hors périmètre strict de la porte de sortie, mais utile pour la
  confiance) : une conversation non liée retombe proprement à
  `cached_tokens: 0` (pas de corruption depuis le ledger précédent), puis
  son propre second tour réutilise le cache à son tour
  (`cached_tokens: 30`). Suite complète rejouée : 233 tests, aucune
  régression.
- Écart au plan : aucun dans le résultat, un raccourci dans la méthode — la
  fiche suggérait de s'inspirer de `Qwen4ExpPromptBuilder.continuationSuffix`
  (diff de rendu au niveau des jetons, technique de Flash-Next qui n'a pas
  d'abstraction de session). Inutile ici : `ChatSession` de MLXLMCommon a
  déjà ce mécanisme intégré (réutilisation de cache + comptage exact via
  `cachedPromptTokenCount`) et une API de réhydratation par historique —
  utiliser directement le niveau d'abstraction adapté plutôt que
  réimplémenter à la main ce qu'il fait déjà. `swift build --product qwen38`
  propre, zéro avertissement dans les fichiers touchés.
- Pas d'agent : 1 (session courante) · appels d'outils : ~60.

## B-6 — Mesures et décision — 2026-09-19 — validée (portée réduite)

- Fait : à la demande explicite de Vincent, la fiche T-2.1 YuE2/`pi`
  (§3 point 3 du plan) est **reportée** — hors du périmètre « porter
  Bonsai 2 » pour cette session. Débit (1 k/10 k/30 k), mémoire et
  comparaison à Flash-Next mesurés en entier, Release + `caffeinate -i`,
  profil d'énergie haute performance. Entrée complète dans
  `docs/knowledge/log.md`, « 2026-09-19 — Bonsai 2 : débit et mémoire ne
  passent pas à l'échelle ».
- Porte de sortie observée : tableau dans `docs/knowledge/log.md`, terminé
  par une ligne Go/No-go argumentée. Ligne initiale : **No-go en l'état** —
  74 Go de mémoire pic à 30 k jetons (mesuré via `footprint`, pas `ps`),
  contre 57,4 Go pour Flash-Next 3-bit (125 B MoE, dix fois plus gros sur
  disque) ; 100 k non tenté, jugé non sûr sur cette machine (96 Go, swap
  déjà à 18/18 Go à 30 k). Débit seul comparable à Flash-Next à prompt
  court (24,2 tok/s chauffe vs 12,9 tok/s référence), mais dégradation plus
  marquée avec le contexte (8,0 tok/s à 30 k).
- Écart au plan : (1) mesures de réflexion par tour non chiffrées
  (dépendaient du chantier YuE2/`pi` reporté — noté comme observation
  qualitative seulement dans `docs/knowledge/log.md`) ; (2) incident de
  méthode découvert et corrigé en cours de route : un process externe
  (`gemma4-cli`, 74 Go résident) contaminait la première passe de mesures
  (décodage à 5 tok/s même à 1 012 jetons) — toutes les mesures retenues
  viennent d'une repasse propre après son arrêt (confirmé par Vincent).
- **Correctif et conclusion rejouée (même jour)** : Vincent a relancé la
  question directement (« ça devrait avoir une empreinte réduite, comprendre
  l'écart, ASK si besoin »). L'hypothèse `Memory.cacheLimit` laissée ouverte
  ci-dessus s'est vérifiée immédiatement (`Memory.snapshot()` : 27 Go de
  `cacheMemory` réclamable sur 40 Go de `phys_footprint` à 10 k jetons,
  chemin `.qwen35` jamais borné contrairement à Flash-Next). Correctif
  d'une ligne dans `Qwen38Runtime.load()` (`Memory.cacheLimit = 8 Go`,
  même motif que Flash-Next H3.3). Repasse complète Release, 233 tests
  rejoués (aucune régression), mesures propres 1 k/10 k/30 k **et 100 k**
  (rendu possible par le correctif — non tenté avant, jugé trop risqué) :
  mémoire divisée par 2,5 à 10 k, par 4,4 à 30 k ; 19 Go résident / 35 Go de
  pic à 100 k jetons, débit inchangé. Détail complet et tableau final dans
  `docs/knowledge/log.md`, section « Correctif (même jour) ». **Aucun ASK
  nécessaire** — gap compris et corrigé dans la session, pas transmis à
  l'agent expert. **Conclusion rejouée : Go** (mémoire réglée, seul motif du
  No-go initial ; Bonsai 2 tient désormais sa promesse d'empreinte réduite
  face à Flash-Next à toutes les tailles de contexte testées).
- Pas d'agent : 1 (session courante) · appels d'outils : ~65.
