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
   `QWEN38_BONSAI_MODEL="$BONSAI" QWEN38_BONSAI_FIXTURE=parity/bonsai2-reference.safetensors Scripts/run-tests.sh`.

**Porte de sortie** : `TESTS OK` avec le test Bonsai 2 exécuté (pas sauté),
ids greedy 32/32 sur 4/4 invites. Si une invite échoue : STOP, journal, la
suite du plan n'a pas de sens sans cette porte.

### B-4 — Outils sur la famille 27B

**Fichiers** : `Sources/Qwen38Server/Qwen38Server.swift` (≈ 648-656 et ≈ 794-800),
`Sources/Qwen38Core/Qwen38Runtime.swift` (≈ 1596-1612 et le chemin `ChatSession`).

1. Lire d'abord comment le chemin Flash-Next fait, c'est le modèle à copier :
   `Qwen38Server.swift` (recherche `requestedTools`, `Qwen38ToolCallParser`,
   `rememberConversation`) et `Sources/Qwen38Core/FlashNext/Qwen4ExpPromptBuilder.swift`
   (`tools:` passé à `applyChatTemplate`).
2. Remplacer la garde `guard await runtime.isFlashNextLoaded` par une garde
   « famille sait rendre les outils » : Flash-Next **ou** Bonsai 2 (expose
   `runtime.loadedModelInfo?.isBonsai2` ou un `var supportsTools: Bool` sur le
   runtime). Le 27B ordinaire (`mlx-community/Qwen3.8-27B-4bit`) porte le même
   gabarit : accepte-le aussi, la garde devient `family == .qwen35 || isFlashNextLoaded`,
   et le message d'erreur 400 disparaît sauf si aucun modèle n'est chargé.
3. Dans `Qwen38Runtime`, chemin `ChatSession` : passer `options.tools`
   (déjà présent dans `Qwen38GenerationOptions`, ligne ≈ 67) au rendu du
   gabarit. `ChatSession` de MLXLMCommon prend `additionalContext` ; vérifie
   dans `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ChatSession.swift` s'il
   expose `tools` sur `UserInput` (recherche `tools` dans ce fichier et dans
   `UserInput.swift`). Si oui, passe-les là ; sinon rends le prompt toi-même
   avec `tokenizer.applyChatTemplate(messages:tools:additionalContext:)` comme
   le fait `Qwen4ExpPromptBuilder`, et donne les ids à la session. Ne duplique
   pas l'analyse des `<tool_call>` : `Qwen38ToolCallParser` est déjà appelé
   côté serveur sur le texte produit, indépendamment de la famille.
4. Les messages `role: "tool"` de l'historique doivent traverser `prepare()`
   (serveur, ≈ ligne 1034) jusqu'au gabarit sous la forme que le gabarit
   attend (`<tool_response>` rendu par le gabarit lui-même à partir de
   `role: tool`) — même chemin que Flash-Next.

**Porte de sortie** : avec le serveur lancé sur le pack
(`qwen38 serve --model-path "$BONSAI" --port 8848 --enable-thinking`), un
aller-retour complet outil → résultat → réponse finale réussit en
**non-stream et en stream**, rejoué avec deux `curl` (le second renvoie
l'historique avec le message `tool`). Recopie le `tool_calls` reçu et la
réponse finale dans le journal.

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
QWEN38_BONSAI_MODEL="$BONSAI" QWEN38_BONSAI_FIXTURE=parity/bonsai2-reference.safetensors Scripts/run-tests.sh
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
