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

#
