# Sonde : Bonsai 2 sans le fork de mlx-swift-lm

Paquet autonome, **hors du paquet racine** : il charge Bonsai 2 contre
mlx-swift-lm **upstream** (révision `604fae710a`, celle que Fluxforge Studio
résout ; vérifié aussi sur la tête de `main` le 2026-09-27), sans
`Vendor/mlx-swift-lm`.

Seuls `main.swift` et `Package.swift` sont propres à la sonde. Les trois autres
fichiers (`Qwen38HadamardModules.swift`, `Qwen38Bonsai2Loader.swift`,
`Qwen38TokenizerLoader.swift`) sont copiés de `Sources/Qwen38Core` au moment de
la construction, pour ne pas dupliquer le code :

```bash
cd Scripts/bonsai2-upstream-probe
cp ../../Sources/Qwen38Core/Bonsai2/Qwen38HadamardModules.swift \
   ../../Sources/Qwen38Core/Bonsai2/Qwen38Bonsai2Loader.swift \
   ../../Sources/Qwen38Core/Qwen38TokenizerLoader.swift Sources/Bonsai2Probe/
xcodebuild -scheme Bonsai2Probe -destination 'platform=macOS' \
  -derivedDataPath .dd -configuration Release build -quiet
./.dd/Build/Products/Release/Bonsai2Probe ~/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
```

Résultat du 2026-09-27 (M3 Max, Release) :

```
chargé en 2.8 s · actif 8203 Mo
réponse : La capitale de l'Australie est Canberra.
génération 3.7 s · pic 8850 Mo · actif 8385 Mo
```

Le seul apport du fork au chemin Bonsai 2 était le filtrage des clés `.signs`
dans `loadWeights` (patch local non poussé, `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift`).
`loadBonsai2(directory:)` le remplace en refaisant les étapes publiques de
`VLMModelFactory._load` et en retirant `.signs` avant `update(verify: .all)`.
