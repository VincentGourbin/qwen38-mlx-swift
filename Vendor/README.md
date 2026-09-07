# Vendor/mlx-swift-lm — checkout local

- Base : `ml-explore/mlx-swift-lm`, branche `pr-545`, commit épinglé `1a562aa00bb66d611a086174e14951f41c43e100` (« Require explicit MTP target compatibility »).
- Ce checkout n'est pas versionné dans ce dépôt (`.gitignore`) ; seules les modifications locales le sont, sous forme de patch.
- 7 fichiers modifiés localement (au-delà des 5 attendus à l'origine — la divergence porte sur les variantes MLXLLM *et* MLXVLM de `Qwen35MTP.swift`, plus le test associé) :
  `Libraries/MLXLLM/Models/Qwen35MTP.swift`, `Libraries/MLXLMCommon/GatedDelta.swift`,
  `Libraries/MLXLMCommon/KVCache.swift`, `Libraries/MLXLMCommon/MTPDrafterModel.swift`,
  `Libraries/MLXLMCommon/SwitchLayers.swift`, `Libraries/MLXVLM/Models/Qwen35MTP.swift`,
  `Tests/MLXLMTests/Qwen35MTPTests.swift`.

## Réappliquer le patch sur un checkout neuf

```sh
git clone https://github.com/ml-explore/mlx-swift-lm.git Vendor/mlx-swift-lm
git -C Vendor/mlx-swift-lm checkout 1a562aa00bb66d611a086174e14951f41c43e100
git -C Vendor/mlx-swift-lm apply ../mlx-swift-lm-local.patch
```

## Vérifier que le patch committé correspond toujours au checkout local

```sh
git -C Vendor/mlx-swift-lm stash
git -C Vendor/mlx-swift-lm apply --check ../mlx-swift-lm-local.patch
git -C Vendor/mlx-swift-lm stash pop
```
