# Références Python vendorées

- `vlm_q4_language.py` — source `mlx_vlm/models/qwen4_exp/language.py` (mlx-vlm,
  licence MIT), la référence de parité Flash-Next. Copiée telle quelle (ne pas
  modifier : c'est la source de vérité des fixtures de `parity/`). À passer aux
  scripts `qwen4-exp-*-reference.py` et `qwen4-exp-mixer-sensitivity.py` via
  `--scratch Scripts/references/vlm_q4_language.py`.
