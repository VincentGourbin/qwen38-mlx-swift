# Qwen3.8 benchmarks

Chaque run conserve : modèle, quantification, version des dépendances, prompt
tokens, tokens générés, TTFT, prefill tok/s, decode tok/s, mémoire MLX et trace
Chrome. Les lignes `conversation-benchmark` correspondent au même processus
sur trois tours ; `cache historique rejoué` est le comportement M1 MTP attendu.

| Date | Modèle | Quant | Prompt | Générés | TTFT | Prefill | Decode | Peak mémoire | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 68 | 16 | 2.50 s | 27 tok/s | 10.4 tok/s | GUI | text, greedy, limit 16 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 319 | 24 | 3.67 s | 87 tok/s | 10.3 tok/s | GUI | local PNG, greedy, limit 24 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-4bit | 4-bit | 67 | 24 | 1.19 s | 57 tok/s | 10.7 tok/s | — | ChatSession, greedy, limit 24 |
| 2026-08-28 | mlx-community/Qwen3.8-27B-8bit | 8-bit | 69 | 24 | 5.85 s | 12 tok/s | 6.9 tok/s | — | ChatSession, greedy, limit 24 |
| 2026-08-29 | mlx-community/Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 59 | 2 | 0.522 s | 118 tok/s | 21.4 tok/s | — | M1 upstream, greedy, MTP actif, 1 drafter apparié |
| 2026-08-29 | mlx-community/Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 572 | 24 | 5.669 s | 102 tok/s | 18.3 tok/s | — | M1 upstream, première inférence VLM, image, 12 proposés / 10 acceptés |
| 2026-08-29 | mlx-community/Qwen3.8-27B-8bit + MTP-8bit | 8-bit + MTP | 59 | 2 | 0.743 s | 82 tok/s | 10.9 tok/s | — | M1 upstream, greedy, MTP chargé, propositions non observées sur ce run court |
| 2026-08-29 | Qwen3.8-27B-4bit | 4-bit | 1279 → 1441 → 2491 | 140 → 1022 → 1031 | 12.504 → 18.183 → 56.484 s | 103 → 79 → 44 tok/s | 10.29 → 2.87 → 8.37 tok/s | 17,075 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-4bit + MTP-4bit | 4-bit + MTP | 1279 → 1441 → 2385 | 140 → 916 → 884 | 13.403 → 15.098 → 23.409 s | 96 → 96 → 102 tok/s | 12.12 → 12.70 → 10.89 tok/s | 17,885 MiB MLX | M1, 78/63 · 508/408 · 480/404 proposés/acceptés, historique rejoué ; probe mémoire |
| 2026-08-29 | Qwen3.8-27B-8bit | 8-bit | 1279 → 1415 → 2263 | 114 → 820 → 936 | 22.256 → 17.794 → 22.986 s | 58 → 80 → 99 tok/s | 4.35 → 6.20 → 7.22 tok/s | 31,483 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-8bit + MTP-8bit | 8-bit + MTP | 1279 → 1419 → 2326 | 118 → 879 → 646 | 30.991 → 27.335 → 29.829 s | 41 → 52 → 78 tok/s | 3.41 → 4.53 → 4.56 tok/s | 31,483 MiB MLX | M1, 64/55 · 502/378 · 363/284 proposés/acceptés, historique rejoué ; probe mémoire |
| 2026-08-29 | Qwen3.8-27B-bf16 | bf16 | 1279 → 1415 → 2280 | 114 → 837 → 934 | 14.324 → 11.663 → 19.805 s | 90 → 122 → 115 tok/s | 4.84 → 4.85 → 3.96 tok/s | 53,093 MiB MLX | 3 tours, image au tour 1, thinking low, max 2048 ; probe mémoire `max 1` |
| 2026-08-29 | Qwen3.8-27B-bf16 + MTP-bf16 | bf16 + MTP | 1279 → 1415 → 2186 | 114 → 743 → 859 | 11.900 → 11.883 → 29.924 s | 108 → 119 → 73 tok/s | 3.64 → 3.80 → 2.22 tok/s | 54,332 MiB MLX | M1, 62/53 · 413/330 · 471/388 proposés/acceptés, historique rejoué ; probe mémoire |

Les pics ci-dessus sont les valeurs `MLX.Memory.peakMemory` des probes
`maxTokens=1` (un probe par variante et mode MTP), et non une estimation RSS.
Pendant les longs runs bf16, l’allocation driver observée a atteint environ
66,8 GiB sans MTP et 78,0 GiB avec MTP ; cette différence est conservée dans
le journal pour distinguer mémoire MLX suivie par le runtime et allocation
driver macOS.

Important : M1 est la baseline de l’itérateur upstream (`blockSize=2`, donc un
seul token drafté par round). Les compteurs d’acceptation sont bien mesurés,
mais les premiers runs historiques ont montré une divergence sur les tours
avec replay d’image. Après correction du rollback GDN et ajout du fallback
M-RoPE, le contrôle trois tours 4-bit à 512 tokens est identique : tour 1 MTP
actif (63/78 acceptés), tours 2–3 en fallback standard (0 proposition), car
M1 ne sait pas transporter la table M-RoPE image vers le cache privé du
drafter. Ce contrôle valide la sécurité de sortie, mais pas encore le gain
net multi-tour; celui-ci attend le pipeline M2 persistant.
