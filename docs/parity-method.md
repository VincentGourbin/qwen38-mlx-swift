# Parity method

How this repository proves that the Swift port computes the same values as the
Python reference implementation — and how to regenerate the evidence, because
the evidence itself is not in the repository.

## Why fixtures instead of "it looks right"

Qwen3.8 Flash-Next (`qwen4_exp`) is a 125 B MoE with 6 B active parameters:
Gated DeltaNet, four-stream hyper-connections, sparse QSA attention, an n-gram
per-layer-embedding table, 512 experts routed top-10, and a multimodal MRoPE.
A port of that can be wrong in ways that still produce fluent French. Several
entries in `docs/knowledge/log.md` exist precisely because a plausible-looking
output was hiding a real numerical bug (the checkpoint norm convention, the
fp32 leak out of GDN/QSA, `lm_head` applied to every prefill position).

So nothing is accepted on output quality. Every seam of the port is compared,
tensor by tensor, against the Python reference that produced it.

## The reference of truth

`Scripts/references/vlm_q4_language.py` is a verbatim copy of
`mlx_vlm/models/qwen4_exp/language.py` (mlx-vlm, MIT). It is **never edited** —
it is the source of truth the fixtures are generated from. Every generator
script receives it through `--scratch`, and loads it with an import shim so it
runs against an older installed `mlx-vlm` without patching the reference
itself.

The scripts run under a pinned Python environment (`mlx-vlm==0.6.17`; the
journal refers to it as `venv617`):

```bash
python3 -m venv venv617
venv617/bin/pip install "mlx-vlm==0.6.17"
```

## The chain of evidence

Each generator produces a `.safetensors` fixture that holds the reference
inputs *and* the reference outputs of one seam. A Swift subcommand reloads the
same inputs, runs the Swift path, and compares.

| Seam | Generator (`Scripts/…`) | Fixture (`parity/…`) | Swift check |
|---|---|---|---|
| Sparse QSA attention | `qwen4-exp-qsa-reference.py` | `qwen4-exp-qsa-reference.safetensors` | `qwen38 flash-qsa-parity <fixture>` |
| Multimodal MRoPE tables | `qwen4-exp-mrope-reference.py` | `qwen4-exp-mrope-reference.safetensors` | `qwen38 flash-mrope-parity <fixture>` |
| Vision tower | `qwen4-exp-vision-reference.py` | `qwen4-exp-vision-reference.safetensors` | `qwen38 flash-vision-parity <model> <fixture>` |
| Decoder layer 0 (real weights) | `qwen4-exp-language-reference.py` | `qwen4-exp-language-reference.safetensors` | `qwen38 flash-language-parity <model> <fixture>` |
| Embedding + final mixer + lm-head | `qwen4-exp-global-reference.py` | `qwen4-exp-global-reference.safetensors` | `qwen38 flash-global-parity <model> <fixture>` |
| Embedding → layer 0 → lm-head | `qwen4-exp-single-layer-reference.py` | `qwen4-exp-single-layer-reference.safetensors` | `qwen38 flash-single-layer-parity <model> <fixture>` |
| One public layer call | `qwen4-exp-public-layer-reference.py` | `qwen4-exp-public-layer-{2,3}-reference.safetensors`, `qwen4-exp-e3-layer-*`, `qwen4-exp-rebased-layer-*`, `qwen4-exp-e5-natural-public-layer-4-*` | `qwen38 flash-public-layer-parity <model> <fixture> [--dequantized]` |
| A chain of layers | `qwen4-exp-selected-layers-reference.py` | `qwen4-exp-selected-layers-reference.safetensors`, `qwen4-exp-e5-natural-layer-0-3-reference.safetensors` | `qwen38 flash-selected-layers-parity <model> <fixture>` |
| Teacher-forced full forward | `qwen4-exp-teacher-forced-reference.py` | `qwen4-exp-e5-natural-*-full-reference.safetensors` | `qwen38 flash-teacher-forced-score … --python-fixture <fixture>` |

Two more scripts are part of the method without producing repository fixtures:

- `Scripts/qwen4-exp-official-teacher-forced.py` scores against the *official*
  Qwen weights rather than the converted checkpoint (`--norm-shift 0` vs `-1`),
  which is how the checkpoint norm convention was pinned down.
- `Scripts/qwen4-exp-checkpoint-vs-hf.py` compares the converted 4-bit MLX
  checkpoint, dequantized, against the official BF16 tensors fetched by HTTP
  range requests — no full download.

The n-gram table has no Python fixture: `qwen38 flash-ngram-parity <model>`
compares the lazy per-row reader against an eager read of the same shard, so
it only needs the checkpoint.

## What "same" means

Comparison is numeric and reported, never a silent boolean. Each tensor pair
yields `maxAbsoluteError`, `relativeRMSError` and `cosineSimilarity`
(`Sources/Qwen38Core/FlashNext/Qwen4ExpSelectedLayersParity.swift`), plus
argmax / token-id agreement where logits are involved. Bit-exactness is
*expected* on most seams and *not required* everywhere:

- On the layer chain, a per-layer error table separates a layer that is
  intrinsically wrong from one that merely receives upstream drift.
- MoE routing membership is **not** treated as an invariant across runtimes.
  `Scripts/qwen4-exp-moe-routing-stability.py` measures the top-10/top-11
  router margin and re-evaluates the same router on CPU and GPU within one
  Python process: when membership already flips there, a Python/Swift
  disagreement on the same boundary is kernel noise, not a port bug.
- `Scripts/qwen4-exp-mixer-sensitivity.py` calibrates how much noise it takes
  to flip an argmax through the hyper-connection mixer (σ = 0.0005 rms is
  enough), which is what makes a "bit-identical or explain why not" rule
  meaningful instead of dogmatic.

Determinism comes from the seed: every generator takes `--seed` (defaults
3802 for the layer-0 fixture, 3803 for the globals, 3804 for the single-layer
chain) and writes it, with the other generation parameters, into the
safetensors **metadata**. A fixture therefore carries the recipe that produced
it — read it back with `mx.load(path, return_metadata=True)` before assuming
what a fixture contains.

## The end-to-end guard

Fixtures prove the seams. One more check proves the assembled model does not
drift: the **Q-B V32 guard**, a teacher-forced scoring of a fixed continuation
on the real checkpoint (`flashTeacherForcedRegressionGuardV32()` in
`Tests/Qwen38Tests/Qwen38Tests.swift`, floor overridable with
`QWEN38_QB_MIN_LOGPROB`). Its reference value, quoted throughout
`BENCHMARKS.md`, is **10/28 confident-argmax agreements and a mean log-prob of
−4.8003182**. Any optimisation that moves that number is rejected or explained.

## Regenerating the fixtures

The fixtures are ~287 MB and useless without the checkpoint, so they are
git-ignored (`parity/*.safetensors`). Regenerating them needs:

- the **full Flash-Next checkpoint** (the 4-bit MLX conversion is ~113 GB, the
  3-bit re-quantisation ~84 GB) — the two synthetic fixtures (QSA, MRoPE) are
  the only exceptions;
- `venv617` as above;
- `export QWEN38_MODELS_DIR=/path/to/models` and
  `FLASH=$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`;
- `REF=Scripts/references/vlm_q4_language.py`.

```bash
# 1. Synthetic, no checkpoint needed
venv617/bin/python Scripts/qwen4-exp-qsa-reference.py \
  --output parity/qwen4-exp-qsa-reference.safetensors
venv617/bin/python Scripts/qwen4-exp-mrope-reference.py \
  --output parity/qwen4-exp-mrope-reference.safetensors

# 2. Real weights, one seam at a time
venv617/bin/python Scripts/qwen4-exp-vision-reference.py \
  --model-dir "$FLASH" --output parity/qwen4-exp-vision-reference.safetensors

venv617/bin/python Scripts/qwen4-exp-language-reference.py \
  --model-dir "$FLASH" --scratch "$REF" \
  --output parity/qwen4-exp-language-reference.safetensors

venv617/bin/python Scripts/qwen4-exp-global-reference.py \
  --model-dir "$FLASH" --scratch "$REF" \
  --output parity/qwen4-exp-global-reference.safetensors

venv617/bin/python Scripts/qwen4-exp-single-layer-reference.py \
  --model-dir "$FLASH" --scratch "$REF" \
  --output parity/qwen4-exp-single-layer-reference.safetensors

# 3. Public layer calls (one fixture per layer index)
for i in 2 3; do
  venv617/bin/python Scripts/qwen4-exp-public-layer-reference.py \
    --model-dir "$FLASH" --scratch "$REF" --layer-index "$i" \
    --output "parity/qwen4-exp-public-layer-$i-reference.safetensors"
done

# 4. A chain of layers, and the same chain on a natural prompt
venv617/bin/python Scripts/qwen4-exp-selected-layers-reference.py \
  --model-dir "$FLASH" --scratch "$REF" --layers 0,1,2,3 \
  --output parity/qwen4-exp-selected-layers-reference.safetensors

# 5. Teacher-forced full forward (--norm-shift -1 for a Vontra-converted
#    checkpoint, 0 for official weights)
venv617/bin/python Scripts/qwen4-exp-teacher-forced-reference.py \
  --model-dir "$FLASH" --prompt-ids <ids> --continuation-ids <ids> \
  --output parity/qwen4-exp-e5-natural-full-reference.safetensors
```

Variants of the fixture names carry the study they belong to:
`-e3-` fixtures use `--dequantized` (FP32 weights, no PLE), `-rebased-` ones
feed each layer with the *previous Python* layer output through
`--input-fixture`, `-e5-natural-` ones use a real chat-template prompt via
`--prompt` instead of a random seeded input, and `-mlx032-` marks a fixture
regenerated under a different MLX version. `docs/knowledge/log.md` says which
question each one was built to answer.

## Running the checks

The Swift side reads fixtures from environment variables, so the whole suite
runs with the real checkpoint and skips cleanly without it:

```bash
QWEN38_FLASH_MODEL="$FLASH" \
QWEN38_QSA_FIXTURE=parity/qwen4-exp-qsa-reference.safetensors \
QWEN38_MROPE_FIXTURE=parity/qwen4-exp-mrope-reference.safetensors \
QWEN38_VISION_FIXTURE=parity/qwen4-exp-vision-reference.safetensors \
QWEN38_LANGUAGE_FIXTURE=parity/qwen4-exp-language-reference.safetensors \
QWEN38_GLOBAL_FIXTURE=parity/qwen4-exp-global-reference.safetensors \
QWEN38_SINGLE_LAYER_FIXTURE=parity/qwen4-exp-single-layer-reference.safetensors \
QWEN38_PUBLIC_LAYER_2_FIXTURE=parity/qwen4-exp-public-layer-2-reference.safetensors \
QWEN38_PUBLIC_LAYER_3_FIXTURE=parity/qwen4-exp-public-layer-3-reference.safetensors \
QWEN38_SELECTED_LAYERS_FIXTURE=parity/qwen4-exp-selected-layers-reference.safetensors \
Scripts/run-tests.sh
```

Without `QWEN38_FLASH_MODEL`, the checkpoint-dependent tests skip and the rest
of the suite still runs — that is the mode CI-style runs use.

Every command also exists standalone on the CLI (`qwen38 flash-*-parity`),
which is the form used while debugging a single seam.
