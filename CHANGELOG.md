# Changelog

## v0.1.1 — 2026-09-27

### Images in the reusable-conversation engine
- The dense engine of `Qwen38Brain` now handles images: prefill goes through
  the model's own `prepare`, so only images new to the conversation run
  through the vision tower, with M-RoPE positions anchored at the cache
  offset. Earlier images and their tokens are reused from the cache.
  3-turn dialogue with two photos: the third turn prefills 32 tokens instead
  of ~2,600 (0.6 s instead of ~25 s). Answers identical to the runtime path,
  including an image added mid-conversation.
- `Qwen38BrainOptions.imageResize` (and `--image-resize` on `qwen38 brain
  ask|replay`): default is the checkpoint's pixel budget (≈ 1,270 vision
  tokens per photo); 512×512 ≈ 200 tokens.
- `Qwen38Runtime.performContext` gives access to the full model context.

## v0.1.0 — 2026-09-27

First tagged release.

### Embeddable brain
- New library product **`Qwen38Brain`**: `load(modelDirectory:profile:)`,
  `respond(to:tools:options:)` streaming `.reasoning`, `.text`, `.toolCall`,
  `.done(usage)`; `resetConversation`, `memoryReport`, `unload`. Depends only
  on `Qwen38Core`.
- **Profiles** `fast` (fp16 KV, 512-token prefill chunks, 4 GB MLX cache) and
  `lean` (8-bit KV, 256-token chunks, memory limits sized from available
  memory, cache cleared after each answer). `textOnlyVariant()` drops the
  vision tower.
- **Conversation reuse** for the dense family (Qwen 3.5 / Bonsai 2): the
  GatedDeltaNet state is snapshotted at the end of the last message and the
  attention KV trimmed, so each agent turn only prefills what is new.
  4-turn agent replay: 78–83 s of prefill instead of 138–148 s, identical
  greedy answers.
- CLI: `qwen38 brain ask|agent|bench|replay`.

### Dependencies
- **No more fork**: `mlx-swift-lm` is upstream `main` instead of a local
  checkout of PR #545. The whole package resolves next to other MLX packages
  (verified in Fluxforge Studio: one `mlx-swift-lm`, the app builds).
- `swift-jinja` ≥ 2.5.1 so `tojson` renders tool specs exactly like Python.

### Models
- **Bonsai 2** (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`): Hadamard-rotated
  2-bit Qwen3.8-27B, loaded with the public mlx-swift-lm API; parity with the
  pack's Python runtime.
- Flash-Next (`qwen4_exp`) and dense Qwen3.8-27B, as before.

### Server
- OpenAI `usage` with `prompt_tokens_details.cached_tokens`, and a per-request
  usage line on stderr.
- `max_tokens <= 0` is refused with HTTP 400.

### Known limitations
- Multi-turn MTP continuation on the dense 27B is gone with the fork: each
  turn re-primes the drafter.
- Because `mlx-swift-lm` is required by branch, SwiftPM does not let another
  package depend on this one by version: use `branch: "master"` or
  `revision:` until mlx-swift-lm publishes a release with the Qwen 3.5 code.
