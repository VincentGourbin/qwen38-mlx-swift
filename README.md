# Qwen3.8 MLX Swift

Native Qwen3.8 inference on Apple Silicon via [MLX Swift](https://github.com/ml-explore/mlx-swift) —
the dense **27B** and the **Flash-Next** (`qwen4_exp`) sparse MoE — 512 experts
per layer, 10 routed per token across 48 layers — with a CLI, an
OpenAI-compatible server (tool calling included) and a SwiftUI bench app.

Flash-Next is not a small model politely quantized. It is Gated DeltaNet,
four-stream hyper-connections, sparse QSA attention, an n-gram per-layer
embedding table read from disk on every token, and 512 experts routed top-10.
Everything below was measured on the port, not estimated — and the failures
were kept too: `PLAN.md` and [`docs/knowledge/log.md`](docs/knowledge/log.md)
are the honest measurement journal, dead ends included.

## Status

| Feature | Status | Details |
|---|---|---|
| Flash-Next text generation | ✅ **Working** | 48 resident layers, 3-bit checkpoint, **13.0–13.5 tok/s** greedy |
| Flash-Next vision (images) | ✅ **Working** | Vision tower + MRoPE merge, parity-checked against Python |
| Dense 27B (4-bit / 8-bit / bf16) | ✅ **Working** | 10.3–10.7 tok/s decode at 4-bit, text + image |
| Thinking mode | ✅ **Working** | `reasoning_content` separated from the answer, per-request or `--enable-thinking` |
| OpenAI-compatible server | ✅ **Working** | `/v1/chat/completions` with SSE streaming, `/v1/models`, `/healthz`, `/metrics` |
| Tool calling | ✅ **Working** | OpenAI `tools` / `tool_calls`, validated end to end. Flash-Next only |
| Conversation & prefix caching | ✅ **Working** | Median TTFT **2.34 s** over 58 turns, with or without `conversation_id` |
| Continuous-ish batching | ✅ **Working** | `--batch-size`: **×2.16** aggregate throughput at 8 clients, no solo regression |
| Agent loop (GUI panel) | ✅ **Working** | 4 sandboxed read-only tools, no shell. Loop ×5.6 faster since suffix-by-diff |
| Bench GUI | ✅ **Working** | 3 tabs: Chat, Server, Agent |
| Parity harness | ✅ **Working** | 9 generator scripts, tensor-by-tensor, see [parity method](docs/parity-method.md) |
| Speculative decoding (MTP) | ⚠️ **Off by default** | Measured at 0.939× greedy after the F7 fix, and crashes the QSA mask at long context. Kept, not recommended |
| Expert offload to disk | ❌ **Not shipped** | Studied and dropped: decoding needs 13.3 GiB/s of expert weights per token, more than any local storage path measured here. See `docs/knowledge/investigations/p3-expert-offload.md` |

## Requirements

- **macOS 15** (Sequoia) or later
- **Apple Silicon**. Flash-Next was developed and measured on an M3 Max
- Xcode 16+, Swift 6.0+
- **96 GB of unified memory for Flash-Next.** This is not a soft
  recommendation: the 3-bit checkpoint peaks at **57–58 GB of resident
  process memory**, and the preflight script refuses to start a run when more
  than ~9 GB is already held by other applications. The dense 27B at 4-bit is
  far more modest (**17,075 MiB** peak MLX memory measured)
- **~90 GB of free disk** for the 3-bit Flash-Next checkpoint (~113 GB for the
  4-bit one). A fast SSD matters: the n-gram table is read at random on every
  token

### GPU wired limit

macOS caps wired GPU memory at roughly 72 GB on an M3 Max. Raise it before a
long resident session:

```bash
sudo sysctl iogpu.wired_limit_mb=85000
```

This resets at reboot. `Scripts/preflight-resident.sh` checks it, along with
anonymous memory, compressor pressure, swap, power source and sleep
assertions, and refuses the run rather than letting it die halfway.

## Quick Start

### Build

```bash
git clone https://github.com/VincentGourbin/qwen38-mlx-swift
cd qwen38-mlx-swift

Scripts/build-release.sh
```

`mlx-swift-lm` is the upstream package (`main` branch, the same one apps using
LTX or Gemma resolve), so this package can be added to such an app without a
dependency conflict. The former local fork is described, for history only, in
[`Vendor/README.md`](Vendor/README.md).

> **Use `Scripts/build-release.sh`, not `swift build`.** MLX inference needs
> the resource bundle Xcode produces —
> `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` — which SwiftPM's
> plain executable output does not carry. The script wraps `xcodebuild` on the
> package scheme so both the CLI and the GUI are refreshed together. Release
> also matters for measurement: MLX host overhead is ~1.2–1.8× higher in Debug.

Binaries land in `.xcodebuild/Build/Products/Release/`:
`qwen38` (CLI) and `qwen38-bench-ui` (GUI).

```bash
Scripts/build.sh      # Debug
Scripts/run-tests.sh  # test suite, its own derived data
```

### Choose where models live

```bash
export QWEN38_MODELS_DIR=/Volumes/YourSSD/models   # default: ~/models
```

### Download a model

```bash
# Dense 27B, a good first run
.xcodebuild/Build/Products/Release/qwen38 download mlx-community/Qwen3.8-27B-4bit

# Flash-Next (large — prefer the resumable script for a multi-hour transfer)
Scripts/download-hf-resumable.sh Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
```

`Scripts/download-hf-resumable.sh` exists because a dropped connection on a
multi-gigabyte LFS shard must not discard the partial file. It resumes
per-file, reuses your local `hf auth` session without ever putting the token
on a command line, and verifies sizes against the Hub manifest.

Optionally re-quantize the experts from 4-bit g32 to 3-bit g64 — this is what
turns a 113 GB checkpoint that barely fits into an 84 GB one that runs
comfortably, with identical greedy output:

```bash
python3 -m venv venv617 && venv617/bin/pip install "mlx-vlm==0.6.17"
venv617/bin/python Scripts/qwen4-exp-requantize-experts.py \
  --src "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
```

(The pinned `mlx-vlm==0.6.17` environment is the same one the parity fixtures
are generated from — see [parity method](docs/parity-method.md).)

### Generate

```bash
qwen38 generate \
  --model-path "$QWEN38_MODELS_DIR/mlx-community/Qwen3.8-27B-4bit" \
  --prompt "Explain unified memory in three sentences" \
  --max-tokens 200

# With an image
qwen38 generate --model-path … --prompt "What is in this photo?" --image photo.jpg
```

### Inspect a checkpoint

```bash
qwen38 info "$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
```

## OpenAI-compatible server

```bash
qwen38 serve --model-path "$QWEN38_MODELS_DIR/local/Qwen3.8-Flash-Next-MLX-e3bit-MTP"
```

Binds `0.0.0.0:8848`. Four routes:

| Route | What it returns |
|---|---|
| `POST /v1/chat/completions` | Chat, JSON or SSE stream |
| `GET /v1/models` | The catalogue discovered under the model root |
| `GET /healthz` | Liveness plus the server's effective settings |
| `GET /metrics` | A JSON snapshot (sessions, cache hits, MTP counters) — not Prometheus format |

```bash
curl http://127.0.0.1:8848/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-MLX-e3bit-MTP",
       "messages":[{"role":"user","content":"Bonjour"}],
       "max_tokens":128,"stream":true}'
```

### Notable options

| Option | Why it exists |
|---|---|
| `--enable-thinking` | Reasoning on by default for clients that do not ask. **Required for agent harnesses**: without it the model chains relevant tool calls and never concludes (measured 2026-09-15). A client sending `enable_thinking` or `reasoning_effort` still wins |
| `--batch-size N` | Group up to N cold requests into one decode step. Default 1 = strictly unchanged behaviour. Measured **×2.16** aggregate at 8 clients, and the last client served goes from 38.57 s to 17.83 s |
| `--batch-max-prompt-tokens` | Default 256. Prefill is dense work with no idle GPU to fill, so batching it adds instead of overlapping — a batched request's TTFT grows roughly linearly with batch size (×4.9 on ~1200-token prompts) |
| `--batch-window-ms` | Grouping window, default 30 ms |
| `--routed-experts N` | Override the checkpoint's `num_experts_per_tok`. Absent = checkpoint value |
| `--conversation-cache-gb` | Per-client conversation LRU budget, default 12 GB |
| `--api-key` | Optional `Authorization: Bearer` |
| `--allow-ablation` | Unlocks the `ablation` request field. Refused by default — ablation produces numerically wrong output by construction and must never be reachable by an ordinary client |
| `--trace <path>` | Profile the whole serving session and write a Chrome trace on Ctrl-C |

Non-standard request fields, accepted at the top level or inside `extra`:
`enable_thinking`, `reasoning_effort`, `mtp` (default `false`),
`mtp_engine`, `mtp_draft_tokens`, `conversation_id`, `routed_experts`,
`repetition_penalty`, `penalty_context_tokens`.

Responses add `reasoning_content` next to `content`. Note there is **no
`usage` object** — token counts live in `/metrics`, not in the reply.

### Tool calling

`tools`, `tool_calls` and the `tool` role work in the OpenAI shape. The model
emits the checkpoint's own `<tool_call><function=…><parameter=…>` grammar;
`Qwen38ToolCalling` parses it with a deliberately non-regex scanner (a
parameter value can contain `<`, `>` and newlines — it is often source code)
and retypes each argument against the declared JSON Schema.

Known boundaries, all deliberate:

- **Flash-Next only.** `tools` against the 27B family returns HTTP 400: that
  chat template only exists in this checkpoint.
- `tool_choice: "none"` is honoured; forcing a specific function is accepted
  but has no effect — the template has no notion of it.
- `tool_call_id` is accepted and ignored: the template pairs tool results **by
  order**.
- In streaming with tools declared, `content` is buffered and delivered in one
  chunk so a raw `<tool_call>` never leaks into the stream. `tool_calls` are
  emitted whole, not token by token. `reasoning_content` still streams.
- A `<tool_call>` truncated by `max_tokens` is folded back into `content`
  verbatim — never guessed, never completed.

### Timeouts

On Flash-Next, the first turn after selecting a model can take ~100 s of layer
loading before prefill even starts. In **streaming** mode the server emits an
SSE `: loading` comment every 10 s so a proxy or client does not cut the
connection on a 60 s idle timeout. In **non-stream** mode no such signal is
possible: set a client timeout of **at least 300 s**, or use streaming.

### Using it from a coding agent

The GUI's Server tab has a copy-ready OpenCode configuration
(`@ai-sdk/openai-compatible` provider, `reasoningField: reasoning_content`).
Note the honest caveat below about tool-heavy clients.

## Bench GUI

```bash
.xcodebuild/Build/Products/Release/qwen38-bench-ui
```

Three tabs:

- **Chat** — talk to the resident model, with live tok/s, TTFT, load duration,
  MTP acceptance rate, per-turn history, and active/peak memory beside it.
- **Serveur** — start and stop the LAN server, pick a model from the
  discovered catalogue, watch sessions refresh every 400 ms.
- **Agent** — an agent loop that talks to the local server over plain HTTP,
  like any third-party client would. It offers the model four tools:
  `list_files`, `read_file`, `grep`, `final_answer`. **No shell**: `grep` is
  reimplemented in Swift, and every path is resolved against a sandbox root
  chosen in a file picker, matched on a `/` boundary so `…/project-evil`
  cannot pass for `…/project`. Tool output is truncated (6000 chars, 200 lines
  per read, 60 grep hits); errors come back to the model as text so it can
  correct itself rather than aborting the loop. The loop stops on
  `final_answer`, on the step budget (default 16), or on user stop.

## Measured performance

Everything here comes from [`BENCHMARKS.md`](BENCHMARKS.md) and
[`docs/knowledge/log.md`](docs/knowledge/log.md), on an M3 Max with 96 GB.

### Flash-Next, 3-bit g64 checkpoint (~84 GB on disk)

| | Value | Context |
|---|---:|---|
| Greedy decode | **13.0–13.5 tok/s** | final default (F7 + concurrent `pread`), short and long runs |
| Prefill, varied prose | **112.1 tok/s** | 4,831 tokens, n-gram shards on internal SSD |
| Prefill, ~30 k tokens | 44 tok/s | throughput halves past ~10 k tokens — cause not identified |
| Resident load (TTFT of turn 1) | 48–63 s | paid once per process; the GUI and server keep it resident |
| Peak process memory | **57.6–58.3 GB** | peak MLX memory 56.6 GB |
| Q-B regression guard | 10/28 · −4.8003182 | teacher-forced, exact value held across every optimisation |

The road there is the interesting part. `flash-layer-bench` said the routed
MoE was 78–88 % of a layer, which was a correct measurement and a wrong
diagnosis: `SwitchGLU` in isolation costs 150–350 µs, unless its input is
`.float32` — which it always was, because GDN and QSA never cast their output
back. Fixing that one dtype (`F7`) gave **×2.74** on the real checkpoint.
Meanwhile a hand-fused `switch_mlp` kernel that was bit-exact on the synthetic
bench turned out **17.6× slower** and +36.7 GB of peak memory on the real
model, and was deleted. Two Metal fusion kernels (F8, F9) were written,
measured at +0.8 % and +1.0 % — slower — and deleted too.

### Server under load

| | Serialised | Batch of 8 |
|---|---:|---:|
| Aggregate throughput, 8 clients | 19.91 tok/s | **43.07 tok/s (×2.16)** |
| Last client served | 38.57 s | **17.83 s** |
| Single client | 20.48 tok/s | 20.38 tok/s (no regression) |

A 20-minute A/B dialogue without `conversation_id` — the ordinary OpenAI-SDK
case where every turn resends the whole history — held a **median TTFT of
2.34 s over 58 turns**, with the implicit prefix cache restoring 56 of them.

### Dense 27B

Measured 2026-08-28/29. The peak column is an `MLX.Memory.peakMemory` value
from a dedicated `maxTokens=1` probe per variant, not an RSS estimate.

| Quant | Peak MLX | Short greedy run | Three turns, 2048-token budget |
|---|---:|---|---|
| 4-bit | 17,075 MiB | 27–87 tok/s prefill, 10.3–10.7 tok/s decode | 103 → 79 → 44 prefill, 10.29 → 2.87 → 8.37 decode |
| 8-bit | 31,483 MiB | 12 tok/s prefill, 6.9 tok/s decode | 58 → 80 → 99 prefill, 4.35 → 6.20 → 7.22 decode |
| bf16 | 53,093 MiB | — | 90 → 122 → 115 prefill, 4.84 → 4.85 → 3.96 decode |

The three-turn runs put an image on turn 1, thinking on low, 2048 max tokens.
Full conditions and the MTP variants are in `BENCHMARKS.md`.

### Agent loop

Rendering only the *new* suffix of a conversation instead of replaying it took
the same ten-step task from **683.7 s to 122.0 s (×5.6)**; cumulative prefill
from 593 s to 74 s. From the third step on, only a few dozen tokens are
prefilled when a tool result is short.

## Known limitations

Kept visible on purpose.

- **MTP is off by default** since 2026-09-16, for two measured reasons: it came
  out **losing** (0.939× greedy — 19.72 against 21.01 tok/s), and the MTP path
  crashed the server at long context with a QSA mask shape mismatch. Earlier,
  at 55 % acceptance, it had been worth ~3 % over greedy; the F7 dtype fix
  sped up greedy enough to erase that. The code stays, the default does not —
  a client sending `mtp: true` still gets it.
- **Prefill throughput halves past ~10 k tokens** (86 → 44 tok/s between 10 k
  and 30 k). Not the QSA sparse budget — the 3 k and 10 k points are already
  above it and flat. Cause not identified.
- **Tool-heavy clients are impractical.** Claude Code declares 59 tools,
  ≈31,700 tokens of schemas in *every* request — about **12 minutes** to first
  token on a cold session. Even at the best prefill rate ever measured here it
  would still be 3.8 minutes. The client's tool payload is the constraint, not
  the server. Leaner harnesses (OpenCode with a small tool set, the built-in
  agent panel) are fine.
- **One model resident per process**, requests serialised outside the batch
  path. This is not vLLM.
- **No `usage` object** in chat completions; counts are in `/metrics`.
- Vision and conversation caching do not combine with batching: a request
  carrying an image, declaring tools, or hitting a cache always takes the
  single-sequence path.
- `mlx-swift-lm` tracks upstream `main` until a tagged release carries the
  Qwen 3.5 / MTP code; multi-turn MTP continuation on the dense 27B path is
  not available upstream, each turn re-primes the drafter. `mlx-swift` is
  pinned to an **exact** version — patch releases have changed APIs before.
- The bench and probe subcommands are shipped in the binary. That is
  intentional (every published number is reproducible from it), but it makes
  `qwen38 --help` long.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `QWEN38_MODELS_DIR` | `~/models`, then `~/Library/Caches/models` | Root of the local model catalogue |
| `QWEN38_FLASH_CHECKPOINT` | `$QWEN38_MODELS_DIR/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | Checkpoint used by `Scripts/qwen4-exp-checkpoint-vs-hf.py` |
| `QWEN38_CONFIGURATION` | `Debug` | Build configuration (`Scripts/build-release.sh` sets `Release`) |
| `QWEN38_DERIVED_DATA` | `.xcodebuild` | Derived data path |
| `QWEN38_SCHEME` | `Qwen38MLXSwift-Package` | Xcode scheme to build |
| `QWEN38_NGRAM_MMAP` | unset | `1` restores the old `mmap` n-gram read path |
| `QWEN38_FUSION_LEVEL` | unset | Test-only: re-runs the Q-B guard at another fusion level (production uses F7) |
| `QWEN38_DTYPE_AUDIT` | unset | Report any tensor that leaves a branch in the wrong dtype |
| `QWEN38_PREFLIGHT_LIMIT_GB` | `9` | Memory headroom threshold in `Scripts/preflight-resident.sh` |
| `QWEN38_REF_IMAGE` | `results/assets/ref-image.jpeg` | Reference image for `Scripts/h6-qualification.sh` |
| `QWEN38_FLASH_MODEL`, `QWEN38_27B_MODEL`, `QWEN38_*_FIXTURE` | unset | Checkpoint and parity fixtures for the test suite — see [parity method](docs/parity-method.md) |
| `QWEN4_HF_CACHE` | `Scripts/hf-cache` | Cache for BF16 tensors fetched by range request |

## CLI

`qwen38 --help` lists about forty subcommands. The ones you actually use:

| Command | What it does |
|---|---|
| `info <model-path>` | Inspect a local checkpoint |
| `download <model-id>` | Fetch a Hugging Face repo |
| `generate` | Stream a single answer, optionally with an image |
| `serve` | Start the OpenAI-compatible server |
| `flash-chat-probe` | Multi-turn Flash-Next chat with full measurement output |
| `flash-layer-bench` | Synthetic single-layer micro-bench — no checkpoint needed |
| `flash-decode-bench` | Per-sub-block decode attribution on the real checkpoint |
| `flash-teacher-forced-score` | Score a fixed continuation (the Q-B regression guard) |
| `flash-*-parity` | Compare one seam against a Python fixture |
| `conversation-benchmark` | Three consecutive turns on the same context |

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the five modules and what
  each one owns
- [`docs/parity-method.md`](docs/parity-method.md) — how the port is proven
  equivalent to the Python reference, and how to regenerate the fixtures
- [`BENCHMARKS.md`](BENCHMARKS.md) — every measurement, with its conditions
- [`docs/knowledge/`](docs/knowledge/index.md) — durable findings,
  regressions, pitfalls, and the dated execution log
- `PLAN.md` — the implementation plan and its journal *(French)*
- [`docs/reference/claude-code-wire-format.md`](docs/reference/claude-code-wire-format.md)
  — what Claude Code actually puts on the wire

`PLAN.md` and `docs/knowledge/log.md` are in French, and stay that way: they
are a working journal, not a rewritten narrative.

## Acknowledgments

- [Qwen](https://github.com/QwenLM) — the model architecture
- [mlx-swift](https://github.com/ml-explore/mlx-swift) and
  [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — Apple's MLX for Swift
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — the Python reference
  implementation the parity fixtures are generated from
- [swift-transformers](https://github.com/huggingface/swift-transformers) — tokenizers
- [hummingbird](https://github.com/hummingbird-project/hummingbird) — the HTTP server
- [swift-mlx-profiler](https://github.com/VincentGourbin/swift-mlx-profiler) — Chrome-trace profiling

## License

MIT License — see [LICENSE](LICENSE).
