# Architecture

Five modules, declared in `Package.swift`. The split is not cosmetic: it is
what lets the agent loop and the server be unit-tested without a 100 GB
checkpoint, a GPU, or a network.

```
Qwen38Core ──┬── Qwen38Server ──┬── Qwen38CLI        (qwen38)
             │                  └── Qwen38BenchUI    (qwen38-bench-ui)
             └── Qwen38Agent ───────┘
```

## `Qwen38Core` — the model

The library. Everything that touches weights, MLX arrays, or the tokenizer.

`FlashNext/` is the port itself — the `qwen4_exp` architecture, roughly sixty
files. The pieces worth knowing about:

- **Streaming decoder** (`Qwen4ExpStreamingDecoder`, `Qwen4ExpStreamingTextModel`,
  `Qwen4ExpCheckpointLayerLoader`): the 48 decoder layers are loaded from the
  checkpoint and kept resident, with an async-eval interval so the GPU is not
  stalled once per layer. This is what makes a 125 B MoE run at all in 96 GB.
- **Attention and recurrence**: `Qwen4ExpQSAAttention` / `Qwen4ExpQSAIndexer` /
  `Qwen4ExpQSAMask` for the sparse attention, `Qwen4ExpGatedDeltaNet` /
  `Qwen4ExpGatedDeltaStates` for the recurrent branch,
  `Qwen4ExpHyperConnection` for the four-stream mixing.
- **`Qwen4ExpSparseMoE`**: 512 experts, top-10 routing, on top of
  `SwitchGLU` from the vendored `mlx-swift-lm`.
- **`Qwen4ExpPLE`**: the n-gram per-layer-embedding table — a random-access
  read per token into shards on disk. It reads rows with concurrent `pread`
  by default (`QWEN38_NGRAM_MMAP=1` restores the old `mmap` path).
- **`Qwen4ExpFusion`**: numbered fusion levels `F1…F9`. The production default
  is `F7` (`f7GatedBranchDtype`), which casts the GDN and QSA outputs back to
  the input dtype so the rest of the layer, MoE included, does not silently
  run in fp32.
- **Vision**: `Qwen4ExpVisionEncoder`, `Qwen4ExpImageProcessor`,
  `Qwen4ExpInputMerger`, `Qwen4ExpMRoPE`.
- **Parity harnesses**: `Qwen4Exp*Parity.swift`, one per seam — see
  [`parity-method.md`](parity-method.md).
- **Benches and probes**: `Qwen4ExpLayerBench` (synthetic single layer, no
  checkpoint), `Qwen4ExpDecodeBench`, `Qwen4ExpBatchProbe`,
  `Qwen4ExpDtypeAudit`.

`MTP/` holds multi-token prediction — the upstream drafter path (M1) and the
local persistent pipeline (M2), with GDN state snapshot/rollback.

At the top level, `Qwen38Runtime` is the engine façade the CLI, GUI and server
all call; `Qwen38ModelCache` resolves the models root (`$QWEN38_MODELS_DIR`,
else `~/models`, else `~/Library/Caches/models`); `Qwen38Download` fetches a
Hugging Face repo over plain HTTPS; `Qwen38ToolCalling` translates between the
checkpoint's XML-ish `<tool_call>` grammar and OpenAI's `tools` / `tool_calls`
JSON; `Qwen38VisibleText` separates reasoning from answer.

## `Qwen38Server` — the OpenAI-compatible API

`Qwen38Server.swift` is a Hummingbird service exposing four routes —
`GET /healthz`, `GET /v1/models`, `GET /metrics`, `POST /v1/chat/completions` —
with optional Bearer auth, SSE streaming, and a `: loading` heartbeat so a
client does not time out during the first cold load.

One model is resident per process, and requests are serialised. On top of that:
a per-conversation LRU cache keyed by `conversation_id`, an implicit prefix
cache for clients that just resend the whole history, and a suffix-by-diff
path so a continuation only prefills what is actually new.

`Qwen38BatchScheduling.swift` is the opt-in batch coordinator (`--batch-size`):
it groups cold requests of similar prompt length inside a short window and
decodes them in one MLX step. Anything that would touch a cache, carry an
image, declare tools, or exceed `--batch-max-prompt-tokens` keeps the
single-sequence path.

## `Qwen38CLI` — the `qwen38` binary

One `ArgumentParser` command tree, about forty subcommands. Three groups:

- **usable**: `info`, `download`, `generate`, `serve`;
- **probes** (`flash-*-probe`, `mtp-probe`, …) — load a slice of the model and
  report what happened, used to bring the port up layer by layer;
- **parity and bench** (`flash-*-parity`, `flash-layer-bench`,
  `flash-decode-bench`, `op-overhead-probe`, `ngram-io-probe`,
  `conversation-benchmark`) — the measurement tooling behind `BENCHMARKS.md`.

The probe and bench commands are kept in the shipped binary on purpose: every
number in `BENCHMARKS.md` and `docs/knowledge/log.md` is reproducible from
this one executable.

## `Qwen38Agent` — the agent loop, headless

Pure Foundation. No AppKit, no SwiftUI, no MLX, no network — which is exactly
why it is testable without a checkpoint.

- `AgentToolCatalog` declares the four tools offered to the model:
  `list_files`, `read_file`, `grep`, `final_answer`. There is deliberately no
  shell tool; `grep` is reimplemented in Swift rather than shelling out.
- `AgentSandbox` resolves every path and rejects anything outside the chosen
  root, matching on a `/` boundary so `…/project-evil` cannot pass for
  `…/project`.
- `AgentToolExecutor` never throws at the loop: a bad path or a missing
  argument comes back to the model as an `ERREUR : …` string it can correct.
- `AgentTruncation` caps what is fed back (6000 chars per tool output, 200
  lines per `read_file`, 60 grep hits, 80 directory entries).
- `AgentLoopEngine` holds the step budget and the stop conditions;
  `AgentRunGate` prevents two loops running at once; `AgentWireFormat` builds
  the OpenAI request body.

The HTTP client that actually talks to the server lives in the GUI, not here.

## `Qwen38BenchUI` — the `qwen38-bench-ui` app

A single-window SwiftUI app with three tabs.

- **Chat** — talk to the resident model, with a metrics panel beside it
  (tok/s, TTFT, load duration, MTP acceptance rate, per-turn history, active
  and peak memory).
- **Serveur** — start and stop the LAN server, pick a model from the
  discovered catalogue, watch live sessions and counters.
- **Agent** — `AgentPanelView`, which drives `Qwen38Agent` against the local
  server over plain HTTP, exactly like any third-party client would. It does
  not call `Qwen38Runtime` directly; if the server is not running, the panel
  says so and offers to start it.

## Vendored and pinned dependencies

`Vendor/mlx-swift-lm` is a pinned local checkout of an upstream PR carrying
Qwen MTP support, and is treated as read-only: the P7 and P10 investigations
stop at its boundary. `mlx-swift` is pinned to an **exact** version because
patch releases have changed APIs before.
