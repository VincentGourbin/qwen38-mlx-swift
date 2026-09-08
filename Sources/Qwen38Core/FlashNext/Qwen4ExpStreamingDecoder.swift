import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// Controls whether decoder weights are released after each layer or retained
/// for subsequent autoregressive tokens.  Resident mode is experimental: the
/// Flash-Next checkpoint is large and must be enabled explicitly by a caller
/// that has measured its available memory budget.
public enum Qwen4ExpLayerLoadingMode: String, Sendable, Equatable {
    case streamed
    case resident
}

/// Immutable-at-the-call-site copy of every cache currently owned by the
/// streaming decoder.  The values are copied through `KVCache.copy()` so a
/// speculative forward cannot mutate the snapshot through a shared cache
/// object.  The type is unchecked-sendable for the same reason as the
/// decoder: MLX arrays are intentionally confined to the owning runtime.
public final class Qwen4ExpStreamingDecoderSnapshot: @unchecked Sendable {
    fileprivate let caches: [Int: any KVCache]

    fileprivate init(caches: [Int: any KVCache]) {
        self.caches = caches
    }
}

/// A layer-at-a-time Flash-Next decoder used to stay below the 96 GB machine
/// budget while the full resident model is being assembled.
public final class Qwen4ExpStreamingDecoder: @unchecked Sendable {
    public let configuration: Qwen4ExpTextConfiguration
    public let directory: URL
    public let quantization: Qwen4ExpQuantizationSpec?
    public let layerLoadingMode: Qwen4ExpLayerLoadingMode
    /// Number of resident layers between MLX evaluation checkpoints.
    ///
    /// Measured 2026-09-05: batching multiple layers before `eval()` does NOT
    /// reduce synchronization overhead — it does the opposite. Deferring 8
    /// layers into one lazy graph produced highly variable, sometimes
    /// catastrophic decode stalls (observed 841s-1436s for 7 decode tokens),
    /// while evaluating every single layer (interval 1) was both fast and
    /// stable (28s for the same 7 tokens, ~30-50x faster). Keep this at 1
    /// unless a future measurement on a different checkpoint/machine shows
    /// otherwise; see docs/knowledge/log.md 2026-09-05.
    public let residentEvaluationInterval: Int
    /// Opt-in per-layer profiling (P0-c). When `false` (the default),
    /// `forward` skips the `profiler.start`/`.end("Flash couche N")` pair
    /// and the per-layer n-gram cache bookkeeping (`observeNGramCache`) for
    /// every decoder layer. Each `start`/`end` pair costs ~4.7 ms (IOKit GPU
    /// sample + `rusage`, see docs/knowledge/log.md "P0 rejoué en Release"),
    /// i.e. ~225 ms/token spread over 48 layers when left on unconditionally
    /// — pure profiler overhead, not decoder cost. The `Prefill`/`Generation`
    /// phases and session metadata (model, prompt tokens, layer visits/load
    /// time…) are unaffected: they live in the generators, not here.
    public let profileLayers: Bool
    /// P2-code (d) → P1 : in resident mode, dispatch every intermediate layer
    /// with `asyncEval` (GPU work starts immediately, the host goes on
    /// building the next layer) and block only on the last layer of the
    /// visit. Measured on the synthetic bench (Release): −13 to −17 % per
    /// layer and GPU busy 82 % (ioreg) instead of ~0 %. Off by default until
    /// P1 has measured it on the real checkpoint; `residentEvaluationInterval`
    /// keeps its meaning (blocking `eval` checkpoints) when this is false.
    public let residentAsyncEval: Bool
    /// P2-mem-a: read resident tensors with `pread` behind `fcntl(F_NOCACHE,
    /// 1)` (`Qwen4ExpUncachedTensorReader`) instead of `loadArraysAndMetadata`,
    /// so the 50-80 GB read from the Lexar during a resident load never fill
    /// the kernel's file cache (measured +30-40 GB otherwise; see
    /// docs/knowledge/log.md "H6 : deux tentatives", 2026-09-08 soir, and the
    /// P2-mem-a entry). Defaults to `true`: validated bit-exact against the
    /// old path with unchanged peak MLX/RSS.
    public let uncachedIO: Bool

    private let checkpointIndex: Qwen4ExpCheckpointLayerIndex
    private var caches: [Int: any KVCache] = [:]
    private var residentLayers: [Int: Qwen4ExpLoadedDecoderLayer] = [:]
    private var ngramCacheTotals = Qwen4ExpNGramCacheStats()
    private var ngramCacheEntriesByLayer: [Int: Int] = [:]
    private var residentNGramCacheSnapshots: [Int: Qwen4ExpNGramCacheStats] = [:]

    public init(
        directory: URL,
        layerLoadingMode: Qwen4ExpLayerLoadingMode = .streamed,
        residentEvaluationInterval: Int = 1,
        profileLayers: Bool = false,
        residentAsyncEval: Bool = false,
        uncachedIO: Bool = true
    ) throws {
        precondition(residentEvaluationInterval > 0)
        self.directory = directory
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        self.configuration = configuration.textConfiguration
        self.quantization = Qwen4ExpQuantizationSpec(configuration.quantization)
        self.layerLoadingMode = layerLoadingMode
        self.residentEvaluationInterval = residentEvaluationInterval
        self.profileLayers = profileLayers
        self.residentAsyncEval = residentAsyncEval
        self.uncachedIO = uncachedIO
        self.checkpointIndex = try Qwen4ExpCheckpointLayerIndex(directory: directory)
    }

    /// Run selected source layers while keeping the recurrent/QSA caches alive.
    ///
    /// `hiddenStates` is already the four-stream `[B,S,4H]` representation.
    /// This seam deliberately does not own token embeddings or the final head;
    /// it is the memory-safe decoder core that those runtime pieces will call.
    public func forward(
        _ hiddenStates: MLXArray,
        inputIDs: MLXArray,
        layerIndices: [Int],
        positionIDs: MLXArray? = nil,
        materializeLayers: Bool = true,
        onLayerVisited: (@Sendable (Int) -> Void)? = nil
    ) throws -> (output: MLXArray, reports: [Qwen4ExpStreamingLayerReport]) {
        precondition(hiddenStates.ndim == 3)
        precondition(hiddenStates.dim(-1) == configuration.hiddenSize * configuration.hcCount)
        precondition(inputIDs.ndim == 2)
        precondition(hiddenStates.dim(0) == inputIDs.dim(0))
        precondition(hiddenStates.dim(1) == inputIDs.dim(1))

        var hidden = hiddenStates
        var reports: [Qwen4ExpStreamingLayerReport] = []
        let synchronizeEachLayer = layerLoadingMode == .streamed
        for (visitIndex, layerIndex) in layerIndices.enumerated() {
            precondition(configuration.layerTypes.indices.contains(layerIndex))
            let profiler = MLXProfiler.shared
            let profileName = "Flash couche \(layerIndex)"
            if profileLayers {
                profiler.start(profileName)
            }
            defer {
                if profileLayers {
                    profiler.end(profileName)
                }
            }
            // In streamed mode the previous layer is released before the next
            // shard group is opened, so clear its now-unused allocations. In
            // resident mode every layer stays live and clearing the allocator
            // here only adds a device-wide synchronization once per layer.
            if synchronizeEachLayer {
                Memory.clearCache()
            }
            let loaded: Qwen4ExpLoadedDecoderLayer
            let loadDuration: Duration
            if layerLoadingMode == .resident,
               let existing = residentLayers[layerIndex] {
                loaded = existing
                loadDuration = .zero
            } else {
                let start = ContinuousClock.now
                loaded = try Qwen4ExpCheckpointLayerLoader.load(
                    layerIndex,
                    from: directory,
                    index: checkpointIndex,
                    materialize: materializeLayers,
                    uncachedIO: uncachedIO)
                loadDuration = ContinuousClock.now - start
                if layerLoadingMode == .resident {
                    residentLayers[layerIndex] = loaded
                }
            }

            let cache: any KVCache
            if let existing = caches[layerIndex] {
                cache = existing
            } else {
                let created = makeCache(for: layerIndex)
                caches[layerIndex] = created
                cache = created
            }

            let attentionMask: MLXArray?
            if configuration.layerTypes[layerIndex] != .fullAttention {
                // GDN's recurrence is causal by construction.  Its optional
                // mask has a different [B,S] contract and is reserved for
                // padded/ragged batches.
                attentionMask = nil
            } else if inputIDs.dim(1) == 1 {
                // P2-code (e): single-token decode has exactly one query
                // position (`cache.offset`), and every cached key position
                // is <= cache.offset < cache.offset + 1 — the causal mask
                // built by `Qwen4ExpQSAAttention.causalMask` is therefore
                // provably all-true for every step, and an all-true boolean
                // mask is numerically the same as no mask at all to SDPA.
                // Skip constructing it: `causalMask` allocates a fresh
                // `Int32` range array of length `cache.offset + 1` — growing
                // with every decoded token — purely to compare it against
                // itself and get back "true" (see docs/knowledge/log.md
                // P2-code (a)/(e)). Multi-token calls (prefill, MTP
                // verification) still build the real mask below.
                attentionMask = nil
            } else {
                attentionMask = Qwen4ExpQSAAttention.causalMask(
                    batch: inputIDs.dim(0), queryLength: inputIDs.dim(1),
                    keyLength: cache.offset + inputIDs.dim(1), offset: cache.offset)
            }

            let forwardStart = ContinuousClock.now
            let output = loaded.layer(
                hidden,
                inputIDs: inputIDs,
            mask: attentionMask,
            cache: cache,
            positionIDs: positionIDs)
            if profileLayers, let currentNGramStats = loaded.layer.ngramCacheStats() {
                observeNGramCache(layerIndex: layerIndex, current: currentNGramStats)
            }
            // Streamed mode must detach the next layer from the previous
            // module before that module is released. Resident mode keeps all
            // modules alive and checkpoints the lazy graph periodically. The
            // final layer is always materialized for a stable result.
            let shouldEvaluate = synchronizeEachLayer ||
                (layerLoadingMode == .resident &&
                 ((visitIndex + 1) % residentEvaluationInterval == 0 ||
                  visitIndex == layerIndices.count - 1))
            if shouldEvaluate {
                eval(output)
            } else if layerLoadingMode == .resident, residentAsyncEval {
                asyncEval(output)
            }
            let forwardDuration = ContinuousClock.now - forwardStart
            hidden = output

            reports.append(
                Qwen4ExpStreamingLayerReport(
                    layerIndex: layerIndex,
                    loadedTensorCount: loaded.tensorCount,
                    loadedShardCount: loaded.shardCount,
                    materializedBytes: loaded.materializedBytes,
                    loadDuration: loadDuration.seconds,
                    forwardDuration: forwardDuration.seconds,
                    outputShape: output.shape))
            onLayerVisited?(layerIndex)
            // `loaded` goes out of scope at the end of this iteration. The
            // cache is intentionally retained for the next conversation turn;
            // the next iteration clears only allocations no longer referenced.
        }
        return (hidden, reports)
    }

    public func resetCaches() {
        caches.removeAll(keepingCapacity: true)
        Memory.clearCache()
    }

    public func ngramCacheStats() -> Qwen4ExpNGramCacheStats {
        var result = ngramCacheTotals
        result.entries = ngramCacheEntriesByLayer.values.reduce(0, +)
        return result
    }

    public func resetNGramCacheStats() {
        ngramCacheTotals = Qwen4ExpNGramCacheStats()
        ngramCacheEntriesByLayer.removeAll(keepingCapacity: true)
        // The snapshots are baselines for delta accounting in resident mode.
        // They must be reset together with the public counters; otherwise the
        // first forward of the next generation can be silently under-counted.
        residentNGramCacheSnapshots.removeAll(keepingCapacity: true)
    }

    /// Capture all hybrid attention state before a speculative verification
    /// forward.  QSA carries both its dense K/V history and its indexer
    /// side-channel; `copy()` preserves both.  GDN and PLE caches are copied
    /// through their public `KVCache` surface as well.
    public func snapshot() -> Qwen4ExpStreamingDecoderSnapshot {
        Qwen4ExpStreamingDecoderSnapshot(
            caches: caches.mapValues { $0.copy() })
    }

    /// Restore a snapshot after a rejected speculative suffix.  Replacing
    /// cache objects, instead of assigning individual arrays into the live
    /// objects, prevents a partially restored layer from being observed by a
    /// later layer in the same round.
    public func restore(_ snapshot: Qwen4ExpStreamingDecoderSnapshot) {
        caches = snapshot.caches.mapValues { $0.copy() }
        Memory.clearCache()
    }

    /// Release the optional resident decoder weights while keeping the decoder
    /// usable in streamed mode for subsequent calls.
    public func unloadResidentLayers() {
        residentLayers.removeAll(keepingCapacity: true)
        Memory.clearCache()
    }

    private func observeNGramCache(layerIndex: Int, current: Qwen4ExpNGramCacheStats) {
        if layerLoadingMode == .resident {
            let previous = residentNGramCacheSnapshots[layerIndex] ?? Qwen4ExpNGramCacheStats()
            ngramCacheTotals.hits += max(0, current.hits - previous.hits)
            ngramCacheTotals.misses += max(0, current.misses - previous.misses)
            residentNGramCacheSnapshots[layerIndex] = current
        } else {
            // Streamed mode constructs a fresh PLE reader for each visit.
            ngramCacheTotals.hits += current.hits
            ngramCacheTotals.misses += current.misses
        }
        ngramCacheEntriesByLayer[layerIndex] = max(
            ngramCacheEntriesByLayer[layerIndex] ?? 0, current.entries)
    }

    private func makeCache(for layerIndex: Int) -> any KVCache {
        if configuration.layerTypes[layerIndex] == .fullAttention {
            return Qwen4ExpQSAKVCache(
                budget: configuration.indexerBudget,
                compressRatio: configuration.indexerCompressRatio)
        }
        if configuration.pleLayerIDs.contains(layerIndex + 1) {
            return ArraysCache(size: 4)
        }
        return MambaCache()
    }
}

public struct Qwen4ExpStreamingLayerReport: Sendable, Equatable {
    public let layerIndex: Int
    public let loadedTensorCount: Int
    public let loadedShardCount: Int
    public let materializedBytes: Int64
    public let loadDuration: Double
    public let forwardDuration: Double
    public let outputShape: [Int]

    public init(
        layerIndex: Int,
        loadedTensorCount: Int,
        loadedShardCount: Int,
        materializedBytes: Int64,
        loadDuration: Double,
        forwardDuration: Double,
        outputShape: [Int]
    ) {
        self.layerIndex = layerIndex
        self.loadedTensorCount = loadedTensorCount
        self.loadedShardCount = loadedShardCount
        self.materializedBytes = materializedBytes
        self.loadDuration = loadDuration
        self.forwardDuration = forwardDuration
        self.outputShape = outputShape
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
