import Foundation
import MLX
import MLXLMCommon

/// The two cache families used by the Flash-Next decoder.
public enum Qwen4ExpAttentionKind: String, Sendable, Equatable {
    case linearAttention = "linear_attention"
    case sparseAttention = "full_attention"
}

/// Cache selection is a property of the layer schedule, not of the prompt.
/// Keeping it explicit prevents a QSA layer from accidentally receiving a
/// recurrent Mamba/GDN cache during a later multi-turn call.
public struct Qwen4ExpCachePlan: Sendable, Equatable {
    public let attentionKinds: [Qwen4ExpAttentionKind]

    public init(configuration: Qwen4ExpTextConfiguration) {
        attentionKinds = configuration.layerTypes.map {
            $0 == .linearAttention ? .linearAttention : .sparseAttention
        }
    }

    public var count: Int { attentionKinds.count }

    public func kind(at layer: Int) -> Qwen4ExpAttentionKind {
        precondition(attentionKinds.indices.contains(layer), "Flash-Next layer hors limites")
        return attentionKinds[layer]
    }

    public var linearLayerIndices: [Int] {
        attentionKinds.indices.filter { attentionKinds[$0] == .linearAttention }
    }

    public var sparseLayerIndices: [Int] {
        attentionKinds.indices.filter { attentionKinds[$0] == .sparseAttention }
    }
}

/// The storage contract for a QSA layer's indexer side-channel.
///
/// The dense main K/V path is intentionally retained for the first correct
/// implementation. Later QSA selection can consume `indexerKeys` and
/// `indexerPositions` without changing the per-layer cache API.
public final class Qwen4ExpQSAKVCache: KVCache, @unchecked Sendable {
    public private(set) var offset: Int = 0
    public var metaState: [String] = [""]
    public let blockSize: Int
    public let budget: Int
    public let compressRatio: Int

    private let mainCache = KVCacheSimple()
    private var indexerKeys: MLXArray?
    private var indexerPositions: MLXArray?

    public init(blockSize: Int = 4, budget: Int = 2_048, compressRatio: Int = 4) {
        precondition(blockSize > 0 && budget > 0 && compressRatio > 0)
        self.blockSize = blockSize
        self.budget = budget
        self.compressRatio = compressRatio
    }

    public var hasIndexerState: Bool {
        indexerKeys != nil && indexerPositions != nil
    }

    public var indexerTokenCount: Int {
        indexerKeys?.dim(1) ?? 0
    }

    public var indexerKeysView: MLXArray? { indexerKeys }
    public var indexerPositionsView: MLXArray? { indexerPositions }

    public func innerState() -> [MLXArray] {
        mainCache.innerState() + [indexerKeys, indexerPositions].compactMap { $0 }
    }

    public var maxSize: Int? { nil }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = mainCache.update(keys: keys, values: values)
        offset = mainCache.offset
        return result
    }

    /// Append the raw indexer keys and multimodal positions for the same
    /// timeline as the main K/V update. Positions may be `[B,S]` or `[B,A,S]`;
    /// in both cases sequence time is the final axis. The single indexer KV
    /// head is stored in canonical `[B,S,D]` form, matching mlx-vlm after its
    /// `squeeze(2)`.
    public func updateIndexer(keys: MLXArray, positions: MLXArray) {
        let canonicalKeys: MLXArray
        if keys.ndim == 3 {
            canonicalKeys = keys
        } else {
            precondition(
                keys.ndim == 4 && keys.dim(1) == 1,
                "QSA indexer keys attendus en [B,S,D] ou [B,1,S,D]")
            canonicalKeys = keys.squeezed(axis: 1)
        }
        precondition(positions.ndim == 2 || positions.ndim == 3)
        precondition(canonicalKeys.dim(1) == positions.dim(-1))

        if let current = indexerKeys {
            precondition(current.dim(0) == canonicalKeys.dim(0))
            precondition(current.dim(2) == canonicalKeys.dim(2))
            precondition(indexerPositions?.ndim == positions.ndim)
            indexerKeys = concatenated([current, canonicalKeys], axis: 1)
            indexerPositions = concatenated([indexerPositions!, positions], axis: positions.ndim - 1)
        } else {
            indexerKeys = canonicalKeys
            indexerPositions = positions
        }
    }

    public var state: [MLXArray] {
        get {
            mainCache.state + [indexerKeys, indexerPositions].compactMap { $0 }
        }
        set {
            guard newValue.count == 2 || newValue.count == 4 else {
                fatalError("Qwen4ExpQSAKVCache state doit contenir 2 ou 4 tableaux")
            }
            mainCache.state = Array(newValue.prefix(2))
            offset = mainCache.offset
            if newValue.count == 4 {
                indexerKeys = newValue[2]
                indexerPositions = newValue[3]
            } else {
                indexerKeys = nil
                indexerPositions = nil
            }
        }
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = mainCache.trim(n)
        offset = mainCache.offset
        // PM4.2 (P-MTP suite) bug fix: `trim` removes the `n` most recently
        // appended positions (matching `mainCache.trim`, which just shrinks
        // `offset` — the newest K/V rows fall out of the visible range).
        // The indexer side-channel must drop the same *tail*, not the head:
        // the previous `[indexerTrimmed...]` kept the newest rows and
        // dropped the oldest ones, silently reordering the indexer's
        // timeline on every rollback. No prior caller exercised a nonzero
        // rollback against nonzero indexer state, so this went unnoticed —
        // see `qwen4ExpQSAKVCacheTracksIndexerState` (only checked counts).
        let indexerTrimmed = min(trimmed, indexerTokenCount)
        if indexerTrimmed > 0 {
            let keep = indexerTokenCount - indexerTrimmed
            indexerKeys = indexerKeys?[.ellipsis, ..<keep, 0...]
            indexerPositions = indexerPositions?[.ellipsis, ..<keep]
        }
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        mainCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        let result = Qwen4ExpQSAKVCache(
            blockSize: blockSize, budget: budget, compressRatio: compressRatio)
        let currentState = state
        if !currentState.isEmpty {
            result.state = currentState.map { $0[.ellipsis] }
        }
        return result
    }
}
