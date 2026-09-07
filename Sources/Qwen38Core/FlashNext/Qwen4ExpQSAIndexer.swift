import MLX
import MLXNN

/// Learned projection at the entrance of a Flash-Next QSA indexer.
///
/// This module intentionally stops before RoPE and block pooling. Those two
/// operations depend on the decoder's multimodal position clock and are kept in
/// the attention layer. Keeping the projection separate also lets us load and
/// parity-check one full-attention layer without instantiating the 512 experts
/// or the 51B-parameter n-gram table.
public final class Qwen4ExpQSAIndexer: Module {
    public let nHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rope: Qwen4ExpMRoPE

    @ModuleInfo(key: "index_qk_proj") public var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") public var qLayerNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") public var kLayerNorm: Qwen4ExpRMSNorm

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        self.nHeads = configuration.indexerNHeads
        self.kvHeads = configuration.indexerKVHeads
        self.headDim = configuration.indexerHeadDim
        self.rope = Qwen4ExpMRoPE(
            rotaryDim: 64,
            base: 10_000_000,
            mropeSections: [11, 11, 10])
        precondition(kvHeads == 1, "Le QSA actuel attend une tête KV indexeur")

        _indexQKProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: (nHeads + kvHeads) * headDim,
            quantization: quantization)
        _qLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: rmsNormEps)
        _kLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: rmsNormEps)
        super.init()
    }

    /// Project hidden states and return normalized queries plus raw keys.
    ///
    /// - hiddenStates: `[B,S,H]`
    /// - queries: `[B,nHeads,S,D]`
    /// - rawKeys: `[B,S,D]`; the singleton indexer KV-head dimension is
    ///   removed to match `QSAKVCache` and mlx-vlm.
    public func callAsFunction(_ hiddenStates: MLXArray) -> (queries: MLXArray, rawKeys: MLXArray) {
        precondition(hiddenStates.ndim == 3, "QSA hidden states attendus en [B,S,H]")
        let batch = hiddenStates.dim(0)
        let sequence = hiddenStates.dim(1)
        let projected = indexQKProj(hiddenStates).reshaped(
            [batch, sequence, nHeads + kvHeads, headDim])
        return split(projected)
    }

    /// Split a projected `[B,S,(nHeads+kvHeads),D]` tensor.
    ///
    /// Exposed for parity fixtures that already contain the projection output
    /// and should not invoke a quantized matrix multiplication twice.
    public func split(_ projected: MLXArray) -> (queries: MLXArray, rawKeys: MLXArray) {
        precondition(projected.ndim == 4)
        precondition(projected.dim(2) == nHeads + kvHeads)
        precondition(projected.dim(3) == headDim)

        let querySlice = projected[.ellipsis, 0..<nHeads, 0...]
        let keySlice = projected[.ellipsis, nHeads..<(nHeads + kvHeads), 0...]
        let queries = qLayerNorm(querySlice).transposed(0, 2, 1, 3)
        let rawKeys = keySlice.squeezed(axis: 2)
        return (queries, rawKeys)
    }

    /// Normalize pooled block keys after averaging their token keys.
    public func normalizePooledKeys(_ pooledKeys: MLXArray) -> MLXArray {
        precondition(pooledKeys.ndim == 3, "QSA blocs attendus en [B,Blocks,D]")
        return kLayerNorm(pooledKeys).expandedDimensions(axis: 1)
    }

    /// Average complete groups of `compressRatio` raw keys in float32.
    ///
    /// The incomplete tail is intentionally excluded: it remains directly
    /// visible through the causal tail of `Qwen4ExpQSAMask`.
    public func poolCompleteKeys(
        _ rawKeys: MLXArray,
        compressRatio: Int
    ) -> MLXArray {
        precondition(rawKeys.ndim == 3, "QSA clés attendues en [B,S,D]")
        precondition(compressRatio > 0)
        let completeTokenCount = (rawKeys.dim(1) / compressRatio) * compressRatio
        precondition(completeTokenCount > 0, "Aucun bloc QSA complet")
        let complete = rawKeys[.ellipsis, 0..<completeTokenCount, 0...]
        return complete.asType(.float32)
            .reshaped([rawKeys.dim(0), completeTokenCount / compressRatio, compressRatio, rawKeys.dim(2)])
            .mean(axis: 2)
            .asType(rawKeys.dtype)
    }

    /// Pool and normalize keys in the shape consumed by the QSA score path.
    public func pooledKeys(
        _ rawKeys: MLXArray,
        compressRatio: Int
    ) -> MLXArray {
        normalizePooledKeys(poolCompleteKeys(rawKeys, compressRatio: compressRatio))
    }

    /// Assemble the learned indexer path through its first sparse mask.
    ///
    /// The cache receives the raw `[B,S,D]` keys before pooling so that a
    /// later turn can reuse the same indexer history as the main KV cache.
    /// `nil` means there is not yet one complete compressed block and the
    /// caller must use ordinary causal attention.
    public func makeMask(
        hiddenStates: MLXArray,
        positionIDs: MLXArray,
        cache: Qwen4ExpQSAKVCache?,
        compressRatio: Int,
        budget: Int
    ) -> MLXArray? {
        let projected = indexQKProj(hiddenStates).reshaped(
            [hiddenStates.dim(0), hiddenStates.dim(1), nHeads + kvHeads, headDim])
        return makeMask(
            fromProjected: projected,
            positionIDs: positionIDs,
            cache: cache,
            compressRatio: compressRatio,
            budget: budget)
    }

    /// Same selector starting from a saved projection tensor.
    ///
    /// This seam is used by the Python parity probe so that quantized linear
    /// loading and the discrete QSA algorithm can be diagnosed independently.
    public func makeMask(
        fromProjected projected: MLXArray,
        positionIDs: MLXArray,
        cache: Qwen4ExpQSAKVCache?,
        compressRatio: Int,
        budget: Int
    ) -> MLXArray? {
        precondition(projected.ndim == 4)
        precondition(projected.dim(2) == nHeads + kvHeads && projected.dim(3) == headDim)
        let splitProjection = split(projected)
        let queries = rope.apply(splitProjection.queries, positionIDs: positionIDs)

        let rawKeys: MLXArray
        let fullPositions: MLXArray
        let pastOffset = cache?.offset ?? 0
        if let cache {
            cache.updateIndexer(keys: splitProjection.rawKeys, positions: positionIDs)
            rawKeys = cache.indexerKeysView!
            fullPositions = cache.indexerPositionsView!
        } else {
            rawKeys = splitProjection.rawKeys
            fullPositions = positionIDs
        }

        let maxCompleteBlocks = rawKeys.dim(1) / compressRatio
        let blockTopK = budget / compressRatio
        // Match mlx-vlm: until there are more complete blocks than the
        // budget can retain, the caller must keep ordinary causal attention.
        // A sparse mask here would add selector overhead without removing any
        // keys and would also differ from the reference at the boundary.
        guard maxCompleteBlocks > blockTopK else {
            return nil
        }
        let pooled = pooledKeys(rawKeys, compressRatio: compressRatio)
        let completeTokenCount = maxCompleteBlocks * compressRatio
        let blockPositions = fullPositions[.ellipsis, .stride(
            from: 0, to: completeTokenCount, by: compressRatio)]
        let rotatedPooled = rope.apply(pooled, positionIDs: blockPositions)

        return Qwen4ExpQSAMask.tokenMask(
            query: queries,
            pooledKeys: rotatedPooled,
            keyLength: rawKeys.dim(1),
            budget: budget,
            compressRatio: compressRatio,
            offset: pastOffset)
    }
}
