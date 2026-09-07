import MLX

/// Pure-MLX part of the QSA selector.
///
/// The learned indexer projection is deliberately kept outside this type. This
/// utility starts at its two observable tensors (query heads and pooled block
/// keys), which makes the discrete selection and its parity tests independent
/// from the 113 GB checkpoint. It is also the safe dense fallback used while a
/// Metal gather kernel is not yet justified by profiling.
public enum Qwen4ExpQSAMask {
    /// Score compressed key blocks with the QSA rule.
    ///
    /// - query: `[B, Hq, Q, D]`
    /// - pooledKeys: `[B, 1, Blocks, D]`
    /// - returns: `[B, Q, Blocks]`, with the positive contribution summed over
    ///   query heads as in the reference implementation.
    public static func blockScores(query: MLXArray, pooledKeys: MLXArray) -> MLXArray {
        precondition(query.ndim == 4, "QSA queries attendus en [B,H,Q,D]")
        precondition(pooledKeys.ndim == 4, "QSA clés compressées attendues en [B,1,B,D]")
        precondition(query.dim(0) == pooledKeys.dim(0))
        precondition(query.dim(3) == pooledKeys.dim(3))

        let products = query.asType(.float32).matmul(
            pooledKeys.asType(.float32).transposed(0, 1, 3, 2))
        return maximum(products, MLXArray(Float(0))).sum(axis: 1)
            / MLXArray(Float(query.dim(3)).squareRoot())
    }

    /// Build a causal sparse mask from block scores.
    ///
    /// The result is a boolean mask `[B, 1, Q, K]` suitable for MLX SDPA.
    /// Complete blocks are selected by score; the incomplete tail is always
    /// retained. If a query sees at most `budget / compressRatio` complete
    /// blocks, the result is exactly the ordinary causal mask.
    public static func tokenMask(
        query: MLXArray,
        pooledKeys: MLXArray,
        keyLength: Int,
        budget: Int,
        compressRatio: Int,
        offset: Int = 0
    ) -> MLXArray {
        precondition(keyLength > 0 && budget > 0 && compressRatio > 0)
        let scores = blockScores(query: query, pooledKeys: pooledKeys)
        let batch = query.dim(0)
        let queryLength = query.dim(2)
        let blockCount = scores.dim(2)
        let blockTopK = budget / compressRatio
        precondition(blockTopK > 0)
        precondition(blockCount == (keyLength / compressRatio))

        let keyIndices = MLXArray(Int32(0) ..< Int32(keyLength))
        let queryEnds = MLXArray(Int32(offset) ..< Int32(offset + queryLength)) + 1
        let blockIndices = MLXArray(Int32(0) ..< Int32(blockCount))
        // `/` on integer MLX arrays follows the numeric division path on
        // this Swift binding.  Floor explicitly: block visibility is a
        // discrete count and 9 visible tokens must mean exactly 2 complete
        // blocks, not 2.25 blocks.
        let completeCounts = minimum(
            floor(queryEnds[.newAxis, .ellipsis].asType(.float32) / Float(compressRatio))
                .asType(.int32),
            MLXArray(Int32(blockCount)))

        let visibleBlocks = blockIndices[.newAxis, .newAxis, .ellipsis]
            .< completeCounts[.ellipsis, .newAxis]
        let maskedScores = `where`(
            visibleBlocks,
            scores,
            MLXArray(-Float.infinity))

        let sparse = blockTopK < blockCount
        let selectedTokens: MLXArray
        if sparse {
            let partitioned = argPartition(
                maskedScores, kth: blockCount - blockTopK, axis: -1)
            let selectedBlocks = partitioned[.ellipsis, (blockCount - blockTopK)..<blockCount]
            let tokenOffsets = MLXArray(Int32(0) ..< Int32(compressRatio))
            let selectedTokenIndices = (
                selectedBlocks[.ellipsis, .newAxis] * Int32(compressRatio)
                    + tokenOffsets[.newAxis, .newAxis, .newAxis, .ellipsis]
            ).reshaped([batch, queryLength, blockTopK * compressRatio])
            let valid = selectedTokenIndices .< MLXArray(Int32(keyLength))
                .&& selectedTokenIndices .< queryEnds[.ellipsis, .newAxis]
            var selected = MLXArray.zeros(
                [batch, queryLength, keyLength + 1], dtype: .bool)
            selected = putAlong(
                selected,
                selectedTokenIndices,
                values: valid,
                axis: -1)

            let tailStarts = completeCounts * Int32(compressRatio)
            let tail = keyIndices[.newAxis, .newAxis, .ellipsis] .>= tailStarts[.ellipsis, .newAxis]
                .&& keyIndices[.newAxis, .newAxis, .ellipsis] .< queryEnds[.ellipsis, .newAxis]
            let causal = keyIndices[.newAxis, .newAxis, .ellipsis] .< queryEnds[.ellipsis, .newAxis]
            let sparseMask = selected[.ellipsis, 0..<keyLength] .|| tail
            let useSparse = completeCounts .> Int32(blockTopK)
            selectedTokens = `where`(useSparse[.ellipsis, .newAxis], sparseMask, causal)
        } else {
            selectedTokens = keyIndices[.newAxis, .newAxis, .ellipsis] .< queryEnds[.ellipsis, .newAxis]
        }

        // Insert the SDPA head axis after the batch axis.  Using a leading
        // `newAxis` happens to work for B=1 but returns [1,B,Q,K] for a
        // batched prefill instead of the required [B,1,Q,K].
        return selectedTokens.expandedDimensions(axis: 1)
    }
}
