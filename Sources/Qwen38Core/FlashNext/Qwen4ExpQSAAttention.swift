import MLX
import MLXLMCommon
import MLXNN

/// Minimal full-attention block for a Flash-Next QSA layer.
///
/// This is intentionally limited to the attention branch: it owns the QSA
/// selector and the regular Q/K/V path, while MoE, hyper-connections and the
/// recurrent layers remain separate integration steps. Keeping this seam
/// small makes it possible to compare one real full-attention layer against
/// mlx-vlm before allocating the rest of the 113 GB checkpoint.
public final class Qwen4ExpQSAAttention: Module {
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rotaryEmbedding: Qwen4ExpMRoPE
    public let indexer: Qwen4ExpQSAIndexer

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: Qwen4ExpRMSNorm

    /// Boundary tensors retained only for the real-checkpoint parity probe.
    /// Keeping them on the attention object avoids a second forward with a
    /// separately reconstructed QSA path.
    public private(set) var lastParityCapture: [String: MLXArray] = [:]
    /// Disabled during normal inference so intermediate graphs are not kept alive.
    public private(set) var captureParity = false

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        self.numAttentionHeads = configuration.numAttentionHeads
        self.numKeyValueHeads = configuration.numKeyValueHeads
        self.headDim = configuration.headDim
        self.scale = 1 / Float(configuration.headDim).squareRoot()
        precondition(headDim >= 64, "QSA full-attention head_dim doit permettre le MRoPE partiel")
        self.rotaryEmbedding = Qwen4ExpMRoPE(
            rotaryDim: 64, base: 10_000_000, mropeSections: [11, 11, 10])
        self.indexer = Qwen4ExpQSAIndexer(
            configuration: configuration, rmsNormEps: rmsNormEps,
            quantization: quantization)

        _qProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: numAttentionHeads * headDim * 2,
            quantization: quantization)
        _kProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: numKeyValueHeads * headDim,
            quantization: quantization)
        _vProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: numKeyValueHeads * headDim,
            quantization: quantization)
        _oProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: numAttentionHeads * headDim,
            outputDimensions: configuration.hiddenSize,
            quantization: quantization)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: rmsNormEps)
        super.init()
    }

    /// Run one QSA attention branch, including cache update and output gate.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        cache: Qwen4ExpQSAKVCache? = nil,
        positionIDs: MLXArray? = nil,
        mask attentionMask: MLXArray? = nil
    ) -> MLXArray {
        precondition(hiddenStates.ndim == 3, "QSA hidden states attendus en [B,S,H]")
        let batch = hiddenStates.dim(0)
        let sequence = hiddenStates.dim(1)
        let positions = positionIDs ?? Qwen4ExpMRoPE.textPositionIDs(
            sequenceLength: sequence, offset: cache?.offset ?? 0)

        let qOutput = qProj(hiddenStates).reshaped(
            [batch, sequence, numAttentionHeads, headDim * 2])
        let qParts = qOutput.split(parts: 2, axis: -1)
        let normalizedQueries = qNorm(qParts[0])
        var queries = normalizedQueries.transposed(0, 2, 1, 3)
        let outputGate = qParts[1].reshaped([batch, sequence, numAttentionHeads * headDim])

        let normalizedKeys = kNorm(kProj(hiddenStates).reshaped(
            [batch, sequence, numKeyValueHeads, headDim]))
        var keys = normalizedKeys.transposed(0, 2, 1, 3)
        let values = vProj(hiddenStates).reshaped(
            [batch, sequence, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)

        queries = rotaryEmbedding.apply(queries, positionIDs: positions)
        keys = rotaryEmbedding.apply(keys, positionIDs: positions)

        if captureParity {
            lastParityCapture = [
                "q_normed": normalizedQueries,
                "k_normed": normalizedKeys,
                "v": values,
                "q_rope": queries,
                "k_rope": keys,
                "output_gate": outputGate,
            ]
        } else {
            lastParityCapture.removeAll(keepingCapacity: true)
        }

        // QSA must inspect the pre-update offset, then the regular cache is
        // advanced with the already-rotated K/V tensors.
        let sparseMask = indexer.makeMask(
            hiddenStates: hiddenStates,
            positionIDs: positions,
            cache: cache,
            compressRatio: cache?.compressRatio ?? 4,
            budget: cache?.budget ?? 2_048)
        let cached: (MLXArray, MLXArray)
        if let cache {
            cached = cache.update(keys: keys, values: values)
        } else {
            cached = (keys, values)
        }

        let keyLength = cached.0.dim(-2)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let sparseMask {
            if let attentionMask {
                // The first implementation only reaches this branch once
                // QSA has selected compressed blocks.  Preserve the target's
                // causal/batch mask as an additional restriction; otherwise
                // a multi-token MTP prefill could attend into padding.
                let causal = attentionMask[.ellipsis, 0..<keyLength]
                mask = .array(MLX.logicalAnd(
                    causal, sparseMask[.ellipsis, 0..<keyLength]))
            } else {
                mask = .array(sparseMask[.ellipsis, 0..<keyLength])
            }
        } else if let attentionMask {
            mask = .array(attentionMask[.ellipsis, 0..<keyLength])
        } else {
            // Match mlx-vlm's public QSA contract: when the indexer has not
            // crossed the sparse-selection boundary, it returns nil and the
            // caller leaves the SDPA mask untouched.  In particular, a
            // layer-at-a-time probe passes mask=nil and the reference uses
            // dense, unmasked SDPA for that prefill.  Do not silently add a
            // causal mask here; the full model supplies its own mask when it
            // wants one.
            mask = .none
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cached.0,
            values: cached.1,
            scale: scale,
            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, sequence, numAttentionHeads * headDim])

        let gatedAttended = attended * sigmoid(outputGate)
        let output = oProj(gatedAttended)
        if captureParity {
            lastParityCapture["attended"] = attended
            lastParityCapture["gated_attended"] = gatedAttended
            lastParityCapture["output"] = output
        }
        return output
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }

    public static func causalMask(
        batch: Int, queryLength: Int, keyLength: Int, offset: Int
    ) -> MLXArray {
        let keys = MLXArray(Int32(0) ..< Int32(keyLength))
        let ends = MLXArray(Int32(offset) ..< Int32(offset + queryLength)) + 1
        return broadcast(
            keys[.newAxis, .newAxis, .ellipsis] .< ends[.newAxis, .ellipsis, .newAxis],
            to: [batch, 1, queryLength, keyLength])
    }
}
