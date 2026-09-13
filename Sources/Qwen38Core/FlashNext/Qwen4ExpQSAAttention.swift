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

    /// F1 (P2-fusion): `q_proj/k_proj/v_proj` concatenated into one matmul
    /// once `fuseInputProjections()` has run. Not an `@ModuleInfo` property —
    /// see the identical comment on `Qwen4ExpGatedDeltaNet.fusedInProj`.
    private var fusedQKV: Linear?

    /// P7.1: which sub-block, if any, `callAsFunction` short-circuits for
    /// `flash-layer-bench --ablate`. `.none` everywhere in production —
    /// see `Qwen4ExpLayerBenchAblation`. P11.2 : mutable — voir
    /// `Qwen4ExpSparseMoE.ablation`'s doc comment.
    public private(set) var ablation: Qwen4ExpLayerBenchAblation

    /// P8.2 (F7): whether query/key are rounded back to the network's
    /// working dtype after `Qwen4ExpMRoPE.apply` — see
    /// `Qwen4ExpFusionLevel.f7GatedBranchDtype`.
    private let fusionLevel: Qwen4ExpFusionLevel

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        ablation: Qwen4ExpLayerBenchAblation = .none,
        fusionLevel: Qwen4ExpFusionLevel = .none
    ) {
        self.ablation = ablation
        self.fusionLevel = fusionLevel
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

        let qWidth = numAttentionHeads * headDim * 2
        let kWidth = numKeyValueHeads * headDim
        let qRaw: MLXArray
        let kRaw: MLXArray
        let vRaw: MLXArray
        if let fusedQKV {
            // F1: one matmul instead of three; split widths match
            // q_proj/k_proj/v_proj's original output widths, in that order.
            let fused = fusedQKV(hiddenStates)
            let parts = MLX.split(fused, indices: [qWidth, qWidth + kWidth], axis: -1)
            qRaw = parts[0]
            kRaw = parts[1]
            vRaw = parts[2]
        } else {
            qRaw = qProj(hiddenStates)
            kRaw = kProj(hiddenStates)
            vRaw = vProj(hiddenStates)
        }

        let qOutput = qRaw.reshaped([batch, sequence, numAttentionHeads, headDim * 2])
        let qParts = qOutput.split(parts: 2, axis: -1)
        // P7.1 (`--ablate norms`): skip q_norm/k_norm, reusing their already
        // shape-correct input. Never numerically correct — see
        // `Qwen4ExpLayerBenchAblation`.
        let normalizedQueries = ablation == .norms ? qParts[0] : qNorm(qParts[0])
        var queries = normalizedQueries.transposed(0, 2, 1, 3)
        let outputGate = qParts[1].reshaped([batch, sequence, numAttentionHeads * headDim])

        let kReshaped = kRaw.reshaped([batch, sequence, numKeyValueHeads, headDim])
        let normalizedKeys = ablation == .norms ? kReshaped : kNorm(kReshaped)
        var keys = normalizedKeys.transposed(0, 2, 1, 3)
        let values = vRaw.reshaped(
            [batch, sequence, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)

        queries = rotaryEmbedding.apply(queries, positionIDs: positions)
        keys = rotaryEmbedding.apply(keys, positionIDs: positions)
        // P8.2 (F7): `Qwen4ExpMRoPE.apply` mixes `queries`/`keys` (the
        // network's working dtype) with `cos`/`sin` tables deliberately kept
        // in fp32 for RoPE precision (piège #6) — MLX's type promotion then
        // makes `apply`'s result fp32, with no downcast back. Below
        // `f7GatedBranchDtype` that fp32 tensor flows straight into SDPA and
        // everything after it (including the whole MoE branch), which then
        // runs its (much slower) fp32 path. Rounding here restores the
        // network's bf16 working dtype at the RoPE boundary, same as F7 does
        // for `Qwen4ExpGatedDeltaNet`'s recurrence output.
        if fusionLevel >= .f7GatedBranchDtype {
            queries = queries.asType(hiddenStates.dtype)
            keys = keys.asType(hiddenStates.dtype)
        }

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
        // P7.1 (`--ablate qsa-attn`): skip the indexer entirely (`nil`
        // reproduces its "not yet past the sparse-selection boundary"
        // return, already handled below by the existing mask branches).
        // Never numerically correct — see `Qwen4ExpLayerBenchAblation`.
        let sparseMask = ablation == .qsaAttn
            ? nil
            : indexer.makeMask(
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

        // P7.1 (`--ablate qsa-attn`): skip SDPA, substituting a zeroed
        // tensor of the exact post-SDPA shape. Never numerically correct —
        // see `Qwen4ExpLayerBenchAblation`.
        let attended = ablation == .qsaAttn
            ? MLXArray.zeros([batch, sequence, numAttentionHeads * headDim], dtype: queries.dtype)
            : MLXFast.scaledDotProductAttention(
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

    /// P11.2 : change `ablation` sur une instance déjà construite, sans
    /// recharger aucun poids — voir `Qwen4ExpSparseMoE.setAblation`.
    public func setAblation(_ new: Qwen4ExpLayerBenchAblation) {
        ablation = new
    }

    /// F1 (P2-fusion): build the fused `q_proj/k_proj/v_proj` matmul once,
    /// after the three checkpoint-shaped modules have their real loaded
    /// weights. Idempotent; the indexer's `index_qk_proj` already covers
    /// query+key in one projection and needs no equivalent step.
    public func fuseInputProjections() {
        guard fusedQKV == nil else { return }
        let fused = qwen4ExpFuseLinear([qProj, kProj, vProj])
        eval(fused.weight)
        if let quantized = fused as? QuantizedLinear {
            eval(quantized.scales)
            if let biases = quantized.biases { eval(biases) }
        }
        fusedQKV = fused
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

    /// P12.2 (lots à longueurs inégales, remplis à gauche — voir
    /// `Qwen4ExpBatchPadding.swift`) : variante qui, en plus de la
    /// restriction causale usuelle, interdit à toute requête d'assister aux
    /// colonnes de remplissage à gauche d'une ligne. Ces colonnes restent
    /// invalides pour toute la durée de vie du cache — bien après le
    /// préremplissage — donc cette variante doit être utilisée à chaque
    /// appel tant qu'un lot porte du remplissage, y compris au décodage à un
    /// seul jeton : contrairement à P2-code (e), le cache contient alors des
    /// colonnes définitivement invalides, et le masque n'est plus jamais
    /// trivialement vrai partout. `leftPadding` doit compter `batch` entrées,
    /// une par ligne.
    public static func causalMask(
        batch: Int, queryLength: Int, keyLength: Int, offset: Int, leftPadding: [Int]
    ) -> MLXArray {
        precondition(
            leftPadding.count == batch, "leftPadding doit compter une entrée par ligne du lot")
        let base = causalMask(batch: batch, queryLength: queryLength, keyLength: keyLength, offset: offset)
        let keyIndices = MLXArray(Int32(0) ..< Int32(keyLength)).reshaped([1, 1, 1, keyLength])
        let padding = MLXArray(leftPadding.map(Int32.init)).reshaped([batch, 1, 1, 1])
        return MLX.logicalAnd(base, keyIndices .>= padding)
    }
}
