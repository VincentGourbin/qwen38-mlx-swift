import MLX
import MLXLMCommon
import MLXNN

/// One Flash-Next decoder layer.
///
/// Flash-Next uses four residual streams. Each attention and MoE branch is
/// reduced by a gated hyper-connection and injected back into all streams;
/// this choreography is part of the checkpoint contract, not an ordinary
/// residual wrapper around an attention module.
public final class Qwen4ExpDecoderLayer: Module {
    public let layerIndex: Int
    public let isLinear: Bool
    public private(set) var captureParity = false

    @ModuleInfo(key: "self_attn") public var selfAttn: Qwen4ExpQSAAttention?
    @ModuleInfo(key: "linear_attn") public var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "mlp") public var mlp: Qwen4ExpSparseMoE
    @ModuleInfo(key: "attn_hyper_connection") public var attnHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") public var mlpHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "ple") public var ple: Qwen4ExpPLELayer?

    public init(
        configuration: Qwen4ExpTextConfiguration,
        layerIndex: Int,
        pleLayerIndex: Int? = nil,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil,
        lazyNGramStorage: Qwen4ExpLazyNGramStorage? = nil
    ) {
        precondition(configuration.layerTypes.indices.contains(layerIndex))
        self.layerIndex = layerIndex
        self.isLinear = configuration.layerTypes[layerIndex] == .linearAttention

        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(
                configuration: configuration, rmsNormEps: rmsNormEps,
                quantization: quantization)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpQSAAttention(
                configuration: configuration, rmsNormEps: rmsNormEps,
                quantization: quantization)
        }

        _mlp.wrappedValue = Qwen4ExpSparseMoE(
            configuration: configuration, quantization: quantization,
            expertsQuantization: expertsQuantization)
        _attnHyperConnection.wrappedValue = Qwen4ExpGatedResidual(
            configuration: configuration, rmsNormEps: rmsNormEps,
            quantization: quantization, parityPrefix: "attn_")
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(
            configuration: configuration, rmsNormEps: rmsNormEps,
            quantization: quantization, parityPrefix: "mlp_")
        if let pleLayerIndex {
            _ple.wrappedValue = Qwen4ExpPLELayer(
                configuration: configuration,
                layerIndex: layerIndex,
                pleLayerIndex: pleLayerIndex,
                quantization: quantization,
                lazyStorage: lazyNGramStorage)
        }
        super.init()
    }

    /// Executes one layer with the cache selected by its attention type.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        inputIDs: MLXArray,
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil,
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        precondition(hiddenStates.ndim == 3)
        precondition(hiddenStates.dim(-1) == attnHyperConnection.streamCount *
            (hiddenStates.dim(-1) / attnHyperConnection.streamCount))

        var state = hiddenStates
        if let ple {
            guard let arrayCache = cache as? ArraysCache else {
                preconditionFailure("La PLE Flash-Next exige un ArraysCache")
            }
            state = state + ple(
                hiddenStates: state, inputIDs: inputIDs, cache: arrayCache, mask: mask)
        }

        let attentionMix = attnHyperConnection(state)
        let attentionBranch: MLXArray
        if isLinear {
            guard let linearAttn, let arrayCache = cache as? ArraysCache else {
                preconditionFailure("Une couche linear_attention exige un ArraysCache")
            }
            attentionBranch = linearAttn(attentionMix.mixedInput, mask: mask, cache: arrayCache)
        } else {
            guard let selfAttn, let qsaCache = cache as? Qwen4ExpQSAKVCache else {
                preconditionFailure("Une couche full_attention exige un cache QSA")
            }
            attentionBranch = selfAttn(
                attentionMix.mixedInput,
                cache: qsaCache,
                positionIDs: positionIDs,
                mask: mask)
        }
        state = Self.inject(
            branch: attentionBranch,
            hyperInput: attentionMix.originalInput,
            weights: attentionMix.injectionWeights)

        let mlpMix = mlpHyperConnection(state)
        let mlpBranch = mlp(mlpMix.mixedInput)
        return Self.inject(
            branch: mlpBranch,
            hyperInput: mlpMix.originalInput,
            weights: mlpMix.injectionWeights)
    }

    /// Enables boundary captures only for a diagnostic parity forward.
    /// Normal generation leaves all child capture dictionaries empty.
    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        selfAttn?.setParityCapture(enabled)
        linearAttn?.setParityCapture(enabled)
        attnHyperConnection.setParityCapture(enabled)
        mlpHyperConnection.setParityCapture(enabled)
        mlp.setParityCapture(enabled)
    }

    public func ngramCacheStats() -> Qwen4ExpNGramCacheStats? {
        ple?.ngramCacheStats()
    }

    private static func inject(
        branch: MLXArray,
        hyperInput: MLXArray,
        weights: MLXArray
    ) -> MLXArray {
        precondition(branch.ndim == 3 && hyperInput.ndim == 3 && weights.ndim == 3)
        let injection = branch.expandedDimensions(axis: -2) * weights.expandedDimensions(axis: -1)
        return hyperInput + injection.reshaped(hyperInput.shape)
    }
}
