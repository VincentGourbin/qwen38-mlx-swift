import MLX
import MLXLMCommon
import MLXNN

/// Text decoder skeleton for Flash-Next.
///
/// This is deliberately a text-only model seam first. Vision embedding merge,
/// lm_head/tied embeddings and the `LanguageModel`/`VLMModelFactory` adapter
/// remain separate so the 113 GB checkpoint is never loaded while the hybrid
/// decoder contract is still being validated.
public final class Qwen4ExpTextModel: Module {
    public let configuration: Qwen4ExpTextConfiguration
    public let layerIndices: [Int]
    public let quantization: Qwen4ExpQuantizationSpec?

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") public var hyperConnectionMixer: Qwen4ExpGatedResidual

    public init(
        configuration: Qwen4ExpTextConfiguration,
        layerIndices: [Int]? = nil,
        quantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        self.configuration = configuration
        self.quantization = quantization
        let selected = layerIndices ?? Array(0 ..< configuration.numHiddenLayers)
        precondition(!selected.isEmpty)
        precondition(selected.allSatisfy(configuration.layerTypes.indices.contains))
        self.layerIndices = selected

        if let quantization {
            _embedTokens.wrappedValue = Qwen4ExpPrequantizedEmbedding(
                embeddingCount: configuration.vocabSize,
                dimensions: configuration.hiddenSize,
                quantization: quantization)
        } else {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: configuration.vocabSize,
                dimensions: configuration.hiddenSize)
        }
        _layers.wrappedValue = selected.map { layerIndex in
            let pleLayerIndex = configuration.pleLayerIDs.firstIndex(of: layerIndex + 1)
            return Qwen4ExpDecoderLayer(
                configuration: configuration,
                layerIndex: layerIndex,
                pleLayerIndex: pleLayerIndex,
                quantization: quantization)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(
            configuration: configuration, useCombine: false,
            quantization: quantization)
        super.init()
    }

    /// Cache plan for the selected layer range.
    public func makeCache(
        budget: Int? = nil,
        compressRatio: Int? = nil
    ) -> [any KVCache] {
        layers.map { layer in
            if layer.isLinear {
                // PLE shares the GDN cache and additionally owns slots 2/3
                // for its short convolution and n-gram history.
                if layer.ple != nil {
                    return ArraysCache(size: 4)
                }
                return MambaCache()
            }
            return Qwen4ExpQSAKVCache(
                budget: budget ?? configuration.indexerBudget,
                compressRatio: compressRatio ?? configuration.indexerCompressRatio)
        }
    }

    /// Runs the selected decoder layers and returns one `[B,S,H]` text stream.
    ///
    /// The input is tiled to four streams once, and every layer advances the
    /// corresponding cache in place. Callers that use a truncated layer range
    /// can therefore validate a real layer without instantiating all experts.
    public func callAsFunction(
        _ inputIDs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        caches: inout [any KVCache],
        mask: MLXArray? = nil,
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        precondition(inputIDs.ndim == 2)
        precondition(caches.count == layers.count)
        var hidden = inputsEmbeds ?? embedTokens(inputIDs)
        precondition(hidden.ndim == 3)
        hidden = tiled(hidden, repetitions: [1, 1, configuration.hcCount])

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                inputIDs: inputIDs,
                mask: mask,
                cache: caches[index],
                positionIDs: positionIDs)
        }
        return hyperConnectionMixer.mixedInput(hidden)
    }
}
