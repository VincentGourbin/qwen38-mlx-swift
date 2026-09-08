import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The small set of text weights that surrounds the layer-at-a-time decoder.
///
/// Flash-Next has a very large n-gram table and 48 decoder layers.  Keeping
/// the embedding, final hyper-connection and language head in their own
/// module lets the streaming executor retain only the cheap global weights
/// while it swaps decoder layers in and out.
public final class Qwen4ExpGlobalTextModel: Module {
    public let configuration: Qwen4ExpTextConfiguration
    public let quantization: Qwen4ExpQuantizationSpec?

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "hyper_connection_mixer") public var hyperConnectionMixer: Qwen4ExpGatedResidual
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public init(
        configuration: Qwen4ExpTextConfiguration,
        quantization: Qwen4ExpQuantizationSpec?
    ) {
        self.configuration = configuration
        self.quantization = quantization

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

        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(
            configuration: configuration,
            useCombine: false,
            quantization: quantization)
        _lmHead.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: configuration.vocabSize,
            quantization: quantization)
        super.init()
    }

    public func embed(_ inputIDs: MLXArray) -> MLXArray {
        embedTokens(inputIDs)
    }

    public func reduceHyperStreams(_ hiddenStates: MLXArray) -> MLXArray {
        hyperConnectionMixer.mixedInput(hiddenStates)
    }

    public func logits(from hiddenStates: MLXArray) -> MLXArray {
        lmHead(hiddenStates)
    }
}

public struct Qwen4ExpGlobalLoadReport: Sendable, Equatable {
    public let tensorCount: Int
    public let shardCount: Int
    public let materializedBytes: Int64

    public init(tensorCount: Int, shardCount: Int, materializedBytes: Int64) {
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.materializedBytes = materializedBytes
    }
}

public final class Qwen4ExpLoadedGlobalTextModel: @unchecked Sendable {
    public let model: Qwen4ExpGlobalTextModel
    public let report: Qwen4ExpGlobalLoadReport

    public init(model: Qwen4ExpGlobalTextModel, report: Qwen4ExpGlobalLoadReport) {
        self.model = model
        self.report = report
    }
}

/// Loads only the global text tensors from the real Flash-Next index.
public enum Qwen4ExpGlobalCheckpointLoader {
    private static let indexName = "model.safetensors.index.json"

    public static func load(
        from directory: URL,
        materialize: Bool = true,
        uncachedIO: Bool = true
    ) throws -> Qwen4ExpLoadedGlobalTextModel {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let selected = try selectedKeys(directory: directory)
        var weights = [String: MLXArray]()
        var loadedShards = Set<String>()

        // See Qwen4ExpCheckpointLayerLoader for why `uncachedIO` reads
        // through `Qwen4ExpUncachedTensorReader` (F_NOCACHE + pread) rather
        // than `loadArraysAndMetadata` (P2-mem-a). The global tensors are a
        // few small tensors, but they still share the same shard files as
        // the huge resident layers, so reading them the old way would still
        // touch the page cache before the F_NOCACHE reader gets to them.
        for (shard, rawKeys) in selected.sorted(by: { $0.key < $1.key }) {
            let shardURL = directory.appendingPathComponent(shard)
            loadedShards.insert(shard)
            let reader: Qwen4ExpUncachedTensorReader? = uncachedIO
                ? try Qwen4ExpUncachedTensorReader(url: shardURL) : nil
            let arrays: [String: MLXArray]? = uncachedIO
                ? nil : try loadArraysAndMetadata(url: shardURL, stream: .cpu).0
            for rawKey in rawKeys {
                let array: MLXArray
                if let reader {
                    do {
                        array = try reader.array(for: rawKey)
                    } catch Qwen4ExpUncachedTensorReaderError.missingTensor {
                        throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                    }
                } else {
                    guard let cached = arrays?[rawKey] else {
                        throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                    }
                    array = cached
                }
                guard let local = localKey(rawKey) else { continue }
                if weights[local] != nil {
                    throw Qwen4ExpCheckpointSliceLoaderError.duplicateTensor(local)
                }
                weights[local] = array
            }
        }

        let normCorrection = Qwen4ExpWeightSanitizer
            .correctShiftedZeroCenteredNormWeights(weights)
        weights = normCorrection.weights

        let model = Qwen4ExpGlobalTextModel(
            configuration: configuration.textConfiguration,
            quantization: Qwen4ExpQuantizationSpec(configuration.quantization))
        // Apply each child independently.  This avoids a subtle MLXNN
        // recursive-update ambiguity when a top-level `ModuleInfo` key and a
        // quantized leaf both contain underscores; every child still gets a
        // strict `.all` update, so missing or extra checkpoint tensors fail.
        let embedWeights = weights.filter { $0.key.hasPrefix("embed_tokens.") }
        let hyperWeights = weights.filter { $0.key.hasPrefix("hyper_connection_mixer.") }
        let headWeights = weights.filter { $0.key.hasPrefix("lm_head.") }
        try model.embedTokens.update(
            parameters: ModuleParameters.unflattened(
                Dictionary(uniqueKeysWithValues: embedWeights.map {
                    (String($0.key.dropFirst("embed_tokens.".count)), $0.value)
                })), verify: [.all])
        try model.hyperConnectionMixer.update(
            parameters: ModuleParameters.unflattened(
                Dictionary(uniqueKeysWithValues: hyperWeights.map {
                    (String($0.key.dropFirst("hyper_connection_mixer.".count)), $0.value)
                })), verify: [.all])
        try model.lmHead.update(
            parameters: ModuleParameters.unflattened(
                Dictionary(uniqueKeysWithValues: headWeights.map {
                    (String($0.key.dropFirst("lm_head.".count)), $0.value)
                })), verify: [.all])

        let bytes: Int64
        if materialize {
            eval(model.parameters())
            bytes = model.parameters().flattened().reduce(Int64(0)) {
                $0 + Int64($1.1.nbytes)
            }
        } else {
            bytes = 0
        }

        return Qwen4ExpLoadedGlobalTextModel(
            model: model,
            report: Qwen4ExpGlobalLoadReport(
                tensorCount: weights.count,
                shardCount: loadedShards.count,
                materializedBytes: bytes))
    }

    private static func selectedKeys(directory: URL) throws -> [String: [String]] {
        let indexURL = directory.appendingPathComponent(indexName)
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(indexURL)
        }

        let prefixes = [
            "language_model.model.embed_tokens.",
            "language_model.model.hyper_connection_mixer.",
            "language_model.lm_head."
        ]
        var result = [String: [String]]()
        for (rawKey, shard) in weightMap where prefixes.contains(where: rawKey.hasPrefix) {
            result[shard, default: []].append(rawKey)
        }
        return result.mapValues { $0.sorted() }
    }

    private static func localKey(_ rawKey: String) -> String? {
        guard let normalized = Qwen4ExpWeightSanitizer.normalize(rawKey) else { return nil }
        let mappings = [
            "language_model.model.embed_tokens.": "embed_tokens.",
            "language_model.model.hyper_connection_mixer.": "hyper_connection_mixer.",
            "language_model.model.lm_head.": "lm_head."
        ]
        for (prefix, replacement) in mappings where normalized.hasPrefix(prefix) {
            return replacement + normalized.dropFirst(prefix.count)
        }
        return nil
    }
}
