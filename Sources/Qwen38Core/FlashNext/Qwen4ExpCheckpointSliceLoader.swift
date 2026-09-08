import Foundation
import MLX
import MLXNN

/// A small, real-checkpoint load result used while the full Flash-Next loader
/// is still being assembled.
public struct Qwen4ExpCheckpointSliceReport: Sendable, Equatable {
    public let layerIndex: Int
    public let tensorCount: Int
    public let shardCount: Int
    public let materializedBytes: Int64
    public let outputShape: [Int]?

    public init(
        layerIndex: Int,
        tensorCount: Int,
        shardCount: Int,
        materializedBytes: Int64,
        outputShape: [Int]? = nil
    ) {
        self.layerIndex = layerIndex
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.materializedBytes = materializedBytes
        self.outputShape = outputShape
    }
}

public final class Qwen4ExpLoadedCheckpointSlice: @unchecked Sendable {
    public let model: Qwen4ExpTextModel
    public let report: Qwen4ExpCheckpointSliceReport

    public init(model: Qwen4ExpTextModel, report: Qwen4ExpCheckpointSliceReport) {
        self.model = model
        self.report = report
    }
}

public enum Qwen4ExpCheckpointSliceLoaderError: LocalizedError, Equatable {
    case invalidIndex(URL)
    case unsupportedLayer(Int)
    case missingTensor(String)
    case duplicateTensor(String)

    public var errorDescription: String? {
        switch self {
        case .invalidIndex(let url): return "Index Flash-Next invalide : \(url.path)"
        case .unsupportedLayer(let layer):
            return "La tranche Flash-Next ne supporte pas encore la couche \(layer)."
        case .missingTensor(let key):
            return "Tenseur Flash-Next absent du shard sélectionné : \(key)"
        case .duplicateTensor(let key):
            return "Deux tenseurs du checkpoint correspondent au même paramètre : \(key)"
        }
    }
}

/// Loads one decoder layer from the actual converted checkpoint.
///
/// This is intentionally not a general model loader yet. It selects keys from
/// `model.safetensors.index.json` before opening each shard, creates the MLX
/// module in its packed shape, and only then calls strict `Module.update`.
/// Consequently the 51B-parameter n-gram table, the other 47 layers, the
/// vision tower and MTP are never instantiated or evaluated by this probe.
public enum Qwen4ExpCheckpointSliceLoader {
    private static let indexName = "model.safetensors.index.json"

    public static func loadLayer(
        _ layerIndex: Int,
        from directory: URL,
        materialize: Bool = true,
        runForward: Bool = false,
        sequenceLength: Int = 4
    ) throws -> Qwen4ExpLoadedCheckpointSlice {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        guard configuration.textConfiguration.layerTypes.indices.contains(layerIndex) else {
            throw Qwen4ExpCheckpointSliceLoaderError.unsupportedLayer(layerIndex)
        }
        let selected = try selectedCheckpointKeys(
            directory: directory, layerIndex: layerIndex)
        var weights = [String: MLXArray]()
        var loadedShards = Set<String>()

        for (shardName, rawKeys) in selected.byShard.sorted(by: { $0.key < $1.key }) {
            let shardURL = directory.appendingPathComponent(shardName)
            let arrays = try loadArraysAndMetadata(url: shardURL, stream: .cpu).0
            loadedShards.insert(shardName)
            for rawKey in rawKeys {
                guard let array = arrays[rawKey] else {
                    throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                }
                guard let localKey = localParameterKey(rawKey, layerIndex: layerIndex) else {
                    continue
                }
                if weights[localKey] != nil {
                    throw Qwen4ExpCheckpointSliceLoaderError.duplicateTensor(localKey)
                }
                weights[localKey] = array
            }
        }

        let quantization = Qwen4ExpQuantizationSpec(configuration.quantization)
        let expertsQuantization = Qwen4ExpQuantizationSpec.experts(from: configuration.quantization)
        let normCorrection = Qwen4ExpWeightSanitizer
            .correctShiftedZeroCenteredNormWeights(weights)
        weights = normCorrection.weights
        let model = Qwen4ExpTextModel(
            configuration: configuration.textConfiguration,
            layerIndices: [layerIndex],
            quantization: quantization,
            expertsQuantization: expertsQuantization)
        try model.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])

        let materializedBytes: Int64
        if materialize {
            eval(model.parameters())
            materializedBytes = model.parameters().flattened().reduce(Int64(0)) {
                total, item in total + Int64(item.1.nbytes)
            }
        } else {
            materializedBytes = 0
        }

        var outputShape: [Int]?
        if runForward {
            precondition(sequenceLength > 0)
            let ids = MLXArray(Array(repeating: Int32(1), count: sequenceLength))
                .reshaped([1, sequenceLength])
            var caches = model.makeCache()
            let output = model(ids, caches: &caches)
            eval(output)
            outputShape = output.shape
        } else {
            outputShape = nil
        }

        return Qwen4ExpLoadedCheckpointSlice(
            model: model,
            report: Qwen4ExpCheckpointSliceReport(
                layerIndex: layerIndex,
                tensorCount: weights.count,
                shardCount: loadedShards.count,
                materializedBytes: materializedBytes,
                outputShape: outputShape))
    }

    private struct SelectedKeys {
        var byShard: [String: [String]]
    }

    private static func selectedCheckpointKeys(
        directory: URL, layerIndex: Int
    ) throws -> SelectedKeys {
        let indexURL = directory.appendingPathComponent(indexName)
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(indexURL)
        }

        let layerPrefix = "language_model.model.layers.\(layerIndex)."
        let globalPrefixes = [
            "language_model.model.embed_tokens.",
            "language_model.model.hyper_connection_mixer."
        ]
        var byShard = [String: [String]]()
        for (rawKey, shardName) in weightMap {
            let isSelected = rawKey.hasPrefix(layerPrefix)
                || globalPrefixes.contains(where: rawKey.hasPrefix)
            guard isSelected else { continue }
            byShard[shardName, default: []].append(rawKey)
        }
        return SelectedKeys(byShard: byShard.mapValues { $0.sorted() })
    }

    private static func localParameterKey(_ rawKey: String, layerIndex: Int) -> String? {
        guard let normalized = Qwen4ExpWeightSanitizer.normalize(rawKey) else { return nil }
        let layerPrefix = "language_model.model.layers.\(layerIndex)."
        if normalized.hasPrefix(layerPrefix) {
            // The reduced model contains one array element at slot 0. Its
            // public `layerIndex` remains the source checkpoint index, but
            // ModuleParameters addresses the selected array element as
            // `layers.0.*`.
            return "layers.0." + normalized.dropFirst(layerPrefix.count)
        }
        if normalized.hasPrefix("language_model.model.embed_tokens.") {
            return "embed_tokens." + normalized.dropFirst("language_model.model.embed_tokens.".count)
        }
        if normalized.hasPrefix("language_model.model.hyper_connection_mixer.") {
            return "hyper_connection_mixer." + normalized.dropFirst(
                "language_model.model.hyper_connection_mixer.".count)
        }
        return nil
    }
}
