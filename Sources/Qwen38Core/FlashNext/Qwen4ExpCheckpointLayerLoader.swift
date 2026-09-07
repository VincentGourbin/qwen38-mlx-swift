import Foundation
import MLX
import MLXNN

public struct Qwen4ExpLoadedDecoderLayer: @unchecked Sendable {
    public let layer: Qwen4ExpDecoderLayer
    public let layerIndex: Int
    public let tensorCount: Int
    public let shardCount: Int
    public let materializedBytes: Int64

    public init(
        layer: Qwen4ExpDecoderLayer,
        layerIndex: Int,
        tensorCount: Int,
        shardCount: Int,
        materializedBytes: Int64
    ) {
        self.layer = layer
        self.layerIndex = layerIndex
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.materializedBytes = materializedBytes
    }
}

/// The immutable layer-to-shard map for a converted Flash-Next checkpoint.
///
/// The streaming decoder visits all 48 layers repeatedly during generation.
/// Parsing the 420 KB safetensors index for every visit is unnecessary and
/// made the validation path pay an avoidable filesystem/JSON cost.  Keeping
/// only the key names here does not materialize any checkpoint tensor.
public struct Qwen4ExpCheckpointLayerIndex: Sendable {
    private let keysByLayer: [Int: [String: [String]]]

    public init(directory: URL) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(indexURL)
        }

        var result = [Int: [String: [String]]]()
        for (key, shard) in weightMap {
            let prefix = "language_model.model.layers."
            guard key.hasPrefix(prefix) else { continue }
            let suffix = key.dropFirst(prefix.count)
            guard let separator = suffix.firstIndex(of: "."),
                  let layer = Int(suffix[..<separator]) else { continue }
            result[layer, default: [:]][shard, default: []].append(key)
        }
        keysByLayer = result.mapValues { byShard in
            byShard.mapValues { $0.sorted() }
        }
    }

    fileprivate func selectedKeys(for layerIndex: Int) -> [String: [String]] {
        keysByLayer[layerIndex] ?? [:]
    }
}

/// Loads only a decoder layer from a converted Flash-Next checkpoint.
///
/// Unlike the slice probe, this loader does not instantiate the global token
/// embedding or final hyper-connection. It is therefore suitable for a
/// layer-at-a-time executor: after `eval(output)`, the layer can be released
/// while its cache remains resident.
public enum Qwen4ExpCheckpointLayerLoader {
    private static let indexName = "model.safetensors.index.json"

    public static func load(
        _ layerIndex: Int,
        from directory: URL,
        index: Qwen4ExpCheckpointLayerIndex? = nil,
        materialize: Bool = true,
        useCheckpointQuantization: Bool = true
    ) throws -> Qwen4ExpLoadedDecoderLayer {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        guard configuration.textConfiguration.layerTypes.indices.contains(layerIndex) else {
            throw Qwen4ExpCheckpointSliceLoaderError.unsupportedLayer(layerIndex)
        }

        let selected: [String: [String]]
        if let index {
            selected = index.selectedKeys(for: layerIndex)
        } else {
            selected = try selectedKeys(directory: directory, layerIndex: layerIndex)
        }
        guard !selected.isEmpty else {
            throw Qwen4ExpCheckpointSliceLoaderError.unsupportedLayer(layerIndex)
        }
        var weights = [String: MLXArray]()
        var rawNGramKeysByShardFile = [String: [String]]()
        var loadedShards = Set<String>()
        var loadedTensorCount = 0
        for (shard, rawKeys) in selected.sorted(by: { $0.key < $1.key }) {
            let arrays = try loadArraysAndMetadata(
                url: directory.appendingPathComponent(shard), stream: .cpu).0
            loadedShards.insert(shard)
            for rawKey in rawKeys {
                guard let array = arrays[rawKey] else {
                    throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                }
                loadedTensorCount += 1
                if useCheckpointQuantization && isNGramTableKey(rawKey) {
                    rawNGramKeysByShardFile[shard, default: []].append(rawKey)
                    continue
                }
                // The regular layer tensors are evaluated below so they remain
                // resident between the shard dictionary and the decoder
                // forward. The large n-gram tensors take a separate path:
                // they are not retained as MLX arrays at all, because a later
                // gather could materialize the complete safetensors mapping.
                eval(array)
                guard let local = localKey(rawKey, layerIndex: layerIndex) else { continue }
                if weights[local] != nil {
                    throw Qwen4ExpCheckpointSliceLoaderError.duplicateTensor(local)
                }
                weights[local] = array
            }
        }

        let checkpointQuantization = Qwen4ExpQuantizationSpec(configuration.quantization)
        if !useCheckpointQuantization {
            guard let checkpointQuantization else {
                throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(
                    directory.appendingPathComponent("config.json"))
            }
            weights = try dequantize(
                weights, specification: checkpointQuantization)
        }

        let normCorrection = Qwen4ExpWeightSanitizer
            .correctShiftedZeroCenteredNormWeights(weights)
        weights = normCorrection.weights

        let lazyNGramStorage: Qwen4ExpLazyNGramStorage?
        if let pleLayerIndex = configuration.textConfiguration.pleLayerIDs.firstIndex(
            of: layerIndex + 1), let checkpointQuantization, !rawNGramKeysByShardFile.isEmpty {
            lazyNGramStorage = try Qwen4ExpLazyNGramStorage(
                directory: directory,
                rawKeysByShardFile: rawNGramKeysByShardFile,
                layerIndex: layerIndex,
                shardCount: configuration.textConfiguration.splitNgramParts,
                dimensions: (configuration.textConfiguration.pleEmbedDim
                    ?? configuration.textConfiguration.hiddenSize)
                    / ((configuration.textConfiguration.ngramSize - 1)
                        * (configuration.textConfiguration.headsPerNgram ?? 8)),
                quantization: checkpointQuantization)
            _ = pleLayerIndex
        } else {
            lazyNGramStorage = nil
        }

        let layer = Qwen4ExpDecoderLayer(
            configuration: configuration.textConfiguration,
            layerIndex: layerIndex,
            pleLayerIndex: configuration.textConfiguration.pleLayerIDs.firstIndex(
                of: layerIndex + 1),
            quantization: useCheckpointQuantization ? checkpointQuantization : nil,
            lazyNGramStorage: useCheckpointQuantization ? lazyNGramStorage : nil)
        try layer.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])

        let bytes: Int64
        if materialize {
            // Use the raw checkpoint keys here rather than the flattened
            // ModuleParameters paths. The latter are allowed to rewrite
            // snake_case into Swift property spelling; the raw keys are the
            // unambiguous boundary for the 32 GB n-gram table.
            let residentWeights = weights.filter {
                !isNGramTableKey($0.key)
            }.map { $0.value }
            eval(residentWeights)
            bytes = residentWeights.reduce(Int64(0)) {
                total, item in total + Int64(item.nbytes)
            }
        } else {
            bytes = 0
        }
        return Qwen4ExpLoadedDecoderLayer(
            layer: layer,
            layerIndex: layerIndex,
            tensorCount: loadedTensorCount,
            shardCount: loadedShards.count,
            materializedBytes: bytes)
    }

    private static func isNGramTableKey(_ key: String) -> Bool {
        // Keep the small hash metadata (`ngram_heads_*`) in the Module update;
        // only the 128 huge shard tensors belong to the external row reader.
        return key.lowercased().contains(".ngram_embedding.shard_")
    }

    private static func selectedKeys(
        directory: URL, layerIndex: Int
    ) throws -> [String: [String]] {
        let indexURL = directory.appendingPathComponent(indexName)
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(indexURL)
        }

        let prefix = "language_model.model.layers.\(layerIndex)."
        var result = [String: [String]]()
        for (key, shard) in weightMap where key.hasPrefix(prefix) {
            result[shard, default: []].append(key)
        }
        return result.mapValues { $0.sorted() }
    }

    private static func localKey(_ rawKey: String, layerIndex: Int) -> String? {
        guard let normalized = Qwen4ExpWeightSanitizer.normalize(rawKey) else { return nil }
        let prefix = "language_model.model.layers.\(layerIndex)."
        guard normalized.hasPrefix(prefix) else { return nil }
        return String(normalized.dropFirst(prefix.count))
    }

    /// Convert a checkpoint-shaped layer to floating-point weights without
    /// instantiating the huge PLE table. E3 uses this only for layers without
    /// PLE; the caller must keep PLE layers on the checkpoint quantization path.
    private static func dequantize(
        _ weights: [String: MLXArray],
        specification: Qwen4ExpQuantizationSpec
    ) throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                continue
            }
            guard key.hasSuffix(".weight") else {
                result[key] = value
                continue
            }
            let base = String(key.dropLast(".weight".count))
            guard let scales = weights[base + ".scales"] else {
                result[key] = value
                continue
            }
            result[key] = MLX.dequantized(
                value,
                scales: scales,
                biases: weights[base + ".biases"],
                groupSize: specification.groupSize,
                bits: specification.bits,
                mode: specification.mode).asType(.float32)
        }
        return result
    }
}
