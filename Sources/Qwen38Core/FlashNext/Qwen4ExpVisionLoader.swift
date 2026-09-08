import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen4ExpVisionLoadReport: Sendable, Equatable {
    public let tensorCount: Int
    public let shardCount: Int
    public let materializedBytes: Int64

    public init(tensorCount: Int, shardCount: Int, materializedBytes: Int64) {
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.materializedBytes = materializedBytes
    }
}

public final class Qwen4ExpLoadedVisionEncoder: @unchecked Sendable {
    public let model: Qwen4ExpVisionEncoder
    public let report: Qwen4ExpVisionLoadReport

    public init(model: Qwen4ExpVisionEncoder, report: Qwen4ExpVisionLoadReport) {
        self.model = model
        self.report = report
    }
}

/// Loads the BF16 vision tower independently from the 113 GB language group.
public enum Qwen4ExpVisionCheckpointLoader {
    private static let indexName = "model.safetensors.index.json"

    public static func load(
        from directory: URL,
        materialize: Bool = true,
        uncachedIO: Bool = true
    ) throws -> Qwen4ExpLoadedVisionEncoder {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let indexURL = directory.appendingPathComponent(indexName)
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(indexURL)
        }

        var selected = [String: [String]]()
        for (key, shard) in weightMap where key.hasPrefix("vision_tower.") {
            selected[shard, default: []].append(key)
        }
        var weights = [String: MLXArray]()
        var shards = Set<String>()
        // See Qwen4ExpCheckpointLayerLoader (P2-mem-a): F_NOCACHE reads keep
        // the vision tower's shard reads from filling the file cache too.
        for (shard, keys) in selected.sorted(by: { $0.key < $1.key }) {
            let shardURL = directory.appendingPathComponent(shard)
            shards.insert(shard)
            let reader: Qwen4ExpUncachedTensorReader? = uncachedIO
                ? try Qwen4ExpUncachedTensorReader(url: shardURL) : nil
            let arrays: [String: MLXArray]? = uncachedIO
                ? nil : try loadArraysAndMetadata(url: shardURL, stream: .cpu).0
            for rawKey in keys.sorted() {
                let value: MLXArray
                if let reader {
                    do {
                        value = try reader.array(for: rawKey)
                    } catch Qwen4ExpUncachedTensorReaderError.missingTensor {
                        throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                    }
                } else {
                    guard let cached = arrays?[rawKey] else {
                        throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                    }
                    value = cached
                }
                guard let local = localKey(rawKey) else { continue }
                weights[local] = value
            }
        }

        let model = Qwen4ExpVisionEncoder(configuration: configuration.visionConfiguration)
        try model.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        let bytes: Int64
        if materialize {
            eval(model.parameters())
            bytes = model.parameters().flattened().reduce(Int64(0)) {
                $0 + Int64($1.1.nbytes)
            }
        } else {
            bytes = 0
        }
        return Qwen4ExpLoadedVisionEncoder(
            model: model,
            report: Qwen4ExpVisionLoadReport(
                tensorCount: weights.count,
                shardCount: shards.count,
                materializedBytes: bytes))
    }

    private static func localKey(_ rawKey: String) -> String? {
        let prefix = "vision_tower."
        guard rawKey.hasPrefix(prefix) else { return nil }
        let suffix = String(rawKey.dropFirst(prefix.count))
        if suffix == "pos_embed.weight" { return "pos_embed" }
        return suffix
    }
}
