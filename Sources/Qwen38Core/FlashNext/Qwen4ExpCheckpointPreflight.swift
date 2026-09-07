import Foundation

/// Header-only validation for a Flash-Next safetensors checkpoint.
///
/// The checkpoint is too large to validate by loading arrays. Safetensors
/// keeps the JSON header at the beginning of every shard, so this preflight
/// checks the complete key contract, tensor headers and quantized companion
/// tensors without materializing model weights.
public struct Qwen4ExpCheckpointPreflightReport: Sendable, Equatable {
    public let tensorCount: Int
    public let shardCount: Int
    public let weightBytes: Int64
    public let quantizedWeightCount: Int
    public let linearLayerCount: Int
    public let fullAttentionLayerCount: Int
    public let ngramShardCount: Int
    public let mtpTensorCount: Int

    public init(
        tensorCount: Int,
        shardCount: Int,
        weightBytes: Int64,
        quantizedWeightCount: Int,
        linearLayerCount: Int,
        fullAttentionLayerCount: Int,
        ngramShardCount: Int,
        mtpTensorCount: Int
    ) {
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.weightBytes = weightBytes
        self.quantizedWeightCount = quantizedWeightCount
        self.linearLayerCount = linearLayerCount
        self.fullAttentionLayerCount = fullAttentionLayerCount
        self.ngramShardCount = ngramShardCount
        self.mtpTensorCount = mtpTensorCount
    }
}

public enum Qwen4ExpCheckpointPreflightError: LocalizedError, Equatable {
    case missingIndex(URL)
    case invalidIndex(URL)
    case missingShard(String)
    case tensorMissingFromShard(String, String)
    case extraTensorInShard(String, String)
    case invalidShardHeader(String)
    case invalidTensorHeader(String, String)
    case invalidTensorShape(String, [Int], [Int])
    case invalidDataOffsets(String)
    case orphanQuantizedCompanion(String)

    public var errorDescription: String? {
        switch self {
        case .missingIndex(let url): return "Index safetensors absent : \(url.path)"
        case .invalidIndex(let url): return "Index safetensors invalide : \(url.path)"
        case .missingShard(let shard): return "Shard safetensors absent : \(shard)"
        case .tensorMissingFromShard(let tensor, let shard):
            return "Tenseur absent du shard référencé : \(tensor) → \(shard)"
        case .extraTensorInShard(let tensor, let shard):
            return "Tenseur non déclaré dans l'index : \(tensor) → \(shard)"
        case .invalidShardHeader(let shard): return "Header safetensors invalide : \(shard)"
        case .invalidTensorHeader(let tensor, let shard):
            return "Header de tenseur invalide : \(tensor) → \(shard)"
        case .invalidTensorShape(let tensor, let expected, let actual):
            return "Forme de tenseur invalide : \(tensor), attendu \(expected), obtenu \(actual)"
        case .invalidDataOffsets(let tensor): return "Offsets safetensors invalides : \(tensor)"
        case .orphanQuantizedCompanion(let tensor):
            return "Compagnon quantifié sans poids .weight : \(tensor)"
        }
    }
}

public enum Qwen4ExpCheckpointPreflight {
    private static let indexName = "model.safetensors.index.json"
    private static let maxHeaderBytes = 128 * 1024 * 1024

    public static func validate(_ directory: URL) throws -> Qwen4ExpCheckpointPreflightReport {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let indexURL = directory.appendingPathComponent(indexName)
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw Qwen4ExpCheckpointPreflightError.missingIndex(indexURL)
        }

        let indexObject: Any
        do {
            indexObject = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
        } catch {
            throw Qwen4ExpCheckpointPreflightError.invalidIndex(indexURL)
        }
        guard let index = indexObject as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpCheckpointPreflightError.invalidIndex(indexURL)
        }

        try Qwen4ExpWeightSanitizer.validateIndexKeys(weightMap.keys)

        let shardNames = Set(weightMap.values).sorted()
        var allHeaderKeys = Set<String>()
        var totalWeightBytes: Int64 = 0
        var allHeaders = [String: [String: Any]]()

        for shardName in shardNames {
            let shardURL = directory.appendingPathComponent(shardName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw Qwen4ExpCheckpointPreflightError.missingShard(shardName)
            }
            let fileSize = try fileByteCount(shardURL)
            totalWeightBytes += fileSize
            let header = try readHeader(from: shardURL, shardName: shardName)
            let headerKeys = Set(header.keys.filter { $0 != "__metadata__" })
            let indexedKeys = Set(weightMap.compactMap { key, value in
                value == shardName ? key : nil
            })

            for key in indexedKeys where !headerKeys.contains(key) {
                throw Qwen4ExpCheckpointPreflightError.tensorMissingFromShard(key, shardName)
            }
            for key in headerKeys where !indexedKeys.contains(key) {
                throw Qwen4ExpCheckpointPreflightError.extraTensorInShard(key, shardName)
            }
            allHeaderKeys.formUnion(headerKeys)
            for key in headerKeys {
                if let value = header[key] as? [String: Any] {
                    allHeaders[key] = value
                }
            }

            try validateTensorHeaders(header, shardName: shardName, fileSize: fileSize)
        }

        let indexKeys = Set(weightMap.keys)
        if allHeaderKeys != indexKeys {
            // The per-shard checks above normally make this unreachable; keep
            // the global equality explicit for diagnostics if an index is odd.
            if let missing = indexKeys.subtracting(allHeaderKeys).sorted().first {
                throw Qwen4ExpCheckpointPreflightError.tensorMissingFromShard(
                    missing, weightMap[missing] ?? "unknown")
            }
            else if let extra = allHeaderKeys.subtracting(indexKeys).sorted().first {
                throw Qwen4ExpCheckpointPreflightError.extraTensorInShard(extra, "unknown")
            }
        }

        let quantizedWeightCount = try validateQuantizedCompanions(indexKeys)
        let linearLayers = layerIndices(in: indexKeys, marker: ".linear_attn.in_proj_qkv.weight")
        let fullLayers = layerIndices(in: indexKeys, marker: ".self_attn.q_proj.weight")
        let ngramShards = indexKeys.filter { $0.contains(".ngram_embedding.shard_") && $0.hasSuffix(".weight") }
        let mtpTensorCount = indexKeys.filter { $0.hasPrefix("language_model.mtp.") }.count

        // Keep configuration and checkpoint topology tied together. This is
        // intentionally stricter than a generic safetensors integrity check.
        guard linearLayers.count + fullLayers.count == configuration.textConfiguration.numHiddenLayers else {
            throw Qwen4ExpCheckpointPreflightError.invalidIndex(indexURL)
        }
        guard linearLayers.count == configuration.textConfiguration.layerTypes.filter({ $0 == .linearAttention }).count,
              fullLayers.count == configuration.textConfiguration.layerTypes.filter({ $0 == .fullAttention }).count,
              ngramShards.count == configuration.textConfiguration.splitNgramParts,
              mtpTensorCount > 0
        else {
            throw Qwen4ExpCheckpointPreflightError.invalidIndex(indexURL)
        }

        try validateAnchorShapes(allHeaders, configuration: configuration)

        return Qwen4ExpCheckpointPreflightReport(
            tensorCount: indexKeys.count,
            shardCount: shardNames.count,
            weightBytes: totalWeightBytes,
            quantizedWeightCount: quantizedWeightCount,
            linearLayerCount: linearLayers.count,
            fullAttentionLayerCount: fullLayers.count,
            ngramShardCount: ngramShards.count,
            mtpTensorCount: mtpTensorCount)
    }

    private static func fileByteCount(_ url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func readHeader(from url: URL, shardName: String) throws -> [String: Any] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw Qwen4ExpCheckpointPreflightError.invalidShardHeader(shardName)
        }
        let bytes = [UInt8](lengthData)
        let length = bytes.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard length > 0, length <= UInt64(maxHeaderBytes), length <= UInt64(Int.max) else {
            throw Qwen4ExpCheckpointPreflightError.invalidShardHeader(shardName)
        }
        guard let headerData = try handle.read(upToCount: Int(length)), headerData.count == Int(length),
              let object = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            throw Qwen4ExpCheckpointPreflightError.invalidShardHeader(shardName)
        }
        return object
    }

    private static func validateTensorHeaders(
        _ header: [String: Any], shardName: String, fileSize: Int64
    ) throws {
        for (key, value) in header where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                  tensor["dtype"] is String,
                  tensor["shape"] is [Any],
                  let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                  let start = offsets[0] as? NSNumber,
                  let end = offsets[1] as? NSNumber
            else {
                throw Qwen4ExpCheckpointPreflightError.invalidTensorHeader(key, shardName)
            }
            let startValue = start.int64Value
            let endValue = end.int64Value
            guard startValue >= 0, endValue >= startValue, endValue <= fileSize else {
                throw Qwen4ExpCheckpointPreflightError.invalidDataOffsets(key)
            }
        }
    }

    private static func validateQuantizedCompanions(_ keys: Set<String>) throws -> Int {
        var quantizedCount = 0
        for key in keys where key.hasSuffix(".scales") || key.hasSuffix(".biases") {
            let base = String(key.dropLast(key.hasSuffix(".scales") ? 7 : 7)) + ".weight"
            guard keys.contains(base) else {
                throw Qwen4ExpCheckpointPreflightError.orphanQuantizedCompanion(key)
            }
        }
        for key in keys where key.hasSuffix(".weight") {
            let stem = String(key.dropLast(7))
            if keys.contains(stem + ".scales") && keys.contains(stem + ".biases") {
                quantizedCount += 1
            }
        }
        return quantizedCount
    }

    /// Validate shapes that define the architecture and quantized packing.
    /// The complete key set is checked above; these anchors catch a checkpoint
    /// whose names look right but whose module orientation or packed width is
    /// incompatible with the Swift implementation.
    private static func validateAnchorShapes(
        _ headers: [String: [String: Any]], configuration: Qwen4ExpConfiguration
    ) throws {
        let text = configuration.textConfiguration
        guard let quantization = configuration.quantization,
              quantization.bits > 0,
              quantization.bits <= 8
        else { return }

        func shape(_ key: String) throws -> [Int] {
            guard let entry = headers[key], let raw = entry["shape"] as? [Any] else {
                throw Qwen4ExpCheckpointPreflightError.invalidTensorHeader(key, "unknown")
            }
            return raw.compactMap { ($0 as? NSNumber)?.intValue }
        }

        func packed(_ dimensions: Int) -> Int {
            precondition(dimensions % 32 == 0 || quantization.bits == 8)
            return dimensions * quantization.bits / 32
        }

        func expect(_ key: String, _ expected: [Int]) throws {
            guard headers[key] != nil else { return }
            let actual = try shape(key)
            guard actual == expected else {
                throw Qwen4ExpCheckpointPreflightError.invalidTensorShape(
                    key, expected, actual)
            }
        }

        try expect(
            "language_model.model.embed_tokens.weight",
            [text.vocabSize, packed(text.hiddenSize)])
        try expect(
            "language_model.lm_head.weight",
            [text.vocabSize, packed(text.hiddenSize)])

        let keyDim = text.linearNumKeyHeads * text.linearKeyHeadDim
        let valueDim = text.linearNumValueHeads * text.linearValueHeadDim
        let firstLinear = "language_model.model.layers.0.linear_attn"
        try expect(
            "\(firstLinear).in_proj_qkv.weight",
            [keyDim * 2 + valueDim, packed(text.hiddenSize)])
        try expect(
            "\(firstLinear).in_proj_z.weight",
            [valueDim, packed(text.hiddenSize)])
        try expect(
            "\(firstLinear).out_proj.weight",
            [text.hiddenSize, packed(valueDim)])

        if let fullIndex = text.layerTypes.firstIndex(of: .fullAttention) {
            let prefix = "language_model.model.layers.\(fullIndex).self_attn"
            try expect(
                "\(prefix).q_proj.weight",
                [text.numAttentionHeads * text.headDim * 2, packed(text.hiddenSize)])
            try expect(
                "\(prefix).k_proj.weight",
                [text.numKeyValueHeads * text.headDim, packed(text.hiddenSize)])
        }

        let moePrefix = "language_model.model.layers.0.mlp.switch_mlp"
        try expect(
            "\(moePrefix).gate_proj.weight",
            [text.numExperts, text.moeIntermediateSize, packed(text.hiddenSize)])
        try expect(
            "\(moePrefix).down_proj.weight",
            [text.numExperts, text.hiddenSize, packed(text.moeIntermediateSize)])
    }

    private static func layerIndices(in keys: Set<String>, marker: String) -> Set<Int> {
        Set(keys.compactMap { key in
            guard key.hasPrefix("language_model.model.layers.") else { return nil }
            guard key.hasSuffix(marker) else { return nil }
            let prefix = String(key.dropLast(marker.count))
            guard let number = prefix.split(separator: ".").last else { return nil }
            return Int(number)
        })
    }
}
