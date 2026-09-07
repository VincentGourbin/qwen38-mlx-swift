import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

public enum Qwen38MTPProviderError: LocalizedError, Equatable {
    case unsupportedTargetDirectory(String)
    case incompatible(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedTargetDirectory(let path):
            return "Aucun drafter MTP apparié pour " + path + "."
        case .incompatible(let message):
            return message
        }
    }
}

/// A model object loaded from an MTP container. mlx-swift-lm exposes the
/// container for safe loading, but generation itself needs the model object to
/// remain alive after the container closure returns. Access is serialized by
/// Qwen38Runtime, so this is the narrow, documented Sendable boundary.
public final class Qwen38MTPDrafterBox: @unchecked Sendable {
    public let model: any MTPDrafterModel
    public let directory: URL

    public init(model: any MTPDrafterModel, directory: URL) {
        self.model = model
        self.directory = directory
    }
}

/// Resolves and loads the small checkpoint paired with a local target model.
/// The target and drafter are intentionally kept separate because the public
/// MLX-community conversions do not embed `mtp.*` in the target shards.
public actor Qwen38MTPDrafterProvider {
    private var loaded: Qwen38MTPDrafterBox?
    private var loadedTargetDirectory: URL?

    public init() {}

    public func pairedDirectory(for targetDirectory: URL) -> URL? {
        let name = targetDirectory.lastPathComponent
        let suffix: String
        if name.hasSuffix("-4bit") {
            suffix = "4bit"
        } else if name.hasSuffix("-8bit") {
            suffix = "8bit"
        } else if name.hasSuffix("-bf16") {
            suffix = "bf16"
        } else {
            return nil
        }
        let drafterName = "Qwen3.8-27B-MTP-\(suffix)"
        let candidate = targetDirectory.deletingLastPathComponent()
            .appendingPathComponent(drafterName, isDirectory: true)
        return FileManager.default.fileExists(atPath: candidate.appendingPathComponent("config.json").path)
            ? candidate : nil
    }

    public func loadIfAvailable(
        for targetDirectory: URL
    ) async -> (availability: Qwen38MTPAvailability, box: Qwen38MTPDrafterBox?) {
        if let loaded, loadedTargetDirectory == targetDirectory {
            return (.active, loaded)
        }

        loaded = nil
        loadedTargetDirectory = nil
        guard let directory = pairedDirectory(for: targetDirectory) else {
            return (.unavailable, nil)
        }

        do {
            await Qwen38MTPRegistration.register()
            let box = try loadStandaloneDrafter(from: directory)
            loaded = box
            loadedTargetDirectory = targetDirectory
            return (.active, box)
        } catch {
            return (.fallback("Échec de chargement du drafter MTP : " + error.localizedDescription), nil)
        }
    }

    public func unload() {
        loaded = nil
        loadedTargetDirectory = nil
    }

    /// The published `*-MTP-*` repositories are split checkpoints. Their
    /// tensors are rooted at `fc.*`/`layers.*`, while the upstream model tree
    /// is rooted at `mtp.fc.*`/`mtp.layers.*`. The upstream factory currently
    /// assumes the latter (the embedded-source layout), so normalize the
    /// standalone artifact here before invoking the upstream Qwen sanitizer.
    private nonisolated func loadStandaloneDrafter(
        from directory: URL
    ) throws -> Qwen38MTPDrafterBox {
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let configuration = try JSONDecoder.json5().decode(
            MTPConfigurationEnvelope.self, from: configData)
        let model = Qwen35VLMNextNDraftModel(
            configuration.textConfiguration,
            preconvertedNorms: true
        )

        let urls = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ).filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else {
            throw Qwen38MTPProviderError.incompatible(
                "Le drafter MTP ne contient aucun fichier safetensors.")
        }

        var rawWeights = [String: MLXArray]()
        var metadata = [String: String]()
        for url in urls {
            let (arrays, fileMetadata) = try loadArraysAndMetadata(url: url, stream: .cpu)
            rawWeights.merge(arrays) { _, new in new }
            if metadata.isEmpty { metadata = fileMetadata }
        }

        let prefixed = Dictionary(uniqueKeysWithValues: rawWeights.map { key, value in
            (key.hasPrefix("mtp.") ? key : "mtp." + key, value)
        })
        let weights = model.sanitize(weights: prefixed, metadata: metadata)
        guard !weights.isEmpty else {
            throw Qwen38MTPProviderError.incompatible(
                "Le sanitizer upstream n'a conservé aucun poids MTP.")
        }

        let quantization = try Self.quantization(from: configData)
        if let quantization {
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mode: quantization.mode
            ) { path, _ in
                weights["\(path).scales"] != nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.all]
        )
        eval(model.parameters())
        return Qwen38MTPDrafterBox(model: model, directory: directory)
    }

    private struct QuantizationSpec {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode
    }

    private struct MTPConfigurationEnvelope: Decodable {
        let textConfiguration: Qwen35Configuration.TextConfiguration

        enum CodingKeys: String, CodingKey {
            case textConfiguration = "text_config"
        }
    }

    private struct QuantizationEnvelope: Decodable {
        let quantization: QuantizationSpecJSON?
    }

    private struct QuantizationSpecJSON: Decodable {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode?

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    private nonisolated static func quantization(from data: Data) throws -> QuantizationSpec? {
        let envelope = try JSONDecoder.json5().decode(QuantizationEnvelope.self, from: data)
        guard let value = envelope.quantization else { return nil }
        return QuantizationSpec(
            groupSize: value.groupSize,
            bits: value.bits,
            mode: value.mode ?? .affine
        )
    }
}
