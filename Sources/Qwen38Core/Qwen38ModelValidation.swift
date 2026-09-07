import Foundation

public enum Qwen38ModelValidationError: LocalizedError, Equatable {
    case missingConfig(URL)
    case invalidJSON(URL)
    case unsupportedModelType(String?)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url): return "config.json absent: \(url.path)"
        case .invalidJSON(let url): return "config.json invalide: \(url.path)"
        case .unsupportedModelType(let type):
            return "model_type non supporté pour Qwen3.8: \(type ?? "absent")"
        }
    }
}

public enum Qwen38ModelFamily: String, Sendable, Equatable, Codable {
    case qwen35 = "qwen3_5"
    case qwen4Exp = "qwen4_exp"
}

public struct Qwen38ModelInfo: Decodable, Sendable, Equatable {
    public let modelType: String
    public let architecture: String?
    public let hiddenSize: Int?
    public let numHiddenLayers: Int?

    public var family: Qwen38ModelFamily? { Qwen38ModelFamily(rawValue: modelType) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try values.decode(String.self, forKey: .modelType)
        architecture = try values.decodeIfPresent([String].self, forKey: .architectures)?.first
        let textConfig = try values.decodeIfPresent(TextConfig.self, forKey: .textConfig)
        hiddenSize = try values.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? textConfig?.hiddenSize
        numHiddenLayers = try values.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? textConfig?.numHiddenLayers
    }

    private struct TextConfig: Decodable {
        let hiddenSize: Int?
        let numHiddenLayers: Int?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
        }
    }
}

public enum Qwen38ModelValidator {
    public static func readInfo(from directory: URL) throws -> Qwen38ModelInfo {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Qwen38ModelValidationError.missingConfig(configURL)
        }
        do {
            return try JSONDecoder().decode(Qwen38ModelInfo.self, from: Data(contentsOf: configURL))
        } catch let error as Qwen38ModelValidationError {
            throw error
        } catch {
            throw Qwen38ModelValidationError.invalidJSON(configURL)
        }
    }

    public static func validate(_ directory: URL) throws -> Qwen38ModelInfo {
        let info = try readInfo(from: directory)
        guard let family = info.family else {
            throw Qwen38ModelValidationError.unsupportedModelType(info.modelType)
        }
        if family == .qwen4Exp {
            try Qwen4ExpConfiguration.load(from: directory).validate()
        }
        return info
    }
}
