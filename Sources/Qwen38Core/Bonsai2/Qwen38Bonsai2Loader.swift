import Foundation
import MLX
import MLXNN
import MLXLMCommon

public enum Qwen38Bonsai2LoaderError: LocalizedError, Equatable {
    case unsupportedBlock(Int)
    case unsupportedDtype(String)
    case missingSigns(String)
    case invalidSigns(String)
    case unexpectedReplacementCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBlock(let block):
            return "Bonsai 2 : bloc Hadamard non supporté (\(block))"
        case .unsupportedDtype(let dtype):
            return "Bonsai 2 : dtype de module non supporté (\(dtype))"
        case .missingSigns(let path):
            return "Bonsai 2 : vecteur de signes absent pour \(path)"
        case .invalidSigns(let path):
            return "Bonsai 2 : vecteur de signes non ±1 pour \(path)"
        case .unexpectedReplacementCount(let expected, let actual):
            return "Bonsai 2 : \(actual) modules remplacés, \(expected) attendus"
        }
    }
}

/// Reads `config.json`'s `modules[]` and the `.signs` vectors from
/// `model.safetensors`, then replaces the 402 packed `QuantizedLinear` /
/// `QuantizedEmbedding` modules of a loaded Bonsai 2 checkpoint with their
/// Hadamard-rotated equivalents (docs/bonsai2/plan.md, fiche B-2).
/// `@unchecked Sendable` for the same "MLX arrays are confined to the
/// owning runtime" reason as every other Flash-Next snapshot type
/// (`Qwen38FlashConversationState`): `signs` is read-only after `init` and
/// only ever touched from within a single `ModelContainer.perform` call.
public struct Qwen38Bonsai2Loader: @unchecked Sendable {
    private struct ModuleEntry: Decodable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }

    private struct Quantization: Decodable {
        let mode: String

        enum CodingKeys: String, CodingKey {
            case mode
        }
    }

    private struct Config: Decodable {
        let modules: [ModuleEntry]
        let quantization: Quantization
    }

    public let expectedReplacementCount: Int
    private let entries: [ModuleEntry]
    private let signs: [String: MLXArray]

    public init(directory: URL) throws {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
        // `quantization.mode` ("affine") is not consumed here: the packed
        // modules keep it via their existing QuantizedLinear/QuantizedEmbedding
        // `mode`, copied by Qwen38HadamardQuantizedLinear/Embedding.
        guard QuantizationMode(rawValue: config.quantization.mode) != nil else {
            throw Qwen38Bonsai2LoaderError.unsupportedDtype(config.quantization.mode)
        }
        for entry in config.modules {
            guard [512, 1024, 2048, 4096].contains(entry.block) else {
                throw Qwen38Bonsai2LoaderError.unsupportedBlock(entry.block)
            }
            guard entry.dtype == "float16" else {
                throw Qwen38Bonsai2LoaderError.unsupportedDtype(entry.dtype)
            }
        }

        // `loadArrays` is lazy: only the `.signs` entries filtered below are
        // ever evaluated, the 1 988 weight/scales/biases tensors in the same
        // file stay as unevaluated graph nodes and are dropped with `all`.
        let safetensorsURL = directory.appendingPathComponent("model.safetensors")
        let all = try loadArrays(url: safetensorsURL)
        var signsByPath = [String: MLXArray]()
        for (key, value) in all where key.hasSuffix(".signs") {
            let modulePath = String(key.dropLast(".signs".count))
            guard allClose(abs(value), MLXArray(1.0)).all().item(Bool.self) else {
                throw Qwen38Bonsai2LoaderError.invalidSigns(modulePath)
            }
            signsByPath[modulePath] = value
        }

        for entry in config.modules {
            let modulePath = "language_model." + entry.path
            guard signsByPath[modulePath] != nil else {
                throw Qwen38Bonsai2LoaderError.missingSigns(modulePath)
            }
        }

        entries = config.modules
        signs = signsByPath
        expectedReplacementCount = config.modules.count
    }

    /// Replaces the packed modules in place. Synchronous and non-`throws`:
    /// all fallible analysis (config, signs) already happened in `init`. A
    /// mismatch here (module missing from the live tree, wrong runtime
    /// type) means the checkpoint does not match the `config.json` we just
    /// validated it against, which is a data-integrity bug, not a normal
    /// error path — hence `fatalError` rather than `throw`. The caller
    /// checks the returned count against `expectedReplacementCount`.
    @discardableResult
    public func install(into model: Module) -> Int {
        precondition(
            ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"] == "0",
            "MLX_QWEN_FOUR_GDN must be \"0\" before installing Bonsai 2 Hadamard modules")

        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var updates = [(String, Module)]()
        updates.reserveCapacity(entries.count)

        for entry in entries {
            let modulePath = "language_model." + entry.path
            guard let signsVector = signs[modulePath] else {
                fatalError("Bonsai 2 : vecteur de signes absent pour \(modulePath)")
            }
            guard let current = leaves[modulePath] else {
                fatalError("Bonsai 2 : module introuvable à \(modulePath)")
            }
            let replacement: Module
            if entry.embedding {
                guard let embedding = current as? QuantizedEmbedding else {
                    fatalError("Bonsai 2 : \(modulePath) n'est pas un QuantizedEmbedding")
                }
                replacement = Qwen38HadamardQuantizedEmbedding(
                    embedding, signs: signsVector, block: entry.block)
            } else {
                guard let linear = current as? QuantizedLinear else {
                    fatalError("Bonsai 2 : \(modulePath) n'est pas un QuantizedLinear")
                }
                replacement = Qwen38HadamardQuantizedLinear(
                    linear, signs: signsVector, block: entry.block)
            }
            updates.append((modulePath, replacement))
        }

        model.update(modules: ModuleChildren.unflattened(updates))
        return updates.count
    }
}
