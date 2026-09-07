import Foundation
import MLX

/// Numeric comparison for the weight-free QSA fixtures emitted by
/// `Scripts/qwen4-exp-qsa-reference.py`.
public struct Qwen4ExpQSAParityReport: Sendable, Equatable {
    public let maxAbsoluteError: [String: Float]
    public let meanAbsoluteError: [String: Float]

    public var worstMaxAbsoluteError: Float {
        maxAbsoluteError.values.max() ?? 0
    }
}

public enum Qwen4ExpQSAParity {
    /// Load a Python fixture and recompute all observable QSA stages in Swift.
    public static func compareFixture(
        at url: URL,
        indexer: Qwen4ExpQSAIndexer,
        budget: Int,
        compressRatio: Int
    ) throws -> Qwen4ExpQSAParityReport {
        let arrays = try loadArrays(url: url, stream: .cpu)
        let required = ["projected", "position_ids", "queries", "raw_keys", "pooled_keys", "scores", "mask"]
        guard required.allSatisfy({ arrays[$0] != nil }) else {
            throw Qwen4ExpQSAParityError.missingTensor
        }

        let projected = arrays["projected"]!
        let positions = arrays["position_ids"]!
        let split = indexer.split(projected)
        let queries = indexer.rope.apply(split.queries, positionIDs: positions)
        let pooled = indexer.pooledKeys(split.rawKeys, compressRatio: compressRatio)
        let blockPositions = positions[.ellipsis, .stride(
            from: 0, to: split.rawKeys.dim(1) / compressRatio * compressRatio,
            by: compressRatio)]
        let rotatedPooled = indexer.rope.apply(pooled, positionIDs: blockPositions)
        let scores = Qwen4ExpQSAMask.blockScores(query: queries, pooledKeys: rotatedPooled)
        guard let mask = indexer.makeMask(
            fromProjected: projected,
            positionIDs: positions,
            cache: nil,
            compressRatio: compressRatio,
            budget: budget) else {
            throw Qwen4ExpQSAParityError.noSparseMask
        }

        let actual: [(String, MLXArray, MLXArray)] = [
            ("queries", arrays["queries"]!, queries),
            ("raw_keys", arrays["raw_keys"]!, split.rawKeys),
            ("pooled_keys", arrays["pooled_keys"]!, rotatedPooled),
            ("scores", arrays["scores"]!, scores),
            ("mask", arrays["mask"]!, mask.asType(.uint8))
        ]
        var maxErrors = [String: Float]()
        var meanErrors = [String: Float]()
        for (name, expected, value) in actual {
            guard expected.shape == value.shape else {
                throw Qwen4ExpQSAParityError.shapeMismatch(
                    name: name, expected: expected.shape, actual: value.shape)
            }
            let difference = abs(expected.asType(.float32) - value.asType(.float32))
            eval(difference)
            maxErrors[name] = difference.max().item(Float.self)
            meanErrors[name] = difference.sum().item(Float.self) / Float(difference.size)
        }
        return Qwen4ExpQSAParityReport(
            maxAbsoluteError: maxErrors, meanAbsoluteError: meanErrors)
    }
}

public enum Qwen4ExpQSAParityError: LocalizedError, Equatable {
    case missingTensor
    case noSparseMask
    case shapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingTensor: return "Fixture QSA incomplet : tenseur intermédiaire absent."
        case .noSparseMask: return "Le fixture ne produit pas de masque sparse avec ce budget."
        case .shapeMismatch(let name, let expected, let actual):
            return "Forme QSA différente pour \(name) : \(expected) vs \(actual)."
        }
    }
}
