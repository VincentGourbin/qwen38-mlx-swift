import Foundation
import MLX

public struct Qwen4ExpMRoPEParityReport: Sendable, Equatable {
    public let maxAbsoluteError: [String: Float]
    public let meanAbsoluteError: [String: Float]

    public var worstMaxAbsoluteError: Float {
        maxAbsoluteError.values.max() ?? 0
    }
}

public enum Qwen4ExpMRoPEParity {
    public static func compareFixture(at url: URL) throws -> Qwen4ExpMRoPEParityReport {
        let arrays = try loadArrays(url: url, stream: .cpu)
        guard let inputIDs = arrays["input_ids"],
              let expectedPositions = arrays["position_ids"],
              let expectedCos = arrays["cos"],
              let expectedSin = arrays["sin"]
        else {
            throw Qwen4ExpMRoPEParityError.missingTensor
        }

        let ids = inputIDs.flattened().asArray(Int32.self)
        let positions = try Qwen4ExpMRoPE.multimodalPositionIDs(
            inputIDs: ids,
            imageTokenID: 99,
            visionStartTokenID: 98,
            grids: [.init(temporal: 1, height: 4, width: 6)],
            spatialMergeSize: 2)
        let tables = Qwen4ExpMRoPE().computeCosSin(positionIDs: positions)
        let actual: [(String, MLXArray, MLXArray)] = [
            ("position_ids", expectedPositions, positions),
            ("cos", expectedCos, tables.cos),
            ("sin", expectedSin, tables.sin),
        ]

        var maximums = [String: Float]()
        var means = [String: Float]()
        for (name, expected, value) in actual {
            guard expected.shape == value.shape else {
                throw Qwen4ExpMRoPEParityError.shapeMismatch(
                    name: name, expected: expected.shape, actual: value.shape)
            }
            let difference = abs(expected.asType(.float32) - value.asType(.float32))
            eval(difference)
            maximums[name] = difference.max().item(Float.self)
            means[name] = difference.sum().item(Float.self) / Float(difference.size)
        }
        return .init(maxAbsoluteError: maximums, meanAbsoluteError: means)
    }
}

public enum Qwen4ExpMRoPEParityError: LocalizedError, Equatable {
    case missingTensor
    case shapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingTensor:
            return "Fixture MRoPE incomplet : tenseur intermédiaire absent."
        case .shapeMismatch(let name, let expected, let actual):
            return "Forme MRoPE différente pour \(name) : \(expected) vs \(actual)."
        }
    }
}
