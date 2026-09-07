import Foundation
import MLX

public struct Qwen4ExpVisionParityReport: Sendable, Equatable {
    public let outputShape: [Int]
    public let maxAbsoluteError: Float
    public let meanAbsoluteError: Float
    public let stageMaxAbsoluteError: [String: Float]

    public init(
        outputShape: [Int], maxAbsoluteError: Float, meanAbsoluteError: Float,
        stageMaxAbsoluteError: [String: Float]
    ) {
        self.outputShape = outputShape
        self.maxAbsoluteError = maxAbsoluteError
        self.meanAbsoluteError = meanAbsoluteError
        self.stageMaxAbsoluteError = stageMaxAbsoluteError
    }
}

public enum Qwen4ExpVisionParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL
    ) throws -> Qwen4ExpVisionParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let pixels = arrays["pixels"] else {
            throw Qwen4ExpVisionParityError.missingTensor
        }
        let loaded = try Qwen4ExpVisionCheckpointLoader.load(from: modelDirectory)
        let trace = loaded.model.forwardTrace(pixels)
        var stageErrors = [String: Float]()
        var outputShape = [Int]()
        var outputMean: Float = 0
        for name in arrays.keys where name != "pixels" {
            guard let expectedStage = arrays[name], let actualStage = trace[name] else {
                throw Qwen4ExpVisionParityError.missingStage(name)
            }
            // Python's tower returns [tokens, hidden], while Swift retains a
            // leading batch dimension for multimodal fusion.
            let comparable = actualStage.ndim == expectedStage.ndim + 1
                && actualStage.dim(0) == 1
                ? actualStage.squeezed(axis: 0) : actualStage
            guard expectedStage.shape == comparable.shape else {
                throw Qwen4ExpVisionParityError.shapeMismatch(
                    expected: expectedStage.shape, actual: comparable.shape)
            }
            let difference = abs(expectedStage.asType(.float32) - comparable.asType(.float32))
            eval(difference)
            stageErrors[name] = difference.max().item(Float.self)
            if name == "output" {
                outputShape = actualStage.shape
                outputMean = difference.sum().item(Float.self) / Float(difference.size)
            }
        }
        return .init(
            outputShape: outputShape,
            maxAbsoluteError: stageErrors["output"] ?? 0,
            meanAbsoluteError: outputMean,
            stageMaxAbsoluteError: stageErrors)
    }
}

public enum Qwen4ExpVisionParityError: LocalizedError, Equatable {
    case missingTensor
    case missingStage(String)
    case shapeMismatch(expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingTensor:
            return "Fixture vision incomplet : pixels ou sortie Python absent."
        case .missingStage(let name):
            return "Étape vision absente dans le fixture ou le runtime : \(name)."
        case .shapeMismatch(let expected, let actual):
            return "Forme vision différente : \(expected) vs \(actual)."
        }
    }
}
