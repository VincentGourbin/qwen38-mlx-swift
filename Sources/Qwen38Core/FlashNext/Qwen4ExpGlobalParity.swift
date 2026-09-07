import Foundation
import MLX

public struct Qwen4ExpGlobalParityReport: Sendable, Equatable {
    public let embeddedMaxAbsoluteError: Float
    public let reducedMaxAbsoluteError: Float
    public let logitsMaxAbsoluteError: Float
    public let embeddedMeanAbsoluteError: Float
    public let reducedMeanAbsoluteError: Float
    public let logitsMeanAbsoluteError: Float

    public init(
        embeddedMaxAbsoluteError: Float,
        reducedMaxAbsoluteError: Float,
        logitsMaxAbsoluteError: Float,
        embeddedMeanAbsoluteError: Float,
        reducedMeanAbsoluteError: Float,
        logitsMeanAbsoluteError: Float
    ) {
        self.embeddedMaxAbsoluteError = embeddedMaxAbsoluteError
        self.reducedMaxAbsoluteError = reducedMaxAbsoluteError
        self.logitsMaxAbsoluteError = logitsMaxAbsoluteError
        self.embeddedMeanAbsoluteError = embeddedMeanAbsoluteError
        self.reducedMeanAbsoluteError = reducedMeanAbsoluteError
        self.logitsMeanAbsoluteError = logitsMeanAbsoluteError
    }
}

/// Compares the resident global Flash-Next weights with a Python MLX fixture.
public enum Qwen4ExpGlobalParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL
    ) throws -> Qwen4ExpGlobalParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let hidden = arrays["hidden"], let inputIDs = arrays["input_ids"],
              let expectedEmbedded = arrays["embedded"],
              let expectedReduced = arrays["reduced"],
              let expectedLogits = arrays["logits"] else {
            throw Qwen4ExpGlobalParityError.missingTensor
        }

        let loaded = try Qwen4ExpGlobalCheckpointLoader.load(
            from: modelDirectory, materialize: true)
        let model = loaded.model
        let embedded = model.embed(inputIDs)
        let reduced = model.reduceHyperStreams(hidden)
        let logits = model.logits(from: reduced)
        eval(embedded, reduced, logits)

        let errors = [
            difference(actual: embedded, expected: expectedEmbedded),
            difference(actual: reduced, expected: expectedReduced),
            difference(actual: logits, expected: expectedLogits),
        ]
        return .init(
            embeddedMaxAbsoluteError: errors[0].max,
            reducedMaxAbsoluteError: errors[1].max,
            logitsMaxAbsoluteError: errors[2].max,
            embeddedMeanAbsoluteError: errors[0].mean,
            reducedMeanAbsoluteError: errors[1].mean,
            logitsMeanAbsoluteError: errors[2].mean)
    }

    private static func difference(
        actual: MLXArray, expected: MLXArray
    ) -> (max: Float, mean: Float) {
        precondition(actual.shape == expected.shape)
        let delta = abs(actual.asType(.float32) - expected.asType(.float32))
        eval(delta)
        return (
            delta.max().item(Float.self),
            delta.sum().item(Float.self) / Float(delta.size))
    }
}

public enum Qwen4ExpGlobalParityError: LocalizedError, Equatable {
    case missingTensor

    public var errorDescription: String? {
        "Fixture global incomplet : hidden, input_ids, embedded, reduced ou logits absent."
    }
}
