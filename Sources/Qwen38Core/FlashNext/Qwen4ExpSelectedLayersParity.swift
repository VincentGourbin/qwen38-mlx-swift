import Foundation
import MLX

public struct Qwen4ExpParityMetrics: Sendable, Equatable {
    public let maxAbsoluteError: Float
    public let rmsError: Float
    public let relativeRMSError: Float
    public let cosineSimilarity: Float

    public init(
        maxAbsoluteError: Float,
        rmsError: Float,
        relativeRMSError: Float,
        cosineSimilarity: Float
    ) {
        self.maxAbsoluteError = maxAbsoluteError
        self.rmsError = rmsError
        self.relativeRMSError = relativeRMSError
        self.cosineSimilarity = cosineSimilarity
    }
}

public struct Qwen4ExpLogitParityMetrics: Sendable, Equatable {
    public let values: Qwen4ExpParityMetrics
    public let pythonTop1Margin: Float
    public let pythonTopToken: Int32
    public let swiftTopToken: Int32
    public let swiftTokenRankInPython: Int
    public let pythonTokenRankInSwift: Int

    public init(
        values: Qwen4ExpParityMetrics,
        pythonTop1Margin: Float,
        pythonTopToken: Int32,
        swiftTopToken: Int32,
        swiftTokenRankInPython: Int,
        pythonTokenRankInSwift: Int
    ) {
        self.values = values
        self.pythonTop1Margin = pythonTop1Margin
        self.pythonTopToken = pythonTopToken
        self.swiftTopToken = swiftTopToken
        self.swiftTokenRankInPython = swiftTokenRankInPython
        self.pythonTokenRankInSwift = pythonTokenRankInSwift
    }
}

public struct Qwen4ExpSelectedLayersParityReport: Sendable, Equatable {
    public let layerMaxAbsoluteError: [Int: Float]
    public let chainedLayerMetrics: [Int: Qwen4ExpParityMetrics]
    public let reanchoredLayerMetrics: [Int: Qwen4ExpParityMetrics]
    public let mixerFromPythonMetrics: Qwen4ExpParityMetrics
    public let logitsFromPythonMetrics: Qwen4ExpLogitParityMetrics
    public let reducedMaxAbsoluteError: Float
    public let logitsMaxAbsoluteError: Float
    public let swiftNextToken: Int32
    public let pythonNextToken: Int32

    public var tokensMatch: Bool { swiftNextToken == pythonNextToken }
}

/// Compares the streaming Swift decoder with a short Python layer chain.
public enum Qwen4ExpSelectedLayersParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL
    ) throws -> Qwen4ExpSelectedLayersParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let inputIDs = arrays["input_ids"],
              let expectedReduced = arrays["reduced"],
              let expectedLogits = arrays["logits"] else {
            throw Qwen4ExpSelectedLayersParityError.missingTensor
        }
        let layerIndices = arrays.keys.compactMap { key -> Int? in
            guard key.hasPrefix("layer_") else { return nil }
            return Int(key.dropFirst("layer_".count))
        }.sorted()
        guard !layerIndices.isEmpty else {
            throw Qwen4ExpSelectedLayersParityError.missingLayer
        }

        let global = try Qwen4ExpGlobalCheckpointLoader.load(
            from: modelDirectory, materialize: true).model
        let decoder = try Qwen4ExpStreamingDecoder(directory: modelDirectory)
        var hidden = tiled(global.embed(inputIDs), repetitions: [1, 1, 4])
        var errors = [Int: Float]()
        var chainedMetrics = [Int: Qwen4ExpParityMetrics]()
        for layerIndex in layerIndices {
            let result = try decoder.forward(
                hidden,
                inputIDs: inputIDs,
                layerIndices: [layerIndex],
                materializeLayers: true)
            hidden = result.output
            guard let expected = arrays["layer_\(layerIndex)"] else {
                throw Qwen4ExpSelectedLayersParityError.missingLayer
            }
            let metrics = metrics(actual: hidden, expected: expected)
            errors[layerIndex] = metrics.maxAbsoluteError
            chainedMetrics[layerIndex] = metrics
        }

        // E2: feed each layer the exact Python output of its predecessor. This
        // removes the feedback amplification from the experiment and tells us
        // whether a layer is intrinsically wrong or merely receives drift.
        let rebasedDecoder = try Qwen4ExpStreamingDecoder(directory: modelDirectory)
        var reanchoredMetrics = [Int: Qwen4ExpParityMetrics]()
        for layerIndex in layerIndices {
            let layerInput: MLXArray
            if layerIndex == layerIndices.first! {
                layerInput = tiled(global.embed(inputIDs), repetitions: [1, 1, 4])
            } else {
                guard let predecessor = arrays["layer_\(layerIndex - 1)"] else {
                    throw Qwen4ExpSelectedLayersParityError.missingLayer
                }
                layerInput = predecessor
            }
            let result = try rebasedDecoder.forward(
                layerInput,
                inputIDs: inputIDs,
                layerIndices: [layerIndex],
                materializeLayers: true)
            guard let expected = arrays["layer_\(layerIndex)"] else {
                throw Qwen4ExpSelectedLayersParityError.missingLayer
            }
            reanchoredMetrics[layerIndex] = metrics(
                actual: result.output, expected: expected)
        }

        // E1: use the Python layer_3 state as input to the Swift global
        // mixer/lm-head. A clean result isolates the layer chain from global
        // checkpoint loading and final reduction.
        guard let pythonLastLayer = arrays["layer_\(layerIndices.last!)"] else {
            throw Qwen4ExpSelectedLayersParityError.missingLayer
        }
        let mixerFromPython = global.reduceHyperStreams(pythonLastLayer)
        let logitsFromPython = global.logits(from: mixerFromPython)
        eval(mixerFromPython, logitsFromPython)
        let mixerMetrics = metrics(actual: mixerFromPython, expected: expectedReduced)
        let logitsMetrics = logitMetrics(actual: logitsFromPython, expected: expectedLogits)

        let reduced = global.reduceHyperStreams(hidden)
        let logits = global.logits(from: reduced)
        eval(reduced, logits)
        let reducedError = metrics(actual: reduced, expected: expectedReduced)
        let logitsError = logitMetrics(actual: logits, expected: expectedLogits)
        return .init(
            layerMaxAbsoluteError: errors,
            chainedLayerMetrics: chainedMetrics,
            reanchoredLayerMetrics: reanchoredMetrics,
            mixerFromPythonMetrics: mixerMetrics,
            logitsFromPythonMetrics: logitsMetrics,
            reducedMaxAbsoluteError: reducedError.maxAbsoluteError,
            logitsMaxAbsoluteError: logitsError.values.maxAbsoluteError,
            swiftNextToken: logitsError.swiftTopToken,
            pythonNextToken: logitsError.pythonTopToken)
    }

    private static func metrics(
        actual: MLXArray, expected: MLXArray
    ) -> Qwen4ExpParityMetrics {
        precondition(actual.shape == expected.shape)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        precondition(actualValues.count == expectedValues.count)
        var maxError: Float = 0
        var sumErrorSquared: Float = 0
        var sumExpectedSquared: Float = 0
        var dot: Float = 0
        for (actualValue, expectedValue) in zip(actualValues, expectedValues) {
            let delta = actualValue - expectedValue
            maxError = max(maxError, abs(delta))
            sumErrorSquared += delta * delta
            sumExpectedSquared += expectedValue * expectedValue
            dot += actualValue * expectedValue
        }
        let count = Float(max(1, actualValues.count))
        let rmsError = (sumErrorSquared / count).squareRoot()
        let expectedRMS = (sumExpectedSquared / count).squareRoot()
        let relative = expectedRMS > 0 ? rmsError / expectedRMS : rmsError
        let actualNorm = sum(actualValues.map { $0 * $0 }).squareRoot()
        let expectedNorm = sum(expectedValues.map { $0 * $0 }).squareRoot()
        let cosine = actualNorm > 0 && expectedNorm > 0
            ? dot / (actualNorm * expectedNorm) : 1
        return .init(
            maxAbsoluteError: maxError,
            rmsError: rmsError,
            relativeRMSError: relative,
            cosineSimilarity: cosine)
    }

    private static func logitMetrics(
        actual: MLXArray, expected: MLXArray
    ) -> Qwen4ExpLogitParityMetrics {
        precondition(actual.ndim == 3 && expected.ndim == 3)
        let values = metrics(actual: actual, expected: expected)
        let actualLast = actual[0..., -1, 0...].asType(.float32).asArray(Float.self)
        let expectedLast = expected[0..., -1, 0...].asType(.float32).asArray(Float.self)
        let pythonTop = argMax(expectedLast)
        let swiftTop = argMax(actualLast)
        let sorted = expectedLast.sorted(by: >)
        let margin = sorted.count > 1 ? sorted[0] - sorted[1] : 0
        let swiftValueInPython = expectedLast[swiftTop]
        let pythonValueInSwift = actualLast[pythonTop]
        let swiftRank = 1 + expectedLast.reduce(into: 0) {
            if $1 > swiftValueInPython { $0 += 1 }
        }
        let pythonRank = 1 + actualLast.reduce(into: 0) {
            if $1 > pythonValueInSwift { $0 += 1 }
        }
        return .init(
            values: values,
            pythonTop1Margin: margin,
            pythonTopToken: Int32(pythonTop),
            swiftTopToken: Int32(swiftTop),
            swiftTokenRankInPython: swiftRank,
            pythonTokenRankInSwift: pythonRank)
    }

    private static func argMax(_ values: [Float]) -> Int {
        guard let first = values.first else { return 0 }
        var bestIndex = 0
        var best = first
        for (index, value) in values.dropFirst().enumerated() where value > best {
            bestIndex = index + 1
            best = value
        }
        return bestIndex
    }

    private static func sum(_ values: [Float]) -> Float {
        values.reduce(0, +)
    }
}

public enum Qwen4ExpSelectedLayersParityError: LocalizedError, Equatable {
    case missingTensor
    case missingLayer

    public var errorDescription: String? {
        switch self {
        case .missingTensor:
            return "Fixture multi-couches incomplet : reduced ou logits absent."
        case .missingLayer:
            return "Fixture multi-couches incomplet : une sortie layer_N est absente."
        }
    }
}
