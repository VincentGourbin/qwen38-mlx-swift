import Foundation
import MLX
import MLXLMCommon

public struct Qwen4ExpPublicLayerParityReport: Sendable, Equatable {
    public let layerIndex: Int
    public let embeddedMaxAbsoluteError: Float
    public let outputMaxAbsoluteError: Float
    public let embeddedMeanAbsoluteError: Float
    public let outputMeanAbsoluteError: Float
    public let stageMaxAbsoluteError: [String: Float]
    public let stageMetrics: [String: Qwen4ExpParityMetrics]
    public let moeRoutingMetrics: Qwen4ExpMoERoutingParityMetrics?

    public init(
        layerIndex: Int,
        embeddedMaxAbsoluteError: Float,
        outputMaxAbsoluteError: Float,
        embeddedMeanAbsoluteError: Float,
        outputMeanAbsoluteError: Float,
        stageMaxAbsoluteError: [String: Float],
        stageMetrics: [String: Qwen4ExpParityMetrics],
        moeRoutingMetrics: Qwen4ExpMoERoutingParityMetrics? = nil
    ) {
        self.layerIndex = layerIndex
        self.embeddedMaxAbsoluteError = embeddedMaxAbsoluteError
        self.outputMaxAbsoluteError = outputMaxAbsoluteError
        self.embeddedMeanAbsoluteError = embeddedMeanAbsoluteError
        self.outputMeanAbsoluteError = outputMeanAbsoluteError
        self.stageMaxAbsoluteError = stageMaxAbsoluteError
        self.stageMetrics = stageMetrics
        self.moeRoutingMetrics = moeRoutingMetrics
    }
}

/// Compares expert membership rather than the numeric order of an
/// `argPartition` slice. The order of the selected experts is unspecified;
/// two equal top-k sets are therefore equivalent even when their arrays differ.
public struct Qwen4ExpMoERoutingParityMetrics: Sendable, Equatable {
    public let positions: Int
    public let positionsWithDifferentMembership: Int
    public let differingExpertAssignments: Int
    public let topK: Int

    public init(
        positions: Int,
        positionsWithDifferentMembership: Int,
        differingExpertAssignments: Int,
        topK: Int
    ) {
        self.positions = positions
        self.positionsWithDifferentMembership = positionsWithDifferentMembership
        self.differingExpertAssignments = differingExpertAssignments
        self.topK = topK
    }
}

/// Compares the public Swift decoder-layer call with a Python MLX fixture.
///
/// The fixture intentionally starts from the shared embedding rather than a
/// full 48-layer forward. This validates each attention family independently
/// while retaining the layer-at-a-time memory bound of the real checkpoint.
public enum Qwen4ExpPublicLayerParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL,
        dequantized: Bool = false
    ) throws -> Qwen4ExpPublicLayerParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let inputIDs = arrays["input_ids"],
              let expectedEmbedded = arrays["embedded"],
              let expectedOutput = arrays["layer_output"],
              let layerIndexArray = arrays["layer_index"] else {
            throw Qwen4ExpPublicLayerParityError.missingTensor
        }
        let layerIndex = Int(layerIndexArray.item(Int32.self))

        let global = try Qwen4ExpGlobalCheckpointLoader.load(
            from: modelDirectory, materialize: true)
        let loaded = try Qwen4ExpCheckpointLayerLoader.load(
            layerIndex,
            from: modelDirectory,
            materialize: true,
            useCheckpointQuantization: !dequantized)
        loaded.layer.setParityCapture(true)
        let embedded = global.model.embed(inputIDs)
        // E2 fixtures provide the exact Python output of the preceding layer.
        // The ordinary public probe has no `hidden` tensor and therefore keeps
        // the historical embedding-derived input.
        let hidden = arrays["hidden"] ?? tiled(embedded, repetitions: [1, 1, 4])
        let cache: any KVCache
        if loaded.layer.isLinear {
            cache = ArraysCache(size: loaded.layer.ple == nil ? 2 : 4)
        } else {
            cache = Qwen4ExpQSAKVCache(
                budget: global.model.configuration.indexerBudget,
                compressRatio: global.model.configuration.indexerCompressRatio)
        }
        let output = loaded.layer(
            hidden,
            inputIDs: inputIDs,
            mask: nil,
            cache: cache,
            positionIDs: nil)
        eval(embedded, output)

        let embeddedError = difference(actual: embedded, expected: expectedEmbedded)
        let outputError = difference(actual: output, expected: expectedOutput)
        var stageErrors = [String: Float]()
        var stageMetrics = [String: Qwen4ExpParityMetrics]()
        if let linearAttn = loaded.layer.linearAttn {
            for (name, actual) in linearAttn.lastParityCapture {
                guard let expected = arrays[name] else { continue }
                let values = parityMetrics(actual: actual, expected: expected)
                stageErrors[name] = values.maxAbsoluteError
                stageMetrics[name] = values
            }
        }
        if let selfAttn = loaded.layer.selfAttn {
            for (name, actual) in selfAttn.lastParityCapture {
                guard let expected = arrays[name] else { continue }
                let values = parityMetrics(actual: actual, expected: expected)
                stageErrors[name] = values.maxAbsoluteError
                stageMetrics[name] = values
            }
        }
        for (name, actual) in loaded.layer.attnHyperConnection.lastParityCapture {
            guard let expected = arrays[name] else { continue }
            let values = parityMetrics(actual: actual, expected: expected)
            stageErrors[name] = values.maxAbsoluteError
            stageMetrics[name] = values
        }
        for (name, actual) in loaded.layer.mlpHyperConnection.lastParityCapture {
            guard let expected = arrays[name] else { continue }
            let values = parityMetrics(actual: actual, expected: expected)
            stageErrors[name] = values.maxAbsoluteError
            stageMetrics[name] = values
        }
        for (name, actual) in loaded.layer.mlp.lastParityCapture {
            guard let expected = arrays[name] else { continue }
            let values = parityMetrics(actual: actual, expected: expected)
            stageErrors[name] = values.maxAbsoluteError
            stageMetrics[name] = values
        }
        let moeRoutingMetrics: Qwen4ExpMoERoutingParityMetrics?
        if let actualIndices = loaded.layer.mlp.lastParityCapture["moe_indices"],
           let expectedIndices = arrays["moe_indices"] {
            moeRoutingMetrics = routingMetrics(
                actual: actualIndices, expected: expectedIndices)
        } else {
            moeRoutingMetrics = nil
        }
        if let moeInput = arrays["moe_input"],
           let expectedMoeOutput = arrays["moe_output"] {
            let isolatedMoeOutput = loaded.layer.mlp(moeInput)
            eval(isolatedMoeOutput)
            let values = parityMetrics(actual: isolatedMoeOutput, expected: expectedMoeOutput)
            stageErrors["isolated_moe_output"] = values.maxAbsoluteError
            stageMetrics["isolated_moe_output"] = values
        }
        return Qwen4ExpPublicLayerParityReport(
            layerIndex: layerIndex,
            embeddedMaxAbsoluteError: embeddedError.max,
            outputMaxAbsoluteError: outputError.max,
            embeddedMeanAbsoluteError: embeddedError.mean,
            outputMeanAbsoluteError: outputError.mean,
            stageMaxAbsoluteError: stageErrors,
            stageMetrics: stageMetrics,
            moeRoutingMetrics: moeRoutingMetrics)
    }

    private static func parityMetrics(
        actual: MLXArray, expected: MLXArray
    ) -> Qwen4ExpParityMetrics {
        precondition(actual.shape == expected.shape)
        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
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
        let actualNorm = (actualValues.reduce(0) { $0 + $1 * $1 }).squareRoot()
        let expectedNorm = (expectedValues.reduce(0) { $0 + $1 * $1 }).squareRoot()
        let cosine = actualNorm > 0 && expectedNorm > 0
            ? dot / (actualNorm * expectedNorm) : 1
        return .init(
            maxAbsoluteError: maxError,
            rmsError: rmsError,
            relativeRMSError: relative,
            cosineSimilarity: cosine)
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

    private static func routingMetrics(
        actual: MLXArray, expected: MLXArray
    ) -> Qwen4ExpMoERoutingParityMetrics {
        precondition(actual.shape == expected.shape)
        precondition(actual.ndim >= 2)
        let topK = actual.dim(-1)
        let actualValues = actual.asType(.uint32).asArray(UInt32.self)
        let expectedValues = expected.asType(.uint32).asArray(UInt32.self)
        let positions = actual.size / max(1, topK)
        var differentPositions = 0
        var differingAssignments = 0
        for position in 0 ..< positions {
            let start = position * topK
            let actualSet = Set(actualValues[start ..< start + topK])
            let expectedSet = Set(expectedValues[start ..< start + topK])
            let symmetricDifference = actualSet.symmetricDifference(expectedSet)
            if !symmetricDifference.isEmpty {
                differentPositions += 1
                differingAssignments += symmetricDifference.count / 2
            }
        }
        return .init(
            positions: positions,
            positionsWithDifferentMembership: differentPositions,
            differingExpertAssignments: differingAssignments,
            topK: topK)
    }
}

public enum Qwen4ExpPublicLayerParityError: LocalizedError, Equatable {
    case missingTensor

    public var errorDescription: String? {
        "Fixture couche publique incomplet : layer_index, input_ids, embedded ou layer_output absent."
    }
}
