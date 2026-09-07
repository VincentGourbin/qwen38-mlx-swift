import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen4ExpLanguageParityReport: Sendable, Equatable {
    public let outputShape: [Int]
    public let outputMaxAbsoluteError: Float
    public let outputMeanAbsoluteError: Float
    public let stageMaxAbsoluteError: [String: Float]
    public let cacheMaxAbsoluteError: [String: Float]

    public init(
        outputShape: [Int], outputMaxAbsoluteError: Float,
        outputMeanAbsoluteError: Float, stageMaxAbsoluteError: [String: Float],
        cacheMaxAbsoluteError: [String: Float]
    ) {
        self.outputShape = outputShape
        self.outputMaxAbsoluteError = outputMaxAbsoluteError
        self.outputMeanAbsoluteError = outputMeanAbsoluteError
        self.stageMaxAbsoluteError = stageMaxAbsoluteError
        self.cacheMaxAbsoluteError = cacheMaxAbsoluteError
    }
}

/// Compares a real checkpoint layer with the MLX Python reference fixture.
///
/// The fixture deliberately stops at decoder layer 0. That is enough to
/// catch projection packing, GDN recurrence, gated RMSNorm, hyper-connections
/// and MoE routing without loading the 51B n-gram embedding or 47 other layers.
public enum Qwen4ExpLanguageParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL
    ) throws -> Qwen4ExpLanguageParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let hidden = arrays["hidden"], let inputIDs = arrays["input_ids"],
              let expected = arrays["output"] else {
            throw Qwen4ExpLanguageParityError.missingTensor
        }

        let loaded = try Qwen4ExpCheckpointLayerLoader.load(0, from: modelDirectory)
        let layer = loaded.layer
        guard let linearAttn = layer.linearAttn else {
            throw Qwen4ExpLanguageParityError.notLinearLayer
        }
        let cache = ArraysCache(size: 4)
        let attentionNormed = layer.attnHyperConnection.hcNorm(hidden)
        let attentionDown = layer.attnHyperConnection.inputMixWeightDown(attentionNormed)
        let attentionDownSilu = MLXNN.silu(
            attentionDown / Float(layer.attnHyperConnection.streamCount))
        let attentionUp = layer.attnHyperConnection.inputMixWeightUp(attentionDownSilu)
        let attentionMixWeights = sigmoid(attentionUp).reshaped(
            [hidden.dim(0), hidden.dim(1), layer.attnHyperConnection.streamCount,
             layer.attnHyperConnection.hiddenSize])
        let attentionStreams = attentionNormed.reshaped(
            [hidden.dim(0), hidden.dim(1), layer.attnHyperConnection.streamCount,
             layer.attnHyperConnection.hiddenSize])
        let attentionMixed = (attentionMixWeights * attentionStreams).mean(axis: -2)
        let attentionInjection = 2 * sigmoid(
            layer.attnHyperConnection.blockInjectWeight!(attentionNormed)
                / Float(layer.attnHyperConnection.streamCount))
        let attentionMix = (
            mixedInput: attentionMixed, originalInput: hidden,
            injectionWeights: attentionInjection)
        let attentionBranch = linearAttn(
            attentionMix.mixedInput, mask: nil, cache: cache)
        let stateAfterAttention = attentionMix.originalInput + Self.inject(
            branch: attentionBranch,
            weights: attentionMix.injectionWeights)
        let mlpMix = layer.mlpHyperConnection(stateAfterAttention)
        let mlpBranch = layer.mlp(mlpMix.mixedInput)
        let actual = mlpMix.originalInput + Self.inject(
            branch: mlpBranch,
            weights: mlpMix.injectionWeights)
        precondition(inputIDs.ndim == 2)
        eval(actual)

        guard actual.shape == expected.shape else {
            throw Qwen4ExpLanguageParityError.shapeMismatch(
                expected: expected.shape, actual: actual.shape)
        }
        let difference = abs(actual.asType(.float32) - expected.asType(.float32))
        eval(difference)

        var stageErrors = [String: Float]()
        let actualStages = [
            "attn_normed": attentionNormed,
            "attn_down": attentionDown,
            "attn_down_silu": attentionDownSilu,
            "attn_up": attentionUp,
            "attn_mix": attentionMixWeights,
            "attn_mixed": attentionMix.mixedInput,
            "attn_branch": attentionBranch,
            "state_after_attention": stateAfterAttention,
            "mlp_mixed": mlpMix.mixedInput,
            "mlp_branch": mlpBranch,
            "output": actual,
        ]
        for (name, value) in actualStages {
            guard let expectedStage = arrays[name] else { continue }
            guard expectedStage.shape == value.shape else {
                throw Qwen4ExpLanguageParityError.stageShapeMismatch(
                    name: name, expected: expectedStage.shape, actual: value.shape)
            }
            let stageDifference = abs(
                value.asType(.float32) - expectedStage.asType(.float32))
            eval(stageDifference)
            stageErrors[name] = stageDifference.max().item(Float.self)
        }

        var cacheErrors = [String: Float]()
        for index in 0..<4 {
            let name = "cache_\(index)"
            guard let expectedCache = arrays[name] else { continue }
            guard let actualCache = cache[index] else {
                throw Qwen4ExpLanguageParityError.missingCache(name)
            }
            guard actualCache.shape == expectedCache.shape else {
                throw Qwen4ExpLanguageParityError.cacheShapeMismatch(
                    name: name, expected: expectedCache.shape, actual: actualCache.shape)
            }
            let cacheDifference = abs(
                actualCache.asType(.float32) - expectedCache.asType(.float32))
            eval(cacheDifference)
            cacheErrors[name] = cacheDifference.max().item(Float.self)
        }

        return .init(
            outputShape: actual.shape,
            outputMaxAbsoluteError: difference.max().item(Float.self),
            outputMeanAbsoluteError: difference.sum().item(Float.self) / Float(difference.size),
            stageMaxAbsoluteError: stageErrors,
            cacheMaxAbsoluteError: cacheErrors)
    }

    private static func inject(branch: MLXArray, weights: MLXArray) -> MLXArray {
        let injection = branch.expandedDimensions(axis: -2)
            * weights.expandedDimensions(axis: -1)
        return injection.reshaped([branch.dim(0), branch.dim(1), -1])
    }
}

public enum Qwen4ExpLanguageParityError: LocalizedError, Equatable {
    case missingTensor
    case notLinearLayer
    case missingCache(String)
    case shapeMismatch(expected: [Int], actual: [Int])
    case stageShapeMismatch(name: String, expected: [Int], actual: [Int])
    case cacheShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingTensor:
            return "Fixture langage incomplet : hidden, input_ids ou output absent."
        case .notLinearLayer:
            return "La parité de la couche 0 attend une couche linear_attention."
        case .missingCache(let name):
            return "État de cache Python absent dans Swift : \(name)."
        case .shapeMismatch(let expected, let actual):
            return "Forme de sortie langage différente : \(expected) vs \(actual)."
        case .stageShapeMismatch(let name, let expected, let actual):
            return "Forme de l'étape \(name) différente : \(expected) vs \(actual)."
        case .cacheShapeMismatch(let name, let expected, let actual):
            return "Forme \(name) différente : \(expected) vs \(actual)."
        }
    }
}
