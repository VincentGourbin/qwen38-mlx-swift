import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen4ExpSingleLayerParityReport: Sendable, Equatable {
    public let embeddedMaxAbsoluteError: Float
    public let layerMaxAbsoluteError: Float
    public let reducedMaxAbsoluteError: Float
    public let logitsMaxAbsoluteError: Float
    public let embeddedMeanAbsoluteError: Float
    public let layerMeanAbsoluteError: Float
    public let reducedMeanAbsoluteError: Float
    public let logitsMeanAbsoluteError: Float
    public let stageMaxAbsoluteError: [String: Float]

    public init(
        embeddedMaxAbsoluteError: Float,
        layerMaxAbsoluteError: Float,
        reducedMaxAbsoluteError: Float,
        logitsMaxAbsoluteError: Float,
        embeddedMeanAbsoluteError: Float,
        layerMeanAbsoluteError: Float,
        reducedMeanAbsoluteError: Float,
        logitsMeanAbsoluteError: Float,
        stageMaxAbsoluteError: [String: Float]
    ) {
        self.embeddedMaxAbsoluteError = embeddedMaxAbsoluteError
        self.layerMaxAbsoluteError = layerMaxAbsoluteError
        self.reducedMaxAbsoluteError = reducedMaxAbsoluteError
        self.logitsMaxAbsoluteError = logitsMaxAbsoluteError
        self.embeddedMeanAbsoluteError = embeddedMeanAbsoluteError
        self.layerMeanAbsoluteError = layerMeanAbsoluteError
        self.reducedMeanAbsoluteError = reducedMeanAbsoluteError
        self.logitsMeanAbsoluteError = logitsMeanAbsoluteError
        self.stageMaxAbsoluteError = stageMaxAbsoluteError
    }
}

/// Compares one real decoder layer plus the surrounding global path.
public enum Qwen4ExpSingleLayerParity {
    public static func compareFixture(
        modelDirectory: URL,
        fixtureURL: URL
    ) throws -> Qwen4ExpSingleLayerParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let inputIDs = arrays["input_ids"],
              let expectedEmbedded = arrays["embedded"],
              let hidden = arrays["hidden"],
              let expectedLayer = arrays["layer_output"],
              let expectedReduced = arrays["reduced"],
              let expectedLogits = arrays["logits"] else {
            throw Qwen4ExpSingleLayerParityError.missingTensor
        }

        let loadedGlobal = try Qwen4ExpGlobalCheckpointLoader.load(
            from: modelDirectory, materialize: true)
        let loadedLayer = try Qwen4ExpCheckpointLayerLoader.load(0, from: modelDirectory)
        let embedded = loadedGlobal.model.embed(inputIDs)
        let layer = loadedLayer.layer
        guard let linearAttn = layer.linearAttn else {
            throw Qwen4ExpSingleLayerParityError.notLinearLayer
        }
        let cache = ArraysCache(size: 4)
        let normed = layer.attnHyperConnection.hcNorm(hidden)
        let down = layer.attnHyperConnection.inputMixWeightDown(normed)
        let downSilu = MLXNN.silu(down / Float(layer.attnHyperConnection.streamCount))
        let up = layer.attnHyperConnection.inputMixWeightUp(downSilu)
        let mix = sigmoid(up).reshaped([
            hidden.dim(0), hidden.dim(1), layer.attnHyperConnection.streamCount,
            layer.attnHyperConnection.hiddenSize])
        let streams = normed.reshaped([
            hidden.dim(0), hidden.dim(1), layer.attnHyperConnection.streamCount,
            layer.attnHyperConnection.hiddenSize])
        let mixed = (mix * streams).mean(axis: -2)
        let injection = 2 * sigmoid(
            layer.attnHyperConnection.blockInjectWeight!(normed)
                / Float(layer.attnHyperConnection.streamCount))
        let qkv = linearAttn.inProjQKV(mixed)
        let z = linearAttn.inProjZ(mixed).reshaped([
            mixed.dim(0), mixed.dim(1), linearAttn.numValueHeads, linearAttn.valueHeadDim])
        let b = linearAttn.inProjB(mixed)
        let a = linearAttn.inProjA(mixed)
        let convState = MLXArray.zeros(
            [mixed.dim(0), max(0, linearAttn.convKernelSize - 1), linearAttn.convDim],
            dtype: mixed.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        cache[0] = contiguous(convInput[.ellipsis, (-(linearAttn.convKernelSize - 1))..., 0...])
        let convOut = MLXNN.silu(linearAttn.conv1d(convInput))
        let split = MLX.split(
            convOut, indices: [linearAttn.keyDim, 2 * linearAttn.keyDim], axis: -1)
        let q = split[0].reshaped([
            mixed.dim(0), mixed.dim(1), linearAttn.numKeyHeads, linearAttn.keyHeadDim])
        let k = split[1].reshaped([
            mixed.dim(0), mixed.dim(1), linearAttn.numKeyHeads, linearAttn.keyHeadDim])
        let v = split[2].reshaped([
            mixed.dim(0), mixed.dim(1), linearAttn.numValueHeads, linearAttn.valueHeadDim])
        let scale = 1 / Float(linearAttn.keyHeadDim).squareRoot()
        let qNormed = q * MLX.rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * MLXArray(scale).asType(q.dtype)
        let kNormed = k * MLX.rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
        let g = exp(-exp(linearAttn.aLog.asType(.float32))
            * MLXNN.softplus(a + linearAttn.dtBias))
        let beta = sigmoid(b).asType(.float32)
        let (gdnOutput, state) = gatedDeltaUpdate(
            q: qNormed, k: kNormed, v: v, a: a, b: b,
            aLog: linearAttn.aLog, dtBias: linearAttn.dtBias,
            state: cache[1], mask: nil)
        cache[1] = state
        cache.advance(hidden.dim(1))
        let normOutput = linearAttn.norm(
            gdnOutput.reshaped([
                mixed.dim(0), mixed.dim(1), linearAttn.numValueHeads, linearAttn.valueHeadDim]),
            gate: z)
        let attnBranch = linearAttn.outProj(
            normOutput.reshaped([mixed.dim(0), mixed.dim(1), linearAttn.valueDim]))
        let stateAfterAttention = hidden + Self.inject(
            branch: attnBranch, weights: injection)
        let mlpMix = layer.mlpHyperConnection(stateAfterAttention)
        let mlpBranch = layer.mlp(mlpMix.mixedInput)
        let layerOutput = mlpMix.originalInput + Self.inject(
            branch: mlpBranch, weights: mlpMix.injectionWeights)
        let reduced = loadedGlobal.model.reduceHyperStreams(layerOutput)
        let logits = loadedGlobal.model.logits(from: reduced)
        eval(embedded, layerOutput, reduced, logits)

        let errors = [
            difference(actual: embedded, expected: expectedEmbedded),
            difference(actual: layerOutput, expected: expectedLayer),
            difference(actual: reduced, expected: expectedReduced),
            difference(actual: logits, expected: expectedLogits),
        ]
        let stageValues = [
            "attn_normed": normed, "attn_down": down,
            "attn_down_silu": downSilu, "attn_up": up, "attn_mix": mix,
            "attn_mixed": mixed, "attn_branch": attnBranch,
            "qkv": qkv, "z": z, "b": b, "a": a,
            "conv_input": convInput, "conv_out": convOut,
            "q": q, "k": k, "v": v, "q_normed": qNormed,
            "k_normed": kNormed, "g": g, "beta": beta,
            "gdn_output": gdnOutput, "norm_output": normOutput,
            "state_after_attention": stateAfterAttention,
            "mlp_mixed": mlpMix.mixedInput, "mlp_branch": mlpBranch,
        ]
        var stageErrors = [String: Float]()
        for (name, value) in stageValues {
            guard let expected = arrays[name] else { continue }
            stageErrors[name] = difference(actual: value, expected: expected).max
        }
        for index in 0 ..< 4 {
            guard let expected = arrays["prefill_cache_\(index)"],
                  let actual = cache[index] else { continue }
            stageErrors["prefill_cache_\(index)"] = difference(
                actual: actual, expected: expected).max
        }

        if let decodeInputIDs = arrays["decode_input_ids"],
           let expectedDecodeEmbedded = arrays["decode_embedded"],
           let expectedDecodeHidden = arrays["decode_hidden"],
           let expectedDecodeLayer = arrays["decode_layer_output"] {
            let decodeEmbedded = loadedGlobal.model.embed(decodeInputIDs)
            let decodeHidden = tiled(
                decodeEmbedded,
                repetitions: [1, 1, layer.attnHyperConnection.streamCount])
            let decodeCache = ArraysCache(size: 4)
            decodeCache[0] = cache[0]
            decodeCache[1] = cache[1]
            let decodeAttentionMix = layer.attnHyperConnection(decodeHidden)
            let decodeQKV = linearAttn.inProjQKV(decodeAttentionMix.mixedInput)
            let decodeZ = linearAttn.inProjZ(decodeAttentionMix.mixedInput).reshaped([
                decodeHidden.dim(0), decodeHidden.dim(1),
                linearAttn.numValueHeads, linearAttn.valueHeadDim])
            let decodeB = linearAttn.inProjB(decodeAttentionMix.mixedInput)
            let decodeA = linearAttn.inProjA(decodeAttentionMix.mixedInput)
            let decodeConvInput = concatenated(
                [decodeCache[0]!, decodeQKV], axis: 1)
            decodeCache[0] = contiguous(
                decodeConvInput[.ellipsis, (-(linearAttn.convKernelSize - 1))..., 0...])
            let decodeConvOut = MLXNN.silu(linearAttn.conv1d(decodeConvInput))
            let decodeSplit = MLX.split(
                decodeConvOut, indices: [linearAttn.keyDim, 2 * linearAttn.keyDim], axis: -1)
            let decodeQ = decodeSplit[0].reshaped([
                1, 1, linearAttn.numKeyHeads, linearAttn.keyHeadDim])
            let decodeK = decodeSplit[1].reshaped([
                1, 1, linearAttn.numKeyHeads, linearAttn.keyHeadDim])
            let decodeV = decodeSplit[2].reshaped([
                1, 1, linearAttn.numValueHeads, linearAttn.valueHeadDim])
            let decodeQNormed = decodeQ * MLX.rsqrt(
                (decodeQ * decodeQ).sum(axis: -1, keepDims: true) + 1e-6)
                * MLXArray(1 / Float(linearAttn.keyHeadDim).squareRoot())
                    .asType(decodeQ.dtype)
            let decodeKNormed = decodeK * MLX.rsqrt(
                (decodeK * decodeK).sum(axis: -1, keepDims: true) + 1e-6)
            let (decodeGDNOutput, decodeState) = gatedDeltaUpdate(
                q: decodeQNormed, k: decodeKNormed, v: decodeV,
                a: decodeA, b: decodeB, aLog: linearAttn.aLog,
                dtBias: linearAttn.dtBias, state: decodeCache[1], mask: nil)
            decodeCache[1] = decodeState
            decodeCache.advance(1)
            let decodeNormOutput = linearAttn.norm(
                decodeGDNOutput.reshaped([
                    1, 1, linearAttn.numValueHeads, linearAttn.valueHeadDim]),
                gate: decodeZ)
            let decodeAttentionBranch = linearAttn.outProj(
                decodeNormOutput.reshaped([1, 1, linearAttn.valueDim]))
            let decodeStateAfterAttention = decodeHidden + Self.inject(
                branch: decodeAttentionBranch,
                weights: decodeAttentionMix.injectionWeights)
            let decodeMLPMix = layer.mlpHyperConnection(decodeStateAfterAttention)
            let decodeMLPBranch = layer.mlp(decodeMLPMix.mixedInput)
            let decodeLayer = decodeMLPMix.originalInput + Self.inject(
                branch: decodeMLPBranch, weights: decodeMLPMix.injectionWeights)
            eval(decodeEmbedded, decodeHidden, decodeLayer, decodeCache[0]!, decodeCache[1]!)
            let layerCallCache = ArraysCache(size: 4)
            layerCallCache[0] = cache[0]
            layerCallCache[1] = cache[1]
            let layerCallOutput = layer(
                decodeHidden, inputIDs: decodeInputIDs, mask: nil,
                cache: layerCallCache, positionIDs: nil)
            eval(layerCallOutput, layerCallCache[0]!, layerCallCache[1]!)
            stageErrors["decode_embedded"] = difference(
                actual: decodeEmbedded, expected: expectedDecodeEmbedded).max
            stageErrors["decode_hidden"] = difference(
                actual: decodeHidden, expected: expectedDecodeHidden).max
            stageErrors["decode_layer_output"] = difference(
                actual: decodeLayer, expected: expectedDecodeLayer).max
            stageErrors["decode_layer_call"] = difference(
                actual: layerCallOutput, expected: expectedDecodeLayer).max
            let decodeValues = [
                "decode_qkv": decodeQKV, "decode_z": decodeZ,
                "decode_a": decodeA, "decode_b": decodeB,
                "decode_conv_input": decodeConvInput, "decode_conv_out": decodeConvOut,
                "decode_q": decodeQ, "decode_k": decodeK, "decode_v": decodeV,
                "decode_q_normed": decodeQNormed, "decode_k_normed": decodeKNormed,
                "decode_gdn_output": decodeGDNOutput, "decode_norm_output": decodeNormOutput,
                "decode_attn_branch": decodeAttentionBranch,
                "decode_state_after_attention": decodeStateAfterAttention,
                "decode_mlp_mixed": decodeMLPMix.mixedInput,
                "decode_mlp_branch": decodeMLPBranch,
            ]
            for (name, value) in decodeValues {
                guard let expected = arrays[name] else { continue }
                stageErrors[name] = difference(actual: value, expected: expected).max
            }
            for index in 0 ..< 4 {
                guard let expected = arrays["cache_\(index)"],
                      let actual = decodeCache[index] else { continue }
                stageErrors["decode_cache_\(index)"] = difference(
                    actual: actual, expected: expected).max
            }
        }
        return .init(
            embeddedMaxAbsoluteError: errors[0].max,
            layerMaxAbsoluteError: errors[1].max,
            reducedMaxAbsoluteError: errors[2].max,
            logitsMaxAbsoluteError: errors[3].max,
            embeddedMeanAbsoluteError: errors[0].mean,
            layerMeanAbsoluteError: errors[1].mean,
            reducedMeanAbsoluteError: errors[2].mean,
            logitsMeanAbsoluteError: errors[3].mean,
            stageMaxAbsoluteError: stageErrors)
    }

    private static func inject(branch: MLXArray, weights: MLXArray) -> MLXArray {
        let injection = branch.expandedDimensions(axis: -2)
            * weights.expandedDimensions(axis: -1)
        return injection.reshaped([branch.dim(0), branch.dim(1), -1])
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

public enum Qwen4ExpSingleLayerParityError: LocalizedError, Equatable {
    case missingTensor
    case notLinearLayer

    public var errorDescription: String? {
        "Fixture single-layer incomplet : input_ids, hidden ou sortie intermédiaire absente."
    }
}
