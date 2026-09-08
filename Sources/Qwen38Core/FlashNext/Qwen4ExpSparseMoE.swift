import MLX
import MLXNN
import MLXLMCommon

/// Dense SwiGLU used by Flash-Next's shared expert.
public final class Qwen4ExpSharedExpert: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(
        inputDimensions: Int,
        hiddenDimensions: Int,
        quantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        _gateProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: inputDimensions, outputDimensions: hiddenDimensions,
            quantization: quantization)
        _upProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: inputDimensions, outputDimensions: hiddenDimensions,
            quantization: quantization)
        _downProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: hiddenDimensions, outputDimensions: inputDimensions,
            quantization: quantization)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Qwen3.8 Flash-Next's routed MoE block.
///
/// The expert projections use MLXLMCommon's `SwitchGLU`, whose checkpoint
/// contract is the fused `[experts, output, input]` layout used by the
/// converted Flash-Next weights. Routing deliberately follows the upstream
/// ascending `argPartition` order because the weighted reduction is
/// order-sensitive for bfloat16 inference.
public final class Qwen4ExpSparseMoE: Module, UnaryLayer {
    public let numExperts: Int
    public let topK: Int
    public let normalizeTopK: Bool

    @ModuleInfo(key: "gate") public var gate: Linear
    @ModuleInfo(key: "switch_mlp") public var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") public var sharedExpert: Qwen4ExpSharedExpert
    @ModuleInfo(key: "shared_expert_gate") public var sharedExpertGate: Linear

    /// Small boundary tensors retained for the real-checkpoint parity probe.
    public private(set) var lastParityCapture: [String: MLXArray] = [:]
    /// Disabled during normal inference so intermediate graphs are not kept alive.
    public private(set) var captureParity = false

    public init(
        configuration: Qwen4ExpTextConfiguration,
        normalizeTopK: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        numExperts = configuration.numExperts
        topK = configuration.numExpertsPerToken
        self.normalizeTopK = normalizeTopK

        precondition(numExperts > 0)
        precondition(topK > 0 && topK <= numExperts)
        // The router is intentionally kept in floating point by the released
        // checkpoint: it has no `.scales`/`.biases` companions. Applying the
        // global 4-bit spec here would create [experts, hidden/8] and strict
        // loading would reject the real [experts, hidden] tensor.
        _gate.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize, outputDimensions: numExperts,
            quantization: nil)
        // The routed experts (`switch_mlp`) get their own quantization spec
        // when the checkpoint declares `quantization.experts` (Q3: 3-bit
        // g64 experts while attention/shared_expert/gate stay 4-bit g32).
        // Absent an override, they fall back to the global spec, matching
        // the unchanged Vontra checkpoint's behavior.
        let switchMLPQuantization = expertsQuantization ?? quantization
        if let switchMLPQuantization {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: numExperts,
                quantization: (
                    groupSize: switchMLPQuantization.groupSize,
                    bits: switchMLPQuantization.bits,
                    mode: switchMLPQuantization.mode))
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: numExperts)
        }
        _sharedExpert.wrappedValue = Qwen4ExpSharedExpert(
            inputDimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.sharedExpertIntermediateSize,
            quantization: quantization)
        _sharedExpertGate.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize, outputDimensions: 1,
            quantization: quantization)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim >= 2)
        let probabilities = MLX.softmax(gate(x), axis: -1, precise: true)
        let kth = numExperts - topK
        let indices = MLX.argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        if normalizeTopK {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped([tokenCount, x.dim(-1)])
        let flatIndices = indices.reshaped([tokenCount, topK])
        let flatScores = scores.reshaped([tokenCount, topK])
        let routed = weightedExpertSum(
            switchMLP(flatX, flatIndices), flatScores).reshaped(x.shape)

        let shared = sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        let output = routed + shared
        if captureParity {
            lastParityCapture = [
                "moe_probabilities": probabilities,
                "moe_indices": indices,
                "moe_scores": scores,
                "moe_routed": routed,
                "moe_shared": shared,
                "moe_output": output,
            ]
        } else {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
        return output
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }
}
