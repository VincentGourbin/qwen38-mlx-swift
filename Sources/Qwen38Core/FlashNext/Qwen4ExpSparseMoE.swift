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

    /// F4 (P2-fusion): whether the router softmax runs in `precise` (fp32
    /// accumulation) mode. `true` until `fusionLevel >= .f4MoE` — see
    /// `callAsFunction` for why dropping `precise` is safe for routing.
    private let preciseRouterSoftmax: Bool

    /// P7.1: which sub-block, if any, `callAsFunction` short-circuits for
    /// `flash-layer-bench --ablate`. `.none` everywhere in production —
    /// see `Qwen4ExpLayerBenchAblation`.
    public let ablation: Qwen4ExpLayerBenchAblation

    public init(
        configuration: Qwen4ExpTextConfiguration,
        normalizeTopK: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil,
        fusionLevel: Qwen4ExpFusionLevel = .none,
        ablation: Qwen4ExpLayerBenchAblation = .none
    ) {
        numExperts = configuration.numExperts
        topK = configuration.numExpertsPerToken
        self.normalizeTopK = normalizeTopK
        self.preciseRouterSoftmax = fusionLevel < .f4MoE
        self.ablation = ablation

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
        // P7.1 (`--ablate moe`): skip routing, expert gather and the shared
        // expert entirely — the branch becomes a no-op residual. Never
        // numerically correct — see `Qwen4ExpLayerBenchAblation`.
        if ablation == .moe {
            return x
        }
        // F4 (P2-fusion): softmax is a strictly monotonic transform of the
        // gate logits (dividing every exp(logit) by the same positive sum
        // preserves relative order), so the top-`topK` *set* selected by
        // `argPartition` below is mathematically identical whether or not
        // `precise` upcasts to fp32 — unless two logits are close enough
        // that the lower-precision reduction flips their order right at the
        // kth boundary. Measured on 200 synthetic single-token gate vectors
        // at this checkpoint's real dimensions (512 experts, top-10): zero
        // such flips (see the "P2-fusion (F4)" test and log.md entry).
        let probabilities = MLX.softmax(gate(x), axis: -1, precise: preciseRouterSoftmax)
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
        // P7.1 sub-probes: the router above always runs for real in every
        // case reaching this point (only `.moe` above skips it entirely);
        // these two zero only the expert computation they name, isolating
        // its cost by subtraction against `.none`. Never numerically
        // correct — see `Qwen4ExpLayerBenchAblation`.
        let routed = (ablation == .moeSwitchMLP || ablation == .moeRouting)
            ? MLXArray.zeros(x.shape, dtype: x.dtype)
            : weightedExpertSum(switchMLP(flatX, flatIndices), flatScores).reshaped(x.shape)
        let shared = (ablation == .moeSharedExpert || ablation == .moeRouting)
            ? MLXArray.zeros(x.shape, dtype: x.dtype)
            : sigmoid(sharedExpertGate(x)) * sharedExpert(x)
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
