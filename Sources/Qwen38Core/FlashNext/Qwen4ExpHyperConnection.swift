import MLX
import MLXFast
import MLXNN

/// Qwen4's four-stream gated residual mixer.
public final class Qwen4ExpGatedResidual: Module {
    public let hiddenSize: Int
    public let streamCount: Int
    private let parityPrefix: String

    /// Optional boundary tensors for public real-checkpoint parity probes.
    public private(set) var lastParityCapture: [String: MLXArray] = [:]
    /// Disabled during normal inference so intermediate graphs are not kept alive.
    public private(set) var captureParity = false

    @ModuleInfo(key: "hc_norm") public var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") public var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") public var inputMixWeightUp: Linear
    @ModuleInfo(key: "block_inject_weight") public var blockInjectWeight: Linear?

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        useCombine: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        parityPrefix: String = ""
    ) {
        hiddenSize = configuration.hiddenSize
        streamCount = configuration.hcCount
        self.parityPrefix = parityPrefix
        let streamHiddenSize = streamCount * hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: streamHiddenSize, groupSize: hiddenSize, eps: rmsNormEps)
        _inputMixWeightDown.wrappedValue = qwen4ExpLinear(
            inputDimensions: streamHiddenSize, outputDimensions: configuration.hcLowrank,
            quantization: quantization)
        _inputMixWeightUp.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hcLowrank, outputDimensions: streamHiddenSize,
            quantization: quantization)
        if useCombine {
            _blockInjectWeight.wrappedValue = qwen4ExpLinear(
                inputDimensions: streamHiddenSize, outputDimensions: streamCount,
                quantization: quantization)
        }
        super.init()
    }

    public func callAsFunction(_ hyperInput: MLXArray) -> (
        mixedInput: MLXArray, originalInput: MLXArray, injectionWeights: MLXArray
    ) {
        precondition(hyperInput.ndim == 3)
        precondition(hyperInput.dim(-1) == streamCount * hiddenSize)
        let normed = hcNorm(hyperInput)
        var mix = silu(inputMixWeightDown(normed) / Float(streamCount))
        mix = sigmoid(inputMixWeightUp(mix))
            .reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        let streams = normed.reshaped(
            [hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        let mixed = (mix * streams).mean(axis: -2)
        guard let blockInjectWeight else {
            preconditionFailure("Cette hyper-connexion n'expose pas de combinaison")
        }
        let injection = 2 * sigmoid(blockInjectWeight(normed) / Float(streamCount))
        if captureParity && !parityPrefix.isEmpty {
            lastParityCapture = [
                "\(parityPrefix)normed": normed,
                "\(parityPrefix)mixed": mixed,
                "\(parityPrefix)injection": injection,
            ]
        }
        return (mixed, hyperInput, injection)
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }

    /// Final four-stream reduction used immediately before the language head.
    /// It intentionally does not allocate or materialize the residual streams.
    public func mixedInput(_ hyperInput: MLXArray) -> MLXArray {
        precondition(hyperInput.ndim == 3)
        precondition(hyperInput.dim(-1) == streamCount * hiddenSize)
        let normed = hcNorm(hyperInput)
        var mix = silu(inputMixWeightDown(normed) / Float(streamCount))
        mix = sigmoid(inputMixWeightUp(mix))
            .reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        let streams = normed.reshaped(
            [hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        return (mix * streams).mean(axis: -2)
    }
}

/// Qwen4 RMSNorm uses zero-centered checkpoint weights.
public final class Qwen4ExpRMSNorm: Module {
    public let eps: Float
    public let groupSize: Int?
    @ParameterInfo(key: "weight") public var weight: MLXArray

    /// F2 (P2-fusion): `1 + weight` baked once by `precomputeEffectiveWeight()`
    /// after the checkpoint (and, where applicable, the Vontra `-1` shift
    /// correction — PLAN.md §6.3 piège 12) has been loaded. Not an
    /// `@ModuleInfo`/`@ParameterInfo` property: it is purely derived from
    /// `weight` and must stay invisible to `parameters()`/`update(parameters:)`.
    /// `nil` means the original per-call `1 + weight` path (unchanged
    /// behavior).
    private var effectiveWeight: MLXArray?

    public init(dimensions: Int, groupSize: Int? = nil, eps: Float = 1e-6) {
        precondition(groupSize == nil || dimensions % groupSize! == 0)
        self.eps = eps
        self.groupSize = groupSize
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        if let effectiveWeight {
            if let groupSize {
                // Grouped case (hc_norm, PLE norms): each of `dimensions /
                // groupSize` groups has its own weight slice, which
                // `MLXFast.rmsNorm`'s single 1-D weight cannot express in one
                // fused call. Keep the manual reduction, but the `1 +`
                // addition and the weight upcast are already baked into
                // `effectiveWeight`, so only the normalization itself still
                // runs per call.
                let inputShape = inputs.shape
                let values = inputs.asType(.float32).reshaped(
                    [inputShape.dropLast().reduce(1, *), inputShape.last! / groupSize, groupSize])
                let groupedWeight = effectiveWeight.reshaped([-1, groupSize])
                let normed = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
                return (normed * groupedWeight).reshaped(inputShape).asType(inputs.dtype)
            }
            // Ungrouped case (q_norm/k_norm, indexer layernorms): a single
            // fused kernel replaces the manual square/mean/rsqrt/mul chain,
            // and MLXFast.rmsNorm handles its own internal precision, so no
            // explicit float32 upcast is needed here either.
            return MLXFast.rmsNorm(inputs, weight: effectiveWeight, eps: eps)
        }
        let inputShape = inputs.shape
        var values = inputs.asType(.float32)
        if let groupSize {
            values = values.reshaped([inputShape.dropLast().reduce(1, *), inputShape.last! / groupSize, groupSize])
            let groupedWeight = weight.reshaped([-1, groupSize]).asType(.float32)
            values = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
            values = values * (1 + groupedWeight)
        } else {
            values = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
            values = values * (1 + weight.asType(.float32))
        }
        return values.reshaped(inputShape).asType(inputs.dtype)
    }

    /// F2 (P2-fusion): bake this checkpoint's `1 + weight` convention into a
    /// cached array once, so `callAsFunction` never adds 1 or upcasts the
    /// weight again. Idempotent; safe to call unconditionally — callers gate
    /// it on `Qwen4ExpFusionLevel`, not this method. Must run after any
    /// convention correction (`Qwen4ExpWeightSanitizer`) has already been
    /// applied to `weight`, i.e. after `Module.update(parameters:)`.
    public func precomputeEffectiveWeight() {
        guard effectiveWeight == nil else { return }
        let baked = (1 + weight.asType(.float32)).asType(weight.dtype)
        eval(baked)
        effectiveWeight = baked
    }
}
