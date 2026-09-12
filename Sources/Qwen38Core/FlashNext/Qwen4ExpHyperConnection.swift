import MLX
import MLXFast
import MLXNN

// MARK: - P10.2 (F8): fused hyper-connection "mix" and "inject" kernels
//
// `Qwen4ExpGatedResidual.callAsFunction`'s mix reduction (sigmoid + two
// reshapes + multiply + mean, 5 ops) and `Qwen4ExpDecoderLayer.inject` (two
// `expandedDimensions` + multiply + reshape + add, 5 ops) are pure glue
// around the two low-rank matmuls and the branch computation — no reduction
// over an axis wider than the 4 hyper-connection streams, called twice per
// layer (attn + mlp). Each op costs a fixed ~8.58 µs of host dispatch
// (docs/knowledge/log.md, "P10.1") regardless of how small the tensor is,
// so collapsing 5 ops into 1 custom kernel call is a plausible win at this
// scale — unlike the matmuls themselves, which stay on MLX's own optimized
// (quantized) matmul path; this fuses only the surrounding elementwise glue.
//
// Both kernels assume the streams are stored contiguously as
// `[..., streamCount * hiddenSize]` (true for every caller here) and that
// `streamCount * hiddenSize` (and `hiddenSize` alone) divide evenly by the
// chosen threadgroup width — enforced by `qwen4ExpKernelThreadGroupWidth`,
// which always returns an exact divisor (falling back to 1), so this holds
// for any hiddenSize, not just Flash-Next's 2560.

/// Largest power-of-two-ish threadgroup width (out of a small fixed set)
/// that evenly divides `total`, so `MLXFastKernel`'s `grid`/`threadGroup`
/// never straddles a non-uniform dispatch. Falls back to 1 (always exact,
/// just less parallel) if nothing bigger divides evenly. `internal` (not
/// `private`): shared with `Qwen4ExpGatedDeltaNet.swift`'s P10.3 kernel.
func qwen4ExpKernelThreadGroupWidth(_ total: Int) -> Int {
    for candidate in [256, 128, 64, 32, 16, 8, 4, 2, 1] where total % candidate == 0 {
        return candidate
    }
    return 1
}

/// Fuses `sigmoid(upOut).reshaped([N, S, H]) * normed.reshaped([N, S, H])`
/// then `.mean(axis: -2)` into one kernel call. `upOut` and `normed` are both
/// `[..., S * H]`, row-contiguous; the result is `[..., H]`, same dtype.
private let qwen4ExpHyperMixKernel = MLXFast.metalKernel(
    name: "qwen4exp_hyper_mix",
    inputNames: ["upOut", "normed"],
    outputNames: ["mixed"],
    source: """
        uint idx = thread_position_in_grid.x;
        uint h = idx % H;
        uint n = idx / H;
        uint base = n * (S * H) + h;
        float acc = 0.0;
        for (uint s = 0; s < S; s++) {
            float u = float(upOut[base + s * H]);
            float x = float(normed[base + s * H]);
            float sig = 1.0f / (1.0f + metal::exp(-u));
            acc += sig * x;
        }
        mixed[idx] = static_cast<T>(acc / float(S));
    """)

/// `upOut`/`normed`: `[..., streamCount * hiddenSize]`, row-contiguous, same
/// dtype. Returns `[..., hiddenSize]`.
func qwen4ExpHyperMixFused(
    upOut rawUpOut: MLXArray, normed: MLXArray, hiddenSize: Int, streamCount: Int
) -> MLXArray {
    precondition(rawUpOut.shape == normed.shape)
    // In production both arrays are already the network's one activation
    // dtype (bf16) — this only differs in the synthetic parity bench, where
    // a quantized `Linear`'s bf16 scales can promote `upOut` away from the
    // bench's float16 convention (`Qwen4ExpLayerBench.checkParity`'s own
    // comment explains why it uses float16). Match `normed` explicitly so
    // the kernel's two inputs always share one type, same as the original
    // unfused path's implicit promotion would settle on before `.mean()`.
    let upOut = rawUpOut.dtype == normed.dtype ? rawUpOut : rawUpOut.asType(normed.dtype)
    var outputShape = upOut.shape
    outputShape[outputShape.count - 1] = hiddenSize
    let total = upOut.size / streamCount
    let width = qwen4ExpKernelThreadGroupWidth(total)
    let outputs = qwen4ExpHyperMixKernel(
        [upOut, normed],
        template: [("H", hiddenSize), ("S", streamCount), ("T", upOut.dtype)],
        grid: (total, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [upOut.dtype])
    return outputs[0]
}

/// Fuses `branch.expandedDimensions(axis: -2) * weights.expandedDimensions(axis: -1)`
/// then `.reshaped(hyperInput.shape)` then `hyperInput + …` into one kernel
/// call. `branch`: `[..., hiddenSize]`; `hyperInput`: `[..., streamCount *
/// hiddenSize]`; `weights`: `[..., streamCount]`; all row-contiguous, same
/// dtype. Returns `hyperInput`'s shape.
private let qwen4ExpHyperInjectKernel = MLXFast.metalKernel(
    name: "qwen4exp_hyper_inject",
    inputNames: ["hyperInput", "branch", "weights"],
    outputNames: ["out"],
    source: """
        uint idx = thread_position_in_grid.x;
        uint h = idx % H;
        uint rem = idx / H;
        uint s = rem % S;
        uint n = rem / S;
        float branchValue = float(branch[n * H + h]);
        float weightValue = float(weights[n * S + s]);
        out[idx] = hyperInput[idx] + static_cast<T>(branchValue * weightValue);
    """)

/// Used by `Qwen4ExpDecoderLayer.inject` (a different file, same target) —
/// hence `internal` rather than `private` like the mix-side helpers above,
/// which are only called from `Qwen4ExpGatedResidual` in this same file.
func qwen4ExpHyperInjectFused(
    branch: MLXArray, hyperInput: MLXArray, weights: MLXArray, hiddenSize: Int, streamCount: Int
) -> MLXArray {
    // `branch`/`weights` are read through an explicit `float(...)` cast
    // inside the kernel (see source above), so they need not share
    // `hyperInput`'s dtype for correctness — only the output does, matching
    // `hyperInput`'s own dtype (the addition's dominant operand, same as the
    // original unfused `hyperInput + injection...` in the common case where
    // all three already share one dtype, true everywhere in production).
    let total = hyperInput.size
    let width = qwen4ExpKernelThreadGroupWidth(total)
    let outputs = qwen4ExpHyperInjectKernel(
        [hyperInput, branch, weights],
        template: [("H", hiddenSize), ("S", streamCount), ("T", hyperInput.dtype)],
        grid: (total, 1, 1),
        threadGroup: (width, 1, 1),
        outputShapes: [hyperInput.shape],
        outputDTypes: [hyperInput.dtype])
    return outputs[0]
}

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

    /// P7.1: which sub-block, if any, `callAsFunction` short-circuits for
    /// `flash-layer-bench --ablate`. `.none` everywhere in production —
    /// see `Qwen4ExpLayerBenchAblation`.
    public let ablation: Qwen4ExpLayerBenchAblation
    /// P10.2 (F8): threaded like `ablation` — a construction-time behavior
    /// flag (like F4's `preciseRouterSoftmax`), not a loaded weight, so it
    /// is not applied via `prepareFusion`.
    private let fusionLevel: Qwen4ExpFusionLevel

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        useCombine: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        parityPrefix: String = "",
        ablation: Qwen4ExpLayerBenchAblation = .none,
        fusionLevel: Qwen4ExpFusionLevel = .none
    ) {
        hiddenSize = configuration.hiddenSize
        streamCount = configuration.hcCount
        self.parityPrefix = parityPrefix
        self.ablation = ablation
        self.fusionLevel = fusionLevel
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

        // P7.1 (`--ablate hyper`): skip the mix (`hc_norm` plus the two
        // low-rank gating matmuls), keeping only the plain stream-mean
        // needed for `mixedInput`'s shape and a constant injection-weight
        // tensor of the right shape. `inject` itself keeps running (see
        // `Qwen4ExpLayerBenchAblation`). Never numerically correct.
        if ablation == .hyper {
            let streams = hyperInput.reshaped(
                [hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
            let mixed = streams.mean(axis: -2)
            let injection = MLXArray.ones(
                [hyperInput.dim(0), hyperInput.dim(1), streamCount], dtype: hyperInput.dtype)
            return (mixed, hyperInput, injection)
        }

        // P7.1 (`--ablate norms`): skip hc_norm, reusing its already
        // shape-correct input for the rest of the mix. Never numerically
        // correct.
        let normed = ablation == .norms ? hyperInput : hcNorm(hyperInput)
        let upOut = inputMixWeightUp(silu(inputMixWeightDown(normed) / Float(streamCount)))
        // P10.2 (F8): `upOut`/`normed` are both `[..., streamCount *
        // hiddenSize]`, row-contiguous — the fused kernel replaces
        // `sigmoid(upOut).reshaped(...) * normed.reshaped(...)).mean(axis: -2)`
        // (5 ops: sigmoid, 2 reshapes, multiply, mean) with 1 kernel call.
        let mixed =
            fusionLevel >= .f8HyperConnectionKernel
            ? qwen4ExpHyperMixFused(
                upOut: upOut, normed: normed, hiddenSize: hiddenSize, streamCount: streamCount)
            : (sigmoid(upOut).reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
                * normed.reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize]))
                .mean(axis: -2)
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
        let upOut = inputMixWeightUp(silu(inputMixWeightDown(normed) / Float(streamCount)))
        if fusionLevel >= .f8HyperConnectionKernel {
            return qwen4ExpHyperMixFused(
                upOut: upOut, normed: normed, hiddenSize: hiddenSize, streamCount: streamCount)
        }
        let mix = sigmoid(upOut)
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
