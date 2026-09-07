import MLX
import MLXLMCommon
import MLXNN

/// Gated DeltaNet branch used by Flash-Next's `linear_attention` layers.
///
/// The recurrence itself is delegated to the public `gatedDeltaUpdate`
/// primitive from mlx-swift-lm. This local wrapper owns the Flash-Next tensor
/// contract: depthwise causal convolution, split projections, float32 state,
/// gated RMSNorm and cache advancement.
public final class Qwen4ExpGatedDeltaNet: Module {
    public let hiddenSize: Int
    public let numValueHeads: Int
    public let numKeyHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let keyDim: Int
    public let valueDim: Int
    public let convKernelSize: Int
    public let convDim: Int

    @ModuleInfo(key: "conv1d") public var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") public var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") public var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") public var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") public var inProjA: Linear
    @ParameterInfo(key: "dt_bias") public var dtBias: MLXArray
    @ParameterInfo(key: "A_log") public var aLog: MLXArray
    @ModuleInfo(key: "norm") public var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    /// Debug-only observability for checkpoint parity probes. It is replaced
    /// on every call and deliberately contains only the small GDN boundary
    /// tensors, not the recurrent history beyond the cache itself.
    public private(set) var lastParityCapture: [String: MLXArray] = [:]
    /// Disabled during normal inference so intermediate graphs are not kept alive.
    public private(set) var captureParity = false

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil
    ) {
        hiddenSize = configuration.hiddenSize
        numValueHeads = configuration.linearNumValueHeads
        numKeyHeads = configuration.linearNumKeyHeads
        keyHeadDim = configuration.linearKeyHeadDim
        valueHeadDim = configuration.linearValueHeadDim
        keyDim = keyHeadDim * numKeyHeads
        valueDim = valueHeadDim * numValueHeads
        convKernelSize = configuration.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        precondition(numValueHeads % numKeyHeads == 0)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: convKernelSize, stride: 1, padding: 0,
            dilation: 1, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = qwen4ExpLinear(
            inputDimensions: hiddenSize, outputDimensions: keyDim * 2 + valueDim,
            quantization: quantization)
        _inProjZ.wrappedValue = qwen4ExpLinear(
            inputDimensions: hiddenSize, outputDimensions: valueDim,
            quantization: quantization)
        _inProjB.wrappedValue = qwen4ExpLinear(
            inputDimensions: hiddenSize, outputDimensions: numValueHeads,
            quantization: quantization)
        _inProjA.wrappedValue = qwen4ExpLinear(
            inputDimensions: hiddenSize, outputDimensions: numValueHeads,
            quantization: quantization)
        _dtBias.wrappedValue = MLXArray.ones([numValueHeads])
        _aLog.wrappedValue = MLXArray.zeros([numValueHeads])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: valueHeadDim, eps: rmsNormEps,
            activation: configuration.outputGateType ?? configuration.hiddenAct ?? "silu")
        _outProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: valueDim, outputDimensions: hiddenSize,
            quantization: quantization)
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: ArraysCache? = nil
    ) -> MLXArray {
        precondition(inputs.ndim == 3)
        let batch = inputs.dim(0)
        let sequence = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped([batch, sequence, numValueHeads, valueHeadDim])
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let state = cache?[0] {
            convState = state
        } else {
            convState = MLXArray.zeros(
                [batch, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
        }
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache, convKernelSize > 1 {
            cache[0] = contiguous(convInput[.ellipsis, (-(convKernelSize - 1))..., 0...])
        }
        let convOut = silu(conv1d(convInput))
        let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = split[0].reshaped([batch, sequence, numKeyHeads, keyHeadDim])
        let k = split[1].reshaped([batch, sequence, numKeyHeads, keyHeadDim])
        let v = split[2].reshaped([batch, sequence, numValueHeads, valueHeadDim])

        // Flash-Next overrides the Qwen3-Next RMS normalization with L2
        // normalization (epsilon after the sum), followed by 1/sqrt(Dk).
        // Keeping this explicit is important: rmsNorm would normalize by the
        // mean square and changes the recurrence scale by sqrt(Dk).
        let scale = 1.0 / Float(keyHeadDim).squareRoot()
        let qNormed = q * MLX.rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * MLXArray(scale).asType(q.dtype)
        // The Flash-Next reference scales only the query after L2
        // normalization. Scaling K as well changes the delta-rule update and
        // is especially visible when a recurrent cache is already warm.
        let kNormed = k * MLX.rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)

        let (out, state) = gatedDeltaUpdate(
            q: qNormed, k: kNormed, v: v, a: a, b: b,
            aLog: aLog, dtBias: dtBias,
            state: cache?[1], mask: mask)
        if captureParity {
            lastParityCapture = [
                "q_normed": qNormed, "k_normed": kNormed,
                "v": v, "gdn_output": out,
            ]
        } else {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
        if let cache {
            cache[1] = state
            cache.advance(sequence)
        }

        let normalized = norm(out.reshaped([batch, sequence, numValueHeads, valueHeadDim]), gate: z)
        let projected = outProj(normalized.reshaped([batch, sequence, valueDim]))
        if captureParity {
            lastParityCapture["norm_output"] = normalized
            lastParityCapture["attn_branch"] = projected
        }
        return projected
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }
}

public final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    public let eps: Float
    public let activation: String

    public init(dimensions: Int, eps: Float = 1e-6, activation: String = "silu") {
        self.eps = eps
        self.activation = activation
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, gate: MLXArray) -> MLXArray {
        // Match mlx-vlm's Qwen4ExpRMSNormGated exactly: normalize in the
        // input dtype, promote both operands to fp32 for the gated product,
        // then round once at the branch boundary.  Multiplying the bf16 norm
        // output directly changes the first sensitive MoE layer much more
        // than the checkpoint's intended bf16 rounding.
        let normalized = MLXFast.rmsNorm(inputs, weight: weight, eps: eps)
            .asType(.float32)
        let activatedGate = (activation == "sigmoid" ? sigmoid(gate) : silu(gate))
            .asType(.float32)
        return (normalized * activatedGate).asType(inputs.dtype)
    }
}

private extension Int {
    var squareRootFloat: Float { Float(self).squareRoot() }
}
