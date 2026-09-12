import MLX
import MLXLMCommon
import MLXNN

// P10.3 (F9, 2026-09-12) — RETIRED: a `MLXFast.metalKernel` used to live
// here, fusing GDN's q/k L2-normalization (4-5 ops) into one kernel call
// per tensor. The gating math named in PLAN.md's P10.3 row
// (`-exp(A_log)·softplus(a+dt_bias)`, `sigmoid(b)`) lives inside the
// vendored `gatedDeltaUpdate` (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/
// GatedDelta.swift`, not patched — PLAN.md §1.1), so the L2-norm glue below
// was this task's actual fusable surface. Correct (bit-exact in isolation,
// bf16-rounding-noise-only at production dtype) but measured on the real
// 3-bit checkpoint, back-to-back alternating with the F7 baseline (6 runs,
// 32-token greedy decode): both within ~1 % of each other, no ≥5 % gain
// (PLAN.md §P10 criterion). Removed rather than kept as a dead opt-in. See
// git history for the removed kernel source, and docs/knowledge/log.md
// "P10.3" for the numbers.

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

    /// F1 (P2-fusion): `in_proj_qkv/z/b/a` concatenated into one matmul once
    /// `fuseInputProjections()` has run. Not an `@ModuleInfo` property: it
    /// carries no checkpoint key of its own and must stay invisible to
    /// `parameters()`/`update(parameters:)`. `nil` means the original,
    /// four-separate-matmuls path (unchanged behavior).
    private var fusedInProj: Linear?

    /// P7.1: which sub-block, if any, `callAsFunction` short-circuits for
    /// `flash-layer-bench --ablate`. `.none` everywhere in production —
    /// see `Qwen4ExpLayerBenchAblation`. P11.2 : mutable — voir
    /// `Qwen4ExpSparseMoE.ablation`'s doc comment pour pourquoi ce n'est pas
    /// un problème sur une instance résidente.
    public private(set) var ablation: Qwen4ExpLayerBenchAblation

    /// P8.2 (F7): whether the recurrence output is rounded to the network's
    /// working dtype before `Qwen4ExpRMSNormGated` — see
    /// `Qwen4ExpFusionLevel.f7GatedBranchDtype`.
    private let fusionLevel: Qwen4ExpFusionLevel

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        ablation: Qwen4ExpLayerBenchAblation = .none,
        fusionLevel: Qwen4ExpFusionLevel = .none
    ) {
        self.ablation = ablation
        self.fusionLevel = fusionLevel
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
        cache: ArraysCache? = nil,
        // PM4.1/PM4.2 (P-MTP suite): non-nil only for the MTP verification
        // forward. When set, the recurrence keeps every per-token state
        // (`gatedDeltaUpdateWithStates`) and records the conv1d window and
        // the state stack so a later partial rejection can roll `cache`
        // back to any prefix without replaying this forward. `nil` (every
        // other caller: prefill, greedy decode, drafting) is the original,
        // unmodified path.
        verificationSink: Qwen4ExpVerificationSink? = nil
    ) -> MLXArray {
        precondition(inputs.ndim == 3)
        let batch = inputs.dim(0)
        let sequence = inputs.dim(1)

        // P7.1 (`--ablate gdn-projections`): skip `in_proj_qkv/z/b/a` and
        // `conv1d` entirely, feeding zeroed but shape-correct q/k/v/z/a/b
        // straight into the real recurrence and output norm/projection.
        // Never numerically correct — a bench-only measurement instrument,
        // see `Qwen4ExpLayerBenchAblation`.
        if ablation == .gdnProjections {
            let qkZero = MLXArray.zeros(
                [batch, sequence, numKeyHeads, keyHeadDim], dtype: inputs.dtype)
            let vZero = MLXArray.zeros(
                [batch, sequence, numValueHeads, valueHeadDim], dtype: inputs.dtype)
            let gateZero = MLXArray.zeros([batch, sequence, numValueHeads], dtype: inputs.dtype)
            let (out, state) = gatedDeltaUpdate(
                q: qkZero, k: qkZero, v: vZero, a: gateZero, b: gateZero,
                aLog: aLog, dtBias: dtBias, state: cache?[1], mask: mask)
            if let cache {
                cache[1] = state
                cache.advance(sequence)
            }
            let normalized = norm(
                out.reshaped([batch, sequence, numValueHeads, valueHeadDim]), gate: vZero)
            return outProj(normalized.reshaped([batch, sequence, valueDim]))
        }

        var qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        if let fusedInProj {
            // F1: one matmul instead of four; the split widths exactly match
            // in_proj_qkv/z/b/a's original output widths, in that order.
            let fused = fusedInProj(inputs)
            let parts = MLX.split(
                fused,
                indices: [
                    keyDim * 2 + valueDim,
                    keyDim * 2 + valueDim + valueDim,
                    keyDim * 2 + valueDim + valueDim + numValueHeads,
                ],
                axis: -1)
            qkv = parts[0]
            z = parts[1].reshaped([batch, sequence, numValueHeads, valueHeadDim])
            b = parts[2]
            a = parts[3]
        } else {
            qkv = inProjQKV(inputs)
            z = inProjZ(inputs).reshaped([batch, sequence, numValueHeads, valueHeadDim])
            b = inProjB(inputs)
            a = inProjA(inputs)
        }

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
        // The Flash-Next reference scales only the query after L2
        // normalization. Scaling K as well changes the delta-rule update and
        // is especially visible when a recurrent cache is already warm.
        let qNormed = q * MLX.rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * MLXArray(scale).asType(q.dtype)
        let kNormed = k * MLX.rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)

        let out: MLXArray
        let state: MLXArray
        if ablation == .gdnRecurrence {
            // P7.1: skip the recurrent delta-rule kernel entirely, reusing
            // `v` (already the exact output shape) and a zeroed state of the
            // shape `gatedDeltaUpdate` would have produced/consumed
            // ([B, Hv, Dv, Dk], fp32). Never numerically correct — see
            // `Qwen4ExpLayerBenchAblation`.
            out = v
            state = cache?[1]
                ?? MLXArray.zeros(
                    [batch, numValueHeads, valueHeadDim, keyHeadDim], dtype: .float32)
        } else if let verificationSink {
            // PM4.1: keep every per-token state so a rejection can roll this
            // cache back to any prefix of the T new tokens without a replay
            // forward. `states[:, T-1]` is exactly the final state
            // `gatedDeltaUpdate` would have produced, so the full-accept
            // path (no rollback) is bit-identical to before.
            let (y, states) = gatedDeltaUpdateWithStates(
                q: qNormed, k: kNormed, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias,
                state: cache?[1], mask: mask)
            out = y
            state = states[0..., -1]
            verificationSink.record(
                slot: 1, entry: .stateAtIndex(source: states))
            if convKernelSize > 1 {
                verificationSink.record(
                    slot: 0,
                    entry: .window(source: convInput, length: convKernelSize - 1))
            }
        } else {
            (out, state) = gatedDeltaUpdate(
                q: qNormed, k: kNormed, v: v, a: a, b: b,
                aLog: aLog, dtBias: dtBias,
                state: cache?[1], mask: mask)
        }
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

        // P7.1 (`--ablate norms`): skip the gated RMSNorm, reusing its
        // already shape-correct input. Never numerically correct — see
        // `Qwen4ExpLayerBenchAblation`.
        // P8.2 (F7): `out` is the recurrence's raw output — kept in fp32 for
        // the recurrent state's own precision (piège #6). `norm` (see
        // `Qwen4ExpRMSNormGated.callAsFunction`) already normalizes and
        // gates entirely in fp32 for precision, and documents "round once
        // at the branch boundary" as its intended final step — but its own
        // `.asType(inputs.dtype)` rounds to *its local `inputs`'* dtype,
        // which is this fp32 `out`, not the network's bf16 working dtype:
        // that "round" is therefore a no-op, and the fp32 leaks into every
        // op downstream, including the whole MoE branch, which then runs
        // its (much slower) fp32 path. Below `f7GatedBranchDtype`, add the
        // missing round to bf16 right here, at the true branch boundary —
        // *after* `norm`'s fp32 reduction/gating, not before (rounding
        // `out` to bf16 before normalizing would instead throw away
        // precision `norm`'s own fp32 upcast is meant to preserve).
        let normalized: MLXArray
        if ablation == .norms {
            normalized = out.reshaped([batch, sequence, numValueHeads, valueHeadDim])
                .asType(inputs.dtype)
        } else {
            let gatedNorm = norm(out.reshaped([batch, sequence, numValueHeads, valueHeadDim]), gate: z)
            normalized = fusionLevel >= .f7GatedBranchDtype
                ? gatedNorm.asType(inputs.dtype) : gatedNorm
        }
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

    /// P11.2 : change `ablation` sur une instance déjà construite, sans
    /// recharger aucun poids — voir `Qwen4ExpSparseMoE.setAblation`.
    public func setAblation(_ new: Qwen4ExpLayerBenchAblation) {
        ablation = new
    }

    /// F1 (P2-fusion): build the fused `in_proj_qkv/z/b/a` matmul once, after
    /// the four checkpoint-shaped modules have their real loaded weights.
    /// Idempotent (a second call is a no-op) and safe to call unconditionally
    /// — callers gate it on `Qwen4ExpFusionLevel`, not this method.
    public func fuseInputProjections() {
        guard fusedInProj == nil else { return }
        let fused = qwen4ExpFuseLinear([inProjQKV, inProjZ, inProjB, inProjA])
        eval(fused.weight)
        if let quantized = fused as? QuantizedLinear {
            eval(quantized.scales)
            if let biases = quantized.biases { eval(biases) }
        }
        fusedInProj = fused
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
