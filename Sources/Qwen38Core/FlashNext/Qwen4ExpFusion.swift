import MLX
import MLXNN

/// P2-fusion (docs/knowledge/log.md, "P2-fusion : leviers F1-F7") — cumulative
/// kernel-reduction levers applied to one decoder layer after its checkpoint
/// weights are loaded. Level `n` enables levers `F1...Fn`; `.none` (the
/// default everywhere) is the original, already-validated numerical path.
///
/// This is threaded through constructors/loaders as an ordinary parameter
/// (like `quantization`), not a global mutable variable: Swift 6 strict
/// concurrency makes a shared mutable global awkward, and constructor
/// injection matches the existing style of this codebase.
public enum Qwen4ExpFusionLevel: Int, Sendable, Comparable, CaseIterable {
    case none = 0
    /// F1: fuse quantized input projections that share the same input and
    /// quantization spec (GDN `in_proj_{qkv,z,b,a}`, QSA `q/k/v_proj`) into
    /// one matmul, split after the call.
    case f1InputProjections = 1
    /// F2: precompute the checkpoint's `1 + weight` RMSNorm convention once
    /// at load time and route the ungrouped case through `MLXFast.rmsNorm`.
    case f2PrecomputedNorms = 2
    case f3HyperConnections = 3
    case f4MoE = 4
    case f5Casts = 5
    case f6Compile = 6
    /// F7 (P8.2, 2026-09-11 — unrelated to the retired P7.4 gate/up fusion
    /// that once held this number): restores the checkpoint's intended
    /// working dtype (bf16) at two points where it was silently promoted to
    /// float32 and never rounded back down — `Qwen4ExpGatedDeltaNet`'s
    /// recurrence output before `Qwen4ExpRMSNormGated` (whose own
    /// `.asType(inputs.dtype)` rounds to *whatever it is handed*, which was
    /// the recurrence's raw fp32 output, not the network's bf16), and
    /// `Qwen4ExpQSAAttention`'s query/key after `Qwen4ExpMRoPE.apply` (whose
    /// float32 `cos`/`sin` tables promote the RoPE output to fp32 with no
    /// downcast). Both leaks make every downstream op — including the
    /// entire MoE branch — run its fp32 path, ~12-27× slower per
    /// `op-overhead-probe`'s `SwitchGLU` measurements (docs/knowledge/log.md,
    /// "P8 : ..."). `false` (`.none` through `.f6Compile`) is the original,
    /// leaking behavior.
    /// Défaut de production depuis le 2026-09-11. Ce n'est pas une
    /// optimisation mais une **mise en conformité avec la référence** :
    /// `Scripts/references/vlm_q4_language.py` termine `Qwen4ExpRMSNorm`
    /// (l. 597) et `Qwen4ExpRMSNormGated` (l. 615) par `.astype(dtype)`,
    /// c'est-à-dire un retour au dtype d'entrée (bf16). Notre chemin laissait
    /// fuir le float32 de l'état GDN et des tables MRoPE dans toute la suite
    /// de la couche, ce qui faisait basculer le MoE sur son chemin lent
    /// (SwitchGLU isolé : 150 µs en bf16, 4022 µs en fp32). Mesuré sur le
    /// checkpoint 3-bit réel : 5,91 → 12,92 tok/s, IDs bit-identiques,
    /// +0,8 Go de pic. Repasser à `.none` pour comparer.
    case f7GatedBranchDtype = 7
    /// F8 (P10.2, 2026-09-12): fuses the hyper-connection "mix" reduction
    /// (`sigmoid(upOut).reshaped(...) * normed.reshaped(...)).mean(axis: -2)`,
    /// 5 ops) and `Qwen4ExpDecoderLayer.inject`'s broadcast-multiply-add
    /// (`expandedDimensions` ×2, multiply, reshape, add — also 5 ops) into
    /// two `MLXFast.metalKernel` calls, called twice per layer
    /// (`attn_hyper_connection`, `mlp_hyper_connection`) plus once per token
    /// at the final reduction before `lm_head`. Pure glue around the
    /// existing (quantized) matmuls, which are untouched.
    ///
    /// **Deliberately given a number above `.f7GatedBranchDtype`, not reusing
    /// the long-unused `.f3HyperConnections` slot**: `Qwen4ExpFusionLevel`'s
    /// `>=` comparisons are cumulative (a level activates every lever at or
    /// below its own number), and production has defaulted to F7 since
    /// 2026-09-11. Wiring new behavior into a number below 7 would have
    /// silently activated it in production the moment it was wired in,
    /// skipping the bench-then-checkpoint validation gate every other lever
    /// in this campaign went through. `.f3HyperConnections` (and `.f4MoE`'s
    /// original P2-fusion sense — MoE softmax precision is a separate,
    /// already-cumulative concern) stay as-is; see P10.6 for the promised
    /// cleanup of this numbering.
    case f8HyperConnectionKernel = 8
    /// F9 (P10.3, 2026-09-12): fuses GDN's q/k L2-normalization
    /// (`x * rsqrt((x*x).sum(axis:-1,keepDims:true) + eps) * scale`, 4-5 ops
    /// each) into one `MLXFast.metalKernel` call per tensor. Explicitly
    /// **outside** `gatedDeltaUpdate` (Vendor/mlx-swift-lm, not patched —
    /// see PLAN.md §1.1): the gating transforms named in PLAN.md's P10.3 row
    /// (`-exp(A_log)·softplus(a+dt_bias)`, `sigmoid(b)`) already live inside
    /// that vendored function's own body
    /// (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift`), so
    /// this only touches `Qwen4ExpGatedDeltaNet`'s own q/k-normalization
    /// glue.
    case f9GdnL2NormKernel = 9

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

    fileprivate func isReached(by level: Qwen4ExpFusionLevel) -> Bool {
        level.rawValue >= rawValue
    }
}

/// Fuse several `Linear`/`QuantizedLinear` modules that all consume the same
/// input into a single module, by concatenating their output axis (axis 0).
///
/// This is exact, not an approximation: `QuantizedLinear` packs weights,
/// scales and affine biases per output row (`[out, in*bits/32]`,
/// `[out, in/g]`) — see `Qwen4ExpPrequantized.swift`. Concatenating several
/// such tensors on axis 0 and running one matmul, then splitting the result
/// on the matching output widths, produces bit-identical results to calling
/// each original module separately: every output row's dot product only
/// depends on that row's own weight/scale/bias, never on any other row.
/// Preconditions enforce that every input actually shares one input width
/// and one quantization spec (or is plain `Linear`), which is the only case
/// this reasoning holds for.
func qwen4ExpFuseLinear(_ layers: [Linear]) -> Linear {
    precondition(layers.count >= 2, "La fusion n'a de sens qu'à partir de deux projections")
    if layers.allSatisfy({ $0 is QuantizedLinear }) {
        let quantized = layers.map { $0 as! QuantizedLinear }
        let groupSize = quantized[0].groupSize
        let bits = quantized[0].bits
        let mode = quantized[0].mode
        precondition(
            quantized.allSatisfy { $0.groupSize == groupSize && $0.bits == bits && $0.mode == mode },
            "Les projections fusionnées doivent partager le même spec de quantification")
        let weight = concatenated(quantized.map { $0.weight }, axis: 0)
        let scales = concatenated(quantized.map { $0.scales }, axis: 0)
        let hasBiases = quantized[0].biases != nil
        precondition(
            quantized.allSatisfy { ($0.biases != nil) == hasBiases },
            "Les projections fusionnées doivent toutes avoir (ou non) des biais affines")
        let biases = hasBiases ? concatenated(quantized.map { $0.biases! }, axis: 0) : nil
        let bias = qwen4ExpFusedLinearBias(quantized)
        return QuantizedLinear(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode)
    }
    precondition(
        !layers.contains(where: { $0 is QuantizedLinear }),
        "Impossible de mélanger des projections quantifiées et non quantifiées")
    let weight = concatenated(layers.map { $0.weight }, axis: 0)
    let bias = qwen4ExpFusedLinearBias(layers)
    return Linear(weight: weight, bias: bias)
}

private func qwen4ExpFusedLinearBias(_ layers: [Linear]) -> MLXArray? {
    guard layers.contains(where: { $0.bias != nil }) else { return nil }
    return concatenated(layers.map { $0.bias ?? MLXArray.zeros([$0.shape.0]) }, axis: 0)
}

extension Qwen4ExpDecoderLayer {
    /// Applies the P2-fusion levers reached by `level` to an already-loaded
    /// layer (called once, right after `update(parameters:verify:)`, both by
    /// `Qwen4ExpCheckpointLayerLoader` and `Qwen4ExpLayerBench`). `.none`
    /// does nothing, leaving the original path exactly as before.
    public func prepareFusion(level: Qwen4ExpFusionLevel) {
        guard level != .none else { return }
        if Qwen4ExpFusionLevel.f1InputProjections.isReached(by: level) {
            linearAttn?.fuseInputProjections()
            selfAttn?.fuseInputProjections()
        }
        if Qwen4ExpFusionLevel.f2PrecomputedNorms.isReached(by: level) {
            for module in modules() {
                (module as? Qwen4ExpRMSNorm)?.precomputeEffectiveWeight()
            }
        }
    }
}
