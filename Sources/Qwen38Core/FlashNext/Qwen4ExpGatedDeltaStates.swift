import MLX
import MLXNN

/// PM4.1 (2026-09-09, P-MTP suite): verification-mode variant of
/// `gatedDeltaUpdate` (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/GatedDelta.swift`)
/// that returns the recurrent state after **each** of the T fed tokens, not
/// just the state after the last one.
///
/// Neither the fused Metal kernel nor the ops fallback in the vendored
/// package expose intermediate states — both only return `(y, finalState)` —
/// and `Vendor/` is off limits (PLAN.md §0). This file is therefore a
/// from-scratch reimplementation of the ops fallback's per-token loop
/// (`gatedDeltaOps`/`gatedDeltaStepOps`, same file), copied rather than
/// called because those symbols carry no access modifier and are not visible
/// outside the `MLXLMCommon` module. It is used **only** by the MTP
/// verification forward (`Qwen4ExpVerificationCapture` non-nil in
/// `Qwen4ExpGatedDeltaNet`); the hot greedy decode path keeps calling the
/// public `gatedDeltaUpdate` (kernel when available) unchanged.
///
/// `gatedDeltaStepOps`/`gatedDeltaG` below are drop-in copies of the private
/// vendor helpers so the per-step math stays byte-for-byte identical to the
/// ops fallback used for parity elsewhere in this codebase.
func qwen4ExpGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
}

private func qwen4ExpGatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (y: MLXArray, state: MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

/// Runs the gated delta-rule recurrence one token at a time and returns the
/// recurrent state after every step, in addition to the usual output.
///
/// - Returns: `y` shaped like `gatedDeltaUpdate`'s output (`[B, T, Hv, Dv]`)
///   and `states` shaped `[B, T, Hv, Dv, Dk]` where `states[:, i]` is the
///   state after processing the first `i + 1` input tokens (so
///   `states[:, T - 1]` equals the final state `gatedDeltaUpdate` would
///   have returned for the same inputs).
public func gatedDeltaUpdateWithStates(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (y: MLXArray, states: MLXArray) {
    let beta = sigmoid(b).asType(.float32)
    let g = qwen4ExpGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let Dk = q.dim(3)

    var qr = q
    var kr = k
    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        qr = MLX.repeated(q, count: repeatFactor, axis: -2)
        kr = MLX.repeated(k, count: repeatFactor, axis: -2)
    }

    var currentState = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if currentState.dtype != .float32 {
        currentState = currentState.asType(.float32)
    }

    var ys = [MLXArray]()
    var states = [MLXArray]()
    ys.reserveCapacity(T)
    states.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = qr[0..., t]
        let kT = kr[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = qwen4ExpGatedDeltaStepOps(
            q: qT, k: kT, v: vT, g: gT, beta: betaT, state: currentState, mask: maskT)
        ys.append(y)
        currentState = newState
        states.append(newState)
    }

    let yOut = MLX.stacked(ys, axis: 1)
    let statesOut = MLX.stacked(states, axis: 1)
    return (yOut, statesOut)
}
