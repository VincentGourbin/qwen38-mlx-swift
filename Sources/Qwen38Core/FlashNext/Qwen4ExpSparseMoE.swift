import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// P8.1: accumulates the wall-clock cost of each `Qwen4ExpSparseMoE.
/// callAsFunction` sub-stage across many bench steps, forcing an `eval`
/// after every stage so MLX's laziness cannot silently attribute an earlier
/// stage's compute to whichever stage happens to trigger the graph's
/// synchronization (see docs/knowledge/log.md, "P7 revisité", point 5, and
/// PLAN.md P8.1). **Diagnostic instrument only**: non-nil exclusively behind
/// `flash-layer-bench --moe-stages`; `nil` everywhere else (including every
/// production call site), so the extra synchronizations this class forces
/// never reach `Qwen4ExpStreamingDecoder`. Because each `eval` is itself a
/// host/GPU synchronization barrier, the sum of stage totals over-counts the
/// undisturbed layer's cost by roughly (stage count − 1) sync barriers —
/// `flash-layer-bench --moe-stages` reports and subtracts an independently
/// measured per-`eval` baseline rather than leaving that caveat implicit.
public final class Qwen4ExpMoEStageProfiler: @unchecked Sendable {
    public private(set) var totalSeconds: [String: Double] = [:]
    public private(set) var stageOrder: [String] = []
    public private(set) var callCount: Int = 0

    public init() {}

    /// Times `body`, forces an `eval` on its result before stopping the
    /// clock, and accumulates the elapsed wall time under `label`.
    func time(_ label: String, _ body: () -> MLXArray) -> MLXArray {
        let start = ContinuousClock.now
        let result = body()
        eval(result)
        let elapsed = (ContinuousClock.now - start).seconds
        if totalSeconds[label] == nil { stageOrder.append(label) }
        totalSeconds[label, default: 0] += elapsed
        return result
    }

    func markStepDone() { callCount += 1 }
}

private extension Duration {
    var seconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

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

/// P11.1 : erreur de validation pour `routedExpertCount`, la largeur de
/// routage du MoE réglable à l'exécution (`num_experts_per_tok` du
/// checkpoint, surchargeable sans toucher aux poids).
public enum Qwen4ExpRoutedExpertCountError: LocalizedError, Equatable {
    case outOfBounds(requested: Int, numExperts: Int)

    public var errorDescription: String? {
        switch self {
        case .outOfBounds(let requested, let numExperts):
            return
                "routedExpertCount doit être compris entre 1 et \(numExperts) (nombre total d'experts du checkpoint) ; reçu \(requested)."
        }
    }
}

/// P11.1 : résout la largeur de routage MoE effective. `override`, quand
/// fourni, remplace `checkpointDefault` (`num_experts_per_tok` décodé par
/// `Qwen4ExpConfiguration`) ; `nil` laisse le comportement inchangé — c'est
/// le défaut partout (Edge0 divise K par deux sur son tier phare
/// `edge0-35b`, PLAN.md P11.1 : le but est de pouvoir balayer K sans
/// retoucher le checkpoint). `topK` ne dimensionne aucun tenseur — il ne
/// choisit que combien d'experts `argPartition` sélectionne par token — donc
/// cette résolution ne recharge jamais de poids.
///
/// Fonction pure, testable sans charger de checkpoint réel — voir
/// `RoutedExpertCountTests`.
public func qwen4ExpResolveRoutedExpertCount(
    override: Int?,
    checkpointDefault: Int,
    numExperts: Int
) throws -> Int {
    guard let override else { return checkpointDefault }
    guard override >= 1, override <= numExperts else {
        throw Qwen4ExpRoutedExpertCountError.outOfBounds(requested: override, numExperts: numExperts)
    }
    return override
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
    /// P11.1 : largeur de routage effective (`num_experts_per_tok` du
    /// checkpoint, sauf surcharge explicite — voir
    /// `qwen4ExpResolveRoutedExpertCount`). Mutable (contrairement au reste
    /// de cette classe) : c'est un entier qui pilote `argPartition` dans
    /// `callAsFunction`, pas la forme d'un tenseur chargé, donc le changer
    /// après coup sur une couche déjà résidente ne touche à aucun poids —
    /// voir `setRoutedExpertCount`.
    public private(set) var topK: Int
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

    /// P8.1: non-nil only under `flash-layer-bench --moe-stages` — see
    /// `Qwen4ExpMoEStageProfiler`.
    public let moeStageProfiler: Qwen4ExpMoEStageProfiler?

    public init(
        configuration: Qwen4ExpTextConfiguration,
        normalizeTopK: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil,
        fusionLevel: Qwen4ExpFusionLevel = .none,
        ablation: Qwen4ExpLayerBenchAblation = .none,
        moeStageProfiler: Qwen4ExpMoEStageProfiler? = nil,
        /// P11.1 : surcharge de `num_experts_per_tok`. `nil` (partout par
        /// défaut) reproduit exactement le comportement précédent. Un
        /// appelant qui connaît déjà `numExperts` (le décodeur, qui a
        /// validé via `qwen4ExpResolveRoutedExpertCount` avant d'arriver
        /// ici) peut passer une valeur hors bornes par erreur — la
        /// precondition ci-dessous reste le filet de sécurité local, mais
        /// le message d'erreur clair et récupérable vit dans cette
        /// fonction de résolution, pas ici.
        routedExpertCount: Int? = nil
    ) {
        numExperts = configuration.numExperts
        let checkpointDefault = configuration.numExpertsPerToken
        let resolvedTopK = routedExpertCount ?? checkpointDefault
        precondition(
            resolvedTopK >= 1 && resolvedTopK <= numExperts,
            "routedExpertCount (\(resolvedTopK)) doit être compris entre 1 et numExperts (\(numExperts))")
        topK = resolvedTopK
        self.normalizeTopK = normalizeTopK
        self.preciseRouterSoftmax = fusionLevel < .f4MoE
        self.ablation = ablation
        self.moeStageProfiler = moeStageProfiler

        precondition(numExperts > 0)
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
        // P8.1 diagnostic path: forces an `eval` after every sub-stage so
        // `flash-layer-bench --moe-stages` measures real, synchronized
        // wall-clock cost per stage instead of attributing everything to
        // whichever stage happens to trigger MLX's lazy graph. Never taken
        // in production (`moeStageProfiler` is `nil` at every real call
        // site) — see `Qwen4ExpMoEStageProfiler`.
        if let moeStageProfiler {
            return callWithStageProfiling(x, moeStageProfiler)
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

    /// P8.1: same computation as `callAsFunction`'s normal path, staged
    /// through `Qwen4ExpMoEStageProfiler.time` so each sub-block's cost is
    /// measured with a forced `eval` boundary. Deliberately a separate
    /// method (not an `if` inside the single expression above) so the
    /// production path above stays exactly as lazy/fused as before this
    /// task — this method is only ever reached when `moeStageProfiler` is
    /// non-nil, i.e. only from `flash-layer-bench --moe-stages`.
    private func callWithStageProfiling(
        _ x: MLXArray, _ stageProfiler: Qwen4ExpMoEStageProfiler
    ) -> MLXArray {
        let gateLogits = stageProfiler.time("gate") { gate(x) }
        let probabilities = stageProfiler.time("softmax") {
            MLX.softmax(gateLogits, axis: -1, precise: preciseRouterSoftmax)
        }
        let kth = numExperts - topK
        let indices = stageProfiler.time("argPartition") {
            MLX.argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
        }
        let scores = stageProfiler.time("takeAlong+normalize") { () -> MLXArray in
            var s = MLX.takeAlong(probabilities, indices, axis: -1)
            if normalizeTopK {
                s = s / s.sum(axis: -1, keepDims: true)
            }
            return s
        }
        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped([tokenCount, x.dim(-1)])
        let flatIndices = indices.reshaped([tokenCount, topK])
        let flatScores = scores.reshaped([tokenCount, topK])
        let switchOutput = stageProfiler.time("switchMLP") { switchMLP(flatX, flatIndices) }
        let routed = stageProfiler.time("weightedExpertSum") {
            weightedExpertSum(switchOutput, flatScores).reshaped(x.shape)
        }
        let sharedGateValue = stageProfiler.time("sharedExpertGate") { sigmoid(sharedExpertGate(x)) }
        let sharedExpertOutput = stageProfiler.time("sharedExpert") { sharedExpert(x) }
        let shared = stageProfiler.time("shared-combine") { sharedGateValue * sharedExpertOutput }
        let output = stageProfiler.time("add") { routed + shared }
        stageProfiler.markStepDone()
        return output
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }

    /// P11.1 : change `topK` sur une instance déjà construite (couche
    /// résidente ou non). Ne touche à aucun poids ni au graphe MLX — un
    /// forward ultérieur voit simplement une autre largeur d'`argPartition`.
    /// Contrairement à la precondition de l'initialiseur, cette voie est
    /// `throws` : elle est appelée depuis un chemin où l'appelant (le
    /// serveur, sur une requête) doit pouvoir rapporter une erreur claire au
    /// client plutôt que faire tomber le process.
    public func setRoutedExpertCount(_ count: Int) throws {
        guard count >= 1, count <= numExperts else {
            throw Qwen4ExpRoutedExpertCountError.outOfBounds(requested: count, numExperts: numExperts)
        }
        topK = count
    }
}
