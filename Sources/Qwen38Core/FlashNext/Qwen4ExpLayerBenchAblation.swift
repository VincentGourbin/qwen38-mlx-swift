import Foundation

/// P7.1 — per-sub-block cost attribution on `flash-layer-bench`
/// (`--ablate <bloc>`). Each case short-circuits one sub-block of a
/// Flash-Next decoder layer with a shape-correct placeholder instead of the
/// real computation, so the rest of the layer still runs exactly as before
/// (same downstream shapes, same call sequence): the difference between the
/// ablated bench median and the unablated (`.none`) median attributes that
/// sub-block's cost by subtraction. See PLAN.md, "P7 — Débit de génération :
/// attribuer le coût par sous-bloc, puis fusionner", tâche P7.1.
///
/// This is a **measurement instrument only** — never numerically correct,
/// never reached by production code. Every module that accepts an
/// `ablation` parameter defaults to `.none`, and every existing production
/// call site (`Qwen4ExpCheckpointLayerLoader`, `Qwen4ExpStreamingDecoder`,
/// `Qwen4ExpLayerBench.checkParity`) never passes anything else — only
/// `flash-layer-bench --ablate <bloc>` ever constructs a non-`.none` value.
public enum Qwen4ExpLayerBenchAblation: String, Sendable, CaseIterable {
    /// No ablation: the original, unmodified layer (the baseline every
    /// other case is measured against).
    case none

    /// Short-circuits `Qwen4ExpSparseMoE.callAsFunction` (router softmax,
    /// `argPartition`, the `SwitchGLU` expert gather, the shared expert) to
    /// `return x` — the routed+shared MoE branch becomes a no-op residual.
    case moe

    /// P7.1 sub-probe (informs P7.4's fusion target inside `moe`, once `moe`
    /// is designated dominant): keeps the router (`gate`, `softmax`,
    /// `argPartition`, `takeAlong`, top-k normalization) running for real,
    /// but zeroes both the routed `SwitchGLU` output and the shared expert.
    /// `T(.moe) ` vs. `T(.moeRouting)` isolates the router's own cost.
    case moeRouting = "moe-routing"
    /// P7.1 sub-probe: zeroes only the routed `SwitchGLU` expert gather,
    /// keeping the router and the shared expert real. `T(.none) -
    /// T(.moeSwitchMLP)` isolates `SwitchGLU`'s cost.
    case moeSwitchMLP = "moe-switch-mlp"
    /// P7.1 sub-probe: zeroes only the shared expert (and its gate),
    /// keeping the router and `SwitchGLU` real. `T(.none) -
    /// T(.moeSharedExpert)` isolates the shared expert's cost.
    case moeSharedExpert = "moe-shared-expert"

    /// Short-circuits `gatedDeltaUpdate`/`gatedDeltaUpdateWithStates` (the
    /// recurrent delta-rule kernel, `Vendor/mlx-swift-lm`) inside
    /// `Qwen4ExpGatedDeltaNet`, substituting `out = v` and a zeroed state of
    /// the same shape. The four input projections, `conv1d` and the output
    /// norm/projection keep running unchanged.
    case gdnRecurrence = "gdn-recurrence"

    /// Short-circuits the four input projections (`in_proj_qkv/z/b/a`) and
    /// `conv1d` inside `Qwen4ExpGatedDeltaNet` with zeroed placeholders,
    /// then runs the real recurrence and output norm/projection on them.
    case gdnProjections = "gdn-projections"

    /// Short-circuits the QSA indexer (`Qwen4ExpQSAIndexer.makeMask`) and
    /// `MLXFast.scaledDotProductAttention` inside `Qwen4ExpQSAAttention`
    /// with a zeroed `attended` tensor, keeping q/k/v projections, RoPE, the
    /// KV-cache update and the output gate/projection intact.
    case qsaAttn = "qsa-attn"

    /// Short-circuits the hyper-connection mix (`hc_norm` plus the two
    /// low-rank gating matmuls) inside `Qwen4ExpGatedResidual`, replacing it
    /// with the plain stream-mean needed to keep `mixedInput`'s shape
    /// correct and a constant injection-weight tensor. The final elementwise
    /// injection (`Qwen4ExpDecoderLayer.inject`) is left running: P2-fusion
    /// F3 (docs/knowledge/log.md, 2026-09-09) already measured it at ~2
    /// kernels, judged negligible to isolate further.
    case hyper

    /// Short-circuits every `Qwen4ExpRMSNorm`/`Qwen4ExpRMSNormGated` call in
    /// the layer (`hc_norm` ×2 in both hyper-connections, QSA `q_norm`/
    /// `k_norm`, GDN's gated output norm), substituting each with its
    /// already shape-correct input. Out of scope: GDN's inline q/k
    /// L2-normalization (not a `Module`, not RMSNorm) and the QSA indexer's
    /// internal layernorms — see docs/knowledge/log.md, 2026-09-11 "P7 :
    /// débit de génération" for the exact boundary this case covers.
    case norms
}

/// P11.2 : erreur de résolution pour `qwen4ExpResolveAblation` — une chaîne
/// reçue en CLI (`--ablate`) ou dans le champ de requête serveur `ablation`
/// qui ne correspond à aucun cas de `Qwen4ExpLayerBenchAblation`.
public enum Qwen4ExpAblationResolutionError: LocalizedError, Equatable {
    case invalidValue(String, choices: [String])

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let value, let choices):
            return
                "ablation doit être l'un de : \(choices.joined(separator: ", ")) ; reçu \"\(value)\"."
        }
    }
}

/// P11.2 : résout une chaîne (le `rawValue` d'un cas de
/// `Qwen4ExpLayerBenchAblation`, `"none"` compris) en la valeur d'ablation
/// correspondante. Ne décide pas de ce que signifie une valeur *absente* —
/// c'est au caller de choisir : `flash-chat-probe --ablate` (absent ⇒
/// `.none`, un process CLI neuf part toujours de zéro) et le champ de
/// requête serveur `ablation` (absent ⇒ ne touche pas au réglage en
/// vigueur, voir `Qwen38GenerationOptions.ablation`) ont des conventions
/// différentes pour ce cas — d'où une fonction qui prend une chaîne non
/// optionnelle plutôt que de trancher elle-même.
///
/// Fonction pure, testable sans charger de checkpoint réel — voir
/// `AblationResolutionTests`.
public func qwen4ExpResolveAblation(rawValue: String) throws -> Qwen4ExpLayerBenchAblation {
    guard let parsed = Qwen4ExpLayerBenchAblation(rawValue: rawValue) else {
        throw Qwen4ExpAblationResolutionError.invalidValue(
            rawValue, choices: Qwen4ExpLayerBenchAblation.allCases.map(\.rawValue))
    }
    return parsed
}
