import MLX
import MLXNN

/// P2-fusion (docs/knowledge/log.md, "P2-fusion : leviers F1-F7") — cumulative
/// levers applied to one decoder layer, threaded through constructors/
/// loaders as an ordinary parameter (like `quantization`, not a global
/// mutable variable — Swift 6 strict concurrency makes a shared mutable
/// global awkward). Level `n` activates every lever numbered at or below
/// `n` (`isReached(by:)`); `.none` is the original, pre-P2-fusion path.
///
/// **Two different kinds of lever share this one numbering, on purpose**
/// (P10.6 cleanup, 2026-09-12 — documented here rather than split into two
/// enums, which would have meant re-threading every call site for a
/// renaming-only change):
/// - **Collage fusions** (F1, F2, F4): exact reorderings of existing ops —
///   concatenate-then-split matmuls, precomputed norm weights, an
///   imprecise-but-provably-equivalent MoE routing softmax. Individually
///   measured *neutral* on the real checkpoint (docs/knowledge/log.md "P7",
///   "P8") — kept because they cost nothing and the default level (F7)
///   already includes them cumulatively.
/// - **Correctness fix** (F7): **not an optimization**. It restores the
///   reference implementation's dtype behavior at a genuine bug (see its
///   own doc comment below) — this is why F7, not `.none`, is the default
///   everywhere, and why a reader should not treat it as an optional perf
///   knob the way F1/F2/F4 are.
///
/// `.f3HyperConnections`, `.f5Casts` and `.f6Compile` are reserved slots
/// from an earlier revision of this plan and were **never implemented** —
/// no code anywhere checks for them. Left in place (not removed) so old
/// logs/scripts that refer to "F1-F6" as a contiguous range stay
/// meaningful, but they do nothing if passed to `--fusion-level`.
///
/// F8 et F9 ont d'abord désigné une paire de noyaux `MLXFast.metalKernel`
/// (mix/inject des hyper-connexions, P10.2 ; L2-norm q/k du GDN, P10.3),
/// construits, validés pour l'exactitude, puis mesurés sur le checkpoint
/// réel — F8 une petite régression (~1 %) mais régulière, F9 dans le bruit
/// de F7 dans un sens comme dans l'autre, aucun des deux ne franchissant la
/// barre des ≥5 % exigée par cette campagne (docs/knowledge/log.md,
/// "P10.2"/"P10.3"). Leur code a été retiré, pas seulement leur défaut,
/// suivant le même précédent que le noyau `switch_mlp` retiré du
/// P2-fusion — voir l'historique git et les commentaires de tête de
/// `Qwen4ExpHyperConnection.swift`/`Qwen4ExpGatedDeltaNet.swift` pour le
/// détail de ce qui a été retiré.
///
/// **2026-09-13 (P11, « le bon toit ») réattribue ces deux numéros** à une
/// approche différente sur un diagnostic différent : `MLX.compile` plutôt
/// qu'un noyau Metal écrit à la main, ciblant le coût de lancement par
/// opération des hyper-connexions et de l'expert partagé — mesurés à 21,8×
/// et 32,2× leur toit de bande passante malgré le trafic d'octets le plus
/// faible du modèle (docs/knowledge/log.md, 2026-09-13, « Le bon toit »).
/// Voir le détail de chaque niveau ci-dessous. La réutilisation des numéros
/// est délibérée, pas un oubli — consulter l'historique git avant de
/// retoucher l'un ou l'autre cas.
public enum Qwen4ExpFusionLevel: Int, Sendable, Comparable, CaseIterable {
    case none = 0
    /// F1 (collage): fuse quantized input projections that share the same
    /// input and quantization spec (GDN `in_proj_{qkv,z,b,a}`, QSA
    /// `q/k/v_proj`) into one matmul, split after the call.
    case f1InputProjections = 1
    /// F2 (collage): precompute the checkpoint's `1 + weight` RMSNorm
    /// convention once at load time and route the ungrouped case through
    /// `MLXFast.rmsNorm`.
    case f2PrecomputedNorms = 2
    /// Reserved, never implemented — see this enum's top-level comment.
    case f3HyperConnections = 3
    /// F4 (collage): `Qwen4ExpSparseMoE`'s router uses `softmax(precise:
    /// false)` — a strictly monotonic transform, so the top-k *set* it
    /// selects does not depend on softmax precision barring a tie flip
    /// right at the boundary (verified: 0/200 mismatches on synthetic
    /// logits at the real router's dimensions).
    case f4MoE = 4
    /// Reserved, never implemented — see this enum's top-level comment.
    case f5Casts = 5
    /// Reserved, never implemented — see this enum's top-level comment.
    case f6Compile = 6
    /// F7 (**correctness fix**, not a collage fusion — P8.2, 2026-09-11;
    /// unrelated to the retired P7.4 gate/up fusion that once held this
    /// number): restores the checkpoint's intended working dtype (bf16) at
    /// two points where it was silently promoted to float32 and never
    /// rounded back down — `Qwen4ExpGatedDeltaNet`'s recurrence output
    /// before `Qwen4ExpRMSNormGated` (whose own `.asType(inputs.dtype)`
    /// rounds to *whatever it is handed*, which was the recurrence's raw
    /// fp32 output, not the network's bf16), and `Qwen4ExpQSAAttention`'s
    /// query/key after `Qwen4ExpMRoPE.apply` (whose float32 `cos`/`sin`
    /// tables promote the RoPE output to fp32 with no downcast). Both leaks
    /// make every downstream op — including the entire MoE branch — run
    /// its fp32 path, ~12-27× slower per `op-overhead-probe`'s `SwitchGLU`
    /// measurements (docs/knowledge/log.md, "P8 : ..."). `false` (`.none`
    /// through `.f6Compile`) is the original, leaking behavior.
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
    /// F8 (P11-fusion, 2026-09-13, opt-in) : enveloppe le chemin normal
    /// (hors branches d'ablation) de `Qwen4ExpGatedResidual.callAsFunction`
    /// — `hc_norm` compris, la branche groupée de `Qwen4ExpRMSNorm` étant la
    /// moitié du gisement — dans un seul `MLX.compile`. Compte d'opérations
    /// mesuré dans le code, par hyper-connexion et par appel, chemin de
    /// production (F2 déjà appliqué, donc `hc_norm` sur sa branche à poids
    /// précalculé) :
    /// - `hc_norm` (branche groupée, `Qwen4ExpRMSNorm.callAsFunction`,
    ///   cas `effectiveWeight` + `groupSize`) : 11 lancements (`asType`,
    ///   2×`reshaped`, `values*values`, `mean`, `+eps`, `rsqrt`, 2×`*`,
    ///   `reshaped`, `asType`).
    /// - mélange bas-rang + injection (le reste de `callAsFunction`) :
    ///   13 lancements (3 matmuls down/up/inject, 2 divisions scalaires,
    ///   `silu`, `sigmoid`×2, 2 `reshaped`, 2 multiplications, `mean`).
    /// Soit **24 lancements par hyper-connexion**, ×2 par couche
    /// (`attn_hyper_connection`, `mlp_hyper_connection`), ×48 couches =
    /// **2 304 lancements par jeton décodé** avant fusion, contre 2 appels
    /// à une fermeture `MLX.compile` déjà tracée par couche après (le
    /// nombre de lancements Metal *à l'intérieur* du graphe compilé dépend
    /// de ce que MLX choisit de fusionner et n'est pas compté ici — voir
    /// docs/knowledge/log.md, 2026-09-13). Ne s'applique jamais aux
    /// branches d'ablation (P7.1) ni quand `captureParity` est actif (le
    /// graphe compilé n'expose pas les tenseurs intermédiaires nommés que
    /// la capture de parité a besoin de publier) — voir
    /// `Qwen4ExpGatedResidual.prepareCompiledPath()`.
    case f8HyperConnectionsCompiled = 8
    /// F9 (P11-fusion, 2026-09-13, opt-in) : réutilise `compiledSiluProduct`
    /// (`MLXLMCommon`, déjà partagé par `SwitchGLU`, donc pas de second
    /// noyau compilé) pour la partie `silu(gate) * up` de
    /// `Qwen4ExpSharedExpert.callAsFunction` — `downProj(silu(gateProj(x))
    /// * upProj(x))` devient `downProj(compiledSiluProduct(gateProj(x),
    /// upProj(x)))`. Fusionne 2 lancements (`silu`, puis la multiplication)
    /// en 1 appel à la fermeture compilée, sans toucher aux 3 matmuls. Un
    /// expert partagé par couche, 48 couches : **48 lancements économisés
    /// par jeton décodé** (un par couche).
    case f9SharedExpertCompiled = 9

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
        // F8 : préparé après F2 ci-dessus (ordre textuel), donc la
        // fermeture compilée trace `hc_norm` sur sa branche à poids
        // précalculé, pas la branche `1 + weight` par appel — voir le
        // commentaire de `Qwen4ExpFusionLevel.f8HyperConnectionsCompiled`.
        if Qwen4ExpFusionLevel.f8HyperConnectionsCompiled.isReached(by: level) {
            attnHyperConnection.prepareCompiledPath()
            mlpHyperConnection.prepareCompiledPath()
        }
        if Qwen4ExpFusionLevel.f9SharedExpertCompiled.isReached(by: level) {
            mlp.sharedExpert.prepareCompiledActivation()
        }
    }
}
