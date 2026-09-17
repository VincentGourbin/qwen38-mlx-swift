import Foundation
import MLX

/// Target state required to roll back one or more speculative verification
/// tokens.  It couples decoder caches with the logical M-RoPE clock; restoring
/// either one without the other silently shifts every later position.
public final class Qwen4ExpStreamingTextModelSnapshot: @unchecked Sendable {
    fileprivate let decoder: Qwen4ExpStreamingDecoderSnapshot
    fileprivate let logicalOffset: Int
    /// P5.1: device bytes held by the copied per-layer caches (GDN/PLE/QSA).
    /// Does not include the resident decoder *weights* (those are shared
    /// across every conversation, not per-snapshot).
    public var byteCount: Int { decoder.byteCount }

    fileprivate init(decoder: Qwen4ExpStreamingDecoderSnapshot, logicalOffset: Int) {
        self.decoder = decoder
        self.logicalOffset = logicalOffset
    }
}

/// Memory-bounded text executor: global weights stay resident, decoder
/// layers are loaded one at a time, and recurrent/QSA caches survive turns.
public final class Qwen4ExpStreamingTextModel: @unchecked Sendable {
    public let configuration: Qwen4ExpTextConfiguration
    public let global: Qwen4ExpGlobalTextModel
    public let decoder: Qwen4ExpStreamingDecoder
    public let globalLoadReport: Qwen4ExpGlobalLoadReport
    /// P4.2: wall time of the last `forward` call's end-of-token segment —
    /// hyper-stream reduction + `lm_head` (248 320 × 2 560, quantized) +
    /// the blocking `eval` that materializes `logits`. `ContinuousClock`
    /// only (no `MLXProfiler` phase, ~4.7 ms/boundary — P0-c): cheap enough
    /// to leave on unconditionally, like `Qwen4ExpFlashMTPStepTimings`.
    public private(set) var lastLMHeadDuration: TimeInterval = 0

    private var logicalOffset = 0

    public init(
        directory: URL,
        materializeGlobal: Bool = true,
        layerLoadingMode: Qwen4ExpLayerLoadingMode = .streamed,
        residentEvaluationInterval: Int = 1,
        profileLayers: Bool = false,
        residentAsyncEval: Bool = false,
        residentAsyncInterval: Int = 1,
        uncachedIO: Bool = true,
        fusionLevel: Qwen4ExpFusionLevel = .f7GatedBranchDtype,
        /// P11.1 : surcharge de `num_experts_per_tok`, threadée jusqu'au
        /// décodeur — voir `Qwen4ExpStreamingDecoder`'s doc comment. `nil`
        /// (le défaut) laisse le comportement inchangé.
        routedExpertCount: Int? = nil,
        /// P11.2 : quel sous-bloc, le cas échéant, court-circuiter — voir
        /// `Qwen4ExpStreamingDecoder`'s doc comment. `.none` (le défaut)
        /// laisse le comportement inchangé.
        ablation: Qwen4ExpLayerBenchAblation = .none
    ) throws {
        let loadedGlobal = try Qwen4ExpGlobalCheckpointLoader.load(
            from: directory, materialize: materializeGlobal, uncachedIO: uncachedIO)
        self.global = loadedGlobal.model
        self.globalLoadReport = loadedGlobal.report
        self.decoder = try Qwen4ExpStreamingDecoder(
            directory: directory,
            layerLoadingMode: layerLoadingMode,
            residentEvaluationInterval: residentEvaluationInterval,
            profileLayers: profileLayers,
            residentAsyncEval: residentAsyncEval,
            residentAsyncInterval: residentAsyncInterval,
            uncachedIO: uncachedIO,
            fusionLevel: fusionLevel,
            routedExpertCount: routedExpertCount,
            ablation: ablation)
        self.configuration = decoder.configuration
    }

    /// P11.1 : largeur de routage MoE actuellement effective — voir
    /// `Qwen4ExpStreamingDecoder.routedExpertCount`.
    public var routedExpertCount: Int { decoder.routedExpertCount }

    /// P11.1 : change la largeur de routage MoE sans recharger le
    /// checkpoint — voir `Qwen4ExpStreamingDecoder.updateRoutedExpertCount`.
    @discardableResult
    public func updateRoutedExpertCount(_ override: Int?) throws -> Int {
        try decoder.updateRoutedExpertCount(override)
    }

    /// P11.2 : ablation actuellement effective — voir
    /// `Qwen4ExpStreamingDecoder.ablation`.
    public var ablation: Qwen4ExpLayerBenchAblation { decoder.ablation }

    /// P11.2 : change l'ablation sans recharger le checkpoint — voir
    /// `Qwen4ExpStreamingDecoder.updateAblation`.
    public func setAblation(_ new: Qwen4ExpLayerBenchAblation) {
        decoder.updateAblation(new)
    }

    public func forward(
        inputIDs: MLXArray,
        layerIndices: [Int]? = nil,
        positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil,
        imageTokenID: Int32? = nil,
        materializeLayers: Bool = true,
        // PM4.2 (P-MTP suite): pass a fresh `Qwen4ExpVerificationCapture()`
        // to make this an MTP verification forward. Every GDN/PLE layer
        // records into it the materials `rollbackVerification` needs to
        // undo a partially rejected draft block without replaying this
        // forward. `nil` (every other caller) is the original path,
        // unchanged.
        verificationCapture: Qwen4ExpVerificationCapture? = nil,
        onLayerVisited: (@Sendable (Int) -> Void)? = nil,
        /// P12.2 : décalage à gauche de chaque ligne du lot — voir
        /// `Qwen4ExpBatchPaddingLayout` et le commentaire de
        /// `Qwen4ExpStreamingDecoder.forward`'s propre paramètre du même
        /// nom, vers lequel celui-ci n'est qu'un relais direct. `nil` (le
        /// défaut) laisse `logicalOffset` gouverner comme avant : cette
        /// sonde n'a d'effet que pour un appelant qui fournit explicitement
        /// ses propres `positionIDs` par ligne, ce que ce paramètre seul ne
        /// change pas.
        leftPadding: [Int]? = nil,
        /// Coupe-circuit mémoire (2026-09-17, crash `metal::malloc` en
        /// production — 67,8 Go demandés pour un `lm_head` sur ~68 250
        /// positions de préfill contre un plafond Metal de 62,6 Go) :
        /// quand `true`, les états cachés sont tronqués à la dernière
        /// position **avant** `global.reduceHyperStreams`/`global.logits`,
        /// au lieu de calculer les logits de chaque position du préfill
        /// pour n'en garder qu'une (`logits[0..., -1, 0...]` côté
        /// appelant). `reduceHyperStreams` (RMSNorm + porte sigmoïde) est
        /// strictement position-locale, donc la dernière position de sa
        /// sortie est bit-à-bit identique, tronquée avant ou après — voir
        /// le test de parité `Qwen4ExpStreamingTextModelLastPositionOnly*`.
        /// `false` (le défaut) ne change rien : `output` reste calculé sur
        /// toutes les positions, exactement comme avant ce paramètre.
        ///
        /// `preMixerHidden` (le retour `result.output`, ci-dessous) N'EST
        /// JAMAIS tronqué par ce paramètre — seul `output` (les logits,
        /// dimensionnés par le vocabulaire, donc responsables de
        /// l'allocation qui plante) l'est. Les appelants qui ont besoin de
        /// `preMixerHidden` sur toutes les positions (brouillonnage/
        /// vérification MTP) restent donc corrects même s'ils passaient
        /// `true` par erreur — mais ne devraient jamais avoir besoin de le
        /// faire : seuls les générateurs qui ne lisent que la dernière
        /// position des logits (`Qwen4ExpStreamingGenerator`,
        /// `Qwen4ExpGreedyGenerator`, `Qwen4ExpBatchStreamingGenerator` via
        /// `batchForward`) le passent à `true`. Tout appelant qui a besoin
        /// des logits de plusieurs positions (`scoreTeacherForced`, les
        /// parités `Qwen4ExpGlobalParity`/`Qwen4ExpSelectedLayersParity`,
        /// la vérification MTP avec `verificationCapture`) DOIT garder
        /// `false`.
        lastPositionOnly: Bool = false
    ) throws -> (
        logits: MLXArray,
        preMixerHidden: MLXArray,
        reports: [Qwen4ExpStreamingLayerReport]
    ) {
        precondition(inputIDs.ndim == 2)
        if visionEmbeddings != nil {
            precondition(imageTokenID != nil)
        }
        let selectedLayers = layerIndices ?? Array(0 ..< configuration.numHiddenLayers)
        let positions = positionIDs ?? Qwen4ExpMRoPE.textPositionIDs(
            sequenceLength: inputIDs.dim(1), offset: logicalOffset)
        var embeddings = global.embed(inputIDs)
        if let visionEmbeddings, let imageTokenID {
            embeddings = try Qwen4ExpInputMerger.merge(
                inputIDs: inputIDs,
                textEmbeddings: embeddings,
                visionEmbeddings: visionEmbeddings,
                imageTokenID: imageTokenID)
        }
        let hyperInput = tiled(
            embeddings, repetitions: [1, 1, configuration.hcCount])
        let result = try decoder.forward(
            hyperInput,
            inputIDs: inputIDs,
            layerIndices: selectedLayers,
            positionIDs: positions,
            materializeLayers: materializeLayers,
            verificationCapture: verificationCapture,
            onLayerVisited: onLayerVisited,
            leftPadding: leftPadding)
        let lmHeadStart = ContinuousClock.now
        let mixerInput: MLXArray
        if lastPositionOnly {
            let seqLen = result.output.dim(1)
            mixerInput = result.output[0..., (seqLen - 1)..<seqLen, 0...]
        } else {
            mixerInput = result.output
        }
        let reduced = global.reduceHyperStreams(mixerInput)
        let output = global.logits(from: reduced)
        eval(output)
        lastLMHeadDuration = (ContinuousClock.now - lmHeadStart).seconds
        logicalOffset += inputIDs.dim(1)
        return (output, result.output, result.reports)
    }

    public func resetConversation() {
        logicalOffset = 0
        decoder.resetCaches()
    }

    public func ngramCacheStats() -> Qwen4ExpNGramCacheStats {
        decoder.ngramCacheStats()
    }

    public func resetNGramCacheStats() {
        decoder.resetNGramCacheStats()
    }

    public func ngramLookupStats() -> Qwen4ExpPLELookupStats {
        decoder.ngramLookupStats()
    }

    public func resetNGramLookupStats() {
        decoder.resetNGramLookupStats()
    }

    /// Capture the complete target continuation point.  This is the public
    /// Flash-Next equivalent of the upstream MTP cache snapshot: it includes
    /// QSA keys/indexer positions, GDN recurrent state and the M-RoPE offset.
    public func snapshot() -> Qwen4ExpStreamingTextModelSnapshot {
        Qwen4ExpStreamingTextModelSnapshot(
            decoder: decoder.snapshot(), logicalOffset: logicalOffset)
    }

    /// Restore the target to a previously captured continuation point.
    public func restore(_ snapshot: Qwen4ExpStreamingTextModelSnapshot) {
        decoder.restore(snapshot.decoder)
        logicalOffset = snapshot.logicalOffset
    }

    /// PM4.2 (P-MTP suite): undo the rejected tail of the last verification
    /// forward without replaying it. `capture` must be the object passed to
    /// that forward's `verificationCapture:`; `totalNewTokens` is the number
    /// of new positions it fed (`inputIDs.dim(1)` of that call);
    /// `committedNewTokens` (>= 1, the bonus token is always kept) is how
    /// many of those the speculative walk actually accepted. Adjusts every
    /// GDN/PLE/QSA cache (`Qwen4ExpStreamingDecoder.rollbackVerification`)
    /// and rewinds the M-RoPE clock (`logicalOffset`) by the same rejected
    /// count so the next round's positions stay correct. A no-op when
    /// nothing was rejected.
    public func rollbackVerification(
        capture: Qwen4ExpVerificationCapture,
        committedNewTokens: Int,
        totalNewTokens: Int
    ) {
        decoder.rollbackVerification(
            capture: capture,
            committedNewTokens: committedNewTokens,
            totalNewTokens: totalNewTokens)
        logicalOffset -= (totalNewTokens - committedNewTokens)
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
