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
        fusionLevel: Qwen4ExpFusionLevel = .none
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
            fusionLevel: fusionLevel)
        self.configuration = decoder.configuration
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
        onLayerVisited: (@Sendable (Int) -> Void)? = nil
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
            onLayerVisited: onLayerVisited)
        let lmHeadStart = ContinuousClock.now
        let reduced = global.reduceHyperStreams(result.output)
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
