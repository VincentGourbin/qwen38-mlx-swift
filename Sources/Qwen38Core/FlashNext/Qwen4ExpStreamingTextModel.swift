import Foundation
import MLX

/// Target state required to roll back one or more speculative verification
/// tokens.  It couples decoder caches with the logical M-RoPE clock; restoring
/// either one without the other silently shifts every later position.
public final class Qwen4ExpStreamingTextModelSnapshot: @unchecked Sendable {
    fileprivate let decoder: Qwen4ExpStreamingDecoderSnapshot
    fileprivate let logicalOffset: Int

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

    private var logicalOffset = 0

    public init(
        directory: URL,
        materializeGlobal: Bool = true,
        layerLoadingMode: Qwen4ExpLayerLoadingMode = .streamed,
        residentEvaluationInterval: Int = 1,
        profileLayers: Bool = false,
        residentAsyncEval: Bool = false,
        uncachedIO: Bool = true
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
            uncachedIO: uncachedIO)
        self.configuration = decoder.configuration
    }

    public func forward(
        inputIDs: MLXArray,
        layerIndices: [Int]? = nil,
        positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil,
        imageTokenID: Int32? = nil,
        materializeLayers: Bool = true,
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
            onLayerVisited: onLayerVisited)
        let reduced = global.reduceHyperStreams(result.output)
        let output = global.logits(from: reduced)
        eval(output)
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
}
