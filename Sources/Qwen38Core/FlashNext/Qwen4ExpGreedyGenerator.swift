import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// A deliberately small generation contract for the first Flash-Next runtime.
///
/// The model is still loaded layer-by-layer, so this type owns only the
/// autoregressive policy. It accepts already-rendered token IDs; tokenizer and
/// multimodal template concerns stay at the pipeline boundary. This makes the
/// first real generation probe useful without pretending that the public
/// ChatSession path can load a qwen4_exp model yet.
public struct Qwen4ExpGreedyGenerationOptions: Sendable, Equatable {
    public var maxNewTokens: Int
    public var stopTokenIDs: Set<Int32>

    public init(maxNewTokens: Int = 32, stopTokenIDs: Set<Int32> = []) {
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
    }
}

extension Qwen4ExpGreedyGenerator {
    /// Publishes cumulative row-cache counters into the active profiler
    /// session without retaining any MLX tensor or computation graph.
    func recordNGramCacheStats(_ profiler: MLXProfiler) {
        guard profiler.isEnabled, let session = profiler.activeSession else { return }
        let stats = model.ngramCacheStats()
        session.addCounterEvent(
            name: "Flash n-gram cache",
            timestampUs: session.currentTimestampUsPublic(),
            values: [
                "hits": Double(stats.hits),
                "misses": Double(stats.misses),
                "entries": Double(stats.entries),
                "hit_rate": stats.hitRate ?? 0,
            ])
    }
}

public struct Qwen4ExpLogitCandidate: Sendable, Equatable {
    public let tokenID: Int32
    public let logit: Float

    public init(tokenID: Int32, logit: Float) {
        self.tokenID = tokenID
        self.logit = logit
    }
}

public struct Qwen4ExpGreedyGenerationResult: Sendable {
    public let tokenIDs: [Int32]
    public let promptTokenCount: Int
    public let prefillTime: TimeInterval
    public let generationTime: TimeInterval
    public let timeToFirstToken: TimeInterval?
    public let layerReports: [[Qwen4ExpStreamingLayerReport]]
    public let firstTokenCandidates: [Qwen4ExpLogitCandidate]

    /// Cumulative time spent loading checkpoint tensors across all forwards.
    /// In streamed mode this is the cost paid again for every autoregressive
    /// token; in resident mode it should be close to the first visit only.
    public var layerLoadTime: TimeInterval {
        layerReports.flatMap { $0 }.reduce(0) { $0 + $1.loadDuration }
    }

    /// Cumulative time spent executing loaded decoder layers, excluding the
    /// checkpoint load itself.  Keeping this separate prevents a streamed
    /// checkpoint from being mistaken for a slow MLX kernel.
    public var layerForwardTime: TimeInterval {
        layerReports.flatMap { $0 }.reduce(0) { $0 + $1.forwardDuration }
    }

    /// Number of layer visits recorded by the streaming executor.
    public var layerVisitCount: Int {
        layerReports.reduce(0) { $0 + $1.count }
    }

    public init(
        tokenIDs: [Int32], promptTokenCount: Int, prefillTime: TimeInterval,
        generationTime: TimeInterval, timeToFirstToken: TimeInterval?,
        layerReports: [[Qwen4ExpStreamingLayerReport]],
        firstTokenCandidates: [Qwen4ExpLogitCandidate] = []
    ) {
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.prefillTime = prefillTime
        self.generationTime = generationTime
        self.timeToFirstToken = timeToFirstToken
        self.layerReports = layerReports
        self.firstTokenCandidates = firstTokenCandidates
    }
}

public enum Qwen4ExpGreedyGenerationError: LocalizedError, Equatable {
    case emptyPrompt
    case invalidMaxNewTokens
    case invalidImageArguments

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt: return "Le prompt Flash-Next ne peut pas être vide."
        case .invalidMaxNewTokens: return "maxNewTokens doit être positif."
        case .invalidImageArguments:
            return "Les embeddings image et l'image token doivent être fournis ensemble."
        }
    }
}

/// Greedy autoregressive loop over the memory-bounded Flash-Next executor.
///
/// The first call is a prefill and may contain image embeddings. Every next
/// call contains exactly one token, so the recurrent/QSA caches and the logical
/// M-RoPE offset stay alive inside `Qwen4ExpStreamingTextModel`.
public final class Qwen4ExpGreedyGenerator: @unchecked Sendable {
    public let model: Qwen4ExpStreamingTextModel

    public init(model: Qwen4ExpStreamingTextModel) {
        self.model = model
    }

    public func generate(
        promptTokenIDs: [Int32],
        positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil,
        imageTokenID: Int32? = nil,
        options: Qwen4ExpGreedyGenerationOptions = .init(),
        firstTokenTopK: Int = 0,
        profiler: MLXProfiler = .shared
    ) throws -> Qwen4ExpGreedyGenerationResult {
        guard !promptTokenIDs.isEmpty else {
            throw Qwen4ExpGreedyGenerationError.emptyPrompt
        }
        guard options.maxNewTokens > 0 else {
            throw Qwen4ExpGreedyGenerationError.invalidMaxNewTokens
        }
        guard (visionEmbeddings == nil) == (imageTokenID == nil) else {
            throw Qwen4ExpGreedyGenerationError.invalidImageArguments
        }

        model.resetConversation()
        model.resetNGramCacheStats()
        let prompt = MLXArray(promptTokenIDs).reshaped([1, promptTokenIDs.count])
        let started = Date()
        profiler.startPrefill()
        let prefill = try model.forward(
            inputIDs: prompt,
            positionIDs: positionIDs,
            visionEmbeddings: visionEmbeddings,
            imageTokenID: imageTokenID,
            // `firstTokenTopK` (ci-dessous) ne lit lui aussi que la
            // dernière position — voir le commentaire de
            // `forward(lastPositionOnly:)`.
            lastPositionOnly: true)
        eval(prefill.logits)
        recordNGramCacheStats(profiler)
        let firstTokenCandidates: [Qwen4ExpLogitCandidate]
        if firstTokenTopK > 0 {
            let firstLogits = prefill.logits[0..., -1, 0...]
                .asType(.float32).asArray(Float.self)
            let top = firstLogits.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(firstTokenTopK)
            firstTokenCandidates = top.map {
                Qwen4ExpLogitCandidate(tokenID: Int32($0.offset), logit: $0.element)
            }
        } else {
            firstTokenCandidates = []
        }
        let prefillEnd = Date()
        profiler.endPrefill()

        var tokenIDs = [Int32]()
        tokenIDs.reserveCapacity(options.maxNewTokens)
        var reports = [prefill.reports]
        var logits = prefill.logits[0..., -1, 0...]
        var firstTokenTime: TimeInterval?
        var generationStarted = false

        while tokenIDs.count < options.maxNewTokens {
            let token = Int32(ArgMaxSampler().sample(logits: logits).item(Int32.self))
            if !generationStarted {
                generationStarted = true
                firstTokenTime = Date().timeIntervalSince(started)
                profiler.startGeneration()
            }
            tokenIDs.append(token)
            if options.stopTokenIDs.contains(token) {
                break
            }
            if tokenIDs.count >= options.maxNewTokens {
                break
            }

            let step = try model.forward(
                inputIDs: MLXArray([token]).reshaped([1, 1]), lastPositionOnly: true)
            eval(step.logits)
            recordNGramCacheStats(profiler)
            logits = step.logits[0..., -1, 0...]
            reports.append(step.reports)
        }

        if generationStarted {
            profiler.endGeneration(tokenCount: tokenIDs.count)
        }
        let generationEnd = Date()
        return Qwen4ExpGreedyGenerationResult(
            tokenIDs: tokenIDs,
            promptTokenCount: promptTokenIDs.count,
            prefillTime: prefillEnd.timeIntervalSince(started),
            generationTime: generationStarted
                ? generationEnd.timeIntervalSince(prefillEnd)
                : 0,
            timeToFirstToken: firstTokenTime,
            layerReports: reports,
            firstTokenCandidates: firstTokenCandidates)
    }
}
