import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// Sampling presets from the shared Qwen3.8 contract (§2.1). Building the
/// sampler through `MLXLMCommon.GenerateParameters` keeps Flash-Next on the
/// exact same sampler implementations as the 27B path — no custom sampler.
public enum Qwen4ExpSamplingPreset: Sendable, Equatable {
    case thinking
    case instruct
    case custom(temperature: Float, topP: Float, topK: Int)

    public var temperature: Float {
        switch self {
        case .thinking: return 1.0
        case .instruct: return 0.7
        case .custom(let temperature, _, _): return temperature
        }
    }

    public var topP: Float {
        switch self {
        case .thinking: return 0.95
        case .instruct: return 0.80
        case .custom(_, let topP, _): return topP
        }
    }

    public var topK: Int {
        switch self {
        case .thinking, .instruct: return 20
        case .custom(_, _, let topK): return topK
        }
    }

    public func sampler(seed: UInt64? = nil) -> LogitSampler {
        GenerateParameters(temperature: temperature, topP: topP, topK: topK, seed: seed).sampler()
    }
}

public struct Qwen4ExpStreamingGenerationOptions: Sendable, Equatable {
    public var maxNewTokens: Int
    public var stopTokenIDs: Set<Int32>
    public var preset: Qwen4ExpSamplingPreset
    /// H2.3: when true, the decoder's recurrent/QSA caches and the logical
    /// M-RoPE offset are kept from the previous call instead of being reset
    /// — this is what makes a second turn cheap (`promptTokenIDs` then only
    /// needs to carry the new suffix).
    public var continueConversation: Bool
    public var seed: UInt64?
    /// P5.3: OpenAI-style presence penalty (flat, applied to every id already
    /// generated this turn, not scaled by how many times it recurred — see
    /// `Qwen4ExpLogitPenalizer`). 0 = no-op. Only ever consulted when
    /// `preset.temperature > 0` — greedy decoding (temperature 0) is
    /// byte-for-byte unchanged regardless of this value (PLAN.md P5.3
    /// contract: "greedy inchangé").
    public var presencePenalty: Float
    /// P5.3: multiplicative repetition penalty, same greedy-only exemption
    /// as `presencePenalty`. 1.0 = no-op.
    public var repetitionPenalty: Float

    public init(
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32> = [],
        preset: Qwen4ExpSamplingPreset = .instruct,
        continueConversation: Bool = false,
        seed: UInt64? = nil,
        presencePenalty: Float = 0,
        repetitionPenalty: Float = 1.0
    ) {
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.preset = preset
        self.continueConversation = continueConversation
        self.seed = seed
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
    }
}

public enum Qwen4ExpStreamingGenerationError: LocalizedError, Equatable {
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

public enum Qwen4ExpGenerationEvent: Sendable {
    case token(Int32)
    case finished(Qwen4ExpGenerationSummary)
}

public struct Qwen4ExpGenerationSummary: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let promptTokenCount: Int
    public let timeToFirstToken: TimeInterval?
    public let prefillTime: TimeInterval
    public let decodeTime: TimeInterval
    public let layerVisitCount: Int
    public let layerLoadTime: TimeInterval
    public let activeMemoryBytes: Int
    public let peakMemoryBytes: Int
    public let ngramCacheStats: Qwen4ExpNGramCacheStats
    /// P4.2: cumulative end-of-token cost over every decode step (excludes
    /// the prefill's own lm_head/sample) — `ContinuousClock`, always on.
    public let lmHeadTime: TimeInterval
    public let samplerSampleTime: TimeInterval
    public let itemTime: TimeInterval

    public init(
        tokenIDs: [Int32], promptTokenCount: Int, timeToFirstToken: TimeInterval?,
        prefillTime: TimeInterval, decodeTime: TimeInterval, layerVisitCount: Int,
        layerLoadTime: TimeInterval, activeMemoryBytes: Int, peakMemoryBytes: Int,
        ngramCacheStats: Qwen4ExpNGramCacheStats,
        lmHeadTime: TimeInterval = 0, samplerSampleTime: TimeInterval = 0,
        itemTime: TimeInterval = 0
    ) {
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.timeToFirstToken = timeToFirstToken
        self.prefillTime = prefillTime
        self.decodeTime = decodeTime
        self.layerVisitCount = layerVisitCount
        self.layerLoadTime = layerLoadTime
        self.activeMemoryBytes = activeMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.ngramCacheStats = ngramCacheStats
        self.lmHeadTime = lmHeadTime
        self.samplerSampleTime = samplerSampleTime
        self.itemTime = itemTime
    }
}

/// P5.3: pure, GPU-side presence/repetition penalty over a `[1, vocab]`
/// logits row. Kept separate from `Qwen4ExpStreamingGenerator.run` so a test
/// can exercise it on synthetic logits without a model. No `.item()` call —
/// every op below stays a lazy MLX array op, and the whole thing is skipped
/// entirely by the caller whenever both penalties are no-ops (greedy
/// decoding, or a sampling request that didn't ask for either).
public enum Qwen4ExpLogitPenalizer {
    /// `seenMask` is a `[vocab]` float array where a strictly positive entry
    /// means that vocabulary id has already been generated this turn
    /// (`>0`, not the raw count — presence, not frequency: PLAN.md P5.3
    /// explicitly folds `frequency_penalty` into the same flat presence
    /// treatment). `presence` is subtracted from every already-seen id's
    /// logit; `repetition` divides positive logits and multiplies negative
    /// ones at those same ids (the standard HF repetition-penalty formula).
    public static func apply(
        logits: MLXArray, seenMask: MLXArray, presence: Float, repetition: Float
    ) -> MLXArray {
        guard presence != 0 || repetition != 1.0 else { return logits }
        let seen = seenMask .> 0
        var result = logits
        if repetition != 1.0 {
            let rescaled = MLX.where(result .> 0, result / repetition, result * repetition)
            result = MLX.where(seen, rescaled, result)
        }
        if presence != 0 {
            result = MLX.where(seen, result - presence, result)
        }
        return result
    }

    /// Scatter-marks `token` as seen in `seenMask` (`[vocab]`, in place via
    /// reassignment — MLX arrays are copy-on-write value handles, so this is
    /// the idiomatic "update" for a lazily-evaluated array, matching
    /// `Qwen4ExpPLE`'s `result.at[...].add(...)` scatter pattern).
    public static func markSeen(_ seenMask: MLXArray, token: Int32) -> MLXArray {
        seenMask.at[MLXArray([token])].add(MLXArray(Float(1)))
    }
}

/// Streamed, sampling-capable autoregressive loop over the Flash-Next
/// executor (H2). Unlike `Qwen4ExpGreedyGenerator` — kept untouched as the
/// greedy oracle for tests — this type samples through
/// `MLXLMCommon.LogitSampler`, emits tokens as they are produced, and
/// supports multi-turn continuation without replaying the prompt.
public final class Qwen4ExpStreamingGenerator: @unchecked Sendable {
    public let model: Qwen4ExpStreamingTextModel

    public init(model: Qwen4ExpStreamingTextModel) {
        self.model = model
    }

    /// MLXArray is a class-backed, non-Sendable handle. Boxing the prompt
    /// arguments lets the generation loop run on a detached `Task` (needed
    /// for streaming + cancellation) without the compiler flagging a data
    /// race — the box is only ever touched by that one task.
    private struct Input: @unchecked Sendable {
        let promptTokenIDs: [Int32]
        let positionIDs: MLXArray?
        let visionEmbeddings: MLXArray?
        let imageTokenID: Int32?
    }

    public func generate(
        promptTokenIDs: [Int32],
        positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil,
        imageTokenID: Int32? = nil,
        options: Qwen4ExpStreamingGenerationOptions = .init(),
        profiler: MLXProfiler = .shared
    ) -> AsyncThrowingStream<Qwen4ExpGenerationEvent, Error> {
        let input = Input(
            promptTokenIDs: promptTokenIDs, positionIDs: positionIDs,
            visionEmbeddings: visionEmbeddings, imageTokenID: imageTokenID)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.run(
                        promptTokenIDs: input.promptTokenIDs,
                        positionIDs: input.positionIDs,
                        visionEmbeddings: input.visionEmbeddings,
                        imageTokenID: input.imageTokenID,
                        options: options,
                        profiler: profiler,
                        continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        promptTokenIDs: [Int32],
        positionIDs: MLXArray?,
        visionEmbeddings: MLXArray?,
        imageTokenID: Int32?,
        options: Qwen4ExpStreamingGenerationOptions,
        profiler: MLXProfiler,
        continuation: AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>.Continuation
    ) throws {
        guard !promptTokenIDs.isEmpty else {
            throw Qwen4ExpStreamingGenerationError.emptyPrompt
        }
        guard options.maxNewTokens > 0 else {
            throw Qwen4ExpStreamingGenerationError.invalidMaxNewTokens
        }
        guard (visionEmbeddings == nil) == (imageTokenID == nil) else {
            throw Qwen4ExpStreamingGenerationError.invalidImageArguments
        }

        if !options.continueConversation {
            model.resetConversation()
            model.resetNGramCacheStats()
        }
        let sampler = options.preset.sampler(seed: options.seed)
        let prompt = MLXArray(promptTokenIDs).reshaped([1, promptTokenIDs.count])
        let started = Date()
        profiler.startPrefill()
        let prefill = try model.forward(
            inputIDs: prompt,
            positionIDs: positionIDs,
            visionEmbeddings: visionEmbeddings,
            imageTokenID: imageTokenID)
        eval(prefill.logits)
        recordNGramCacheStats(profiler)
        var layerVisitCount = prefill.reports.count
        var layerLoadTime = prefill.reports.reduce(0) { $0 + $1.loadDuration }
        let prefillEnd = Date()
        profiler.endPrefill()

        var tokenIDs = [Int32]()
        tokenIDs.reserveCapacity(options.maxNewTokens)
        var logits = prefill.logits[0..., -1, 0...]
        var firstTokenTime: TimeInterval?
        var generationStarted = false
        // P4.2: end-of-token cost, ContinuousClock only (always on, no
        // MLXProfiler phase boundary — see `lastLMHeadDuration`).
        var lmHeadTimeTotal: TimeInterval = 0
        var samplerSampleTimeTotal: TimeInterval = 0
        var itemTimeTotal: TimeInterval = 0
        // P5.3: presence/repetition penalties are strictly opt-in and
        // sampling-only — greedy decoding (`preset.temperature == 0`) never
        // even allocates the mask, let alone touches `logits`, so it stays
        // byte-for-byte identical to pre-P5.3 behavior.
        let penaltiesActive =
            options.preset.temperature > 0
            && (options.presencePenalty != 0 || options.repetitionPenalty != 1.0)
        var seenMask = penaltiesActive ? MLXArray.zeros([logits.dim(-1)]) : nil

        while tokenIDs.count < options.maxNewTokens {
            if Task.isCancelled { break }
            let sampledLogits: MLXArray
            if let seenMask {
                sampledLogits = Qwen4ExpLogitPenalizer.apply(
                    logits: logits, seenMask: seenMask, presence: options.presencePenalty,
                    repetition: options.repetitionPenalty)
            } else {
                sampledLogits = logits
            }
            // Materialize the token eagerly (piège 11) — no deferred graph
            // survives across autoregressive steps.
            let sampleStart = ContinuousClock.now
            let sampled = sampler.sample(logits: sampledLogits)
            samplerSampleTimeTotal += (ContinuousClock.now - sampleStart).seconds
            let itemStart = ContinuousClock.now
            let token = Int32(sampled.item(Int32.self))
            itemTimeTotal += (ContinuousClock.now - itemStart).seconds
            if let mask = seenMask {
                seenMask = Qwen4ExpLogitPenalizer.markSeen(mask, token: token)
            }
            if !generationStarted {
                generationStarted = true
                firstTokenTime = Date().timeIntervalSince(started)
                profiler.startGeneration()
            }
            tokenIDs.append(token)
            continuation.yield(.token(token))
            if options.stopTokenIDs.contains(token) {
                break
            }
            if tokenIDs.count >= options.maxNewTokens {
                break
            }
            if Task.isCancelled { break }

            let step = try model.forward(inputIDs: MLXArray([token]).reshaped([1, 1]))
            eval(step.logits)
            lmHeadTimeTotal += model.lastLMHeadDuration
            recordNGramCacheStats(profiler)
            logits = step.logits[0..., -1, 0...]
            layerVisitCount += step.reports.count
            layerLoadTime += step.reports.reduce(0) { $0 + $1.loadDuration }
        }

        if generationStarted {
            profiler.endGeneration(tokenCount: tokenIDs.count)
        }
        let generationEnd = Date()
        let summary = Qwen4ExpGenerationSummary(
            tokenIDs: tokenIDs,
            promptTokenCount: promptTokenIDs.count,
            timeToFirstToken: firstTokenTime,
            prefillTime: prefillEnd.timeIntervalSince(started),
            decodeTime: generationStarted ? generationEnd.timeIntervalSince(prefillEnd) : 0,
            layerVisitCount: layerVisitCount,
            layerLoadTime: layerLoadTime,
            activeMemoryBytes: Memory.activeMemory,
            peakMemoryBytes: Memory.peakMemory,
            ngramCacheStats: model.ngramCacheStats(),
            lmHeadTime: lmHeadTimeTotal,
            samplerSampleTime: samplerSampleTimeTotal,
            itemTime: itemTimeTotal)
        continuation.yield(.finished(summary))
    }
}

extension Qwen4ExpStreamingGenerator {
    /// Publishes cumulative row-cache counters into the active profiler
    /// session (H4.4: the GUI's exported trace must show the "Flash n-gram
    /// cache" counter alongside the per-layer phases the decoder already
    /// emits). Mirrors `Qwen4ExpGreedyGenerator.recordNGramCacheStats`.
    fileprivate func recordNGramCacheStats(_ profiler: MLXProfiler) {
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

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
