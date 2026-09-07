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

    public init(
        maxNewTokens: Int = 256,
        stopTokenIDs: Set<Int32> = [],
        preset: Qwen4ExpSamplingPreset = .instruct,
        continueConversation: Bool = false,
        seed: UInt64? = nil
    ) {
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.preset = preset
        self.continueConversation = continueConversation
        self.seed = seed
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

    public init(
        tokenIDs: [Int32], promptTokenCount: Int, timeToFirstToken: TimeInterval?,
        prefillTime: TimeInterval, decodeTime: TimeInterval, layerVisitCount: Int,
        layerLoadTime: TimeInterval, activeMemoryBytes: Int, peakMemoryBytes: Int,
        ngramCacheStats: Qwen4ExpNGramCacheStats
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
        var layerVisitCount = prefill.reports.count
        var layerLoadTime = prefill.reports.reduce(0) { $0 + $1.loadDuration }
        let prefillEnd = Date()
        profiler.endPrefill()

        var tokenIDs = [Int32]()
        tokenIDs.reserveCapacity(options.maxNewTokens)
        var logits = prefill.logits[0..., -1, 0...]
        var firstTokenTime: TimeInterval?
        var generationStarted = false

        while tokenIDs.count < options.maxNewTokens {
            if Task.isCancelled { break }
            // Materialize the token eagerly (piège 11) — no deferred graph
            // survives across autoregressive steps.
            let token = Int32(sampler.sample(logits: logits).item(Int32.self))
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
            ngramCacheStats: model.ngramCacheStats())
        continuation.yield(.finished(summary))
    }
}
