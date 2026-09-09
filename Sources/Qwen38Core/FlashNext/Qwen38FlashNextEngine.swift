import Foundation
import MLX
import MLXLMCommon
import MLXProfiler
import Tokenizers

public enum Qwen38FlashNextEngineError: LocalizedError, Equatable {
    case multipleImagesUnsupported
    case imageOnContinuationUnsupported
    case statelessImagesUnsupported

    public var errorDescription: String? {
        switch self {
        case .multipleImagesUnsupported:
            return "Flash-Next n'accepte qu'une seule image par tour."
        case .imageOnContinuationUnsupported:
            return "Flash-Next n'accepte une image qu'au premier tour d'une conversation."
        case .statelessImagesUnsupported:
            return
                "Flash-Next (LAN) : une image n'est acceptée que sur le dernier message utilisateur, sans tour assistant précédent."
        }
    }
}

/// Runtime-facing surface for the Flash-Next engine. A protocol — rather
/// than the concrete `Qwen38FlashNextEngine` — so `Qwen38Runtime`'s family
/// dispatch (H3.1) can be exercised in tests through a lightweight mock,
/// without loading the real ~80 GB resident checkpoint.
public protocol Qwen38FlashNextEngineProtocol: AnyObject, Sendable {
    var directory: URL { get }
    func resetConversation()
    func unload()
    func decode(tokenIDs: [Int32]) -> String
    func generate(
        prompt: String, systemPrompt: String?, imageURLs: [URL], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error>
    func generateFromMessages(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error>
    /// H4.2: forces every decoder layer to be loaded from disk once, up
    /// front, instead of paying that cost inside the first real turn's
    /// TTFT. Yields the index of each layer as it finishes loading (0-based,
    /// `numHiddenLayers` values total) so a caller can drive a progress bar.
    /// A best-effort warm-up: failures surface later, on the first real
    /// `generate` call, rather than here.
    func warmUp() -> AsyncStream<Int>
}

public protocol Qwen38FlashNextEngineFactory: Sendable {
    func makeEngine(directory: URL) async throws -> any Qwen38FlashNextEngineProtocol
}

public struct Qwen38DefaultFlashNextEngineFactory: Qwen38FlashNextEngineFactory {
    public init() {}

    public func makeEngine(directory: URL) async throws -> any Qwen38FlashNextEngineProtocol {
        try await Qwen38FlashNextEngine(directory: directory)
    }
}

/// Wraps the Flash-Next (`qwen4_exp`) streaming pieces — resident text
/// model, tokenizer, sampling — behind the same `Qwen38GenerationEvent`
/// shape as the 27B `ChatSession` path (H3.2), so `Qwen38Runtime` can
/// dispatch on model family without either path knowing about the other.
public final class Qwen38FlashNextEngine: Qwen38FlashNextEngineProtocol, @unchecked Sendable {
    public let directory: URL
    private let configuration: Qwen4ExpConfiguration
    private let tokenizer: any Tokenizers.Tokenizer
    private let model: Qwen4ExpStreamingTextModel
    private let generator: Qwen4ExpStreamingGenerator
    private let stopTokenIDs: Set<Int32>
    private let visibleTokenFilter: Qwen38VisibleTokenFilter
    private var hasConversationHistory = false
    private var turnIndex = 0

    /// Keeps macOS from idle-sleeping while a Flash-Next model is resident:
    /// P1 (2026-09-08) showed the Mac entering 'Idle Sleep' 73 s into a
    /// resident load (1-minute idle sleep in the power profile), which
    /// froze every run of the previous three days. Released in `deinit`.
    private let sleepActivity: NSObjectProtocol

    /// `residentAsyncEval` defaults to `true` since P1 (2026-09-08): on the
    /// real checkpoint, `asyncEval` per layer decoded 6 tokens in 2.33 s
    /// against 2.98 s with a blocking `eval` per layer (-22 %) and 10.25 s
    /// with a single deferred `eval` per token. Same peak memory (75.2 GB).
    public init(
        directory: URL, profileLayers: Bool = false, residentAsyncEval: Bool = true,
        uncachedIO: Bool = true
    ) async throws {
        self.sleepActivity = ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled, .userInitiated],
            reason: "Qwen3.8 Flash-Next resident model")
        self.directory = directory
        self.configuration = try Qwen4ExpConfiguration.load(from: directory)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        // Global weights resident, decoder layers resident too (H3.1):
        // Flash-Next is the sole resident model per process (contrat
        // §5.1.1) so paying the ~80 GB peak here is the intended trade-off
        // against per-token streaming reload cost.
        self.model = try Qwen4ExpStreamingTextModel(
            directory: directory, layerLoadingMode: .resident, residentEvaluationInterval: 1,
            profileLayers: profileLayers, residentAsyncEval: residentAsyncEval,
            uncachedIO: uncachedIO)
        self.generator = Qwen4ExpStreamingGenerator(model: model)
        self.stopTokenIDs = [
            configuration.textConfiguration.eosTokenID, Int32(248044), Int32(248046),
        ].compactMap { $0 }.reduce(into: Set<Int32>()) { $0.insert($1) }
        self.visibleTokenFilter = Qwen38VisibleTokenFilter(
            convertTokenToId: tokenizer.convertTokenToId)
    }

    deinit {
        ProcessInfo.processInfo.endActivity(sleepActivity)
    }

    public func resetConversation() {
        model.resetConversation()
        hasConversationHistory = false
        turnIndex = 0
    }

    public func unload() {
        model.decoder.unloadResidentLayers()
    }

    public func decode(tokenIDs: [Int32]) -> String {
        tokenizer.decode(tokens: tokenIDs.map(Int.init), skipSpecialTokens: false)
    }

    public func warmUp() -> AsyncStream<Int> {
        AsyncStream { continuation in
            let task = Task {
                let dummyToken = Int32(tokenizer.convertTokenToId("<|im_start|>") ?? 0)
                // Any generate() call right after this resets caches and the
                // logical M-RoPE offset unconditionally (hasConversationHistory
                // is still false), so this dummy forward's own state does not
                // leak into the first real turn — only the now-resident layer
                // weights do.
                _ = try? model.forward(
                    inputIDs: MLXArray([dummyToken]).reshaped([1, 1]),
                    onLayerVisited: { layerIndex in continuation.yield(layerIndex + 1) })
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func generate(
        prompt: String, systemPrompt: String?, imageURLs: [URL], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        guard imageURLs.count <= 1 else {
            throw Qwen38FlashNextEngineError.multipleImagesUnsupported
        }
        if hasConversationHistory && !imageURLs.isEmpty {
            throw Qwen38FlashNextEngineError.imageOnContinuationUnsupported
        }

        let built: Qwen4ExpBuiltPrompt
        let continueConversation = hasConversationHistory
        if continueConversation {
            built = Qwen4ExpPromptBuilder.buildContinuationTurn(
                tokenizer: tokenizer, prompt: prompt, thinking: options.enableThinking)
        } else {
            built = try Qwen4ExpPromptBuilder.buildFirstTurn(
                tokenizer: tokenizer, configuration: configuration, directory: directory,
                prompt: prompt, imageURL: imageURLs.first, thinking: options.enableThinking,
                reasoningEffort: options.reasoningEffort, systemPrompt: systemPrompt)
        }
        hasConversationHistory = true

        let inputDescription = imageURLs.isEmpty ? "Texte" : "Texte + image"
        return try runGenerationStream(
            built: built, options: options, continueConversation: continueConversation,
            inputDescription: inputDescription)
    }

    /// Stateless server path (H3.2 / Jalon 2): the whole message history is
    /// rendered as a single turn through the chat template — there is no
    /// per-client persistent cache to continue (contrat §5.1.1, "Stateless
    /// v1"). Images are rejected explicitly rather than silently dropped;
    /// the manual ChatML assembly Flash-Next uses for images only knows how
    /// to attach one image to the first rendered turn (see `generate`).
    public func generateFromMessages(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        // H6.3 (2026-09-08): a single image on the last user message, with
        // at most a system message before it, is exactly the first-turn case
        // `generate` already handles — route it there instead of failing the
        // whole LAN request. An image buried in an earlier turn of a replayed
        // history still has no manual ChatML rendering and stays rejected.
        if let last = messages.last, last.role == .user, !last.imageURLs.isEmpty,
           messages.dropLast().allSatisfy({ $0.role == .system && $0.imageURLs.isEmpty }) {
            guard last.imageURLs.count == 1 else {
                throw Qwen38FlashNextEngineError.multipleImagesUnsupported
            }
            resetConversation()
            let systemPrompt = messages.dropLast().map(\.content).joined(separator: "\n")
            return try generate(
                prompt: last.content, systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt,
                imageURLs: last.imageURLs, options: options)
        }
        guard messages.allSatisfy({ $0.imageURLs.isEmpty }) else {
            throw Qwen38FlashNextEngineError.statelessImagesUnsupported
        }
        resetConversation()
        let hfMessages: [Tokenizers.Message] = messages.map {
            ["role": $0.role.rawValue, "content": $0.content]
        }
        let built = try Qwen4ExpPromptBuilder.buildFromMessages(
            tokenizer: tokenizer, messages: hfMessages, thinking: options.enableThinking,
            reasoningEffort: options.reasoningEffort)
        hasConversationHistory = true
        return try runGenerationStream(
            built: built, options: options, continueConversation: false,
            inputDescription: "Texte")
    }

    private func runGenerationStream(
        built: Qwen4ExpBuiltPrompt, options: Qwen38GenerationOptions,
        continueConversation: Bool, inputDescription: String
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        turnIndex += 1
        let currentTurnIndex = turnIndex
        let maxNewTokens = max(options.maxTokens, 1)
        let preset = Qwen4ExpSamplingPreset.custom(
            temperature: options.temperature, topP: options.topP, topK: options.topK)
        if options.mtp.enabled {
            // P-MTP (2026-09-09, PM3) → P-MTP suite (2026-09-09, PM4):
            // measured, still not wired in, but for a different and much
            // smaller reason than PM3's. The Vontra-norm fix (loader +
            // pre_fc_norm_*) raised block-2 acceptance from 24% to 47.6%,
            // and PM4.1/PM4.2 removed the verify round's
            // `model.snapshot()`/`restore()` + replay forward entirely
            // (per-token GDN states + a capture-based cache rollback instead
            // — `stats.replayedTokens` is now always 0). Net effect on the
            // 3-bit checkpoint: block 2 went from ~1.19x SLOWER than greedy
            // (PM3) to ~0.81-0.86x — genuinely *faster* than greedy at 32
            // and 128 tokens, bit-identical token ids — but that stays just
            // short of PLAN.md's ≤0.8x bar for auto-branching (PM4.3, table
            // in docs/knowledge/log.md 2026-09-09 "P-MTP (suite) : PM4"),
            // margin ~2-8% depending on run/context length. Blocks 3 and 4
            // regress further (accept rate falls with block size). Log and
            // ignore rather than fail the request (H3.2: "options.mtp
            // ignoré, un log, pas d'erreur").
            FileHandle.standardError.write(
                Data(
                    "qwen38: Flash-Next ignore options.mtp (P-MTP : sans rejeu mais encore ~0,8-0,86x greedy, sous le seuil ≤0,8x — voir log.md 2026-09-09 PM4)\n"
                        .utf8))
        }

        let profiler = MLXProfiler.shared
        let profileSession = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
        profileSession.title = "QWEN3.8 FLASH-NEXT INFERENCE"
        profileSession.metadata["model"] = directory.lastPathComponent
        profileSession.metadata["turn"] = String(currentTurnIndex)
        profiler.activeSession = profileSession
        profiler.enable()

        let inner = generator.generate(
            promptTokenIDs: built.tokenIDs, positionIDs: built.positionIDs,
            visionEmbeddings: built.visionEmbeddings, imageTokenID: built.imageTokenID,
            options: .init(
                maxNewTokens: maxNewTokens, stopTokenIDs: stopTokenIDs, preset: preset,
                continueConversation: continueConversation),
            profiler: profiler)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in inner {
                        if Task.isCancelled { break }
                        switch event {
                        case .token(let token):
                            guard self.visibleTokenFilter.shouldEmit(Int(token)) else { continue }
                            let piece = Qwen38VisibleText.sanitize(
                                self.tokenizer.decode(
                                    tokens: [Int(token)], skipSpecialTokens: false))
                            if !piece.isEmpty {
                                continuation.yield(.chunk(piece))
                            }
                        case .finished(let summary):
                            let stopReason: GenerateStopReason
                            if let last = summary.tokenIDs.last, self.stopTokenIDs.contains(last) {
                                stopReason = .stop
                            } else if summary.tokenIDs.count >= maxNewTokens {
                                stopReason = .length
                            } else {
                                stopReason = .cancelled
                            }
                            let llmMetrics = LLMMetrics(
                                prefillTime: summary.prefillTime,
                                generationTime: summary.decodeTime,
                                promptTokens: summary.promptTokenCount,
                                generatedTokens: summary.tokenIDs.count)
                            continuation.yield(
                                .metrics(
                                    Qwen38RunMetrics(
                                        metrics: llmMetrics,
                                        stopReason: stopReason,
                                        report: profileSession.generateReport(),
                                        chromeTrace: ChromeTraceExporter.export(
                                            session: profileSession),
                                        activeMemoryBytes: summary.activeMemoryBytes,
                                        peakMemoryBytes: summary.peakMemoryBytes,
                                        acceptRate: nil,
                                        timeToFirstToken: summary.timeToFirstToken,
                                        turnIndex: currentTurnIndex,
                                        cacheReused: continueConversation,
                                        conversationReplayed: false,
                                        inputDescription: inputDescription,
                                        mtpStatus: Qwen38MTPRunStatus(
                                            availability: .fallback(
                                                "Flash-Next : MTP local mesuré ~0,8-0,86x greedy (PM4), sous le seuil ≤0,8x"))
                                    )))
                        }
                    }
                    continuation.finish()
                    profiler.disable()
                } catch {
                    continuation.finish(throwing: error)
                    profiler.disable()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
