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
    /// PM4.3 (branchement, 2026-09-09): dynamic MTP availability for this
    /// loaded engine. Unlike the 27B path (drafter presence known at load
    /// time), Flash-Next's predictor loads lazily on the first turn that
    /// requests it (`options.mtp.enabled`), so this starts as `.fallback`
    /// and flips to `.active` once that load has happened.
    var mtpState: Qwen38MTPAvailability { get }
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

    /// PM4.3 (branchement, 2026-09-09): loaded lazily on the first turn
    /// that requests `options.mtp.enabled` (`Qwen4ExpMTPLoader`,
    /// `uncachedIO` — same F_NOCACHE contract as the decoder/global
    /// loaders, PLAN.md §6.3-4/8). `nil` means "not requested yet", not
    /// "unavailable": every Flash-Next checkpoint used in this codebase
    /// ships an MTP head.
    private var mtpPredictor: Qwen4ExpMTPPredictor?
    /// The drafter's persistent per-conversation cache (PM4.3): created
    /// once alongside `mtpPredictor` and reused across turns so
    /// `continueConversation` can extend it (`prepareContinuation`)
    /// instead of re-priming from scratch every turn. Cleared by
    /// `resetConversation()` together with the target's own caches.
    private var mtpDraftState: Qwen4ExpFlashMTPState?

    /// Keeps macOS from idle-sleeping while a Flash-Next model is resident:
    /// P1 (2026-09-08) showed the Mac entering 'Idle Sleep' 73 s into a
    /// resident load (1-minute idle sleep in the power profile), which
    /// froze every run of the previous three days. Released in `deinit`.
    private let sleepActivity: NSObjectProtocol

    /// `residentAsyncEval` defaults to `true` since P1 (2026-09-08): on the
    /// real checkpoint, `asyncEval` per layer decoded 6 tokens in 2.33 s
    /// against 2.98 s with a blocking `eval` per layer (-22 %) and 10.25 s
    /// with a single deferred `eval` per token. Same peak memory (75.2 GB).
    ///
    /// P4.0/P4.1 (2026-09-09) found that P1's `residentAsyncEval` was, on
    /// its own, a no-op in production: `residentEvaluationInterval == 1`
    /// made `shouldEvaluate` unconditionally `true`
    /// (`Qwen4ExpStreamingDecoder`), so every layer still took a blocking
    /// `eval` regardless of this flag — Metal System Trace on the real
    /// checkpoint measured only 14.2 % GPU-busy over the prefill+decode
    /// window, far below the synthetic bench's 82-85 %. `residentAsyncInterval`
    /// is the real, separate knob P4.1 added; sweeping N=1/2/4/8/12 on the
    /// real 3-bit checkpoint (32 tokens, 2 runs each, IDs bit-identical to
    /// greedy in all 10 runs) gave 0.164/0.150/0.144/0.141/0.140 s/token —
    /// N=8 (-14.4 %) and N=12 (-14.6 %) are within noise of each other with
    /// diminishing returns past 8, so N=8 (already the reference interval
    /// used throughout `flash-layer-bench --async-interval`) is the default.
    public init(
        directory: URL, profileLayers: Bool = false, residentAsyncEval: Bool = true,
        residentAsyncInterval: Int = 8, uncachedIO: Bool = true
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
            residentAsyncInterval: residentAsyncInterval,
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

    /// PM4.3 (branchement): `.active` once the predictor has been loaded by
    /// a prior MTP-enabled turn, `.fallback` (with a reason a caller can
    /// surface, e.g. the GUI's `mtpHelp` text) until then.
    public var mtpState: Qwen38MTPAvailability {
        mtpPredictor != nil
            ? .active
            : .fallback("Flash-Next : MTP local chargé à la demande au premier tour MTP")
    }

    public func resetConversation() {
        model.resetConversation()
        hasConversationHistory = false
        turnIndex = 0
        // The drafter's cache is tied to the target's own conversation
        // history; the predictor's *weights* stay loaded (no need to pay
        // Lexar IO again), only its per-conversation state is discarded.
        mtpDraftState = nil
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

        let profiler = MLXProfiler.shared
        let profileSession = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
        profileSession.title = "QWEN3.8 FLASH-NEXT INFERENCE"
        profileSession.metadata["model"] = directory.lastPathComponent
        profileSession.metadata["turn"] = String(currentTurnIndex)
        profiler.activeSession = profileSession
        profiler.enable()

        // PM4.3 (branchement, 2026-09-09): opt-in local MTP — default off
        // (`options.mtp.enabled`). PLAN.md P-MTP suite PM4.3 measured block
        // 2 at 0.81-0.86x greedy on the 3-bit checkpoint (bit-identical
        // token ids, `stats.replayedTokens == 0`): faster than greedy but
        // short of the 0.8x bar set for auto-branching, hence a caller has
        // to ask for it explicitly rather than it being the default. Text
        // only — `Qwen4ExpFlashMTPDraftEngine`'s multimodal M-RoPE
        // continuation isn't wired into the drafter (see its doc comment).
        let requestedMTP = options.mtp.enabled
        let hasImage = built.visionEmbeddings != nil
        // The local MTP path is greedy-only: honour it only when the caller
        // asked for greedy decoding (same rule as the 27B path, where MTP
        // requires `temperature == 0`). The server defaults `mtp` to true
        // when the field is omitted, so without this guard a sampled request
        // (temperature 0.7, presets) would silently become greedy.
        let isGreedy = options.temperature <= 0
        if requestedMTP && !hasImage && isGreedy {
            return runMTPGenerationStream(
                built: built, options: options, continueConversation: continueConversation,
                inputDescription: inputDescription, currentTurnIndex: currentTurnIndex,
                maxNewTokens: maxNewTokens, profiler: profiler, profileSession: profileSession)
        }
        if requestedMTP && hasImage {
            FileHandle.standardError.write(
                Data(
                    "qwen38: Flash-Next ignore options.mtp pour ce tour (image présente, MTP local texte seul)\n"
                        .utf8))
        } else if requestedMTP && !isGreedy {
            FileHandle.standardError.write(
                Data(
                    "qwen38: Flash-Next ignore options.mtp pour ce tour (échantillonnage demandé, MTP local greedy seul)\n"
                        .utf8))
        }

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
                            let mtpStatus: Qwen38MTPRunStatus
                            if requestedMTP && hasImage {
                                mtpStatus = Qwen38MTPRunStatus(
                                    availability: .fallback("MTP Flash-Next : texte seul"),
                                    engine: options.mtp.engine)
                            } else if requestedMTP && !isGreedy {
                                mtpStatus = Qwen38MTPRunStatus(
                                    availability: .fallback("MTP Flash-Next : greedy seul (température > 0)"),
                                    engine: options.mtp.engine)
                            } else {
                                mtpStatus = Qwen38MTPRunStatus(availability: .unavailable)
                            }
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
                                        mtpStatus: mtpStatus)))
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

    /// PM4.3 (branchement, 2026-09-09): drives `Qwen4ExpGreedyGenerator.generateMTP`
    /// behind the same `Qwen38GenerationEvent` stream the greedy/sampling
    /// path uses. The predictor loads lazily here (first MTP-enabled turn)
    /// and `mtpDraftState` persists on `self` across turns so a later
    /// continuation turn can extend the drafter's cache instead of
    /// re-priming it (see `generateMTP`'s `state`/`continueConversation`
    /// docs). MTP is greedy-only (`generateMTP` samples via `ArgMaxSampler`
    /// throughout, target and drafter alike, matching every CLI probe);
    /// `options.temperature`/`topP`/`topK` are not applied to this path.
    private func runMTPGenerationStream(
        built: Qwen4ExpBuiltPrompt, options: Qwen38GenerationOptions,
        continueConversation: Bool, inputDescription: String,
        currentTurnIndex: Int, maxNewTokens: Int,
        profiler: MLXProfiler, profileSession: ProfilingSession
    ) -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        let requestedDrafts = options.mtp.draftDepth.requestedDraftTokens
        let blockSize = min(max(requestedDrafts + 1, 2), 4)
        let mtpEngineKind = options.mtp.engine

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Loading (Lexar IO, first MTP turn only) and the
                    // draft-state lookup both happen inside the task so a
                    // caller gets its stream back immediately, exactly like
                    // the greedy path above.
                    let predictor: Qwen4ExpMTPPredictor
                    if let loaded = self.mtpPredictor {
                        predictor = loaded
                    } else {
                        let loaded = try Qwen4ExpMTPLoader.load(
                            from: self.directory, uncachedIO: true)
                        predictor = loaded.model
                        self.mtpPredictor = predictor
                    }
                    let state = self.mtpDraftState
                        ?? Qwen4ExpFlashMTPDraftEngine(target: self.model, predictor: predictor)
                            .makeState()
                    self.mtpDraftState = state

                    let greedyGenerator = Qwen4ExpGreedyGenerator(model: self.model)
                    let result = try greedyGenerator.generateMTP(
                        promptTokenIDs: built.tokenIDs,
                        predictor: predictor,
                        options: .init(maxNewTokens: maxNewTokens, stopTokenIDs: self.stopTokenIDs),
                        blockSize: blockSize,
                        profiler: profiler,
                        continueConversation: continueConversation,
                        state: state,
                        onToken: { token in
                            if Task.isCancelled { return }
                            guard self.visibleTokenFilter.shouldEmit(Int(token)) else { return }
                            let piece = Qwen38VisibleText.sanitize(
                                self.tokenizer.decode(
                                    tokens: [Int(token)], skipSpecialTokens: false))
                            if !piece.isEmpty {
                                continuation.yield(.chunk(piece))
                            }
                        })

                    let stopReason: GenerateStopReason
                    if let last = result.tokenIDs.last, self.stopTokenIDs.contains(last) {
                        stopReason = .stop
                    } else if result.tokenIDs.count >= maxNewTokens {
                        stopReason = .length
                    } else {
                        stopReason = .cancelled
                    }
                    let llmMetrics = LLMMetrics(
                        prefillTime: result.prefillTime,
                        generationTime: result.generationTime,
                        promptTokens: result.promptTokenCount,
                        generatedTokens: result.tokenIDs.count)
                    let mtpStatus = Qwen38MTPRunStatus(
                        availability: .active,
                        engine: mtpEngineKind,
                        blockSize: blockSize,
                        proposedTokens: result.stats.proposedTokens,
                        acceptedTokens: result.stats.acceptedTokens,
                        rounds: result.stats.rounds)
                    continuation.yield(
                        .metrics(
                            Qwen38RunMetrics(
                                metrics: llmMetrics,
                                stopReason: stopReason,
                                report: profileSession.generateReport(),
                                chromeTrace: ChromeTraceExporter.export(session: profileSession),
                                activeMemoryBytes: Memory.activeMemory,
                                peakMemoryBytes: Memory.peakMemory,
                                acceptRate: result.stats.acceptanceRate,
                                timeToFirstToken: result.timeToFirstToken,
                                turnIndex: currentTurnIndex,
                                cacheReused: continueConversation,
                                conversationReplayed: false,
                                inputDescription: inputDescription,
                                mtpStatus: mtpStatus)))
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
