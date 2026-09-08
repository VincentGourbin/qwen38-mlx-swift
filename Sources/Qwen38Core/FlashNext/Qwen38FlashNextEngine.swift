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
                "Flash-Next : les images ne sont pas encore supportées en mode stateless (LAN)."
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
    public init(directory: URL, profileLayers: Bool = false, residentAsyncEval: Bool = true) async throws {
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
            profileLayers: profileLayers, residentAsyncEval: residentAsyncEval)
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
            // Flash-Next MTP is a separate, opt-in chantier (P-MTP) not yet
            // wired into the runtime — log and ignore rather than fail the
            // request (H3.2: "options.mtp ignoré, un log, pas d'erreur").
            FileHandle.standardError.write(
                Data(
                    "qwen38: Flash-Next ignore options.mtp (P-MTP en chantier)\n".utf8))
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
                                                "Flash-Next : MTP local en chantier P-MTP"))
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
