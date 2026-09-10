import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import MLXProfiler

public struct Qwen38GenerationOptions: Sendable, Equatable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var enableThinking: Bool
    public var reasoningEffort: String
    public var kvBits: Int?
    public var mtp: Qwen38MTPOptions
    /// P5.3: OpenAI `presence_penalty` (and `frequency_penalty`, accepted as
    /// an alias — see the server's field mapping). Applied only by the
    /// Flash-Next streaming generator, only when `temperature > 0`
    /// (`Qwen4ExpStreamingGenerator`); the 27B path and greedy decoding
    /// ignore it. Default 0 (no-op, matches pre-P5.3 behavior); the server
    /// applies its own default (1.5) when a sampling request omits the
    /// field entirely (PLAN.md §2.1 instruct preset).
    public var presencePenalty: Float
    /// P5.3: multiplicative repetition penalty (`extra.repetition_penalty`).
    /// 1.0 is a no-op (default). Same scope restriction as `presencePenalty`.
    public var repetitionPenalty: Float

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        topP: Float = 0.95,
        topK: Int = 20,
        enableThinking: Bool = true,
        reasoningEffort: String = "xhigh",
        kvBits: Int? = 4,
        mtp: Qwen38MTPOptions = .init(),
        presencePenalty: Float = 0,
        repetitionPenalty: Float = 1.0
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.kvBits = kvBits
        self.mtp = mtp
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
    }

    public var parameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            kvBits: kvBits,
            kvGroupSize: 64,
            quantizedKVStart: 5000,
            temperature: temperature,
            topP: topP,
            topK: topK,
            seed: temperature == 0 ? nil : 42
        )
    }
}

public struct Qwen38RunMetrics: Sendable {
    public let metrics: LLMMetrics
    public let stopReason: GenerateStopReason
    public let report: String
    public let chromeTrace: Data
    public let activeMemoryBytes: Int
    public let peakMemoryBytes: Int
    public let acceptRate: Double?
    public let turnIndex: Int
    /// Human-readable modality of the user input for the benchmark panel.
    public let inputDescription: String
    /// Whether this turn appended to an already-prefilled ChatSession cache.
    public let cacheReused: Bool
    /// Wall-clock time from request start (after model loading) to the first
    /// non-empty streamed output chunk.
    public let timeToFirstToken: TimeInterval?
    /// True when this turn had to replay the accumulated conversation instead
    /// of appending to ChatSession's persistent KV cache (M1 MTP path).
    public let conversationReplayed: Bool
    public let mtpStatus: Qwen38MTPRunStatus

    public init(
        metrics: LLMMetrics,
        stopReason: GenerateStopReason,
        report: String,
        chromeTrace: Data,
        activeMemoryBytes: Int = 0,
        peakMemoryBytes: Int = 0,
        acceptRate: Double? = nil,
        timeToFirstToken: TimeInterval? = nil,
        turnIndex: Int = 1,
        cacheReused: Bool = false,
        conversationReplayed: Bool = false,
        inputDescription: String = "Texte",
        mtpStatus: Qwen38MTPRunStatus = .init(availability: .unavailable)
    ) {
        self.metrics = metrics
        self.stopReason = stopReason
        self.report = report
        self.chromeTrace = chromeTrace
        self.activeMemoryBytes = activeMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.acceptRate = acceptRate
        self.timeToFirstToken = timeToFirstToken
        self.turnIndex = turnIndex
        self.cacheReused = cacheReused
        self.conversationReplayed = conversationReplayed
        self.inputDescription = inputDescription
        self.mtpStatus = mtpStatus
    }
}

/// Token-level comparison between the local M2 loop and the upstream M1
/// iterator.  Text equality alone is not sufficient here: a tokenizer can
/// hide a divergence until several tokens later.
public struct Qwen38MTPParityResult: Sendable, Equatable {
    public let localTokenIDs: [Int32]
    public let upstreamTokenIDs: [Int32]

    public init(localTokenIDs: [Int32], upstreamTokenIDs: [Int32]) {
        self.localTokenIDs = localTokenIDs
        self.upstreamTokenIDs = upstreamTokenIDs
    }

    public var isIdentical: Bool { localTokenIDs == upstreamTokenIDs }

    public var firstDifference: Int? {
        let commonCount = min(localTokenIDs.count, upstreamTokenIDs.count)
        if let index = (0 ..< commonCount).first(where: {
            localTokenIDs[$0] != upstreamTokenIDs[$0]
        }) {
            return index
        }
        return localTokenIDs.count == upstreamTokenIDs.count ? nil : commonCount
    }
}

public enum Qwen38GenerationEvent: Sendable {
    case chunk(String)
    case metrics(Qwen38RunMetrics)
}

/// A transport-neutral chat message used by the LAN server. Keeping this in
/// Core lets the server replay a complete OpenAI-style conversation without
/// exposing MLXLMCommon's non-Sendable Chat.Message type.
public struct Qwen38ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable { case system, user, assistant }

    public let role: Role
    public let content: String
    public let imageURLs: [URL]

    public init(role: Role, content: String, imageURLs: [URL] = []) {
        self.role = role
        self.content = content
        self.imageURLs = imageURLs
    }
}

private struct Qwen38ConversationTurn: Sendable {
    let role: Chat.Message.Role
    let text: String
    let imageURLs: [URL]
}

private final class Qwen38MTPTokenSink: @unchecked Sendable {
    let continuation: AsyncStream<Int32>.Continuation

    init(_ continuation: AsyncStream<Int32>.Continuation) {
        self.continuation = continuation
    }

    func yield(_ token: Int32) { continuation.yield(token) }
    func finish() { continuation.finish() }
}

/// First-stage runtime: uses mlx-swift-lm's upstream qwen3_5 implementation.
/// The actor boundary is also the serialization point used by the future LAN server.
public actor Qwen38Runtime {
    private var container: ModelContainer?
    private var chatSession: ChatSession?
    private let mtpProvider = Qwen38MTPDrafterProvider()
    private var mtpDrafter: Qwen38MTPDrafterBox?
    private var mtpAvailability: Qwen38MTPAvailability = .unavailable
    private var conversationTurns: [Qwen38ConversationTurn] = []
    private var m2Conversation: Qwen38MTPConversation?
    private var m2ConversationTurns: [Qwen38ConversationTurn] = []
    private var conversationTurnCount = 0
    /// M1's standalone upstream MTP path rebuilds the prompt through
    /// `MLXLMCommon.generate`, so it cannot share ChatSession's persistent KV
    /// cache yet. Once a conversation uses MTP, keep subsequent turns on the
    /// same replay path until reset, even if MTP is toggled off, so history is
    /// not split between two incompatible cache implementations.
    private var directConversationMode = false
    public private(set) var loadedDirectory: URL?
    private let flashNextEngineFactory: any Qwen38FlashNextEngineFactory
    private var flashEngine: (any Qwen38FlashNextEngineProtocol)?

    public init(flashNextEngineFactory: any Qwen38FlashNextEngineFactory = Qwen38DefaultFlashNextEngineFactory()) {
        self.flashNextEngineFactory = flashNextEngineFactory
    }

    public var isLoaded: Bool { container != nil || flashEngine != nil }

    public func load(
        from directory: URL,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        preloadMTP: Bool = true
    ) async throws {
        let info = try Qwen38ModelValidator.validate(directory)
        guard let family = info.family else {
            throw Qwen38ModelValidationError.unsupportedModelType(info.modelType)
        }
        if loadedDirectory == directory, container != nil || flashEngine != nil {
            return
        }
        chatSession = nil
        container = nil
        mtpDrafter = nil
        mtpAvailability = .unavailable
        conversationTurns = []
        m2Conversation = nil
        m2ConversationTurns = []
        directConversationMode = false
        flashEngine?.unload()
        flashEngine = nil
        Memory.clearCache()
        // The local MLXLMCommon overload does not expose a progress callback;
        // progress is available on the remote-loading overload only.
        _ = progressHandler

        switch family {
        case .qwen4Exp:
            // Netflix-void pattern (H3.3): bound the Metal buffer cache
            // while Flash-Next is resident — an unmeasured starting value,
            // to revisit once P (débit) profiles the resident path.
            Memory.cacheLimit = 8 * 1024 * 1024 * 1024
            flashEngine = try await flashNextEngineFactory.makeEngine(directory: directory)
            // PM4.3 (branchement, 2026-09-09): `mtpState` now delegates to
            // the loaded engine's own dynamic availability (predictor loads
            // lazily on the first MTP-enabled turn) instead of a fixed
            // snapshot taken here — see the `mtpState` getter below.
            mtpAvailability = .unavailable
        case .qwen35:
            await Qwen38MTPRegistration.register()
            // The generic helper tries registered factories in order. The LLM
            // factory also accepts qwen3_5 and would silently load the text-only
            // implementation, dropping vision inputs. Select the VLM factory
            // explicitly so Qwen35.prepare() receives the processed image.
            container = try await VLMModelFactory.shared.loadContainer(
                from: directory,
                using: Qwen38TokenizerLoader()
            )
            // Leave processing overrides empty so the Qwen processor uses the
            // checkpoint's own min/max pixel contract for each image.
            chatSession = ChatSession(container!, processing: .init())
            if preloadMTP {
                let mtpResult = await mtpProvider.loadIfAvailable(for: directory)
                mtpAvailability = mtpResult.availability
                mtpDrafter = mtpResult.box
            }
        }
        conversationTurnCount = 0
        loadedDirectory = directory
    }

    public func unload() async {
        chatSession = nil
        container = nil
        mtpDrafter = nil
        mtpAvailability = .unavailable
        await mtpProvider.unload()
        conversationTurns = []
        m2Conversation = nil
        m2ConversationTurns = []
        flashEngine?.unload()
        flashEngine = nil
        loadedDirectory = nil
        conversationTurnCount = 0
        directConversationMode = false
        Memory.clearCache()
    }

    /// Clears the conversation history and KV cache while keeping the model
    /// weights resident for clean repeated benchmarks.
    public func resetConversation() {
        if let flashEngine {
            flashEngine.resetConversation()
            conversationTurnCount = 0
            return
        }
        guard let container else { return }
        // Recreating the lightweight session clears its history and KV cache
        // without sending a non-Sendable ChatSession across actor isolation.
        chatSession = ChatSession(container, processing: .init())
        conversationTurns = []
        m2Conversation = nil
        m2ConversationTurns = []
        conversationTurnCount = 0
        directConversationMode = false
    }

    /// PM4.3 (branchement, 2026-09-09): Flash-Next reports its own dynamic
    /// availability (predictor loaded lazily on the first MTP-enabled
    /// turn); the 27B path keeps the fixed snapshot taken at `load()`.
    public var mtpState: Qwen38MTPAvailability {
        if let flashEngine { return flashEngine.mtpState }
        return mtpAvailability
    }

    /// H4.2: whether the resident model currently loaded is Flash-Next —
    /// callers use this to decide whether `flashNextWarmUp()` is meaningful
    /// before showing a per-layer loading progress bar.
    public var isFlashNextLoaded: Bool { flashEngine != nil }

    /// Drives the GUI's Flash-Next loading progress bar (H4.2): forces every
    /// decoder layer to load from the Lexar up front instead of inside the
    /// first turn's TTFT, yielding how many of the (typically 48) layers
    /// have finished loading so far. `nil` when no Flash-Next engine is
    /// resident.
    public func flashNextWarmUp() -> AsyncStream<Int>? {
        flashEngine?.warmUp()
    }

    /// P5.2: exposes the resident Flash-Next engine's export/restore surface
    /// to the server's per-conversation LRU without leaking the concrete
    /// engine type. `nil` when no Flash-Next model is resident (27B keeps
    /// its own single `ChatSession` cache, untouched by the LRU — contrat
    /// §5.1.1 / PLAN.md P5 scope).
    public func exportFlashConversationState(
        ledger: [Qwen38ChatMessage]
    ) -> (any Qwen38FlashConversationStateProtocol)? {
        flashEngine?.exportConversationState(ledger: ledger)
    }

    /// Restores a previously exported state into the resident Flash-Next
    /// engine. A no-op when no Flash-Next model is resident.
    public func restoreFlashConversationState(_ state: any Qwen38FlashConversationStateProtocol) {
        flashEngine?.restoreConversationState(state)
    }

    /// P5.2: releases MLX's allocator cache after the server's LRU drops
    /// evicted conversation snapshots, so the device memory those
    /// `MLXArray`s held is actually returned to the system instead of
    /// sitting in MLX's buffer pool.
    public func clearMLXCache() {
        Memory.clearCache()
    }

    /// Executes the local M2 loop on one prepared request.
    ///
    /// This diagnostic entry point is intentionally separate from `generate`:
    /// M2 currently returns a completed token block, while production
    /// streaming still uses the validated M1 path. It gives us a real
    /// checkpoint probe before changing the GUI's hot path.
    public func runLocalMTP(
        prompt: String,
        imageURLs: [URL] = [],
        options: Qwen38GenerationOptions = .init(),
        blockSize: Int = 3
    ) async throws -> Qwen38MTPPipeline.Result {
        guard let container, let mtpDrafter else {
            throw Qwen38RuntimeError.incompatibleMTPDrafter
        }
        guard options.temperature == 0 else {
            throw Qwen38MTPPipeline.Error.nonGreedySampling
        }
        let additionalContext: [String: any Sendable] = [
            "enable_thinking": options.enableThinking,
            "reasoning_effort": options.reasoningEffort,
        ]
        return try await container.perform { context in
            let input = UserInput(
                chat: [
                    Chat.Message(
                        role: .user,
                        content: prompt,
                        images: imageURLs.map(UserInput.Image.url))
                ],
                additionalContext: additionalContext)
            let prepared = try await context.processor.prepare(input: input)
            let stopTokenIDs = Self.stopTokenIDs(context: context)
            return try Qwen38MTPPipeline.run(
                input: prepared,
                target: context.model,
                drafter: mtpDrafter.model,
                parameters: options.parameters,
                blockSize: blockSize,
                stopTokenIDs: stopTokenIDs)
        }
    }

    /// Compares M2 against the upstream M1 MTP iterator on the exact same
    /// prepared multimodal input.  The comparison is intentionally raw-token
    /// based and greedy; it is a correctness probe, not the production path.
    public func compareLocalMTPWithUpstream(
        prompt: String,
        imageURLs: [URL] = [],
        options: Qwen38GenerationOptions = .init(),
        blockSize: Int = 2
    ) async throws -> Qwen38MTPParityResult {
        guard let container, let mtpDrafter else {
            throw Qwen38RuntimeError.incompatibleMTPDrafter
        }
        guard options.temperature == 0 else {
            throw Qwen38MTPPipeline.Error.nonGreedySampling
        }
        let additionalContext: [String: any Sendable] = [
            "enable_thinking": options.enableThinking,
            "reasoning_effort": options.reasoningEffort,
        ]
        return try await container.perform { context in
            let input = UserInput(
                chat: [
                    Chat.Message(
                        role: .user,
                        content: prompt,
                        images: imageURLs.map(UserInput.Image.url))
                ],
                additionalContext: additionalContext)
            let prepared = try await context.processor.prepare(input: input)

            let local = try Qwen38MTPPipeline.run(
                input: prepared,
                target: context.model,
                drafter: mtpDrafter.model,
                parameters: options.parameters,
                blockSize: blockSize,
                stopTokenIDs: Self.stopTokenIDs(context: context))

            // generateTokens creates a fresh target cache internally, so the
            // local M2 cache above cannot contaminate the upstream baseline.
            let upstreamStream = try MLXLMCommon.generateTokens(
                input: prepared,
                parameters: options.parameters,
                context: context,
                mtpDrafter: mtpDrafter.model,
                blockSize: blockSize)
            var upstream = [Int32]()
            for await event in upstreamStream {
                if let token = event.token {
                    upstream.append(Int32(token))
                }
            }
            return Qwen38MTPParityResult(
                localTokenIDs: local.tokenIDs,
                upstreamTokenIDs: upstream)
        }
    }

    /// Runs two or three consecutive M2 turns while retaining target and
    /// drafter caches. This remains a diagnostic path until streaming metrics
    /// are exposed by the production runtime.
    public func runLocalMTPConversation(
        prompt: String,
        secondPrompt: String,
        thirdPrompt: String? = nil,
        imageURLs: [URL] = [],
        systemPrompt: String? = nil,
        options: Qwen38GenerationOptions = .init(),
        blockSize: Int = 3
    ) async throws -> [Qwen38MTPPipeline.Result] {
        guard let container, let mtpDrafter else {
            throw Qwen38RuntimeError.incompatibleMTPDrafter
        }
        guard options.temperature == 0 else {
            throw Qwen38MTPPipeline.Error.nonGreedySampling
        }
        var turns = m2ConversationTurns
        if turns.isEmpty, let systemPrompt {
            turns.append(.init(role: .system, text: systemPrompt, imageURLs: []))
        }

        let firstTurn = Qwen38ConversationTurn(
            role: .user, text: prompt, imageURLs: imageURLs)
        turns.append(firstTurn)
        let first = try await runM2Turn(
            turns: turns,
            targetContainer: container,
            drafter: mtpDrafter,
            options: options,
            blockSize: blockSize)
        m2Conversation = first.session
        let firstText = await container.decode(tokenIds: first.result.tokenIDs.map(Int.init))
        turns.append(.init(role: .assistant, text: firstText, imageURLs: []))

        turns.append(.init(role: .user, text: secondPrompt, imageURLs: []))
        let second = try await runM2Turn(
            turns: turns,
            targetContainer: container,
            drafter: mtpDrafter,
            options: options,
            blockSize: blockSize)
        let secondText = await container.decode(tokenIds: second.result.tokenIDs.map(Int.init))
        turns.append(.init(role: .assistant, text: secondText, imageURLs: []))

        var results = [first.result, second.result]
        if let thirdPrompt {
            turns.append(.init(role: .user, text: thirdPrompt, imageURLs: []))
            let third = try await runM2Turn(
                turns: turns,
                targetContainer: container,
                drafter: mtpDrafter,
                options: options,
                blockSize: blockSize)
            let thirdText = await container.decode(tokenIds: third.result.tokenIDs.map(Int.init))
            turns.append(.init(role: .assistant, text: thirdText, imageURLs: []))
            results.append(third.result)
        }

        m2ConversationTurns = turns
        return results
    }

    private struct M2ConversationCall: @unchecked Sendable {
        let session: Qwen38MTPConversation
        let result: Qwen38MTPPipeline.Result
        let promptTokenCount: Int
    }

    private func runM2Turn(
        turns: [Qwen38ConversationTurn],
        targetContainer: ModelContainer,
        drafter: Qwen38MTPDrafterBox,
        options: Qwen38GenerationOptions,
        blockSize: Int,
        tokenSink: Qwen38MTPTokenSink? = nil,
        didStartGeneration: (@Sendable () -> Void)? = nil
    ) async throws -> M2ConversationCall {
        let existingSession = m2Conversation
        return try await targetContainer.perform { context in
            let input = UserInput(
                chat: turns.map { turn in
                    Chat.Message(
                        role: turn.role,
                        content: turn.text,
                        images: turn.imageURLs.map(UserInput.Image.url))
                },
                additionalContext: [
                    "enable_thinking": options.enableThinking,
                    "reasoning_effort": options.reasoningEffort,
                ])
            if let session = existingSession {
                guard let lastTurn = turns.last,
                    lastTurn.role == .user,
                    lastTurn.imageURLs.isEmpty
                else {
                    throw Qwen38MTPConversation.Error.unsupportedContinuation
                }

                // The target cache already contains the previous assistant
                // stream. Append only the exact Qwen structural suffix; do
                // not render the old assistant text again, because the
                // template's `reasoning_content` field is not representable
                // by Chat.Message and would shift the first token boundary.
                let assistantPrompt = options.enableThinking
                    ? "<|im_start|>assistant\n<think>\n"
                    : "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                let imEndTokenID = context.tokenizer.convertTokenToId("<|im_end|>")
                    .map(Int32.init)
                let assistantClose = session.pendingStopTokenID == imEndTokenID
                    ? "\n<|im_start|>user\n"
                    : "<|im_end|>\n<|im_start|>user\n"
                let suffixText = assistantClose
                    + lastTurn.text
                    + "<|im_end|>\n"
                    + assistantPrompt
                let suffix = context.tokenizer.encode(
                    text: suffixText, addSpecialTokens: false).map(Int32.init)
                let result = try session.append(
                    suffixTokens: suffix,
                    target: context.model,
                    drafter: drafter.model,
                    didStartGeneration: didStartGeneration,
                    didGenerate: { token in tokenSink?.yield(token) })
                return M2ConversationCall(
                    session: session, result: result, promptTokenCount: suffix.count)
            }
            let prepared = try await context.processor.prepare(input: input)
            let session = try Qwen38MTPConversation(
                target: context.model,
                parameters: options.parameters,
                blockSize: blockSize,
                stopTokenIDs: Self.stopTokenIDs(context: context))
            let result = try session.start(
                input: prepared,
                target: context.model,
                drafter: drafter.model,
                didStartGeneration: didStartGeneration,
                didGenerate: { token in tokenSink?.yield(token) })
            return M2ConversationCall(
                session: session, result: result,
                promptTokenCount: prepared.text.tokens.size)
        }
    }

    private func makeLocalMTPStream(
        turns: [Qwen38ConversationTurn],
        targetContainer: ModelContainer,
        drafter: Qwen38MTPDrafterBox,
        options: Qwen38GenerationOptions,
        blockSize: Int,
        profiler: MLXProfiler
    ) async throws -> AsyncThrowingStream<Generation, Error> {
        let tokenizer = await targetContainer.perform { context in context.tokenizer }
        var tokenContinuation: AsyncStream<Int32>.Continuation?
        let tokenStream = AsyncStream<Int32> { continuation in
            tokenContinuation = continuation
        }
        guard let tokenContinuation else {
            throw Qwen38RuntimeError.localMTPStreamUnavailable
        }
        let sink = Qwen38MTPTokenSink(tokenContinuation)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    profiler.startPrefill()
                    let callTask = Task {
                        defer { sink.finish() }
                        return try await self.runM2Turn(
                            turns: turns,
                            targetContainer: targetContainer,
                            drafter: drafter,
                            options: options,
                            blockSize: blockSize,
                            tokenSink: sink,
                            didStartGeneration: {
                                profiler.endPrefill()
                                profiler.startGeneration()
                            })
                    }

                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                    let visibleTokenFilter = Qwen38VisibleTokenFilter(tokenizer: tokenizer)
                    for await token in tokenStream {
                        guard visibleTokenFilter.shouldEmit(Int(token)) else { continue }
                        let decodeStart = Date()
                        detokenizer.append(token: Int(token))
                        let chunk = Qwen38VisibleText.sanitize(detokenizer.next() ?? "")
                        profiler.addDecodingTime(Date().timeIntervalSince(decodeStart))
                        if !chunk.isEmpty {
                            continuation.yield(.chunk(chunk))
                        }
                    }

                    let call = try await callTask.value
                    profiler.endGeneration(tokenCount: call.result.tokenIDs.count)
                    let measured = profiler.getLLMMetrics()
                    let info = GenerateCompletionInfo(
                        promptTokenCount: call.promptTokenCount,
                        generationTokenCount: call.result.tokenIDs.count,
                        promptTime: measured.prefillTime,
                        generationTime: measured.generationTime,
                        stopReason: call.result.stopReason,
                        proposedDraftTokens: call.result.stats.proposedTokens,
                        acceptedDraftTokens: call.result.stats.acceptedTokens)
                    continuation.yield(.info(info))

                    let assistantText = Qwen38VisibleText.sanitize(tokenizer.decode(
                        tokenIds: call.result.tokenIDs.map(Int.init),
                        skipSpecialTokens: false))
                    // The session keeps an EOS boundary for the next turn;
                    // don't downgrade it to replay just because generation
                    // ended normally on a stop token.
                    self.m2Conversation = call.session
                    self.m2ConversationTurns = turns + [
                        .init(role: .assistant, text: assistantText, imageURLs: [])
                    ]
                    continuation.finish()
                } catch {
                    sink.finish()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func decode(tokenIDs: [Int32]) async -> String {
        if let flashEngine {
            return flashEngine.decode(tokenIDs: tokenIDs)
        }
        guard let container else { return "" }
        return await container.decode(tokenIds: tokenIDs.map(Int.init))
    }

    /// Stateless entry point for the HTTP server. The caller supplies the
    /// complete conversation; the runtime uses a fresh replay path and then
    /// restores its local interactive conversation state. This prevents one
    /// LAN client from leaking KV history into another client while retaining
    /// the one-model-resident memory policy.
    public func generateStateless(
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions = .init()
    ) async throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        if let flashEngine {
            // Flash-Next has no per-client persistent cache (contrat
            // §5.1.1, "Stateless v1"): the whole history is rendered as one
            // turn instead of replaying it through the in-process
            // conversation state used by the 27B path below.
            return try flashEngine.generateFromMessages(messages: messages, options: options)
        }
        guard let lastUserIndex = messages.lastIndex(where: { $0.role == .user }) else {
            throw Qwen38RuntimeError.missingUserMessage
        }
        guard lastUserIndex == messages.count - 1 else {
            throw Qwen38RuntimeError.missingUserMessage
        }
        let priorTurns = messages[..<lastUserIndex].map {
            Qwen38ConversationTurn(
                role: Chat.Message.Role(rawValue: $0.role.rawValue) ?? .user,
                text: $0.content,
                imageURLs: $0.imageURLs)
        }
        let previousTurns = conversationTurns
        let previousCount = conversationTurnCount
        let previousDirectMode = directConversationMode
        conversationTurns = Array(priorTurns)
        conversationTurnCount = priorTurns.filter { $0.role == .user }.count
        directConversationMode = false
        do {
            let last = messages[lastUserIndex]
            let stream = try await generate(
                prompt: last.content,
                imageURLs: last.imageURLs,
                options: options,
                forceConversationReplay: true)
            return AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        for try await event in stream { continuation.yield(event) }
                        self.conversationTurns = previousTurns
                        self.conversationTurnCount = previousCount
                        self.directConversationMode = previousDirectMode
                        self.resetConversation()
                        continuation.finish()
                    } catch {
                        self.conversationTurns = previousTurns
                        self.conversationTurnCount = previousCount
                        self.directConversationMode = previousDirectMode
                        self.resetConversation()
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        } catch {
            conversationTurns = previousTurns
            conversationTurnCount = previousCount
            directConversationMode = previousDirectMode
            throw error
        }
    }

    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        imageURLs: [URL] = [],
        options: Qwen38GenerationOptions = .init(),
        forceConversationReplay: Bool = false
    ) async throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        if let flashEngine {
            conversationTurnCount += 1
            return try flashEngine.generate(
                prompt: prompt, systemPrompt: systemPrompt, imageURLs: imageURLs,
                options: options)
        }
        guard let chatSession else { throw Qwen38RuntimeError.modelNotLoaded }

        chatSession.generateParameters = options.parameters
        chatSession.additionalContext = [
            "enable_thinking": options.enableThinking,
            "reasoning_effort": options.reasoningEffort,
        ]
        if let systemPrompt {
            chatSession.instructions = systemPrompt
        }
        let turnIndex = conversationTurnCount + 1
        conversationTurnCount = turnIndex
        let imageCount = imageURLs.count
        let images = imageURLs.map(UserInput.Image.url)

        let requestedMTP = options.mtp.enabled
        let localMTPRequested = requestedMTP && options.mtp.engine == .local
        // M1 replays the complete rendered conversation.  The upstream Qwen
        // MTP drafter can prefill a text-only private cache, but it does not
        // receive the full per-token 3-axis M-RoPE table for image tokens.
        // Replaying an image-bearing history would therefore give the drafter
        // an invalid prefix.  Keep MTP for a cold first image turn, and fail
        // closed on later image-bearing replays until M2 owns the persistent
        // target/drafter state and position table.
        let imageBearingReplay = turnIndex > 1
            && (!imageURLs.isEmpty || conversationTurns.contains { !$0.imageURLs.isEmpty })
        let canUseLocalMTP = localMTPRequested
            && options.temperature == 0
            && mtpDrafter != nil
            && (turnIndex == 1 || imageURLs.isEmpty)
            && (turnIndex == 1 || m2Conversation?.isStarted == true)
        let canUseUpstreamMTP = !localMTPRequested
            && requestedMTP
            && options.temperature == 0
            && mtpDrafter != nil
            && !imageBearingReplay
        let canUseMTP = canUseLocalMTP || canUseUpstreamMTP
        let mtpFallback: String?
        if !requestedMTP {
            mtpFallback = nil
        } else if options.temperature != 0 {
            mtpFallback = "Le MTP upstream requiert un échantillonnage greedy (température 0)."
        } else if localMTPRequested && !canUseLocalMTP {
            if turnIndex > 1 && !imageURLs.isEmpty {
                mtpFallback = "M2 local ne réutilise pas un cache après l'ajout d'une nouvelle image : réinitialiser la conversation."
            } else if turnIndex > 1 && m2Conversation?.isStarted != true {
                mtpFallback = "État M2 arrêté ou non initialisé : replay contrôlé de la conversation."
            } else {
                mtpFallback = "M2 local indisponible pour cette transition de conversation."
            }
        } else if imageBearingReplay {
            mtpFallback =
                "MTP désactivé sur le replay d'un historique avec image : positions M-RoPE "
                + "du préfixe non transportées par l'API upstream."
        } else if mtpDrafter == nil {
            if case .fallback(let reason) = mtpAvailability {
                mtpFallback = reason
            } else {
                mtpFallback = "Poids du drafter MTP absents pour cette variante."
            }
        } else {
            mtpFallback = nil
        }
        let useDirectConversation = canUseMTP || directConversationMode || forceConversationReplay
        if canUseMTP {
            directConversationMode = true
        }
        let activeDrafter = mtpDrafter
        // The ordinary ChatSession path reuses its KV cache. The M1 direct
        // path intentionally replays the complete conversation; report that
        // accurately until M2 adds target/drafter snapshot and replay.
        let cacheReused = canUseLocalMTP
            ? turnIndex > 1
            : conversationTurnCount > 1 && !useDirectConversation
        let mtpStatusBeforeRun = canUseMTP
            ? Qwen38MTPRunStatus(
                availability: .active,
                engine: options.mtp.engine,
                blockSize: canUseLocalMTP
                    ? options.mtp.draftDepth.requestedDraftTokens + 1
                    : min(options.mtp.draftDepth.requestedDraftTokens + 1, 2))
            : Qwen38MTPRunStatus(
                availability: requestedMTP
                    ? .fallback(mtpFallback ?? "MTP indisponible")
                    : .unavailable,
                engine: requestedMTP ? options.mtp.engine : nil)

        let userTurn = Qwen38ConversationTurn(
            role: .user, text: prompt, imageURLs: imageURLs)
        if conversationTurns.isEmpty, let systemPrompt {
            conversationTurns.append(.init(role: .system, text: systemPrompt, imageURLs: []))
        }
        conversationTurns.append(userTurn)

        // Start before message reconstruction, image preprocessing, target
        // prefill, and drafter initialization. Otherwise MTP would report a
        // deceptively tiny TTFT because those phases happen before its stream
        // is returned.
        let profiler = MLXProfiler.shared
        let (profileSession, ownsSession, requestPhase) = Qwen38Profiling.beginRequestSession(
            title: "QWEN3.8 INFERENCE",
            metadata: [
                "model": loadedDirectory?.lastPathComponent ?? "Qwen3.8",
                "kvBits": options.kvBits.map(String.init) ?? "none",
                "mtpRequested": String(requestedMTP),
                "mtpEngine": options.mtp.engine.rawValue,
                "mtpSelectedPath": canUseLocalMTP ? "local" : canUseUpstreamMTP ? "upstream" : "fallback",
            ],
            phase: "Requête \(turnIndex)")
        let requestStart = Date()
        profiler.start("Turn")
        profiler.start("Time to first token")

        let generationStream: AsyncThrowingStream<Generation, Error>
        do {
            if canUseLocalMTP, let container, let activeDrafter {
                var turns = m2ConversationTurns
                if turns.isEmpty, let systemPrompt {
                    turns.append(.init(role: .system, text: systemPrompt, imageURLs: []))
                }
                turns.append(userTurn)
                generationStream = try await makeLocalMTPStream(
                    turns: turns,
                    targetContainer: container,
                    drafter: activeDrafter,
                    options: options,
                    blockSize: mtpStatusBeforeRun.blockSize ?? 3,
                    profiler: profiler)
            } else if useDirectConversation, let container {
                let turns = conversationTurns
                let mtpStream = try await container.perform { context in
                    let messages = turns.map { turn in
                        Chat.Message(
                            role: turn.role,
                            content: turn.text,
                            images: turn.imageURLs.map(UserInput.Image.url)
                        )
                    }
                    let input = UserInput(
                        chat: messages,
                        additionalContext: [
                            "enable_thinking": options.enableThinking,
                            "reasoning_effort": options.reasoningEffort,
                        ]
                    )
                    let preparedInput = try await context.processor.prepare(input: input)
                    if canUseMTP, let activeDrafter {
                        guard activeDrafter.model.isCompatible(with: context.model) else {
                            throw Qwen38RuntimeError.incompatibleMTPDrafter
                        }
                        return try MLXLMCommon.generate(
                            input: preparedInput,
                            parameters: options.parameters,
                            context: context,
                            mtpDrafter: activeDrafter.model,
                            blockSize: min(options.mtp.draftDepth.requestedDraftTokens + 1, 2)
                        )
                    }
                    return try MLXLMCommon.generate(
                        input: preparedInput,
                        parameters: options.parameters, context: context
                    )
                }
                generationStream = AsyncThrowingStream { continuation in
                    Task {
                        for await event in mtpStream {
                            continuation.yield(event)
                        }
                        continuation.finish()
                    }
                }
            } else {
                generationStream = chatSession.streamDetails(to: prompt, images: images)
            }
        } catch {
            Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
            removeLastPendingUserMessage()
            throw error
        }

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var completionInfo: GenerateCompletionInfo?
                    var timeToFirstToken: TimeInterval?
                    var outputText = ""
                    for try await event in generationStream {
                        switch event {
                        case .chunk(let text):
                            if timeToFirstToken == nil, !text.isEmpty {
                                timeToFirstToken = Date().timeIntervalSince(requestStart)
                                profiler.end("Time to first token")
                                profiler.start("Decode")
                            }
                            let visibleText = Qwen38VisibleText.sanitize(text)
                            outputText += visibleText
                            if !visibleText.isEmpty {
                                continuation.yield(.chunk(visibleText))
                            }
                        case .info(let info): completionInfo = info
                        case .toolCall, .rejectedToolCall: break
                        }
                    }
                    profiler.end("Decode")
                    profiler.end("Turn")
                    guard let info = completionInfo else {
                        throw Qwen38RuntimeError.missingCompletionInfo
                    }
                    let acceptRate: Double?
                    if let proposed = info.proposedDraftTokens,
                       let accepted = info.acceptedDraftTokens,
                       proposed > 0 {
                        acceptRate = Double(accepted) / Double(proposed)
                    } else {
                        acceptRate = nil
                    }
                    let mtpStatus: Qwen38MTPRunStatus
                    if canUseMTP {
                        let proposed = info.proposedDraftTokens ?? 0
                        let draftPerRound = max((mtpStatusBeforeRun.blockSize ?? 2) - 1, 1)
                        mtpStatus = Qwen38MTPRunStatus(
                            availability: info.passthroughReason == nil
                                ? .active
                                : .fallback(info.passthroughReason!),
                            engine: options.mtp.engine,
                            blockSize: mtpStatusBeforeRun.blockSize,
                            proposedTokens: proposed,
                            acceptedTokens: info.acceptedDraftTokens ?? 0,
                            rounds: proposed == 0
                                ? 0
                                : (proposed + draftPerRound - 1) / draftPerRound,
                            passthroughReason: info.passthroughReason
                        )
                    } else {
                        mtpStatus = mtpStatusBeforeRun
                    }
                    let measured = profiler.getLLMMetrics()
                    let metrics = LLMMetrics(
                        tokenizationTime: measured.tokenizationTime,
                        prefillTime: info.promptTime,
                        generationTime: info.generateTime,
                        decodingTime: measured.decodingTime,
                        promptTokens: info.promptTokenCount,
                        generatedTokens: info.generationTokenCount
                    )
                    continuation.yield(.metrics(Qwen38RunMetrics(
                        metrics: metrics,
                        stopReason: info.stopReason,
                        report: ownsSession ? profileSession.generateReport() : "",
                        chromeTrace: ownsSession ? ChromeTraceExporter.export(session: profileSession) : Data(),
                        activeMemoryBytes: Memory.activeMemory,
                        peakMemoryBytes: Memory.peakMemory,
                        acceptRate: acceptRate,
                        timeToFirstToken: timeToFirstToken,
                        turnIndex: turnIndex,
                        cacheReused: cacheReused,
                        conversationReplayed: useDirectConversation
                            && turnIndex > 1
                            && !canUseLocalMTP,
                        inputDescription: imageCount == 0
                            ? "Texte"
                            : "Texte + \(imageCount) image\(imageCount == 1 ? "" : "s")",
                        mtpStatus: mtpStatus
                    )))
                    self.finishConversationTurn(with: outputText)
                    continuation.finish()
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
                } catch {
                    self.removeLastPendingUserMessage()
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func finishConversationTurn(with text: String) {
        conversationTurns.append(.init(role: .assistant, text: text, imageURLs: []))
    }

    private static func stopTokenIDs(context: ModelContext) -> Set<Int32> {
        var ids = Set(context.configuration.eosTokenIds.map(Int32.init))
        if let tokenizerEOS = context.tokenizer.eosTokenId {
            ids.insert(Int32(tokenizerEOS))
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                ids.insert(Int32(id))
            }
        }
        return ids
    }

    private func removeLastPendingUserMessage() {
        guard conversationTurns.last?.role == .user else { return }
        conversationTurns.removeLast()
        conversationTurnCount = max(conversationTurnCount - 1, 0)
    }
}

public enum Qwen38RuntimeError: LocalizedError, Equatable {
    case modelNotLoaded
    case missingCompletionInfo
    case missingUserMessage
    case incompatibleMTPDrafter
    case localMTPStreamUnavailable

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Aucun modèle Qwen3.8 n'est chargé."
        case .missingCompletionInfo: return "Le runtime n'a pas reçu les métriques de fin de génération."
        case .missingUserMessage: return "La conversation doit se terminer par un message utilisateur."
        case .incompatibleMTPDrafter: return "Le drafter MTP n'est pas compatible avec la cible Qwen3.8 chargée."
        case .localMTPStreamUnavailable: return "Le flux MTP local n'a pas pu être initialisé."
        }
    }
}
