import CoreGraphics
import Foundation
import MLX
import MLXLMCommon
import Qwen38Core

/// What an answer is made of, in stream order: reasoning (when thinking is
/// on), visible text, tool calls (after the text, once the answer is
/// complete), then usage.
public enum Qwen38BrainEvent: Sendable {
    case reasoning(String)
    case text(String)
    /// `argumentsJSON` is typed against the tool's JSON schema (numbers stay
    /// numbers). Replay it in the next request as an assistant message's
    /// `toolCalls`, followed by one `.tool` message per result.
    case toolCall(Qwen38ToolCall)
    case done(Qwen38BrainUsage)
}

public struct Qwen38BrainUsage: Sendable, Equatable {
    /// Prompt tokens prefilled by this answer.
    public let promptTokens: Int
    /// Prompt tokens reused from the previous answer's cache.
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let promptTokensPerSecond: Double
    /// Wall time spent prefilling this answer's new prompt tokens.
    public let prefillSeconds: Double
    public let tokensPerSecond: Double
    public let timeToFirstToken: TimeInterval?
    public let peakMemoryBytes: Int
    /// Decode time per token, median and 90th percentile, in seconds (dense
    /// engine only; `nil` on the runtime path).
    public let stepMedian: Double?
    public let stepP90: Double?
    /// `length` when `maxTokens` cut the answer, `stop` otherwise; `toolCalls`
    /// when the answer ended by asking for tools.
    public let finishReason: FinishReason

    public enum FinishReason: String, Sendable { case stop, length, toolCalls }
}

public struct Qwen38BrainOptions: Sendable, Equatable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var enableThinking: Bool
    /// `low`, `medium` or `xhigh` — the Qwen template has no `high`.
    public var reasoningEffort: String
    /// Resize every image to this size before the vision tower. `nil` keeps
    /// the checkpoint's own pixel budget (≈ 1,280 vision tokens per image for
    /// Bonsai 2): most detail, longest prefill. 512×512 ≈ 170 tokens.
    public var imageResize: CGSize?

    public init(
        maxTokens: Int = 2048, temperature: Float = 0.7, topP: Float = 0.95, topK: Int = 20,
        enableThinking: Bool = false, reasoningEffort: String = "low",
        imageResize: CGSize? = nil
    ) {
        self.imageResize = imageResize
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }
}

public struct Qwen38BrainMemoryReport: Sendable, Equatable {
    public let activeBytes: Int
    public let cacheBytes: Int
    public let peakBytes: Int
}

public enum Qwen38BrainError: LocalizedError {
    case invalidReasoningEffort(String)
    case emptyConversation
    case imagesNeedVision

    public var errorDescription: String? {
        switch self {
        case .invalidReasoningEffort(let value):
            return "reasoningEffort \(value) inconnu : low, medium ou xhigh"
        case .emptyConversation:
            return "conversation vide : au moins un message utilisateur ou outil"
        case .imagesNeedVision:
            return "ce profil charge le modèle sans vision : pas d'image possible"
        }
    }
}

/// The embeddable entry point: one model resident, OpenAI-shaped
/// conversations in, typed events out. The whole conversation is passed on
/// every call, like an HTTP chat API; the runtime reuses its cache when the
/// new conversation extends the previous one.
///
/// ```swift
/// let brain = try await Qwen38Brain.load(modelDirectory: url, profile: .lean)
/// for try await event in await brain.respond(to: messages, tools: tools) { … }
/// ```
public actor Qwen38Brain {
    public let modelDirectory: URL
    public let profile: Qwen38BrainProfile
    private let runtime: Qwen38Runtime
    /// Dense hybrid family (Qwen 3.5 / Bonsai 2): answered by
    /// `Qwen38BrainDenseEngine`, which reuses the conversation cache across
    /// requests. Flash-Next keeps the runtime's own path.
    private let isDense: Bool
    private let conversation = Qwen38BrainConversationCache()
    private let stopTokenIDs: Set<Int>
    /// Prefill chunk, in tokens; starts at the profile's value. On Bonsai 2 at
    /// 10k tokens, 512 measured 100 tok/s / 16 GB peak against 88 tok/s /
    /// 32 GB at 2048 and 94 tok/s / 52 GB at 4096; lean uses 256 (32k:
    /// 11.3 GB peak, 73 tok/s) — `docs/bonsai2-brain/plan.md`, K-7.
    public var prefillStepSize: Int

    private init(
        modelDirectory: URL, profile: Qwen38BrainProfile, runtime: Qwen38Runtime,
        isDense: Bool, stopTokenIDs: Set<Int>
    ) {
        self.modelDirectory = modelDirectory
        self.profile = profile
        self.runtime = runtime
        self.isDense = isDense
        self.stopTokenIDs = stopTokenIDs
        self.prefillStepSize = profile.prefillStepSize
    }

    /// A chat-template message: the checkpoint's own shape (ordered tool-call
    /// arguments), with images as `{"type": "image"}` content parts before the
    /// text, as Qwen's vision template expects.
    static func templateMessage(_ message: Qwen38ChatMessage) -> [String: any Sendable] {
        var rendered = Qwen4ExpPromptBuilder.hfMessage(from: message)
        if !message.imageURLs.isEmpty {
            let parts: [[String: any Sendable]] =
                message.imageURLs.map { _ in ["type": "image"] }
                + [["type": "text", "text": message.content]]
            rendered["content"] = parts
        }
        return rendered
    }

    static func percentile(_ values: [Double], _ q: Double) -> Double? {
        // The first interval includes the queueing of the first step: skip it.
        let sorted = values.dropFirst().sorted()
        guard !sorted.isEmpty else { return nil }
        return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * q))]
    }

    public func setPrefillStepSize(_ value: Int) {
        prefillStepSize = max(64, value)
    }

    /// Loads the model (Bonsai 2 or any checkpoint `Qwen38ModelValidator`
    /// accepts) and applies the profile's process-wide memory policy.
    public static func load(
        modelDirectory: URL, profile: Qwen38BrainProfile = .fast
    ) async throws -> Qwen38Brain {
        let info = try Qwen38ModelValidator.validate(modelDirectory)
        let runtime = Qwen38Runtime()
        try await runtime.load(
            from: modelDirectory, preloadMTP: false, textOnly: profile.textOnly)
        profile.applyGlobalPolicy()
        // Stop tokens only matter to the dense engine; Flash-Next's runtime
        // path has no `ModelContainer` and handles its own stops.
        var stops = Set<Int>()
        if info.family == .qwen35 {
            let tokenizerStops = try await runtime.performRaw { _, tokenizer in
                ["<|im_end|>", "<|endoftext|>"].compactMap { tokenizer.convertTokenToId($0) }
            }
            stops.formUnion(tokenizerStops)
        }
        if let data = try? Data(
            contentsOf: modelDirectory.appending(component: "generation_config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let one = object["eos_token_id"] as? Int { stops.insert(one) }
            if let many = object["eos_token_id"] as? [Int] { stops.formUnion(many) }
        }
        return Qwen38Brain(
            modelDirectory: modelDirectory, profile: profile, runtime: runtime,
            // `QWEN38_BRAIN_ENGINE=runtime` forces the runtime's own path on
            // the dense family too — to check the dense engine against it.
            isDense: info.family == .qwen35
                && ProcessInfo.processInfo.environment["QWEN38_BRAIN_ENGINE"] != "runtime",
            stopTokenIDs: stops)
    }

    public func respond(
        to messages: [Qwen38ChatMessage], tools: [Qwen38ToolSpec] = [],
        options: Qwen38BrainOptions = .init()
    ) -> AsyncThrowingStream<Qwen38BrainEvent, Error> {
        let runtime = runtime
        let profile = profile
        let isDense = isDense
        let conversation = conversation
        let stopTokenIDs = stopTokenIDs
        let prefillStepSize = prefillStepSize
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard ["low", "medium", "xhigh"].contains(options.reasoningEffort) else {
                        throw Qwen38BrainError.invalidReasoningEffort(options.reasoningEffort)
                    }
                    guard !messages.isEmpty else { throw Qwen38BrainError.emptyConversation }
                    guard !profile.textOnly || messages.allSatisfy({ $0.imageURLs.isEmpty }) else {
                        throw Qwen38BrainError.imagesNeedVision
                    }
                    // Shared by both engines: reasoning / text / tool calls.
                    var parser = Qwen38ThinkingStreamParser(primedInside: options.enableThinking)
                    var splitter = Qwen38ToolCallTextSplitter(enabled: !tools.isEmpty)
                    var fullContent = ""
                    func absorb(_ chunk: String) {
                        let output = parser.append(chunk)
                        if !output.reasoning.isEmpty { continuation.yield(.reasoning(output.reasoning)) }
                        fullContent += output.content
                        let visible = splitter.append(output.content)
                        if !visible.isEmpty { continuation.yield(.text(visible)) }
                    }
                    func finish(stoppedByLength: Bool) -> Qwen38BrainUsage.FinishReason {
                        let tail = parser.finish()
                        if !tail.reasoning.isEmpty { continuation.yield(.reasoning(tail.reasoning)) }
                        fullContent += tail.content
                        let visibleTail = splitter.append(tail.content) + splitter.finish()
                        if !visibleTail.isEmpty { continuation.yield(.text(visibleTail)) }
                        var finish: Qwen38BrainUsage.FinishReason = stoppedByLength ? .length : .stop
                        if !tools.isEmpty {
                            let parsed = Qwen38ToolCallParser.parse(fullContent)
                            for call in parsed.calls {
                                let schema = tools.first(where: { $0.name == call.name })?.parameters
                                let arguments = Qwen38ToolArgumentTyper.typedArguments(
                                    call.parameters, schema: schema
                                ).toJSONString()
                                let id = "call_" + UUID().uuidString
                                    .replacingOccurrences(of: "-", with: "").prefix(24)
                                continuation.yield(.toolCall(
                                    Qwen38ToolCall(id: id, name: call.name, argumentsJSON: arguments)))
                            }
                            if !parsed.calls.isEmpty, finish != .length { finish = .toolCalls }
                        }
                        return finish
                    }

                    if isDense {
                        let hfMessages = messages.map(Qwen38Brain.templateMessage)
                        let imageURLs = messages.flatMap(\.imageURLs)
                        let context = Qwen4ExpPromptBuilder.templateContext(
                            thinking: options.enableThinking,
                            reasoningEffort: options.reasoningEffort,
                            tools: tools.isEmpty ? nil : tools)
                        let toolDictionaries = tools.isEmpty ? nil : tools.map(\.toolSpecDictionary)
                        Memory.peakMemory = 0
                        let chunks = Qwen38BrainChunkBuffer()
                        let result = try await runtime.performContext { modelContext in
                            let tokenizer = modelContext.tokenizer
                            // Same rendering with or without images: our own
                            // messages (ordered tool arguments); the processor
                            // only adds image pixels and expands the padding.
                            let tokens: [Int]
                            let image: LMInput.ProcessedImage?
                            if imageURLs.isEmpty {
                                tokens = try tokenizer.applyChatTemplate(
                                    messages: hfMessages, tools: toolDictionaries,
                                    additionalContext: context)
                                image = nil
                            } else {
                                let input = try await modelContext.processor.prepare(
                                    input: UserInput(
                                        prompt: .messages(hfMessages),
                                        images: imageURLs.map(UserInput.Image.url),
                                        processing: .init(resize: options.imageResize),
                                        tools: toolDictionaries, additionalContext: context))
                                tokens = input.text.tokens.asArray(Int.self)
                                image = input.image
                            }
                            let request = Qwen38BrainDenseRequest(
                                tokens: tokens, image: image,
                                visionStartTokenID: tokenizer.convertTokenToId("<|vision_start|>"),
                                maxTokens: options.maxTokens,
                                temperature: options.temperature, topP: options.topP,
                                topK: options.topK, kvBits: profile.kvBits,
                                prefillStepSize: prefillStepSize, stopTokenIDs: stopTokenIDs,
                                conversationStartTokenID: tokenizer.convertTokenToId("<|im_start|>"))
                            return try Qwen38BrainDenseEngine.run(
                                request, model: modelContext.model, tokenizer: tokenizer,
                                conversation: conversation
                            ) { piece in
                                chunks.append(piece)
                                return !Task.isCancelled
                            }
                        }
                        // `perform` runs synchronously; replay its text in order.
                        for piece in chunks.drain() { absorb(piece) }
                        let finishReason = finish(stoppedByLength: result.hitLength)
                        continuation.yield(.done(Qwen38BrainUsage(
                            promptTokens: result.promptTokens,
                            cachedPromptTokens: result.cachedPromptTokens,
                            completionTokens: result.completionTokens,
                            promptTokensPerSecond: result.prefillSeconds > 0
                                ? Double(result.promptTokens) / result.prefillSeconds : 0,
                            prefillSeconds: result.prefillSeconds,
                            tokensPerSecond: result.decodeSeconds > 0
                                ? Double(result.completionTokens) / result.decodeSeconds : 0,
                            timeToFirstToken: result.timeToFirstToken,
                            peakMemoryBytes: Memory.peakMemory,
                            stepMedian: Qwen38Brain.percentile(result.stepDurations, 0.5),
                            stepP90: Qwen38Brain.percentile(result.stepDurations, 0.9),
                            finishReason: finishReason)))
                    } else {
                        let generation = Qwen38GenerationOptions(
                            maxTokens: options.maxTokens, temperature: options.temperature,
                            topP: options.topP, topK: options.topK,
                            enableThinking: options.enableThinking,
                            reasoningEffort: options.reasoningEffort, kvBits: profile.kvBits,
                            tools: tools)
                        let stream = try await runtime.generateStateless(
                            messages: messages, options: generation)
                        for try await event in stream {
                            try Task.checkCancellation()
                            switch event {
                            case .chunk(let chunk):
                                absorb(chunk)
                            case .metrics(let metrics):
                                let finishReason = finish(
                                    stoppedByLength: metrics.stopReason == .length)
                                continuation.yield(.done(Qwen38BrainUsage(
                                    promptTokens: metrics.metrics.promptTokens,
                                    cachedPromptTokens: metrics.cachedPromptTokens,
                                    completionTokens: metrics.metrics.generatedTokens,
                                    promptTokensPerSecond: metrics.metrics.prefillTokensPerSecond,
                                    prefillSeconds: metrics.metrics.prefillTime,
                                    tokensPerSecond: metrics.metrics.generationTokensPerSecond,
                                    timeToFirstToken: metrics.timeToFirstToken,
                                    peakMemoryBytes: metrics.peakMemoryBytes,
                                    stepMedian: nil, stepP90: nil,
                                    finishReason: finishReason)))
                            }
                        }
                    }
                    if profile.clearCacheAfterAnswer { Memory.clearCache() }
                    continuation.finish()
                } catch {
                    if profile.clearCacheAfterAnswer { Memory.clearCache() }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Forgets the cached conversation; the next answer prefills from scratch.
    public func resetConversation() async {
        conversation.reset()
        await runtime.resetConversation()
    }

    public func memoryReport() -> Qwen38BrainMemoryReport {
        let snapshot = Memory.snapshot()
        return Qwen38BrainMemoryReport(
            activeBytes: snapshot.activeMemory, cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory)
    }

    public func unload() async {
        await runtime.unload()
    }
}

/// Text pieces produced inside `ModelContainer.perform` (synchronous), handed
/// back to the async side in order.
final class Qwen38BrainChunkBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pieces: [String] = []

    func append(_ piece: String) {
        lock.lock()
        pieces.append(piece)
        lock.unlock()
    }

    func drain() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let result = pieces
        pieces = []
        return result
    }
}

/// Streams answer text but never lets a `<tool_call>` block through: once
/// the marker starts, the rest is held back for `Qwen38ToolCallParser`.
/// A partial marker at the end of a chunk is held until the next one says
/// whether it really is one.
struct Qwen38ToolCallTextSplitter {
    private static let marker = "<tool_call>"
    private let enabled: Bool
    private var pending = ""
    private var inToolCall = false

    init(enabled: Bool) { self.enabled = enabled }

    mutating func append(_ text: String) -> String {
        guard enabled else { return text }
        guard !inToolCall else { return "" }
        pending += text
        if let range = pending.range(of: Self.marker) {
            inToolCall = true
            let visible = String(pending[..<range.lowerBound])
            pending = ""
            return visible
        }
        // Hold back the longest suffix that could still grow into the marker.
        var keep = 0
        for length in stride(from: min(Self.marker.count - 1, pending.count), to: 0, by: -1)
        where Self.marker.hasPrefix(String(pending.suffix(length))) {
            keep = length
            break
        }
        let visible = String(pending.dropLast(keep))
        pending = String(pending.suffix(keep))
        return visible
    }

    mutating func finish() -> String {
        defer { pending = "" }
        return inToolCall ? "" : pending
    }
}
