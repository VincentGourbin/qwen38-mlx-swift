import CoreGraphics
import Foundation
import MLX
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
    /// Dense hybrid family (Qwen 3.5 / Bonsai 2): answered by the runtime's
    /// reusable-conversation engine. Flash-Next keeps the runtime's own path.
    private let isDense: Bool

    private init(
        modelDirectory: URL, profile: Qwen38BrainProfile, runtime: Qwen38Runtime, isDense: Bool
    ) {
        self.modelDirectory = modelDirectory
        self.profile = profile
        self.runtime = runtime
        self.isDense = isDense
    }

    /// Prefill chunk of the dense engine, in tokens (starts at the profile's).
    public var prefillStepSize: Int {
        get async { await runtime.densePrefillStepSize }
    }

    static func percentile(_ values: [Double], _ q: Double) -> Double? {
        // The first interval includes the queueing of the first step: skip it.
        let sorted = values.dropFirst().sorted()
        guard !sorted.isEmpty else { return nil }
        return sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * q))]
    }

    public func setPrefillStepSize(_ value: Int) async {
        await runtime.setDensePrefillStepSize(value)
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
        await runtime.setDensePrefillStepSize(profile.prefillStepSize)
        return Qwen38Brain(
            modelDirectory: modelDirectory, profile: profile, runtime: runtime,
            // `QWEN38_BRAIN_ENGINE=runtime` forces the runtime's historical
            // path on the dense family too — to check the engine against it.
            isDense: info.family == .qwen35
                && ProcessInfo.processInfo.environment["QWEN38_BRAIN_ENGINE"] != "runtime")
    }

    public func respond(
        to messages: [Qwen38ChatMessage], tools: [Qwen38ToolSpec] = [],
        options: Qwen38BrainOptions = .init()
    ) -> AsyncThrowingStream<Qwen38BrainEvent, Error> {
        let runtime = runtime
        let profile = profile
        let isDense = isDense
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
                    var generation = Qwen38GenerationOptions(
                        maxTokens: options.maxTokens, temperature: options.temperature,
                        topP: options.topP, topK: options.topK,
                        enableThinking: options.enableThinking,
                        reasoningEffort: options.reasoningEffort, kvBits: profile.kvBits,
                        tools: tools)
                    generation.quantizedKVStart = profile.quantizedKVStart
                    let stream = isDense
                        ? try await runtime.generateDenseConversation(
                            messages: messages, options: generation, imageResize: options.imageResize)
                        : try await runtime.generateStateless(messages: messages, options: generation)

                    var parser = Qwen38ThinkingStreamParser(primedInside: options.enableThinking)
                    var splitter = Qwen38ToolCallTextSplitter(enabled: !tools.isEmpty)
                    var fullContent = ""
                    for try await event in stream {
                        try Task.checkCancellation()
                        switch event {
                        case .chunk(let chunk):
                            let output = parser.append(chunk)
                            if !output.reasoning.isEmpty { continuation.yield(.reasoning(output.reasoning)) }
                            fullContent += output.content
                            let visible = splitter.append(output.content)
                            if !visible.isEmpty { continuation.yield(.text(visible)) }
                        case .metrics(let metrics):
                            let tail = parser.finish()
                            if !tail.reasoning.isEmpty { continuation.yield(.reasoning(tail.reasoning)) }
                            fullContent += tail.content
                            let visibleTail = splitter.append(tail.content) + splitter.finish()
                            if !visibleTail.isEmpty { continuation.yield(.text(visibleTail)) }
                            var finish: Qwen38BrainUsage.FinishReason =
                                metrics.stopReason == .length ? .length : .stop
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
                            continuation.yield(.done(Qwen38BrainUsage(
                                promptTokens: metrics.metrics.promptTokens,
                                cachedPromptTokens: metrics.cachedPromptTokens,
                                completionTokens: metrics.metrics.generatedTokens,
                                promptTokensPerSecond: metrics.metrics.prefillTokensPerSecond,
                                prefillSeconds: metrics.metrics.prefillTime,
                                tokensPerSecond: metrics.metrics.generationTokensPerSecond,
                                timeToFirstToken: metrics.timeToFirstToken,
                                peakMemoryBytes: metrics.peakMemoryBytes,
                                stepMedian: Qwen38Brain.percentile(metrics.decodeStepDurations, 0.5),
                                stepP90: Qwen38Brain.percentile(metrics.decodeStepDurations, 0.9),
                                finishReason: finish)))
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
