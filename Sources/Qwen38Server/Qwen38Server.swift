import Foundation
import Hummingbird
import NIOCore
import Qwen38Core

public enum Qwen38ServerStatus: String, Sendable, Equatable, Codable { case stopped, starting, running, stopping, failed }
public enum Qwen38ServerSessionStatus: String, Sendable, Equatable, Codable { case queued, running, completed, failed }

public struct Qwen38ServerSession: Sendable, Equatable, Codable, Identifiable {
    public let id: UUID
    public let client: String
    public let path: String
    public var model: String
    public var conversationID: String?
    public let startedAt: Date
    public var status: Qwen38ServerSessionStatus
    public var inputDescription: String
    public var promptTokens: Int
    public var generatedTokens: Int
    public var timeToFirstToken: TimeInterval?
    public var tokensPerSecond: Double?
    public var lastToken: String
    public var cacheReused: Bool
    /// P5.2: true when `cacheReused` came from an LRU restore (a different
    /// conversation was live in the resident engine just before this
    /// request) rather than the conversation already being live. GUI:
    /// `sessionRow`'s "Cache" line — restauré / réutilisé / rejoué.
    public var cacheRestored: Bool
    /// P5.2: true when this request had a `conversation_id` with prior
    /// history but still had to fall back to a full stateless replay
    /// (`Qwen38Runtime.generateStateless`) instead of the persistent-cache
    /// path — set directly by the server's `prepareConversation`, since the
    /// engine-level `conversationReplayed` metric only ever reflects the 27B
    /// M1 replay path today (Flash-Next always reports `false` there).
    public var cacheReplayed: Bool
    public var conversationReplayed: Bool
    public var mtp: String
    public var mtpProposed: Int
    public var mtpAccepted: Int
    public var mtpAcceptRate: Double?
    public var error: String?
    /// Set when the session completes or fails (GUI: durée totale).
    public var finishedAt: Date?
    public init(id: UUID = UUID(), client: String, path: String, model: String = "Qwen3.8", conversationID: String? = nil, startedAt: Date = Date(), status: Qwen38ServerSessionStatus = .queued, inputDescription: String = "Texte", promptTokens: Int = 0, generatedTokens: Int = 0, timeToFirstToken: TimeInterval? = nil, tokensPerSecond: Double? = nil, lastToken: String = "", cacheReused: Bool = false, cacheRestored: Bool = false, cacheReplayed: Bool = false, conversationReplayed: Bool = false, mtp: String = "indisponible", mtpProposed: Int = 0, mtpAccepted: Int = 0, mtpAcceptRate: Double? = nil, error: String? = nil) {
        self.id = id; self.client = client; self.path = path; self.model = model; self.conversationID = conversationID; self.startedAt = startedAt; self.status = status; self.inputDescription = inputDescription; self.promptTokens = promptTokens; self.generatedTokens = generatedTokens; self.timeToFirstToken = timeToFirstToken; self.tokensPerSecond = tokensPerSecond; self.lastToken = lastToken; self.cacheReused = cacheReused; self.cacheRestored = cacheRestored; self.cacheReplayed = cacheReplayed; self.conversationReplayed = conversationReplayed; self.mtp = mtp; self.mtpProposed = mtpProposed; self.mtpAccepted = mtpAccepted; self.mtpAcceptRate = mtpAcceptRate; self.error = error
    }
}

public struct Qwen38ServerSnapshot: Sendable, Equatable, Codable {
    public let status: Qwen38ServerStatus
    public let port: Int
    public let url: String
    public let activeSessions: Int
    public let queuedSessions: Int
    public let sessions: [Qwen38ServerSession]
    public let availableModels: [String]
    public let loadedModel: String?
    public let lastError: String?
    /// P5.2: LRU counters for the Flash-Next per-conversation cache
    /// (contrat §5.1.1 — the 27B path keeps its single `ChatSession` cache
    /// and never touches these). `cacheBudgetBytes == 0` means the LRU is
    /// disabled (`--conversation-cache-gb 0`, legacy single-active-cache
    /// behavior).
    public let cacheMisses: Int
    public let cachedConversations: Int
    public let cacheBytes: Int64
    public let cacheBudgetBytes: Int64
    public init(status: Qwen38ServerStatus, port: Int, url: String, activeSessions: Int, queuedSessions: Int, sessions: [Qwen38ServerSession], availableModels: [String] = [], loadedModel: String? = nil, lastError: String? = nil, cacheMisses: Int = 0, cachedConversations: Int = 0, cacheBytes: Int64 = 0, cacheBudgetBytes: Int64 = 0) { self.status = status; self.port = port; self.url = url; self.activeSessions = activeSessions; self.queuedSessions = queuedSessions; self.sessions = sessions; self.availableModels = availableModels; self.loadedModel = loadedModel; self.lastError = lastError; self.cacheMisses = cacheMisses; self.cachedConversations = cachedConversations; self.cacheBytes = cacheBytes; self.cacheBudgetBytes = cacheBudgetBytes }
}

public enum Qwen38ServerError: LocalizedError, Equatable {
    case invalidPort, alreadyRunning, unauthorized, unsupportedImageURL, modelNotLoaded, noModelsAvailable
    case modelNotFound(String)
    case invalidRequest(String)
    public var errorDescription: String? {
        switch self { case .invalidPort: return "Le port du serveur doit être compris entre 1 et 65535."; case .alreadyRunning: return "Le serveur Qwen3.8 est déjà démarré."; case .unauthorized: return "Clé API absente ou invalide."; case .unsupportedImageURL: return "Les images doivent être envoyées en data URL base64 ou en file:// local."; case .modelNotLoaded: return "Chargez un modèle avant de démarrer le serveur ou configurez un catalogue de modèles."; case .noModelsAvailable: return "Aucun modèle Qwen3.8 valide n'a été trouvé dans le catalogue."; case .modelNotFound(let model): return "Modèle indisponible dans le catalogue local : \(model)."; case .invalidRequest(let message): return message }
    }
}

/// Models exposed by the LAN server are discovered below one explicitly
/// configured directory.  Keeping the resolver here prevents a remote client
/// from turning `model` into an arbitrary filesystem path.
public enum Qwen38ModelCatalog {
    public static func discover(in root: URL) -> [String: URL] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else { return [:] }
        var result: [String: URL] = [:]
        for entry in entries {
            guard (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
            let id = entry.lastPathComponent
            guard !id.isEmpty, (try? Qwen38ModelValidator.validate(entry)) != nil else { continue }
            result[id] = entry.standardizedFileURL
        }
        return result
    }

    /// Sum of the `*.safetensors` file sizes below `url`, read from file
    /// attributes only — never opens or parses the weights.
    public static func sizeOnDisk(_ url: URL) -> Int64 {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]) else { return 0 }
        var total: Int64 = 0
        for entry in entries {
            guard (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true,
                  entry.pathExtension == "safetensors" else { continue }
            let size = (try? entry.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            total += Int64(size)
        }
        return total
    }
}

private struct ChatCompletionRequest: Codable, Sendable {
    let model: String?; let messages: [ChatCompletionMessage]; let stream: Bool?; let maxTokens: Int?; let maxCompletionTokens: Int?; let temperature: Float?; let topP: Float?; let presencePenalty: Float?; let frequencyPenalty: Float?; let reasoningEffort: String?; let reasoning: ChatCompletionReasoning?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; let extra: ChatCompletionExtra?
    enum CodingKeys: String, CodingKey { case model, messages, stream, maxTokens = "max_tokens", maxCompletionTokens = "max_completion_tokens", temperature, topP = "top_p", presencePenalty = "presence_penalty", frequencyPenalty = "frequency_penalty", reasoningEffort = "reasoning_effort", reasoning, enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id", extra }

    var effectiveMaxTokens: Int? { maxCompletionTokens ?? maxTokens }
    var effectiveReasoningEffort: String? { reasoningEffort ?? reasoning?.effort ?? extra?.reasoningEffort }
    var effectiveThinking: Bool? { enableThinking ?? extra?.enableThinking }
    var effectiveMTP: Bool? { mtp ?? extra?.mtp ?? true }
    var effectiveMTPEngine: Qwen38MTPEngine { Qwen38MTPEngine(rawValue: (mtpEngine ?? extra?.mtpEngine ?? "local").lowercased()) ?? .local }
    var effectiveMTPDraftTokens: Int { min(max(mtpDraftTokens ?? extra?.mtpDraftTokens ?? 1, 1), 8) }
    var effectiveConversationID: String? { (conversationID ?? extra?.conversationID)?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty }
    /// P5.3: `presence_penalty` wins over `frequency_penalty` (both accepted,
    /// both treated as presence — PLAN.md P5.3). An explicit `0` from either
    /// field disables the penalty even under sampling; only the complete
    /// absence of both falls through to the server's own default (applied
    /// by the caller, since it depends on `temperature`).
    var explicitPresencePenalty: Float? { presencePenalty ?? frequencyPenalty }
    var effectiveRepetitionPenalty: Float { extra?.repetitionPenalty ?? 1.0 }
}
private struct ChatCompletionReasoning: Codable, Sendable { let effort: String? }
private struct ChatCompletionExtra: Codable, Sendable { let reasoningEffort: String?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; let repetitionPenalty: Float?; enum CodingKeys: String, CodingKey { case reasoningEffort = "reasoning_effort", enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id", repetitionPenalty = "repetition_penalty" } }
private struct ChatCompletionMessage: Codable, Sendable { let role: String; let content: ChatCompletionContent? }
private enum ChatCompletionContent: Codable, Sendable {
    case text(String); case parts([ChatCompletionPart])
    init(from decoder: Decoder) throws { if let value = try? decoder.singleValueContainer().decode(String.self) { self = .text(value) } else { self = .parts(try decoder.singleValueContainer().decode([ChatCompletionPart].self)) } }
    func encode(to encoder: Encoder) throws { switch self { case .text(let value): try value.encode(to: encoder); case .parts(let values): try values.encode(to: encoder) } }
}
private struct ChatCompletionPart: Codable, Sendable { let type: String; let text: String?; let imageURL: ChatCompletionImageURL?; enum CodingKeys: String, CodingKey { case type, text, imageURL = "image_url" } }
private struct ChatCompletionImageURL: Codable, Sendable { let url: String }
private struct ChatCompletionChoice: Codable, Sendable { let index: Int; let message: ChatCompletionMessageResponse?; let delta: ChatCompletionDelta?; let finishReason: String?; enum CodingKeys: String, CodingKey { case index, message, delta, finishReason = "finish_reason" } }
private struct ChatCompletionMessageResponse: Codable, Sendable {
    let role: String
    let content: String
    let reasoningContent: String?
    enum CodingKeys: String, CodingKey { case role, content, reasoningContent = "reasoning_content" }
}
private struct ChatCompletionDelta: Codable, Sendable {
    let role: String?
    let content: String?
    let reasoningContent: String?
    enum CodingKeys: String, CodingKey { case role, content, reasoningContent = "reasoning_content" }
}
private struct ChatCompletionResponse: Codable, Sendable { let id: String; let object: String; let created: Int; let model: String; let choices: [ChatCompletionChoice] }
private struct ModelListResponse: Codable, Sendable { let object: String; let data: [ModelDescription] }
private struct ModelDescription: Codable, Sendable { let id: String; let object: String; let ownedBy: String; let loaded: Bool; let family: String?; enum CodingKeys: String, CodingKey { case id, object, ownedBy = "owned_by", loaded, family } }
private struct HealthResponse: Codable, Sendable { let status: String; let modelLoaded: Bool; let model: String?; let queue: String; enum CodingKeys: String, CodingKey { case status, modelLoaded = "model_loaded", model, queue } }
private struct ErrorResponse: Codable, Sendable { let error: ErrorPayload }
private struct ErrorPayload: Codable, Sendable { let message: String; let type: String; let code: String? }

private actor FIFORequestQueue {
    private var occupied = false; private var waiters: [CheckedContinuation<Void, Never>] = []
    var queuedCount: Int { waiters.count }
    func acquire() async { if !occupied { occupied = true; return }; await withCheckedContinuation { waiters.append($0) } }
    func release() { if let next = waiters.first { waiters.removeFirst(); next.resume() } else { occupied = false } }
}

/// Sendable-safe "has the first event arrived yet" signal shared between
/// the stream consumer and the heartbeat ticker in `mergingHeartbeat`.
private actor Qwen38SSEProgressFlag {
    private(set) var hasProgressed = false
    func markProgress() { hasProgressed = true }
}

enum Qwen38SSEHeartbeatItem: Sendable {
    case heartbeat
    case event(Qwen38GenerationEvent)
}

extension Qwen38InferenceServer {
    /// H5.3: Flash-Next's first turn can take ~100 s (layer load) + prefill
    /// before the first real event arrives. This merges `stream` with a
    /// repeating heartbeat tick that only fires while no event has arrived
    /// yet (later per-chunk gaps are well under any realistic idle
    /// timeout), so a caller can turn ticks into an SSE keep-alive comment
    /// and keep proxies/clients from treating the connection as dead.
    ///
    /// The two internal producer tasks only ever touch `continuation`
    /// (Sendable) and `stream` itself — never the caller's writer, which in
    /// `ResponseBody { writer in … }` is an `inout` parameter and therefore
    /// cannot be captured by an escaping/task closure at all. The merged
    /// stream keeps all actual writes on the caller's single task.
    fileprivate static func mergingHeartbeat(
        _ stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>,
        interval: Duration = .seconds(10)
    ) -> AsyncThrowingStream<Qwen38SSEHeartbeatItem, Error> {
        AsyncThrowingStream { continuation in
            let flag = Qwen38SSEProgressFlag()
            let eventTask = Task {
                do {
                    for try await event in stream {
                        await flag.markProgress()
                        continuation.yield(.event(event))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            let heartbeatTask = Task {
                while await !flag.hasProgressed {
                    try? await Task.sleep(for: interval)
                    if Task.isCancelled { return }
                    if await !flag.hasProgressed {
                        continuation.yield(.heartbeat)
                    }
                }
            }
            continuation.onTermination = { _ in
                eventTask.cancel()
                heartbeatTask.cancel()
            }
        }
    }
}

/// OpenAI-compatible LAN transport. One model stays resident and inference is FIFO.
public actor Qwen38InferenceServer {
    public static let defaultPort = 8848
    private let runtime: Qwen38Runtime; private let queue = FIFORequestQueue()
    private var serverTask: Task<Void, Never>?
    private var sessions: [UUID: Qwen38ServerSession] = [:]; private var sessionOrder: [UUID] = []
    private var serverStatus: Qwen38ServerStatus = .stopped; private var serverPort = 8848; private var apiKey: String?; private var lastError: String?
    private var modelsRoot: URL?; private var modelDirectories: [String: URL] = [:]; private var loadedModel: String?
    /// Only one target cache is resident. A conversation id makes that cache
    /// explicit: switching ids resets/replays rather than leaking one client's
    /// history into another request.
    private var activeConversationID: String?
    private var activeConversationModel: String?
    private var activeConversationMessages: [Qwen38ChatMessage] = []
    private var activeConversationOptions: Qwen38GenerationOptions?
    /// P5.2: LRU of exported Flash-Next conversation states that are *not*
    /// currently live in the resident engine (the live one stays only in
    /// `activeConversation*` above until a different id displaces it —
    /// exporting on every turn would be wasted `KVCache.copy()` work).
    /// qwen4Exp only; 27B never populates this (contrat §5.1.1).
    private struct CachedConversation {
        let model: String
        let options: Qwen38GenerationOptions
        let ledger: [Qwen38ChatMessage]
        let state: any Qwen38FlashConversationStateProtocol
    }
    private var conversationCache: [String: CachedConversation] = [:]
    /// Least-recently-used at the front, most-recently-used at the back.
    private var conversationCacheOrder: [String] = []
    private var conversationCacheBudgetBytes: Int64 = 12 * 1024 * 1024 * 1024
    private var cacheMissCount = 0
    public init(runtime: Qwen38Runtime) { self.runtime = runtime }

    public func start(port: Int = 8848, apiKey: String? = nil, modelsDirectory: URL? = nil, conversationCacheGB: Double = 12) async throws {
        guard (1 ... 65_535).contains(port) else { throw Qwen38ServerError.invalidPort }
        conversationCacheBudgetBytes = conversationCacheGB > 0
            ? Int64(conversationCacheGB * 1024 * 1024 * 1024) : 0
        let currentDirectory = await runtime.loadedDirectory
        let root = modelsDirectory ?? currentDirectory?.deletingLastPathComponent()
        guard let root else { throw Qwen38ServerError.modelNotLoaded }
        modelsRoot = root.standardizedFileURL
        refreshModelCatalog()
        guard !modelDirectories.isEmpty else { throw Qwen38ServerError.noModelsAvailable }
        if let currentDirectory, let id = modelDirectories.first(where: { sameDirectory($0.value, currentDirectory) })?.key { loadedModel = id }
        guard serverStatus == .stopped || serverStatus == .failed else { throw Qwen38ServerError.alreadyRunning }
        serverPort = port; self.apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty; lastError = nil; serverStatus = .starting
        let router = Router()
        router.get("healthz") { [self] _, _ in await self.healthResponse() }
        router.get("v1/models") { [self] request, _ in try await self.modelsResponse(request: request) }
        router.get("metrics") { [self] _, _ in await self.metricsResponse() }
        router.post("v1/chat/completions") { [self] request, _ in try await self.chatCompletionsResponse(request: request) }
        let application = Application(router: router, configuration: .init(address: .hostname("0.0.0.0", port: port), serverName: "Qwen38Inference"))
        serverTask = Task { [weak self, application] in
            do { try await application.run(); await self?.serverDidStop() }
            catch is CancellationError { await self?.serverDidStop() }
            catch { await self?.serverDidFail(String(describing: error)) }
        }
        await Task.yield(); serverStatus = .running
    }

    public func stop() async {
        guard serverStatus != .stopped else { return }; serverStatus = .stopping; serverTask?.cancel(); if let serverTask { await serverTask.value }; self.serverTask = nil; serverStatus = .stopped
    }
    public func snapshot() async -> Qwen38ServerSnapshot { refreshModelCatalog(); let current = sessionOrder.compactMap { sessions[$0] }; return .init(status: serverStatus, port: serverPort, url: "http://127.0.0.1:\(serverPort)", activeSessions: current.filter { $0.status == .queued || $0.status == .running }.count, queuedSessions: await queue.queuedCount, sessions: current, availableModels: modelDirectories.keys.sorted(), loadedModel: loadedModel, lastError: lastError, cacheMisses: cacheMissCount, cachedConversations: conversationCache.count, cacheBytes: totalCacheBytes(), cacheBudgetBytes: conversationCacheBudgetBytes) }
    private func serverDidStop() { if serverStatus != .stopping { serverStatus = .stopped } }
    private func serverDidFail(_ error: String) { lastError = error; serverStatus = .failed }

    private func healthResponse() async -> Response { Self.jsonResponse(HealthResponse(status: serverStatus.rawValue, modelLoaded: await runtime.isLoaded, model: loadedModel, queue: String(sessions.values.filter { $0.status == .queued }.count))) }
    private func modelsResponse(request: Request) async throws -> Response { try authorize(request); refreshModelCatalog(); let current = loadedModel; let models = modelDirectories.keys.sorted().map { id -> ModelDescription in let family = modelDirectories[id].flatMap { try? Qwen38ModelValidator.readInfo(from: $0) }?.family; return ModelDescription(id: id, object: "model", ownedBy: "local", loaded: id == current, family: family?.rawValue) }; return Self.jsonResponse(ModelListResponse(object: "list", data: models)) }
    private func metricsResponse() async -> Response { let current = await snapshot(); return Self.jsonResponse(current) }

    private func chatCompletionsResponse(request: Request) async throws -> Response {
        try authorize(request); var request = request; let buffer = try await request.collectBody(upTo: 64 * 1024 * 1024)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else { throw Qwen38ServerError.invalidRequest("Le corps JSON est vide.") }
        let input: ChatCompletionRequest
        do { input = try JSONDecoder().decode(ChatCompletionRequest.self, from: data) } catch { throw Qwen38ServerError.invalidRequest("Requête chat invalide : \(error.localizedDescription)") }
        guard !input.messages.isEmpty else { throw Qwen38ServerError.invalidRequest("La requête doit contenir au moins un message.") }
        guard input.messages.last?.role == "user" else { throw Qwen38ServerError.invalidRequest("Le dernier message doit avoir le rôle user.") }
        let requestedModel = input.model?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let id = UUID(); sessions[id] = .init(id: id, client: "LAN", path: "/v1/chat/completions", model: requestedModel ?? loadedModel ?? "default", conversationID: input.effectiveConversationID); sessionOrder.append(id); trimSessions(); await queue.acquire(); updateSession(id) { $0.status = .running }; defer { Task { await queue.release() } }
        do {
            let selectedModel = try await ensureModelLoaded(requestedModel)
            updateSession(id) { $0.model = selectedModel }
            let prepared = try prepare(input.messages)
            let temperature = input.temperature ?? 0
            // P5.3: server default when a sampling request (temperature > 0)
            // omits both `presence_penalty` and `frequency_penalty` — the
            // instruct preset's `presence 1.5` (PLAN.md §2.1). An explicit
            // `presence_penalty: 0` (or `frequency_penalty: 0`) disables it;
            // greedy requests (temperature 0) never get this default since
            // the generator ignores penalties there regardless.
            let presencePenalty = input.explicitPresencePenalty ?? (temperature > 0 ? 1.5 : 0)
            let options = Qwen38GenerationOptions(maxTokens: min(max(input.effectiveMaxTokens ?? 256, 1), 131_072), temperature: temperature, topP: input.topP ?? 0.95, enableThinking: input.effectiveThinking ?? (input.effectiveReasoningEffort != nil), reasoningEffort: input.effectiveReasoningEffort ?? "low", mtp: .init(enabled: input.effectiveMTP ?? true, draftDepth: .fixed(input.effectiveMTPDraftTokens), engine: input.effectiveMTPEngine), presencePenalty: presencePenalty, repetitionPenalty: input.effectiveRepetitionPenalty)
            let conversationID = input.effectiveConversationID
            let (usePersistentCache, cacheRestored) = try await prepareConversation(
                id: conversationID,
                model: selectedModel,
                messages: prepared.messages,
                options: options)
            // Ground truth for the GUI's "Cache" tri-state (P5.2): a request
            // that named a conversation with prior turns but still fell back
            // to a full stateless replay. Computed here, not from engine
            // metrics — see `cacheReplayed`'s doc comment.
            let cacheReplayed = !usePersistentCache && conversationID != nil && prepared.messages.count > 1
            updateSession(id) { $0.cacheRestored = cacheRestored; $0.cacheReplayed = cacheReplayed }
            let stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>
            if usePersistentCache, let last = prepared.messages.last {
                let systemPrompt = prepared.messages.first(where: { $0.role == .system })?.content
                stream = try await runtime.generate(
                    prompt: last.content,
                    systemPrompt: systemPrompt,
                    imageURLs: last.imageURLs,
                    options: options)
            } else {
                stream = try await runtime.generateStateless(messages: prepared.messages, options: options)
            }
            for url in prepared.temporaryFiles { try? FileManager.default.removeItem(at: url) }
            // Must mirror `options.enableThinking` exactly: the parser assumes the
            // prompt ends inside `<think>` only when thinking was rendered.
            let thinkingIsPrimed = options.enableThinking
            // Always pass the real `conversationID` (not gated on
            // `usePersistentCache`): `rememberConversation` itself decides
            // whether to append to the already-active conversation or
            // establish a brand new one from a replay's resulting state (the
            // "nouvel état après la réponse" half of P5.2 — see its doc
            // comment).
            if input.stream == true {
                return try await makeStreamingResponse(
                    stream: stream, sessionID: id, model: selectedModel,
                    primedInside: thinkingIsPrimed,
                    conversationID: conversationID,
                    requestMessages: prepared.messages, options: options)
            }
            return try await makeJSONResponse(
                stream: stream, sessionID: id, model: selectedModel,
                primedInside: thinkingIsPrimed,
                conversationID: conversationID,
                requestMessages: prepared.messages, options: options)
        } catch {
            if input.effectiveConversationID != nil {
                clearActiveConversation()
                await runtime.resetConversation()
            }
            updateSession(id) { $0.status = .failed; $0.error = error.localizedDescription; $0.finishedAt = Date() }
            return Self.errorResponse(Self.status(for: error), message: error.localizedDescription)
        }
    }

    /// P5.2: returns whether this request can use the resident engine's
    /// persistent cache, and — when it can — whether that meant restoring a
    /// different conversation's exported state into the (single, §5.1.1)
    /// resident engine rather than continuing the conversation that was
    /// already live.
    func prepareConversation(
        id: String?,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> (usePersistentCache: Bool, cacheRestored: Bool) {
        guard let id else {
            clearActiveConversation()
            return (false, false)
        }
        // The LRU only ever manages Flash-Next conversations (contrat
        // §5.1.1 / PLAN.md P5 scope) — the 27B path, or an operator who set
        // `--conversation-cache-gb 0`, keeps the original single-active-
        // conversation behavior with no export/restore machinery at all.
        guard familyOf(model) == .qwen4Exp, conversationCacheBudgetBytes > 0 else {
            return (try await legacyPrepareConversation(id: id, model: model, messages: messages, options: options), false)
        }

        if activeConversationID == id, activeConversationModel == model,
           activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true,
           messages.count == activeConversationMessages.count + 1,
           Array(messages.dropLast()) == activeConversationMessages {
            return (true, false)
        }

        // A different conversation is about to become live: export the
        // current one into the LRU first (a no-op if none was active) so
        // switching back to it later can restore instead of replaying.
        if let previousID = activeConversationID, let previousModel = activeConversationModel,
           let previousOptions = activeConversationOptions {
            await storeActiveConversationIntoLRU(id: previousID, model: previousModel, options: previousOptions)
        }
        clearActiveConversation()

        if let cached = conversationCache[id], cached.model == model,
           cacheOptionsCompatible(cached.options, options),
           messages.count == cached.ledger.count + 1,
           Array(messages.dropLast()) == cached.ledger {
            await runtime.restoreFlashConversationState(cached.state)
            conversationCache.removeValue(forKey: id)
            conversationCacheOrder.removeAll { $0 == id }
            activeConversationID = id
            activeConversationModel = model
            activeConversationMessages = cached.ledger
            activeConversationOptions = options
            return (true, true)
        }

        // A new or non-contiguous session is deliberately cold. A short
        // system/user prompt can start a persistent cache directly; longer
        // histories use the stateless replay path and are not advertised as
        // cached because reconstructing assistant hidden states is impossible
        // without rerunning them.
        cacheMissCount += 1
        await runtime.resetConversation()
        let userCount = messages.filter { $0.role == .user }.count
        guard messages.last?.role == .user,
              userCount == 1,
              messages.allSatisfy({ $0.role == .system || $0.role == .user }) else {
            return (false, false)
        }
        activeConversationID = id
        activeConversationModel = model
        activeConversationOptions = options
        return (true, false)
    }

    /// Legacy behavior (pre-P5.2 / LRU disabled / non-Flash-Next family):
    /// exactly one conversation's cache can ever be live; switching ids
    /// always resets and cold-starts, never restores.
    private func legacyPrepareConversation(
        id: String,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> Bool {
        let isContinuation = activeConversationID == id
            && activeConversationModel == model
            && activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true
            && messages.count == activeConversationMessages.count + 1
            && Array(messages.dropLast()) == activeConversationMessages
        if isContinuation {
            return true
        }

        await runtime.resetConversation()
        clearActiveConversation()
        let userCount = messages.filter { $0.role == .user }.count
        guard messages.last?.role == .user,
              userCount == 1,
              messages.allSatisfy({ $0.role == .system || $0.role == .user }) else {
            return false
        }
        activeConversationID = id
        activeConversationModel = model
        activeConversationOptions = options
        return true
    }

    private func familyOf(_ model: String) -> Qwen38ModelFamily? {
        guard let directory = modelDirectories[model] else { return nil }
        return (try? Qwen38ModelValidator.readInfo(from: directory))?.family
    }

    private func cacheOptionsCompatible(_ a: Qwen38GenerationOptions, _ b: Qwen38GenerationOptions) -> Bool {
        a.temperature == b.temperature
            && a.topP == b.topP
            && a.topK == b.topK
            && a.enableThinking == b.enableThinking
            && a.reasoningEffort == b.reasoningEffort
            && a.kvBits == b.kvBits
            && a.mtp == b.mtp
            && a.presencePenalty == b.presencePenalty
            && a.repetitionPenalty == b.repetitionPenalty
    }

    private func clearActiveConversation() {
        activeConversationID = nil
        activeConversationModel = nil
        activeConversationMessages = []
        activeConversationOptions = nil
    }

    /// P5.2: exports the currently-live conversation's engine state
    /// (`Qwen38Runtime.exportFlashConversationState`, `KVCache.copy()`
    /// under the hood — P5.1) into the LRU, then evicts the oldest entries
    /// until the budget is met again.
    private func storeActiveConversationIntoLRU(
        id: String, model: String, options: Qwen38GenerationOptions
    ) async {
        guard !activeConversationMessages.isEmpty,
              let exported = await runtime.exportFlashConversationState(ledger: activeConversationMessages)
        else { return }
        conversationCache[id] = CachedConversation(
            model: model, options: options, ledger: activeConversationMessages, state: exported)
        conversationCacheOrder.removeAll { $0 == id }
        conversationCacheOrder.append(id)
        await evictIfNeeded()
    }

    private func totalCacheBytes() -> Int64 {
        conversationCache.values.reduce(Int64(0)) { $0 + Int64($1.state.byteCount) }
    }

    /// Drops the least-recently-used cached conversations until the total
    /// exported byte count is back under budget. `MLXArray`s referenced only
    /// by the evicted `CachedConversation` are released by ARC when the
    /// dictionary entry is removed; `Memory.clearCache()` then returns that
    /// freed device memory to the system (PLAN.md P5.2 criterion, verified
    /// in a gated test against `Memory.activeMemory`).
    private func evictIfNeeded() async {
        guard conversationCacheBudgetBytes > 0 else { return }
        var total = totalCacheBytes()
        var evictedAny = false
        while total > conversationCacheBudgetBytes, !conversationCacheOrder.isEmpty {
            let oldest = conversationCacheOrder.removeFirst()
            if let removed = conversationCache.removeValue(forKey: oldest) {
                total -= Int64(removed.state.byteCount)
                evictedAny = true
            }
        }
        if evictedAny {
            await runtime.clearMLXCache()
        }
    }

    func rememberConversation(
        id: String,
        model: String,
        requestMessages: [Qwen38ChatMessage],
        assistantContent: String,
        options: Qwen38GenerationOptions
    ) {
        if activeConversationID == id, activeConversationModel == model {
            activeConversationMessages = requestMessages + [
                Qwen38ChatMessage(role: .assistant, content: assistantContent)
            ]
            return
        }
        // P5.2, missed in the first pass: "sinon rejeu et **nouvel état après
        // la réponse**" (PLAN.md). A stateless replay (`Qwen38Runtime.
        // generateStateless` → `generateFromMessages`) still leaves the
        // resident Flash-Next engine's live state matching this exact
        // history when it finishes — nothing resets it afterward. Registering
        // it here as the new active conversation means the *next* turn for
        // this id can restore/continue instead of replaying again. Real-world
        // impact found in P5.6's dialogue run: an agent whose very first
        // message already carries a synthetic assistant turn (so the
        // cold-start gate in `prepareConversation` never accepts it) was
        // permanently stuck on stateless replay for the whole conversation
        // without this. Only for the LRU-managed family with a non-zero
        // budget; 27B and `--conversation-cache-gb 0` keep the untouched
        // legacy behavior (no bookkeeping at all).
        guard activeConversationID == nil,
              familyOf(model) == .qwen4Exp,
              conversationCacheBudgetBytes > 0 else { return }
        activeConversationID = id
        activeConversationModel = model
        activeConversationOptions = options
        activeConversationMessages = requestMessages + [
            Qwen38ChatMessage(role: .assistant, content: assistantContent)
        ]
    }

    private func ensureModelLoaded(_ requestedModel: String?) async throws -> String {
        refreshModelCatalog()
        let currentDirectory = await runtime.loadedDirectory
        let selection: (String, URL)
        if let requestedModel {
            guard let directory = modelDirectories[requestedModel] else { throw Qwen38ServerError.modelNotFound(requestedModel) }
            selection = (requestedModel, directory)
        } else if let currentDirectory, let current = modelDirectories.first(where: { sameDirectory($0.value, currentDirectory) }) {
            selection = (current.key, current.value)
        } else if let only = modelDirectories.first, modelDirectories.count == 1 {
            selection = (only.key, only.value)
        } else {
            throw Qwen38ServerError.invalidRequest("Le champ model est obligatoire lorsque plusieurs modèles sont disponibles.")
        }
        if let currentDirectory, sameDirectory(currentDirectory, selection.1) {
            loadedModel = selection.0
            return selection.0
        }
        await runtime.unload()
        clearActiveConversation()
        // A different resident model invalidates every exported state: they
        // reference the previous decoder's own caches (§5.1.1, one model
        // resident at a time).
        conversationCache.removeAll()
        conversationCacheOrder.removeAll()
        try await runtime.load(from: selection.1, preloadMTP: true)
        loadedModel = selection.0
        return selection.0
    }

    private func refreshModelCatalog() {
        guard let modelsRoot else { return }
        modelDirectories = Qwen38ModelCatalog.discover(in: modelsRoot)
    }

    private func sameDirectory(_ lhs: URL, _ rhs: URL) -> Bool {
        lhs.standardizedFileURL.resolvingSymlinksInPath() == rhs.standardizedFileURL.resolvingSymlinksInPath()
    }

    private struct PreparedMessages: Sendable { let messages: [Qwen38ChatMessage]; let temporaryFiles: [URL] }
    private func prepare(_ messages: [ChatCompletionMessage]) throws -> PreparedMessages {
        var result = [Qwen38ChatMessage](), temporaryFiles = [URL]()
        for message in messages {
            var text = "", images = [URL]()
            switch message.content { case .text(let value): text = value; case .parts(let parts): for part in parts { if part.type == "text" { text += part.text ?? "" }; if part.type == "image_url", let value = part.imageURL?.url { let materialized = try materializeImage(value); images.append(materialized.url); if materialized.isTemporary { temporaryFiles.append(materialized.url) } } }; case .none: break }
            let role: Qwen38ChatMessage.Role; switch message.role { case "system": role = .system; case "assistant": role = .assistant; default: role = .user }
            result.append(.init(role: role, content: text, imageURLs: images))
        }
        return .init(messages: result, temporaryFiles: temporaryFiles)
    }
    private func materializeImage(_ value: String) throws -> (url: URL, isTemporary: Bool) {
        if value.hasPrefix("file://"), let url = URL(string: value) { return (url, false) }
        guard value.hasPrefix("data:"), let comma = value.firstIndex(of: ",") else { throw Qwen38ServerError.unsupportedImageURL }
        let metadata = String(value[..<comma]); guard metadata.contains(";base64") else { throw Qwen38ServerError.unsupportedImageURL }
        guard let data = Data(base64Encoded: String(value[value.index(after: comma)...])) else { throw Qwen38ServerError.invalidRequest("Image base64 invalide.") }
        let ext = metadata.split(separator: "/").last.map(String.init)?.split(separator: ";").first.map(String.init) ?? "bin"; let url = FileManager.default.temporaryDirectory.appendingPathComponent("qwen38-server-\(UUID().uuidString).\(ext)" ); try data.write(to: url, options: .atomic); return (url, true)
    }

    private func makeJSONResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, conversationID: String?, requestMessages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) async throws -> Response {
        var text = "", metrics: Qwen38RunMetrics?
        var parser = Qwen38ThinkingStreamParser(primedInside: primedInside)
        var reasoning = ""
        for try await event in stream {
            switch event {
            case .chunk(let chunk):
                let output = parser.append(chunk)
                reasoning += output.reasoning
                text += output.content
                updateSession(sessionID) { $0.generatedTokens += 1 }
            case .metrics(let value):
                let tail = parser.finish()
                reasoning += tail.reasoning
                text += tail.content
                metrics = value
                completeSession(sessionID, metrics: value)
            }
        }
        if let conversationID {
            rememberConversation(id: conversationID, model: model, requestMessages: requestMessages, assistantContent: text, options: options)
        }
        return Self.jsonResponse(ChatCompletionResponse(id: "chatcmpl-\(sessionID.uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970), model: model, choices: [.init(index: 0, message: .init(role: "assistant", content: text, reasoningContent: reasoning.nilIfEmpty), delta: nil, finishReason: Self.finishReason(metrics?.stopReason))]))
    }
    private func makeStreamingResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, conversationID: String?, requestMessages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) async throws -> Response {
        let body = ResponseBody { writer in
            func writeDelta(content: String? = nil, reasoning: String? = nil, finishReason: String? = nil) async throws {
                let value = ChatCompletionResponse(
                    id: "chatcmpl-\(sessionID.uuidString)", object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970), model: model,
                    choices: [.init(index: 0, message: nil, delta: .init(role: nil, content: content, reasoningContent: reasoning), finishReason: finishReason)])
                let payload = try JSONEncoder().encode(value)
                var line = ByteBuffer(string: "data: ")
                line.writeBytes(payload)
                line.writeString("\n\n")
                try await writer.write(line)
            }

            var parser = Qwen38ThinkingStreamParser(primedInside: primedInside)
            var responseContent = ""
            do {
                for try await item in Self.mergingHeartbeat(stream) {
                    switch item {
                    case .heartbeat:
                        try await writer.write(ByteBuffer(string: ": loading\n\n"))
                    case .event(.chunk(let chunk)):
                        await self.updateSessionAsync(sessionID, chunk: chunk)
                        let output = parser.append(chunk)
                        responseContent += output.content
                        if !output.reasoning.isEmpty { try await writeDelta(reasoning: output.reasoning) }
                        if !output.content.isEmpty { try await writeDelta(content: output.content) }
                    case .event(.metrics(let metrics)):
                        await self.completeSessionAsync(sessionID, metrics: metrics)
                        let tail = parser.finish()
                        responseContent += tail.content
                        if !tail.reasoning.isEmpty { try await writeDelta(reasoning: tail.reasoning) }
                        if !tail.content.isEmpty { try await writeDelta(content: tail.content) }
                        try await writeDelta(finishReason: Self.finishReason(metrics.stopReason))
                        if let conversationID {
                            await self.rememberConversation(
                                id: conversationID, model: model,
                                requestMessages: requestMessages,
                                assistantContent: responseContent, options: options)
                        }
                    }
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                try await writer.finish(nil)
            } catch {
                await self.failSessionAsync(sessionID, error: error.localizedDescription)
                throw error
            }
        }
        var headers = HTTPFields(); headers[.contentType] = "text/event-stream; charset=utf-8"; headers[.cacheControl] = "no-cache"; headers[.connection] = "keep-alive"; return .init(status: .ok, headers: headers, body: body)
    }
    private func authorize(_ request: Request) throws { guard let apiKey, !apiKey.isEmpty else { return }; guard request.headers[.authorization] == "Bearer \(apiKey)" else { throw Qwen38ServerError.unauthorized } }
    private func updateSession(_ id: UUID, _ body: (inout Qwen38ServerSession) -> Void) { guard var session = sessions[id] else { return }; body(&session); sessions[id] = session }
    private func updateSessionAsync(_ id: UUID, chunk: String) { updateSession(id) { $0.generatedTokens += 1; $0.lastToken = String(chunk.suffix(48)) } }
    private func completeSession(_ id: UUID, metrics: Qwen38RunMetrics) { updateSession(id) { $0.status = .completed; $0.finishedAt = Date(); $0.promptTokens = metrics.metrics.promptTokens; $0.generatedTokens = metrics.metrics.generatedTokens; $0.timeToFirstToken = metrics.timeToFirstToken; $0.tokensPerSecond = metrics.metrics.generationTokensPerSecond; $0.inputDescription = metrics.inputDescription; $0.cacheReused = metrics.cacheReused; $0.conversationReplayed = metrics.conversationReplayed; $0.mtp = Self.mtpLabel(metrics.mtpStatus); $0.mtpProposed = metrics.mtpStatus.proposedTokens; $0.mtpAccepted = metrics.mtpStatus.acceptedTokens; $0.mtpAcceptRate = metrics.mtpStatus.acceptanceRate } }
    private func completeSessionAsync(_ id: UUID, metrics: Qwen38RunMetrics) { completeSession(id, metrics: metrics) }
    private func failSessionAsync(_ id: UUID, error: String) { updateSession(id) { $0.status = .failed; $0.error = error; $0.finishedAt = Date() } }
    private func trimSessions() { while sessionOrder.count > 32 { sessions.removeValue(forKey: sessionOrder.removeFirst()) } }
    private static func finishReason(_ reason: Any?) -> String { guard let reason else { return "stop" }; return String(describing: reason).lowercased().contains("length") ? "length" : "stop" }
    private static func mtpLabel(_ status: Qwen38MTPRunStatus) -> String { switch status.availability { case .active: return "actif"; case .unavailable: return "indisponible"; case .fallback(let reason): return "fallback: \(reason)" } }
    private static func jsonResponse<T: Encodable>(_ value: T) -> Response { let data = (try? JSONEncoder().encode(value)) ?? Data(); var buffer = ByteBufferAllocator().buffer(capacity: data.count); buffer.writeBytes(data); var headers = HTTPFields(); headers[.contentType] = "application/json; charset=utf-8"; return .init(status: .ok, headers: headers, body: .init(byteBuffer: buffer)) }
    private static func errorResponse(_ status: HTTPResponse.Status, message: String) -> Response {
        var response = jsonResponse(ErrorResponse(error: .init(message: message, type: status.code >= 500 ? "server_error" : "invalid_request_error", code: nil)))
        response.status = status
        return response
    }

    /// LAN test 2026-09-09 (T8): a `Qwen38ServerError` escaping the handler used
    /// to surface as an empty HTTP 500. Map it to an OpenAI-style JSON error
    /// with a meaningful status instead.
    private static func status(for error: any Error) -> HTTPResponse.Status {
        guard let serverError = error as? Qwen38ServerError else { return .internalServerError }
        switch serverError {
        case .modelNotFound: return .notFound
        case .unauthorized: return .unauthorized
        case .invalidRequest, .unsupportedImageURL, .invalidPort: return .badRequest
        case .modelNotLoaded, .noModelsAvailable: return .serviceUnavailable
        case .alreadyRunning: return .conflict
        }
    }
}

private extension String { var nilIfEmpty: String? { isEmpty ? nil : self } }
