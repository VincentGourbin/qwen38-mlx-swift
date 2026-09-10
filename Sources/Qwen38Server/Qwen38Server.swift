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
    public var conversationReplayed: Bool
    public var mtp: String
    public var mtpProposed: Int
    public var mtpAccepted: Int
    public var mtpAcceptRate: Double?
    public var error: String?
    /// Set when the session completes or fails (GUI: durée totale).
    public var finishedAt: Date?
    public init(id: UUID = UUID(), client: String, path: String, model: String = "Qwen3.8", conversationID: String? = nil, startedAt: Date = Date(), status: Qwen38ServerSessionStatus = .queued, inputDescription: String = "Texte", promptTokens: Int = 0, generatedTokens: Int = 0, timeToFirstToken: TimeInterval? = nil, tokensPerSecond: Double? = nil, lastToken: String = "", cacheReused: Bool = false, conversationReplayed: Bool = false, mtp: String = "indisponible", mtpProposed: Int = 0, mtpAccepted: Int = 0, mtpAcceptRate: Double? = nil, error: String? = nil) {
        self.id = id; self.client = client; self.path = path; self.model = model; self.conversationID = conversationID; self.startedAt = startedAt; self.status = status; self.inputDescription = inputDescription; self.promptTokens = promptTokens; self.generatedTokens = generatedTokens; self.timeToFirstToken = timeToFirstToken; self.tokensPerSecond = tokensPerSecond; self.lastToken = lastToken; self.cacheReused = cacheReused; self.conversationReplayed = conversationReplayed; self.mtp = mtp; self.mtpProposed = mtpProposed; self.mtpAccepted = mtpAccepted; self.mtpAcceptRate = mtpAcceptRate; self.error = error
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
    public init(status: Qwen38ServerStatus, port: Int, url: String, activeSessions: Int, queuedSessions: Int, sessions: [Qwen38ServerSession], availableModels: [String] = [], loadedModel: String? = nil, lastError: String? = nil) { self.status = status; self.port = port; self.url = url; self.activeSessions = activeSessions; self.queuedSessions = queuedSessions; self.sessions = sessions; self.availableModels = availableModels; self.loadedModel = loadedModel; self.lastError = lastError }
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
    let model: String?; let messages: [ChatCompletionMessage]; let stream: Bool?; let maxTokens: Int?; let maxCompletionTokens: Int?; let temperature: Float?; let topP: Float?; let reasoningEffort: String?; let reasoning: ChatCompletionReasoning?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; let extra: ChatCompletionExtra?
    enum CodingKeys: String, CodingKey { case model, messages, stream, maxTokens = "max_tokens", maxCompletionTokens = "max_completion_tokens", temperature, topP = "top_p", reasoningEffort = "reasoning_effort", reasoning, enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id", extra }

    var effectiveMaxTokens: Int? { maxCompletionTokens ?? maxTokens }
    var effectiveReasoningEffort: String? { reasoningEffort ?? reasoning?.effort ?? extra?.reasoningEffort }
    var effectiveThinking: Bool? { enableThinking ?? extra?.enableThinking }
    var effectiveMTP: Bool? { mtp ?? extra?.mtp ?? true }
    var effectiveMTPEngine: Qwen38MTPEngine { Qwen38MTPEngine(rawValue: (mtpEngine ?? extra?.mtpEngine ?? "local").lowercased()) ?? .local }
    var effectiveMTPDraftTokens: Int { min(max(mtpDraftTokens ?? extra?.mtpDraftTokens ?? 1, 1), 8) }
    var effectiveConversationID: String? { (conversationID ?? extra?.conversationID)?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty }
}
private struct ChatCompletionReasoning: Codable, Sendable { let effort: String? }
private struct ChatCompletionExtra: Codable, Sendable { let reasoningEffort: String?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; enum CodingKeys: String, CodingKey { case reasoningEffort = "reasoning_effort", enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id" } }
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
    public init(runtime: Qwen38Runtime) { self.runtime = runtime }

    public func start(port: Int = 8848, apiKey: String? = nil, modelsDirectory: URL? = nil) async throws {
        guard (1 ... 65_535).contains(port) else { throw Qwen38ServerError.invalidPort }
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
    public func snapshot() async -> Qwen38ServerSnapshot { refreshModelCatalog(); let current = sessionOrder.compactMap { sessions[$0] }; return .init(status: serverStatus, port: serverPort, url: "http://127.0.0.1:\(serverPort)", activeSessions: current.filter { $0.status == .queued || $0.status == .running }.count, queuedSessions: await queue.queuedCount, sessions: current, availableModels: modelDirectories.keys.sorted(), loadedModel: loadedModel, lastError: lastError) }
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
            let options = Qwen38GenerationOptions(maxTokens: min(max(input.effectiveMaxTokens ?? 256, 1), 131_072), temperature: input.temperature ?? 0, topP: input.topP ?? 0.95, enableThinking: input.effectiveThinking ?? (input.effectiveReasoningEffort != nil), reasoningEffort: input.effectiveReasoningEffort ?? "low", mtp: .init(enabled: input.effectiveMTP ?? true, draftDepth: .fixed(input.effectiveMTPDraftTokens), engine: input.effectiveMTPEngine))
            let conversationID = input.effectiveConversationID
            let usePersistentCache = try await prepareConversation(
                id: conversationID,
                model: selectedModel,
                messages: prepared.messages,
                options: options)
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
            if input.stream == true {
                return try await makeStreamingResponse(
                    stream: stream, sessionID: id, model: selectedModel,
                    primedInside: thinkingIsPrimed,
                    conversationID: usePersistentCache ? conversationID : nil,
                    requestMessages: prepared.messages)
            }
            return try await makeJSONResponse(
                stream: stream, sessionID: id, model: selectedModel,
                primedInside: thinkingIsPrimed,
                conversationID: usePersistentCache ? conversationID : nil,
                requestMessages: prepared.messages)
        } catch {
            if input.effectiveConversationID != nil {
                clearActiveConversation()
                await runtime.resetConversation()
            }
            updateSession(id) { $0.status = .failed; $0.error = error.localizedDescription; $0.finishedAt = Date() }
            return Self.errorResponse(Self.status(for: error), message: error.localizedDescription)
        }
    }

    private func prepareConversation(
        id: String?,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> Bool {
        guard let id else {
            clearActiveConversation()
            return false
        }
        let isContinuation = activeConversationID == id
            && activeConversationModel == model
            && activeConversationOptions.map(cacheCompatible(_:)) == true
            && messages.count == activeConversationMessages.count + 1
            && Array(messages.dropLast()) == activeConversationMessages
        if isContinuation {
            return true
        }

        // A new or non-contiguous session is deliberately cold. A short
        // system/user prompt can start a persistent cache directly; longer
        // histories use the stateless replay path and are not advertised as
        // cached because reconstructing assistant hidden states is impossible
        // without rerunning them.
        await runtime.resetConversation()
        activeConversationID = nil
        activeConversationModel = nil
        activeConversationMessages = []
        activeConversationOptions = nil
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

    private func cacheCompatible(_ options: Qwen38GenerationOptions) -> Bool {
        guard let active = activeConversationOptions else { return false }
        return active.temperature == options.temperature
            && active.topP == options.topP
            && active.topK == options.topK
            && active.enableThinking == options.enableThinking
            && active.reasoningEffort == options.reasoningEffort
            && active.kvBits == options.kvBits
            && active.mtp == options.mtp
    }

    private func clearActiveConversation() {
        activeConversationID = nil
        activeConversationModel = nil
        activeConversationMessages = []
        activeConversationOptions = nil
    }

    private func rememberConversation(
        id: String,
        model: String,
        requestMessages: [Qwen38ChatMessage],
        assistantContent: String
    ) {
        guard activeConversationID == id, activeConversationModel == model else { return }
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

    private func makeJSONResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, conversationID: String?, requestMessages: [Qwen38ChatMessage]) async throws -> Response {
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
            rememberConversation(id: conversationID, model: model, requestMessages: requestMessages, assistantContent: text)
        }
        return Self.jsonResponse(ChatCompletionResponse(id: "chatcmpl-\(sessionID.uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970), model: model, choices: [.init(index: 0, message: .init(role: "assistant", content: text, reasoningContent: reasoning.nilIfEmpty), delta: nil, finishReason: Self.finishReason(metrics?.stopReason))]))
    }
    private func makeStreamingResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, conversationID: String?, requestMessages: [Qwen38ChatMessage]) async throws -> Response {
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
                                assistantContent: responseContent)
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
