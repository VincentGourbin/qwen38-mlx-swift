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
    /// Jetons du prompt servis depuis le cache de conversation (préfixe
    /// réutilisé) ; `promptTokens` ne compte que les jetons préremplis par
    /// ce tour. Voir `Qwen38RunMetrics.cachedPromptTokens`.
    public var cachedPromptTokens: Int = 0
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
    /// P11.1 : largeur de routage MoE effectivement utilisée par ce tour
    /// (`Qwen38RunMetrics.routedExpertCount`) — `nil` tant que le tour n'a
    /// pas terminé ou sur la famille 27B. Publié pour la même raison que
    /// `/healthz` : ne pas croire à tort avoir mesuré un K qu'on n'a pas
    /// réellement appliqué (PLAN.md P11.1).
    public var routedExpertCount: Int?
    /// P11.2 : ablation effectivement utilisée par ce tour
    /// (`Qwen38RunMetrics.ablation`) — `nil` tant que le tour n'a pas
    /// terminé. Publié pour la même raison que `routedExpertCount` et que
    /// `/healthz` : ne jamais laisser croire à tort qu'une mesure a été
    /// prise avec (ou sans) ablation active (PLAN.md P11.2).
    public var ablation: String?
    /// P12.3 : taille du lot dans lequel cette session a été effectivement
    /// servie (`1` pour le chemin mono-séquence, `N > 1` pour un lot réel)
    /// — `nil` tant que ce n'est pas encore décidé. Publiée uniquement au
    /// moment où l'ordonnanceur a réellement tranché, jamais devinée à
    /// l'avance : même garde de publication que `routedExpertCount`/
    /// `ablation` (PLAN.md P12.3, « ne jamais mesurer en croyant à tort
    /// avoir groupé »). Toujours `nil` quand `serve --batch-size` vaut 1.
    public var batchSizeServed: Int?
    public init(id: UUID = UUID(), client: String, path: String, model: String = "Qwen3.8", conversationID: String? = nil, startedAt: Date = Date(), status: Qwen38ServerSessionStatus = .queued, inputDescription: String = "Texte", promptTokens: Int = 0, generatedTokens: Int = 0, timeToFirstToken: TimeInterval? = nil, tokensPerSecond: Double? = nil, lastToken: String = "", cacheReused: Bool = false, cacheRestored: Bool = false, cacheReplayed: Bool = false, conversationReplayed: Bool = false, mtp: String = "indisponible", mtpProposed: Int = 0, mtpAccepted: Int = 0, mtpAcceptRate: Double? = nil, error: String? = nil, routedExpertCount: Int? = nil, ablation: String? = nil, batchSizeServed: Int? = nil) {
        self.id = id; self.client = client; self.path = path; self.model = model; self.conversationID = conversationID; self.startedAt = startedAt; self.status = status; self.inputDescription = inputDescription; self.promptTokens = promptTokens; self.generatedTokens = generatedTokens; self.timeToFirstToken = timeToFirstToken; self.tokensPerSecond = tokensPerSecond; self.lastToken = lastToken; self.cacheReused = cacheReused; self.cacheRestored = cacheRestored; self.cacheReplayed = cacheReplayed; self.conversationReplayed = conversationReplayed; self.mtp = mtp; self.mtpProposed = mtpProposed; self.mtpAccepted = mtpAccepted; self.mtpAcceptRate = mtpAcceptRate; self.error = error; self.routedExpertCount = routedExpertCount; self.ablation = ablation; self.batchSizeServed = batchSizeServed
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
    /// P6.1: implicit-prefix cache counters — requests **without** a
    /// `conversation_id` (Open WebUI, plain OpenAI SDK clients) that either
    /// found (`prefixHits`) or missed (`prefixMisses`) a matching LRU/active
    /// ledger by rendered token IDs. Disjoint from `cacheMisses`, which only
    /// ever counts explicit-`conversation_id` misses.
    public let prefixHits: Int
    public let prefixMisses: Int
    /// P12.3 : taille de lot configurée au démarrage (`serve --batch-size`,
    /// défaut 1) — toujours publiée, jamais devinée : c'est la garde de
    /// publication qui permet de distinguer « le lot est actif » de « je
    /// crois qu'il l'est » (même contrat que `routedExpertCount`/`ablation`
    /// sur `/healthz`, PLAN.md P11/P12.3).
    public let batchSizeConfigured: Int
    /// Défaut 2026-09-14 : seuil de longueur de prompt (`serve
    /// --batch-max-prompt-tokens`, défaut 256) au-delà duquel une requête
    /// froide n'est jamais proposée au lot — voir `chatCompletionsResponseBatched`
    /// et le rapport du défaut A. Même garde de publication que
    /// `batchSizeConfigured` : toujours publiée, jamais devinée.
    public let batchMaxPromptTokensConfigured: Int
    public init(status: Qwen38ServerStatus, port: Int, url: String, activeSessions: Int, queuedSessions: Int, sessions: [Qwen38ServerSession], availableModels: [String] = [], loadedModel: String? = nil, lastError: String? = nil, cacheMisses: Int = 0, cachedConversations: Int = 0, cacheBytes: Int64 = 0, cacheBudgetBytes: Int64 = 0, prefixHits: Int = 0, prefixMisses: Int = 0, batchSizeConfigured: Int = 1, batchMaxPromptTokensConfigured: Int = 256) { self.status = status; self.port = port; self.url = url; self.activeSessions = activeSessions; self.queuedSessions = queuedSessions; self.sessions = sessions; self.availableModels = availableModels; self.loadedModel = loadedModel; self.lastError = lastError; self.cacheMisses = cacheMisses; self.cachedConversations = cachedConversations; self.cacheBytes = cacheBytes; self.cacheBudgetBytes = cacheBudgetBytes; self.prefixHits = prefixHits; self.prefixMisses = prefixMisses; self.batchSizeConfigured = batchSizeConfigured; self.batchMaxPromptTokensConfigured = batchMaxPromptTokensConfigured }
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
    let model: String?; let messages: [ChatCompletionMessage]; let stream: Bool?; let maxTokens: Int?; let maxCompletionTokens: Int?; let temperature: Float?; let topP: Float?; let presencePenalty: Float?; let frequencyPenalty: Float?; let reasoningEffort: String?; let reasoning: ChatCompletionReasoning?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; let routedExperts: Int?; let ablation: String?; let extra: ChatCompletionExtra?
    /// P13.1 : outils au format OpenAI (`{"type":"function","function":{name,description,parameters}}`)
    /// et le sélecteur associé — voir `effectiveTools`.
    let tools: [ChatCompletionRequestTool]?
    let toolChoice: Qwen38JSONValue?
    enum CodingKeys: String, CodingKey { case model, messages, stream, maxTokens = "max_tokens", maxCompletionTokens = "max_completion_tokens", temperature, topP = "top_p", presencePenalty = "presence_penalty", frequencyPenalty = "frequency_penalty", reasoningEffort = "reasoning_effort", reasoning, enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id", routedExperts = "routed_experts", ablation, extra, tools, toolChoice = "tool_choice" }

    /// P13.1 : liste normalisée — vide si `tools` est absent/vide ou si
    /// `tool_choice` vaut explicitement `"none"`. Le gabarit du checkpoint
    /// n'a aucune notion de forcer un outil précis : un `tool_choice` objet
    /// (`{"type":"function","function":{"name":…}}`) est accepté sans
    /// erreur mais n'a aucun effet au-delà de "les outils sont bien là" —
    /// voir le rapport à Vincent.
    var effectiveTools: [ChatCompletionRequestTool] {
        if case .string("none")? = toolChoice { return [] }
        return tools ?? []
    }

    var effectiveMaxTokens: Int? { maxCompletionTokens ?? maxTokens }
    var effectiveReasoningEffort: String? { reasoningEffort ?? reasoning?.effort ?? extra?.reasoningEffort }
    var effectiveThinking: Bool? { enableThinking ?? extra?.enableThinking }
    /// Défaut passé à `false` le 2026-09-16. Deux raisons mesurées :
    /// (a) P11.4 a montré le MTP **perdant** (0,939× le greedy : 19,72 contre
    /// 21,01 tok/s) — il coûtait donc du débit à tout client qui ne le
    /// désactivait pas ; (b) le chemin MTP fait planter le serveur à long
    /// contexte (`[broadcast_shapes] Shapes (1,1,20310,20575) et
    /// (1,1,20310,20577)` dans le masque QSA, écart de 2 = les jetons de
    /// brouillon), défaut jamais vu avant parce que toutes les validations
    /// de §P13 envoyaient explicitement `mtp: false`. Un client qui demande
    /// `mtp: true` garde la main, et retombe sur le même chemin qu'avant.
    var effectiveMTP: Bool? { mtp ?? extra?.mtp ?? false }
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
    /// P6.3: `extra.penalty_context_tokens` — how many trailing assistant-
    /// turn tokens the presence/repetition mask is seeded with. Absent
    /// falls through to `Qwen38GenerationOptions`'s own default (2048); an
    /// explicit `0` restores the pre-P6.3 per-turn-only mask.
    var effectivePenaltyContextTokens: Int? { extra?.penaltyContextTokens }
    /// P11.1 : `routed_experts` au premier niveau ou dans `extra` (même
    /// convention double que `mtp`/`conversation_id`) — surcharge
    /// ponctuelle de `num_experts_per_tok` pour ce tour, Flash-Next
    /// uniquement. Absent des deux : ne touche pas au réglage en vigueur
    /// (voir `Qwen38GenerationOptions.routedExpertCount`).
    var effectiveRoutedExperts: Int? { routedExperts ?? extra?.routedExperts }
    /// P11.2 : `ablation` au premier niveau ou dans `extra` (même convention
    /// double que `routed_experts`/`mtp`) — chaîne (`rawValue` de
    /// `Qwen4ExpLayerBenchAblation`, `"none"` pour désactiver). Absent des
    /// deux : ne touche pas au réglage en vigueur. Présent alors que le
    /// serveur n'a pas démarré avec `--allow-ablation` : requête refusée en
    /// HTTP 400 avant même la résolution de la chaîne — voir
    /// `chatCompletionsResponse`.
    var effectiveAblation: String? { ablation ?? extra?.ablation }
}
private struct ChatCompletionReasoning: Codable, Sendable { let effort: String? }
private struct ChatCompletionExtra: Codable, Sendable { let reasoningEffort: String?; let enableThinking: Bool?; let mtp: Bool?; let mtpEngine: String?; let mtpDraftTokens: Int?; let conversationID: String?; let repetitionPenalty: Float?; let penaltyContextTokens: Int?; let routedExperts: Int?; let ablation: String?; enum CodingKeys: String, CodingKey { case reasoningEffort = "reasoning_effort", enableThinking = "enable_thinking", mtp, mtpEngine = "mtp_engine", mtpDraftTokens = "mtp_draft_tokens", conversationID = "conversation_id", repetitionPenalty = "repetition_penalty", penaltyContextTokens = "penalty_context_tokens", routedExperts = "routed_experts", ablation } }
private struct ChatCompletionMessage: Codable, Sendable {
    let role: String
    let content: ChatCompletionContent?
    /// P13.1 : présent sur un tour assistant qui a appelé un outil.
    let toolCalls: [ChatCompletionRequestToolCall]?
    /// P13.1 : présent sur un message `role: "tool"` — accepté pour
    /// compatibilité avec le format OpenAI, mais non utilisé pour le rendu :
    /// le gabarit du checkpoint apparie les réponses d'outils par ordre,
    /// jamais par identifiant (voir `chat_template.jinja`).
    let toolCallID: String?
    enum CodingKeys: String, CodingKey { case role, content, toolCalls = "tool_calls", toolCallID = "tool_call_id" }
}

/// P13.1 : un `tools[]` de requête, format OpenAI.
private struct ChatCompletionRequestToolFunction: Codable, Sendable {
    let name: String
    let description: String?
    let parameters: Qwen38JSONValue?
}
private struct ChatCompletionRequestTool: Codable, Sendable {
    let type: String?
    let function: ChatCompletionRequestToolFunction
    func toSpec() -> Qwen38ToolSpec {
        Qwen38ToolSpec(
            type: type ?? "function", name: function.name, description: function.description,
            parameters: function.parameters)
    }
}
/// P13.1 : un `message.tool_calls[]` d'un tour assistant renvoyé par le
/// client — `arguments` est, comme dans le format OpenAI, une chaîne JSON
/// (jamais un objet imbriqué).
private struct ChatCompletionRequestToolCallFunction: Codable, Sendable {
    let name: String
    let arguments: String?
}
private struct ChatCompletionRequestToolCall: Codable, Sendable {
    let id: String?
    let type: String?
    let function: ChatCompletionRequestToolCallFunction
}
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
    /// P13.1 : `nil` (donc omis, `encodeIfPresent`) sauf si au moins un
    /// `<tool_call>` a été reconnu dans la réponse — la forme JSON d'une
    /// réponse sans outils reste bit-identique à avant P13.1.
    let toolCalls: [ChatCompletionToolCallOut]?
    enum CodingKeys: String, CodingKey { case role, content, reasoningContent = "reasoning_content", toolCalls = "tool_calls" }
}
private struct ChatCompletionDelta: Codable, Sendable {
    let role: String?
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ChatCompletionToolCallOut]?
    enum CodingKeys: String, CodingKey { case role, content, reasoningContent = "reasoning_content", toolCalls = "tool_calls" }
}
/// P13.1 : un `tool_calls[]` de réponse, réutilisé pour le message final
/// (`index: nil`, omis) et pour un fragment `delta.tool_calls` en diffusion
/// (`index` renseigné) — voir le rapport à Vincent sur le choix "un seul
/// fragment une fois complet".
private struct ChatCompletionToolCallOut: Codable, Sendable {
    let index: Int?
    let id: String?
    let type: String?
    let function: ChatCompletionToolCallFunctionOut
}
private struct ChatCompletionToolCallFunctionOut: Codable, Sendable {
    let name: String
    let arguments: String
}
private struct ChatCompletionResponse: Codable, Sendable { let id: String; let object: String; let created: Int; let model: String; let choices: [ChatCompletionChoice]; var usage: ChatCompletionUsage? = nil }
/// Bloc `usage` OpenAI. `prompt_tokens` est la longueur TOTALE du prompt vue
/// par le modèle (préfixe en cache compris) et `prompt_tokens_details.
/// cached_tokens` la part servie depuis le cache — la convention qu'un client
/// comme pi lit pour suivre son contexte et chiffrer une session (issue #1).
/// En diffusion, il est porté par le dernier fragment (celui du
/// `finish_reason`), comme le fait `stream_options.include_usage`.
private struct ChatCompletionUsage: Codable, Sendable {
    struct PromptTokensDetails: Codable, Sendable { let cachedTokens: Int; enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" } }
    let promptTokens: Int; let completionTokens: Int; let totalTokens: Int; let promptTokensDetails: PromptTokensDetails
    enum CodingKeys: String, CodingKey { case promptTokens = "prompt_tokens", completionTokens = "completion_tokens", totalTokens = "total_tokens", promptTokensDetails = "prompt_tokens_details" }
    init(metrics: Qwen38RunMetrics) {
        promptTokens = metrics.cachedPromptTokens + metrics.metrics.promptTokens
        completionTokens = metrics.metrics.generatedTokens
        totalTokens = promptTokens + completionTokens
        promptTokensDetails = .init(cachedTokens: metrics.cachedPromptTokens)
    }
}
private struct ModelListResponse: Codable, Sendable { let object: String; let data: [ModelDescription] }
private struct ModelDescription: Codable, Sendable { let id: String; let object: String; let ownedBy: String; let loaded: Bool; let family: String?; enum CodingKeys: String, CodingKey { case id, object, ownedBy = "owned_by", loaded, family } }
private struct HealthResponse: Codable, Sendable { let status: String; let modelLoaded: Bool; let model: String?; let queue: String; let defaultEnableThinking: Bool; let routedExpertCount: Int?; let ablation: String; let batchSizeConfigured: Int; let batchMaxPromptTokensConfigured: Int; enum CodingKeys: String, CodingKey { case status, modelLoaded = "model_loaded", model, queue, routedExpertCount = "routed_expert_count", defaultEnableThinking = "default_enable_thinking", ablation, batchSizeConfigured = "batch_size_configured", batchMaxPromptTokensConfigured = "batch_max_prompt_tokens_configured" } }
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
    /// P11.1 : option de démarrage (`serve --routed-experts`), appliquée à
    /// tout chargement de modèle Flash-Next — y compris un changement de
    /// modèle déclenché par une requête ultérieure (`ensureModelLoaded`),
    /// pas seulement le tout premier chargement. `nil` (le défaut) laisse
    /// le comportement inchangé.
    private var startupRoutedExpertCount: Int?
    /// P11.2 : option de démarrage (`serve --allow-ablation`, défaut
    /// `false`). Tant que c'est `false`, un champ de requête `ablation` est
    /// refusé en HTTP 400 avant d'atteindre le moteur — l'ablation produit
    /// des sorties numériquement fausses par construction et ne doit
    /// jamais être déclenchable par un client ordinaire (PLAN.md P11.2).
    private var allowAblation = false
    /// P12.3 : `serve --batch-size` (défaut 1). À 1, `start` n'enregistre
    /// même pas la route batchée : `/v1/chat/completions` reste
    /// `chatCompletionsResponse`, strictement inchangée — voir `start`'s
    /// commentaire. `> 1` construit `batchCoordinator` et enregistre
    /// `chatCompletionsResponseBatched` à la place.
    private var batchSize = 1
    /// Valeur de repli du mode réflexion pour les clients qui ne
    /// l'expriment pas (`serve --enable-thinking`). Nécessaire pour les
    /// harnais d'agent : mesuré le 2026-09-15, sans réflexion le modèle
    /// enchaîne des appels d'outils pertinents mais ne conclut jamais
    /// (22 appels, 0 réponse finale ; avec réflexion, 4 pas). Un client
    /// qui envoie explicitement `enable_thinking` ou `reasoning_effort`
    /// garde toujours la main : ce n'est qu'un défaut.
    private var defaultEnableThinking = false
    /// Défaut A (2026-09-14) : `serve --batch-max-prompt-tokens` (défaut
    /// 256). Une requête froide dont le prompt RENDU dépasse ce seuil ne
    /// rejoint jamais `batchCoordinator` — elle garde le chemin
    /// chaud/mono-séquence de `chatCompletionsResponseBatched`, avec son
    /// TTFT d'aujourd'hui. Voir `chatCompletionsResponseBatched` pour
    /// l'endroit exact du test, et le rapport du défaut A pour la
    /// justification du défaut :
    ///
    /// Le préfill est un calcul DENSE (contrairement au décodage, qui a de
    /// la capacité GPU libre à remplir) : le regrouper ne fait qu'additionner
    /// le travail de chaque ligne (plus le gâchis du remplissage à la
    /// longueur du plus long prompt du lot), sans aucun recouvrement
    /// possible. Le TTFT d'une requête groupée croît donc à peu près
    /// linéairement avec la taille du lot — mesuré : un lot de 4 sur des
    /// prompts d'environ 1 200 jetons porte le TTFT de 7,53 s (seul) à
    /// 37,20 s (×4,9). Sur les prompts courts (~25 jetons) qui ont mesuré le
    /// ×2,28 de P12.3/P12.4, ce même facteur reste sous la seconde et n'est
    /// pas perceptible. 256 est choisi comme un ordre de grandeur sous le
    /// point mesuré catastrophique (1 200) et un ordre de grandeur au-dessus
    /// du point mesuré sans dégradation (~25), donc avec de la marge des
    /// deux côtés ; c'est aussi une longueur qui couvre un tour de
    /// conversation ordinaire (quelques phrases, un petit historique) sans
    /// couvrir un prompt qui colle un document. Non mesuré finement au-delà
    /// de ces deux points — un opérateur qui connaît son trafic doit
    /// recalibrer avec `--batch-max-prompt-tokens`.
    private var batchMaxPromptTokens = 256
    /// P12.3 : verrou d'exécution du chemin batché — séparé de `queue`
    /// (utilisé uniquement par le chemin `batchSize == 1` inchangé) pour ne
    /// jamais toucher son comportement. Tenu depuis `ensureModelLoaded`
    /// jusqu'à ce que l'exécution qui touche réellement le modèle résident
    /// (chemin chaud/seul ou lot) soit terminée — jamais seulement jusqu'à
    /// la construction de la `Response` HTTP, et jamais seulement jusqu'à
    /// la consommation des flux (correctif du 2026-09-13, crash mémoire :
    /// voir `Qwen38BatchGenerationResult`'s commentaire pour le chemin lot,
    /// `Qwen38BatchCompletionGate`'s pour le chemin chaud/seul).
    private let batchExecutionLock = FIFORequestQueue()
    /// P12.3 : `nil` quand `batchSize == 1`. Construit par `start` avec un
    /// `runBatch` qui referme sur `self.runtime`.
    private var batchCoordinator: Qwen38BatchCoordinator?
    // P6.4: the per-conversation LRU (id/prefix matching, restore/export,
    // budget, prefix hit/miss counters) moved to `Qwen38Runtime` so the GUI
    // shares it too (`Qwen38Runtime.generate`'s Flash-Next branch) instead
    // of only ever resetting on its first turn. The server keeps ownership
    // of the *budget* (`start(conversationCacheGB:)` below) and thin
    // forwarders (`prepareConversation`/`rememberConversation`) so existing
    // call sites and tests are unaffected.
    public init(runtime: Qwen38Runtime) { self.runtime = runtime }

    public func start(port: Int = 8848, apiKey: String? = nil, modelsDirectory: URL? = nil, conversationCacheGB: Double = 12, routedExpertCount: Int? = nil, allowAblation: Bool = false, batchSize: Int = 1, enableThinking: Bool = false, batchMaxPromptTokens: Int = 256, batchWindowMs: Int = 30) async throws {
        guard (1 ... 65_535).contains(port) else { throw Qwen38ServerError.invalidPort }
        // P12.3 : mémorisé pour toute la durée de vie du serveur — voir
        // `batchSize`'s doc comment. `<= 1` désactive le regroupement,
        // exactement comme avant P12.3 (aucun `Qwen38BatchCoordinator`
        // construit, route `chatCompletionsResponse` inchangée ci-dessous).
        self.batchSize = max(batchSize, 1)
        self.defaultEnableThinking = enableThinking
        // Défaut A (2026-09-14) : voir `batchMaxPromptTokens`'s doc comment.
        self.batchMaxPromptTokens = max(batchMaxPromptTokens, 0)
        if self.batchSize > 1 {
            // P12.3 : `batchExecutionLock` doit être tenu pour TOUTE
            // exécution qui touche réellement le modèle résident, lot
            // compris — pas seulement le chemin chaud/seul de
            // `chatCompletionsResponseBatched`. Sans ce verrou ici, un lot
            // pourrait démarrer pendant qu'une requête chaude — ou un AUTRE
            // lot — est encore en train d'appeler `model.forward()`,
            // corrompant l'état partagé.
            //
            // Correctif du 2026-09-13 (crash mémoire, `EXC_BAD_ACCESS` dans
            // les couches résidentes) : le verrou n'est PAS relâché quand
            // les flux du lot sont consommés (`rawStreams` retournés ici
            // sont livrés immédiatement au client, avant même que la
            // génération n'ait commencé). Il est relâché uniquement quand
            // `result.completion` se termine — le signal que
            // `Qwen4ExpBatchStreamingGenerator.run()` a réellement fini de
            // toucher le modèle, y compris sa propre réinitialisation de
            // fin de lot (voir `Qwen38BatchGenerationResult`'s
            // commentaire). La libération via consommation des flux
            // laissait un second lot démarrer sa réinitialisation pendant
            // que le premier exécutait encore la sienne, après avoir déjà
            // refermé ses flux — deux `resetConversation()` concurrents sur
            // le même modèle résident, cause vérifiée du crash.
            batchCoordinator = Qwen38BatchCoordinator(
                batchSize: self.batchSize, window: .milliseconds(max(batchWindowMs, 0))
            ) {
                [runtime, batchExecutionLock] requests in
                await batchExecutionLock.acquire()
                do {
                    let result = try await runtime.generateBatchFlashConversations(
                        requests: requests.map { .init(messages: $0.messages, options: $0.options) })
                    Task {
                        _ = await result.completion.value
                        await batchExecutionLock.release()
                    }
                    return result.streams
                } catch {
                    await batchExecutionLock.release()
                    throw error
                }
            }
        } else {
            batchCoordinator = nil
        }
        // P11.1 : mémorisé pour tout (re)chargement ultérieur — voir
        // `startupRoutedExpertCount`'s doc comment. Ne touche pas le modèle
        // déjà résident au moment de cet appel (chargé séparément par le
        // caller, `serve` en CLI, avant `start`) ; un modèle déjà chargé
        // avec un K différent reste tel quel jusqu'à sa prochaine
        // (re)sélection ou une surcharge par requête.
        startupRoutedExpertCount = routedExpertCount
        // P11.2 : mémorisé pour toute la durée de vie du serveur — voir
        // `allowAblation`'s doc comment.
        self.allowAblation = allowAblation
        await runtime.configureConversationCacheBudget(gb: conversationCacheGB)
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
        router.get("v1/models") { [self] request, _ in await self.catchingHTTPErrors { try await self.modelsResponse(request: request) } }
        router.get("metrics") { [self] _, _ in await self.metricsResponse() }
        // P12.3 : à `batchSize == 1` (le défaut), `/v1/chat/completions`
        // reste très exactement `chatCompletionsResponse` — pas une variante
        // qui passerait par le nouveau code avec un lot de taille 1, la
        // fonction non modifiée d'avant P12.3. `batchSize > 1` enregistre à
        // la place `chatCompletionsResponseBatched`, qui seule connaît
        // `batchCoordinator`/`batchExecutionLock`.
        if self.batchSize > 1 {
            router.post("v1/chat/completions") { [self] request, _ in await self.catchingHTTPErrors { try await self.chatCompletionsResponseBatched(request: request) } }
        } else {
            router.post("v1/chat/completions") { [self] request, _ in await self.catchingHTTPErrors { try await self.chatCompletionsResponse(request: request) } }
        }
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
        // P12.3 : ne laisse jamais un client suspendu derrière une fenêtre
        // de regroupement qui ne se déclenchera plus.
        if let batchCoordinator { await batchCoordinator.drain() }
    }
    public func snapshot() async -> Qwen38ServerSnapshot {
        refreshModelCatalog()
        let current = sessionOrder.compactMap { sessions[$0] }
        let cache = await runtime.flashConversationCacheSnapshot()
        return .init(
            status: serverStatus, port: serverPort, url: "http://127.0.0.1:\(serverPort)",
            activeSessions: current.filter { $0.status == .queued || $0.status == .running }.count,
            queuedSessions: await queue.queuedCount, sessions: current,
            availableModels: modelDirectories.keys.sorted(), loadedModel: loadedModel,
            lastError: lastError, cacheMisses: cache.cacheMisses,
            cachedConversations: cache.cachedConversations, cacheBytes: cache.cacheBytes,
            cacheBudgetBytes: cache.cacheBudgetBytes, prefixHits: cache.prefixHits,
            prefixMisses: cache.prefixMisses, batchSizeConfigured: batchSize,
            batchMaxPromptTokensConfigured: batchMaxPromptTokens)
    }
    private func serverDidStop() { if serverStatus != .stopping { serverStatus = .stopped } }
    private func serverDidFail(_ error: String) { lastError = error; serverStatus = .failed }

    // P11.1 : `routed_expert_count` publié systématiquement (pas seulement
    // en cas de surcharge) pour qu'on ne puisse jamais croire à tort avoir
    // changé K sans le vérifier ici — voir PLAN.md P11.1, "métriques et
    // /healthz". `nil` seulement quand aucun engin Flash-Next n'est chargé.
    // P11.2 : `ablation` publié systématiquement, `"none"` quand il n'y en a
    // pas (aucun engin Flash-Next chargé, ou aucune ablation demandée) —
    // même garde de publication que `routed_expert_count`, PLAN.md P11.2.
    private func healthResponse() async -> Response { Self.jsonResponse(HealthResponse(status: serverStatus.rawValue, modelLoaded: await runtime.isLoaded, model: loadedModel, queue: String(sessions.values.filter { $0.status == .queued }.count), defaultEnableThinking: defaultEnableThinking, routedExpertCount: await runtime.flashRoutedExpertCount, ablation: await runtime.flashAblation?.rawValue ?? "none", batchSizeConfigured: batchSize, batchMaxPromptTokensConfigured: batchMaxPromptTokens)) }
    private func modelsResponse(request: Request) async throws -> Response { try authorize(request); refreshModelCatalog(); let current = loadedModel; let models = modelDirectories.keys.sorted().map { id -> ModelDescription in let family = modelDirectories[id].flatMap { try? Qwen38ModelValidator.readInfo(from: $0) }?.family; return ModelDescription(id: id, object: "model", ownedBy: "local", loaded: id == current, family: family?.rawValue) }; return Self.jsonResponse(ModelListResponse(object: "list", data: models)) }
    private func metricsResponse() async -> Response { let current = await snapshot(); return Self.jsonResponse(current) }

    private func chatCompletionsResponse(request: Request) async throws -> Response {
        try authorize(request); var request = request; let buffer = try await request.collectBody(upTo: 64 * 1024 * 1024)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else { throw Qwen38ServerError.invalidRequest("Le corps JSON est vide.") }
        let input: ChatCompletionRequest
        do { input = try JSONDecoder().decode(ChatCompletionRequest.self, from: data) } catch { throw Qwen38ServerError.invalidRequest("Requête chat invalide : \(error.localizedDescription)") }
        guard !input.messages.isEmpty else { throw Qwen38ServerError.invalidRequest("La requête doit contenir au moins un message.") }
        // P13.1 : un tour "tool" (le résultat d'un appel, renvoyé au serveur
        // pour que le modèle produise sa réponse finale) est désormais un
        // dernier message valide, au même titre que "user" — c'est
        // exactement la forme d'une requête de suivi OpenAI après un appel
        // d'outil.
        //
        // Défaut du 2026-09-15 : "assistant" rejoint la liste. Une boucle
        // d'agent réelle (OpenCode, Claude Code) peut parfaitement produire
        // ce dernier rôle — le modèle a atteint `max_tokens` en pleine
        // réflexion, sans texte ni appel d'outil, et le client a ajouté un
        // tour assistant (vide, ou porteur du texte tronqué) puis a
        // renvoyé l'historique pour que le serveur termine ce tour. Ce
        // n'est pas un protocole étranger à ce serveur : c'est exactement
        // ce que `dispatchConversationTurn`/`continueConversationTurn`
        // (P13.3) savent déjà faire — le suffixe à préfiller est calculé
        // par différence de rendu complet, donc indépendant du rôle du
        // dernier message (voir `Qwen4ExpPromptBuilder.continuationSuffix`) —
        // et le rejeu complet (`generateFromMessages`/`buildFromMessages`)
        // qui sert de repli n'a lui non plus aucune hypothèse sur le rôle
        // du dernier message. Rejeter cette forme obligerait exactement le
        // client qui a motivé P13.3 (une boucle d'agent) à réémettre tout
        // l'historique sous un rôle différent pour contourner un refus
        // artificiel — moins cohérent avec le reste du serveur que de
        // l'accepter tel quel.
        guard let lastMessageRole = input.messages.last?.role,
              lastMessageRole == "user" || lastMessageRole == "tool" || lastMessageRole == "assistant"
        else { throw Qwen38ServerError.invalidRequest("Le dernier message doit avoir le rôle user, tool (après un appel d'outil), ou assistant (pour continuer un tour interrompu).") }
        let requestedTools = Self.orderedToolSpecs(input.effectiveTools.map { $0.toSpec() }, body: data)
        guard requestedTools.allSatisfy({ !$0.name.isEmpty }) else { throw Qwen38ServerError.invalidRequest("Chaque outil déclaré dans tools doit avoir un nom (function.name).") }
        let requestedModel = input.model?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let id = UUID(); sessions[id] = .init(id: id, client: "LAN", path: "/v1/chat/completions", model: requestedModel ?? loadedModel ?? "default", conversationID: input.effectiveConversationID); sessionOrder.append(id); trimSessions(); await queue.acquire(); updateSession(id) { $0.status = .running }; defer { Task { await queue.release() } }
        do {
            let selectedModel = try await ensureModelLoaded(requestedModel)
            updateSession(id) { $0.model = selectedModel }
            // P13.1/P14.4 : l'appel d'outils dépend du gabarit dédié du
            // checkpoint (chat_template.jinja) — Flash-Next et la famille
            // 27B (Bonsai 2 y compris) partagent le même gabarit, donc le
            // même rendu d'outils ; seul l'absence de tout modèle chargé
            // justifie un refus explicite plutôt qu'un silence trompeur.
            if !requestedTools.isEmpty {
                guard await runtime.isLoaded else {
                    throw Qwen38ServerError.invalidRequest("Le champ tools nécessite qu'un modèle soit chargé.")
                }
            }
            let prepared = try prepare(input.messages)
            let temperature = input.temperature ?? 0
            // P5.3: server default when a sampling request (temperature > 0)
            // omits both `presence_penalty` and `frequency_penalty` — the
            // instruct preset's `presence 1.5` (PLAN.md §2.1). An explicit
            // `presence_penalty: 0` (or `frequency_penalty: 0`) disables it;
            // greedy requests (temperature 0) never get this default since
            // the generator ignores penalties there regardless.
            let presencePenalty = input.explicitPresencePenalty ?? (temperature > 0 ? 1.5 : 0)
            // P11.2 : un champ `ablation` n'est même résolu que si le
            // serveur a démarré avec `--allow-ablation` — sinon la requête
            // est refusée en HTTP 400 avant d'atteindre le moteur.
            // L'ablation produit des sorties numériquement fausses par
            // construction ; elle ne doit jamais pouvoir être déclenchée par
            // un client ordinaire (PLAN.md P11.2).
            let requestedAblation: Qwen4ExpLayerBenchAblation?
            if let rawAblation = input.effectiveAblation {
                guard allowAblation else {
                    throw Qwen38ServerError.invalidRequest(
                        "Le champ ablation est refusé : redémarrez le serveur avec --allow-ablation pour l'autoriser. L'ablation produit des sorties numériquement fausses par construction.")
                }
                requestedAblation = try qwen4ExpResolveAblation(rawValue: rawAblation)
            } else {
                requestedAblation = nil
            }
            let options = Qwen38GenerationOptions(maxTokens: try Self.resolvedMaxTokens(input.effectiveMaxTokens), temperature: temperature, topP: input.topP ?? 0.95, enableThinking: input.effectiveThinking ?? (input.effectiveReasoningEffort != nil || defaultEnableThinking), reasoningEffort: input.effectiveReasoningEffort ?? "low", mtp: .init(enabled: input.effectiveMTP ?? true, draftDepth: .fixed(input.effectiveMTPDraftTokens), engine: input.effectiveMTPEngine), presencePenalty: presencePenalty, repetitionPenalty: input.effectiveRepetitionPenalty, penaltyContextTokens: max(0, input.effectivePenaltyContextTokens ?? 2048), routedExpertCount: input.effectiveRoutedExperts, ablation: requestedAblation, tools: requestedTools)
            let conversationID = input.effectiveConversationID
            // P13.2 : une requête outillée touche désormais le cache de
            // conversation Flash-Next (LRU/préfixe implicite) exactement
            // comme une requête ordinaire — `cacheOptionsCompatible` compare
            // déjà `options.tools`, donc un changement de la liste d'outils
            // invalide proprement la comparaison (voir le test dédié). C'est
            // un renversement délibéré de la décision de P13.1 : une boucle
            // d'agent renvoie tout l'historique à chaque tour et n'ajoute
            // que quelques centaines de jetons à la fin d'un prompt par
            // ailleurs identique — le cas idéal du cache de préfixe, mesuré
            // à 86 % du temps d'une boucle perdu en préfill sans lui (PLAN.md
            // P13.2).
            let (usePersistentCache, cacheRestored, trackingID) = try await prepareConversation(
                id: conversationID,
                model: selectedModel,
                messages: prepared.messages,
                options: options)
            // P13.3 : la garde P13.2 (un tour se terminant par `role: tool`
            // repartait toujours en rejeu complet) n'est plus nécessaire —
            // `dispatchConversationTurn` continue maintenant n'importe quel
            // rôle de dernier message par différence de jetons plutôt qu'en
            // reconstruisant le tour à la main (voir le rapport PLAN.md
            // P13.3) et retombe elle-même sur un rejeu complet si le
            // suffixe ne peut pas être calculé sûrement.
            let (stream, dispatchedViaPersistentCache) = try await dispatchConversationTurn(
                usePersistentCache: usePersistentCache, messages: prepared.messages, options: options)
            // Ground truth for the GUI's "Cache" tri-state (P5.2): a request
            // that named a conversation with prior turns but still fell back
            // to a full stateless replay. Computed here, not from engine
            // metrics — see `cacheReplayed`'s doc comment. P6.1: `trackingID`
            // covers both an explicit `conversation_id` and the synthetic id
            // the implicit-prefix path hands out when there is none.
            let cacheReplayed = !dispatchedViaPersistentCache && trackingID != nil && prepared.messages.count > 1
            updateSession(id) { $0.cacheRestored = cacheRestored && dispatchedViaPersistentCache; $0.cacheReplayed = cacheReplayed }
            for url in prepared.temporaryFiles { try? FileManager.default.removeItem(at: url) }
            // Must mirror `options.enableThinking` exactly: the parser assumes the
            // prompt ends inside `<think>` only when thinking was rendered.
            let thinkingIsPrimed = options.enableThinking
            // Always pass the real `trackingID` (not gated on
            // `usePersistentCache`): `rememberConversation` itself decides
            // whether to append to the already-active conversation or
            // establish a brand new one from a replay's resulting state (the
            // "nouvel état après la réponse" half of P5.2 — see its doc
            // comment). P6.1: this is the client's `conversation_id` when it
            // gave one, or the synthetic id `prepareConversation` handed out
            // for an implicit-prefix match/miss otherwise.
            if input.stream == true {
                return try await makeStreamingResponse(
                    stream: stream, sessionID: id, model: selectedModel,
                    primedInside: thinkingIsPrimed,
                    trackingID: trackingID,
                    requestMessages: prepared.messages, options: options)
            }
            return try await makeJSONResponse(
                stream: stream, sessionID: id, model: selectedModel,
                primedInside: thinkingIsPrimed,
                trackingID: trackingID,
                requestMessages: prepared.messages, options: options)
        } catch {
            if input.effectiveConversationID != nil {
                await runtime.clearActiveFlashConversation()
                await runtime.resetConversation()
            }
            updateSession(id) { $0.status = .failed; $0.error = error.localizedDescription; $0.finishedAt = Date() }
            return Self.errorResponse(Self.status(for: error), message: error.localizedDescription)
        }
    }

    /// P12.3 : jumelle de `chatCompletionsResponse` pour `serve --batch-size
    /// N > 1` — un fichier séparé plutôt qu'un `if batchSize > 1` glissé
    /// dans la fonction d'aujourd'hui, précisément pour que celle-ci reste
    /// intouchée (voir `start`'s commentaire et le critère « bit-identique
    /// à batch-size 1 » de PLAN.md P12.3).
    ///
    /// Différences avec le chemin d'aujourd'hui :
    ///  - Une requête « chaude » (elle touche une conversation active ou
    ///    restaurable — `Qwen38Runtime.flashConversationCacheWouldHit`, une
    ///    sonde en LECTURE SEULE) ou porteuse d'une image suit exactement
    ///    la même logique que `chatCompletionsResponse`
    ///    (`prepareConversation` puis `generate`/`generateStateless`), mais
    ///    sous `batchExecutionLock` plutôt que `queue` — un verrou séparé
    ///    tenu jusqu'à la fin RÉELLE de la génération (voir
    ///    `qwen38AttachCompletionGate`), pas seulement jusqu'à la
    ///    construction de la réponse HTTP, pour rester mutuellement exclusif
    ///    avec un lot en cours.
    ///  - Une requête « froide » rejoint `batchCoordinator` : si elle se
    ///    retrouve seule à l'issue de la fenêtre de regroupement (`.solo`),
    ///    elle est traitée exactement comme une requête chaude (même
    ///    `prepareConversation`, donc éligible à devenir une conversation
    ///    active pour un futur tour) ; si elle rejoint un lot (`.batched`),
    ///    elle ne touche JAMAIS `prepareConversation`/`rememberConversation`
    ///    — `trackingID` reste `nil` (voir PLAN.md P12.3, « le lot et le
    ///    cache de conversations sont incompatibles »).
    private func chatCompletionsResponseBatched(request: Request) async throws -> Response {
        try authorize(request); var request = request; let buffer = try await request.collectBody(upTo: 64 * 1024 * 1024)
        guard let data = buffer.getData(at: buffer.readerIndex, length: buffer.readableBytes) else { throw Qwen38ServerError.invalidRequest("Le corps JSON est vide.") }
        let input: ChatCompletionRequest
        do { input = try JSONDecoder().decode(ChatCompletionRequest.self, from: data) } catch { throw Qwen38ServerError.invalidRequest("Requête chat invalide : \(error.localizedDescription)") }
        guard !input.messages.isEmpty else { throw Qwen38ServerError.invalidRequest("La requête doit contenir au moins un message.") }
        // P13.1 / défaut du 2026-09-15 : voir le même commentaire dans
        // `chatCompletionsResponse`.
        guard let lastMessageRole = input.messages.last?.role,
              lastMessageRole == "user" || lastMessageRole == "tool" || lastMessageRole == "assistant"
        else { throw Qwen38ServerError.invalidRequest("Le dernier message doit avoir le rôle user, tool (après un appel d'outil), ou assistant (pour continuer un tour interrompu).") }
        let requestedTools = Self.orderedToolSpecs(input.effectiveTools.map { $0.toSpec() }, body: data)
        guard requestedTools.allSatisfy({ !$0.name.isEmpty }) else { throw Qwen38ServerError.invalidRequest("Chaque outil déclaré dans tools doit avoir un nom (function.name).") }
        let requestedModel = input.model?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        let id = UUID(); sessions[id] = .init(id: id, client: "LAN", path: "/v1/chat/completions", model: requestedModel ?? loadedModel ?? "default", conversationID: input.effectiveConversationID); sessionOrder.append(id); trimSessions(); updateSession(id) { $0.status = .running }
        do {
            await batchExecutionLock.acquire()
            let selectedModel: String
            do { selectedModel = try await ensureModelLoaded(requestedModel) }
            catch { await batchExecutionLock.release(); throw error }
            await batchExecutionLock.release()
            updateSession(id) { $0.model = selectedModel }
            // P13.1/P14.4 : voir le commentaire jumeau dans
            // chatCompletionsResponse (non-stream) — même gabarit partagé,
            // même garde.
            if !requestedTools.isEmpty {
                guard await runtime.isLoaded else {
                    throw Qwen38ServerError.invalidRequest("Le champ tools nécessite qu'un modèle soit chargé.")
                }
            }
            let prepared = try prepare(input.messages)
            let temperature = input.temperature ?? 0
            let presencePenalty = input.explicitPresencePenalty ?? (temperature > 0 ? 1.5 : 0)
            let requestedAblation: Qwen4ExpLayerBenchAblation?
            if let rawAblation = input.effectiveAblation {
                guard allowAblation else {
                    throw Qwen38ServerError.invalidRequest(
                        "Le champ ablation est refusé : redémarrez le serveur avec --allow-ablation pour l'autoriser. L'ablation produit des sorties numériquement fausses par construction.")
                }
                requestedAblation = try qwen4ExpResolveAblation(rawValue: rawAblation)
            } else {
                requestedAblation = nil
            }
            let options = Qwen38GenerationOptions(maxTokens: try Self.resolvedMaxTokens(input.effectiveMaxTokens), temperature: temperature, topP: input.topP ?? 0.95, enableThinking: input.effectiveThinking ?? (input.effectiveReasoningEffort != nil || defaultEnableThinking), reasoningEffort: input.effectiveReasoningEffort ?? "low", mtp: .init(enabled: input.effectiveMTP ?? true, draftDepth: .fixed(input.effectiveMTPDraftTokens), engine: input.effectiveMTPEngine), presencePenalty: presencePenalty, repetitionPenalty: input.effectiveRepetitionPenalty, penaltyContextTokens: max(0, input.effectivePenaltyContextTokens ?? 2048), routedExpertCount: input.effectiveRoutedExperts, ablation: requestedAblation, tools: requestedTools)
            let conversationID = input.effectiveConversationID

            // P12.3 : une image, ou une requête qui touche le cache de
            // conversations/de préfixe, suit toujours le chemin
            // chaud/mono-séquence — jamais le coordinateur de lot. Voir le
            // commentaire de fonction.
            let hasImages = prepared.messages.contains { !$0.imageURLs.isEmpty }
            let flashNextLoaded = await runtime.isFlashNextLoaded
            // P13.1 : une requête porteuse d'outils suit toujours le chemin
            // chaud/mono-séquence ci-dessous (`.solo`), jamais le
            // coordinateur de lot — voir le rapport à Vincent : ce choix
            // simple garantit que `tools` ne peut jamais interagir avec le
            // regroupement.
            var isEligibleForBatching = !hasImages && requestedTools.isEmpty && batchCoordinator != nil && flashNextLoaded
            if isEligibleForBatching {
                isEligibleForBatching = !(await runtime.flashConversationCacheWouldHit(
                    id: conversationID, model: selectedModel, messages: prepared.messages, options: options))
            }

            var joinResult: Qwen38BatchJoinResult = .solo
            if isEligibleForBatching, let batchCoordinator {
                let promptTokenCount = ((try? await runtime.renderedFlashTokenIDs(
                    messages: prepared.messages, options: options)).flatMap { $0 })?.count
                // Défaut A (2026-09-14) : le préfill est dense — le
                // regrouper additionne le travail de chaque ligne (plus le
                // remplissage) sans aucun recouvrement, contrairement au
                // décodage. Une requête dont le prompt rendu dépasse
                // `batchMaxPromptTokens` (`serve --batch-max-prompt-tokens`,
                // défaut 256, voir sa doc comment) n'est donc JAMAIS
                // proposée au coordinateur : elle garde le chemin
                // chaud/mono-séquence ci-dessous (`.solo`), avec son TTFT
                // d'aujourd'hui — mesuré : 37,20 s en lot de 4 contre
                // 7,53 s seul sur des prompts d'environ 1 200 jetons.
                if let promptTokenCount, promptTokenCount <= batchMaxPromptTokens {
                    joinResult = try await batchCoordinator.join(
                        promptTokenCount: promptTokenCount,
                        request: .init(messages: prepared.messages, options: options))
                }
            }

            let thinkingIsPrimed = options.enableThinking
            switch joinResult {
            case .batched(let rawStream, let batchSizeServed):
                // Jamais mémorisé — voir le commentaire de fonction et
                // `Qwen4ExpBatchStreamingGenerator`, « toujours stateless ».
                updateSession(id) { $0.batchSizeServed = batchSizeServed }
                for url in prepared.temporaryFiles { try? FileManager.default.removeItem(at: url) }
                if input.stream == true {
                    return try await makeStreamingResponse(
                        stream: rawStream, sessionID: id, model: selectedModel,
                        primedInside: thinkingIsPrimed, trackingID: nil,
                        requestMessages: prepared.messages, options: options)
                }
                return try await makeJSONResponse(
                    stream: rawStream, sessionID: id, model: selectedModel,
                    primedInside: thinkingIsPrimed, trackingID: nil,
                    requestMessages: prepared.messages, options: options)

            case .solo:
                // Exactement la logique de `chatCompletionsResponse` (une
                // requête chaude, une image, ou une requête froide restée
                // seule à l'issue de la fenêtre de regroupement) — sous
                // `batchExecutionLock`, tenu jusqu'à la fin réelle de la
                // génération plutôt que jusqu'à la réponse HTTP (voir le
                // commentaire de fonction).
                await batchExecutionLock.acquire()
                let usePersistentCache: Bool
                let cacheRestored: Bool
                let trackingID: String?
                do {
                    // P13.2 : voir le même commentaire dans
                    // `chatCompletionsResponse` — une requête outillée
                    // touche désormais le cache de conversation comme
                    // n'importe quelle autre.
                    (usePersistentCache, cacheRestored, trackingID) = try await prepareConversation(
                        id: conversationID, model: selectedModel, messages: prepared.messages, options: options)
                } catch {
                    await batchExecutionLock.release()
                    throw error
                }
                // P13.3 : même dispatch que `chatCompletionsResponse` — la
                // garde P13.2 sur un tour se terminant par `role: tool`
                // n'est plus nécessaire, voir son commentaire.
                let rawStream: AsyncThrowingStream<Qwen38GenerationEvent, Error>
                let dispatchedViaPersistentCache: Bool
                do {
                    (rawStream, dispatchedViaPersistentCache) = try await dispatchConversationTurn(
                        usePersistentCache: usePersistentCache, messages: prepared.messages, options: options)
                } catch {
                    await batchExecutionLock.release()
                    throw error
                }
                let cacheReplayed = !dispatchedViaPersistentCache && trackingID != nil && prepared.messages.count > 1
                updateSession(id) { $0.cacheRestored = cacheRestored && dispatchedViaPersistentCache; $0.cacheReplayed = cacheReplayed; $0.batchSizeServed = 1 }
                for url in prepared.temporaryFiles { try? FileManager.default.removeItem(at: url) }
                // Sûr ici (contrairement au lot, voir le commentaire de
                // `Qwen38BatchCompletionGate`) : `Qwen4ExpStreamingGenerator.
                // run()` ne touche plus jamais `model` après avoir yield
                // `.finished` — rien à attendre de plus que la clôture du
                // flux avant de relâcher le verrou.
                let gate = Qwen38BatchCompletionGate(rowCount: 1) { [batchExecutionLock] in
                    Task { await batchExecutionLock.release() }
                }
                let stream = qwen38AttachCompletionGate(rawStream, gate: gate)
                if input.stream == true {
                    return try await makeStreamingResponse(
                        stream: stream, sessionID: id, model: selectedModel,
                        primedInside: thinkingIsPrimed, trackingID: trackingID,
                        requestMessages: prepared.messages, options: options)
                }
                return try await makeJSONResponse(
                    stream: stream, sessionID: id, model: selectedModel,
                    primedInside: thinkingIsPrimed, trackingID: trackingID,
                    requestMessages: prepared.messages, options: options)
            }
        } catch {
            if input.effectiveConversationID != nil {
                await runtime.clearActiveFlashConversation()
                await runtime.resetConversation()
            }
            updateSession(id) { $0.status = .failed; $0.error = error.localizedDescription; $0.finishedAt = Date() }
            return Self.errorResponse(Self.status(for: error), message: error.localizedDescription)
        }
    }

    /// P6.4: thin forwarder — the actual LRU lives on `Qwen38Runtime` now
    /// (shared with the GUI). Kept with this signature so existing call
    /// sites and tests are unaffected by the move.
    func prepareConversation(
        id: String?,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> (usePersistentCache: Bool, cacheRestored: Bool, trackingID: String?) {
        try await runtime.prepareFlashConversation(id: id, model: model, messages: messages, options: options)
    }

    /// P6.4: thin forwarder — see `prepareConversation` above. P13.2:
    /// `assistantToolCalls` threads through to `Qwen38Runtime.
    /// rememberFlashConversation` — see its doc comment.
    func rememberConversation(
        id: String,
        model: String,
        requestMessages: [Qwen38ChatMessage],
        assistantContent: String,
        assistantToolCalls: [Qwen38ToolCall] = [],
        options: Qwen38GenerationOptions
    ) async {
        await runtime.rememberFlashConversation(
            id: id, model: model, requestMessages: requestMessages,
            assistantContent: assistantContent, assistantToolCalls: assistantToolCalls, options: options)
    }

    /// P13.3 : dispatch commun aux deux chemins serveur
    /// (`chatCompletionsResponse`, `chatCompletionsResponseBatched`) pour un
    /// tour dont `prepareConversation` a déjà décidé qu'il PEUT emprunter le
    /// cache persistant (`usePersistentCache`). Remplace l'ancienne garde
    /// « un tour se terminant par `role: tool` repart toujours en rejeu
    /// complet » (P13.2) : elle n'est plus nécessaire —
    /// `runtime.generateFlashConversationTurn` continue maintenant n'importe
    /// quel rôle de dernier message par différence de jetons plutôt qu'en
    /// reconstruisant le tour à la main (voir le rapport PLAN.md P13.3), et
    /// retombe elle-même sur un rejeu complet si le suffixe ne peut pas être
    /// calculé sûrement (`Qwen38FlashNextEngineError.
    /// continuationSuffixUnavailable` — historique divergent, options
    /// incompatibles, ou rien de nouveau à préfiller).
    ///
    /// La famille 27B (`chatSession`) n'a pas d'équivalent à
    /// `generateFlashConversationTurn` — elle garde `generate(prompt:...)`,
    /// inchangé.
    ///
    /// Le second membre du résultat dit ce qui a RÉELLEMENT été utilisé
    /// (jamais `usePersistentCache` tel quel) pour que l'appelant reporte
    /// une télémétrie de cache honnête même après un tel repli.
    private func dispatchConversationTurn(
        usePersistentCache: Bool,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> (
        stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, usedPersistentCache: Bool
    ) {
        guard usePersistentCache, let last = messages.last else {
            return (try await runtime.generateStateless(messages: messages, options: options), false)
        }
        guard await runtime.isFlashNextLoaded else {
            let systemPrompt = messages.first(where: { $0.role == .system })?.content
            let stream = try await runtime.generate(
                prompt: last.content, systemPrompt: systemPrompt, imageURLs: last.imageURLs,
                options: options)
            return (stream, true)
        }
        do {
            let stream = try await runtime.generateFlashConversationTurn(
                messages: messages, options: options)
            return (stream, true)
        } catch Qwen38FlashNextEngineError.continuationSuffixUnavailable {
            return (try await runtime.generateStateless(messages: messages, options: options), false)
        }
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
        // A different resident model invalidates every exported state: they
        // reference the previous decoder's own caches (§5.1.1, one model
        // resident at a time).
        await runtime.discardFlashConversationCache()
        try await runtime.load(
            from: selection.1, preloadMTP: true, routedExpertCount: startupRoutedExpertCount)
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
            let role: Qwen38ChatMessage.Role; switch message.role { case "system": role = .system; case "assistant": role = .assistant; case "tool": role = .tool; default: role = .user }
            // P13.1 : un tour assistant qui a appelé un outil — rejoué tel
            // quel au tour suivant (voir `Qwen4ExpPromptBuilder.hfMessage`).
            let toolCalls: [Qwen38ToolCall] = (message.toolCalls ?? []).map { raw in
                Qwen38ToolCall(
                    id: raw.id ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                    name: raw.function.name, argumentsJSON: raw.function.arguments ?? "{}")
            }
            result.append(.init(role: role, content: text, imageURLs: images, toolCalls: toolCalls))
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

    private func makeJSONResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, trackingID: String?, requestMessages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) async throws -> Response {
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
        // P13.1 : n'analyse les `<tool_call>` que si CETTE requête a déclaré
        // des outils — sans quoi une réponse ordinaire qui contiendrait par
        // hasard ce texte (hallucination) resterait un `content` brut,
        // comportement strictement inchangé sans `tools` (critère PLAN.md
        // P13.1).
        // P13.2 : ce parsing doit avoir lieu AVANT de mémoriser le tour dans
        // le ledger (déplacé plus bas, cf. avant P13.2 il avait lieu après)
        // — sinon le tour assistant mémorisé aurait gardé le XML
        // `<tool_call>` brut comme `content`, alors que le PROCHAIN tour
        // reconstruit ce même tour assistant depuis les `tool_calls[]`
        // structurés que le client renvoie (voir `Qwen38Server.prepare`) :
        // deux représentations différentes du même tour ne rendent pas les
        // mêmes jetons, ce qui aurait fait manquer à tort la comparaison de
        // préfixe P6.1 sur le tour suivant.
        var finishReason = Self.finishReason(metrics?.stopReason)
        var toolCallsOut: [ChatCompletionToolCallOut]? = nil
        var rememberedToolCalls: [Qwen38ToolCall] = []
        if !options.tools.isEmpty {
            let parsed = Qwen38ToolCallParser.parse(text)
            if !parsed.calls.isEmpty {
                text = parsed.content
                let built = parsed.calls.map { call -> (out: ChatCompletionToolCallOut, remembered: Qwen38ToolCall) in
                    let schema = options.tools.first(where: { $0.name == call.name })?.parameters
                    let argumentsJSON = Qwen38ToolArgumentTyper.typedArguments(call.parameters, schema: schema).toJSONString()
                    let callID = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                    return (
                        ChatCompletionToolCallOut(index: nil, id: callID, type: "function", function: .init(name: call.name, arguments: argumentsJSON)),
                        Qwen38ToolCall(id: callID, name: call.name, argumentsJSON: argumentsJSON))
                }
                toolCallsOut = built.map(\.out)
                rememberedToolCalls = built.map(\.remembered)
                // Un appel tronqué par max_tokens ne devient jamais un appel
                // malformé (voir `Qwen38ToolCallParser`) ; symétriquement,
                // un finish_reason "length" reste "length" même si un ou
                // plusieurs appels complets ont pu être reconnus avant la
                // coupure — jamais "tool_calls" dans ce cas.
                if finishReason != "length" { finishReason = "tool_calls" }
            }
        }
        if let trackingID {
            await rememberConversation(id: trackingID, model: model, requestMessages: requestMessages, assistantContent: text, assistantToolCalls: rememberedToolCalls, options: options)
        }
        return Self.jsonResponse(ChatCompletionResponse(id: "chatcmpl-\(sessionID.uuidString)", object: "chat.completion", created: Int(Date().timeIntervalSince1970), model: model, choices: [.init(index: 0, message: .init(role: "assistant", content: text, reasoningContent: reasoning.nilIfEmpty, toolCalls: toolCallsOut), delta: nil, finishReason: finishReason)], usage: metrics.map(ChatCompletionUsage.init(metrics:))))
    }
    private func makeStreamingResponse(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, sessionID: UUID, model: String, primedInside: Bool, trackingID: String?, requestMessages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) async throws -> Response {
        // P13.1 : quand la requête porte des outils, le `content` n'est
        // jamais diffusé morceau par morceau — un `<tool_call>` XML
        // pourrait sinon apparaître tel quel dans le flux avant d'être
        // reconnu comme un appel, dupliqué avec le `tool_calls` structuré
        // émis ensuite. Il est mis en mémoire tampon et livré en un seul
        // fragment une fois la génération terminée (voir plus bas) — un
        // choix simple et sûr, signalé à Vincent : la diffusion incrémentale
        // du texte "réponse" est sacrifiée sur un tour porteur d'outils
        // (généralement court), jamais sur un tour ordinaire. Le
        // raisonnement (`<think>`), jamais concerné par ce marqueur,
        // continue de s'afficher au fil de l'eau dans tous les cas.
        let hasTools = !options.tools.isEmpty
        let body = ResponseBody { writer in
            func writeDelta(content: String? = nil, reasoning: String? = nil, toolCalls: [ChatCompletionToolCallOut]? = nil, finishReason: String? = nil, usage: ChatCompletionUsage? = nil) async throws {
                let value = ChatCompletionResponse(
                    id: "chatcmpl-\(sessionID.uuidString)", object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970), model: model,
                    choices: [.init(index: 0, message: nil, delta: .init(role: nil, content: content, reasoningContent: reasoning, toolCalls: toolCalls), finishReason: finishReason)],
                    usage: usage)
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
                        if !hasTools, !output.content.isEmpty { try await writeDelta(content: output.content) }
                    case .event(.metrics(let metrics)):
                        await self.completeSessionAsync(sessionID, metrics: metrics)
                        let tail = parser.finish()
                        responseContent += tail.content
                        if !tail.reasoning.isEmpty { try await writeDelta(reasoning: tail.reasoning) }
                        if !hasTools, !tail.content.isEmpty { try await writeDelta(content: tail.content) }

                        var finishReason = Self.finishReason(metrics.stopReason)
                        // P13.2 : mêmes calls que `makeJSONResponse` — voir
                        // son commentaire sur pourquoi le ledger doit garder
                        // les `tool_calls` structurés plutôt que le XML brut.
                        var rememberedToolCalls: [Qwen38ToolCall] = []
                        if hasTools {
                            let parsed = Qwen38ToolCallParser.parse(responseContent)
                            if !parsed.content.isEmpty { try await writeDelta(content: parsed.content) }
                            if !parsed.calls.isEmpty {
                                let built = parsed.calls.enumerated().map { index, call -> (out: ChatCompletionToolCallOut, remembered: Qwen38ToolCall) in
                                    let schema = options.tools.first(where: { $0.name == call.name })?.parameters
                                    let argumentsJSON = Qwen38ToolArgumentTyper.typedArguments(call.parameters, schema: schema).toJSONString()
                                    let callID = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                                    return (
                                        ChatCompletionToolCallOut(index: index, id: callID, type: "function", function: .init(name: call.name, arguments: argumentsJSON)),
                                        Qwen38ToolCall(id: callID, name: call.name, argumentsJSON: argumentsJSON))
                                }
                                let toolCallDeltas = built.map(\.out)
                                rememberedToolCalls = built.map(\.remembered)
                                // Livré en un seul fragment complet plutôt
                                // que morceau par morceau — voir le
                                // commentaire de fonction et le rapport à
                                // Vincent.
                                try await writeDelta(toolCalls: toolCallDeltas)
                                if finishReason != "length" { finishReason = "tool_calls" }
                            }
                            responseContent = parsed.content
                        }
                        try await writeDelta(finishReason: finishReason, usage: ChatCompletionUsage(metrics: metrics))
                        if let trackingID {
                            await self.rememberConversation(
                                id: trackingID, model: model,
                                requestMessages: requestMessages,
                                assistantContent: responseContent, assistantToolCalls: rememberedToolCalls, options: options)
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
    private func completeSession(_ id: UUID, metrics: Qwen38RunMetrics) {
        // Une ligne par requête sur stderr : de quoi additionner une session
        // d'agent (entrée, sortie, part en cache) et la chiffrer au tarif d'un
        // modèle du marché — voir Scripts/pi-session-cost.py.
        if let session = sessions[id] {
            let usage = ChatCompletionUsage(metrics: metrics)
            let line = "qwen38 serve · usage · client \(session.client) · prompt \(usage.promptTokens) jetons (dont \(usage.promptTokensDetails.cachedTokens) en cache) · sortie \(usage.completionTokens) jetons · \(String(format: "%.1f", metrics.metrics.generationTokensPerSecond)) tok/s\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        updateSession(id) { $0.status = .completed; $0.finishedAt = Date(); $0.promptTokens = metrics.metrics.promptTokens; $0.cachedPromptTokens = metrics.cachedPromptTokens; $0.generatedTokens = metrics.metrics.generatedTokens; $0.timeToFirstToken = metrics.timeToFirstToken; $0.tokensPerSecond = metrics.metrics.generationTokensPerSecond; $0.inputDescription = metrics.inputDescription; $0.cacheReused = metrics.cacheReused; $0.conversationReplayed = metrics.conversationReplayed; $0.mtp = Self.mtpLabel(metrics.mtpStatus); $0.mtpProposed = metrics.mtpStatus.proposedTokens; $0.mtpAccepted = metrics.mtpStatus.acceptedTokens; $0.mtpAcceptRate = metrics.mtpStatus.acceptanceRate; $0.routedExpertCount = metrics.routedExpertCount; $0.ablation = metrics.ablation } }
    private func completeSessionAsync(_ id: UUID, metrics: Qwen38RunMetrics) { completeSession(id, metrics: metrics) }
    private func failSessionAsync(_ id: UUID, error: String) { updateSession(id) { $0.status = .failed; $0.error = error; $0.finishedAt = Date() } }
    private func trimSessions() { while sessionOrder.count > 32 { sessions.removeValue(forKey: sessionOrder.removeFirst()) } }
    /// Un `max_tokens` (ou `max_completion_tokens`) nul ou négatif est refusé
    /// en HTTP 400 plutôt que ramené à 1. pi — et tout client qui calcule
    /// « fenêtre − prompt estimé » — envoie 0 ou moins quand son prompt
    /// déborde la fenêtre qu'il s'est fixée. Générer un seul jeton puis
    /// répondre `finish_reason: "length"` ressemblait à une réponse tronquée :
    /// le client compactait, la compaction était elle-même surdimensionnée,
    /// et la boucle n'avait pas de sortie (issue #1). Le message contient
    /// volontairement « exceeds the context window » : c'est un des motifs
    /// que pi reconnaît comme débordement de contexte, ce qui l'aiguille
    /// vers sa récupération bornée (une compaction, un seul réessai).
    /// Rattache à chaque outil sa forme JSON telle que le client l'a
    /// envoyée, ordre des membres compris : `JSONDecoder` perd cet ordre,
    /// or le gabarit rend `{{ tool | tojson }}` et transformers — le rendu
    /// d'entraînement du modèle — écrit les membres dans l'ordre reçu. Le
    /// corps est relu une fois avec `Qwen38OrderedJSON` ; si sa liste
    /// `tools` ne correspond pas (absente, autre longueur), on garde les
    /// spécifications décodées, rendues alors à clés triées.
    static func orderedToolSpecs(_ specs: [Qwen38ToolSpec], body: Data) -> [Qwen38ToolSpec] {
        guard !specs.isEmpty, let root = try? Qwen38OrderedJSON.parse(body),
            let ordered = root["tools"]?.arrayValue, ordered.count == specs.count
        else { return specs }
        return zip(specs, ordered).map { spec, entry in
            Qwen38ToolSpec(
                type: spec.type, name: spec.name, description: spec.description,
                parameters: spec.parameters, orderedSpec: entry)
        }
    }

    static func resolvedMaxTokens(_ requested: Int?) throws -> Int {
        guard let requested else { return 256 }
        guard requested > 0 else {
            throw Qwen38ServerError.invalidRequest(
                "max_tokens must be positive (got \(requested)): the prompt exceeds the context window the client budgeted for it. Shorten the conversation or raise the client's contextWindow.")
        }
        return min(requested, 131_072)
    }
    private static func finishReason(_ reason: Any?) -> String { guard let reason else { return "stop" }; return String(describing: reason).lowercased().contains("length") ? "length" : "stop" }
    private static func mtpLabel(_ status: Qwen38MTPRunStatus) -> String { switch status.availability { case .active: return "actif"; case .unavailable: return "indisponible"; case .fallback(let reason): return "fallback: \(reason)" } }
    private static func jsonResponse<T: Encodable>(_ value: T) -> Response { let data = (try? JSONEncoder().encode(value)) ?? Data(); var buffer = ByteBufferAllocator().buffer(capacity: data.count); buffer.writeBytes(data); var headers = HTTPFields(); headers[.contentType] = "application/json; charset=utf-8"; return .init(status: .ok, headers: headers, body: .init(byteBuffer: buffer)) }
    private static func errorResponse(_ status: HTTPResponse.Status, message: String) -> Response {
        var response = jsonResponse(ErrorResponse(error: .init(message: message, type: status.code >= 500 ? "server_error" : "invalid_request_error", code: nil)))
        response.status = status
        return response
    }

    /// Défaut du 2026-09-15 : le filet posé par T8 (le `catch` interne de
    /// `chatCompletionsResponse`/`chatCompletionsResponseBatched`, plus bas)
    /// ne couvrait que les erreurs levées APRÈS son `do {` — les gardes de
    /// validation placées avant (corps vide, liste de messages vide, dernier
    /// message d'un rôle refusé, `tools[].name` manquant, `authorize` sur
    /// `v1/models`…) s'échappaient telles quelles jusqu'au routeur
    /// Hummingbird. `RouterResponder.respond` (voir
    /// `.build/checkouts/hummingbird/Sources/Hummingbird/Router/
    /// RouterResponder.swift`) ne rattrape que les erreurs conformes à
    /// `HTTPResponseError` ; tout le reste retombe sur le filet générique
    /// d'`Application.run()`, qui répond `Response(status: .internalServerError,
    /// body: .init())` — un HTTP 500 au corps vide, indiagnosticable côté
    /// client. Reproduit avec un dernier message `assistant` après un aller-
    /// retour d'outil complet (voir le test dédié).
    ///
    /// Plutôt que d'étendre le `do/catch` interne de chaque gestionnaire (un
    /// correctif au cas par cas, qui laisserait le même piège ouvert à toute
    /// future garde ou tout futur point d'entrée), ce filet s'enregistre UNE
    /// SEULE FOIS à l'endroit où chaque route est déclarée (`start()`) : il
    /// capture absolument toute erreur Swift qui s'échapperait du
    /// gestionnaire — gardes de validation comprises — et la transforme via
    /// `status(for:)`/`errorResponse` en réponse JSON de style OpenAI, avec
    /// un statut et un corps toujours présents. Le `do/catch` interne des
    /// deux gestionnaires `/v1/chat/completions` reste néanmoins en place :
    /// lui seul sait faire le ménage de session/conversation avant de
    /// renvoyer l'erreur (voir son propre commentaire) ; ce filet-ci ne fait
    /// que garantir qu'aucune erreur ne peut plus jamais ressortir sans
    /// corps, quel que soit l'endroit d'où elle est levée.
    private func catchingHTTPErrors(_ handler: () async throws -> Response) async -> Response {
        do { return try await handler() }
        catch { return Self.errorResponse(Self.status(for: error), message: error.localizedDescription) }
    }

    /// LAN test 2026-09-09 (T8): a `Qwen38ServerError` escaping the handler used
    /// to surface as an empty HTTP 500. Map it to an OpenAI-style JSON error
    /// with a meaningful status instead.
    private static func status(for error: any Error) -> HTTPResponse.Status {
        // Les refus explicites du moteur Flash-Next (image sur une
        // continuation, images multiples, image en mode stateless) sont des
        // erreurs de requête, pas des pannes : sans cela elles sortaient en
        // 500 sans corps (constaté le 2026-09-12 sur un tour 2 après image).
        if error is Qwen38FlashNextEngineError { return .badRequest }
        // P11.2 : une chaîne `ablation` invalide (ni un cas de
        // `Qwen4ExpLayerBenchAblation` ni "none") est une erreur de requête,
        // pas une panne serveur.
        if error is Qwen4ExpAblationResolutionError { return .badRequest }
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
