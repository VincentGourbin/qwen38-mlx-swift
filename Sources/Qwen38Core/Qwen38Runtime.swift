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
    /// P6.3: how many of the most recent **assistant-turn** tokens the
    /// presence/repetition mask is seeded with before the first token of a
    /// new turn, instead of starting empty every turn (`extra.
    /// penalty_context_tokens`). 0 restores the pre-P6.3 per-turn-only
    /// behavior. Only ever consulted alongside `presencePenalty`/
    /// `repetitionPenalty` (temperature > 0) — greedy decoding is
    /// unaffected regardless of this value.
    public var penaltyContextTokens: Int
    /// P11.1 : surcharge ponctuelle de la largeur de routage MoE
    /// (`num_experts_per_tok`) pour Flash-Next, appliquée à l'engin résident
    /// *avant* ce tour (`Qwen38Runtime.generate`/`generateStateless`) —
    /// `nil` (le défaut) laisse en vigueur le réglage de démarrage ou le
    /// dernier réglage explicite, sans jamais revenir silencieusement à la
    /// valeur du checkpoint. Non consulté par le chemin 27B. Le changement
    /// n'invalide pas le cache KV en cours (voir `Qwen38FlashNextEngine.
    /// setRoutedExpertCount`) : comparer des K différents sur la même
    /// conversation continuée mélangerait des tours calculés avec des
    /// largeurs différentes — repartir d'une conversation neuve pour
    /// chaque K du balayage P11.1.
    public var routedExpertCount: Int?
    /// P11.2 : surcharge ponctuelle de l'ablation (`Qwen4ExpLayerBenchAblation`)
    /// pour Flash-Next, appliquée à l'engin résident *avant* ce tour — même
    /// contrat que `routedExpertCount` : `nil` (le défaut) laisse en
    /// vigueur le réglage en cours (démarrage ou dernier réglage explicite),
    /// jamais un retour silencieux à `.none`. Non consulté par le chemin
    /// 27B. Comme pour `routedExpertCount`, changer l'ablation en cours de
    /// conversation continuée mélange des tours calculés différemment —
    /// repartir d'une conversation neuve. Le serveur ne peuple ce champ que
    /// lorsque `--allow-ablation` a été passé au démarrage ; sinon un champ
    /// de requête `ablation` est refusé en HTTP 400 avant d'atteindre ce
    /// point (voir `Qwen38InferenceServer`).
    public var ablation: Qwen4ExpLayerBenchAblation?
    /// P13.1 : outils OpenAI déclarés pour ce tour, Flash-Next uniquement —
    /// vide (le défaut) laisse `applyChatTemplate(tools:)` recevoir `nil`,
    /// comportement strictement inchangé. Le serveur ne peuple ce champ que
    /// lorsque la requête porte elle-même `tools`, et route alors la
    /// requête entière autour du cache de conversation Flash-Next (voir
    /// `Qwen38InferenceServer`) : ce champ n'entre donc jamais en jeu dans
    /// `Qwen38Runtime`'s prefix/LRU comparisons en pratique, mais reste
    /// comparé ci-dessous par défense en profondeur.
    public var tools: [Qwen38ToolSpec]

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
        repetitionPenalty: Float = 1.0,
        penaltyContextTokens: Int = 2048,
        routedExpertCount: Int? = nil,
        ablation: Qwen4ExpLayerBenchAblation? = nil,
        tools: [Qwen38ToolSpec] = []
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
        self.penaltyContextTokens = penaltyContextTokens
        self.routedExpertCount = routedExpertCount
        self.ablation = ablation
        self.tools = tools
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
    /// P11.1 : largeur de routage MoE effectivement utilisée par ce tour
    /// (`Qwen38FlashNextEngine.routedExpertCount` au moment de la requête),
    /// pour qu'un run ne puisse pas être attribué au mauvais K — voir
    /// PLAN.md P11.1. `nil` sur la famille 27B, qui n'a pas ce réglage.
    public let routedExpertCount: Int?
    /// P11.2 : ablation effectivement utilisée par ce tour
    /// (`Qwen4ExpLayerBenchAblation.rawValue`) — toujours publiée, `"none"`
    /// quand il n'y en a pas (y compris sur la famille 27B, qui n'a pas ce
    /// concept), pour la même raison que `routedExpertCount` : ne jamais
    /// laisser croire à tort qu'une mesure a été prise avec (ou sans)
    /// ablation.
    public let ablation: String

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
        mtpStatus: Qwen38MTPRunStatus = .init(availability: .unavailable),
        routedExpertCount: Int? = nil,
        ablation: String = "none"
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
        self.routedExpertCount = routedExpertCount
        self.ablation = ablation
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
    /// P13.1 : `.tool` s'ajoute à la famille — un message `role: "tool"`
    /// OpenAI (le résultat d'un appel), rendu par le gabarit du checkpoint
    /// comme un `<tool_response>` fusionné dans le tour utilisateur suivant
    /// (voir `chat_template.jinja`, aucune référence à `tool_call_id` : les
    /// réponses sont appariées par ordre, pas par identifiant).
    public enum Role: String, Sendable, Equatable { case system, user, assistant, tool }

    public let role: Role
    public let content: String
    public let imageURLs: [URL]
    /// P13.1 : uniquement significatif pour `role == .assistant` — les
    /// appels d'outils de ce tour, dans l'ordre OpenAI, rejoués dans le
    /// rendu du tour suivant comme autant de blocs `<tool_call>` (voir
    /// `Qwen4ExpPromptBuilder.hfMessage(from:)`). Vide pour un tour
    /// assistant ordinaire.
    public let toolCalls: [Qwen38ToolCall]

    public init(role: Role, content: String, imageURLs: [URL] = [], toolCalls: [Qwen38ToolCall] = []) {
        self.role = role
        self.content = content
        self.imageURLs = imageURLs
        self.toolCalls = toolCalls
    }
}

/// P12.3 : une ligne d'un lot soumis à `Qwen38FlashNextEngineProtocol.
/// generateBatch` — le pendant transport-neutre de `Qwen38ChatMessage`
/// pour l'ordonnanceur du serveur (`serve --batch-size N`). Chaque ligne
/// garde ses propres `options` (température, top-p/top-k, pénalités,
/// `maxTokens` — voir PLAN.md P12.3, « échantillonnage par séquence »).
public struct Qwen38BatchGenerationRequest: Sendable {
    public let messages: [Qwen38ChatMessage]
    public let options: Qwen38GenerationOptions

    public init(messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) {
        self.messages = messages
        self.options = options
    }
}

/// P12.3 (correctif du 2026-09-13, crash mémoire) : `completion` est le
/// SEUL signal sûr pour savoir qu'une exécution de lot a fini de toucher le
/// modèle résident — jamais la clôture des `streams`. Un appelant (le
/// coordinateur du serveur) qui libérerait un verrou d'exclusion mutuelle
/// dès que les flux sont consommés pourrait laisser un second lot démarrer
/// pendant que celui-ci exécute encore son propre nettoyage après avoir
/// déjà refermé ses flux — deux `resetConversation()` concurrents sur le
/// même modèle résident, cause vérifiée du crash (`EXC_BAD_ACCESS` dans les
/// couches résidentes). Voir `Qwen4ExpBatchStreamingGenerator.Result`, dont
/// ceci est le pendant côté `Qwen38FlashNextEngineProtocol`.
public struct Qwen38BatchGenerationResult: Sendable {
    public let streams: [AsyncThrowingStream<Qwen38GenerationEvent, Error>]
    public let completion: Task<Void, Never>

    public init(streams: [AsyncThrowingStream<Qwen38GenerationEvent, Error>], completion: Task<Void, Never>) {
        self.streams = streams
        self.completion = completion
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
    /// P6.4: the GUI's own Flash-Next conversation ledger — unlike the 27B
    /// `chatSession` path, the Flash-Next engine keeps no structured
    /// message history at all (only KV caches), so this is what
    /// `prepareFlashConversation`/`rememberFlashConversation` compare
    /// against and extend. Reset alongside `conversationTurnCount`.
    private var flashGUIMessages: [Qwen38ChatMessage] = []

    /// P6.4: `generate`'s streaming completion runs in a detached `Task`
    /// (needed for cancellation), which is not statically isolated to this
    /// actor — an ordinary property write from inside it needs an `await`
    /// through a method exactly like this one.
    private func setFlashGUIMessages(_ messages: [Qwen38ChatMessage]) {
        flashGUIMessages = messages
    }
    public private(set) var loadedDirectory: URL?

    // MARK: - P5.2/P6.1/P6.4: shared Flash-Next conversation LRU
    //
    // Moved here from `Qwen38Server` in P6.4 so the GUI (`generate`, below)
    // and the LAN server share the exact same cache instead of the GUI
    // reaching for a blunt "reset on turn 1" workaround while a LAN request
    // silently invalidates whatever the resident engine held. The server
    // stays the budget's owner (`configureConversationCacheBudget`, called
    // from `Qwen38InferenceServer.start`); this actor just owns the storage
    // and matching logic both callers need.
    //
    /// Only one target cache is resident. A conversation id makes that cache
    /// explicit: switching ids resets/replays rather than leaking one
    /// client's history into another request.
    private var activeConversationID: String?
    private var activeConversationModel: String?
    private var activeConversationMessages: [Qwen38ChatMessage] = []
    private var activeConversationOptions: Qwen38GenerationOptions?
    /// P5.2: LRU of exported Flash-Next conversation states that are *not*
    /// currently live in the resident engine (the live one stays only in
    /// `activeConversation*` above until a different id displaces it —
    /// exporting on every turn would be wasted `KVCache.copy()` work). Only
    /// ever populated while a Flash-Next model is resident (§5.1.1) — the
    /// 27B path never touches it.
    private struct CachedConversation {
        let model: String
        let options: Qwen38GenerationOptions
        let ledger: [Qwen38ChatMessage]
        let state: any Qwen38FlashConversationStateProtocol
    }
    private var conversationCache: [String: CachedConversation] = [:]
    /// Least-recently-used at the front, most-recently-used at the back.
    private var conversationCacheOrder: [String] = []
    /// Defaults to the server's own historical default (12 GB) so the GUI
    /// gets a usable cache even when the LAN server was never started;
    /// `configureConversationCacheBudget` lets the server override it.
    private var conversationCacheBudgetBytes: Int64 = 12 * 1024 * 1024 * 1024
    private var cacheMissCount = 0
    /// P6.1: implicit-prefix cache counters (no `conversation_id`) — see
    /// `Qwen38ServerSnapshot.prefixHits`/`prefixMisses`.
    private var prefixHitCount = 0
    private var prefixMissCount = 0
    private let flashNextEngineFactory: any Qwen38FlashNextEngineFactory
    private var flashEngine: (any Qwen38FlashNextEngineProtocol)?

    public init(flashNextEngineFactory: any Qwen38FlashNextEngineFactory = Qwen38DefaultFlashNextEngineFactory()) {
        self.flashNextEngineFactory = flashNextEngineFactory
    }

    public var isLoaded: Bool { container != nil || flashEngine != nil }

    public func load(
        from directory: URL,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        preloadMTP: Bool = true,
        /// P11.1 : option de démarrage — surcharge `num_experts_per_tok`
        /// pour tout l'engin Flash-Next résident (cible et drafter MTP).
        /// `nil` (le défaut) laisse le comportement inchangé. Sans effet
        /// sur la famille 27B.
        routedExpertCount: Int? = nil,
        /// P11.2 : option de démarrage — quel sous-bloc, le cas échéant,
        /// court-circuiter dans tout l'engin Flash-Next résident. `.none`
        /// (le défaut) laisse le comportement inchangé. Sans effet sur la
        /// famille 27B.
        ablation: Qwen4ExpLayerBenchAblation = .none
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
            flashEngine = try await flashNextEngineFactory.makeEngine(
                directory: directory, routedExpertCount: routedExpertCount, ablation: ablation)
            // PM4.3 (branchement, 2026-09-09): `mtpState` now delegates to
            // the loaded engine's own dynamic availability (predictor loads
            // lazily on the first MTP-enabled turn) instead of a fixed
            // snapshot taken here — see the `mtpState` getter below.
            mtpAvailability = .unavailable
        case .qwen35:
            await Qwen38MTPRegistration.register()
            if info.isBonsai2 {
                await Qwen38Bonsai2.register()
                // Belt and braces: the fused 4-way GDN projection cannot fuse
                // mixed packed/float projections anyway (FusedQuantizedLinear
                // returns ineligible), but never let it try on this checkpoint.
                setenv("MLX_QWEN_FOUR_GDN", "0", 1)
            }
            // The generic helper tries registered factories in order. The LLM
            // factory also accepts qwen3_5 and would silently load the text-only
            // implementation, dropping vision inputs. Select the VLM factory
            // explicitly so Qwen35.prepare() receives the processed image.
            container = try await VLMModelFactory.shared.loadContainer(
                from: directory,
                using: Qwen38TokenizerLoader()
            )
            if info.isBonsai2 {
                let installer = try Qwen38Bonsai2Loader(directory: directory)
                // `context.model` is a class reference: mutating it in place via
                // `perform` (not `update`) avoids capturing a `var` across the
                // `@Sendable` closure boundary just to read the count back.
                let replacedCount = await container!.perform { context in
                    installer.install(into: context.model)
                }
                guard replacedCount == installer.expectedReplacementCount else {
                    throw Qwen38Bonsai2LoaderError.unexpectedReplacementCount(
                        expected: installer.expectedReplacementCount, actual: replacedCount)
                }
            }
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
            flashGUIMessages = []
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

    /// P11.1 : largeur de routage MoE actuellement effective sur l'engin
    /// Flash-Next résident, pour `/healthz` et l'affichage GUI. `nil` quand
    /// aucun engin Flash-Next n'est chargé (y compris famille 27B).
    public var flashRoutedExpertCount: Int? { flashEngine?.routedExpertCount }

    /// P11.1 : change la largeur de routage MoE de l'engin résident sans
    /// recharger le checkpoint — no-op (renvoie `nil`) si aucun engin
    /// Flash-Next n'est chargé.
    @discardableResult
    public func setFlashRoutedExpertCount(_ override: Int?) throws -> Int? {
        try flashEngine?.setRoutedExpertCount(override)
    }

    /// P11.2 : ablation actuellement effective sur l'engin Flash-Next
    /// résident, pour `/healthz`. `nil` quand aucun engin Flash-Next n'est
    /// chargé (y compris famille 27B) — le serveur publie alors `"none"`
    /// directement, sans distinguer "pas de Flash-Next" de "pas
    /// d'ablation" (voir `Qwen38InferenceServer.healthResponse`).
    public var flashAblation: Qwen4ExpLayerBenchAblation? { flashEngine?.ablation }

    /// P11.2 : change l'ablation de l'engin résident sans recharger le
    /// checkpoint — no-op si aucun engin Flash-Next n'est chargé.
    public func setFlashAblation(_ new: Qwen4ExpLayerBenchAblation) {
        flashEngine?.setAblation(new)
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

    /// P6.1: exposes `Qwen38FlashNextEngineProtocol.renderedTokenIDs` to the
    /// server's implicit-prefix cache. `nil` when no Flash-Next model is
    /// resident (same scope restriction as the rest of the LRU, §5.1.1).
    public func renderedFlashTokenIDs(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> [Int32]? {
        try flashEngine?.renderedTokenIDs(messages: messages, options: options)
    }

    /// P13.3 : point d'entrée du serveur pour un tour de conversation
    /// Flash-Next à cache persistant, une fois `prepareFlashConversation`
    /// (via `Qwen38Server.prepareConversation`) déjà appelé pour CETTE
    /// requête — l'engin résident est donc déjà dans l'état qu'il a établi
    /// (conversation active confirmée, restaurée depuis le LRU, ou démarrée
    /// à froid). Délibérément distinct de `generate(prompt:...)` : celui-ci
    /// refait sa PROPRE préparation sous l'id interne partagé avec la GUI
    /// ("gui", voir son commentaire) et reconstruit `messages` à partir de
    /// son propre ledger interne — qui garde le XML brut d'un tour outillé,
    /// jamais reparsé en `tool_calls` structurés. Cette méthode-ci prend
    /// directement les `messages` du serveur (`Qwen38Server.prepare`, déjà
    /// reconstruits avec les `tool_calls` structurés — voir
    /// `rememberFlashConversation`'s commentaire P13.2), sans toucher au LRU
    /// ni à aucun ledger : elle se contente de faire avancer l'engin résident
    /// d'un tour. Le serveur reste seul responsable de mémoriser le résultat
    /// après coup (`rememberFlashConversation`, avec le vrai `trackingID`).
    ///
    /// À la continuation, le nouveau suffixe de jetons est calculé par
    /// différence entre deux rendus complets plutôt que reconstruit à la
    /// main (`Qwen4ExpPromptBuilder.continuationSuffix`, via
    /// `Qwen38FlashNextEngine.continueConversationTurn`) — ce qui couvre
    /// n'importe quel rôle de dernier message, `tool` compris (voir le
    /// rapport PLAN.md P13.3). Peut lever
    /// `Qwen38FlashNextEngineError.continuationSuffixUnavailable` : c'est au
    /// serveur de rattraper cette erreur précise et de retomber sur
    /// `generateStateless` (rejeu complet), jamais de la laisser échouer la
    /// requête.
    public func generateFlashConversationTurn(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) async throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        guard let flashEngine else { throw Qwen38RuntimeError.modelNotLoaded }
        // P11.1/P11.2 : même contrat que `generate(prompt:...)` — appliqué
        // avant de générer, jamais après.
        if let requestedRoutedExpertCount = options.routedExpertCount {
            try flashEngine.setRoutedExpertCount(requestedRoutedExpertCount)
        }
        if let requestedAblation = options.ablation {
            flashEngine.setAblation(requestedAblation)
        }
        return try flashEngine.continueConversationTurn(messages: messages, options: options)
    }

    /// P5.2: releases MLX's allocator cache after the server's LRU drops
    /// evicted conversation snapshots, so the device memory those
    /// `MLXArray`s held is actually returned to the system instead of
    /// sitting in MLX's buffer pool.
    public func clearMLXCache() {
        Memory.clearCache()
    }

    /// P6.4: sets the shared LRU's byte budget. Called by
    /// `Qwen38InferenceServer.start` (`--conversation-cache-gb`) — the
    /// server remains the budget's owner; the GUI never calls this and
    /// simply inherits whatever is configured (or this actor's own 12 GB
    /// default if the LAN server was never started). `<= 0` disables the
    /// LRU entirely (legacy single-active-conversation behavior).
    public func configureConversationCacheBudget(gb: Double) {
        conversationCacheBudgetBytes = gb > 0 ? Int64(gb * 1024 * 1024 * 1024) : 0
    }

    /// P6.4: read-only counters for `/metrics` and the GUI's session list —
    /// unchanged shape from the pre-move `Qwen38ServerSnapshot` fields.
    public func flashConversationCacheSnapshot() -> (
        cachedConversations: Int, cacheBytes: Int64, cacheBudgetBytes: Int64,
        cacheMisses: Int, prefixHits: Int, prefixMisses: Int
    ) {
        (
            conversationCache.count, totalCacheBytes(), conversationCacheBudgetBytes,
            cacheMissCount, prefixHitCount, prefixMissCount
        )
    }

    /// P12.3 : sonde en LECTURE SEULE — aucun effet de bord sur l'état de
    /// conversation actif ni sur le LRU, contrairement à
    /// `prepareFlashConversation`. Réservée au nouvel ordonnanceur de lot du
    /// serveur (`serve --batch-size N > 1`) pour classer une requête
    /// « chaude » (elle correspond à une conversation active ou restaurable)
    /// avant de décider si elle rejoint un lot — décider APRÈS aurait déjà
    /// déclenché l'effet de bord de démarrage de cache de
    /// `prepareFlashConversation`/`coldStartConversation` pour une requête
    /// finalement détournée vers le lot, laissant l'engin croire à tort
    /// qu'une conversation est active alors qu'elle n'a jamais tourné (voir
    /// PLAN.md P12.3 : « le lot et le cache de conversations sont
    /// incompatibles en l'état »).
    ///
    /// Duplique délibérément les conditions de correspondance de
    /// `prepareFlashConversation`/`prepareImplicitConversation` plutôt que de
    /// les appeler : ces méthodes commencent par muter l'état (export vers
    /// le LRU, `clearActiveConversation`) avant même de savoir si la requête
    /// est un succès ou un échec, ce qu'une sonde en lecture seule ne peut
    /// pas se permettre.
    public func flashConversationCacheWouldHit(
        id: String?,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) -> Bool {
        guard isFlashNextLoaded, conversationCacheBudgetBytes > 0 else { return false }
        if let id {
            if activeConversationID == id, activeConversationModel == model,
               activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true,
               messages.count == activeConversationMessages.count + 1,
               Array(messages.dropLast()) == activeConversationMessages {
                return true
            }
            if let cached = conversationCache[id], cached.model == model,
               cacheOptionsCompatible(cached.options, options),
               messages.count == cached.ledger.count + 1,
               Array(messages.dropLast()) == cached.ledger {
                return true
            }
            return false
        }
        // Chemin implicite (pas de `conversation_id`) : un succès suppose
        // des tours antérieurs à comparer — même garde que
        // `prepareImplicitConversation`.
        guard messages.count > 1 else { return false }
        let priorMessages = Array(messages.dropLast())
        guard
            let priorRenderedIDs = (try? renderedFlashTokenIDs(messages: priorMessages, options: options))
                .flatMap({ $0.isEmpty ? nil : $0 })
        else { return false }
        if activeConversationModel == model, activeConversationID != nil,
           activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true,
           !activeConversationMessages.isEmpty,
           let activeRenderedIDs = try? renderedFlashTokenIDs(
               messages: activeConversationMessages, options: options),
           activeRenderedIDs == priorRenderedIDs {
            return true
        }
        for candidateID in conversationCacheOrder.reversed() {
            guard let cached = conversationCache[candidateID], cached.model == model,
                  cacheOptionsCompatible(cached.options, options), !cached.ledger.isEmpty,
                  let candidateRenderedIDs = try? renderedFlashTokenIDs(
                      messages: cached.ledger, options: options),
                  candidateRenderedIDs == priorRenderedIDs
            else { continue }
            return true
        }
        return false
    }

    /// P12.3 : décodage en lot pour l'ordonnanceur du serveur — voir
    /// `Qwen38FlashNextEngineProtocol.generateBatch`. Chemin strictement
    /// séparé de `generate`/`generateStateless` : jamais de conversation
    /// active touchée, jamais de LRU consulté (voir
    /// `Qwen4ExpBatchStreamingGenerator`, « toujours stateless »). N'est
    /// atteint par le serveur qu'après vérification de `isFlashNextLoaded`
    /// et de `flashConversationCacheWouldHit` — l'erreur ci-dessous est un
    /// filet de sécurité, pas un chemin normal.
    public func generateBatchFlashConversations(
        requests: [Qwen38BatchGenerationRequest]
    ) throws -> Qwen38BatchGenerationResult {
        guard let flashEngine else {
            throw Qwen38RuntimeError.modelNotLoaded
        }
        return try flashEngine.generateBatch(requests: requests)
    }

    /// Invalidates every cached Flash-Next conversation state — called when
    /// a different model becomes resident (states reference the previous
    /// decoder's own caches, §5.1.1).
    public func discardFlashConversationCache() {
        clearActiveConversation()
        conversationCache.removeAll()
        conversationCacheOrder.removeAll()
    }

    /// Drops only the currently-*live* conversation (not the LRU) — used by
    /// the server's error path, which must not leave a half-finished
    /// request's id looking active for the next request.
    public func clearActiveFlashConversation() {
        clearActiveConversation()
    }

    /// P5.2/P6.1: returns whether this request can use the resident
    /// engine's persistent cache, and — when it can — whether that meant
    /// restoring a different conversation's exported state into the
    /// (single, §5.1.1) resident engine rather than continuing the
    /// conversation that was already live. `model` is an opaque caller-
    /// supplied compatibility key (the server's catalog id, or the GUI's
    /// loaded-directory name) — family/eligibility is decided by
    /// `isFlashNextLoaded`, not by looking `model` up anywhere.
    public func prepareFlashConversation(
        id: String?,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) async throws -> (usePersistentCache: Bool, cacheRestored: Bool, trackingID: String?) {
        guard let id else {
            // P6.1: no explicit `conversation_id` — the common case for
            // Open WebUI and plain OpenAI SDK clients, which resend the
            // whole history every turn instead of naming a conversation.
            // An explicit id always takes priority over this path (it is
            // only ever reached when `id == nil`).
            return prepareImplicitConversation(model: model, messages: messages, options: options)
        }
        // The LRU only ever manages Flash-Next conversations (contrat
        // §5.1.1 / PLAN.md P5 scope) — the 27B path, or an operator who set
        // `--conversation-cache-gb 0`, keeps the original single-active-
        // conversation behavior with no export/restore machinery at all.
        guard isFlashNextLoaded, conversationCacheBudgetBytes > 0 else {
            return (legacyPrepareConversation(id: id, model: model, messages: messages, options: options), false, id)
        }

        if activeConversationID == id, activeConversationModel == model,
           activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true,
           messages.count == activeConversationMessages.count + 1,
           Array(messages.dropLast()) == activeConversationMessages {
            return (true, false, id)
        }

        // A different conversation is about to become live: export the
        // current one into the LRU first (a no-op if none was active) so
        // switching back to it later can restore instead of replaying.
        if let previousID = activeConversationID, let previousModel = activeConversationModel,
           let previousOptions = activeConversationOptions {
            storeActiveConversationIntoLRU(id: previousID, model: previousModel, options: previousOptions)
        }
        clearActiveConversation()

        if let cached = conversationCache[id], cached.model == model,
           cacheOptionsCompatible(cached.options, options),
           messages.count == cached.ledger.count + 1,
           Array(messages.dropLast()) == cached.ledger {
            restoreFlashConversationState(cached.state)
            conversationCache.removeValue(forKey: id)
            conversationCacheOrder.removeAll { $0 == id }
            activeConversationID = id
            activeConversationModel = model
            activeConversationMessages = cached.ledger
            activeConversationOptions = options
            return (true, true, id)
        }

        // A new or non-contiguous session is deliberately cold. A short
        // system/user prompt can start a persistent cache directly; longer
        // histories use the stateless replay path and are not advertised as
        // cached because reconstructing assistant hidden states is impossible
        // without rerunning them.
        cacheMissCount += 1
        resetConversation()
        let started = coldStartConversation(id: id, model: model, messages: messages, options: options)
        return (started, false, id)
    }

    /// P6.1: attempts the same restore-instead-of-replay optimization as
    /// the explicit-`conversation_id` path above, but without a client-
    /// supplied key. Instead of a dictionary lookup, it renders the
    /// incoming request and every candidate ledger (active conversation,
    /// then the LRU) with the same `Qwen4ExpPromptBuilder` path and
    /// compares **rendered token IDs**, never message structs or text — a
    /// system-prompt edit or a client-truncated (sliding-window) history
    /// renders differently and is therefore a clean miss, not a wrong
    /// restore. Comparing `messages.dropLast()` against a candidate's own
    /// ledger (rather than the ledger against a prefix of the full
    /// request) sidesteps a subtlety of the chat template: a ledger always
    /// ends on an assistant turn, the full request always ends on the new
    /// user turn, and whatever priming tokens the template adds at the
    /// very end depends on that trailing role — comparing two renders that
    /// both end in the same role (the shared history, minus the new
    /// message) cancels that out on both sides instead of requiring the
    /// exact priming behavior to be known here. A hit is tracked under the
    /// matched entry's existing id (synthetic for a conversation that was
    /// itself found this way); a miss synthesizes a fresh internal id so
    /// `rememberFlashConversation` can register the resulting state for the
    /// *next* implicit turn to find, exactly like the explicit-id path's
    /// own "replay now, remember for next time" fallback.
    private func prepareImplicitConversation(
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) -> (usePersistentCache: Bool, cacheRestored: Bool, trackingID: String?) {
        guard isFlashNextLoaded, conversationCacheBudgetBytes > 0 else {
            clearActiveConversation()
            return (false, false, nil)
        }
        // `nil` when there is no prior turn to compare against (the very
        // first message of a conversation) or when rendering it failed
        // (e.g. an image in the history — H6.3's manual ChatML has no
        // multi-turn form): both cases skip straight to the cold-start
        // fallback below, same as the explicit-id path's catch-all branch.
        let priorMessages = messages.count > 1 ? Array(messages.dropLast()) : []
        let priorRenderedIDs: [Int32]? = priorMessages.isEmpty
            ? nil
            : (try? renderedFlashTokenIDs(messages: priorMessages, options: options))
                .flatMap { $0.isEmpty ? nil : $0 }

        if let priorRenderedIDs, activeConversationModel == model,
           let activeID = activeConversationID,
           activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true,
           !activeConversationMessages.isEmpty,
           let activeRenderedIDs = try? renderedFlashTokenIDs(
               messages: activeConversationMessages, options: options),
           activeRenderedIDs == priorRenderedIDs {
            prefixHitCount += 1
            return (true, false, activeID)
        }

        // A different conversation is about to become live: export the
        // current one first (a no-op if none was active), same as the
        // explicit-id path.
        if let previousID = activeConversationID, let previousModel = activeConversationModel,
           let previousOptions = activeConversationOptions {
            storeActiveConversationIntoLRU(id: previousID, model: previousModel, options: previousOptions)
        }
        clearActiveConversation()

        if let priorRenderedIDs {
            for candidateID in conversationCacheOrder.reversed() {
                guard let cached = conversationCache[candidateID], cached.model == model,
                      cacheOptionsCompatible(cached.options, options), !cached.ledger.isEmpty,
                      let candidateRenderedIDs = try? renderedFlashTokenIDs(
                          messages: cached.ledger, options: options),
                      candidateRenderedIDs == priorRenderedIDs else { continue }
                restoreFlashConversationState(cached.state)
                conversationCache.removeValue(forKey: candidateID)
                conversationCacheOrder.removeAll { $0 == candidateID }
                activeConversationID = candidateID
                activeConversationModel = model
                activeConversationMessages = cached.ledger
                activeConversationOptions = options
                prefixHitCount += 1
                return (true, true, candidateID)
            }
        }

        prefixMissCount += 1
        resetConversation()
        return implicitColdStart(model: model, messages: messages, options: options)
    }

    /// P6.1 miss path: synthesizes a fresh internal id (never sent to the
    /// client) so the resulting state — whether this turn starts a live
    /// cache directly (`coldStartConversation` below) or falls through to
    /// a stateless replay — can be registered by `rememberFlashConversation`
    /// for the next implicit turn to find, mirroring the explicit-id
    /// path's "sinon rejeu et nouvel état après la réponse" contract.
    private func implicitColdStart(
        model: String, messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) -> (usePersistentCache: Bool, cacheRestored: Bool, trackingID: String?) {
        let syntheticID = "auto:" + UUID().uuidString
        let started = coldStartConversation(
            id: syntheticID, model: model, messages: messages, options: options)
        return (started, false, syntheticID)
    }

    /// Shared cold-start gate for both the explicit-id and implicit-prefix
    /// paths: a short system/user-only prompt can start a persistent cache
    /// directly; anything else (an already multi-turn or non-user-final
    /// history) uses the stateless replay path — reconstructing assistant
    /// hidden states without rerunning them is impossible — and is not
    /// registered as active here (leaves `activeConversationID` untouched).
    private func coldStartConversation(
        id: String, model: String, messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) -> Bool {
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

    /// Legacy behavior (pre-P5.2 / LRU disabled / non-Flash-Next family):
    /// exactly one conversation's cache can ever be live; switching ids
    /// always resets and cold-starts, never restores.
    private func legacyPrepareConversation(
        id: String,
        model: String,
        messages: [Qwen38ChatMessage],
        options: Qwen38GenerationOptions
    ) -> Bool {
        let isContinuation = activeConversationID == id
            && activeConversationModel == model
            && activeConversationOptions.map({ cacheOptionsCompatible($0, options) }) == true
            && messages.count == activeConversationMessages.count + 1
            && Array(messages.dropLast()) == activeConversationMessages
        if isContinuation {
            return true
        }

        resetConversation()
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
            && a.tools == b.tools
    }

    private func clearActiveConversation() {
        activeConversationID = nil
        activeConversationModel = nil
        activeConversationMessages = []
        activeConversationOptions = nil
    }

    /// P5.2: exports the currently-live conversation's engine state
    /// (`exportFlashConversationState`, `KVCache.copy()` under the hood —
    /// P5.1) into the LRU, then evicts the oldest entries until the budget
    /// is met again.
    private func storeActiveConversationIntoLRU(
        id: String, model: String, options: Qwen38GenerationOptions
    ) {
        guard !activeConversationMessages.isEmpty,
              let exported = exportFlashConversationState(ledger: activeConversationMessages)
        else { return }
        conversationCache[id] = CachedConversation(
            model: model, options: options, ledger: activeConversationMessages, state: exported)
        conversationCacheOrder.removeAll { $0 == id }
        conversationCacheOrder.append(id)
        evictIfNeeded()
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
    private func evictIfNeeded() {
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
            clearMLXCache()
        }
    }

    /// P5.2/P6.1: registers the result of a turn (continuation, restore, or
    /// stateless replay) as the new live conversation for `id`, so the
    /// *next* turn for that id — explicit or, via P6.1's matching, implicit
    /// — can restore/continue instead of replaying again. See PLAN.md P5.2:
    /// "sinon rejeu et nouvel état après la réponse".
    /// P13.2 : `assistantToolCalls` (par défaut vide, donc sans effet sur les
    /// appelants existants) préserve la structure `tool_calls` du tour
    /// assistant dans le ledger mémorisé — sans cela, un tour outillé était
    /// mémorisé comme un simple `content` texte (le XML `<tool_call>` brut),
    /// alors que le PROCHAIN tour reconstruit ce même tour depuis ce que le
    /// client renvoie (`tool_calls[]` structuré, voir `Qwen38Server.prepare`
    /// et `Qwen4ExpPromptBuilder.hfMessage`) : les deux représentations ne
    /// rendent pas les mêmes jetons, ce qui aurait fait manquer à tort la
    /// comparaison de préfixe (P6.1) sur le tour suivant — jamais une
    /// mauvaise restauration (la comparaison se fait sur les jetons rendus,
    /// jamais sur les structs), seulement un manque évitable. Voir le
    /// rapport à Vincent (PLAN.md P13.2).
    public func rememberFlashConversation(
        id: String,
        model: String,
        requestMessages: [Qwen38ChatMessage],
        assistantContent: String,
        assistantToolCalls: [Qwen38ToolCall] = [],
        options: Qwen38GenerationOptions
    ) {
        let assistantMessage = Qwen38ChatMessage(
            role: .assistant, content: assistantContent, toolCalls: assistantToolCalls)
        if activeConversationID == id, activeConversationModel == model {
            activeConversationMessages = requestMessages + [assistantMessage]
            return
        }
        guard activeConversationID == nil,
              isFlashNextLoaded,
              conversationCacheBudgetBytes > 0 else { return }
        activeConversationID = id
        activeConversationModel = model
        activeConversationOptions = options
        activeConversationMessages = requestMessages + [assistantMessage]
    }

    /// P6.4: write-through variant for the GUI. Unlike the LAN path (which
    /// leaves a turn's result live in the resident engine until a *different*
    /// conversation displaces it — lazy eviction, cheaper for back-to-back
    /// same-conversation turns), the GUI cannot assume nothing else will
    /// touch the shared resident engine between two of its own turns (a LAN
    /// request's `generateStateless` resets it unconditionally). Exporting
    /// immediately after `rememberFlashConversation` and clearing the
    /// "live" pointer means the GUI's *next* turn always restores from the
    /// LRU explicitly instead of trusting stale liveness bookkeeping.
    public func flushGUIConversationToLRU(
        id: String, model: String, options: Qwen38GenerationOptions
    ) {
        guard activeConversationID == id, activeConversationModel == model else { return }
        storeActiveConversationIntoLRU(id: id, model: model, options: options)
        clearActiveConversation()
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
            // P11.1 : appliqué avant de générer, jamais après — voir le
            // commentaire de `Qwen38GenerationOptions.routedExpertCount`.
            if let requestedRoutedExpertCount = options.routedExpertCount {
                try flashEngine.setRoutedExpertCount(requestedRoutedExpertCount)
            }
            // P11.2 : même contrat, voir `Qwen38GenerationOptions.ablation`.
            if let requestedAblation = options.ablation {
                flashEngine.setAblation(requestedAblation)
            }
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
            // P11.1 : appliqué avant de générer, jamais après — voir le
            // commentaire de `Qwen38GenerationOptions.routedExpertCount`.
            if let requestedRoutedExpertCount = options.routedExpertCount {
                try flashEngine.setRoutedExpertCount(requestedRoutedExpertCount)
            }
            // P11.2 : même contrat, voir `Qwen38GenerationOptions.ablation`.
            if let requestedAblation = options.ablation {
                flashEngine.setAblation(requestedAblation)
            }
            // The resident engine is shared with the LAN server (single model,
            // §5.1.1): after a server request — or a P5 LRU restore — it holds
            // someone else's conversation. A new GUI conversation must start
            // from a clean state, otherwise its first turn is treated as a
            // continuation ("Flash-Next n'accepte une image qu'au premier
            // tour", seen 2026-09-10 with an image on turn 1).
            if conversationTurnCount == 0 {
                flashEngine.resetConversation()
                flashGUIMessages = []
                if let systemPrompt, !systemPrompt.isEmpty {
                    flashGUIMessages.append(.init(role: .system, content: systemPrompt))
                }
            }
            // P6.4: the GUI is now a client of the same LRU the LAN server
            // uses (`prepareFlashConversation`/`rememberFlashConversation`,
            // moved to this actor), under the fixed internal id "gui" — an
            // explicit id, never the P6.1 implicit-prefix path. Before this,
            // a LAN request between two GUI turns (`generateStateless`
            // resets the engine unconditionally) silently corrupted the
            // GUI's next turn; now that turn restores its own exported
            // state instead of trusting the engine's live bookkeeping.
            let modelKey = flashEngine.directory.lastPathComponent
            let fullMessages = flashGUIMessages + [
                Qwen38ChatMessage(role: .user, content: prompt, imageURLs: imageURLs)
            ]
            let (usePersistentCache, _, trackingID) = try await prepareFlashConversation(
                id: "gui", model: modelKey, messages: fullMessages, options: options)
            if !usePersistentCache {
                // Cold-start-ineligible (an already multi-turn GUI history
                // whose live/cached state could not be found — e.g. a LAN
                // model switch discarded it): fall back to a clean reset so
                // `flashEngine.generate` still deterministically starts a
                // fresh first turn instead of silently continuing into
                // whatever the engine happens to hold.
                flashEngine.resetConversation()
            }
            conversationTurnCount += 1
            let inner = try flashEngine.generate(
                prompt: prompt, systemPrompt: systemPrompt, imageURLs: imageURLs,
                options: options)
            guard let trackingID else { return inner }
            return AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        // Same reasoning/content split as the server's
                        // `makeJSONResponse` — the ledger only ever stores
                        // visible reply content, not `<think>` text.
                        var parser = Qwen38ThinkingStreamParser(primedInside: options.enableThinking)
                        var responseText = ""
                        for try await event in inner {
                            if case .chunk(let chunk) = event {
                                responseText += parser.append(chunk).content
                            }
                            continuation.yield(event)
                        }
                        responseText += parser.finish().content
                        // Write-through: export immediately and clear the
                        // "live" pointer, so the GUI's *next* turn always
                        // restores explicitly instead of assuming nothing
                        // touched the shared engine in between.
                        self.rememberFlashConversation(
                            id: trackingID, model: modelKey, requestMessages: fullMessages,
                            assistantContent: responseText, options: options)
                        self.flushGUIConversationToLRU(
                            id: trackingID, model: modelKey, options: options)
                        self.setFlashGUIMessages(
                            fullMessages + [Qwen38ChatMessage(role: .assistant, content: responseText)])
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
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
