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

/// P5.1: opaque, exportable snapshot of one Flash-Next conversation's
/// generation state. `Qwen38FlashNextEngine` produces the real thing
/// (decoder caches via `Qwen4ExpStreamingTextModel.snapshot()`); a test
/// mock can hand back its own lightweight conformer without loading the
/// ~57-84 GB real checkpoint (H3.1's existing mock-engine pattern).
public protocol Qwen38FlashConversationStateProtocol: Sendable {
    /// The exact message history (system/user/assistant turns, assistant
    /// replies included) this state was captured after — the server's LRU
    /// contract (§5.1.1, P5.2) compares an incoming request's
    /// `messages.dropLast()` against this to decide whether the state is
    /// still a valid continuation point.
    var ledger: [Qwen38ChatMessage] { get }
    /// Device bytes held by the captured caches — what the server's
    /// `--conversation-cache-gb` budget is measured against.
    var byteCount: Int { get }
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
    /// P11.1 : largeur de routage MoE effective (`num_experts_per_tok` du
    /// checkpoint, sauf surcharge) — publiée telle quelle par `/healthz` et
    /// dans les métriques de chaque tour pour qu'une mesure ne puisse pas
    /// se croire à un K qui n'est plus en vigueur (PLAN.md P11.1).
    var routedExpertCount: Int { get }
    /// P11.1 : change la largeur de routage MoE de l'engin résident sans
    /// recharger le checkpoint. `nil` revient à la valeur du checkpoint ;
    /// une valeur hors bornes lève une erreur claire (`Qwen4ExpRoutedExpertCountError`)
    /// au lieu de faire tomber le process — appelable en toute sécurité
    /// depuis une requête serveur. Répercutée sur le drafter MTP s'il est
    /// déjà chargé, avec la même valeur que la cible (voir
    /// `Qwen4ExpMTPPredictor.setRoutedExpertCount`).
    @discardableResult
    func setRoutedExpertCount(_ override: Int?) throws -> Int
    /// P11.2 : quel sous-bloc, le cas échéant, l'engin résident court-
    /// circuite (`Qwen4ExpLayerBenchAblation`) — publiée par `/healthz` et
    /// dans les métriques de chaque tour pour la même raison que
    /// `routedExpertCount` : ne jamais mesurer en croyant à tort avoir (ou
    /// ne pas avoir) une ablation active.
    var ablation: Qwen4ExpLayerBenchAblation { get }
    /// P11.2 : change l'ablation de l'engin résident sans recharger le
    /// checkpoint. Contrairement à `setRoutedExpertCount`, aucune erreur
    /// possible : toute valeur de `Qwen4ExpLayerBenchAblation` est valide.
    func setAblation(_ new: Qwen4ExpLayerBenchAblation)
    func resetConversation()
    func unload()
    func decode(tokenIDs: [Int32]) -> String
    func generate(
        prompt: String, systemPrompt: String?, imageURLs: [URL], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error>
    func generateFromMessages(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> AsyncThrowingStream<Qwen38GenerationEvent, Error>
    /// P12.3 : décodage en lot de B requêtes indépendantes pour
    /// l'ordonnanceur du serveur (`serve --batch-size N > 1`) — voir
    /// `Qwen4ExpBatchStreamingGenerator`'s doc comment pour le contrat
    /// complet (toujours stateless, jamais de MTP, gâchis assumé sur une
    /// ligne finie). Un flux par requête, dans le même ordre que
    /// `requests` ; chaque flux ne porte jamais que les événements de sa
    /// propre ligne (critère de non-contamination, PLAN.md P12.3).
    ///
    /// `Qwen38BatchGenerationResult.completion` (correctif du 2026-09-13,
    /// crash mémoire) est le seul signal sûr pour savoir que cette
    /// exécution a fini de toucher le modèle résident — un appelant qui
    /// libère un verrou d'exclusion mutuelle envers une autre exécution
    /// DOIT attendre `completion`, jamais seulement la consommation des
    /// flux. Voir le commentaire de `Qwen38BatchGenerationResult`.
    func generateBatch(
        requests: [Qwen38BatchGenerationRequest]
    ) throws -> Qwen38BatchGenerationResult
    /// P6.1: pure rendering, no generation side effect — the exact token
    /// IDs `generateFromMessages` would feed the model for this message
    /// list, via the same `Qwen4ExpPromptBuilder.buildFromMessages` path.
    /// The server's implicit-prefix cache (no `conversation_id`) uses this
    /// to compare a candidate LRU entry's `ledger` against an incoming
    /// request on rendered token IDs rather than message structs, so a
    /// system-prompt edit or a client-truncated history is a clean miss
    /// instead of a wrong restore.
    func renderedTokenIDs(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> [Int32]
    /// H4.2: forces every decoder layer to be loaded from disk once, up
    /// front, instead of paying that cost inside the first real turn's
    /// TTFT. Yields the index of each layer as it finishes loading (0-based,
    /// `numHiddenLayers` values total) so a caller can drive a progress bar.
    /// A best-effort warm-up: failures surface later, on the first real
    /// `generate` call, rather than here.
    func warmUp() -> AsyncStream<Int>
    /// P5.1: capture the complete per-conversation generation state (decoder
    /// caches, M-RoPE offset, turn index) bundled with `ledger` — the
    /// message history already rendered — so the server's LRU (P5.2) can
    /// key, budget and later restore it as one unit. MTP's own drafter
    /// cache is deliberately not preserved (see `Qwen38FlashNextEngine`'s
    /// doc comment on `exportConversationState`): a restored conversation
    /// re-primes the drafter from scratch on its next MTP-enabled turn.
    func exportConversationState(ledger: [Qwen38ChatMessage]) -> any Qwen38FlashConversationStateProtocol
    /// Replace this engine's live conversation state with a previously
    /// exported one. The engine is a single resident model (§5.1.1): this
    /// overwrites whatever conversation was live before the call.
    func restoreConversationState(_ state: any Qwen38FlashConversationStateProtocol)
}

public protocol Qwen38FlashNextEngineFactory: Sendable {
    /// P11.1 : `routedExpertCount` surcharge `num_experts_per_tok` du
    /// checkpoint — `nil` (comportement inchangé partout ailleurs) laisse
    /// le décodeur utiliser sa propre valeur. Pas de valeur par défaut ici :
    /// Swift n'applique jamais un défaut de paramètre à un appel fait à
    /// travers un type protocole/existentiel — seul le type concret en
    /// bénéficie (voir `Qwen38DefaultFlashNextEngineFactory`).
    func makeEngine(
        directory: URL, routedExpertCount: Int?, ablation: Qwen4ExpLayerBenchAblation
    ) async throws -> any Qwen38FlashNextEngineProtocol
}

public struct Qwen38DefaultFlashNextEngineFactory: Qwen38FlashNextEngineFactory {
    public init() {}

    public func makeEngine(
        directory: URL, routedExpertCount: Int? = nil,
        ablation: Qwen4ExpLayerBenchAblation = .none
    ) async throws -> any Qwen38FlashNextEngineProtocol {
        try await Qwen38FlashNextEngine(
            directory: directory, routedExpertCount: routedExpertCount, ablation: ablation)
    }
}

/// P5.1: `Qwen38FlashNextEngine`'s concrete conversation state. Holds the
/// decoder's copied caches (`Qwen4ExpStreamingTextModelSnapshot`, itself
/// `@unchecked Sendable` for the same "MLX arrays are confined to the
/// owning runtime" reason as every other Flash-Next snapshot type) plus the
/// bookkeeping `restoreConversationState` needs to put the engine back in
/// exactly the state `exportConversationState` found it in.
public final class Qwen38FlashConversationState: Qwen38FlashConversationStateProtocol, @unchecked Sendable {
    fileprivate let modelSnapshot: Qwen4ExpStreamingTextModelSnapshot
    fileprivate let hasConversationHistory: Bool
    fileprivate let turnIndex: Int
    public let ledger: [Qwen38ChatMessage]
    public let byteCount: Int
    /// P6.3: the rolling buffer of prior-assistant-turn tokens the
    /// presence/repetition mask seeds from — captured too, or an LRU
    /// restore (P5.2/P6.1) would silently forget every turn generated
    /// before the export and reopen the same verbatim-loop risk P6.3
    /// fixes for a plain continuation.
    fileprivate let recentAssistantTokenIDs: [Int32]

    fileprivate init(
        modelSnapshot: Qwen4ExpStreamingTextModelSnapshot, hasConversationHistory: Bool,
        turnIndex: Int, ledger: [Qwen38ChatMessage], recentAssistantTokenIDs: [Int32]
    ) {
        self.modelSnapshot = modelSnapshot
        self.hasConversationHistory = hasConversationHistory
        self.turnIndex = turnIndex
        self.ledger = ledger
        self.recentAssistantTokenIDs = recentAssistantTokenIDs
        self.byteCount = modelSnapshot.byteCount
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
    /// P6.3: rolling buffer of the most recent assistant-turn tokens this
    /// conversation generated, trimmed to `options.penaltyContextTokens`
    /// after every turn — what a new turn's presence/repetition mask seeds
    /// from instead of starting empty (see `Qwen4ExpLogitPenalizer.seedMask`).
    private var recentAssistantTokenIDs: [Int32] = []

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
    /// diminishing returns past 8, so N=8 was the default.
    ///
    /// **Révisé le 2026-09-13 (P11) : le défaut passe de 8 à 48**, c'est-à-dire
    /// un seul `eval` bloquant par forward. Le balayage P4.1 ci-dessus datait
    /// d'**avant** la correction de dtype F7 (P8.2), qui a multiplié le débit
    /// par 2,74 et déplacé l'optimum. Une trace Metal System Trace du décodage
    /// a montré **197 tampons de commandes par pas et 33 % de GPU inactif**
    /// (46 trous par pas, 17,4 ms perdus sur 52,7) ; remesuré à séquence
    /// forcée sur le checkpoint réel, N ∈ {8, 16, 24, 48} donne
    /// 46,82 / 45,51 / 44,89 / **44,36** ms par pas — soit **+5,5 %** à N=48,
    /// avec des identifiants greedy strictement identiques et un pic MLX
    /// inchangé (57,42 Go). Voir docs/knowledge/log.md, 2026-09-13.
    public init(
        directory: URL, profileLayers: Bool = false, residentAsyncEval: Bool = true,
        residentAsyncInterval: Int = 48, uncachedIO: Bool = true,
        /// P11.1 : surcharge de `num_experts_per_tok` pour le modèle
        /// résident — voir `Qwen4ExpStreamingTextModel`'s doc comment.
        /// `nil` (le défaut) laisse le comportement inchangé.
        routedExpertCount: Int? = nil,
        /// P11.2 : quel sous-bloc, le cas échéant, court-circuiter dans le
        /// modèle résident. `.none` (le défaut) laisse le comportement
        /// inchangé.
        ablation: Qwen4ExpLayerBenchAblation = .none
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
            uncachedIO: uncachedIO, routedExpertCount: routedExpertCount, ablation: ablation)
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

    /// P11.1 : largeur de routage MoE effective du modèle résident.
    public var routedExpertCount: Int { model.routedExpertCount }

    /// P11.1 : change la largeur de routage MoE de l'engin résident, cible
    /// et drafter MTP (s'il est déjà chargé) ensemble — voir le commentaire
    /// du protocole. Ne recharge aucun poids, donc appelable entre deux
    /// requêtes sans reproduire le coût d'E/S du chargement initial.
    @discardableResult
    public func setRoutedExpertCount(_ override: Int?) throws -> Int {
        let resolved = try model.updateRoutedExpertCount(override)
        if let mtpPredictor {
            try mtpPredictor.setRoutedExpertCount(resolved)
        }
        return resolved
    }

    /// P11.2 : ablation effective du modèle résident.
    public var ablation: Qwen4ExpLayerBenchAblation { model.ablation }

    /// P11.2 : change l'ablation de l'engin résident sans recharger le
    /// checkpoint. Non répercutée sur le drafter MTP, qui n'a pas été câblé
    /// pour cette tâche (hors périmètre — voir le rapport P11.2) : un tour
    /// MTP-activé continue de brouillonner sans aucune ablation même quand
    /// la cible en a une.
    public func setAblation(_ new: Qwen4ExpLayerBenchAblation) {
        model.setAblation(new)
    }

    public func resetConversation() {
        model.resetConversation()
        hasConversationHistory = false
        turnIndex = 0
        recentAssistantTokenIDs = []
        // The drafter's cache is tied to the target's own conversation
        // history; the predictor's *weights* stay loaded (no need to pay
        // Lexar IO again), only its per-conversation state is discarded.
        mtpDraftState = nil
    }

    /// P6.3: appends a turn's generated tokens to the rolling
    /// presence/repetition context and trims it to `limit` tokens (`<= 0`
    /// clears it — the option's "0 = old behavior" contract).
    private func recordAssistantTokens(_ tokenIDs: [Int32], limit: Int) {
        guard limit > 0 else {
            recentAssistantTokenIDs = []
            return
        }
        recentAssistantTokenIDs.append(contentsOf: tokenIDs)
        if recentAssistantTokenIDs.count > limit {
            recentAssistantTokenIDs.removeFirst(recentAssistantTokenIDs.count - limit)
        }
    }

    public func unload() {
        model.decoder.unloadResidentLayers()
    }

    /// P5.1: capture caches + M-RoPE offset (`model.snapshot()`, `KVCache.copy()`
    /// under the hood — measured cost and size documented in
    /// docs/knowledge/log.md "P5.1") together with the turn bookkeeping a
    /// later `restoreConversationState` needs to resume exactly where this
    /// conversation left off.
    ///
    /// MTP's drafter cache (`mtpDraftState`) is intentionally *not* captured:
    /// it has no `copy()`-based deep-snapshot support today, and MTP requires
    /// greedy decoding while the LRU's main use case (P5.2) is sampled
    /// multi-client dialogue. A conversation restored through this state
    /// simply re-primes its drafter from scratch on its next MTP-enabled
    /// turn (`runMTPGenerationStream` treats a missing `mtpDraftState` as
    /// "prime fresh" already).
    public func exportConversationState(
        ledger: [Qwen38ChatMessage]
    ) -> any Qwen38FlashConversationStateProtocol {
        Qwen38FlashConversationState(
            modelSnapshot: model.snapshot(), hasConversationHistory: hasConversationHistory,
            turnIndex: turnIndex, ledger: ledger,
            recentAssistantTokenIDs: recentAssistantTokenIDs)
    }

    /// Replace the engine's live state with a previously exported one. Only
    /// ever called by the server's LRU (P5.2) with a state this same engine
    /// produced (states never cross model families or checkpoints), hence
    /// the force-cast — a mismatch here would be a server-side bug, not a
    /// recoverable runtime condition.
    public func restoreConversationState(_ state: any Qwen38FlashConversationStateProtocol) {
        guard let state = state as? Qwen38FlashConversationState else {
            preconditionFailure(
                "restoreConversationState: état d'un autre moteur (bug du LRU serveur)")
        }
        model.restore(state.modelSnapshot)
        hasConversationHistory = state.hasConversationHistory
        turnIndex = state.turnIndex
        recentAssistantTokenIDs = state.recentAssistantTokenIDs
        // See exportConversationState's doc comment: the drafter cache is
        // never preserved, so any stale one from before this restore must
        // not survive into the resumed conversation.
        mtpDraftState = nil
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
    /// P6.1: same rendering `generateFromMessages` performs before it calls
    /// `resetConversation()` and starts generating — extracted so it can be
    /// called read-only, on the server's actor, without touching
    /// `hasConversationHistory` or any decoder state.
    public func renderedTokenIDs(
        messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions
    ) throws -> [Int32] {
        guard messages.allSatisfy({ $0.imageURLs.isEmpty }) else {
            throw Qwen38FlashNextEngineError.statelessImagesUnsupported
        }
        let hfMessages: [Tokenizers.Message] = messages.map {
            ["role": $0.role.rawValue, "content": $0.content]
        }
        let built = try Qwen4ExpPromptBuilder.buildFromMessages(
            tokenizer: tokenizer, messages: hfMessages, thinking: options.enableThinking,
            reasoningEffort: options.reasoningEffort)
        return built.tokenIDs
    }

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

    /// P12.3 : voir le contrat complet sur `Qwen38FlashNextEngineProtocol.
    /// generateBatch` et `Qwen4ExpBatchStreamingGenerator`'s doc comment.
    /// Refuse toute image (le rendu ChatML manuel de Flash-Next n'a qu'une
    /// forme mono-tour/mono-image, jamais branchée sur un lot) — le serveur
    /// n'est de toute façon censé router une requête avec image que par le
    /// chemin chaud/mono-séquence, jamais vers ce lot (PLAN.md P12.3).
    public func generateBatch(
        requests: [Qwen38BatchGenerationRequest]
    ) throws -> Qwen38BatchGenerationResult {
        guard !requests.isEmpty else { return Qwen38BatchGenerationResult(streams: [], completion: Task {}) }
        guard requests.allSatisfy({ $0.messages.allSatisfy { $0.imageURLs.isEmpty } }) else {
            throw Qwen38FlashNextEngineError.statelessImagesUnsupported
        }
        // Toujours stateless (voir Qwen4ExpBatchStreamingGenerator) : ce lot
        // ne doit ni hériter d'une conversation active, ni empiéter sur la
        // bibliothèque de tours (turnIndex) des conversations persistantes.
        resetConversation()

        let built = try requests.map { request -> Qwen4ExpBuiltPrompt in
            let hfMessages: [Tokenizers.Message] = request.messages.map {
                ["role": $0.role.rawValue, "content": $0.content]
            }
            return try Qwen4ExpPromptBuilder.buildFromMessages(
                tokenizer: tokenizer, messages: hfMessages,
                thinking: request.options.enableThinking,
                reasoningEffort: request.options.reasoningEffort)
        }

        let padTokenID = configuration.textConfiguration.eosTokenID ?? 0
        let rows = zip(built, requests).map { builtPrompt, request in
            Qwen4ExpBatchStreamingGenerator.Row(
                tokenIDs: builtPrompt.tokenIDs,
                maxNewTokens: max(request.options.maxTokens, 1),
                stopTokenIDs: stopTokenIDs,
                preset: .custom(
                    temperature: request.options.temperature, topP: request.options.topP,
                    topK: request.options.topK),
                presencePenalty: request.options.presencePenalty,
                repetitionPenalty: request.options.repetitionPenalty)
        }

        let batchGenerator = Qwen4ExpBatchStreamingGenerator(model: model)
        let inner = try batchGenerator.generate(rows: rows, padTokenID: padTokenID)
        let mappedStreams = zip(inner.streams, requests).map { innerStream, request in
            mapBatchRowStream(innerStream, ablationLabel: model.ablation.rawValue, options: request.options)
        }
        // `inner.completion` — pas un flux quelconque — est propagée telle
        // quelle : c'est la tâche `Qwen4ExpBatchStreamingGenerator.run()`
        // elle-même, indépendante de tout habillage ultérieur
        // (`mapBatchRowStream` relaie dans SA PROPRE tâche, qui ne touche
        // jamais `model`). Voir le commentaire de `Qwen38BatchGenerationResult`.
        return Qwen38BatchGenerationResult(streams: mappedStreams, completion: inner.completion)
    }

    /// P12.3 : convertit le flux `Qwen4ExpGenerationEvent` d'une ligne de
    /// lot vers le même `Qwen38GenerationEvent` que le chemin mono-séquence
    /// — pas de branchement MTP (jamais actif ici), `cacheReused`/
    /// `conversationReplayed` toujours `false` (toujours stateless),
    /// `turnIndex` fixé à 1 (une ligne de lot ne fait jamais partie d'une
    /// conversation continuée, ce compteur n'a pas de sens pour elle).
    private func mapBatchRowStream(
        _ inner: AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>,
        ablationLabel: String, options: Qwen38GenerationOptions
    ) -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in inner {
                        if Task.isCancelled { break }
                        switch event {
                        case .token(let token):
                            guard self.visibleTokenFilter.shouldEmit(Int(token)) else { continue }
                            let piece = Qwen38VisibleText.sanitize(
                                self.tokenizer.decode(tokens: [Int(token)], skipSpecialTokens: false))
                            if !piece.isEmpty {
                                continuation.yield(.chunk(piece))
                            }
                        case .finished(let summary):
                            let stopReason: GenerateStopReason
                            if let last = summary.tokenIDs.last, self.stopTokenIDs.contains(last) {
                                stopReason = .stop
                            } else if summary.tokenIDs.count >= max(options.maxTokens, 1) {
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
                                        report: "",
                                        chromeTrace: Data(),
                                        activeMemoryBytes: summary.activeMemoryBytes,
                                        peakMemoryBytes: summary.peakMemoryBytes,
                                        acceptRate: nil,
                                        timeToFirstToken: summary.timeToFirstToken,
                                        turnIndex: 1,
                                        cacheReused: false,
                                        conversationReplayed: false,
                                        inputDescription: "Texte",
                                        mtpStatus: Qwen38MTPRunStatus(availability: .unavailable),
                                        routedExpertCount: self.model.routedExpertCount,
                                        ablation: ablationLabel)))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
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
        let (profileSession, ownsSession, requestPhase) = Qwen38Profiling.beginRequestSession(
            title: "QWEN3.8 FLASH-NEXT INFERENCE",
            metadata: ["model": directory.lastPathComponent, "turn": String(currentTurnIndex)],
            phase: "Requête \(currentTurnIndex)")

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
                maxNewTokens: maxNewTokens, profiler: profiler, profileSession: profileSession,
                ownsSession: ownsSession, requestPhase: requestPhase)
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

        // P6.3: seed this turn's mask with the rolling buffer of prior
        // assistant-turn tokens (already trimmed to `penaltyContextTokens`
        // by `recordAssistantTokens`) instead of starting empty every turn.
        let inner = generator.generate(
            promptTokenIDs: built.tokenIDs, positionIDs: built.positionIDs,
            visionEmbeddings: built.visionEmbeddings, imageTokenID: built.imageTokenID,
            options: .init(
                maxNewTokens: maxNewTokens, stopTokenIDs: stopTokenIDs, preset: preset,
                continueConversation: continueConversation,
                presencePenalty: options.presencePenalty,
                repetitionPenalty: options.repetitionPenalty,
                initialPenaltyTokenIDs: recentAssistantTokenIDs),
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
                            // P6.3: extend the rolling penalty-context buffer
                            // with this turn's own reply, ready for the
                            // *next* turn's mask — done regardless of
                            // whether penalties were active this turn, so
                            // enabling them mid-conversation still sees
                            // whatever history already accumulated.
                            self.recordAssistantTokens(
                                summary.tokenIDs, limit: options.penaltyContextTokens)
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
                                        report: ownsSession ? profileSession.generateReport() : "",
                                        chromeTrace: ownsSession
                                            ? ChromeTraceExporter.export(session: profileSession) : Data(),
                                        activeMemoryBytes: summary.activeMemoryBytes,
                                        peakMemoryBytes: summary.peakMemoryBytes,
                                        acceptRate: nil,
                                        timeToFirstToken: summary.timeToFirstToken,
                                        turnIndex: currentTurnIndex,
                                        cacheReused: continueConversation,
                                        conversationReplayed: false,
                                        inputDescription: inputDescription,
                                        mtpStatus: mtpStatus,
                                        routedExpertCount: self.model.routedExpertCount,
                                        ablation: self.model.ablation.rawValue)))
                        }
                    }
                    continuation.finish()
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
                } catch {
                    continuation.finish(throwing: error)
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
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
        profiler: MLXProfiler, profileSession: ProfilingSession,
        ownsSession: Bool, requestPhase: String
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
                        // P11.1 : le drafter reçoit la même largeur de
                        // routage que la cible, sauf distinction explicite
                        // (aucune ici) — voir `Qwen4ExpMTPLoader.load`'s
                        // doc comment.
                        let loaded = try Qwen4ExpMTPLoader.load(
                            from: self.directory, uncachedIO: true,
                            routedExpertCount: self.model.routedExpertCount)
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

                    // P6.3: MTP is greedy-only (no mask ever applied here),
                    // but a later sampled turn in the same conversation
                    // still needs this turn's reply in its penalty context.
                    self.recordAssistantTokens(
                        result.tokenIDs, limit: options.penaltyContextTokens)
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
                                report: ownsSession ? profileSession.generateReport() : "",
                                chromeTrace: ownsSession
                                    ? ChromeTraceExporter.export(session: profileSession) : Data(),
                                activeMemoryBytes: Memory.activeMemory,
                                peakMemoryBytes: Memory.peakMemory,
                                acceptRate: result.stats.acceptanceRate,
                                timeToFirstToken: result.timeToFirstToken,
                                turnIndex: currentTurnIndex,
                                cacheReused: continueConversation,
                                conversationReplayed: false,
                                inputDescription: inputDescription,
                                mtpStatus: mtpStatus,
                                routedExpertCount: self.model.routedExpertCount,
                                ablation: self.model.ablation.rawValue)))
                    continuation.finish()
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
                } catch {
                    continuation.finish(throwing: error)
                    Qwen38Profiling.endRequestSession(ownsSession: ownsSession, phase: requestPhase)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
