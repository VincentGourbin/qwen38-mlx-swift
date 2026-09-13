import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// P12.3 (voir PLAN.md « P12 » et docs/knowledge/log.md, 2026-09-13
/// « P12.3 ») : génération en lot pour B requêtes indépendantes, au service
/// de l'ordonnanceur du serveur (`serve --batch-size N`). Construit
/// directement sur les briques déjà prouvées par P12.1/P12.2 :
/// `Qwen4ExpStreamingTextModel.forward(leftPadding:)` généralise déjà sur
/// `dim(0)` (parité inter-séquences vérifiée par `flash-batch-probe`), et
/// `Qwen4ExpBatchPadding.swift` porte la logique pure de remplissage à
/// gauche / `positionIDs` / masque de validité, déjà testée sans device
/// Metal.
///
/// Différences volontaires avec `Qwen4ExpStreamingGenerator` (le chemin
/// mono-séquence, strictement inchangé par ce fichier) :
///
///  - **Toujours stateless.** Une requête traitée ici ne devient jamais une
///    conversation active ni n'entre dans le LRU de conversations :
///    `Qwen38FlashNextEngine.generateBatch` réinitialise le modèle avant et
///    après l'appel. C'est le prix du lot — voir PLAN.md P12.3, « le lot et
///    le cache de conversations sont incompatibles en l'état ».
///  - **Jamais de MTP.** Le brouillonnage spéculatif suppose une séquence
///    unique (arbre de vérification) ; un tour groupé décode toujours
///    jeton par jeton, quelles que soient les options du client.
///  - **Échantillonnage et pénalités par ligne**, appliqués individuellement
///    à chaque pas (B est borné par `--batch-size`, donc B appels de
///    sampler par pas n'est pas le goulot — le lot lui-même reste un seul
///    `forward` MLX).
///  - **Gâchis assumé.** Une ligne qui atteint son propre jeton d'arrêt ou
///    son propre `maxNewTokens` continue d'être traitée par le `forward`
///    partagé (remplie du jeton de remplissage) jusqu'à ce que TOUTES les
///    lignes du lot soient terminées : faire sortir une ligne finie du lot
///    en cours de route est explicitement hors périmètre de cette étape
///    (PLAN.md P12.3, « lot continu » non traité ici).
public final class Qwen4ExpBatchStreamingGenerator: @unchecked Sendable {
    public let model: Qwen4ExpStreamingTextModel

    public init(model: Qwen4ExpStreamingTextModel) {
        self.model = model
    }

    /// Une ligne du lot : tout ce qui, contrairement au prompt et à
    /// `padTokenID` (partagés), reste réglable **par requête** — température,
    /// top-p/top-k (via `preset`), pénalités, jetons d'arrêt et
    /// `maxNewTokens` propres.
    public struct Row: Sendable {
        public let tokenIDs: [Int32]
        public let maxNewTokens: Int
        public let stopTokenIDs: Set<Int32>
        public let preset: Qwen4ExpSamplingPreset
        public let seed: UInt64?
        public let presencePenalty: Float
        public let repetitionPenalty: Float

        public init(
            tokenIDs: [Int32], maxNewTokens: Int, stopTokenIDs: Set<Int32>,
            preset: Qwen4ExpSamplingPreset, seed: UInt64? = nil,
            presencePenalty: Float = 0, repetitionPenalty: Float = 1.0
        ) {
            self.tokenIDs = tokenIDs
            self.maxNewTokens = maxNewTokens
            self.stopTokenIDs = stopTokenIDs
            self.preset = preset
            self.seed = seed
            self.presencePenalty = presencePenalty
            self.repetitionPenalty = repetitionPenalty
        }
    }

    /// Un flux par ligne, dans l'ordre de `rows` : chaque flux ne reçoit
    /// JAMAIS que les événements `.token`/`.finished` de SA PROPRE ligne —
    /// c'est le critère de non-contamination (PLAN.md P12.3, « deux
    /// requêtes identiques envoyées ensemble doivent rendre exactement les
    /// mêmes identifiants que la même requête seule »), déjà garanti au
    /// niveau numérique par P12.1/P12.2 et préservé ici au niveau du
    /// branchement : `run` n'écrit jamais dans `continuations[i]` avec une
    /// donnée calculée pour une autre ligne que `i`.
    ///
    /// P12.3 (correctif du 2026-09-13, crash mémoire) : `completion` est le
    /// SEUL signal sûr pour savoir que ce lot a fini de toucher `model` —
    /// jamais la clôture des flux. La clôture d'un flux (`continuation.
    /// finish()`) réveille son lecteur en aval de façon asynchrone, sur une
    /// autre tâche que celle qui a produit l'événement ; un appelant qui
    /// libérait un verrou d'exclusion mutuelle dès que tous les flux d'un
    /// lot sont consommés pouvait donc laisser un second lot démarrer SA
    /// PROPRE réinitialisation pendant que ce lot-ci exécutait encore la
    /// sienne après avoir déjà refermé ses continuations — deux
    /// `resetCaches()` concurrents sur le même modèle résident, cause
    /// vérifiée du crash (`EXC_BAD_ACCESS` dans les couches résidentes,
    /// voir docs/knowledge/log.md). `completion` ne se termine, lui, que
    /// lorsque `run()` a réellement fini de s'exécuter (succès ou échec),
    /// quelle que soit la vitesse des lecteurs.
    public struct Result: Sendable {
        public let streams: [AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>]
        public let completion: Task<Void, Never>
    }

    public func generate(
        rows: [Row], padTokenID: Int32, profiler: MLXProfiler = .shared
    ) throws -> Result {
        guard !rows.isEmpty else { throw Qwen4ExpBatchPaddingError.emptyBatch }
        guard rows.allSatisfy({ $0.maxNewTokens > 0 }) else {
            throw Qwen4ExpStreamingGenerationError.invalidMaxNewTokens
        }
        guard rows.allSatisfy({ !$0.tokenIDs.isEmpty }) else {
            throw Qwen4ExpStreamingGenerationError.emptyPrompt
        }

        var continuations: [AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>.Continuation] = []
        var streams: [AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>] = []
        streams.reserveCapacity(rows.count)
        for _ in rows {
            var boxed: AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>.Continuation!
            let stream = AsyncThrowingStream<Qwen4ExpGenerationEvent, Error> { boxed = $0 }
            streams.append(stream)
            continuations.append(boxed)
        }

        // La tâche pilote les B lignes depuis un seul `forward` MLX partagé
        // — voir le commentaire de fichier, « toujours stateless » : elle
        // n'est jamais annulée par le départ d'un client isolé (un flux
        // abandonné laisse simplement ses événements sans lecteur), les
        // autres lignes du lot continuant normalement. `completion` est
        // cette même tâche : elle se termine exactement quand `run()`
        // retourne, jamais avant — voir le commentaire de `Result`.
        let completion = Task {
            do {
                try self.run(rows: rows, padTokenID: padTokenID, profiler: profiler, continuations: continuations)
            } catch {
                for continuation in continuations {
                    continuation.finish(throwing: error)
                }
            }
        }
        return Result(streams: streams, completion: completion)
    }

    private func run(
        rows: [Row], padTokenID: Int32, profiler: MLXProfiler,
        continuations: [AsyncThrowingStream<Qwen4ExpGenerationEvent, Error>.Continuation]
    ) throws {
        let batchSize = rows.count
        // Toujours stateless (voir le commentaire de fichier) : jamais de
        // continuation d'une conversation précédente, jamais d'état laissé
        // pour la requête suivante — que celle-ci rejoigne un futur lot ou
        // le chemin mono-séquence.
        model.resetConversation()
        model.resetNGramCacheStats()

        let tokenCounts = rows.map(\.tokenIDs.count)
        let layout = try qwen4ExpComputeBatchPaddingLayout(tokenCounts: tokenCounts)
        let paddedRows = qwen4ExpLeftPadTokenIDs(
            rows.map(\.tokenIDs), layout: layout, padTokenID: padTokenID)
        let promptArray = MLXArray(paddedRows.flatMap { $0 }).reshaped([batchSize, layout.maxLength])
        let prefillPositionIDs = qwen4ExpLeftPaddedPositionIDsArray(layout: layout)

        let started = Date()
        profiler.startPrefill()
        let prefill = try model.forward(
            inputIDs: promptArray, positionIDs: prefillPositionIDs, leftPadding: layout.leftPadding)
        eval(prefill.logits)
        let prefillEnd = Date()
        profiler.endPrefill()

        var logits = prefill.logits[0..., -1, 0...]
        var generated: [[Int32]] = Array(repeating: [], count: batchSize)
        var finished = Array(repeating: false, count: batchSize)
        var firstTokenTime: [TimeInterval?] = Array(repeating: nil, count: batchSize)
        var generationStarted = false

        // Un échantillonneur par ligne : contrairement au chemin
        // mono-séquence, les lignes d'un même lot n'ont aucune raison de
        // partager température/top-p/top-k (PLAN.md P12.3, « échantillonnage
        // par séquence »).
        let samplers = rows.map { $0.preset.sampler(seed: $0.seed) }
        let penaltiesActive = rows.map {
            $0.preset.temperature > 0 && ($0.presencePenalty != 0 || $0.repetitionPenalty != 1.0)
        }
        var seenMasks: [MLXArray?] = penaltiesActive.map {
            $0 ? Qwen4ExpLogitPenalizer.seedMask(vocabSize: logits.dim(-1), tokenIDs: []) : nil
        }

        let globalMaxNewTokens = rows.map(\.maxNewTokens).max() ?? 0
        for step in 0 ..< globalMaxNewTokens {
            if finished.allSatisfy({ $0 }) { break }
            var nextTokens = [Int32](repeating: padTokenID, count: batchSize)
            for row in 0 ..< batchSize where !finished[row] {
                let rowLogits = logits[row ..< (row + 1), 0...]
                let sampledLogits: MLXArray
                if let mask = seenMasks[row] {
                    sampledLogits = Qwen4ExpLogitPenalizer.apply(
                        logits: rowLogits, seenMask: mask, presence: rows[row].presencePenalty,
                        repetition: rows[row].repetitionPenalty)
                } else {
                    sampledLogits = rowLogits
                }
                let sampled = samplers[row].sample(logits: sampledLogits)
                let token = Int32(sampled.item(Int32.self))
                if let mask = seenMasks[row] {
                    seenMasks[row] = Qwen4ExpLogitPenalizer.markSeen(mask, token: token)
                }
                if !generationStarted {
                    generationStarted = true
                    profiler.startGeneration()
                }
                if firstTokenTime[row] == nil {
                    firstTokenTime[row] = Date().timeIntervalSince(started)
                }
                generated[row].append(token)
                nextTokens[row] = token
                // Écrit uniquement dans le flux `row` — voir le critère de
                // non-contamination dans le commentaire de `generate`.
                continuations[row].yield(.token(token))
                if rows[row].stopTokenIDs.contains(token) || generated[row].count >= rows[row].maxNewTokens {
                    finished[row] = true
                }
            }
            if finished.allSatisfy({ $0 }) { break }
            let nextInput = MLXArray(nextTokens).reshaped([batchSize, 1])
            let decodePositionIDs = qwen4ExpLeftPaddedDecodePositionIDsArray(
                tokenCounts: tokenCounts, step: step)
            let stepResult = try model.forward(
                inputIDs: nextInput, positionIDs: decodePositionIDs, leftPadding: layout.leftPadding)
            eval(stepResult.logits)
            logits = stepResult.logits[0..., -1, 0...]
        }

        if generationStarted {
            profiler.endGeneration(tokenCount: generated.reduce(0) { $0 + $1.count })
        }
        let generationEnd = Date()
        // P12.3 (correctif du 2026-09-13, crash mémoire) : toujours
        // stateless — ne laisse aucun état de ce lot survivre pour la
        // requête suivante — mais cette réinitialisation doit s'exécuter
        // AVANT de refermer la moindre continuation ci-dessous, jamais
        // après. `run()` s'exécute entièrement sur une seule tâche, sans
        // aucun point de suspension avant son retour : placer la
        // réinitialisation ici garantit qu'elle est terminée avant que qui
        // que ce soit, en aval, ne puisse observer la fin d'une seule ligne
        // du lot — indépendamment de `completion` (voir son commentaire),
        // en défense en profondeur.
        model.resetConversation()
        // Les temps de préremplissage/décodage sont ceux du lot entier —
        // un `forward` partagé ne permet pas de les répartir par ligne
        // honnêtement ; chaque ligne rapporte donc le coût réel qu'elle a
        // subi (celui du lot), pas un coût individuel fictif.
        for row in 0 ..< batchSize {
            let summary = Qwen4ExpGenerationSummary(
                tokenIDs: generated[row],
                promptTokenCount: rows[row].tokenIDs.count,
                timeToFirstToken: firstTokenTime[row],
                prefillTime: prefillEnd.timeIntervalSince(started),
                decodeTime: generationStarted ? generationEnd.timeIntervalSince(prefillEnd) : 0,
                layerVisitCount: prefill.reports.count,
                layerLoadTime: prefill.reports.reduce(0) { $0 + $1.loadDuration },
                activeMemoryBytes: Memory.activeMemory,
                peakMemoryBytes: Memory.peakMemory,
                ngramCacheStats: model.ngramCacheStats())
            continuations[row].yield(.finished(summary))
            continuations[row].finish()
        }
    }
}
