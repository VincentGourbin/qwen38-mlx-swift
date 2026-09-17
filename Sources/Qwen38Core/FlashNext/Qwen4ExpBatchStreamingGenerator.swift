import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// P12.3 (défaut 1 du 2026-09-14) : seam de test — `run()` ne s'adresse
/// jamais qu'à ces quatre membres de `Qwen4ExpStreamingTextModel`. Isoler ce
/// sous-ensemble derrière un protocole permet à un test d'injecter un
/// modèle factice pour exercer `run()` (en particulier l'ordre de fermeture
/// des continuations) sans jamais charger de checkpoint ni toucher un
/// device Metal au-delà des `MLXArray` que le test construit lui-même.
///
/// `forward` est délibérément renommé `batchForward` dans ce protocole : un
/// témoin (`extension Qwen4ExpStreamingTextModel` ci-dessous) qui relaierait
/// vers `forward(inputIDs:positionIDs:leftPadding:)` sous le MÊME nom
/// entrerait en concurrence avec la méthode d'origine (neuf paramètres,
/// six par défaut) pour exactement les mêmes trois arguments — la
/// résolution de surcharge de Swift préfère la correspondance exacte (donc
/// SOI-MÊME) à celle qui comble des paramètres par défaut, ce qui
/// bouclerait indéfiniment plutôt que d'appeler la méthode d'origine.
public protocol Qwen4ExpBatchForwardModel: AnyObject, Sendable {
    func batchForward(
        inputIDs: MLXArray, positionIDs: MLXArray?, leftPadding: [Int]?,
        // 2026-09-17 (crash mémoire `metal::malloc` en production, lot ou
        // non — voir `Qwen4ExpStreamingTextModel.forward(lastPositionOnly:)`) :
        // relayé tel quel, sans défaut ici, pour que chaque appelant de ce
        // protocole (production comme témoins de test) décide explicitement.
        lastPositionOnly: Bool
    ) throws -> (logits: MLXArray, preMixerHidden: MLXArray, reports: [Qwen4ExpStreamingLayerReport])
    func resetConversation()
    func resetNGramCacheStats()
    func ngramCacheStats() -> Qwen4ExpNGramCacheStats
}

extension Qwen4ExpStreamingTextModel: Qwen4ExpBatchForwardModel {
    public func batchForward(
        inputIDs: MLXArray, positionIDs: MLXArray?, leftPadding: [Int]?,
        lastPositionOnly: Bool
    ) throws -> (logits: MLXArray, preMixerHidden: MLXArray, reports: [Qwen4ExpStreamingLayerReport]) {
        try forward(
            inputIDs: inputIDs, positionIDs: positionIDs, leftPadding: leftPadding,
            lastPositionOnly: lastPositionOnly)
    }
}

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
///    à chaque pas (B est borné par `--batch-size`, donc des réglages
///    hétérogènes par ligne ne sont pas le goulot). Le goulot mesuré
///    (défaut 2 du 2026-09-14, facteur ×10 contre `flash-batch-probe`)
///    n'était pas le nombre de graphes construits mais le nombre de
///    synchronisations GPU : `run()` construit toujours B graphes
///    d'échantillonnage paresseux distincts (un par ligne, avec SON propre
///    sampler/masque — jamais mélangés), mais ne les matérialise plus
///    qu'avec UN seul `eval` par pas, exactement comme `flash-batch-probe`
///    le fait pour son cas homogène `[B, V]`.
///  - **Gâchis de calcul assumé, livraison immédiate.** Une ligne qui
///    atteint son propre jeton d'arrêt ou son propre `maxNewTokens`
///    continue d'être traitée par le `forward` partagé (remplie du jeton de
///    remplissage) jusqu'à ce que TOUTES les lignes du lot soient
///    terminées : faire sortir une ligne finie du lot de calcul en cours de
///    route est explicitement hors périmètre de cette étape (PLAN.md P12.3,
///    « lot continu » non traité ici). En revanche (défaut 1 du
///    2026-09-14), la LIVRAISON au client — la clôture de SA continuation —
///    n'attend plus la fin du lot : elle a lieu dès que la ligne est
///    terminée, voir `finishRow` dans `run()`.
public final class Qwen4ExpBatchStreamingGenerator: @unchecked Sendable {
    private let model: any Qwen4ExpBatchForwardModel

    /// Entrée de production, inchangée : `Qwen38FlashNextEngine.
    /// generateBatch` ne connaît toujours que le modèle réel.
    public init(model: Qwen4ExpStreamingTextModel) {
        self.model = model
    }

    /// P12.3 (défaut 1 du 2026-09-14) : seam de test — voir le commentaire
    /// de `Qwen4ExpBatchForwardModel`. `internal`, jamais appelé en
    /// production.
    init(forwardModel: any Qwen4ExpBatchForwardModel) {
        self.model = forwardModel
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
    ///
    /// P12.3 (défaut 1 du 2026-09-14) : `run()` referme maintenant la
    /// continuation d'une ligne dès qu'elle est terminée, pas seulement
    /// toutes ensemble à la fin — voir `finishRow` dans `run()`. Cela ne
    /// réintroduit PAS le risque ci-dessus : le serveur (`Qwen38Server`)
    /// tient son verrou d'exécution jusqu'à `completion.value`, jamais
    /// jusqu'à la consommation d'un flux individuel, donc aucune autre
    /// tâche ne peut toucher `model` avant que `run()` ne retourne
    /// réellement — qu'une ligne ait fermé son flux tôt ou tard n'y change
    /// rien.
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
        let prefill = try model.batchForward(
            inputIDs: promptArray, positionIDs: prefillPositionIDs, leftPadding: layout.leftPadding,
            lastPositionOnly: true)
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

        // P12.3 (défaut 1 du 2026-09-14) : `closed[row]` est vrai dès que la
        // continuation `row` a été refermée — que ce soit tôt, dans la
        // boucle ci-dessous, ou tardivement, dans le nettoyage final. Rend
        // `finishRow` idempotente : elle peut être appelée une fois de plus
        // sans risque (défense en profondeur, voir son commentaire).
        var closed = Array(repeating: false, count: batchSize)

        // P12.3 (défaut 1 du 2026-09-14) : livre `row` à son client dès
        // qu'elle est terminée, au lieu d'attendre que TOUTES les lignes du
        // lot le soient — c'était le bogue mesuré (un client à 16 jetons
        // attendait les 512 pas des trois autres). Capture `Date()` à
        // l'appel : `decodeTime` devient donc le temps réellement écoulé
        // pour CETTE ligne jusqu'à SA livraison, pas celui du lot entier —
        // ce n'est plus un « coût individuel fictif » (l'ancien commentaire
        // à cet endroit) puisque la livraison, désormais, a réellement lieu
        // à cet instant.
        //
        // Garantie de verrou préservée : cette fonction ne fait QUE refermer
        // des continuations, jamais toucher `model` au-delà des lectures en
        // lecture seule ci-dessous (`ngramCacheStats()`) — la seule écriture
        // sur `model` de tout `run()`, `resetConversation()`, reste à SA
        // place actuelle, après la boucle `for step` et avant le nettoyage
        // final ci-dessous. Les appels précoces à `finishRow` (dans la
        // boucle) s'exécutent donc tous AVANT cette réinitialisation dans
        // l'ordre du programme ; seul le nettoyage final — les continuations
        // qui restent encore ouvertes une fois la boucle terminée — s'exécute
        // après. `resetConversation()` s'exécute donc toujours avant la
        // fermeture des continuations RESTANTES, exactement la garantie du
        // correctif du 2026-09-13 (voir son commentaire ci-dessus) ; elle
        // n'a jamais promis d'ordonner les lignes déjà refermées tôt, qui ne
        // dépendent pas de cette réinitialisation.
        func finishRow(_ row: Int) {
            guard !closed[row] else { return }
            closed[row] = true
            let now = Date()
            let summary = Qwen4ExpGenerationSummary(
                tokenIDs: generated[row],
                promptTokenCount: rows[row].tokenIDs.count,
                timeToFirstToken: firstTokenTime[row],
                prefillTime: prefillEnd.timeIntervalSince(started),
                decodeTime: generationStarted ? now.timeIntervalSince(prefillEnd) : 0,
                layerVisitCount: prefill.reports.count,
                layerLoadTime: prefill.reports.reduce(0) { $0 + $1.loadDuration },
                activeMemoryBytes: Memory.activeMemory,
                peakMemoryBytes: Memory.peakMemory,
                ngramCacheStats: model.ngramCacheStats())
            continuations[row].yield(.finished(summary))
            continuations[row].finish()
        }

        let globalMaxNewTokens = rows.map(\.maxNewTokens).max() ?? 0
        for step in 0 ..< globalMaxNewTokens {
            if finished.allSatisfy({ $0 }) { break }
            var nextTokens = [Int32](repeating: padTokenID, count: batchSize)

            // P12.3 (défaut 2 du 2026-09-14) : construit les graphes
            // d'échantillonnage des lignes actives SANS les évaluer —
            // `LogitSampler.sample(logits:)` (MLXLMCommon) et
            // `Qwen4ExpLogitPenalizer.apply` restent tous les deux
            // paresseux, aucun `.item()`/`eval` caché à l'intérieur. Chaque
            // ligne garde SON propre sampler (`samplers[row]`) et SON propre
            // masque (`seenMasks[row]`) — le calcul par ligne, donc sa
            // parité avec le chemin mono-séquence et le critère de
            // non-contamination, reste identique octet pour octet à avant
            // ce correctif ; seule la matérialisation change.
            var activeRows: [Int] = []
            activeRows.reserveCapacity(batchSize)
            var sampledPerRow: [MLXArray] = []
            sampledPerRow.reserveCapacity(batchSize)
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
                activeRows.append(row)
                sampledPerRow.append(samplers[row].sample(logits: sampledLogits))
            }

            // UNE seule synchronisation GPU pour tout le pas, quel que soit
            // le nombre de lignes actives — contre B avant ce correctif
            // (mesuré : 864 ms/pas à B=4, contre 86 ms/pas pour
            // `flash-batch-probe` sur le même modèle et le même lot).
            let batchSampled =
                sampledPerRow.count == 1 ? sampledPerRow[0] : concatenated(sampledPerRow, axis: 0)
            eval(batchSampled)
            let sampledTokens = batchSampled.asArray(Int32.self)

            for (index, row) in activeRows.enumerated() {
                let token = sampledTokens[index]
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
                    // Défaut 1 : livraison immédiate — voir `finishRow`.
                    finishRow(row)
                }
            }
            if finished.allSatisfy({ $0 }) { break }
            let nextInput = MLXArray(nextTokens).reshaped([batchSize, 1])
            let decodePositionIDs = qwen4ExpLeftPaddedDecodePositionIDsArray(
                tokenCounts: tokenCounts, step: step)
            let stepResult = try model.batchForward(
                inputIDs: nextInput, positionIDs: decodePositionIDs, leftPadding: layout.leftPadding,
                lastPositionOnly: true)
            eval(stepResult.logits)
            logits = stepResult.logits[0..., -1, 0...]
        }

        if generationStarted {
            profiler.endGeneration(tokenCount: generated.reduce(0) { $0 + $1.count })
        }
        // P12.3 (correctif du 2026-09-13, crash mémoire) : toujours
        // stateless — ne laisse aucun état de ce lot survivre pour la
        // requête suivante — mais cette réinitialisation doit s'exécuter
        // AVANT de refermer les continuations RESTANTES ci-dessous (celles
        // qu'aucune ligne finie tôt n'a déjà refermées — voir `finishRow` et
        // `closed` plus haut), jamais après. `run()` s'exécute entièrement
        // sur une seule tâche, sans aucun point de suspension avant son
        // retour : placer la réinitialisation ici garantit qu'elle est
        // terminée avant que qui que ce soit, en aval, ne puisse observer la
        // fin du LOT — indépendamment de `completion` (voir son
        // commentaire), en défense en profondeur.
        model.resetConversation()
        // Nettoyage final : referme toute ligne qui ne l'aurait pas déjà été
        // par `finishRow` dans la boucle ci-dessus. En pratique, avec
        // `maxNewTokens > 0` garanti par la validation en tête de `run()`,
        // chaque ligne atteint forcément son propre jeton d'arrêt ou son
        // propre `maxNewTokens` au plus tard à l'itération
        // `rows[row].maxNewTokens - 1` de la boucle `for step`, donc avant
        // la sortie de cette boucle — cette seconde passe ne devrait plus
        // rien avoir à fermer ; elle reste en défense en profondeur (une
        // ligne dont `finished[row]` ne serait jamais devenu vrai, par
        // exemple) plutôt qu'un chemin normalement emprunté.
        for row in 0 ..< batchSize {
            finishRow(row)
        }
    }
}
