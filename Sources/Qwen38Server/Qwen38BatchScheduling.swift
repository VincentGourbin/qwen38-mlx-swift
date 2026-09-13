import Foundation
import Qwen38Core

/// P12.3 (voir PLAN.md « P12 » et docs/knowledge/log.md, 2026-09-13
/// « P12.3 ») : ordonnanceur de lot pour `serve --batch-size N`. Ce fichier
/// est entièrement additif — rien ici n'est référencé quand
/// `--batch-size` vaut 1 (le défaut) : `Qwen38InferenceServer.start` ne
/// construit un `Qwen38BatchCoordinator` et n'enregistre la route batchée
/// que pour `batchSize > 1` (voir `Qwen38Server.swift`).
///
/// ## La règle de regroupement : ancre + voisins de longueur
///
/// P12.2 a mesuré qu'un lot de longueurs mêlées coûte **+24 %** par pas
/// contre un lot homogène (le lot avance au rythme de sa plus longue
/// séquence, remplissage à gauche oblige). `qwen38FormBatch` retient donc,
/// parmi les requêtes froides en attente, la plus ANCIENNE comme ancre, puis
/// complète le lot avec celles dont la longueur de prompt est la plus
/// proche de la sienne — l'ancienneté départageant les égalités. Deux
/// propriétés en découlent, toutes deux voulues :
///  - Choisir l'ancre la plus ancienne borne l'attente d'une requête à la
///    fenêtre de regroupement, quelle que soit sa longueur — pas de famine.
///  - Le critère ignore délibérément « remplir le lot à tout prix » : mieux
///    vaut un lot de 2 séquences très proches en longueur qu'un lot de 8
///    dont une seule fait diverger le remplissage.
public struct Qwen38BatchCandidate: Sendable, Equatable {
    public let id: UUID
    public let promptTokenCount: Int
    public let arrivalIndex: Int

    public init(id: UUID, promptTokenCount: Int, arrivalIndex: Int) {
        self.id = id
        self.promptTokenCount = promptTokenCount
        self.arrivalIndex = arrivalIndex
    }
}

/// Constitue un lot d'au plus `batchSize` candidats à partir de `waiting`
/// (peut être vide, auquel cas le résultat est vide). Logique pure, sans
/// aucun effet de bord — testable sans checkpoint ni device Metal.
public func qwen38FormBatch(
    waiting: [Qwen38BatchCandidate], batchSize: Int
) -> [Qwen38BatchCandidate] {
    guard !waiting.isEmpty, batchSize > 0 else { return [] }
    guard let anchor = waiting.min(by: { $0.arrivalIndex < $1.arrivalIndex }) else { return [] }
    let ranked = waiting.sorted { a, b in
        let distanceA = abs(a.promptTokenCount - anchor.promptTokenCount)
        let distanceB = abs(b.promptTokenCount - anchor.promptTokenCount)
        if distanceA != distanceB { return distanceA < distanceB }
        return a.arrivalIndex < b.arrivalIndex
    }
    return Array(ranked.prefix(batchSize))
}

/// Erreurs internes à l'ordonnanceur — jamais une condition normale, un
/// filet de sécurité si `runBatch` rend un nombre de flux différent du
/// nombre de requêtes qui les attendaient.
public enum Qwen38BatchCoordinatorError: LocalizedError, Equatable {
    case streamCountMismatch(expected: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .streamCountMismatch(let expected, let got):
            return "Lot P12.3 : \(got) flux rendus pour \(expected) requête(s) attendue(s) (bug de l'ordonnanceur)."
        }
    }
}

/// Résultat de `Qwen38BatchCoordinator.join` : soit la requête doit être
/// servie seule par l'appelant (le chemin mono-séquence d'aujourd'hui,
/// inchangé — voir PLAN.md P12.3, « une requête sans compagnie tourne comme
/// avant »), soit elle a rejoint un lot et reçoit directement son propre
/// flux d'événements.
public enum Qwen38BatchJoinResult: Sendable {
    case solo
    case batched(stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, batchSizeServed: Int)
}

/// Coordonne le regroupement des requêtes « froides » (voir
/// `Qwen38InferenceServer`'s nouvelle route batchée) derrière une fenêtre
/// d'attente courte. Un acteur : la file d'attente et l'horloge de fenêtre
/// n'ont besoin d'aucune synchronisation supplémentaire.
///
/// **Fenêtre de regroupement : 30 ms**, choisie sans mesure formelle (le
/// point n'a pas encore été instrumenté — à affiner par la mesure demandée
/// en P12.3) mais défendable par construction : c'est un ordre de grandeur
/// sous le TTFT du chemin froid observé en LAN (plusieurs centaines de
/// millisecondes à plusieurs secondes, PLAN.md P12), donc son coût sur le
/// TTFT d'un client isolé (qui attend la fenêtre entière avant de partir
/// seul) reste marginal, tout en étant assez large pour absorber des
/// requêtes qui arrivent à quelques dizaines de millisecondes d'écart sur
/// un LAN. Non exposée en option pour l'instant — voir le rapport P12.3.
public actor Qwen38BatchCoordinator {
    public struct Request: Sendable {
        public let messages: [Qwen38ChatMessage]
        public let options: Qwen38GenerationOptions

        public init(messages: [Qwen38ChatMessage], options: Qwen38GenerationOptions) {
            self.messages = messages
            self.options = options
        }
    }

    private struct Pending {
        let candidate: Qwen38BatchCandidate
        let request: Request
        let continuation: CheckedContinuation<Qwen38BatchJoinResult, Error>
    }

    private var pending: [Pending] = []
    private var nextArrivalIndex = 0
    private var windowTask: Task<Void, Never>?
    private let batchSize: Int
    private let window: Duration
    private let runBatch: @Sendable ([Request]) async throws -> [AsyncThrowingStream<Qwen38GenerationEvent, Error>]

    public init(
        batchSize: Int,
        window: Duration = .milliseconds(30),
        runBatch: @escaping @Sendable ([Request]) async throws -> [AsyncThrowingStream<Qwen38GenerationEvent, Error>]
    ) {
        self.batchSize = max(batchSize, 1)
        self.window = window
        self.runBatch = runBatch
    }

    /// Enregistre une requête froide et attend soit d'être détournée vers le
    /// chemin mono-séquence (`.solo`), soit de recevoir son propre flux au
    /// sein d'un lot (`.batched`). Ne lève que si le lot auquel cette
    /// requête a fini par appartenir a échoué à démarrer.
    public func join(promptTokenCount: Int, request: Request) async throws -> Qwen38BatchJoinResult {
        try await withCheckedThrowingContinuation { continuation in
            let candidate = Qwen38BatchCandidate(
                id: UUID(), promptTokenCount: promptTokenCount, arrivalIndex: nextArrivalIndex)
            nextArrivalIndex += 1
            pending.append(Pending(candidate: candidate, request: request, continuation: continuation))
            if pending.count >= batchSize {
                dispatch()
            } else {
                scheduleWindow()
            }
        }
    }

    /// Résout immédiatement toute requête en attente avec `.solo` — appelé à
    /// l'arrêt du serveur pour ne jamais laisser un client suspendu
    /// indéfiniment derrière une fenêtre qui ne se déclenchera plus.
    public func drain() {
        windowTask?.cancel()
        windowTask = nil
        let remaining = pending
        pending.removeAll()
        for member in remaining {
            member.continuation.resume(returning: .solo)
        }
    }

    private func scheduleWindow() {
        guard windowTask == nil else { return }
        windowTask = Task { [weak self, window] in
            try? await Task.sleep(for: window)
            guard !Task.isCancelled else { return }
            await self?.windowFired()
        }
    }

    private func windowFired() {
        windowTask = nil
        guard !pending.isEmpty else { return }
        dispatch()
    }

    private func dispatch() {
        windowTask?.cancel()
        windowTask = nil
        guard !pending.isEmpty else { return }
        let candidates = pending.map(\.candidate)
        let chosenIDs = Set(qwen38FormBatch(waiting: candidates, batchSize: batchSize).map(\.id))
        let group = pending.filter { chosenIDs.contains($0.candidate.id) }
        pending.removeAll { chosenIDs.contains($0.candidate.id) }

        if group.count <= 1 {
            for member in group {
                member.continuation.resume(returning: .solo)
            }
        } else {
            let requests = group.map(\.request)
            let batchSizeServed = group.count
            Task {
                do {
                    let rawStreams = try await self.runBatch(requests)
                    guard rawStreams.count == group.count else {
                        let error = Qwen38BatchCoordinatorError.streamCountMismatch(
                            expected: group.count, got: rawStreams.count)
                        for member in group { member.continuation.resume(throwing: error) }
                        return
                    }
                    for (member, stream) in zip(group, rawStreams) {
                        member.continuation.resume(
                            returning: .batched(stream: stream, batchSizeServed: batchSizeServed))
                    }
                } catch {
                    for member in group {
                        member.continuation.resume(throwing: error)
                    }
                }
            }
        }

        // Les candidats non retenus restent en file : leur fenêtre doit
        // repartir, celle-ci vient d'être consommée par ce round.
        if !pending.isEmpty {
            scheduleWindow()
        }
    }
}

/// Referme un flux dérivé d'une exécution mono-séquence (le chemin
/// chaud/seul de `chatCompletionsResponseBatched`, une seule ligne) :
/// `onDone` n'est appelé qu'une fois que ce flux est épuisé.
///
/// **N'est plus utilisée pour le lot proprement dit depuis le correctif du
/// 2026-09-13** (crash mémoire) : la consommation des flux d'un
/// `Qwen4ExpBatchStreamingGenerator` n'est PAS un signal sûr de fin
/// d'exécution — voir `Qwen38BatchGenerationResult`'s commentaire — parce
/// que `continuation.finish()` réveille son lecteur en aval de façon
/// asynchrone, potentiellement avant que le producteur n'ait fini
/// d'exécuter son propre nettoyage. Ce type reste correct pour une seule
/// ligne mono-séquence (`Qwen4ExpStreamingGenerator`) précisément parce que
/// ce générateur-là ne touche plus jamais `model` après avoir refermé sa
/// continuation — voir le commentaire d'appel dans `Qwen38Server.swift`.
actor Qwen38BatchCompletionGate {
    private var remaining: Int
    private var onDone: (@Sendable () -> Void)?

    init(rowCount: Int, onDone: @escaping @Sendable () -> Void) {
        remaining = rowCount
        self.onDone = onDone
    }

    func rowFinished() {
        remaining -= 1
        guard remaining <= 0, let onDone else { return }
        onDone()
        self.onDone = nil
    }
}

/// Relaie `stream` événement par événement (sans en retarder ni en
/// coalescer aucun — le vrai flux incrémental produit par le générateur,
/// pas une version tamponnée) et signale `gate` une fois que ce flux
/// particulier est épuisé, avec ou sans erreur.
func qwen38AttachCompletionGate(
    _ stream: AsyncThrowingStream<Qwen38GenerationEvent, Error>, gate: Qwen38BatchCompletionGate
) -> AsyncThrowingStream<Qwen38GenerationEvent, Error> {
    AsyncThrowingStream { continuation in
        let task = Task {
            do {
                for try await event in stream {
                    continuation.yield(event)
                }
                await gate.rowFinished()
                continuation.finish()
            } catch {
                await gate.rowFinished()
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}
