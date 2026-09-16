import AppKit
import Foundation
import Qwen38Agent
import SwiftUI

/// Panneau « Agent » : fait piloter au modèle local les outils
/// `list_files`/`read_file`/`grep` sur un dossier réel, en parlant au
/// serveur d'inférence en HTTP — comme n'importe quel client.
///
/// Contrainte d'architecture délibérée : ce panneau ne rebranche jamais
/// `runtime.generate`. Tout le chemin outils (analyse des `<tool_call>`,
/// typage des arguments, messages `role: tool`, cache de préfixe) a déjà été
/// validé de bout en bout côté serveur en §P13.1-P13.3 ; le dupliquer ici
/// rouvrirait des bugs déjà fermés. Si le serveur n'est pas démarré, le
/// panneau le dit et propose de le démarrer plutôt que de contourner.
@MainActor
final class AgentPanelViewModel: ObservableObject {
    @Published var rootURL: URL?
    @Published var task: String = ""
    /// Défaut mesuré §P13.4 : un pas coûte ~12 s une fois le cache de
    /// préfixe chaud, une tâche prend donc une à trois minutes à 16 pas.
    @Published var maxSteps: Int = 16
    /// Défaut §P13.4.
    @Published var maxTokensPerStep: Int = 900
    @Published var isRunning = false
    @Published var log: [AgentStepRecord] = []
    @Published var finalAnswer: String?
    @Published var errorMessage: String?
    @Published var stats = AgentStats()
    @Published var statusLabel = "Prêt"

    private var runTask: Task<Void, Never>?
    /// Empêche un second `start()` de faire tourner deux boucles à la fois
    /// (double-clic, ou relance pendant qu'une tâche tourne déjà en fond —
    /// ce dernier cas est redevenu possible une fois cet objet hissé dans
    /// `BenchViewModel` : avant, changer d'onglet détruisait cette instance
    /// avec sa tâche, ce qui masquait le problème). Extrait dans
    /// `Qwen38Agent.AgentRunGate` pour rester testable sans réseau.
    private var runGate = AgentRunGate()

    func chooseRootDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choisir le dossier racine que l'agent pourra explorer"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        rootURL = url
    }

    func start(baseURL: String, apiKey: String) {
        guard runGate.tryStart() else { return }
        guard let rootURL else {
            runGate.finish()
            errorMessage = "Choisis d'abord un dossier racine."
            return
        }
        let trimmedTask = task.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTask.isEmpty else {
            runGate.finish()
            errorMessage = "Écris la tâche à confier à l'agent."
            return
        }
        isRunning = true
        log = []
        finalAnswer = nil
        errorMessage = nil
        stats = AgentStats()
        statusLabel = "Connexion au serveur…"
        let steps = max(1, maxSteps)
        let tokens = max(64, maxTokensPerStep)
        runTask = Task { [weak self] in
            await self?.performRun(
                rootURL: rootURL, task: trimmedTask, maxSteps: steps, maxTokensPerStep: tokens,
                baseURL: baseURL, apiKey: apiKey)
        }
    }

    /// Interrompt la boucle proprement : la requête HTTP en cours, si elle
    /// n'est pas encore revenue, est annulée par `URLSession` ; le pas en
    /// cours n'est jamais appliqué à l'état.
    func stop() {
        runTask?.cancel()
        runTask = nil
        runGate.finish()
        isRunning = false
        statusLabel = "Arrêté par l'utilisateur"
    }

    /// Filet de sécurité si cette instance est un jour désallouée pendant
    /// qu'une boucle tourne (aujourd'hui elle ne l'est plus tant que l'app
    /// vit, `BenchViewModel` la retient — voir `Qwen38BenchUIApp.swift`) :
    /// `Task.cancel()` est sûr à appeler depuis n'importe quel contexte.
    deinit {
        runTask?.cancel()
    }

    private func performRun(
        rootURL: URL, task: String, maxSteps: Int, maxTokensPerStep: Int, baseURL: String, apiKey: String
    ) async {
        defer {
            isRunning = false
            runTask = nil
            runGate.finish()
        }

        let model: String
        do {
            model = try await fetchModelName(baseURL: baseURL)
        } catch {
            errorMessage = "Impossible de joindre le serveur : \(Self.describe(error))"
            statusLabel = "Échec de connexion"
            return
        }

        var engine = AgentLoopEngine(rootURL: rootURL, task: task)
        while true {
            if Task.isCancelled {
                statusLabel = "Arrêté par l'utilisateur"
                return
            }
            statusLabel = "Pas \(engine.stats.steps + 1)/\(maxSteps) en cours… (~12 s attendues)"
            let body = AgentWireFormat.buildRequestBody(
                model: model, messages: engine.messages, maxTokens: maxTokensPerStep, enableThinking: true)

            let turn: AgentModelTurn
            do {
                turn = try await postChatCompletion(baseURL: baseURL, apiKey: apiKey, body: body)
            } catch is CancellationError {
                statusLabel = "Arrêté par l'utilisateur"
                return
            } catch {
                errorMessage =
                    "L'appel au serveur a échoué au pas \(engine.stats.steps + 1) : \(Self.describe(error))"
                statusLabel = "Échec de l'appel serveur"
                return
            }

            let (record, outcome) = engine.apply(turn)
            log.append(record)
            stats = engine.stats

            if let result = AgentBudget.evaluate(
                outcome: outcome, stepsUsed: engine.stats.steps, maxSteps: maxSteps)
            {
                switch result {
                case .finalAnswer(let answer):
                    finalAnswer = answer
                    statusLabel = "Terminé · réponse finale obtenue en \(engine.stats.steps) pas"
                case .budgetExhausted:
                    errorMessage = "Budget de \(maxSteps) pas épuisé sans réponse finale."
                    statusLabel = "Budget épuisé"
                }
                return
            }
        }
    }

    private func fetchModelName(baseURL: String) async throws -> String {
        guard let url = URL(string: baseURL + "/healthz") else { throw AgentPanelError.invalidServerURL }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.checkHTTP(response, data: data)
        guard let model = AgentWireFormat.parseHealthResponse(data) else {
            throw AgentPanelError.noModelReported
        }
        return model
    }

    private func postChatCompletion(baseURL: String, apiKey: String, body: Data) async throws -> AgentModelTurn {
        guard let url = URL(string: baseURL + "/v1/chat/completions") else {
            throw AgentPanelError.invalidServerURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        request.httpBody = body
        // Mesuré §P13.3 : ~12 s par pas une fois le cache chaud, mais le
        // tout premier pas (aucun cache) peut coûter largement plus sur un
        // dépôt volumineux — large marge plutôt qu'un abandon prématuré.
        request.timeoutInterval = 1800
        let (data, response) = try await URLSession.shared.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try AgentWireFormat.parseChatCompletionResponse(data)
    }

    private static func checkHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw AgentPanelError.serverUnreachable("réponse HTTP invalide")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw AgentPanelError.httpStatus(http.statusCode, String(body.prefix(600)))
        }
    }

    private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return (error as NSError).localizedDescription
    }
}

private enum AgentPanelError: LocalizedError {
    case invalidServerURL
    case serverUnreachable(String)
    case httpStatus(Int, String)
    case noModelReported

    var errorDescription: String? {
        switch self {
        case .invalidServerURL: return "URL du serveur invalide."
        case .serverUnreachable(let detail): return "serveur injoignable (\(detail))"
        case .httpStatus(let code, let body): return "HTTP \(code) — \(body)"
        case .noModelReported: return "le serveur ne rapporte aucun modèle chargé (/healthz)"
        }
    }
}

struct AgentPanelView: View {
    @ObservedObject var model: BenchViewModel
    // §Agent panel : cet objet vit dans `BenchViewModel`, pas ici — un
    // `@StateObject` local était détruit (et sa tâche en cours avec lui) dès
    // que l'onglet changeait, puisque `ContentView` retire alors cette vue
    // de la hiérarchie (`if selectedTab == … else … AgentPanelView(...)`).
    // `@ObservedObject` sur l'instance partagée de `model` fait que revenir
    // sur l'onglet retrouve l'état exact, journal et compteurs compris.
    @ObservedObject var agent: AgentPanelViewModel

    init(model: BenchViewModel) {
        self.model = model
        self.agent = model.agentPanel
    }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            controlColumn
            Divider()
            journalColumn
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: colonne de contrôle

    private var controlColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Agent").font(.title3.weight(.semibold))
                    Text("Le modèle local pilote list_files, read_file et grep sur un dossier réel, en parlant au serveur en HTTP.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                serverBox
                rootBox
                taskBox
                budgetBox
                runControls

                Text(agent.statusLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if agent.stats.steps > 0 {
                    statsBox
                }
                if let errorMessage = agent.errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Spacer(minLength: 0)
            }
            .padding(20)
        }
        .frame(minWidth: 340, maxWidth: 380, maxHeight: .infinity, alignment: .topLeading)
    }

    private var serverBox: some View {
        GroupBox("Serveur") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Circle().fill(serverColor).frame(width: 8, height: 8)
                    Text(serverLabel).font(.caption).lineLimit(1)
                    Spacer()
                    if model.serverSnapshot.status != .running {
                        Button("Démarrer") { model.toggleServer() }
                            .buttonStyle(.bordered)
                            .disabled(
                                model.serverSnapshot.status == .starting
                                    || model.serverSnapshot.status == .stopping || !model.isLoaded)
                    }
                }
                Text(
                    "Le panneau parle au serveur en HTTP, comme n'importe quel client — c'est le chemin outils déjà validé de bout en bout (§P13)."
                )
                .font(.caption2)
                .foregroundStyle(.tertiary)
                if model.serverSnapshot.status != .running {
                    Text(
                        model.isLoaded
                            ? "Démarre le serveur ci-dessus avant de lancer une tâche."
                            : "Charge d'abord un modèle (onglet Chat), puis démarre le serveur."
                    )
                    .font(.caption2)
                    .foregroundStyle(.orange)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var rootBox: some View {
        GroupBox("Dossier racine") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(agent.rootURL?.path ?? "Aucun dossier choisi")
                        .font(.caption.monospaced())
                        .lineLimit(2)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Choisir…") { agent.chooseRootDirectory() }
                        .buttonStyle(.bordered)
                        .disabled(agent.isRunning)
                }
                Text("Tout chemin utilisé par un outil est résolu sous ce dossier ; ce qui en sort est refusé, jamais suivi.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.vertical, 4)
        }
    }

    private var taskBox: some View {
        GroupBox("Tâche") {
            TextEditor(text: $agent.task)
                .font(.callout)
                .frame(minHeight: 70, maxHeight: 110)
                .scrollContentBackground(.hidden)
                .padding(5)
                .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
                .disabled(agent.isRunning)
                .padding(.vertical, 4)
        }
    }

    private var budgetBox: some View {
        GroupBox("Budget") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Pas maximum").foregroundStyle(.secondary)
                    Spacer()
                    Stepper("\(agent.maxSteps)", value: $agent.maxSteps, in: 1...64)
                        .frame(width: 150)
                }
                HStack {
                    Text("Jetons par tour").foregroundStyle(.secondary)
                    Spacer()
                    TextField("900", value: $agent.maxTokensPerStep, format: .number)
                        .frame(width: 70)
                        .multilineTextAlignment(.trailing)
                    Stepper("", value: $agent.maxTokensPerStep, in: 64...8192, step: 50)
                        .labelsHidden()
                }
                Text(
                    "Réflexion active (obligatoire) : mesuré §P13.4 — sans elle, le modèle enchaîne des appels pertinents mais ne conclut jamais ; avec elle, quatre pas suffisent. Un pas coûte environ 12 s, donc une à trois minutes par tâche."
                )
                .font(.caption2)
                .foregroundStyle(.tertiary)
            }
            .padding(.vertical, 4)
        }
        .disabled(agent.isRunning)
    }

    private var runControls: some View {
        HStack(spacing: 10) {
            Button {
                agent.start(baseURL: model.serverSnapshot.url, apiKey: model.serverAPIKey)
            } label: {
                Label("Lancer", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(agent.isRunning || model.serverSnapshot.status != .running)

            Button {
                agent.stop()
            } label: {
                Label("Arrêter", systemImage: "stop.fill")
            }
            .buttonStyle(.bordered)
            .tint(.red)
            .disabled(!agent.isRunning)

            if agent.isRunning {
                ProgressView().controlSize(.small)
            }
            Spacer()
        }
    }

    private var statsBox: some View {
        GroupBox("Compteur") {
            // C'est la métrique qui a servi à juger tout le chantier §P13 :
            // nombre d'appels, dont combien bien formés (voir
            // docs/knowledge/log.md, P13.2/P13.3).
            HStack(spacing: 14) {
                statTile("Pas", "\(agent.stats.steps)")
                statTile("Appels", "\(agent.stats.calls)")
                statTile("Bien formés", "\(agent.stats.wellFormed)/\(agent.stats.calls)")
                statTile("Valides", "\(agent.stats.valid)/\(agent.stats.calls)")
            }
            .padding(.vertical, 4)
        }
    }

    private func statTile(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.callout.monospacedDigit())
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var serverColor: Color {
        switch model.serverSnapshot.status {
        case .running: return .green
        case .failed: return .orange
        case .starting, .stopping: return .yellow
        case .stopped: return .secondary
        }
    }

    private var serverLabel: String {
        switch model.serverSnapshot.status {
        case .stopped: return "Arrêté"
        case .starting: return "Démarrage…"
        case .running: return "En écoute · \(model.serverSnapshot.url)"
        case .stopping: return "Arrêt…"
        case .failed: return "Erreur"
        }
    }

    // MARK: journal

    private var journalColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Journal").font(.headline)
                Spacer()
                if !agent.log.isEmpty {
                    Text("\(agent.log.count) pas").font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .padding([.horizontal, .top], 18)
            .padding(.bottom, 8)

            if let finalAnswer = agent.finalAnswer {
                finalAnswerBanner(finalAnswer)
                    .padding(.horizontal, 18)
                    .padding(.bottom, 10)
            }

            if agent.log.isEmpty {
                ContentUnavailableView(
                    "Aucun pas encore",
                    systemImage: "list.bullet.rectangle",
                    description: Text("Choisis un dossier, écris une tâche, puis lance l'agent."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(agent.log) { step in
                            stepRow(step)
                        }
                    }
                    .padding(.horizontal, 18)
                    .padding(.bottom, 18)
                }
            }
        }
        .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func finalAnswerBanner(_ answer: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Réponse finale", systemImage: "checkmark.seal.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.green)
            Text(answer)
                .font(.body)
                .textSelection(.enabled)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.green.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private func stepRow(_ step: AgentStepRecord) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Pas \(step.stepIndex)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            if let textOnly = step.textOnly {
                Text("Réponse texte, sans appel d'outil — la tâche n'avance pas :")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                Text(textOnly.isEmpty ? "(vide)" : textOnly)
                    .font(.callout)
                    .textSelection(.enabled)
            } else {
                ForEach(Array(step.toolCalls.enumerated()), id: \.offset) { _, call in
                    toolCallRow(call)
                }
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
    }

    private func toolCallRow(_ call: AgentToolCallExecutionRecord) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                if call.isFinalAnswer {
                    Text("Termine la boucle avec cette réponse.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text(call.resultPreview.isEmpty ? "(résultat vide)" : call.resultPreview)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: call.isFinalAnswer ? "checkmark.circle" : "wrench.and.screwdriver")
                    .foregroundStyle(call.isFinalAnswer ? .green : .cyan)
                Text("\(call.name)(\(call.argumentsJSON.prefix(80)))")
                    .font(.caption.monospaced())
                    .lineLimit(1)
                if !call.wellFormed {
                    Label("malformé", systemImage: "exclamationmark.triangle")
                        .font(.caption2)
                        .foregroundStyle(.red)
                } else if !call.valid {
                    Label("incomplet", systemImage: "questionmark.circle")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
        }
    }
}
