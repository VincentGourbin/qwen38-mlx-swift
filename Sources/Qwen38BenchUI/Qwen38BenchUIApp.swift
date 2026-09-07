import AppKit
import Foundation
import Qwen38Core
import Qwen38Server
import MLXProfiler
import SwiftUI
import UniformTypeIdentifiers

final class Qwen38AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct ChatMessage: Identifiable {
    enum Role {
        case user
        case assistant
    }

    let id = UUID()
    let role: Role
    var text: String
    var reasoning: String = ""
    var imageURL: URL?
    var isStreaming = false
}

/// One entry of the H4.1 model picker: a directory below `modelPath`'s
/// parent that `Qwen38ModelValidator` accepts, with its family and size —
/// discovery is the single source of truth, no hardcoded variant list.
struct Qwen38DiscoveredModel: Identifiable, Equatable {
    let id: String
    let url: URL
    let family: Qwen38ModelFamily?
    let sizeBytes: Int64

    var sizeDescription: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    var familyDescription: String {
        switch family {
        case .qwen35: return "27B"
        case .qwen4Exp: return "Flash-Next"
        case nil: return "?"
        }
    }
}

@MainActor
final class BenchViewModel: ObservableObject {
    @Published var modelPath = "/Volumes/Lexar/models/mlx-community/Qwen3.8-27B-4bit"
    @Published var catalog: [Qwen38DiscoveredModel] = []
    @Published var flashLoadProgress: (visited: Int, total: Int)?
    @Published var loadedFamily: Qwen38ModelFamily?

    var loadedFamilyLabel: String {
        switch loadedFamily {
        case .qwen35: return "27B"
        case .qwen4Exp: return "Flash-Next"
        case nil: return "Qwen3.8"
        }
    }
    @Published var prompt = ""
    @Published var output = ""
    @Published var status = "Prêt"
    @Published var isBusy = false
    @Published var isLoaded = false
    @Published var thinking = true
    @Published var reasoningEffort = "xhigh"
    @Published var maxTokens = 2048
    @Published var mtpEnabled = true
    @Published var mtpEngine: Qwen38MTPEngine = .local
    @Published var mtpDraftTokens = 1
    @Published var mtpAvailability: Qwen38MTPAvailability = .unavailable
    @Published var imageURL: URL?
    @Published var metrics: LLMMetrics?
    @Published var report = ""
    @Published var activeMemoryBytes = 0
    @Published var peakMemoryBytes = 0
    @Published var acceptRate: Double?
    @Published var timeToFirstToken: TimeInterval?
    @Published var loadDuration: TimeInterval?
    @Published var turnHistory: [Qwen38RunMetrics] = []
    @Published var messages: [ChatMessage] = []
    @Published var serverPort = "8848"
    @Published var serverAPIKey = ""
    @Published var serverSnapshot = Qwen38ServerSnapshot(
        status: .stopped, port: Qwen38InferenceServer.defaultPort,
        url: "http://127.0.0.1:8848", activeSessions: 0, queuedSessions: 0, sessions: [])

    let runtime: Qwen38Runtime
    private let inferenceServer: Qwen38InferenceServer
    private var traceData: Data?

    init() {
        let runtime = Qwen38Runtime()
        self.runtime = runtime
        self.inferenceServer = Qwen38InferenceServer(runtime: runtime)
    }

    func loadModel() {
        guard !isBusy else { return }
        isBusy = true
        status = "Chargement du modèle…"
        flashLoadProgress = nil
        let path = modelPath
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        // Read family/layer count up front — the H4.2 progress bar's
        // denominator, and the header's family label. A read failure here
        // just falls back to defaults instead of blocking the load; the
        // real error, if any, still surfaces from runtime.load below.
        let info = try? Qwen38ModelValidator.readInfo(from: directory)
        let expectedLayers = info?.numHiddenLayers ?? 48
        Task { @MainActor in
            let start = Date()
            do {
                try await runtime.load(from: directory)
                loadedFamily = info?.family
                if await runtime.isFlashNextLoaded {
                    status = "Chargement Flash-Next (résident, ~100 s depuis le Lexar)…"
                    flashLoadProgress = (0, expectedLayers)
                    if let progress = await runtime.flashNextWarmUp() {
                        for await visited in progress {
                            flashLoadProgress = (visited, expectedLayers)
                        }
                    }
                }
                mtpAvailability = await runtime.mtpState
                loadDuration = Date().timeIntervalSince(start)
                isLoaded = true
                status = mtpAvailability == .active
                    ? "Modèle VLM chargé · drafter MTP prêt"
                    : "Modèle VLM chargé — prêt à inférer"
            } catch {
                isLoaded = false
                loadedFamily = nil
                status = "Erreur de chargement : \(error.localizedDescription)"
            }
            flashLoadProgress = nil
            isBusy = false
        }
    }

    func unloadModel() {
        guard !isBusy else { return }
        isLoaded = false
        loadedFamily = nil
        status = "Modèle déchargé"
        Task {
            await runtime.unload()
        }
    }

    func resetConversation() {
        guard !isBusy else { return }
        output = ""
        metrics = nil
        report = ""
        traceData = nil
        acceptRate = nil
        timeToFirstToken = nil
        turnHistory = []
        messages = []
        status = "Nouvelle conversation"
        Task {
            await runtime.resetConversation()
        }
    }

    func modelPathChanged() {
        guard isLoaded else { return }
        isLoaded = false
        loadDuration = nil
        metrics = nil
        timeToFirstToken = nil
        status = "Chemin modifié — recharge nécessaire"
    }

    /// H4.1: discover every valid Qwen3.8 checkpoint below `modelPath`'s
    /// parent directory — replaces the old hardcoded 27B 4-bit/8-bit/BF16
    /// button list, and is how Flash-Next becomes selectable at all.
    func refreshCatalog() {
        let parent = URL(fileURLWithPath: modelPath, isDirectory: true)
            .deletingLastPathComponent()
        let discovered = Qwen38ModelCatalog.discover(in: parent)
        catalog = discovered.map { id, url in
            let info = try? Qwen38ModelValidator.readInfo(from: url)
            return Qwen38DiscoveredModel(
                id: id, url: url, family: info?.family,
                sizeBytes: Qwen38ModelCatalog.sizeOnDisk(url))
        }.sorted { $0.id < $1.id }
    }

    func selectDiscoveredModel(_ discovered: Qwen38DiscoveredModel) {
        guard !isBusy else { return }
        modelPath = discovered.url.path
        isLoaded = false
        mtpAvailability = .unavailable
        metrics = nil
        turnHistory = []
        output = ""
        messages = []
        flashLoadProgress = nil
        status = "\(discovered.id) sélectionné — charge le modèle"
    }

    func run() {
        guard !isBusy else { return }
        guard isLoaded else {
            status = "Charge le modèle avant d’inférer"
            return
        }
        let cleanedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedPrompt.isEmpty else {
            status = "Écris un message avant d’inférer"
            return
        }
        isBusy = true
        let turnNumber = turnHistory.count + 1
        let userMessage = ChatMessage(role: .user, text: cleanedPrompt, imageURL: imageURL)
        let assistantMessage = ChatMessage(role: .assistant, text: "", isStreaming: true)
        messages.append(userMessage)
        messages.append(assistantMessage)
        let assistantMessageID = assistantMessage.id
        // Capture the attachment before clearing the composer. The visual
        // message keeps its URL, while the runtime must receive it as input.
        let imageURL = self.imageURL
        // The next turn is entered in the composer. The actual conversation
        // state remains in Qwen38Runtime's persistent ChatSession/KV cache.
        self.prompt = ""
        self.imageURL = nil
        metrics = nil
        acceptRate = nil
        activeMemoryBytes = 0
        peakMemoryBytes = 0
        timeToFirstToken = nil
        let inputDescription = imageURL == nil ? "texte" : "1 image envoyée"
        status = "Inférence du tour \(turnNumber) · \(inputDescription)…"
        let prompt = cleanedPrompt
        let maxTokens = self.maxTokens
        let thinking = self.thinking
        let reasoningEffort = self.reasoningEffort
        let mtpEnabled = self.mtpEnabled
        let mtpEngine = self.mtpEngine
        let mtpDraftTokens = self.mtpDraftTokens

        Task { @MainActor in
            do {
                let options = Qwen38GenerationOptions(
                    maxTokens: maxTokens,
                    enableThinking: thinking,
                    reasoningEffort: reasoningEffort,
                    mtp: Qwen38MTPOptions(
                        enabled: mtpEnabled,
                        draftDepth: .fixed(mtpDraftTokens),
                        engine: mtpEngine
                    )
                )
                let stream = try await runtime.generate(
                    prompt: prompt,
                    imageURLs: imageURL.map { [$0] } ?? [],
                    options: options
                )
                // Qwen streams reasoning and answer content through the same
                // runtime event.  The server already exposes the two fields
                // separately; keep the GUI equally strict and route reasoning
                // to its collapsible bubble instead of the answer bubble. With thinking on,
                // the template starts inside <think>; with thinking off it
                // starts directly in the response channel.
                var visibleParser = Qwen38ThinkingStreamParser(
                    primedInside: thinking)
                for try await event in stream {
                    switch event {
                    case .chunk(let chunk):
                        let parsed = visibleParser.append(chunk)
                        if !parsed.reasoning.isEmpty,
                           let index = messages.firstIndex(where: { $0.id == assistantMessageID }) {
                            messages[index].reasoning += parsed.reasoning
                        }
                        if !parsed.content.isEmpty {
                            output += parsed.content
                            if let index = messages.firstIndex(where: { $0.id == assistantMessageID }) {
                                messages[index].text += parsed.content
                            }
                        }
                    case .metrics(let run):
                        let tail = visibleParser.finish()
                        if let index = messages.firstIndex(where: { $0.id == assistantMessageID }) {
                            if !tail.reasoning.isEmpty {
                                messages[index].reasoning += tail.reasoning
                            }
                            if !tail.content.isEmpty {
                                messages[index].text += tail.content
                            }
                        }
                        if !tail.content.isEmpty {
                            output += tail.content
                        }
                        metrics = run.metrics
                        report = run.report
                        traceData = run.chromeTrace
                        activeMemoryBytes = run.activeMemoryBytes
                        peakMemoryBytes = run.peakMemoryBytes
                        acceptRate = run.acceptRate
                        mtpAvailability = run.mtpStatus.availability
                        timeToFirstToken = run.timeToFirstToken
                        turnHistory.append(run)
                        if let index = messages.firstIndex(where: { $0.id == assistantMessageID }) {
                            messages[index].isStreaming = false
                        }
                        status = run.conversationReplayed
                            ? "Tour \(run.turnIndex) terminé · historique rejoué (M1 MTP)"
                            : run.cacheReused
                                ? "Tour \(run.turnIndex) terminé · cache KV réutilisé"
                                : "Tour \(run.turnIndex) terminé · cache prêt pour la suite"
                    }
                }
            } catch {
                if let index = messages.firstIndex(where: { $0.id == assistantMessageID }) {
                    messages[index].text = "Erreur : \(error.localizedDescription)"
                    messages[index].isStreaming = false
                }
                status = "Erreur : \(error.localizedDescription)"
            }
            isBusy = false
        }
    }

    func exportTrace() {
        guard let traceData else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "qwen38-run.trace.json"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            try? traceData.write(to: url, options: .atomic)
        }
    }

    func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK else { return }
        imageURL = panel.url
    }

    func clearImage() {
        imageURL = nil
    }

    func toggleServer() {
        if serverSnapshot.status == .running || serverSnapshot.status == .starting {
            Task { @MainActor in
                await inferenceServer.stop()
                await refreshServer()
            }
            return
        }
        guard let port = Int(serverPort), (1 ... 65_535).contains(port) else {
            status = "Port serveur invalide"
            return
        }
        let key = serverAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        Task { @MainActor in
            do {
                try await inferenceServer.start(
                    port: port,
                    apiKey: key.isEmpty ? nil : key,
                    modelsDirectory: URL(fileURLWithPath: modelPath, isDirectory: true)
                        .deletingLastPathComponent())
                await refreshServer()
                status = "Serveur LAN actif sur le port \(port)"
            } catch {
                status = "Erreur serveur : \(error.localizedDescription)"
                await refreshServer()
            }
        }
    }

    func refreshServer() async {
        serverSnapshot = await inferenceServer.snapshot()
    }

    var openCodeConfiguration: String {
        let discovered = serverSnapshot.availableModels
        let modelIDs = discovered.isEmpty
            ? ["Qwen3.8-27B-4bit", "Qwen3.8-27B-8bit", "Qwen3.8-27B-bf16"]
            : discovered
        let defaultID = serverSnapshot.loadedModel ?? modelIDs.first ?? "Qwen3.8-27B-4bit"
        let port = serverSnapshot.port
        let apiKey = serverAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)

        let modelEntries = Dictionary(uniqueKeysWithValues: modelIDs.map { id in
            (id, [
                "name": "Qwen3.8 local — (displayName(for: id))",
                "reasoning": true,
                "compatibility": ["reasoningField": "reasoning_content"],
            ] as [String: Any])
        })
        let configuration: [String: Any] = [
            "$schema": "https://opencode.ai/config.json",
            "model": "qwen38/\(defaultID)",
            "provider": [
                "qwen38": [
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Qwen3.8 local",
                    "options": [
                        "baseURL": "http://127.0.0.1:\(port)/v1",
                        "apiKey": apiKey.isEmpty ? "local" : apiKey,
                    ],
                    "models": modelEntries,
                ] as [String: Any]
            ] as [String: Any],
        ]

        guard JSONSerialization.isValidJSONObject(configuration),
              let data = try? JSONSerialization.data(
                withJSONObject: configuration, options: [.prettyPrinted, .sortedKeys]),
              let result = String(data: data, encoding: .utf8)
        else { return "" }
        return result
    }

    func copyOpenCodeConfiguration() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(openCodeConfiguration, forType: .string)
        status = "Configuration OpenCode copiée"
    }

    private func displayName(for id: String) -> String {
        id
            .replacingOccurrences(of: "Qwen3.8-27B-", with: "")
            .replacingOccurrences(of: "4bit", with: "4-bit")
            .replacingOccurrences(of: "8bit", with: "8-bit")
            .replacingOccurrences(of: "bf16", with: "BF16")
    }
}

struct ContentView: View {
    @StateObject private var model = BenchViewModel()
    @State private var showingSettings = false
    @State private var selectedTab = 0

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Picker("Vue", selection: $selectedTab) {
                Label("Chat", systemImage: "bubble.left.and.bubble.right").tag(0)
                Label("Serveur", systemImage: "network").tag(1)
            }
            .pickerStyle(.segmented)
            .frame(width: 220)
            .padding(.vertical, 9)
            if selectedTab == 0 { chatLayout } else { ServerView(model: model) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .frame(minWidth: 980, minHeight: 680, alignment: .top)
        .preferredColorScheme(.dark)
        .onChange(of: model.modelPath) { _, _ in model.modelPathChanged() }
        .task {
            while !Task.isCancelled {
                await model.refreshServer()
                try? await Task.sleep(for: .milliseconds(400))
            }
        }
    }

    private var chatLayout: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                conversation
                Divider()
                composer
            }
            .frame(minWidth: 620, maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            MetricsView(
                metrics: model.metrics,
                acceptRate: model.acceptRate,
                mtpStatus: model.turnHistory.last?.mtpStatus,
                timeToFirstToken: model.timeToFirstToken,
                loadDuration: model.loadDuration,
                turnHistory: model.turnHistory,
                activeMemoryBytes: model.activeMemoryBytes,
                peakMemoryBytes: model.peakMemoryBytes
            )
            .frame(minWidth: 255, idealWidth: 275, maxWidth: 310)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(model.isLoaded ? Color.green : Color.orange)
                .frame(width: 9, height: 9)
            VStack(alignment: .leading, spacing: 2) {
                Text("Qwen3.8")
                    .font(.headline)
                Text(model.isLoaded
                    ? (model.turnHistory.isEmpty
                        ? "\(model.loadedFamilyLabel) · prêt"
                        : "\(model.loadedFamilyLabel) · session active · cache KV")
                    : "Modèle non chargé")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(model.status)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Button {
                model.resetConversation()
            } label: {
                Label("Nouvelle conversation", systemImage: "square.and.pencil")
            }
            .buttonStyle(.borderless)
            .disabled(model.isBusy || !model.isLoaded)
            Button {
                showingSettings.toggle()
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.title3)
            }
            .buttonStyle(.borderless)
            .help("Réglages du modèle et de la génération")
            .popover(isPresented: $showingSettings, arrowEdge: .top) {
                SettingsView(model: model)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 13)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if model.messages.isEmpty {
                    EmptyConversationView(isLoaded: model.isLoaded) {
                        model.loadModel()
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    LazyVStack(alignment: .leading, spacing: 18) {
                        ForEach(model.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 24)
                }
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: model.messages.count) { _, _ in
                if let lastID = model.messages.last?.id {
                    withAnimation(.easeOut(duration: 0.18)) {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var composer: some View {
        VStack(spacing: 8) {
            if let imageURL = model.imageURL {
                HStack(spacing: 6) {
                    Image(systemName: "photo")
                    Text(imageURL.lastPathComponent)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button {
                        model.clearImage()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(alignment: .bottom, spacing: 10) {
                Button {
                    model.chooseImage()
                } label: {
                    Image(systemName: "paperclip")
                        .font(.title3)
                }
                .buttonStyle(.borderless)
                .help("Joindre une image")
                .disabled(model.isBusy)

                TextField("Écris un message…", text: $model.prompt, axis: .vertical)
                    .font(.body)
                    .textFieldStyle(.plain)
                    .lineLimit(1 ... 4)
                    .frame(minHeight: 30, maxHeight: 82)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.10))
                }

                Button {
                    model.run()
                } label: {
                    Image(systemName: model.isBusy ? "hourglass" : "arrow.up")
                        .font(.headline)
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.borderedProminent)
                .clipShape(Circle())
                .help(model.isLoaded ? "Envoyer" : "Charge le modèle dans les réglages")
                .disabled(model.isBusy || !model.isLoaded || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text(model.isLoaded
                ? (model.turnHistory.isEmpty
                    ? "Utilise le bouton ↑ pour envoyer · la conversation restera disponible pour les tours suivants"
                    : "Tour suivant prêt · conversation conservée · réglages via le bouton curseurs")
                : "Charge un modèle pour commencer")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .padding(.horizontal, 22)
        .padding(.top, 12)
        .padding(.bottom, 14)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.72))
    }
}

private struct ServerView: View {
    @ObservedObject var model: BenchViewModel
    @State private var showingOpenCodeConfiguration = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Serveur d’inférence LAN")
                            .font(.title3.weight(.semibold))
                        Text("Un seul modèle résident · file FIFO · API compatible OpenAI")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Circle()
                        .fill(statusColor)
                        .frame(width: 10, height: 10)
                    Text(statusLabel)
                        .foregroundStyle(.secondary)
                }

                GroupBox("Accès") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Port")
                                .foregroundStyle(.secondary)
                            TextField("8848", text: $model.serverPort)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 90)
                            Text("Clé API")
                                .foregroundStyle(.secondary)
                            SecureField("optionnelle", text: $model.serverAPIKey)
                                .textFieldStyle(.roundedBorder)
                        }
                        Text(model.serverSnapshot.status == .running
                            ? "LAN : http://<adresse-du-Mac>:\(model.serverSnapshot.port)"
                            : "Le catalogue sera chargé à la demande sur le premier appel.")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                        HStack {
                            Button(model.serverSnapshot.status == .running ? "Arrêter le serveur" : "Démarrer le serveur") {
                                model.toggleServer()
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(model.serverSnapshot.status == .running ? .red : .accentColor)
                            .disabled(model.serverSnapshot.status == .starting || model.serverSnapshot.status == .stopping)
                            Spacer()
                            Text("Local : 127.0.0.1:\(model.serverSnapshot.port)")
                                .font(.caption.monospaced())
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.vertical, 4)
                }

                if !model.serverSnapshot.availableModels.isEmpty {
                    GroupBox("Catalogue local") {
                        VStack(alignment: .leading, spacing: 5) {
                            ForEach(model.serverSnapshot.availableModels, id: \.self) { name in
                                HStack(spacing: 6) {
                                    Circle()
                                        .fill(name == model.serverSnapshot.loadedModel ? Color.green : Color.secondary)
                                        .frame(width: 6, height: 6)
                                    Text(name)
                                        .font(.caption.monospaced())
                                    if name == model.serverSnapshot.loadedModel {
                                        Text("chargé")
                                            .font(.caption2)
                                            .foregroundStyle(.green)
                                    }
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                HStack(spacing: 12) {
                    serverStat("Actives", "\(model.serverSnapshot.activeSessions)")
                    serverStat("En attente", "\(model.serverSnapshot.queuedSessions)")
                    serverStat("Port", "\(model.serverSnapshot.port)")
                }

                GroupBox {
                    DisclosureGroup("Configuration OpenCode", isExpanded: $showingOpenCodeConfiguration) {
                        VStack(alignment: .leading, spacing: 8) {
                            TextEditor(text: .constant(model.openCodeConfiguration))
                                .font(.system(.caption, design: .monospaced))
                                .frame(minHeight: 150, maxHeight: 230)
                                .scrollContentBackground(.hidden)
                                .padding(5)
                                .background(Color.black.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
                                .textSelection(.enabled)
                            HStack {
                                Button("Copier pour OpenCode") {
                                    model.copyOpenCodeConfiguration()
                                }
                                .buttonStyle(.borderedProminent)
                                Spacer()
                                Text("Défaut : \(model.serverSnapshot.loadedModel ?? "4-bit")")
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        .padding(.top, 8)
                    }
                    .font(.headline)
                }

                if let lastError = model.serverSnapshot.lastError {
                    Label(lastError, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Spacer()
            }
            .padding(24)
            .frame(minWidth: 500, maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            Divider()
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Sessions")
                        .font(.headline)
                    Spacer()
                    Text("actualisation 400 ms")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                if model.serverSnapshot.sessions.isEmpty {
                    ContentUnavailableView("Aucune session", systemImage: "network.slash", description: Text("Les requêtes du réseau apparaîtront ici."))
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(model.serverSnapshot.sessions.reversed()) { session in
                                sessionRow(session)
                            }
                        }
                    }
                }
            }
            .padding(18)
            .frame(minWidth: 350, idealWidth: 390, maxWidth: 470, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var statusLabel: String {
        switch model.serverSnapshot.status {
        case .stopped: return "Arrêté"
        case .starting: return "Démarrage…"
        case .running: return "En écoute"
        case .stopping: return "Arrêt…"
        case .failed: return "Erreur"
        }
    }

    private var statusColor: Color {
        switch model.serverSnapshot.status {
        case .running: return .green
        case .failed: return .orange
        case .starting, .stopping: return .yellow
        case .stopped: return .secondary
        }
    }

    private func serverStat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.title2.monospacedDigit())
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
    }

    private func sessionRow(_ session: Qwen38ServerSession) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Circle()
                    .fill(sessionColor(session.status))
                    .frame(width: 7, height: 7)
                Text(session.status.rawValue.capitalized)
                    .font(.caption.weight(.semibold))
                Spacer()
                Text("\(session.model) · \(session.path)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
            HStack(spacing: 10) {
                        Text(session.model)
                        Text(session.inputDescription)
                Text("\(session.generatedTokens) tokens")
                if let ttft = session.timeToFirstToken {
                    Text(String(format: "TTFT %.0f ms", ttft * 1000))
                }
                if let speed = session.tokensPerSecond {
                    Text(String(format: "%.1f tok/s", speed))
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            if !session.lastToken.isEmpty {
                Text(session.lastToken)
                    .font(.caption.monospaced())
                    .foregroundStyle(.cyan)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(7)
                    .background(Color.cyan.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
            }
            if let error = session.error {
                Text(error).font(.caption2).foregroundStyle(.orange)
            }
        }
        .padding(10)
        .background(Color.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
    }

    private func sessionColor(_ status: Qwen38ServerSessionStatus) -> Color {
        switch status {
        case .queued: return .yellow
        case .running: return .cyan
        case .completed: return .green
        case .failed: return .orange
        }
    }
}

private struct EmptyConversationView: View {
    let isLoaded: Bool
    let load: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(.secondary)
            Text("Commencer une conversation")
                .font(.title3.weight(.semibold))
            Text(isLoaded ? "Écris ton premier message ci-dessous." : "Charge le modèle, puis écris ton premier message ci-dessous.")
                .foregroundStyle(.secondary)
            if !isLoaded {
                Button("Charger le modèle") { load() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: 420)
        .padding(30)
    }
}

private struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .assistant {
                Image(systemName: "sparkles")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.cyan)
                    .frame(width: 25, height: 25)
                    .background(Color.cyan.opacity(0.12), in: Circle())
                VStack(alignment: .leading, spacing: 6) {
                    if !message.reasoning.isEmpty {
                        reasoningBubble
                    }
                    bubble
                }
                Spacer(minLength: 40)
            } else {
                Spacer(minLength: 100)
                bubble
                    .foregroundStyle(.white)
            }
        }
    }

    @ViewBuilder
    private var reasoningBubble: some View {
        DisclosureGroup {
            Text(message.reasoning)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 3)
        } label: {
            Label("Thinking · \(message.reasoning.count) caractères", systemImage: "brain")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(Color.cyan.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
    }

    private var bubble: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let imageURL = message.imageURL, let image = NSImage(contentsOf: imageURL) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            if !message.text.isEmpty {
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if message.isStreaming {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 10)
        .background(
            message.role == .user ? Color.accentColor.opacity(0.85) : Color.white.opacity(0.07),
            in: RoundedRectangle(cornerRadius: 14)
        )
    }
}

private struct SettingsView: View {
    @ObservedObject var model: BenchViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Réglages")
                .font(.headline)

            VStack(alignment: .leading, spacing: 5) {
                Text("Modèle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("Répertoire du modèle", text: $model.modelPath)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.refreshCatalog() }
            }

            HStack {
                Text("Catalogue")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Rafraîchir") { model.refreshCatalog() }
                    .buttonStyle(.bordered)
                    .disabled(model.isBusy)
            }
            if model.catalog.isEmpty {
                Text("Aucun modèle Qwen3.8 valide trouvé à côté du chemin ci-dessus.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(model.catalog) { discovered in
                            Button {
                                model.selectDiscoveredModel(discovered)
                            } label: {
                                HStack {
                                    Text(discovered.id)
                                        .lineLimit(1)
                                    Spacer()
                                    Text(discovered.familyDescription)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                    Text(discovered.sizeDescription)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .monospacedDigit()
                                }
                            }
                            .buttonStyle(.plain)
                            .padding(.vertical, 3)
                            .padding(.horizontal, 6)
                            .background(
                                model.modelPath == discovered.url.path
                                    ? Color.accentColor.opacity(0.18) : Color.clear,
                                in: RoundedRectangle(cornerRadius: 6))
                        }
                    }
                }
                .frame(maxHeight: 150)
                .disabled(model.isBusy)
            }

            HStack {
                Button(model.isLoaded ? "Modèle chargé" : "Charger") {
                    model.loadModel()
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isBusy || model.isLoaded)
                Button("Décharger") { model.unloadModel() }
                    .buttonStyle(.bordered)
                    .disabled(model.isBusy || !model.isLoaded)
            }
            if let progress = model.flashLoadProgress {
                ProgressView(
                    value: Double(progress.visited), total: Double(max(progress.total, 1)))
                Text("Couches Flash-Next chargées : \(progress.visited)/\(progress.total)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Divider()

            Toggle("Thinking", isOn: $model.thinking)
            HStack {
                Text("Effort")
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Effort thinking", selection: $model.reasoningEffort) {
                    Text("Faible").tag("low")
                    Text("Moyen").tag("medium")
                    Text("Élevé").tag("xhigh")
                }
                .labelsHidden()
                .frame(width: 120)
            }
            HStack {
                Text("Tokens maximum")
                    .foregroundStyle(.secondary)
                Spacer()
                TextField("1024", value: $model.maxTokens, format: .number)
                    .frame(width: 70)
                    .multilineTextAlignment(.trailing)
                Stepper("", value: $model.maxTokens, in: 128...16_384, step: 128)
                    .labelsHidden()
            }

            Divider()
            Toggle("MTP spéculatif", isOn: $model.mtpEnabled)
                .disabled(model.mtpAvailability != .active)
            Picker("Moteur MTP", selection: $model.mtpEngine) {
                Text("M1 upstream").tag(Qwen38MTPEngine.upstream)
                Text("M2 local").tag(Qwen38MTPEngine.local)
            }
            .disabled(!model.mtpEnabled)
            HStack {
                Text("Tokens draftés")
                    .foregroundStyle(.secondary)
                Spacer()
                Stepper(value: $model.mtpDraftTokens, in: 1...8) {
                    Text("\(model.mtpDraftTokens)")
                        .monospacedDigit()
                        .frame(width: 24, alignment: .trailing)
                }
                .disabled(!model.mtpEnabled)
            }
            Text(mtpHelp)
                .font(.caption2)
                .foregroundStyle(.tertiary)

            Divider()
            HStack {
                Button("Nouvelle conversation") { model.resetConversation() }
                    .disabled(model.isBusy || !model.isLoaded)
                Spacer()
                Button("Exporter trace") { model.exportTrace() }
                    .disabled(model.metrics == nil)
            }
        }
        .padding(18)
        .frame(width: 390)
        .onAppear { model.refreshCatalog() }
    }

    private var mtpHelp: String {
        switch model.mtpAvailability {
        case .active:
            return model.mtpEngine == .local
                ? "M2 local : greedy, cache target + drafter persistant entre tours."
                : "M1 upstream : greedy uniquement, un token drafté par round."
        case .fallback(let reason):
            return reason
        case .unavailable:
            return "Drafter MTP non présent pour cette variante."
        }
    }
}

struct MetricsView: View {
    let metrics: LLMMetrics?
    let acceptRate: Double?
    let mtpStatus: Qwen38MTPRunStatus?
    let timeToFirstToken: TimeInterval?
    let loadDuration: TimeInterval?
    let turnHistory: [Qwen38RunMetrics]
    let activeMemoryBytes: Int
    let peakMemoryBytes: Int

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Mesures").font(.headline)
                    Spacer()
                    Image(systemName: "gauge.with.dots.needle.33percent")
                        .foregroundStyle(.secondary)
                }
            if let loadDuration {
                metric("Chargement modèle", String(format: "%.2f s", loadDuration))
            }
            if let metrics {
                if let latest = turnHistory.last {
                    metric("Entrée", latest.inputDescription)
                }
                if let timeToFirstToken {
                    metric("TTFT réel", String(format: "%.0f ms", timeToFirstToken * 1000))
                }
                metric("Cache KV", runCacheLabel)
                if !turnHistory.isEmpty {
                    Divider().padding(.vertical, 4)
                    Text("Tours (\(turnHistory.count))").font(.headline)
                    ForEach(turnHistory, id: \.turnIndex) { turn in
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text("Tour \(turn.turnIndex)")
                                Spacer()
                                Text(turn.timeToFirstToken.map {
                                    String(format: "%.0f ms", $0 * 1000)
                                } ?? "—")
                                    .monospaced()
                            }
                            Text("\(turn.inputDescription) · \(turn.metrics.generatedTokens) tokens · \(String(format: "%.1f", turn.metrics.generationTokensPerSecond)) tok/s")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                            Text(mtpSummary(turn.mtpStatus))
                                .font(.caption2.monospaced())
                                .foregroundStyle(turn.mtpStatus.availability.isActive ? .green : .orange)
                        }
                        .foregroundStyle(.secondary)
                    }
                }
                metric("Prefill modèle", String(format: "%.0f ms", metrics.prefillTime * 1000))
                metric("Prefill", String(format: "%.1f tok/s", metrics.prefillTokensPerSecond))
                metric("Decode", String(format: "%.1f tok/s", metrics.generationTokensPerSecond))
                metric("Prompt", "\(metrics.promptTokens) tokens")
                metric("Générés", "\(metrics.generatedTokens) tokens")
                metric("Total", String(format: "%.2f s", metrics.totalTime))
                if let acceptRate {
                    metric("Accept rate MTP", String(format: "%.1f%%", acceptRate * 100))
                }
                if let mtpStatus {
                    switch mtpStatus.availability {
                    case .active:
                        metric("MTP", "actif · bloc \(mtpStatus.blockSize ?? 0)")
                    case .fallback(let reason):
                        metric("MTP", "fallback")
                        Text(reason)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    case .unavailable:
                        metric("MTP", "indisponible")
                    }
                    if mtpStatus.proposedTokens > 0 {
                        metric("MTP proposés", "\(mtpStatus.proposedTokens)")
                        metric("MTP acceptés", "\(mtpStatus.acceptedTokens)")
                        metric("Rounds MTP", "\(mtpStatus.rounds)")
                    }
                }
                if peakMemoryBytes > 0 {
                    metric("Mémoire active", ByteCountFormatter.string(
                        fromByteCount: Int64(activeMemoryBytes), countStyle: .memory
                    ))
                    metric("Mémoire peak", ByteCountFormatter.string(
                        fromByteCount: Int64(peakMemoryBytes), countStyle: .memory
                    ))
                }
            } else {
                Text("Aucune génération").foregroundStyle(.secondary)
            }
                Spacer(minLength: 0)
            }
            .padding()
        }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        HStack { Text(title).foregroundStyle(.secondary); Spacer(); Text(value).monospaced() }
    }

    private var runCacheLabel: String {
        guard let latest = turnHistory.last else { return "—" }
        return latest.cacheReused ? "réutilisé" : "initialisé"
    }

    private func mtpSummary(_ status: Qwen38MTPRunStatus) -> String {
        let engine = status.engine == .local ? "M2 local" : "M1 upstream"
        switch status.availability {
        case .active:
            if status.proposedTokens > 0 {
                return "\(engine) actif · \(status.acceptedTokens)/\(status.proposedTokens) acceptés · \(status.rounds) rounds"
            }
            return "\(engine) actif"
        case .fallback:
            return "\(engine) fallback · replay"
        case .unavailable:
            return "MTP indisponible"
        }
    }
}

@main
struct Qwen38BenchUIApp: App {
    @NSApplicationDelegateAdaptor(Qwen38AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("Qwen3.8 Bench") { ContentView() }
    }
}
