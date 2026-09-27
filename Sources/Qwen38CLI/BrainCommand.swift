import ArgumentParser
import CoreGraphics
import Darwin
import Foundation
import MLXProfiler
import Qwen38Brain
import Qwen38Core

/// P16 : le cerveau embarquable (`Qwen38Brain`) piloté depuis la CLI — une
/// réponse, une boucle d'agent enregistrée, un banc par taille de contexte et
/// un rejeu de conversation d'agent. Ce sont les mesures de
/// `docs/bonsai2-brain/plan.md`.
struct Brain: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Cerveau embarquable (Qwen38Brain) : réponse, agent, banc, rejeu",
        subcommands: [BrainAsk.self, BrainAgent.self, BrainBench.self, BrainReplay.self])
}

struct BrainCommonOptions: ParsableArguments {
    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Profil : fast ou lean")
    var profile: String = "fast"

    @Option(name: .long, help: "Tranche de préfill en jetons (moteur dense)")
    var prefillStep: Int?

    @Flag(name: .long, help: "Charger le modèle sans tour de vision")
    var textOnly = false

    @Option(name: .long, help: "Dossier où écrire la trace du profiler (trace.json, report.txt, phases.jsonl)")
    var trace: String?

    /// Opens a profiling session (GPU/CPU/memory sampling) before the model
    /// loads; `finishTrace` writes it out. The runtime's own request sessions
    /// (Flash-Next path) join the same session.
    func startTrace() -> ProfilingSession? {
        guard trace != nil else { return nil }
        let session = ProfilingSession(config: .fineGrained, subsystem: "com.qwen38mlx.brain")
        session.title = "qwen38 brain · \(URL(fileURLWithPath: modelPath).lastPathComponent) · \(profile)"
        MLXProfiler.shared.activeSession = session
        MLXProfiler.shared.enable()
        Qwen38Profiling.sharedSession = session
        return session
    }

    func finishTrace(_ session: ProfilingSession?) throws {
        guard let session, let trace else { return }
        session.finish()
        MLXProfiler.shared.disable()
        let directory = URL(fileURLWithPath: trace, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try ChromeTraceExporter.export(session: session).write(to: directory.appending(path: "trace.json"))
        try session.generateReport().write(
            to: directory.appending(path: "report.txt"), atomically: true, encoding: .utf8)
        var lines: [String] = []
        for phase in session.phaseSummaries() {
            var fields: [String: Any] = [
                "phase": phase.name, "ms": BrainMeasure.decimal(phase.durationMs, 0),
            ]
            if let gpu = phase.gpu {
                fields["gpu_mean"] = BrainMeasure.decimal(gpu.mean, 0)
                fields["gpu_p90"] = BrainMeasure.decimal(gpu.p90, 0)
            }
            if let cpu = phase.cpuPercent { fields["cpu_pct"] = BrainMeasure.decimal(cpu, 0) }
            if let peak = phase.peakMLXActiveMB { fields["peak_mlx_mb"] = BrainMeasure.decimal(peak, 0) }
            lines.append(BrainMeasure.jsonLine(fields))
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appending(path: "phases.jsonl"), atomically: true, encoding: .utf8)
        FileHandle.standardError.write(Data("trace : \(directory.path)\n".utf8))
    }

    /// Size of the weights read per decoded token for a dense model (all
    /// safetensors of the pack, symlinks followed), in GB.
    func weightGigabytes() -> Double {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        let bytes = files.filter { $0.pathExtension == "safetensors" }.reduce(0) { total, url in
            let resolved = url.resolvingSymlinksInPath()
            let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + size
        }
        return Double(bytes) / 1e9
    }

    func loadBrain() async throws -> Qwen38Brain {
        guard var profile = Qwen38BrainProfile.named(profile) else {
            throw ValidationError("profil inconnu : \(self.profile) (fast ou lean)")
        }
        if textOnly { profile = profile.textOnlyVariant() }
        let started = Date()
        let brain = try await Qwen38Brain.load(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true), profile: profile)
        if let prefillStep { await brain.setPrefillStepSize(prefillStep) }
        let report = await brain.memoryReport()
        FileHandle.standardError.write(Data(String(
            format: "brain · profil %@ · chargé en %.1f s · actif %d Mo · footprint %d Mo\n",
            profile.id, Date().timeIntervalSince(started), report.activeBytes / 1_048_576,
            BrainMeasure.physFootprintMB()).utf8))
        return brain
    }
}

enum BrainMeasure {
    /// `phys_footprint` — what Activity Monitor shows, cache included.
    static func physFootprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) / 1_048_576 : -1
    }

    static func jsonLine(_ object: [String: Any]) -> String {
        let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    /// A decimal JSON number with `digits` decimals (a rounded `Double`
    /// still prints as 22.100000000000001 through JSONSerialization).
    static func decimal(_ value: Double, _ digits: Int = 1) -> NSDecimalNumber {
        NSDecimalNumber(string: String(format: "%.\(digits)f", value))
    }

    static func usageFields(_ usage: Qwen38BrainUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "cached_prompt_tokens": usage.cachedPromptTokens,
            "completion_tokens": usage.completionTokens,
            "prefill_tok_s": decimal(usage.promptTokensPerSecond),
            "prefill_s": decimal(usage.prefillSeconds, 2),
            "decode_tok_s": decimal(usage.tokensPerSecond),
            "ttft_s": usage.timeToFirstToken.map { decimal($0, 2) } ?? NSNull(),
            "peak_mb": usage.peakMemoryBytes / 1_048_576,
            "step_p50_ms": usage.stepMedian.map { decimal($0 * 1000, 1) } ?? NSNull(),
            "step_p90_ms": usage.stepP90.map { decimal($0 * 1000, 1) } ?? NSNull(),
            "finish": usage.finishReason.rawValue,
        ]
    }
}

// MARK: - ask

struct BrainAsk: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ask", abstract: "Une réponse, en flux")

    @OptionGroup var common: BrainCommonOptions

    @Argument(help: "Question")
    var prompt: String

    @Flag(name: .long, help: "Activer la réflexion")
    var thinking = false

    @Option(name: .long, help: "Image locale à joindre à la question")
    var image: String?

    @Option(name: .long, help: "Redimensionner l'image en N×N (défaut : budget du checkpoint)")
    var imageResize: Int?

    @Option(name: .long, help: "Température, 0 = greedy")
    var temperature: Float = 0

    @Option(name: .long, help: "Jetons de sortie au plus")
    var maxTokens: Int = 512

    func run() async throws {
        let brain = try await common.loadBrain()
        let stream = await brain.respond(
            to: [.init(
                role: .user, content: prompt,
                imageURLs: image.map { [URL(fileURLWithPath: $0)] } ?? [])],
            options: .init(
                maxTokens: maxTokens, temperature: temperature, enableThinking: thinking,
                imageResize: imageResize.map { CGSize(width: $0, height: $0) }))
        for try await event in stream {
            switch event {
            case .reasoning(let text): FileHandle.standardError.write(Data(text.utf8))
            case .text(let text): print(text, terminator: ""); fflush(stdout)
            case .toolCall(let call): print("\n[appel] \(call.name) \(call.argumentsJSON)")
            case .done(let usage):
                print("")
                FileHandle.standardError.write(Data(
                    (BrainMeasure.jsonLine(BrainMeasure.usageFields(usage)) + "\n").utf8))
            }
        }
    }
}

// MARK: - agent (boucle outillée, lecture seule, enregistrée)

/// Outils en lecture seule sur un dossier : de quoi dérouler une vraie boucle
/// d'agent sans risque, et enregistrer une conversation pour `replay`.
enum BrainReadOnlyTools {
    static let specs: [Qwen38ToolSpec] = [
        tool("list_files", "Liste les fichiers du dépôt (chemins relatifs).", [:], required: []),
        tool(
            "read_file", "Lit un fichier texte du dépôt.",
            ["path": ["type": "string", "description": "Chemin relatif du fichier"]],
            required: ["path"]),
        tool(
            "search", "Cherche un texte dans les fichiers du dépôt ; renvoie fichier:ligne: texte.",
            ["query": ["type": "string", "description": "Texte à chercher"]],
            required: ["query"]),
    ]

    private static func tool(
        _ name: String, _ description: String, _ properties: [String: [String: String]],
        required: [String]
    ) -> Qwen38ToolSpec {
        var props: [String: Qwen38JSONValue] = [:]
        for (key, value) in properties {
            props[key] = .object(value.mapValues { .string($0) })
        }
        return Qwen38ToolSpec(
            name: name, description: description,
            parameters: .object([
                "type": .string("object"), "properties": .object(props),
                "required": .array(required.map { .string($0) }),
            ]))
    }

    static func run(_ call: Qwen38ToolCall, root: URL) -> String {
        let arguments =
            (try? JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8))) as? [String: Any]
            ?? [:]
        let files = listFiles(root)
        switch call.name {
        case "list_files":
            return files.joined(separator: "\n")
        case "read_file":
            guard let path = arguments["path"] as? String, files.contains(path),
                let text = try? String(contentsOf: root.appending(path: path), encoding: .utf8)
            else { return "erreur : fichier introuvable" }
            return String(text.prefix(6000))
        case "search":
            guard let query = arguments["query"] as? String, !query.isEmpty else {
                return "erreur : query vide"
            }
            var hits: [String] = []
            for path in files {
                guard let text = try? String(contentsOf: root.appending(path: path), encoding: .utf8)
                else { continue }
                for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                    .enumerated() where line.contains(query)
                {
                    hits.append("\(path):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
                    if hits.count >= 40 { return hits.joined(separator: "\n") }
                }
            }
            return hits.isEmpty ? "aucun résultat" : hits.joined(separator: "\n")
        default:
            return "erreur : outil inconnu \(call.name)"
        }
    }

    private static func listFiles(_ root: URL) -> [String] {
        let base = root.standardizedFileURL.path + "/"
        var result: [String] = []
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])
        while let url = enumerator?.nextObject() as? URL {
            let path = url.standardizedFileURL.path
            if path.contains("/.build/") { continue }
            if (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                result.append(String(path.dropFirst(base.count)))
            }
        }
        return result.sorted()
    }
}

/// Transcript on disk: OpenAI-shaped messages, enough to replay a
/// conversation through `Qwen38Brain.respond` turn by turn.
struct BrainTranscript: Codable {
    struct Message: Codable {
        struct ToolCall: Codable {
            var id: String
            var name: String
            var arguments: String
        }
        var role: String
        var content: String
        var toolCalls: [ToolCall]?
        /// Local image paths attached to this message.
        var images: [String]?
    }
    var tools: [String]
    var messages: [Message]

    func chatMessages(upTo end: Int) -> [Qwen38ChatMessage] {
        messages[..<end].map { message in
            Qwen38ChatMessage(
                role: Qwen38ChatMessage.Role(rawValue: message.role) ?? .user,
                content: message.content,
                imageURLs: (message.images ?? []).map { URL(fileURLWithPath: $0) },
                toolCalls: (message.toolCalls ?? []).map {
                    Qwen38ToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.arguments)
                })
        }
    }
}

struct BrainAgent: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "agent",
        abstract: "Boucle d'agent en lecture seule sur un dossier ; enregistre la conversation")

    @OptionGroup var common: BrainCommonOptions

    @Option(name: .long, help: "Dossier de travail (lecture seule)")
    var workspace: String

    @Option(name: .long, help: "Demande de l'utilisateur")
    var task: String

    @Option(name: .long, help: "Appels d'outils au plus")
    var maxSteps: Int = 10

    @Option(name: .long, help: "Fichier JSON où enregistrer la conversation")
    var record: String?

    @Option(name: .long, help: "Température, 0 = greedy")
    var temperature: Float = 0

    func run() async throws {
        let brain = try await common.loadBrain()
        let root = URL(fileURLWithPath: workspace, isDirectory: true)
        let system =
            "Tu es un agent de code. Tu travailles dans un dépôt en lecture seule avec les outils "
            + "list_files, read_file et search. Lis avant de répondre. Réponds en français, brièvement."
        var transcript = BrainTranscript(
            tools: BrainReadOnlyTools.specs.map(\.name),
            messages: [.init(role: "system", content: system), .init(role: "user", content: task)])
        for step in 1...(maxSteps + 1) {
            var text = ""
            var calls: [Qwen38ToolCall] = []
            let stream = await brain.respond(
                to: transcript.chatMessages(upTo: transcript.messages.count),
                tools: step <= maxSteps ? BrainReadOnlyTools.specs : [],
                options: .init(maxTokens: 1024, temperature: temperature))
            for try await event in stream {
                switch event {
                case .text(let chunk): text += chunk
                case .toolCall(let call): calls.append(call)
                case .reasoning: break
                case .done(let usage):
                    var fields = BrainMeasure.usageFields(usage)
                    fields["step"] = step
                    fields["tool_calls"] = calls.map(\.name)
                    print(BrainMeasure.jsonLine(fields))
                }
            }
            transcript.messages.append(.init(
                role: "assistant", content: text,
                toolCalls: calls.isEmpty
                    ? nil : calls.map { .init(id: $0.id, name: $0.name, arguments: $0.argumentsJSON) }))
            if calls.isEmpty {
                print("réponse : " + text.trimmingCharacters(in: .whitespacesAndNewlines))
                break
            }
            for call in calls {
                transcript.messages.append(.init(
                    role: "tool", content: BrainReadOnlyTools.run(call, root: root)))
            }
        }
        if let record {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(transcript).write(to: URL(fileURLWithPath: record))
            FileHandle.standardError.write(Data("conversation enregistrée : \(record)\n".utf8))
        }
    }
}

// MARK: - replay

struct BrainReplay: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "replay",
        abstract: "Rejoue une conversation enregistrée tour par tour : TTFT et jetons réutilisés")

    @OptionGroup var common: BrainCommonOptions

    @Option(name: .long, help: "Conversation enregistrée par `brain agent --record`")
    var transcript: String

    @Option(name: .long, help: "Jetons générés par tour (la suite vient de l'enregistrement)")
    var maxTokens: Int = 16

    @Option(name: .long, help: "Redimensionner les images en N×N (défaut : budget du checkpoint)")
    var imageResize: Int?

    func run() async throws {
        let recorded = try JSONDecoder().decode(
            BrainTranscript.self, from: Data(contentsOf: URL(fileURLWithPath: transcript)))
        let tools = BrainReadOnlyTools.specs.filter { recorded.tools.contains($0.name) }
        let session = common.startTrace()
        let brain = try await common.loadBrain()
        // Chaque tour assistant enregistré devient une requête : l'historique
        // qui le précède, comme le renverrait un client d'API.
        let turns = recorded.messages.indices.filter { recorded.messages[$0].role == "assistant" }
        var totalPrompt = 0
        var totalCached = 0
        for (index, end) in turns.enumerated() {
            let stream = await brain.respond(
                to: recorded.chatMessages(upTo: end), tools: tools,
                options: .init(
                    maxTokens: maxTokens, temperature: 0,
                    imageResize: imageResize.map { CGSize(width: $0, height: $0) }))
            var answer = ""
            for try await event in stream {
                switch event {
                case .text(let text): answer += text
                case .toolCall(let call): answer += "[\(call.name) \(call.argumentsJSON)]"
                case .reasoning: break
                case .done(let usage):
                    totalPrompt += usage.promptTokens + usage.cachedPromptTokens
                    totalCached += usage.cachedPromptTokens
                    var fields = BrainMeasure.usageFields(usage)
                    fields["turn"] = index + 1
                    fields["profile"] = common.profile
                    fields["answer"] = String(answer.prefix(120))
                    print(BrainMeasure.jsonLine(fields))
                }
            }
        }
        print(String(
            format: "replay · %d tours · %d jetons de prompt dont %d réutilisés (%.0f %%)",
            turns.count, totalPrompt, totalCached,
            totalPrompt > 0 ? Double(totalCached) * 100 / Double(totalPrompt) : 0))
        try common.finishTrace(session)
    }
}

// MARK: - bench

struct BrainBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Préfill, TTFT, décodage et mémoire par taille de contexte (une ligne JSON chacun)")

    @OptionGroup var common: BrainCommonOptions

    @Option(name: .long, help: "Fichiers de prompt, séparés par des virgules")
    var promptFiles: String

    @Option(name: .long, help: "Jetons générés")
    var maxTokens: Int = 128

    @Option(name: .long, help: "Pause entre deux mesures (s)")
    var cooldown: Int = 0

    func run() async throws {
        let session = common.startTrace()
        session?.beginPhase("Chargement", category: .modelLoad)
        let brain = try await common.loadBrain()
        session?.endPhase("Chargement", category: .modelLoad)
        let weightsGB = common.weightGigabytes()
        for (index, path) in promptFiles.split(separator: ",").map(String.init).enumerated() {
            if index > 0, cooldown > 0 { try await Task.sleep(for: .seconds(cooldown)) }
            await brain.resetConversation()
            let prompt = try String(contentsOfFile: path, encoding: .utf8)
            let stream = await brain.respond(
                to: [.init(role: .user, content: prompt)],
                options: .init(maxTokens: maxTokens, temperature: 0))
            for try await event in stream {
                if case .done(let usage) = event {
                    var fields = BrainMeasure.usageFields(usage)
                    fields["prompt_file"] = (path as NSString).lastPathComponent
                    fields["profile"] = common.profile
                    fields["prefill_step"] = await brain.prefillStepSize
                    fields["footprint_mb"] = BrainMeasure.physFootprintMB()
                    fields["date"] = ISO8601DateFormatter().string(from: Date())
                    fields["model"] = URL(fileURLWithPath: common.modelPath).lastPathComponent
                    fields["weights_gb"] = BrainMeasure.decimal(weightsGB, 2)
                    // Every weight is read once per decoded token on a dense
                    // model: this is the bandwidth the decode actually drew.
                    fields["weights_bw_gbps"] = BrainMeasure.decimal(weightsGB * usage.tokensPerSecond, 0)
                    print(BrainMeasure.jsonLine(fields))
                }
            }
        }
        try common.finishTrace(session)
    }
}
