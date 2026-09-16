// Tests du panneau Agent (Qwen38Agent) : uniquement ce qui est testable
// sans checkpoint ni réseau — la garde de chemin sous la racine (y compris
// les tentatives d'évasion), la troncature des sorties d'outil, et la
// machine à états de la boucle (appel → résultat → appel → réponse finale,
// et le budget épuisé). Voir le rapport de portage pour le détail des choix.
import Foundation
import Testing
@testable import Qwen38Agent

// MARK: - AgentSandbox : résolution de chemin sous la racine

@Test("La racine elle-même et ses sous-dossiers résolvent sous le dossier choisi")
func sandboxResolvesInsideRoot() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let sandbox = AgentSandbox(root: root)

    #expect(try sandbox.resolve(nil).path == root.standardizedFileURL.path)
    #expect(try sandbox.resolve(".").path == root.standardizedFileURL.path)
    #expect(try sandbox.resolve("Sources").path == root.appendingPathComponent("Sources").standardizedFileURL.path)
    #expect(
        try sandbox.resolve("Sources/Qwen38Agent/AgentSandbox.swift").path
            == root.appendingPathComponent("Sources/Qwen38Agent/AgentSandbox.swift").standardizedFileURL.path)
}

@Test("Une évasion par .. hors de la racine est refusée")
func sandboxRejectsDotDotEscape() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let sandbox = AgentSandbox(root: root)

    #expect(throws: AgentPathError.self) { try sandbox.resolve("..") }
    #expect(throws: AgentPathError.self) { try sandbox.resolve("../etc/passwd") }
    #expect(throws: AgentPathError.self) { try sandbox.resolve("sous-dossier/../../etc") }
}

@Test("Un chemin absolu hors racine est refusé")
func sandboxRejectsAbsoluteEscape() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let sandbox = AgentSandbox(root: root)

    #expect(throws: AgentPathError.self) { try sandbox.resolve("/etc/passwd") }
}

@Test("Un dossier voisin qui partage le préfixe de la racine n'est pas confondu avec elle")
func sandboxRejectsSiblingPrefixCollision() throws {
    // Le piège classique d'une garde par préfixe naïve : "/tmp/projet-evil"
    // commence bien par "/tmp/projet" en tant que chaîne, mais n'est pas
    // sous "/tmp/projet/". La frontière de "/" dans AgentSandbox.resolve
    // doit l'empêcher.
    let root = try makeTempDirectory(named: "qwen38-agent-sandbox-projet")
    defer { try? FileManager.default.removeItem(at: root) }
    let evilSibling = root.deletingLastPathComponent()
        .appendingPathComponent(root.lastPathComponent + "-evil", isDirectory: true)
    let sandbox = AgentSandbox(root: root)

    #expect(throws: AgentPathError.self) { try sandbox.resolve(evilSibling.path) }
}

@Test("Le message d'erreur d'évasion cite le chemin demandé")
func sandboxErrorMessageCitesRequestedPath() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let sandbox = AgentSandbox(root: root)

    do {
        _ = try sandbox.resolve("../secret")
        Issue.record("la résolution aurait dû lever AgentPathError")
    } catch let error as AgentPathError {
        #expect(error.requested == "../secret")
        #expect(error.errorDescription?.contains("../secret") == true)
    }
}

// MARK: - AgentTruncation : troncature des sorties

@Test("Un texte plus court que la limite n'est pas modifié")
func truncationLeavesShortTextUnchanged() {
    let text = "bonjour"
    #expect(AgentTruncation.truncate(text, limit: 100) == text)
}

@Test("Un texte plus long que la limite est coupé exactement à la limite")
func truncationCutsAtExactLimit() {
    let text = String(repeating: "a", count: 500)
    let truncated = AgentTruncation.truncate(text, limit: 200)
    #expect(truncated.count == 200)
    #expect(truncated == String(repeating: "a", count: 200))
}

@Test("Un texte de longueur exactement égale à la limite n'est pas coupé")
func truncationLeavesExactLengthTextUnchanged() {
    let text = String(repeating: "b", count: 4000)
    #expect(AgentTruncation.truncate(text, limit: 4000) == text)
}

@Test("Les constantes de troncature reprennent celles de Scripts/agent-loop.py")
func truncationConstantsMatchPythonReference() {
    #expect(AgentTruncation.toolOutputCharLimit == 6000)
    #expect(AgentTruncation.messageCharLimit == 4000)
    #expect(AgentTruncation.readFileLineLimit == 200)
    #expect(AgentTruncation.grepLineLimit == 60)
    #expect(AgentTruncation.listFilesEntryLimit == 80)
}

// MARK: - AgentToolExecutor : outils de lecture seule sous la racine

@Test("list_files trie les entrées et signale un dossier vide")
func toolExecutorListFiles() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try "a".write(to: root.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
    try "a".write(to: root.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
    let empty = root.appendingPathComponent("vide", isDirectory: true)
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)

    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    #expect(executor.run(name: "list_files", argumentsJSON: #"{"path":"."}"#) == "a.txt\nb.txt\nvide")
    #expect(executor.run(name: "list_files", argumentsJSON: #"{"path":"vide"}"#) == "(vide)")
}

@Test("list_files hors racine renvoie une erreur, jamais le contenu")
func toolExecutorListFilesRejectsEscape() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    let result = executor.run(name: "list_files", argumentsJSON: #"{"path":"../.."}"#)
    #expect(result.hasPrefix("ERREUR"))
}

@Test("read_file numérote les lignes à partir de start_line et plafonne à 200 lignes")
func toolExecutorReadFile() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let lines = (1...250).map { "ligne \($0)" }
    try lines.joined(separator: "\n").write(
        to: root.appendingPathComponent("fichier.txt"), atomically: true, encoding: .utf8)

    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    let fromStart = executor.run(name: "read_file", argumentsJSON: #"{"path":"fichier.txt"}"#)
    let fromStartLines = fromStart.split(separator: "\n")
    #expect(fromStartLines.count == 200)
    #expect(fromStartLines.first == "1\tligne 1")
    #expect(fromStartLines.last == "200\tligne 200")

    let fromOffset = executor.run(name: "read_file", argumentsJSON: #"{"path":"fichier.txt","start_line":240}"#)
    let fromOffsetLines = fromOffset.split(separator: "\n")
    #expect(fromOffsetLines.first == "240\tligne 240")
    #expect(fromOffsetLines.last == "250\tligne 250")
}

@Test("read_file sans le paramètre requis renvoie une erreur explicite")
func toolExecutorReadFileMissingPath() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    let result = executor.run(name: "read_file", argumentsJSON: "{}")
    #expect(result.hasPrefix("ERREUR"))
}

@Test("grep ne cherche que dans les fichiers .swift/.md et respecte la racine")
func toolExecutorGrep() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try "let cibleUnique = 1\nautre ligne".write(
        to: root.appendingPathComponent("Fichier.swift"), atomically: true, encoding: .utf8)
    try "cibleUnique dans un fichier ignoré".write(
        to: root.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)

    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    let result = executor.run(name: "grep", argumentsJSON: #"{"pattern":"cibleUnique"}"#)
    #expect(result.contains("Fichier.swift:1:let cibleUnique = 1"))
    #expect(!result.contains("notes.txt"))
}

// MARK: - AgentLoopEngine : machine à états de la boucle

@Test("Un appel d'outil bien formé alimente l'historique et la boucle continue")
func loopEngineToolCallThenContinues() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try "contenu".write(to: root.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)

    var engine = AgentLoopEngine(rootURL: root, task: "trouve la réponse")
    let turn = AgentModelTurn(
        content: "", reasoning: "je cherche",
        toolCalls: [AgentToolCallRecord(id: "call_1", name: "list_files", argumentsJSON: #"{"path":"."}"#)],
        finishReason: "tool_calls")

    let (record, outcome) = engine.apply(turn)

    #expect(outcome == .continuing)
    #expect(record.toolCalls.count == 1)
    #expect(record.toolCalls[0].wellFormed)
    #expect(record.toolCalls[0].valid)
    #expect(record.toolCalls[0].resultPreview.contains("f.txt"))
    #expect(engine.stats.steps == 1)
    #expect(engine.stats.calls == 1)
    #expect(engine.stats.wellFormed == 1)
    #expect(engine.stats.valid == 1)
    // system + user + assistant(tool_calls) + tool(résultat)
    #expect(engine.messages.count == 4)
    #expect(engine.messages.last?.role == .tool)
    #expect(engine.messages.last?.toolCallID == "call_1")
}

@Test("La séquence appel → résultat → appel → réponse finale aboutit avec les bons compteurs")
func loopEngineFullSequenceReachesFinalAnswer() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try "let cible = 42".write(to: root.appendingPathComponent("f.swift"), atomically: true, encoding: .utf8)

    var engine = AgentLoopEngine(rootURL: root, task: "quelle est la valeur de cible ?")

    let step1 = AgentModelTurn(
        toolCalls: [AgentToolCallRecord(id: "1", name: "grep", argumentsJSON: #"{"pattern":"cible"}"#)])
    let (_, outcome1) = engine.apply(step1)
    #expect(outcome1 == .continuing)

    let step2 = AgentModelTurn(
        toolCalls: [
            AgentToolCallRecord(id: "2", name: "final_answer", argumentsJSON: #"{"answer":"42"}"#)
        ])
    let (record2, outcome2) = engine.apply(step2)

    #expect(outcome2 == .finalAnswer("42"))
    #expect(record2.toolCalls.first?.isFinalAnswer == true)
    #expect(engine.stats.steps == 2)
    #expect(engine.stats.calls == 2)
    #expect(engine.stats.wellFormed == 2)
    #expect(engine.stats.valid == 2)
    // final_answer ne produit jamais de message role: tool (court-circuite,
    // comme le `return` immédiat de Scripts/agent-loop.py).
    #expect(engine.messages.last?.role == .assistant)
}

@Test("final_answer dans un tour à plusieurs appels court-circuite les appels suivants")
func loopEngineFinalAnswerShortCircuitsRemainingCalls() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    var engine = AgentLoopEngine(rootURL: root, task: "tâche")
    let turn = AgentModelTurn(toolCalls: [
        AgentToolCallRecord(id: "1", name: "list_files", argumentsJSON: #"{"path":"."}"#),
        AgentToolCallRecord(id: "2", name: "final_answer", argumentsJSON: #"{"answer":"fini"}"#),
        AgentToolCallRecord(id: "3", name: "list_files", argumentsJSON: #"{"path":"."}"#),
    ])
    let (record, outcome) = engine.apply(turn)

    #expect(outcome == .finalAnswer("fini"))
    #expect(record.toolCalls.count == 2)
    #expect(engine.stats.calls == 2)
}

@Test("Un appel aux arguments malformés est compté à part, et l'outil renvoie une erreur récupérable")
func loopEngineMalformedArguments() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    var engine = AgentLoopEngine(rootURL: root, task: "tâche")
    let turn = AgentModelTurn(
        toolCalls: [AgentToolCallRecord(id: "1", name: "read_file", argumentsJSON: "pas-du-json")])
    let (record, outcome) = engine.apply(turn)

    #expect(outcome == .continuing)
    #expect(record.toolCalls[0].wellFormed == false)
    #expect(record.toolCalls[0].valid == false)
    #expect(record.toolCalls[0].resultPreview.hasPrefix("ERREUR"))
    #expect(engine.stats.calls == 1)
    #expect(engine.stats.wellFormed == 0)
    #expect(engine.stats.valid == 0)
    // La conversation continue malgré tout : le modèle peut se corriger au
    // pas suivant, la boucle ne s'arrête jamais sur une erreur d'outil.
    #expect(engine.messages.last?.role == .tool)
}

@Test("Un tour sans appel d'outil est compté et n'arrête pas la boucle")
func loopEngineTextOnlyTurnContinues() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    var engine = AgentLoopEngine(rootURL: root, task: "tâche")
    let (record, outcome) = engine.apply(AgentModelTurn(content: "je réfléchis encore"))

    #expect(outcome == .continuing)
    #expect(record.textOnly == "je réfléchis encore")
    #expect(engine.stats.textOnlyTurns == 1)
    #expect(engine.stats.calls == 0)
}

@Test("Le budget de pas s'épuise sans réponse finale : la boucle doit s'arrêter et le dire")
func loopEngineBudgetExhausted() throws {
    let root = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    var engine = AgentLoopEngine(rootURL: root, task: "tâche sans réponse")
    let maxSteps = 3
    var lastResult: AgentRunOutcome?

    for _ in 0 ..< maxSteps {
        let turn = AgentModelTurn(
            toolCalls: [AgentToolCallRecord(id: "n", name: "list_files", argumentsJSON: #"{"path":"."}"#)])
        let (_, outcome) = engine.apply(turn)
        lastResult = AgentBudget.evaluate(outcome: outcome, stepsUsed: engine.stats.steps, maxSteps: maxSteps)
        if lastResult != nil { break }
    }

    #expect(engine.stats.steps == maxSteps)
    #expect(lastResult == .budgetExhausted)
}

@Test("Une réponse finale avant l'épuisement du budget l'emporte")
func loopEngineFinalAnswerBeforeBudgetExhausted() throws {
    let outcome = AgentStepOutcome.finalAnswer("réponse")
    #expect(AgentBudget.evaluate(outcome: outcome, stepsUsed: 1, maxSteps: 16) == .finalAnswer("réponse"))
}

@Test("Continuer avant le dernier pas ne déclenche ni fin ni épuisement")
func loopEngineContinuingBeforeLastStepIsNil() throws {
    #expect(AgentBudget.evaluate(outcome: .continuing, stepsUsed: 2, maxSteps: 16) == nil)
}

// MARK: - AgentRunGate : une seule boucle à la fois

@Test("Une deuxième tentative de démarrage pendant qu'une boucle tourne est refusée")
func runGateRejectsConcurrentStart() throws {
    var gate = AgentRunGate()

    #expect(gate.tryStart() == true)
    #expect(gate.isRunning == true)
    // La relance (double clic, ou tâche déjà en fond après le hissage de
    // l'état dans BenchViewModel) ne doit jamais laisser deux boucles vivre :
    // le deuxième appel ne doit pas remettre `isRunning` à `true` "à
    // nouveau" au sens où l'appelant recommencerait quoi que ce soit.
    #expect(gate.tryStart() == false)
    #expect(gate.isRunning == true)
}

@Test("Après finish(), une nouvelle boucle peut démarrer")
func runGateAllowsRestartAfterFinish() throws {
    var gate = AgentRunGate()

    #expect(gate.tryStart() == true)
    gate.finish()
    #expect(gate.isRunning == false)
    #expect(gate.tryStart() == true)
}

@Test("finish() sans démarrage préalable ne fait rien (idempotent)")
func runGateFinishWithoutStartIsHarmless() throws {
    var gate = AgentRunGate()
    gate.finish()
    #expect(gate.isRunning == false)
}

// MARK: - AgentWireFormat : construction/lecture JSON (sans réseau)

@Test("buildRequestBody produit un corps JSON conforme au contrat serveur")
func wireFormatBuildRequestBody() throws {
    let messages: [AgentMessage] = [
        AgentMessage(role: .system, content: "consignes"),
        AgentMessage(role: .user, content: "question"),
        AgentMessage(
            role: .assistant, content: "",
            toolCalls: [AgentToolCallRecord(id: "1", name: "grep", argumentsJSON: #"{"pattern":"x"}"#)]),
        AgentMessage(role: .tool, content: "résultat", toolCallID: "1"),
    ]
    let data = AgentWireFormat.buildRequestBody(
        model: "Qwen3.8-Flash-Next", messages: messages, maxTokens: 900, enableThinking: true)
    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(object["model"] as? String == "Qwen3.8-Flash-Next")
    #expect(object["max_tokens"] as? Int == 900)
    #expect(object["enable_thinking"] as? Bool == true)
    #expect(object["mtp"] as? Bool == false)
    let renderedMessages = try #require(object["messages"] as? [[String: Any]])
    #expect(renderedMessages.count == 4)
    #expect(renderedMessages[3]["tool_call_id"] as? String == "1")
    let toolCalls = try #require(renderedMessages[2]["tool_calls"] as? [[String: Any]])
    let function = try #require(toolCalls.first?["function"] as? [String: Any])
    #expect(function["name"] as? String == "grep")
    #expect(function["arguments"] as? String == #"{"pattern":"x"}"#)
    let tools = try #require(object["tools"] as? [[String: Any]])
    #expect(tools.count == 4)
}

@Test("parseChatCompletionResponse lit les tool_calls et le finish_reason")
func wireFormatParseToolCallResponse() throws {
    let json = """
        {"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"m",
         "choices":[{"index":0,"finish_reason":"tool_calls","message":{
            "role":"assistant","content":"","reasoning_content":"je cherche",
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"grep","arguments":"{\\"pattern\\":\\"x\\"}"}}]
         }}]}
        """
    let turn = try AgentWireFormat.parseChatCompletionResponse(Data(json.utf8))
    #expect(turn.finishReason == "tool_calls")
    #expect(turn.reasoning == "je cherche")
    #expect(turn.toolCalls.count == 1)
    #expect(turn.toolCalls[0].name == "grep")
    #expect(turn.toolCalls[0].argumentsJSON == #"{"pattern":"x"}"#)
}

@Test("parseChatCompletionResponse lit une réponse texte sans outil")
func wireFormatParsePlainTextResponse() throws {
    let json = """
        {"id":"chatcmpl-2","object":"chat.completion","created":0,"model":"m",
         "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"voilà"}}]}
        """
    let turn = try AgentWireFormat.parseChatCompletionResponse(Data(json.utf8))
    #expect(turn.content == "voilà")
    #expect(turn.toolCalls.isEmpty)
    #expect(turn.finishReason == "stop")
}

@Test("parseChatCompletionResponse lève une erreur claire sur une forme inattendue")
func wireFormatParseRejectsUnexpectedShape() throws {
    #expect(throws: AgentWireError.self) {
        _ = try AgentWireFormat.parseChatCompletionResponse(Data(#"{"choices":[]}"#.utf8))
    }
}

@Test("parseHealthResponse lit le modèle chargé, ou renvoie nil s'il est absent")
func wireFormatParseHealthResponse() {
    let withModel = Data(#"{"status":"ok","model_loaded":true,"model":"Qwen3.8-Flash-Next"}"#.utf8)
    #expect(AgentWireFormat.parseHealthResponse(withModel) == "Qwen3.8-Flash-Next")

    let withoutModel = Data(#"{"status":"ok","model_loaded":false,"model":null}"#.utf8)
    #expect(AgentWireFormat.parseHealthResponse(withoutModel) == nil)
}

// MARK: - fixtures

private func makeTempDirectory(named name: String = "qwen38-agent-\(UUID().uuidString)") throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
