import Foundation

/// Compteurs affichés dans le panneau — « le compteur qui compte vraiment »
/// (nombre d'appels, dont combien bien formés), la métrique qui a servi à
/// juger tout le chantier §P13 (voir `docs/knowledge/log.md`, P13.2/P13.3).
public struct AgentStats: Equatable, Sendable {
    public var steps = 0
    /// Tours sans aucun appel d'outil (le modèle a répondu en texte seul).
    public var textOnlyTurns = 0
    /// Nombre total d'appels d'outil émis, tous outils confondus.
    public var calls = 0
    /// Appel « bien formé » : les arguments sont un JSON valide qui décode
    /// en objet. Un appel dont les arguments ne parsent pas est compté ici
    /// comme malformé — cas qui n'a jamais été observé sur le checkpoint
    /// réel (voir P13.2 : « 29 appels sur 29 bien formés »), mais que le
    /// panneau doit pouvoir signaler s'il se produisait.
    public var wellFormed = 0
    /// Appel « valide » : bien formé, outil connu, et tous les paramètres
    /// requis par son schéma sont présents.
    public var valid = 0

    public init() {}
}

/// Un appel d'outil exécuté à un pas donné — ce que le journal pas à pas
/// affiche (outil, arguments, extrait du résultat, repliable).
public struct AgentToolCallExecutionRecord: Equatable, Sendable {
    public var name: String
    public var argumentsJSON: String
    public var wellFormed: Bool
    public var valid: Bool
    /// Résultat déjà tronqué (voir `AgentTruncation`) — vide pour
    /// `final_answer`, qui n'a pas de résultat d'outil, seulement la réponse
    /// elle-même portée par `AgentStepOutcome.finalAnswer`.
    public var resultPreview: String
    public var isFinalAnswer: Bool

    public init(
        name: String, argumentsJSON: String, wellFormed: Bool, valid: Bool, resultPreview: String,
        isFinalAnswer: Bool
    ) {
        self.name = name
        self.argumentsJSON = argumentsJSON
        self.wellFormed = wellFormed
        self.valid = valid
        self.resultPreview = resultPreview
        self.isFinalAnswer = isFinalAnswer
    }
}

/// Une entrée du journal pas à pas — un pas complet (un aller-retour avec le
/// serveur), qu'il ait appelé un ou plusieurs outils ou juste répondu en
/// texte.
public struct AgentStepRecord: Equatable, Sendable, Identifiable {
    public var stepIndex: Int
    public var reasoning: String
    public var toolCalls: [AgentToolCallExecutionRecord]
    /// Non-nil quand le tour n'a appelé aucun outil : le contenu texte brut
    /// du tour, pour affichage — ce cas ne fait pas avancer la tâche (voir
    /// le journal §P13.4 : « la réflexion doit être active, sinon le modèle
    /// enchaîne des tours pertinents mais ne conclut jamais »).
    public var textOnly: String?

    public var id: Int { stepIndex }

    public init(stepIndex: Int, reasoning: String, toolCalls: [AgentToolCallExecutionRecord], textOnly: String?) {
        self.stepIndex = stepIndex
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.textOnly = textOnly
    }
}

/// Le résultat d'un pas : soit la boucle continue, soit `final_answer` a été
/// appelé et elle s'arrête là.
public enum AgentStepOutcome: Equatable, Sendable {
    case continuing
    case finalAnswer(String)
}

/// L'issue d'une exécution complète, budget de pas compris.
public enum AgentRunOutcome: Equatable, Sendable {
    case finalAnswer(String)
    case budgetExhausted
}

/// Décide si la boucle doit s'arrêter pour épuisement du budget de pas —
/// fonction pure, séparée de `AgentLoopEngine.apply` pour rester testable
/// indépendamment (« le cas du budget épuisé », demandé explicitement).
/// `nil` veut dire : ni réponse finale, ni budget épuisé, la boucle continue.
public enum AgentBudget {
    public static func evaluate(outcome: AgentStepOutcome, stepsUsed: Int, maxSteps: Int) -> AgentRunOutcome? {
        switch outcome {
        case .finalAnswer(let answer):
            return .finalAnswer(answer)
        case .continuing:
            return stepsUsed >= maxSteps ? .budgetExhausted : nil
        }
    }
}

/// La machine à états de la boucle d'agent : accumule l'historique de
/// conversation, exécute les outils demandés, et compte ce qui compte.
/// Porté depuis la boucle `for step in range(a.max_steps): …` de
/// `Scripts/agent-loop.py`, en enlevant tout ce qui est réseau (ça, c'est le
/// travail du client HTTP côté GUI — `apply` ne fait qu'appliquer un tour
/// déjà reçu).
///
/// Type valeur : chaque appel à `apply` avance l'état par mutation, sans
/// jamais toucher au réseau ni au disque au-delà de ce que l'exécuteur
/// d'outils fait en son sein — ce qui le rend directement testable en
/// séquence, sans checkpoint ni serveur.
public struct AgentLoopEngine: Sendable {
    public static let systemPrompt = """
        Tu es un assistant qui explore un dossier avec les outils fournis.

        Méthode :
        1. Cherche avec grep un terme précis (un nom de fichier, un mot-clé), \
        pas une expression du langage courant.
        2. Dès qu'une sortie d'outil contient la réponse, appelle final_answer \
        immédiatement. N'explore pas plus loin par précaution.
        3. Si trois recherches de suite ne donnent rien, change complètement d'angle.

        Tu DOIS terminer par un appel à final_answer. Une exploration sans réponse est un échec.
        """

    public let sandbox: AgentSandbox
    public let executor: AgentToolExecutor
    public private(set) var messages: [AgentMessage]
    public private(set) var stats = AgentStats()

    public init(rootURL: URL, task: String) {
        self.sandbox = AgentSandbox(root: rootURL)
        self.executor = AgentToolExecutor(sandbox: sandbox)
        self.messages = [
            AgentMessage(role: .system, content: Self.systemPrompt),
            AgentMessage(role: .user, content: task),
        ]
    }

    /// Constructeur pour les tests : un exécuteur déjà construit (permet de
    /// pointer vers un dossier de fixtures sans dupliquer `AgentSandbox`).
    public init(executor: AgentToolExecutor, task: String) {
        self.sandbox = executor.sandbox
        self.executor = executor
        self.messages = [
            AgentMessage(role: .system, content: Self.systemPrompt),
            AgentMessage(role: .user, content: task),
        ]
    }

    /// Applique un tour reçu du serveur : exécute chaque outil demandé dans
    /// l'ordre, alimente l'historique, et arrête au premier `final_answer`
    /// rencontré (les appels suivants dans le même tour, s'il y en avait,
    /// ne sont pas exécutés — même comportement que le `return` immédiat de
    /// `Scripts/agent-loop.py`).
    @discardableResult
    public mutating func apply(_ turn: AgentModelTurn) -> (record: AgentStepRecord, outcome: AgentStepOutcome) {
        stats.steps += 1

        guard !turn.toolCalls.isEmpty else {
            stats.textOnlyTurns += 1
            messages.append(AgentMessage(role: .assistant, content: turn.content))
            let record = AgentStepRecord(
                stepIndex: stats.steps, reasoning: turn.reasoning, toolCalls: [], textOnly: turn.content)
            return (record, .continuing)
        }

        messages.append(AgentMessage(role: .assistant, content: turn.content, toolCalls: turn.toolCalls))

        var executions: [AgentToolCallExecutionRecord] = []
        for call in turn.toolCalls {
            stats.calls += 1
            let argumentKeys = Self.decodeArgumentKeys(call.argumentsJSON)
            let wellFormed = argumentKeys != nil
            if wellFormed { stats.wellFormed += 1 }
            let requiredKeys = AgentToolCatalog.requiredArguments[call.name]
            let valid = wellFormed && requiredKeys != nil
                && requiredKeys!.allSatisfy { (argumentKeys ?? []).contains($0) }
            if valid { stats.valid += 1 }

            if call.name == "final_answer" {
                let answer = Self.decodeAnswer(call.argumentsJSON) ?? ""
                executions.append(
                    AgentToolCallExecutionRecord(
                        name: call.name, argumentsJSON: call.argumentsJSON, wellFormed: wellFormed,
                        valid: valid, resultPreview: "", isFinalAnswer: true))
                let record = AgentStepRecord(
                    stepIndex: stats.steps, reasoning: turn.reasoning, toolCalls: executions, textOnly: nil)
                return (record, .finalAnswer(answer))
            }

            let rawResult = executor.run(name: call.name, argumentsJSON: call.argumentsJSON)
            let truncated = AgentTruncation.truncate(rawResult, limit: AgentTruncation.messageCharLimit)
            executions.append(
                AgentToolCallExecutionRecord(
                    name: call.name, argumentsJSON: call.argumentsJSON, wellFormed: wellFormed, valid: valid,
                    resultPreview: truncated, isFinalAnswer: false))
            messages.append(AgentMessage(role: .tool, content: truncated, toolCallID: call.id))
        }

        let record = AgentStepRecord(
            stepIndex: stats.steps, reasoning: turn.reasoning, toolCalls: executions, textOnly: nil)
        return (record, .continuing)
    }

    private static func decodeArgumentKeys(_ json: String) -> Set<String>? {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return Set(object.keys)
    }

    private static func decodeAnswer(_ json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object["answer"] as? String
    }
}
