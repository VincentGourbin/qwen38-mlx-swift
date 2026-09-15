import Foundation

/// Un message de la conversation outillée, tel que gardé par
/// `AgentLoopEngine` et rendu vers le format de requête du serveur. Miroir
/// Swift de ce que `Scripts/agent-loop.py` accumule dans `msgs`.
public struct AgentMessage: Equatable, Sendable {
    public enum Role: String, Sendable, Equatable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: String
    /// Présent sur un tour assistant qui a appelé un ou plusieurs outils.
    public var toolCalls: [AgentToolCallRecord]
    /// Présent sur un message `role: tool` — le serveur l'accepte pour
    /// compatibilité mais apparie en réalité par ordre (P13.1).
    public var toolCallID: String?

    public init(
        role: Role, content: String, toolCalls: [AgentToolCallRecord] = [], toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/// Un `tool_call` — émis par le serveur dans une réponse, ou renvoyé par le
/// client dans le tour assistant suivant. `argumentsJSON` est une chaîne
/// JSON, jamais un objet imbriqué (même convention que le fil OpenAI et que
/// `ChatCompletionRequestToolCallFunction` côté serveur).
public struct AgentToolCallRecord: Equatable, Sendable {
    public var id: String
    public var name: String
    public var argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// Le tour renvoyé par le serveur pour un pas de la boucle — extrait d'une
/// réponse `/v1/chat/completions` non diffusée (la même forme que
/// `d["choices"][0]["message"]` côté Python).
public struct AgentModelTurn: Equatable, Sendable {
    public var content: String
    public var reasoning: String
    public var toolCalls: [AgentToolCallRecord]
    public var finishReason: String?

    public init(
        content: String = "", reasoning: String = "", toolCalls: [AgentToolCallRecord] = [],
        finishReason: String? = nil
    ) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.finishReason = finishReason
    }
}

public struct AgentWireError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// Construction de la requête et lecture de la réponse — les deux fonctions
/// pures et testables sans réseau (elles ne font qu'assembler/lire du JSON).
/// L'appel HTTP lui-même vit côté GUI (`Qwen38BenchUI`), pas ici.
public enum AgentWireFormat {
    /// Corps JSON de la requête, format OpenAI + les champs propres au
    /// serveur (`enable_thinking`, `mtp: false` — la boucle d'agent n'a
    /// aucun usage du drafter spéculatif).
    public static func buildRequestBody(
        model: String, messages: [AgentMessage], maxTokens: Int, enableThinking: Bool
    ) -> Data {
        let messagesJSON: [[String: Any]] = messages.map { message in
            var dict: [String: Any] = ["role": message.role.rawValue, "content": message.content]
            if !message.toolCalls.isEmpty {
                dict["tool_calls"] = message.toolCalls.map { call in
                    [
                        "id": call.id, "type": "function",
                        "function": ["name": call.name, "arguments": call.argumentsJSON],
                    ] as [String: Any]
                }
            }
            if let toolCallID = message.toolCallID {
                dict["tool_call_id"] = toolCallID
            }
            return dict
        }
        let body: [String: Any] = [
            "model": model,
            "messages": messagesJSON,
            "tools": AgentToolCatalog.schema(),
            "temperature": 0,
            "max_tokens": maxTokens,
            "enable_thinking": enableThinking,
            "mtp": false,
        ]
        // `JSONSerialization` sur un dictionnaire construit localement ne
        // peut échouer que si un type non sérialisable s'y est glissé — ce
        // qui n'arrive jamais ici (Strings/Ints/Bools/tableaux/dicos).
        return (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
    }

    /// Lit une réponse `/v1/chat/completions` non diffusée et en extrait le
    /// premier choix. Lève `AgentWireError` sur une forme inattendue — le
    /// corps d'un 4xx/5xx serveur n'a pas cette forme et doit être traité en
    /// amont, avant l'appel à cette fonction.
    public static func parseChatCompletionResponse(_ data: Data) throws -> AgentModelTurn {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AgentWireError("réponse JSON illisible")
        }
        guard let choices = object["choices"] as? [[String: Any]], let first = choices.first else {
            throw AgentWireError("aucun choix dans la réponse")
        }
        let message = (first["message"] as? [String: Any]) ?? [:]
        let content = (message["content"] as? String) ?? ""
        let reasoning = (message["reasoning_content"] as? String) ?? ""
        let finishReason = first["finish_reason"] as? String

        var toolCalls: [AgentToolCallRecord] = []
        if let rawToolCalls = message["tool_calls"] as? [[String: Any]] {
            for raw in rawToolCalls {
                let id = (raw["id"] as? String) ?? ""
                let function = (raw["function"] as? [String: Any]) ?? [:]
                let name = (function["name"] as? String) ?? ""
                let arguments = (function["arguments"] as? String) ?? "{}"
                toolCalls.append(AgentToolCallRecord(id: id, name: name, argumentsJSON: arguments))
            }
        }
        return AgentModelTurn(
            content: content, reasoning: reasoning, toolCalls: toolCalls, finishReason: finishReason)
    }

    /// Lit `/healthz` et en extrait le nom du modèle chargé — `nil` si le
    /// serveur ne rapporte aucun modèle résident (`model_loaded: false`).
    public static func parseHealthResponse(_ data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return (object["model"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
    }
}

extension String {
    fileprivate var nilIfEmpty: String? { isEmpty ? nil : self }
}
