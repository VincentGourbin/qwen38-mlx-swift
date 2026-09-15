import Foundation

/// P13.1 : le socle pur (aucun tokenizer, aucun checkpoint) de l'appel
/// d'outils au format OpenAI — voir PLAN.md, section "P13". Trois
/// responsabilités séparées :
///
///  - `Qwen38JSONValue` : un arbre JSON `Sendable` (le schéma
///    `tools[].function.parameters` d'une requête, et les valeurs typées
///    d'un appel une fois reconnu) — nécessaire parce que `Tokenizers.
///    ToolSpec`/`Message` sont typés `[String: any Sendable]` et qu'un `Any`
///    brut ne satisfait pas Swift 6 en mode concurrence stricte.
///  - `Qwen38ToolCallParser` : reconnaît les blocs `<tool_call>` du texte
///    déjà généré (après retrait du `<think>`) — le format XML-ish du
///    gabarit du checkpoint Vontra, jamais du JSON.
///  - `Qwen38ToolArgumentTyper` : convertit les valeurs texte brutes d'un
///    appel reconnu vers leur type JSON Schema déclaré (entier, nombre,
///    booléen, tableau/objet), avec repli sur la chaîne.

// MARK: - Qwen38JSONValue

/// A minimal `Sendable` JSON value tree. Used both to decode an arbitrary
/// `tools[].function.parameters` JSON Schema object from a request, and to
/// hold a tool call's typed argument values before they are serialized back
/// into the OpenAI-wire-format `function.arguments` JSON string.
public enum Qwen38JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([Qwen38JSONValue])
    case object([String: Qwen38JSONValue])
}

extension Qwen38JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([Qwen38JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: Qwen38JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Valeur JSON non prise en charge.")
        }
    }

    /// Not actually needed for P13.1's request-decode-only use (the
    /// `parameters`/`tool_choice` fields never round-trip back through
    /// `JSONEncoder`), but declaring `Codable` rather than `Decodable` keeps
    /// this a drop-in `Codable` field on the request structs above without
    /// their own custom `Encodable` — those structs are only ever decoded
    /// in practice.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

extension Qwen38JSONValue {
    /// Parses a standalone JSON document (e.g. a tool call parameter's raw
    /// text when its schema says `array`/`object`, or a client's
    /// `function.arguments` string being replayed into history).
    public static func parse(_ text: String) throws -> Qwen38JSONValue {
        try JSONDecoder().decode(Qwen38JSONValue.self, from: Data(text.utf8))
    }

    /// Bridges into the tokenizer's `any Sendable`-typed dictionaries
    /// (`Tokenizers.ToolSpec`/`Message`, both `[String: any Sendable]`).
    /// `Jinja.Value.init(any:)` has no case for a bare null inside a
    /// non-Optional `any Sendable` container, so a JSON `null` degrades to
    /// an empty string here — harmless in practice, since a
    /// `tools[].function.parameters` JSON Schema essentially never carries a
    /// literal `null` the model needs to see (reported to Vincent, PLAN.md
    /// P13.1).
    public var sendableValue: any Sendable {
        switch self {
        case .null: return ""
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { $0.sendableValue }
        case .object(let values): return values.mapValues { $0.sendableValue }
        }
    }

    /// Compact JSON text — used to build the OpenAI-wire-format
    /// `function.arguments` string for a parsed tool call. Object keys are
    /// emitted in sorted order for determinism: the source `<parameter=…>`
    /// blocks carry no ordering guarantee worth preserving.
    public func toJSONString() -> String {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .string(let value): return Self.encodedJSONString(value)
        case .array(let values): return "[" + values.map { $0.toJSONString() }.joined(separator: ",") + "]"
        case .object(let values):
            let pairs = values.keys.sorted().map { key -> String in
                "\(Self.encodedJSONString(key)):\(values[key]!.toJSONString())"
            }
            return "{" + pairs.joined(separator: ",") + "}"
        }
    }

    private static func encodedJSONString(_ value: String) -> String {
        var result = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            default:
                if scalar.value < 0x20 {
                    result += String(format: "\\u%04x", scalar.value)
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        result += "\""
        return result
    }
}

// MARK: - Tool specs and calls (transport-neutral)

/// One `tools[]` entry in the OpenAI wire format, threaded as-is into
/// `Tokenizers.ToolSpec` (`applyChatTemplate(tools:)`) — Flash-Next only,
/// since the "Tools" system-message convention below lives in this
/// checkpoint's own `chat_template.jinja`, not in the 27B family's.
public struct Qwen38ToolSpec: Sendable, Equatable {
    public let type: String
    public let name: String
    public let description: String?
    public let parameters: Qwen38JSONValue?

    public init(
        type: String = "function", name: String, description: String? = nil,
        parameters: Qwen38JSONValue? = nil
    ) {
        self.type = type
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// The `Tokenizers.ToolSpec` (`[String: any Sendable]`) dictionary this
    /// renders as for `applyChatTemplate` — the checkpoint's own template
    /// just does `tool | tojson`, so whatever shape is handed to it is what
    /// the model sees; this mirrors the OpenAI `tools[]` shape exactly.
    public var toolSpecDictionary: [String: any Sendable] {
        var function: [String: any Sendable] = ["name": name]
        if let description { function["description"] = description }
        if let parameters { function["parameters"] = parameters.sendableValue }
        return ["type": type, "function": function]
    }
}

/// One assistant-turn tool call — the transport-neutral pendant of an
/// OpenAI `message.tool_calls[]` entry, carried on `Qwen38ChatMessage` so a
/// tool round trip can be replayed into the next turn's chat-template
/// rendering.
public struct Qwen38ToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    /// Compact JSON text of the arguments object — the OpenAI wire format
    /// for `function.arguments` (e.g. `{"command":"ls","timeout":300}`).
    public let argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

// MARK: - Parsing the checkpoint's `<tool_call>` XML-ish output

/// One `<parameter=NAME>` entry inside a recognized `<tool_call>` block, in
/// the order the model emitted it. `rawValue` is the untyped, unescaped
/// text between the tags — typing per the tool's JSON Schema happens
/// separately, in `Qwen38ToolArgumentTyper`.
public struct Qwen38ToolCallParameter: Sendable, Equatable {
    public let name: String
    public let rawValue: String

    public init(name: String, rawValue: String) {
        self.name = name
        self.rawValue = rawValue
    }
}

/// One fully-closed `<tool_call>` block recognized in the model's output.
public struct Qwen38ParsedToolCall: Sendable, Equatable {
    public let name: String
    public let parameters: [Qwen38ToolCallParameter]

    public init(name: String, parameters: [Qwen38ToolCallParameter]) {
        self.name = name
        self.parameters = parameters
    }
}

/// Recognizes the checkpoint's `<tool_call>` blocks in already-generated
/// text (post-`<think>` stripping — see `Qwen38ThinkingStreamParser`). Pure
/// string scanning, deliberately **not** a generic XML/regex parser: a
/// parameter's raw value is allowed to contain angle brackets and newlines
/// (source code, for instance), so the only structural tokens this ever
/// looks for are the checkpoint's own literal markers
/// (`<tool_call>`/`</tool_call>`, `<function=…>`/`</function>`,
/// `<parameter=…>`/`</parameter>`) — never a bare `<` or `>`.
public enum Qwen38ToolCallParser {
    private static let openTag = "<tool_call>"
    private static let closeTag = "</tool_call>"
    private static let functionOpenPrefix = "<function="
    private static let functionClose = "</function>"
    private static let parameterOpenPrefix = "<parameter="
    private static let parameterClose = "</parameter>"

    /// Splits `text` into the free-form content the model produced and the
    /// well-formed `<tool_call>` blocks within it, in order.
    ///
    /// Only a **fully closed** block becomes a call: `<tool_call>` with a
    /// matching `</tool_call>`, itself containing a fully closed
    /// `<function=…>…</function>` with fully closed `<parameter=…>` entries.
    /// An unterminated `<tool_call>` — the shape a `max_tokens` cutoff mid-
    /// call leaves behind — is folded back into `content` verbatim instead
    /// of ever being guessed at or completed: this is what keeps a
    /// truncated generation from producing a malformed call (PLAN.md P13.1's
    /// truncation pitfall). A `<tool_call>…</tool_call>` block that *is*
    /// closed but doesn't match the `<function=…>` grammar is preserved the
    /// same way, for the same reason.
    public static func parse(_ text: String) -> (content: String, calls: [Qwen38ParsedToolCall]) {
        var content = ""
        var calls: [Qwen38ParsedToolCall] = []
        var cursor = text.startIndex
        while let openRange = text.range(of: openTag, range: cursor..<text.endIndex) {
            content += text[cursor..<openRange.lowerBound]
            guard let closeRange = text.range(of: closeTag, range: openRange.upperBound..<text.endIndex)
            else {
                content += text[openRange.lowerBound..<text.endIndex]
                cursor = text.endIndex
                break
            }
            let body = text[openRange.upperBound..<closeRange.lowerBound]
            if let call = parseCall(body) {
                calls.append(call)
            } else {
                content += text[openRange.lowerBound..<closeRange.upperBound]
            }
            cursor = closeRange.upperBound
        }
        content += text[cursor..<text.endIndex]
        return (content, calls)
    }

    private static func parseCall(_ body: Substring) -> Qwen38ParsedToolCall? {
        guard let openRange = body.range(of: functionOpenPrefix) else { return nil }
        guard let nameEnd = body[openRange.upperBound...].firstIndex(of: ">") else { return nil }
        let name = String(body[openRange.upperBound..<nameEnd])
        guard !name.isEmpty else { return nil }
        let innerStart = body.index(after: nameEnd)
        guard let closeRange = body.range(of: functionClose, range: innerStart..<body.endIndex)
        else { return nil }
        let parameters = parseParameters(body[innerStart..<closeRange.lowerBound])
        return Qwen38ParsedToolCall(name: name, parameters: parameters)
    }

    private static func parseParameters(_ text: Substring) -> [Qwen38ToolCallParameter] {
        var parameters: [Qwen38ToolCallParameter] = []
        var cursor = text.startIndex
        while let openRange = text.range(of: parameterOpenPrefix, range: cursor..<text.endIndex) {
            guard let nameEnd = text[openRange.upperBound...].firstIndex(of: ">") else { break }
            let name = String(text[openRange.upperBound..<nameEnd])
            let valueStart = text.index(after: nameEnd)
            guard let closeRange = text.range(of: parameterClose, range: valueStart..<text.endIndex)
            else { break }
            var raw = text[valueStart..<closeRange.lowerBound]
            // The template always wraps a value in exactly one structural
            // newline on each side (`'<parameter=' + name + '>\n'` … value …
            // `'\n</parameter>'`) — strip only those, never a value's own
            // intentional blank lines.
            if raw.first == "\n" { raw = raw.dropFirst() }
            if raw.last == "\n" { raw = raw.dropLast() }
            parameters.append(Qwen38ToolCallParameter(name: name, rawValue: String(raw)))
            cursor = closeRange.upperBound
        }
        return parameters
    }
}

// MARK: - Typing a parsed call's raw parameter text per JSON Schema

/// Converts a parsed call's raw (always-string) parameter text into typed
/// JSON values per the tool's declared `parameters` JSON Schema — "an
/// `integer` must come out as a number, not a string," falling back to a
/// plain string wherever the schema says nothing or the value doesn't
/// parse as declared (PLAN.md P13.1).
public enum Qwen38ToolArgumentTyper {
    public static func typedArguments(
        _ parameters: [Qwen38ToolCallParameter], schema: Qwen38JSONValue?
    ) -> Qwen38JSONValue {
        let properties: [String: Qwen38JSONValue]
        if case .object(let root)? = schema, case .object(let props)? = root["properties"] {
            properties = props
        } else {
            properties = [:]
        }
        var result: [String: Qwen38JSONValue] = [:]
        for parameter in parameters {
            result[parameter.name] = typedValue(raw: parameter.rawValue, schema: properties[parameter.name])
        }
        return .object(result)
    }

    static func typedValue(raw: String, schema: Qwen38JSONValue?) -> Qwen38JSONValue {
        guard case .object(let propertySchema)? = schema, case .string(let type)? = propertySchema["type"]
        else {
            return .string(raw)
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case "integer":
            return Int(trimmed).map(Qwen38JSONValue.int) ?? .string(raw)
        case "number":
            return Double(trimmed).map(Qwen38JSONValue.double) ?? .string(raw)
        case "boolean":
            switch trimmed.lowercased() {
            case "true": return .bool(true)
            case "false": return .bool(false)
            default: return .string(raw)
            }
        case "array", "object":
            return (try? Qwen38JSONValue.parse(raw)) ?? .string(raw)
        default:
            return .string(raw)
        }
    }
}
