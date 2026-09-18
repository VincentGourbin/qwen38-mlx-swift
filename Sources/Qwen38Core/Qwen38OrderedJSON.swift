import Foundation
import Jinja

/// A JSON value that keeps object members in document order.
///
/// Foundation's `JSONDecoder` and `JSONSerialization` both hand back
/// unordered dictionaries, and `Jinja.Value.init(any:)` sorts dictionary
/// keys to stay deterministic. Neither is what the model saw at training
/// time: transformers renders `{{ tool | tojson }}` and the `|items` loop
/// over `tool_call.arguments` with Python `json.dumps` and dict insertion
/// order, i.e. **the order the client sent**. This type parses the client's
/// bytes once, in order, and converts straight to a `Jinja.Value` whose
/// objects are ordered — `Value.init(any:)` passes an existing `Value`
/// through untouched, so the order survives all the way into the template
/// (see `Qwen38ToolSpec.templateValue` and `Qwen4ExpPromptBuilder.hfMessage`).
///
/// Number semantics follow Python's `json` module: a literal without `.`,
/// `e` or `E` is an integer, anything else a double; an integer that does
/// not fit `Int` degrades to a double.
public indirect enum Qwen38OrderedJSON: Sendable, Equatable {
    public struct Member: Sendable, Equatable {
        public let key: String
        public let value: Qwen38OrderedJSON
        public init(key: String, value: Qwen38OrderedJSON) {
            self.key = key
            self.value = value
        }
    }

    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([Qwen38OrderedJSON])
    case object([Member])

    /// First member with this key (JSON allows duplicates; Python keeps the
    /// last one, but a chat request never carries duplicates in practice).
    public subscript(key: String) -> Qwen38OrderedJSON? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.key == key }?.value
    }

    public var arrayValue: [Qwen38OrderedJSON]? {
        guard case .array(let values) = self else { return nil }
        return values
    }

    public var isObject: Bool {
        if case .object = self { return true }
        return false
    }

    /// Ordered `Jinja.Value`: objects become `OrderedDictionary`s in member
    /// order, which `tojson` (swift-jinja ≥ 2.5, `json.dumps` semantics) and
    /// `|items` both honour.
    public var jinjaValue: Jinja.Value {
        switch self {
        case .null: return .null
        case .bool(let value): return .boolean(value)
        case .int(let value): return .int(value)
        case .double(let value): return .double(value)
        case .string(let value): return .string(value)
        case .array(let values): return .array(values.map(\.jinjaValue))
        case .object(let members):
            var ordered = OrderedDictionary<String, Jinja.Value>()
            for member in members { ordered[member.key] = member.value.jinjaValue }
            return .object(ordered)
        }
    }

    public static func parse(_ text: String) throws -> Qwen38OrderedJSON {
        try parse(Data(text.utf8))
    }

    public static func parse(_ data: Data) throws -> Qwen38OrderedJSON {
        var parser = Parser(bytes: [UInt8](data))
        let value = try parser.parseValue()
        parser.skipWhitespace()
        guard parser.isAtEnd else { throw parser.error("trailing characters") }
        return value
    }
}

public struct Qwen38OrderedJSONError: Error, CustomStringConvertible, Sendable {
    public let offset: Int
    public let reason: String
    public var description: String { "JSON invalide à l'octet \(offset) : \(reason)" }
}

// MARK: - Parser

extension Qwen38OrderedJSON {
    fileprivate struct Parser {
        let bytes: [UInt8]
        var index = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        var isAtEnd: Bool { index >= bytes.count }

        func error(_ reason: String) -> Qwen38OrderedJSONError {
            Qwen38OrderedJSONError(offset: index, reason: reason)
        }

        mutating func skipWhitespace() {
            while index < bytes.count {
                switch bytes[index] {
                case 0x20, 0x09, 0x0A, 0x0D: index += 1
                default: return
                }
            }
        }

        private mutating func expect(_ byte: UInt8) throws {
            guard index < bytes.count, bytes[index] == byte else {
                throw error("'\(Character(UnicodeScalar(byte)))' attendu")
            }
            index += 1
        }

        private mutating func expectLiteral(_ literal: String) throws {
            let expected = Array(literal.utf8)
            guard index + expected.count <= bytes.count,
                Array(bytes[index..<index + expected.count]) == expected
            else { throw error("littéral \(literal) attendu") }
            index += expected.count
        }

        mutating func parseValue() throws -> Qwen38OrderedJSON {
            skipWhitespace()
            guard index < bytes.count else { throw error("fin prématurée") }
            switch bytes[index] {
            case UInt8(ascii: "{"): return try parseObject()
            case UInt8(ascii: "["): return try parseArray()
            case UInt8(ascii: "\""): return .string(try parseString())
            case UInt8(ascii: "t"): try expectLiteral("true"); return .bool(true)
            case UInt8(ascii: "f"): try expectLiteral("false"); return .bool(false)
            case UInt8(ascii: "n"): try expectLiteral("null"); return .null
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"): return try parseNumber()
            default: throw error("caractère inattendu")
            }
        }

        private mutating func parseObject() throws -> Qwen38OrderedJSON {
            try expect(UInt8(ascii: "{"))
            var members: [Member] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
                index += 1
                return .object(members)
            }
            while true {
                skipWhitespace()
                let key = try parseString()
                skipWhitespace()
                try expect(UInt8(ascii: ":"))
                let value = try parseValue()
                members.append(Member(key: key, value: value))
                skipWhitespace()
                guard index < bytes.count else { throw error("fin prématurée dans un objet") }
                if bytes[index] == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                try expect(UInt8(ascii: "}"))
                return .object(members)
            }
        }

        private mutating func parseArray() throws -> Qwen38OrderedJSON {
            try expect(UInt8(ascii: "["))
            var values: [Qwen38OrderedJSON] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "]") {
                index += 1
                return .array(values)
            }
            while true {
                values.append(try parseValue())
                skipWhitespace()
                guard index < bytes.count else { throw error("fin prématurée dans un tableau") }
                if bytes[index] == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                try expect(UInt8(ascii: "]"))
                return .array(values)
            }
        }

        private mutating func parseNumber() throws -> Qwen38OrderedJSON {
            let start = index
            var isInteger = true
            while index < bytes.count {
                let byte = bytes[index]
                switch byte {
                case UInt8(ascii: "0")...UInt8(ascii: "9"), UInt8(ascii: "-"), UInt8(ascii: "+"):
                    index += 1
                case UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E"):
                    isInteger = false
                    index += 1
                default:
                    guard let text = String(bytes: bytes[start..<index], encoding: .utf8) else {
                        throw error("nombre illisible")
                    }
                    return try number(from: text, isInteger: isInteger)
                }
            }
            guard let text = String(bytes: bytes[start..<index], encoding: .utf8) else {
                throw error("nombre illisible")
            }
            return try number(from: text, isInteger: isInteger)
        }

        private func number(from text: String, isInteger: Bool) throws -> Qwen38OrderedJSON {
            if isInteger, let value = Int(text) { return .int(value) }
            guard let value = Double(text) else { throw error("nombre invalide \(text)") }
            return .double(value)
        }

        private mutating func parseString() throws -> String {
            try expect(UInt8(ascii: "\""))
            var scalars = String.UnicodeScalarView()
            var raw: [UInt8] = []
            func flushRaw() throws {
                guard !raw.isEmpty else { return }
                guard let text = String(bytes: raw, encoding: .utf8) else { throw error("UTF-8 invalide") }
                scalars.append(contentsOf: text.unicodeScalars)
                raw.removeAll(keepingCapacity: true)
            }
            while index < bytes.count {
                let byte = bytes[index]
                index += 1
                switch byte {
                case UInt8(ascii: "\""):
                    try flushRaw()
                    return String(scalars)
                case UInt8(ascii: "\\"):
                    try flushRaw()
                    guard index < bytes.count else { throw error("échappement tronqué") }
                    let escape = bytes[index]
                    index += 1
                    switch escape {
                    case UInt8(ascii: "\""): scalars.append("\"")
                    case UInt8(ascii: "\\"): scalars.append("\\")
                    case UInt8(ascii: "/"): scalars.append("/")
                    case UInt8(ascii: "b"): scalars.append("\u{08}")
                    case UInt8(ascii: "f"): scalars.append("\u{0C}")
                    case UInt8(ascii: "n"): scalars.append("\n")
                    case UInt8(ascii: "r"): scalars.append("\r")
                    case UInt8(ascii: "t"): scalars.append("\t")
                    case UInt8(ascii: "u"):
                        var code = try parseHex4()
                        if (0xD800...0xDBFF).contains(code) {
                            // Surrogate pair: the low half must follow as \uXXXX.
                            guard index + 1 < bytes.count, bytes[index] == UInt8(ascii: "\\"),
                                bytes[index + 1] == UInt8(ascii: "u")
                            else { throw error("paire de substitution incomplète") }
                            index += 2
                            let low = try parseHex4()
                            guard (0xDC00...0xDFFF).contains(low) else { throw error("paire de substitution invalide") }
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        }
                        guard let scalar = UnicodeScalar(code) else { throw error("point de code invalide") }
                        scalars.append(scalar)
                    default:
                        throw error("échappement inconnu")
                    }
                default:
                    raw.append(byte)
                }
            }
            throw error("chaîne non terminée")
        }

        private mutating func parseHex4() throws -> UInt32 {
            guard index + 4 <= bytes.count,
                let text = String(bytes: bytes[index..<index + 4], encoding: .utf8),
                let value = UInt32(text, radix: 16)
            else { throw error("\\uXXXX attendu") }
            index += 4
            return value
        }
    }
}
