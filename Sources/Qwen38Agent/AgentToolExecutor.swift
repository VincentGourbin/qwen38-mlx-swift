import Foundation

/// Un argument requis manque, ou n'a pas le type attendu.
struct AgentToolInputError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// Exécute les trois outils de lecture seule sous la racine d'un
/// `AgentSandbox`. Volontairement sans `Process`/sous-processus : pas
/// d'exécution de commande shell (contrainte du panneau Agent), `grep` est
/// donc réimplémenté en Swift pur plutôt que de déléguer au binaire système.
///
/// Ne lève jamais : toute erreur (chemin hors racine, fichier absent,
/// argument manquant) revient comme texte `"ERREUR : ..."`, exactement le
/// contrat de `run_tool` dans `Scripts/agent-loop.py` — pour que le modèle
/// puisse la lire et se corriger au pas suivant, plutôt que de faire
/// planter la boucle.
public struct AgentToolExecutor: Sendable {
    public let sandbox: AgentSandbox

    public init(sandbox: AgentSandbox) {
        self.sandbox = sandbox
    }

    /// `name` un des trois outils de lecture ; `final_answer` n'est jamais
    /// routé ici (il termine la boucle avant d'atteindre l'exécuteur — voir
    /// `AgentLoopEngine.apply`).
    public func run(name: String, argumentsJSON: String) -> String {
        let arguments = Self.decodeArguments(argumentsJSON)
        do {
            switch name {
            case "list_files":
                return try listFiles(path: arguments["path"] as? String)
            case "read_file":
                guard let path = arguments["path"] as? String else {
                    throw AgentToolInputError(message: "paramètre requis manquant : path")
                }
                let startLine = Self.intValue(arguments["start_line"]) ?? 1
                return try readFile(path: path, startLine: startLine)
            case "grep":
                guard let pattern = arguments["pattern"] as? String else {
                    throw AgentToolInputError(message: "paramètre requis manquant : pattern")
                }
                return try grep(pattern: pattern, path: arguments["path"] as? String)
            default:
                return "outil inconnu : \(name)"
            }
        } catch {
            return "ERREUR : \(Self.describe(error))"
        }
    }

    // MARK: outils

    private func listFiles(path: String?) throws -> String {
        let directory = try sandbox.resolve(path)
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
        guard !entries.isEmpty else { return "(vide)" }
        let limited = entries.prefix(AgentTruncation.listFilesEntryLimit)
        return AgentTruncation.truncate(limited.joined(separator: "\n"), limit: AgentTruncation.toolOutputCharLimit)
    }

    private func readFile(path: String, startLine: Int) throws -> String {
        let file = try sandbox.resolve(path)
        let text = try String(contentsOf: file, encoding: .utf8)
        let lines = text.components(separatedBy: "\n")
        let start = max(startLine, 1)
        let startIndex = start - 1
        guard startIndex < lines.count else { return "" }
        let endIndex = min(startIndex + AgentTruncation.readFileLineLimit, lines.count)
        var rendered = [String]()
        rendered.reserveCapacity(endIndex - startIndex)
        for index in startIndex ..< endIndex {
            rendered.append("\(index + 1)\t\(lines[index])")
        }
        return AgentTruncation.truncate(rendered.joined(separator: "\n"), limit: AgentTruncation.toolOutputCharLimit)
    }

    private func grep(pattern: String, path: String?) throws -> String {
        let root = try sandbox.resolve(path)
        let allowedExtensions: Set<String> = ["swift", "md"]
        let fileManager = FileManager.default

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
            return "(aucune correspondance)"
        }

        let files: [URL]
        if isDirectory.boolValue {
            let enumerator = fileManager.enumerator(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            var collected: [URL] = []
            while let next = enumerator?.nextObject() as? URL {
                if allowedExtensions.contains(next.pathExtension) { collected.append(next) }
            }
            files = collected.sorted { $0.path < $1.path }
        } else {
            files = allowedExtensions.contains(root.pathExtension) ? [root] : []
        }

        // `pattern` peut ne pas être une regex valide (un motif de langage
        // courant, par exemple) : on retombe alors sur une recherche de
        // sous-chaîne littérale plutôt que d'échouer.
        let regex = try? NSRegularExpression(pattern: pattern)

        var matches: [String] = []
        for file in files {
            guard let content = try? String(contentsOf: file, encoding: .utf8) else { continue }
            let lines = content.components(separatedBy: "\n")
            for (index, line) in lines.enumerated() {
                let isMatch: Bool
                if let regex {
                    isMatch = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil
                } else {
                    isMatch = line.contains(pattern)
                }
                guard isMatch else { continue }
                matches.append("\(file.path):\(index + 1):\(line)")
                if matches.count >= AgentTruncation.grepLineLimit { break }
            }
            if matches.count >= AgentTruncation.grepLineLimit { break }
        }

        let joined = matches.isEmpty ? "(aucune correspondance)" : matches.joined(separator: "\n")
        return AgentTruncation.truncate(joined, limit: AgentTruncation.toolOutputCharLimit)
    }

    // MARK: décodage des arguments

    private static func decodeArguments(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return object
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String { return Int(string) }
        return nil
    }

    private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return String(describing: error)
    }
}
