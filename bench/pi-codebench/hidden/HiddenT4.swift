// Test d'acceptation caché de T4.
import Foundation
import Testing
@testable import AgentKit

@Test("T4 caché : count_lines, catalogue et garde de chemin")
func hiddenT4CountLines() throws {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("hidden-t4-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let files = ["vide.txt": "", "un.txt": "a", "deux.txt": "a\nb", "deux-final.txt": "a\nb\n", "trois.txt": "a\n\nc\n"]
    for (name, text) in files {
        try text.write(to: root.appendingPathComponent(name), atomically: true, encoding: .utf8)
    }
    let executor = AgentToolExecutor(sandbox: AgentSandbox(root: root))
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"vide.txt"}"#) == "0 lignes")
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"un.txt"}"#) == "1 lignes")
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"deux.txt"}"#) == "2 lignes")
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"deux-final.txt"}"#) == "2 lignes")
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"trois.txt"}"#) == "3 lignes")
    #expect(executor.run(name: "count_lines", argumentsJSON: #"{"path":"../../etc/hosts"}"#).hasPrefix("ERREUR : "))
    #expect(executor.run(name: "count_lines", argumentsJSON: "{}") == "ERREUR : paramètre requis manquant : path")

    #expect(AgentToolCatalog.requiredArguments["count_lines"] == ["path"])
    #expect(AgentToolCatalog.toolNames.contains("count_lines"))
    let names = AgentToolCatalog.schema().compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
    #expect(names.contains("count_lines"))
    #expect(names.count == 5)
}
