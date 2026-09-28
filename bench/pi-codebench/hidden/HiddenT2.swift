// Test d'acceptation caché de T2.
import Foundation
import Testing
@testable import AgentKit

@Test("T2 caché : les jetons se cumulent sur tous les types de tours")
func hiddenT2Accumulation() throws {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("hidden-t2-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try "bonjour\n".write(to: root.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)

    var engine = AgentLoopEngine(rootURL: root, task: "question")
    engine.apply(AgentModelTurn(content: "je réfléchis",
                                usage: AgentTokenUsage(promptTokens: 100, completionTokens: 10, cachedTokens: 0)))
    engine.apply(AgentModelTurn(
        toolCalls: [AgentToolCallRecord(id: "1", name: "read_file", argumentsJSON: #"{"path":"a.md"}"#)],
        usage: AgentTokenUsage(promptTokens: 200, completionTokens: 20, cachedTokens: 90)))
    engine.apply(AgentModelTurn(content: "sans usage"))
    let final = engine.apply(AgentModelTurn(
        toolCalls: [AgentToolCallRecord(id: "2", name: "final_answer", argumentsJSON: #"{"answer":"fini"}"#)],
        usage: AgentTokenUsage(promptTokens: 300, completionTokens: 30, cachedTokens: 190)))

    #expect(final.outcome == .finalAnswer("fini"))
    #expect(engine.stats.promptTokens == 600)
    #expect(engine.stats.completionTokens == 60)
    #expect(engine.stats.cachedTokens == 280)
    #expect(engine.stats.steps == 4)
    #expect(engine.stats.textOnlyTurns == 2)
    #expect(engine.stats.calls == 2)
    #expect(engine.stats.valid == 2)
    #expect(AgentStats().promptTokens == 0 && AgentStats().cachedTokens == 0)
}
