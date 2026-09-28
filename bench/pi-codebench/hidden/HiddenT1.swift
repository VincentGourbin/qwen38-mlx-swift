// Test d'acceptation caché de T1 (copié par le banc après le run, jamais vu par l'agent).
import Foundation
import Testing
@testable import AgentKit

private func hiddenT1Parse(_ usage: String?) throws -> AgentModelTurn {
    let usagePart = usage.map { ", \"usage\": \($0)" } ?? ""
    let json = """
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]\(usagePart)}
        """
    return try AgentWireFormat.parseChatCompletionResponse(Data(json.utf8))
}

@Test("T1 caché : usage complet, partiel, absent")
func hiddenT1Usage() throws {
    let full = try hiddenT1Parse("""
        {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350,
         "prompt_tokens_details": {"cached_tokens": 800}}
        """)
    #expect(full.usage == AgentTokenUsage(promptTokens: 1200, completionTokens: 150, cachedTokens: 800))
    #expect(full.content == "ok")

    let noDetails = try hiddenT1Parse(#"{"prompt_tokens": 10, "completion_tokens": 3}"#)
    #expect(noDetails.usage == AgentTokenUsage(promptTokens: 10, completionTokens: 3))
    #expect(noDetails.usage?.cachedTokens == 0)

    let partial = try hiddenT1Parse(#"{"completion_tokens": 7, "prompt_tokens_details": {}}"#)
    #expect(partial.usage == AgentTokenUsage(promptTokens: 0, completionTokens: 7, cachedTokens: 0))

    #expect(try hiddenT1Parse(nil).usage == nil)
    #expect(AgentModelTurn(content: "x").usage == nil)
}
