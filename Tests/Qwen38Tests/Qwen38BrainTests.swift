import Foundation
import Testing
@testable import Qwen38Brain

@Test("P16 : le texte d'une réponse outillée s'arrête au marqueur <tool_call>, même coupé entre deux fragments")
func brainSplitterHoldsBackToolCall() {
    var splitter = Qwen38ToolCallTextSplitter(enabled: true)
    var visible = splitter.append("Je lis le fichier.<tool")
    visible += splitter.append("_call>\n<function=read_file>")
    visible += splitter.append("</function></tool_call>")
    visible += splitter.finish()
    #expect(visible == "Je lis le fichier.")
}

@Test("P16 : un faux début de marqueur est rendu tel quel")
func brainSplitterReleasesFalseMarker() {
    var splitter = Qwen38ToolCallTextSplitter(enabled: true)
    var visible = splitter.append("a <tool")
    visible += splitter.append("s> b")
    visible += splitter.finish()
    #expect(visible == "a <tools> b")
}

@Test("P16 : sans outils déclarés, le texte passe sans filtrage")
func brainSplitterDisabled() {
    var splitter = Qwen38ToolCallTextSplitter(enabled: false)
    #expect(splitter.append("x <tool_call> y") == "x <tool_call> y")
}

@Test("P16 : les profils nommés existent et lean est plus sobre que fast")
func brainProfiles() throws {
    let fast = try #require(Qwen38BrainProfile.named("fast"))
    let lean = try #require(Qwen38BrainProfile.named("lean"))
    #expect(fast.kvBits == nil)
    #expect(lean.kvBits == 8)
    #expect((lean.cacheLimitMB ?? .max) < (fast.cacheLimitMB ?? .max))
    #expect(lean.clearCacheAfterAnswer)
    #expect(!lean.textOnly && !fast.textOnly)
    #expect(lean.textOnlyVariant().textOnly)
    #expect(lean.prefillStepSize < fast.prefillStepSize)
    #expect(Qwen38BrainProfile.named("max") == nil)
}
