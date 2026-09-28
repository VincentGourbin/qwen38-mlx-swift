// Test d'acceptation caché de T3.
import Foundation
import Testing
@testable import AgentKit

@Test("T3 caché : taux de cache et résumé exact")
func hiddenT3Summary() {
    var stats = AgentStats()
    #expect(stats.cacheHitRate == 0)
    #expect(stats.summary == "0 pas · 0 appels (0 valides) · 0 jetons d'entrée dont 0 en cache (0,0 %) · 0 jetons de sortie")

    stats.steps = 3
    stats.calls = 5
    stats.valid = 4
    stats.promptTokens = 1200
    stats.cachedTokens = 800
    stats.completionTokens = 150
    #expect(abs(stats.cacheHitRate - 2.0 / 3.0) < 1e-9)
    #expect(stats.summary == "3 pas · 5 appels (4 valides) · 1200 jetons d'entrée dont 800 en cache (66,7 %) · 150 jetons de sortie")

    stats.promptTokens = 12_345
    stats.cachedTokens = 12_345
    #expect(stats.summary.contains("12345 jetons d'entrée dont 12345 en cache (100,0 %)"))
}
