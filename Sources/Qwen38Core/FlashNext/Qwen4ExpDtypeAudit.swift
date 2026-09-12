import Foundation
import MLX
import MLXLMCommon

/// Contrôle de dtype aux frontières de couche, activé par `QWEN38_DTYPE_AUDIT=1`.
///
/// Motivation (2026-09-12) : deux fuites float32 ont chacune coûté un facteur
/// ~2 sur le décodage — les normes GDN/QSA qui ne revenaient pas au dtype
/// d'entrée (corrigé par F7), puis la tour vision fusionnée telle quelle dans
/// l'état caché. Les deux étaient invisibles sauf à mesurer. Ce module rend le
/// contrôle systématique : tout `float32` sur un tenseur **qui propage**
/// (état caché, contenu de cache) est une fuite. Le float32 interne et voulu
/// (état récurrent GDN, tables RoPE, scores de sélection QSA) ne passe pas
/// par ici.
public enum Qwen4ExpDtypeAudit {
    public static let isEnabled = ProcessInfo.processInfo.environment["QWEN38_DTYPE_AUDIT"] == "1"

    nonisolated(unsafe) private static var seen = Set<Int>()
    private static let lock = NSLock()

    public static func report(
        layer: Int, kind: String, input: MLXArray, output: MLXArray, cache: any KVCache
    ) {
        lock.lock()
        let first = seen.insert(layer).inserted
        lock.unlock()
        guard first else { return }
        let caches = cache.innerState().map { "\($0.dtype)" }.joined(separator: ",")
        let leak = (output.dtype == .float32 || input.dtype == .float32
            || cache.innerState().contains { $0.dtype == .float32 && kind == "QSA" })
        print("DTYPE-AUDIT couche \(String(format: "%2d", layer)) \(kind) · entrée \(input.dtype)"
            + " · sortie \(output.dtype) · cache [\(caches)]\(leak ? "   <-- FUITE ?" : "")")
    }

    public static func reset() { lock.lock(); seen.removeAll(); lock.unlock() }
}
