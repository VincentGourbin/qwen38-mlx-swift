import Foundation
import MLXLLM
import MLXVLM

/// Installs the Qwen MTP creators into mlx-swift-lm's registries.
///
/// The upstream registries are actor-isolated and last-write-wins. Calling
/// both registrations is intentional: the VLM factory uses the
/// M-RoPE-aware registry, while the text registration keeps this module
/// usable for a future text-only target and for diagnostics.
public enum Qwen38MTPRegistration {
    public static func register() async {
        // The upstream actor registries overwrite the same creator
        // idempotently. Avoid a process-global mutable guard here: Swift 6
        // correctly rejects lock-protected globals in async code.
        await Qwen35TextMTPRegistration.register()
        await Qwen35VLMMTPRegistration.register()
    }
}
