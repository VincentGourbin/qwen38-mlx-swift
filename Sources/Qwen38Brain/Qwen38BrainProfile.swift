import Foundation
import MLX
#if os(iOS)
import os
#endif

/// A named, pre-measured set of runtime knobs for embedding a model as an
/// app's "brain" — the same idea as YuE2's reference profiles: `fast` keeps
/// everything Mac-sized, `lean` trades a little speed for a much smaller
/// footprint. Every field maps to an existing knob; the numbers behind each
/// choice live in `docs/bonsai2-brain/plan.md` and `BENCHMARKS.md`.
public struct Qwen38BrainProfile: Sendable, Equatable, Identifiable {
    public enum Kind: String, Sendable, CaseIterable { case fast, lean }

    public let kind: Kind
    /// KV-cache quantization for the full-attention layers (`nil` = fp16).
    public let kvBits: Int?
    /// `Memory.cacheLimit` in MB, or `nil` to size it from available memory.
    public let cacheLimitMB: Int?
    /// `Memory.memoryLimit` in MB (mlx's cache-GC threshold, not a hard cap),
    /// or `nil` to leave mlx's default.
    public let memoryLimitMB: Int?
    /// Free cached (not active) buffers after every answer.
    public let clearCacheAfterAnswer: Bool
    /// Skip the vision tower (Bonsai 2: −0.92 GB); images are then refused.
    /// Off in both named profiles; set it only for an app that never sends
    /// images.
    public let textOnly: Bool
    /// Prefill chunk in tokens: smaller chunks lower the transient peak
    /// (attention scores are materialised per chunk with a quantized KV).
    public let prefillStepSize: Int

    public var id: String { kind.rawValue }

    public init(
        kind: Kind, kvBits: Int?, cacheLimitMB: Int?, memoryLimitMB: Int?,
        clearCacheAfterAnswer: Bool, textOnly: Bool = false, prefillStepSize: Int = 512
    ) {
        self.kind = kind
        self.kvBits = kvBits
        self.cacheLimitMB = cacheLimitMB
        self.memoryLimitMB = memoryLimitMB
        self.clearCacheAfterAnswer = clearCacheAfterAnswer
        self.textOnly = textOnly
        self.prefillStepSize = prefillStepSize
    }

    /// Everything resident, fp16 KV, a Mac-sized buffer cache.
    public static let fast = Qwen38BrainProfile(
        kind: .fast, kvBits: nil, cacheLimitMB: 4096, memoryLimitMB: nil,
        clearCacheAfterAnswer: false, textOnly: false, prefillStepSize: 512)

    /// 8-bit KV, 256-token prefill chunks, a small buffer cache and a GC threshold sized from what the
    /// machine actually has, cache cleared between answers.
    public static var lean: Qwen38BrainProfile {
        let available = availableMemoryMB()
        return Qwen38BrainProfile(
            kind: .lean, kvBits: 8,
            cacheLimitMB: min(1024, max(256, available / 6)),
            memoryLimitMB: max(4096, available - 2048),
            clearCacheAfterAnswer: true, textOnly: false, prefillStepSize: 256)
    }

    public static func named(_ id: String) -> Qwen38BrainProfile? {
        switch id {
        case "fast": return .fast
        case "lean": return .lean
        default: return nil
        }
    }

    /// The same profile without the vision tower.
    public func textOnlyVariant() -> Qwen38BrainProfile {
        Qwen38BrainProfile(
            kind: kind, kvBits: kvBits, cacheLimitMB: cacheLimitMB, memoryLimitMB: memoryLimitMB,
            clearCacheAfterAnswer: clearCacheAfterAnswer, textOnly: true,
            prefillStepSize: prefillStepSize)
    }

    /// Process-wide knobs. Call after the model is loaded: the runtime sets
    /// its own `Memory.cacheLimit` while loading.
    public func applyGlobalPolicy() {
        if let cacheLimitMB { Memory.cacheLimit = cacheLimitMB * 1024 * 1024 }
        if let memoryLimitMB { Memory.memoryLimit = memoryLimitMB * 1024 * 1024 }
    }

    /// Memory the process can reasonably use, in MB. On iOS the OS says; on
    /// macOS physical memory minus 8 GB for the system and other apps.
    /// `QWEN38_BRAIN_AVAILABLE_MB` overrides it, to size `lean` for a 16 GB
    /// Mac while measuring on a bigger one.
    public static func availableMemoryMB() -> Int {
        if let raw = ProcessInfo.processInfo.environment["QWEN38_BRAIN_AVAILABLE_MB"],
            let value = Int(raw)
        {
            return value
        }
        #if os(iOS)
        return Int(os_proc_available_memory() / (1024 * 1024))
        #else
        let physical = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
        return max(4096, physical - 8192)
        #endif
    }
}
