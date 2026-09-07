import Foundation

public enum Qwen38MTPEngine: String, Sendable, Equatable, CaseIterable {
    /// The iterator implemented by mlx-swift-lm; the stable default.
    case upstream
    /// The local multi-token loop with persistent target/drafter state.
    case local
}

/// Controls the upstream MTP baseline. The house multi-token pipeline will
/// consume the same option once it lands; keeping the option here avoids a UI
/// and server API break between M1 and M2.
public enum Qwen38MTPDraftDepth: Sendable, Equatable {
    case fixed(Int)
    case automatic

    public static let `default`: Self = .fixed(1)

    /// Number of draft tokens requested by the caller, clamped to the safe
    /// range used by the current MTP implementation.
    public var requestedDraftTokens: Int {
        switch self {
        case .fixed(let value): return min(max(value, 1), 8)
        case .automatic: return 1
        }
    }
}

public struct Qwen38MTPOptions: Sendable, Equatable {
    public var enabled: Bool
    public var draftDepth: Qwen38MTPDraftDepth
    public var engine: Qwen38MTPEngine

    public init(
        enabled: Bool = false,
        draftDepth: Qwen38MTPDraftDepth = .default,
        engine: Qwen38MTPEngine = .upstream
    ) {
        self.enabled = enabled
        self.draftDepth = draftDepth
        self.engine = engine
    }
}
