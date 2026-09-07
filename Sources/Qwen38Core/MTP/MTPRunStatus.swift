import Foundation

public enum Qwen38MTPAvailability: Sendable, Equatable {
    case active
    case unavailable
    case fallback(String)

    public var isActive: Bool {
        if case .active = self { return true }
        return false
    }
}

public struct Qwen38MTPRunStatus: Sendable, Equatable {
    public let availability: Qwen38MTPAvailability
    public let engine: Qwen38MTPEngine?
    public let blockSize: Int?
    public let proposedTokens: Int
    public let acceptedTokens: Int
    public let rounds: Int
    public let passthroughReason: String?

    public init(
        availability: Qwen38MTPAvailability,
        engine: Qwen38MTPEngine? = nil,
        blockSize: Int? = nil,
        proposedTokens: Int = 0,
        acceptedTokens: Int = 0,
        rounds: Int = 0,
        passthroughReason: String? = nil
    ) {
        self.availability = availability
        self.engine = engine
        self.blockSize = blockSize
        self.proposedTokens = proposedTokens
        self.acceptedTokens = acceptedTokens
        self.rounds = rounds
        self.passthroughReason = passthroughReason
    }

    public var acceptanceRate: Double? {
        guard proposedTokens > 0 else { return nil }
        return Double(acceptedTokens) / Double(proposedTokens)
    }
}
