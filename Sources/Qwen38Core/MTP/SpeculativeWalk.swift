import Foundation

/// Greedy acceptance walk shared by the local M2 pipeline and its tests.
///
/// `targets[i]` is the target prediction for the token at draft position `i`.
/// The final target entry is the correction/bonus prediction after the last
/// draft. No MLX work happens here, which keeps the acceptance policy easy to
/// test independently from cache mutation.
public enum Qwen38SpeculativeWalk {
    public struct Result: Sendable, Equatable {
        public let accepted: Int
        public let emitted: [Int32]

        public init(accepted: Int, emitted: [Int32]) {
            self.accepted = accepted
            self.emitted = emitted
        }
    }

    public static func walk(
        drafts: [Int32],
        targets: [Int32],
        budget: Int
    ) -> Result {
        precondition(targets.count == drafts.count + 1)
        precondition(budget >= 0)

        var accepted = drafts.count
        for index in drafts.indices where drafts[index] != targets[index] {
            accepted = index
            break
        }

        var emitted = Array(drafts.prefix(accepted))
        emitted.append(targets[accepted])
        if emitted.count > budget {
            emitted.removeLast(emitted.count - budget)
        }
        return Result(accepted: accepted, emitted: emitted)
    }
}
