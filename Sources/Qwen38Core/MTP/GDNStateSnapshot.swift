import Foundation
import MLX
import MLXLMCommon

/// Public-API snapshot of the recurrent state carried by Qwen's Gated
/// DeltaNet layers.
///
/// This is deliberately a value describing only `MambaCache` entries. Full
/// attention caches remain owned by the upstream cache machinery. A snapshot
/// is captured before a speculative verify pass and restored when the target
/// rejects a suffix of the proposed block.
public struct Qwen38GDNStateSnapshot {
    public struct Entry {
        public let state: [MLXArray]
        public let offset: Int

        fileprivate init(state: [MLXArray], offset: Int) {
            self.state = state
            self.offset = offset
        }
    }

    public let entries: [Entry]

    public init(caches: [any KVCache]) throws {
        var entries = [Entry]()
        entries.reserveCapacity(caches.count)
        for cache in caches {
            guard let mamba = cache as? MambaCache else {
                continue
            }
            guard mamba.state.count == 2 else {
                throw Qwen38GDNStateSnapshotError.invalidStateCount(mamba.state.count)
            }
            entries.append(Entry(state: mamba.state, offset: mamba.offset))
        }
        self.entries = entries
    }

    /// Restore all recurrent entries in cache order.
    ///
    /// The method refuses a topology mismatch instead of restoring a prefix
    /// and leaving the remaining layers at speculative positions.
    @discardableResult
    public func restore(to caches: [any KVCache]) throws -> Int {
        let mambaCaches = caches.compactMap { $0 as? MambaCache }
        guard mambaCaches.count == entries.count else {
            throw Qwen38GDNStateSnapshotError.topologyMismatch(
                expected: entries.count, actual: mambaCaches.count)
        }
        for (cache, entry) in zip(mambaCaches, entries) {
            guard entry.state.count == 2 else {
                throw Qwen38GDNStateSnapshotError.invalidStateCount(entry.state.count)
            }
            cache.state = entry.state
            cache.offset = entry.offset
        }
        return entries.count
    }
}

public enum Qwen38GDNStateSnapshotError: LocalizedError, Equatable {
    case invalidStateCount(Int)
    case topologyMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidStateCount(let count):
            return "État GDN inattendu : \(count) tenseur(s), 2 attendus."
        case .topologyMismatch(let expected, let actual):
            return "Topologie GDN incohérente : \(actual) cache(s), \(expected) attendu(s)."
        }
    }
}
