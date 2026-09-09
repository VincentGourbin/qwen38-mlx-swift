import MLX

/// PM4.2 (2026-09-09, P-MTP suite): materials captured during an MTP
/// verification forward (`Qwen4ExpStreamingTextModel.forward(...,
/// verificationCapture:)`) so a partially rejected draft block can roll
/// every `ArraysCache`-backed layer (GDN, PLE) back to the state after
/// exactly `committedNewTokens` of the newly fed tokens — without replaying
/// any forward pass.
///
/// Every captured tensor is already fully materialized by the single
/// verification forward; rolling back is a cheap host-side slice, not a
/// device computation:
/// - GDN's conv1d input, PLE's short-conv input and PLE's raw-ID history
///   window are plain concatenations of "old state" + "new tokens" with no
///   recurrence beyond that concatenation, so any prefix window can be read
///   back after the fact (`.window`).
/// - GDN's recurrent state genuinely is a step-by-step recurrence and needs
///   the per-token state stack produced by `gatedDeltaUpdateWithStates`
///   (PM4.1) to recover the state after an arbitrary prefix (`.stateAtIndex`).
///
/// QSA layers need no entry here: `Qwen4ExpQSAKVCache`'s backing arrays only
/// ever grow, and the visible length is tracked by `offset`
/// (`KVCacheSimple.trim` truncates `offset`, it never rewrites history), so
/// `trim(_:)` alone is exact — see `Qwen4ExpStreamingDecoder.rollbackVerification`.
public final class Qwen4ExpVerificationCapture: @unchecked Sendable {
    /// How to reconstruct one cache slot's value after `j` of the T new
    /// verified tokens have been committed (`j` counts tokens, 1-based: `j`
    /// tokens committed means indices `0..<j` of the newly fed sequence).
    public enum Entry {
        /// `cache[slot] = source[..., j ..< (j + length), ...]`.
        case window(source: MLXArray, length: Int)
        /// `cache[slot] = source[..., j - 1, ...]` (requires `j >= 1`).
        case stateAtIndex(source: MLXArray)
    }

    /// `[layerIndex: [cacheSlot: Entry]]`.
    public private(set) var entries: [Int: [Int: Entry]] = [:]

    public init() {}

    public func record(layerIndex: Int, slot: Int, entry: Entry) {
        entries[layerIndex, default: [:]][slot] = entry
    }
}

/// Binds a `Qwen4ExpVerificationCapture` to the one layer index currently
/// executing, so `Qwen4ExpGatedDeltaNet`/`Qwen4ExpPLELayer` do not need to
/// know their own position in the decoder stack.
public struct Qwen4ExpVerificationSink: Sendable {
    public let layerIndex: Int
    public let capture: Qwen4ExpVerificationCapture

    public init(layerIndex: Int, capture: Qwen4ExpVerificationCapture) {
        self.layerIndex = layerIndex
        self.capture = capture
    }

    public func record(slot: Int, entry: Qwen4ExpVerificationCapture.Entry) {
        capture.record(layerIndex: layerIndex, slot: slot, entry: entry)
    }
}
