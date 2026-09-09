import Foundation
import MLX

/// Key-level normalization shared by the future Flash-Next loader.
///
/// The function is intentionally pure and does not touch tensor data. This
/// makes it safe to run while inspecting a safetensors index and avoids
/// materializing the n-gram table before the model topology is available.
public enum Qwen4ExpWeightSanitizer {
    /// The Vontra Flash-Next conversion applied the legacy `+1` checkpoint
    /// shift to the zero-centered RMSNorm weights.  The model primitive still
    /// evaluates `1 + weight`, so converted checkpoints must be shifted back
    /// once, at the loading boundary.  GDN's `linear_attn.norm` is a regular
    /// gated norm and is deliberately not part of this list.
    private static let shiftedNormSuffixes = [
        "hc_norm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
        // MTP head (2026-09-09, verified against the official HF BF16 shards):
        // both pre-fc norms are zero-centered too and shifted by exactly +1.000
        // in the Vontra checkpoint (HF means -0.764 / -0.328).
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ]

    /// Returns the MLX tree key for a Transformers or converted checkpoint
    /// key. `nil` means that the tensor is a derived RoPE buffer and must not
    /// be loaded as a parameter.
    public static func normalize(_ key: String) -> String? {
        var normalized = key
        if normalized.hasPrefix("model.") {
            normalized.removeFirst("model.".count)
        }

        if normalized.hasPrefix("language_model."),
           !normalized.hasPrefix("language_model.model.") {
            normalized = normalized.replacingOccurrences(
                of: "language_model.", with: "language_model.model.", options: [], range: normalized.startIndex ..< normalized.index(normalized.startIndex, offsetBy: "language_model.".count))
        }

        // The converted checkpoint names dynamic shards as `shard_0`, while
        // MLXNN materializes a Swift array as `shards.0`. Keep this remap
        // local to the n-gram table so ordinary model keys remain untouched.
        normalized = normalized.replacingOccurrences(
            of: ".ngram_embedding.shard_", with: ".ngram_embedding.shards_")
        if let marker = normalized.range(of: ".ngram_embedding.shards_") {
            let suffix = normalized[marker.upperBound...]
            if let separator = suffix.firstIndex(of: "."),
               suffix[..<separator].allSatisfy({ $0.isNumber }) {
                let number = suffix[..<separator]
                let numberEnd = normalized.index(marker.upperBound, offsetBy: number.count)
                normalized.replaceSubrange(
                    marker.lowerBound ..< numberEnd,
                    with: ".ngram_embedding.shards.\(number)")
            }
        }

        if normalized.contains("rotary_emb") || normalized.hasSuffix(".rope.freqs") {
            return nil
        }
        return normalized.isEmpty ? nil : normalized
    }

    /// Returns whether a normalized language-model key is one of the
    /// zero-centered norms covered by the converted-checkpoint correction.
    public static func isShiftedZeroCenteredNormKey(_ key: String) -> Bool {
        shiftedNormSuffixes.contains { key.hasSuffix($0) }
    }

    /// Correct the Vontra checkpoint convention without changing the RMSNorm
    /// module's normal `1 + weight` semantics.  The correction is enabled only
    /// when an anchor norm has the characteristic shifted mean (approximately
    /// `0.94` instead of the official zero-centered mean near `-0.06`).
    ///
    /// This keeps synthetic fixtures and already-correct checkpoints intact.
    /// Evaluating one small anchor is intentional: it is independent of the
    /// large n-gram table and does not alter the memory-bounded load path.
    public static func correctShiftedZeroCenteredNormWeights(
        _ weights: [String: MLXArray]
    ) -> (weights: [String: MLXArray], applied: Bool) {
        guard let anchorKey = weights.keys.sorted().first(where: {
            $0.hasSuffix("hc_norm.weight")
        }), let anchor = weights[anchorKey] else {
            return (weights, false)
        }

        let anchorMean = anchor.asType(.float32).mean().item(Float.self)
        guard anchorMean > 0.5 else {
            return (weights, false)
        }

        let offset = MLXArray(Float(1.0)).asType(anchor.dtype)
        var corrected = weights
        for key in weights.keys where isShiftedZeroCenteredNormKey(key) {
            guard let value = weights[key] else { continue }
            corrected[key] = (value - offset).asType(value.dtype)
        }
        return (corrected, true)
    }

    /// Checks the naming contract of a converted Flash-Next index without
    /// loading any safetensors shard.
    public static func validateIndexKeys<S: Sequence>(_ keys: S) throws
    where S.Element == String {
        let values = Array(keys)
        guard values.contains("language_model.model.embed_tokens.weight") else {
            throw Qwen4ExpWeightSanitizerError.missingRequiredKey(
                "language_model.model.embed_tokens.weight")
        }
        guard values.contains(where: { $0.contains("linear_attn.in_proj_qkv") }) else {
            throw Qwen4ExpWeightSanitizerError.missingRequiredKey("linear_attn.in_proj_qkv")
        }
        guard values.contains(where: { $0.contains("hyper_connection_mixer") }) else {
            throw Qwen4ExpWeightSanitizerError.missingRequiredKey("hyper_connection_mixer")
        }
        guard values.contains(where: { $0.contains("mtp") }) else {
            throw Qwen4ExpWeightSanitizerError.missingRequiredKey("mtp")
        }
    }
}

public enum Qwen4ExpWeightSanitizerError: LocalizedError, Equatable {
    case missingRequiredKey(String)

    public var errorDescription: String? {
        switch self {
        case .missingRequiredKey(let key):
            return "Index Flash-Next incomplet : clé requise absente (\(key))."
        }
    }
}
