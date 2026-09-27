import Foundation
import MLX
import MLXLMCommon
import Qwen38Core

/// Conversation cache for the dense hybrid family (Qwen 3.5 / Bonsai 2):
/// 16 full-attention layers whose KV can be trimmed, and 48 GatedDeltaNet
/// layers whose recurrent state cannot. So the state is snapshotted at one
/// point — the end of the last message, just before the generation prompt —
/// and the next request resumes from there when its rendered tokens start
/// with the same prefix, which an append-only chat history always does.
///
/// Why that point: the chat template re-renders earlier assistant turns
/// (reasoning dropped, tool calls re-serialised), so what the model generated
/// is not a reliable prefix of the next request; the history up to the last
/// `<|im_end|>` is.
final class Qwen38BrainConversationCache: @unchecked Sendable {
    fileprivate var cache: [KVCache]?
    fileprivate var state: LMOutput.State?
    fileprivate var prefixTokens: [Int] = []
    fileprivate var recurrentSnapshot: [(index: Int, arrays: [MLXArray], offset: Int)] = []
    fileprivate var kvBits: Int?

    func reset() {
        cache = nil
        state = nil
        prefixTokens = []
        recurrentSnapshot = []
    }
}

struct Qwen38BrainDenseRequest: Sendable {
    let tokens: [Int]
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let topK: Int
    let kvBits: Int?
    let prefillStepSize: Int
    let stopTokenIDs: Set<Int>
    let conversationStartTokenID: Int?
}

struct Qwen38BrainDenseResult: Sendable {
    let promptTokens: Int
    let cachedPromptTokens: Int
    let completionTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let timeToFirstToken: TimeInterval?
    let hitLength: Bool
}

enum Qwen38BrainDenseEngineError: LocalizedError {
    case emptyPrompt
    case cacheOffsetMismatch(layer: Int, offset: Int, expected: Int)

    var errorDescription: String? {
        switch self {
        case .emptyPrompt: return "prompt vide"
        case .cacheOffsetMismatch(let layer, let offset, let expected):
            return "cache incohérent : couche \(layer) à l'offset \(offset), \(expected) attendu"
        }
    }
}

enum Qwen38BrainDenseEngine {
    /// Runs one answer synchronously (inside `ModelContainer.perform`),
    /// reusing `conversation` when possible. `emit` receives decoded text
    /// pieces; returning `false` stops generation (cancellation).
    static func run(
        _ request: Qwen38BrainDenseRequest, model: any LanguageModel, tokenizer: any Tokenizer,
        conversation: Qwen38BrainConversationCache, emit: (String) -> Bool
    ) throws -> Qwen38BrainDenseResult {
        let tokens = request.tokens
        guard !tokens.isEmpty else { throw Qwen38BrainDenseEngineError.emptyPrompt }
        let started = Date()

        // Resume point of this request: just before the last `<|im_start|>`
        // (the generation prompt), or the whole prompt minus its last token.
        let lastStart = request.conversationStartTokenID.flatMap { id in
            tokens.lastIndex(of: id)
        }
        let snapshotLength = min(lastStart ?? (tokens.count - 1), tokens.count - 1)

        var start = 0
        if let cache = conversation.cache, conversation.kvBits == request.kvBits,
            !conversation.prefixTokens.isEmpty,
            conversation.prefixTokens.count <= snapshotLength,
            tokens.starts(with: conversation.prefixTokens)
        {
            start = conversation.prefixTokens.count
            for entry in conversation.recurrentSnapshot {
                guard let layer = cache[entry.index] as? ArraysCache else { continue }
                layer.state = entry.arrays
                layer.offset = entry.offset
            }
            for (index, layer) in cache.enumerated() where !(layer is ArraysCache) {
                if layer.offset > start { layer.trim(layer.offset - start) }
                guard layer.offset == start else {
                    conversation.reset()
                    throw Qwen38BrainDenseEngineError.cacheOffsetMismatch(
                        layer: index, offset: layer.offset, expected: start)
                }
            }
        } else {
            conversation.reset()
            var cache = try model.newCache(parameters: nil)
            if let bits = request.kvBits {
                cache = cache.map { layer in
                    layer is ArraysCache ? layer : QuantizedKVCache(groupSize: 64, bits: bits)
                }
            }
            conversation.cache = cache
            conversation.kvBits = request.kvBits
        }
        let cache = conversation.cache!

        func forward(_ input: MLXArray) -> LMOutput {
            let output = model(LMInput.Text(tokens: input), cache: cache, state: conversation.state)
            conversation.state = output.state ?? conversation.state
            return output
        }
        func forward(_ slice: ArraySlice<Int>) -> LMOutput {
            forward(MLXArray(slice.map(Int32.init)).reshaped(1, slice.count))
        }
        func prefill(_ range: Range<Int>) {
            var position = range.lowerBound
            while position < range.upperBound {
                let end = min(position + request.prefillStepSize, range.upperBound)
                _ = forward(tokens[position ..< end])
                eval(cache.flatMap { $0.innerState() })
                position = end
            }
        }

        // 1. history up to the resume point, then snapshot it;
        // 2. the generation prompt except its last token;
        // 3. decode from the last prompt token.
        prefill(start ..< snapshotLength)
        conversation.prefixTokens = Array(tokens[..<snapshotLength])
        conversation.recurrentSnapshot = cache.enumerated().compactMap { index, layer in
            layer is ArraysCache ? (index, layer.state, layer.offset) : nil
        }
        prefill(snapshotLength ..< tokens.count - 1)
        let prefillSeconds = Date().timeIntervalSince(started)

        let parameters = GenerateParameters(
            temperature: request.temperature, topP: request.topP, topK: request.topK)
        let sampler = parameters.sampler()
        // Lazy all the way: the next step is queued (`asyncEval`) before the
        // current token is read back, so the GPU never waits on the host.
        func step(_ token: MLXArray) -> MLXArray {
            let output = forward(token.reshaped(1, 1))
            return sampler.sample(logits: output.logits[0..., -1, 0...])
        }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var generated = 0
        var timeToFirstToken: TimeInterval?
        var hitLength = true
        let decodeStarted = Date()
        var current = step(MLXArray([Int32(tokens[tokens.count - 1])]))
        asyncEval(current)
        while generated < request.maxTokens {
            let next = generated + 1 < request.maxTokens ? step(current) : nil
            if let next { asyncEval(next) }
            let token = current.item(Int.self)
            if request.stopTokenIDs.contains(token) {
                hitLength = false
                break
            }
            generated += 1
            if let next { current = next }
            detokenizer.append(token: token)
            if let piece = detokenizer.next(), !piece.isEmpty {
                if timeToFirstToken == nil { timeToFirstToken = Date().timeIntervalSince(started) }
                guard emit(piece) else {
                    hitLength = false
                    break
                }
            }
        }
        return Qwen38BrainDenseResult(
            promptTokens: tokens.count - start, cachedPromptTokens: start,
            completionTokens: generated, prefillSeconds: prefillSeconds,
            decodeSeconds: Date().timeIntervalSince(decodeStarted),
            timeToFirstToken: timeToFirstToken, hitLength: hitLength)
    }
}
