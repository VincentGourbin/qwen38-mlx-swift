import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

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
final class Qwen38DenseConversationCache: @unchecked Sendable {
    fileprivate var cache: [KVCache]?
    fileprivate var state: LMOutput.State?
    /// Positional state at the snapshot point: carries the M-RoPE delta that
    /// earlier images introduced, which every later token is anchored on.
    fileprivate var snapshotState: LMOutput.State?
    fileprivate var prefixTokens: [Int] = []
    fileprivate var recurrentSnapshot: [(index: Int, arrays: [MLXArray], offset: Int)] = []
    fileprivate var kvBits: Int?

    func reset() {
        cache = nil
        state = nil
        snapshotState = nil
        prefixTokens = []
        recurrentSnapshot = []
    }
}

struct Qwen38DenseRequest {
    let tokens: [Int]
    /// Pixels of every image in the conversation, in order, as the model's
    /// processor produced them; `nil` for a text-only conversation.
    let image: LMInput.ProcessedImage?
    /// `<|vision_start|>`: one per image, used to find which images a token
    /// range contains.
    let visionStartTokenID: Int?
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let topK: Int
    let kvBits: Int?
    /// Tokens kept in full precision before the KV cache is quantized to
    /// `kvBits` (the runtime's `GenerateParameters.quantizedKVStart`).
    let quantizedKVStart: Int
    let prefillStepSize: Int
    let stopTokenIDs: Set<Int>
    let conversationStartTokenID: Int?
}

struct Qwen38DenseResult: Sendable {
    let promptTokens: Int
    let cachedPromptTokens: Int
    let completionTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let timeToFirstToken: TimeInterval?
    let hitLength: Bool
    /// Wall time between consecutive decoded tokens, in seconds.
    let stepDurations: [Double]
}

enum Qwen38DenseEngineError: LocalizedError {
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

enum Qwen38DenseEngine {
    /// Runs one answer synchronously (inside `ModelContainer.perform`),
    /// reusing `conversation` when possible. `emit` receives decoded text
    /// pieces; returning `false` stops generation (cancellation).
    static func run(
        _ request: Qwen38DenseRequest, model: any LanguageModel, tokenizer: any Tokenizer,
        conversation: Qwen38DenseConversationCache, emit: (String) -> Bool
    ) throws -> Qwen38DenseResult {
        let tokens = request.tokens
        guard !tokens.isEmpty else { throw Qwen38DenseEngineError.emptyPrompt }
        let started = Date()
        // Phases land in the active profiling session, if any (`qwen38 brain
        // … --trace`); a disabled profiler makes these calls no-ops.
        let session = MLXProfiler.shared.isEnabled ? MLXProfiler.shared.activeSession : nil

        // Resume point of this request: just before the last `<|im_start|>`
        // (the generation prompt), or the whole prompt minus its last token.
        let lastStart = request.conversationStartTokenID.flatMap { id in
            tokens.lastIndex(of: id)
        }
        let snapshotLength = min(lastStart ?? (tokens.count - 1), tokens.count - 1)
        // (the generation prompt, from the last `<|im_start|>`, is never empty)

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
                    throw Qwen38DenseEngineError.cacheOffsetMismatch(
                        layer: index, offset: layer.offset, expected: start)
                }
            }
            conversation.state = conversation.snapshotState
        } else {
            conversation.reset()
            conversation.cache = try model.newCache(parameters: nil)
            conversation.kvBits = request.kvBits
        }
        // Same rule as the runtime's iterator: full precision until
        // `quantizedKVStart` tokens, then `kvBits` (the quantized layers
        // replace the plain ones in place, trimming still works).
        func quantizeIfDue() {
            guard request.kvBits != nil, var layers = conversation.cache else { return }
            maybeQuantizeKVCache(
                cache: &layers, kvBits: request.kvBits, kvGroupSize: 64,
                quantizedKVStart: request.quantizedKVStart)
            conversation.cache = layers
        }
        var cache: [KVCache] { conversation.cache! }

        func forward(_ input: MLXArray) -> LMOutput {
            let output = model(LMInput.Text(tokens: input), cache: cache, state: conversation.state)
            conversation.state = output.state ?? conversation.state
            return output
        }

        // Images whose `<|vision_start|>` falls inside `range`, sliced out of
        // the conversation's pixels (rows are t·h·w patches per image). Range
        // bounds are message boundaries, so an image is never split.
        func images(in range: Range<Int>) -> LMInput.ProcessedImage? {
            guard let image = request.image, let frames = image.frames,
                let visionStart = request.visionStartTokenID
            else { return nil }
            let before = tokens[..<range.lowerBound].filter { $0 == visionStart }.count
            let inside = tokens[range].filter { $0 == visionStart }.count
            guard inside > 0 else { return nil }
            let rows = frames.map { $0.t * $0.h * $0.w }
            let firstRow = rows[..<before].reduce(0, +)
            let rowCount = rows[before ..< before + inside].reduce(0, +)
            return LMInput.ProcessedImage(
                pixels: image.pixels[firstRow ..< firstRow + rowCount],
                frames: Array(frames[before ..< before + inside]))
        }

        // The model's own prepare: vision tower on the new images only, M-RoPE
        // positions anchored at the cache offset, chunked prefill. Returns the
        // logits of the range's last position.
        func prefill(_ range: Range<Int>) throws -> MLXArray? {
            guard !range.isEmpty else { return nil }
            let text = LMInput.Text(
                tokens: MLXArray(tokens[range].map(Int32.init)).reshaped(1, range.count))
            let input = LMInput(text: text, image: images(in: range))
            switch try model.prepare(
                input, cache: cache, state: conversation.state,
                prefill: PrefillParameters(stepSize: request.prefillStepSize))
            {
            case .logits(let output):
                conversation.state = output.state ?? conversation.state
                eval(cache.flatMap { $0.innerState() })
                quantizeIfDue()
                return output.logits
            case .tokens(let remaining):
                // Text-only remainder left to the caller (not the Qwen 3.5 path).
                return forward(remaining.tokens.ndim == 1
                    ? remaining.tokens.reshaped(1, -1) : remaining.tokens).logits
            }
        }

        // 1. history up to the resume point, then snapshot it;
        // 2. the generation prompt, whose last logits give the first token.
        session?.beginPhase("Préfill historique", category: .prefill)
        _ = try prefill(start ..< snapshotLength)
        session?.endPhase("Préfill historique", category: .prefill)
        conversation.prefixTokens = Array(tokens[..<snapshotLength])
        conversation.snapshotState = conversation.state
        conversation.recurrentSnapshot = cache.enumerated().compactMap { index, layer in
            layer is ArraysCache ? (index, layer.state, layer.offset) : nil
        }
        session?.beginPhase("Préfill invite", category: .prefill)
        let promptLogitsOrNil = try prefill(snapshotLength ..< tokens.count)
        session?.endPhase("Préfill invite", category: .prefill)
        guard let promptLogits = promptLogitsOrNil else {
            throw Qwen38DenseEngineError.emptyPrompt
        }
        let prefillSeconds = Date().timeIntervalSince(started)

        let parameters = GenerateParameters(
            temperature: request.temperature, topP: request.topP, topK: request.topK)
        let sampler = parameters.sampler()
        // Lazy all the way: the next step is queued (`asyncEval`) before the
        // current token is read back, so the GPU never waits on the host.
        func step(_ token: MLXArray) -> MLXArray {
            let output = forward(token.reshaped(1, 1))
            quantizeIfDue()
            return sampler.sample(logits: output.logits[0..., -1, 0...])
        }

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var generated = 0
        var timeToFirstToken: TimeInterval?
        var hitLength = true
        let decodeStarted = Date()
        var stepDurations: [Double] = []
        var lastStep = decodeStarted
        session?.beginPhase("Décodage", category: .generation)
        defer { session?.endPhase("Décodage", category: .generation) }
        var current = sampler.sample(logits: promptLogits[0..., -1, 0...])
        asyncEval(current)
        while generated < request.maxTokens {
            let next = generated + 1 < request.maxTokens ? step(current) : nil
            if let next { asyncEval(next) }
            let token = current.item(Int.self)
            let now = Date()
            stepDurations.append(now.timeIntervalSince(lastStep))
            lastStep = now
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
        return Qwen38DenseResult(
            promptTokens: tokens.count - start, cachedPromptTokens: start,
            completionTokens: generated, prefillSeconds: prefillSeconds,
            decodeSeconds: Date().timeIntervalSince(decodeStarted),
            timeToFirstToken: timeToFirstToken, hitLength: hitLength,
            stepDurations: stepDurations)
    }
}
