import Foundation
import MLXLMCommon

/// Keeps ChatML control tokens out of the UI while leaving the token ledger
/// used by the target and drafter untouched.
public enum Qwen38VisibleText {
    private static let structuralTokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>"
    ]

    public static func structuralTokenIDs(
        tokenizer: any Tokenizer
    ) -> Set<Int> {
        structuralTokenIDs(convertTokenToId: tokenizer.convertTokenToId)
    }

    /// Tokenizer-agnostic variant: Flash-Next builds its prompts through
    /// `Tokenizers.Tokenizer` (swift-transformers) rather than
    /// `MLXLMCommon.Tokenizer` — both expose `convertTokenToId`, so the
    /// structural-token contract stays the single source of truth here.
    public static func structuralTokenIDs(
        convertTokenToId: (String) -> Int?
    ) -> Set<Int> {
        Set(structuralTokens.compactMap(convertTokenToId))
    }

    /// Defensive cleanup for already decoded chunks, including output from
    /// the upstream iterator. Thinking text and `<think>` markers are kept.
    public static func sanitize(_ text: String) -> String {
        let rolePrefixes = [
            "<|im_start|>assistant\n",
            "<|im_start|>user\n",
            "<|im_start|>system\n",
            "<|im_start|>tool\n"
        ]
        let withRolesRemoved = rolePrefixes.reduce(text) { partial, prefix in
            partial.replacingOccurrences(of: prefix, with: "")
        }
        return structuralTokens.reduce(withRolesRemoved) { partial, token in
            partial.replacingOccurrences(of: token, with: "")
        }
    }
}

public struct Qwen38VisibleTokenFilter: Sendable {
    private let structuralIDs: Set<Int>

    public init(tokenizer: any Tokenizer) {
        self.structuralIDs = Qwen38VisibleText.structuralTokenIDs(tokenizer: tokenizer)
    }

    public init(convertTokenToId: (String) -> Int?) {
        self.structuralIDs = Qwen38VisibleText.structuralTokenIDs(convertTokenToId: convertTokenToId)
    }

    public func shouldEmit(_ tokenID: Int) -> Bool {
        !structuralIDs.contains(tokenID)
    }
}

/// Splits Qwen's interleaved thinking stream into the two fields expected by
/// OpenAI-compatible clients.  The generator is token streamed, so either
/// marker can be split over multiple chunks; `pending` keeps only the small
/// suffix that could still become a marker.
public struct Qwen38ThinkingStreamParser: Sendable {
    public struct Output: Sendable, Equatable {
        public let reasoning: String
        public let content: String

        public init(reasoning: String = "", content: String = "") {
            self.reasoning = reasoning
            self.content = content
        }
    }

    private var emitter: ReasoningEventEmitter

    /// Qwen's thinking-enabled template already puts `<think>` in the
    /// generation prompt, so the normal stream starts *inside* reasoning and
    /// only emits `</think>`.  Callers that feed a raw stream containing the
    /// opening marker can pass `primedInside: false`.
    public init(primedInside: Bool = true) {
        emitter = ReasoningEventEmitter(
            config: .thinkTagsWithEnableThinking,
            primedInside: primedInside)
    }

    public mutating func append(_ chunk: String) -> Output {
        Self.output(from: emitter.process(chunk))
    }

    public mutating func finish() -> Output {
        Self.output(from: emitter.finalize())
    }

    private static func output(from segments: [ReasoningEventEmitter.Segment]) -> Output {
        var reasoning = ""
        var content = ""
        for segment in segments {
            switch segment {
            case .reasoning(let value): reasoning += value
            case .response(let value): content += value
            }
        }
        return Output(reasoning: reasoning, content: content)
    }
}
