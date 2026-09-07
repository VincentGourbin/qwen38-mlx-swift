import Foundation
import MLX
import Tokenizers

public enum Qwen4ExpPromptBuilderError: LocalizedError, Equatable {
    case missingImageTokenIDs

    public var errorDescription: String? {
        switch self {
        case .missingImageTokenIDs:
            return "La configuration Flash-Next ne fournit pas les identifiants de marqueurs image."
        }
    }
}

/// A rendered prompt ready for `Qwen4ExpStreamingTextModel.forward`.
public struct Qwen4ExpBuiltPrompt: @unchecked Sendable {
    public let tokenIDs: [Int32]
    public let positionIDs: MLXArray?
    public let visionEmbeddings: MLXArray?
    public let imageTokenID: Int32?

    public init(
        tokenIDs: [Int32], positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil, imageTokenID: Int32? = nil
    ) {
        self.tokenIDs = tokenIDs
        self.positionIDs = positionIDs
        self.visionEmbeddings = visionEmbeddings
        self.imageTokenID = imageTokenID
    }
}

/// Shared prompt/image rendering for the Flash-Next CLI probes and the
/// streaming generator (H2.4). Extracted from `FlashGenerateProbe` so the
/// rendering used by CLI, GUI and server stays a single source of truth.
public enum Qwen4ExpPromptBuilder {
    /// First-turn rendering: the full chat template for text-only prompts,
    /// or the manual ChatML assembly around the vision markers when an
    /// image is attached — identical to `flash-generate-probe`'s image
    /// branch, including the thinking prefix (V51).
    public static func buildFirstTurn(
        tokenizer: any Tokenizer,
        configuration: Qwen4ExpConfiguration,
        directory: URL,
        prompt: String,
        imageURL: URL?,
        thinking: Bool,
        reasoningEffort: String = "low",
        systemPrompt: String? = nil
    ) throws -> Qwen4ExpBuiltPrompt {
        guard let imageURL else {
            var messages: [Message] = []
            if let systemPrompt {
                messages.append(["role": "system", "content": systemPrompt])
            }
            messages.append(["role": "user", "content": prompt])
            let tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: [
                    "enable_thinking": thinking,
                    "reasoning_effort": reasoningEffort,
                ]
            ).map(Int32.init)
            return Qwen4ExpBuiltPrompt(tokenIDs: tokenIDs)
        }

        guard let imageToken = configuration.imageTokenID,
              let visionStart = configuration.visionStartTokenID,
              let visionEnd = configuration.visionEndTokenID else {
            throw Qwen4ExpPromptBuilderError.missingImageTokenIDs
        }
        let vision = try Qwen4ExpVisionCheckpointLoader.load(from: directory)
        let processed = try Qwen4ExpImageProcessor.load(
            from: imageURL,
            patchSize: configuration.visionConfiguration.patchSize,
            mergeSize: configuration.visionConfiguration.spatialMergeSize)
        let embeddings = vision.model(processed.pixels)
        eval(embeddings)

        func encode(_ value: String) -> [Int32] {
            tokenizer.encode(text: value, addSpecialTokens: false).map(Int32.init)
        }
        let markerCount = embeddings.dim(1)
        let systemPrefix = systemPrompt.map { encode("<|im_start|>system\n" + $0 + "\n<|im_end|>\n") } ?? []
        let tokenIDs =
            systemPrefix
            + encode("<|im_start|>user\n")
            + [visionStart]
            + Array(repeating: imageToken, count: markerCount)
            + [visionEnd]
            + encode(prompt + "\n<|im_end|>\n<|im_start|>assistant\n")
            + thinkingPrefix(thinking, encode: encode)
        let positionIDs = try Qwen4ExpMRoPE.multimodalPositionIDs(
            inputIDs: tokenIDs,
            imageTokenID: imageToken,
            visionStartTokenID: visionStart,
            grids: [
                Qwen4ExpVisionGrid(
                    height: processed.patchGrid.height, width: processed.patchGrid.width)
            ])
        return Qwen4ExpBuiltPrompt(
            tokenIDs: tokenIDs, positionIDs: positionIDs, visionEmbeddings: embeddings,
            imageTokenID: imageToken)
    }

    /// Full-history rendering for stateless callers (H3, the LAN server):
    /// the whole message list goes through the HF chat template in one
    /// call, exactly like `buildFirstTurn`'s text-only branch — the
    /// template itself knows how to render several turns. Text only: a
    /// stateless multi-turn request carrying an image has no single point
    /// in the manual ChatML assembly to attach it, so callers should reject
    /// that case explicitly rather than silently dropping the image.
    public static func buildFromMessages(
        tokenizer: any Tokenizer,
        messages: [Message],
        thinking: Bool,
        reasoningEffort: String = "low"
    ) throws -> Qwen4ExpBuiltPrompt {
        let tokenIDs = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: [
                "enable_thinking": thinking,
                "reasoning_effort": reasoningEffort,
            ]
        ).map(Int32.init)
        return Qwen4ExpBuiltPrompt(tokenIDs: tokenIDs)
    }

    /// Continuation-turn rendering (H2.3): only the new suffix is
    /// tokenized — no system prompt, no history replay — because the
    /// decoder's recurrent/QSA caches already carry the earlier turns.
    /// Follows the same ChatML convention validated in V51.
    public static func buildContinuationTurn(
        tokenizer: any Tokenizer,
        prompt: String,
        thinking: Bool
    ) -> Qwen4ExpBuiltPrompt {
        func encode(_ value: String) -> [Int32] {
            tokenizer.encode(text: value, addSpecialTokens: false).map(Int32.init)
        }
        let tokenIDs =
            encode("<|im_start|>user\n" + prompt + "\n<|im_end|>\n<|im_start|>assistant\n")
            + thinkingPrefix(thinking, encode: encode)
        return Qwen4ExpBuiltPrompt(tokenIDs: tokenIDs)
    }

    private static func thinkingPrefix(
        _ thinking: Bool, encode: (String) -> [Int32]
    ) -> [Int32] {
        thinking ? encode("<think>\n") : encode("<think>\n\n</think>\n\n")
    }
}
