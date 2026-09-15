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
    /// P13.2 : `tools` s'ajoute pour que le premier tour d'une conversation
    /// à cache persistant (`Qwen38FlashNextEngine.generate`, démarré à froid
    /// par `Qwen38Runtime.coldStartConversation`) rende exactement le même
    /// bloc d'outils que `buildFromMessages` — sans ce paramètre, le premier
    /// tour d'une boucle d'agent perdait silencieusement la déclaration des
    /// outils (le modèle ne savait même pas qu'ils existaient) dès que la
    /// requête empruntait ce chemin plutôt que le chemin stateless. Reste
    /// `nil` sur la branche image : le rendu manuel ChatML ci-dessous n'a
    /// aucune notion d'outils (limitation préexistante, non couverte ici).
    public static func buildFirstTurn(
        tokenizer: any Tokenizer,
        configuration: Qwen4ExpConfiguration,
        directory: URL,
        prompt: String,
        imageURL: URL?,
        thinking: Bool,
        reasoningEffort: String = "low",
        systemPrompt: String? = nil,
        tools: [ToolSpec]? = nil
    ) throws -> Qwen4ExpBuiltPrompt {
        guard let imageURL else {
            var messages: [Message] = []
            if let systemPrompt {
                messages.append(["role": "system", "content": systemPrompt])
            }
            messages.append(["role": "user", "content": prompt])
            let tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
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
        reasoningEffort: String = "low",
        tools: [ToolSpec]? = nil
    ) throws -> Qwen4ExpBuiltPrompt {
        let tokenIDs = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: [
                "enable_thinking": thinking,
                "reasoning_effort": reasoningEffort,
            ]
        ).map(Int32.init)
        return Qwen4ExpBuiltPrompt(tokenIDs: tokenIDs)
    }

    /// P13.1 : convertit un `Qwen38ChatMessage` transport-neutre vers le
    /// `Tokenizers.Message` attendu par `applyChatTemplate` — le point de
    /// conversion partagé par les trois sites d'appel stateless de
    /// `Qwen38FlashNextEngine` (`renderedTokenIDs`, `generateFromMessages`,
    /// `generateBatch`). Un tour assistant portant des `toolCalls` se rend
    /// exactement comme la forme `message.tool_calls[].function.
    /// {name,arguments}` du gabarit du checkpoint (chat_template.jinja) —
    /// `arguments` doit y être un objet JSON, jamais la chaîne JSON du
    /// format fil OpenAI (`Qwen38ToolCall.argumentsJSON`), parce que le
    /// gabarit boucle dessus avec Jinja `|items` ; d'où le nouveau parsing
    /// ci-dessous. Un `role: .tool` se rend en `{"role":"tool","content":…}`
    /// — le gabarit n'utilise jamais d'identifiant d'appel, seulement
    /// l'ordre des messages (voir `chat_template.jinja`, aucune référence à
    /// `tool_call_id`).
    public static func hfMessage(from message: Qwen38ChatMessage) -> Message {
        switch message.role {
        case .system, .user:
            return ["role": message.role.rawValue, "content": message.content]
        case .tool:
            return ["role": "tool", "content": message.content]
        case .assistant:
            var dict: Message = ["role": "assistant", "content": message.content]
            guard !message.toolCalls.isEmpty else { return dict }
            dict["tool_calls"] = message.toolCalls.map { call -> [String: any Sendable] in
                let arguments: any Sendable
                if case .object(let fields)? = try? Qwen38JSONValue.parse(call.argumentsJSON) {
                    arguments = fields.mapValues { $0.sendableValue }
                } else {
                    arguments = [String: any Sendable]()
                }
                return [
                    "id": call.id, "type": "function",
                    "function": ["name": call.name, "arguments": arguments],
                ]
            }
            return dict
        }
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

    /// P13.3 : le bloc de jetons que `applyChatTemplate(..., addGenerationPrompt: true)`
    /// ajoute après le dernier tour réel pour amorcer une réponse assistant
    /// — littéralement le même appel que `buildContinuationTurn` utilise
    /// déjà pour ce même bloc, exposé ici pour que
    /// `Qwen38FlashNextEngine.continueConversationTurn` (P13.3) puisse le
    /// retirer de la fin d'un rendu complet (`renderedTokenIDs`) et
    /// retrouver ainsi le contenu littéral déjà présent dans le cache — sans
    /// deviner le gabarit : c'est la brique déjà validée qui construit ce
    /// bloc, pas une nouvelle reconstruction.
    public static func assistantGenerationPromptTokenIDs(
        tokenizer: any Tokenizer, thinking: Bool
    ) -> [Int32] {
        func encode(_ value: String) -> [Int32] {
            tokenizer.encode(text: value, addSpecialTokens: false).map(Int32.init)
        }
        return encode("<|im_start|>assistant\n") + thinkingPrefix(thinking, encode: encode)
    }

    /// P13.3 : calcule le suffixe de jetons à préfiller pour continuer une
    /// conversation à cache persistant, par DIFFÉRENCE entre deux rendus
    /// complets — jamais en reconstruisant le tour à la main — afin de
    /// couvrir n'importe quel rôle de dernier message (`user` comme `tool`),
    /// contrairement à `buildContinuationTurn` qui encadre systématiquement
    /// le nouveau message comme un tour `user` nu. Voir le rapport à Vincent
    /// (PLAN.md P13.3) pour le contexte complet.
    ///
    /// `priorRenderedTokenIDs` est le rendu complet (`renderedTokenIDs`,
    /// donc avec `addGenerationPrompt: true`) des messages déjà dans le
    /// cache ; `fullRenderedTokenIDs` est le même rendu pour la
    /// conversation actuelle, un message de plus. Les deux se terminent par
    /// le même bloc d'amorçage (`generationPromptTokenIDs`, voir
    /// `assistantGenerationPromptTokenIDs`) puisque `applyChatTemplate`
    /// l'ajoute inconditionnellement — le retirer de la fin de
    /// `priorRenderedTokenIDs` retrouve donc exactement le contenu littéral
    /// déjà écrit dans le cache du moteur.
    ///
    /// Retourne `nil` — jamais un état incohérent — dans chacun des cas
    /// dégradés :
    ///  - `priorRenderedTokenIDs` ne se termine pas par le bloc d'amorçage
    ///    attendu (garde de cohérence : ne devrait jamais arriver si l'appelant
    ///    a bien rendu les deux côtés avec les mêmes options) ;
    ///  - le contenu littéral qui en résulte n'est PAS un préfixe exact de
    ///    `fullRenderedTokenIDs` (historique divergent : un message plus tôt
    ///    dans la conversation a été édité, tronqué, ou la liste d'outils a
    ///    changé) ;
    ///  - rien de nouveau à préfiller au-delà du bloc d'amorçage lui-même
    ///    (le rendu complet actuel est identique au rendu déjà en cache).
    /// Dans les trois cas, l'appelant doit retomber sur un rejeu complet.
    public static func continuationSuffix(
        priorRenderedTokenIDs: [Int32],
        fullRenderedTokenIDs: [Int32],
        generationPromptTokenIDs: [Int32]
    ) -> [Int32]? {
        guard priorRenderedTokenIDs.count >= generationPromptTokenIDs.count,
              Array(priorRenderedTokenIDs.suffix(generationPromptTokenIDs.count))
                == generationPromptTokenIDs
        else { return nil }
        let cachedLength = priorRenderedTokenIDs.count - generationPromptTokenIDs.count
        // `fullRenderedTokenIDs` ends in the SAME priming block (both sides
        // are rendered with `addGenerationPrompt: true`, and the compared
        // conversations share `options.enableThinking` — `cacheOptionsCompatible`
        // guarantees that upstream) — so a genuine "nothing new" case (the
        // full render is identical to the prior one) would otherwise slip
        // through as a one-token-shorter-than-expected "suffix" containing
        // only the priming block itself, never actually empty. The `>`
        // below (rather than `>=`) rejects that case explicitly: a real
        // continuation must add at least one token of new message content
        // beyond the priming block.
        guard fullRenderedTokenIDs.count > cachedLength + generationPromptTokenIDs.count,
              Array(fullRenderedTokenIDs.prefix(cachedLength))
                == Array(priorRenderedTokenIDs.prefix(cachedLength))
        else { return nil }
        return Array(fullRenderedTokenIDs[cachedLength...])
    }
}
