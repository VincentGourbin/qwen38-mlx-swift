import Foundation
import MLX
import MLXLMCommon

/// Per-conversation state for the Flash-Next MTP predictor.
///
/// The target owns its own hybrid caches; this object owns only the private
/// one-layer QSA cache used by `mtp.layers.0`.  Keeping the two states separate
/// is what makes a rejected target suffix reversible without replaying the
/// whole conversation through the drafter.
public final class Qwen4ExpFlashMTPState: @unchecked Sendable {
    public fileprivate(set) var cache: Qwen4ExpQSAKVCache
    public fileprivate(set) var nextPosition: Int = 0
    public fileprivate(set) var seedToken: MLXArray?
    public fileprivate(set) var seedHidden: MLXArray?
    public fileprivate(set) var proposalAppended: Int = 0

    public init(cache: Qwen4ExpQSAKVCache) {
        self.cache = cache
    }

    public func reset() {
        cache = Qwen4ExpQSAKVCache(
            blockSize: cache.blockSize,
            budget: cache.budget,
            compressRatio: cache.compressRatio)
        nextPosition = 0
        seedToken = nil
        seedHidden = nil
        proposalAppended = 0
    }
}

public enum Qwen4ExpFlashMTPEngineError: LocalizedError, Equatable {
    case emptyPrompt
    case invalidPromptHidden
    case invalidBlockSize
    case invalidAcceptance
    case invalidVerificationHidden

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            return "Le prompt MTP Flash-Next ne peut pas être vide."
        case .invalidPromptHidden:
            return "Les hidden states du prompt doivent être [1,L,4H]."
        case .invalidBlockSize:
            return "La taille d'un round MTP doit être au moins 2."
        case .invalidAcceptance:
            return "Le nombre de tokens acceptés est hors du bloc proposé."
        case .invalidVerificationHidden:
            return "Les hidden states de vérification doivent couvrir la frontière commit."
        }
    }
}

/// Correctness-first local MTP seam for qwen4_exp.
///
/// This type intentionally does not conform to `StatefulMTPDrafterModel` yet:
/// that protocol is tied to `LanguageModel`, while the Flash target is still a
/// streamed layer executor.  It implements the same state machine locally so
/// its draft/commit behavior can be validated before an adapter is introduced
/// into the public ChatSession path.
public final class Qwen4ExpFlashMTPDraftEngine: @unchecked Sendable {
    public let target: Qwen4ExpStreamingTextModel
    public let predictor: Qwen4ExpMTPPredictor

    public init(target: Qwen4ExpStreamingTextModel, predictor: Qwen4ExpMTPPredictor) {
        self.target = target
        self.predictor = predictor
    }

    public func makeState() -> Qwen4ExpFlashMTPState {
        Qwen4ExpFlashMTPState(cache: predictor.makeCache())
    }

    /// Prefill the private MTP cache from the target's prompt hidden states.
    /// `firstBonus` is the token sampled from the target prompt logits and is
    /// therefore the last input of the shifted MTP prefill. Resets `state`
    /// first: use this for a fresh conversation (turn 1, or any turn where
    /// the target's own cache was also reset).
    public func prepare(
        promptTokenIDs: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionIDs: MLXArray? = nil,
        state: Qwen4ExpFlashMTPState
    ) throws {
        try primePredictor(
            promptTokenIDs: promptTokenIDs, targetHidden: targetHidden, firstBonus: firstBonus,
            positionIDs: positionIDs, state: state, resetState: true)
    }

    /// PM4.3 (branchement, 2026-09-09): continuation variant of `prepare`
    /// for a turn where the target model's own cache was *not* reset
    /// (`Qwen4ExpStreamingGenerator`'s `continueConversation` contract).
    /// Identical priming pass over this turn's prompt suffix, but the
    /// predictor's existing cache/history survives instead of being
    /// recreated — otherwise every new turn would silently discard the
    /// drafter's memory of the conversation while the target kept its own.
    /// `positionIDs` is already absolute (`Qwen4ExpMRoPE.textPositionIDs`
    /// continues from the target's ongoing `logicalOffset`), so no
    /// state-relative position adjustment is needed beyond skipping the
    /// reset itself.
    public func prepareContinuation(
        promptTokenIDs: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionIDs: MLXArray? = nil,
        state: Qwen4ExpFlashMTPState
    ) throws {
        try primePredictor(
            promptTokenIDs: promptTokenIDs, targetHidden: targetHidden, firstBonus: firstBonus,
            positionIDs: positionIDs, state: state, resetState: false)
    }

    private func primePredictor(
        promptTokenIDs: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionIDs: MLXArray?,
        state: Qwen4ExpFlashMTPState,
        resetState: Bool
    ) throws {
        let prompt = promptTokenIDs.ndim == 1
            ? promptTokenIDs.reshaped([1, promptTokenIDs.dim(0)])
            : promptTokenIDs
        guard prompt.ndim == 2, prompt.dim(0) == 1, prompt.dim(1) > 0,
              targetHidden.ndim == 3,
              targetHidden.shape == [1, prompt.dim(1), target.configuration.hiddenSize * target.configuration.hcCount]
        else { throw Qwen4ExpFlashMTPEngineError.invalidPromptHidden }

        let bonus = firstBonus.ndim == 1
            ? firstBonus.reshaped([1, firstBonus.dim(0)])
            : firstBonus
        guard bonus.shape == [1, 1] else {
            throw Qwen4ExpFlashMTPEngineError.invalidPromptHidden
        }

        if resetState {
            state.reset()
        }
        let shifted = concatenated([prompt[0..., 1...], bonus], axis: 1)
        let shiftedHidden = targetHidden[0..., 0..<shifted.dim(1), 0...]
        let shiftedPositions = shiftedPositionIDs(
            positionIDs, promptLength: prompt.dim(1), state: state)
        let output = predictor(
            inputEmbeddings: target.global.embed(shifted),
            targetHidden: shiftedHidden,
            inputIDs: shifted,
            cache: state.cache,
            positionIDs: shiftedPositions)
        eval(output)
        state.nextPosition = state.cache.offset
        state.seedHidden = output[0..., (-1)..., 0...]
        state.seedToken = greedyToken(
            logits: target.global.logits(
                from: predictor.logitsHidden(from: state.seedHidden!)))
    }

    /// Produce `blockSize - 1` draft tokens.  A prepared seed is returned
    /// without advancing the private cache; asking for a larger block feeds
    /// that seed back once with its already-computed hidden state, matching
    /// the upstream Qwen MTP state machine.
    public func draftBlock(
        lastToken: MLXArray,
        lastHidden: MLXArray,
        blockSize: Int,
        state: Qwen4ExpFlashMTPState
    ) throws -> MLXArray {
        guard blockSize >= 2 else { throw Qwen4ExpFlashMTPEngineError.invalidBlockSize }
        var token = normalizedColumn(lastToken)
        var hidden = lastHidden.ndim == 2
            ? lastHidden.expandedDimensions(axis: 0)
            : lastHidden
        var drafts = [MLXArray]()
        drafts.reserveCapacity(blockSize - 1)
        state.proposalAppended = 0

        if let seed = state.seedToken, let seedHidden = state.seedHidden {
            token = normalizedColumn(seed)
            hidden = seedHidden
            drafts.append(token)
            state.seedToken = nil
            state.seedHidden = nil
        }

        while drafts.count < blockSize - 1 {
            let output = try advance(
                token: token, hidden: hidden, state: state)
            hidden = output
            token = greedyToken(
                logits: target.global.logits(
                    from: predictor.logitsHidden(from: output)))
            drafts.append(token)
            state.proposalAppended += 1
        }
        return concatenated(drafts, axis: 1)
    }

    /// Reconcile tentative private-cache writes after target verification.
    /// `targetHidden` must be the hidden rows for the replayed committed input
    /// `[bonus, acceptedDrafts]`; it has `acceptedCount + 1` rows and gives the
    /// predictor the target hidden associated with the correction token.
    public func commit(
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        state: Qwen4ExpFlashMTPState
    ) throws {
        let drafts = draftTokens.ndim == 1
            ? draftTokens.reshaped([1, draftTokens.dim(0)])
            : draftTokens
        guard drafts.ndim == 2, drafts.dim(0) == 1,
              acceptedCount >= 0, acceptedCount <= drafts.dim(1),
              acceptedCount <= state.proposalAppended + 1
        else { throw Qwen4ExpFlashMTPEngineError.invalidAcceptance }

        let expectedRows = acceptedCount + 1
        guard targetHidden.ndim == 3,
              targetHidden.dim(0) == 1,
              targetHidden.dim(1) >= expectedRows,
              targetHidden.dim(2) == target.configuration.hiddenSize * target.configuration.hcCount
        else { throw Qwen4ExpFlashMTPEngineError.invalidVerificationHidden }

        let keepAppended = min(acceptedCount, state.proposalAppended)
        let trim = state.proposalAppended - keepAppended
        if trim > 0 {
            _ = state.cache.trim(trim)
        }
        state.nextPosition = state.cache.offset

        var tokens = [MLXArray]()
        var hiddens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(drafts[0..., index ..< index + 1])
            hiddens.append(targetHidden[0..., index ..< index + 1, 0...])
        }
        tokens.append(normalizedColumn(finalToken))
        hiddens.append(targetHidden[0..., acceptedCount ..< acceptedCount + 1, 0...])

        var lastOutput: MLXArray?
        for (token, hidden) in zip(tokens, hiddens) {
            lastOutput = try advance(token: token, hidden: hidden, state: state)
        }
        if let lastOutput {
            // The last predictor output is the seed for the next round.  It
            // is computed from the committed correction, not from a rejected
            // target suffix.
            state.seedHidden = lastOutput
            state.seedToken = greedyToken(
                logits: target.global.logits(
                    from: predictor.logitsHidden(from: lastOutput)))
        }
        state.proposalAppended = 0
    }

    private func advance(
        token: MLXArray,
        hidden: MLXArray,
        state: Qwen4ExpFlashMTPState
    ) throws -> MLXArray {
        try advanceOutputOnly(token: token, hidden: hidden, state: state)
    }

    private func advanceOutputOnly(
        token: MLXArray,
        hidden: MLXArray,
        state: Qwen4ExpFlashMTPState
    ) throws -> MLXArray {
        let ids = normalizedColumn(token)
        let fullHidden = hidden.ndim == 2 ? hidden.expandedDimensions(axis: 0) : hidden
        guard fullHidden.shape == [1, 1, target.configuration.hiddenSize * target.configuration.hcCount]
        else { throw Qwen4ExpFlashMTPEngineError.invalidVerificationHidden }
        let output = predictor(
            inputEmbeddings: target.global.embed(ids),
            targetHidden: fullHidden,
            inputIDs: ids,
            cache: state.cache,
            positionIDs: Qwen4ExpMRoPE.textPositionIDs(
                sequenceLength: 1, offset: state.cache.offset))
        eval(output)
        state.nextPosition = state.cache.offset
        return output
    }

    private func shiftedPositionIDs(
        _ positions: MLXArray?, promptLength: Int, state: Qwen4ExpFlashMTPState
    ) -> MLXArray? {
        guard let positions else { return nil }
        precondition(positions.dim(-1) == promptLength)
        let prefix = positions[.ellipsis, 1..<promptLength]
        let last = positions[.ellipsis, (promptLength - 1)..<promptLength] + 1
        return concatenated([prefix, last], axis: -1)
    }

    private func normalizedColumn(_ value: MLXArray) -> MLXArray {
        let column = value.ndim == 0 ? value.reshaped([1, 1])
            : value.ndim == 1 ? value.reshaped([value.dim(0), 1]) : value
        precondition(column.shape == [1, 1])
        return column.asType(.int32)
    }

    private func greedyToken(logits: MLXArray) -> MLXArray {
        let sampled = ArgMaxSampler().sample(logits: logits[0..., -1, 0...])
        let token = normalizedColumn(sampled)
        // PM2 (b) (2026-09-09, V54 suspect #2): materialize the token here
        // instead of returning a lazy graph node. Without this, `token`
        // (stored as `state.seedToken`/consumed as the next round's draft
        // input) deferred the quantized 248K-vocab `lm_head` matmul this
        // logits came from to whichever later call first forced evaluation —
        // unpredictable and never accounted by any profiled phase.
        eval(token)
        return token
    }
}
