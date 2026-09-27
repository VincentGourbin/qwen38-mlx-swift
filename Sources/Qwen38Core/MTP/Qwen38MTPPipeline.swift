import Foundation
import MLX
import MLXLMCommon

/// Local M2 speculative loop for Qwen3.8.
///
/// This first implementation intentionally accepts a prepared input and owns
/// one complete generation. Keeping the loop outside `ChatSession` lets us
/// validate block sizes and GDN rollback independently before wiring the
/// persistent conversation controller around it.
public enum Qwen38MTPPipeline {
    public struct Result: Sendable, Equatable {
        public let tokenIDs: [Int32]
        public let stats: Stats
        public let stopReason: GenerateStopReason

        public init(
            tokenIDs: [Int32], stats: Stats,
            stopReason: GenerateStopReason = .length
        ) {
            self.tokenIDs = tokenIDs
            self.stats = stats
            self.stopReason = stopReason
        }
    }

    public struct Stats: Sendable, Equatable {
        public var rounds = 0
        public var proposedTokens = 0
        public var acceptedTokens = 0
        public var targetVerifiedTokens = 0
        public var gdnRestores = 0

        public var acceptanceRate: Double? {
            guard proposedTokens > 0 else { return nil }
            return Double(acceptedTokens) / Double(proposedTokens)
        }
    }

    public enum Error: LocalizedError, Equatable {
        case nonGreedySampling
        case unsupportedPrefill
        case missingDrafterState
        case invalidDrafterOutput

        public var errorDescription: String? {
            switch self {
            case .nonGreedySampling:
                return "Le pipeline M2 requiert un échantillonnage greedy."
            case .unsupportedPrefill:
                return "Le pipeline M2 attend un résultat de préfill avec logits."
            case .missingDrafterState:
                return "La cible n'a pas fourni l'état requis par le drafter MTP."
            case .invalidDrafterOutput:
                return "Le drafter MTP a renvoyé un bloc de tokens invalide."
            }
        }
    }

    /// Runs one greedy speculative generation from a prepared input.
    ///
    /// `blockSize` is the total verification width: one bonus token plus
    /// `blockSize - 1` drafts. Unlike the upstream baseline, the local loop
    /// does not clamp to the drafter's conservative `maximumBlockSize`.
    public static func run(
        input: LMInput,
        target: any LanguageModel,
        drafter: any MTPDrafterModel,
        parameters: GenerateParameters,
        blockSize: Int,
        stopTokenIDs: Set<Int32> = [],
        didStartGeneration: (@Sendable () -> Void)? = nil,
        didGenerate: (@Sendable (Int32) -> Void)? = nil
    ) throws -> Result {
        guard parameters.temperature == 0 else { throw Error.nonGreedySampling }
        precondition(blockSize >= 2, "blockSize doit être >= 2")

        let maxTokens = parameters.maxTokens ?? Int.max
        guard maxTokens > 0 else { return Result(tokenIDs: [], stats: Stats()) }

        let mainCache = try target.newCache(parameters: parameters)
        var mainState = LMOutput.State()
        mainState[mtpEmitFlagKey] = true

        // Force the reference whole-prompt path for the first M2 slice. The
        // production controller will select a bounded prefill once parity is
        // established; this avoids mixing chunking and speculative changes in
        // the first correctness result.
        let prefill = PrefillParameters(stepSize: Int.max, chunking: .unchunked)
        let prepared = try target.prepare(
            input, cache: mainCache, state: mainState, prefill: prefill)
        guard case .logits(let prefillOutput) = prepared else {
            throw Error.unsupportedPrefill
        }
        guard let promptHidden = prefillOutput.state?[mtpLastHiddenStatesKey] else {
            throw Error.missingDrafterState
        }
        mainState = prefillOutput.state ?? mainState

        let sampler = parameters.sampler()
        var bonus = sampler.sample(logits: prefillOutput.logits[0..., -1, 0...])
        eval(bonus, promptHidden)

        var output = [Int32]()
        let bonusID = bonus.item(Int32.self)
        if stopTokenIDs.contains(bonusID) {
            return Result(tokenIDs: output, stats: Stats(), stopReason: .stop)
        }
        output.append(bonusID)
        didGenerate?(bonusID)
        guard output.count < maxTokens else {
            return Result(tokenIDs: output, stats: Stats())
        }

        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
            throw Error.invalidDrafterOutput
        }
        var drafterState = statefulDrafter.makeState(parameters: parameters)
        statefulDrafter.prepareDrafterState(
            target: target,
            promptTokens: input.text.tokens,
            targetHidden: promptHidden,
            firstBonus: bonus,
            positionDeltas: mainState[mtpPositionDeltasKey],
            state: &drafterState,
            sampler: sampler)
        didStartGeneration?()

        var stats = Stats()
        var stopReason: GenerateStopReason = .length
        while output.count < maxTokens {
            guard let targetState = mainState[mtpLastHiddenStatesKey] else {
                throw Error.missingDrafterState
            }

            let lastHidden = targetState[0..., (-1)..., 0...]
            let requestedDrafts = min(
                blockSize - 1, maxTokens - output.count - 1)
            guard requestedDrafts > 0 else { break }

            // The Qwen drafter clears `seedHidden` when it consumes the seed;
            // retain the value needed to continue the same proposal block.
            let pendingSeedHidden = drafterState.seedHidden
            let firstDraft = statefulDrafter.draftBlock(
                target: target,
                lastToken: bonus,
                lastHidden: lastHidden,
                sharedKV: [:],
                positionDeltas: mainState[mtpPositionDeltasKey],
                queryOffset: mainCache.first?.offset ?? 0,
                blockSize: requestedDrafts + 1,
                state: &drafterState,
                sampler: sampler)
            guard firstDraft.ndim == 2, firstDraft.dim(0) == 1 else {
                throw Error.invalidDrafterOutput
            }
            let drafts: MLXArray
            if firstDraft.dim(1) == requestedDrafts {
                // A drafter without a precomputed seed can produce the whole
                // block in one call.
                drafts = firstDraft
            } else if firstDraft.dim(1) == 1, pendingSeedHidden != nil,
                requestedDrafts > 1
            {
                // Qwen MTP-1 returns its first proposal from `seedToken`
                // without appending that token to the private MTP cache. Feed
                // the seed back once with its already-computed MTP hidden;
                // this appends the seed to the cache and produces the rest of
                // the requested block. `commitDrafterState` then sees exactly
                // the number of tentative cache entries it must trim.
                let seed = firstDraft
                let seedHidden = pendingSeedHidden!
                let continuation = statefulDrafter.draftBlock(
                    target: target,
                    lastToken: seed,
                    lastHidden: seedHidden,
                    sharedKV: [:],
                    positionDeltas: mainState[mtpPositionDeltasKey],
                    queryOffset: mainCache.first?.offset ?? 0,
                    blockSize: requestedDrafts,
                    state: &drafterState,
                    sampler: sampler)
                guard continuation.ndim == 2,
                    continuation.dim(0) == 1,
                    continuation.dim(1) == requestedDrafts - 1
                else {
                    throw Error.invalidDrafterOutput
                }
                drafts = concatenated([seed, continuation], axis: 1)
            } else if firstDraft.dim(1) == 1, requestedDrafts == 1 {
                drafts = firstDraft
            } else {
                throw Error.invalidDrafterOutput
            }
            let flatDrafts = drafts.flattened()
            eval(flatDrafts)
            let draftIDs = flatDrafts.asArray(Int32.self)

            let preVerifyState = mainState
            let gdnSnapshot = try Qwen38GDNStateSnapshot(caches: mainCache)
            var verifyState = mainState
            verifyState[mtpEmitFlagKey] = true
            let verifyTokens = concatenated([bonus.flattened(), flatDrafts])
                .asType(.int32)
                .reshaped(1, requestedDrafts + 1)
            let verifyOutput = target(
                LMInput.Text(tokens: verifyTokens), cache: mainCache, state: verifyState)
            guard let verifyHidden = verifyOutput.state?[mtpLastHiddenStatesKey] else {
                throw Error.missingDrafterState
            }
            eval(verifyOutput.logits, verifyHidden)

            var targetIDs = [Int32]()
            targetIDs.reserveCapacity(requestedDrafts + 1)
            for index in 0 ... requestedDrafts {
                let token = sampler.sample(logits: verifyOutput.logits[0..., index, 0...])
                eval(token)
                targetIDs.append(token.item(Int32.self))
            }

            let walk = Qwen38SpeculativeWalk.walk(
                drafts: draftIDs,
                targets: targetIDs,
                budget: maxTokens - output.count)
            let stopIndex = walk.emitted.firstIndex(where: stopTokenIDs.contains)
            let visible = stopIndex.map { Array(walk.emitted[..<$0]) } ?? walk.emitted
            output.append(contentsOf: visible)
            visible.forEach { didGenerate?($0) }
            stats.rounds += 1
            stats.proposedTokens += draftIDs.count
            stats.acceptedTokens += walk.accepted
            stats.targetVerifiedTokens += requestedDrafts + 1

            let rejected = draftIDs.count - walk.accepted
            let finalToken = MLXArray([walk.emitted.last ?? targetIDs[walk.accepted]])
            let postVerifyState: LMOutput.State
            let hiddenForCommit: MLXArray
            if rejected > 0 {
                _ = trimPromptCache(mainCache, numTokens: rejected)
                _ = try gdnSnapshot.restore(to: mainCache)
                stats.gdnRestores += 1

                // Re-forward only the committed prefix so recurrent layers
                // and the hidden boundary agree with the trimmed attention
                // caches. This is the correctness-first M2 rollback path.
                let kept = concatenated([
                    bonus.flattened(),
                    flatDrafts[0 ..< walk.accepted],
                ]).asType(.int32).reshaped(1, walk.accepted + 1)
                var replayState = preVerifyState
                replayState[mtpEmitFlagKey] = true
                let replay = target(
                    LMInput.Text(tokens: kept), cache: mainCache, state: replayState)
                guard let replayHidden = replay.state?[mtpLastHiddenStatesKey] else {
                    throw Error.missingDrafterState
                }
                postVerifyState = replay.state ?? replayState
                hiddenForCommit = replayHidden
            } else {
                postVerifyState = verifyOutput.state ?? verifyState
                hiddenForCommit = verifyHidden
            }

            statefulDrafter.commitDrafterState(
                target: target,
                targetHidden: hiddenForCommit,
                draftTokens: drafts,
                acceptedCount: walk.accepted,
                finalToken: finalToken,
                positionDeltas: postVerifyState[mtpPositionDeltasKey],
                state: &drafterState,
                sampler: sampler)

            mainState = postVerifyState
            bonus = finalToken
            if stopIndex != nil {
                stopReason = .stop
                break
            }
        }

        // A speculative round needs at least one draft plus one verifier
        // position.  If only one output slot remains, finish it with the
        // target's ordinary one-token step instead of silently returning one
        // token short of `maxTokens`.
        if output.count < maxTokens {
            var finalState = mainState
            finalState[mtpEmitFlagKey] = true
            let finalInput = LMInput.Text(
                tokens: bonus.flattened().asType(.int32).reshaped(1, 1))
            let finalOutput = target(
                finalInput, cache: mainCache, state: finalState)
            eval(finalOutput.logits)
            let final = sampler.sample(logits: finalOutput.logits[0..., -1, 0...])
            eval(final)
            let finalID = final.item(Int32.self)
            if stopTokenIDs.contains(finalID) {
                stopReason = .stop
            } else {
                output.append(finalID)
                didGenerate?(finalID)
            }
        }

        return Result(tokenIDs: output, stats: stats, stopReason: stopReason)
    }
}
