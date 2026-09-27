import Foundation
import MLX
import MLXLMCommon

/// Stateful M2 controller for a single in-memory conversation.
///
/// The object is kept behind `Qwen38Runtime`'s actor and is only entered from
/// `ModelContainer.perform`. It owns the target cache, the private drafter
/// cache, the target state, and the exact token ledger as one unit. It is
/// deliberately not a disk prompt-cache format: Qwen VLM M-RoPE state is not
/// safe to restore independently from these caches.
public final class Qwen38MTPConversation: @unchecked Sendable {
    public enum Error: LocalizedError, Equatable {
        case notStarted
        case nonContiguousPrompt(
            expected: Int, actual: Int, firstDifference: Int?, expectedToken: Int32?, actualToken: Int32?)
        case missingTargetHidden
        case unsupportedContinuation

        public var errorDescription: String? {
            switch self {
            case .notStarted: return "La conversation M2 n'a pas encore été initialisée."
            case .nonContiguousPrompt(
                let expected, let actual, let firstDifference, let expectedToken, let actualToken):
                let expectedTokenText = expectedToken.map(String.init) ?? "?"
                let actualTokenText = actualToken.map(String.init) ?? "?"
                return "Le prompt M2 ne prolonge pas le ledger : attendu \(expected) "
                    + "tokens, reçu \(actual), première différence "
                    + (firstDifference.map(String.init) ?? "inconnue")
                    + " (ledger=\(expectedTokenText), prompt=\(actualTokenText))."
            case .missingTargetHidden:
                return "La cible n'a pas fourni les hidden states nécessaires à M2."
            case .unsupportedContinuation:
                return "Le drafter chargé ne sait pas prolonger un état MTP existant."
            }
        }
    }

    private let parameters: GenerateParameters
    private let blockSize: Int
    private let stopTokenIDs: Set<Int32>
    private var mainCache: [KVCache]
    private var mainState = LMOutput.State()
    private var drafterState: MTPDrafterState?
    private var ledger: [Int32] = []
    private var pendingToken: Int32?
    /// A stop token predicted after the committed visible output. It remains
    /// pending at the cache boundary so the next user turn can consume it.
    private var pendingStopToken: Int32?
    private var started = false

    public init(
        target: any LanguageModel,
        parameters: GenerateParameters,
        blockSize: Int,
        stopTokenIDs: Set<Int32> = []
    ) throws {
        guard parameters.temperature == 0, blockSize >= 2 else {
            throw Qwen38MTPPipeline.Error.nonGreedySampling
        }
        self.parameters = parameters
        self.blockSize = blockSize
        self.stopTokenIDs = stopTokenIDs
        self.mainCache = try target.newCache(parameters: parameters)
    }

    public var cachedTokenCount: Int { ledger.count }
    public var isStarted: Bool { started }

    /// Used by the runtime to avoid appending a duplicate `<|im_end|>` after
    /// a generation stopped on that structural token.
    public var pendingStopTokenID: Int32? { pendingStopToken }

    /// Start the conversation with a cold prepared prompt.
    public func start(
        input: LMInput,
        target: any LanguageModel,
        drafter: any MTPDrafterModel,
        didStartGeneration: (@Sendable () -> Void)? = nil,
        didGenerate: (@Sendable (Int32) -> Void)? = nil
    ) throws -> Qwen38MTPPipeline.Result {
        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
            throw Error.unsupportedContinuation
        }

        mainState[mtpEmitFlagKey] = true
        let prefill = PrefillParameters(stepSize: Int.max, chunking: .unchunked)
        let prepared = try target.prepare(
            input, cache: mainCache, state: mainState, prefill: prefill)
        guard case .logits(let output) = prepared,
            let targetHidden = output.state?[mtpLastHiddenStatesKey]
        else {
            throw Qwen38MTPPipeline.Error.unsupportedPrefill
        }
        mainState = output.state ?? mainState

        let sampler = parameters.sampler()
        let bonus = sampler.sample(logits: output.logits[0..., -1, 0...])
        eval(bonus, targetHidden)

        var state = statefulDrafter.makeState(parameters: parameters)
        statefulDrafter.prepareDrafterState(
            target: target,
            promptTokens: input.text.tokens,
            targetHidden: targetHidden,
            firstBonus: bonus,
            positionDeltas: mainState[mtpPositionDeltasKey],
            state: &state,
            sampler: sampler)
        drafterState = state
        started = true
        didStartGeneration?()

        let result = try generateFromBoundary(
            target: target, drafter: drafter, initialBonus: bonus,
            didGenerate: didGenerate)
        let promptTokens = input.text.tokens.asType(.int32).flattened().asArray(Int32.self)
        ledger = promptTokens + result.tokenIDs
        return result
    }

    /// Append the suffix of a newly prepared full conversation prompt.
    ///
    /// The caller supplies the processor's full rendered prompt. Only its
    /// unrepresented suffix is evaluated; the common prefix is checked against
    /// the ledger and is never replayed. For a text-only continuation, the
    /// prepared media fields are intentionally ignored because the media was
    /// already consumed by the first turn.
    public func append(
        input: LMInput,
        target: any LanguageModel,
        drafter: any MTPDrafterModel
    ) throws -> Qwen38MTPPipeline.Result {
        guard started, pendingToken != nil, drafterState != nil else {
            throw Error.notStarted
        }
        guard drafter is any StatefulMTPDrafterModel else {
            throw Error.unsupportedContinuation
        }

        let fullTokens = input.text.tokens.asType(.int32).flattened()
        let fullIDs = fullTokens.asArray(Int32.self)
        guard fullIDs.count >= ledger.count else {
            throw Error.nonContiguousPrompt(
                expected: ledger.count, actual: fullIDs.count, firstDifference: nil,
                expectedToken: nil, actualToken: nil)
        }
        let prefix = Array(fullIDs.prefix(ledger.count))
        guard prefix == ledger else {
            let firstDifference = (0 ..< ledger.count).first {
                prefix[$0] != ledger[$0]
            }
            throw Error.nonContiguousPrompt(
                expected: ledger.count, actual: fullIDs.count,
                firstDifference: firstDifference,
                expectedToken: firstDifference.map { ledger[$0] },
                actualToken: firstDifference.map { fullIDs[$0] })
        }
        let suffix = Array(fullIDs.dropFirst(ledger.count))
        guard !suffix.isEmpty else {
            throw Error.nonContiguousPrompt(
                expected: ledger.count, actual: fullIDs.count, firstDifference: nil,
                expectedToken: nil, actualToken: nil)
        }
        return try append(
            suffixTokens: suffix,
            target: target,
            drafter: drafter)
    }

    /// Appends an exact token suffix after the current target cache boundary.
    ///
    /// This is the persistent-conversation path. Re-rendering all previous
    /// messages through `Chat.Message` is intentionally not used here: the
    /// Qwen template has a separate `reasoning_content` field, while the
    /// generated token stream contains the raw continuation after the opening
    /// `<think>` marker. Re-decoding it as assistant content changes the
    /// token sequence at the first thinking boundary. The caller therefore
    /// builds only the structural/user suffix with the loaded tokenizer.
    public func append(
        suffixTokens: [Int32],
        target: any LanguageModel,
        drafter: any MTPDrafterModel,
        didStartGeneration: (@Sendable () -> Void)? = nil,
        didGenerate: (@Sendable (Int32) -> Void)? = nil
    ) throws -> Qwen38MTPPipeline.Result {
        guard started, let pendingToken, let previousDrafterState = drafterState else {
            throw Error.notStarted
        }
        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel else {
            throw Error.unsupportedContinuation
        }
        guard !suffixTokens.isEmpty else {
            throw Error.nonContiguousPrompt(
                expected: ledger.count, actual: ledger.count,
                firstDifference: nil, expectedToken: nil, actualToken: nil)
        }

        let suffixCount = suffixTokens.count
        let suffix = MLXArray(suffixTokens).reshaped(1, suffixCount)
        let pending = MLXArray([pendingToken]).reshaped(1, 1)
        let targetInput = LMInput.Text(tokens: concatenated([pending, suffix], axis: 1))
        var continuationState = mainState
        continuationState[mtpEmitFlagKey] = true
        let output = target(targetInput, cache: mainCache, state: continuationState)
        guard let targetHidden = output.state?[mtpLastHiddenStatesKey],
            targetHidden.dim(1) >= suffixCount + 1
        else {
            throw Error.missingTargetHidden
        }
        mainState = output.state ?? continuationState

        let sampler = parameters.sampler()
        let bonus = sampler.sample(logits: output.logits[0..., -1, 0...])
        eval(bonus, targetHidden)

        // The drafter must see each new suffix token paired with the target
        // hidden immediately before it, then the target bonus paired with the
        // final suffix hidden. The pending output token is already represented
        // by the seed produced at the end of the previous turn.
        let drafterTokens = concatenated([
            suffix,
            bonus.reshaped(1, 1),
        ], axis: 1)
        let drafterHidden = targetHidden[0..., ..<(suffixCount + 1), 0...]
        // UPSTREAM PROBE: `appendDrafterState` is a local patch; unsupported here.
        _ = (statefulDrafter, drafterTokens, drafterHidden, previousDrafterState)
        throw Error.unsupportedContinuation
        didStartGeneration?()

        let result = try generateFromBoundary(
            target: target, drafter: drafter, initialBonus: bonus,
            didGenerate: didGenerate)
        ledger.append(contentsOf: suffixTokens)
        ledger.append(contentsOf: result.tokenIDs)
        return result
    }

    private func generateFromBoundary(
        target: any LanguageModel,
        drafter: any MTPDrafterModel,
        initialBonus: MLXArray,
        didGenerate: (@Sendable (Int32) -> Void)?
    ) throws -> Qwen38MTPPipeline.Result {
        guard let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        else {
            throw Error.unsupportedContinuation
        }

        let sampler = parameters.sampler()
        let maxTokens = parameters.maxTokens ?? Int.max
        var output = [Int32]()
        var bonus = initialBonus
        let initialBonusID = bonus.item(Int32.self)
        pendingStopToken = nil
        if stopTokenIDs.contains(initialBonusID) {
            pendingToken = initialBonusID
            pendingStopToken = initialBonusID
            return Qwen38MTPPipeline.Result(
                tokenIDs: [], stats: Qwen38MTPPipeline.Stats(), stopReason: .stop)
        }
        output.append(initialBonusID)
        didGenerate?(initialBonusID)
        var stats = Qwen38MTPPipeline.Stats()
        var stopReason: GenerateStopReason = .length

        while output.count < maxTokens {
            guard let targetState = mainState[mtpLastHiddenStatesKey] else {
                throw Qwen38MTPPipeline.Error.missingDrafterState
            }
            let lastHidden = targetState[0..., (-1)..., 0...]
            let requestedDrafts = min(blockSize - 1, maxTokens - output.count - 1)
            guard requestedDrafts > 0 else { break }

            let pendingSeedHidden = currentDrafterState.seedHidden
            let firstDraft = statefulDrafter.draftBlock(
                target: target,
                lastToken: bonus,
                lastHidden: lastHidden,
                sharedKV: [:],
                positionDeltas: mainState[mtpPositionDeltasKey],
                queryOffset: mainCache.first?.offset ?? 0,
                blockSize: requestedDrafts + 1,
                state: &currentDrafterState,
                sampler: sampler)
            guard firstDraft.ndim == 2, firstDraft.dim(0) == 1 else {
                throw Qwen38MTPPipeline.Error.invalidDrafterOutput
            }
            let drafts: MLXArray
            if firstDraft.dim(1) == requestedDrafts {
                drafts = firstDraft
            } else if firstDraft.dim(1) == 1, let pendingSeedHidden,
                requestedDrafts > 1
            {
                let continuation = statefulDrafter.draftBlock(
                    target: target,
                    lastToken: firstDraft,
                    lastHidden: pendingSeedHidden,
                    sharedKV: [:],
                    positionDeltas: mainState[mtpPositionDeltasKey],
                    queryOffset: mainCache.first?.offset ?? 0,
                    blockSize: requestedDrafts,
                    state: &currentDrafterState,
                    sampler: sampler)
                guard continuation.ndim == 2,
                    continuation.dim(0) == 1,
                    continuation.dim(1) == requestedDrafts - 1
                else { throw Qwen38MTPPipeline.Error.invalidDrafterOutput }
                drafts = concatenated([firstDraft, continuation], axis: 1)
            } else if firstDraft.dim(1) == 1, requestedDrafts == 1 {
                drafts = firstDraft
            } else {
                throw Qwen38MTPPipeline.Error.invalidDrafterOutput
            }

            let flatDrafts = drafts.flattened()
            eval(flatDrafts)
            let draftIDs = flatDrafts.asArray(Int32.self)
            let preVerifyState = mainState
            let gdnSnapshot = try Qwen38GDNStateSnapshot(caches: mainCache)
            var verifyState = mainState
            verifyState[mtpEmitFlagKey] = true
            let verifyTokens = concatenated([bonus.flattened(), flatDrafts])
                .asType(.int32).reshaped(1, requestedDrafts + 1)
            let verifyOutput = target(
                LMInput.Text(tokens: verifyTokens), cache: mainCache, state: verifyState)
            guard let verifyHidden = verifyOutput.state?[mtpLastHiddenStatesKey] else {
                throw Qwen38MTPPipeline.Error.missingDrafterState
            }
            eval(verifyOutput.logits, verifyHidden)

            var targetIDs = [Int32]()
            for index in 0 ... requestedDrafts {
                let token = sampler.sample(logits: verifyOutput.logits[0..., index, 0...])
                eval(token)
                targetIDs.append(token.item(Int32.self))
            }
            let walk = Qwen38SpeculativeWalk.walk(
                drafts: draftIDs, targets: targetIDs,
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
                let kept = concatenated([
                    bonus.flattened(), flatDrafts[0 ..< walk.accepted],
                ]).asType(.int32).reshaped(1, walk.accepted + 1)
                var replayState = preVerifyState
                replayState[mtpEmitFlagKey] = true
                let replay = target(
                    LMInput.Text(tokens: kept), cache: mainCache, state: replayState)
                guard let replayHidden = replay.state?[mtpLastHiddenStatesKey] else {
                    throw Qwen38MTPPipeline.Error.missingDrafterState
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
                state: &currentDrafterState,
                sampler: sampler)
            mainState = postVerifyState
            bonus = finalToken
            if stopIndex != nil {
                stopReason = .stop
                // The stop token is not part of the committed visible prefix;
                // retain it as the next target input instead of destroying a
                // still-valid M2 boundary.
                pendingStopToken = walk.emitted.last
                break
            }
        }

        if stopReason != .stop && output.count < maxTokens {
            var finalState = mainState
            finalState[mtpEmitFlagKey] = true
            let finalOutput = target(
                LMInput.Text(tokens: bonus.flattened().asType(.int32).reshaped(1, 1)),
                cache: mainCache, state: finalState)
            eval(finalOutput.logits)
            let final = sampler.sample(logits: finalOutput.logits[0..., -1, 0...])
            eval(final)
            let finalID = final.item(Int32.self)
            if stopTokenIDs.contains(finalID) {
                stopReason = .stop
                pendingStopToken = finalID
            } else {
                output.append(finalID)
                didGenerate?(finalID)
            }
            mainState = finalOutput.state ?? finalState
        }

        drafterState = currentDrafterState
        self.pendingToken = pendingStopToken ?? output.last
        // EOS ends generation, not the reusable target/drafter boundary. The
        // next append consumes the pending stop token and adds the new user
        // suffix on the same caches.
        self.started = true
        return Qwen38MTPPipeline.Result(
            tokenIDs: output, stats: stats, stopReason: stopReason)
    }
}
