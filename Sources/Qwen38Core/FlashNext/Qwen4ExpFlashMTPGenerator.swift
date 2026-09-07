import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// Counters emitted by the local Flash-Next MTP loop.
public struct Qwen4ExpFlashMTPGenerationStats: Sendable, Equatable {
    public var rounds = 0
    public var proposedTokens = 0
    public var acceptedTokens = 0
    public var targetVerifiedTokens = 0
    public var rollbacks = 0
    public var replayedTokens = 0

    public var acceptanceRate: Double? {
        guard proposedTokens > 0 else { return nil }
        return Double(acceptedTokens) / Double(proposedTokens)
    }
}

public struct Qwen4ExpFlashMTPGenerationResult: Sendable {
    public let tokenIDs: [Int32]
    public let promptTokenCount: Int
    public let prefillTime: TimeInterval
    public let generationTime: TimeInterval
    public let timeToFirstToken: TimeInterval?
    public let stats: Qwen4ExpFlashMTPGenerationStats
    public let layerReports: [[Qwen4ExpStreamingLayerReport]]

    public init(
        tokenIDs: [Int32], promptTokenCount: Int, prefillTime: TimeInterval,
        generationTime: TimeInterval, timeToFirstToken: TimeInterval?,
        stats: Qwen4ExpFlashMTPGenerationStats,
        layerReports: [[Qwen4ExpStreamingLayerReport]]
    ) {
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.prefillTime = prefillTime
        self.generationTime = generationTime
        self.timeToFirstToken = timeToFirstToken
        self.stats = stats
        self.layerReports = layerReports
    }

    public var layerLoadTime: TimeInterval {
        layerReports.flatMap { $0 }.reduce(0) { $0 + $1.loadDuration }
    }

    public var layerForwardTime: TimeInterval {
        layerReports.flatMap { $0 }.reduce(0) { $0 + $1.forwardDuration }
    }

    public var layerVisitCount: Int {
        layerReports.reduce(0) { $0 + $1.count }
    }
}

public extension Qwen4ExpGreedyGenerator {
    /// Opt-in local MTP generation for the Flash-Next text path.
    ///
    /// The target remains authoritative: every draft block is verified in one
    /// target forward, and a rejected suffix restores the complete target
    /// snapshot before replaying the committed prefix.  Multimodal M-RoPE
    /// position deltas are deliberately rejected here until their continuation
    /// contract is wired into the drafter; the one-round probe already covers
    /// the text cache/rollback path.
    func generateMTP(
        promptTokenIDs: [Int32],
        predictor: Qwen4ExpMTPPredictor,
        options: Qwen4ExpGreedyGenerationOptions = .init(),
        blockSize: Int = 2,
        profiler: MLXProfiler = .shared
    ) throws -> Qwen4ExpFlashMTPGenerationResult {
        guard !promptTokenIDs.isEmpty else {
            throw Qwen4ExpGreedyGenerationError.emptyPrompt
        }
        guard options.maxNewTokens > 0 else {
            throw Qwen4ExpGreedyGenerationError.invalidMaxNewTokens
        }
        guard blockSize >= 2 else {
            throw Qwen4ExpFlashMTPEngineError.invalidBlockSize
        }

        model.resetConversation()
        model.resetNGramCacheStats()
        let prompt = MLXArray(promptTokenIDs).reshaped([1, promptTokenIDs.count])
        let started = Date()
        profiler.startPrefill()
        let prefill = try model.forward(inputIDs: prompt)
        eval(prefill.logits)
        eval(prefill.preMixerHidden)
        recordNGramCacheStats(profiler)
        let prefillEnd = Date()
        profiler.endPrefill()

        let engine = Qwen4ExpFlashMTPDraftEngine(target: model, predictor: predictor)
        let state = engine.makeState()
        let firstBonus = greedyToken(from: prefill.logits[0..., -1, 0...])
        let firstBonusID = firstBonus.item(Int32.self)
        if options.stopTokenIDs.contains(firstBonusID) {
            return Qwen4ExpFlashMTPGenerationResult(
                tokenIDs: [], promptTokenCount: promptTokenIDs.count,
                prefillTime: prefillEnd.timeIntervalSince(started), generationTime: 0,
                timeToFirstToken: nil, stats: .init(), layerReports: [prefill.reports])
        }

        try engine.prepare(
            promptTokenIDs: prompt,
            targetHidden: prefill.preMixerHidden,
            firstBonus: firstBonus,
            state: state)

        var output = [firstBonusID]
        var bonus = firstBonus
        var lastHidden = prefill.preMixerHidden[0..., (-1)..., 0...]
        var reports = [prefill.reports]
        var stats = Qwen4ExpFlashMTPGenerationStats()
        let firstTokenTime = Date().timeIntervalSince(started)
        profiler.startGeneration()
        let generationStarted = Date()

        while output.count < options.maxNewTokens {
            // The first bonus is already one generated token, but the next
            // speculative round may still emit up to the remaining budget.
            // Subtracting one here made maxNewTokens == 2 silently skip the
            // first MTP round and return only the bonus token.
            let requestedDrafts = Self.requestedDraftCount(
                maxNewTokens: options.maxNewTokens,
                generatedCount: output.count,
                blockSize: blockSize)
            guard requestedDrafts > 0 else { break }

            let drafts = try engine.draftBlock(
                lastToken: bonus,
                lastHidden: lastHidden,
                blockSize: requestedDrafts + 1,
                state: state)
            eval(drafts)
            let draftIDs = drafts.flattened().asArray(Int32.self)

            let targetSnapshot = model.snapshot()
            let verifyTokens = concatenated([
                bonus.reshaped([1, 1]), drafts
            ], axis: 1).asType(.int32)
            let verification = try model.forward(inputIDs: verifyTokens)
            eval(verification.logits)
            eval(verification.preMixerHidden)
            recordNGramCacheStats(profiler)
            reports.append(verification.reports)
            let targetIDs = (0 ..< verifyTokens.dim(1)).map { index in
                greedyToken(from: verification.logits[0..., index, 0...])
                    .item(Int32.self)
            }
            let walk = Qwen38SpeculativeWalk.walk(
                drafts: draftIDs, targets: targetIDs,
                budget: options.maxNewTokens - output.count)
            let finalToken = MLXArray([walk.emitted.last ?? targetIDs[walk.accepted]])

            let hiddenForCommit: MLXArray
            if walk.accepted < draftIDs.count {
                model.restore(targetSnapshot)
                let replayTokens = concatenated([
                    bonus.reshaped([1, 1]),
                    drafts[0..., 0 ..< walk.accepted]
                ], axis: 1).asType(.int32)
                let replay = try model.forward(inputIDs: replayTokens)
                eval(replay.preMixerHidden)
                recordNGramCacheStats(profiler)
                reports.append(replay.reports)
                hiddenForCommit = replay.preMixerHidden
                stats.rollbacks += 1
                stats.replayedTokens += replayTokens.dim(1)
            } else {
                hiddenForCommit = verification.preMixerHidden
            }

            try engine.commit(
                targetHidden: hiddenForCommit,
                draftTokens: drafts,
                acceptedCount: walk.accepted,
                finalToken: finalToken,
                state: state)

            stats.rounds += 1
            stats.proposedTokens += draftIDs.count
            stats.acceptedTokens += walk.accepted
            stats.targetVerifiedTokens += requestedDrafts + 1

            for token in walk.emitted {
                if options.stopTokenIDs.contains(token) {
                    output.append(token)
                    break
                }
                output.append(token)
                if output.count >= options.maxNewTokens { break }
            }

            lastHidden = hiddenForCommit[0..., (-1)..., 0...]
            bonus = finalToken
            if output.last.map(options.stopTokenIDs.contains) == true { break }
        }

        profiler.endGeneration(tokenCount: output.count)
        return Qwen4ExpFlashMTPGenerationResult(
            tokenIDs: Array(output.prefix(options.maxNewTokens)),
            promptTokenCount: promptTokenIDs.count,
            prefillTime: prefillEnd.timeIntervalSince(started),
            generationTime: Date().timeIntervalSince(generationStarted),
            timeToFirstToken: firstTokenTime,
            stats: stats,
            layerReports: reports)
    }

    private func greedyToken(from logits: MLXArray) -> MLXArray {
        let token = ArgMaxSampler().sample(logits: logits)
        eval(token)
        return token
    }
}

extension Qwen4ExpGreedyGenerator {
    /// Computes the number of draft tokens still allowed in a speculative
    /// round. Kept separate so the bonus-token boundary is unit-testable.
    static func requestedDraftCount(
        maxNewTokens: Int, generatedCount: Int, blockSize: Int
    ) -> Int {
        guard maxNewTokens > generatedCount, blockSize >= 2 else { return 0 }
        return min(blockSize - 1, maxNewTokens - generatedCount)
    }
}
