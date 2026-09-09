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

/// PM1 (2026-09-09): cumulative wall time per phase of `generateMTP`,
/// measured with `ContinuousClock` — independent of `MLXProfiler` and always
/// on (no IOKit GPU sample, no `rusage`, unlike `profiler.start`/`.end` whose
/// ~4.7 ms per boundary made per-layer profiling opt-in, see
/// `Qwen4ExpStreamingDecoder.profileLayers`). This is the direct answer to the
/// V54 question "where do the ~880s outside any profiled phase go": every
/// step between two `model.forward` calls is now individually timed.
public struct Qwen4ExpFlashMTPStepTimings: Sendable, Equatable {
    /// `engine.draftBlock` — drafter forward(s) + greedy sampling of the
    /// speculative suffix.
    public var draftBlock: TimeInterval = 0
    /// The target's verification `forward` over `[bonus, drafts]`.
    public var verifyForward: TimeInterval = 0
    /// Extracting one greedy token id per verified position from the
    /// verification logits (`targetIDs`).
    public var targetIDs: TimeInterval = 0
    /// PM4.2 (P-MTP suite): `model.rollbackVerification(...)` — host-side
    /// cache slicing only, paid on a rollback round. Replaces the PM1-era
    /// `model.snapshot()` (every round) + `model.restore(...)` and replay
    /// forward (rollback rounds only), both removed: no target forward is
    /// replayed anymore.
    public var rollback: TimeInterval = 0
    /// `engine.commit` — reconciling the drafter's private cache.
    public var commit: TimeInterval = 0

    public init(
        draftBlock: TimeInterval = 0,
        verifyForward: TimeInterval = 0, targetIDs: TimeInterval = 0,
        rollback: TimeInterval = 0, commit: TimeInterval = 0
    ) {
        self.draftBlock = draftBlock
        self.verifyForward = verifyForward
        self.targetIDs = targetIDs
        self.rollback = rollback
        self.commit = commit
    }

    public var total: TimeInterval {
        draftBlock + verifyForward + targetIDs + rollback + commit
    }

    fileprivate mutating func add(_ other: Qwen4ExpFlashMTPStepTimings) {
        draftBlock += other.draftBlock
        verifyForward += other.verifyForward
        targetIDs += other.targetIDs
        rollback += other.rollback
        commit += other.commit
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
    /// PM1: cumulative per-step wall time, see `Qwen4ExpFlashMTPStepTimings`.
    public let stepTimings: Qwen4ExpFlashMTPStepTimings

    public init(
        tokenIDs: [Int32], promptTokenCount: Int, prefillTime: TimeInterval,
        generationTime: TimeInterval, timeToFirstToken: TimeInterval?,
        stats: Qwen4ExpFlashMTPGenerationStats,
        layerReports: [[Qwen4ExpStreamingLayerReport]],
        stepTimings: Qwen4ExpFlashMTPStepTimings = .init()
    ) {
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.prefillTime = prefillTime
        self.generationTime = generationTime
        self.timeToFirstToken = timeToFirstToken
        self.stats = stats
        self.layerReports = layerReports
        self.stepTimings = stepTimings
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
        profiler: MLXProfiler = .shared,
        // PM1 (2026-09-09): opt-in `MLXProfiler` phases around each step of
        // the round loop, mirroring `Qwen4ExpStreamingDecoder.profileLayers`
        // (~4.7 ms per start/end boundary — not paid unless explicitly asked
        // for). The cheap `ContinuousClock` totals in
        // `Qwen4ExpFlashMTPStepTimings` are always collected regardless of
        // this flag.
        profileMTP: Bool = false
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
        var stepTimings = Qwen4ExpFlashMTPStepTimings()
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

            if profileMTP { profiler.start("MTP draftBlock") }
            let draftBlockStart = ContinuousClock.now
            let drafts = try engine.draftBlock(
                lastToken: bonus,
                lastHidden: lastHidden,
                blockSize: requestedDrafts + 1,
                state: state)
            eval(drafts)
            let draftIDs = drafts.flattened().asArray(Int32.self)
            stepTimings.draftBlock += (ContinuousClock.now - draftBlockStart).seconds
            if profileMTP { profiler.end("MTP draftBlock") }

            if profileMTP { profiler.start("MTP verify") }
            let verifyStart = ContinuousClock.now
            let verifyTokens = concatenated([
                bonus.reshaped([1, 1]), drafts
            ], axis: 1).asType(.int32)
            // PM4.2 (P-MTP suite): the verification forward now always
            // captures the per-token materials (`Qwen4ExpVerificationCapture`)
            // needed to roll a partial rejection back without replaying a
            // second target forward — see `rollbackVerification` below.
            let verificationCapture = Qwen4ExpVerificationCapture()
            let verification = try model.forward(
                inputIDs: verifyTokens, verificationCapture: verificationCapture)
            eval(verification.logits)
            eval(verification.preMixerHidden)
            recordNGramCacheStats(profiler)
            reports.append(verification.reports)
            stepTimings.verifyForward += (ContinuousClock.now - verifyStart).seconds
            if profileMTP { profiler.end("MTP verify") }

            if profileMTP { profiler.start("MTP targetIDs") }
            let targetIDsStart = ContinuousClock.now
            // PM2 (a) (2026-09-09): a single batched `argMax(axis: -1)` over
            // every verified position, followed by one `asArray` host sync,
            // instead of one `.item()` per position (V54's suspect #3 — a
            // CPU sync per verified token). Bit-identical result: `argMax`
            // over the vocabulary axis at each position is exactly what the
            // per-position `ArgMaxSampler` loop computed, just batched.
            let targetIDs = argMax(verification.logits, axis: -1).asArray(Int32.self)
            stepTimings.targetIDs += (ContinuousClock.now - targetIDsStart).seconds
            if profileMTP { profiler.end("MTP targetIDs") }

            let walk = Qwen38SpeculativeWalk.walk(
                drafts: draftIDs, targets: targetIDs,
                budget: options.maxNewTokens - output.count)
            let finalToken = MLXArray([walk.emitted.last ?? targetIDs[walk.accepted]])

            // PM4.2: `verification.preMixerHidden` already holds the hidden
            // state at every one of the `verifyTokens.dim(1)` new positions,
            // computed by the single verify forward above. Because the
            // model is strictly causal (QSA attention is masked, GDN's
            // recurrence only flows forward in time, PLE only looks
            // backward), the hidden state at position i does not depend on
            // any token fed at a position > i — it is bit-identical to what
            // a forward over just the first i+1 positions would have
            // produced. The committed prefix's hidden rows are therefore
            // already sitting in `verification.preMixerHidden`; no replay
            // forward is needed to obtain them.
            let committedNewTokens = walk.accepted + 1
            let hiddenForCommit = verification.preMixerHidden[
                0..., 0 ..< committedNewTokens, 0...]
            if walk.accepted < draftIDs.count {
                if profileMTP { profiler.start("MTP rollback") }
                let rollbackStart = ContinuousClock.now
                model.rollbackVerification(
                    capture: verificationCapture,
                    committedNewTokens: committedNewTokens,
                    totalNewTokens: verifyTokens.dim(1))
                stats.rollbacks += 1
                stepTimings.rollback += (ContinuousClock.now - rollbackStart).seconds
                if profileMTP { profiler.end("MTP rollback") }
            }

            if profileMTP { profiler.start("MTP commit") }
            let commitStart = ContinuousClock.now
            try engine.commit(
                targetHidden: hiddenForCommit,
                draftTokens: drafts,
                acceptedCount: walk.accepted,
                finalToken: finalToken,
                state: state)
            stepTimings.commit += (ContinuousClock.now - commitStart).seconds
            if profileMTP { profiler.end("MTP commit") }

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
            layerReports: reports,
            stepTimings: stepTimings)
    }

    private func greedyToken(from logits: MLXArray) -> MLXArray {
        let token = ArgMaxSampler().sample(logits: logits)
        eval(token)
        return token
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
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
