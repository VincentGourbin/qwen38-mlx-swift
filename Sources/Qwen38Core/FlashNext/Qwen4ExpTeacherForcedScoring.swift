import Foundation
import MLX

/// One teacher-forced observation for a continuation token.
public struct Qwen4ExpTeacherForcedTokenScore: Sendable, Equatable {
    public let tokenID: Int32
    public let logProbability: Float
    public let argmaxTokenID: Int32
    public let targetRank: Int
    public let argmaxMargin: Float

    public init(
        tokenID: Int32,
        logProbability: Float,
        argmaxTokenID: Int32,
        targetRank: Int,
        argmaxMargin: Float
    ) {
        self.tokenID = tokenID
        self.logProbability = logProbability
        self.argmaxTokenID = argmaxTokenID
        self.targetRank = targetRank
        self.argmaxMargin = argmaxMargin
    }
}

/// Aggregate used by Q-B.  The metrics are deliberately computed from the
/// same logits tensor: no sampling, stop-token handling or second forward can
/// contaminate the comparison with the Python reference.
public struct Qwen4ExpTeacherForcedScore: Sendable, Equatable {
    public let promptTokenCount: Int
    public let continuationTokenCount: Int
    public let meanLogProbability: Float
    public let argmaxAgreement: Float
    public let confidentAgreement: Float
    public let confidentTokenCount: Int
    public let meanTargetRank: Float
    public let tokens: [Qwen4ExpTeacherForcedTokenScore]
    public let prefillTime: TimeInterval
    public let layerReports: [Qwen4ExpStreamingLayerReport]

    public init(
        promptTokenCount: Int,
        continuationTokenCount: Int,
        meanLogProbability: Float,
        argmaxAgreement: Float,
        confidentAgreement: Float,
        confidentTokenCount: Int,
        meanTargetRank: Float,
        tokens: [Qwen4ExpTeacherForcedTokenScore],
        prefillTime: TimeInterval,
        layerReports: [Qwen4ExpStreamingLayerReport]
    ) {
        self.promptTokenCount = promptTokenCount
        self.continuationTokenCount = continuationTokenCount
        self.meanLogProbability = meanLogProbability
        self.argmaxAgreement = argmaxAgreement
        self.confidentAgreement = confidentAgreement
        self.confidentTokenCount = confidentTokenCount
        self.meanTargetRank = meanTargetRank
        self.tokens = tokens
        self.prefillTime = prefillTime
        self.layerReports = layerReports
    }
}

public enum Qwen4ExpTeacherForcedScoringError: LocalizedError, Equatable {
    case emptyPrompt
    case emptyContinuation
    case invalidPositionIDs

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            return "Le prompt teacher-forced ne peut pas être vide."
        case .emptyContinuation:
            return "La continuation teacher-forced ne peut pas être vide."
        case .invalidPositionIDs:
            return "Les position IDs doivent couvrir prompt + continuation."
        }
    }
}

extension Qwen4ExpStreamingTextModel {
    /// Scores a fixed continuation in one causal forward.
    ///
    /// The logits at positions `prompt.count - 1 ... end - 1` predict the
    /// continuation tokens.  This is the central Q-B primitive: run the same
    /// token sequence through Swift and Python, then compare log-probability,
    /// argmax agreement above a margin, and target ranks.  It is intentionally
    /// host-reduced after `eval` so the result is independent of sampler code.
    public func scoreTeacherForced(
        promptTokenIDs: [Int32],
        continuationTokenIDs: [Int32],
        positionIDs: MLXArray? = nil,
        visionEmbeddings: MLXArray? = nil,
        imageTokenID: Int32? = nil,
        confidentMargin: Float = 0.5
    ) throws -> Qwen4ExpTeacherForcedScore {
        guard !promptTokenIDs.isEmpty else {
            throw Qwen4ExpTeacherForcedScoringError.emptyPrompt
        }
        guard !continuationTokenIDs.isEmpty else {
            throw Qwen4ExpTeacherForcedScoringError.emptyContinuation
        }
        if let positionIDs {
            let expected = promptTokenIDs.count + continuationTokenIDs.count
            guard positionIDs.ndim == 2 || positionIDs.ndim == 3 else {
                throw Qwen4ExpTeacherForcedScoringError.invalidPositionIDs
            }
            let length = positionIDs.ndim == 2 ? positionIDs.dim(1) : positionIDs.dim(2)
            guard length == expected else {
                throw Qwen4ExpTeacherForcedScoringError.invalidPositionIDs
            }
        }

        resetConversation()
        let allIDs = promptTokenIDs + continuationTokenIDs
        let inputIDs = MLXArray(allIDs).reshaped([1, allIDs.count])
        let started = Date()
        let forward = try forward(
            inputIDs: inputIDs,
            positionIDs: positionIDs,
            visionEmbeddings: visionEmbeddings,
            imageTokenID: imageTokenID)
        eval(forward.logits)
        let logits = forward.logits.asType(.float32).asArray(Float.self)
        let vocabulary = forward.logits.dim(-1)
        let promptCount = promptTokenIDs.count
        let continuationCount = continuationTokenIDs.count
        let rows = allIDs.count

        var scores: [Qwen4ExpTeacherForcedTokenScore] = []
        scores.reserveCapacity(continuationCount)
        var logProbabilitySum: Float = 0
        var argmaxMatches = 0
        var confidentMatches = 0
        var confidentCount = 0
        var rankSum = 0

        for continuationIndex in 0..<continuationCount {
            // Row r predicts token r + 1.  The first continuation is predicted
            // by the final prompt row.
            let row = promptCount - 1 + continuationIndex
            let offset = row * vocabulary
            let target = continuationTokenIDs[continuationIndex]
            let targetIndex = Int(target)
            guard row >= 0, row < rows, targetIndex >= 0, targetIndex < vocabulary else {
                continue
            }

            var maximum = -Float.infinity
            var argmax = 0
            var second = -Float.infinity
            for index in 0..<vocabulary {
                let value = logits[offset + index]
                if value > maximum {
                    second = maximum
                    maximum = value
                    argmax = index
                } else if value > second {
                    second = value
                }
            }

            let targetLogit = logits[offset + targetIndex]
            var expSum: Double = 0
            for index in 0..<vocabulary {
                expSum += Foundation.exp(Double(logits[offset + index] - maximum))
            }
            let logProbability = targetLogit - maximum - Float(Foundation.log(expSum))
            let rank = 1 + (0..<vocabulary).reduce(into: 0) { count, index in
                if logits[offset + index] > targetLogit { count += 1 }
            }
            let margin = maximum - second
            let match = argmax == targetIndex
            if match { argmaxMatches += 1 }
            if margin >= confidentMargin {
                confidentCount += 1
                if match { confidentMatches += 1 }
            }
            logProbabilitySum += logProbability
            rankSum += rank
            scores.append(.init(
                tokenID: target,
                logProbability: logProbability,
                argmaxTokenID: Int32(argmax),
                targetRank: rank,
                argmaxMargin: margin))
        }

        let measuredCount = max(scores.count, 1)
        return Qwen4ExpTeacherForcedScore(
            promptTokenCount: promptCount,
            continuationTokenCount: continuationCount,
            meanLogProbability: logProbabilitySum / Float(measuredCount),
            argmaxAgreement: Float(argmaxMatches) / Float(measuredCount),
            confidentAgreement: confidentCount == 0
                ? 0 : Float(confidentMatches) / Float(confidentCount),
            confidentTokenCount: confidentCount,
            meanTargetRank: Float(rankSum) / Float(measuredCount),
            tokens: scores,
            prefillTime: Date().timeIntervalSince(started),
            layerReports: forward.reports)
    }
}
