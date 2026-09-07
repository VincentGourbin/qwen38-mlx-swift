import Foundation
import MLX

public struct Qwen4ExpTeacherForcedParityReport: Sendable, Equatable {
    public let tokenCount: Int
    public let continuationIDsMatch: Bool
    public let maxLogProbabilityAbsoluteError: Float
    public let meanLogProbabilityAbsoluteError: Float
    public let maxMarginAbsoluteError: Float
    public let maxTargetRankDifference: Int
    public let argmaxAgreement: Float
    public let pythonArgmaxAgreement: Float
    public let targetRankMeanDifference: Float

    public var isAligned: Bool { continuationIDsMatch && tokenCount > 0 }

    public init(
        tokenCount: Int,
        continuationIDsMatch: Bool,
        maxLogProbabilityAbsoluteError: Float,
        meanLogProbabilityAbsoluteError: Float,
        maxMarginAbsoluteError: Float,
        maxTargetRankDifference: Int,
        argmaxAgreement: Float,
        pythonArgmaxAgreement: Float,
        targetRankMeanDifference: Float
    ) {
        self.tokenCount = tokenCount
        self.continuationIDsMatch = continuationIDsMatch
        self.maxLogProbabilityAbsoluteError = maxLogProbabilityAbsoluteError
        self.meanLogProbabilityAbsoluteError = meanLogProbabilityAbsoluteError
        self.maxMarginAbsoluteError = maxMarginAbsoluteError
        self.maxTargetRankDifference = maxTargetRankDifference
        self.argmaxAgreement = argmaxAgreement
        self.pythonArgmaxAgreement = pythonArgmaxAgreement
        self.targetRankMeanDifference = targetRankMeanDifference
    }
}

public enum Qwen4ExpTeacherForcedParityError: LocalizedError, Equatable {
    case missingTensor
    case shapeMismatch
    case continuationMismatch

    public var errorDescription: String? {
        switch self {
        case .missingTensor:
            return "Fixture teacher-forced incomplète : métriques cibles absentes."
        case .shapeMismatch:
            return "Fixture teacher-forced incompatible avec le nombre de tokens scorés."
        case .continuationMismatch:
            return "Les IDs de continuation Swift et Python ne sont pas identiques."
        }
    }
}

/// Compares the already-computed Swift score with a Python MLX fixture.
/// Loading this fixture never runs another model forward.
public enum Qwen4ExpTeacherForcedParity {
    public static func compare(
        score: Qwen4ExpTeacherForcedScore,
        fixtureURL: URL
    ) throws -> Qwen4ExpTeacherForcedParityReport {
        let arrays = try loadArrays(url: fixtureURL, stream: .cpu)
        guard let expectedIDs = arrays["continuation_ids"],
              let expectedLogProbabilities = arrays["target_logprob"],
              let expectedArgmax = arrays["argmax"],
              let expectedRanks = arrays["target_rank"],
              let expectedMargins = arrays["argmax_margin"] else {
            throw Qwen4ExpTeacherForcedParityError.missingTensor
        }

        let pythonIDs = expectedIDs.asType(.int32).asArray(Int32.self)
        let pythonLogProbabilities = expectedLogProbabilities
            .asType(.float32).asArray(Float.self)
        let pythonArgmax = expectedArgmax.asType(.int32).asArray(Int32.self)
        let pythonRanks = expectedRanks.asType(.int32).asArray(Int32.self)
        let pythonMargins = expectedMargins.asType(.float32).asArray(Float.self)
        let count = score.tokens.count
        guard pythonIDs.count == count,
              pythonLogProbabilities.count == count,
              pythonArgmax.count == count,
              pythonRanks.count == count,
              pythonMargins.count == count else {
            throw Qwen4ExpTeacherForcedParityError.shapeMismatch
        }

        guard score.tokens.enumerated().allSatisfy({ index, token in
            token.tokenID == pythonIDs[index]
        }) else {
            throw Qwen4ExpTeacherForcedParityError.continuationMismatch
        }

        var maxLogProbabilityError: Float = 0
        var sumLogProbabilityError: Float = 0
        var maxMarginError: Float = 0
        var maxRankDifference = 0
        var swiftArgmaxMatches = 0
        var pythonArgmaxMatches = 0
        var pythonRankSum = 0
        for index in 0..<count {
            let token = score.tokens[index]
            let logProbabilityError = abs(token.logProbability - pythonLogProbabilities[index])
            maxLogProbabilityError = max(maxLogProbabilityError, logProbabilityError)
            sumLogProbabilityError += logProbabilityError
            maxMarginError = max(
                maxMarginError, abs(token.argmaxMargin - pythonMargins[index]))
            maxRankDifference = max(
                maxRankDifference, abs(token.targetRank - Int(pythonRanks[index])))
            if token.argmaxTokenID == token.tokenID { swiftArgmaxMatches += 1 }
            if pythonArgmax[index] == token.tokenID { pythonArgmaxMatches += 1 }
            pythonRankSum += Int(pythonRanks[index])
        }
        let denominator = Float(max(count, 1))
        let pythonAgreement = Float(pythonArgmaxMatches) / denominator
        let pythonMeanRank = Float(pythonRankSum) / denominator

        return .init(
            tokenCount: count,
            continuationIDsMatch: true,
            maxLogProbabilityAbsoluteError: maxLogProbabilityError,
            meanLogProbabilityAbsoluteError: sumLogProbabilityError / denominator,
            maxMarginAbsoluteError: maxMarginError,
            maxTargetRankDifference: maxRankDifference,
            argmaxAgreement: Float(swiftArgmaxMatches) / denominator,
            pythonArgmaxAgreement: pythonAgreement,
            targetRankMeanDifference: abs(score.meanTargetRank - pythonMeanRank))
    }
}
