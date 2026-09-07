import Foundation
import MLX

public enum Qwen4ExpInputMergeError: LocalizedError, Equatable {
    case missingImageTokenID
    case markerCountMismatch(expected: Int, actual: Int)
    case unsupportedBatch(Int)

    public var errorDescription: String? {
        switch self {
        case .missingImageTokenID:
            return "Le checkpoint Flash-Next ne fournit pas image_token_id."
        case .markerCountMismatch(let expected, let actual):
            return "Nombre de marqueurs image incorrect : \(actual), embeddings vision \(expected)."
        case .unsupportedBatch(let batch):
            return "Le mergeur image Flash-Next attend un batch de 1, reçu \(batch)."
        }
    }
}

/// Replace image marker embeddings without splicing the sequence on the host.
/// This keeps token positions and all subsequent cache accounting unchanged.
public enum Qwen4ExpInputMerger {
    public static func merge(
        inputIDs: MLXArray,
        textEmbeddings: MLXArray,
        visionEmbeddings: MLXArray,
        imageTokenID: Int32
    ) throws -> MLXArray {
        guard inputIDs.ndim == 2,
              textEmbeddings.ndim == 3,
              visionEmbeddings.ndim == 3
        else { preconditionFailure("Formes Flash-Next invalides pour le merge image") }
        guard inputIDs.dim(0) == 1 else {
            throw Qwen4ExpInputMergeError.unsupportedBatch(inputIDs.dim(0))
        }
        precondition(textEmbeddings.shape[0] == 1)
        precondition(visionEmbeddings.shape[0] == 1)
        precondition(textEmbeddings.dim(-1) == visionEmbeddings.dim(-1))

        let markerMask = (inputIDs .== imageTokenID).asType(.int32).flattened()
        eval(markerMask)
        let actual = markerMask.asArray(Int32.self).reduce(0) { $0 + Int($1) }
        let expected = visionEmbeddings.dim(1)
        guard actual == expected else {
            throw Qwen4ExpInputMergeError.markerCountMismatch(
                expected: expected, actual: actual)
        }

        let positions = MLX.cumsum(markerMask, axis: 0) - 1
        let aligned = visionEmbeddings.reshaped([-1, visionEmbeddings.dim(-1)])[positions]
        let mask = markerMask.asType(.bool).expandedDimensions(axis: -1)
        return MLX.where(
            mask,
            aligned,
            textEmbeddings.reshaped([-1, textEmbeddings.dim(-1)])
        ).reshaped(textEmbeddings.shape)
    }
}
