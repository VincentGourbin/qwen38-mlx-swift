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

        // 2026-09-12 : la tour vision remonte du float32 (interpolation des
        // positions, l. 241-242 de Qwen4ExpVisionEncoder). Sans ce cast, l'état
        // caché fusionné devient fp32 et contamine les caches remplis au
        // préfill : toutes les couches — y compris les GDN, dont le coût ne
        // dépend pourtant pas du contexte — décodent ensuite ~1,8× plus
        // lentement. Même nature que la fuite corrigée par F7.
        let vision = visionEmbeddings.dtype == textEmbeddings.dtype
            ? visionEmbeddings
            : visionEmbeddings.asType(textEmbeddings.dtype)
        let positions = MLX.cumsum(markerMask, axis: 0) - 1
        let aligned = vision.reshaped([-1, vision.dim(-1)])[positions]
        let mask = markerMask.asType(.bool).expandedDimensions(axis: -1)
        return MLX.where(
            mask,
            aligned,
            textEmbeddings.reshaped([-1, textEmbeddings.dim(-1)])
        ).reshaped(textEmbeddings.shape)
    }
}
