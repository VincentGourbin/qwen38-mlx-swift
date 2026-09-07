import MLX
import MLXNN
import Foundation

public struct Qwen4ExpVisionGrid: Sendable, Equatable {
    public let temporal: Int
    public let height: Int
    public let width: Int

    public init(temporal: Int = 1, height: Int, width: Int) {
        precondition(temporal > 0 && height > 0 && width > 0)
        self.temporal = temporal
        self.height = height
        self.width = width
    }
}

public enum Qwen4ExpMRoPEPositionError: LocalizedError, Equatable {
    case missingVisionStart
    case missingImageMarker(Int)
    case imageMarkerCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingVisionStart:
            return "Position MRoPE multimodale : vision_start_token_id absent."
        case .missingImageMarker(let imageIndex):
            return "Position MRoPE multimodale : marqueurs image absents pour l'image \(imageIndex)."
        case .imageMarkerCountMismatch(let expected, let actual):
            return "Position MRoPE multimodale : \(actual) marqueurs, \(expected) attendus."
        }
    }
}

/// Interleaved partial MRoPE used by the QSA indexer and full attention.
///
/// Flash-Next keeps the Qwen3.5 convention: only the first 64 dimensions of
/// the attention head are rotated, with temporal/height/width frequencies
/// interleaved according to `[11, 11, 10]`. Tables are built in float32 even
/// when the model activations are float16/bfloat16.
public final class Qwen4ExpMRoPE: Module {
    public let rotaryDim: Int
    public let mropeSections: [Int]
    // Keep derived RoPE frequencies out of Module parameters. A plain
    // MLXArray stored on a Module is reflected as a parameter by MLXNN, but
    // `inv_freq` is not a checkpoint tensor in qwen4_exp.
    private let invFreq: [Float]

    public init(
        rotaryDim: Int = 64,
        base: Float = 10_000_000,
        mropeSections: [Int] = [11, 11, 10]
    ) {
        precondition(rotaryDim > 0 && rotaryDim.isMultiple(of: 2))
        precondition(mropeSections.count == 3)
        precondition(mropeSections.reduce(0, +) == rotaryDim / 2)
        self.rotaryDim = rotaryDim
        self.mropeSections = mropeSections

        let halfDim = rotaryDim / 2
        self.invFreq = (0 ..< halfDim).map { index in
            1.0 / Foundation.pow(base, (2.0 * Float(index)) / Float(rotaryDim))
        }
        super.init()
    }

    /// Compute interleaved MRoPE cosine and sine tables.
    ///
    /// Accepts text positions `[B,S]` or multimodal positions `[3,B,S]`.
    /// Returns `[B,S,rotaryDim]` for broadcasting over attention heads.
    public func computeCosSin(positionIDs: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let positions: MLXArray
        let batch: Int
        let sequence: Int
        if positionIDs.ndim == 2 {
            batch = positionIDs.dim(0)
            sequence = positionIDs.dim(1)
            let temporal = positionIDs
            let zeros = MLXArray.zeros([batch, sequence], dtype: .int32)
            positions = stacked([temporal, zeros, zeros])
        } else {
            precondition(positionIDs.ndim == 3 && positionIDs.dim(0) == 3)
            batch = positionIDs.dim(1)
            sequence = positionIDs.dim(2)
            positions = positionIDs
        }

        let halfDim = rotaryDim / 2
        let frequencies = (
            MLXArray(invFreq).reshaped([1, 1, halfDim, 1])
                * positions.reshaped([3, batch, 1, sequence]).asType(.float32)
        ).transposed(0, 1, 3, 2)

        var interleaved = frequencies[0]
        for axis in 1..<3 {
            let sectionLength = mropeSections[axis]
            let axisFrequencies = frequencies[axis]
            for sectionIndex in 0..<sectionLength {
                let index = axis + sectionIndex * 3
                if index < halfDim {
                    let value = axisFrequencies[0..., 0..., index..<(index + 1)]
                    if index == 0 {
                        interleaved = concatenated(
                            [value, interleaved[0..., 0..., 1..<halfDim]], axis: -1)
                    } else if index == halfDim - 1 {
                        interleaved = concatenated(
                            [interleaved[0..., 0..., 0..<index], value], axis: -1)
                    } else {
                        interleaved = concatenated(
                            [
                                interleaved[0..., 0..., 0..<index], value,
                                interleaved[0..., 0..., (index + 1)..<halfDim]
                            ], axis: -1)
                    }
                }
            }
        }

        let full = concatenated([interleaved, interleaved], axis: -1)
        return (MLX.cos(full), MLX.sin(full))
    }

    /// Apply the partial GPT-NeoX/rotate-half transform to `[B,H,S,D]`.
    public func apply(_ x: MLXArray, positionIDs: MLXArray) -> MLXArray {
        precondition(x.ndim == 4 && x.dim(-1) >= rotaryDim)
        let (cos, sin) = computeCosSin(positionIDs: positionIDs)
        let rotated = x[.ellipsis, 0..<rotaryDim]
        let passthrough = x[.ellipsis, rotaryDim..<x.dim(-1)]
        let half = rotaryDim / 2
        let x1 = rotated[.ellipsis, 0..<half]
        let x2 = rotated[.ellipsis, half..<rotaryDim]
        let rotateHalf = concatenated([-x2, x1], axis: -1)
        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)
        let output = rotated * cosB + rotateHalf * sinB
        return concatenated([output, passthrough], axis: -1)
    }

    public static func textPositionIDs(sequenceLength: Int, offset: Int = 0) -> MLXArray {
        let temporal = MLXArray(
            stride(from: Int32(offset), to: Int32(offset + sequenceLength), by: 1))
            .reshaped([1, sequenceLength])
        // The Python Qwen path expands scalar text positions to all three
        // MRoPE axes.  Zeroing height/width is tempting for text-only input,
        // but it changes the phase on the interleaved dimensions owned by
        // those axes and breaks the ordinary language-attention path.
        return stacked([temporal, temporal, temporal])
    }

    /// Build Qwen's three MRoPE position rows for a ChatML image sequence.
    /// Grids are expressed in pre-merge patch units and are divided by the
    /// spatial merge factor before the height/width coordinates are emitted.
    public static func multimodalPositionIDs(
        inputIDs: [Int32],
        imageTokenID: Int32,
        visionStartTokenID: Int32,
        grids: [Qwen4ExpVisionGrid],
        spatialMergeSize: Int = 2,
        offset: Int = 0
    ) throws -> MLXArray {
        let actualMarkers = inputIDs.reduce(0) { $0 + ($1 == imageTokenID ? 1 : 0) }
        let expectedMarkers = grids.reduce(0) {
            $0 + $1.temporal * ($1.height / spatialMergeSize) * ($1.width / spatialMergeSize)
        }
        guard actualMarkers == expectedMarkers else {
            throw Qwen4ExpMRoPEPositionError.imageMarkerCountMismatch(
                expected: expectedMarkers, actual: actualMarkers)
        }

        var rows = [[Int32]](repeating: [], count: 3)
        var cursor = 0
        var imageIndex = 0
        var maximum = Int32(offset - 1)
        while imageIndex < grids.count {
            guard let start = inputIDs[cursor...].firstIndex(of: visionStartTokenID) else {
                throw Qwen4ExpMRoPEPositionError.missingVisionStart
            }
            guard let marker = inputIDs[(start + 1)...].firstIndex(of: imageTokenID) else {
                throw Qwen4ExpMRoPEPositionError.missingImageMarker(imageIndex)
            }
            let textLength = marker - cursor
            let textStart = maximum + 1
            appendTextPositions(to: &rows, count: textLength, start: textStart)

            let grid = grids[imageIndex]
            precondition(grid.height % spatialMergeSize == 0)
            precondition(grid.width % spatialMergeSize == 0)
            let gridT = grid.temporal
            let gridH = grid.height / spatialMergeSize
            let gridW = grid.width / spatialMergeSize
            let imageTokenCount = gridT * gridH * gridW
            let imageEnd = marker + imageTokenCount
            guard imageEnd <= inputIDs.count,
                  inputIDs[marker..<imageEnd].allSatisfy({ $0 == imageTokenID })
            else {
                let remaining = inputIDs[marker...].prefix(while: { $0 == imageTokenID }).count
                throw Qwen4ExpMRoPEPositionError.imageMarkerCountMismatch(
                    expected: imageTokenCount, actual: remaining)
            }

            let imageStart = textStart + Int32(textLength)
            for t in 0..<gridT {
                for h in 0..<gridH {
                    for w in 0..<gridW {
                        rows[0].append(imageStart + Int32(t))
                        rows[1].append(imageStart + Int32(h))
                        rows[2].append(imageStart + Int32(w))
                    }
                }
            }
            maximum = imageStart + Int32(max(gridT - 1, max(gridH - 1, gridW - 1)))
            cursor = imageEnd
            imageIndex += 1
        }

        appendTextPositions(to: &rows, count: inputIDs.count - cursor, start: maximum + 1)
        guard rows.allSatisfy({ $0.count == inputIDs.count }) else {
            throw Qwen4ExpMRoPEPositionError.imageMarkerCountMismatch(
                expected: inputIDs.count, actual: rows[0].count)
        }
        return MLXArray(rows.flatMap { $0 }).reshaped([3, 1, inputIDs.count])
    }

    private static func appendTextPositions(
        to rows: inout [[Int32]], count: Int, start: Int32
    ) {
        guard count > 0 else { return }
        let positions = (0..<count).map { start + Int32($0) }
        for axis in 0..<3 { rows[axis].append(contentsOf: positions) }
    }
}
