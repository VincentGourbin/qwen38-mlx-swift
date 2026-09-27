import Foundation
import MLX
import MLXNN

/// `x · signs`, then a normalized Walsh-Hadamard transform over blocks of
/// `block` along the last axis (forward); the inverse applies the transform
/// first and the signs after. Computed in float32 like the reference
/// runtime (`runtime/runtime.py` in the Bonsai 2 pack), cast back to the
/// input dtype.
func hadamardRotate(_ x: MLXArray, signs: MLXArray, block: Int, inverse: Bool) -> MLXArray {
    precondition(x.dim(-1) % block == 0, "Hadamard block does not divide activation width")
    let shape = x.shape
    var y = x.asType(.float32)
    if !inverse { y = y * signs }
    y = hadamardTransform(y.reshaped([-1, block]), scale: 1 / Float(block).squareRoot())
        .reshaped(shape)
    if inverse { y = y * signs }
    return y.asType(x.dtype)
}

/// A `QuantizedLinear` whose input is first Hadamard-rotated (Bonsai 2 /
/// Prism Hadamard packs). Signs are applied before the transform, matching
/// the projection path of `runtime/runtime.py`.
public final class Qwen38HadamardQuantizedLinear: QuantizedLinear {
    let signs: MLXArray
    let block: Int

    public init(_ source: QuantizedLinear, signs: MLXArray, block: Int) {
        self.signs = signs
        self.block = block
        super.init(
            weight: source.weight, bias: source.bias, scales: source.scales,
            biases: source.biases, groupSize: source.groupSize,
            bits: source.bits, mode: source.mode)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(hadamardRotate(x, signs: signs, block: block, inverse: false))
    }
}

/// An embedding lookup over a Bonsai 2 2-bit packed table, dequantized and
/// then Hadamard-rotated (signs applied after the transform, matching the
/// embedding path of `runtime/runtime.py`).
///
/// `QuantizedEmbedding`'s own initializers re-quantize a floating-point
/// weight, so this subclasses `Embedding` directly and reproduces
/// `QuantizedEmbedding.callAsFunction` instead.
public final class Qwen38HadamardQuantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    @ParameterInfo(key: "scales") public var scales: MLXArray
    @ParameterInfo(key: "biases") public var biases: MLXArray
    let signs: MLXArray
    let block: Int

    public init(_ source: QuantizedEmbedding, signs: MLXArray, block: Int) {
        groupSize = source.groupSize
        bits = source.bits
        mode = source.mode
        self.signs = signs
        self.block = block
        super.init(weight: source.weight)
        _scales.wrappedValue = source.scales
        // Bonsai 2 packs always carry biases (== -scales) alongside scales.
        _biases.wrappedValue = source.biases!
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.flattened()
        let out = dequantized(
            weight[flat], scales: scales[flat], biases: biases[flat],
            groupSize: groupSize, bits: bits, mode: mode
        ).reshaped(shape + [-1])
        return hadamardRotate(out, signs: signs, block: block, inverse: true)
    }
}
