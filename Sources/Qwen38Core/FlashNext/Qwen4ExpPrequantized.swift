import MLX
import MLXLMCommon
import MLXNN

/// Quantization parameters needed to create a checkpoint-shaped module before
/// its safetensor arrays are attached.  This is deliberately separate from
/// `quantize(model:)`: quantizing a freshly constructed 512-expert MoE would
/// first materialize the enormous floating-point initializer.
public struct Qwen4ExpQuantizationSpec: Sendable, Equatable {
    /// MLX's affine quantization packs `bits`-wide codes into 32-bit words;
    /// these are the widths it (and this codebase's kernels) support. The
    /// real constraint on a given tensor is not "32 % bits == 0" (which
    /// rejects the valid 3-bit case used by the Q3 expert requantification,
    /// since 32 % 3 != 0) but that `inputDimensions * bits % 32 == 0` for
    /// every quantized tensor shape — checked per-call in
    /// `qwen4ExpPackedInput` below.
    private static let supportedBitWidths: Set<Int> = [2, 3, 4, 5, 6, 8]

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int, bits: Int, mode: QuantizationMode = .affine) {
        precondition(groupSize > 0 && Self.supportedBitWidths.contains(bits),
                     "bits doit appartenir à \(Self.supportedBitWidths)")
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    public init?(_ configuration: Qwen4ExpQuantization?) {
        guard let configuration,
              let mode = configuration.mode,
              let parsedMode = QuantizationMode(rawValue: mode)
        else { return nil }
        self.init(
            groupSize: configuration.groupSize,
            bits: configuration.bits,
            mode: parsedMode)
    }

    /// Resolves the spec that packs routed-expert (`switch_mlp`) weights.
    /// A checkpoint's `quantization.experts` override wins when present
    /// (the Q3 3-bit-g64 expert requantification); otherwise the experts
    /// share the outer/global spec, which is the Vontra checkpoint's
    /// current (unchanged) shape.
    public static func experts(from configuration: Qwen4ExpQuantization?) -> Qwen4ExpQuantizationSpec? {
        guard let configuration else { return nil }
        if let override = configuration.experts {
            guard let mode = override.mode,
                  let parsedMode = QuantizationMode(rawValue: mode)
            else { return nil }
            return Qwen4ExpQuantizationSpec(
                groupSize: override.groupSize, bits: override.bits, mode: parsedMode)
        }
        return Qwen4ExpQuantizationSpec(configuration)
    }
}

private func qwen4ExpPackedInput(_ inputDimensions: Int, bits: Int) -> Int {
    precondition((inputDimensions * bits) % 32 == 0,
                 "Les dimensions quantifiées Flash-Next doivent être compatibles avec 32 bits")
    return inputDimensions * bits / 32
}

private func qwen4ExpQuantizedScales(
    outputDimensions: Int, inputDimensions: Int, spec: Qwen4ExpQuantizationSpec
) -> MLXArray {
    precondition(inputDimensions % spec.groupSize == 0)
    return MLXArray.zeros(
        [outputDimensions, inputDimensions / spec.groupSize], dtype: .bfloat16)
}

/// Build a `QuantizedLinear` without quantizing a floating-point initializer.
/// The arrays are placeholders: the loader must replace all of them before
/// inference, and strict `Module.update` catches an incomplete checkpoint.
public func qwen4ExpLinear(
    inputDimensions: Int,
    outputDimensions: Int,
    bias: Bool = false,
    quantization: Qwen4ExpQuantizationSpec?
) -> Linear {
    guard let quantization else {
        return Linear(inputDimensions, outputDimensions, bias: bias)
    }
    let packed = MLXArray.zeros(
        [outputDimensions, qwen4ExpPackedInput(inputDimensions, bits: quantization.bits)],
        dtype: .uint32)
    let scales = qwen4ExpQuantizedScales(
        outputDimensions: outputDimensions, inputDimensions: inputDimensions,
        spec: quantization)
    // Scales and affine biases must not share storage. Module.update assigns
    // both keys independently; aliasing them makes the final value depend on
    // dictionary iteration order and produces nondeterministic matmuls.
    let biases = quantization.mode == .affine
        ? MLXArray.zeros(scales.shape, dtype: .bfloat16) : nil
    let linearBias = bias ? MLXArray.zeros([outputDimensions], dtype: .bfloat16) : nil
    return QuantizedLinear(
        weight: packed,
        bias: linearBias,
        scales: scales,
        biases: biases,
        groupSize: quantization.groupSize,
        bits: quantization.bits,
        mode: quantization.mode)
}

/// Prequantized embedding with the same parameter names and math as MLX's
/// `QuantizedEmbedding`, but without a floating-point source matrix.
public final class Qwen4ExpPrequantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    @ModuleInfo(key: "scales") public var scales: MLXArray
    @ModuleInfo(key: "biases") public var biases: MLXArray?

    public override var shape: (Int, Int) {
        let (count, dimensions) = super.shape
        return (count, dimensions * 32 / bits)
    }

    public init(
        embeddingCount: Int,
        dimensions: Int,
        quantization: Qwen4ExpQuantizationSpec
    ) {
        self.groupSize = quantization.groupSize
        self.bits = quantization.bits
        self.mode = quantization.mode
        precondition(dimensions % quantization.groupSize == 0)
        let packed = MLXArray.zeros(
            [embeddingCount, qwen4ExpPackedInput(dimensions, bits: quantization.bits)],
            dtype: .uint32)
        let scales = MLXArray.zeros(
            [embeddingCount, dimensions / quantization.groupSize], dtype: .bfloat16)
        _scales.wrappedValue = scales
        _biases.wrappedValue = quantization.mode == .affine
            ? MLXArray.zeros(scales.shape, dtype: .bfloat16) : nil
        super.init(weight: packed)
        freeze()
    }

    public override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let inputShape = input.shape
        let flat = input.flattened()
        let dequantized = MLX.dequantized(
            weight[flat],
            scales: scales[flat],
            biases: biases == nil ? nil : biases![flat],
            groupSize: groupSize,
            bits: bits,
            mode: mode)
        return dequantized.reshaped(inputShape + [-1])
    }

    public override func asLinear(_ input: MLXArray) -> MLXArray {
        MLX.quantizedMM(
            input, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Build a checkpoint-shaped quantized `SwitchLinear` without first creating
/// a floating-point expert matrix.
public func qwen4ExpSwitchLinear(
    inputDimensions: Int,
    outputDimensions: Int,
    numExperts: Int,
    quantization: Qwen4ExpQuantizationSpec?
) -> SwitchLinear {
    guard let quantization else {
        return SwitchLinear(
            inputDims: inputDimensions,
            outputDims: outputDimensions,
            numExperts: numExperts,
            bias: false)
    }
    precondition(inputDimensions % quantization.groupSize == 0)
    let packed = MLXArray.zeros(
        [numExperts, outputDimensions,
         qwen4ExpPackedInput(inputDimensions, bits: quantization.bits)],
        dtype: .uint32)
    let scales = MLXArray.zeros(
        [numExperts, outputDimensions, inputDimensions / quantization.groupSize],
        dtype: .bfloat16)
    let biases = quantization.mode == .affine
        ? MLXArray.zeros(scales.shape, dtype: .bfloat16) : nil
    return QuantizedSwitchLinear(
        inputDims: inputDimensions,
        outputDims: outputDimensions,
        numExperts: numExperts,
        weight: packed,
        scales: scales,
        biases: biases,
        groupSize: quantization.groupSize,
        bits: quantization.bits,
        mode: quantization.mode)
}
