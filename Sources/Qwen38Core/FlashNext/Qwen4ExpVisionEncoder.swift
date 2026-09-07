import Foundation
import MLX
import MLXFast
import MLXNN

@inline(__always)
private func qwen4ExpVisionGELUTanh(_ inputs: MLXArray) -> MLXArray {
    let coefficient = MLXArray(Float(0.044715))
    let scale = MLXArray(Float(0.7978845608028654))
    return MLXArray(Float(0.5)) * inputs * (
        MLXArray(Float(1)) + MLX.tanh(scale * (inputs + coefficient * inputs * inputs * inputs)))
}

/// The Flash-Next vision tower.  It intentionally mirrors the Qwen-family
/// ViT used by the 27B model: 3D patch projection, learned 2D positions,
/// rotary 2D self-attention, then a 2x2 spatial merger.
public final class Qwen4ExpVisionAttention: Module {
    private let heads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "qkv") public var qkv: Linear
    @ModuleInfo(key: "proj") public var proj: Linear

    public init(configuration: Qwen4ExpVisionConfiguration) {
        self.heads = configuration.numHeads
        self.headDim = configuration.hiddenSize / configuration.numHeads
        self.scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize * 3, bias: true)
        _proj.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, rotary: MLXArray) -> MLXArray {
        let batch = inputs.dim(0)
        let sequence = inputs.dim(1)
        let qkvValues = qkv(inputs).reshaped([batch, sequence, 3, heads, headDim])
        var q = qkvValues[.ellipsis, 0, 0..., 0...]
        var k = qkvValues[.ellipsis, 1, 0..., 0...]
        let v = qkvValues[.ellipsis, 2, 0..., 0...]
        q = applyRotary(q, rotary)
        k = applyRotary(k, rotary)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: scale,
            mask: nil)
        return proj(attended.transposed(0, 2, 1, 3).reshaped([batch, sequence, -1]))
    }

    private func applyRotary(_ inputs: MLXArray, _ frequencies: MLXArray) -> MLXArray {
        let half = inputs.dim(-1) / 2
        let x1 = inputs[.ellipsis, 0..<half]
        let x2 = inputs[.ellipsis, half..<(2 * half)]
        let cosines = MLX.cos(frequencies).reshaped([1, frequencies.dim(0), 1, frequencies.dim(1)])
        let sines = MLX.sin(frequencies).reshaped([1, frequencies.dim(0), 1, frequencies.dim(1)])
        return concatenated([x1 * cosines - x2 * sines, x1 * sines + x2 * cosines], axis: -1)
    }
}

public final class Qwen4ExpVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") public var linearFC1: Linear
    @ModuleInfo(key: "linear_fc2") public var linearFC2: Linear

    public init(configuration: Qwen4ExpVisionConfiguration) {
        _linearFC1.wrappedValue = Linear(
            configuration.hiddenSize, configuration.intermediateSize, bias: true)
        _linearFC2.wrappedValue = Linear(
            configuration.intermediateSize, configuration.hiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        linearFC2(qwen4ExpVisionGELUTanh(linearFC1(inputs)))
    }
}

public final class Qwen4ExpVisionBlock: Module {
    @ModuleInfo(key: "attn") public var attention: Qwen4ExpVisionAttention
    @ModuleInfo(key: "mlp") public var mlp: Qwen4ExpVisionMLP
    @ModuleInfo(key: "norm1") public var norm1: LayerNorm
    @ModuleInfo(key: "norm2") public var norm2: LayerNorm

    public init(configuration: Qwen4ExpVisionConfiguration) {
        _attention.wrappedValue = Qwen4ExpVisionAttention(configuration: configuration)
        _mlp.wrappedValue = Qwen4ExpVisionMLP(configuration: configuration)
        _norm1.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-6)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, rotary: MLXArray) -> MLXArray {
        forwardTrace(inputs, rotary: rotary)["output"]!
    }

    public func forwardTrace(_ inputs: MLXArray, rotary: MLXArray) -> [String: MLXArray] {
        let attentionOutput = attention(norm1(inputs), rotary: rotary)
        let afterAttention = inputs + attentionOutput
        let mlpOutput = mlp(norm2(afterAttention))
        return [
            "attention": attentionOutput,
            "after_attention": afterAttention,
            "mlp": mlpOutput,
            "output": afterAttention + mlpOutput,
        ]
    }
}

public final class Qwen4ExpVisionPatchEmbed: Module {
    private let patchSize: Int
    private let temporalPatchSize: Int
    private let hiddenSize: Int
    private let mergeSize: Int

    @ModuleInfo(key: "proj") public var proj: Conv3d

    public init(configuration: Qwen4ExpVisionConfiguration) {
        patchSize = configuration.patchSize
        temporalPatchSize = configuration.temporalPatchSize
        hiddenSize = configuration.hiddenSize
        mergeSize = configuration.spatialMergeSize
        let kernel: IntOrTriple = [temporalPatchSize, patchSize, patchSize]
        _proj.wrappedValue = Conv3d(
            inputChannels: 3,
            outputChannels: configuration.hiddenSize,
            kernelSize: kernel,
            stride: kernel,
            padding: 0,
            bias: true)
        super.init()
    }

    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        precondition(pixels.ndim == 4 && pixels.dim(-1) == 3)
        let batch = pixels.dim(0)
        let height = pixels.dim(1)
        let width = pixels.dim(2)
        let temporal = concatenated(
            [pixels.expandedDimensions(axis: 1), pixels.expandedDimensions(axis: 1)], axis: 1)
        var hidden = proj(temporal).squeezed(axis: 1)
        let gridH = height / patchSize
        let gridW = width / patchSize
        hidden = hidden.reshaped([batch, gridH / mergeSize, mergeSize, gridW / mergeSize, mergeSize, hiddenSize])
        hidden = hidden.transposed(0, 1, 3, 2, 4, 5)
        return hidden.reshaped([batch, gridH * gridW, hiddenSize])
    }
}

public final class Qwen4ExpVisionMerger: Module {
    private let mergeSize: Int
    private let mergedDimension: Int

    @ModuleInfo(key: "norm") public var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") public var linearFC1: Linear
    @ModuleInfo(key: "linear_fc2") public var linearFC2: Linear

    public init(configuration: Qwen4ExpVisionConfiguration) {
        mergeSize = configuration.spatialMergeSize
        mergedDimension = configuration.hiddenSize * mergeSize * mergeSize
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-6)
        _linearFC1.wrappedValue = Linear(mergedDimension, mergedDimension, bias: true)
        _linearFC2.wrappedValue = Linear(
            mergedDimension, configuration.outHiddenSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let normalized = norm(inputs)
        let batch = normalized.dim(0)
        let patches = normalized.dim(1)
        let merged = normalized.reshaped([batch, patches / (mergeSize * mergeSize), mergedDimension])
        return linearFC2(qwen4ExpVisionGELUTanh(linearFC1(merged)))
    }
}

public final class Qwen4ExpVisionEncoder: Module {
    public let configuration: Qwen4ExpVisionConfiguration
    private let rotaryDim: Int
    private let patchSize: Int

    @ModuleInfo(key: "patch_embed") public var patchEmbed: Qwen4ExpVisionPatchEmbed
    @ParameterInfo(key: "pos_embed") public var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") public var blocks: [Qwen4ExpVisionBlock]
    @ModuleInfo(key: "merger") public var merger: Qwen4ExpVisionMerger

    public init(configuration: Qwen4ExpVisionConfiguration) {
        self.configuration = configuration
        rotaryDim = configuration.hiddenSize / configuration.numHeads
        patchSize = configuration.patchSize
        _patchEmbed.wrappedValue = Qwen4ExpVisionPatchEmbed(configuration: configuration)
        _posEmbed.wrappedValue = MLXArray.zeros(
            [configuration.depth == 0 ? 1 : 2304, configuration.hiddenSize])
        _blocks.wrappedValue = (0..<configuration.depth).map { _ in
            Qwen4ExpVisionBlock(configuration: configuration)
        }
        _merger.wrappedValue = Qwen4ExpVisionMerger(configuration: configuration)
        super.init()
    }

    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        forwardTrace(pixels)["output"]!
    }

    /// Exposes stage outputs for Python/Swift parity diagnosis. This is kept
    /// separate from the hot call path so normal inference does not allocate
    /// a dictionary or retain all 27 block activations.
    public func forwardTrace(_ pixels: MLXArray) -> [String: MLXArray] {
        let gridH = pixels.dim(1) / patchSize
        let gridW = pixels.dim(2) / patchSize
        var hidden = patchEmbed(pixels)
        let position = interpolatedPositionEmbeddings(gridH: gridH, gridW: gridW)
        var trace = ["patch_embed": hidden, "position_embed": position]
        hidden = hidden + position
        trace["pre_block"] = hidden
        let rotary = rotaryEmbeddings(gridH: gridH, gridW: gridW)
        for (index, block) in blocks.enumerated() {
            let blockTrace = block.forwardTrace(hidden, rotary: rotary)
            for (stage, value) in blockTrace {
                trace["block_\(index)_\(stage)"] = value
            }
            hidden = blockTrace["output"]!
            trace["block_\(index)"] = hidden
        }
        trace["output"] = merger(hidden)
        return trace
    }

    private func interpolatedPositionEmbeddings(gridH: Int, gridW: Int) -> MLXArray {
        let sourceSide = Int(sqrt(Float(posEmbed.dim(0))))
        let h = gridH == 1 ? MLXArray([Float(0)]) : MLXArray(
            stride(from: Float(0), through: Float(sourceSide - 1),
                   by: Float(sourceSide - 1) / Float(gridH - 1)))
        let w = gridW == 1 ? MLXArray([Float(0)]) : MLXArray(
            stride(from: Float(0), through: Float(sourceSide - 1),
                   by: Float(sourceSide - 1) / Float(gridW - 1)))
        let hf = h.asType(.int32)
        let wf = w.asType(.int32)
        let hc = MLX.minimum(hf + 1, MLXArray(Int32(sourceSide - 1)))
        let wc = MLX.minimum(wf + 1, MLXArray(Int32(sourceSide - 1)))
        let dh = h - hf.asType(.float32)
        let dw = w - wf.asType(.float32)
        let strideValue = Int32(sourceSide)
        let baseH = hf * strideValue
        let baseHC = hc * strideValue
        let idx00 = (baseH.reshaped([gridH, 1]) + wf.reshaped([1, gridW])).reshaped([-1])
        let idx01 = (baseH.reshaped([gridH, 1]) + wc.reshaped([1, gridW])).reshaped([-1])
        let idx10 = (baseHC.reshaped([gridH, 1]) + wf.reshaped([1, gridW])).reshaped([-1])
        let idx11 = (baseHC.reshaped([gridH, 1]) + wc.reshaped([1, gridW])).reshaped([-1])
        let w00 = ((1 - dh).reshaped([gridH, 1]) * (1 - dw).reshaped([1, gridW])).reshaped([-1])
        let w01 = ((1 - dh).reshaped([gridH, 1]) * dw.reshaped([1, gridW])).reshaped([-1])
        let w10 = (dh.reshaped([gridH, 1]) * (1 - dw).reshaped([1, gridW])).reshaped([-1])
        let w11 = (dh.reshaped([gridH, 1]) * dw.reshaped([1, gridW])).reshaped([-1])
        var result = posEmbed[idx00] * w00.expandedDimensions(axis: -1)
        result = result + posEmbed[idx01] * w01.expandedDimensions(axis: -1)
        result = result + posEmbed[idx10] * w10.expandedDimensions(axis: -1)
        result = result + posEmbed[idx11] * w11.expandedDimensions(axis: -1)
        result = result.reshaped([gridH, gridW, configuration.hiddenSize])
        result = result.reshaped([
            gridH / configuration.spatialMergeSize, configuration.spatialMergeSize,
            gridW / configuration.spatialMergeSize, configuration.spatialMergeSize,
            configuration.hiddenSize])
        return result.transposed(0, 2, 1, 3, 4).reshaped([1, -1, configuration.hiddenSize])
    }

    private func rotaryEmbeddings(gridH: Int, gridW: Int) -> MLXArray {
        let dim = rotaryDim / 2
        let frequencies = 1.0 / MLX.pow(
            MLXArray(Float(10000)),
            MLXArray(stride(from: Float(0), to: Float(dim), by: 2)) / Float(dim))
        let table = MLX.outer(
            MLXArray(stride(from: Float(0), to: Float(max(gridH, gridW)), by: 1)), frequencies)
        var hs: [Int32] = []
        var ws: [Int32] = []
        for blockH in 0..<(gridH / 2) {
            for blockW in 0..<(gridW / 2) {
                for innerH in 0..<2 {
                    for innerW in 0..<2 {
                        hs.append(Int32(blockH * 2 + innerH))
                        ws.append(Int32(blockW * 2 + innerW))
                    }
                }
            }
        }
        return concatenated([table[MLXArray(hs)], table[MLXArray(ws)]], axis: -1)
    }
}
