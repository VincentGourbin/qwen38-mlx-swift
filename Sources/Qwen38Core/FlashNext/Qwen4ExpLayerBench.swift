import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXProfiler

/// P0 — synthetic single-layer micro-bench (no checkpoint, no Lexar access).
///
/// See PLAN.md, "RÉPONSE — chantier P avant G-8 : le GPU est bien inactif,
/// la cause première est la mémoire, protocole court (2026-09-07)", tableau
/// R1, ligne P0. The question this answers: is the Flash-Next compute path
/// itself fast when a layer's weights are already hot in RAM, independent of
/// the 51B-parameter n-gram table, the 77 GB resident checkpoint and the
/// Lexar I/O? If a synthetic layer at real dimensions is slow with the GPU
/// idle, the bottleneck is in the graph/compute path (H-B); if it is fast,
/// the ~25 ms/couche measured in real residency (docs/knowledge/log.md,
/// 2026-09-07) points at memory pressure/paging instead (H-A).

/// Which Flash-Next decoder layer flavor to bench.
public enum Qwen4ExpLayerBenchKind: String, Sendable, CaseIterable {
    case gdn
    case qsa
}

/// Timing and system-metric sample for one synthetic decode step.
public struct Qwen4ExpLayerBenchStepMetrics: Sendable, Equatable {
    public let durationSeconds: Double
    public let cpuPercent: Double
    public let gpuPercent: Double

    public init(durationSeconds: Double, cpuPercent: Double, gpuPercent: Double) {
        self.durationSeconds = durationSeconds
        self.cpuPercent = cpuPercent
        self.gpuPercent = gpuPercent
    }
}

/// Result of benching one layer kind for `measuredSteps` decode steps
/// (warm-up steps are run but not included here).
public struct Qwen4ExpLayerBenchResult: Sendable {
    public let kind: Qwen4ExpLayerBenchKind
    public let steps: [Qwen4ExpLayerBenchStepMetrics]
    public let materializedBytes: Int64

    public init(
        kind: Qwen4ExpLayerBenchKind,
        steps: [Qwen4ExpLayerBenchStepMetrics],
        materializedBytes: Int64
    ) {
        self.kind = kind
        self.steps = steps
        self.materializedBytes = materializedBytes
    }
}

/// Dimensions needed to build a single Flash-Next decoder layer without a
/// checkpoint or `config.json`. `.real` hardcodes the released `qwen4_exp`
/// (Qwen3.8 Flash-Next) checkpoint's dimensions, taken from the
/// `Qwen4ExpTextConfiguration` fixtures already used by
/// Tests/Qwen38Tests/Qwen38Tests.swift ("Le plan Flash-Next sépare cache
/// récurrent et cache QSA", "Le contrat Flash-Next rejette une topologie de
/// couches incohérente").
public struct Qwen4ExpLayerBenchDimensions: Sendable {
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let indexerBudget: Int
    public let indexerCompressRatio: Int
    public let indexerHeadDim: Int
    public let indexerKVHeads: Int
    public let indexerNHeads: Int
    public let hcCount: Int
    public let hcLowrank: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int

    public init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        linearNumKeyHeads: Int,
        linearNumValueHeads: Int,
        linearKeyHeadDim: Int,
        linearValueHeadDim: Int,
        linearConvKernelDim: Int,
        numExperts: Int,
        numExpertsPerToken: Int,
        moeIntermediateSize: Int,
        sharedExpertIntermediateSize: Int,
        indexerBudget: Int,
        indexerCompressRatio: Int,
        indexerHeadDim: Int,
        indexerKVHeads: Int,
        indexerNHeads: Int,
        hcCount: Int,
        hcLowrank: Int,
        vocabSize: Int,
        maxPositionEmbeddings: Int
    ) {
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.indexerHeadDim = indexerHeadDim
        self.indexerKVHeads = indexerKVHeads
        self.indexerNHeads = indexerNHeads
        self.hcCount = hcCount
        self.hcLowrank = hcLowrank
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    /// Real `qwen4_exp` (Qwen3.8 Flash-Next) checkpoint dimensions: hidden
    /// 2560, 4 hyper-connection streams (rank 320), 512 experts of dim 640
    /// (10 routed + 1 shared, intermediate 640), GDN 48 value heads / 16
    /// key heads / head_dim 128 / conv kernel 4, QSA 24 query heads / 2 KV
    /// heads / head_dim 256 (partial rotary 64, handled internally by
    /// `Qwen4ExpQSAAttention` — not a configuration knob).
    public static let real = Qwen4ExpLayerBenchDimensions(
        hiddenSize: 2_560,
        numAttentionHeads: 24,
        numKeyValueHeads: 2,
        headDim: 256,
        linearNumKeyHeads: 16,
        linearNumValueHeads: 48,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4,
        numExperts: 512,
        numExpertsPerToken: 10,
        moeIntermediateSize: 640,
        sharedExpertIntermediateSize: 640,
        indexerBudget: 2_048,
        indexerCompressRatio: 4,
        indexerHeadDim: 128,
        indexerKVHeads: 1,
        indexerNHeads: 4,
        hcCount: 4,
        hcLowrank: 320,
        vocabSize: 248_320,
        maxPositionEmbeddings: 262_144)

    /// A single-layer `Qwen4ExpTextConfiguration` selecting `layerType` at
    /// index 0. P0 never calls `Qwen4ExpConfiguration.validate()`, so this
    /// does not need the real 48-entry `layer_types` array or a PLE layer.
    fileprivate func textConfiguration(
        layerType: Qwen4ExpTextConfiguration.LayerType
    ) -> Qwen4ExpTextConfiguration {
        Qwen4ExpTextConfiguration(
            hiddenSize: hiddenSize,
            numHiddenLayers: 1,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            layerTypes: [layerType],
            fullAttentionInterval: 1,
            linearNumKeyHeads: linearNumKeyHeads,
            linearNumValueHeads: linearNumValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            linearConvKernelDim: linearConvKernelDim,
            numExperts: numExperts,
            numExpertsPerToken: numExpertsPerToken,
            moeIntermediateSize: moeIntermediateSize,
            sharedExpertIntermediateSize: sharedExpertIntermediateSize,
            indexerBudget: indexerBudget,
            indexerCompressRatio: indexerCompressRatio,
            indexerHeadDim: indexerHeadDim,
            indexerKVHeads: indexerKVHeads,
            indexerNHeads: indexerNHeads,
            hcCount: hcCount,
            hcLowrank: hcLowrank,
            ngramSize: 3,
            ngramVocabSizeBase: 20_000_000,
            splitNgramParts: 128,
            pleLayerIDs: [],
            pleConvKernelSize: 4,
            vocabSize: vocabSize,
            maxPositionEmbeddings: maxPositionEmbeddings)
    }
}

/// Builds one Flash-Next decoder layer with weights created directly in
/// their packed 4-bit shapes (`qwen4ExpLinear` / `qwen4ExpSwitchLinear`,
/// see `Qwen4ExpPrequantized.swift` and the test "Les modules Flash-Next
/// peuvent naître directement empaquetés"), then decodes one synthetic
/// token repeatedly to see whether the compute path itself is fast once
/// weights are hot in RAM.
public enum Qwen4ExpLayerBench {
    public static func run(
        kind: Qwen4ExpLayerBenchKind,
        dimensions: Qwen4ExpLayerBenchDimensions = .real,
        warmupSteps: Int = 20,
        measuredSteps: Int = 200,
        quantization: Qwen4ExpQuantizationSpec = Qwen4ExpQuantizationSpec(groupSize: 32, bits: 4),
        profiler: MLXProfiler = .shared
    ) -> Qwen4ExpLayerBenchResult {
        precondition(warmupSteps >= 0, "--warmup doit être positif ou nul")
        precondition(measuredSteps > 0, "--steps doit être positif")
        // Match FlashGenerateProbe: touch the default Metal device before any
        // profiler phase samples system memory, otherwise a fresh process can
        // see an empty device list.
        _ = Device.defaultDevice()

        let layerType: Qwen4ExpTextConfiguration.LayerType =
            kind == .gdn ? .linearAttention : .fullAttention
        let configuration = dimensions.textConfiguration(layerType: layerType)

        // This is THE layer under benchmark: PLAN.md §0 forbids a global
        // `eval(model.parameters())` on Flash-Next (it would materialize the
        // n-gram table), not an isolated bench layer whose entire point is
        // to be materialized once, like the checkpoint loader already does
        // for a real layer (Qwen4ExpCheckpointLayerLoader.load).
        let layer = Qwen4ExpDecoderLayer(
            configuration: configuration,
            layerIndex: 0,
            pleLayerIndex: nil,
            quantization: quantization)
        let weightArrays = layer.parameters().flattened().map { $0.1 }
        eval(weightArrays)
        let materializedBytes = weightArrays.reduce(Int64(0)) { $0 + Int64($1.nbytes) }

        // Same cache selection as `Qwen4ExpStreamingDecoder.makeCache(for:)`
        // (that method is `private`; PLE is intentionally out of scope for
        // P0 — see PLAN.md — so only the GDN/QSA branches are reproduced).
        let cache: any KVCache = kind == .gdn
            ? MambaCache()
            : Qwen4ExpQSAKVCache(
                budget: dimensions.indexerBudget,
                compressRatio: dimensions.indexerCompressRatio)

        let hiddenDimensions = dimensions.hiddenSize * dimensions.hcCount
        let inputIDs = MLXArray([Int32(1)]).reshaped([1, 1])
        let phaseName = "Bench couche \(kind.rawValue)"

        func decodeOneStep() -> Qwen4ExpLayerBenchStepMetrics {
            let hidden = MLXRandom.uniform(
                low: Float(-1), high: Float(1), [1, 1, hiddenDimensions], dtype: .float16)

            let gpuBefore = Double(SystemMetrics.gpuUtilization())
            let cpuBefore = SystemMetrics.processCPUTime()
            let start = ContinuousClock.now

            let output: MLXArray
            if kind == .gdn {
                output = layer(hidden, inputIDs: inputIDs, cache: cache)
            } else {
                let qsaCache = cache as! Qwen4ExpQSAKVCache
                // Same causal mask construction as
                // `Qwen4ExpStreamingDecoder.forward` for a full-attention
                // layer visit.
                let mask = Qwen4ExpQSAAttention.causalMask(
                    batch: 1, queryLength: 1,
                    keyLength: qsaCache.offset + 1, offset: qsaCache.offset)
                output = layer(
                    hidden, inputIDs: inputIDs, mask: mask, cache: cache, positionIDs: nil)
            }
            eval(output)

            let elapsed = ContinuousClock.now - start
            let cpuAfter = SystemMetrics.processCPUTime()
            let gpuAfter = Double(SystemMetrics.gpuUtilization())
            let wallSeconds = elapsed.seconds
            let cpuPercent = wallSeconds > 0 ? (cpuAfter - cpuBefore) / wallSeconds * 100 : 0
            return Qwen4ExpLayerBenchStepMetrics(
                durationSeconds: wallSeconds,
                cpuPercent: cpuPercent,
                gpuPercent: (gpuBefore + gpuAfter) / 2)
        }

        for _ in 0..<warmupSteps {
            _ = decodeOneStep()
        }

        var measured: [Qwen4ExpLayerBenchStepMetrics] = []
        measured.reserveCapacity(measuredSteps)
        for _ in 0..<measuredSteps {
            profiler.start(phaseName)
            let metrics = decodeOneStep()
            profiler.end(phaseName)
            measured.append(metrics)
        }

        return Qwen4ExpLayerBenchResult(
            kind: kind, steps: measured, materializedBytes: materializedBytes)
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
