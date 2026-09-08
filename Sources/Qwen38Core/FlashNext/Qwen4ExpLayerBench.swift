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
    /// Flattened last measured step's output (all bench weights are
    /// deterministic zero placeholders — see `qwen4ExpLinear` — so with the
    /// same RNG seed this is reproducible across eager and `compile`d runs;
    /// used by the P2-code (c) correctness check that `--compiled` does not
    /// silently change the layer's numerics).
    public let lastOutput: [Float]

    public init(
        kind: Qwen4ExpLayerBenchKind,
        steps: [Qwen4ExpLayerBenchStepMetrics],
        materializedBytes: Int64,
        lastOutput: [Float] = []
    ) {
        self.kind = kind
        self.steps = steps
        self.materializedBytes = materializedBytes
        self.lastOutput = lastOutput
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

/// P2-code (c): whether `flash-layer-bench` runs the layer's forward
/// eagerly (baseline) or wrapped in `MLX.compile`. `--compiled` alone means
/// "recompile whenever the traced shapes change" (the QSA mask and cache
/// grow every step, so this is expected to recompile on every call and
/// therefore measure worse than eager — see docs/knowledge/log.md P2-code
/// (c)); `shapeless: true` asks MLX not to recompile purely for a shape
/// change.
public struct Qwen4ExpLayerBenchComputeMode: Sendable, Equatable {
    public var compiled: Bool
    public var shapeless: Bool
    /// P2-code (d): number of decode steps between blocking `eval` calls.
    /// `1` (the default) matches the production decoder's behavior — see
    /// `Qwen4ExpStreamingDecoder.residentEvaluationInterval` and PLAN.md
    /// §6.3 piège 11 — and is unaffected by this bench-only knob. Values
    /// above 1 use `asyncEval` for the steps in between and only block
    /// (`eval`) on the last step of each group, to measure whether
    /// deferring synchronization helps in Release with no memory pressure
    /// (the bench never approaches the ~80 GB resident checkpoint).
    public var syncEvery: Int
    /// P2-code (e): the bench always decodes one token at a time
    /// (`queryLength == 1`), so its QSA causal mask is provably all-true on
    /// every step (see `Qwen4ExpStreamingDecoder.forward`, which applies
    /// this unconditionally in production). `false` (default) reproduces
    /// the bench's original behavior of building that mask anyway, so its
    /// cost can be measured in isolation; `true` skips it, matching what
    /// production now does.
    public var skipTrivialCausalMask: Bool

    public init(
        compiled: Bool = false, shapeless: Bool = false, syncEvery: Int = 1,
        skipTrivialCausalMask: Bool = false
    ) {
        precondition(syncEvery > 0, "syncEvery doit être positif")
        self.compiled = compiled
        self.shapeless = shapeless
        self.syncEvery = syncEvery
        self.skipTrivialCausalMask = skipTrivialCausalMask
    }

    public static let eager = Qwen4ExpLayerBenchComputeMode()
}

/// Boxes a `KVCache` behind MLX's `Updatable` protocol so its arrays can be
/// registered as `compile(inputs:outputs:)` state. `KVCache` and `Updatable`
/// both require only `innerState() -> [MLXArray]`, but Swift does not infer
/// cross-protocol conformance from a matching signature, so this forwards
/// explicitly. `innerState()` is called fresh on every compiled invocation,
/// so it always reflects whatever arrays the cache holds *right now* (the
/// layer replaces `cache[0]`/`cache[1]` with new array objects each step —
/// see `Qwen4ExpGatedDeltaNet.callAsFunction` — rather than mutating them in
/// place, which this indirection makes transparent to `compile`).
private final class Qwen4ExpLayerBenchCacheBox: Updatable {
    let cache: any KVCache
    init(_ cache: any KVCache) { self.cache = cache }
    func innerState() -> [MLXArray] { cache.innerState() }
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
        /// P0/Q3.2: independent spec for the routed experts (`switch_mlp`).
        /// `nil` (the default) means "same as `quantization`" — the bench's
        /// original, unchanged behavior. Passing a 3-bit/g64 spec here lets
        /// `flash-layer-bench --expert-bits 3` measure the QSA/GDN layer
        /// throughput with 3-bit-packed experts without touching the rest
        /// of the layer's quantization.
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil,
        profiler: MLXProfiler = .shared,
        computeMode: Qwen4ExpLayerBenchComputeMode = .eager
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
            quantization: quantization,
            expertsQuantization: expertsQuantization)
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

        // P2-code (c): GDN's cache holds fixed-shape state every step (a
        // constant-width conv window plus a constant-shape recurrent state,
        // see `Qwen4ExpGatedDeltaNet`), so its one-argument forward
        // (hidden -> output) is a plausible `compile` target. QSA's cache
        // instead concatenates a new key/value onto the KV/indexer history
        // every step (`Qwen4ExpQSAKVCache`/`KVCacheSimple.update`) and its
        // causal mask (built in Swift from `cache.offset`) grows with it, so
        // both a positional argument (the mask) and the traced state shape
        // change on every single call — compiled without `shapeless` this
        // recompiles every step; the obstacle and its measured cost are
        // documented in docs/knowledge/log.md P2-code (c) rather than
        // skipped.
        let cacheBox = Qwen4ExpLayerBenchCacheBox(cache)
        let compiledGDNForward: (@Sendable (MLXArray) -> MLXArray)? = {
            guard kind == .gdn, computeMode.compiled else { return nil }
            return compile(
                inputs: [cacheBox], outputs: [cacheBox], shapeless: computeMode.shapeless
            ) { hidden in
                layer(hidden, inputIDs: inputIDs, cache: cache)
            }
        }()
        let compiledQSAForward: (@Sendable (MLXArray, MLXArray) -> MLXArray)? = {
            guard kind == .qsa, computeMode.compiled else { return nil }
            return compile(
                inputs: [cacheBox], outputs: [cacheBox], shapeless: computeMode.shapeless
            ) { hidden, mask in
                layer(hidden, inputIDs: inputIDs, mask: mask, cache: cache, positionIDs: nil)
            }
        }()

        var lastOutput: MLXArray?

        // Builds one step's graph without evaluating it, so both the
        // per-step (`decodeOneStep`) and the grouped `asyncEval` (P2-code
        // (d)) paths share exactly the same forward.
        func buildStep() -> MLXArray {
            let hidden = MLXRandom.uniform(
                low: Float(-1), high: Float(1), [1, 1, hiddenDimensions], dtype: .float16)
            if kind == .gdn {
                if let compiledGDNForward {
                    return compiledGDNForward(hidden)
                }
                return layer(hidden, inputIDs: inputIDs, cache: cache)
            }
            let qsaCache = cache as! Qwen4ExpQSAKVCache
            if computeMode.skipTrivialCausalMask {
                // P2-code (e): queryLength is always 1 in this bench, so
                // this mask is provably all-true — skip building it
                // (matches `Qwen4ExpStreamingDecoder.forward` in
                // production). Not combined with `--compiled`, whose 2-arg
                // QSA closure expects a concrete mask array; eager only.
                return layer(
                    hidden, inputIDs: inputIDs, mask: nil, cache: cache, positionIDs: nil)
            }
            // Same causal mask construction as `Qwen4ExpStreamingDecoder.forward`
            // did before P2-code (e) for a full-attention layer visit.
            let mask = Qwen4ExpQSAAttention.causalMask(
                batch: 1, queryLength: 1,
                keyLength: qsaCache.offset + 1, offset: qsaCache.offset)
            if let compiledQSAForward {
                return compiledQSAForward(hidden, mask)
            }
            return layer(hidden, inputIDs: inputIDs, mask: mask, cache: cache, positionIDs: nil)
        }

        func decodeOneStep() -> Qwen4ExpLayerBenchStepMetrics {
            let gpuBefore = Double(SystemMetrics.gpuUtilization())
            let cpuBefore = SystemMetrics.processCPUTime()
            let start = ContinuousClock.now

            let output = buildStep()
            eval(output)
            lastOutput = output

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

        // P2-code (d): `asyncEval` every step but one in the group, then a
        // single blocking `eval` on the last step — CPU/GPU are sampled
        // once around the whole group and the group's wall time is
        // amortized evenly across its `count` steps so the returned array
        // still has one entry per measured step (unchanged shape for
        // existing callers/tests).
        func decodeStepGroup(_ count: Int) -> [Qwen4ExpLayerBenchStepMetrics] {
            let gpuBefore = Double(SystemMetrics.gpuUtilization())
            let cpuBefore = SystemMetrics.processCPUTime()
            let start = ContinuousClock.now

            var output: MLXArray!
            for stepInGroup in 0..<count {
                output = buildStep()
                if stepInGroup == count - 1 {
                    eval(output)
                } else {
                    asyncEval(output)
                }
            }
            lastOutput = output

            let elapsed = ContinuousClock.now - start
            let cpuAfter = SystemMetrics.processCPUTime()
            let gpuAfter = Double(SystemMetrics.gpuUtilization())
            let wallSeconds = elapsed.seconds
            let cpuPercent = wallSeconds > 0 ? (cpuAfter - cpuBefore) / wallSeconds * 100 : 0
            let gpuPercent = (gpuBefore + gpuAfter) / 2
            let perStepSeconds = wallSeconds / Double(count)
            return Array(
                repeating: Qwen4ExpLayerBenchStepMetrics(
                    durationSeconds: perStepSeconds, cpuPercent: cpuPercent,
                    gpuPercent: gpuPercent),
                count: count)
        }

        for _ in 0..<warmupSteps {
            _ = decodeOneStep()
        }

        var measured: [Qwen4ExpLayerBenchStepMetrics] = []
        measured.reserveCapacity(measuredSteps)
        if computeMode.syncEvery <= 1 {
            for _ in 0..<measuredSteps {
                profiler.start(phaseName)
                let metrics = decodeOneStep()
                profiler.end(phaseName)
                measured.append(metrics)
            }
        } else {
            var remaining = measuredSteps
            while remaining > 0 {
                let groupSize = min(computeMode.syncEvery, remaining)
                profiler.start(phaseName)
                let groupMetrics = decodeStepGroup(groupSize)
                profiler.end(phaseName)
                measured.append(contentsOf: groupMetrics)
                remaining -= groupSize
            }
        }

        return Qwen4ExpLayerBenchResult(
            kind: kind, steps: measured, materializedBytes: materializedBytes,
            lastOutput: lastOutput?.asType(.float32).asArray(Float.self) ?? [])
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
