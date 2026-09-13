import MLX
import MLXFast
import MLXNN

// P10.2 (F8, 2026-09-12) — RETIRED: a `MLXFast.metalKernel` pair used to
// live here, fusing the hyper-connection "mix" reduction and
// `Qwen4ExpDecoderLayer.inject`'s broadcast-multiply-add (5 ops each) into
// one kernel call. Correct (bit-exact in isolation, bf16-rounding-noise-only
// against the original path at production dtype) but measured on the real
// 3-bit checkpoint, back-to-back alternating with the F7 baseline (6 runs,
// 32-token greedy decode, `flash-chat-probe --fusion-level 7/8`): F7 mean
// 2.372 s, F8 mean 2.392 s — a small, consistent ~1 % **regression**, not
// the required ≥5 % gain (PLAN.md §P10 criterion). Removed rather than kept
// as a dead opt-in, same fate as the P2-fusion `switch_mlp` kernel and the
// per-layer/per-step `MLX.compile` attempts before it (docs/knowledge/log.md,
// "P10.2" and "P7"). See git history for the removed kernel source.

/// Qwen4's four-stream gated residual mixer.
public final class Qwen4ExpGatedResidual: Module {
    public let hiddenSize: Int
    public let streamCount: Int
    private let parityPrefix: String

    /// Optional boundary tensors for public real-checkpoint parity probes.
    public private(set) var lastParityCapture: [String: MLXArray] = [:]
    /// Disabled during normal inference so intermediate graphs are not kept alive.
    public private(set) var captureParity = false

    @ModuleInfo(key: "hc_norm") public var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") public var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") public var inputMixWeightUp: Linear
    @ModuleInfo(key: "block_inject_weight") public var blockInjectWeight: Linear?

    /// P7.1: which sub-block, if any, `callAsFunction` short-circuits for
    /// `flash-layer-bench --ablate`. `.none` everywhere in production —
    /// see `Qwen4ExpLayerBenchAblation`. P11.2 : mutable — voir
    /// `Qwen4ExpSparseMoE.ablation`'s doc comment.
    public private(set) var ablation: Qwen4ExpLayerBenchAblation

    /// F8 (P11-fusion, `--fusion-level 8`) : fermeture `MLX.compile`
    /// mémorisée pour le chemin normal de `callAsFunction` (`hc_norm` +
    /// mélange bas-rang + injection). `nil` tant que F8 n'a pas été demandé
    /// — voir `prepareCompiledPath()`. Jamais consultée par les branches
    /// d'ablation (P7.1) ni quand `captureParity` est actif : le graphe
    /// compilé ne renvoie que `(mixed, injection)`, pas `normed`, que la
    /// capture de parité doit publier.
    private var compiledPath: (@Sendable ([MLXArray]) -> [MLXArray])?

    public init(
        configuration: Qwen4ExpTextConfiguration,
        rmsNormEps: Float = 1e-6,
        useCombine: Bool = true,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        parityPrefix: String = "",
        ablation: Qwen4ExpLayerBenchAblation = .none
    ) {
        hiddenSize = configuration.hiddenSize
        streamCount = configuration.hcCount
        self.parityPrefix = parityPrefix
        self.ablation = ablation
        let streamHiddenSize = streamCount * hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: streamHiddenSize, groupSize: hiddenSize, eps: rmsNormEps)
        _inputMixWeightDown.wrappedValue = qwen4ExpLinear(
            inputDimensions: streamHiddenSize, outputDimensions: configuration.hcLowrank,
            quantization: quantization)
        _inputMixWeightUp.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hcLowrank, outputDimensions: streamHiddenSize,
            quantization: quantization)
        if useCombine {
            _blockInjectWeight.wrappedValue = qwen4ExpLinear(
                inputDimensions: streamHiddenSize, outputDimensions: streamCount,
                quantization: quantization)
        }
        super.init()
    }

    public func callAsFunction(_ hyperInput: MLXArray) -> (
        mixedInput: MLXArray, originalInput: MLXArray, injectionWeights: MLXArray
    ) {
        precondition(hyperInput.ndim == 3)
        precondition(hyperInput.dim(-1) == streamCount * hiddenSize)

        // P7.1 (`--ablate hyper`): skip the mix (`hc_norm` plus the two
        // low-rank gating matmuls), keeping only the plain stream-mean
        // needed for `mixedInput`'s shape and a constant injection-weight
        // tensor of the right shape. `inject` itself keeps running (see
        // `Qwen4ExpLayerBenchAblation`). Never numerically correct.
        if ablation == .hyper {
            let streams = hyperInput.reshaped(
                [hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
            let mixed = streams.mean(axis: -2)
            let injection = MLXArray.ones(
                [hyperInput.dim(0), hyperInput.dim(1), streamCount], dtype: hyperInput.dtype)
            return (mixed, hyperInput, injection)
        }

        // F8 (`--fusion-level 8`) : chemin compilé, seulement hors ablation
        // et hors capture de parité (le graphe compilé n'expose pas
        // `normed`, que `lastParityCapture` doit publier — voir
        // `prepareCompiledPath()`).
        if let compiledPath, ablation == .none, !(captureParity && !parityPrefix.isEmpty) {
            let outputs = compiledPath([hyperInput])
            return (outputs[0], hyperInput, outputs[1])
        }

        // P7.1 (`--ablate norms`): skip hc_norm, reusing its already
        // shape-correct input for the rest of the mix. Never numerically
        // correct.
        let normed = ablation == .norms ? hyperInput : hcNorm(hyperInput)
        let upOut = inputMixWeightUp(silu(inputMixWeightDown(normed) / Float(streamCount)))
        let mixed = (sigmoid(upOut)
            .reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
            * normed.reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize]))
            .mean(axis: -2)
        guard let blockInjectWeight else {
            preconditionFailure("Cette hyper-connexion n'expose pas de combinaison")
        }
        let injection = 2 * sigmoid(blockInjectWeight(normed) / Float(streamCount))
        if captureParity && !parityPrefix.isEmpty {
            lastParityCapture = [
                "\(parityPrefix)normed": normed,
                "\(parityPrefix)mixed": mixed,
                "\(parityPrefix)injection": injection,
            ]
        }
        return (mixed, hyperInput, injection)
    }

    /// F8 (P11-fusion, `--fusion-level 8`) : trace `hc_norm` + le mélange
    /// bas-rang + l'injection en un seul graphe `MLX.compile`, remplaçant
    /// ~24 lancements de noyaux élémentaires par appel (voir le commentaire
    /// de `Qwen4ExpFusionLevel.f8HyperConnectionsCompiled` pour le compte
    /// détaillé). Idempotent — comme `precomputeEffectiveWeight()`, sûr à
    /// appeler plusieurs fois. Doit être appelé après que les poids réels
    /// (et, le cas échéant, la F2 de `hc_norm`) sont en place : les
    /// tenseurs de poids lus par la fermeture au premier passage de trace
    /// sont figés dans le graphe compilé, exactement comme `effectiveWeight`
    /// fige `1 + weight` — voir `Qwen4ExpDecoderLayer.prepareFusion`, où F8
    /// est appliqué textuellement après F2.
    ///
    /// La forme est constante en décodage (`[1, 1, streamCount*hiddenSize]`)
    /// donc pas de `shapeless` ici. Piège documenté, non vérifié sans
    /// checkpoint réel : un préfill (`[1, seq, …]`, longueur variable d'une
    /// conversation à l'autre) et le décodage (`[1, 1, …]`) forcent chacun
    /// leur propre compilation la première fois qu'ils sont vus — MLX met
    /// en cache un graphe par forme rencontrée, donc l'alternance
    /// préfill/décodage à *longueur de préfill fixe* ne recompile qu'une
    /// fois par forme, mais une longueur de préfill différente à chaque
    /// nouvelle conversation recompile à chaque fois. À mesurer : le coût
    /// de cette recompilation au premier jeton d'une conversation.
    public func prepareCompiledPath() {
        guard compiledPath == nil else { return }
        guard blockInjectWeight != nil else {
            preconditionFailure(
                "F8 exige une hyper-connexion avec injection (useCombine), voir prepareCompiledPath()")
        }
        compiledPath = compile { [unowned self] inputs in
            let hyperInput = inputs[0]
            let normed = self.hcNorm(hyperInput)
            let upOut = self.inputMixWeightUp(
                silu(self.inputMixWeightDown(normed) / Float(self.streamCount)))
            let mixed = (sigmoid(upOut)
                .reshaped([hyperInput.dim(0), hyperInput.dim(1), self.streamCount, self.hiddenSize])
                * normed.reshaped(
                    [hyperInput.dim(0), hyperInput.dim(1), self.streamCount, self.hiddenSize]))
                .mean(axis: -2)
            let injection = 2 * sigmoid(self.blockInjectWeight!(normed) / Float(self.streamCount))
            return [mixed, injection]
        }
    }

    public func setParityCapture(_ enabled: Bool) {
        captureParity = enabled
        if !enabled {
            lastParityCapture.removeAll(keepingCapacity: true)
        }
    }

    /// P11.2 : change `ablation` sur une instance déjà construite, sans
    /// recharger aucun poids — voir `Qwen4ExpSparseMoE.setAblation`.
    public func setAblation(_ new: Qwen4ExpLayerBenchAblation) {
        ablation = new
    }

    /// Final four-stream reduction used immediately before the language head.
    /// It intentionally does not allocate or materialize the residual streams.
    public func mixedInput(_ hyperInput: MLXArray) -> MLXArray {
        precondition(hyperInput.ndim == 3)
        precondition(hyperInput.dim(-1) == streamCount * hiddenSize)
        let normed = hcNorm(hyperInput)
        let upOut = inputMixWeightUp(silu(inputMixWeightDown(normed) / Float(streamCount)))
        let mix = sigmoid(upOut)
            .reshaped([hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        let streams = normed.reshaped(
            [hyperInput.dim(0), hyperInput.dim(1), streamCount, hiddenSize])
        return (mix * streams).mean(axis: -2)
    }
}

/// Qwen4 RMSNorm uses zero-centered checkpoint weights.
public final class Qwen4ExpRMSNorm: Module {
    public let eps: Float
    public let groupSize: Int?
    @ParameterInfo(key: "weight") public var weight: MLXArray

    /// F2 (P2-fusion): `1 + weight` baked once by `precomputeEffectiveWeight()`
    /// after the checkpoint (and, where applicable, the Vontra `-1` shift
    /// correction — PLAN.md §6.3 piège 12) has been loaded. Not an
    /// `@ModuleInfo`/`@ParameterInfo` property: it is purely derived from
    /// `weight` and must stay invisible to `parameters()`/`update(parameters:)`.
    /// `nil` means the original per-call `1 + weight` path (unchanged
    /// behavior).
    private var effectiveWeight: MLXArray?

    public init(dimensions: Int, groupSize: Int? = nil, eps: Float = 1e-6) {
        precondition(groupSize == nil || dimensions % groupSize! == 0)
        self.eps = eps
        self.groupSize = groupSize
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        if let effectiveWeight {
            if let groupSize {
                // Grouped case (hc_norm, PLE norms): each of `dimensions /
                // groupSize` groups has its own weight slice, which
                // `MLXFast.rmsNorm`'s single 1-D weight cannot express in one
                // fused call. Keep the manual reduction, but the `1 +`
                // addition and the weight upcast are already baked into
                // `effectiveWeight`, so only the normalization itself still
                // runs per call.
                let inputShape = inputs.shape
                let values = inputs.asType(.float32).reshaped(
                    [inputShape.dropLast().reduce(1, *), inputShape.last! / groupSize, groupSize])
                let groupedWeight = effectiveWeight.reshaped([-1, groupSize])
                let normed = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
                return (normed * groupedWeight).reshaped(inputShape).asType(inputs.dtype)
            }
            // Ungrouped case (q_norm/k_norm, indexer layernorms): a single
            // fused kernel replaces the manual square/mean/rsqrt/mul chain,
            // and MLXFast.rmsNorm handles its own internal precision, so no
            // explicit float32 upcast is needed here either.
            return MLXFast.rmsNorm(inputs, weight: effectiveWeight, eps: eps)
        }
        let inputShape = inputs.shape
        var values = inputs.asType(.float32)
        if let groupSize {
            values = values.reshaped([inputShape.dropLast().reduce(1, *), inputShape.last! / groupSize, groupSize])
            let groupedWeight = weight.reshaped([-1, groupSize]).asType(.float32)
            values = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
            values = values * (1 + groupedWeight)
        } else {
            values = values * MLX.rsqrt((values * values).mean(axis: -1, keepDims: true) + eps)
            values = values * (1 + weight.asType(.float32))
        }
        return values.reshaped(inputShape).asType(inputs.dtype)
    }

    /// F2 (P2-fusion): bake this checkpoint's `1 + weight` convention into a
    /// cached array once, so `callAsFunction` never adds 1 or upcasts the
    /// weight again. Idempotent; safe to call unconditionally — callers gate
    /// it on `Qwen4ExpFusionLevel`, not this method. Must run after any
    /// convention correction (`Qwen4ExpWeightSanitizer`) has already been
    /// applied to `weight`, i.e. after `Module.update(parameters:)`.
    public func precomputeEffectiveWeight() {
        guard effectiveWeight == nil else { return }
        let baked = (1 + weight.asType(.float32)).asType(weight.dtype)
        eval(baked)
        effectiveWeight = baked
    }
}
