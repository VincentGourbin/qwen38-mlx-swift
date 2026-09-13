import Foundation
import MLX
import MLXLMCommon
import MLXProfiler

/// Controls whether decoder weights are released after each layer or retained
/// for subsequent autoregressive tokens.  Resident mode is experimental: the
/// Flash-Next checkpoint is large and must be enabled explicitly by a caller
/// that has measured its available memory budget.
public enum Qwen4ExpLayerLoadingMode: String, Sendable, Equatable {
    case streamed
    case resident
}

/// Immutable-at-the-call-site copy of every cache currently owned by the
/// streaming decoder.  The values are copied through `KVCache.copy()` so a
/// speculative forward cannot mutate the snapshot through a shared cache
/// object.  The type is unchecked-sendable for the same reason as the
/// decoder: MLX arrays are intentionally confined to the owning runtime.
public final class Qwen4ExpStreamingDecoderSnapshot: @unchecked Sendable {
    fileprivate let caches: [Int: any KVCache]
    /// P5.1: total device bytes held by this snapshot's copied caches
    /// (`KVCache.state` is the generic array surface every cache family —
    /// `ArraysCache`, `MambaCache`, `Qwen4ExpQSAKVCache` — already exposes).
    /// Summed once at snapshot time so a caller (the server's LRU, P5.2)
    /// never has to walk the cache dictionary itself.
    public let byteCount: Int

    fileprivate init(caches: [Int: any KVCache]) {
        self.caches = caches
        self.byteCount = caches.values.reduce(0) { total, cache in
            total + cache.state.reduce(0) { $0 + $1.nbytes }
        }
    }
}

/// A layer-at-a-time Flash-Next decoder used to stay below the 96 GB machine
/// budget while the full resident model is being assembled.
public final class Qwen4ExpStreamingDecoder: @unchecked Sendable {
    public let configuration: Qwen4ExpTextConfiguration
    public let directory: URL
    public let quantization: Qwen4ExpQuantizationSpec?
    public let layerLoadingMode: Qwen4ExpLayerLoadingMode
    /// Number of resident layers between MLX evaluation checkpoints.
    ///
    /// Measured 2026-09-05: batching multiple layers before `eval()` does NOT
    /// reduce synchronization overhead — it does the opposite. Deferring 8
    /// layers into one lazy graph produced highly variable, sometimes
    /// catastrophic decode stalls (observed 841s-1436s for 7 decode tokens),
    /// while evaluating every single layer (interval 1) was both fast and
    /// stable (28s for the same 7 tokens, ~30-50x faster). Keep this at 1
    /// unless a future measurement on a different checkpoint/machine shows
    /// otherwise; see docs/knowledge/log.md 2026-09-05.
    public let residentEvaluationInterval: Int
    /// Opt-in per-layer profiling (P0-c). When `false` (the default),
    /// `forward` skips the `profiler.start`/`.end("Flash couche N")` pair
    /// and the per-layer n-gram cache bookkeeping (`observeNGramCache`) for
    /// every decoder layer. Each `start`/`end` pair costs ~4.7 ms (IOKit GPU
    /// sample + `rusage`, see docs/knowledge/log.md "P0 rejoué en Release"),
    /// i.e. ~225 ms/token spread over 48 layers when left on unconditionally
    /// — pure profiler overhead, not decoder cost. The `Prefill`/`Generation`
    /// phases and session metadata (model, prompt tokens, layer visits/load
    /// time…) are unaffected: they live in the generators, not here.
    public let profileLayers: Bool
    /// P2-code (d) → P1 : in resident mode, dispatch every intermediate layer
    /// with `asyncEval` (GPU work starts immediately, the host goes on
    /// building the next layer) and block only on the last layer of the
    /// visit. Measured on the synthetic bench (Release): −13 to −17 % per
    /// layer and GPU busy 82 % (ioreg) instead of ~0 %. Off by default until
    /// P1 has measured it on the real checkpoint; `residentEvaluationInterval`
    /// keeps its meaning (blocking `eval` checkpoints) when this is false.
    ///
    /// P4.0 found that this flag, on its own, does nothing at the production
    /// default `residentEvaluationInterval == 1`: `shouldEvaluate` reduces to
    /// `(visitIndex + 1) % 1 == 0`, which is `true` for every integer, so the
    /// blocking `eval` branch fired on *every* layer regardless of this flag
    /// and the `asyncEval` branch below was dead code. `residentAsyncInterval`
    /// (P4.1) is the real, separate knob for this flag's own frequency;
    /// `residentEvaluationInterval` is untouched and stays at 1 everywhere in
    /// production (piège 11 — it still governs the dangerous fully-deferred
    /// path taken when this flag is `false`).
    public let residentAsyncEval: Bool
    /// P4.1: when `residentAsyncEval` is `true`, the number of resident
    /// layers between blocking `eval` checkpoints — intermediate layers get
    /// `asyncEval` (dispatched to the GPU immediately, non-blocking) instead.
    /// This is *not* the same mechanism as `residentEvaluationInterval`
    /// batching without any eval/asyncEval at all (piège 11's 30-50×
    /// regression, reconfirmed by P1 variant (iii)): every visited layer
    /// still gets a real `eval`/`asyncEval` call here, just not always a
    /// blocking one. The last layer of every visit is always blocking,
    /// regardless of this value. Default 1 reproduces exactly the blocking-
    /// every-layer behavior this flag had before P4.1 fixed the dead-code
    /// bug above, so existing callers that never touch this parameter see no
    /// behavior change.
    public let residentAsyncInterval: Int
    /// P2-mem-a: read resident tensors with `pread` behind `fcntl(F_NOCACHE,
    /// 1)` (`Qwen4ExpUncachedTensorReader`) instead of `loadArraysAndMetadata`,
    /// so the 50-80 GB read from the Lexar during a resident load never fill
    /// the kernel's file cache (measured +30-40 GB otherwise; see
    /// docs/knowledge/log.md "H6 : deux tentatives", 2026-09-08 soir, and the
    /// P2-mem-a entry). Defaults to `true`: validated bit-exact against the
    /// old path with unchanged peak MLX/RSS.
    public let uncachedIO: Bool
    /// P2-fusion (F1-F6, docs/knowledge/log.md "P2-fusion"): kernel-reduction
    /// level applied to every layer right after it loads. `.none` (the
    /// default) is the original path, unchanged until F7 validates the fused
    /// path bit-for-bit on the real checkpoint.
    public let fusionLevel: Qwen4ExpFusionLevel
    /// P11.1 : largeur de routage MoE effective (`num_experts_per_tok` du
    /// checkpoint, sauf surcharge à la construction ou via
    /// `updateRoutedExpertCount`). Toujours une valeur concrète et déjà
    /// validée — jamais `nil` — même quand aucune surcharge n'a été
    /// demandée, pour que ce champ soit directement publiable dans
    /// `/healthz`/les métriques sans distinguer "réglé" de "défaut".
    public private(set) var routedExpertCount: Int
    /// P11.2 : quel sous-bloc, le cas échéant, chaque couche court-circuite
    /// (`--ablate` de `flash-chat-probe`, jamais réglé côté serveur sans
    /// `--allow-ablation`). `.none` reproduit exactement le comportement
    /// précédent partout ailleurs. Toujours une valeur concrète — jamais
    /// `nil` — pour la même raison de publication que `routedExpertCount`.
    public private(set) var ablation: Qwen4ExpLayerBenchAblation

    private let checkpointIndex: Qwen4ExpCheckpointLayerIndex
    private var caches: [Int: any KVCache] = [:]
    private var residentLayers: [Int: Qwen4ExpLoadedDecoderLayer] = [:]
    private var ngramCacheTotals = Qwen4ExpNGramCacheStats()
    private var ngramCacheEntriesByLayer: [Int: Int] = [:]
    private var residentNGramCacheSnapshots: [Int: Qwen4ExpNGramCacheStats] = [:]
    /// P6.2: same delta-accounting pattern as the n-gram cache counters
    /// above, for the lookup-path instrumentation (arrays constructed,
    /// dequantize calls, host read time).
    private var ngramLookupTotals = Qwen4ExpPLELookupStats()
    private var residentNGramLookupSnapshots: [Int: Qwen4ExpPLELookupStats] = [:]

    public init(
        directory: URL,
        layerLoadingMode: Qwen4ExpLayerLoadingMode = .streamed,
        residentEvaluationInterval: Int = 1,
        profileLayers: Bool = false,
        residentAsyncEval: Bool = false,
        residentAsyncInterval: Int = 1,
        uncachedIO: Bool = true,
        fusionLevel: Qwen4ExpFusionLevel = .f7GatedBranchDtype,
        /// P11.1 : surcharge de `num_experts_per_tok`. `nil` (le défaut)
        /// laisse le comportement inchangé — la valeur du checkpoint est
        /// utilisée telle quelle, exactement comme avant cette option.
        /// Validée ici (contre `numExperts` du checkpoint tout juste
        /// chargé) via `qwen4ExpResolveRoutedExpertCount`, qui lève une
        /// erreur claire plutôt qu'un crash pour une valeur hors bornes.
        routedExpertCount: Int? = nil,
        /// P11.2 : quel sous-bloc, le cas échéant, court-circuiter dans
        /// chaque couche — `.none` (le défaut) laisse le comportement
        /// inchangé.
        ablation: Qwen4ExpLayerBenchAblation = .none
    ) throws {
        precondition(residentEvaluationInterval > 0)
        precondition(residentAsyncInterval > 0)
        self.directory = directory
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        self.configuration = configuration.textConfiguration
        self.quantization = Qwen4ExpQuantizationSpec(configuration.quantization)
        self.layerLoadingMode = layerLoadingMode
        self.residentEvaluationInterval = residentEvaluationInterval
        self.profileLayers = profileLayers
        self.residentAsyncEval = residentAsyncEval
        self.residentAsyncInterval = residentAsyncInterval
        self.uncachedIO = uncachedIO
        self.fusionLevel = fusionLevel
        self.routedExpertCount = try qwen4ExpResolveRoutedExpertCount(
            override: routedExpertCount,
            checkpointDefault: self.configuration.numExpertsPerToken,
            numExperts: self.configuration.numExperts)
        self.ablation = ablation
        self.checkpointIndex = try Qwen4ExpCheckpointLayerIndex(directory: directory)
    }

    /// Run selected source layers while keeping the recurrent/QSA caches alive.
    ///
    /// `hiddenStates` is already the four-stream `[B,S,4H]` representation.
    /// This seam deliberately does not own token embeddings or the final head;
    /// it is the memory-safe decoder core that those runtime pieces will call.
    public func forward(
        _ hiddenStates: MLXArray,
        inputIDs: MLXArray,
        layerIndices: [Int],
        positionIDs: MLXArray? = nil,
        materializeLayers: Bool = true,
        // PM4.2 (P-MTP suite): non-nil only for an MTP verification forward.
        // Every GDN/PLE layer visited records into it the materials needed
        // to roll its `ArraysCache` back to any prefix of the newly fed
        // tokens — see `rollbackVerification` and `Qwen4ExpVerificationCapture`.
        verificationCapture: Qwen4ExpVerificationCapture? = nil,
        onLayerVisited: (@Sendable (Int) -> Void)? = nil,
        /// P12.2 : décalage à gauche de chaque ligne du lot (voir
        /// `Qwen4ExpBatchPaddingLayout`), constant pour toute la durée de la
        /// conversation — un lot est constitué une fois, au départ (pas
        /// d'ordonnanceur ici). `nil` (le défaut, tout appelant existant)
        /// laisse le comportement inchangé : ni masque de remplissage GDN/PLE,
        /// ni restriction de validité de clé QSA au-delà de la restriction
        /// causale usuelle.
        leftPadding: [Int]? = nil
    ) throws -> (output: MLXArray, reports: [Qwen4ExpStreamingLayerReport]) {
        precondition(hiddenStates.ndim == 3)
        precondition(hiddenStates.dim(-1) == configuration.hiddenSize * configuration.hcCount)
        precondition(inputIDs.ndim == 2)
        precondition(hiddenStates.dim(0) == inputIDs.dim(0))
        precondition(hiddenStates.dim(1) == inputIDs.dim(1))

        var hidden = hiddenStates
        var reports: [Qwen4ExpStreamingLayerReport] = []
        let synchronizeEachLayer = layerLoadingMode == .streamed
        for (visitIndex, layerIndex) in layerIndices.enumerated() {
            precondition(configuration.layerTypes.indices.contains(layerIndex))
            let profiler = MLXProfiler.shared
            let profileName = "Flash couche \(layerIndex)"
            if profileLayers {
                profiler.start(profileName)
            }
            defer {
                if profileLayers {
                    profiler.end(profileName)
                }
            }
            // In streamed mode the previous layer is released before the next
            // shard group is opened, so clear its now-unused allocations. In
            // resident mode every layer stays live and clearing the allocator
            // here only adds a device-wide synchronization once per layer.
            if synchronizeEachLayer {
                Memory.clearCache()
            }
            let loaded: Qwen4ExpLoadedDecoderLayer
            let loadDuration: Duration
            if layerLoadingMode == .resident,
               let existing = residentLayers[layerIndex] {
                loaded = existing
                loadDuration = .zero
            } else {
                let start = ContinuousClock.now
                loaded = try Qwen4ExpCheckpointLayerLoader.load(
                    layerIndex,
                    from: directory,
                    index: checkpointIndex,
                    materialize: materializeLayers,
                    uncachedIO: uncachedIO,
                    fusionLevel: fusionLevel,
                    routedExpertCount: routedExpertCount,
                    ablation: ablation)
                loadDuration = ContinuousClock.now - start
                if layerLoadingMode == .resident {
                    residentLayers[layerIndex] = loaded
                }
            }

            let cache: any KVCache
            if let existing = caches[layerIndex] {
                cache = existing
            } else {
                let created = makeCache(for: layerIndex)
                caches[layerIndex] = created
                cache = created
            }

            let isFullAttention = configuration.layerTypes[layerIndex] == .fullAttention
            let attentionMask: MLXArray?
            if let leftPadding {
                // P12.2 : lot à longueurs inégales, rempli à gauche — voir
                // `Qwen4ExpBatchPaddingLayout`. Le remplissage laisse des
                // colonnes de cache définitivement invalides pour la durée
                // de la conversation : contrairement au chemin par défaut
                // ci-dessous, P2-code (e) ne s'applique plus jamais (même à
                // un seul jeton décodé, le masque n'est pas trivialement
                // vrai partout) et le masque GDN/PLE `[B,S]`, documenté mais
                // jamais alimenté jusqu'ici (« reserved for padded/ragged
                // batches »), est enfin construit.
                if isFullAttention {
                    attentionMask = Qwen4ExpQSAAttention.causalMask(
                        batch: inputIDs.dim(0), queryLength: inputIDs.dim(1),
                        keyLength: cache.offset + inputIDs.dim(1), offset: cache.offset,
                        leftPadding: leftPadding)
                } else if inputIDs.dim(1) > 1 {
                    // Le remplissage n'existe que dans le bloc préremplissage
                    // (un seul appel multi-jeton, à décalage absolu 0 — un
                    // lot est constitué une fois, au départ). Les pas de
                    // décodage suivants n'introduisent plus jamais de
                    // remplissage : chaque ligne y avance d'exactement un
                    // jeton réel, donc `inputIDs.dim(1) == 1` n'a besoin
                    // d'aucun masque GDN/PLE.
                    attentionMask = qwen4ExpBatchPaddingValidityMaskArray(
                        leftPadding: leftPadding, columnOffset: 0, columnCount: inputIDs.dim(1))
                } else {
                    attentionMask = nil
                }
            } else if !isFullAttention {
                // GDN's recurrence is causal by construction.  Its optional
                // mask has a different [B,S] contract and is reserved for
                // padded/ragged batches.
                attentionMask = nil
            } else if inputIDs.dim(1) == 1 {
                // P2-code (e): single-token decode has exactly one query
                // position (`cache.offset`), and every cached key position
                // is <= cache.offset < cache.offset + 1 — the causal mask
                // built by `Qwen4ExpQSAAttention.causalMask` is therefore
                // provably all-true for every step, and an all-true boolean
                // mask is numerically the same as no mask at all to SDPA.
                // Skip constructing it: `causalMask` allocates a fresh
                // `Int32` range array of length `cache.offset + 1` — growing
                // with every decoded token — purely to compare it against
                // itself and get back "true" (see docs/knowledge/log.md
                // P2-code (a)/(e)). Multi-token calls (prefill, MTP
                // verification) still build the real mask below.
                attentionMask = nil
            } else {
                attentionMask = Qwen4ExpQSAAttention.causalMask(
                    batch: inputIDs.dim(0), queryLength: inputIDs.dim(1),
                    keyLength: cache.offset + inputIDs.dim(1), offset: cache.offset)
            }

            let verificationSink = verificationCapture.map {
                Qwen4ExpVerificationSink(layerIndex: layerIndex, capture: $0)
            }
            let forwardStart = ContinuousClock.now
            let output = loaded.layer(
                hidden,
                inputIDs: inputIDs,
            mask: attentionMask,
            cache: cache,
            positionIDs: positionIDs,
            verificationSink: verificationSink)
            // P5.5: bookkeeping only (a dictionary/struct read on the PLE's
            // already-resident LRU state, no GPU op) — always on, unlike the
            // `profiler.start`/`.end` pair above which stays gated by
            // `profileLayers` (P0-c's ~4.7 ms/boundary cost). Before this fix
            // `model.ngramCacheStats()` — and therefore the generators'
            // `recordNGramCacheStats` counter published into the shared
            // profiling session — stayed at zero unless `--profile-layers`
            // was also on (docs/knowledge/log.md 2026-09-10, "Dialogue A/B").
            if let currentNGramStats = loaded.layer.ngramCacheStats() {
                observeNGramCache(layerIndex: layerIndex, current: currentNGramStats)
            }
            if let currentLookupStats = loaded.layer.ngramLookupStats() {
                observeNGramLookup(layerIndex: layerIndex, current: currentLookupStats)
            }
            // Streamed mode must detach the next layer from the previous
            // module before that module is released. Resident mode keeps all
            // modules alive and checkpoints the lazy graph periodically. The
            // final layer is always materialized for a stable result.
            let isLastVisitedLayer = visitIndex == layerIndices.count - 1
            let shouldEvaluate: Bool
            if synchronizeEachLayer {
                shouldEvaluate = true
            } else if layerLoadingMode == .resident, residentAsyncEval {
                // P4.1: every visited layer gets a real eval/asyncEval call
                // below (never a fully-deferred graph — piège 11), but only
                // every `residentAsyncInterval` layers (and always the last
                // one) blocks the host. `residentEvaluationInterval` is
                // deliberately not consulted on this branch.
                shouldEvaluate =
                    isLastVisitedLayer || (visitIndex + 1) % residentAsyncInterval == 0
            } else if layerLoadingMode == .resident {
                shouldEvaluate =
                    (visitIndex + 1) % residentEvaluationInterval == 0 || isLastVisitedLayer
            } else {
                shouldEvaluate = false
            }
            if shouldEvaluate {
                eval(output)
            } else if layerLoadingMode == .resident, residentAsyncEval {
                asyncEval(output)
            }
            // Audit des dtypes (QWEN38_DTYPE_AUDIT=1) : deux fuites fp32 ont
            // coûté un facteur 2 chacune (F7 pour les normes GDN/QSA, la tour
            // vision pour le merge image). Ce contrôle imprime, pour la
            // première visite de chaque couche, le dtype de l'état caché en
            // entrée/sortie et celui des tenseurs du cache — tout ce qui
            // propage. Un `float32` ici est une fuite ; le fp32 interne
            // (état récurrent GDN, tables RoPE, scores QSA) est voulu.
            if Qwen4ExpDtypeAudit.isEnabled {
                Qwen4ExpDtypeAudit.report(
                    layer: layerIndex,
                    kind: configuration.layerTypes[layerIndex] == .fullAttention ? "QSA" : "GDN",
                    input: hidden, output: output, cache: cache)
            }
            let forwardDuration = ContinuousClock.now - forwardStart
            hidden = output

            reports.append(
                Qwen4ExpStreamingLayerReport(
                    layerIndex: layerIndex,
                    loadedTensorCount: loaded.tensorCount,
                    loadedShardCount: loaded.shardCount,
                    materializedBytes: loaded.materializedBytes,
                    loadDuration: loadDuration.seconds,
                    forwardDuration: forwardDuration.seconds,
                    outputShape: output.shape))
            onLayerVisited?(layerIndex)
            // `loaded` goes out of scope at the end of this iteration. The
            // cache is intentionally retained for the next conversation turn;
            // the next iteration clears only allocations no longer referenced.
        }
        return (hidden, reports)
    }

    public func resetCaches() {
        caches.removeAll(keepingCapacity: true)
        Memory.clearCache()
    }

    /// P11.1 : change la largeur de routage MoE effective sans recharger le
    /// checkpoint. `override` suit le même contrat qu'à la construction
    /// (`nil` revient à la valeur du checkpoint) ; la valeur résolue est
    /// renvoyée pour qu'un appelant (le serveur, `/healthz`) puisse la
    /// publier immédiatement.
    ///
    /// En mode streamé, seul `routedExpertCount` change : le prochain appel
    /// à `Qwen4ExpCheckpointLayerLoader.load` en tiendra compte. En mode
    /// résident, les couches déjà chargées ne repasseraient jamais par ce
    /// loader (`forward` réutilise `residentLayers`) — cette méthode répercute
    /// donc aussi le changement directement sur chaque couche déjà résidente
    /// via `Qwen4ExpDecoderLayer.setRoutedExpertCount`, qui ne touche à aucun
    /// poids ni au graphe MLX (voir son commentaire).
    @discardableResult
    public func updateRoutedExpertCount(_ override: Int?) throws -> Int {
        let resolved = try qwen4ExpResolveRoutedExpertCount(
            override: override,
            checkpointDefault: configuration.numExpertsPerToken,
            numExperts: configuration.numExperts)
        for loaded in residentLayers.values {
            try loaded.layer.setRoutedExpertCount(resolved)
        }
        routedExpertCount = resolved
        return resolved
    }

    /// P11.2 : change l'ablation effective sans recharger le checkpoint —
    /// même mécanisme que `updateRoutedExpertCount`. En mode streamé, seul
    /// `ablation` change (le prochain `Qwen4ExpCheckpointLayerLoader.load`
    /// en tiendra compte) ; en mode résident, chaque couche déjà chargée
    /// est aussi mise à jour directement via `Qwen4ExpDecoderLayer.
    /// setAblation`, qui ne touche à aucun poids ni au graphe MLX.
    /// Contrairement à `updateRoutedExpertCount`, aucune borne à valider :
    /// toute valeur de `Qwen4ExpLayerBenchAblation` est acceptable.
    public func updateAblation(_ new: Qwen4ExpLayerBenchAblation) {
        for loaded in residentLayers.values {
            loaded.layer.setAblation(new)
        }
        ablation = new
    }

    public func ngramCacheStats() -> Qwen4ExpNGramCacheStats {
        var result = ngramCacheTotals
        result.entries = ngramCacheEntriesByLayer.values.reduce(0, +)
        return result
    }

    public func resetNGramCacheStats() {
        ngramCacheTotals = Qwen4ExpNGramCacheStats()
        ngramCacheEntriesByLayer.removeAll(keepingCapacity: true)
        // The snapshots are baselines for delta accounting in resident mode.
        // They must be reset together with the public counters; otherwise the
        // first forward of the next generation can be silently under-counted.
        residentNGramCacheSnapshots.removeAll(keepingCapacity: true)
    }

    public func ngramLookupStats() -> Qwen4ExpPLELookupStats {
        ngramLookupTotals
    }

    public func resetNGramLookupStats() {
        ngramLookupTotals = Qwen4ExpPLELookupStats()
        residentNGramLookupSnapshots.removeAll(keepingCapacity: true)
    }

    /// Capture all hybrid attention state before a speculative verification
    /// forward.  QSA carries both its dense K/V history and its indexer
    /// side-channel; `copy()` preserves both.  GDN and PLE caches are copied
    /// through their public `KVCache` surface as well.
    public func snapshot() -> Qwen4ExpStreamingDecoderSnapshot {
        Qwen4ExpStreamingDecoderSnapshot(
            caches: caches.mapValues { $0.copy() })
    }

    /// Restore a snapshot after a rejected speculative suffix.  Replacing
    /// cache objects, instead of assigning individual arrays into the live
    /// objects, prevents a partially restored layer from being observed by a
    /// later layer in the same round.
    public func restore(_ snapshot: Qwen4ExpStreamingDecoderSnapshot) {
        caches = snapshot.caches.mapValues { $0.copy() }
        Memory.clearCache()
    }

    /// PM4.2 (P-MTP suite): roll every cache touched by the last
    /// verification forward back to the state after exactly
    /// `committedNewTokens` of the `totalNewTokens` tokens that forward fed,
    /// without replaying any forward pass.
    ///
    /// - GDN/PLE (`ArraysCache`): reconstructed from `capture.entries`, a
    ///   cheap host-side slice of tensors the verification forward already
    ///   materialized (see `Qwen4ExpVerificationCapture`).
    /// - QSA (`Qwen4ExpQSAKVCache`): `trim(rejected)` — its backing arrays
    ///   only ever grow and the visible length is tracked by `offset`, so no
    ///   capture is needed.
    /// A no-op when nothing was rejected.
    public func rollbackVerification(
        capture: Qwen4ExpVerificationCapture,
        committedNewTokens: Int,
        totalNewTokens: Int
    ) {
        let rejected = totalNewTokens - committedNewTokens
        guard rejected > 0 else { return }
        precondition(committedNewTokens >= 1, "Le token bonus doit toujours être conservé")

        for (layerIndex, slots) in capture.entries {
            guard let arrayCache = caches[layerIndex] as? ArraysCache else { continue }
            for (slot, entry) in slots {
                switch entry {
                case .window(let source, let length):
                    // Only axis 1 (the token axis) is indexed explicitly;
                    // trailing feature axes (present for GDN/PLE-conv
                    // sources, absent for PLE's 2-D raw-ID history) are
                    // implicitly kept in full, matching the `q[0..., t]`
                    // convention used throughout the ops fallback this
                    // mirrors.
                    arrayCache[slot] = contiguous(
                        source[0..., committedNewTokens ..< (committedNewTokens + length)])
                case .stateAtIndex(let source):
                    arrayCache[slot] = contiguous(source[0..., committedNewTokens - 1])
                }
            }
        }
        for cache in caches.values {
            if let qsaCache = cache as? Qwen4ExpQSAKVCache {
                qsaCache.trim(rejected)
            }
        }
    }

    /// Release the optional resident decoder weights while keeping the decoder
    /// usable in streamed mode for subsequent calls.
    public func unloadResidentLayers() {
        residentLayers.removeAll(keepingCapacity: true)
        Memory.clearCache()
    }

    private func observeNGramCache(layerIndex: Int, current: Qwen4ExpNGramCacheStats) {
        if layerLoadingMode == .resident {
            let previous = residentNGramCacheSnapshots[layerIndex] ?? Qwen4ExpNGramCacheStats()
            ngramCacheTotals.hits += max(0, current.hits - previous.hits)
            ngramCacheTotals.misses += max(0, current.misses - previous.misses)
            residentNGramCacheSnapshots[layerIndex] = current
        } else {
            // Streamed mode constructs a fresh PLE reader for each visit.
            ngramCacheTotals.hits += current.hits
            ngramCacheTotals.misses += current.misses
        }
        ngramCacheEntriesByLayer[layerIndex] = max(
            ngramCacheEntriesByLayer[layerIndex] ?? 0, current.entries)
    }

    /// P6.2: same delta-vs-snapshot accounting as `observeNGramCache`, for
    /// the lookup instrumentation counters.
    private func observeNGramLookup(layerIndex: Int, current: Qwen4ExpPLELookupStats) {
        if layerLoadingMode == .resident {
            let previous = residentNGramLookupSnapshots[layerIndex] ?? Qwen4ExpPLELookupStats()
            ngramLookupTotals.lookupCalls += max(0, current.lookupCalls - previous.lookupCalls)
            ngramLookupTotals.arraysConstructed += max(
                0, current.arraysConstructed - previous.arraysConstructed)
            ngramLookupTotals.dequantizeCalls += max(
                0, current.dequantizeCalls - previous.dequantizeCalls)
            ngramLookupTotals.hostReadSeconds += max(
                0, current.hostReadSeconds - previous.hostReadSeconds)
            residentNGramLookupSnapshots[layerIndex] = current
        } else {
            ngramLookupTotals.lookupCalls += current.lookupCalls
            ngramLookupTotals.arraysConstructed += current.arraysConstructed
            ngramLookupTotals.dequantizeCalls += current.dequantizeCalls
            ngramLookupTotals.hostReadSeconds += current.hostReadSeconds
        }
    }

    private func makeCache(for layerIndex: Int) -> any KVCache {
        if configuration.layerTypes[layerIndex] == .fullAttention {
            return Qwen4ExpQSAKVCache(
                budget: configuration.indexerBudget,
                compressRatio: configuration.indexerCompressRatio)
        }
        if configuration.pleLayerIDs.contains(layerIndex + 1) {
            return ArraysCache(size: 4)
        }
        return MambaCache()
    }
}

public struct Qwen4ExpStreamingLayerReport: Sendable, Equatable {
    public let layerIndex: Int
    public let loadedTensorCount: Int
    public let loadedShardCount: Int
    public let materializedBytes: Int64
    public let loadDuration: Double
    public let forwardDuration: Double
    public let outputShape: [Int]

    public init(
        layerIndex: Int,
        loadedTensorCount: Int,
        loadedShardCount: Int,
        materializedBytes: Int64,
        loadDuration: Double,
        forwardDuration: Double,
        outputShape: [Int]
    ) {
        self.layerIndex = layerIndex
        self.loadedTensorCount = loadedTensorCount
        self.loadedShardCount = loadedShardCount
        self.materializedBytes = materializedBytes
        self.loadDuration = loadDuration
        self.forwardDuration = forwardDuration
        self.outputShape = outputShape
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
