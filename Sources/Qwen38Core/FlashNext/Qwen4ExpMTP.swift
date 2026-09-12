import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The native Flash-Next MTP head.
///
/// Flash-Next does not use the Qwen3.5 `fc([embedding, hidden])` contract.
/// It keeps the target's four-stream hidden state and projects the two inputs
/// independently before adding them.  Keeping this distinction explicit is
/// important: silently flattening the state to H produces plausible-shaped,
/// but semantically wrong, draft logits.
public final class Qwen4ExpMTPPredictor: Module {
    public let configuration: Qwen4ExpTextConfiguration
    public let fullAttentionLayerIndex: Int

    @ModuleInfo(key: "fc_embedding") public var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") public var fcHidden: Linear
    @ModuleInfo(key: "pre_fc_norm_embedding") public var preFCNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") public var preFCNormHidden: Qwen4ExpRMSNorm
    @ModuleInfo(key: "layers") public var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") public var hyperConnectionMixer: Qwen4ExpGatedResidual

    public init(
        configuration: Qwen4ExpTextConfiguration,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        expertsQuantization: Qwen4ExpQuantizationSpec? = nil,
        fusionLevel: Qwen4ExpFusionLevel = .f7GatedBranchDtype,
        /// P11.1 : le drafter reçoit la même largeur de routage que la
        /// cible sauf distinction explicite du caller — voir
        /// `Qwen4ExpMTPLoader.load` et `Qwen38FlashNextEngine`, qui passent
        /// systématiquement la valeur déjà résolue de
        /// `Qwen4ExpStreamingTextModel.routedExpertCount`.
        routedExpertCount: Int? = nil
    ) {
        self.configuration = configuration
        guard let fullAttentionLayerIndex = configuration.layerTypes.firstIndex(
            of: .fullAttention) else {
            preconditionFailure("Le predictor MTP Flash exige une couche full_attention")
        }
        self.fullAttentionLayerIndex = fullAttentionLayerIndex

        _fcEmbedding.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: configuration.hiddenSize,
            quantization: quantization)
        _fcHidden.wrappedValue = qwen4ExpLinear(
            inputDimensions: configuration.hiddenSize,
            outputDimensions: configuration.hiddenSize,
            quantization: quantization)
        _preFCNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: configuration.hiddenSize)
        _preFCNormHidden.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: configuration.hiddenSize * configuration.hcCount,
            groupSize: configuration.hiddenSize)

        // MTP has one full-attention layer.  Reusing the already parity-tested
        // Flash layer keeps QSA, gated residuals and the MoE weight contract in
        // one place; the loader below maps its local checkpoint namespace.
        _layers.wrappedValue = [Qwen4ExpDecoderLayer(
            configuration: configuration,
            layerIndex: fullAttentionLayerIndex,
            pleLayerIndex: nil,
            quantization: quantization,
            expertsQuantization: expertsQuantization,
            fusionLevel: fusionLevel,
            routedExpertCount: routedExpertCount)]
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(
            configuration: configuration,
            useCombine: false,
            quantization: quantization)
        super.init()
    }

    public func makeCache() -> Qwen4ExpQSAKVCache {
        Qwen4ExpQSAKVCache(
            budget: configuration.indexerBudget,
            compressRatio: configuration.indexerCompressRatio)
    }

    /// Produces the four-stream MTP hidden state used by the next MTP step.
    /// The target hidden input must remain the complete `[B,S,4H]` state captured
    /// immediately before the target's final hyper-connection mixer.
    public func callAsFunction(
        inputEmbeddings: MLXArray,
        targetHidden: MLXArray,
        inputIDs: MLXArray,
        cache: Qwen4ExpQSAKVCache,
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        precondition(inputEmbeddings.ndim == 3)
        precondition(targetHidden.ndim == 3)
        precondition(inputIDs.ndim == 2)
        precondition(inputEmbeddings.shape == [inputIDs.dim(0), inputIDs.dim(1), configuration.hiddenSize])
        precondition(targetHidden.shape == [inputIDs.dim(0), inputIDs.dim(1), configuration.hiddenSize * configuration.hcCount])

        let projectedEmbedding = fcEmbedding(preFCNormEmbedding(inputEmbeddings))
        // The checkpoint stores fc_hidden as H -> H even though its input norm
        // is defined over the complete four-stream state (4H).  The MTP
        // contract is therefore: normalize the 4H state, reduce the streams,
        // then apply fc_hidden.  Flattening directly into fc_hidden would make
        // the Swift model expect scales [H, 4H/group], which cannot load the
        // real Flash-Next tensors ([H, H/group]).
        let normalizedHidden = preFCNormHidden(targetHidden)
        let hiddenShape = normalizedHidden.shape
        let reducedHidden = normalizedHidden
            .reshaped([
                hiddenShape[0], hiddenShape[1],
                configuration.hcCount, configuration.hiddenSize
            ])
            .mean(axis: -2)
        let projectedHidden = fcHidden(reducedHidden)
        let merged = projectedEmbedding + projectedHidden
        let streams = tiled(merged, repetitions: [1, 1, configuration.hcCount])
        let positions = positionIDs ?? Qwen4ExpMRoPE.textPositionIDs(
            sequenceLength: inputIDs.dim(1), offset: cache.offset)
        let attentionMask = Qwen4ExpQSAAttention.causalMask(
            batch: inputIDs.dim(0), queryLength: inputIDs.dim(1),
            keyLength: cache.offset + inputIDs.dim(1), offset: cache.offset)
        let layerOutput = layers[0](
            streams,
            inputIDs: inputIDs,
            mask: attentionMask,
            cache: cache,
            positionIDs: positions)
        // Keep the complete HC state.  Flash-Next's target contract feeds
        // this [B,S,4H] tensor into the next MTP prediction; reducing it here
        // would make the first token look plausible while making token 2 use
        // the wrong input topology.  The final mixer is exposed separately
        // through `logitsHidden(from:)`.
        return layerOutput
    }

    /// Converts a four-stream MTP state to the H-wide representation consumed
    /// by the target embedding/language head.
    public func logitsHidden(from mtpHidden: MLXArray) -> MLXArray {
        precondition(mtpHidden.ndim == 3)
        precondition(mtpHidden.dim(-1) == configuration.hiddenSize * configuration.hcCount)
        return hyperConnectionMixer.mixedInput(mtpHidden)
    }

    /// P11.1 : répercute un changement de largeur de routage MoE sur le
    /// drafter déjà chargé — même mécanisme que
    /// `Qwen4ExpDecoderLayer.setRoutedExpertCount`, aucun poids touché.
    public func setRoutedExpertCount(_ count: Int) throws {
        try layers[0].setRoutedExpertCount(count)
    }
}

public struct Qwen4ExpLoadedMTPPredictor: @unchecked Sendable {
    public let model: Qwen4ExpMTPPredictor
    public let tensorCount: Int
    public let shardCount: Int
    public let materializedBytes: Int64

    public init(
        model: Qwen4ExpMTPPredictor,
        tensorCount: Int,
        shardCount: Int,
        materializedBytes: Int64
    ) {
        self.model = model
        self.tensorCount = tensorCount
        self.shardCount = shardCount
        self.materializedBytes = materializedBytes
    }
}

public enum Qwen4ExpMTPLoaderError: LocalizedError, Equatable {
    case missingMTPWeights
    case missingTensor(String)
    case duplicateTensor(String)

    public var errorDescription: String? {
        switch self {
        case .missingMTPWeights:
            return "Le checkpoint Flash-Next ne contient pas de poids MTP utilisables."
        case .missingTensor(let key):
            return "Poids MTP absent : \(key)."
        case .duplicateTensor(let key):
            return "Poids MTP dupliqué : \(key)."
        }
    }
}

/// Loads only `language_model.mtp.*` from the target checkpoint.
///
/// This deliberately does not use the generic Qwen3.5 MTP loader: Flash-Next
/// stores its MTP tensors in the target shards and its hidden-state topology is
/// four-stream/QSA-specific.
public enum Qwen4ExpMTPLoader {
    public static func load(
        from directory: URL,
        materialize: Bool = true,
        useCheckpointQuantization: Bool = true,
        uncachedIO: Bool = true,
        fusionLevel: Qwen4ExpFusionLevel = .f7GatedBranchDtype,
        /// P11.1 : surcharge de la largeur de routage MoE — voir
        /// `Qwen4ExpMTPPredictor.init`. `nil` (le défaut) laisse le
        /// comportement inchangé.
        routedExpertCount: Int? = nil
    ) throws -> Qwen4ExpLoadedMTPPredictor {
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpMTPLoaderError.missingMTPWeights
        }

        let selected = weightMap.filter { key, _ in
            key.contains("language_model.mtp.")
        }
        guard !selected.isEmpty else {
            throw Qwen4ExpMTPLoaderError.missingMTPWeights
        }

        var weights = [String: MLXArray]()
        var shards = Set<String>()
        // See Qwen4ExpCheckpointLayerLoader (P2-mem-a): F_NOCACHE reads keep
        // MTP's shard reads from filling the file cache too.
        for shard in Set(selected.values).sorted() {
            let shardURL = directory.appendingPathComponent(shard)
            shards.insert(shard)
            let reader: Qwen4ExpUncachedTensorReader? = uncachedIO
                ? try Qwen4ExpUncachedTensorReader(url: shardURL) : nil
            let arrays: [String: MLXArray]? = uncachedIO
                ? nil : try loadArraysAndMetadata(url: shardURL, stream: .cpu).0
            for (rawKey, mappedShard) in selected where mappedShard == shard {
                let value: MLXArray
                if let reader {
                    do {
                        value = try reader.array(for: rawKey)
                    } catch Qwen4ExpUncachedTensorReaderError.missingTensor {
                        throw Qwen4ExpMTPLoaderError.missingTensor(rawKey)
                    }
                } else {
                    guard let cached = arrays?[rawKey] else {
                        throw Qwen4ExpMTPLoaderError.missingTensor(rawKey)
                    }
                    eval(cached)
                    value = cached
                }
                guard let local = localKey(rawKey) else { continue }
                if weights[local] != nil {
                    throw Qwen4ExpMTPLoaderError.duplicateTensor(local)
                }
                weights[local] = value
            }
        }

        // 2026-09-09 : the MTP head has the same zero-centered norms as the
        // decoder layers (hc_norm ×3, q/k_norm, indexer q/k layernorm) and the
        // Vontra converter shifted them by +1 too (means +1.1 … +4.8 in the
        // checkpoint). The layer/global loaders already correct this (piège
        // §6.3-12); the MTP loader did not, which is the likely cause of the
        // 8-24 % acceptance measured in P-MTP PM3. Anchor and detection are
        // the same as for the decoder, so already-correct checkpoints are
        // left untouched.
        let normCorrection = Qwen4ExpWeightSanitizer.correctShiftedZeroCenteredNormWeights(weights)
        weights = normCorrection.weights
        let quantization = Qwen4ExpQuantizationSpec(configuration.quantization)
        let expertsQuantization = Qwen4ExpQuantizationSpec.experts(from: configuration.quantization)
        // 2026-09-11 : le drafter est une couche Flash complète ; sans le
        // niveau de fusion il subissait la fuite fp32 corrigée par F7 (couche
        // ~4 ms au lieu de ~1 ms) ET travaillait dans une arithmétique
        // différente de la cible, ce qui pénalisait l'acceptation.
        let model = Qwen4ExpMTPPredictor(
            configuration: configuration.textConfiguration,
            quantization: useCheckpointQuantization ? quantization : nil,
            expertsQuantization: useCheckpointQuantization ? expertsQuantization : nil,
            fusionLevel: fusionLevel,
            routedExpertCount: routedExpertCount)
        try update(model: model, weights: weights)

        let bytes: Int64
        if materialize {
            eval(model.parameters())
            bytes = model.parameters().flattened().reduce(Int64(0)) {
                $0 + Int64($1.1.nbytes)
            }
        } else {
            bytes = 0
        }
        return Qwen4ExpLoadedMTPPredictor(
            model: model,
            tensorCount: weights.count,
            shardCount: shards.count,
            materializedBytes: bytes)
    }

    private static func localKey(_ rawKey: String) -> String? {
        guard let normalized = Qwen4ExpWeightSanitizer.normalize(rawKey),
              normalized.hasPrefix("language_model.model.mtp.") else { return nil }
        return "mtp." + normalized.dropFirst("language_model.model.mtp.".count)
    }

    private static func update(
        model: Qwen4ExpMTPPredictor,
        weights: [String: MLXArray]
    ) throws {
        func child(_ prefix: String) -> ModuleParameters {
            ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights.compactMap {
                guard $0.key.hasPrefix(prefix) else { return nil }
                return (String($0.key.dropFirst(prefix.count)), $0.value)
            }))
        }

        try model.fcEmbedding.update(parameters: child("mtp.fc_embedding."), verify: [.all])
        try model.fcHidden.update(parameters: child("mtp.fc_hidden."), verify: [.all])
        try model.preFCNormEmbedding.update(
            parameters: child("mtp.pre_fc_norm_embedding."), verify: [.all])
        try model.preFCNormHidden.update(
            parameters: child("mtp.pre_fc_norm_hidden."), verify: [.all])
        try model.hyperConnectionMixer.update(
            parameters: child("mtp.hyper_connection_mixer."), verify: [.all])
        try model.layers[0].update(
            parameters: child("mtp.layers.0."), verify: [.all])
    }
}
