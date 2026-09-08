import Foundation

/// Configuration contract for Qwen3.8 Flash-Next (`qwen4_exp`).
///
/// This type deliberately contains configuration only. Loading it must not
/// instantiate the 51B-parameter hashed n-gram table; the Flash-Next model
/// implementation will consume it lazily once the architecture is ported.
public struct Qwen4ExpConfiguration: Decodable, Sendable, Equatable {
    public let modelType: String
    public let architectures: [String]
    public let imageTokenID: Int32?
    public let videoTokenID: Int32?
    public let visionStartTokenID: Int32?
    public let visionEndTokenID: Int32?
    public let textConfiguration: Qwen4ExpTextConfiguration
    public let visionConfiguration: Qwen4ExpVisionConfiguration
    public let quantization: Qwen4ExpQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try values.decode(String.self, forKey: .modelType)
        architectures = try values.decodeIfPresent([String].self, forKey: .architectures) ?? []
        imageTokenID = try values.decodeIfPresent(Int32.self, forKey: .imageTokenID)
        videoTokenID = try values.decodeIfPresent(Int32.self, forKey: .videoTokenID)
        visionStartTokenID = try values.decodeIfPresent(Int32.self, forKey: .visionStartTokenID)
        visionEndTokenID = try values.decodeIfPresent(Int32.self, forKey: .visionEndTokenID)
        textConfiguration = try values.decode(Qwen4ExpTextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try values.decodeIfPresent(
            Qwen4ExpVisionConfiguration.self, forKey: .visionConfiguration)
            ?? .init()
        quantization = try values.decodeIfPresent(Qwen4ExpQuantization.self, forKey: .quantization)
            ?? values.decodeIfPresent(Qwen4ExpQuantization.self, forKey: .quantizationConfig)
    }

    public func validate() throws {
        guard modelType == "qwen4_exp" else {
            throw Qwen4ExpConfigurationError.unsupportedModelType(modelType)
        }
        guard textConfiguration.numHiddenLayers == textConfiguration.layerTypes.count else {
            throw Qwen4ExpConfigurationError.layerTypeCount(
                expected: textConfiguration.numHiddenLayers,
                actual: textConfiguration.layerTypes.count)
        }
        guard textConfiguration.layerTypes.filter({ $0 == .fullAttention }).count > 0 else {
            throw Qwen4ExpConfigurationError.missingFullAttention
        }
        guard textConfiguration.layerTypes.allSatisfy({ $0 == .linearAttention || $0 == .fullAttention }) else {
            throw Qwen4ExpConfigurationError.unknownLayerType
        }
        guard textConfiguration.hcCount == 4 else {
            throw Qwen4ExpConfigurationError.invariant("hc_count doit valoir 4")
        }
        guard textConfiguration.splitNgramParts == 128 else {
            throw Qwen4ExpConfigurationError.invariant("split_ngram_parts doit valoir 128")
        }
        guard textConfiguration.ngramSize == 3 else {
            throw Qwen4ExpConfigurationError.invariant("ngram_size doit valoir 3")
        }
        guard textConfiguration.indexerBudget > 0,
              textConfiguration.indexerCompressRatio > 0 else {
            throw Qwen4ExpConfigurationError.invariant("les paramètres QSA indexer doivent être positifs")
        }
    }

    public static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Qwen4ExpConfigurationError.missingConfig(url)
        }
        do {
            let configuration = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
            try configuration.validate()
            return configuration
        } catch let error as Qwen4ExpConfigurationError {
            throw error
        } catch {
            throw Qwen4ExpConfigurationError.invalidJSON(url)
        }
    }
}

public struct Qwen4ExpTextConfiguration: Decodable, Sendable, Equatable {
    public enum LayerType: String, Decodable, Sendable, Equatable {
        case linearAttention = "linear_attention"
        case fullAttention = "full_attention"
    }

    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    /// Optional for compatibility with reduced test/config fixtures; the
    /// released checkpoint provides it and runtime code falls back to `silu`.
    public let hiddenAct: String?
    public let outputGateType: String?
    public let layerTypes: [LayerType]
    public let fullAttentionInterval: Int
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
    public let ngramSize: Int
    public let ngramVocabSizeBase: Int
    public let splitNgramParts: Int
    public let headsPerNgram: Int?
    public let pleEmbedDim: Int?
    public let makeNgramVocabSizeDivisibleBy: Int?
    public let eosTokenID: Int32?
    public let pleLayerIDs: [Int]
    public let pleConvKernelSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case outputGateType = "output_gate_type"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case indexerHeadDim = "indexer_head_dim"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerNHeads = "indexer_n_heads"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case ngramSize = "ngram_size"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case splitNgramParts = "split_ngram_parts"
        case headsPerNgram = "heads_per_ngram"
        case pleEmbedDim = "ple_embed_dim"
        case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case eosTokenID = "eos_token_id"
        case pleLayerIDs = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(
        hiddenSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, headDim: Int, layerTypes: [LayerType],
        fullAttentionInterval: Int, linearNumKeyHeads: Int,
        linearNumValueHeads: Int, linearKeyHeadDim: Int,
        linearValueHeadDim: Int, linearConvKernelDim: Int,
        numExperts: Int, numExpertsPerToken: Int, moeIntermediateSize: Int,
        sharedExpertIntermediateSize: Int, indexerBudget: Int,
        indexerCompressRatio: Int, indexerHeadDim: Int, indexerKVHeads: Int,
        indexerNHeads: Int, hcCount: Int, hcLowrank: Int, ngramSize: Int,
        ngramVocabSizeBase: Int, splitNgramParts: Int, pleLayerIDs: [Int],
        pleConvKernelSize: Int, vocabSize: Int, maxPositionEmbeddings: Int,
        hiddenAct: String? = "silu", outputGateType: String? = nil,
        headsPerNgram: Int? = nil, pleEmbedDim: Int? = nil,
        makeNgramVocabSizeDivisibleBy: Int? = nil, eosTokenID: Int32? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.outputGateType = outputGateType
        self.layerTypes = layerTypes
        self.fullAttentionInterval = fullAttentionInterval
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
        self.ngramSize = ngramSize
        self.ngramVocabSizeBase = ngramVocabSizeBase
        self.splitNgramParts = splitNgramParts
        self.headsPerNgram = headsPerNgram
        self.pleEmbedDim = pleEmbedDim
        self.makeNgramVocabSizeDivisibleBy = makeNgramVocabSizeDivisibleBy
        self.eosTokenID = eosTokenID
        self.pleLayerIDs = pleLayerIDs
        self.pleConvKernelSize = pleConvKernelSize
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }
}

public struct Qwen4ExpVisionConfiguration: Decodable, Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let depth: Int
    public let numHeads: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let outHiddenSize: Int

    public init(
        hiddenSize: Int = 1152,
        intermediateSize: Int = 4304,
        depth: Int = 27,
        numHeads: Int = 16,
        patchSize: Int = 16,
        spatialMergeSize: Int = 2,
        temporalPatchSize: Int = 2,
        outHiddenSize: Int = 2560
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.depth = depth
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.spatialMergeSize = spatialMergeSize
        self.temporalPatchSize = temporalPatchSize
        self.outHiddenSize = outHiddenSize
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case depth
        case numHeads = "num_heads"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case outHiddenSize = "out_hidden_size"
    }
}

public struct Qwen4ExpQuantization: Decodable, Sendable, Equatable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String?
    /// Optional per-checkpoint override for the routed-expert
    /// (`*.mlp.switch_mlp.*`) weights. Absent (the Vontra checkpoint's
    /// current shape) means the experts share this same spec; a Q3
    /// requantified checkpoint sets this to `{group_size: 64, bits: 3,
    /// mode: "affine"}` while attention/shared-expert/gate/norms stay on
    /// the outer spec.
    ///
    /// This is a distinct (non-recursive) type rather than
    /// `Qwen4ExpQuantization?` again: a struct cannot store a property of
    /// its own type (Swift rejects it as an infinite-size value type), and
    /// an experts-of-experts override has no meaning anyway.
    public let experts: Qwen4ExpQuantizationOverride?

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
        case experts
    }

    public init(groupSize: Int, bits: Int, mode: String?, experts: Qwen4ExpQuantizationOverride? = nil) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.experts = experts
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try values.decode(Int.self, forKey: .groupSize)
        bits = try values.decode(Int.self, forKey: .bits)
        mode = try values.decodeIfPresent(String.self, forKey: .mode)
        experts = try values.decodeIfPresent(Qwen4ExpQuantizationOverride.self, forKey: .experts)
    }
}

/// The `quantization.experts` override block: same shape as
/// `Qwen4ExpQuantization` minus the (meaningless) nested `experts` field.
public struct Qwen4ExpQuantizationOverride: Decodable, Sendable, Equatable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String?

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(groupSize: Int, bits: Int, mode: String?) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }
}

public enum Qwen4ExpConfigurationError: LocalizedError, Equatable {
    case missingConfig(URL)
    case invalidJSON(URL)
    case unsupportedModelType(String)
    case layerTypeCount(expected: Int, actual: Int)
    case missingFullAttention
    case unknownLayerType
    case invariant(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig(let url): return "config.json absent : \(url.path)"
        case .invalidJSON(let url): return "config.json invalide : \(url.path)"
        case .unsupportedModelType(let type): return "model_type Flash-Next inattendu : \(type)"
        case .layerTypeCount(let expected, let actual):
            return "Nombre de layer_types invalide : attendu \(expected), obtenu \(actual)"
        case .missingFullAttention: return "La configuration Flash-Next ne contient aucune couche full_attention."
        case .unknownLayerType: return "La configuration contient un type de couche inconnu."
        case .invariant(let message): return "Invariant Flash-Next invalide : \(message)"
        }
    }
}
