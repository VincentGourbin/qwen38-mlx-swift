import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

// Chargement Bonsai 2 sans fork : même étapes que VLMModelFactory._load, mais
// les vecteurs `.signs` sont retirés avant `update(verify: .all)` au lieu d'un
// patch de Load.swift dans mlx-swift-lm.
func loadBonsai2(directory: URL) async throws -> ModelContainer {
    setenv("MLX_QWEN_FOUR_GDN", "0", 1)
    let configData = try Data(contentsOf: directory.appending(component: "config.json"))
    let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
    let qwenConfig = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configData)
    let model = Qwen35(qwenConfig)

    var weights = try loadArrays(url: directory.appending(component: "model.safetensors"))
    weights = model.sanitize(weights: weights, metadata: [:])
    weights = weights.filter { !$0.key.hasSuffix(".signs") }
    if let perLayer = base.perLayerQuantization {
        quantize(model: model) { path, _ in
            weights["\(path).scales"] != nil ? perLayer.quantization(layer: path)?.asTuple : nil
        }
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    eval(model)

    let installer = try Qwen38Bonsai2Loader(directory: directory)
    let replaced = installer.install(into: model)
    guard replaced == installer.expectedReplacementCount else {
        throw Qwen38Bonsai2LoaderError.unexpectedReplacementCount(
            expected: installer.expectedReplacementCount, actual: replaced)
    }

    let tokenizer = try await Qwen38TokenizerLoader().load(from: directory)
    let processorData = try Data(contentsOf: directory.appending(component: "preprocessor_config.json"))
    let processor = try await VLMProcessorTypeRegistry.shared.createModel(
        configuration: processorData, processorType: "Qwen3VLProcessor", tokenizer: tokenizer)
    var eos = base.effectiveEOSTokenIds
    struct Gen: Decodable { let eos_token_id: IntOrArray? }
    enum IntOrArray: Decodable {
        case one(Int), many([Int])
        init(from d: Decoder) throws {
            let c = try d.singleValueContainer()
            if let i = try? c.decode(Int.self) { self = .one(i) } else { self = .many(try c.decode([Int].self)) }
        }
        var values: [Int] { switch self { case .one(let i): [i]; case .many(let a): a } }
    }
    if let g = try? JSONDecoder().decode(Gen.self, from: Data(contentsOf: directory.appending(component: "generation_config.json"))),
       let ids = g.eos_token_id?.values { eos = Set(ids) }
    let configuration = ModelConfiguration(directory: directory, eosTokenIds: eos)
    return ModelContainer(context: ModelContext(
        configuration: configuration, model: model, processor: processor, tokenizer: tokenizer))
}

let directory = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1]
    : NSHomeDirectory() + "/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
Memory.cacheLimit = 2 * 1024 * 1024 * 1024
let t0 = Date()
let container = try await loadBonsai2(directory: directory)
print(String(format: "chargé en %.1f s · actif %d Mo", Date().timeIntervalSince(t0), Memory.activeMemory / 1_048_576))

let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: 120, temperature: 0),
                          additionalContext: ["enable_thinking": false])
let t1 = Date()
let answer = try await session.respond(to: "Quelle est la capitale de l'Australie ? Réponds en une phrase.")
print("réponse :", answer)
print(String(format: "génération %.1f s · pic %d Mo · actif %d Mo", Date().timeIntervalSince(t1),
             Memory.peakMemory / 1_048_576, Memory.activeMemory / 1_048_576))
