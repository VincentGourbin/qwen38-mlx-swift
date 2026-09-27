import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

extension Qwen38Bonsai2 {
    static let visionPrefix = "vision_tower."

    /// Loads a Bonsai 2 pack into a `ModelContainer` with upstream
    /// mlx-swift-lm only. `VLMModelFactory.loadContainer` cannot be used: its
    /// `loadWeights` verifies every safetensors key (`verify: [.all]`), and
    /// the pack's `.signs` vectors belong to no module. This replays the
    /// public steps of `VLMModelFactory._load` and drops them first; the
    /// Hadamard installer reads them from the file itself.
    ///
    /// `textOnly`: the vision tower's weights are neither loaded nor
    /// evaluated (−0.92 GB resident). Its modules still exist, holding lazy,
    /// never-materialised initial values, so an image would run on garbage:
    /// callers that pass `textOnly` must refuse images.
    public static func loadContainer(
        directory: URL, tokenizerLoader: any TokenizerLoader, textOnly: Bool = false
    ) async throws -> ModelContainer {
        setenv("MLX_QWEN_FOUR_GDN", "0", 1)
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let configuration = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configData)
        let model = Qwen35(configuration)

        var weights = try loadArrays(url: directory.appending(component: "model.safetensors"))
        weights = model.sanitize(weights: weights, metadata: [:])
        weights = weights.filter { !$0.key.hasSuffix(".signs") }
        if textOnly {
            weights = weights.filter { !$0.key.hasPrefix(visionPrefix) }
        }
        if let perLayer = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? perLayer.quantization(layer: path)?.asTuple : nil
            }
        }
        if textOnly {
            // Every language key must still be present and shaped right;
            // only the vision keys are allowed to stay at their initial value.
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys, .shapeMismatch])
            let languageParameters = model.parameters().flattened()
                .filter { !$0.0.hasPrefix(visionPrefix) }
            let missing = languageParameters.filter { weights[$0.0] == nil }.map(\.0)
            guard missing.isEmpty else {
                throw Qwen38Bonsai2LoaderError.missingWeights(Array(missing.prefix(3)))
            }
            eval(languageParameters.map(\.1))
        } else {
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model)
        }

        let installer = try Qwen38Bonsai2Loader(directory: directory)
        let replaced = installer.install(into: model)
        guard replaced == installer.expectedReplacementCount else {
            throw Qwen38Bonsai2LoaderError.unexpectedReplacementCount(
                expected: installer.expectedReplacementCount, actual: replaced)
        }

        let tokenizer = try await tokenizerLoader.load(from: directory)
        let processorData = try Data(
            contentsOf: directory.appending(component: "preprocessor_config.json"))
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: processorData, processorType: "Qwen3VLProcessor", tokenizer: tokenizer)
        var eosTokenIds = base.effectiveEOSTokenIds
        if let data = try? Data(contentsOf: directory.appending(component: "generation_config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let one = object["eos_token_id"] as? Int {
                eosTokenIds = [one]
            } else if let many = object["eos_token_id"] as? [Int] {
                eosTokenIds = Set(many)
            }
        }
        let modelConfiguration = ModelConfiguration(directory: directory, eosTokenIds: eosTokenIds)
        return ModelContainer(context: ModelContext(
            configuration: modelConfiguration, model: model, processor: processor,
            tokenizer: tokenizer))
    }
}
