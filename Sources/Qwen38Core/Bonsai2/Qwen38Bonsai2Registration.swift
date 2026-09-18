import Foundation
import MLXLMCommon
import MLXVLM

/// `prism_hadamard_qwen35` (Bonsai 2, Prism ML): Qwen3.8-27B with a Hadamard-
/// rotated 2-bit checkpoint. Same module topology as `qwen3_5`, so it reuses
/// `Qwen35Configuration`/`Qwen35` — the Hadamard modules are installed
/// afterwards by `Qwen38Bonsai2Loader` (docs/bonsai2/plan.md).
public enum Qwen38Bonsai2 {
    public static let modelType = "prism_hadamard_qwen35"

    /// Idempotent; call before `VLMModelFactory.shared.loadContainer`.
    /// Mirrors the private `create(Qwen35Configuration.self, Qwen35.init)`
    /// entry for "qwen3_5" in VLMModelFactory.swift (line ≈ 95): same
    /// configuration type, same model class, same validation hook.
    public static func register() async {
        await VLMTypeRegistry.shared.registerModelType(modelType) { data in
            let configuration = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
            if let validating = configuration as? ModelConfigurationValidating {
                try validating.validateModelConfiguration()
            }
            return Qwen35(configuration)
        }
    }
}
