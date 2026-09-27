import Foundation

/// `prism_hadamard_qwen35` (Bonsai 2, Prism ML): Qwen3.8-27B with a Hadamard-
/// rotated 2-bit checkpoint. Same module topology as `qwen3_5`, so it reuses
/// upstream `Qwen35Configuration`/`Qwen35`; `loadContainer(directory:tokenizerLoader:)`
/// loads it and installs the Hadamard modules (docs/bonsai2/plan.md).
public enum Qwen38Bonsai2 {
    public static let modelType = "prism_hadamard_qwen35"
}
